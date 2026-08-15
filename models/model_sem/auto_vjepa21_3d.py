import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models import register
from models.model_sem.base.transformer import ResidualAttentionBlock, ResidualAttentionBlock_sem
from models.model_sem.base.utils import get_model_dims, init_weights
from models.model_sem.base.rope import get_freqs
from vjepa2.app.vjepa_2_1.models.vision_transformer import vit_base


def load_pretrained_vjepa21_pt_weights(model, pretrained_weights_path: str):
    ckpt = torch.load(pretrained_weights_path, map_location="cpu")
    if "encoder" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'encoder'. Keys={list(ckpt.keys())[:20]}")
    sd = ckpt["encoder"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[VJEPA2.1] Loaded encoder weights from {pretrained_weights_path} with msg: {msg}")


class VFMFeatureCompressor3D(nn.Module):
    """
    Compress VJEPA teacher tokens into a 3D latent volume with convolutions
    and a lightweight 3D transformer.

    Example (256px, 16 frames):
      teacher grid (8, 16, 16), dim=768  ->  latent (4, 16, 16), dim=64
    """

    def __init__(
        self,
        in_channels=768,
        out_channels=64,
        in_grid=(8, 16, 16),
        out_grid=(4, 16, 16),
        model_size="small",
    ):
        super().__init__()
        self.in_grid = list(in_grid)
        self.out_grid = list(out_grid)
        self.out_channels = out_channels
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)

        t_ratio = self.in_grid[0] // self.out_grid[0]
        if t_ratio < 1 or self.in_grid[0] % self.out_grid[0] != 0:
            raise ValueError(
                f"Temporal downsample from {self.in_grid[0]} to {self.out_grid[0]} is unsupported."
            )

        self.temporal_down = nn.Conv3d(
            in_channels,
            self.width,
            kernel_size=(t_ratio, 1, 1),
            stride=(t_ratio, 1, 1),
            bias=True,
        )
        self.freqs = get_freqs(0, self.out_grid, head_dim=self.width // self.heads)
        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers,
        )
        self.proj_out = nn.Conv3d(self.width, out_channels, kernel_size=1, bias=True)
        self.apply(init_weights)

    def forward(self, vfm_feats):
        """
        Args:
            vfm_feats: [B, N, D_teacher]
        Returns:
            z: [B, C, T, H, W]
        """
        B, N, _ = vfm_feats.shape
        t, h, w = self.in_grid
        if N != t * h * w:
            raise ValueError(f"Expected {t * h * w} teacher tokens, got {N}.")

        x = rearrange(vfm_feats, "b (t h w) c -> b c t h w", t=t, h=h, w=w)
        x = self.temporal_down(x)

        if x.shape[2:] != tuple(self.out_grid):
            x = F.adaptive_avg_pool3d(x, output_size=self.out_grid)

        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = self.model_layers(x, freqs=self.freqs.to(x.device))
        x = rearrange(
            x,
            "b (t h w) c -> b c t h w",
            t=self.out_grid[0],
            h=self.out_grid[1],
            w=self.out_grid[2],
        )
        x = self.proj_out(x)
        return x


class LatentDecoder3D(nn.Module):
    """
    Decode a 3D latent volume back to video with a 3D transformer + ConvTranspose3D.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=3,
        in_grid=(4, 16, 16),
        patch_size=(4, 16, 16),
        out_grid=(16, 256, 256),
        model_size="base",
        inter_layer_indices=(2,),
    ):
        super().__init__()
        self.in_grid = list(in_grid)
        self.patch_size = list(patch_size)
        self.out_grid = list(out_grid)
        self.grid_size = self.in_grid[0] * self.in_grid[1] * self.in_grid[2]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        self.inter_layer_indices = tuple(inter_layer_indices)
        self.multi_layer_inter = len(self.inter_layer_indices) > 1

        self.proj_in = nn.Conv3d(in_channels, self.width, kernel_size=1, bias=True)
        self.freqs = get_freqs(0, self.in_grid, head_dim=self.width // self.heads)
        self.model_layers = ResidualAttentionBlock_sem(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers,
            inter_layer_indices=self.inter_layer_indices,
        )
        self.proj_out = nn.ConvTranspose3d(
            in_channels=self.width,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.apply(init_weights)

    def forward(self, z):
        """
        Args:
            z: [B, C, T, H, W]
        Returns:
            pred_video: [B, 3, T_out, H_out, W_out]
            inner_feat: [B, N, width]
        """
        x = self.proj_in(z)
        x = rearrange(
            x,
            "b c t h w -> b (t h w) c",
            t=self.in_grid[0],
            h=self.in_grid[1],
            w=self.in_grid[2],
        )
        x, inter_out = self.model_layers(x, freqs=self.freqs.to(x.device))

        if self.multi_layer_inter:
            inter_feats = inter_out[list(inter_out.keys())[-1]]
        else:
            inter_feats = inter_out

        x = rearrange(
            x,
            "b (t h w) c -> b c t h w",
            t=self.in_grid[0],
            h=self.in_grid[1],
            w=self.in_grid[2],
        )
        pred_video = self.proj_out(x)
        return pred_video, inter_feats


class FeatureAligner(nn.Module):
    def __init__(self, student_dim=512, teacher_dim=768, global_loss_weight=0.2):
        super().__init__()
        self.global_loss_weight = global_loss_weight
        self.proj = nn.Sequential(
            nn.Linear(student_dim, student_dim * 2),
            nn.GELU(),
            nn.Linear(student_dim * 2, teacher_dim),
        )

    def forward(self, student_feat, teacher_feat, teacher_grid, student_grid):
        tt, ht, wt = teacher_grid
        st, sh, sw = student_grid
        B = student_feat.shape[0]

        if teacher_feat.shape[1] == tt * ht * wt + 1:
            teacher_feat = teacher_feat[:, 1:, :]

        student_feat_proj = self.proj(student_feat)

        teacher_3d = teacher_feat.transpose(1, 2).view(B, -1, tt, ht, wt).contiguous()
        student_3d = student_feat_proj.transpose(1, 2).view(B, -1, st, sh, sw).contiguous()

        teacher_aligned = teacher_3d
        student_aligned = F.interpolate(
            student_3d,
            size=(tt, ht, wt),
            mode="trilinear",
            align_corners=False,
        )

        teacher_local_flat = teacher_aligned.flatten(2).transpose(1, 2)
        student_local_flat = student_aligned.flatten(2).transpose(1, 2)

        teacher_local_norm = F.normalize(teacher_local_flat, dim=-1, eps=1e-8)
        student_local_norm = F.normalize(student_local_flat, dim=-1, eps=1e-8)

        local_cos_sim = (student_local_norm * teacher_local_norm).sum(dim=-1)
        local_loss = (-local_cos_sim).mean()

        teacher_global = F.adaptive_avg_pool3d(teacher_aligned, (1, 1, 1)).flatten(1)
        student_global = F.adaptive_avg_pool3d(student_aligned, (1, 1, 1)).flatten(1)

        teacher_global_norm = F.normalize(teacher_global, dim=-1, eps=1e-8)
        student_global_norm = F.normalize(student_global, dim=-1, eps=1e-8)

        global_cos_sim = (student_global_norm * teacher_global_norm).sum(dim=-1)
        global_loss = (-global_cos_sim).mean()

        total_loss = local_loss + self.global_loss_weight * global_loss
        loss_dict = {
            "repa_loss_total": total_loss.item(),
            "repa_loss_local": local_loss.item(),
            "repa_loss_global": global_loss.item(),
        }
        return total_loss, loss_dict


@register("autoencoder_vjepa21_3d_c64_repa")
class AutoEncoderVJepa21_3D_C64Repa(nn.Module):
    """
    V-JEPA 2.1 (ViT-B distilled from ViT-G) autoencoder with:
    - frozen VJEPA2.1 teacher
    - 3D conv + transformer latent compressor: (8,16,16) x 768 -> (4,16,16) x 64
    - 3D transformer decoder with ConvTranspose3D upsampling
    - REPA alignment loss
    """

    def __init__(
        self,
        bottleneck=None,
        prior_model=None,
        latent_dim=64,
        latent_temporal=4,
        latent_spatial=16,
        input_size=256,
        repa_loss_weight=1.0,
        global_loss_weight=0.2,
        vjepa21_encoder_ckpt="/xxx/vjepa2_1_vitb_dist_vitG_384.pt",
        vjepa21_num_frames=16,
        vjepa21_tubelet_size=2,
        vjepa21_patch_size=16,
        vjepa21_use_bf16=False,
        vjepa21_teacher_out_layers=None,
        compressor_model_size="small",
        decoder_model_size="base",
        **kwargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.repa_loss_weight = repa_loss_weight
        self.input_size = int(input_size)
        self.vjepa21_num_frames = int(vjepa21_num_frames)
        self.vjepa21_tubelet_size = int(vjepa21_tubelet_size)
        self.vjepa21_patch_size = int(vjepa21_patch_size)
        self.vjepa21_use_bf16 = bool(vjepa21_use_bf16)
        self.vjepa21_encoder_ckpt = vjepa21_encoder_ckpt
        if vjepa21_teacher_out_layers is None:
            # ViT-B depth=12 hierarchical layers in VJEPA2.1
            vjepa21_teacher_out_layers = [2, 5, 8, 11]
        self.vjepa21_teacher_out_layers = list(vjepa21_teacher_out_layers)

        teacher_t = self.vjepa21_num_frames // self.vjepa21_tubelet_size
        teacher_h = self.input_size // self.vjepa21_patch_size
        teacher_w = self.input_size // self.vjepa21_patch_size
        self.vfm_grid = (teacher_t, teacher_h, teacher_w)
        self.latent_grid = (latent_temporal, latent_spatial, latent_spatial)
        self.num_latent_tokens = latent_temporal * latent_spatial * latent_spatial

        self.latent_compressor = VFMFeatureCompressor3D(
            in_channels=768,
            out_channels=latent_dim,
            in_grid=self.vfm_grid,
            out_grid=self.latent_grid,
            model_size=compressor_model_size,
        )

        decoder_patch = (
            self.vjepa21_num_frames // latent_temporal,
            self.input_size // latent_spatial,
            self.input_size // latent_spatial,
        )
        self.decoder = LatentDecoder3D(
            in_channels=latent_dim,
            out_channels=3,
            in_grid=self.latent_grid,
            patch_size=decoder_patch,
            out_grid=[self.vjepa21_num_frames, self.input_size, self.input_size],
            model_size=decoder_model_size,
        )

        self.teacher_model = None
        self._init_vjepa21_teacher()

        self.feature_aligner = FeatureAligner(
            student_dim=self.decoder.width,
            teacher_dim=self.teacher_dim,
            global_loss_weight=global_loss_weight,
        )
        self.prior_model = None

    def _init_vjepa21_teacher(self):
        if (self.vjepa21_encoder_ckpt is None) or (not os.path.exists(self.vjepa21_encoder_ckpt)):
            print(f"ERROR: vjepa21_encoder_ckpt not found: {self.vjepa21_encoder_ckpt}")
            self.teacher_dim = 768
            return

        print("[VJEPA2.1] Initializing teacher (ViT-B) ...")
        self.teacher_model = vit_base(
            img_size=(self.input_size, self.input_size),
            num_frames=self.vjepa21_num_frames,
            tubelet_size=self.vjepa21_tubelet_size,
            use_rope=True,
            modality_embedding=True,
            handle_nonsquare_inputs=True,
            out_layers=self.vjepa21_teacher_out_layers,
        )
        load_pretrained_vjepa21_pt_weights(self.teacher_model, self.vjepa21_encoder_ckpt)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        if self.vjepa21_use_bf16:
            self.teacher_model = self.teacher_model.to(dtype=torch.bfloat16)

        self.teacher_dim = self.teacher_model.embed_dim
        print(
            f"[VJEPA2.1] Teacher loaded. embed_dim={self.teacher_dim}, "
            f"img_size={self.input_size}, num_frames={self.vjepa21_num_frames}, "
            f"teacher_grid={self.vfm_grid}, latent_grid={self.latent_grid}, "
            f"teacher_out_layers={self.vjepa21_teacher_out_layers}"
        )

    def preprocess_for_teacher(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    def _get_teacher_feats_list(self, x: torch.Tensor):
        assert self.teacher_model is not None, "Teacher model is required for encoding."
        if next(self.teacher_model.parameters()).device != x.device:
            self.teacher_model.to(x.device)
        target_dtype = next(self.teacher_model.parameters()).dtype
        return self.teacher_model(x.to(dtype=target_dtype), training=False)

    @staticmethod
    def _sum_teacher_feats(feats_list):
        if isinstance(feats_list, torch.Tensor):
            return feats_list.float()
        return sum(feats_list).float()

    def _latent_to_tokens(self, z):
        return rearrange(z, "b c t h w -> b (t h w) c")

    def _tokens_to_latent(self, z_tokens):
        t, h, w = self.latent_grid
        return rearrange(z_tokens, "b (t h w) c -> b c t h w", t=t, h=h, w=w)

    @torch.no_grad()
    def encode(self, x, **kwargs):
        """
        x: [B, 3, T, H, W] in [-1, 1]
        Returns: [B, num_latent_tokens, latent_dim]
        """
        x_vjepa = self.preprocess_for_teacher(x)
        feats_list = self._get_teacher_feats_list(x_vjepa)
        vfm_feats = self._sum_teacher_feats(feats_list)
        z = self.latent_compressor(vfm_feats)
        return self._latent_to_tokens(z)

    def decode(self, z):
        """
        z: [B, num_latent_tokens, latent_dim] or [B, C, T, H, W]
        """
        if z.ndim == 3:
            z = self._tokens_to_latent(z)
        pred_video, _ = self.decoder(z)
        return pred_video

    def forward(self, data, **kwargs):
        x = data
        x_vjepa = self.preprocess_for_teacher(x)

        with torch.no_grad():
            feats_list = self._get_teacher_feats_list(x_vjepa)
            vfm_feats = self._sum_teacher_feats(feats_list)
            if isinstance(feats_list, torch.Tensor):
                target_teacher_feat = feats_list.float()
            else:
                target_teacher_feat = feats_list[-1].float()

        z = self.latent_compressor(vfm_feats)
        pred_video, inner_feat = self.decoder(z)

        teacher_grid = self.vfm_grid
        student_grid = self.latent_grid

        repa_loss, repa_loss_dict = self.feature_aligner(
            student_feat=inner_feat,
            teacher_feat=target_teacher_feat,
            teacher_grid=teacher_grid,
            student_grid=student_grid,
        )

        return_dict = {
            "pred_frames": pred_video.contiguous(),
            "repa_loss": repa_loss * self.repa_loss_weight,
            "loss_kl": torch.tensor(0.0, device=x.device),
            "loss_commit": torch.tensor(0.0, device=x.device),
            "loss_codebook": torch.tensor(0.0, device=x.device),
        }
        return_dict.update(repa_loss_dict)
        return return_dict

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True, version="sd"):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location="cpu")
        else:
            assert isinstance(ckpt, dict), "checkpoint must be a dict or a path to a checkpoint"

        kwargs = ckpt["model"]["args"]
        model = cls(**kwargs)

        if load_state_dict:
            if version == "sd":
                sd = ckpt["model"]["sd"]
            elif version.startswith("ema"):
                alpha = float(version.split("_")[1])
                sd = ckpt["model"]["ema_sd"][alpha]
            else:
                raise ValueError(f"Unknown version: {version}")
            model.load_state_dict(sd, strict=False)
        return model
