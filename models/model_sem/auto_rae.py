
import os
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from models.model_sem.base.blocks import TokenizerEncoder1D, Decoder_sem
from models import register
from vjepa2.src.models.vision_transformer import vit_large_rope


def download_file(url, local_path, chunk_size=1024):
    if (url is None) or (str(url).strip() == ""):
        raise ValueError("download_file got empty url")
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {url} -> {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def load_pretrained_vjepa2_pt_weights(model, pretrained_weights_path: str):
    ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
    if "encoder" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'encoder'. Keys={list(ckpt.keys())[:20]}")
    sd = ckpt["encoder"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[VJEPA2] Loaded encoder weights from {pretrained_weights_path} with msg: {msg}")





class FeatureAligner(nn.Module):
    def __init__(self, student_dim=512, teacher_dim=1024, global_loss_weight=0.2):
        super().__init__()
        self.global_loss_weight = global_loss_weight
        self.proj = nn.Sequential(
            nn.Linear(student_dim, student_dim * 2),
            nn.GELU(),
            nn.Linear(student_dim * 2, teacher_dim)
        )

    def forward(self, student_feat, teacher_feat, teacher_grid, student_grid):
        """
        Args:
            student_feat: [B, N_s, student_dim]
            teacher_feat: [B, N_t, teacher_dim]
            teacher_grid: (tt, ht, wt)
            student_grid: (st, sh, sw)
        """
        tt, ht, wt = teacher_grid
        st, sh, sw = student_grid
        B = student_feat.shape[0]

        if teacher_feat.shape[1] == tt * ht * wt + 1:
            teacher_feat = teacher_feat[:, 1:, :]

        student_feat_proj = self.proj(student_feat)

        teacher_3d = teacher_feat.transpose(1, 2).view(B, -1, tt, ht, wt).contiguous()
        student_3d = student_feat_proj.transpose(1, 2).view(B, -1, st, sh, sw).contiguous()

        target_t, target_h, target_w = tt, ht, wt

        teacher_aligned = teacher_3d
        student_aligned = F.interpolate(
            student_3d,
            size=(target_t, target_h, target_w),
            mode='trilinear',
            align_corners=False
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
            "repa_loss_global": global_loss.item()
        }
        return total_loss, loss_dict


@register("autoencoder_vfm_ae_c32_repa")
class AutoEncoderC32Repa(nn.Module):
    """
    Plain autoencoder (no KL loss) with:
    - 512 latent tokens
    - 32-dim bottleneck
    - REPA loss (local + global cosine similarity with VJEPA teacher)
    """

    def __init__(
        self,
        bottleneck=None,
        prior_model=None,
        num_latent_tokens=1024,
        latent_dim=64,
        input_size=256,
        bypass_tokenizer=True,
        use_vjepa_loss=True,
        repa_loss_weight=1.0,
        global_loss_weight=0.2,
        vjepa2_encoder_ckpt="",
        vjepa2_img_size=256,
        vjepa2_num_frames=16,
        vjepa2_sample_strategy="repeat",
        vjepa2_tubelet_size=2,
        vjepa2_patch_size=16,
        vjepa2_use_bf16=False,
        **kwargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.repa_loss_weight = repa_loss_weight
        self.vfm_grid = (8, 16, 16)
        token_size = 512

        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='small',
            in_channels=1024,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )

        # 512 → 32 (compress to low-dim bottleneck)
        self.enc_proj = nn.Linear(token_size, latent_dim)
        # 32 → 512 (project back up before decoding)
        self.dec_proj = nn.Linear(latent_dim, token_size)

        self.decoder = Decoder_sem(
            model_size='base',
            patch_size=[4, 16, 16],
            in_channels=token_size,
            out_channels=3,
            in_tokens=num_latent_tokens,
            out_grid=[16, 256, 256],
        )

        self.teacher_model = None
        self.vjepa2_encoder_ckpt = vjepa2_encoder_ckpt
        self.vjepa2_img_size = int(vjepa2_img_size)
        self.vjepa2_num_frames = int(vjepa2_num_frames)
        self.vjepa2_sample_strategy = str(vjepa2_sample_strategy)
        self.vjepa2_tubelet_size = int(vjepa2_tubelet_size)
        self.vjepa2_patch_size = int(vjepa2_patch_size)
        self.vjepa2_use_bf16 = bool(vjepa2_use_bf16)
        self._init_vjepa2_teacher()

        self.feature_aligner = FeatureAligner(
            student_dim=self.decoder.width,
            teacher_dim=self.teacher_dim,
            global_loss_weight=global_loss_weight,
        )

    def _init_vjepa2_teacher(self):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            self.teacher_dim = 1024
            return
        print("[VJEPA2] Initializing teacher (PyTorch) ...")
        self.teacher_model = vit_large_rope(
            img_size=(self.vjepa2_img_size, self.vjepa2_img_size),
            num_frames=self.vjepa2_num_frames,
        )
        load_pretrained_vjepa2_pt_weights(self.teacher_model, self.vjepa2_encoder_ckpt)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        if self.vjepa2_use_bf16:
            self.teacher_model = self.teacher_model.to(dtype=torch.bfloat16)
        teacher_dim = getattr(self.teacher_model, "embed_dim", None)
        if teacher_dim is None:
            raise AttributeError("teacher_model has no attribute embed_dim; please check VJEPA2 model class.")
        self.teacher_dim = teacher_dim
        print(f"[VJEPA2] Teacher loaded. embed_dim={teacher_dim}, img_size={self.vjepa2_img_size}, num_frames={self.vjepa2_num_frames}")
        self.prior_model = None

    def preprocess_for_teacher(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        x_vjepa = (x - mean) / std
        return x_vjepa

    def _get_teacher_feats_list(self, x: torch.Tensor):
        assert self.teacher_model is not None, "Teacher model is required for encoding."
        if next(self.teacher_model.parameters()).device != x.device:
            self.teacher_model.to(x.device)
        target_dtype = next(self.teacher_model.parameters()).dtype
        feats_list = self.teacher_model(x.to(dtype=target_dtype))
        return feats_list

    @torch.no_grad()
    def encode(self, x, **kwargs):
        """
        x: [B, 3, T, H, W] range [-1, 1]
        Returns: z [B, 512, 32]
        """
        x_vjepa = self.preprocess_for_teacher(x)
        feats_list = self._get_teacher_feats_list(x_vjepa)
        vfm_feats = sum(feats_list).float()               # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)        # [B, 512, 512]
        z = self.enc_proj(latent)                         # [B, 512, 32]
        return z

    def decode(self, z):
        """
        z: [B, 512, 32]
        """
        z_up = self.dec_proj(z)                           # [B, 512, 512]
        pred_video, _ = self.decoder(z_up)                # [B, 3, T, H, W]
        return pred_video

    def forward(self, data, **kwargs):
        x = data
        x_vjepa = self.preprocess_for_teacher(x)

        with torch.no_grad():
            feats_list = self._get_teacher_feats_list(x_vjepa)
            vfm_feats = sum(feats_list).float()
            target_teacher_feat = feats_list[-1].float()

        latent = self.tokenizer_encoder(vfm_feats)        # [B, 512, 512]
        z = self.enc_proj(latent)                         # [B, 512, 32]
        z_up = self.dec_proj(z)                           # [B, 512, 512]

        pred_video, inner_feat = self.decoder(z_up)       # [B, 3, T, H, W], [B, N_grid, decoder_width]

        tt = self.vjepa2_num_frames // self.vjepa2_tubelet_size
        ht = self.vjepa2_img_size // self.vjepa2_patch_size
        wt = self.vjepa2_img_size // self.vjepa2_patch_size
        teacher_grid = (tt, ht, wt)
        student_grid = tuple(self.decoder.grid)

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
    def from_checkpoint(cls, ckpt, load_state_dict=True, version='sd'):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location='cpu')
        else:
            assert isinstance(ckpt, dict), "checkpoint must be a dict or a path to a checkpoint"

        kwargs = ckpt["model"]["args"]
        model = cls(**kwargs)

        if load_state_dict:
            if version == 'sd':
                sd = ckpt["model"]["sd"]
            elif version.startswith('ema'):
                alpha = float(version.split('_')[1])
                sd = ckpt["model"]['ema_sd'][alpha]
            else:
                raise ValueError(f"Unknown version: {version}")
            model.load_state_dict(sd, strict=False)
        return model










