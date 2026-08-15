
import os
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.model_sem.base.blocks import TokenizerEncoder1D,Decoder_sem
from models import register
from vjepa2.src.models.vision_transformer import vit_large_rope
from vq import SimVQMultiCodebookManager
# =========================================================
# 1) 工具：下载（可选）
# =========================================================
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


# =========================================================
# 2) VJEPA2: 权重加载
# =========================================================
def load_pretrained_vjepa2_pt_weights(model, pretrained_weights_path: str):
    """
    兼容你贴的 VJEPA2 demo 权重格式：
      ckpt["encoder"]，并且 key 可能带 module./backbone. 前缀
    """
    ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
    if "encoder" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'encoder'. Keys={list(ckpt.keys())[:20]}")
    sd = ckpt["encoder"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[VJEPA2] Loaded encoder weights from {pretrained_weights_path} with msg: {msg}")







class FeatureAligner21(nn.Module):
    def __init__(self, student_dim=512, teacher_dim=1024, global_loss_weight=0.2): 
        """
        Args:
            student_dim: Student 特征维度 (如 512)
            teacher_dim: Teacher 特征维度 (如 1024)
            global_loss_weight: Global Sim Loss 在总 Loss 中的占比权重
        """
        super().__init__()
        self.global_loss_weight = global_loss_weight
        
        # 把 Student (512) 映射到 Teacher 的维度 (1024)
        self.proj = nn.Sequential(
            nn.Linear(student_dim, student_dim * 2),
            nn.GELU(),
            nn.Linear(student_dim * 2, teacher_dim)
        )

    def forward(self, student_feat, teacher_feat, teacher_grid, student_grid):
        """
        Args:
            student_feat: [B, N_s, student_dim] (如 [B, 1024, 512], N_s = 4*16*16)
            teacher_feat: [B, N_t, teacher_dim] (如 [B, 4608, 1024], N_t = 8*24*24, 可能含 CLS)
            teacher_grid: (tt=8, ht=24, wt=24)
            student_grid: (st=4, sh=16, sw=16)
        """
        tt, ht, wt = teacher_grid
        st, sh, sw = student_grid
        B = student_feat.shape[0]

        # 去掉可能存在的 CLS token
        if teacher_feat.shape[1] == tt * ht * wt + 1:
            teacher_feat = teacher_feat[:, 1:, :]

        # Student 维度投影到 teacher_dim
        student_feat_proj = self.proj(student_feat)

        # reshape 成 3D 体素特征
        # teacher: [B, D, tt, ht, wt]
        teacher_3d = teacher_feat.transpose(1, 2).view(B, -1, tt, ht, wt).contiguous()
        # student: [B, D, st, sh, sw]
        student_3d = student_feat_proj.transpose(1, 2).view(B, -1, st, sh, sw).contiguous()

        # ---------------------------------------------------------
        # 4. 对齐到 Teacher 的时空尺寸 (上采样 Student 到 Teacher)
        # ---------------------------------------------------------
        target_t, target_h, target_w = tt, ht, wt  # 8, 24, 24

        # Teacher 保持不变
        teacher_aligned = teacher_3d

        # Student 上采样: 时间 4->8, 空间 16x16 -> 24x24 (trilinear 插值)
        student_aligned = F.interpolate(
            student_3d,
            size=(target_t, target_h, target_w),
            mode='trilinear',
            align_corners=False
        )

        # ---------------------------------------------------------
        # 5. 展平并计算 Local Cosine Similarity (Patch-level)
        # ---------------------------------------------------------
        # [B, D, 8, 24, 24] -> [B, D, 4608] -> [B, 4608, D]
        teacher_local_flat = teacher_aligned.flatten(2).transpose(1, 2)
        student_local_flat = student_aligned.flatten(2).transpose(1, 2)

        # L2 归一化
        teacher_local_norm = F.normalize(teacher_local_flat, dim=-1, eps=1e-8)
        student_local_norm = F.normalize(student_local_flat, dim=-1, eps=1e-8)

        # 沿着特征维度计算点积 -> 余弦相似度 [B, N]
        local_cos_sim = (student_local_norm * teacher_local_norm).sum(dim=-1)

        # 取负号的均值作为 Local Loss
        local_loss = (-local_cos_sim).mean()

        # ---------------------------------------------------------
        # 6. 计算 Global Cosine Similarity (Video-level)
        # ---------------------------------------------------------
        # 在空间和时间维度上做全局平均池化
        # [B, D, 8, 24, 24] -> [B, D, 1, 1, 1] -> [B, D]
        teacher_global = F.adaptive_avg_pool3d(teacher_aligned, (1, 1, 1)).flatten(1)
        student_global = F.adaptive_avg_pool3d(student_aligned, (1, 1, 1)).flatten(1)

        # L2 归一化
        teacher_global_norm = F.normalize(teacher_global, dim=-1, eps=1e-8)
        student_global_norm = F.normalize(student_global, dim=-1, eps=1e-8)

        # 沿着特征维度计算点积 -> 余弦相似度 [B]
        global_cos_sim = (student_global_norm * teacher_global_norm).sum(dim=-1)
        # 取负号的均值作为 Global Loss
        global_loss = (-global_cos_sim).mean()

        total_loss = local_loss + self.global_loss_weight * global_loss
        loss_dict = {
            "repa_loss_total": total_loss.item(),
            "repa_loss_local": local_loss.item(),
            "repa_loss_global": global_loss.item()
        }
        return total_loss, loss_dict





@register("autoencoder_vfm_add_simple_repa")
class AutoEncoder_repa(nn.Module):
    def __init__(
        self,
        bottleneck,
        prior_model,
        num_latent_tokens=1024,
        input_size=256,
        bypass_tokenizer=True,
        use_vjepa_loss=True,
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

        self.vfm_grid = (8, 16, 16)  # student tokens grid: t,h,w
        token_size = 512             # codebook 维度

        self.tokenizer_encoder = TokenizerEncoder1D(
            model_size='small',
            in_channels=1024,
            out_channels=token_size,
            in_tokens=2048,
            out_tokens=num_latent_tokens,
            in_grid=self.vfm_grid,
        )
        
        self.quantize = SimVQMultiCodebookManager(
            num_codebooks=4,             
            codebook_size=4096,          
            codebook_dim=token_size,     
            beta=0.25,                   
            use_per_codebook_size=True
        )

        self.decoder = Decoder_sem(
            model_size='base',
            patch_size=[4, 16, 16],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
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

        # 【核心修改】初始化 FeatureAligner21
        # teacher_dim 在 _init_vjepa2_teacher 中获取
        self.feature_aligner = FeatureAligner21(
            student_dim=self.decoder.width,    # Decoder 内部特征维度 (base一般为768)
            teacher_dim=self.teacher_dim,      # VJEPA 特征维度 (vitl一般为1024)
            global_loss_weight=0.2
        )

    def _init_vjepa2_teacher(self):
        if (self.vjepa2_encoder_ckpt is None) or (not os.path.exists(self.vjepa2_encoder_ckpt)):
            print(f"ERROR: vjepa2_encoder_ckpt not found: {self.vjepa2_encoder_ckpt}")
            self.use_vjepa_loss = False
            self.teacher_dim = 1024 # 兜底
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
            
        self.teacher_dim = teacher_dim # 保存下来供给 Aligner 使用
        print(f"[VJEPA2] Teacher loaded. embed_dim={teacher_dim}, img_size={self.vjepa2_img_size}, num_frames={self.vjepa2_num_frames}")
        self.prior_model = None

    def preprocess_for_teacher(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
        x_vjepa = (x - mean) / std
        return x_vjepa

    def _get_teacher_feats_list(self, x: torch.Tensor):
        """返回 VJEPA 的多层特征列表"""
        assert self.teacher_model is not None, "Teacher model is required for encoding."
        if next(self.teacher_model.parameters()).device != x.device:
            self.teacher_model.to(x.device)
            
        target_dtype = next(self.teacher_model.parameters()).dtype
        feats_list = self.teacher_model(x.to(dtype=target_dtype))
        return feats_list

    @torch.no_grad()
    def encode(self, x, **kwargs):
        x_vjepa = self.preprocess_for_teacher(x)
        feats_list = self._get_teacher_feats_list(x_vjepa)
        vfm_feats = sum(feats_list).float()                 # [B, 2048, D_teacher]
        latent = self.tokenizer_encoder(vfm_feats)          # [B, 1024, token_size]
        latent_q, q_dict = self.quantize(latent)
        return latent_q, q_dict
    
    def decode(self, x_q):
        # 仅返回预测视频，抛弃 inner_feat，保持原 API 兼容
        pred_video, _ = self.decoder(x_q)                   
        return pred_video
    
    def forward(self, data, **kwargs):
        x = data
        x_vjepa = self.preprocess_for_teacher(x)
        
        # 1. 提取教师特征
        with torch.no_grad():
            feats_list = self._get_teacher_feats_list(x_vjepa)
            vfm_feats = sum(feats_list).float()                 # 供 Encoder 使用
            target_teacher_feat = feats_list[-1].float()        # 提取最后一层作为 REPA target
            
        # 2. Encoder 降维
        latent = self.tokenizer_encoder(vfm_feats)            
        
        # 3. 量化（返回量化结果和损失字典）
        latent_q, q_dict = self.quantize(latent)              
        
        # 4. Decoder 解码 (获取视频 AND 第3层内部特征)
        pred_video, inner_feat = self.decoder(latent_q)      
        
        # 5. 【核心修改】计算 REPA Loss
        # 动态计算 VJEPA teacher grid (例如 16帧/tubelet_size2 = 8, 256/16 = 16)
        tt = self.vjepa2_num_frames // self.vjepa2_tubelet_size
        ht = self.vjepa2_img_size // self.vjepa2_patch_size
        wt = self.vjepa2_img_size // self.vjepa2_patch_size
        teacher_grid = (tt, ht, wt)
        
        # 获取 Student grid (Decoder 里的 self.grid)
        student_grid = tuple(self.decoder.grid)
        
        # 输入 Feature Aligner
        repa_loss, repa_loss_dict = self.feature_aligner(
            student_feat=inner_feat, 
            teacher_feat=target_teacher_feat, 
            teacher_grid=teacher_grid, 
            student_grid=student_grid
        )

        # 6. 组装返回结果
        return_dict = {
            "pred_frames": pred_video.contiguous(),
            "repa_loss": repa_loss  # 总 REPA Loss
        }
        
        # 注入量化 Loss
        return_dict.update({k: v for k, v in q_dict.items()})
        # 注入 REPA 具体指标 (repa_loss_total, repa_loss_local, repa_loss_global)
        return_dict.update(repa_loss_dict)
        
        return return_dict
    
    # =====================================================================
    # AR generation 相关接口
    # =====================================================================
    @property
    def bottleneck_token_num(self):
        return self.tokenizer_encoder.out_tokens  # 1024

    @property
    def codebook_size(self):
        return self.quantize.num_codebooks * self.quantize.sub_n_e

    @torch.no_grad()
    def encode_tokens(self, x):
        """
        Encode video to discrete token indices for AR training.
        x: [B, 3, T, H, W]
        Returns: dict with 'bottleneck_rep': [B, N, M] indices
        """
        x_vjepa = self.preprocess_for_teacher(x)
        feats_list = self._get_teacher_feats_list(x_vjepa)
        vfm_feats = sum(feats_list).float()
        latent = self.tokenizer_encoder(vfm_feats)
        indices = self.quantize.encode_to_indices(latent)
        return {'bottleneck_rep': indices}

    def decode_from_bottleneck(self, indices):
        """
        Decode from discrete token indices to video.
        indices: [B, N, M]
        Returns: [B, C, T, H, W]
        """
        z_q = self.quantize.decode_from_indices(indices)
        pred_video, _ = self.decoder(z_q)
        return pred_video

    def set_vq_eval_deterministic(self, flag):
        pass

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








