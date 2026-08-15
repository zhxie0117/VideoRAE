import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import models
from models import register


# ---------------------------------------------------------------------------
# 通用工具: 与 LARP 中保持一致
# ---------------------------------------------------------------------------
def entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """Calculates the entropy loss (与 LARP 一致)。"""
    flat_affinity = affinity.view(-1, affinity.shape[-1])
    flat_affinity = flat_affinity / temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, dim=-1)
        onehots = F.one_hot(codes, num_classes=flat_affinity.shape[-1]).to(flat_affinity.dtype)
        onehots = probs - (probs - onehots).detach()
        target_probs = onehots
    else:
        raise ValueError(f"Entropy loss {loss_type} not supported")

    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss, sample_entropy, avg_entropy


class AbstractRegularizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError


# ===========================================================================
# 1) Multi-Codebook Quantizer (MCQ)
# ===========================================================================
@register("mcq")
class MultiCodebookQuantizer(AbstractRegularizer):
    """
    多码本量化 (Product Quantization 风格)。
    把每个 token 的 D 维特征沿通道切成 M 份 (每份 sub_dim = D / M),
    每一份独立查一个 sub-codebook (大小 = codebook_size / M)。

    输入:  z [B, N, D]
    输出 dict 字段与 LARP `SimpleVectorQuantizer` 对齐, 其中
        bottleneck_rep: [B, N, M] 的 long index
    """

    def __init__(
        self,
        dim,                                 # 由 Bottleneck 注入
        token_nums,                          # 由 Bottleneck 注入
        codebook_size,
        num_codebooks=4,
        commitment_loss_weight=0.25,
        codebook_loss_weight=1.0,
        entropy_loss_weight=0.0,
        entropy_loss_temperature=0.01,
        l2_normalized=False,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_codebooks == 0, \
            f"dim ({dim}) must be divisible by num_codebooks ({num_codebooks})"
        assert codebook_size % num_codebooks == 0, \
            f"codebook_size ({codebook_size}) must be divisible by num_codebooks ({num_codebooks})"

        self.dim = dim
        self.token_nums = token_nums
        self.num_codebooks = num_codebooks
        self.sub_dim = dim // num_codebooks
        self.sub_codebook_size = codebook_size // num_codebooks
        self.codebook_size = codebook_size  # 仅作记录, 等于 sub_K * M

        self.beta = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature

        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature

        if self.stochastic:
            if stochastic_temperature > 0:
                self.stochastic_temperature_inv = 1.0 / stochastic_temperature
            else:
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        # M 个子码本
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.sub_codebook_size, self.sub_dim)
            for _ in range(num_codebooks)
        ])
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight)

    # ----------------- 与 LARP 对齐的 API -----------------
    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1.0 / temperature

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self, codebook_idx):
        emb = self.embeddings[codebook_idx].weight
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        assert emb.dtype == torch.float32, f"Embedding dtype is {emb.dtype}, expected float32"
        return emb

    # ----------------- 内部: 单个子码本量化 -----------------
    def _quantize_single(self, z_sub, codebook_idx):
        """
        z_sub: [B, N, sub_dim] (float32)
        return: dict
        """
        if self.l2_normalized:
            z_sub = F.normalize(z_sub, p=2, dim=-1)

        emb = self.get_emb(codebook_idx)               # [K, sub_dim]
        z_flat = rearrange(z_sub, 'b n d -> (b n) d')  # [BN, sub_dim]

        if self.stochastic:
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flat, emb)            # [BN, K]
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
            # 为 entropy loss 提供 logits (越大 = 越近), 与 LARP 用 -d 对齐
            logits_for_entropy = cos_sim
        else:
            d = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True)
                + torch.sum(emb ** 2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flat, rearrange(emb, "n d -> d n"))
            )
            q_indices = torch.argmin(d, dim=1)
            logits_for_entropy = -d

        quantized = F.embedding(q_indices, emb).view(z_sub.shape)       # [B, N, sub_dim]

        # 注意: 与 LARP 严格对齐的损失分解
        loss_commit   = ((quantized.detach() - z_sub) ** 2).mean()  # encoder pulls toward codebook
        loss_codebook = ((quantized - z_sub.detach()) ** 2).mean()  # codebook pulls toward encoder

        if self.entropy_loss_weight > 0:
            loss_ent, samp_ent, avg_ent = entropy_loss(
                logits_for_entropy, temperature=self.entropy_loss_temperature)
        else:
            loss_ent = z_sub.new_zeros(())
            samp_ent = z_sub.new_zeros(())
            avg_ent  = z_sub.new_zeros(())

        # straight-through
        quantized_ste = z_sub + (quantized - z_sub).detach()
        q_indices = q_indices.view(z_sub.shape[0], z_sub.shape[1])      # [B, N]

        return {
            'quantized':   quantized_ste,
            'indices':     q_indices,
            'z_in':        z_sub,            # 已 l2 normalize 过 (若启用)
            'loss_commit': loss_commit,
            'loss_codebook': loss_codebook,
            'loss_entropy': loss_ent,
            'sample_entropy': samp_ent,
            'avg_entropy':  avg_ent,
        }

    # ----------------- 主入口 -----------------
    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, z):
        z = z.float()
        assert z.dim() == 3, "Input shape must be (batch, n_tokens, e_dim)"

        chunks = torch.chunk(z, self.num_codebooks, dim=-1)
        sub_outs = [self._quantize_single(c, i) for i, c in enumerate(chunks)]

        quantized = torch.cat([o['quantized'] for o in sub_outs], dim=-1)         # [B, N, D]
        indices   = torch.stack([o['indices']   for o in sub_outs], dim=-1)       # [B, N, M]
        z_in      = torch.cat([o['z_in']      for o in sub_outs], dim=-1)         # [B, N, D]

        loss_commit    = torch.stack([o['loss_commit']    for o in sub_outs]).mean()
        loss_codebook  = torch.stack([o['loss_codebook']  for o in sub_outs]).mean()
        loss_ent       = torch.stack([o['loss_entropy']   for o in sub_outs]).mean()
        sample_ent     = torch.stack([o['sample_entropy'] for o in sub_outs]).mean()
        avg_ent        = torch.stack([o['avg_entropy']    for o in sub_outs]).mean()

        loss_q = (self.beta * loss_commit
                  + self.codebook_loss_weight * loss_codebook
                  + self.entropy_loss_weight * loss_ent)

        return {
            'unregularized_z':   z_in,
            'regularized_z':     quantized,
            'bottleneck_rep':    indices,            # [B, N, M]
            'loss_q':            loss_q,
            'loss_commit':       loss_commit,
            'loss_codebook':     loss_codebook,
            'loss_entropy':      loss_ent,
            'per_sample_entropy': sample_ent,
            'codebook_entropy':  avg_ent,
        }

    # ----------------- 解码 -----------------
    def get_codebook_entry(self, indices, shape=None):
        """
        indices: 任意形状, 但最后一维必须是 num_codebooks。
                  常见形状 [B, N, M]。
        """
        assert indices.shape[-1] == self.num_codebooks, \
            f"Last dim of indices must be num_codebooks ({self.num_codebooks}), got {indices.shape}"

        sub_zqs = []
        for i in range(self.num_codebooks):
            emb = self.get_emb(i)
            sub_zqs.append(F.embedding(indices[..., i], emb))   # [..., sub_dim]
        z_q = torch.cat(sub_zqs, dim=-1)                         # [..., D]

        if shape is not None:
            z_q = z_q.reshape(shape)
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)


# ===========================================================================
# 2) SimVQ + Multi-Codebook Quantizer (SimVQ-MCQ)
# ===========================================================================
@register("simvq_mcq")
class SimVQMultiCodebookQuantizer(AbstractRegularizer):
    """
    在 MCQ 的基础上, 每个子码本采用 SimVQ 风格:
        - 真正的 codebook 表是 frozen 的高斯随机表 C ∈ R^{K × d_sub}
        - 学一个可训练的线性投影 W_i: R^{d_sub} -> R^{d_sub}
        - 实际查询的码本 = W_i(C)
    其余行为与 MCQ 完全一致, 包含 bottleneck_rep: [B, N, M]。
    """

    def __init__(
        self,
        dim,
        token_nums,
        codebook_size,
        num_codebooks=4,
        commitment_loss_weight=0.25,
        codebook_loss_weight=1.0,
        entropy_loss_weight=0.0,
        entropy_loss_temperature=0.01,
        l2_normalized=False,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_codebooks == 0, \
            f"dim ({dim}) must be divisible by num_codebooks ({num_codebooks})"
        assert codebook_size % num_codebooks == 0, \
            f"codebook_size ({codebook_size}) must be divisible by num_codebooks ({num_codebooks})"

        self.dim = dim
        self.token_nums = token_nums
        self.num_codebooks = num_codebooks
        self.sub_dim = dim // num_codebooks
        self.sub_codebook_size = codebook_size // num_codebooks
        self.codebook_size = codebook_size

        self.beta = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature

        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature

        if self.stochastic:
            if stochastic_temperature > 0:
                self.stochastic_temperature_inv = 1.0 / stochastic_temperature
            else:
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        # 1) frozen base codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.sub_codebook_size, self.sub_dim)
            for _ in range(num_codebooks)
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=self.sub_dim ** -0.5)
            for p in emb.parameters():
                p.requires_grad = False

        # 2) learnable linear projections
        self.projections = nn.ModuleList([
            nn.Linear(self.sub_dim, self.sub_dim)
            for _ in range(num_codebooks)
        ])

    # ----------------- API -----------------
    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1.0 / temperature

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self, codebook_idx):
        # 强制 float32, 与 LARP 一致
        base = self.embeddings[codebook_idx].weight.float()
        # projections 也在 float32 下执行
        proj = self.projections[codebook_idx].float()
        emb = proj(base)
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    # ----------------- 内部: 单个子码本量化 -----------------
    def _quantize_single(self, z_sub, codebook_idx):
        if self.l2_normalized:
            z_sub = F.normalize(z_sub, p=2, dim=-1)

        emb = self.get_emb(codebook_idx)
        z_flat = rearrange(z_sub, 'b n d -> (b n) d')

        if self.stochastic:
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flat, emb)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
            logits_for_entropy = cos_sim
        else:
            d = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True)
                + torch.sum(emb ** 2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flat, rearrange(emb, "n d -> d n"))
            )
            q_indices = torch.argmin(d, dim=1)
            logits_for_entropy = -d

        quantized = F.embedding(q_indices, emb).view(z_sub.shape)

        loss_commit   = ((quantized.detach() - z_sub) ** 2).mean()
        loss_codebook = ((quantized - z_sub.detach()) ** 2).mean()

        if self.entropy_loss_weight > 0:
            loss_ent, samp_ent, avg_ent = entropy_loss(
                logits_for_entropy, temperature=self.entropy_loss_temperature)
        else:
            loss_ent = z_sub.new_zeros(())
            samp_ent = z_sub.new_zeros(())
            avg_ent  = z_sub.new_zeros(())

        quantized_ste = z_sub + (quantized - z_sub).detach()
        q_indices = q_indices.view(z_sub.shape[0], z_sub.shape[1])

        return {
            'quantized':     quantized_ste,
            'indices':       q_indices,
            'z_in':          z_sub,
            'loss_commit':   loss_commit,
            'loss_codebook': loss_codebook,
            'loss_entropy':  loss_ent,
            'sample_entropy': samp_ent,
            'avg_entropy':   avg_ent,
        }

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, z):
        z = z.float()
        assert z.dim() == 3, "Input shape must be (batch, n_tokens, e_dim)"

        chunks = torch.chunk(z, self.num_codebooks, dim=-1)
        sub_outs = [self._quantize_single(c, i) for i, c in enumerate(chunks)]

        quantized = torch.cat([o['quantized'] for o in sub_outs], dim=-1)
        indices   = torch.stack([o['indices']   for o in sub_outs], dim=-1)
        z_in      = torch.cat([o['z_in']      for o in sub_outs], dim=-1)

        loss_commit    = torch.stack([o['loss_commit']    for o in sub_outs]).mean()
        loss_codebook  = torch.stack([o['loss_codebook']  for o in sub_outs]).mean()
        loss_ent       = torch.stack([o['loss_entropy']   for o in sub_outs]).mean()
        sample_ent     = torch.stack([o['sample_entropy'] for o in sub_outs]).mean()
        avg_ent        = torch.stack([o['avg_entropy']    for o in sub_outs]).mean()

        loss_q = (self.beta * loss_commit
                  + self.codebook_loss_weight * loss_codebook
                  + self.entropy_loss_weight * loss_ent)

        return {
            'unregularized_z':   z_in,
            'regularized_z':     quantized,
            'bottleneck_rep':    indices,
            'loss_q':            loss_q,
            'loss_commit':       loss_commit,
            'loss_codebook':     loss_codebook,
            'loss_entropy':      loss_ent,
            'per_sample_entropy': sample_ent,
            'codebook_entropy':  avg_ent,
        }

    def get_codebook_entry(self, indices, shape=None):
        assert indices.shape[-1] == self.num_codebooks, \
            f"Last dim of indices must be num_codebooks ({self.num_codebooks}), got {indices.shape}"
        sub_zqs = []
        for i in range(self.num_codebooks):
            emb = self.get_emb(i)
            sub_zqs.append(F.embedding(indices[..., i], emb))
        z_q = torch.cat(sub_zqs, dim=-1)
        if shape is not None:
            z_q = z_q.reshape(shape)
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)





