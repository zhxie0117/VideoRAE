
# models/larp_ar.py
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import functional as F

import models
from models.embed import LabelEmbedder
from models.norm import RMSNorm

from .embed import get_1d_sincos_pos_embed_from_grid


def is_master():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32

    # === RQ === inner AR head 深度（1 或 2 即可）
    n_output_layer: int = 1

    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 101
    class_dropout_prob: float = 0.1
    model_type: str = 'class_cond'

    # vocab_size = 总 codebook size = num_codebooks * sub_vocab_size
    vocab_size: int = 16384
    num_codebooks: int = 1
    cls_token_num: int = 1

    max_batch_size: int = 32
    max_seq_len: int = 1024

    use_fixed_pe: bool = False
    frame_prediction: bool = False


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(v_out.dtype)
        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(self, x, input_pos=None, mask=None):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=True if mask is None else False,
            dropout_p=self.attn_dropout_p if self.training else 0
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.resid_dropout(self.wo(output))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path_p: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0. else nn.Identity()

    def forward(self, x, input_pos=None, mask=None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), input_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


# ---------------------------------------------------------------------------
# Multi-codebook token embedding (sum-based)
# ---------------------------------------------------------------------------
class TokenEmbedder(nn.Module):
    """
    Sum-based multi-codebook embedding (RQ-Transformer / MusicGen 风格).
    indices: [B, N, M] (or [B, N] when M==1).
    return:  [B, N, dim]
    """
    def __init__(self, sub_vocab_size, hidden_size, num_codebooks, extra=0):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(sub_vocab_size + extra, hidden_size)
            for _ in range(num_codebooks)
        ])

    def forward(self, indices):
        if indices.ndim == 2:
            assert self.num_codebooks == 1, \
                f'indices must be [B, N, M] when num_codebooks > 1'
            indices = indices.unsqueeze(-1)
        assert indices.shape[-1] == self.num_codebooks, \
            f'indices last dim {indices.shape[-1]} != num_codebooks {self.num_codebooks}'
        embs = [self.codebooks[i](indices[..., i]) for i in range(self.num_codebooks)]
        return torch.stack(embs, dim=0).sum(dim=0)


# ---------------------------------------------------------------------------
# Inner AR head (RQ-Transformer style)
# ---------------------------------------------------------------------------
class AutoRegressiveHead(nn.Module):
    """
    给定主 transformer 输出的 hidden state h_t，按码本维做一个小 AR：
        p(c¹_t | h_t)
        p(c²_t | h_t, c¹_t)
        p(c³_t | h_t, c¹_t, c²_t)
        ...
        p(c^M_t | h_t, c¹_t, ..., c^{M-1}_t)
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.sub_vocab_size = config.vocab_size // config.num_codebooks

        # 用 M-1 个 embedding（c^M 不会被作为输入）
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.sub_vocab_size, config.dim)
            for _ in range(self.num_codebooks - 1)
        ])

        self.layers = nn.ModuleList([
            TransformerBlock(config, drop_path_p=0.)
            for _ in range(config.n_output_layer)
        ])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.linear_head = nn.Linear(config.dim, self.sub_vocab_size, bias=False)

    def forward_train(self, base_tokens: torch.Tensor, targets: torch.Tensor):
        """
        base_tokens: [B, L, C]   主 transformer 在 N 个空间位置上的 hidden state
        targets:     [B, L, M]   每个空间位置的 M 个码本 GT
        return:      [B, L, M, V_sub]
        """
        B, L, C = base_tokens.shape
        M = self.num_codebooks

        # 把 (B, L) 展平：每个空间位置内部独立做 inner AR
        base = base_tokens.reshape(B * L, 1, C)             # [B*L, 1, C]

        if M > 1:
            tgt_in = targets.reshape(B * L, M)[:, :-1]      # [B*L, M-1]
            idx_embs = [self.codebooks[i](tgt_in[:, i]) for i in range(M - 1)]
            idx_embs = torch.stack(idx_embs, dim=1)         # [B*L, M-1, C]
            h = torch.cat([base, idx_embs], dim=1)          # [B*L, M, C]
        else:
            h = base                                         # [B*L, 1, C]

        # 因果小 transformer（mask=None -> SDPA is_causal=True）
        for layer in self.layers:
            h = layer(h, input_pos=None, mask=None)
        h = self.norm(h)
        logits = self.linear_head(h)                         # [B*L, M, V_sub]
        return logits.reshape(B, L, M, -1)

    def sample_inner(self, base_token: torch.Tensor, sample_fn):
        """
        base_token: [T, 1, C]   主 transformer 在 *某一个* 空间位置的 hidden state
                                T = bsz 或 2*bsz（cfg 时）
        sample_fn:  callable(logits[T, V_sub], m_idx) -> tokens[bsz]
                    （内部完成 cfg 合并 + temperature/topk/topp + multinomial）

        return: [bsz, M] 该位置生成的 M 个子码本 token
        """
        M = self.num_codebooks
        T = base_token.shape[0]
        h_seq = base_token                                   # [T, 1, C]
        generated = []

        for m in range(M):
            h = h_seq
            for layer in self.layers:
                h = layer(h, input_pos=None, mask=None)
            h = self.norm(h)
            logits_m = self.linear_head(h[:, -1])            # [T, V_sub]
            token_m = sample_fn(logits_m, m)                 # [bsz]
            generated.append(token_m)

            if m < M - 1:
                # 把采样的 token 回填到 inner 序列里；cfg 时两条 batch 用同一个 token
                if T > token_m.shape[0]:
                    token_full = torch.cat([token_m, token_m], dim=0)
                else:
                    token_full = token_m
                emb = self.codebooks[m](token_full).unsqueeze(1)   # [T, 1, C]
                h_seq = torch.cat([h_seq, emb], dim=1)

        return torch.stack(generated, dim=-1)                # [bsz, M]


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------
def _sample_token(logits, temperature, top_k, top_p):
    """
    logits: [B, V] -> tokens: [B]
    """
    logits = logits.float() / max(float(temperature), 1e-5)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        logits = torch.where(logits < v[..., -1:], torch.full_like(logits, float('-inf')), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LARP_AR(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # --- multi-codebook ---
        assert config.vocab_size % config.num_codebooks == 0, (
            f'vocab_size ({config.vocab_size}) must be divisible by '
            f'num_codebooks ({config.num_codebooks})'
        )
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.sub_vocab_size = config.vocab_size // self.num_codebooks

        self.n_layer = config.n_layer
        self.max_seq_length = config.max_seq_len    # N（token 位置数）
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.is_sampling = False
        self.frame_prediction = config.frame_prediction

        # --- conditioning ---
        if self.frame_prediction:
            self.cls_embedding = None
            extra = 1     # 给每个子码本一个 sep
        elif self.model_type == 'class_cond':
            self.cls_embedding = LabelEmbedder(
                config.num_classes, config.dim, config.class_dropout_prob
            )
            extra = 0
        else:
            raise Exception('please check model type')

        # --- token embedding (sum across codebooks) ---
        self.tok_embeddings = TokenEmbedder(
            self.sub_vocab_size, config.dim, self.num_codebooks, extra=extra
        )
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # --- main transformer ---
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = nn.ModuleList([
            TransformerBlock(config, dpr[i]) for i in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # --- output head ---
        # M=1: 直接在主 transformer hidden 上做线性投影（与旧版一致）
        # M>1: RQ-Transformer inner AR，建模同一 spatial 位置内子码本依赖
        self.use_inner_ar = config.num_codebooks > 1
        if self.use_inner_ar:
            self.output = AutoRegressiveHead(config)
        else:
            self.output = nn.Linear(config.dim, self.sub_vocab_size, bias=False)

        # --- 1D abs PE ---
        pe_len = config.max_seq_len + config.cls_token_num - 1
        if config.use_fixed_pe:
            self.register_buffer('abs_pe', torch.zeros(1, pe_len, config.dim))
            abs_pe = get_1d_sincos_pos_embed_from_grid(
                embed_dim=config.dim, pos=np.arange(pe_len)
            )
            self.abs_pe.copy_(torch.from_numpy(abs_pe).float().reshape_as(self.abs_pe))
            if is_master():
                print('[LARP_AR] Using fixed 1D abs PE')
        else:
            self.abs_pe = nn.Parameter(torch.randn(1, pe_len, config.dim) * 0.02)
            if is_master():
                print('[LARP_AR] Using learned 1D abs PE')

        self.initialize_weights()

    # ------------------------- init -------------------------
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero-init 输出层：初始 loss ≈ log(V_sub)，同时保证主 transformer 能收到梯度
        if self.use_inner_ar:
            nn.init.constant_(self.output.linear_head.weight, 0)
        else:
            nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @contextmanager
    def sampling(self):
        self.is_sampling = True
        try:
            yield
        finally:
            self.is_sampling = False

    # ------------------------- KV cache -------------------------
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        assert max_seq_length == self.max_seq_length + self.cls_token_num, \
            f'{max_seq_length} != {self.max_seq_length} + {self.cls_token_num}'
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        device = next(self.parameters()).device

        for b in self.layers:
            kv = KVCache(
                max_batch_size, max_seq_length,
                self.config.n_head, head_dim, dtype
            ).to(device)
            b.attention.kv_cache = kv

        causal_mask = torch.tril(
            torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1)

    def reset_caches(self):
        for b in self.layers:
            b.attention.kv_cache = None

    # ------------------------- main transformer forward -------------------------
    def forward_main(self, idx, cond_idx, input_pos=None, mask=None):
        """
        Run main transformer body, return hidden state h: [B, T, dim]
        """
        if idx is not None and cond_idx is not None:
            # training / naive inference: cond + tokens 一起送
            if self.frame_prediction:
                cond_embeddings = self.tok_embeddings(cond_idx)
                assert cond_embeddings.shape[1] == self.cls_token_num
            else:
                cond_embeddings = self.cls_embedding(
                    cond_idx, train=self.training
                ).unsqueeze(1)[:, :self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat([cond_embeddings, token_embeddings], dim=1)
            h = self.tok_dropout(token_embeddings)
        else:
            if cond_idx is not None:
                # prefill 阶段
                if self.frame_prediction:
                    token_embeddings = self.tok_embeddings(cond_idx)
                    assert token_embeddings.shape[1] == self.cls_token_num
                else:
                    token_embeddings = self.cls_embedding(
                        cond_idx, train=self.training
                    ).unsqueeze(1)[:, :self.cls_token_num]
            else:
                # decode 阶段，单步
                token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)

        # 1D abs PE
        if self.is_sampling:
            h = h + self.abs_pe[:, input_pos]
        else:
            h = h + self.abs_pe[:, :h.shape[1]]

        for layer in self.layers:
            h = layer(h, input_pos, mask)
        h = self.norm(h)
        return h

    def _compute_logits(self, h_for_logits: torch.Tensor, targets: torch.Tensor):
        """Return logits [B, T, M, V_sub] for training."""
        if self.use_inner_ar:
            return self.output.forward_train(h_for_logits, targets)
        return self.output(h_for_logits).unsqueeze(2)

    def _sample_logits_with_cfg(
        self, logits, step, bsz, cfg_enabled, cfg_scale, cfg_interval,
        temperature, top_k, top_p,
    ):
        """logits: [T, V_sub] -> tokens: [bsz]"""
        if cfg_enabled:
            logits_cond = logits[:bsz]
            logits_uncond = logits[bsz:]
            apply_cfg = (
                cfg_interval is None or cfg_interval < 0 or step < cfg_interval
            )
            if apply_cfg:
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond
        return _sample_token(logits, temperature, top_k, top_p)

    # ------------------------- forward (训练/算 loss) -------------------------
    def forward(
        self,
        idx: Optional[torch.Tensor],         # [B, N, M] (训练) / [B, 1, M] (单步) / None
        cond_idx: Optional[torch.Tensor],    # [B] 类别 或 [B, cls_token_num, M]
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,    # [B, N, M]
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        h = self.forward_main(idx, cond_idx, input_pos, mask)

        if targets is None:
            # 没给 target：返回主干 hidden state，配合 sample() 使用
            return h, None

        # 训练阶段：通过 inner AR head 拿 logits
        if targets.ndim == 2:
            assert self.num_codebooks == 1
            targets = targets.unsqueeze(-1)

        # 训练时把 cond 部分裁掉（cls_token_num=1 时是 no-op）
        h_for_logits = h[:, self.cls_token_num - 1:].contiguous()

        logits = self._compute_logits(h_for_logits, targets)   # [B, T, M, V_sub]
        B, T, M, V = logits.shape
        assert targets.shape == (B, T, M), \
            f'targets shape {targets.shape} != ({B}, {T}, {M})'

        if valid is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction='none'
            ).view(B, T, M).mean(dim=-1)                 # 在 M 上求平均，得 [B, T]
            valid_all = valid[:, None].expand(-1, T)
            denom = valid_all.sum().clamp(min=1)
            loss = (ce * valid_all).sum() / denom
        else:
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

        return logits, loss

    # ------------------------- sampling -------------------------
    @torch.inference_mode()
    def sample(
        self, c,
        cfg_scale=1.0,
        cfg_interval=-1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        seq_length=None,
    ):
        """
        多码本 AR 采样（外层时间维 + 内层码本维双 AR）。
        return: [B, N, M] long tokens
        """
        seq_length = self.max_seq_length if seq_length is None else seq_length
        bsz = c.shape[0]
        device = c.device
        dtype = self.dtype

        cfg_enabled = cfg_scale is not None and cfg_scale > 1.0

        if cfg_enabled:
            cond_null = torch.full_like(c, self.num_classes)
            c_in = torch.cat([c, cond_null], dim=0)
        else:
            c_in = c

        T = c_in.shape[0]
        max_seq_length = seq_length + self.cls_token_num
        generated = torch.zeros(
            bsz, seq_length, self.num_codebooks, dtype=torch.long, device=device
        )

        def make_sample_fn(step):
            """构造 inner head 用的采样回调，封装 cfg + temp/topk/topp。"""
            def sample_fn(logits, m_idx):
                # logits: [T, V_sub]
                if cfg_enabled:
                    logits_cond = logits[:bsz]
                    logits_uncond = logits[bsz:]
                    apply_cfg = (
                        cfg_interval is None or cfg_interval < 0 or step < cfg_interval
                    )
                    if apply_cfg:
                        l = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                    else:
                        l = logits_cond
                else:
                    l = logits
                return _sample_token(l, temperature, top_k, top_p)
            return sample_fn

        with self.sampling():
            self.setup_caches(T, max_seq_length, dtype=dtype)

            if self.use_inner_ar:
                # ---- prefill (cls / cond) ----
                input_pos = torch.arange(0, self.cls_token_num, device=device)
                h = self.forward_main(idx=None, cond_idx=c_in, input_pos=input_pos)
                base_token = h[:, -1:].contiguous()                    # [T, 1, dim]

                next_tokens = self.output.sample_inner(
                    base_token, make_sample_fn(0)
                )                                                        # [bsz, M]
                generated[:, 0] = next_tokens

                # ---- decode loop ----
                for i in range(1, seq_length):
                    input_pos = torch.tensor(
                        [self.cls_token_num + i - 1], device=device
                    )

                    if cfg_enabled:
                        idx_in = torch.cat([next_tokens, next_tokens], dim=0).unsqueeze(1)
                    else:
                        idx_in = next_tokens.unsqueeze(1)               # [T, 1, M]

                    h = self.forward_main(idx=idx_in, cond_idx=None, input_pos=input_pos)
                    base_token = h[:, -1:].contiguous()                  # [T, 1, dim]

                    next_tokens = self.output.sample_inner(
                        base_token, make_sample_fn(i)
                    )                                                     # [bsz, M]
                    generated[:, i] = next_tokens
            else:
                # ---- single-codebook: 直接在主 transformer hidden 上预测 ----
                input_pos = torch.arange(0, self.cls_token_num, device=device)
                h = self.forward_main(idx=None, cond_idx=c_in, input_pos=input_pos)
                logits = self.output(h[:, -1]).float()
                next_tokens = self._sample_logits_with_cfg(
                    logits, 0, bsz, cfg_enabled, cfg_scale, cfg_interval,
                    temperature, top_k, top_p,
                ).unsqueeze(-1)
                generated[:, 0] = next_tokens

                for i in range(1, seq_length):
                    input_pos = torch.tensor(
                        [self.cls_token_num + i - 1], device=device
                    )
                    if cfg_enabled:
                        idx_in = torch.cat([next_tokens, next_tokens], dim=0)
                    else:
                        idx_in = next_tokens
                    h = self.forward_main(
                        idx=idx_in, cond_idx=None, input_pos=input_pos
                    )
                    logits = self.output(h[:, -1]).float()
                    next_tokens = self._sample_logits_with_cfg(
                        logits, i, bsz, cfg_enabled, cfg_scale, cfg_interval,
                        temperature, top_k, top_p,
                    ).unsqueeze(-1)
                    generated[:, i] = next_tokens

            self.reset_caches()

        return generated

    # ------------------------- ckpt -------------------------
    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        import gc
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location='cpu')
        else:
            assert isinstance(ckpt, dict)
        model_spec = ckpt['model']
        del ckpt
        gc.collect()
        model = models.make(model_spec, load_sd=load_state_dict)
        return model


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def LLAMA_ABS_XXXL(**kwargs):
    return LARP_AR(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))

def LLAMA_ABS_XXL(**kwargs):
    return LARP_AR(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))

def LLAMA_ABS_XL(**kwargs):
    return LARP_AR(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))

def LLAMA_ABS_LP(**kwargs):
    return LARP_AR(ModelArgs(n_layer=30, n_head=20, dim=1280, **kwargs))

def LLAMA_ABS_L(**kwargs):
    return LARP_AR(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))

def LLAMA_ABS_B(**kwargs):
    return LARP_AR(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))

def LLAMA_ABS_S(**kwargs):
    return LARP_AR(ModelArgs(n_layer=12, n_head=6, dim=384, **kwargs))


larp_ar_models = {
    'llama-abs-S':    LLAMA_ABS_S,
    'llama-abs-B':    LLAMA_ABS_B,
    'llama-abs-L':    LLAMA_ABS_L,
    'llama-abs-LP':   LLAMA_ABS_LP,
    'llama-abs-XL':   LLAMA_ABS_XL,
    'llama-abs-XXL':  LLAMA_ABS_XXL,
    'llama-abs-XXXL': LLAMA_ABS_XXXL,
}

models.models.update(larp_ar_models)