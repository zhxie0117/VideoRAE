import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from einops import einsum, rearrange, reduce

"""
References:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

# def apply_rotary_emb(x, freqs_cis):
#     x = x.contiguous()
#     with torch.autocast(x.device.type, enabled=False):
#         x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
#         freqs_cis = freqs_cis.unsqueeze(-2) # unsqueeze head dim -> [..., H, D]
#         x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(-2)

#     return x_out.type_as(x)
def apply_rotary_emb(x, freqs_cis):
    # 不需要强制 contiguous 了，省掉一次内存拷贝
    with torch.autocast(x.device.type, enabled=False):
        # 1. 转换类型并重塑，拆分出“实部”和“虚部”
        x_float = x.float()
        x_shaped = x_float.reshape(*x_float.shape[:-1], -1, 2)
        x_real = x_shaped[..., 0]
        x_imag = x_shaped[..., 1]

        # 2. 对 freqs_cis 进行形状对齐并提取实部和虚部
        freqs_cis = freqs_cis.unsqueeze(-2) # 原有的广播逻辑: unsqueeze head dim
        f_real = freqs_cis.real
        f_imag = freqs_cis.imag

        # 3. 按照复数乘法公式计算: (A + Bi) * (C + Di) = (AC - BD) + (AD + BC)i
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real

        # 4. 重新拼接成对，并展平最后两个维度
        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)

    return x_out.type_as(x)

def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, freqs_dtype=torch.float64): 
    assert dim % 2 == 0

    if type(pos) is int:
        pos = torch.arange(pos)

    start = 1.0
    end = theta

    freqs = theta ** torch.linspace(
        math.log(start, theta), # 0.0?
        math.log(end, theta), # 1.0?
        dim//2,
        device=pos.device,
        dtype=freqs_dtype,
    )
    freqs = freqs * math.pi / 2.0
    freqs = freqs * pos.unsqueeze(-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

        
def get_grid(in_grid, in_tokens):
    frames, height, width = in_grid
    seq_len = math.prod(in_grid) + in_tokens

    # Create position IDs -> [L, 3], all zeros. 3 = [frames, height, width]. Tokens are packed into THW dims like orig m-rope.
    position_ids = torch.zeros(seq_len, len(in_grid), dtype=torch.int64)
    position_ids[:in_tokens] = torch.arange(in_tokens, dtype=torch.int64).unsqueeze(-1) # assign to THW dims.

    # add THW position ids
    position_ids[in_tokens:, 0] = ( # frames
        torch.arange(frames, dtype=torch.int64)
        .view(-1, 1, 1)
        .repeat(1, height, width)
        .flatten()
    )

    position_ids[in_tokens:, 1] = ( # height
        torch.arange(height, dtype=torch.int64)
        .view(1, -1, 1)
        .repeat(frames, 1, width)
        .flatten()
    )

    position_ids[in_tokens:, 2] = ( # width
        torch.arange(width, dtype=torch.int64)
        .view(1, 1, -1)
        .repeat(frames, height, 1)
        .flatten()
    )

    position_ids[in_tokens:] += in_tokens # offset THW to increment from 1D enc
    return position_ids


def interleave_freqs(freqs):
    dim = sum([f.shape[-1] for f in freqs]) # // 2 # 32
    out = torch.zeros(*freqs[0].shape[:-1], dim, device=freqs[0].device, dtype=freqs[0].dtype)
    freqs = sorted(freqs, key=lambda i: i.shape[-1], reverse=True) # high to low

    # interleave eg. THWTHW...THTH...TTT
    offset = 0
    last_len = 0
    for _ in range(len(freqs)): # 3
        indices = torch.arange(freqs[-1].shape[-1]-offset) # aka 10-12 -> 0-2
        for i, f in enumerate(freqs):
            out[..., (indices*len(freqs))+i+last_len] = f[..., indices+offset]

        offset += indices.shape[0]
        last_len += indices.shape[0] * len(freqs)
        freqs.pop(-1)

    return out


def get_freqs(in_tokens=2048, in_grid=[16, 64, 64], head_dim=64, theta=10000.0):
    axes_dim = head_dim/len(in_grid)
    axes_dim = [int(axes_dim - (axes_dim % 2))] * len(in_grid)
    axes_dim[0] += head_dim - sum(axes_dim) # add remainder to T dim
    rope_grid = get_grid(in_grid, in_tokens)

    result = []
    for i in range(len(axes_dim)):
        freqs = get_1d_rotary_pos_embed(axes_dim[i], rope_grid[:, i], theta=theta)
        result.append(freqs)

    # return torch.cat(result, dim=-1)
    return interleave_freqs(result)