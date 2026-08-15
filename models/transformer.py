import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as BlockTimm

from models import register


@register('transformer_encoder_fused')
class TransformerEncoderFused(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        assert ff_dim is None
        assert dim == head_dim * n_head

        self.blocks = nn.Sequential(
            *[
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# @register('transformer_encoder_parallel')
# class TransformerEncoderParallel(nn.Module):
#     def __init__(
#         self,
#         dim,
#         depth,
#         n_head,
#         head_dim,
#         ff_dim=None,
#         dropout=0.0
#     ):
#         super().__init__()
#         self.is_encoder_decoder = True
#         assert ff_dim is None
#         assert dim == head_dim * n_head
#         self.blocks = nn.ModuleList()
#         for _ in range(depth):
#             self.blocks.append(
#                 BlockTimm(
#                     dim=dim,
#                     num_heads=n_head,
#                     mlp_ratio=4,
#                     qkv_bias=False,
#                     proj_drop=dropout,
#                     attn_drop=dropout,
#                 )
#             )

#     def forward(self, context, query):
#         query_length = query.size(1)
#         h = torch.cat([context, query], dim=1)

#         for block in self.blocks:
#             h = block(h)

#         h = h[:, -query_length:, :]
#         return h
@register('transformer_encoder_parallel')
class TransformerEncoderParallel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        n_head,
        head_dim,
        ff_dim=None,
        dropout=0.0,
        inter_layer_idx=4,   # 第几层作为中间特征输出（1-based，第 4 层 → idx=4）
    ):
        super().__init__()
        self.is_encoder_decoder = True
        assert ff_dim is None
        assert dim == head_dim * n_head
        self.depth = depth
        self.inter_layer_idx = inter_layer_idx
        assert 1 <= inter_layer_idx <= depth, \
            f"inter_layer_idx={inter_layer_idx} 必须在 [1, depth={depth}] 范围内"

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
            )

    def forward(self, context, query, return_interfeat=False):
        """
        Args:
            context: (B, N_ctx, D)
            query:   (B, N_q,   D)
            return_interfeat: 是否返回第 `inter_layer_idx` 层后 query 部分的特征
        Returns:
            如果 return_interfeat=False:
                h_query: (B, N_q, D)  最后一层 query 输出
            如果 return_interfeat=True:
                h_query: (B, N_q, D)
                inter_feat: (B, N_q, D)  指定中间层的 query 部分特征
        """
        query_length = query.size(1)
        h = torch.cat([context, query], dim=1)

        inter_feat = None
        for i, block in enumerate(self.blocks):
            h = block(h)
            # i 是 0-based，第 inter_layer_idx 层对应 i == inter_layer_idx - 1
            if return_interfeat and (i == self.inter_layer_idx - 1):
                # 只取 query 部分，与最终输出保持形状一致
                inter_feat = h[:, -query_length:, :].clone()

        h_query = h[:, -query_length:, :]

        if return_interfeat:
            return h_query, inter_feat
        else:
            return h_query