import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_sem.base.transformer import ResidualAttentionBlock,ResidualAttentionBlock_sem
from models.model_sem.base.utils import get_model_dims, init_weights
from models.model_sem.base.rope import get_freqs
from einops.layers.torch import Rearrange
from einops import rearrange
import math

import torch
import torch.nn as nn
import math
from einops import rearrange


class Encoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3,  # RGB
            out_channels=5,  # len(fsq_levels)
            in_grid=(32, 256, 256),
            out_tokens=2048,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels
        self.out_tokens = out_tokens
        self.grid = [x // y for x, y in zip(in_grid, patch_size)]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        # ========== 改动：Linear → Conv3d ==========
        self.proj_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        # ============================================

        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(out_tokens, self.grid, head_dim=self.width // self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        x = self.proj_in(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        mask_tokens = self.mask_token.expand(B, self.out_tokens, self.width)
        x = torch.cat([mask_tokens, x], dim=1)
        x = self.model_layers(x, freqs=self.freqs.to(device))
        x = x[:, :self.out_tokens]
        x = self.proj_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,
            out_grid=(32, 256, 256),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.grid = [x // y for x, y in zip(out_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(in_tokens, self.grid, head_dim=self.width // self.heads)
        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )
        self.proj_out = nn.ConvTranspose3d(
            in_channels=self.width,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.apply(init_weights)

    def forward(self, x):
        # x: quantized latent tokens [B, in_tokens, C]
        B = x.shape[0]
        device = x.device
        x = self.proj_in(x)
        mask_tokens = self.mask_token.expand(B, self.grid_size, self.width)
        x = torch.cat([x, mask_tokens], dim=1)
        x = self.model_layers(x, freqs=self.freqs.to(device))
        x = x[:, self.in_tokens:]  # [B, grid_size, width]
        x = rearrange(
            x, 'b (t h w) c -> b c t h w',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
        )
        x = self.proj_out(x)
        return x


class TokenizerEncoder1D(nn.Module):
    """Compress N_vfm VFM tokens → N_latent tokens (for FSQ quantization)."""
    
    def __init__(
        self,
        model_size="base_thin",
        in_channels=1024,       # D_teacher (VFM embed dim)
        out_channels=6,         # token_size (FSQ code dim)
        in_tokens=2048,         # number of input VFM tokens
        out_tokens=1024,        # number of output latent tokens
        in_grid=(8, 16, 16),    # spatial grid of VFM tokens
    ):
        super().__init__()
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.in_grid = list(in_grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(in_channels, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        # freqs covers [mask_tokens (out_tokens), grid_tokens (prod(in_grid))]
        self.freqs = get_freqs(out_tokens, self.in_grid, head_dim=self.width // self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers,
        )
        self.proj_out = nn.Linear(self.width, out_channels, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        """
        x: [B, in_tokens, D_teacher]
        Returns: [B, out_tokens, token_size]
        """
        B = x.shape[0]
        device = x.device
        x = self.proj_in(x)                                          # [B, 2048, width]
        mask_tokens = self.mask_token.expand(B, self.out_tokens, self.width)  # [B, 1024, width]
        x = torch.cat([mask_tokens, x], dim=1)                       # [B, 3072, width]
        x = self.model_layers(x, freqs=self.freqs.to(device))
        x = x[:, :self.out_tokens]                                    # [B, 1024, width]
        x = self.proj_out(x)                                          # [B, 1024, token_size]
        return x


class TokenizerDecoder1D(nn.Module):
    """Expand N_latent quantized tokens → N_vfm tokens."""
    
    def __init__(
        self,
        model_size="base_thin",
        in_channels=6,          # token_size (FSQ code dim)
        in_tokens=1024,         # number of input latent tokens
        out_tokens=2048,        # number of output tokens
        out_grid=(8, 16, 16),   # spatial grid of output tokens
    ):
        super().__init__()
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.out_grid = list(out_grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(in_channels, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        # freqs covers [input_tokens (in_tokens), mask_tokens (prod(out_grid))]
        self.freqs = get_freqs(in_tokens, self.out_grid, head_dim=self.width // self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers,
        )
        # Output remains in width dimension — no projection
        self.apply(init_weights)

    @property
    def output_dim(self):
        return self.width

    def forward(self, x):
        """
        x: [B, in_tokens, token_size]
        Returns: [B, out_tokens, width]
        """
        B = x.shape[0]
        device = x.device
        x = self.proj_in(x)                                           # [B, 1024, width]
        mask_tokens = self.mask_token.expand(B, self.out_tokens, self.width)  # [B, 2048, width]
        x = torch.cat([x, mask_tokens], dim=1)                        # [B, 3072, width]
        x = self.model_layers(x, freqs=self.freqs.to(device))
        x = x[:, self.in_tokens:]                                      # [B, 2048, width]
        return x




class VideoDecoder(nn.Module):
    """Decode spatial tokens → video via attention + ConvTranspose3d."""
    
    def __init__(
        self,
        model_size="base_thin",
        in_channels=None,       # input token dim (from TokenizerDecoder1D)
        out_channels=3,         # RGB
        num_tokens=2048,
        token_grid=(8, 16, 16),
        patch_size=(2, 16,16),
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_grid = list(token_grid)
        self.patch_size = list(patch_size)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)

        if in_channels is not None and in_channels != self.width:
            self.proj_in = nn.Linear(in_channels, self.width, bias=True)
        else:
            self.proj_in = nn.Identity()

        # positional encoding: only grid tokens (no extra tokens)
        self.freqs = get_freqs(0, self.token_grid, head_dim=self.width // self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers,
        )

        self.proj_out = nn.ConvTranspose3d(
            in_channels=self.width,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.apply(init_weights)

    def forward(self, x):
        """
        x: [B, num_tokens, in_channels]
        Returns: [B, out_channels, T, H, W]
        """
        x = self.proj_in(x)          # [B, 2048, width]
        x = self.model_layers(x, freqs=self.freqs.to(x.device))
        x = rearrange(
            x, 'b (t h w) c -> b c t h w',
            t=self.token_grid[0], h=self.token_grid[1], w=self.token_grid[2],
        )
        x = self.proj_out(x)          # [B, 3, T*pt, H*ph, W*pw]
        return x







class Decoder_sem(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,
            out_grid=(32, 256, 256),
            inter_layer_indices=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.grid = [x // y for x, y in zip(out_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        if inter_layer_indices is None:
            inter_layer_indices = (2,)
        self.inter_layer_indices = tuple(inter_layer_indices)
        self.multi_layer_inter = len(self.inter_layer_indices) > 1
        scale = self.width ** -0.5
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(in_tokens, self.grid, head_dim=self.width // self.heads)
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

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        x = self.proj_in(x)
        mask_tokens = self.mask_token.expand(B, self.grid_size, self.width)
        
        # x 包含 condition tokens (in_tokens) 和 mask tokens
        x = torch.cat([x, mask_tokens], dim=1)
        
        x, inter_out = self.model_layers(x, freqs=self.freqs.to(device))

        x = x[:, self.in_tokens:]

        if self.multi_layer_inter:
            inter_feats = {k: v[:, self.in_tokens:] for k, v in inter_out.items()}
        else:
            inter_feats = inter_out[:, self.in_tokens:]

        x = rearrange(
            x, 'b (t h w) c -> b c t h w',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
        )
        x = self.proj_out(x)
        return x, inter_feats