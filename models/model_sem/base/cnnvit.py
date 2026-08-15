import torch
import torch.nn as nn
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=16):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention (flatten T, H, W)
        b, c, t, h, w = q.shape
        q = q.reshape(b, c, t*h*w).permute(0, 2, 1)   # b, thw, c
        k = k.reshape(b, c, t*h*w)                    # b, c, thw
        w_ = torch.bmm(q, k)     # b, thw, thw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, t*h*w)
        w_ = w_.permute(0, 2, 1)   # b, thw, thw
        h_ = torch.bmm(v, w_)      # b, c, thw
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # stride example: (1, 2, 2) or (2, 2, 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample3D(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor # (t_scale, h_scale, w_scale)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Nearest Neighbor Upsampling first
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)

class Encoder_cnn(nn.Module):
    def __init__(self, in_channels=3, ch=16, 
                 ch_mult=(1, 2, 4, 4), # 使用这个配置来获得 3 次下采样
                 num_res_blocks=2, 
                 norm_type='group', dropout=0.0, z_channels=256,
                 use_attn=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.use_attn = use_attn
        
        # 初始卷积
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        prev_out_dim = ch
        self.conv_blocks = nn.ModuleList()

        # 定义每一步下采样的 Stride (Time, Height, Width)
        # 目标: Input(16,128,128) -> Output(4,16,16) (Total /4, /8, /8)
        # 我们有 3 个下采样阶段 (len(ch_mult)=4, 所以有 3 次 transition)
        # 1. (1, 2, 2) -> T:16, H:64
        # 2. (2, 2, 2) -> T:8,  H:16
        # 3. (2, 2, 2) -> T:4,  H:16
        downsample_strides = [
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2)
        ]

        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            
            block_in = prev_out_dim
            block_out = ch * ch_mult[i_level]
            
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock3D(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn:
                    # 只在最后的高语义层加 Attention，节省显存
                    if i_level == self.num_resolutions - 1:
                        attn_block.append(AttnBlock3D(block_in, norm_type))
            
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            
            # Downsample logic
            if i_level != self.num_resolutions - 1:
                # 获取预定义的 stride
                curr_stride = downsample_strides[i_level]
                conv_block.downsample = Downsample3D(block_in, block_in, stride=curr_stride)
                prev_out_dim = block_in # downsample 不改变通道，只改变尺寸

            self.conv_blocks.append(conv_block)

        # Middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock3D(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock3D(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock3D(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # End
        self.norm_out = Normalize(block_in, num_groups=16) 
        self.conv_out = nn.Conv3d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        h = self.conv_in(x)
        
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        for mid_block in self.mid:
            h = mid_block(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder_cnn(nn.Module):
    def __init__(self, z_channels=256, ch=16, 
                 ch_mult=(1, 2, 4, 4), 
                 num_res_blocks=2, norm_type="group",
                 dropout=0.0, out_channels=3, use_attn=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.use_attn = use_attn

        # 计算最后一层的通道数
        block_in = ch * ch_mult[self.num_resolutions-1]
        
        # z to block_in
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock3D(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock3D(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock3D(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # Upsampling (Encoder 的逆过程)
        # Encoder Down: (1,2,2) -> (2,2,2) -> (2,2,2)
        # Decoder Up:   (2,2,2) -> (2,2,2) -> (1,2,2)
        upsample_scales = [
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2)
        ]

        self.conv_blocks = nn.ModuleList()
        
        prev_out_dim = block_in

        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            
            block_out = ch * ch_mult[i_level]
            
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock3D(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn and i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock3D(block_in, norm_type))
            
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            
            if i_level != 0:
                # 对应 Encoder 的逆序 stride
                # Encoder index: 0, 1, 2. Decoder reversed: starts at level 3 down to 0.
                # Upsample happens when i_level != 0. 
                # Specifically: 
                # i_level 3 -> 2: upsample_scales[0] (2,2,2)
                # i_level 2 -> 1: upsample_scales[1] (2,2,2)
                # i_level 1 -> 0: upsample_scales[2] (1,2,2)
                scale = upsample_scales[self.num_resolutions - 1 - i_level]
                conv_block.upsample = Upsample3D(block_in, scale_factor=scale)
                prev_out_dim = block_in

            self.conv_blocks.append(conv_block)

        # End
        self.norm_out = Normalize(block_in, num_groups=16)
        self.conv_out = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z: [B, Z, T', H', W']
        h = self.conv_in(z)

        # Middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # Upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h