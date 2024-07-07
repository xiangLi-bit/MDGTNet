import numbers
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


##########################################################################
# Implementation of layer normalization introduced in:
# 'Restormer: Efficient Transformer for High-Resolution Image Restoration'(Zamir et al., 2022)
# https://github.com/swz30/Restormer
def to_3d(x):
    # input: [b,c,h,w] output: [b,(h,w),c]
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    # input: [b,(h,w),c] output: [b,c,h,w]
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # imput:[b,(h,w),c]
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # imput:[b,(h,w),c]
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # input:[b,c,h,w]
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Multi-Depth Gated Module
class MDGM(nn.Module):
    def __init__(self, dim, hidden_features=None, dropout=0., bias=False):
        super().__init__()
        hidden_features = hidden_features or dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_shallow = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                        groups=hidden_features, bias=bias)
        self.dwconv_deep1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                      groups=hidden_features, bias=bias)
        self.dwconv_deep2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                      groups=hidden_features, bias=bias)

        self.dwconv_gate = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)

        # self.dropout = torch.nn.Dropout(dropout)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x, x_gate = self.project_in(x).chunk(2, dim=1)
        # shallow
        x_shallow = self.dwconv_shallow(x)
        # deep
        x_deep = self.dwconv_deep2(F.gelu(self.dwconv_deep1(x)))
        x = x_shallow + x_deep
        # gated
        x_gate = F.gelu(self.dwconv_gate(x_gate))

        x = x_gate * x

        # x = self.dropout(x)
        x = self.project_out(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sparse_weight = nn.Parameter(0.5 * torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        print('[chattn]:', 'dims=', dim, ', num_heads=', num_heads)

    def forward(self, x):
        # input: [B,C,H,W]
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # softmax
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, dropout=0., bias=False):
        super().__init__()

        self.norm1 = LayerNorm(dim=dim)
        self.ch_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = LayerNorm(dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = MDGM(dim=dim, hidden_features=mlp_hidden_dim, dropout=dropout, bias=bias)


    def forward(self, x):
        x = x + self.ch_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Backbone(nn.Module):
    def __init__(self, config, num_blocks=[4, 6, 6, 8], bias=False):
        super().__init__()
        self.config = config
        channels, channels_mult = config.backbone.channels, tuple(config.backbone.channels_mult)
        dropout = config.backbone.dropout
        in_channels = 3
        out_channels = 3

        self.ch = channels
        self.ch_level = [self.ch * channels_mult[i] for i in range(len(channels_mult))]
        self.in_channels = in_channels
        self.mlp_ratio = 2.00

        # input
        self.conv_in = nn.Conv2d(in_channels, self.ch_level[0], kernel_size=3, stride=1, padding=1, bias=bias)

        # E_1
        self.attn1_1 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[0], num_heads=1, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[0])])

        self.down1_2 = nn.Conv2d(self.ch_level[0], self.ch_level[1], kernel_size=3, stride=2, padding=1, bias=bias)

        # E_2
        self.attn2_1 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[1], num_heads=2, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[1])])

        self.down2_3 = nn.Conv2d(self.ch_level[1], self.ch_level[2], kernel_size=3, stride=2, padding=1, bias=bias)

        # E_3
        self.attn3_1 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[2], num_heads=4, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[2])])

        self.down3_4 = nn.Conv2d(self.ch_level[2], self.ch_level[3], kernel_size=3, stride=2, padding=1, bias=bias)

        # middle
        self.attn4 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[3], num_heads=8, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[3])])

        self.up4_3 = nn.ConvTranspose2d(self.ch_level[3], self.ch_level[2], kernel_size=4, padding=1, stride=2)

        # D_3
        self.chcat3 = nn.Conv2d(self.ch_level[2] * 2, self.ch_level[2], kernel_size=1, bias=bias)
        self.attn3_2 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[2], num_heads=4, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[2])])

        self.up3_2 = nn.ConvTranspose2d(self.ch_level[2], self.ch_level[1], kernel_size=4, padding=1, stride=2)

        # D_2
        self.chcat2 = nn.Conv2d(self.ch_level[1] * 2, self.ch_level[1], kernel_size=1, bias=bias)
        self.attn2_2 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[1], num_heads=2, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[1])])

        self.up2_1 = nn.ConvTranspose2d(self.ch_level[1], self.ch_level[0], kernel_size=4, padding=1, stride=2)

        # D_1
        self.attn1_2 = nn.Sequential(
            *[TransformerBlock(dim=self.ch_level[0] * 2, num_heads=1, mlp_ratio=self.mlp_ratio, dropout=dropout) for i in range(num_blocks[0])])

        # output
        self.conv_out = torch.nn.Conv2d(self.ch_level[0] * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):

        # input
        x1 = self.conv_in(x)

        # E_1
        x1 = self.attn1_1(x1)
        x2 = self.down1_2(x1)

        # E_2
        x2 = self.attn2_1(x2)
        x3 = self.down2_3(x2)

        # E_3
        x3 = self.attn3_1(x3)
        h = self.down3_4(x3)

        # middle
        h = self.attn4(h)
        h = self.up4_3(h)

        # D_3
        h = self.chcat3(torch.cat([h, x3], 1))
        h = self.attn3_2(h)
        h = self.up3_2(h)

        # D_2
        h = self.chcat2(torch.cat([h, x2], 1))
        h = self.attn2_2(h)
        h = self.up2_1(h)

        # D_1
        h = self.attn1_2(torch.cat([h, x1], 1))

        # output
        h = self.conv_out(h)

        return x + h











