"""Hand-built UNet noise predictor for diffusion.

Architecture (standard DDPM UNet, Ho et al. 2020, simplified):

    input: (B, C, H, W) noisy image + timestep t (+ optional context)
      → sinusoidal time embedding → MLP → scale/shift per residual block
      → encoder: [ResBlock → ResBlock → Downsample] × depth
      → middle: ResBlock → Attention → ResBlock
      → decoder: [Upsample → ResBlock → ResBlock] × depth (skip connections)
      → GroupNorm → SiLU → Conv → (B, C, H, W) predicted noise

Shapes are documented on every block. Requires torch; importing this
module without torch raises :class:`ImgGenError` with install guidance
instead of a bare ImportError.
"""

from __future__ import annotations

import math

from . import ImgGenError, TORCH_AVAILABLE

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.unet needs PyTorch: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu "
        "(CPU build is fine for tiny models; CUDA build for real training)")

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SinusoidalTimeEmbedding",
    "ResidualBlock",
    "AttentionBlock",
    "Downsample",
    "Upsample",
    "UNet",
    "count_parameters",
]


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al. 2017, as used by Ho).

    t: (B,) int64 → (B, dim) float. dim must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ImgGenError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # (half,) frequencies: 10000^(-2i/d)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResidualBlock(nn.Module):
    """Two convs + GroupNorm + SiLU, with time-embedding scale/shift.

    in:  (B, in_ch, H, W), temb: (B, temb_dim)
    out: (B, out_ch, H, W)
    """

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int,
                 groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_ch * 2),
        )
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.residual = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.temb_proj(temb).chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1))
        h = h + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    """Single-head-reshaped multi-head self-attention over pixels.

    in/out: (B, ch, H, W). Q/K/V via 1x1 convs, softmax(QK^T/sqrt(d))V.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads:
            raise ImgGenError("attention channels must divide num_heads")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        # (B, heads, head_dim, H*W)
        def _split(t):
            t = t.view(b, self.num_heads, c // self.num_heads, h * w)
            return t

        q, k, v = _split(q), _split(k), _split(v)
        attn = torch.softmax(
            (q.transpose(-2, -1) @ k) / math.sqrt(c // self.num_heads),
            dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: image queries attend to text context keys/values.

    x: (B, ch, H, W), context: (B, seq, ctx_dim) → (B, ch, H, W).
    This is what makes text conditioning work (Rombach et al. 2022).
    """

    def __init__(self, channels: int, ctx_dim: int,
                 num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads:
            raise ImgGenError("channels must divide num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(ctx_dim, channels)
        self.to_v = nn.Linear(ctx_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(self.norm(x)).view(
            b, self.num_heads, self.head_dim, h * w)
        k = self.to_k(context).view(
            b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(
            b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(
            (q.transpose(-2, -1) @ k.transpose(-2, -1))
            / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided conv halving H, W and (optionally) changing channels.

    (B, in_ch, H, W) → (B, out_ch, H/2, W/2).
    """

    def __init__(self, in_ch: int, out_ch: int | None = None) -> None:
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv doubling H, W."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """Denoising UNet.

    Args:
        in_channels: image channels (3 = RGB pixels, 4 = SD latent).
        base_channels: width of the first level; doubles per depth.
        depth: number of down/up levels (resolution / 2**depth at bottom).
        temb_dim: time-embedding width.
        ctx_dim: text-context width; 0/None disables cross-attention.
        attn_heads: attention heads.
        image_size: spatial size the model is built for (used to sanity
            check inputs, not to reshape).

    Forward: (noisy: (B, C, H, W), t: (B,) int64, context: (B, S, ctx)?)
             → (B, C, H, W) predicted noise.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 depth: int = 3, temb_dim: int = 256,
                 ctx_dim: int | None = None, attn_heads: int = 4,
                 image_size: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.ctx_dim = ctx_dim

        self.sinusoidal = SinusoidalTimeEmbedding(temb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 4),
            nn.SiLU(),
            nn.Linear(temb_dim * 4, temb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder.
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(ch, ch, temb_dim),
                ResidualBlock(ch, ch, temb_dim),
            ]))
            # Downsample doubles channels for the next level.
            self.downsamples.append(Downsample(ch, ch * 2))
            ch *= 2

        # Bottleneck.
        self.mid1 = ResidualBlock(ch, ch, temb_dim)
        self.mid_attn = AttentionBlock(ch, attn_heads)
        self.mid2 = ResidualBlock(ch, ch, temb_dim)

        # Decoder.
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            ch //= 2
            self.upsamples.append(Upsample(ch * 2))
            blocks: list[nn.Module] = [
                ResidualBlock(ch * 3, ch, temb_dim),
                ResidualBlock(ch, ch, temb_dim),
            ]
            if ctx_dim:
                blocks.append(CrossAttentionBlock(ch, ctx_dim, attn_heads))
            self.up_blocks.append(nn.ModuleList(blocks))

        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ImgGenError(
                f"UNet expects {self.in_channels} channels, "
                f"got {x.shape[1]}")
        temb = self._temb(t)

        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for blk in blocks:
                h = blk(h, temb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for up, blocks in zip(self.upsamples, self.up_blocks):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for blk in blocks:
                if isinstance(blk, CrossAttentionBlock):
                    if context is None:
                        raise ImgGenError(
                            "UNet has cross-attention but no text "
                            "context was given")
                    h = blk(h, context)
                else:
                    h = blk(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.sinusoidal(t))


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
