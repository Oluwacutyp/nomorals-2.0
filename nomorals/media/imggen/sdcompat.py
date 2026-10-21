"""SD-format compatibility: Devon's own SD1.5 architecture, open weights.

The pipeline code is ours; weights are data. This module implements
the Stable Diffusion 1.5 UNet, CLIP ViT-L/14 text encoder, and
AutoencoderKL **from their documented architectures** and loads
``.safetensors`` checkpoints in the original compvis/LDM key format
(``model.diffusion_model.*``, ``cond_stage_model.*``,
``first_stage_model.*``) — the format CivitAI and most HF repos ship.

What this unlocks: any SD1.5 checkpoint becomes a Devon pipeline
backend with zero diffusers dependency. LoRA adapters in diffusers
format map onto :mod:`.lora` the same way.

Honest scope: UNet + CLIP-L + VAE only (the SD1.5 trio). SDXL has a
different UNet and two text encoders — not covered; the error says so.
"""

from __future__ import annotations

import math
import os
import re

from . import ImgGenError, TORCH_AVAILABLE

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.sdcompat needs PyTorch: "
        "pip install torch")

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SDUNet",
    "CLIPTextEncoder",
    "AutoencoderKL",
    "load_safetensors",
    "SD15Pipeline",
    "SD15StudioAdapter",
]


# ---------------------------------------------------------------------------
# Building blocks (SD-flavored)
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    """Gated linear unit with GeLU: the SD feed-forward activation."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(dim, dim * mult),
            nn.Dropout(0.0),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Multi-head attention with optional context (cross-attention)."""

    def __init__(self, query_dim: int, context_dim: int | None = None,
                 heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = x if context is None else context
        k, v = self.to_k(context), self.to_v(context)

        def _split(t):
            b, n, _ = t.shape
            return t.view(b, n, h, -1).transpose(1, 2)

        q, k, v = _split(q), _split(k), _split(v)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale,
                             dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], -1,
                                                h * (q.shape[-1]))
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """attn1 (self) → attn2 (cross) → ff, each with norm + residual."""

    def __init__(self, dim: int, heads: int, dim_head: int,
                 context_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, None, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, dim_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """The transformer that sits inside each SD UNet block.

    (B, C, H, W) → norm → proj_in → flatten → N×BasicTransformerBlock
    → proj_out → reshape → residual. dim_head = channels // heads,
    matching SD (8 heads; dim_head grows with width).
    """

    def __init__(self, channels: int, heads: int = 8,
                 depth: int = 1, context_dim: int = 768) -> None:
        super().__init__()
        if channels % heads:
            raise ImgGenError(
                f"SD channels {channels} must divide heads {heads}")
        dim_head = channels // heads
        inner = heads * dim_head  # == channels
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, inner, 1)
        self.blocks = nn.ModuleList(
            BasicTransformerBlock(inner, heads, dim_head, context_dim)
            for _ in range(depth))
        self.proj_out = nn.Conv2d(inner, channels, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        for blk in self.blocks:
            x = blk(x, context)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + residual


class ResnetBlock(nn.Module):
    """SD resnet block: norm→SiLU→conv→temb→norm→SiLU→conv + shortcut."""

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Sequential(nn.SiLU(),
                                       nn.Linear(temb_ch, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                         else nn.Identity())

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Resnets only (the 4th SD down block).

    LDM's last down level has NO downsampler — the latent enters the
    middle block at the third level's resolution. ``downsample``
    exists because an earlier shape of this class always downsampled,
    which both broke the spatial chain at real image sizes and left
    two parameters no SD checkpoint could ever fill.
    """

    def __init__(self, in_ch, out_ch, temb_ch, layers=2,
                 downsample: bool = False):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch, temb_ch)
            for i in range(layers))
        self.downsampler = (nn.Conv2d(out_ch, out_ch, 3, stride=2,
                                      padding=1)
                            if downsample else None)

    def forward(self, x, temb):
        for r in self.resnets:
            x = r(x, temb)
        skip = x
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x, skip


class CrossAttnDownBlock(nn.Module):
    """Resnet + transformer pairs (SD down blocks 0–2)."""

    def __init__(self, in_ch, out_ch, temb_ch, heads,
                 layers=2, context_dim=768, down=True):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(layers):
            ic = in_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock(ic, out_ch, temb_ch))
            self.transformers.append(
                SpatialTransformer(out_ch, heads, 1,
                                   context_dim))
        self.downsampler = (nn.Conv2d(out_ch, out_ch, 3, stride=2,
                                      padding=1) if down else None)

    def forward(self, x, temb, context=None):
        for r, t in zip(self.resnets, self.transformers):
            x = r(x, temb)
            x = t(x, context)
        skip = x
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x, skip


class CrossAttnUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, heads,
                 layers=3, context_dim=768, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.resnets = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(layers):
            ic = in_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock(ic, out_ch, temb_ch))
            self.transformers.append(
                SpatialTransformer(out_ch, heads, 1,
                                   context_dim))
        # The conv is the LDM upsample conv of this group's last
        # block (output_blocks 2/5/8 ".2.conv"); the final group
        # (up3) has no upsample in LDM, hence no conv — an earlier
        # shape always built one, leaving parameters no checkpoint
        # could fill. Interpolation still only happens when
        # ``upsample`` is set; the conv runs whenever it exists.
        self.upsampler = (nn.Conv2d(out_ch, out_ch, 3, padding=1)
                          if upsample else nn.Identity())

    def forward(self, x, skip, temb, context=None):
        x = torch.cat([x, skip], dim=1)
        for r, t in zip(self.resnets, self.transformers):
            x = r(x, temb)
            x = t(x, context)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.upsampler(x)


# ---------------------------------------------------------------------------
# SD1.5 UNet
# ---------------------------------------------------------------------------

class SDUNet(nn.Module):
    """Stable Diffusion 1.5 UNet — our implementation, open weights.

    Channels: 320 → 640 → 1280 → 1280; cross-attention context 768
    (CLIP ViT-L/14). Time embedding 320 → 1280. Latent in/out: 4 ch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 4
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280))

        self.down0 = CrossAttnDownBlock(320, 320, 1280, 8)
        self.down1 = CrossAttnDownBlock(320, 640, 1280, 8)
        self.down2 = CrossAttnDownBlock(640, 1280, 1280, 8)
        self.down3 = DownBlock(1280, 1280, 1280)

        self.mid_res1 = ResnetBlock(1280, 1280, 1280)
        self.mid_trans = SpatialTransformer(1280, 8, 1, 768)
        self.mid_res2 = ResnetBlock(1280, 1280, 1280)

        # Skip channels: up0 takes mid(1280)+down3(1280)=2560;
        # up1 takes 1280+down2(1280)=2560; up2 takes 1280+down1(640);
        # up3 takes 640+down0(320)=960. Only up0..up2 upsample.
        self.up0 = CrossAttnUpBlock(2560, 1280, 1280, 8,
                                   upsample=True)
        self.up1 = CrossAttnUpBlock(2560, 1280, 1280, 8,
                                   upsample=True)
        self.up2 = CrossAttnUpBlock(1920, 640, 1280, 8,
                                    upsample=True)
        self.up3 = CrossAttnUpBlock(960, 320, 1280, 8,
                                    upsample=False)

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_out = nn.Conv2d(320, 4, 3, padding=1)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int = 320) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device).float()
                          / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    def forward(self, x, t, context=None):
        temb = self.time_embed(self.timestep_embedding(t))
        h = self.conv_in(x)
        skips = []
        for blk in (self.down0, self.down1, self.down2):
            h, s = blk(h, temb, context)
            skips.append(s)
        h, s = self.down3(h, temb)
        skips.append(s)
        h = self.mid_res1(h, temb)
        h = self.mid_trans(h, context)
        h = self.mid_res2(h, temb)
        # Skips pop in reverse: down3, down2, down1, down0.
        h = self.up0(h, skips[3], temb, context)
        h = self.up1(h, skips[2], temb, context)
        h = self.up2(h, skips[1], temb, context)
        h = self.up3(h, skips[0], temb, context)
        h = self.conv_norm_out(h)
        return self.conv_out(F.silu(h))


# ---------------------------------------------------------------------------
# CLIP text encoder (ViT-L/14 text tower)
# ---------------------------------------------------------------------------

class CLIPAttention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        def _s(t):
            return t.view(b, n, h, -1).transpose(1, 2)

        q, k, v = _s(self.q_proj(x)), _s(self.k_proj(x)), _s(
            self.v_proj(x))
        # Causal mask: CLIP text is autoregressive.
        mask = torch.full((n, n), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale
                             + mask, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out_proj(out)


class CLIPLayer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = CLIPAttention(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 3072), nn.GELU(),
                                 nn.Linear(3072, dim))

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPTextEncoder(nn.Module):
    """CLIP ViT-L/14 text encoder (SD1.5's conditioner).

    Tokenization: when a real CLIP BPE tokenizer is attached
    (``self.tokenizer``, set by the checkpoint loader from vocab data
    found near the weights — see :mod:`.tokenizers`), :meth:`encode`
    produces true CLIP token ids. Without one it falls back to a
    stable *hashing* tokenizer that is NOT BPE; that mode is exposed
    as ``tokenizer_mode == "hashing-fallback"`` and surfaced to the
    user by the pipeline — never silently treated as equivalent.
    """

    def __init__(self, vocab_size=49408, dim=768, layers=12,
                 max_tokens=77):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_tokens, dim)
        self.layers = nn.ModuleList(CLIPLayer(dim)
                                    for _ in range(layers))
        self.final_layer_norm = nn.LayerNorm(dim)
        #: A .tokenizers.CLIPBPETokenizer, or None (hashing fallback).
        self.tokenizer = None

    @property
    def tokenizer_mode(self) -> str:
        """``"bpe"`` with a real tokenizer attached, else fallback."""
        return "bpe" if self.tokenizer is not None \
            else "hashing-fallback"

    @staticmethod
    def _hash_tokenize(text: str, vocab_size: int,
                       max_tokens: int) -> list[int]:
        """Stable hashing tokenizer (NOT CLIP BPE — see class docstring).

        BOS=49406, EOS=49407 match CLIP's specials; the rest hash.
        """
        import hashlib

        ids = [49406]
        for word in text.lower().split()[:max_tokens - 2]:
            h = hashlib.md5(word.encode()).digest()
            ids.append(49400 - (int.from_bytes(h[:2], "little") % 49000))
        ids.append(49407)
        ids += [49407] * (max_tokens - len(ids))
        return ids[:max_tokens]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        pos = torch.arange(self.max_tokens,
                           device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer_norm(x)

    def encode(self, prompts: list[str]) -> torch.Tensor:
        device = self.token_embedding.weight.device
        if self.tokenizer is not None:
            rows = [self.tokenizer.encode(p, self.max_tokens)
                    for p in prompts]
        else:
            rows = [self._hash_tokenize(p, 49408, self.max_tokens)
                    for p in prompts]
        ids = torch.tensor(rows, dtype=torch.long, device=device)
        return self(ids)


# ---------------------------------------------------------------------------
# AutoencoderKL (SD VAE)
# ---------------------------------------------------------------------------

class ResnetVAE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class VAEAttention(nn.Module):
    """LDM AttnBlock: single-head self-attention over spatial positions.

    Lives in the VAE mid blocks (``attn_1``) in the real SD weights.
    The previous AutoencoderKL omitted it entirely, which is why the
    old VAE key map could never be complete; the class exists so
    every ``first_stage_model.*`` key has a real target.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)   # (b, hw, c)
        k = k.reshape(b, c, hh * ww)                     # (b, c, hw)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=2)
        v = v.reshape(b, c, hh * ww).permute(0, 2, 1)    # (b, hw, c)
        out = torch.bmm(attn, v).reshape(b, hh, ww, c).permute(0, 3, 1, 2)
        return x + self.proj_out(out)


class _VAEUpsample(nn.Module):
    """LDM Upsample: nearest-neighbor ×2 then a 3×3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0,
                                       mode="nearest"))


class AutoencoderKL(nn.Module):
    """SD's VAE: 8× downsample, 4-channel latent, KL-regularized.

    The structure mirrors the LDM AutoencoderKL block for block —
    four encoder levels (the last without downsample), the mid trio
    (resnet → attention → resnet) on both sides, and four decoder
    levels of *three* resnets each (channel changes happen in the
    first resnet of a level, as in LDM) — so the checkpoint's
    ``first_stage_model.*`` key set maps onto this class completely,
    with nothing dropped and nothing approximated (see
    :func:`_load_ldm_vae`). An earlier shape of this class folded
    levels together and could never have held the real weights.
    """

    def __init__(self, latent_ch: int = 4) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        # Encoder (LDM down.0..3): 3 → 128 → 256 → 512 → 512.
        self.enc_in = nn.Conv2d(3, 128, 3, padding=1)
        self.enc_b1 = nn.Sequential(ResnetVAE(128, 128),
                                    ResnetVAE(128, 128),
                                    nn.Conv2d(128, 128, 3, stride=2,
                                              padding=1))
        self.enc_b2 = nn.Sequential(ResnetVAE(128, 256),
                                    ResnetVAE(256, 256),
                                    nn.Conv2d(256, 256, 3, stride=2,
                                              padding=1))
        self.enc_b3 = nn.Sequential(ResnetVAE(256, 512),
                                    ResnetVAE(512, 512),
                                    nn.Conv2d(512, 512, 3, stride=2,
                                              padding=1))
        self.enc_b4 = nn.Sequential(ResnetVAE(512, 512),
                                    ResnetVAE(512, 512))
        self.enc_mid = nn.Sequential(ResnetVAE(512, 512),
                                     VAEAttention(512),
                                     ResnetVAE(512, 512))
        self.quant_conv = nn.Conv2d(512, latent_ch * 2, 1)
        # Decoder (LDM up.3..0, in execution order): post-quant →
        # mid → 512 (up.3) → 512 (up.2) → 256 (up.1) → 128 (up.0,
        # no upsample) → out.
        self.post_quant = nn.Conv2d(latent_ch, 512, 1)
        self.dec_mid = nn.Sequential(ResnetVAE(512, 512),
                                     VAEAttention(512),
                                     ResnetVAE(512, 512))
        self.dec_b1 = nn.Sequential(ResnetVAE(512, 512),
                                    ResnetVAE(512, 512),
                                    ResnetVAE(512, 512),
                                    _VAEUpsample(512))
        self.dec_b2 = nn.Sequential(ResnetVAE(512, 512),
                                    ResnetVAE(512, 512),
                                    ResnetVAE(512, 512),
                                    _VAEUpsample(512))
        self.dec_b3 = nn.Sequential(ResnetVAE(512, 256),
                                    ResnetVAE(256, 256),
                                    ResnetVAE(256, 256),
                                    _VAEUpsample(256))
        self.dec_b4 = nn.Sequential(ResnetVAE(256, 128),
                                    ResnetVAE(128, 128),
                                    ResnetVAE(128, 128))
        self.dec_out = nn.Sequential(nn.GroupNorm(32, 128), nn.SiLU(),
                                     nn.Conv2d(128, 3, 3, padding=1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """RGB [-1,1] → latent (scaled by SD's 0.18215 factor)."""
        h = self.enc_in(x)
        h = self.enc_b1(h)
        h = self.enc_b2(h)
        h = self.enc_b3(h)
        h = self.enc_b4(h)
        h = self.enc_mid(h)
        mean, logvar = self.quant_conv(h).chunk(2, dim=1)
        return mean * 0.18215

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.post_quant(z / 0.18215)
        h = self.dec_mid(h)
        h = self.dec_b1(h)
        h = self.dec_b2(h)
        h = self.dec_b3(h)
        h = self.dec_b4(h)
        return self.dec_out(h).clamp(-1, 1)


# ---------------------------------------------------------------------------
# Safetensors loading with compvis key mapping
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load a .safetensors file → {name: tensor} (CPU)."""
    if not os.path.exists(path):
        raise ImgGenError(f"weights not found: {path}")
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImgGenError(
            "safetensors not installed: pip install safetensors "
            "(needed to read open SD weights)")
    try:
        return load_file(path, device="cpu")
    except Exception as exc:
        raise ImgGenError(
            f"could not read {path}: {exc}") from exc


class _KeyMapper:
    """Rename LDM checkpoint keys → our module paths, with accounting.

    Every loader in this module maps through a _KeyMapper so the set
    of *consumed* source keys is known exactly. Loaders then enforce
    completeness in both directions: a target parameter left unfilled
    is an error, and a checkpoint key inside a mapped section that no
    mapping consumed is an error too (format drift must fail loudly,
    never drop weights silently).
    """

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state
        self.mapped: dict[str, torch.Tensor] = {}
        self.consumed: set[str] = set()

    def put(self, ldm: str, ours: str) -> None:
        if ldm in self.state:
            self.mapped[ours] = self.state[ldm]
            self.consumed.add(ldm)


def _check_section(state: dict[str, torch.Tensor], prefix: str,
                   consumed: set[str], what: str,
                   known_unused: set[str] | None = None) -> None:
    """Fail if any ``prefix`` key in the checkpoint went unconsumed."""
    unused = sorted(k for k in state
                    if k.startswith(prefix) and k not in consumed
                    and k not in (known_unused or set()))
    if unused:
        raise ImgGenError(
            f"{what}: {len(unused)} checkpoint keys under {prefix!r} "
            f"have no target in Devon's implementation "
            f"(e.g. {unused[0]}); the checkpoint format has drifted "
            f"from the documented layout — refusing to drop weights "
            f"silently")


def _map_ldm_unet(state: dict[str, torch.Tensor]):
    """Map compvis LDM UNet keys → our SDUNet attribute paths.

    LDM layout: input_blocks 0..11 (each: resnet(s) + optional
    transformer + optional downsampler), middle_block (0,1,2),
    output_blocks 0..11. Our layout: down0..down3, mid_*, up0..up4.
    Returns (mapped_state_dict, consumed_source_keys).
    """
    mapper = _KeyMapper(state)
    put = mapper.put
    out = mapper.mapped

    put("model.diffusion_model.time_embed.0.weight",
        "time_embed.0.weight")
    put("model.diffusion_model.time_embed.0.bias", "time_embed.0.bias")
    put("model.diffusion_model.time_embed.2.weight",
        "time_embed.2.weight")
    put("model.diffusion_model.time_embed.2.bias", "time_embed.2.bias")
    put("model.diffusion_model.input_blocks.0.0.weight", "conv_in.weight")
    put("model.diffusion_model.input_blocks.0.0.bias", "conv_in.bias")

    # input_blocks 1..11 → down0..down3.
    # down0: blocks 1,2 (320ch, transformer, downsample)
    # down1: blocks 4,5 (640ch, transformer, downsample)
    # down2: blocks 7,8 (1280ch, transformer, downsample)
    # down3: blocks 10,11 (1280ch, no transformer, no downsample)
    down_map = [
        ("down0", [1, 2], 320), ("down1", [4, 5], 640),
        ("down2", [7, 8], 1280), ("down3", [10, 11], 1280),
    ]
    for dname, blocks, _ch in down_map:
        for li, bi in enumerate(blocks):
            p = f"model.diffusion_model.input_blocks.{bi}.0."
            o = f"{dname}.resnets.{li}."
            for k in ("in_layers.0.weight", "in_layers.0.bias",
                      "in_layers.2.weight", "in_layers.2.bias",
                      "emb_layers.1.weight", "emb_layers.1.bias",
                      "out_layers.0.weight", "out_layers.0.bias",
                      "out_layers.3.weight", "out_layers.3.bias",
                      "skip_connection.weight", "skip_connection.bias"):
                ours_k = (k.replace("in_layers.0.", "norm1.")
                          .replace("in_layers.2.", "conv1.")
                          .replace("emb_layers.1.", "temb_proj.1.")
                          .replace("out_layers.0.", "norm2.")
                          .replace("out_layers.3.", "conv2.")
                          .replace("skip_connection.", "shortcut."))
                put(p + k, o + ours_k)
            # Transformer (blocks 1,2,4,5,7,8 have .1).
            tp = f"model.diffusion_model.input_blocks.{bi}.1."
            to = f"{dname}.transformers.{li}."
            _map_transformer(state, tp, to, put)
        # Downsampler lives on input_blocks.{3,6,9}.0.{weight,bias}
        ds_block = {1: 3, 4: 6, 7: 9}.get(blocks[0])
        if ds_block is not None:
            put(f"model.diffusion_model.input_blocks.{ds_block}.0.weight",
                f"{dname}.downsampler.weight")
            put(f"model.diffusion_model.input_blocks.{ds_block}.0.bias",
                f"{dname}.downsampler.bias")

    # Middle block.
    for k in ("in_layers.0.weight", "in_layers.0.bias",
              "in_layers.2.weight", "in_layers.2.bias",
              "emb_layers.1.weight", "emb_layers.1.bias",
              "out_layers.0.weight", "out_layers.0.bias",
              "out_layers.3.weight", "out_layers.3.bias",
              "skip_connection.weight", "skip_connection.bias"):
        ours_k = (k.replace("in_layers.0.", "norm1.")
                  .replace("in_layers.2.", "conv1.")
                  .replace("emb_layers.1.", "temb_proj.1.")
                  .replace("out_layers.0.", "norm2.")
                  .replace("out_layers.3.", "conv2.")
                  .replace("skip_connection.", "shortcut."))
        put("model.diffusion_model.middle_block.0." + k,
            "mid_res1." + ours_k)
        put("model.diffusion_model.middle_block.2." + k,
            "mid_res2." + ours_k)
    _map_transformer(state, "model.diffusion_model.middle_block.1.",
                     "mid_trans.", put)

    # Output blocks 0..11 → up0..up3 (3 resnet+transformer pairs each).
    # up0←0,1,2 (2560→1280); up1←3,4,5 (2560→1280);
    # up2←6,7,8 (1920→640); up3←9,10,11 (960→320, no upsample).
    # The LDM upsample conv of each group's last block
    # (output_blocks.{2,5,8}.2.conv) is our group's final
    # ``upsampler`` conv — same position in the chain. (An earlier
    # map skipped these keys entirely; they are real weights and are
    # mapped like everything else.)
    up_map = [
        ("up0", [0, 1, 2]), ("up1", [3, 4, 5]),
        ("up2", [6, 7, 8]), ("up3", [9, 10, 11]),
    ]
    for uname, blocks in up_map:
        for li, bi in enumerate(blocks):
            p = f"model.diffusion_model.output_blocks.{bi}.0."
            o = f"{uname}.resnets.{li}."
            for k in ("in_layers.0.weight", "in_layers.0.bias",
                      "in_layers.2.weight", "in_layers.2.bias",
                      "emb_layers.1.weight", "emb_layers.1.bias",
                      "out_layers.0.weight", "out_layers.0.bias",
                      "out_layers.3.weight", "out_layers.3.bias",
                      "skip_connection.weight", "skip_connection.bias"):
                ours_k = (k.replace("in_layers.0.", "norm1.")
                          .replace("in_layers.2.", "conv1.")
                          .replace("emb_layers.1.", "temb_proj.1.")
                          .replace("out_layers.0.", "norm2.")
                          .replace("out_layers.3.", "conv2.")
                          .replace("skip_connection.", "shortcut."))
                put(p + k, o + ours_k)
            _map_transformer(
                state, f"model.diffusion_model.output_blocks.{bi}.1.",
                f"{uname}.transformers.{li}.", put)
        last = blocks[-1]
        put(f"model.diffusion_model.output_blocks.{last}.2.conv.weight",
            f"{uname}.upsampler.weight")
        put(f"model.diffusion_model.output_blocks.{last}.2.conv.bias",
            f"{uname}.upsampler.bias")

    put("model.diffusion_model.out.0.weight", "conv_norm_out.weight")
    put("model.diffusion_model.out.0.bias", "conv_norm_out.bias")
    put("model.diffusion_model.out.2.weight", "conv_out.weight")
    put("model.diffusion_model.out.2.bias", "conv_out.bias")
    return out, mapper.consumed


def _map_transformer(state, prefix: str, out_prefix: str, put) -> None:
    """Map one LDM transformer block → our SpatialTransformer."""
    put(prefix + "norm.weight", out_prefix + "norm.weight")
    put(prefix + "norm.bias", out_prefix + "norm.bias")
    put(prefix + "proj_in.weight", out_prefix + "proj_in.weight")
    put(prefix + "proj_in.bias", out_prefix + "proj_in.bias")
    put(prefix + "proj_out.weight", out_prefix + "proj_out.weight")
    put(prefix + "proj_out.bias", out_prefix + "proj_out.bias")
    for bi in range(1):  # SD1.5 depth=1 per transformer
        bp = f"{prefix}transformer_blocks.{bi}."
        bo = f"{out_prefix}blocks.{bi}."
        put(bp + "norm1.weight", bo + "norm1.weight")
        put(bp + "norm1.bias", bo + "norm1.bias")
        put(bp + "norm2.weight", bo + "norm2.weight")
        put(bp + "norm2.bias", bo + "norm2.bias")
        put(bp + "norm3.weight", bo + "norm3.weight")
        put(bp + "norm3.bias", bo + "norm3.bias")
        for attn, ours in (("attn1", "attn1"), ("attn2", "attn2")):
            for proj in ("to_q", "to_k", "to_v"):
                put(f"{bp}{attn}.{proj}.weight",
                    f"{bo}{ours}.{proj}.weight")
            put(f"{bp}{attn}.to_out.0.weight",
                f"{bo}{ours}.to_out.0.weight")
            put(f"{bp}{attn}.to_out.0.bias",
                f"{bo}{ours}.to_out.0.bias")
        put(bp + "ff.net.0.proj.weight", bo + "ff.net.0.proj.weight")
        put(bp + "ff.net.0.proj.bias", bo + "ff.net.0.proj.bias")
        put(bp + "ff.net.2.weight", bo + "ff.net.2.weight")
        put(bp + "ff.net.2.bias", bo + "ff.net.2.bias")


class SD15Pipeline:
    """Text-to-image with SD1.5 open weights in Devon's own code.

    UNet operates in VAE latent space (64×64×4 for 512px); the VAE
    decodes to pixels. Same scheduler math as the native pipeline.
    """

    def __init__(self, unet: SDUNet, text_encoder: CLIPTextEncoder,
                 vae: AutoencoderKL, timesteps: int = 1000,
                 device: str = "") -> None:
        self.unet = unet.eval()
        self.text_encoder = text_encoder.eval()
        self.vae = vae.eval()
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        for m in (self.unet, self.text_encoder, self.vae):
            m.to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
        self.timesteps = timesteps

    @property
    def tokenizer_mode(self) -> str:
        """How prompts are tokenized: ``"bpe"`` or the fallback."""
        return self.text_encoder.tokenizer_mode

    @property
    def warnings(self) -> list[str]:
        """Standing disclosures about this pipeline's fidelity."""
        if self.tokenizer_mode == "bpe":
            return []
        return [
            "tokenizer: hashing fallback active — no CLIP BPE vocab "
            "found near this checkpoint, so prompts are tokenized "
            "approximately and fidelity is reduced. Add vocab.json + "
            "merges.txt (or tokenizer.json) next to the checkpoint "
            "or in ~/.devon-models/tokenizer/ for true CLIP BPE."]

    @classmethod
    def from_safetensors(cls, path: str, device: str = "") -> "SD15Pipeline":
        """Load an SD1.5 .safetensors checkpoint (compvis key format)."""
        state = load_safetensors(path)
        keys = set(state)
        is_ldm = any(k.startswith("model.diffusion_model.") for k in keys)
        is_diffusers = any(k.startswith("unet.") for k in keys)
        if not is_ldm and not is_diffusers:
            # Heuristic: SDXL has different channel counts.
            raise ImgGenError(
                f"{path}: unrecognized SD key format "
                "(need compvis LDM or diffusers SD1.5 keys)")
        if is_diffusers and not is_ldm:
            raise ImgGenError(
                f"{path}: diffusers-format keys detected — convert to "
                "compvis LDM format first (one-time script; the loader "
                "speaks LDM natively)")
        unet = SDUNet()
        mapped, consumed = _map_ldm_unet(state)
        missing, _unexpected = unet.load_state_dict(mapped, strict=False)
        if missing:
            raise ImgGenError(
                f"{path}: {len(missing)} UNet weights missing "
                f"(e.g. {missing[0]}); is this really SD1.5?")
        _check_section(state, "model.diffusion_model.", consumed,
                       f"{path} (UNet)")
        text_encoder = CLIPTextEncoder()
        _load_ldm_clip(state, text_encoder)
        # Real CLIP BPE if vocab data sits near the checkpoint; the
        # hashing fallback otherwise — disclosed via .warnings.
        from .tokenizers import load_clip_tokenizer

        text_encoder.tokenizer = load_clip_tokenizer(path)
        vae = AutoencoderKL()
        _load_ldm_vae(state, vae)
        return cls(unet, text_encoder, vae, device=device)

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 30,
                 guidance_scale: float = 7.5, seed: int | None = None,
                 width: int = 512, height: int = 512,
                 negative_prompt: str = "", eta: float = 0.0) -> list:
        """Latent diffusion sampling → PIL images."""
        from PIL import Image

        from .diffusion import DDIMScheduler

        sched = DDIMScheduler(timesteps=self.timesteps,
                              schedule="linear")
        lw, lh = width // 8, height // 8
        ctx = self.text_encoder.encode([prompt]).to(self.device)
        uncond = self.text_encoder.encode(
            [negative_prompt or ""]).to(self.device)
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        xt = torch.randn((1, 4, lh, lw), generator=g,
                         device=self.device)
        ts = torch.linspace(self.timesteps - 1, 0, steps,
                            dtype=torch.long)
        for t in ts:
            ti = int(t)
            t_batch = torch.full((2,), ti, dtype=torch.long,
                                 device=self.device)
            x_in = torch.cat([xt, xt], dim=0)
            context = torch.cat([ctx, uncond], dim=0)
            eps_c, eps_u = self.unet(x_in, t_batch,
                                     context).chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            xt = sched.p_sample(lambda _x, _t: eps, xt, ti, eta=eta)
        img = self.vae.decode(xt)
        arr = (img[0].clamp(-1, 1) * 0.5 + 0.5) * 255
        arr = arr.byte().permute(1, 2, 0).cpu().numpy()
        return [Image.fromarray(arr)]


#: CLIP keys present in real checkpoints that Devon's encoder
#: legitimately has no target for: SD conditions on the transformer's
#: final-LN sequence output, never on the projected pooled vector,
#: so text_projection is unused *by architecture* — named here so the
#: completeness check below can account for it explicitly instead of
#: dropping it silently.
_CLIP_KNOWN_UNUSED = {"cond_stage_model.transformer.text_projection"}


def _load_ldm_clip(state: dict, enc: CLIPTextEncoder) -> None:
    """Map cond_stage_model (OpenAI CLIP-L) keys → our encoder.

    Complete or explicit: every target parameter must be filled and
    every cond_stage_model key must be consumed (or named in
    _CLIP_KNOWN_UNUSED); anything else is an ImgGenError.
    """
    mapper = _KeyMapper(state)
    mapped = mapper.mapped

    def put(ldm, ours):
        mapper.put(ldm, ours)

    p = "cond_stage_model.transformer."
    put(p + "token_embedding.weight", "token_embedding.weight")
    put(p + "positional_embedding", "position_embedding.weight")
    for i in range(12):
        lp, op = f"{p}resblocks.{i}.", f"layers.{i}."
        put(lp + "ln_1.weight", op + "layer_norm1.weight")
        put(lp + "ln_1.bias", op + "layer_norm1.bias")
        put(lp + "ln_2.weight", op + "layer_norm2.weight")
        put(lp + "ln_2.bias", op + "layer_norm2.bias")
        # CLIP stores qkv as one in_proj; split into q/k/v.
        key = lp + "attn.in_proj_weight"
        if key in state:
            q, k, v = state[key].chunk(3, dim=0)
            mapped[op + "self_attn.q_proj.weight"] = q
            mapped[op + "self_attn.k_proj.weight"] = k
            mapped[op + "self_attn.v_proj.weight"] = v
            mapper.consumed.add(key)
        key = lp + "attn.in_proj_bias"
        if key in state:
            qb, kb, vb = state[key].chunk(3, dim=0)
            mapped[op + "self_attn.q_proj.bias"] = qb
            mapped[op + "self_attn.k_proj.bias"] = kb
            mapped[op + "self_attn.v_proj.bias"] = vb
            mapper.consumed.add(key)
        put(lp + "attn.out_proj.weight", op + "self_attn.out_proj.weight")
        put(lp + "attn.out_proj.bias", op + "self_attn.out_proj.bias")
        put(lp + "mlp.c_fc.weight", op + "mlp.0.weight")
        put(lp + "mlp.c_fc.bias", op + "mlp.0.bias")
        put(lp + "mlp.c_proj.weight", op + "mlp.2.weight")
        put(lp + "mlp.c_proj.bias", op + "mlp.2.bias")
    put(p + "ln_final.weight", "final_layer_norm.weight")
    put(p + "ln_final.bias", "final_layer_norm.bias")
    missing, _ = enc.load_state_dict(mapped, strict=False)
    if missing:
        raise ImgGenError(
            f"CLIP weights incomplete: {len(missing)} missing "
            f"(e.g. {missing[0]})")
    _check_section(state, "cond_stage_model.", mapper.consumed,
                   "CLIP text encoder", known_unused=_CLIP_KNOWN_UNUSED)


def _load_ldm_vae(state: dict, vae: AutoencoderKL) -> None:
    """Map the COMPLETE first_stage_model key set → our AutoencoderKL.

    The mapping is total by construction (AutoencoderKL mirrors the
    LDM VAE block for block, including the mid attention the old
    loader approximated away):

    - ``encoder.conv_in`` → ``enc_in``; ``encoder.down.{0,1,2,3}``
      (``block.{0,1}`` resnets each; ``downsample.conv`` on levels
      0–2 only) → ``enc_b{1,2,3,4}`` (resnets at ``.{0,1}``,
      downsample conv at ``.2``);
    - ``encoder.mid`` / ``decoder.mid``: ``block_1`` → ``.0``,
      ``attn_1`` → ``.1``, ``block_2`` → ``.2`` of ``enc_mid`` /
      ``dec_mid``;
    - ``quant_conv`` → ``quant_conv``; ``post_quant_conv`` →
      ``post_quant``;
    - ``decoder.up.{3,2,1,0}`` (``block.{0,1,2}`` resnets each;
      ``upsample.conv`` on levels 3–1 only) → ``dec_b{1,2,3,4}``
      (resnets at ``.{0,1,2}``, upsample conv at ``.3.conv``);
    - ``decoder.norm_out`` → ``dec_out.0``; ``decoder.conv_out`` →
      ``dec_out.2``.

    Completeness is enforced in both directions: an unfilled target
    parameter OR an unconsumed first_stage_model key raises
    ImgGenError naming the key. There is no partial-map degrade.
    """
    mapper = _KeyMapper(state)
    put = mapper.put
    p = "first_stage_model."

    def resnet(ldm_prefix: str, ours_prefix: str) -> None:
        for sub in ("norm1", "conv1", "norm2", "conv2"):
            put(ldm_prefix + sub + ".weight", ours_prefix + sub + ".weight")
            put(ldm_prefix + sub + ".bias", ours_prefix + sub + ".bias")
        # LDM calls the 1×1 channel-change conv "nin_shortcut"; ours
        # is "shortcut". Absent on both sides when channels match.
        put(ldm_prefix + "nin_shortcut.weight",
            ours_prefix + "shortcut.weight")
        put(ldm_prefix + "nin_shortcut.bias",
            ours_prefix + "shortcut.bias")

    def attn(ldm_prefix: str, ours_prefix: str) -> None:
        for sub in ("norm", "q", "k", "v", "proj_out"):
            put(ldm_prefix + sub + ".weight", ours_prefix + sub + ".weight")
            put(ldm_prefix + sub + ".bias", ours_prefix + sub + ".bias")

    def mid(ldm_prefix: str, ours_prefix: str) -> None:
        resnet(ldm_prefix + "block_1.", ours_prefix + "0.")
        attn(ldm_prefix + "attn_1.", ours_prefix + "1.")
        resnet(ldm_prefix + "block_2.", ours_prefix + "2.")

    # Encoder.
    put(p + "encoder.conv_in.weight", "enc_in.weight")
    put(p + "encoder.conv_in.bias", "enc_in.bias")
    for i in range(4):
        ours = f"enc_b{i + 1}."
        ldm = f"{p}encoder.down.{i}."
        resnet(ldm + "block.0.", ours + "0.")
        resnet(ldm + "block.1.", ours + "1.")
        if i < 3:  # the last LDM down level has no downsample
            put(ldm + "downsample.conv.weight", ours + "2.weight")
            put(ldm + "downsample.conv.bias", ours + "2.bias")
    mid(p + "encoder.mid.", "enc_mid.")
    put(p + "quant_conv.weight", "quant_conv.weight")
    put(p + "quant_conv.bias", "quant_conv.bias")

    # Decoder.
    put(p + "post_quant_conv.weight", "post_quant.weight")
    put(p + "post_quant_conv.bias", "post_quant.bias")
    mid(p + "decoder.mid.", "dec_mid.")
    for i, up_idx in enumerate((3, 2, 1, 0)):
        ours = f"dec_b{i + 1}."
        ldm = f"{p}decoder.up.{up_idx}."
        resnet(ldm + "block.0.", ours + "0.")
        resnet(ldm + "block.1.", ours + "1.")
        resnet(ldm + "block.2.", ours + "2.")
        if up_idx > 0:  # up.0 is the final level: no upsample
            put(ldm + "upsample.conv.weight", ours + "3.conv.weight")
            put(ldm + "upsample.conv.bias", ours + "3.conv.bias")
    put(p + "decoder.norm_out.weight", "dec_out.0.weight")
    put(p + "decoder.norm_out.bias", "dec_out.0.bias")
    put(p + "decoder.conv_out.weight", "dec_out.2.weight")
    put(p + "decoder.conv_out.bias", "dec_out.2.bias")

    missing, _unexpected = vae.load_state_dict(mapper.mapped,
                                               strict=False)
    if missing:
        raise ImgGenError(
            f"VAE weights incomplete: {len(missing)} target keys "
            f"unfilled (e.g. {missing[0]}); the checkpoint's "
            f"first_stage_model does not match the SD VAE layout")
    _check_section(state, p, mapper.consumed, "VAE")


# ---------------------------------------------------------------------------
# Studio protocol adapter
# ---------------------------------------------------------------------------

class SD15StudioAdapter:
    """Adapt :class:`SD15Pipeline` to the protocol Studio speaks.

    Studio (and ``edit.py`` / ``upscale.py`` through it) drives a
    pipeline as ``generate(prompt, PipelineConfig)`` plus attribute
    access (``.device``, ``.unet``, ``.text_encoder``,
    ``._scheduler()``, ``._to_pil()``). SD1.5 generation maps onto
    that cleanly; the *edit* operations do not — they are written
    against the native pipeline's pixel-space UNet, while SD1.5
    edits belong in VAE latent space. Rather than fake them, every
    edit-protocol surface raises :class:`ImgGenError` naming the
    operation and the model class; latent-space SD editing is its
    own later piece of work.
    """

    model_class = "sd15"

    def __init__(self, pipeline: "SD15Pipeline") -> None:
        self._pipe = pipeline
        #: Disclosures from the most recent generate() call; Studio
        #: reads this to surface them (sidecar + log).
        self.last_warnings: list[str] = list(pipeline.warnings)

    # -- honest passthroughs -------------------------------------------
    @property
    def device(self) -> str:
        return self._pipe.device

    @property
    def tokenizer_mode(self) -> str:
        return self._pipe.tokenizer_mode

    @property
    def warnings(self) -> list[str]:
        return list(self._pipe.warnings)

    @property
    def controlnets(self) -> dict:
        return {}

    # -- generation ------------------------------------------------------
    def generate(self, prompt, cfg=None, conditioning=None) -> list:
        """PipelineConfig → SD15Pipeline.generate kwargs, faithfully."""
        from .pipeline import PipelineConfig

        cfg = cfg or PipelineConfig()
        if conditioning:
            raise ImgGenError(
                "ControlNet conditioning is not available for SD1.5 "
                "(sd15) checkpoints through the Studio adapter in "
                "this pass — the SD pipeline has no ControlNet branch "
                "wired; refusing to silently ignore the control input")
        if cfg.sampler != "ddim":
            raise ImgGenError(
                f"SD1.5 (sd15) checkpoints sample with DDIM only; "
                f"sampler {cfg.sampler!r} is not available for this "
                f"model class")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        out: list = []
        for text in prompts:
            for i in range(max(1, cfg.batch_size)):
                seed = None if cfg.seed is None else cfg.seed + i
                out.extend(self._pipe.generate(
                    text, steps=cfg.steps,
                    guidance_scale=cfg.guidance_scale, seed=seed,
                    width=cfg.width, height=cfg.height,
                    negative_prompt=cfg.negative_prompt, eta=cfg.eta))
        self.last_warnings = list(self._pipe.warnings)
        return out

    # -- edit protocol: honest gates, never fake results -----------------
    def _edit_gate(self, what: str):
        raise ImgGenError(
            f"{what} is not available for SD1.5 (sd15) checkpoints "
            f"through the Studio adapter: Devon's edit operations run "
            f"against the native pixel-space pipeline, and SD1.5 "
            f"img2img/inpaint/outpaint need a latent-space adapter "
            f"that does not exist yet. Text-to-image works; the edit "
            f"was refused rather than faked.")

    @property
    def unet(self):
        return self._edit_gate("the UNet (edit protocol)")

    @property
    def text_encoder(self):
        return self._edit_gate("the text encoder (edit protocol)")

    def _scheduler(self, sampler: str):
        return self._edit_gate(f"the {sampler} scheduler "
                               f"(edit protocol)")

    def _to_pil(self, tensor):
        return self._edit_gate("tensor→PIL conversion (edit protocol)")

    def attach_controlnet(self, kind, controlnet):
        return self._edit_gate(f"ControlNet attach ({kind})")

    def attach_diffusers_controlnet(self, kind, diffusers_controlnet):
        return self._edit_gate(f"ControlNet attach ({kind})")
