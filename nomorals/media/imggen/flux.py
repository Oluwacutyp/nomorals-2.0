"""FLUX.1 text-to-image compatibility (BFL single-file format).

The pipeline code is ours; weights are data. This module implements
the FLUX.1 MMDiT transformer, its T5-XXL text encoder (SentencePiece)
and CLIP-L pooled encoder, and the FLUX VAE **from their documented
architectures** and loads ``.safetensors`` checkpoints in the BFL
single-file key format (``double_blocks.*`` / ``single_blocks.*``,
optionally under a ``model.diffusion_model.`` prefix) — the format
black-forest-labs ships for FLUX.1-dev.

Architecture, verified against the live BFL reference
(github.com/black-forest-labs/flux, src/flux/model.py and
src/flux/modules/layers.py, read 2026-11-17) — the HF config for
FLUX.1-dev itself is gated (401: the repo requires accepting the
FLUX.1-dev Non-Commercial License first), so the BFL source is the
primary evidence and the diffusers FLUX.1-dev config values are the
cross-check:

- Transformer (MMDiT): hidden 3072, 24 heads (head dim 128),
  19 double-stream blocks + 38 single-stream blocks, context 4096
  (T5-XXL), pooled 768 (CLIP-L), in_channels 64, out 64,
  axes_dim [16, 56, 56], theta 10000, qkv_bias True, guidance
  embedding present (dev only; schnell has none). One double
  modulation produces BOTH modulation outs (6 x 3072 rows); one
  single modulation produces one (3 x 3072 rows). The single
  blocks concatenate attention (3 x 3072) + MLP (4 x 3072) into ONE
  linear (21504 rows in). RoPE is per-axis (t, h, w) with the
  rotation matrices [[cos, -sin], [sin, cos]]; timestep embedding
  is cos-first with time_factor 1000 (unlike SD's sin-first).
- Text: T5-XXL (google/t5-v1_1-xxl: d_model 4096, d_ff 10240,
  64 heads, 24 layers, vocab 32128) with the v1.1 GATED-GELU
  feed-forward (wi_0 * gelu(wi_1) — NOT relu), SentencePiece
  tokenizer (512 tokens, pad 0), and a relative-position bias
  carried by block 0 only. CLIP-L (transformers CLIPTextModel
  layout) supplies the 768-wide pooled vector.
- VAE: the diffusers AutoencoderKL layout (ae.safetensors):
  16 latent channels, scale 0.3611, shift 0.1159, channels
  [128, 256, 512, 512], two resnets per decoder level, split I/O
  convs (encoder conv_out 3x3 + quant_conv 1x1; post_quant_conv
  1x1 + decoder conv_in 3x3) and an encoder norm_out + SiLU.
  The sdcompat VAE grows exactly those (``diffusers_layout`` +
  ``decoder_resnets`` + ``shift``), defaults unchanged.
- Sampling: rectified flow (FlowMatchScheduler) — BFL's
  get_schedule: linspace 1 -> 0, dynamic shift (mu from the image
  sequence length, base 0.5 / max 1.15, base_seq 256 / max_seq
  4096), Euler step x += (t_prev - t_curr) * v. The static shift
  3.0 some callers pass is accepted but unused under dynamic
  shifting, as upstream.

Companions resolve at load time: the T5 file
(``t5xxl_fp16.safetensors`` / ``t5xxl.safetensors``), the CLIP-L
file (``clip_l.safetensors``), and the VAE (``ae.safetensors``) are
searched in the transformer's own directory, then
:func:`checkpoint_dir`, then ``~/.devon-models``. A missing
companion is an ImgGenError naming the file and every directory
searched — never a silent fallback.

Honest scope: the full FLUX.1-dev text-to-image loader
(transformer + T5 + CLIP-L + VAE + flow-match Euler). No
diffusers-transformer keys — the error names the one-time convert
to the BFL single-file layout, as the SDXL loader does for its own
diffusers layout. No fp16/bf16 (pass C). FLUX.1-dev is
guidance-distilled: there is no CFG branch, so a negative prompt is
accepted, ignored, and disclosed by the adapter — never rendered.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from . import TORCH_AVAILABLE, ImgGenError, checkpoint_dir

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.flux needs PyTorch: pip install torch")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline import PipelineConfig
from .sdcompat import (
    AutoencoderKL,
    CLIPTextEncoder,
    _check_section,
    _KeyMapper,
    load_safetensors,
)
from .tokenizers import load_clip_tokenizer

# ---------------------------------------------------------------------------
# FLUX.1-dev dimensions (BFL FluxParams for the dev checkpoint)
# ---------------------------------------------------------------------------

FLUX_HIDDEN = 3072
FLUX_HEADS = 24
FLUX_HEAD_DIM = 128
FLUX_DEPTH = 19
FLUX_DEPTH_SINGLE = 38
FLUX_CONTEXT_DIM = 4096  # T5-XXL
FLUX_POOLED_DIM = 768  # CLIP-L
FLUX_IN_CHANNELS = 64
FLUX_AXES_DIM = [16, 56, 56]
FLUX_THETA = 10000
FLUX_MLP_RATIO = 4.0

T5_D_MODEL = 4096
T5_D_FF = 10240
T5_HEADS = 64
T5_LAYERS = 24
T5_VOCAB = 32128
T5_MAX_TOKENS = 512

FLUX_VAE_SCALE = 0.3611
FLUX_VAE_SHIFT = 0.1159

T5_FILE_NAMES = ("t5xxl_fp16.safetensors", "t5xxl.safetensors")
CLIP_FILE_NAMES = ("clip_l.safetensors",)
VAE_FILE_NAMES = ("ae.safetensors",)


# ---------------------------------------------------------------------------
# RoPE (BFL flux/math.py: rope / apply_rope, einops-free)
# ---------------------------------------------------------------------------

def _rope(pos, dim, theta=10000):
    """(..., L, dim//2, 2, 2) rotation matrices for one axis.

    BFL computes the frequencies in float64 and returns float32.
    """
    scale = (torch.arange(0, dim, 2, dtype=torch.float64,
                          device=pos.device) / dim)
    omega = 1.0 / (theta ** scale)
    out = pos.unsqueeze(-1).float() * omega
    cos = out.cos()
    sin = out.sin()
    mat = torch.stack(
        [torch.stack([cos, -sin], dim=-1),
         torch.stack([sin, cos], dim=-1)],
        dim=-2)
    return mat.float()


def _apply_rope(xq, xk, freqs_cis):
    """Rotate q and k by the precomputed RoPE matrices."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = (freqs_cis[..., 0] * xq_[..., 0]
              + freqs_cis[..., 1] * xq_[..., 1])
    xk_out = (freqs_cis[..., 0] * xk_[..., 0]
              + freqs_cis[..., 1] * xk_[..., 1])
    return (xq_out.reshape(*xq.shape).type_as(xq),
            xk_out.reshape(*xk.shape).type_as(xk))


class EmbedND(nn.Module):
    """Per-axis RoPE: (B, L, 3) ids -> (B, 1, L, head_dim//2, 2, 2)."""

    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta)
             for i in range(n_axes)],
            dim=-3)
        return emb.unsqueeze(1)


def _flux_timestep_embedding(t, dim, max_period=10000,
                             time_factor=1000.0):
    """Sinusoidal embedding, cos-first, t scaled by 1000 (BFL).

    Unlike SD's timestep embedding (sin-first), FLUX puts cosine
    first — and unlike SD the input lives in [0, 1], so the
    time_factor 1000 is what makes the frequencies land.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32,
                       device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """Linear -> SiLU -> Linear (time_in / vector_in / guidance_in)."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    """Root-mean-square norm. The parameter is named ``scale`` —
    that is the checkpoint key (``...norm.query_norm.scale``)."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """Separate RMSNorms for q and k (BFL)."""

    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def _attention(q, k, v, pe):
    """RoPE + scaled dot-product attention, (B, H, L, D) in/out."""
    q, k = _apply_rope(q, k, pe)
    x = F.scaled_dot_product_attention(q, k, v)
    return x.transpose(1, 2).reshape(
        x.shape[0], x.shape[2], x.shape[1] * x.shape[3])


class SelfAttention(nn.Module):
    """Fused-qkv attention. The double-stream blocks inline its
    qkv / norm / proj (they need joint txt+img attention); this
    forward is the single-sequence case."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pe):
        qkv = self.qkv(x)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3,
                           self.num_heads, -1).unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = self.norm(q, k, v)
        return self.proj(_attention(q, k, v, pe))


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """One linear producing shift/scale/gate (x1) or two outs (x2).

    ``double=True`` -> 6 x dim rows, two ModulationOuts; False ->
    3 x dim rows, one ModulationOut. The double-stream blocks hold
    ONE Modulation per stream (not two) — the single linear
    produces both outs.
    """

    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(F.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]),
                ModulationOut(*out[3:]) if self.is_double else None)


class DoubleStreamBlock(nn.Module):
    """MMDiT double-stream block: separate img/txt weights, joint
    attention over the concatenated sequence (txt first)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio,
                 qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, vec, pe):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        img_modulated = self.img_norm1(img)
        img_modulated = ((1 + img_mod1.scale) * img_modulated
                         + img_mod1.shift)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3,
            self.num_heads, -1).unbind(dim=2)
        img_q = img_q.permute(0, 2, 1, 3)
        img_k = img_k.permute(0, 2, 1, 3)
        img_v = img_v.permute(0, 2, 1, 3)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = ((1 + txt_mod1.scale) * txt_modulated
                         + txt_mod1.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3,
            self.num_heads, -1).unbind(dim=2)
        txt_q = txt_q.permute(0, 2, 1, 3)
        txt_k = txt_k.permute(0, 2, 1, 3)
        txt_v = txt_v.permute(0, 2, 1, 3)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = _attention(q, k, v, pe)
        txt_attn = attn[:, : txt.shape[1]]
        img_attn = attn[:, txt.shape[1]:]
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img)
            + img_mod2.shift)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt)
            + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """DiT block with parallel linear layers (arXiv:2302.05442):
    one linear produces qkv + the MLP input, a second produces the
    output from attention + activated MLP side by side."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        # BFL computes this scale but the attention call does not
        # use it (SDPA's own 1/sqrt(head_dim) applies); kept for
        # parity with the reference, not as a live knob.
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, vec, pe):
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3,
                           self.num_heads, -1).unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = self.norm(q, k, v)
        attn = _attention(q, k, v, pe)
        output = self.linear2(
            torch.cat((attn, self.mlp_act(mlp)), dim=2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    """Final adaLN + projection back to the packed latent (64)."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels,
            bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) \
            + shift[:, None, :]
        return self.linear(x)


class FluxTransformer(nn.Module):
    """The FLUX.1 transformer (BFL's ``Flux``), dev defaults.

    Constructor kwargs are the BFL FluxParams flattened; the small
    sizes exist so tests can run a real forward on CPU, the dev
    sizes are the defaults.
    """

    def __init__(self, hidden_size=FLUX_HIDDEN, num_heads=FLUX_HEADS,
                 depth=FLUX_DEPTH,
                 depth_single_blocks=FLUX_DEPTH_SINGLE,
                 in_channels=FLUX_IN_CHANNELS,
                 out_channels=FLUX_IN_CHANNELS,
                 vec_in_dim=FLUX_POOLED_DIM,
                 context_in_dim=FLUX_CONTEXT_DIM,
                 mlp_ratio=FLUX_MLP_RATIO,
                 axes_dim=None, theta=FLUX_THETA, qkv_bias=True,
                 guidance_embed=True):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ImgGenError(
                f"Hidden size {hidden_size} must be divisible by "
                f"num_heads {num_heads}")
        pe_dim = hidden_size // num_heads
        axes_dim = list(axes_dim) if axes_dim is not None \
            else list(FLUX_AXES_DIM)
        if sum(axes_dim) != pe_dim:
            raise ImgGenError(
                f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = pe_dim
        self.axes_dim = axes_dim
        self.theta = theta
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guidance_embed = guidance_embed
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta,
                                   axes_dim=axes_dim)
        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
            if guidance_embed else nn.Identity())
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
             for _ in range(depth)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads,
                               mlp_ratio=mlp_ratio)
             for _ in range(depth_single_blocks)])
        self.final_layer = LastLayer(hidden_size, 1, out_channels)

    def forward(self, img, img_ids, txt, txt_ids, timesteps, y,
                guidance=None):
        if img.ndim != 3 or txt.ndim != 3:
            raise ImgGenError(
                "Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        vec = self.time_in(_flux_timestep_embedding(timesteps, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ImgGenError(
                    "Didn't get guidance strength for guidance "
                    "distilled model.")
            vec = vec + self.guidance_in(
                _flux_timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]
        return self.final_layer(img, vec)


# ---------------------------------------------------------------------------
# Latent packing (BFL sampling.prepare / unpack)
# ---------------------------------------------------------------------------

def pack_latents(x):
    """(B, C, H, W) -> (B, (H/2)*(W/2), C*4): 2x2 pixel blocks."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(x, height, width):
    """Packed tokens -> (B, C, H, W) for a latent grid of
    height x width (pixel height // 8, pixel width // 8)."""
    b = x.shape[0]
    h = height // 2
    w = width // 2
    x = x.reshape(b, h, w, -1, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, -1, height, width)


def _image_ids(latent_h, latent_w):
    """(latent_h//2 * latent_w//2, 3) RoPE ids: (t=0, h, w)."""
    ids = torch.zeros(latent_h // 2, latent_w // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(
        latent_h // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(
        latent_w // 2)[None, :]
    return ids.reshape(-1, 3)


# ---------------------------------------------------------------------------
# FlowMatchScheduler (BFL sampling.get_schedule + Euler step)
# ---------------------------------------------------------------------------

def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _lin_mu(image_seq_len, base_shift=0.5, max_shift=1.15,
            base_seq=256, max_seq=4096):
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return m * image_seq_len + b


class FlowMatchScheduler:
    """Rectified-flow schedule: linspace 1 -> 0, dynamic shift by
    image sequence length (BFL get_schedule), Euler integration.

    ``set_timesteps`` returns the full sigma list (n + 1 entries,
    first 1.0, last 0.0). The static shift 3.0 some callers pass is
    accepted but unused under dynamic shifting, as upstream.
    """

    def __init__(self):
        self.timesteps = None

    def set_timesteps(self, num_steps, image_seq_len=None,
                      shift=True, static_shift=3.0):
        timesteps = torch.linspace(1, 0, num_steps + 1)
        if shift and image_seq_len is not None:
            mu = _lin_mu(image_seq_len)
            timesteps = _time_shift(mu, 1.0, timesteps)
        self.timesteps = timesteps
        return timesteps

    def step(self, v, t_curr, t_prev, x):
        """One Euler step: x += (t_prev - t_curr) * v."""
        return x + (t_prev - t_curr) * v

# ---------------------------------------------------------------------------
# T5 text encoder (transformers T5EncoderModel layout, v1.1 gated-gelu)
# ---------------------------------------------------------------------------

def _relative_position_bucket(relative_position, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """transformers' bucketing: near positions exact, far ones log."""
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class T5LayerNorm(nn.Module):
    """T5's RMSNorm: no mean subtraction, weight only, eps 1e-6."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5Attention(nn.Module):
    """T5 self-attention: separate q/k/v/o, no bias, no causal mask
    (the encoder is bidirectional). Block 0 alone carries the
    relative-position bias embedding (32 buckets)."""

    def __init__(self, d_model, num_heads,
                 has_relative_attention_bias=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                32, num_heads)

    def _compute_bias(self, seq_len, device):
        position_bias = torch.arange(
            seq_len, dtype=torch.long, device=device)
        ctx = position_bias[:, None]
        mem = position_bias[None, :]
        rel = _relative_position_bucket(mem - ctx)
        values = self.relative_attention_bias(rel)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x, position_bias=None):
        b, seq_len, _ = x.shape
        q = self.q(x).view(b, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        if position_bias is None:
            if not hasattr(self, "relative_attention_bias"):
                raise ImgGenError(
                    "T5 block without a relative-position bias was "
                    "asked to compute one — the first block owns it")
            position_bias = self._compute_bias(seq_len, x.device)
        scores = scores + position_bias
        scores = scores / math.sqrt(self.head_dim)
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, seq_len, self.d_model)
        return self.o(out), position_bias


class T5LayerSelfAttention(nn.Module):
    """One T5 encoder layer's attention half (transformers'
    T5LayerSelfAttention: ``.SelfAttention`` + ``.layer_norm``)."""

    def __init__(self, d_model, num_heads,
                 has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            d_model, num_heads, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, x, position_bias=None):
        normed = self.layer_norm(x)
        out, position_bias = self.SelfAttention(normed, position_bias)
        return x + out, position_bias


class T5DenseGatedGelu(nn.Module):
    """T5 v1.1 feed-forward: gated GELU, NOT relu.

    wi_0 and wi_1 both project up to d_ff; the output is
    GELU(wi_0(x)) * wi_1(x) projected back down by wo. A v1.0
    DenseReluDense would compute relu(wi(x)) here and be wrong.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        hidden = F.gelu(self.wi_0(x), approximate="tanh")
        hidden = hidden * self.wi_1(x)
        return self.wo(hidden)


class T5LayerFF(nn.Module):
    """One T5 encoder layer's feed-forward half (transformers'
    T5LayerFF: ``.ffn`` + ``.layer_norm``)."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = T5DenseGatedGelu(d_model, d_ff)
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, x):
        return x + self.ffn(self.layer_norm(x))


class T5Block(nn.Module):
    """One T5 encoder block: attention half + FF half."""

    def __init__(self, d_model, d_ff, num_heads,
                 has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(
                d_model, num_heads, has_relative_attention_bias),
            T5LayerFF(d_model, d_ff),
        ])

    def forward(self, x, position_bias=None):
        x, position_bias = self.layer[0](x, position_bias)
        x = self.layer[1](x)
        return x, position_bias


class T5TextEncoder(nn.Module):
    """google/t5-v1_1-xxl encoder (transformers T5EncoderModel).

    Constructor kwargs are the config values; the small sizes exist
    so tests can run a real forward on CPU, the XXL sizes are the
    defaults.
    """

    def __init__(self, d_model=T5_D_MODEL, d_ff=T5_D_FF,
                 num_heads=T5_HEADS, num_layers=T5_LAYERS,
                 vocab_size=T5_VOCAB):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.shared = nn.Embedding(vocab_size, d_model)
        self.block = nn.ModuleList([
            T5Block(d_model, d_ff, num_heads,
                    has_relative_attention_bias=(i == 0))
            for i in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(d_model)
        self.tokenizer = None

    @property
    def tokenizer_mode(self):
        return "spiece" if self.tokenizer is not None \
            else "hashing-fallback"

    def forward(self, ids):
        x = self.shared(ids)
        position_bias = None
        for block in self.block:
            x, position_bias = block(x, position_bias)
        return self.final_layer_norm(x)

    def encode(self, prompts):
        """[prompt] -> (B, 512, d_model) via SentencePiece or the
        hashing fallback (disclosed in the pipeline warnings)."""
        if self.tokenizer is not None:
            rows = []
            for p in prompts:
                ids = self.tokenizer.encode(p)[:T5_MAX_TOKENS]
                ids = ids + [0] * (T5_MAX_TOKENS - len(ids))
                rows.append(ids)
        else:
            rows = [_hash_tokenize_t5(p, self.vocab_size,
                                      T5_MAX_TOKENS)
                    for p in prompts]
        ids = torch.tensor(rows, dtype=torch.long,
                           device=self.shared.weight.device)
        return self(ids)


def _hash_tokenize_t5(text, vocab_size, max_tokens):
    """Deterministic word-hash tokenization (fallback only).

    Same shape as the CLIP hashing fallback: words hashed into
    [1, vocab_size), padded with 0. Prompt fidelity is reduced —
    the pipeline says so in its warnings.
    """
    import hashlib

    ids = []
    for word in (text or "").split():
        h = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(1 + int.from_bytes(h[:4], "little")
                   % (vocab_size - 1))
    ids = ids[:max_tokens]
    return ids + [0] * (max_tokens - len(ids))


def _load_hf_t5(state, dtype: str | None = None):
    """Map transformers T5EncoderModel keys -> T5TextEncoder.

    Renames are exactly two: the ``encoder.`` prefix, and
    ``layer.1.DenseReluDense.`` -> ``layer.1.ffn.`` (the class is
    gated-gelu in v1.1, whatever transformers calls it). Sizes are
    read off the checkpoint itself (XXL for a real T5-XXL file) so
    the loader matches the weights, not an assumption.
    Completeness is enforced both ways. ``dtype`` casts the
    encoder before its weights load (see dtype.cast_module).
    """
    from .dtype import cast_module

    shared = state.get("shared.weight")
    if shared is None:
        raise ImgGenError(
            "T5 encoder: no shared.weight — the checkpoint is not "
            "a transformers T5EncoderModel state dict")
    vocab_size, d_model = (int(s) for s in shared.shape)
    block_idxs = set()
    for k in state:
        if k.startswith("encoder.block."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_idxs.add(int(parts[2]))
    if not block_idxs:
        raise ImgGenError(
            "T5 encoder: no encoder.block.* keys — the checkpoint "
            "is not a transformers T5EncoderModel state dict")
    num_layers = max(block_idxs) + 1
    wi_0 = state.get(
        "encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    rel_bias = state.get(
        "encoder.block.0.layer.0.SelfAttention."
        "relative_attention_bias.weight")
    if wi_0 is None or rel_bias is None:
        raise ImgGenError(
            "T5 encoder: block 0 is missing its feed-forward or "
            "relative-position bias — the checkpoint is not a "
            "transformers T5EncoderModel state dict")
    d_ff = int(wi_0.shape[0])
    num_heads = int(rel_bias.shape[1])
    enc = T5TextEncoder(d_model=d_model, d_ff=d_ff,
                        num_heads=num_heads, num_layers=num_layers,
                        vocab_size=vocab_size)
    if dtype:
        cast_module(enc, dtype)
    mapper = _KeyMapper(state)
    put = mapper.put
    put("shared.weight", "shared.weight")
    put("encoder.final_layer_norm.weight", "final_layer_norm.weight")
    for i in range(num_layers):
        src = f"encoder.block.{i}.layer.0.SelfAttention."
        dst = f"block.{i}.layer.0.SelfAttention."
        for tail in ("q.weight", "k.weight", "v.weight", "o.weight"):
            put(src + tail, dst + tail)
        if i == 0:
            put(src + "relative_attention_bias.weight",
                dst + "relative_attention_bias.weight")
        put(f"encoder.block.{i}.layer.0.layer_norm.weight",
            f"block.{i}.layer.0.layer_norm.weight")
        for tail in ("wi_0.weight", "wi_1.weight", "wo.weight"):
            put(f"encoder.block.{i}.layer.1.DenseReluDense.{tail}",
                f"block.{i}.layer.1.ffn.{tail}")
        put(f"encoder.block.{i}.layer.1.layer_norm.weight",
            f"block.{i}.layer.1.layer_norm.weight")
    missing, _ = enc.load_state_dict(mapper.mapped, strict=False)
    if missing:
        raise ImgGenError(
            f"T5 encoder: {len(missing)} parameters unfilled "
            f"(e.g. {missing[0]}) — the checkpoint is not a "
            f"transformers T5EncoderModel state dict")
    _check_section(state, "shared.weight", mapper.consumed, "T5 encoder")
    _check_section(state, "encoder.", mapper.consumed, "T5 encoder")
    return enc


def load_t5_tokenizer(t5_path):
    """(tokenizer, warning) — SentencePiece if the model file and
    the sentencepiece package are both present, else (None, warn).

    The spiece.model file is looked for next to the T5 weights,
    then in the checkpoint dir, then in ~/.devon-models.
    """
    dirs = []
    if t5_path:
        dirs.append(os.path.dirname(os.path.abspath(t5_path)))
    dirs.append(checkpoint_dir())
    dirs.append(os.path.join(os.path.expanduser("~"), ".devon-models"))
    seen = set()
    dirs = [d for d in dirs if not (d in seen or seen.add(d))]
    for d in dirs:
        cand = os.path.join(d, "spiece.model")
        if not os.path.exists(cand):
            continue
        try:
            import sentencepiece as spm
        except ImportError:
            return None, (
                "T5 tokenizer: spiece.model found but the "
                "sentencepiece package is not installed — hashing "
                "fallback (reduced prompt fidelity)")
        proc = spm.SentencePieceProcessor()
        proc.Load(cand)
        return proc, ""
    return None, (
        "T5 tokenizer: no spiece.model next to the T5 weights — "
        "hashing fallback (reduced prompt fidelity — add "
        "spiece.model next to the T5 file)")


def _load_hf_clip(state, dtype: str | None = None):
    """Map transformers CLIPTextModel keys -> CLIPTextEncoder.

    The pooled output is the EOT hidden state through
    ``text_projection`` (transformers stores it bias-free; ours
    keeps a zero bias, as A2 did for CLIP-G). Sizes are read off
    the checkpoint itself (768-wide, 12 layers for a real CLIP-L
    file) so the loader matches the weights, not an assumption.
    Completeness is enforced both ways. ``dtype`` casts the
    encoder before its weights load (see dtype.cast_module).
    """
    from .dtype import cast_module

    tok = state.get("text_model.embeddings.token_embedding.weight")
    if tok is None:
        raise ImgGenError(
            "CLIP-L encoder: no token embedding — the checkpoint "
            "is not a transformers CLIPTextModel state dict")
    dim = int(tok.shape[1])
    layer_idxs = set()
    for k in state:
        if k.startswith("text_model.encoder.layers."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                layer_idxs.add(int(parts[3]))
    if not layer_idxs:
        raise ImgGenError(
            "CLIP-L encoder: no encoder layer keys — the "
            "checkpoint is not a transformers CLIPTextModel state "
            "dict")
    num_layers = max(layer_idxs) + 1
    enc = CLIPTextEncoder(dim=dim, layers=num_layers)
    enc.text_projection = nn.Linear(dim, dim)
    if dtype:
        cast_module(enc, dtype)
    mapper = _KeyMapper(state)
    put = mapper.put
    put("text_model.embeddings.token_embedding.weight",
        "token_embedding.weight")
    put("text_model.embeddings.position_embedding.weight",
        "position_embedding.weight")
    put("text_projection.weight", "text_projection.weight")
    put("text_model.final_layer_norm.weight", "final_layer_norm.weight")
    put("text_model.final_layer_norm.bias", "final_layer_norm.bias")
    for i in range(num_layers):
        src = f"text_model.encoder.layers.{i}."
        dst = f"layers.{i}."
        for tail in ("self_attn.q_proj.weight",
                     "self_attn.q_proj.bias",
                     "self_attn.k_proj.weight",
                     "self_attn.k_proj.bias",
                     "self_attn.v_proj.weight",
                     "self_attn.v_proj.bias",
                     "self_attn.out_proj.weight",
                     "self_attn.out_proj.bias",
                     "layer_norm1.weight", "layer_norm1.bias",
                     "layer_norm2.weight", "layer_norm2.bias"):
            put(src + tail, dst + tail)
        put(src + "mlp.fc1.weight", dst + "mlp.0.weight")
        put(src + "mlp.fc1.bias", dst + "mlp.0.bias")
        put(src + "mlp.fc2.weight", dst + "mlp.2.weight")
        put(src + "mlp.fc2.bias", dst + "mlp.2.bias")
    missing, _ = enc.load_state_dict(mapper.mapped, strict=False)
    if missing:
        raise ImgGenError(
            f"CLIP-L encoder: {len(missing)} parameters unfilled "
            f"(e.g. {missing[0]}) — the checkpoint is not a "
            f"transformers CLIPTextModel state dict")
    _check_section(state, "text_model.", mapper.consumed, "CLIP-L")
    _check_section(state, "text_projection.", mapper.consumed, "CLIP-L")
    return enc


# ---------------------------------------------------------------------------
# VAE (diffusers AutoencoderKL layout -> sdcompat AutoencoderKL)
# ---------------------------------------------------------------------------

def _load_diffusers_vae(state, vae):
    """Map a diffusers AutoencoderKL state dict -> ``vae``.

    The diffusers VAE splits the I/O convs the LDM VAE folds into
    one (encoder conv_out 3x3 + quant_conv 1x1; post_quant_conv 1x1
    + decoder conv_in 3x3), adds an encoder norm_out + SiLU, and
    carries two resnets per decoder level where the LDM VAE carries
    three — so the target must be built with ``diffusers_layout``
    and the matching ``decoder_resnets`` (see AutoencoderKL).
    Completeness is enforced both ways.
    """
    mapper = _KeyMapper(state)
    put = mapper.put
    if not getattr(vae, "diffusers_layout", False):
        raise ImgGenError(
            "FLUX VAE: the target module was not built with "
            "diffusers_layout=True — the diffusers VAE's split I/O "
            "convs and encoder norm_out have nowhere to load")
    up_idx = getattr(vae, "decoder_resnets", 3)

    def resnet(src_prefix, dst_prefix):
        for sub in ("norm1", "conv1", "norm2", "conv2"):
            put(src_prefix + sub + ".weight",
                dst_prefix + sub + ".weight")
            put(src_prefix + sub + ".bias", dst_prefix + sub + ".bias")
        # Diffusers calls the 1x1 channel-change conv
        # "conv_shortcut"; ours is "shortcut". Absent on both sides
        # when channels match.
        put(src_prefix + "conv_shortcut.weight",
            dst_prefix + "shortcut.weight")
        put(src_prefix + "conv_shortcut.bias",
            dst_prefix + "shortcut.bias")

    def attn(src_prefix, dst_prefix):
        put(src_prefix + "group_norm.weight", dst_prefix + "norm.weight")
        put(src_prefix + "group_norm.bias", dst_prefix + "norm.bias")
        for s, d in (("query", "q"), ("key", "k"), ("value", "v")):
            put(src_prefix + s + ".weight", dst_prefix + d + ".weight")
            put(src_prefix + s + ".bias", dst_prefix + d + ".bias")
        put(src_prefix + "proj_attn.weight",
            dst_prefix + "proj_out.weight")
        put(src_prefix + "proj_attn.bias",
            dst_prefix + "proj_out.bias")

    put("encoder.conv_in.weight", "enc_in.weight")
    put("encoder.conv_in.bias", "enc_in.bias")
    put("encoder.conv_norm_out.weight", "enc_norm_out.weight")
    put("encoder.conv_norm_out.bias", "enc_norm_out.bias")
    put("encoder.conv_out.weight", "enc_out.weight")
    put("encoder.conv_out.bias", "enc_out.bias")
    put("quant_conv.weight", "quant_conv.weight")
    put("quant_conv.bias", "quant_conv.bias")
    put("post_quant_conv.weight", "post_quant.weight")
    put("post_quant_conv.bias", "post_quant.bias")
    put("decoder.conv_in.weight", "dec_in.weight")
    put("decoder.conv_in.bias", "dec_in.bias")
    put("decoder.conv_norm_out.weight", "dec_out.0.weight")
    put("decoder.conv_norm_out.bias", "dec_out.0.bias")
    put("decoder.conv_out.weight", "dec_out.2.weight")
    put("decoder.conv_out.bias", "dec_out.2.bias")
    for i in range(4):
        resnet(f"encoder.down_blocks.{i}.resnets.0.",
               f"enc_b{i + 1}.0.")
        resnet(f"encoder.down_blocks.{i}.resnets.1.",
               f"enc_b{i + 1}.1.")
        if i < 3:
            put(f"encoder.down_blocks.{i}.downsamplers.0.conv.weight",
                f"enc_b{i + 1}.2.weight")
            put(f"encoder.down_blocks.{i}.downsamplers.0.conv.bias",
                f"enc_b{i + 1}.2.bias")
    resnet("encoder.mid_block.resnets.0.", "enc_mid.0.")
    attn("encoder.mid_block.attentions.0.", "enc_mid.1.")
    resnet("encoder.mid_block.resnets.1.", "enc_mid.2.")
    resnet("decoder.mid_block.resnets.0.", "dec_mid.0.")
    attn("decoder.mid_block.attentions.0.", "dec_mid.1.")
    resnet("decoder.mid_block.resnets.1.", "dec_mid.2.")
    for i in range(4):
        for j in range(up_idx):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}.",
                   f"dec_b{i + 1}.{j}.")
        if i < 3:
            put(f"decoder.up_blocks.{i}.upsamplers.0.conv.weight",
                f"dec_b{i + 1}.{up_idx}.conv.weight")
            put(f"decoder.up_blocks.{i}.upsamplers.0.conv.bias",
                f"dec_b{i + 1}.{up_idx}.conv.bias")
    missing, _ = vae.load_state_dict(mapper.mapped, strict=False)
    if missing:
        raise ImgGenError(
            f"FLUX VAE: {len(missing)} parameters unfilled "
            f"(e.g. {missing[0]}) — the checkpoint is not a "
            f"diffusers AutoencoderKL state dict")
    for prefix in ("encoder.", "decoder.", "quant_conv.",
                   "post_quant_conv."):
        _check_section(state, prefix, mapper.consumed, "FLUX VAE")


# ---------------------------------------------------------------------------
# BFL transformer key map
# ---------------------------------------------------------------------------

def _map_bfl_transformer(state):
    """BFL single-file keys -> FluxTransformer state dict.

    The module's parameter names ARE the BFL names, so the map is a
    prefix strip (``model.diffusion_model.`` when present) plus
    completeness in both directions: an unfilled target parameter
    or an unconsumed ``double_blocks.`` / ``single_blocks.`` /
    ``*_in`` / ``final_layer.`` key raises naming the key.
    """
    prefix = ""
    if any(k.startswith("model.diffusion_model.double_blocks.")
           or k.startswith("model.diffusion_model.single_blocks.")
           for k in state):
        prefix = "model.diffusion_model."
    with torch.device("meta"):
        probe = FluxTransformer()
    want = set(probe.state_dict())
    mapped = {}
    consumed = set()
    for key in want:
        src = prefix + key
        if src in state:
            mapped[key] = state[src]
            consumed.add(src)
    missing = sorted(want - set(mapped))
    if missing:
        raise ImgGenError(
            f"FLUX transformer: {len(missing)} parameters unfilled "
            f"(e.g. {missing[0]}) — the checkpoint is not a complete "
            f"BFL single-file FLUX transformer")
    for src_prefix in (prefix + "double_blocks.",
                       prefix + "single_blocks.", prefix + "img_in.",
                       prefix + "txt_in.", prefix + "time_in.",
                       prefix + "vector_in.", prefix + "guidance_in.",
                       prefix + "final_layer."):
        _check_section(state, src_prefix, consumed,
                       "FLUX transformer")
    return mapped, consumed


# ---------------------------------------------------------------------------
# Companion resolution
# ---------------------------------------------------------------------------

def _find_companion(transformer_path, names, explicit=None):
    """First existing companion file, else an error naming the file
    and every directory searched.

    Order: an explicit path, the transformer's own directory,
    :func:`checkpoint_dir`, ``~/.devon-models``.
    """
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise ImgGenError(
            f"FLUX companion not found: {explicit}")
    dirs = []
    if transformer_path:
        dirs.append(os.path.dirname(os.path.abspath(transformer_path)))
    dirs.append(checkpoint_dir())
    dirs.append(os.path.join(os.path.expanduser("~"), ".devon-models"))
    seen = set()
    dirs = [d for d in dirs if not (d in seen or seen.add(d))]
    for d in dirs:
        for name in names:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
    joined = " or ".join(names)
    raise ImgGenError(
        f"FLUX companion not found: tried {joined} in "
        f"{', '.join(dirs)} — drop the file next to the FLUX "
        f"checkpoint")


def _split_companion_sections(state):
    """(t5_state, clip_state, vae_state) pulled out of a state dict
    that may carry the companions in-file (the round-trip fixture
    does; real BFL files do not — they resolve from disk)."""
    t5_state = {}
    clip_state = {}
    vae_state = {}
    for k, v in state.items():
        if (k == "shared.weight"
                or k.startswith("encoder.block.")
                or k == "encoder.final_layer_norm.weight"):
            t5_state[k] = v
        elif k.startswith("text_model.") or k == "text_projection.weight":
            clip_state[k] = v
        elif (k.startswith("encoder.conv_in.")
                or k.startswith("encoder.down_blocks.")
                or k.startswith("encoder.mid_block.")
                or k.startswith("encoder.conv_norm_out.")
                or k.startswith("encoder.conv_out.")
                or k.startswith("decoder.")
                or k.startswith("quant_conv.")
                or k.startswith("post_quant_conv.")):
            vae_state[k] = v
    return t5_state, clip_state, vae_state


# ---------------------------------------------------------------------------
# FLUXPipeline
# ---------------------------------------------------------------------------

class FLUXPipeline:
    """FLUX.1-dev text-to-image: MMDiT + T5-XXL + CLIP-L + VAE,
    sampled by rectified flow (FlowMatchScheduler, Euler)."""

    def __init__(self, transformer, t5, clip, vae, device="cpu",
                 warnings=None, dtype: str = "fp32",
                 quant: str = "none"):
        self.transformer = transformer
        self.t5 = t5
        self.clip = clip
        self.vae = vae
        self.device = device or "cpu"
        self.warnings = list(warnings or [])
        #: The dtype the weights were cast to before loading, and the
        #: quantization applied after ("none" | "int8"). The adapter
        #: and Studio report these; the sidecar records them.
        self.dtype = dtype
        self.quant = quant
        self.scheduler = FlowMatchScheduler()

    @property
    def tokenizer_mode(self):
        if getattr(self.t5, "tokenizer", None) is not None:
            return "spiece"
        if getattr(self.clip, "tokenizer", None) is not None:
            return "bpe"
        return "hashing-fallback"

    @classmethod
    def from_safetensors(cls, path, device="", *, t5_path=None,
                         clip_path=None, vae_path=None, dtype="",
                         quant=""):
        """Load a FLUX .safetensors checkpoint (BFL single-file layout).

        ``dtype`` ("fp32"/"fp16"/"bf16", "" = profile default) is
        cast into each module *before* its weights load;
        ``quant="int8"`` applies torch dynamic quantization to the
        Linear modules after the load (CPU only — see
        dtype.validate_request). Both are resolved and validated
        before the file is touched.
        """
        from .dtype import resolve_request

        want_dtype, want_quant, _device = resolve_request(
            dtype, quant, device=device)
        state = load_safetensors(path)
        return cls._from_state(state, path, device,
                               t5_path=t5_path, clip_path=clip_path,
                               vae_path=vae_path, dtype=want_dtype,
                               quant=want_quant)

    @classmethod
    def _from_state(cls, state, path, device="", *, t5_path=None,
                    clip_path=None, vae_path=None, dtype="",
                    quant=""):
        from .dtype import cast_module, quantize_linear_int8, resolve_request

        want_dtype, want_quant, _device = resolve_request(
            dtype, quant, device=device)
        keys = set(state)
        has_bfl = any(
            k.startswith("double_blocks.")
            or k.startswith("single_blocks.")
            or k.startswith("model.diffusion_model.double_blocks.")
            or k.startswith("model.diffusion_model.single_blocks.")
            for k in keys)
        has_diffusers = any(
            k.startswith("transformer_blocks.")
            or k.startswith("single_transformer_blocks.")
            for k in keys)
        if has_diffusers and not has_bfl:
            raise ImgGenError(
                "FLUX checkpoint in diffusers transformer layout — "
                "the loader speaks the BFL single-file layout "
                "(double_blocks/single_blocks); convert the "
                "transformer to a single BFL .safetensors first "
                "(diffusers-to-BFL key map is a named follow-up, "
                "not this pass)")
        if not has_bfl:
            if any("label_emb" in k for k in keys):
                raise ImgGenError(
                    f"{path} looks like an SDXL checkpoint "
                    "(label_emb present), not FLUX — load it with "
                    "the SDXL loader")
            if any("cond_stage_model" in k or "first_stage_model" in k
                   for k in keys):
                raise ImgGenError(
                    f"{path} looks like an SD1.5 checkpoint, not "
                    "FLUX — load it with the SD1.5 loader")
            raise ImgGenError(
                f"{path} is not a FLUX checkpoint: no "
                "double_blocks/single_blocks keys found")
        # Companions: in-file sections first, else files on disk.
        t5_state, clip_state, vae_state = _split_companion_sections(
            state)
        t5_source = path
        if not t5_state:
            t5_source = _find_companion(path, T5_FILE_NAMES, t5_path)
            t5_state = load_safetensors(t5_source)
        clip_source = path
        if not clip_state:
            clip_source = _find_companion(path, CLIP_FILE_NAMES,
                                          clip_path)
            clip_state = load_safetensors(clip_source)
        if not vae_state:
            vae_source = _find_companion(path, VAE_FILE_NAMES, vae_path)
            vae_state = load_safetensors(vae_source)
        mapped, _consumed = _map_bfl_transformer(state)
        transformer = FluxTransformer()
        cast_module(transformer, want_dtype)
        transformer.load_state_dict(mapped, strict=True)
        t5 = _load_hf_t5(t5_state, dtype=want_dtype)
        clip = _load_hf_clip(clip_state, dtype=want_dtype)
        vae = AutoencoderKL(latent_ch=16, scale=FLUX_VAE_SCALE,
                            shift=FLUX_VAE_SHIFT, decoder_resnets=2,
                            diffusers_layout=True)
        cast_module(vae, want_dtype)
        _load_diffusers_vae(vae_state, vae)
        warnings = []
        t5_tok, t5_warn = load_t5_tokenizer(t5_source)
        t5.tokenizer = t5_tok
        if t5_warn:
            warnings.append(t5_warn)
        clip_tok = None
        if clip_source != path:
            # Returns None when no tokenizer data is present; raises
            # (honestly, not swallowed) on a partial vocab/merges
            # pair — half a tokenizer is a data problem, not a
            # fallback.
            clip_tok = load_clip_tokenizer(clip_source)
        clip.tokenizer = clip_tok
        if clip_tok is None:
            warnings.append(
                "CLIP tokenizer: no vocab.json + merges.txt next "
                "to the CLIP-L file — hashing fallback for the "
                "pooled vector (reduced prompt fidelity)")
        device = device or _device
        transformer.to(device)
        t5.to(device)
        clip.to(device)
        vae.to(device)
        pipe = cls(transformer, t5, clip, vae, device=device,
                   warnings=warnings, dtype=want_dtype,
                   quant=want_quant)
        if want_quant == "int8":
            pipe.transformer = quantize_linear_int8(pipe.transformer)
            pipe.t5 = quantize_linear_int8(pipe.t5)
            pipe.clip = quantize_linear_int8(pipe.clip)
        return pipe

    def generate(self, prompt, *, steps=28, guidance_scale=3.5,
                 seed=None, width=1024, height=1024,
                 negative_prompt="", sampler="flow"):
        """Render one image. ``negative_prompt`` is accepted and
        ignored — FLUX.1-dev is guidance-distilled, there is no CFG
        branch; the adapter discloses that per call."""
        if sampler not in ("flow", "euler"):
            raise ImgGenError(
                f"sampler {sampler!r} is not available for FLUX — "
                "FLUX samples with flow-match Euler (sampler "
                "'flow')")
        if width % 16 or height % 16:
            raise ImgGenError(
                f"FLUX size {width}x{height} is not a multiple of "
                "16 — the VAE downsamples 8x and packing 2x")
        latent_h, latent_w = height // 8, width // 8
        gen = (torch.Generator(device="cpu").manual_seed(seed)
               if seed is not None else None)
        noise = torch.randn(1, FLUX_IN_CHANNELS, latent_h, latent_w,
                            generator=gen, device="cpu")
        txt = self.t5.encode([prompt])
        txt_ids = torch.zeros(1, txt.shape[1], 3)
        _ctx, vec = self.clip.encode_states([prompt])
        img = pack_latents(noise).to(self.device)
        img_ids = _image_ids(latent_h, latent_w).unsqueeze(0)
        img_ids = img_ids.to(self.device)
        txt = txt.to(self.device)
        txt_ids = txt_ids.to(self.device)
        vec = vec.to(self.device)
        timesteps = self.scheduler.set_timesteps(
            steps, image_seq_len=img.shape[1])
        guidance = torch.full((1,), guidance_scale,
                              device=self.device, dtype=img.dtype)
        with torch.no_grad():
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:],
                                      strict=True):
                t_vec = torch.full((1,), float(t_curr),
                                   device=self.device,
                                   dtype=img.dtype)
                v = self.transformer(
                    img, img_ids, txt, txt_ids, t_vec, vec, guidance)
                img = self.scheduler.step(
                    v, float(t_curr), float(t_prev), img)
        lat = unpack_latents(img.to("cpu"), latent_h, latent_w)
        with torch.no_grad():
            pixels = self.vae.decode(lat)
        arr = ((pixels.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        arr = arr[0].permute(1, 2, 0).cpu().numpy()
        from PIL import Image

        return [Image.fromarray(arr)]


# ---------------------------------------------------------------------------
# Studio adapter
# ---------------------------------------------------------------------------

class FLUXStudioAdapter:
    """Studio-facing FLUX adapter (mirrors SDXLStudioAdapter).

    The sampler question, settled: FLUX samples with flow-match
    Euler only. ``sampler="flow"`` (or ``"euler"``) is the honest
    request. ``sampler="ddim"`` is accepted ONLY as the
    PipelineConfig default — the caller passed no sampler, so the
    adapter runs flow-match Euler and says so in last_warnings.
    Any other sampler is refused by name.
    """

    model_class = "flux"

    def __init__(self, pipe):
        self._pipe = pipe
        self.last_warnings = []
        self.device = getattr(pipe, "device", "cpu")

    @property
    def dtype(self) -> str:
        """The dtype the pipeline's weights were cast to."""
        return getattr(self._pipe, "dtype", "fp32")

    @property
    def quant(self) -> str:
        """The quantization applied to the pipeline ("none"|"int8")."""
        return getattr(self._pipe, "quant", "none")

    @property
    def unet(self):
        raise ImgGenError(
            "flux has no UNet — the FLUX transformer is not a UNet; "
            "use the pipeline's transformer instead")

    def _scheduler(self, sampler):
        raise ImgGenError(
            "flux has no separate scheduler object — the flow-match "
            "schedule lives on the pipeline (FlowMatchScheduler)")

    def generate(self, prompt, cfg=None, conditioning=None):
        cfg = cfg or PipelineConfig()
        self.last_warnings = list(
            getattr(self._pipe, "warnings", []))
        if conditioning:
            raise ImgGenError(
                "flux does not support conditioning (ControlNet "
                "etc.) — pass no conditioning")
        sampler = (cfg.sampler or "flow").lower()
        if sampler not in ("flow", "euler", "ddim"):
            raise ImgGenError(
                f"sampler {cfg.sampler!r} is not available for FLUX "
                "— FLUX samples with flow-match Euler (sampler "
                "'flow')")
        if sampler == "ddim":
            self.last_warnings.append(
                "sampler 'ddim' is the Studio default — FLUX "
                "samples with flow-match Euler (sampler 'flow')")
            sampler = "flow"
        if cfg.negative_prompt:
            self.last_warnings.append(
                "FLUX.1-dev is guidance-distilled (single pass, no "
                "CFG branch): negative prompt "
                f"{cfg.negative_prompt!r} ignored")
        return self._pipe.generate(
            prompt, steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed, width=cfg.width, height=cfg.height,
            negative_prompt="", sampler=sampler)
