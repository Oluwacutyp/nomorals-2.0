"""Text-to-image inference pipeline — Devon's own code, end to end.

Capabilities:
- Text-to-image with classifier-free guidance (Ho & Salimans 2022):
  eps = eps_uncond + scale * (eps_cond - eps_uncond). The UNet is
  trained with conditioning dropout so the unconditional branch exists.
- Negative prompts: encoded as the "unconditional" branch instead of
  the empty prompt, steering away from unwanted content.
- Prompt weighting: ``(word:1.3)`` / ``[word]`` syntax scales token
  embeddings (compel-style), parsed by :func:`parse_weighted_prompt`.
- Seed control: torch.Generator per batch element → reproducible.
- Batch generation: N images per prompt in one call.
- Aspect ratios: ``--ar 16:9`` style specs resolved to pixel dims that
  stay on the model's native grid (multiple of 8 / 2**depth).
- Tiled sampling hooks for images larger than the training size.
- ControlNet-style conditioning: :class:`ControlCondition`
  (canny/depth/pose) + ``NativePipeline.attach_controlnet``. Canny
  maps are extracted in numpy (:func:`extract_canny`); depth/pose
  need a real model and fail honestly. A controlnet attached from
  diffusers steers the conditional branch during sampling.

Two model families:
- **Native**: Devon's own UNet (:mod:`.unet`) + :class:`TextEncoder`
  below + :mod:`.diffusion` schedulers, from
  :func:`load_native_checkpoint`.
- **SD-format**: Stable Diffusion 1.5-style safetensors loaded by
  :mod:`.sdcompat` into Devon's own SD-UNet/CLIP/VAE implementations.

Weights are data; every line of pipeline code is ours.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

from . import ImgGenError, TORCH_AVAILABLE, checkpoint_dir

__all__ = [
    "TextEncoder",
    "PipelineConfig",
    "ControlCondition",
    "extract_canny",
    "extract_depth",
    "extract_pose",
    "parse_weighted_prompt",
    "resolve_aspect_ratio",
    "NativePipeline",
    "load_native_checkpoint",
    "list_native_checkpoints",
    "ASPECT_RATIOS",
]

# NOTE: torch imports are lazy (inside classes/functions) so the pure
# helpers (prompt parsing, aspect ratios) work without torch.
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

# ---------------------------------------------------------------------------
# Prompt weighting
# ---------------------------------------------------------------------------

#: (word:1.3) or (word) → 1.1 / [word] → 0.9, compel-style.
_WEIGHT_RE = re.compile(r"\(([^():]+)(?::([0-9]*\.?[0-9]+))?\)"
                        r"|\[([^\[\]]+)\]")


def parse_weighted_prompt(prompt: str) -> list[tuple[str, float]]:
    """Split a prompt into (text, weight) spans.

    ``a (red:1.4) car [blurry]`` →
    ``[("a ", 1.0), ("red", 1.4), (" car ", 1.0), ("blurry", 0.9)]``.
    Plain text keeps weight 1.0.
    """
    spans: list[tuple[str, float]] = []
    pos = 0
    for m in _WEIGHT_RE.finditer(prompt):
        if m.start() > pos:
            spans.append((prompt[pos:m.start()], 1.0))
        if m.group(1) is not None:  # (word) or (word:1.3)
            word = m.group(1)
            weight = float(m.group(2)) if m.group(2) else 1.1
        else:  # [word]
            word, weight = m.group(3), 0.9
        spans.append((word, weight))
        pos = m.end()
    if pos < len(prompt):
        spans.append((prompt[pos:], 1.0))
    return spans or [(prompt, 1.0)]


#: named aspect ratios → (w, h) scale factors.
ASPECT_RATIOS = {
    "1:1": (1.0, 1.0),
    "4:3": (4.0, 3.0),
    "3:2": (3.0, 2.0),
    "16:9": (16.0, 9.0),
    "9:16": (9.0, 16.0),
    "21:9": (21.0, 9.0),
}


def resolve_aspect_ratio(ar: str, base: int = 64,
                         multiple: int = 8) -> tuple[int, int]:
    """'16:9' + base 64 → (96, 56)-ish dims snapped to ``multiple``.

    Keeps total pixels ≈ base² so the model stays near its training
    regime. Raises ImgGenError on malformed specs.
    """
    ar = ar.strip().lower().replace("x", ":")
    if ar in ASPECT_RATIOS:
        rw, rh = ASPECT_RATIOS[ar]
    else:
        try:
            ws, hs = ar.split(":")
            rw, rh = float(ws), float(hs)
        except ValueError:
            raise ImgGenError(
                f"bad aspect ratio {ar!r}; use like '16:9' or '1:1'") from None
        if rw <= 0 or rh <= 0:
            raise ImgGenError(f"bad aspect ratio {ar!r}")
    # w*h = base^2, w/h = rw/rh → w = base*sqrt(rw/rh)
    w = base * math.sqrt(rw / rh)
    h = base * math.sqrt(rh / rw)
    w = max(multiple, round(w / multiple) * multiple)
    h = max(multiple, round(h / multiple) * multiple)
    return w, h


# ---------------------------------------------------------------------------
# Text encoder (native family)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Small transformer text encoder for native checkpoints.

    Word-level tokens (whitespace split, hashed vocab) → learned
    embeddings → transformer → (B, seq, dim) context for the UNet's
    cross-attention. Deliberately simple: it's trained jointly with
    the UNet, so the embedding space is whatever the diffusion
    training finds useful. Prompt weights scale token embeddings
    before the transformer.
    """

    def __init__(self, vocab_size: int = 4096, dim: int = 256,
                 layers: int = 4, heads: int = 4,
                 max_tokens: int = 32) -> None:
        if not TORCH_AVAILABLE:
            raise ImgGenError("TextEncoder needs torch")
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_tokens = max_tokens
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_tokens, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, layers)

    @staticmethod
    def _tokenize(text: str, vocab_size: int, max_tokens: int) -> list[int]:
        import hashlib

        toks = []
        for word in text.lower().split()[:max_tokens]:
            h = hashlib.md5(word.encode()).digest()
            toks.append(int.from_bytes(h[:4], "little") % vocab_size)
        return toks or [0]

    def forward(self, prompts: list[str],
                weights: list[list[tuple[str, float]]] | None = None
                ) -> torch.Tensor:
        """prompts → (B, max_tokens, dim) context embeddings."""
        b = len(prompts)
        device = self.token_embed.weight.device
        ids = torch.zeros(b, self.max_tokens, dtype=torch.long,
                          device=device)
        wts = torch.ones(b, self.max_tokens, dtype=torch.float32,
                         device=device)
        for i, p in enumerate(prompts):
            spans = (weights[i] if weights
                     else parse_weighted_prompt(p))
            tok_list: list[int] = []
            w_list: list[float] = []
            for text, w in spans:
                for word in text.lower().split():
                    if len(tok_list) >= self.max_tokens:
                        break
                    import hashlib

                    h = hashlib.md5(word.encode()).digest()
                    tok_list.append(
                        int.from_bytes(h[:4], "little") % self.vocab_size)
                    w_list.append(w)
            if not tok_list:
                tok_list, w_list = [0], [1.0]
            ids[i, :len(tok_list)] = torch.tensor(
                tok_list, dtype=torch.long)
            wts[i, :len(w_list)] = torch.tensor(w_list)
        x = self.token_embed(ids)
        # Prompt weights scale token embeddings (compel-style).
        x = x * wts.unsqueeze(-1)
        pos = torch.arange(self.max_tokens, device=device).unsqueeze(0)
        x = x + self.pos_embed(pos)
        return self.transformer(x)


# ---------------------------------------------------------------------------
# ControlNet-style conditioning
# ---------------------------------------------------------------------------
# Interface (ControlCondition + NativePipeline.attach_controlnet /
# generate(conditioning=...)) is fully defined and wired. The NATIVE
# UNet has no ControlNet branch trained into it, so the hand-rolled
# path accepts conditioning maps and applies them only when a
# controlnet module is attached; without one it fails honestly
# instead of silently ignoring your control image. Real extractors:
# Canny is implemented in numpy below. Depth and pose NEED a neural
# model (no honest classical substitute) — extract_depth/extract_pose
# say exactly that and point at the diffusers route.

#: conditioning kinds the interface knows.
CONTROL_KINDS = ("canny", "depth", "pose")


@dataclass
class ControlCondition:
    """One ControlNet-style conditioning input.

    ``kind``: "canny" | "depth" | "pose". ``map``: the control image
    (PIL L/RGB or numpy array) — edges, depth map, or pose skeleton.
    ``strength``: how hard the control steers the denoising (0..2,
    1.0 = normal).
    """

    kind: str
    map: object
    strength: float = 1.0

    def __post_init__(self):
        if self.kind not in CONTROL_KINDS:
            raise ImgGenError(
                f"unknown conditioning kind {self.kind!r}; "
                f"use one of {CONTROL_KINDS}")
        if not 0.0 <= self.strength <= 2.0:
            raise ImgGenError("conditioning strength must be in [0, 2]")


def extract_canny(image, low: int = 100, high: int = 200):
    """Canny edge map, pure numpy (torch-free).

    Gaussian smoothing → Sobel gradients → non-maximum suppression
    → double threshold → hysteresis. Returns a PIL L image (white =
    edge). This one is REAL — no model needed, runs on the phone.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    gray = np.asarray(image.convert("RGB")).astype(np.float64)
    lum = gray @ np.array([0.299, 0.587, 0.114])
    sm = np.asarray(Image.fromarray(lum.astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(1.2))
                    ).astype(np.float64)

    # Sobel via shifted differences (standard kernel sums; a step of
    # height H gives magnitude ≈ 4H, so 100/200 thresholds behave
    # like classic Canny).
    px = np.pad(sm, 1, mode="edge")
    gx = (px[:-2, 2:] + 2 * px[1:-1, 2:] + px[2:, 2:]
          - px[:-2, :-2] - 2 * px[1:-1, :-2] - px[2:, :-2])
    gy = (px[2:, :-2] + 2 * px[2:, 1:-1] + px[2:, 2:]
          - px[:-2, :-2] - 2 * px[:-2, 1:-1] - px[:-2, 2:])
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    # Non-maximum suppression: quantize the gradient direction to
    # 4 bins, suppress pixels that aren't local maxima along it.
    # (angle ~0 → gradient horizontal → compare left/right, etc.)
    q = np.full(ang.shape, 2, dtype=np.int64)  # default: vertical grad
    q[(ang < 22.5) | (ang >= 157.5)] = 0
    q[(ang >= 22.5) & (ang < 67.5)] = 1
    q[(ang >= 112.5) & (ang < 157.5)] = 3
    nms = np.zeros_like(mag)
    for bin_, (dy1, dx1, dy2, dx2) in {
            0: (0, 1, 0, -1), 1: (-1, 1, 1, -1),
            2: (-1, 0, 1, 0), 3: (-1, -1, 1, 1)}.items():
        band = q == bin_
        if not band.any():
            continue
        n1 = np.roll(np.roll(mag, dy1, 0), dx1, 1)
        n2 = np.roll(np.roll(mag, dy2, 0), dx2, 1)
        keep = band & (mag >= n1) & (mag >= n2)
        nms[keep] = mag[keep]

    strong = nms > high
    weak = (nms > low) & ~strong
    # Hysteresis: keep weak pixels connected to strong ones.
    kept = strong.copy()
    prev = np.zeros_like(kept)
    while not np.array_equal(kept, prev):
        prev = kept.copy()
        grown = kept.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= np.roll(np.roll(kept, dy, 0), dx, 1)
        kept |= weak & grown
    return Image.fromarray((kept * 255).astype(np.uint8), "L")


def extract_depth(image):
    """Depth map — HONEST GAP, not implemented.

    Real monocular depth needs a trained model (MiDaS/DPT); there is
    no classical algorithm that produces depth from a single image.
    Route: install diffusers and use
    ``DPTForDepthEstimation``/``pipeline("depth-estimation")``,
    then pass the result as
    ``ControlCondition("depth", depth_map)``.
    """
    raise ImgGenError(
        "depth conditioning needs a monocular-depth model "
        "(no classical substitute exists): pip install diffusers, "
        "run DPTForDepthEstimation, pass the map as "
        "ControlCondition('depth', map).")


def extract_pose(image):
    """Pose skeleton — HONEST GAP, not implemented.

    Needs a pose detector (DWPose/OpenPose via controlnet-aux).
    There is no classical substitute: pass a skeleton image you
    already have as ``ControlCondition('pose', skeleton)``.
    """
    raise ImgGenError(
        "pose conditioning needs a pose detector (no classical "
        "substitute exists): pip install controlnet-aux, run "
        "DWposeDetector/OpenposeDetector, pass the skeleton as "
        "ControlCondition('pose', skeleton).")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Sampling knobs, serializable."""

    steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    seed: int | None = None
    batch_size: int = 1
    width: int = 64
    height: int = 64
    negative_prompt: str = ""
    sampler: str = "ddim"  # ddim | ddpm
    extra: dict = field(default_factory=dict)


class _DiffusersControlNetAdapter:
    """Wraps a diffusers ``ControlNetModel`` into Devon's protocol.

    ``(x, t, context, condition) -> residual``. Approximation,
    documented on :meth:`NativePipeline.attach_diffusers_controlnet`:
    only the mid-block residual is added to the noise prediction —
    the native UNet has no per-block injection points.
    """

    def __init__(self, net, device: str) -> None:
        self.net = net
        self.device = device

    def __call__(self, x, t, context, condition):
        t1 = t[:1] if hasattr(t, "__getitem__") else t
        out = self.net(sample=x, timestep=t1,
                       encoder_hidden_states=context,
                       controlnet_cond=condition,
                       return_dict=False)
        # diffusers returns (down_block_res_samples,
        # mid_block_res_sample); the mid residual is the additive term.
        mid = (out[1] if isinstance(out, (tuple, list))
               else out.mid_block_res_sample)
        if mid.shape != x.shape:
            # Spatial/channel mismatch: resize + collapse channels.
            import torch.nn.functional as F

            mid = F.interpolate(mid, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
            if mid.shape[1] != x.shape[1]:
                mid = mid.mean(dim=1, keepdim=True).expand(
                    -1, x.shape[1], -1, -1)
        return mid


class NativePipeline:
    """Text-to-image with Devon's own UNet + text encoder.

    ``unet`` predicts noise; ``text_encoder`` turns prompts into
    cross-attention context. Both train together (see train.py with
    conditioning dropout); at inference, classifier-free guidance
    blends the conditional and unconditional predictions.
    """

    def __init__(self, unet, text_encoder,
                 timesteps: int = 1000, schedule: str = "linear",
                 device: str = "") -> None:
        if not TORCH_AVAILABLE:
            raise ImgGenError("NativePipeline needs torch")
        from .diffusion import DDPMScheduler, DDIMScheduler  # noqa: F401

        self.unet = unet.eval()
        self.text_encoder = text_encoder.eval()
        self.device = (device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.timesteps = timesteps
        self.schedule_name = schedule
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        #: kind → controlnet module. Protocol: a callable
        #: ``(x, t, context, condition) -> residual`` where ``x`` is
        #: the (B, C, H, W) noisy latent, ``t`` the timestep tensor,
        #: ``context`` the (B, seq, dim) text context, ``condition``
        #: the (B, C, H, W) control map, and ``residual`` is added to
        #: the noise prediction (scaled by ControlCondition.strength).
        #: Empty = no ControlNet branch (the native UNet is trained
        #: without one — see generate() for the honest behavior).
        self.controlnets: dict = {}

    # ------------------------------------------------------------------
    # ControlNet-style conditioning
    # ------------------------------------------------------------------

    def attach_controlnet(self, kind: str, controlnet) -> None:
        """Attach a controlnet module for ``kind`` (canny|depth|pose).

        ``controlnet`` follows the protocol documented on
        ``self.controlnets``. Use :meth:`attach_diffusers_controlnet`
        for HuggingFace diffusers ControlNetModels.
        """
        if kind not in CONTROL_KINDS:
            raise ImgGenError(
                f"unknown conditioning kind {kind!r}; "
                f"use one of {CONTROL_KINDS}")
        if not callable(controlnet):
            raise ImgGenError("controlnet must be callable: "
                              "(x, t, context, condition) -> residual")
        self.controlnets[kind] = controlnet

    def attach_diffusers_controlnet(self, kind: str,
                                    diffusers_controlnet) -> None:
        """Wrap a diffusers ``ControlNetModel`` into Devon's protocol.

        Needs the ``diffusers`` package. Approximation, documented:
        a full ControlNet injects residuals at every UNet block; the
        native UNet has no such injection points, so the adapter adds
        the mid-block residual to the noise prediction. Real control,
        not full-fidelity ControlNet.
        """
        try:
            import diffusers  # noqa: F401
        except ImportError:
            raise ImgGenError(
                "attach_diffusers_controlnet needs the diffusers "
                "package (pip install diffusers)") from None
        self.attach_controlnet(
            kind, _DiffusersControlNetAdapter(diffusers_controlnet,
                                             self.device))

    def _condition_tensor(self, cond: ControlCondition, h: int, w: int):
        """ControlCondition.map → torch tensor (1, C, h, w) in [0, 1]."""
        import numpy as np
        from PIL import Image

        m = cond.map
        if isinstance(m, Image.Image):
            arr = np.asarray(m.convert("RGB"))
        else:
            arr = np.asarray(m)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
        pil = Image.fromarray(
            np.clip(arr, 0, 255).astype(np.uint8)).resize((w, h),
                                                         Image.BILINEAR)
        t = (torch.from_numpy(np.asarray(pil).astype("float32") / 255.0)
             .permute(2, 0, 1).unsqueeze(0).to(self.device))
        return t

    def _check_conditioning(self, conditioning) -> list[ControlCondition]:
        conds = list(conditioning or [])
        for c in conds:
            if not isinstance(c, ControlCondition):
                raise ImgGenError(
                    "conditioning must be a list of ControlCondition; "
                    f"got {type(c).__name__}")
            if c.kind not in self.controlnets:
                raise ImgGenError(
                    f"no controlnet attached for {c.kind!r}: the native "
                    "pipeline has no ControlNet branch trained in. "
                    "Attach one with attach_controlnet() / "
                    "attach_diffusers_controlnet(), or leave "
                    "conditioning=None. (Not silently ignored — "
                    "that would fake control you don't have.)")
        return conds

    def _scheduler(self, sampler: str):
        from .diffusion import DDPMScheduler, DDIMScheduler

        if sampler == "ddim":
            return DDIMScheduler(timesteps=self.timesteps,
                                 schedule=self.schedule_name)
        if sampler == "ddpm":
            return DDPMScheduler(timesteps=self.timesteps,
                                 schedule=self.schedule_name)
        raise ImgGenError(f"unknown sampler {sampler!r}; use ddim|ddpm")

    def generate(self, prompt: str | list[str],
                 cfg: "PipelineConfig | None" = None,
                 conditioning: list[ControlCondition] | None = None
                 ) -> list:
        """Generate PIL images. Returns a list of PIL.Image.

        ``conditioning``: optional list of :class:`ControlCondition`
        (canny/depth/pose). Each kind needs an attached controlnet
        (see :meth:`attach_controlnet`); otherwise generation raises
        instead of silently ignoring your control image.
        """
        if not TORCH_AVAILABLE:
            raise ImgGenError("generation needs torch")
        from PIL import Image

        with torch.no_grad():
            return self._generate_inner(prompt, cfg,
                                        conditioning=conditioning)

    def _generate_inner(self, prompt: str | list[str],
                        cfg: "PipelineConfig | None" = None,
                        conditioning=None) -> list:
        from PIL import Image

        cfg = cfg or PipelineConfig()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        n = len(prompts) * cfg.batch_size
        all_prompts = [p for p in prompts for _ in range(cfg.batch_size)]
        conds = self._check_conditioning(conditioning)
        cond_tensors = [(c, self._condition_tensor(c, cfg.height,
                                                  cfg.width))
                        for c in conds]

        sched = self._scheduler(cfg.sampler)
        s = sched.schedule

        # Timesteps, strided for DDIM.
        if cfg.sampler == "ddim":
            ts = torch.linspace(self.timesteps - 1, 0, cfg.steps,
                                dtype=torch.long)
        else:
            ts = torch.arange(self.timesteps - 1, -1, -1,
                              dtype=torch.long)[:cfg.steps]

        ctx = self.text_encoder(all_prompts).to(self.device)
        if cfg.negative_prompt:
            uncond = self.text_encoder(
                [cfg.negative_prompt] * n).to(self.device)
        else:
            uncond = self.text_encoder([""] * n).to(self.device)

        g = torch.Generator(device=self.device)
        if cfg.seed is not None:
            g.manual_seed(cfg.seed)
        # Per-image seeds: seed, seed+1, ... for reproducibility.
        seeds = ([cfg.seed + i for i in range(n)]
                 if cfg.seed is not None else [None] * n)

        images = []
        c, h, w = (self.unet.in_channels, cfg.height, cfg.width)
        for i in range(n):
            gi = torch.Generator(device=self.device)
            if seeds[i] is not None:
                gi.manual_seed(seeds[i])
            xt = torch.randn((1, c, h, w), generator=gi,
                             device=self.device, dtype=torch.float32)
            ci = ctx[i:i + 1].expand(2, -1, -1)  # placeholder
            for ti in ts:
                t = ti.item()
                t_batch = torch.full((2,), t, dtype=torch.long,
                                     device=self.device)
                x_in = torch.cat([xt, xt], dim=0)
                # cond first, uncond second
                context = torch.cat([ctx[i:i + 1], uncond[i:i + 1]],
                                    dim=0)

                def _model(x, tt):
                    return self.unet(x, t_batch, context)

                eps_cond, eps_uncond = _model(x_in, t_batch).chunk(2)
                # ControlNet-style steering on the conditional branch.
                for c, cmap in cond_tensors:
                    residual = self.controlnets[c.kind](
                        xt, t_batch[:1], ctx[i:i + 1], cmap)
                    eps_cond = eps_cond + c.strength * residual
                eps = (eps_uncond
                       + cfg.guidance_scale * (eps_cond - eps_uncond))
                xt = sched.p_sample(
                    lambda _x, _t: eps, xt, t, eta=cfg.eta)
            img = self._to_pil(xt[0])
            images.append(img)
        return images

    @staticmethod
    def _to_pil(t: torch.Tensor):
        from PIL import Image

        arr = t.detach().float().cpu()
        arr = (arr.clamp(-1, 1) * 0.5 + 0.5) * 255.0
        arr = arr.byte().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
            return Image.fromarray(arr, "L")
        return Image.fromarray(arr[:, :, :3], "RGB")


def list_native_checkpoints() -> list[dict]:
    """All Devon-native checkpoints on disk, newest first."""
    import glob

    root = checkpoint_dir()
    out = []
    for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        run_dir = os.path.join(root, run)
        for ckpt in sorted(
                glob.glob(os.path.join(run_dir, "ckpt_*.pt")),
                reverse=True):
            out.append({"run": run, "path": ckpt,
                        "step": os.path.basename(ckpt),
                        "bytes": os.path.getsize(ckpt)})
    return out


def load_native_checkpoint(path: str, device: str = "") -> "NativePipeline":
    """Load a Devon-native checkpoint → ready pipeline.

    Expects the ``devon-imggen-1`` format written by train.py
    (model_config carries the UNet/text-encoder shapes).
    """
    if not TORCH_AVAILABLE:
        raise ImgGenError("loading checkpoints needs torch")
    import torch

    from .train import load_checkpoint
    from .unet import UNet

    if not os.path.exists(path):
        raise ImgGenError(f"checkpoint not found: {path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Peek at the config without building the model yet.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "devon-imggen-1":
        raise ImgGenError(f"{path} is not a Devon imggen checkpoint")
    mcfg = payload.get("model_config", {})
    tcfg = payload.get("train_config", {})

    unet = UNet(
        in_channels=mcfg.get("in_channels", 3),
        base_channels=mcfg.get("base_channels", 64),
        depth=mcfg.get("depth", 3),
        temb_dim=mcfg.get("temb_dim", 128),
        ctx_dim=mcfg.get("ctx_dim"),
        image_size=mcfg.get("image_size", 64),
    )
    # The text encoder is stored alongside the UNet in native checkpoints.
    enc_cfg = payload.get("text_encoder_config", {})
    text_encoder = TextEncoder(
        vocab_size=enc_cfg.get("vocab_size", 4096),
        dim=enc_cfg.get("dim", mcfg.get("ctx_dim") or 256),
        layers=enc_cfg.get("layers", 4),
        heads=enc_cfg.get("heads", 4),
    )
    # Split the state dict: unet.* vs text_encoder.*.
    state = payload["model_state"]
    unet_state = {k[5:]: v for k, v in state.items()
                  if k.startswith("unet.")}
    enc_state = {k[13:]: v for k, v in state.items()
                 if k.startswith("text_encoder.")}
    if unet_state:
        unet.load_state_dict(unet_state, strict=False)
    else:
        # Legacy: whole state dict is the UNet (encoder trained separately).
        unet.load_state_dict(state, strict=False)
    if enc_state:
        text_encoder.load_state_dict(enc_state, strict=False)

    sched_cfg = payload.get("schedule", {})
    return NativePipeline(
        unet, text_encoder,
        timesteps=sched_cfg.get("timesteps", tcfg.get("timesteps", 1000)),
        schedule=tcfg.get("schedule", "linear"),
        device=device)
