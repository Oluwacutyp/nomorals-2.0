"""Image-to-image, inpainting, and outpainting — Devon's own code.

- **img2img** (SDEdit, Meng et al. 2021, arXiv:2108.01073): noise the
  input to an intermediate timestep ``t0 = strength * T`` with the
  forward process, then run the reverse chain from there conditioned
  on the new prompt. Low strength ≈ faithful edit; high strength ≈
  near-regeneration.
- **inpainting** (blended diffusion): at every reverse step the known
  region is re-noised from the original (``q_sample(x0, t-1)``) and
  composited under the mask, so unmasked pixels stay pixel-identical
  while masked ones are regenerated. Mask is feathered to avoid seams.
- **outpainting**: canvas extension is inpainting where the mask
  covers the new border regions.
"""

from __future__ import annotations

import torch

from . import ImgGenError, TORCH_AVAILABLE

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.edit needs PyTorch: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu")

from PIL import Image, ImageFilter

__all__ = [
    "img2img",
    "inpaint",
    "outpaint",
    "feather_mask",
]


def _pil_to_tensor(img: Image.Image, size: tuple[int, int],
                   device: str) -> torch.Tensor:
    img = img.convert("RGB").resize(size, Image.BILINEAR)
    import numpy as np

    arr = np.asarray(img).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def feather_mask(mask: Image.Image, radius: int = 8) -> Image.Image:
    """Gaussian-feather a binary/L mask to hide inpaint seams."""
    m = mask.convert("L")
    if radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius))
    return m


def img2img(pipeline, image: Image.Image, prompt: str,
            strength: float = 0.6, **gen_kwargs) -> list[Image.Image]:
    """SDEdit: ``strength`` ∈ (0,1) controls how far to re-noise.

    strength=0.2 → subtle edit, 0.8 → almost a fresh generation that
    keeps composition. Implemented by starting the reverse chain at
    t0 = round(strength * T) instead of T-1.
    """
    from .pipeline import PipelineConfig

    if not 0.0 < strength < 1.0:
        raise ImgGenError("strength must be in (0, 1)")
    cfg = gen_kwargs.pop("cfg", None) or PipelineConfig(**gen_kwargs)
    sched = pipeline._scheduler(cfg.sampler)
    s = sched.schedule
    t0 = min(sched.timesteps - 1, round(strength * sched.timesteps))

    w, h = cfg.width, cfg.height
    x0 = _pil_to_tensor(image, (w, h), pipeline.device)

    # Forward-noise the input to t0 (exact q_sample, not approximate).
    eps = torch.randn_like(x0)
    ab = float(s.alphas_cumprod[t0])
    xt = (ab ** 0.5) * x0 + ((1 - ab) ** 0.5) * eps

    ctx = pipeline.text_encoder([prompt]).to(pipeline.device)
    uncond = pipeline.text_encoder(
        [cfg.negative_prompt or ""]).to(pipeline.device)

    g = torch.Generator(device=pipeline.device)
    if cfg.seed is not None:
        g.manual_seed(cfg.seed)

    if cfg.sampler == "ddim":
        import numpy as np

        ts = np.linspace(t0, 0, cfg.steps).astype(int).tolist()
    else:
        ts = list(range(t0, -1, -1))[:cfg.steps]

    with torch.no_grad():
        for t in ts:
            t_batch = torch.full((2,), t, dtype=torch.long,
                                 device=pipeline.device)
            x_in = torch.cat([xt, xt], dim=0)
            context = torch.cat([ctx, uncond], dim=0)
            eps_c, eps_u = pipeline.unet(x_in, t_batch,
                                         context).chunk(2)
            eps_pred = eps_u + cfg.guidance_scale * (eps_c - eps_u)
            xt = sched.p_sample(lambda _x, _t: eps_pred, xt, t,
                                eta=cfg.eta)
    return [pipeline._to_pil(xt[0])]


def inpaint(pipeline, image: Image.Image, mask: Image.Image,
            prompt: str, feather: int = 8, **gen_kwargs) -> list[Image.Image]:
    """Regenerate masked regions; unmasked pixels stay identical.

    ``mask``: white = repaint, black = keep. At each reverse step the
    known region is replaced by the correctly-noised original, so the
    final image matches the input outside the mask exactly (up to the
    feather blend).
    """
    from .pipeline import PipelineConfig

    cfg = gen_kwargs.pop("cfg", None) or PipelineConfig(**gen_kwargs)
    sched = pipeline._scheduler(cfg.sampler)
    s = sched.schedule
    w, h = cfg.width, cfg.height
    device = pipeline.device

    x0 = _pil_to_tensor(image, (w, h), device)
    m = feather_mask(mask.resize((w, h), Image.BILINEAR),
                     radius=feather)
    import numpy as np

    m_t = (torch.from_numpy(np.asarray(m).astype("float32") / 255.0)
           .to(device).unsqueeze(0).unsqueeze(0))  # 1= repaint

    ctx = pipeline.text_encoder([prompt]).to(device)
    uncond = pipeline.text_encoder(
        [cfg.negative_prompt or ""]).to(device)

    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(cfg.seed)
    xt = torch.randn((1, pipeline.unet.in_channels, h, w),
                     generator=g, device=device)

    if cfg.sampler == "ddim":
        import numpy as np

        ts = np.linspace(sched.timesteps - 1, 0, cfg.steps
                         ).astype(int).tolist()
    else:
        ts = list(range(sched.timesteps - 1, -1, -1))[:cfg.steps]

    with torch.no_grad():
        for t in ts:
            t_batch = torch.full((2,), t, dtype=torch.long, device=device)
            x_in = torch.cat([xt, xt], dim=0)
            context = torch.cat([ctx, uncond], dim=0)
            eps_c, eps_u = pipeline.unet(x_in, t_batch,
                                         context).chunk(2)
            eps_pred = eps_u + cfg.guidance_scale * (eps_c - eps_u)
            xt = sched.p_sample(lambda _x, _t: eps_pred, xt, t,
                                eta=cfg.eta)
            if t > 0:
                # Re-noise the known region to timestep t-1 and blend.
                ab = float(s.alphas_cumprod[t - 1])
                known = ((ab ** 0.5) * x0 + ((1 - ab) ** 0.5)
                         * torch.randn_like(x0))
                xt = m_t * xt + (1 - m_t) * known
    return [pipeline._to_pil(xt[0])]


def outpaint(pipeline, image: Image.Image, prompt: str,
             left: int = 0, right: int = 0, top: int = 0,
             bottom: int = 0, feather: int = 12,
             **gen_kwargs) -> list[Image.Image]:
    """Extend the canvas; new regions are inpainted from the prompt.

    ``left/right/top/bottom`` are pixel counts added on each side.
    """
    from .pipeline import PipelineConfig

    cfg = gen_kwargs.pop("cfg", None) or PipelineConfig(**gen_kwargs)
    w, h = image.size
    nw, nh = w + left + right, h + top + bottom
    if nw <= 0 or nh <= 0:
        raise ImgGenError("outpaint extents must keep a positive canvas")

    canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
    canvas.paste(image, (left, top))
    mask = Image.new("L", (nw, nh), 255)  # repaint everything...
    keep = Image.new("L", (w, h), 0)      # ...except the original
    mask.paste(keep, (left, top))

    # Sample at the extended size snapped to the model grid.
    multiple = 8
    sw = max(multiple, round(nw / multiple) * multiple)
    sh = max(multiple, round(nh / multiple) * multiple)
    gen_kwargs = dict(gen_kwargs)
    gen_kwargs["width"], gen_kwargs["height"] = sw, sh
    cfg.width, cfg.height = sw, sh
    return inpaint(pipeline, canvas, mask, prompt, feather=feather,
                   cfg=cfg, **gen_kwargs)
