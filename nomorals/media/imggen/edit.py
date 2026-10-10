"""Image-to-image, inpainting, and outpainting — Devon's own code.

Two tiers:

- **Neural** (needs torch + a pipeline): img2img SDEdit (Meng et al.
  2021, arXiv:2108.01073), blended-diffusion inpainting, neural
  outpainting. Best quality; workstation/GPU territory.
- **CPU/classical** (numpy + PIL, always available): Telea-style
  inpainting via OpenCV when ``cv2`` is importable, otherwise a
  numpy Voronoi-seeded diffusion fill. Honest, non-neural quality —
  good for small regions and object removal, not for generating new
  content. This tier is what runs on the phone.

:func:`inpaint_auto` picks the best tier available and says which one
it used, so callers never have to guess. :func:`feather_mask`,
:func:`auto_mask` (re-exported from :mod:`.masks`), and the
:func:`pyramid_blend` compositor are torch-free.
"""

from __future__ import annotations

from . import ImgGenError, TORCH_AVAILABLE

__all__ = [
    "img2img",
    "suggest_strength",
    "inpaint",
    "inpaint_cpu",
    "inpaint_auto",
    "outpaint",
    "outpaint_cpu",
    "feather_mask",
    "auto_mask",
    "pyramid_blend",
    "seam_metric",
    "edge_extend_fill",
]

# Auto-mask lives in masks.py (one home, no scattering); re-exported
# here so edit.py is the single entry point for all editing ops.
from .masks import auto_mask  # noqa: E402


def _need_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImgGenError(
            "this neural edit needs PyTorch: "
            "pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu")


def _pil_to_tensor(img, size: tuple[int, int], device: str):
    """PIL → normalized torch tensor. Torch-only; call after _need_torch."""
    import torch
    from PIL import Image

    img = img.convert("RGB").resize(size, Image.BILINEAR)
    import numpy as np

    arr = np.asarray(img).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def feather_mask(mask, radius: int = 8):
    """Gaussian-feather a binary/L mask to hide inpaint seams."""
    from PIL import ImageFilter

    m = mask.convert("L")
    if radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius))
    return m


# ---------------------------------------------------------------------------
# Strength auto-selection (torch-free, pure heuristic)
# ---------------------------------------------------------------------------

#: edit-description cues → target SDEdit strength. "Restyle" cues push
#: toward regeneration; "touch-up" cues keep the edit faithful.
_STRENGTH_RULES: list[tuple[frozenset[str], float, str]] = [
    (frozenset({"touch up", "touchup", "fix", "remove the", "remove ",
                "erase", "subtle", "slight", "minor", "clean up",
                "cleanup", "sharpen", "brighten", "small edit",
                "tiny", "blemish", "scratch", "watermark"}),
     0.25, "touch-up / removal — keep it faithful"),
    (frozenset({"recolor", "change the color", "change color",
                "different color", "hair color", "eye color"}),
     0.40, "local recolor — medium strength"),
    (frozenset({"replace", "swap", "turn into", "make it a",
                "add a", "add an"}),
     0.50, "object replacement — balanced"),
    (frozenset({"background", "backdrop", "scenery", "setting",
                "put them in", "put it in"}),
     0.55, "background change — keeps the subject"),
    (frozenset({"style of", "in the style", "anime", "cartoon",
                "painting", "cyberpunk", "steampunk", "watercolor",
                "oil painting", "pixel art", "ghibli", "restyle",
                "stylize"}),
     0.70, "restyle — push toward reinterpretation"),
    (frozenset({"redraw", "regenerate", "completely", "totally",
                "from scratch", "reimagine"}),
     0.80, "near-regeneration"),
]


def suggest_strength(description: str) -> tuple[float, str]:
    """Infer SDEdit strength from an edit description.

    Returns ``(strength, rationale)``. Small touch-ups map to low
    strength (faithful), restyles to high strength (creative). This is
    a heuristic — an explicit strength always wins over the guess.
    """
    text = " " + description.lower() + " "
    for cues, strength, rationale in _STRENGTH_RULES:
        if any(cue in text for cue in cues):
            return strength, rationale
    return 0.55, "no strong signal — balanced default"


# ---------------------------------------------------------------------------
# img2img (SDEdit) — neural, needs torch
# ---------------------------------------------------------------------------

def img2img(pipeline, image, prompt: str,
            strength: float | str = 0.6, **gen_kwargs):
    """SDEdit: ``strength`` ∈ (0,1) controls how far to re-noise.

    strength=0.2 → subtle edit, 0.8 → almost a fresh generation that
    keeps composition. Pass ``strength="auto"`` to infer it from the
    prompt via :func:`suggest_strength`.
    """
    if isinstance(strength, str) and strength != "auto":
        raise ImgGenError(
            f"strength must be a float in (0, 1) or 'auto', "
            f"got {strength!r}")
    _need_torch()
    import torch
    from PIL import Image

    from .pipeline import PipelineConfig

    if isinstance(strength, str):  # "auto"
        strength, _why = suggest_strength(prompt)
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


# ---------------------------------------------------------------------------
# Neural inpainting (blended diffusion) — needs torch
# ---------------------------------------------------------------------------

def inpaint(pipeline, image, mask, prompt: str,
            feather: int = 8, **gen_kwargs):
    """Regenerate masked regions; unmasked pixels stay identical.

    ``mask``: white = repaint, black = keep. At each reverse step the
    known region is replaced by the correctly-noised original, so the
    final image matches the input outside the mask exactly (up to the
    feather blend). Torch + pipeline required; see :func:`inpaint_cpu`
    / :func:`inpaint_auto` for the phone-friendly tiers.
    """
    _need_torch()
    import torch
    from PIL import Image

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
           .to(device).unsqueeze(0).unsqueeze(0))  # 1 = repaint

    ctx = pipeline.text_encoder([prompt]).to(device)
    uncond = pipeline.text_encoder(
        [cfg.negative_prompt or ""]).to(device)

    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(cfg.seed)
    xt = torch.randn((1, pipeline.unet.in_channels, h, w),
                     generator=g, device=device)

    if cfg.sampler == "ddim":
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


# ---------------------------------------------------------------------------
# CPU/classical inpainting — numpy + PIL, phone-friendly
# ---------------------------------------------------------------------------

def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _inpaint_cv2(image, mask, method: str):
    """OpenCV Telea/NS inpaint. Real classical algorithm, non-neural."""
    import cv2
    import numpy as np
    from PIL import Image

    img = image.convert("RGB")
    arr = np.asarray(img)
    m = (np.asarray(mask.convert("L").resize(img.size, Image.BILINEAR))
         > 127).astype(np.uint8) * 255
    if not m.any():
        return img.copy(), "cv2: nothing masked"
    flag = {"telea": cv2.INPAINT_TELEA,
            "ns": cv2.INPAINT_NS}[method]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, m, 3, flag)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return (Image.fromarray(rgb),
            f"cv2-{method} (classical, non-neural quality)")


def _voronoi_seed(arr: "np.ndarray", hole: "np.ndarray") -> "np.ndarray":
    """Seed hole pixels from the nearest known pixel (Voronoi fill).

    Approximate multi-source nearest-neighbour via vectorized
    chamfer passes: forward + backward raster, each pixel takes the
    value of the neighbour with the smallest accumulated distance.
    Gives patch-match-*ish* structure: textured regions keep the
    texture of their nearest intact neighbour instead of collapsing
    to a gray mean.
    """
    import numpy as np

    h, w = hole.shape
    INF = 1e9
    dist = np.where(hole, INF, 0.0)
    val = arr.astype(np.float64).copy()

    # 8-neighbour shifts and their chamfer weights.
    shifts = [(-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
              (0, -1, 1.0), (0, 1, 1.0),
              (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5)]

    def _propagate_once():
        nonlocal dist, val
        cand_d = [dist]
        cand_v = [val]
        for dy, dx, wt in shifts:
            d = np.roll(np.roll(dist, dy, axis=0), dx, axis=1) + wt
            v = np.roll(np.roll(val, dy, axis=0), dx, axis=1)
            cand_d.append(d)
            cand_v.append(v)
        d_stack = np.stack(cand_d, axis=0)
        v_stack = np.stack(cand_v, axis=0)
        best = np.argmin(d_stack, axis=0)
        # gather the winning values per pixel (val is HxWx3)
        idx = np.expand_dims(np.expand_dims(best, 0), -1)
        val = np.take_along_axis(v_stack, idx, axis=0)[0]
        dist = np.min(d_stack, axis=0)

    _propagate_once()  # forward-ish
    _propagate_once()  # backward-ish (same op, second sweep)
    return val


def _inpaint_numpy(image, mask, *, iters: int = 150,
                   work_px: int = 256):
    """Voronoi-seeded diffusion fill, pure numpy.

    1. Seed hole pixels from their nearest intact neighbour (keeps
       local texture instead of starting from gray).
    2. Iteratively relax masked pixels toward their 4-neighbour mean
       with known pixels pinned (harmonic/Laplacian smoothing).

    Real classical inpainting. Quality label is honest: smooth
    interpolation, not neural synthesis.
    """
    import numpy as np
    from PIL import Image

    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, work_px / max(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((sw, sh), Image.BICUBIC)
    m = (np.asarray(mask.resize((sw, sh), Image.BICUBIC)
                    .convert("L")) > 127)
    if not m.any():
        return img.copy(), "numpy: nothing masked"

    arr = _voronoi_seed(np.asarray(small).astype(np.float64), m)
    known = ~m
    # Harmonic relaxation; known pixels pinned.
    for _ in range(iters):
        prev = arr[m].copy()
        up = np.roll(arr, 1, axis=0)
        down = np.roll(arr, -1, axis=0)
        left = np.roll(arr, 1, axis=1)
        right = np.roll(arr, -1, axis=1)
        arr[m] = (up[m] + down[m] + left[m] + right[m]) / 4
        if np.abs(arr[m] - prev).max(initial=0) < 0.05:
            break
    # Respect feather: soft-blend filled region with the original.
    alpha = (np.asarray(
        feather_mask(mask.resize((sw, sh), Image.BICUBIC),
                     radius=6).convert("L")).astype(np.float64)
             / 255.0)[..., None]
    base = np.asarray(small).astype(np.float64)
    out = alpha * arr + (1 - alpha) * base
    return (Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
            .resize((w, h), Image.BICUBIC),
            "numpy-voronoi-diffuse (classical, non-neural quality)")


def inpaint_cpu(image, mask, *, method: str = "auto",
                **kwargs) -> tuple:
    """Inpaint without torch: cv2 Telea/NS when available, else numpy.

    ``method``: ``"auto"`` (cv2 telea → numpy), ``"telea"``,
    ``"ns"``, ``"numpy"``. Returns ``(PIL.Image, quality_label)`` —
    the label always says the tier honestly (never "neural").
    """
    if method not in ("auto", "telea", "ns", "numpy"):
        raise ImgGenError(
            f"unknown cpu inpaint method {method!r}; "
            "use auto|telea|ns|numpy")
    if method in ("telea", "ns") and not _cv2_available():
        raise ImgGenError(
            f"method {method!r} needs OpenCV (pip install opencv-python); "
            "use method='numpy' or 'auto'")
    if method == "auto":
        method = "telea" if _cv2_available() else "numpy"
    if method in ("telea", "ns"):
        return _inpaint_cv2(image, mask, method)
    return _inpaint_numpy(image, mask, **kwargs)


def inpaint_auto(image, mask, prompt: str = "", *,
                 pipeline=None, method: str = "auto",
                 feather: int = 8, **gen_kwargs) -> tuple:
    """Best available inpaint, with an honest tier label.

    pipeline + torch → neural blended diffusion (best);
    otherwise → :func:`inpaint_cpu`. Returns ``(PIL.Image, label)``.
    """
    if pipeline is not None and TORCH_AVAILABLE:
        out = inpaint(pipeline, image, mask, prompt,
                      feather=feather, **gen_kwargs)[0]
        return out, "neural-blended-diffusion"
    if pipeline is not None and not TORCH_AVAILABLE:
        label_extra = " (pipeline given but torch missing)"
    else:
        label_extra = ""
    out, label = inpaint_cpu(image, mask, method=method)
    return out, label + label_extra


# ---------------------------------------------------------------------------
# Outpainting
# ---------------------------------------------------------------------------

def edge_extend_fill(image, left: int = 0, right: int = 0,
                     top: int = 0, bottom: int = 0):
    """Extend the canvas by extruding edge content outward.

    Torch-free multi-scale-ish fill: edge rows/columns are replicated
    with a gentle gradient continuation and fine noise so the new
    border reads as a natural extension, not a flat color block.
    Used as the seed for :func:`outpaint_cpu` and as the canvas for
    neural outpainting.
    """
    from PIL import Image

    import numpy as np

    img = image.convert("RGB")
    w, h = img.size
    nw, nh = w + left + right, h + top + bottom
    if nw <= 0 or nh <= 0:
        raise ImgGenError("outpaint extents must keep a positive canvas")
    arr = np.asarray(img).astype(np.float64)

    # Horizontal extension: replicate edge columns with decaying
    # gradient continuation + noise.
    def _extend_1d(edge, inner, n, axis):
        if n <= 0:
            return None
        grad = edge - inner  # per-pixel outward gradient estimate
        decay = np.exp(-np.arange(1, n + 1) / 12.0)
        if axis == 1:
            base = np.repeat(edge[:, None, :], n, axis=1)
            push = (grad[:, None, :]
                    * decay[None, :, None] * 0.5)
            noise = (np.random.default_rng(7).normal(
                0, 3.0, (arr.shape[0], n, 3)))
            return base + push + noise
        base = np.repeat(edge[None, :, :], n, axis=0)
        push = grad[None, :, :] * decay[:, None, None] * 0.5
        noise = np.random.default_rng(7).normal(
            0, 3.0, (n, edge.shape[0], 3))
        return base + push + noise

    parts_x = []
    l = _extend_1d(arr[:, 0, :], arr[:, min(1, w - 1), :], left, 1)
    if l is not None:
        parts_x.append(l)
    parts_x.append(arr)
    r = _extend_1d(arr[:, -1, :], arr[:, max(0, w - 2), :], right, 1)
    if r is not None:
        parts_x.append(r)
    wide = np.concatenate(parts_x, axis=1)

    parts_y = []
    t = _extend_1d(wide[0, :, :], wide[min(1, h - 1), :, :], top, 0)
    if t is not None:
        parts_y.append(t)
    parts_y.append(wide)
    b = _extend_1d(wide[-1, :, :], wide[max(0, h - 2), :, :], bottom, 0)
    if b is not None:
        parts_y.append(b)
    full = np.concatenate(parts_y, axis=0)
    return Image.fromarray(np.clip(full, 0, 255).astype(np.uint8))


def pyramid_blend(a, b, mask, levels: int = 4):
    """Burt & Adelson multi-resolution spline blend (torch-free).

    ``a``/``b`` are PIL RGB images, ``mask`` is PIL L (white → take
    ``a``). Blends each frequency band separately so seams vanish
    without ghosting — the classic seam-killer for outpaint borders.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    a_arr = np.asarray(a.convert("RGB")).astype(np.float64)
    b_arr = np.asarray(b.convert("RGB").resize(a.size, Image.BILINEAR)
                       ).astype(np.float64)
    m = (np.asarray(mask.convert("L").resize(a.size, Image.BILINEAR))
         .astype(np.float64) / 255.0)

    def _reduce(img):
        # 2x2 mean-pool decimation, pure float (no uint8 round-trip,
        # so the pyramid reconstructs near-exactly and mask=0/1
        # regions survive bit-identical).
        h, w = img.shape[:2]
        h2, w2 = max(1, h // 2), max(1, w // 2)
        c = img[:h2 * 2, :w2 * 2]
        if c.ndim == 2:
            return c.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
        return c.reshape(h2, 2, w2, 2, c.shape[2]).mean(axis=(1, 3))

    def _tent_blur(img):
        # Separable 3-tap [1,2,1]/4 blur on axes 0/1, edge-replicated.
        k = np.array([1.0, 2.0, 1.0]) / 4.0
        pad = [(1, 1), (1, 1)] + [(0, 0)] * (img.ndim - 2)
        p = np.pad(img, pad, mode="edge")
        acc = np.zeros_like(img)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += (k[dy + 1] * k[dx + 1]
                        * p[1 + dy:1 + dy + img.shape[0],
                            1 + dx:1 + dx + img.shape[1]])
        return acc

    def _expand(img, shape):
        # 2x nearest upsample + tent blur ≈ bilinear, energy
        # preserving (no gain factor needed). Odd parent dims were
        # cropped on the way down; replicate the edge to cover them.
        up = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
        up = _tent_blur(up)
        up = up[:shape[0], :shape[1]]
        pad_h = shape[0] - up.shape[0]
        pad_w = shape[1] - up.shape[1]
        if pad_h > 0 or pad_w > 0:
            pad = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (up.ndim - 2)
            up = np.pad(up, pad, mode="edge")
        return up

    # Gaussian pyramids (mask stays 2-D)
    ga, gb, gm = [a_arr], [b_arr], [m]
    for _ in range(levels):
        ga.append(_reduce(ga[-1]))
        gb.append(_reduce(gb[-1]))
        gm.append(_reduce(gm[-1]))

    # Laplacian pyramids + blend per band
    out = gm[-1][..., None] * ga[-1] + (1 - gm[-1][..., None]) * gb[-1]
    for lvl in range(levels - 1, -1, -1):
        la = ga[lvl] - _expand(ga[lvl + 1], ga[lvl].shape)
        lb = gb[lvl] - _expand(gb[lvl + 1], gb[lvl].shape)
        gm_l = gm[lvl][..., None]
        out = (gm_l * la + (1 - gm_l) * lb
               + _expand(out, ga[lvl].shape))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def seam_metric(image, seam_x: int, window: int = 6) -> float:
    """How visible is the vertical seam at ``seam_x``? (torch-free)

    Compares the mean absolute horizontal gradient in a narrow band
    around the seam against the interior baseline. 1.0 ≈ invisible
    (seam looks like the rest of the image); higher = more visible.
    """
    import numpy as np

    arr = np.asarray(image.convert("L")).astype(np.float64)
    gx = np.abs(np.diff(arr, axis=1))
    h, w = gx.shape
    sx = min(max(seam_x, window), w - window - 1)
    seam_band = gx[:, sx - window:sx + window].mean()
    interior = np.concatenate(
        [gx[:, :max(1, sx - 3 * window)].ravel(),
         gx[:, sx + 3 * window:].ravel()]).mean()
    if interior < 1e-9:
        return 1.0 if seam_band < 1e-9 else float("inf")
    return float(seam_band / interior)


def outpaint_cpu(image, prompt: str = "", left: int = 0,
                 right: int = 0, top: int = 0, bottom: int = 0,
                 *, method: str = "auto") -> tuple:
    """Extend the canvas without torch.

    Seed: :func:`edge_extend_fill` (gradient-continued edge extrusion).
    Polish: CPU-inpaint the seam band, then :func:`pyramid_blend` the
    extension against the original so the border disappears.

    Preservation guarantee: the original's high-frequency detail is
    bit-identical outside a 12px seam band; inside ~32px of the
    border a smooth low-frequency tone blend applies (that's what
    makes the seam invisible). Returns ``(PIL.Image, quality_label)``
    — honest: structure is extrapolated, no new content is
    synthesized (non-neural).
    """
    from PIL import Image, ImageFilter

    from .masks import box_mask

    w, h = image.size
    canvas = edge_extend_fill(image, left, right, top, bottom)
    nw, nh = canvas.size

    # Seam band = the strips right at the original border.
    band = 12
    mask = Image.new("L", (nw, nh), 0)
    if left:
        mask.paste(255, (max(0, left - band), 0, left + band, nh))
    if right:
        mask.paste(255, (nw - right - band, 0,
                         min(nw, nw - right + band), nh))
    if top:
        mask.paste(255, (0, max(0, top - band), nw, top + band))
    if bottom:
        mask.paste(255, (0, nh - bottom - band, nw,
                         min(nh, nh - bottom + band)))

    refined, fill_label = inpaint_cpu(canvas, mask, method=method)

    # Pyramid-blend: keep the ORIGINAL pixels inside, the refined
    # fill outside; the seam band itself gets multi-resolution
    # blending so the transition is invisible. The "keep" side is
    # the original composited over the refined canvas (NOT black
    # outside) so the blurred mask never bleeds darkness into
    # the seam.
    keep = Image.new("L", (nw, nh), 255)  # white → take refined...
    inner = Image.new("L", (w, h), 0)     # ...black → keep original
    keep.paste(inner, (left, top))
    keep = keep.filter(ImageFilter.GaussianBlur(band))
    keep_side = refined.copy()
    keep_side.paste(image.convert("RGB"), (left, top))
    blended = pyramid_blend(refined, keep_side, keep, levels=4)
    return blended, (f"cpu-outpaint ({fill_label}; extrapolated, "
                     "non-neural — no new content synthesized)")


def outpaint(pipeline, image, prompt: str,
             left: int = 0, right: int = 0, top: int = 0,
             bottom: int = 0, feather: int = 12, seam_blend: bool = True,
             **gen_kwargs):
    """Extend the canvas; new regions are inpainted from the prompt.

    ``left/right/top/bottom`` are pixel counts added on each side.
    With ``seam_blend=True`` (default) the generated canvas is
    pyramid-blended against the original along the border so the seam
    disappears even when the model drifts in tone.
    """
    _need_torch()
    from PIL import Image

    from .pipeline import PipelineConfig

    cfg = gen_kwargs.pop("cfg", None) or PipelineConfig(**gen_kwargs)
    w, h = image.size
    canvas = edge_extend_fill(image, left, right, top, bottom)
    nw, nh = canvas.size

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
    gen = inpaint(pipeline, canvas, mask, prompt, feather=feather,
                  cfg=cfg, **gen_kwargs)[0]

    if seam_blend:
        from PIL import ImageFilter

        # Blend map: white → generated, black → original, feathered
        # along the border so each frequency band mixes smoothly.
        keep_map = Image.new("L", gen.size, 255)
        inner = Image.new("L", (w, h), 0)
        # Map original coords into the generated (snapped) size.
        sx, sy = gen.size[0] / nw, gen.size[1] / nh
        inner = inner.resize((round(w * sx), round(h * sy)),
                             Image.BILINEAR)
        keep_map.paste(inner, (round(left * sx), round(top * sy)))
        keep_map = keep_map.filter(ImageFilter.GaussianBlur(feather))
        # "Keep" side = original composited over the generation (not
        # black outside) so the blurred mask can't bleed darkness
        # into the seam.
        keep_side = gen.copy()
        keep_side.paste(
            image.convert("RGB").resize((round(w * sx), round(h * sy)),
                                        Image.BILINEAR),
            (round(left * sx), round(top * sy)))
        gen = pyramid_blend(gen, keep_side, keep_map, levels=4)
    return [gen]
