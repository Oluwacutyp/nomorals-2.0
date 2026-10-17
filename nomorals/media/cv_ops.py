"""Classical CV ops — pure numpy/PIL, no cv2 required.

inpaint_diffuse: iterative Laplacian diffusion fill for masked regions.
A real inpainting algorithm (harmonic fill): masked pixels converge to
the smooth interpolation of their surroundings. Good for small regions
(object removal, scratch repair); large regions get downscaled fill.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

NUMPY_AVAILABLE = np is not None


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "nomorals.media.cv_ops is unavailable: numpy is not installed "
            "(pip install numpy)")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise RuntimeError(
            "nomorals.media.cv_ops is unavailable: Pillow is not installed "
            "(pip install Pillow)")



def inpaint_diffuse(image: Image.Image, mask: Image.Image,
                    *, iters: int = 400) -> Image.Image:
    """Fill masked (white) regions by diffusion from known pixels."""
    img = image.convert("RGB")
    w, h = img.size
    # work at most at 256px on the long side for speed
    scale = min(1.0, 256 / max(w, h))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    _require_pil()
    small = img.resize((sw, sh), Image.BICUBIC)
    _require_numpy()
    m = np.array(mask.resize((sw, sh), Image.BICUBIC).convert("L")) > 127
    if not m.any():
        return img.copy()
    arr = np.array(small).astype(np.float64)
    known = ~m
    # init masked pixels to the mean of known (fast start)
    for c in range(3):
        ch = arr[..., c]
        ch[m] = ch[known].mean() if known.any() else 128
    # diffuse: masked pixel <- mean of 4-neighbours, known pixels pinned
    for _ in range(iters):
        prev = arr[m].copy()
        up = np.roll(arr, 1, axis=0)
        down = np.roll(arr, -1, axis=0)
        left = np.roll(arr, 1, axis=1)
        right = np.roll(arr, -1, axis=1)
        arr[m] = (up[m] + down[m] + left[m] + right[m]) / 4
        if np.abs(arr[m] - prev).max() < 0.05:
            break
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out.resize((w, h), Image.BICUBIC)


def feather(mask: Image.Image, radius: int = 8) -> Image.Image:
    _require_pil()
    from PIL import ImageFilter
    return mask.convert("L").filter(ImageFilter.GaussianBlur(radius))
