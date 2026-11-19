"""Texture preview rendering (TextureCraft port, census 3A.15).

Renders a composited channel to a PIL image through Devon's own
PIL path (the same seam the photo editor and the vector/design
renderers use). A mipmap level may be selected; uncovered texels
show the channel's default value; a checker background can be
requested for the transparent (uncovered) region. The renderer
never fakes a feature: anything it does not draw is named in the
``unrendered`` list (unsupported layer effects, height masks,
geometry masks).
"""
from __future__ import annotations

from PIL import Image

from .compose import _default_u16, composite
from .errors import TextureError, TextureErrorKind
from .filters import SUPPORTED_FILTERS
from .mipmaps import mipmap_chain
from .model import Channel, TextureDocument


def render_channel(doc: TextureDocument, set_name: str, channel: str,
                   level: int = 0, checker: bool = False
                   ) -> tuple[Image.Image, list[str]]:
    """Render one channel to a PIL image. Returns ``(image,
    unrendered)``."""
    tset = None
    for s in doc.sets:
        if s.name == set_name:
            tset = s
            break
    if tset is None:
        raise TextureError(TextureErrorKind.NOT_FOUND,
                           f"no texture set {set_name!r}")
    unrendered: list[str] = []
    for layer in tset.layers:
        for spec in layer.effects:
            name = str(spec[0]).lower() if spec else ""
            if name and name not in SUPPORTED_FILTERS:
                unrendered.append(f"effect {name!r} on layer "
                                  f"{layer.name!r}")
        if layer.mask is not None and layer.mask.get("height_mask"):
            unrendered.append(f"height mask on layer {layer.name!r}")
        if layer.mask is not None and layer.mask.get("geometry_mask"):
            unrendered.append(f"geometry mask on layer {layer.name!r}")
    img = composite(doc, set_name, channel)
    chain = mipmap_chain(img)
    lvl = min(max(0, level), len(chain) - 1)
    img = chain[lvl]
    ch = Channel.parse(channel)
    comps = ch.components() if ch is not None else 1
    default = _default_u16(channel)
    if comps == 3:
        pil = Image.new("RGB", (img.width, img.height))
        px = pil.load()
        for y in range(img.height):
            for x in range(img.width):
                v = img.get(x, y)
                cov = v[comps] / 65535.0
                if cov <= 0.0:
                    if checker and ((x // 8) + (y // 8)) % 2 == 0:
                        px[x, y] = (200, 200, 200)
                    else:
                        px[x, y] = tuple(int(d / 65535.0 * 255.0 + 0.5)
                                         for d in default[:3])
                else:
                    px[x, y] = tuple(int(v[c] / 65535.0 * 255.0 + 0.5)
                                     for c in range(3))
    else:
        pil = Image.new("L", (img.width, img.height))
        px = pil.load()
        for y in range(img.height):
            for x in range(img.width):
                v = img.get(x, y)
                cov = v[comps] / 65535.0
                if cov <= 0.0:
                    px[x, y] = int(default[0] / 65535.0 * 255.0 + 0.5)
                else:
                    px[x, y] = int(v[0] / 65535.0 * 255.0 + 0.5)
    return pil, sorted(set(unrendered))


def save_preview(doc: TextureDocument, set_name: str, channel: str,
                 path, level: int = 0,
                 checker: bool = False) -> dict:
    """Render and save a PNG preview. Returns the report."""
    pil, unrendered = render_channel(doc, set_name, channel, level, checker)
    pil.save(str(path), format="PNG")
    return {"path": str(path), "width": pil.width, "height": pil.height,
            "channel": channel, "level": level,
            "unrendered": unrendered}
