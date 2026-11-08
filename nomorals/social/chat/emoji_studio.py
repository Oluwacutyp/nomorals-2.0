"""The studio: Devon's own hand — a deterministic local art
renderer for emoji, stickers, animations and banners.

Why this exists (owner correction, 2026-11-07): emoji creation
used to mean "call a configured image backend, and when none is
configured, fail" — on the owner's install none ever was, so
creation was dead while surfaces called its sticker output
"emoji".  This module is the CODE RUNG of the creation ladder:
a real renderer that needs nothing outside the install (Pillow
is in the stack) and produces, from a described design:

* **custom-emoji art** — a 100×100 transparent PNG, the exact
  shape Telegram custom-emoji packs carry;
* **sticker art** — a 512×512 master the house sticker seam
  normalizes (``media_convert.image_to_sticker``);
* **mood animations** — small GIFs (a face/shape in motion) for
  the expression library's self-created floor;
* **the menu banner** — her picture head for the live menu.

A design is a plain dict — ``{"shape": ..., "palette": ...,
"face": ..., "letter": ..., "sparkles": bool, "bg": ...}`` —
composed from words by :func:`parse_design` (the command/NL
path) or built directly (the moment system).  Rendering is
deterministic: the same design draws the same art, on any host,
with no backend, no key and no network.  When an image backend
IS configured it stays the top rung of /emoji make — this rung
is the floor that can never be missing, not a replacement.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any

__all__ = [
    "PALETTES",
    "SHAPES",
    "parse_design",
    "render_master",
    "emoji_png",
    "sticker_master_png",
    "mood_animation",
    "menu_banner",
    "fallback_char_for",
]


def _pil():  # type: ignore[no-untyped-def]
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


# ── palettes ─────────────────────────────────────────────────
#: name → (main, dark, light, accent).  Warm, readable at
#: 100×100: bold shapes, limited detail — the emoji rule.
PALETTES: dict[str, tuple[str, str, str, str]] = {
    "red": ("#e0342b", "#9c1f18", "#ff7a6e", "#ffd9d4"),
    "pink": ("#ef5da8", "#b23177", "#ff9ecb", "#ffe0ef"),
    "gold": ("#f2b01e", "#b97e0a", "#ffd95e", "#fff0bd"),
    "orange": ("#f07d1a", "#b35508", "#ffa94d", "#ffe3c2"),
    "purple": ("#8b5cf6", "#5f36c4", "#bb9bff", "#eadfff"),
    "blue": ("#2f7de1", "#1a529c", "#6fb1ff", "#d6e9ff"),
    "teal": ("#12b5a5", "#087f74", "#4fe0cf", "#ccf7f1"),
    "green": ("#35b558", "#1d7c39", "#6fe08c", "#d8f7df"),
    "mint": ("#7fe3c0", "#3aa97f", "#b8f4dc", "#eafff5"),
    "night": ("#2b3350", "#171c30", "#5a6a9e", "#c9d4ff"),
    "mono": ("#e8e8ec", "#8a8a94", "#ffffff", "#3a3a42"),
}

_COLOR_WORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("purple", "violet", "lavender"), "purple"),
    (("pink", "rose", "magenta"), "pink"),
    (("gold", "golden", "yellow"), "gold"),
    (("orange", "amber"), "orange"),
    (("teal", "cyan", "turquoise"), "teal"),
    (("green", "emerald"), "green"),
    (("mint",), "mint"),
    (("blue", "azure", "sky"), "blue"),
    (("red", "crimson", "scarlet"), "red"),
    (("night", "midnight", "dark"), "night"),
    (("white", "silver", "mono"), "mono"),
)

#: shape word → shape name.  The face words compose a face;
#: a bare mood word with no shape is a face too (that's what
#: "a happy emoji" means).
SHAPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("heart", "love"), "heart"),
    (("star",), "star"),
    (("moon", "crescent"), "moon"),
    (("sun", "sunny"), "sun"),
    (("flame", "fire", "lit"), "flame"),
    (("drop", "tear", "raindrop", "water"), "drop"),
    (("bolt", "lightning", "thunder", "zap"), "bolt"),
    (("ring", "halo", "circle", "orb", "ball"), "disc"),
    (("cat",), "cat"),
    (("flower", "bloom", "petal"), "flower"),
    (("crown",), "crown"),
    (("rocket",), "rocket"),
)

_FACE_WORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("laugh", "laughing", "haha", "joy"), "laugh"),
    (("wink",), "wink"),
    (("sleepy", "sleep", "tired"), "sleepy"),
    (("wow", "surprised", "shock"), "wow"),
    (("cool", "sunglasses", "shades"), "cool"),
    (("happy", "smile", "smiling", "grin", "face"), "happy"),
)

#: A created emoji's unicode fallback: the closest standing
#: emoji for each shape/face, so the set entry renders as
#: SOMETHING everywhere (WhatsApp, plain text) even where
#: custom emoji don't exist.
_FALLBACKS: dict[str, str] = {
    "heart": "❤️", "star": "⭐", "moon": "🌙", "sun": "☀️",
    "flame": "🔥", "drop": "💧", "bolt": "⚡", "disc": "🔵",
    "cat": "🐱", "flower": "🌸", "crown": "👑", "rocket": "🚀",
    "face": "🙂",
}
_FACE_FALLBACKS: dict[str, str] = {
    "happy": "😊", "laugh": "😂", "wink": "😉", "sleepy": "😴",
    "wow": "😮", "cool": "😎",
}


def parse_design(description: str) -> dict[str, Any]:
    """Compose a design dict from plain words — "a purple
    laughing cat", "gold heart with sparkles", "letter d".
    Deterministic keyword composition; unknown words simply
    don't move the design (the default is a happy gold face —
    an emoji-shaped answer, never an error)."""
    text = " ".join(str(description or "").lower().split())
    spec: dict[str, Any] = {"shape": "", "palette": "gold",
                           "face": "", "letter": "",
                           "sparkles": "sparkle" in text
                           or "glitter" in text,
                           "bg": "disc" if "badge" in text
                           or "button" in text else ""}
    for words, palette in _COLOR_WORDS:
        if any(w in text for w in words):
            spec["palette"] = palette
            break
    for words, face in _FACE_WORDS:
        if any(w in text for w in words):
            spec["face"] = face
            break
    for words, shape in SHAPES:
        if any(w in text for w in words):
            spec["shape"] = shape
            break
    words = text.split()
    if "letter" in words:
        i = words.index("letter")
        if i + 1 < len(words) and words[i + 1]:
            spec["letter"] = words[i + 1][0]
    if not spec["shape"] and len(words) == 1 and len(words[0]) == 1:
        spec["letter"] = words[0]
    if spec["letter"]:
        spec["shape"] = "letter"
    elif not spec["shape"]:
        spec["shape"] = "face"
        spec["face"] = spec["face"] or "happy"
    return spec


def fallback_char_for(spec: dict[str, Any]) -> str:
    """The unicode face a created emoji falls back to."""
    if spec.get("shape") == "letter":
        return str(spec.get("letter") or "✨")
    if spec.get("shape") == "face" or spec.get("face"):
        return _FACE_FALLBACKS.get(str(spec.get("face") or "happy"),
                                   "🙂")
    return _FALLBACKS.get(str(spec.get("shape") or ""), "✨")


def _seed_for(spec: dict[str, Any]) -> int:
    blob = repr(sorted((str(k), str(v)) for k, v in spec.items()))
    return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)


def _font(size: int):  # type: ignore[no-untyped-def]
    """A bold font at ``size``: DejaVu where the host carries
    it (Linux), the Windows staples on EC2, else Pillow's own
    sized default — letter art never depends on one host."""
    _Image, _Draw, ImageFont = _pil()
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        try:
            if Path(path).is_file():
                return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 - next candidate
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001 - ancient Pillow
        return ImageFont.load_default()


def _heart_points(cx: float, cy: float, s: float) -> list:
    pts = []
    for i in range(64):
        t = 2 * math.pi * i / 64
        x = 16 * math.sin(t) ** 3
        y = (13 * math.cos(t) - 5 * math.cos(2 * t)
             - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + s * x / 17.0, cy - s * y / 17.0))
    return pts


def _star_points(cx: float, cy: float, r: float,
                 points: int = 5, inner: float = 0.48) -> list:
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * inner
        a = -math.pi / 2 + i * math.pi / points
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts


def _draw_shape(draw, spec: dict[str, Any], box: tuple,
                palette: tuple[str, str, str, str],
                img=None) -> None:  # type: ignore[no-untyped-def]
    """Draw the design's subject inside ``box`` (x0, y0, x1, y1).
    ``img`` is the canvas itself, for the one shape (the moon)
    whose crescent is a CUT, not a paint."""
    Image, _Draw, _Font = _pil()
    main, dark, light, accent = palette
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(w, h) / 2
    shape = str(spec.get("shape") or "face")
    if shape == "heart":
        draw.polygon(_heart_points(cx, cy, s * 0.95), fill=main)
        draw.polygon(_heart_points(cx - s * 0.18, cy - s * 0.22,
                                   s * 0.34), fill=light)
    elif shape == "star":
        draw.polygon(_star_points(cx, cy, s * 0.98), fill=main,
                     outline=dark)
    elif shape == "moon":
        if img is None:  # pragma: no cover - callers pass the canvas
            draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=main)
        else:
            from PIL import ImageColor

            mask = Image.new("L", img.size, 0)
            md = _Draw(mask)
            md.ellipse([cx - s, cy - s, cx + s, cy + s], fill=255)
            md.ellipse([cx - s * 0.25, cy - s * 1.05,
                        cx + s * 1.35, cy + s * 0.55], fill=0)
            layer = Image.new(
                "RGBA", img.size,
                ImageColor.getrgb(main) + (255,))
            layer.putalpha(mask)
            img.alpha_composite(layer)
    elif shape == "sun":
        for i in range(8):
            a = i * math.pi / 4
            tip = (cx + s * 1.02 * math.cos(a),
                   cy + s * 1.02 * math.sin(a))
            base_a = (cx + s * 0.62 * math.cos(a - 0.32),
                      cy + s * 0.62 * math.sin(a - 0.32))
            base_b = (cx + s * 0.62 * math.cos(a + 0.32),
                      cy + s * 0.62 * math.sin(a + 0.32))
            draw.polygon([tip, base_a, base_b], fill=dark)
        draw.ellipse([cx - s * 0.66, cy - s * 0.66,
                      cx + s * 0.66, cy + s * 0.66], fill=main)
    elif shape == "flame":
        pts = []
        for i in range(48):
            t = 2 * math.pi * i / 48
            r = s * (0.72 + 0.28 * math.sin(t * 1.0 + 1.2))
            squash = 1.0 + 0.45 * math.sin(t - math.pi / 2)
            pts.append((cx + r * math.cos(t) * 0.78,
                        cy + r * math.sin(t) * squash))
        draw.polygon(pts, fill=main)
        pts2 = [(cx + (px - cx) * 0.52, cy + (py - cy) * 0.52
                 + s * 0.22) for px, py in pts]
        draw.polygon(pts2, fill=light)
    elif shape == "drop":
        draw.polygon([(cx, cy - s * 1.05),
                      (cx + s * 0.72, cy + s * 0.28),
                      (cx - s * 0.72, cy + s * 0.28)], fill=main)
        draw.ellipse([cx - s * 0.74, cy - s * 0.30,
                      cx + s * 0.74, cy + s * 1.02], fill=main)
        draw.ellipse([cx - s * 0.30, cy + s * 0.30,
                      cx - s * 0.05, cy + s * 0.62], fill=light)
    elif shape == "bolt":
        draw.polygon([(cx + s * 0.28, cy - s * 1.0),
                      (cx - s * 0.52, cy + s * 0.18),
                      (cx - s * 0.02, cy + s * 0.18),
                      (cx - s * 0.28, cy + s * 1.0),
                      (cx + s * 0.52, cy - s * 0.18),
                      (cx + s * 0.02, cy - s * 0.18)], fill=main,
                     outline=dark)
    elif shape == "disc":
        draw.ellipse([cx - s * 0.95, cy - s * 0.95,
                      cx + s * 0.95, cy + s * 0.95], fill=main)
        draw.ellipse([cx - s * 0.62, cy - s * 0.62,
                      cx + s * 0.62, cy + s * 0.62], fill=light)
        draw.ellipse([cx - s * 0.30, cy - s * 0.30,
                      cx + s * 0.30, cy + s * 0.30], fill=accent)
    elif shape == "flower":
        for i in range(6):
            a = i * math.pi / 3
            px, py = cx + s * 0.52 * math.cos(a), \
                cy + s * 0.52 * math.sin(a)
            draw.ellipse([px - s * 0.42, py - s * 0.42,
                          px + s * 0.42, py + s * 0.42], fill=light)
        draw.ellipse([cx - s * 0.34, cy - s * 0.34,
                      cx + s * 0.34, cy + s * 0.34], fill=main)
    elif shape == "crown":
        draw.polygon([(cx - s * 0.95, cy + s * 0.62),
                      (cx - s * 0.95, cy - s * 0.30),
                      (cx - s * 0.45, cy + s * 0.05),
                      (cx, cy - s * 0.62),
                      (cx + s * 0.45, cy + s * 0.05),
                      (cx + s * 0.95, cy - s * 0.30),
                      (cx + s * 0.95, cy + s * 0.62)], fill=main,
                     outline=dark)
        for dx in (-0.62, 0.0, 0.62):
            draw.ellipse([cx + s * dx - s * 0.10, cy + s * 0.28,
                          cx + s * dx + s * 0.10, cy + s * 0.48],
                         fill=accent)
    elif shape == "rocket":
        draw.polygon([(cx, cy - s * 1.02),
                      (cx + s * 0.46, cy - s * 0.1),
                      (cx + s * 0.46, cy + s * 0.52),
                      (cx - s * 0.46, cy + s * 0.52),
                      (cx - s * 0.46, cy - s * 0.1)], fill=light)
        draw.polygon([(cx - s * 0.46, cy + s * 0.3),
                      (cx - s * 0.92, cy + s * 0.78),
                      (cx - s * 0.46, cy + s * 0.62)], fill=dark)
        draw.polygon([(cx + s * 0.46, cy + s * 0.3),
                      (cx + s * 0.92, cy + s * 0.78),
                      (cx + s * 0.46, cy + s * 0.62)], fill=dark)
        draw.ellipse([cx - s * 0.22, cy - s * 0.52,
                      cx + s * 0.22, cy - s * 0.08], fill=main)
        draw.polygon([(cx - s * 0.18, cy + s * 0.98),
                      (cx, cy + s * 0.6),
                      (cx + s * 0.18, cy + s * 0.98)], fill=main)
    elif shape == "cat":
        draw.ellipse([cx - s * 0.82, cy - s * 0.55,
                      cx + s * 0.82, cy + s * 0.85], fill=main)
        draw.polygon([(cx - s * 0.78, cy - s * 0.42),
                      (cx - s * 0.62, cy - s * 1.0),
                      (cx - s * 0.18, cy - s * 0.62)], fill=main)
        draw.polygon([(cx + s * 0.78, cy - s * 0.42),
                      (cx + s * 0.62, cy - s * 1.0),
                      (cx + s * 0.18, cy - s * 0.62)], fill=main)
        _draw_face_features(draw, spec, cx, cy + s * 0.12, s,
                            palette, cat=True)
    elif shape == "letter":
        ch = str(spec.get("letter") or "✨")[:1] or "✨"
        font = _font(int(s * 1.9))
        draw.text((cx, cy), ch, font=font, fill=main,
                  anchor="mm", stroke_width=max(2, int(s * 0.06)),
                  stroke_fill=dark)
    else:  # face
        _draw_face(draw, spec, cx, cy, s, palette)


def _draw_face_features(draw, spec, cx, cy, s, palette,  # type: ignore[no-untyped-def]
                        *, cat: bool = False) -> None:
    main, dark, light, accent = palette
    ink = "#26221f" if not cat else "#26221f"
    face = str(spec.get("face") or "happy")
    ex, ey, r = s * 0.34, s * 0.10, max(3.0, s * 0.085)
    if face == "laugh":
        for dx in (-ex, ex):
            draw.arc([cx + dx - r * 1.7, cy - ey - r * 1.4,
                      cx + dx + r * 1.7, cy - ey + r * 1.4],
                     start=15, end=165, fill=ink,
                     width=max(3, int(s * 0.07)))
        draw.pieslice([cx - s * 0.34, cy + s * 0.16,
                       cx + s * 0.34, cy + s * 0.72],
                      start=0, end=180, fill=ink)
    elif face == "wink":
        draw.ellipse([cx - ex - r, cy - ey - r,
                      cx - ex + r, cy - ey + r], fill=ink)
        draw.line([cx + ex - r * 1.6, cy - ey,
                   cx + ex + r * 1.6, cy - ey], fill=ink,
                  width=max(3, int(s * 0.07)))
        draw.arc([cx - s * 0.32, cy + s * 0.02,
                  cx + s * 0.32, cy + s * 0.52],
                 start=15, end=165, fill=ink,
                 width=max(3, int(s * 0.07)))
    elif face == "sleepy":
        for dx in (-ex, ex):
            draw.arc([cx + dx - r * 1.7, cy - ey - r * 1.2,
                      cx + dx + r * 1.7, cy - ey + r * 1.2],
                     start=195, end=345, fill=ink,
                     width=max(3, int(s * 0.07)))
        draw.ellipse([cx - r * 0.9, cy + s * 0.34,
                      cx + r * 0.9, cy + s * 0.34 + r * 1.8],
                     outline=ink, width=max(3, int(s * 0.06)))
    elif face == "wow":
        for dx in (-ex, ex):
            draw.ellipse([cx + dx - r * 1.25, cy - ey - r * 1.25,
                          cx + dx + r * 1.25, cy - ey + r * 1.25],
                         fill=ink)
        draw.ellipse([cx - r * 1.5, cy + s * 0.30,
                      cx + r * 1.5, cy + s * 0.30 + r * 3.0],
                     fill=ink)
    elif face == "cool":
        draw.rectangle([cx - s * 0.78, cy - ey - r * 1.7,
                        cx + s * 0.78, cy - ey + r * 1.2],
                       fill=ink)
        draw.arc([cx - s * 0.30, cy + s * 0.10,
                  cx + s * 0.30, cy + s * 0.5],
                 start=20, end=160, fill=ink,
                 width=max(3, int(s * 0.07)))
    else:  # happy + default
        for dx in (-ex, ex):
            draw.ellipse([cx + dx - r, cy - ey - r,
                          cx + dx + r, cy - ey + r], fill=ink)
        draw.arc([cx - s * 0.32, cy + s * 0.02,
                  cx + s * 0.32, cy + s * 0.52],
                 start=15, end=165, fill=ink,
                 width=max(3, int(s * 0.07)))


def _draw_face(draw, spec, cx, cy, s, palette) -> None:  # type: ignore[no-untyped-def]
    main, dark, light, accent = palette
    draw.ellipse([cx - s * 0.98, cy - s * 0.98,
                  cx + s * 0.98, cy + s * 0.98], fill=main,
                 outline=dark, width=max(2, int(s * 0.05)))
    draw.ellipse([cx - s * 0.62, cy + s * 0.34,
                  cx - s * 0.30, cy + s * 0.62], fill=light)
    draw.ellipse([cx + s * 0.30, cy + s * 0.34,
                  cx + s * 0.62, cy + s * 0.62], fill=light)
    _draw_face_features(draw, spec, cx, cy, s, palette)


def render_master(spec: dict[str, Any], *, size: int = 512):  # type: ignore[no-untyped-def]
    """Render one design at ``size``×``size`` (RGBA, transparent
    ground unless the design asks for a badge disc)."""
    Image, ImageDraw, _Font = _pil()
    spec = dict(spec or {})
    palette = PALETTES.get(str(spec.get("palette") or "gold"),
                           PALETTES["gold"])
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    pad = size * 0.06
    if spec.get("bg") == "disc":
        draw.ellipse([pad, pad, size - pad, size - pad],
                     fill=palette[3])
    _draw_shape(draw, spec, (pad, pad, size - pad, size - pad),
                palette, img=img)
    if spec.get("sparkles"):
        rng = random.Random(_seed_for(spec))
        main, _d, light, _a = palette
        for _i in range(5):
            sx = rng.uniform(0.08, 0.92) * size
            sy = rng.uniform(0.08, 0.92) * size
            r = rng.uniform(0.02, 0.045) * size
            draw.polygon(_star_points(sx, sy, r, points=4,
                                      inner=0.22), fill=light)
    return img


def _save(img, out_dir: Any, name: str) -> Path:  # type: ignore[no-untyped-def]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    img.save(str(path))
    return path


def emoji_png(spec: dict[str, Any], out_dir: Any, *,
              name: str = "emoji") -> Path:
    """The custom-emoji file: the design at exactly 100×100
    PNG — the format Telegram custom-emoji packs carry."""
    Image, _Draw, _Font = _pil()
    img = render_master(spec, size=400)
    img = img.resize((100, 100), Image.LANCZOS)
    return _save(img, out_dir, f"{name}.png")


def sticker_master_png(spec: dict[str, Any], out_dir: Any, *,
                       name: str = "sticker") -> Path:
    """The 512×512 master PNG the sticker seam normalizes."""
    img = render_master(spec, size=512)
    return _save(img, out_dir, f"{name}_master.png")


#: mood word → (design, motion) for the library's self-created
#: animation floor: her own GIFs, rendered — no provider, no key.
_MOODS: dict[str, dict[str, Any]] = {
    "welcome": {"shape": "face", "face": "happy",
                "palette": "teal", "sparkles": True},
    "celebration": {"shape": "star", "palette": "gold",
                    "sparkles": True},
    "party": {"shape": "star", "palette": "pink",
              "sparkles": True},
    "win": {"shape": "crown", "palette": "gold",
            "sparkles": True},
    "jackpot": {"shape": "crown", "palette": "gold",
                "sparkles": True},
    "quest": {"shape": "rocket", "palette": "purple"},
    "ban": {"shape": "bolt", "palette": "red"},
    "hug": {"shape": "heart", "palette": "pink"},
    "love": {"shape": "heart", "palette": "red"},
    "laugh": {"shape": "face", "face": "laugh", "palette": "gold"},
    "dance": {"shape": "face", "face": "laugh", "palette": "purple",
              "sparkles": True},
    "wave": {"shape": "face", "face": "happy", "palette": "blue"},
    "fire": {"shape": "flame", "palette": "orange"},
    "sleep": {"shape": "moon", "palette": "night"},
    "default": {"shape": "face", "face": "happy",
                "palette": "purple", "sparkles": True},
}


def mood_design(mood: str) -> dict[str, Any]:
    """The studio design for a mood word (default when the
    mood is hers to invent)."""
    return dict(_MOODS.get(str(mood or "").lower().strip())
                or _MOODS["default"])


def mood_animation(mood: str, out_dir: Any, *,
                   name: str = "moment", frames: int = 14,
                   size: int = 320) -> Path:
    """A small looping GIF of the mood's design in motion —
    bounce + sway + (for sparkled designs) twinkling confetti.
    Deterministic per mood."""
    Image, ImageDraw, _Font = _pil()
    spec = mood_design(mood)
    palette = PALETTES.get(str(spec.get("palette") or "gold"),
                           PALETTES["gold"])
    rng = random.Random(_seed_for(spec) + 7)
    confetti = [(rng.uniform(0, size), rng.uniform(0, size),
                 rng.uniform(2.5, 6.0),
                 rng.choice([palette[0], palette[2], palette[3]]))
                for _i in range(26)] if spec.get("sparkles") else []
    out_frames = []
    for f in range(frames):
        phase = 2 * math.pi * f / frames
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        for (gx, gy, gr, color) in confetti:
            fall = (gy + f * size / frames * 1.7) % size
            draw.ellipse([gx - gr, fall - gr, gx + gr, fall + gr],
                         fill=color)
        bounce = abs(math.sin(phase)) * size * 0.07
        sway = math.sin(phase) * size * 0.03
        m = size * 0.16
        _draw_shape(draw, spec,
                    (m + sway, m + bounce * 0.4,
                     size - m + sway, size - m - bounce),
                    palette, img=img)
        out_frames.append(img.convert("P", palette=Image.ADAPTIVE,
                                      colors=128))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.gif"
    out_frames[0].save(
        str(path), save_all=True, append_images=out_frames[1:],
        duration=70, loop=0, disposal=2)
    return path


def menu_banner(out_dir: Any, *, name: str = "menu_banner") -> Path:
    """Her menu picture head: a 1024×512 banner — gradient
    field, her ring-and-bolt emblem, her name in type.  Drawn
    locally like everything else in the studio."""
    Image, ImageDraw, _Font = _pil()
    w, h = 1024, 512
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img, "RGBA")
    top = (43, 32, 84)
    bottom = (8, 110, 110)
    for y in range(h):
        t = y / h
        draw.line([(0, y), (w, y)],
                  fill=tuple(int(top[i] + (bottom[i] - top[i]) * t)
                             for i in range(3)))
    rng = random.Random(20261107)
    for _i in range(70):
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(1.2, 3.2)
        draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                     fill=(255, 255, 255, 90))
    palette = PALETTES["gold"]
    _draw_shape(draw, {"shape": "disc"},
                (70, 106, 370, 406), palette, img=img)
    _draw_shape(draw, {"shape": "bolt"},
                (130, 150, 310, 362), PALETTES["night"], img=img)
    font = _font(150)
    draw.text((440, 218), "DEVON", font=font, fill="#ffffff",
              anchor="lm")
    small = _font(44)
    draw.text((446, 318), "she does things herself",
              font=small, fill=(255, 255, 255, 190), anchor="lm")
    return _save(img, out_dir, f"{name}.png")
