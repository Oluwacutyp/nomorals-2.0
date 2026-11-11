"""Makers & tools (FEI gap wave E): the card makers, the
background remover, the standalone QR, the link shortener and
the bible lookup — one organ, blended into the systems Devon
already runs.

- **Quote cards** (/qcard + the census skins) render through
  Pillow — the same imaging stack as ``media_edit`` — as a
  polished chat-style CARD: a name, a medallion with the
  author's initial, the words.  The skins (TikTok-comment,
  SMS-chat, game-profile, meme-caption, two-person duo) are
  STYLES of the one renderer, never parallel code.
- **Platform designs** (owner's order, 2026-11-10 — this
  paragraph REPLACES the old "no platform chrome" law):
  beside the legacy skins, the one renderer carries real
  platform designs — Facebook, X, TikTok, Reddit, Instagram,
  Mastodon, Tinder, POF — each a data spec in
  ``PLATFORM_DESIGNS`` drawn in that platform's own visual
  language: its chrome, avatar shape, name/handle forms,
  action row and comment section (nested where the platform
  nests, flat where it doesn't).  Emulating a platform's
  layout on a generated card IS the feature the owner
  ordered; a card is still Devon's render, produced on
  request from words the user supplied — the designs are a
  registry, so a new platform is a new data entry, never new
  layout code.  The same registry carries the feature
  designs ``bigquote`` (the /quote card) and ``motivation``
  (the Motivation class's card): full-bleed backdrop cards
  with typography-service faces.
- **/removebg** rides the existing segmentation organ
  (``media_edit.segment`` — rembg, profile-aware models).  What
  this wave adds is the Devon self-provisioning rung on top
  (one bounded ``pip install`` attempt per process, in the
  ``ensure_ytdlp`` idiom) and a bounded first-run model pick.
  When no rung can run, the answer is one honest line — never
  a silent downgrade to chroma keying (the segment organ's own
  law).
- **/qr** generates with the ``qrcode`` library (the in-tree
  pattern: payments, Discord login) behind the same
  self-provisioning rung, and READS through the vision organ's
  real decoder (``vision.native.read_qr``, pyzbar) — its
  unavailable state surfaces as-is, never a guessed decode.
- **/shorturl** walks keyless shortener seeds with the shared
  health breaker and always shows the full destination next to
  the short link; **/unshorten** walks redirect hops without
  downloading bodies.
- **/bible** is a plain lookup over bible-api.com (primary)
  and bolls.life (chapter fallback + the search the primary
  lacks) — verse text quoted exactly from the response,
  attributed to its source.  It is NOT WisdomKeeper: that
  organ is corpus and practice; this one fetches a passage.
- **/couplepp** takes ONE artwork from the picture pools
  and splits it into a matched pair, delivered as TWO
  separate profile-ready photos — the left half for one
  person, the right half for the other — never two unrelated
  pictures passed off as a designed pair, and never one
  combined poster the couple must crop themselves
  (delivery-shape pass, 2026-11-07: a couple pfp IS two
  pictures).  The pools are seeds, not a closed list, and
  not anime-only: anime pools, a safe-rated general anime
  pool and a general photography pool all feed the chain,
  with her own studio render as the floor.

Like ``fun.py`` beside it, the organ is transport-free:
providers fetch, the service walks health-checked chains,
outcomes are plain dicts, and the runtime mixin
(``agents/partner/runtime_makers.py``) owns the live wiring —
the §33 group gate and media delivery through the expression
send path.  Media kinds ride ``expression.MEDIA_DELIVERED_KINDS``:
success returns "" because the picture WAS the reply.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .expression import ProviderHealth
from .failures import record_failure
from .fun import _sniff_image, _trim_caption, picture_shape

__all__ = [
    "MAKER_COMMANDS", "CARD_KINDS", "CARD_STYLES",
    "CARD_DESIGNS", "PLATFORM_DESIGNS", "FEATURE_DESIGNS",
    "MEDIA_KINDS", "MakerError", "MakerService",
    "default_maker_chains",
    "render_card_png", "parse_card_request", "card_backdrop",
    "rembg_available", "ensure_rembg", "pick_rembg_model",
    "remove_background_bytes",
    "qrcode_available", "ensure_qrcode", "make_qr_png",
    "read_qr_codes",
    "IsGdProvider", "VGdProvider", "TinyUrlProvider",
    "validate_long_url", "resolve_redirect_chain",
    "BIBLE_BOOKS", "BibleRef", "parse_bible_tail",
    "BibleApiProvider", "BollsProvider", "clean_bolls_text",
    "WaifuPoolProvider", "NekosPoolProvider",
    "NekosApiPoolProvider", "PicsumPoolProvider",
    "make_diptych", "split_diptych",
]


class MakerError(Exception):
    """A named failure whose message is fit to reach the user:
    it says what actually happened (a rate limit, an
    unprovisioned rung, a shapeless source) — never a guess
    dressed as a result."""


#: Every wave-E kind, in registry order.  Naming (recorded):
#: /quote is wave D's text-quote command and was NOT taken;
#: the card maker is /qcard.  The FEI census skins are
#: separate kinds over the one maker (the /tl alias
#: precedent): canvasquote/iqc/wndwqute → the quote style,
#: meme2 → the meme style, fakettc → ttc, nokiamsg → chat,
#: fakeff/ffgirl → ff, fakeffduo → duo, pp → /couplepp.
MAKER_COMMANDS: tuple[str, ...] = (
    "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
    "fakemsg", "fakettc", "nokiamsg", "fakeff", "fakeffduo",
    "ffgirl",
    "removebg", "qr", "shorturl", "unshorten", "bible",
    "couplepp", "pp",
)

#: Card kinds → the style they imply.  /qcard and /fakemsg
#: take a style as their first word when it names one.
CARD_KINDS: dict[str, str] = {
    "qcard": "quote",
    "canvasquote": "quote",
    "iqc": "quote",
    "wndwqute": "quote",
    "meme2": "meme",
    "fakemsg": "quote",
    "fakettc": "ttc",
    "nokiamsg": "chat",
    "fakeff": "ff",
    "fakeffduo": "duo",
    "ffgirl": "ff",
}

#: The styles the one renderer draws.
CARD_STYLES: tuple[str, ...] = ("quote", "ttc", "chat", "ff",
                                "meme", "duo")

#: Kinds whose success is a media send through the expression
#: path — they join expression.MEDIA_DELIVERED_KINDS.
MEDIA_KINDS: frozenset[str] = frozenset({
    "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
    "fakemsg", "fakettc", "nokiamsg", "fakeff", "fakeffduo",
    "ffgirl", "removebg", "qr", "couplepp", "pp",
})


# ── http (the house urllib shape, makers UA) ─────────────────

def _http_json(url: str, *, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-makers/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(url: str, *, timeout: float = 15.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-makers/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _http_bytes(url: str, *, timeout: float = 30.0,
                cap: int = 20 * 1024 * 1024) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-makers/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(cap + 1)
    if len(data) > cap:
        raise ValueError("media exceeds the 20MB cap")
    return data


def _need_text(value: Any, what: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"maker source shape: missing {what}")
    return text


# ── the quote-card renderer (Pillow) ─────────────────────────

#: Medallion colours, picked deterministically from the name —
#: the same author wears the same colour on every card.
_MEDALLION_COLORS = (
    (0x5E, 0xB2, 0xF2), (0xF2, 0x8C, 0x5E), (0x7D, 0xC8, 0x6A),
    (0xC8, 0x6A, 0xB8), (0xE8, 0xC3, 0x4A), (0x6A, 0xC8, 0xB8),
    (0xB8, 0x6A, 0x6A), (0x8C, 0x7D, 0xE8),
)

_CARD_W = 920
_PAD = 40


def _font(size: int, *, bold: bool = False, intent: str = "text"):
    """Card faces come from the typography service (semantic
    selection over the parsed catalogue); the legacy ladder —
    DejaVu / Noto / Liberation on Linux, Arial on Windows (the
    EC2 host), Pillow's bundled default last — remains the
    fallback when the service has no face, so a card renders on
    every profile.  ``intent`` is the typography service's own
    vocabulary (text / display / editorial / …): designs pick
    the intent that matches their platform's typographic feel
    — the ONE theme system's faces, never a private font
    hunt."""
    import os

    from PIL import ImageFont

    try:
        from ...media.typography import get_service
        picked = get_service().pil_font(
            intent, size, weight=700 if bold else 400)
    except Exception:  # noqa: BLE001 - legacy ladder below
        picked = None
    if picked is not None:
        return picked
    names = (("DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf",
              "LiberationSans-Bold.ttf", "arialbd.ttf")
             if bold else
             ("DejaVuSans.ttf", "NotoSans-Regular.ttf",
              "LiberationSans-Regular.ttf", "arial.ttf"))
    roots = ("/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/truetype/noto",
             "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/TTF",
             os.path.join(os.environ.get("WINDIR", "C:\\Windows"),
                          "Fonts"))
    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
    return ImageFont.load_default()


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", str(name or "").strip())
             if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:1].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


def _medallion_color(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(
        str(name or "").strip().lower().encode("utf-8")).digest()
    return _MEDALLION_COLORS[digest[0] % len(_MEDALLION_COLORS)]


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap by measured pixel width (never a char-count
    guess): long words split rather than overflow the card."""
    lines: list[str] = []
    for para in str(text or "").split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        line = ""
        for word in words:
            while draw.textlength(word, font=font) > max_width:
                cut = max(1, len(word) // 2)
                head, word = word[:cut], word[cut:]
                if line:
                    lines.append(line)
                    line = ""
                lines.append(head)
            trial = f"{line} {word}".strip()
            if line and draw.textlength(trial, font=font) > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
    return lines or [""]


def _draw_medallion(draw, xy: tuple[int, int], size: int,
                    name: str) -> None:
    x, y = xy
    draw.ellipse([x, y, x + size, y + size],
                 fill=_medallion_color(name))
    font = _font(int(size * 0.42), bold=True)
    label = _initials(name)
    box = draw.textbbox((0, 0), label, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text((x + (size - tw) / 2 - box[0],
               y + (size - th) / 2 - box[1]),
              label, font=font, fill=(255, 255, 255))


def _load_avatar_image(avatar):
    """Normalise an avatar hand-off — PIL image, raw bytes, or
    a local file path — into a PIL image, or None when the
    hand-off carries nothing usable.  Never raises: a bad
    avatar degrades to the placeholder disc, never a broken
    card."""
    if avatar is None:
        return None
    try:
        from PIL import Image

        if isinstance(avatar, Image.Image):
            return avatar
        if isinstance(avatar, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(avatar)))
            img.load()
            return img
        if isinstance(avatar, (str, Path)):
            img = Image.open(avatar)
            img.load()
            return img
    except Exception:  # noqa: BLE001 - placeholder is the fallback
        return None
    return None


def _paste_avatar(canvas, xy: tuple[int, int], size: int,
                  avatar) -> bool:
    """Paste a supplied avatar image into the medallion disc.
    Returns False when no avatar was supplied — the caller
    draws the initial medallion instead.  Chat surfaces hand
    avatar bytes over through ``MakerService.card(avatars=…)``
    (the runtime resolves them from message meta when a
    surface stashes them); a surface with none gets the
    placeholder disc, never a broken box."""
    img = _load_avatar_image(avatar)
    if img is None:
        return False
    from PIL import Image, ImageDraw

    disc = img.convert("RGB").resize((size, size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
    canvas.paste(disc, xy, mask)
    return True


def _paste_avatar_shape(canvas, xy: tuple[int, int], size: int,
                        avatar, shape: str) -> bool:
    """The platform-design avatar paste: ``circle`` discs, or
    ``rounded`` squares (Mastodon / POF wear rounded squares).
    Same hand-off contract as :func:`_paste_avatar`."""
    img = _load_avatar_image(avatar)
    if img is None:
        return False
    from PIL import Image, ImageDraw

    x, y = xy
    tile = img.convert("RGB").resize((size, size))
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    if shape == "rounded":
        md.rounded_rectangle([0, 0, size, size],
                             radius=max(4, size // 5), fill=255)
    else:
        md.ellipse([0, 0, size, size], fill=255)
    canvas.paste(tile, (x, y), mask)
    return True


def _draw_placeholder(draw, xy: tuple[int, int], size: int,
                      name: str, fill, shape: str = "circle",
                      ring=None) -> None:
    """The no-avatar placeholder: the author's initials on a
    disc/square in the DESIGN's own colours (never a broken
    box, never a grey shrug).  ``ring`` draws the design's
    avatar ring (Instagram's gradient band reads as a solid
    warm ring at card scale)."""
    x, y = xy
    if ring is not None:
        grow = max(3, size // 14)
        box = [x - grow, y - grow, x + size + grow, y + size + grow]
        if shape == "rounded":
            draw.rounded_rectangle(
                box, radius=max(4, size // 5) + grow, fill=ring)
        else:
            draw.ellipse(box, fill=ring)
    if shape == "rounded":
        draw.rounded_rectangle([x, y, x + size, y + size],
                               radius=max(4, size // 5), fill=fill)
    else:
        draw.ellipse([x, y, x + size, y + size], fill=fill)
    font = _font(int(size * 0.42), bold=True)
    label = _initials(name)
    box = draw.textbbox((0, 0), label, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text((x + (size - tw) / 2 - box[0],
               y + (size - th) / 2 - box[1]),
              label, font=font, fill=(255, 255, 255))


#: Per-style canvas grammar: (background, bubble, name colour,
#: text colour).  The styles are skins of ONE layout — quote
#: (dark bubble), ttc (white comment row), chat (SMS beige),
#: ff (game profile: navy + gold frame, name as player tag).
_STYLE_THEMES = {
    "quote": {"bg": (0x0F, 0x14, 0x1A), "bubble": (0x1C, 0x27, 0x33),
              "name": (0x5E, 0xB2, 0xF2), "text": (0xF5, 0xF8, 0xFA),
              "frame": None},
    "ttc": {"bg": (0xFF, 0xFF, 0xFF), "bubble": (0xFF, 0xFF, 0xFF),
            "name": (0x16, 0x18, 0x23), "text": (0x16, 0x18, 0x23),
            "frame": None},
    "chat": {"bg": (0xE5, 0xDD, 0xD2), "bubble": (0xFF, 0xFF, 0xFF),
             "name": (0x07, 0x5E, 0x54), "text": (0x11, 0x11, 0x11),
             "frame": None},
    "ff": {"bg": (0x0B, 0x1E, 0x3A), "bubble": (0x12, 0x2C, 0x52),
           "name": (0xF2, 0xC1, 0x4A), "text": (0xF5, 0xF8, 0xFA),
           "frame": (0xF2, 0xC1, 0x4A)},
    "meme": {"bg": (0x11, 0x11, 0x11), "bubble": (0x11, 0x11, 0x11),
             "name": (0x8A, 0x8A, 0x8A), "text": (0xFF, 0xFF, 0xFF),
             "frame": None},
}


def render_card_png(style: str, entries: list[dict], *,
                    avatar=None, backdrop=None
                    ) -> tuple[bytes, dict[str, Any]]:
    """Render ONE card → (PNG bytes, drawn record).

    ``style`` names a legacy skin (``CARD_STYLES``), a
    platform design (``PLATFORM_DESIGNS`` — entries[0] is the
    post, the rest its comment section) or a feature design
    (``FEATURE_DESIGNS`` — one entry over a backdrop).
    Entries are ``{"name", "text"}`` dicts; platform entries
    may also carry ``handle`` / ``meta`` / ``likes`` /
    ``depth`` (comment nesting) / ``avatar`` (image bytes or
    a PIL image handed over by the surface).  The drawn
    record is the renderer's own seam — what was drawn, at
    what size, in which regions — so tests assert against it
    instead of OCR-ing pixels.
    """
    from PIL import Image, ImageDraw

    style = str(style or "quote").strip().lower()
    if style not in CARD_DESIGNS:
        raise MakerError(f"unknown card style {style!r} — "
                         f"pick from: {', '.join(CARD_DESIGNS)}")
    clean = [{"name": str(e.get("name") or "").strip()[:48],
              "text": str(e.get("text") or "").strip()[:600],
              **{k: e[k] for k in
                 ("handle", "meta", "likes", "depth", "avatar",
                  "title") if k in e}}
             for e in entries if str(e.get("text") or "").strip()]
    if not clean:
        raise MakerError("a card needs words — nothing to draw")
    if style in FEATURE_DESIGNS:
        return _render_feature(style, clean, backdrop=backdrop)
    if style in PLATFORM_DESIGNS:
        return _render_platform(style, clean)
    entries = clean
    if style == "duo" and len(entries) < 2:
        raise MakerError("a duo card needs two speakers")
    theme = _STYLE_THEMES[style if style != "duo" else "quote"]
    name_font = _font(34, bold=True)
    text_font = _font(37)
    big_font = _font(54, bold=True)

    probe = ImageDraw.Draw(Image.new("RGB", (_CARD_W, 100)))
    med = 84
    if style == "meme":
        text_w = _CARD_W - 2 * _PAD
        wrapped = [_wrap(probe, e["text"].upper(), big_font, text_w)
                   for e in entries]
        heights = [len(w) * 68 + 30 for w in wrapped]
        height = _PAD * 2 + sum(heights) + 44
    elif style == "duo":
        text_w = int(_CARD_W * 0.68)
        wrapped = [_wrap(probe, e["text"], text_font, text_w)
                   for e in entries[:2]]
        heights = [58 + len(w) * 48 + 26 for w in wrapped]
        height = _PAD * 2 + sum(heights) + 18 * (len(heights) - 1)
    else:
        text_w = _CARD_W - 2 * _PAD - med - 28
        wrapped = [_wrap(probe, entries[0]["text"], text_font,
                         text_w)]
        height = _PAD * 2 + max(med, 46 + len(wrapped[0]) * 48) + 8

    img = Image.new("RGB", (_CARD_W, int(height)), theme["bg"])
    draw = ImageDraw.Draw(img)
    if theme["frame"]:
        draw.rectangle([6, 6, _CARD_W - 7, int(height) - 7],
                       outline=theme["frame"], width=4)

    drawn: dict[str, Any] = {"style": style,
                             "size": (_CARD_W, int(height)),
                             "names": [], "texts": []}
    if style == "meme":
        y = _PAD
        for entry, lines in zip(entries, wrapped, strict=True):
            for line in lines:
                tw = draw.textlength(line, font=big_font)
                draw.text(((_CARD_W - tw) / 2, y), line,
                          font=big_font, fill=theme["text"])
                y += 68
            y += 30
            drawn["names"].append(entry["name"])
            drawn["texts"].append(entry["text"])
        who = " — ".join(n for n in drawn["names"] if n)
        if who:
            font = _font(26)
            tw = draw.textlength(f"— {who}", font=font)
            draw.text(((_CARD_W - tw) / 2, y), f"— {who}",
                      font=font, fill=theme["name"])
        return _png_bytes(img), drawn

    if style == "duo":
        y = _PAD
        for idx, (entry, lines) in enumerate(
                zip(entries[:2], wrapped, strict=True)):
            bw = int(_CARD_W * 0.78)
            x = _PAD if idx == 0 else _CARD_W - _PAD - bw
            bubble_fill = (0x1C, 0x27, 0x33) if idx == 0 \
                else (0x14, 0x3D, 0x2B)
            name_fill = (0x5E, 0xB2, 0xF2) if idx == 0 \
                else (0x6A, 0xC8, 0x7D)
            draw.rounded_rectangle(
                [x, y, x + bw, y + heights[idx]], radius=22,
                fill=bubble_fill)
            draw.text((x + 24, y + 14), entry["name"],
                      font=_font(28, bold=True), fill=name_fill)
            ty = y + 58
            for line in lines:
                draw.text((x + 24, ty), line, font=text_font,
                          fill=(0xF5, 0xF8, 0xFA))
                ty += 48
            drawn["names"].append(entry["name"])
            drawn["texts"].append(entry["text"])
            y += heights[idx] + 18
        return _png_bytes(img), drawn

    entry = entries[0]
    draw.rounded_rectangle(
        [_PAD - 14, _PAD - 14, _CARD_W - _PAD + 14, height - _PAD + 14],
        radius=26, fill=theme["bubble"])
    if not _paste_avatar(img, (_PAD, _PAD), med,
                         avatar if avatar is not None
                         else entry.get("avatar")):
        _draw_medallion(draw, (_PAD, _PAD), med, entry["name"])
    tx = _PAD + med + 28
    draw.text((tx, _PAD), entry["name"], font=name_font,
              fill=theme["name"])
    ty = _PAD + 46
    for line in wrapped[0]:
        draw.text((tx, ty), line, font=text_font,
                  fill=theme["text"])
        ty += 48
    drawn["names"].append(entry["name"])
    drawn["texts"].append(entry["text"])
    return _png_bytes(img), drawn


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── the design registry (platforms + feature cards) ──────────
#
# A design is DATA: colours, avatar shape, name/handle form,
# header kind, action row (or rail), comment geometry.  The
# renderer below is the only layout code; a new platform is a
# new entry here.  Colour values are the platforms' own brand
# colours as they render on their real surfaces.

PLATFORM_DESIGNS: dict[str, dict[str, Any]] = {
    "facebook": {
        "label": "Facebook", "dark": False,
        "bg": (0xF0, 0xF2, 0xF5), "surface": (0xFF, 0xFF, 0xFF),
        "text": (0x05, 0x05, 0x05), "sub": (0x65, 0x76, 0x7B),
        "accent": (0x18, 0x76, 0xF2), "divider": (0xE4, 0xE6, 0xEB),
        "avatar_shape": "circle", "avatar_ring": None,
        "avatar_bg": (0x18, 0x76, 0xF2),
        "name_form": "name", "badge": (0x18, 0x76, 0xF2),
        "header": "bar", "rail": "",
        "actions": ("Like", "Comment", "Share"), "like_at": 0,
        "comment_indent": 44, "comment_actions": True,
        "thread_line": False, "font": "text",
    },
    "x": {
        "label": "X", "dark": True,
        "bg": (0x00, 0x00, 0x00), "surface": (0x00, 0x00, 0x00),
        "text": (0xE7, 0xE9, 0xEA), "sub": (0x71, 0x72, 0x74),
        "accent": (0x1D, 0x9B, 0xF0), "divider": (0x2F, 0x33, 0x35),
        "avatar_shape": "circle", "avatar_ring": None,
        "avatar_bg": (0x1D, 0x9B, 0xF0),
        "name_form": "name_handle", "badge": (0x1D, 0x9B, 0xF0),
        "header": "minimal", "rail": "",
        "actions": ("Reply", "Repost", "Like", "Views"),
        "like_at": 2,
        "comment_indent": 0, "comment_actions": False,
        "thread_line": True, "font": "text",
    },
    "tiktok": {
        "label": "TikTok", "dark": True,
        "bg": (0x12, 0x12, 0x12), "surface": (0x12, 0x12, 0x12),
        "comment_surface": (0x1E, 0x1E, 0x1E),
        "text": (0xFF, 0xFF, 0xFF), "sub": (0xA8, 0xA8, 0xA8),
        "accent": (0xFE, 0x2C, 0x55), "divider": (0x2A, 0x2A, 0x2A),
        "avatar_shape": "circle", "avatar_ring": None,
        "avatar_bg": (0x25, 0xF4, 0xEE),
        "name_form": "handle", "badge": (0xFE, 0x2C, 0x55),
        "header": "none", "rail": "right",
        "actions": ("Like", "Comment", "Share"), "like_at": 0,
        "comment_indent": 0, "comment_actions": True,
        "thread_line": False, "font": "text",
    },
    "reddit": {
        "label": "Reddit", "dark": False,
        "bg": (0xDA, 0xE0, 0xE6), "surface": (0xFF, 0xFF, 0xFF),
        "text": (0x1A, 0x1A, 0x1B), "sub": (0x7C, 0x7C, 0x7C),
        "accent": (0xFF, 0x45, 0x00), "divider": (0xE3, 0xE6, 0xE8),
        "avatar_shape": "circle", "avatar_ring": None,
        "avatar_bg": (0xFF, 0x45, 0x00),
        "name_form": "u/name", "badge": None,
        "header": "bar", "rail": "left-vote",
        "actions": ("Comments", "Share", "Save"), "like_at": -1,
        "comment_indent": 36, "comment_actions": True,
        "thread_line": True, "font": "text",
    },
    "instagram": {
        "label": "Instagram", "dark": False,
        "bg": (0xFF, 0xFF, 0xFF), "surface": (0xFF, 0xFF, 0xFF),
        "text": (0x26, 0x26, 0x26), "sub": (0x8E, 0x8E, 0x8E),
        "accent": (0x00, 0x95, 0xF6), "divider": (0xDB, 0xDB, 0xDB),
        "avatar_shape": "circle",
        "avatar_ring": (0xF5, 0x85, 0x3C),
        "avatar_bg": (0x8A, 0x3F, 0xC8),
        "name_form": "name", "badge": (0x00, 0x95, 0xF6),
        "header": "bar", "rail": "",
        "actions": ("Like", "Comment", "Share", "Save"),
        "like_at": 0,
        "comment_indent": 0, "comment_actions": False,
        "thread_line": False, "font": "text",
    },
    "mastodon": {
        "label": "Mastodon", "dark": True,
        "bg": (0x19, 0x1B, 0x22), "surface": (0x28, 0x2C, 0x37),
        "text": (0xD9, 0xE1, 0xE8), "sub": (0x9B, 0xA8, 0xB8),
        "accent": (0x63, 0x64, 0xFF), "divider": (0x39, 0x3E, 0x52),
        "avatar_shape": "rounded", "avatar_ring": None,
        "avatar_bg": (0x63, 0x64, 0xFF),
        "name_form": "name_handle", "badge": None,
        "header": "bar", "rail": "",
        "actions": ("Reply", "Boost", "Favourite", "Share"),
        "like_at": 2,
        "comment_indent": 30, "comment_actions": True,
        "thread_line": False, "font": "text",
    },
    "tinder": {
        "label": "Tinder", "dark": False,
        "bg": (0xFF, 0xFF, 0xFF), "surface": (0xFF, 0xFF, 0xFF),
        "band": ((0xFD, 0x26, 0x6D), (0xFF, 0x78, 0x57)),
        "text": (0x2B, 0x2B, 0x2B), "sub": (0x8E, 0x8E, 0x93),
        "accent": (0xFD, 0x26, 0x6D), "divider": (0xEC, 0xEC, 0xEC),
        "avatar_shape": "circle", "avatar_ring": None,
        "avatar_bg": (0xFD, 0x26, 0x6D),
        "name_form": "name", "badge": None,
        "header": "band", "rail": "",
        "actions": ("Pass", "Like", "Super Like"), "like_at": 1,
        "comment_indent": 0, "comment_actions": False,
        "thread_line": False, "comment_bubbles": True,
        "font": "text",
    },
    "pof": {
        "label": "POF", "dark": False,
        "bg": (0xE8, 0xEE, 0xF3), "surface": (0xFF, 0xFF, 0xFF),
        "text": (0x1F, 0x2A, 0x37), "sub": (0x5F, 0x74, 0x8A),
        "accent": (0x00, 0x7A, 0xB8), "divider": (0xC9, 0xD6, 0xE2),
        "avatar_shape": "rounded", "avatar_ring": None,
        "avatar_bg": (0x00, 0x7A, 0xB8),
        "name_form": "name", "badge": None,
        "header": "bar", "rail": "",
        "actions": ("Message", "Favourite", "Meet Me"),
        "like_at": 1,
        "comment_indent": 0, "comment_actions": True,
        "thread_line": False, "font": "text",
    },
}

#: The feature designs: full-bleed backdrop cards (the /quote
#: card and the Motivation class's card).  ``gradients`` are
#: (top, bottom) pairs drawn when no backdrop image rides;
#: ``meta`` on the entry picks the variant (motivation:
#: "evening" wears the night gradient, anything else dawn).
FEATURE_DESIGNS: dict[str, dict[str, Any]] = {
    "bigquote": {
        "label": "Quote", "size": 1080,
        "gradient": ((0x0B, 0x1E, 0x3A), (0x0E, 0x4D, 0x5C)),
        "accent": (0xE8, 0xC3, 0x4A), "font": "editorial",
    },
    "motivation": {
        "label": "Motivation", "size": 1080,
        "gradient": ((0x1B, 0x1B, 0x3A), (0xB4, 0x5A, 0x1B)),
        "gradient_evening": ((0x05, 0x08, 0x1A), (0x1D, 0x2A, 0x52)),
        "accent": (0xF2, 0xC1, 0x4A), "font": "display",
    },
}

#: Every style/design name the one renderer accepts: the six
#: legacy skins + the platform designs + the feature designs.
CARD_DESIGNS: tuple[str, ...] = (
    CARD_STYLES + tuple(PLATFORM_DESIGNS) + tuple(FEATURE_DESIGNS))


def _handle_of(name: str) -> str:
    """A display name as a platform handle: lower-cased, one
    token, platform-safe characters only."""
    handle = re.sub(r"[^a-z0-9_.]+", "", str(name or "").lower())
    return handle or "someone"


def _name_line(spec: dict[str, Any], entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or "Someone")
    form = spec.get("name_form", "name")
    if form == "handle":
        return "@" + _handle_of(name)
    if form == "u/name":
        return "u/" + _handle_of(name)
    if form == "name_handle":
        return f"{name}  @{_handle_of(name)}"
    return name


def _draw_badge(draw, xy: tuple[int, int], size: int, fill) -> None:
    """A verified badge: a filled disc with a white check —
    the shape every platform's badge reduces to at card
    scale."""
    x, y = xy
    draw.ellipse([x, y, x + size, y + size], fill=fill)
    s = size
    draw.line([x + s * 0.28, y + s * 0.52, x + s * 0.44, y + s * 0.68],
              fill=(255, 255, 255), width=max(2, s // 6))
    draw.line([x + s * 0.44, y + s * 0.68, x + s * 0.74, y + s * 0.34],
              fill=(255, 255, 255), width=max(2, s // 6))


def _avatar_at(canvas, draw, xy, size, entry, spec) -> None:
    """One entry's avatar: the handed-over image when one
    rides the entry, else the design-coloured initials
    placeholder."""
    avatar = entry.get("avatar")
    if avatar is not None and _paste_avatar_shape(
            canvas, xy, size, avatar, spec["avatar_shape"]):
        return
    _draw_placeholder(draw, xy, size, str(entry.get("name") or "?"),
                      spec["avatar_bg"], spec["avatar_shape"],
                      ring=spec.get("avatar_ring"))


def _render_platform(key: str, entries: list[dict[str, Any]]
                     ) -> tuple[bytes, dict[str, Any]]:
    """Draw one platform design: entries[0] is the post,
    entries[1:] its comment section.  The drawn record
    carries the layout's regions (header / post / actions /
    comment boxes / avatar boxes) so tests assert structure,
    not pixels alone."""
    from PIL import Image, ImageDraw

    spec = PLATFORM_DESIGNS[key]
    post, comments = entries[0], entries[1:]
    width = _CARD_W
    rail_right = spec.get("rail") == "right"
    vote_rail = spec.get("rail") == "left-vote"
    content_x = _PAD + (64 if vote_rail else 0)
    content_w = width - content_x - _PAD - (96 if rail_right else 0)
    header_h = {"bar": 64, "minimal": 52, "band": 120}.get(
        spec["header"], 0)
    name_font = _font(32, bold=True, intent=spec["font"])
    meta_font = _font(24, intent=spec["font"])
    text_font = _font(36, intent=spec["font"])
    c_name_font = _font(27, bold=True, intent=spec["font"])
    c_text_font = _font(29, intent=spec["font"])
    act_font = _font(25, bold=True, intent=spec["font"])

    probe = ImageDraw.Draw(Image.new("RGB", (width, 100)))
    post_lines = _wrap(probe, post["text"], text_font,
                       content_w - 84 - 24)
    comment_rows: list[tuple[dict[str, Any], list[str]]] = []
    for entry in comments:
        indent = int(entry.get("depth") or 0) * int(
            spec["comment_indent"])
        bubble = 26 if spec.get("comment_bubbles") else 0
        rows = _wrap(probe, entry["text"], c_text_font,
                     content_w - 56 - 20 - indent - bubble)
        comment_rows.append((entry, rows))

    post_h = 84 + 46 + len(post_lines) * 47 + 56
    comments_h = 0
    if comment_rows:
        comments_h = 54 + sum(
            40 + len(rows) * 38 + (30 if spec["comment_actions"]
                                   else 8) + 18
            for _e, rows in comment_rows)
    height = header_h + post_h + comments_h + _PAD

    img = Image.new("RGB", (width, int(height)), spec["bg"])
    draw = ImageDraw.Draw(img)
    regions: dict[str, Any] = {"avatars": [], "comments": []}
    drawn: dict[str, Any] = {
        "style": key, "design": key, "platform": spec["label"],
        "size": (width, int(height)),
        "names": [str(e.get("name") or "") for e in entries],
        "texts": [str(e.get("text") or "") for e in entries],
        "fingerprint": {
            "bg": spec["bg"], "surface": spec["surface"],
            "accent": spec["accent"],
            "avatar_shape": spec["avatar_shape"],
            "name_form": spec["name_form"],
            "header": spec["header"], "rail": spec.get("rail", ""),
            "actions": tuple(spec["actions"]),
            "comment_indent": int(spec["comment_indent"]),
        },
        "regions": regions,
    }

    y = 0
    if spec["header"] == "bar":
        draw.rectangle([0, 0, width, header_h], fill=spec["surface"])
        draw.rectangle([0, header_h - 2, width, header_h],
                       fill=spec["divider"])
        draw.ellipse([_PAD, 14, _PAD + 36, 50], fill=spec["accent"])
        draw.text((_PAD + 52, 14), spec["label"],
                  font=_font(30, bold=True, intent="display"),
                  fill=spec["accent"])
        regions["header"] = (0, 0, width, header_h)
        y = header_h
    elif spec["header"] == "minimal":
        draw.text((_PAD, 10), "✕",
                  font=_font(34, bold=True, intent="display"),
                  fill=spec["text"])
        regions["header"] = (0, 0, width, header_h)
        y = header_h
    elif spec["header"] == "band":
        top, bottom = spec["band"]
        for i in range(header_h):
            frac = i / max(1, header_h - 1)
            row = tuple(int(top[c] + (bottom[c] - top[c]) * frac)
                        for c in range(3))
            draw.line([0, i, width, i], fill=row)
        draw.text((_PAD, 22), spec["label"],
                  font=_font(30, bold=True, intent="display"),
                  fill=(255, 255, 255))
        regions["header"] = (0, 0, width, header_h)
        y = header_h

    post_y = y + 20
    draw.rounded_rectangle(
        [content_x - 16, post_y - 12, content_x + content_w + 16,
         post_y + post_h - 30], radius=18, fill=spec["surface"])
    if vote_rail:
        vx = _PAD + 8
        draw.polygon([(vx + 14, post_y + 2), (vx, post_y + 20),
                      (vx + 28, post_y + 20)], fill=spec["accent"])
        votes = str(post.get("likes") or "128")
        draw.text((vx, post_y + 28), votes, font=meta_font,
                  fill=spec["text"])
        draw.polygon([(vx, post_y + 62), (vx + 28, post_y + 62),
                      (vx + 14, post_y + 80)], fill=spec["sub"])
        regions["vote_rail"] = (vx, post_y, 30, 84)
    _avatar_at(img, draw, (content_x, post_y), 84, post, spec)
    draw = ImageDraw.Draw(img)
    regions["avatars"].append((content_x, post_y, 84, 84))
    tx = content_x + 84 + 24
    name = _name_line(spec, post)
    draw.text((tx, post_y), name, font=name_font, fill=spec["text"])
    if spec.get("badge") is not None:
        nw = draw.textlength(name, font=name_font)
        _draw_badge(draw, (int(tx + nw + 10), post_y + 6), 22,
                    spec["badge"])
    meta = str(post.get("meta") or "Just now")
    draw.text((tx, post_y + 42), meta, font=meta_font,
              fill=spec["sub"])
    ty = post_y + 84 + 8
    for line in post_lines:
        draw.text((content_x, ty), line, font=text_font,
                  fill=spec["text"])
        ty += 47
    act_y = ty + 10
    if rail_right:
        rail_x = width - _PAD - 64
        ry = post_y
        regions["rail"] = (rail_x, post_y, 64,
                           3 * 92)
        for i, action in enumerate(spec["actions"]):
            cy = ry + i * 92
            fill = spec["accent"] if i == int(spec["like_at"]) \
                else spec["surface"]
            if key == "tiktok":
                fill = (0x2A, 0x2A, 0x2A) if i != int(
                    spec["like_at"]) else spec["accent"]
            draw.ellipse([rail_x, cy, rail_x + 64, cy + 64],
                         fill=fill)
            glyph = {"Like": "♥", "Comment": "…",
                     "Share": "↗"}.get(action, "•")
            gf = _font(30, bold=True)
            gb = draw.textbbox((0, 0), glyph, font=gf)
            draw.text((rail_x + (64 - (gb[2] - gb[0])) / 2 - gb[0],
                       cy + (64 - (gb[3] - gb[1])) / 2 - gb[1]),
                      glyph, font=gf, fill=(255, 255, 255))
            count = str(post.get("likes") or "1.2K") if i == 0 \
                else ""
            if count:
                cf = _font(20)
                cw = draw.textlength(count, font=cf)
                draw.text((rail_x + (64 - cw) / 2, cy + 66), count,
                          font=cf, fill=spec["sub"])
        regions["actions"] = regions["rail"]
    else:
        draw.line([content_x, act_y - 6, content_x + content_w,
                   act_y - 6], fill=spec["divider"])
        step = content_w // max(1, len(spec["actions"]))
        boxes = []
        for i, action in enumerate(spec["actions"]):
            ax = content_x + i * step
            fill = spec["accent"] if i == int(spec["like_at"]) \
                else spec["sub"]
            label = action
            if key == "tinder":
                draw.ellipse([ax, act_y, ax + 44, act_y + 44],
                             fill=fill)
                glyph = ("✕" if action == "Pass" else
                         "♥" if action == "Like" else "★")
                gf = _font(24, bold=True)
                gb = draw.textbbox((0, 0), glyph, font=gf)
                draw.text((ax + (44 - (gb[2] - gb[0])) / 2 - gb[0],
                           act_y + (44 - (gb[3] - gb[1])) / 2 - gb[1]),
                          glyph, font=gf, fill=(255, 255, 255))
                boxes.append((ax, act_y, 44, 44))
                continue
            if i == int(spec["like_at"]):
                label = f"♥ {action}"
            draw.text((ax, act_y), label, font=act_font, fill=fill)
            boxes.append((ax, act_y, step, 32))
        regions["actions"] = boxes
    regions["post"] = (content_x, post_y, content_w, post_h - 20)
    y = post_y + post_h - 20

    if comment_rows:
        if "comment_surface" in spec:
            draw.rectangle([0, y, width, int(height)],
                           fill=spec["comment_surface"])
        draw.text((content_x, y + 12),
                  f"Comments · {len(comment_rows)}",
                  font=_font(24, bold=True), fill=spec["sub"])
        cy = y + 54
        for entry, rows in comment_rows:
            depth = int(entry.get("depth") or 0)
            indent = depth * int(spec["comment_indent"])
            cx = content_x + indent
            if spec.get("thread_line") and depth:
                for lvl in range(1, depth + 1):
                    lx = content_x + (lvl - 1) * int(
                        spec["comment_indent"]) + 18
                    draw.line([lx, cy, lx, cy + 40 + len(rows) * 38],
                              fill=spec["divider"], width=3)
            _avatar_at(img, draw, (cx, cy), 56, entry, spec)
            draw = ImageDraw.Draw(img)
            regions["avatars"].append((cx, cy, 56, 56))
            nx = cx + 56 + 18
            cname = _name_line(spec, entry)
            if spec.get("comment_bubbles"):
                bw = content_w - indent - 30
                bh = 40 + len(rows) * 38 + 12
                draw.rounded_rectangle(
                    [cx + 56 + 8, cy - 4, cx + 56 + 8 + bw,
                     cy - 4 + bh], radius=20,
                    fill=spec["divider"] if spec["dark"] else
                    (0xF2, 0xF2, 0xF7))
                nx = cx + 56 + 28
            draw.text((nx, cy), cname, font=c_name_font,
                      fill=spec["text"])
            if spec.get("badge") is not None:
                nw = draw.textlength(cname, font=c_name_font)
                _draw_badge(draw, (int(nx + nw + 8), cy + 4), 18,
                            spec["badge"])
            tty = cy + 40
            for line in rows:
                draw.text((nx, tty), line, font=c_text_font,
                          fill=spec["text"])
                tty += 38
            if spec["comment_actions"]:
                draw.text((nx, tty + 2),
                          "Reply · Like" + (
                              f" · {entry['likes']}"
                              if entry.get("likes") else ""),
                          font=meta_font, fill=spec["sub"])
                tty += 30
            row_h = (tty - cy) + 18
            regions["comments"].append((cx, cy, content_w - indent,
                                        row_h))
            cy += row_h
    return _png_bytes(img), drawn


def _render_feature(key: str, entries: list[dict[str, Any]], *,
                    backdrop=None) -> tuple[bytes, dict[str, Any]]:
    """Draw one feature design (bigquote / motivation): a
    full-bleed square card — backdrop photograph or generated
    art when one rides, else the design's gradient — with the
    words set large in the typography service's display /
    editorial faces."""
    from PIL import Image, ImageDraw

    spec = FEATURE_DESIGNS[key]
    entry = entries[0]
    size = int(spec["size"])
    variant = str(entry.get("meta") or "").strip().lower()
    if variant == "evening" and "gradient_evening" in spec:
        gradient = spec["gradient_evening"]
    else:
        gradient = spec["gradient"]
    bg_img = _load_avatar_image(backdrop)
    if bg_img is not None:
        img = bg_img.convert("RGB")
        scale = max(size / img.width, size / img.height)
        img = img.resize((int(img.width * scale) + 1,
                          int(img.height * scale) + 1))
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        shade = Image.new("RGB", (size, size), (0, 0, 0))
        img = Image.blend(img, shade, 0.45)
        backdrop_kind = "image"
    else:
        img = Image.new("RGB", (size, size), gradient[0])
        top_c, bot_c = gradient
        for i in range(size):
            frac = i / max(1, size - 1)
            row = tuple(int(top_c[c] + (bot_c[c] - top_c[c]) * frac)
                        for c in range(3))
            ImageDraw.Draw(img).line([0, i, size, i], fill=row)
        backdrop_kind = "gradient"
    draw = ImageDraw.Draw(img)
    white = (0xFF, 0xFF, 0xFF)
    drawn: dict[str, Any] = {
        "style": key, "design": key, "size": (size, size),
        "names": [str(entry.get("name") or "")],
        "texts": [str(entry.get("text") or "")],
        "backdrop": backdrop_kind,
        "regions": {},
    }
    title = str(entry.get("title") or "").strip()
    if title:
        tfont = _font(34, bold=True, intent="caption")
        tw = draw.textlength(title.upper(), font=tfont)
        draw.text(((size - tw) / 2, 84), title.upper(), font=tfont,
                  fill=spec["accent"])
        drawn["regions"]["title"] = (int((size - tw) / 2), 84,
                                     int(tw), 40)
    if key == "bigquote":
        qfont = _font(150, bold=True, intent=spec["font"])
        draw.text((90, 120), "“", font=qfont, fill=spec["accent"])
    body_font = _font(58, intent=spec["font"])
    probe = ImageDraw.Draw(Image.new("RGB", (size, 100)))
    lines = _wrap(probe, str(entry.get("text") or ""), body_font,
                  size - 220)
    total_h = len(lines) * 76
    ty = (size - total_h) // 2 + (30 if title else 0)
    for line in lines:
        tw = draw.textlength(line, font=body_font)
        draw.text(((size - tw) / 2, ty), line, font=body_font,
                  fill=white)
        ty += 76
    drawn["regions"]["body"] = (110, (size - total_h) // 2,
                                size - 220, total_h)
    author = str(entry.get("name") or "").strip()
    if author:
        draw.line([size // 2 - 60, size - 190, size // 2 + 60,
                   size - 190], fill=spec["accent"], width=4)
        afont = _font(36, bold=True, intent="caption")
        label = author.upper()
        tw = draw.textlength(label, font=afont)
        draw.text(((size - tw) / 2, size - 160), label, font=afont,
                  fill=white)
        drawn["regions"]["author"] = (int((size - tw) / 2),
                                      size - 160, int(tw), 44)
    return _png_bytes(img), drawn


def card_backdrop(*, seed: str = "",
                  fetcher: Callable[[str], bytes] | None = None):
    """A backdrop image for the feature designs, best-effort:
    first a real photograph from the shipped picsum pool
    (keyless; the same provider the couplepp chain walks),
    then Devon's own studio art (``emoji_studio``) generated
    from the seed words.  Returns a PIL image or None — the
    renderer draws its gradient when both rungs miss, so a
    card never waits on the network to exist."""
    fetch = fetcher or (lambda url: _http_bytes(url, timeout=12.0))
    try:
        url = PicsumPoolProvider().image_url()
        data = fetch(url)
        if _sniff_image(data) is not None:
            img = _load_avatar_image(data)
            if img is not None:
                return img
    except Exception:  # noqa: BLE001 - next rung
        pass
    try:
        from .emoji_studio import parse_design, render_master

        spec = parse_design(
            f"serene landscape inspired by: {seed[:80]}")
        return render_master(spec, size=1080)
    except Exception:  # noqa: BLE001 - gradient is the floor
        return None


def _parse_comment_line(spec: str, depth: int,
                        sender_name: str) -> dict[str, Any]:
    """One comment line's body → a comment entry.  Shapes:
    ``Name | words``, ``Name: words``, or bare words (the
    commenter is the sender)."""
    spec = spec.strip()
    if "|" in spec:
        name, _, text = spec.partition("|")
        name, text = name.strip(), text.strip()
    elif ": " in spec and len(spec.split(": ", 1)[0]) <= 48:
        name, text = (p.strip() for p in spec.split(": ", 1))
    else:
        name, text = "", spec
    if not text:
        raise MakerError(
            "a comment line needs words — `> Name | their words`")
    return {"name": name or sender_name or "Someone",
            "text": text, "depth": depth}


def parse_card_request(kind: str, tail: str, *,
                       reply_name: str = "",
                       reply_text: str = "",
                       sender_name: str = ""
                       ) -> tuple[str, list[dict]]:
    """(style, entries) for a card command — or a MakerError
    carrying the usage line.  Grammar:

    - reply form: the replied message supplies name + text
      (single styles only);
    - explicit single: ``<name> | <text>``; bare words with
      no ``|`` quote the SENDER (name = the sender's);
    - /qcard and /fakemsg take a style OR a platform design
      as first word when it names one (``/qcard ttc Ada | hi``,
      ``/qcard facebook Ada | hi``);
    - duo: ``<name> | <text> | <name> | <text>``;
    - comments (platform designs): each following line that
      starts with ``>`` adds a comment — ``> Name | words``;
      ``>>`` replies to the comment above it (nesting depth
      1, ``>>>`` depth 2) where the platform nests.
    """
    kind = str(kind or "").strip().lower()
    style = CARD_KINDS.get(kind, "quote")
    tail = str(tail or "").strip()
    if kind in ("qcard", "fakemsg") and tail:
        first, _, rest = tail.partition(" ")
        if first.strip().lower() in CARD_DESIGNS:
            style = first.strip().lower()
            tail = rest.strip()
    reply_name = str(reply_name or "").strip()
    reply_text = str(reply_text or "").strip()
    usage = (f"usage: reply to a message with /{kind}, or "
             f"/{kind} <name> | <text>")
    if style == "duo":
        parts = [p.strip() for p in tail.split("|")]
        if len(parts) == 4 and all(parts):
            return style, [{"name": parts[0], "text": parts[1]},
                           {"name": parts[2], "text": parts[3]}]
        raise MakerError(
            f"usage: /{kind} <name> | <text> | <name> | <text> "
            f"— a duo card carries two speakers")
    lines = tail.split("\n") if tail else []
    head = lines[0] if lines else ""
    comment_specs: list[tuple[int, str]] = []
    extra: list[str] = []
    for raw in lines[1:]:
        stripped = raw.strip()
        if stripped.startswith(">"):
            marks = len(stripped) - len(stripped.lstrip(">"))
            comment_specs.append(
                (min(marks - 1, 4), stripped[marks:].strip()))
        elif stripped:
            extra.append(stripped)
    if extra:
        head = (head + "\n" + "\n".join(extra)).strip()
    if comment_specs and style not in PLATFORM_DESIGNS:
        raise MakerError(
            "comment sections ride the platform designs — "
            f"pick one ({', '.join(PLATFORM_DESIGNS)}): "
            f"/{kind} <design> <name> | <post> then one "
            f"`> Name | comment` per line")
    if reply_text:
        entries: list[dict] = [{"name": reply_name or sender_name
                                or "Someone", "text": reply_text}]
    elif "|" in head:
        name, _, text = head.partition("|")
        if not text.strip():
            raise MakerError(usage)
        entries = [{"name": name.strip() or sender_name
                    or "Someone", "text": text.strip()}]
    elif head:
        entries = [{"name": sender_name or "Someone",
                    "text": head}]
    else:
        raise MakerError(usage)
    for depth, spec in comment_specs:
        entries.append(_parse_comment_line(spec, depth,
                                           sender_name))
    return style, entries


# ── self-provisioning (the ensure_ytdlp idiom) ───────────────

#: One provisioning attempt per package per process: a missing
#: tool is Devon's job to fix, but a broken network must not
#: turn every command into a pip run.
_PROVISION_ATTEMPTED: set[str] = set()


def _spec_present(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - a broken lookup is "absent"
        return False


def _pip_install(pip_name: str, *, timeout: float) -> bool:
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 - provisioning is best-effort;
        # the honest unavailable line stands when it can't run
        return False


def _ensure_package(import_name: str, pip_name: str, *,
                    timeout: float = 180.0,
                    available_fn: Callable[[], bool] | None = None,
                    runner: Callable[[], bool] | None = None) -> bool:
    """Make ``import_name`` importable, installing it herself
    when it's missing — one bounded attempt per process.
    Never raises.  Under a test runner the environment is
    simulated, so a real install never runs there (the
    injectable seams are how tests stage provisioning)."""
    import sys

    available = available_fn or (lambda: _spec_present(import_name))
    if available():
        return True
    if import_name in _PROVISION_ATTEMPTED:
        return False
    _PROVISION_ATTEMPTED.add(import_name)
    if runner is None and "pytest" in sys.modules:
        return False
    run = runner or (lambda: _pip_install(pip_name,
                                          timeout=timeout))
    try:
        if not run():
            return False
    except Exception:  # noqa: BLE001 - best-effort, see above
        return False
    return bool(available())


# ── /removebg: the segmentation organ + provisioning ─────────

def rembg_available() -> bool:
    return _spec_present("rembg")


def ensure_rembg(**seams: Any) -> bool:
    """One bounded attempt at installing rembg herself
    (onnxruntime rides along — the timeout is sized for it)."""
    return _ensure_package("rembg", "rembg", timeout=600.0,
                           **seams)


def _rembg_weights_cached(model: str) -> bool:
    import os

    home = os.environ.get("U2NET_HOME") or str(Path.home()
                                               / ".u2net")
    return (Path(home) / f"{model}.onnx").exists()


def pick_rembg_model(*, cached: Callable[[str], bool] | None = None,
                     default: str = "") -> str:
    """The model a /removebg run uses: the segment organ's
    profile default — EXCEPT a first run whose default's
    weights aren't cached yet and weigh >250 MB (the
    birefnet pair), which rides the 4.7 MB u2netp instead.
    Bounded first use; the quality default returns once its
    weights are on disk by any route."""
    from ...media_edit.segment import MODELS, default_model

    model = (default or default_model()).strip().lower()
    info = MODELS.get(model) or {}
    cached = cached or _rembg_weights_cached
    if float(info.get("size_mb") or 0) > 250 and not cached(model):
        return "u2netp"
    return model


def remove_background_bytes(data: bytes, *,
                            remover: Callable[..., Any] | None = None,
                            model: str = "") -> tuple[bytes, str]:
    """Image bytes → (transparent PNG bytes, model used).
    ``remover`` is the model seam (tests stub it); the real
    path is the segment organ's own remove_background — its
    SegmentationError becomes this command's honest line."""
    from ...media_edit.images import load_image
    from ...media_edit.segment import SegmentationError, remove_background

    try:
        img = load_image(io.BytesIO(bytes(data)))
    except Exception as exc:  # noqa: BLE001 - not an image we can open
        raise MakerError(
            "that file isn't an image I can open — send a "
            "photo and I'll cut its background out") from exc
    use_model = model or pick_rembg_model()
    fn = remover or remove_background
    try:
        out = fn(img, model=use_model)
    except SegmentationError as exc:
        raise MakerError(str(exc)) from exc
    out = out.convert("RGBA")
    return _png_bytes(out), use_model


# ── /qr: generate (qrcode lib) + read (vision organ) ─────────

def qrcode_available() -> bool:
    return _spec_present("qrcode")


def ensure_qrcode(**seams: Any) -> bool:
    return _ensure_package("qrcode", "qrcode[pil]",
                           timeout=180.0, **seams)


def make_qr_png(text: str, *, qr_lib: Any = None) -> bytes:
    """Encode ``text`` as a QR PNG.  ``qr_lib`` is the library
    seam; the real module is the ``qrcode`` package — the same
    generator the payments and Discord-login paths use."""
    text = str(text or "").strip()
    if not text:
        raise MakerError("usage: /qr <text or a link>")
    lib = qr_lib
    if lib is None:
        try:
            import qrcode as lib
        except ImportError as exc:
            raise MakerError(
                "QR making needs the qrcode library on this "
                "box — and my self-install didn't land it"
            ) from exc
    img = lib.make(text)
    if hasattr(img, "get_image"):  # qrcode's QrImage wrapper
        img = img.get_image()
    return _png_bytes(img.convert("RGB"))


def read_qr_codes(data: bytes, *,
                  reader: Callable[[bytes], dict] | None = None
                  ) -> list[dict[str, str]]:
    """Decode every code in an image through the vision
    organ's real decoder.  Its NativeUnavailable (pyzbar /
    zbar absent) is surfaced verbatim — never a guessed
    decode, never a fake 'no code found'."""
    from ...vision.native import NativeUnavailable

    fn = reader
    if fn is None:
        from ...vision.native import read_qr as fn
    try:
        out = fn(bytes(data))
    except NativeUnavailable as exc:
        raise MakerError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - undecodable bytes
        raise MakerError(
            "I couldn't read that file as an image — send "
            "the QR as a photo and I'll decode it") from exc
    codes = []
    for item in out.get("codes") or []:
        codes.append({"type": str(item.get("type") or "QRCODE"),
                      "data": str(item.get("data") or "")})
    return codes


# ── /shorturl + /unshorten ───────────────────────────────────

def validate_long_url(raw: str) -> str:
    """A URL this organ will shorten or follow: http(s) with a
    real host.  Anything else is refused in plain words —
    never handed to a shortener to launder."""
    url = str(raw or "").strip()
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise MakerError(
            f"“{url or '(nothing)'}” isn't a web link I can "
            f"work with — give me a full http(s) URL")
    return url


class _PlainTextShortener:
    """The keyless shorteners share one shape: GET the create
    endpoint, the short URL comes back as the whole body."""

    name = "shortener"
    endpoint = ""

    def __init__(self, *, fetcher: Callable[[str], str] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_text

    def shorten(self, url: str) -> dict[str, Any]:
        out = self._fetch(self.endpoint
                          + urllib.parse.quote(url, safe=""))
        short = str(out or "").strip()
        if short.lower().startswith("error") or \
                not short.startswith(("http://", "https://")):
            raise MakerError(f"{self.name} refused: "
                             f"{short[:120] or 'empty reply'}")
        return {"short": short}


class IsGdProvider(_PlainTextShortener):
    name = "is.gd"
    endpoint = "https://is.gd/create.php?format=simple&url="


class VGdProvider(_PlainTextShortener):
    name = "v.gd"
    endpoint = "https://v.gd/create.php?format=simple&url="


class TinyUrlProvider(_PlainTextShortener):
    name = "tinyurl"
    endpoint = "https://tinyurl.com/api-create.php?url="


class _StopRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers,
                         newurl):  # noqa: D102 - stdlib override
        return None


def _hop_once(url: str) -> tuple[int, str]:
    """One redirect hop: (status, Location) — the body is
    never read, so /unshorten inspects a link without
    downloading what it points at."""
    opener = urllib.request.build_opener(_StopRedirect)
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(
            url, method=method,
            headers={"User-Agent": "nomorals-makers/1.0"})
        try:
            with opener.open(req, timeout=10) as resp:
                return int(resp.status), str(
                    resp.headers.get("Location") or "")
        except urllib.error.HTTPError as exc:
            if exc.code in (405, 501) and method == "HEAD":
                continue
            return int(exc.code), str(
                exc.headers.get("Location") or "")
    return 0, ""


def resolve_redirect_chain(
        url: str, *,
        hop: Callable[[str], tuple[int, str]] | None = None,
        max_hops: int = 10) -> dict[str, Any]:
    """Walk a link's redirects to its landing page."""
    hop = hop or _hop_once
    seen = {url}
    current, hops = url, 0
    while hops < max_hops:
        status, location = hop(current)
        if status not in (301, 302, 303, 307, 308) or not location:
            return {"final": current, "hops": hops,
                    "status": status}
        nxt = urllib.parse.urljoin(current, location)
        if nxt in seen:
            raise MakerError("that link loops back on itself — "
                             "it never lands anywhere")
        seen.add(nxt)
        current, hops = nxt, hops + 1
    raise MakerError(f"that link redirected {max_hops} times "
                     f"without landing — I stopped following it")


# ── /bible ───────────────────────────────────────────────────

#: The canon in order — the number is bolls.life's book id
#: (verified against live payloads: 43 John, 45 Romans,
#: 62 1 John…).  68/70/71 are the apocrypha ids the same
#: source's search returns (Tobit, Wisdom, Sirach).
BIBLE_BOOKS: tuple[tuple[int, str], ...] = tuple(
    enumerate((
        "Genesis", "Exodus", "Leviticus", "Numbers",
        "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel",
        "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles",
        "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
        "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
        "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
        "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
        "Zechariah", "Malachi", "Matthew", "Mark", "Luke",
        "John", "Acts", "Romans", "1 Corinthians",
        "2 Corinthians", "Galatians", "Ephesians", "Philippians",
        "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter", "1 John",
        "2 John", "3 John", "Jude", "Revelation",
    ), start=1)) + ((68, "Tobit"), (70, "Wisdom"),
                    (71, "Sirach"))

_BOOK_BY_NAME = {name.lower(): (num, name)
                 for num, name in BIBLE_BOOKS}
_BOOK_BY_NAME.update({
    "psalm": (19, "Psalms"),
    "song of songs": (22, "Song of Solomon"),
    "canticles": (22, "Song of Solomon"),
    "revelations": (66, "Revelation"),
    "acts of the apostles": (44, "Acts"),
})
_BOOK_BY_NUMBER = dict(BIBLE_BOOKS)

#: Translations both seeds carry (bible-api ids lowercase;
#: bolls takes the same codes upper-cased).
BIBLE_VERSIONS = ("kjv", "asv", "web", "darby", "ylt", "bbe")

_REF_RE = re.compile(
    r"^(?P<book>.+?)\s+(?P<chapter>\d+)"
    r"(?:\s*:\s*(?P<v1>\d+)(?:\s*-\s*(?P<v2>\d+))?)?$")


class BibleRef:
    """One parsed reference: book + chapter + optional verse
    span (0 = from the chapter's start / to its end)."""

    def __init__(self, book: str, number: int, chapter: int,
                 verse_from: int = 0, verse_to: int = 0,
                 version: str = "kjv") -> None:
        self.book = book
        self.number = number
        self.chapter = chapter
        self.verse_from = verse_from
        self.verse_to = verse_to
        self.version = version

    def label(self) -> str:
        base = f"{self.book} {self.chapter}"
        if self.verse_from:
            if self.verse_to and self.verse_to != self.verse_from:
                return f"{base}:{self.verse_from}-{self.verse_to}"
            return f"{base}:{self.verse_from}"
        return base

    def in_span(self, verse: int) -> bool:
        if self.verse_from and verse < self.verse_from:
            return False
        return not (self.verse_to and verse > self.verse_to)


def parse_bible_tail(tail: str) -> tuple[str, Any]:
    """Parse a /bible tail → ("passage", BibleRef) or
    ("search", (query, version)).  Raises MakerError with a
    usage-shaped message for anything unreadable — an unknown
    book names the miss instead of guessing a neighbour."""
    raw = " ".join(str(tail or "").split()).strip()
    if not raw:
        raise MakerError("usage: /bible <book> <chapter>[:<verse>]"
                         " — e.g. /bible John 3:16, or "
                         "/bible search <words>")
    version = "kjv"
    words = raw.split()
    if len(words) > 1 and words[-1].lower() in BIBLE_VERSIONS:
        version = words[-1].lower()
        raw = " ".join(words[:-1])
    if raw.lower().startswith("search "):
        query = raw[len("search "):].strip()
        if not query:
            raise MakerError("usage: /bible search <words>")
        return "search", (query, version)
    match = _REF_RE.match(raw)
    if not match:
        raise MakerError(
            f"I can't read “{raw}” as a reference — the shape "
            f"is /bible <book> <chapter>[:<verse>[-<verse>]], "
            f"e.g. /bible 1 Samuel 1:1")
    book_key = " ".join(match.group("book").lower().split())
    found = _BOOK_BY_NAME.get(book_key)
    if found is None:
        raise MakerError(
            f"“{match.group('book')}” isn't a book I know — "
            f"check the spelling (e.g. Psalms, 1 Samuel, "
            f"Song of Solomon)")
    number, book = found
    v1 = int(match.group("v1") or 0)
    v2 = int(match.group("v2") or 0)
    if v2 and v2 < v1:
        v1, v2 = v2, v1
    return "passage", BibleRef(book, number,
                               int(match.group("chapter")),
                               v1, v2, version)


def clean_bolls_text(text: str) -> str:
    """bolls.life text carries Strong's-number markup
    (``word~~123~~``) and search highlighting (``==word==``) —
    markup, not words: strip both, quote the rest exactly."""
    text = re.sub(r"~~\d+~~", "", str(text or ""))
    text = text.replace("==", "")
    return " ".join(text.split())


class BibleApiProvider:
    """bible-api.com — the primary seed.  Answers
    ``{reference, verses: [{verse, text, …}],
    translation_name}``; an unknown reference answers an
    ``error`` body / 404."""

    name = "bible-api.com"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def passage(self, ref: BibleRef) -> dict[str, Any]:
        query = urllib.parse.quote(ref.label())
        try:
            data = self._fetch(
                f"https://bible-api.com/{query}"
                f"?translation={ref.version}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise MakerError(
                    f"no such passage — {ref.label()} isn't in "
                    f"the {ref.version.upper()} text I can reach"
                ) from exc
            raise
        if not isinstance(data, dict) or data.get("error"):
            raise MakerError(
                f"no such passage — {ref.label()} "
                f"({data.get('error') if isinstance(data, dict) else 'shapeless reply'})")
        verses = [{"verse": int(v.get("verse") or 0),
                   "text": " ".join(str(v.get("text") or "")
                                    .split())}
                  for v in data.get("verses") or []]
        verses = [v for v in verses if v["text"]]
        if not verses:
            raise MakerError(f"no such passage — {ref.label()}")
        return {"reference": str(data.get("reference")
                                 or ref.label()),
                "translation": str(data.get("translation_name")
                                   or ref.version.upper()),
                "source": self.name, "verses": verses}


class BollsProvider:
    """bolls.life — the second seed: whole-chapter text
    (``[{pk, verse, text}]``) and the full-text search
    bible-api.com doesn't offer
    (``[{pk, book, chapter, verse, text}]``)."""

    name = "bolls.life"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def passage(self, ref: BibleRef) -> dict[str, Any]:
        data = self._fetch(
            f"https://bolls.life/get-text/"
            f"{ref.version.upper()}/{ref.number}/{ref.chapter}/")
        if not isinstance(data, list):
            raise MakerError("bolls.life answered a non-list")
        verses = [{"verse": int(v.get("verse") or 0),
                   "text": clean_bolls_text(v.get("text"))}
                  for v in data if isinstance(v, dict)]
        verses = [v for v in verses
                  if v["text"] and ref.in_span(v["verse"])]
        if not verses:
            raise MakerError(f"no such passage — {ref.label()}")
        return {"reference": ref.label(),
                "translation": ref.version.upper(),
                "source": self.name, "verses": verses}

    def search(self, query: str, version: str) -> dict[str, Any]:
        data = self._fetch(
            f"https://bolls.life/search/{version.upper()}/"
            f"?search={urllib.parse.quote(query)}")
        if not isinstance(data, list):
            raise MakerError("bolls.life answered a non-list")
        hits = []
        for item in data:
            if not isinstance(item, dict):
                continue
            book = _BOOK_BY_NUMBER.get(int(item.get("book")
                                           or 0))
            if book is None:
                continue  # an id outside the shipped canon
                # table can't be named honestly — skip it
            hits.append({
                "reference": f"{book} {int(item.get('chapter') or 0)}"
                             f":{int(item.get('verse') or 0)}",
                "text": clean_bolls_text(item.get("text"))})
        return {"hits": hits, "source": self.name,
                "translation": version.upper()}


# ── /couplepp: one artwork, a matched pair ───────────────────
#
# The chain's providers are DELEGATES, not implementations
# (2026-11-10 consolidation): every endpoint's URL, response
# shape and safety check lives in the anime organ's unified
# pool table (nomorals.anime.sources) — the single pool organ
# the whole picture family draws from.  These classes keep
# the chain's contract (a name the health breaker and the
# captions speak, an ``image_url()``) and forward to that
# organ's client over a one-entry pool.  The duplicate
# endpoint implementations that used to live here are gone:
# a provider fix now lands once, for every picture command.

class _FetcherTransport:
    """Adapt a makers JSON fetcher to the anime Transport
    seam, so a delegate handed a fetcher (the test shape)
    still speaks to the one pool client."""

    def __init__(self, fetcher: Callable[[str], Any]) -> None:
        self._fetch = fetcher

    def request(self, method: str, url: str, **kw: Any) -> Any:
        from ...anime.transport import HttpResult, TransportError

        try:
            data = self._fetch(url)
        except Exception as exc:  # noqa: BLE001 - one seam, named
            raise TransportError(str(exc)) from exc
        return HttpResult(200, data=data)


class _PoolDelegate:
    """One named rung of the couplepp chain over the unified
    pool client: the entry is read out of the shared table
    (the ``waifu`` pool's entry of the same name — the /waifu
    category, which is the category this chain has always
    drawn)."""

    name = "?"
    _entry_name = "?"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None,
                 client: Any = None,
                 transport: Any = None,
                 rng: random.Random | None = None) -> None:
        self._fetcher = fetcher
        self._client = client
        self._transport = transport
        self._rng = rng or random.Random()

    def _client_or_build(self) -> Any:
        if self._client is not None:
            return self._client
        from ...anime.sources import IMAGE_POOLS, ImagePoolClient
        from ...anime.transport import core_http_transport

        entries = [e for e in IMAGE_POOLS["waifu"]
                   if e.name == self._entry_name]
        if self._transport is not None:
            transport = self._transport
        elif self._fetcher is not None:
            transport = _FetcherTransport(self._fetcher)
        else:
            transport = core_http_transport()
        self._client = ImagePoolClient(
            transport, pools={"_one": entries})
        return self._client

    def image_url(self) -> str:
        from ...anime.sources import SourceFailure

        try:
            url, _source = self._client_or_build().fetch_url(
                "_one", rng=self._rng)
        except SourceFailure as exc:
            raise MakerError(str(exc)) from exc
        return url


class WaifuPoolProvider(_PoolDelegate):
    """waifu.pics SFW pool — delegated to the unified pool
    organ (its ``{"url": …}`` shape is checked there)."""

    name = "waifu.pics"
    _entry_name = "waifu.pics"


class NekosPoolProvider(_PoolDelegate):
    """nekos.best pool — delegated to the unified pool organ
    (its ``{"results": [{"url": …}]}`` shape is checked
    there)."""

    name = "nekos.best"
    _entry_name = "nekos.best"


class NekosApiPoolProvider(_PoolDelegate):
    """nekosapi.com v4 random pool — delegated to the unified
    pool organ, whose extractor keeps the safety law: the
    unfiltered pool is NOT safe (it serves suggestive/
    explicit rows), so only a row the pool itself rates
    ``safe`` can ever be returned — the filter is the
    source's claim, the check is the pool's."""

    name = "nekosapi"
    _entry_name = "nekosapi"


class PicsumPoolProvider:
    """picsum.photos — general photography (the Lorem Picsum
    pool of real photos), keyless: a seeded URL answers a
    redirect to one real photograph.  This is the chain's
    non-anime rung: a couple pair drawn from an actual photo,
    named as such in the caption.  The seeded-URL template
    lives in the unified pool table; this adapter only fills
    it with its rng."""

    name = "picsum"

    def __init__(self, *,
                 rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def image_url(self) -> str:
        from ...anime.sources import direct_source_url

        return direct_source_url("picsum", rng=self._rng)


def _diptych_crops(data: bytes):
    """The two matched halves of ONE artwork: complementary
    3:4 crops (left third / right third of the frame), each
    480×640 — one profile picture each.  Returns
    ``(left_image, right_image, info)``."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(bytes(data)))
        img.load()
    except Exception as exc:  # noqa: BLE001 - not an image
        raise MakerError("the pool sent bytes that aren't an "
                         "image") from exc
    img = img.convert("RGB")
    w, h = img.size
    half_w, half_h = 480, 640
    win_w = min(w // 2, int(h * 0.75))
    win_h = int(win_w / 0.75)
    if win_h > h:
        win_h, win_w = h, int(h * 0.75)
    win_w, win_h = max(8, win_w), max(8, win_h)

    def crop(cx: float):
        x = min(max(int(w * cx - win_w / 2), 0), w - win_w)
        y = min(max(int(h / 2 - win_h / 2), 0), h - win_h)
        return img.crop((x, y, x + win_w, y + win_h)).resize(
            (half_w, half_h), Image.LANCZOS)

    info = {"half": (half_w, half_h)}
    return crop(0.30), crop(0.70), info


def make_diptych(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """The two halves composed into ONE canvas side by side
    — a preview shape.  Kept for callers that want the pair
    on one canvas; /couplepp itself delivers the halves as
    two separate photos (see :func:`split_diptych`)."""
    from PIL import Image

    left, right, info = _diptych_crops(data)
    half_w, half_h = info["half"]
    canvas = Image.new("RGB", (half_w * 2 + 8, half_h),
                       (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (half_w + 8, 0))
    return _png_bytes(canvas), {"size": canvas.size, **info}


def split_diptych(data: bytes) -> tuple[bytes, bytes,
                                        dict[str, Any]]:
    """Split ONE artwork into its matched pair AS TWO
    PICTURES: ``(left_png, right_png, info)`` — each half a
    standalone 480×640 photo, ready to be set as a profile
    picture the moment it lands.  This is the /couplepp
    deliverable: two photos, not one poster of a pair."""
    left, right, info = _diptych_crops(data)
    return _png_bytes(left), _png_bytes(right), info


# ── the service ──────────────────────────────────────────────

def default_maker_chains() -> dict[str, list[Any]]:
    return {
        "shorturl": [IsGdProvider(), VGdProvider(),
                     TinyUrlProvider()],
        "bible": [BibleApiProvider(), BollsProvider()],
        "bible_search": [BollsProvider()],
        "couplepp": [WaifuPoolProvider(), NekosPoolProvider(),
                     NekosApiPoolProvider(), PicsumPoolProvider()],
    }


class MakerService:
    """Walks the maker chains and renders the makers' media.
    Every outcome is a plain dict; the service never raises —
    a MakerError's message IS the user-fit reason."""

    def __init__(self, *, media_dir: Path | str = "",
                 chains: dict[str, list[Any]] | None = None,
                 health: ProviderHealth | None = None,
                 bytes_fetcher: Callable[[str], bytes] | None = None,
                 qr_lib: Any = None,
                 qr_reader: Callable[[bytes], dict] | None = None,
                 remover: Callable[..., Any] | None = None,
                 ensure_rembg_fn: Callable[[], bool] | None = None,
                 ensure_qrcode_fn: Callable[[], bool] | None = None,
                 hop_fetcher: Callable[[str], tuple[int, str]] | None = None
                 ) -> None:
        self.media_dir = Path(media_dir) if media_dir else None
        self.health = health or ProviderHealth()
        self.bytes_fetcher = bytes_fetcher or _http_bytes
        self.qr_lib = qr_lib
        self.qr_reader = qr_reader
        self.remover = remover
        self.ensure_rembg_fn = ensure_rembg_fn or ensure_rembg
        self.ensure_qrcode_fn = ensure_qrcode_fn or ensure_qrcode
        self.hop_fetcher = hop_fetcher
        merged = default_maker_chains()
        for kind, chain in (chains or {}).items():
            merged[kind] = list(chain)
        self.chains: dict[str, list[Any]] = merged

    # -- plumbing ---------------------------------------------
    def _save(self, data: bytes, ext: str) -> Path:
        if self.media_dir is None:
            raise MakerError("no media directory on this runtime")
        self.media_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(data).hexdigest()[:16] + ext
        path = self.media_dir / name
        if not path.exists():
            path.write_bytes(data)
        return path

    def _deliver(self, data: bytes, ext: str, mime: str,
                 caption: str, send, *, audience: str = ""
                 ) -> dict[str, Any]:
        from .base import MediaRef

        try:
            path = self._save(data, ext)
        except MakerError as exc:
            return {"sent": False, "reason": str(exc)}
        ref = MediaRef(path=str(path), mime=mime, kind="image",
                       name=path.name, audience=audience)
        try:
            result = send(ref, caption)
        except Exception as exc:  # noqa: BLE001 - send failed
            return {"sent": False,
                    "reason": f"the send failed: {exc}"}
        if result is not None and getattr(result, "ok", True):
            return {"sent": True}
        return {"sent": False,
                "reason": str(getattr(result, "error", "")
                              or "this surface couldn't send it")}

    def _walk(self, chain_key: str, method: str, *args: Any
              ) -> tuple[dict[str, Any] | None, str]:
        """(result, source) from the first healthy seed that
        answers in shape.  A MakerError is a NAMED answer (a
        refusal, a not-found) — it ends the walk at once;
        anything else sheds to the next seed."""
        last = ""
        for provider in self.chains.get(chain_key) or []:
            name = str(getattr(provider, "name", "?"))
            if not self.health.available(name):
                continue
            try:
                result = getattr(provider, method)(*args)
            except MakerError:
                raise
            except Exception as exc:  # noqa: BLE001 - shed
                self.health.note(name, False)
                last = str(exc)[:160]
                continue
            if not isinstance(result, dict) or not result:
                self.health.note(name, False)
                last = f"{name} answered a shapeless result"
                continue
            self.health.note(name, True)
            return result, name
        return None, last

    # -- cards --------------------------------------------------
    def card(self, kind: str, tail: str, *, reply_name: str = "",
             reply_text: str = "", sender_name: str = "",
             avatars: dict[int, Any] | None = None,
             backdrop: Any = None,
             audience: str = "",
             send) -> dict[str, Any]:
        """Render + deliver one card.  ``avatars`` maps an
        entry index to the avatar the surface handed over
        (bytes / PIL image / path) — the renderer pastes it,
        else the design's placeholder disc stands in.
        ``backdrop`` feeds the feature designs; when none
        rides, one is resolved best-effort (photo pool, then
        her studio, then the design's gradient).
        ``audience`` is the artifact's public/personal
        classification, carried on the MediaRef: the runtime
        passes "public" when the card is made in a group chat
        (the send seam then burns the brand mark in), "personal"
        in the owner's own DM.  The card bytes at rest are never
        marked — the mark is a property of the send."""
        try:
            style, entries = parse_card_request(
                kind, tail, reply_name=reply_name,
                reply_text=reply_text, sender_name=sender_name)
            for idx, avatar in (avatars or {}).items():
                if 0 <= int(idx) < len(entries):
                    entries[int(idx)]["avatar"] = avatar
            if style in FEATURE_DESIGNS and backdrop is None:
                backdrop = card_backdrop(
                    seed=entries[0]["text"] if entries else "")
            data, drawn = render_card_png(style, entries,
                                          backdrop=backdrop)
        except MakerError as exc:
            return {"sent": False, "reason": str(exc)}
        names = drawn.get("names") or []
        if style == "duo":
            caption = (f"💬💬 duo card — "
                       f"{' × '.join(n for n in names if n)}")
        elif style == "meme":
            caption = "🖼 meme card"
        elif style in PLATFORM_DESIGNS:
            caption = (f"🖼 {PLATFORM_DESIGNS[style]['label']} "
                       f"card — {names[0] if names else ''}")
        elif style in FEATURE_DESIGNS:
            caption = (f"💬 quote card — "
                       f"{names[0] if names else ''}")
        else:
            caption = f"💬 quote card — {names[0] if names else ''}"
        out = self._deliver(data, ".png", "image/png",
                            _trim_caption(caption), send,
                            audience=audience)
        out.update({"style": style, "drawn": drawn})
        return out

    # -- removebg -----------------------------------------------
    def removebg(self, data: bytes, *, send,
                 audience: str = "") -> dict[str, Any]:
        """Cut the background out + deliver.  ``audience`` is
        the artifact's public/personal classification, carried
        on the MediaRef exactly as :meth:`card` carries it: the
        runtime passes "public" when the cutout is made in a
        group chat (the send seam then burns the brand mark
        in), "personal" in the owner's own DM.  The cutout
        bytes at rest are never marked — the mark is a property
        of the send."""
        if (self.remover is None and not rembg_available()
                and not self.ensure_rembg_fn()):
            return {"sent": False, "reason": (
                "background removal runs on the rembg model "
                "rung, and it isn't on this box — I tried "
                "installing it myself and the install "
                "didn't land, so there's no honest way to "
                "cut this one yet.")}
        try:
            png, model = remove_background_bytes(
                data, remover=self.remover)
        except MakerError as exc:
            return {"sent": False, "reason": str(exc)}
        out = self._deliver(png, ".png", "image/png",
                            "✂️ background removed", send,
                            audience=audience)
        out["model"] = model
        return out

    # -- qr -------------------------------------------------------
    def qr_make(self, text: str, *, send,
                audience: str = "") -> dict[str, Any]:
        """Make a QR code + deliver it.  ``audience`` is the
        artifact's public/personal classification, carried on
        the MediaRef exactly as :meth:`card` carries it: the
        runtime passes "public" when the code is made in a
        group chat (the send seam then burns the brand mark
        in), "personal" in the owner's own DM.  The code's
        bytes at rest are never marked — the mark is a property
        of the send."""
        if (self.qr_lib is None and not qrcode_available()
                and not self.ensure_qrcode_fn()):
            return {"sent": False, "reason": (
                "QR making needs the qrcode library, and "
                "it isn't on this box — I tried installing "
                "it myself and the install didn't land.")}
        try:
            png = make_qr_png(text, qr_lib=self.qr_lib)
        except MakerError as exc:
            return {"sent": False, "reason": str(exc)}
        return self._deliver(png, ".png", "image/png",
                             _trim_caption(str(text).strip(),
                                           200), send,
                             audience=audience)

    def qr_read(self, data: bytes) -> dict[str, Any]:
        try:
            codes = read_qr_codes(data, reader=self.qr_reader)
        except MakerError as exc:
            return {"ok": False, "reason": str(exc)}
        return {"ok": True, "codes": codes}

    # -- shorturl / unshorten -------------------------------------
    def shorten(self, raw: str) -> dict[str, Any]:
        try:
            url = validate_long_url(raw)
            result, _src = self._walk("shorturl", "shorten", url)
        except MakerError as exc:
            return {"ok": False, "reason": str(exc)}
        if result is None:
            return {"ok": False, "reason": (
                "every shortener seed is down right now — "
                "try again in a bit")}
        return {"ok": True, "short": result["short"], "url": url,
                "source": _src}

    def unshorten(self, raw: str) -> dict[str, Any]:
        try:
            url = validate_long_url(raw)
            out = resolve_redirect_chain(url,
                                         hop=self.hop_fetcher)
        except MakerError as exc:
            return {"ok": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - network down
            return {"ok": False, "reason": (
                f"I couldn't reach that link to follow it: "
                f"{str(exc)[:120]}")}
        return {"ok": True, **out}

    # -- bible ------------------------------------------------------
    def bible(self, tail: str) -> dict[str, Any]:
        try:
            mode, parsed = parse_bible_tail(tail)
            if mode == "search":
                query, version = parsed
                result, _src = self._walk("bible_search", "search",
                                          query, version)
                if result is None:
                    return {"ok": False, "reason": (
                        "the bible search source is down right "
                        "now — try again in a bit")}
                return {"ok": True, "mode": "search",
                        "query": query, **result}
            result, _src = self._walk("bible", "passage", parsed)
        except MakerError as exc:
            return {"ok": False, "reason": str(exc)}
        if result is None:
            return {"ok": False, "reason": (
                "every bible source is down right now — try "
                "again in a bit")}
        return {"ok": True, "mode": "passage", **result}

    # -- couplepp ---------------------------------------------------
    def _deliver_pair(self, left_png: bytes, right_png: bytes,
                      first_caption: str, second_caption: str,
                      send, buttons=None,
                      audience: str = "") -> dict[str, Any]:
        """Deliver a matched pair as TWO separate photos —
        the delivery-shape law: each half is its own picture,
        settable as a profile photo the moment it lands.  The
        follow-up ``buttons`` (when the caller offers them)
        ride the second photo, where a surface carries buttons
        on media at all.  A pair that only half-lands says so —
        an incomplete pair is never reported as delivered."""
        from .base import MediaRef

        try:
            left_path = self._save(left_png, ".png")
            right_path = self._save(right_png, ".png")
        except MakerError as exc:
            return {"sent": False, "reason": str(exc)}

        def _one(path: Path, caption: str,
                 with_buttons: bool, label: str = "") -> Any:
            ref = MediaRef(path=str(path), mime="image/png",
                           kind="image",
                           name=f"couple pp {label}.png",
                           audience=audience)
            if with_buttons and buttons:
                try:
                    return send(ref, caption, buttons=buttons)
                except TypeError as exc:
                    record_failure("makers._one", exc)
            return send(ref, caption)

        try:
            first = _one(left_path, first_caption, False, "left")
        except Exception as exc:  # noqa: BLE001 - send failed
            return {"sent": False,
                    "reason": f"the send failed: {exc}"}
        if first is not None and not getattr(first, "ok", True):
            return {"sent": False,
                    "reason": str(getattr(first, "error", "")
                                  or "this surface couldn't send it")}
        try:
            second = _one(right_path, second_caption, True, "right")
        except Exception as exc:  # noqa: BLE001 - send failed
            return {"sent": False, "parts": 1, "reason": (
                f"the left half landed, then the send failed: "
                f"{exc} — the pair is incomplete")}
        if second is not None and not getattr(second, "ok", True):
            return {"sent": False, "parts": 1, "reason": (
                "the left half landed but the right half "
                "wouldn't send on this surface — the pair is "
                "incomplete, try again")}
        return {"sent": True, "parts": 2}

    def couplepp(self, *, send, buttons=None,
                 audience: str = "") -> dict[str, Any]:
        """Deliver a matched pair.  ``audience`` is the pair's
        public/personal classification, carried on BOTH halves'
        MediaRefs exactly as :meth:`card` carries it: the
        runtime passes "public" when the pair is made in a
        group chat (the send seam then burns the brand mark
        into each half), "personal" in the owner's own DM.
        The halves at rest are never marked — the mark is a
        property of the send."""
        last = ""
        providers = list(self.chains.get("couplepp") or [])
        # The chain is a pool, not a queue: shuffle the walk
        # so the sources genuinely rotate across calls instead
        # of the first healthy seed winning every time.
        random.shuffle(providers)
        for provider in providers:
            name = str(getattr(provider, "name", "?"))
            if not self.health.available(name):
                continue
            # A pair needs a STILL (a diptych is two profile
            # pictures): the diversified pools can answer with
            # an animation, so a provider whose draw comes
            # back animated is re-drawn a few times before the
            # walk sheds to the next source.  An animated draw
            # is not a source failure — no health penalty; the
            # source is healthy, the asset's shape is wrong
            # for a pair.
            for _draw in range(3):
                try:
                    url = provider.image_url()
                    data = self.bytes_fetcher(url)
                except Exception as exc:  # noqa: BLE001 - shed
                    self.health.note(name, False)
                    last = str(exc)[:160]
                    break
                if _sniff_image(data) is None:
                    self.health.note(name, False)
                    last = f"{name} sent bytes that aren't an image"
                    break
                shape = picture_shape(data)
                if shape is not None and shape[2] != "image":
                    last = (f"{name} sent an animation — "
                            f"a pair needs a still")
                    continue
                self.health.note(name, True)
                try:
                    left, right, info = split_diptych(data)
                except MakerError as exc:
                    return {"sent": False, "reason": str(exc)}
                out = self._deliver_pair(
                    left, right,
                    f"💞 couple pp — one artwork via {name}, split "
                    f"into a matched pair. This is the left half — "
                    f"for one of you (1 of 2)",
                    "💞 and the right half — for the other (2 of 2). "
                    "Each half is its own picture: set it as your "
                    "profile photo just as it is",
                    send, buttons=buttons, audience=audience)
                out.update({"source": name, **info})
                return out
        # Her own hand (expression completion, 2026-11-07):
        # every pool can be down — the pair still lands, drawn
        # by HER studio: one of her own artworks, split into
        # the same matched pair (the pair law holds: never two
        # unrelated pictures dressed as a pair).
        try:
            import io

            from . import emoji_studio

            spec = emoji_studio.parse_design(
                "two hearts together, gold and pink")
            img = emoji_studio.render_master(spec, size=800)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            left, right, info = split_diptych(buf.getvalue())
        except Exception:  # noqa: BLE001 - every rung failed
            return {"sent": False, "reason": (
                "the picture pools are down right now — "
                f"try again in a bit ({last})" if last else
                "the picture pools are down right now — "
                "try again in a bit")}
        out = self._deliver_pair(
            left, right,
            "💞 couple pp — my own artwork this time (the "
            "picture pools are down), drawn by me and split "
            "into a matched pair. This is the left half — for "
            "one of you (1 of 2)",
            "💞 and the right half — for the other (2 of 2). "
            "Each half is its own picture: set it as your "
            "profile photo just as it is",
            send, buttons=buttons, audience=audience)
        out.update({"source": "self:render", **info})
        return out
