"""Deterministic natural-language intent parsing for media editing.

The 20 most common instructions map straight to explicit op chains — no model
call, instant and predictable. Anything else raises
:class:`AmbiguousInstructionError` with a hint of supported phrasings; callers
may then fall through to a model (which must return ops validated against the
allowlist) or ask the user to rephrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .images import MediaEditError, validate_ops

_log = get_logger(__name__)


class AmbiguousInstructionError(MediaEditError):
    """The instruction didn't match any deterministic intent."""


@dataclass
class ParsedIntent:
    kind: str  # "image" | "video"
    ops: list[dict[str, Any]] = field(default_factory=list)  # image op chain
    action: dict[str, Any] = field(default_factory=dict)  # video descriptor
    summary: str = ""

    def describe(self) -> str:
        lines = [f"plan ({self.kind}): {self.summary}"]
        if self.kind == "image":
            for i, op in enumerate(self.ops, 1):
                params = ", ".join(f"{k}={v}" for k, v in op.items()
                                   if k != "op")
                lines.append(f"  {i}. {op['op']}({params})")
        else:
            a = self.action
            params = ", ".join(f"{k}={v}" for k, v in a.items()
                               if k != "video_op")
            lines.append(f"  1. {a.get('video_op')}({params})  [background job]")
        return "\n".join(lines)


_HINTS = (
    "supported image intents: 'make it square', 'resize to 1080', "
    "'convert to webp', 'watermark with logo.png', 'rotate 90', 'grayscale', "
    "'circle the <thing>', 'add text \"hello\"', 'make a thumbnail', "
    "'flip horizontal', 'sharpen', 'crop to 16:9', 'meme top: ... bottom: ...'; "
    "studio intents: 'cinematic look', 'make it warmer/cooler', 'letterbox', "
    "'smart crop to 4:5', 'title: My Video'; "
    "AI instruction edits: 'make a bird sit on the tree', "
    "'put him in a grand room', 'make it sunset', 'change the car to red', "
    "'remove the trash can' (need a configured generative backend); "
    "supported video intents: 'trim the first 30 seconds', 'trim 0:30-1:00', "
    "'extract the audio', 'make a gif', 'extract 5 thumbnails', "
    "'convert to mp4', 'resize to 720p'; "
    "more image intents: 'remove background', 'cartoonize', 'pencil sketch', "
    "'upscale', 'duotone', 'round corners', 'gradient overlay', "
    "'apply lut film.cube'"
)


def _fail(instruction: str) -> AmbiguousInstructionError:
    return AmbiguousInstructionError(
        f"could not understand {instruction!r} deterministically. {_HINTS}")


# ---------------------------------------------------------------------------
# generative (AI instruction) image intents
# ---------------------------------------------------------------------------

# "make a bird sit on the tree", "put him in a grand room", "make it sunset",
# "change the car to a convertible", "turn day into night", "remove the bins".
# Kept strictly separate from the mechanical edit intents above: these map to
# the generative_edit op (needs an AI backend), never to deterministic ops.
_GEN_PATTERNS = (
    r"put (?:the |a |an )?(.+?) in(?:to)? (?:a |an |the )?(.+)",
    r"place (?:the |a |an )?(.+?) in(?:to)? (?:a |an |the )?(.+)",
    r"change (?:the |a |an )?(.+?) to (?:a |an |the )?(.+)",
    r"turn (?:the |a |an )?(.+?) into (?:a |an |the )?(.+)",
    r"add (?:a |an |the )?(.+)",
    r"remove (?:the |a |an )?(.+)",
    r"make (?:a |an |the )?(.+?) (?:sit|stand|lie|fly|swim|run|walk|sleep|"
    r"smile|laugh|cry|dance)(?:\s+(?:on|in|under|behind|next to|beside)\s+"
    r"(?:the |a |an )?(.+))?",
)


def _gen_intent(text: str, instruction: str) -> ParsedIntent | None:
    """Match AI-instruction phrasing → generative_edit op."""
    for pat in _GEN_PATTERNS:
        if re.search(pat, text):
            return ParsedIntent(
                kind="image",
                ops=[{"op": "generative_edit", "instruction": instruction}],
                summary=f"AI edit: {instruction}")
    # "make it sunset" / "make her smile bigger" — but never the mechanical
    # "make it square" / "make a thumbnail" (those matched earlier anyway).
    m = re.search(r"^make (?!(?:it square|a thumbnail|an? gif|a meme)\b)(.+)$",
                  text)
    if m:
        return ParsedIntent(
            kind="image",
            ops=[{"op": "generative_edit", "instruction": instruction}],
            summary=f"AI edit: {instruction}")
    return None


# ---------------------------------------------------------------------------
# image intents
# ---------------------------------------------------------------------------


def _merge_enhance_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive ``enhance`` ops into one.

    "make it brighter and increase contrast" produces two enhance ops;
    merging them avoids running the enhance pipeline twice.  Non-enhance
    ops pass through untouched, and enhance ops separated by other ops
    are NOT merged (order matters: e.g. enhance → crop → enhance).
    """
    out: list[dict[str, Any]] = []
    for op in ops:
        if op.get("op") == "enhance" and out and out[-1].get("op") == "enhance":
            merged = dict(out[-1])
            merged.update({k: v for k, v in op.items() if k != "op"})
            out[-1] = merged
        else:
            out.append(op)
    return out


def _parse_image(text: str, raw: str | None = None) -> ParsedIntent | None:
    """Match mechanical image intents, collecting ALL that apply.

    Compound instructions like "make it brighter and crop to square" now
    produce a single op chain with every matched intent, in pattern order.
    Previously only the first match was returned and the rest were silently
    dropped.
    """
    hits: list[tuple[list[dict[str, Any]], str]] = []  # (ops, summary)
    # 1. square (instagram default 1080)
    if re.search(r"\bsquare\b", text):
        size = 1080 if "insta" in text else None
        ops: list[dict[str, Any]] = [
            {"op": "crop", "aspect": "1:1", "anchor": "center"}]
        if size:
            ops.append({"op": "resize", "width": size, "height": size,
                        "mode": "exact"})
        hits.append((ops, f"square crop" + (f" at {size}x{size}" if size else "")))
    # 2. resize: "resize to 1080", "resize to 1920x1080", "resize to 720p"
    m = re.search(r"resize to (\d+)\s*x\s*(\d+)", text)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        hits.append(([{"op": "resize", "width": w,
                                                "height": h, "mode": "exact"}], f"resize to {w}x{h}"))
    m = re.search(r"resize to (\d+)\s*p\b", text)
    if m:
        h = int(m.group(1))
        hits.append(([{"op": "resize", "width": h,
                                                "height": h, "mode": "fit"}], f"resize to {h}p (fit)"))
    m = re.search(r"resize to (\d{2,5})\b", text)
    if m:
        n = int(m.group(1))
        hits.append(([{"op": "resize", "width": n,
                                                "height": n, "mode": "fit"}], f"resize to fit {n}x{n}"))
    # 3. convert
    m = re.search(r"convert to (png|jpe?g|webp|avif|bmp|tiff?)", text)
    if m:
        fmt = m.group(1).upper().replace("JPG", "JPEG")
        hits.append(([{"op": "convert", "format": fmt}], f"convert to {fmt}"))
    # 4. watermark
    m = re.search(r"watermark with (\S+)", text)
    if m:
        logo = m.group(1).strip("'\"")
        hits.append(([{"op": "watermark",
                                                "logo": logo}], f"watermark with {logo}"))
    # 5. rotate
    m = re.search(r"rotate (?:by )?(\d+)(?:\s*deg(?:rees)?)?", text)
    if m:
        angle = float(m.group(1))
        hits.append(([{"op": "rotate",
                                                "angle": angle}], f"rotate {angle:g}°"))
    if re.search(r"rotate left", text):
        hits.append(([{"op": "rotate", "angle": 90}], "rotate 90° left"))
    if re.search(r"rotate right", text):
        hits.append(([{"op": "rotate", "angle": -90}], "rotate 90° right"))
    # 6. grayscale
    if re.search(r"gr[ae]yscale|black[ -]?and[ -]?white|\bb\s*&\s*w\b", text):
        hits.append(([{"op": "enhance", "grayscale": True}], "grayscale"))
    # 7. brightness
    if re.search(r"\bbrighten\b|\bbrighter\b|increase brightness", text):
        hits.append(([{"op": "enhance", "brightness": 1.25}], "brighten"))
    if re.search(r"\bdarken\b|\bdarker\b|decrease brightness", text):
        hits.append(([{"op": "enhance", "brightness": 0.8}], "darken"))
    # 8. circle/highlight the <thing> — needs vision locate at run time
    m = re.search(r"(?:circle|highlight|mark|point at|point out|ring) the (.+)",
                  text)
    if m:
        thing = m.group(1).strip().rstrip(".")
        hits.append(([{"op": "annotate_shape", "shape": "circle",
                  "locate": thing, "outline": "red", "width": 4}], f"circle the {thing} (vision locate)"))
    # 9. add text / caption
    m = re.search(r"(?:add\s+)?text\s+[\"'](.+?)[\"']", text)
    if not m:
        m = re.search(r"^caption\s*[:\-]?\s*(.+)", text)
    if not m:
        m = re.search(r"(?:add\s+(?:a\s+)?caption\s*[:\-]?\s*)(.+)", text)
    if m:
        caption = m.group(1).strip().strip("\"'")
        pos = "top" if text.startswith("top") else "bottom"
        hits.append(([{"op": "annotate_text", "text": caption,
                                  "position": pos}], f"text overlay: {caption!r}"))
    # 10. thumbnail
    if re.search(r"\bthumbnail\b", text):
        m2 = re.search(r"(\d{2,4})\s*(?:px)?\s*thumbnail|thumbnail\s*(\d{2,4})",
                       text)
        size = int(m2.group(1) or m2.group(2)) if m2 else 256
        hits.append(([{"op": "thumbnail", "size": size}], f"{size}px thumbnail"))
    # 11. flip
    m = re.search(r"flip (horizontal|vertical)", text)
    if m:
        hits.append(([{"op": "flip",
                                  "direction": m.group(1)}], f"flip {m.group(1)}"))
    # 12. sharpen
    if re.search(r"\bsharpen\b", text):
        hits.append(([{"op": "enhance", "sharpness": 1.8}], "sharpen"))
    # 13. crop to aspect (not "smart crop", which is a studio intent below)
    m = re.search(r"(?<!smart )crop to (\d+\s*:\s*\d+)", text)
    if m:
        aspect = m.group(1).replace(" ", "")
        hits.append(([{"op": "crop", "aspect": aspect,
                                  "anchor": "center"}], f"crop to {aspect}"))
    # 14. meme
    m = re.search(r"meme.*?top\s*[:\-]\s*(.+?)\s+bottom\s*[:\-]\s*(.+)",
                  text)
    if m:
        hits.append(([{"op": "meme", "top": m.group(1).strip(),
                                  "bottom": m.group(2).strip()}], "meme caption"))
    # 15. auto contrast / enhance
    if re.search(r"auto[ -]?contrast|auto[ -]?enhance|\benhance\b", text):
        hits.append(([{"op": "enhance", "autocontrast": True}], "auto contrast"))
    # 16. contrast up/down
    if re.search(r"increase contrast", text):
        hits.append(([{"op": "enhance", "contrast": 1.3}], "more contrast"))
    if re.search(r"decrease contrast", text):
        hits.append(([{"op": "enhance", "contrast": 0.7}], "less contrast"))
    # -- studio mechanical intents (lazy import registers the studio ops) --
    from . import studio as _studio  # noqa: F401
    # 17. filter presets: "cinematic look", "apply vintage filter"
    m = re.search(r"\b(portrait|cinematic|vintage|bw-drama|vibrant|"
                  r"teal-orange|noir|golden-hour|cool-matte|warm-fade)\b",
                  text)
    if m and re.search(r"look|filter|style|preset|apply|make it|give it", text):
        hits.append(([{"op": "filter", "preset": m.group(1),
                                  "strength": 1.0}], f"filter: {m.group(1)}"))
    # 18. quick grades: warmer / cooler / moodier / more vivid
    if re.search(r"\bwarmer\b", text):
        hits.append(([{"op": "grade", "temperature": 800}], "grade: warmer"))
    if re.search(r"\bcooler\b", text):
        hits.append(([{"op": "grade", "temperature": -800}], "grade: cooler"))
    if re.search(r"\bmoodier\b|\bmoodier look\b", text):
        hits.append(([{"op": "grade", "vignette": 0.6,
                                  "lift": [-0.04, -0.04, -0.04]}], "grade: moodier"))
    if re.search(r"\bmore vivid\b|\bmore vibrant\b", text):
        hits.append(([{"op": "grade", "vibrance": 0.5,
                                  "saturation": 1.2}], "grade: more vivid"))
    # 19. letterbox: "letterbox", "cinematic bars", "anamorphic"
    if re.search(r"letterbox|cinematic bars|anamorphic|\b2\.39:1\b|\b21:9\b",
                 text):
        hits.append(([{"op": "letterbox", "aspect": "21:9",
                                  "color": "black"}], "letterbox 21:9"))
    # 20. smart crop / reframe to an aspect
    m = re.search(r"(?:smart[ -]?crop|reframe)(?:\s+to)?\s+"
                  r"(\d+(?:\.\d+)?\s*[:x]\s*\d+(?:\.\d+)?)", text)
    if m:
        hits.append(([{"op": "smart_crop",
                                  "aspect": m.group(1).replace(" ", ""),
                                  "mode": "saliency"}], f"smart crop {m.group(1)}"))
    # 21. title text: "title: My Day"
    m = re.search(r"^title\s*:\s*(.+)$", text)
    if m:
        hits.append(([{"op": "text_layer", "text": m.group(1),
                                  "position": "top", "size": 72,
                                  "color": "white", "stroke_width": 3,
                                  "shadow": True, "margin": 40}], f"title: {m.group(1)}"))
    # 22. remove background → dedicated segmentation module
    if re.search(r"remove (?:the )?background|cut ?out (?:the )?subject|"
                 r"transparent background|make the background transparent", text):
        hits.append(([{"op": "bg_remove_v2"}], "remove background (AI)"))
    # 23. cartoonize / pencil sketch
    if re.search(r"\bcartooni[sz]e\b|cartoon (?:effect|look|style)", text):
        hits.append(([{"op": "cartoonize"}], "cartoonize"))
    if re.search(r"pencil sketch|sketch (?:effect|look)", text):
        hits.append(([{"op": "pencil_sketch"}], "pencil sketch"))
    # 24. upscale / super-resolution
    if re.search(r"\bupscale\b|super[ -]?resolution|increase resolution|"
                 r"make it (?:hd|4k|sharper)", text):
        hits.append(([{"op": "upscale_sr", "scale": 4.0}],
                     "AI upscale 4x"))
    # 25. duotone
    if re.search(r"\bduotone\b|two[ -]?tone", text):
        hits.append(([{"op": "duotone"}], "duotone grade"))
    # 26. rounded corners
    if re.search(r"round(?:ed)? corners", text):
        hits.append(([{"op": "round_corners", "radius": 48}],
                     "rounded corners"))
    # 27. gradient overlay / scrim
    if re.search(r"gradient overlay|fade to black|vignette scrim", text):
        hits.append(([{"op": "gradient_overlay"}], "gradient overlay"))
    # 28. apply a .cube LUT: "apply lut film.cube"
    m = re.search(r"apply (?:the )?lut (\S+\.cube)", text)
    if m:
        hits.append(([{"op": "cube_lut", "lut": m.group(1).strip("'\"")}],
                     f"apply LUT {m.group(1)}"))
    # -- generative (AI instruction) edits --------------------------------
    # These come AFTER every mechanical intent so "make it square",
    # "make a thumbnail", "add text ..." etc. never land here.
    from . import generate as _generate  # noqa: F401
    # Combine all matched mechanical intents into one op chain.
    # Generative (AI) intents are only tried when NO mechanical intent
    # matched — mixing them would send the mechanical phrasing to the
    # generative backend.
    if hits:
        all_ops: list[dict[str, Any]] = []
        summaries: list[str] = []
        for ops, summary in hits:
            all_ops.extend(ops)
            summaries.append(summary)
        # Merge adjacent enhance ops (brightness + contrast + etc.) into
        # one op so the pipeline doesn't run enhance 3 times.
        all_ops = _merge_enhance_ops(all_ops)
        return ParsedIntent(
            kind="image",
            ops=all_ops,
            summary=" + ".join(summaries))
    gen = _gen_intent(text, raw or text)
    if gen is not None:
        return gen
    return None


# ---------------------------------------------------------------------------
# video intents
# ---------------------------------------------------------------------------

def _parse_video(text: str) -> ParsedIntent | None:
    # 15. trim first N seconds / trim A-B
    m = re.search(
        r"(?:trim|cut)(?:\s+the)?\s+first\s+(\d+(?:\.\d+)?)\s*"
        r"(s|sec|secs|second|seconds)?\b", text)
    if m:
        secs = float(m.group(1))
        return ParsedIntent(kind="video",
                            action={"video_op": "trim", "start": 0,
                                    "end": secs},
                            summary=f"trim first {secs:g}s")
    m = re.search(r"(?:trim|cut)\s+(\S+)\s*(?:-|to)\s*(\S+)", text)
    if m:
        return ParsedIntent(kind="video",
                            action={"video_op": "trim",
                                    "start": m.group(1), "end": m.group(2)},
                            summary=f"trim {m.group(1)}-{m.group(2)}")
    # 16. extract audio
    if re.search(r"extract (?:the )?audio|audio (?:only|track)|"
                 r"\bto mp3\b|\bto wav\b", text):
        ext = ".wav" if "wav" in text else ".mp3"
        return ParsedIntent(kind="video",
                            action={"video_op": "extract_audio", "ext": ext},
                            summary=f"extract audio ({ext})")
    # 17. gif
    if re.search(r"\bgif\b", text):
        m2 = re.search(r"first\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds?)?", text)
        dur = float(m2.group(1)) if m2 else 3.0
        return ParsedIntent(kind="video",
                            action={"video_op": "make_gif",
                                    "start": 0, "duration": dur},
                            summary=f"gif of first {dur:g}s")
    # 18. thumbnails
    m = re.search(r"(?:extract|pull|get|make)?\s*(\d+)?\s*thumbnails?", text)
    if m and "thumbnail" in text:
        if m.group(1):
            return ParsedIntent(
                kind="video",
                action={"video_op": "extract_frames",
                        "count": int(m.group(1))},
                summary=f"{m.group(1)} thumbnails")
        return ParsedIntent(kind="video",
                            action={"video_op": "extract_frames",
                                    "interval": 10},
                            summary="thumbnails every 10s")
    # 19. convert video container
    m = re.search(r"convert to (mp4|webm|mov|mkv)", text)
    if m:
        return ParsedIntent(kind="video",
                            action={"video_op": "transcode",
                                    "ext": f".{m.group(1)}"},
                            summary=f"convert to {m.group(1)}")
    # 20. resize video
    m = re.search(r"(?:resize|scale)(?: to)? (\d+)\s*p\b", text)
    if m:
        h = int(m.group(1))
        return ParsedIntent(kind="video",
                            action={"video_op": "transcode", "height": h},
                            summary=f"resize to {h}p")
    m = re.search(r"(?:resize|scale)(?: to)? (\d+)\s*x\s*(\d+)", text)
    if m:
        return ParsedIntent(kind="video",
                            action={"video_op": "transcode",
                                    "width": int(m.group(1)),
                                    "height": int(m.group(2))},
                            summary=f"resize to {m.group(1)}x{m.group(2)}")
    return None


def parse_instruction(instruction: str, *,
                      kind: str = "auto") -> ParsedIntent:
    """Parse a natural-language editing instruction deterministically.

    ``kind``: "image", "video", or "auto" (video intents win on overlap —
    callers that know the file type should pass it explicitly).
    """
    raw = instruction.strip()
    if not raw:
        raise _fail(instruction)
    text = raw.lower()

    def _only_video() -> ParsedIntent:
        hit = _parse_video(text)
        if hit:
            return hit
        raise _fail(instruction)

    def _only_image() -> ParsedIntent:
        hit = _parse_image(text)
        if hit:
            return hit
        raise _fail(instruction)

    if kind == "video":
        return _only_video()
    if kind == "image":
        return _only_image()
    # Document intents are specific multi-word phrases (Sunday morning,
    # tighten, match look, cut-where-said, ...) — try them before the
    # image generative catch-all ("make ...") can swallow them.
    doc = parse_document_intent(instruction)
    if doc is not None:
        return doc
    hit = _parse_video(text) or _parse_image(text, raw)
    if hit:
        return hit
    raise _fail(instruction)

def describe_plan(intent: ParsedIntent) -> str:
    """Human-readable plan for --dry-run / chat confirmation."""
    return format_plan(intent, theme="plain")


def format_plan(intent: ParsedIntent, theme: str = "rich") -> str:
    """Present a parsed intent as a readable plan.

    ``theme="plain"``: the classic indented text (what ``describe_plan``
    returns). ``theme="rich"``: emoji step markers and a boxed header —
    the version you show the user in chat.
    """
    theme = (theme or "rich").lower()
    if theme not in ("rich", "plain"):
        raise AmbiguousInstructionError(
            f"unknown plan theme {theme!r}; use rich|plain")
    if theme == "plain":
        return intent.describe()
    bar = "─" * 44
    lines = [f"┌{bar}┐",
             f"│ 🗺️  plan ({intent.kind})",
             f"│ {intent.summary[:40]:<40} │",
             f"└{bar}┘"]
    if intent.kind == "image":
        glyph = {"crop": "✂️", "resize": "📐", "rotate": "🔄",
                 "filter": "🎨", "grade": "🌈", "meme": "😂",
                 "watermark": "©️", "upscale_sr": "🔍",
                 "bg_remove_v2": "🪄", "cartoonize": "🖌️",
                 "annotate_text": "🔤", "thumbnail": "🖼️"}
        for i, op in enumerate(intent.ops, 1):
            g = glyph.get(op.get("op", ""), "⚙️")
            params = ", ".join(f"{k}={v}" for k, v in op.items()
                               if k != "op")
            lines.append(f"  {g} {i}. {op['op']}"
                         + (f"  ·  {params}" if params else ""))
    elif intent.kind == "document":
        for i, op in enumerate(intent.ops, 1):
            params = ", ".join(f"{k}={v}" for k, v in op.items()
                               if k != "op")
            lines.append(f"  📄 {i}. {op.get('op')}"
                         + (f"  ·  {params}" if params else ""))
    else:
        a = intent.action
        params = ", ".join(f"{k}={v}" for k, v in a.items()
                           if k != "video_op")
        lines.append(f"  🎬 1. {a.get('video_op')}"
                     + (f"  ·  {params}" if params else "")
                     + "  [background job]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document intents + learning (EDIT_GEN_UPGRADE 2.2)
# ---------------------------------------------------------------------------
# The intent front door, broadened past mechanical ops to *intentions*:
# "make it feel Sunday morning", "make him sound tired",
# "tighten by a third", "match look of that shot". An intention maps to
# a preset STACK (style-agnostic engine + preset packs — the look is
# data in the stack, never code in the engine). Corrections
# ("no, warmer") are recorded as supervised pairs in CorrectionStore
# and replayed as preference adjustments on the next parse.

#: intention → preset stack. Each stack is DATA: ops the document
#: engine already understands (effects, emotion ops, pacing ops,
#: look-lock fields). No aesthetic lives in the engine; the stack
#: *is* the look.
INTENT_PRESET_STACKS: dict[str, dict[str, Any]] = {
    "sunday_morning": {
        "description": "warm, soft, unhurried morning feel",
        "ops": [
            {"op": "look_lock", "grade": "warm", "grain": 4.0,
             "lens": "soft", "palette": [[255, 214, 170],
                                         [255, 236, 200], [120, 90, 70]]},
            {"op": "effect", "name": "color_grade",
             "params": {"grade": "warm"}},
            {"op": "effect", "name": "grain", "params": {"strength": 4.0}},
            {"op": "effect", "name": "ken_burns",
             "params": {"direction": "in", "zoom": 1.08}},
            {"op": "music_bed", "mood": "gentle", "volume": 0.25},
            {"op": "caption_style", "style": "soft"},
        ],
    },
    "tired_voice": {
        "description": "delivery: tired / low-energy voice",
        "ops": [{"op": "emotion", "emotion": "tired", "intensity": 0.8,
                 "scope": "voice"}],
    },
    "energetic_voice": {
        "description": "delivery: energetic voice",
        "ops": [{"op": "emotion", "emotion": "energetic", "intensity": 0.8,
                 "scope": "voice"}],
    },
    "tighten_third": {
        "description": "pacing: tighten by one third",
        "ops": [{"op": "tighten", "fraction": 1.0 / 3.0}],
    },
    "match_look": {
        "description": "match the look of a reference shot",
        "ops": [{"op": "look_lock_from_reference",
                 "source": "reference_shot"}],
    },
    "warmer": {
        "description": "correction direction: warmer",
        "ops": [{"op": "grade", "temperature": 800}],
    },
    "cooler": {
        "description": "correction direction: cooler",
        "ops": [{"op": "grade", "temperature": -800}],
    },
}


def _stack_ops(name: str) -> list[dict[str, Any]]:
    import copy as _copy
    stack = INTENT_PRESET_STACKS.get(name)
    if stack is None:
        raise AmbiguousInstructionError(f"unknown intent stack {name!r}")
    return _copy.deepcopy(stack["ops"])


def parse_document_intent(instruction: str) -> ParsedIntent | None:
    """Deterministic NL → document edit ops (the intent front door).

    Returns ``None`` when nothing matches (caller falls through to the
    mechanical parsers or raises). ``ParsedIntent.kind == "document"``,
    ``ops`` = the document op list, ``action`` carries the stack name.
    """
    raw = (instruction or "").strip()
    if not raw:
        return None
    text = raw.lower()

    def _doc(stack: str, summary: str, extra: list | None = None) -> ParsedIntent:
        ops = _stack_ops(stack) + list(extra or [])
        return ParsedIntent(kind="document", ops=ops,
                            action={"document_op": stack, "ops": ops},
                            summary=summary)

    # "make it feel sunday morning" / "sunday morning feel"
    if "sunday morning" in text:
        return _doc("sunday_morning", "feel: Sunday morning")
    # "make him/her/them sound tired / exhausted / sleepy / energetic"
    m = re.search(r"sound (tired|exhausted|sleepy|energetic|excited|"
                  r"angry|warm|sad|whisper)", text)
    if m:
        emo = {"exhausted": "tired", "sleepy": "tired"}.get(m.group(1),
                                                            m.group(1))
        stack = {"tired": "tired_voice",
                 "energetic": "energetic_voice"}.get(emo, "tired_voice")
        ops = [{"op": "emotion", "emotion": emo, "intensity": 0.8,
                "scope": "voice"}]
        return ParsedIntent(
            kind="document", ops=ops,
            action={"document_op": stack, "ops": ops},
            summary=f"delivery: sound {emo}")
    # "make him sound tired" without "sound" verb variants
    if re.search(r"\b(sound|voice)\b.*\btired\b", text):
        return _doc("tired_voice", "delivery: sound tired")
    # "tighten by a third / by 30% / by half"
    m = re.search(r"tighten(?:\s+it)?\s+by\s+(?:a\s+)?(third|half|quarter"
                  r"|\d+\s*%)", text)
    if m or "tighten by" in text:
        frac = 1.0 / 3.0
        if m:
            val = m.group(1)
            if val == "half":
                frac = 0.5
            elif val == "quarter":
                frac = 0.25
            elif val.endswith("%"):
                frac = float(val.rstrip("%").strip()) / 100.0
        ops = [{"op": "tighten", "fraction": frac}]
        return ParsedIntent(kind="document", ops=ops,
                            action={"document_op": "tighten", "ops": ops},
                            summary=f"tighten by {frac:.0%}")
    if re.search(r"\btighten\b", text):
        return _doc("tighten_third", "tighten by a third")
    # "match look of that shot / match the look"
    if re.search(r"match (?:the )?look", text):
        return _doc("match_look", "match look of reference shot")
    # "cut video where he says 'never again'" / "cut where she says ..."
    m = re.search(r"cut(?:\s+video)?\s+where\s+(?:he|she|they|it)\s+"
                  r"says?\s+[\"'](.+?)[\"']", text)
    if m:
        phrase = m.group(1)
        ops = [{"op": "cut_where_said", "phrase": phrase, "keep": True}]
        return ParsedIntent(kind="document", ops=ops,
                            action={"document_op": "cut_where_said",
                                    "ops": ops},
                            summary=f"cut video where he says {phrase!r}")
    # "use that audio as music bed" / "use that audio as the music bed"
    if re.search(r"use (?:that|this|the) audio as (?:the )?(music bed|bed)",
                 text):
        ops = [{"op": "use_audio_as_music_bed", "volume": 0.35}]
        return ParsedIntent(kind="document", ops=ops,
                            action={"document_op": "use_audio_as_music_bed",
                                    "ops": ops},
                            summary="use that audio as music bed")
    # Bare "warmer"/"cooler" stay image grade ops (existing behaviour) —
    # corrections are handled by CorrectionStore.learn_from_correction,
    # not by claiming the bare word here.
    return None


class CorrectionStore:
    """Supervised pairs from owner corrections ("no, warmer").

    ``record_correction`` stores (instruction, correction, ops_before,
    ops_after). ``preferences`` replays the pairs as adjustments the
    next parse applies: a "warmer" correction after a look stack bumps
    that stack's grade temperature on future parses. JSON-persistable
    via :meth:`to_dict` / :meth:`from_dict` / :meth:`save` / :meth:`load`.
    """

    def __init__(self, pairs: list[dict[str, Any]] | None = None):
        self.pairs: list[dict[str, Any]] = [dict(p) for p in (pairs or [])]

    def record_correction(self, instruction: str, correction: str,
                          ops_before: list[dict[str, Any]],
                          ops_after: list[dict[str, Any]]) -> dict[str, Any]:
        """Record one supervised pair. Returns the stored pair."""
        pair = {"instruction": instruction, "correction": correction,
                "ops_before": [dict(o) for o in ops_before],
                "ops_after": [dict(o) for o in ops_after]}
        self.pairs.append(pair)
        return pair

    def learn_from_correction(self, correction: str,
                              ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply a correction phrase to an op list (the learning step).

        "warmer" appends/increments a warm grade; "cooler" the reverse;
        "tighter" bumps a tighten fraction. Unknown corrections pass
        the ops through unchanged — never invent an adjustment.
        """
        import copy as _copy
        out = _copy.deepcopy(ops)
        text = (correction or "").lower()
        if "warmer" in text:
            out.append({"op": "grade", "temperature": 800,
                        "learned_from": "correction: warmer"})
        elif "cooler" in text:
            out.append({"op": "grade", "temperature": -800,
                        "learned_from": "correction: cooler"})
        elif "tighter" in text:
            for op in out:
                if op.get("op") == "tighten":
                    op["fraction"] = min(0.9, float(op.get("fraction", 1 / 3)) + 0.1)
        return out

    def preferences(self) -> dict[str, Any]:
        """Aggregate learned preferences: correction → count."""
        counts: dict[str, int] = {}
        for p in self.pairs:
            key = str(p.get("correction", "")).strip().lower()
            counts[key] = counts.get(key, 0) + 1
        return {"pairs": len(self.pairs), "corrections": counts}

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [dict(p) for p in self.pairs]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectionStore":
        return cls(pairs=list(data.get("pairs", [])))

    def save(self, path: str) -> str:
        import json as _json
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str) -> "CorrectionStore":
        import json as _json
        from pathlib import Path as _Path
        return cls.from_dict(_json.loads(_Path(path).read_text("utf-8")))


def record_correction(instruction: str, correction: str,
                      ops_before: list[dict[str, Any]],
                      ops_after: list[dict[str, Any]],
                      store: CorrectionStore | None = None) -> dict[str, Any]:
    """Module-level front door: record one supervised correction pair."""
    store = store or CorrectionStore()
    return store.record_correction(instruction, correction, ops_before,
                                   ops_after)
