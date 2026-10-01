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
    "supported video intents: 'trim the first 30 seconds', 'trim 0:30-1:00', "
    "'extract the audio', 'make a gif', 'extract 5 thumbnails', "
    "'convert to mp4', 'resize to 720p'"
)


def _fail(instruction: str) -> AmbiguousInstructionError:
    return AmbiguousInstructionError(
        f"could not understand {instruction!r} deterministically. {_HINTS}")


# ---------------------------------------------------------------------------
# image intents
# ---------------------------------------------------------------------------

def _parse_image(text: str) -> ParsedIntent | None:
    # 1. square (instagram default 1080)
    if re.search(r"\bsquare\b", text):
        size = 1080 if "insta" in text else None
        ops: list[dict[str, Any]] = [
            {"op": "crop", "aspect": "1:1", "anchor": "center"}]
        if size:
            ops.append({"op": "resize", "width": size, "height": size,
                        "mode": "exact"})
        return ParsedIntent(kind="image", ops=ops,
                            summary=f"square crop"
                            + (f" at {size}x{size}" if size else ""))
    # 2. resize: "resize to 1080", "resize to 1920x1080", "resize to 720p"
    m = re.search(r"resize to (\d+)\s*x\s*(\d+)", text)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        return ParsedIntent(kind="image", ops=[{"op": "resize", "width": w,
                                                "height": h, "mode": "exact"}],
                            summary=f"resize to {w}x{h}")
    m = re.search(r"resize to (\d+)\s*p\b", text)
    if m:
        h = int(m.group(1))
        return ParsedIntent(kind="image", ops=[{"op": "resize", "width": h,
                                                "height": h, "mode": "fit"}],
                            summary=f"resize to {h}p (fit)")
    m = re.search(r"resize to (\d{2,5})\b", text)
    if m:
        n = int(m.group(1))
        return ParsedIntent(kind="image", ops=[{"op": "resize", "width": n,
                                                "height": n, "mode": "fit"}],
                            summary=f"resize to fit {n}x{n}")
    # 3. convert
    m = re.search(r"convert to (png|jpe?g|webp|avif|bmp|tiff?)", text)
    if m:
        fmt = m.group(1).upper().replace("JPG", "JPEG")
        return ParsedIntent(kind="image", ops=[{"op": "convert", "format": fmt}],
                            summary=f"convert to {fmt}")
    # 4. watermark
    m = re.search(r"watermark with (\S+)", text)
    if m:
        logo = m.group(1).strip("'\"")
        return ParsedIntent(kind="image", ops=[{"op": "watermark",
                                                "logo": logo}],
                            summary=f"watermark with {logo}")
    # 5. rotate
    m = re.search(r"rotate (?:by )?(\d+)(?:\s*deg(?:rees)?)?", text)
    if m:
        angle = float(m.group(1))
        return ParsedIntent(kind="image", ops=[{"op": "rotate",
                                                "angle": angle}],
                            summary=f"rotate {angle:g}°")
    if re.search(r"rotate left", text):
        return ParsedIntent(kind="image", ops=[{"op": "rotate", "angle": 90}],
                            summary="rotate 90° left")
    if re.search(r"rotate right", text):
        return ParsedIntent(kind="image", ops=[{"op": "rotate", "angle": -90}],
                            summary="rotate 90° right")
    # 6. grayscale
    if re.search(r"gr[ae]yscale|black[ -]?and[ -]?white|\bb\s*&\s*w\b", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "grayscale": True}],
                            summary="grayscale")
    # 7. brightness
    if re.search(r"\bbrighten\b|increase brightness", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "brightness": 1.25}],
                            summary="brighten")
    if re.search(r"\bdarken\b|decrease brightness", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "brightness": 0.8}],
                            summary="darken")
    # 8. circle/highlight the <thing> — needs vision locate at run time
    m = re.search(r"(?:circle|highlight|mark|point at|point out|ring) the (.+)",
                  text)
    if m:
        thing = m.group(1).strip().rstrip(".")
        return ParsedIntent(
            kind="image",
            ops=[{"op": "annotate_shape", "shape": "circle",
                  "locate": thing, "outline": "red", "width": 4}],
            summary=f"circle the {thing} (vision locate)")
    # 9. add text / caption
    m = re.search(r"(?:add\s+)?text\s+[\"'](.+?)[\"']", text)
    if not m:
        m = re.search(r"^caption\s*[:\-]?\s*(.+)", text)
    if not m:
        m = re.search(r"(?:add\s+(?:a\s+)?caption\s*[:\-]?\s*)(.+)", text)
    if m:
        caption = m.group(1).strip().strip("\"'")
        pos = "top" if text.startswith("top") else "bottom"
        return ParsedIntent(kind="image",
                            ops=[{"op": "annotate_text", "text": caption,
                                  "position": pos}],
                            summary=f"text overlay: {caption!r}")
    # 10. thumbnail
    if re.search(r"\bthumbnail\b", text):
        m2 = re.search(r"(\d{2,4})\s*(?:px)?\s*thumbnail|thumbnail\s*(\d{2,4})",
                       text)
        size = int(m2.group(1) or m2.group(2)) if m2 else 256
        return ParsedIntent(kind="image",
                            ops=[{"op": "thumbnail", "size": size}],
                            summary=f"{size}px thumbnail")
    # 11. flip
    m = re.search(r"flip (horizontal|vertical)", text)
    if m:
        return ParsedIntent(kind="image",
                            ops=[{"op": "flip",
                                  "direction": m.group(1)}],
                            summary=f"flip {m.group(1)}")
    # 12. sharpen
    if re.search(r"\bsharpen\b", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "sharpness": 1.8}],
                            summary="sharpen")
    # 13. crop to aspect
    m = re.search(r"crop to (\d+\s*:\s*\d+)", text)
    if m:
        aspect = m.group(1).replace(" ", "")
        return ParsedIntent(kind="image",
                            ops=[{"op": "crop", "aspect": aspect,
                                  "anchor": "center"}],
                            summary=f"crop to {aspect}")
    # 14. meme
    m = re.search(r"meme.*?top\s*[:\-]\s*(.+?)\s+bottom\s*[:\-]\s*(.+)",
                  text)
    if m:
        return ParsedIntent(kind="image",
                            ops=[{"op": "meme", "top": m.group(1).strip(),
                                  "bottom": m.group(2).strip()}],
                            summary="meme caption")
    # 15. auto contrast / enhance
    if re.search(r"auto[ -]?contrast|auto[ -]?enhance|\benhance\b", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "autocontrast": True}],
                            summary="auto contrast")
    # 16. contrast up/down
    if re.search(r"increase contrast", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "contrast": 1.3}],
                            summary="more contrast")
    if re.search(r"decrease contrast", text):
        return ParsedIntent(kind="image",
                            ops=[{"op": "enhance", "contrast": 0.7}],
                            summary="less contrast")
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
    hit = _parse_video(text) or _parse_image(text, raw)
    if hit:
        return hit
    raise _fail(instruction)


def describe_plan(intent: ParsedIntent) -> str:
    """Human-readable plan for --dry-run / chat confirmation."""
    return intent.describe()
