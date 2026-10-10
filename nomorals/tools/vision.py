"""Vision: image understanding (Prompt 09).

One ``vision`` tool with four actions — ``describe``, ``read_text``,
``locate``, ``compare`` — over a single intake layer. Images are read ONLY
from: explicit owner paths under the workspace, ``inbox:<id>`` / ``room:<slug>:
<path>`` references, ``attachment:<n>`` references to the current chat message's
media, raw bytes handed in-process, or an explicit URL.

Description comes from whichever registered provider supports vision. Image
metadata (dimensions, format, size) is parsed from headers directly, so the tool
still reports something useful when no VLM is configured — which is the offline
and Termux case.

Results are cached by content hash: describing the same screenshot twice should
not cost two model calls.

Privacy rules (hard, in code):
- every vision call is audit-logged (timestamp, action, source, byte sizes —
  never the image itself) and the first call per process emits a one-time
  notice that images leave the machine for the model provider;
- ``locate`` coordinates are approximate and the result says so;
- screenshots are privileged: settings gate + explicit per-call confirmation,
  never scheduled, never background.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ..llm.brain import brain_for
from ..core.errors import ModelError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = [
    "analyze",
    "compare",
    "describe",
    "extract",
    "format_chat_summary",
    "image_metadata",
    "layout",
    "locate",
    "read_text",
    "register",
    "resolve_image",
    "vision_capabilities",
]

_log = get_logger(__name__)

DEFAULT_MAX_DIMENSION = 1568
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024

_LOCATE_DISCLAIMER = (
    "Coordinates are approximate (0-1000 normalized, origin top-left). "
    "Re-verify on screen before any click automation."
)

_READ_TEXT_PROMPT = (
    "Transcribe ALL visible text in this image verbatim, top to bottom, left to "
    "right. Do not correct spelling, grammar, or punctuation — return what's "
    "there. For illegible or uncertain regions write [illegible] instead of "
    "guessing. Preserve line breaks."
)

_LOCATE_PROMPT = (
    "Locate {target} in this image. Reply with ONLY a JSON object, no other "
    'text: {{"x": <0-1000>, "y": <0-1000>, "w": <0-1000>, "h": <0-1000>, '
    '"confidence": <0.0-1.0>}}. Coordinates are normalized 0-1000 from the '
    'top-left corner. If it is not visible, reply {{"found": false}}.'
)

_COMPARE_PROMPT = (
    "These are two photos side by side: LEFT is the first/before image, RIGHT "
    "is the second/after image. {extra} List every meaningful difference you "
    "can see. If they look identical, say so plainly instead of inventing "
    "differences."
)

_EXTRACT_PROMPT = (
    "Analyze this image for a chat summary and reply with ONLY a JSON "
    'object, no other text:\n{"summary": "<one or two sentences: what this '
    'image is about>", "key_text": "<the most important text visible, '
    'verbatim; empty string if there is none>", "notable_elements": '
    '["<element 1>", "<element 2>", ... up to 8 items], "scene_type": '
    '"<photo|screenshot|document|chart|diagram|meme|other>"}'
    "{extra}"
)

_EXTRACT_EXTRA = "\nPay special attention to: {prompt}"

#: one-time "images leave the machine" notice, per process
_NOTICE_EMITTED = False


def image_metadata(data: bytes) -> dict[str, Any]:
    """Format, dimensions, and size — parsed from headers, no PIL needed.

    Canonical implementation lives in :mod:`nomorals.vision.native`
    (``header_metadata``); this delegates so the two can never drift.
    """
    from ..vision.native import header_metadata

    return header_metadata(data)


# ── settings / logging ───────────────────────────────────────────────────────


def _vision_settings(context: Any) -> Any:
    settings = getattr(context, "settings", None) if context is not None else None
    partner = getattr(settings, "partner", None) if settings is not None else None
    return getattr(partner, "vision", None)


def _knob(vision: Any, name: str, default: Any) -> Any:
    return getattr(vision, name, default) if vision is not None else default


def _emit_notice_once() -> None:
    global _NOTICE_EMITTED
    if not _NOTICE_EMITTED:
        _NOTICE_EMITTED = True
        _log.warning(
            "vision sends images to the configured vision model provider for "
            "analysis — that is data leaving this machine. No background or "
            "ambient image capture is ever performed."
        )


def _audit(action: str, source: str, original_bytes: int, sent_bytes: int,
           provider: str, model: str, usage: dict[str, Any] | None,
           context: Any) -> None:
    """Per-call audit log: timestamp, action, source, sizes — never pixels."""
    vision = _vision_settings(context)
    if not _knob(vision, "log_calls", True):
        return
    _emit_notice_once()
    _log.info(
        "vision %s source=%s bytes=%d sent=%d provider=%s model=%s usage=%s",
        action, source, original_bytes, sent_bytes,
        provider or "?", model or "?", usage or {},
    )


# ── intake: one resolver for every image source ─────────────────────────────


def _resolve_inbox_ref(context: Any, ref: str) -> tuple[bytes, str]:
    """``inbox:<id>`` → the dropped file's bytes."""
    item_id = ref.split(":", 1)[1]
    settings = getattr(context, "settings", None) if context is not None else None
    workspace = Path(getattr(settings, "workspace_dir", "workspace"))
    db_path = workspace / "inbox" / "inbox.db"
    if not db_path.exists():
        raise ToolError(f"inbox reference {ref!r}: no inbox database at {db_path}")
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT path, name FROM inbox_items WHERE id = ?", (item_id,)
        ).fetchone()
    if row is None:
        raise ToolError(f"inbox reference {ref!r}: no such item")
    target = Path(row[0])
    if not target.exists():
        raise ToolError(f"inbox reference {ref!r}: file missing ({row[1]})")
    return target.read_bytes(), f"inbox:{item_id}"


def _resolve_room_ref(context: Any, ref: str) -> tuple[bytes, str]:
    """``room:<slug>:<path>`` → bytes, resolved inside the room sandbox."""
    _, slug, rel = ref.split(":", 2)
    from ..workspace.rooms import RoomContext, RoomManager

    settings = getattr(context, "settings", None) if context is not None else None
    workspace = Path(getattr(settings, "workspace_dir", "workspace"))
    manager = RoomManager(workspace)
    room = manager.get(slug)
    if room is None:
        raise ToolError(f"room reference {ref!r}: no such room")
    target = RoomContext(manager, room).path(rel)
    if not target.is_file():
        raise ToolError(f"room reference {ref!r}: not a file")
    return target.read_bytes(), f"room:{slug}:{rel}"


def _resolve_attachment(context: Any, ref: str) -> tuple[bytes, str]:
    """``attachment:<n>`` → the nth image attached to the current message.

    The chat runtime stashes the current message's media on
    ``context.extras["attachments"]`` (list of ``{"path","name","mime"}``).
    These paths live under the chat media dirs — outside the workspace — and
    are explicitly allowed here because they came from the owner's own chat.
    """
    try:
        index = int(ref.split(":", 1)[1])
    except (ValueError, IndexError):
        raise ToolError(f"bad attachment reference {ref!r} (want attachment:<n>)")
    extras = getattr(context, "extras", None) if context is not None else None
    attachments = (extras or {}).get("attachments") or []
    if not 0 <= index < len(attachments):
        raise ToolError(
            f"attachment:{index} out of range — this message has "
            f"{len(attachments)} image attachment(s)"
        )
    entry = attachments[index]
    target = Path(entry["path"])
    if not target.is_file():
        raise ToolError(f"attachment:{index} file missing ({entry.get('name', '?')})")
    return target.read_bytes(), f"attachment:{index}"


def resolve_image(
    context: Any,
    *,
    path: str = "",
    url: str = "",
    data: bytes | None = None,
    reference: str = "",
) -> tuple[bytes, str]:
    """Resolve one image from any supported source → (bytes, source label).

    ``reference`` accepts ``inbox:<id>``, ``room:<slug>:<path>`` and
    ``attachment:<n>``. Exactly one source must be given.
    """
    given = [bool(path), bool(url), data is not None, bool(reference)]
    if sum(given) != 1:
        raise ToolError(
            "vision needs exactly one image source: path=, url=, data=, "
            "or reference= (inbox:<id> / room:<slug>:<path> / attachment:<n>)"
        )
    if data is not None:
        return bytes(data), "bytes"
    if reference:
        if reference.startswith("inbox:"):
            return _resolve_inbox_ref(context, reference)
        if reference.startswith("room:"):
            return _resolve_room_ref(context, reference)
        if reference.startswith("attachment:"):
            return _resolve_attachment(context, reference)
        raise ToolError(
            f"unknown reference {reference!r} "
            "(want inbox:<id> / room:<slug>:<path> / attachment:<n>)"
        )
    if path:
        from .filesystem import safe_path

        target = safe_path(context, path, must_exist=True)
        if not target.is_file():
            raise ToolError(f"vision path is not a file: {path!r}")
        return target.read_bytes(), f"path:{path}"
    # url
    from ..core.http import HttpClient

    return HttpClient(timeout=60.0).get(url).body, f"url:{url}"


def _check_size(data: bytes, context: Any) -> None:
    vision = _vision_settings(context)
    cap = _knob(vision, "max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)
    if not data:
        raise ToolError("vision got empty image bytes")
    if len(data) > cap:
        raise ToolError(
            f"image is {len(data)} bytes — over the {cap}-byte vision cap; "
            "downscale it first or raise vision.max_image_bytes"
        )


def _downscale(data: bytes, max_dim: int) -> tuple[bytes, str]:
    """Shrink images over ``max_dim`` before sending. Originals untouched."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return data, "PIL not installed — sent at original size"
    import io

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - undecodable: send as-is, model decides
        return data, "undecodable by PIL — sent as-is"
    width, height = img.size
    if max(width, height) <= max_dim:
        return data, ""
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    fmt = (img.format or "PNG").upper()
    if fmt not in ("PNG", "JPEG", "WEBP"):
        fmt = "PNG"
    if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    note = f"downscaled {width}x{height} → {img.size[0]}x{img.size[1]}"
    return buf.getvalue(), note


# ── the four actions ─────────────────────────────────────────────────────────


def _parse_json_loose(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a model reply as JSON, tolerating fences and chatter.

    Returns ``(obj, True)`` on success, ``(None, False)`` when the model did
    not produce a JSON object — the caller must say so instead of inventing
    structure.
    """
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    if not cleaned:
        return None, False
    for candidate in (cleaned, cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj, True
    return None, False


def extract(
    context: Any,
    data: bytes,
    prompt: str = "",
    *,
    cache: dict[str, dict[str, Any]] | None = None,
    source: str = "bytes",
) -> dict[str, Any]:
    """Chat-ready structured extraction: summary + key text + elements.

    Unlike :func:`describe` (a free-form dump) and :func:`read_text` (a raw
    OCR transcript), this returns a structured result the agent can paste
    straight into chat — see :func:`format_chat_summary`.

    The vision model is **required**: when no vision-capable provider is
    configured this raises :class:`ModelError` with a plain explanation
    instead of returning an empty "success". The model is asked for a JSON
    object; when it does not produce one the result keeps the raw text as
    the summary and says ``parsed: False`` instead of inventing structure.
    """
    router = getattr(context, "router", None) if context is not None else None
    if router is None:
        raise ModelError(
            "vision extraction is unavailable: no vision-capable model "
            "provider is configured on this context (no context.router). "
            "Describe what you need differently, or configure a vision model."
        )
    instruction = _EXTRACT_PROMPT.replace(
        "{extra}",
        _EXTRACT_EXTRA.format(prompt=prompt) if (prompt or "").strip() else "",
    )
    result = _vision_call(
        context, data, instruction, action="extract", source=source,
        cache=cache, cache_extra=instruction, strict=True,
    )
    raw = result.get("description", "")
    parsed, parsed_ok = _parse_json_loose(raw)

    base = {k: v for k, v in result.items() if k != "description"}
    structured: dict[str, Any] = {
        **base,
        "available": True,
        "parsed": parsed_ok,
        "raw_text": raw,
    }
    if parsed_ok and parsed is not None:
        elements = parsed.get("notable_elements") or []
        structured.update(
            summary=str(parsed.get("summary") or "").strip(),
            key_text=str(parsed.get("key_text") or "").strip(),
            notable_elements=[str(e).strip() for e in elements
                              if str(e).strip()][:8],
            scene_type=str(parsed.get("scene_type") or "other").strip(),
        )
    else:
        # honest degradation: raw text as the summary, nothing invented
        structured.update(
            summary=raw.strip()[:600],
            key_text="",
            notable_elements=[],
            scene_type="unknown",
            note=("the model did not return structured JSON — the raw reply "
                  "is kept as the summary; structure was not invented"),
        )
    if not structured["summary"]:
        structured["summary"] = "(the model returned no usable description)"
    return structured


def format_chat_summary(result: dict[str, Any]) -> str:
    """Render an :func:`extract` result as a compact chat-friendly message.

    Output shape::

        🖼️ <summary>
        • <element>
        • <element>
        📝 Text in image: <key text, truncated>
        <format> <W>x<H> · via <provider>/<model> · <seconds>s

    Sections with no content are omitted — the message never carries empty
    headers.
    """
    lines = [f"🖼️ {(result.get('summary') or '').strip()}"]
    for element in result.get("notable_elements") or []:
        element = str(element).strip()
        if element:
            lines.append(f"• {element}")
    key_text = str(result.get("key_text") or "").strip()
    if key_text:
        shown = key_text if len(key_text) <= 400 else key_text[:397] + "…"
        lines.append(f"📝 Text in image: {shown}")
    meta_bits: list[str] = []
    fmt = result.get("format")
    width, height = result.get("width"), result.get("height")
    if fmt:
        meta_bits.append(f"{fmt} {width}x{height}" if width and height else str(fmt))
    provider = result.get("provider") or ""
    model = result.get("model") or ""
    if provider or model:
        meta_bits.append(f"via {provider}/{model}".rstrip("/"))
    seconds = result.get("seconds")
    if seconds is not None:
        meta_bits.append(f"{seconds}s")
    scene = result.get("scene_type")
    if scene and scene not in ("unknown", ""):
        meta_bits.append(str(scene))
    if meta_bits:
        lines.append(" · ".join(meta_bits))
    return "\n".join(line for line in lines if line.strip())


def _vision_call(
    context: Any,
    data: bytes,
    instruction: str,
    *,
    action: str,
    source: str,
    cache: dict[str, dict[str, Any]] | None,
    cache_extra: str = "",
    strict: bool = False,
) -> dict[str, Any]:
    """Shared send path: guards → downscale → route → audit → result."""
    _check_size(data, context)
    vision = _vision_settings(context)
    max_dim = _knob(vision, "max_dimension", DEFAULT_MAX_DIMENSION)
    metadata = image_metadata(data)
    key = f"{action}:{metadata['sha256']}:{hashlib.sha256(cache_extra.encode()).hexdigest()[:16]}"
    if cache is not None and key in cache:
        return {**cache[key], "cached": True}

    payload, downscale_note = _downscale(data, max_dim)
    started = time.perf_counter()
    router = getattr(context, "router", None) if context is not None else None
    description, provider, model, usage, error = "", "", "", {}, ""
    if router is not None:
        try:
            response = router.describe_image(payload, instruction)
            description, provider, model = response.text, response.provider, response.model
            usage_raw = response.usage.as_dict
            usage = dict(usage_raw() if callable(usage_raw) else usage_raw)
        except Exception as exc:  # noqa: BLE001 - vision is optional, metadata is not
            if strict:
                # failed attempts are still privacy-relevant: audit, then raise
                _audit(action, source, len(data), 0, "", "", {}, context)
                raise
            error = f"[vision unavailable: {type(exc).__name__}: {exc}]"
            description = error
    else:
        error = "[vision unavailable: no router]"
        description = error
        if strict:
            raise ModelError("vision called without a router on the context")

    _audit(action, source, len(data), len(payload), provider, model, usage, context)
    result: dict[str, Any] = {
        **metadata,
        "description": description,
        "provider": provider,
        "model": model,
        "tokens_used": usage,
        "prompt": instruction,
        "source": source,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if downscale_note:
        result["downscale_note"] = downscale_note
    if cache is not None and not error:
        cache[key] = result
    return result


def describe(
    context: Any,
    data: bytes,
    prompt: str = "",
    *,
    cache: dict[str, dict[str, Any]] | None = None,
    strict: bool = False,
    source: str = "bytes",
) -> dict[str, Any]:
    """Describe an image with the best available vision provider."""
    instruction = prompt or (
        "Describe this image thoroughly: layout, text, objects, and anything notable. "
        "Transcribe any visible text verbatim."
    )
    result = _vision_call(
        context, data, instruction, action="describe", source=source,
        cache=cache, cache_extra=instruction, strict=strict,
    )
    return result


def _native_ocr_text(data: bytes, context: Any) -> str:
    """Transcribe with the tesseract binary directly — no router needed.

    This is Devon's own OCR floor: deterministic, offline, verbatim. Raises
    :class:`NativeUnavailable` (via the shared resolver) when the binary is
    missing, so callers can chain to the next strategy honestly.
    """
    from ..llm.providers.ocr import ocr_binary, ocr_bytes
    from ..vision.native import NativeUnavailable

    vision = _vision_settings(context)
    explicit = str(_knob(vision, "ocr_binary", "") or "")
    language = str(_knob(vision, "ocr_language", "eng") or "eng")
    if not ocr_binary(explicit):
        raise NativeUnavailable(
            "native OCR needs the tesseract binary (Termux: "
            "pkg install tesseract; or set NM_OCR_BINARY)"
        )
    return ocr_bytes(data, language=language, binary=explicit)


def read_text(
    context: Any,
    data: bytes,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
    strict: bool = False,
    source: str = "bytes",
    strategy: str = "auto",
) -> dict[str, Any]:
    """Transcribe text from an image — strategy chain, native first.

    Strategies:
    - ``"auto"`` (default): native tesseract OCR when the binary is present
      (deterministic, offline, verbatim) → vision model otherwise.
    - ``"native"``: tesseract only — raises a clear error when unavailable.
    - ``"model"``: vision model only — the previous behavior.

    The result always names its ``method`` (``"native-ocr"`` or
    ``"vlm"``/``"vlm-ocr-floor"``). Never silently "fixes" garbled text —
    the model prompt demands verbatim output with [illegible] markers, and
    low-confidence regions are flagged, not invented.
    """
    from ..vision.native import NativeUnavailable

    strategy = (strategy or "auto").lower()
    if strategy not in ("auto", "native", "model"):
        raise ToolError(
            f"unknown read_text strategy {strategy!r} "
            "(want auto | native | model)"
        )
    _check_size(data, context)

    if strategy in ("auto", "native"):
        try:
            text = _native_ocr_text(data, context)
        except NativeUnavailable:
            if strategy == "native":
                raise
            _log.info("native OCR unavailable; chaining to vision model")
        else:
            from ..vision.seer import UNTRUSTED_VISION_PREFIX

            _audit("read_text", source, len(data), len(data),
                   "native", "tesseract", {}, context)
            return {
                "text": UNTRUSTED_VISION_PREFIX + text,
                "confidence_note": (
                    "native tesseract transcription of the raw pixels — "
                    "deterministic and offline; verify proper nouns and "
                    "numbers against the image"
                ),
                "method": "native-ocr",
                "provider": "native",
                "model": "tesseract",
                "source": source,
                **image_metadata(data),
            }

    result = _vision_call(
        context, data, _READ_TEXT_PROMPT, action="read_text", source=source,
        cache=cache, strict=strict,
    )
    text = result.pop("description")
    provider = result.get("provider", "")
    if provider == "ocr":
        method = "vlm-ocr-floor"
        confidence_note = (
            "tesseract transcription of the raw pixels via the router's OCR "
            "floor — verify proper nouns and numbers against the image"
        )
    else:
        method = "vlm"
        confidence_note = (
            "model transcription — regions marked [illegible] are uncertain; "
            "the model was instructed not to guess"
        )
    return {
        "text": text,
        "confidence_note": confidence_note,
        "method": method,
        **result,
    }


def locate(
    context: Any,
    data: bytes,
    target: str,
    *,
    strict: bool = False,
    source: str = "bytes",
    template: bytes | None = None,
) -> dict[str, Any]:
    """Find ``target`` ("the submit button") → 0-1000 bbox + confidence.

    Strategy chain:
    - ``template`` given (bytes of the exact icon/button to find): Devon's
      own normalized cross-correlation match — deterministic, no model,
      honest ``found: False`` below threshold instead of a guessed box.
    - otherwise: the vision model guesses a region from the word
      description — coordinates are approximate and the result says so.

    Any future click automation must re-verify on screen.
    """
    if not (target or "").strip():
        raise ToolError("vision locate needs a target description")
    if template is not None:
        from ..vision.native import template_locate

        _check_size(data, context)
        _check_size(template, context)
        found = template_locate(data, template)
        _audit("locate", source, len(data), len(data),
               "native", "template-match-ncc", {}, context)
        return {"target": target, **found}
    raw = _vision_call(
        context, data, _LOCATE_PROMPT.format(target=target), action="locate",
        source=source, cache=None, strict=strict,
    )
    box = _parse_box(raw.get("description", ""))
    base = {k: raw[k] for k in ("provider", "model", "tokens_used", "source", "seconds")
            if k in raw}
    if box is None:
        return {
            **base, "target": target, "found": False,
            "method": "vlm-guess",
            "disclaimer": _LOCATE_DISCLAIMER,
            "note": "model did not return a parseable region",
        }
    if not box.get("found", True):
        return {**base, "target": target, "found": False,
                "method": "vlm-guess",
                "disclaimer": _LOCATE_DISCLAIMER}
    for key in ("x", "y", "w", "h"):
        box[key] = max(0, min(1000, int(box.get(key, 0))))
    box["confidence"] = max(0.0, min(1.0, float(box.get("confidence", 0.5))))
    return {
        **base, "target": target, "found": True,
        "x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"],
        "confidence": box["confidence"],
        "method": "vlm-guess",
        "approximate": True,
        "disclaimer": _LOCATE_DISCLAIMER,
    }


def _parse_box(text: str) -> dict[str, Any] | None:
    """Pull the first {...} JSON object out of model chatter."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def _native_diff_summary(native: dict[str, Any]) -> str:
    """Honest one-line summary of a native pixel diff (no model)."""
    if native.get("identical"):
        return "native diff: the two images are pixel-identical"
    frac = native.get("changed_fraction") or 0.0
    bbox = native.get("changed_bbox_1000") or {}
    where = (f" around x={bbox.get('x')},y={bbox.get('y')} "
             f"(w={bbox.get('w')},h={bbox.get('h')}, 0-1000)") if bbox else ""
    return (f"native diff: {frac * 100:.1f}% of pixels changed"
            f" (mean abs diff {native.get('mean_abs_diff')}/255){where}")


def compare(
    context: Any,
    data_a: bytes,
    data_b: bytes,
    prompt: str = "",
    *,
    strict: bool = False,
    source: str = "bytes",
    semantic: bool = True,
) -> dict[str, Any]:
    """Spot the difference between two images (before/after, deploy check).

    Strategy chain: Devon's own pixel diff ALWAYS runs when Pillow is
    present (deterministic, offline — changed fraction + changed region),
    and the vision model adds the *semantic* reading ("what changed
    meaningfully") when a router is configured and ``semantic`` is true.
    With no router the native diff is the whole answer — an honest
    measurement, never an error string dressed as a description.
    """
    _check_size(data_a, context)
    _check_size(data_b, context)
    vision = _vision_settings(context)
    max_dim = _knob(vision, "max_dimension", DEFAULT_MAX_DIMENSION)

    # native diff first — works fully offline
    native: dict[str, Any] | None = None
    try:
        from ..vision.native import compare_native

        native = compare_native(data_a, data_b)
        _audit("compare", source, len(data_a) + len(data_b),
               len(data_a) + len(data_b), "native", "pixel-diff", {}, context)
    except Exception as exc:  # noqa: BLE001 - native is best-effort here
        _log.debug("native compare unavailable: %s", exc)

    router = getattr(context, "router", None) if context is not None else None
    if router is None or not semantic:
        if native is None:
            raise ToolError(
                "compare needs Pillow for the native diff or a vision-capable "
                "model for the semantic reading — neither is available"
            )
        return {
            "description": _native_diff_summary(native),
            "native": native,
            "method": "native",
            "semantic": False,
            "note": ("no vision model configured — this is Devon's own "
                     "pixel-level diff; semantic interpretation needs a model"),
            "source": source,
        }

    instruction = _COMPARE_PROMPT.format(extra=prompt or "Spot the differences.")
    downscaled_a, _note_a = _downscale(data_a, max_dim)
    downscaled_b, _note_b = _downscale(data_b, max_dim)

    composite = _side_by_side(downscaled_a, downscaled_b)
    if composite is None:
        # no PIL: describe each separately and present both
        desc_a = _vision_call(
            context, downscaled_a, "Describe this image thoroughly.",
            action="compare", source=source, cache=None, strict=strict,
        )["description"]
        desc_b = _vision_call(
            context, downscaled_b, "Describe this image thoroughly.",
            action="compare", source=source, cache=None, strict=strict,
        )["description"]
        result = {
            "description": "Image A:\n" + desc_a + "\n\nImage B:\n" + desc_b,
            "note": ("PIL unavailable — described separately instead of a "
                     "single side-by-side view"),
            "method": "separate-descriptions",
            "source": source,
        }
        if native is not None:
            result["native"] = native
        return result

    result = _vision_call(
        context, composite, instruction, action="compare", source=source,
        cache=None, strict=strict,
    )
    result["method"] = "side-by-side"
    result["semantic"] = True
    if native is not None:
        result["native"] = native
    return result


def _side_by_side(data_a: bytes, data_b: bytes) -> bytes | None:
    """Stack two images horizontally for one compare call. None without PIL."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    import io

    try:
        a = ImageOps.exif_transpose(Image.open(io.BytesIO(data_a)))
        b = ImageOps.exif_transpose(Image.open(io.BytesIO(data_b)))
    except Exception:  # noqa: BLE001
        return None

    def _fit(img: Any, height: int) -> Any:
        w, h = img.size
        w2 = max(1, round(w * height / h))
        return img.resize((w2, height), Image.LANCZOS)

    height = min(a.size[1], b.size[1])
    a, b = _fit(a, height), _fit(b, height)
    canvas = Image.new("RGB", (a.size[0] + b.size[0] + 8, height), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.size[0] + 8, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ── native analysis (no model, no network) ───────────────────────────────────


def analyze(
    context: Any,
    data: bytes,
    *,
    source: str = "bytes",
) -> dict[str, Any]:
    """Deep native image analysis — Devon's own eyes, no model involved.

    One call returns EXIF forensics, dominant colors, brightness/contrast/
    saturation, sharpness, entropy, perceptual hashes, plus best-effort
    face detection and QR decode (each reported with its method or a clear
    reason it is unavailable on this machine). Images never leave the
    machine here, so there is no provider/model in the result.
    """
    from ..vision.native import analyze as _analyze

    _check_size(data, context)
    started = time.perf_counter()
    report = _analyze(data)
    _audit("analyze", source, len(data), 0, "native", "analyze", {}, context)
    return {
        **report,
        "source": source,
        "seconds": round(time.perf_counter() - started, 3),
    }


def layout(
    context: Any,
    data: bytes,
    *,
    source: str = "bytes",
) -> dict[str, Any]:
    """Document layout analysis — tesseract TSV, no model.

    Returns blocks/lines/words with 0-1000 normalized coordinates and word
    confidence. Raises :class:`NativeUnavailable` with an install hint when
    the tesseract binary is missing.
    """
    from ..vision.native import document_layout as _document_layout

    _check_size(data, context)
    vision = _vision_settings(context)
    language = str(_knob(vision, "ocr_language", "eng") or "eng")
    started = time.perf_counter()
    result = _document_layout(data, language=language)
    _audit("layout", source, len(data), 0, "native", "tesseract-tsv", {}, context)
    return {**result, "source": source,
            "seconds": round(time.perf_counter() - started, 3)}


def vision_capabilities(context: Any = None) -> dict[str, Any]:
    """Honest capability report: what Devon can do with images HERE.

    ``native`` lists every model-free analysis with per-machine
    availability; ``needs_model`` names what genuinely still requires a
    vision-capable model and why. ``router_vision`` says whether a model
    path is currently configured.
    """
    from ..vision.native import capabilities as _capabilities

    report = _capabilities()
    router = getattr(context, "router", None) if context is not None else None
    router_vision = False
    if router is not None:
        try:
            chain = router.providers() if hasattr(router, "providers") else []
            router_vision = any("vision" in getattr(p, "capabilities", set())
                                for p in chain)
        except Exception:  # noqa: BLE001 - capability probe is best-effort
            router_vision = False
    report["router_vision"] = router_vision
    return report


# ── screenshots (privileged) ────────────────────────────────────────────────


def _screenshot_capture(display: int = 0) -> bytes:
    """Grab the local display. Raises ToolError when capture is unavailable."""
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise ToolError(
            "screenshot needs Pillow (pip install pillow); no other "
            "capture backend is bundled"
        ) from exc
    import io

    try:
        img = ImageGrab.grab(all_screens=True) if display < 0 else ImageGrab.grab()
    except Exception as exc:  # noqa: BLE001 - headless, Wayland, permissions…
        raise ToolError(f"screenshot capture failed: {exc}") from exc
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── registration ─────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """Attach the vision tools to a registry."""
    context = registry.context
    cache: dict[str, dict[str, Any]] = {}

    def _load(path: str = "", url: str = "",
              reference: str = "") -> tuple[bytes, str]:
        return resolve_image(context, path=path, url=url, reference=reference)

    @registry.register(
        "vision_describe",
        description=(
            "Describe an image using a vision model. One source: path= "
            "(workspace-relative), url=, or reference= (inbox:<id> / "
            "room:<slug>:<path> / attachment:<n> for the current message's "
            "attached image)."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_describe(path: str = "", url: str = "", prompt: str = "",
                        reference: str = "") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        return describe(context, data, prompt, cache=cache, source=source)

    @registry.register(
        "vision_read_text",
        description=(
            "Transcribe text from an image (OCR-style). Native-first: uses "
            "Devon's own tesseract OCR when available (deterministic, "
            "offline), vision model otherwise. strategy= auto|native|model. "
            "Verbatim output; illegible regions are flagged, never invented. "
            "Same sources as vision_describe."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_read_text(path: str = "", url: str = "",
                         reference: str = "",
                         strategy: str = "auto") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        return read_text(context, data, cache=cache, source=source,
                         strategy=strategy)

    @registry.register(
        "vision_locate",
        description=(
            "Locate a UI element or object ('the submit button') in an image. "
            "Give template_path= with the EXACT icon/button image for "
            "Devon's own deterministic template match (no model, honest "
            "not-found); otherwise the vision model guesses the region. "
            "Returns 0-1000 normalized coordinates — re-verify on screen "
            "before any click automation. Same sources as vision_describe."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_locate(path: str = "", url: str = "", target: str = "",
                      reference: str = "",
                      template_path: str = "") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        template = None
        if template_path:
            template, _ = _load(template_path, "")
        return locate(context, data, target, source=source, template=template)

    @registry.register(
        "vision_compare",
        description=(
            "Spot the difference between two images (before/after, 'did the "
            "deploy change the UI'). Devon's own pixel diff always runs "
            "(changed fraction + region, works offline); the vision model "
            "adds the semantic reading when configured. Give path_a=/url_a= "
            "and path_b=/url_b=."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_compare(path_a: str = "", path_b: str = "", url_a: str = "",
                       url_b: str = "", prompt: str = "",
                       reference_a: str = "", reference_b: str = ""
                       ) -> dict[str, Any]:
        data_a, _ = _load(path_a, url_a, reference=reference_a)
        data_b, _ = _load(path_b, url_b, reference=reference_b)
        return compare(context, data_a, data_b, prompt, source="compare")

    @registry.register(
        "vision_analyze",
        description=(
            "Deep NATIVE image analysis — no model, no network, no key: EXIF "
            "forensics, dominant colors, brightness/contrast/saturation, "
            "sharpness, entropy, perceptual hashes, face detection and QR "
            "decode (each labeled with its method or why it's unavailable "
            "here). Same sources as vision_describe."
        ),
        capability=Capability.FS_READ,
    )
    def vision_analyze(path: str = "", url: str = "",
                       reference: str = "") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        return analyze(context, data, source=source)

    @registry.register(
        "vision_capabilities",
        description=(
            "Honest report of what Devon can do with images on THIS machine: "
            "every native (offline, model-free) analysis with per-machine "
            "availability, plus what genuinely still needs a vision model "
            "and why."
        ),
        capability=Capability.FS_READ,
    )
    def vision_capabilities_tool() -> dict[str, Any]:
        return vision_capabilities(context)

    @registry.register(
        "vision_layout",
        description=(
            "Document layout analysis via tesseract (no model): text blocks, "
            "lines and words with 0-1000 normalized coordinates and word "
            "confidence. Needs the tesseract binary. Same sources as "
            "vision_describe."
        ),
        capability=Capability.FS_READ,
    )
    def vision_layout(path: str = "", url: str = "",
                      reference: str = "") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        return layout(context, data, source=source)

    @registry.register(
        "vision_extract",
        description=(
            "Structured, chat-ready image extraction: returns a summary, the "
            "key visible text, and a list of notable elements (not a raw OCR "
            "dump), plus a chat-friendly rendering. One source: path= "
            "(workspace-relative), url=, or reference= (inbox:<id> / "
            "room:<slug>:<path> / attachment:<n>). Raises a clear error when "
            "no vision-capable model is configured — never an empty success."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_extract(path: str = "", url: str = "", prompt: str = "",
                       reference: str = "") -> dict[str, Any]:
        data, source = _load(path, url, reference=reference)
        result = extract(context, data, prompt, cache=cache, source=source)
        result["chat_summary"] = format_chat_summary(result)
        return result

    @registry.register(
        "vision_screenshot",
        description=(
            "PRIVILEGED: capture the local display where Devon runs. Requires "
            "settings.partner.vision.allow_screenshot=true AND confirm=True "
            "per call. Never run on a schedule or in the background — "
            "screenshots can expose private on-screen content."
        ),
        capability=Capability.MODEL_CALL,
    )
    def vision_screenshot(display: int = 0, confirm: bool = False,
                          prompt: str = "") -> dict[str, Any]:
        vision = _vision_settings(context)
        if not _knob(vision, "allow_screenshot", False):
            raise ToolError(
                "screenshots are disabled: set "
                "settings.partner.vision.allow_screenshot=true first"
            )
        if not confirm:
            raise ToolError(
                "screenshot needs explicit per-call confirmation "
                "(confirm=True) — it captures whatever is on screen"
            )
        data = _screenshot_capture(display)
        return describe(context, data, prompt, cache=cache, source="screenshot")

    @registry.register(
        "vision_metadata",
        description="Read image format, dimensions, and size without calling a model.",
        capability=Capability.FS_READ,
    )
    def vision_metadata(path: str = "", url: str = "",
                        reference: str = "") -> dict[str, Any]:
        data, _ = _load(path, url, reference=reference)
        return image_metadata(data)

    @registry.register(
        "vision_to_base64",
        description="Encode an image as a data URL, for passing to an API by hand.",
        capability=Capability.FS_READ,
    )
    def vision_to_base64(path: str = "", url: str = "", mime: str = "image/png",
                         reference: str = "") -> dict[str, Any]:
        data, _ = _load(path, url, reference=reference)
        encoded = base64.b64encode(data).decode("ascii")
        return {"mime": mime, "bytes": len(data),
                "data_url": f"data:{mime};base64,{encoded}"}


# ── inbox contract (Prompt 09 §2) ─────────────────────────────────────────────


def make_inbox_vision_hook(context: Any) -> Any:
    """Build the ``(data, action, prompt) -> dict`` callable the inbox expects.

    This is the ``image`` intent contract from the Prompt 09 spec: the inbox
    stays model-free by default and only calls vision through this injected
    hook. Strict mode — failures raise so the handler can park the item
    honestly instead of inventing a description.
    """
    def hook(data: bytes, action: str, prompt: str) -> dict[str, Any]:
        if action == "read_text":
            return read_text(context, data, strict=True, source="inbox")
        if action == "locate":
            return locate(context, data, prompt, strict=True, source="inbox")
        return describe(context, data, prompt, strict=True, source="inbox")

    return hook
