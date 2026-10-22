"""Styled outputs (non-chat): command / tool / game / scheduled results.

The owner's correction (2026-10-17): style was built for conversational
replies, but it was MEANT for outputs — the results that are not normal
chats. This module is the output side of the style system:

* every output carries a header (emoji marker + title), a divider, and a
  structured body — never a raw dict / JSON blob;
* bodies render through :func:`core.tool_render.render_tool_value`
  (``key: value`` lines, never mid-JSON cuts, never empty errors);
* long outputs truncate at whole trailing sections first, then at a
  sentence boundary, with an explicit ``…`` marker.

Layering: this module lives in ``core/`` so the scheduler (``agents/``)
can use it. It therefore does NOT import ``partner.style`` (partner
depends on core, not the other way round). Output formatting here is
plain-text (no markdown/HTML) BY DESIGN — structured outputs read as
documents, not chat. (Since formatting pass F1 the Telegram send path
itself converts the house dialect and sets ``parse_mode`` at the
gateway, so this choice is now a style decision, not a workaround
for a transport that dropped markup.)

Conversational replies keep the persona's own voice passes
(``normalize_formatting`` / ``humanize_emoji`` / ``apply_texting_voice`` /
``clamp_to_budget``); they do NOT go through this module.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .tool_render import render_tool_value

__all__ = [
    "OUTPUT_KIND_MARKERS",
    "parse_json_payload",
    "render_output",
    "style_output_text",
]

#: Output kind -> emoji marker. Markers are purposeful section openers
#: (the mined vocabulary), not decoration.
OUTPUT_KIND_MARKERS: dict[str, str] = {
    "scheduled": "⏰",
    "memory": "🧠",
    "search": "🔍",
    "game": "🎮",
    "tool": "🔧",
    "status": "📊",
    "error": "❌",
    "music": "🎵",
    "command": "💬",
    "briefing": "📰",
    "job": "📋",
    "help": "❓",
}

#: Divider between an output's header and its body (mined divider).
_DIVIDER = "━━━━━━━━━━━━━━━"

#: Default hard cap for one styled output (Telegram's 4096, with room
#: for the notifier banner + adaptive buttons).
_DEFAULT_MAX_CHARS = 3900

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_payload(text: str) -> tuple[str | None, Any | None]:
    """Parse a ``key: {json}`` or bare ``{json}`` string, or return None.

    Scheduled job summaries arrive as ``memory: {"mode": "additive",
    ...}`` — a label plus a JSON blob. This splits the label from the
    blob and parses it, so the renderer can show structured lines
    instead of the raw blob. Returns ``(key, parsed)``; ``key`` is None
    for a bare JSON object. Returns ``(None, None)`` when the text is
    not JSON-shaped or fails to parse — the caller falls back to the
    raw string, never to an error.
    """
    if not text:
        return None, None
    t = text.strip()
    # "key: {json}" — label, colon, then a JSON object to end of text.
    m = re.match(r"^([A-Za-z0-9_][\w .-]{0,40}):\s*(\{.*\})\s*$", t, re.DOTALL)
    candidate = None
    key = None
    if m:
        key, candidate = m.group(1), m.group(2)
    elif t.startswith("{"):
        candidate = t
    if candidate is None:
        return None, None
    if not _JSON_OBJ.match(candidate.strip()):
        return None, None
    try:
        return key, json.loads(candidate)
    except (ValueError, TypeError):
        return None, None


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cut at a sentence boundary at or under ``max_chars`` + ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if boundary > max_chars // 3:
        return cut[: boundary + 1].rstrip()
    space = cut.rfind(" ")
    if space > max_chars // 2:
        return cut[:space].rstrip() + "…"
    return cut + "…"


def _render_body(payload: Any) -> str:
    """Human-readable body for a payload — never a raw JSON dump."""
    if payload is None:
        return "(no output)"
    if isinstance(payload, str):
        if not payload.strip():
            return "(empty)"
        key, parsed = parse_json_payload(payload)
        if parsed is not None:
            rendered = render_tool_value(parsed)
            return f"{key}:\n{rendered}" if key else rendered
        return payload
    return render_tool_value(payload)


def render_output(
    *,
    kind: str,
    title: str,
    payload: Any,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Render one styled output: header + divider + structured body.

    ``kind`` picks the emoji marker (unknown kinds get ``📌``);
    ``title`` names the output (the tool / job / command that produced
    it); ``payload`` is the result (dict, list, scalar, or a string
    that may embed ``key: {json}``). The result is never a raw JSON
    blob, never ends mid-section, and never raises.
    """
    try:
        marker = OUTPUT_KIND_MARKERS.get((kind or "").lower(), "📌")
        header = f"{marker} {title}".strip()
        body = _render_body(payload)
        out = f"{header}\n{_DIVIDER}\n{body}" if body else header
        if len(out) > max_chars:
            # Drop whole trailing sections first (an output that ends
            # mid-section reads broken), then sentence-cut.
            sections = re.split(rf"(?m)(?=^{re.escape(_DIVIDER)}$)", out)
            kept: list[str] = []
            total = 0
            for sec in sections:
                if kept and total + len(sec) > max_chars:
                    break
                kept.append(sec)
                total += len(sec)
            candidate = "".join(kept).strip()
            if candidate and len(candidate) >= max_chars // 2:
                out = candidate
            if len(out) > max_chars:
                out = _truncate_at_sentence(out, max_chars)
        return out
    except Exception:  # noqa: BLE001 - rendering never breaks the output
        try:
            return str(payload)[:max_chars] or "(no output)"
        except Exception:  # noqa: BLE001
            return "(unprintable output)"


def style_output_text(
    text: str,
    *,
    kind: str,
    title: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Style an already-built string output (header + structure).

    For outputs whose body is already text (a command handler's reply,
    a scheduler summary): prepend the kind header + divider, parse out
    any embedded ``key: {json}`` blob into structured lines, and
    truncate cleanly. Plain text passes through with just the header.
    """
    try:
        if not (text or "").strip():
            return render_output(kind=kind, title=title, payload="(empty)",
                                 max_chars=max_chars)
        marker = OUTPUT_KIND_MARKERS.get((kind or "").lower(), "📌")
        header = f"{marker} {title}".strip()
        body = _render_body(text)
        # _render_body already expanded any JSON blob; avoid a double
        # header when the text was already styled by render_output.
        if body.startswith(header):
            out = body
        else:
            out = f"{header}\n{_DIVIDER}\n{body}"
        if len(out) > max_chars:
            out = _truncate_at_sentence(out, max_chars)
        return out
    except Exception:  # noqa: BLE001 - rendering never breaks the output
        return (text or "(empty)")[:max_chars]
