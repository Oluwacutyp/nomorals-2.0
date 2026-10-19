"""Human-readable rendering for tool outputs and tool errors.

The rule this module enforces at the chat boundary:

* **No raw dicts / reprs to chat.** A tool that returns ``{"a": 1}`` must
  reach the user as ``a: 1`` lines, never ``{'a': 1}`` or a JSON blob.
* **No truncated JSON.** Values are rendered structurally and capped by
  *lines/items*, with an explicit ``… (N more)`` tail — never a mid-blob
  ``[:2500]`` cut that leaves dangling ``{"a": …``.
* **No empty errors.** ``f"failed ({exc})"`` renders ``failed ()`` when
  the exception carries no message (``Exception()``, bare ``OSError``,
  …). :func:`describe_exc` always yields a non-empty cause: the message,
  else the exception type name, else ``"unknown error"``.
* **No bare generic errors.** :func:`format_tool_error` always carries
  the real cause (status + message, like the STT fix), never just
  ``"failed"`` / ``"something went wrong"``.

Devon-only shared helper: the partner tool loop has its own renderer and
is untouched. This module never raises — worst case it returns a plain
fallback string.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "describe_exc",
    "outcome_message",
    "render_tool_value",
    "format_tool_error",
    "render_mapping",
]

#: Cap on rendered lines for one tool value (structured cap, not a cut).
_MAX_LINES = 60
#: Cap on list items rendered before the "… (N more)" tail.
_MAX_ITEMS = 40
#: Cap on a single rendered scalar line (whole-line clip + marker).
_MAX_LINE_CHARS = 500
#: Cap on nested depth — deeper values collapse to "…".
_MAX_DEPTH = 4


def describe_exc(exc: BaseException | Any) -> str:
    """Non-empty human cause for an exception or error value.

    Prefers ``.message`` (framework errors), then ``str(exc)``; when both
    are blank, falls back to the type name so the caller never renders
    ``()``. Single-line, so it is safe inside ``f"… ({describe_exc(e)})"``.
    """
    if exc is None:
        return "unknown error"
    msg = ""
    if isinstance(exc, BaseException):
        msg = str(getattr(exc, "message", "") or "") or str(exc)
    else:
        msg = str(getattr(exc, "message", "") or "") or str(exc)
    msg = " ".join(str(msg).split())
    if msg:
        return msg
    name = type(exc).__name__
    return name if name and name != "Exception" else "unknown error"


def outcome_message(error: Any) -> str:
    """Non-empty message out of a registry ``Outcome.error``.

    Handles framework errors (``.message``), plain exceptions, and plain
    strings — all of which flow through ``getattr(err, 'message', err)``
    call sites today, where an empty message silently became ``""``.
    """
    if error is None:
        return "unknown error"
    if isinstance(error, str):
        return error.strip() or "unknown error"
    return describe_exc(error)


def format_tool_error(tool: str, exc_or_msg: BaseException | Any) -> str:
    """``"<tool> failed: <real cause>"`` — never bare, never empty."""
    cause = outcome_message(exc_or_msg) if isinstance(exc_or_msg, str) \
        else describe_exc(exc_or_msg)
    return f"{tool} failed: {cause}"


def _scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        # Never collapse a nonzero number to zero: values outside the
        # "human" band keep their full repr (1e-06, not 0.00 → "0").
        # NaN/inf also survive via repr instead of a format crash.
        if value == 0.0:
            return "0"
        if abs(value) < 0.01 or abs(value) >= 1e15 or value != value \
                or value in (float("inf"), float("-inf")):
            return repr(value)
        return f"{value:.2f}".rstrip("0").rstrip(".")
    text = str(value)
    text = " ".join(text.split())
    if len(text) > _MAX_LINE_CHARS:
        return text[:_MAX_LINE_CHARS] + f"… (+{len(text) - _MAX_LINE_CHARS} more chars)"
    return text or "—"


def _render(value: Any, *, depth: int, lines: list[str], indent: str) -> None:
    """Append rendered lines for ``value`` (bounded, never raises)."""
    if len(lines) >= _MAX_LINES:
        return
    if depth > _MAX_DEPTH:
        lines.append(f"{indent}…")
        return
    if isinstance(value, dict):
        if not value:
            lines.append(f"{indent}(empty)")
            return
        shown = 0
        for key, item in value.items():
            if len(lines) >= _MAX_LINES:
                lines.append(f"{indent}… ({len(value) - shown} more fields)")
                return
            if isinstance(item, (dict, list, tuple)) and item:
                lines.append(f"{indent}{key}:")
                _render(item, depth=depth + 1, lines=lines, indent=indent + "  ")
            else:
                lines.append(f"{indent}{key}: {_scalar(item)}")
            shown += 1
        return
    if isinstance(value, (list, tuple)):
        if not value:
            lines.append(f"{indent}(none)")
            return
        for i, item in enumerate(value):
            if i >= _MAX_ITEMS:
                lines.append(f"{indent}… ({len(value) - _MAX_ITEMS} more)")
                return
            if len(lines) >= _MAX_LINES:
                lines.append(f"{indent}… ({len(value) - i} more)")
                return
            if isinstance(item, (dict, list, tuple)) and item:
                lines.append(f"{indent}•")
                _render(item, depth=depth + 1, lines=lines, indent=indent + "  ")
            else:
                lines.append(f"{indent}• {_scalar(item)}")
        return
    lines.append(f"{indent}{_scalar(value)}")


def render_tool_value(value: Any, *, max_lines: int = _MAX_LINES) -> str:
    """Render a tool result for chat: ``key: value`` lines, never JSON.

    * ``str`` → the string itself, clipped at a whole line with an
      explicit ``… (+N more lines)`` tail (never a mid-JSON cut).
    * ``dict`` → one ``key: value`` line per field, nested indented.
    * ``list`` → one ``• item`` line per entry.
    * ``None`` → ``"(no output)"``; other scalars → ``str(value)``.
    Never raises; never returns a raw ``{'a': 1}`` / ``{"a": 1}`` dump.
    """
    try:
        if value is None:
            return "(no output)"
        if isinstance(value, str):
            if not value.strip():
                return "(empty)"
            parts = value.splitlines()
            if len(parts) > max_lines:
                kept = parts[:max_lines]
                return "\n".join(kept) + f"\n… (+{len(parts) - max_lines} more lines)"
            return value
        lines: list[str] = []
        _render(value, depth=0, lines=lines, indent="")
        return "\n".join(lines) if lines else "(no output)"
    except Exception:  # noqa: BLE001 - renderer never breaks the chat
        try:
            return str(value)[:1000] or "(no output)"
        except Exception:  # noqa: BLE001
            return "(unprintable result)"


def render_mapping(mapping: dict[str, Any], *, title: str = "") -> str:
    """``render_tool_value`` for a flat mapping, with an optional title."""
    body = render_tool_value(mapping)
    return f"{title}\n{body}" if title else body


#: Progress callback type for long Devon runs:
#: ``(step_index_1_based, total_steps, tool_name, status)`` where status is
#: "start" | "parked" | "done" | "failed". "parked" means the step's tool
#: is running as a background job with heartbeats — the run moved on and
#: will collect (or already collected) the result. Callbacks must never
#: raise — Devon swallows callback errors so progress can never break
#: the run.
ProgressCallback = Callable[[int, int, str, str], None]


def emit_progress(cb: ProgressCallback | None, step: int, total: int,
                  tool: str, status: str) -> None:
    """Invoke a progress callback, swallowing callback errors."""
    if cb is None:
        return
    try:
        cb(int(step), int(total), str(tool), str(status))
    except Exception:  # noqa: BLE001 - progress never breaks the run
        pass
