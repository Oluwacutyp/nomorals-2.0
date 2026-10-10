"""Styled presentation for the connectors package.

God-tier output for connector surfaces: fleet tables, status panels,
health snapshots, capability sheets, and connect guides. Pure functions
returning strings — the ``nm`` CLI, chat renderers, and dashboards all
share them.

Design notes (mirroring ``nomorals.cmdline.style`` conventions without
importing it — connectors are L5, the CLI is L7, and the layering test
forbids the import):

* semantic styles only (success/error/warning/info/accent/dim);
* no ANSI when piped, when ``NO_COLOR``/``NM_NO_COLOR`` is set, or when
  ``TERM=dumb``;
* unicode icons on capable TTYs, ASCII fallbacks otherwise — never a bare
  emoji into a pipe.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Iterable, Sequence

__all__ = [
    "render_capabilities",
    "render_checkpoint_list",
    "render_connect_guide",
    "render_connector_table",
    "render_health_snapshot",
    "render_status",
]


# ── theme ─────────────────────────────────────────────────────────────

_ANSI: dict[str, str] = {
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "cyan": "36", "bright_black": "90",
    "bold": "1", "dim": "2",
}

_OK = ("✓", "[ok]")
_FAIL = ("✗", "[!!]")
_WARN = ("⚠", "[!]")
_INFO = ("ℹ", "[i]")
_DOT = ("•", "-")
_ARROW = ("→", "->")


def _color_enabled() -> bool:
    if os.environ.get("NM_NO_COLOR") or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


def _unicode_ok() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return enc.lower().replace("-", "") in ("utf8", "utf8sig")


def _st(text: str, *styles: str) -> str:
    if not styles or not _color_enabled():
        return text
    codes: list[str] = []
    for chunk in styles:
        for part in chunk.split():
            code = _ANSI.get(part, "")
            if code and code not in codes:
                codes.append(code)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _ic(pair: tuple[str, str]) -> str:
    uni, ascii_ = pair
    if not _color_enabled() or not _unicode_ok():
        return ascii_
    return uni


def _ok_dot(good: bool) -> str:
    return _st(_ic(_OK), "green") if good else _st(_ic(_FAIL), "red")


# ── table ─────────────────────────────────────────────────────────────

def _table(headers: Sequence[str],
           rows: Iterable[Sequence[Any]]) -> str:
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    max_width = shutil.get_terminal_size((100, 24)).columns
    total = sum(widths) + 2 * (len(widths) - 1)
    if total > max_width and widths:
        widths[-1] = max(8, widths[-1] - (total - max_width))

    def fit(text: str, width: int) -> str:
        return text.ljust(width) if len(text) <= width else (
            text[: max(0, width - 1)] + "…")

    lines = [
        _st("  ".join(fit(h, w) for h, w in zip(headers, widths)).rstrip(),
            "bold")
    ]
    for row in str_rows:
        padded = list(row) + [""] * (len(widths) - len(row))
        lines.append("  ".join(
            fit(c, w) for c, w in zip(padded, widths)).rstrip())
    return "\n".join(lines)


def _kv(mapping: dict[str, Any]) -> str:
    items = [(str(k), str(v)) for k, v in mapping.items() if v not in ("", None)]
    if not items:
        return ""
    width = max(len(k) for k, _ in items)
    return "\n".join(
        f"{_st(k.ljust(width), 'dim')}  {v}" for k, v in items)


# ── renderers ─────────────────────────────────────────────────────────

def render_connector_table(
    infos: Sequence[dict[str, Any]],
    statuses: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Fleet table: state dot, id, name, category, auth, account/detail.

    ``statuses`` optionally maps connector id → ``{"connected": bool,
    "account": str|None, "detail": str}`` (from ``health_snapshot``).
    """
    statuses = statuses or {}
    rows = []
    for info in infos:
        cid = str(info.get("id", ""))
        st = statuses.get(cid, {})
        if "connected" in st:
            state = _ok_dot(bool(st["connected"]))
            account = st.get("account") or st.get("detail") or ""
        else:
            state = _st(_ic(_DOT), "dim")
            account = ""
        auth = ",".join(info.get("auth_methods") or [])
        rows.append((
            state, cid, info.get("name", ""),
            info.get("category", "") or _st("-", "dim"),
            auth, account,
        ))
    body = _table(["", "id", "name", "category", "auth", "account / detail"],
                  rows)
    return f"{_st('Connectors', 'bold cyan')} ({len(rows)})\n{body}"


def render_status(name: str, status: Any) -> str:
    """One connector's live status as a titled kv panel."""
    connected = bool(getattr(status, "connected", False))
    head = f"{_ok_dot(connected)} {_st(name, 'bold cyan')}"
    state = "connected" if connected else "disconnected"
    panel = _kv({
        "state": _st(state, "green" if connected else "red"),
        "account": getattr(status, "account", None),
        "scopes": ", ".join(getattr(status, "scopes", None) or []),
        "detail": getattr(status, "detail", ""),
    })
    return f"{head}\n{panel}" if panel else head


def render_health_snapshot(snapshot: dict[str, Any]) -> str:
    """Fleet health rollup: counts, then failures, then the full roster."""
    total = snapshot.get("total", 0)
    connected = snapshot.get("connected", 0)
    details = snapshot.get("details", [])
    pct = (100.0 * connected / total) if total else 0.0
    head = (
        f"{_st('Connector health', 'bold cyan')}  "
        f"{_st(f'{connected}/{total}', 'green' if connected == total else 'yellow')} "
        f"connected ({pct:.0f}%)"
    )
    lines = [head]
    failures = [d for d in details if not d.get("connected")]
    if failures:
        lines.append("")
        lines.append(_st("Needs attention:", "bold"))
        for d in failures:
            lines.append(
                f"  {_ic(_FAIL)} {_st(d['id'], 'cyan')} — "
                f"{_st(d.get('detail', 'disconnected'), 'dim')}"
            )
    lines.append("")
    rows = [
        (_ok_dot(bool(d.get("connected"))), d["id"],
         d.get("account") or _st("—", "dim"),
         f"{d.get('latency_ms', 0):.0f}ms")
        for d in details
    ]
    lines.append(_table(["", "id", "account", "check"], rows))
    return "\n".join(lines)


def render_capabilities(manifest: dict[str, Any]) -> str:
    """Capability sheet for one connector: identity + feature inventory."""
    head = f"{_st(manifest.get('name', manifest.get('id', '')), 'bold cyan')}"
    sub = str(manifest.get("description", "") or "").strip()
    lines = [head]
    if sub:
        lines.append(_st(sub, "dim"))
    lines.append("")
    lines.append(_kv({
        "id": manifest.get("id", ""),
        "category": manifest.get("category", "") or "—",
        "auth": ", ".join(manifest.get("auth_methods", [])) or "—",
        "provisionable": ", ".join(manifest.get("provisionable", [])) or "—",
    }))
    features = manifest.get("features", [])
    if features:
        lines.append("")
        lines.append(_st("Features", "bold"))
        cols = 3
        col_w = max(len(f) for f in features) + 4
        for i in range(0, len(features), cols):
            chunk = features[i:i + cols]
            lines.append("  " + "".join(
                f"{_st(_ic(_DOT), 'cyan')} {f.ljust(col_w - 2)}"
                for f in chunk).rstrip())
    return "\n".join(lines)


def render_connect_guide(
    name: str,
    steps: Sequence[str],
    *,
    note: str = "",
) -> str:
    """Numbered connect guide: what the owner does, in order."""
    lines = [f"{_st('Connect', 'bold cyan')} {_st(name, 'bold')}"]
    for i, step in enumerate(steps, 1):
        lines.append(f"  {_st(str(i) + '.', 'cyan')} {step}")
    if note:
        lines.append("")
        lines.append(f"{_ic(_INFO)} {_st(note, 'dim')}")
    return "\n".join(lines)


def render_checkpoint_list(
    checkpoints: Sequence[Any],
    *,
    title_text: str = "Pending checkpoints",
) -> str:
    """Human-checkpoint queue with kind, connector, and age."""
    import time as _time
    items = list(checkpoints)
    lines = [f"{_st(title_text, 'bold cyan')} ({len(items)})"]
    if not items:
        lines.append(_st("  nothing waiting on a human", "dim"))
        return "\n".join(lines)
    for cp in items:
        kind = getattr(getattr(cp, "kind", ""), "value", str(getattr(cp, "kind", "")))
        created = float(getattr(cp, "created_at", 0) or 0)
        age_h = (_time.time() - created) / 3600 if created else 0
        age = f"{age_h:.1f}h old" if age_h >= 1 else f"{age_h * 60:.0f}m old"
        lines.append(
            f"  {_st(_ic(_WARN), 'yellow')} {_st(cp.id, 'cyan')} "
            f"[{kind}] {_st(str(getattr(cp, 'connector_id', '')), 'dim')} — "
            f"{cp.title} {_st('(' + age + ')', 'dim')}"
        )
    return "\n".join(lines)
