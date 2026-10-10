"""Media style spine — god-tier output formatting for the whole media stack.

Every human-facing media report (song sheets, DJ set plans, render
reports, highlight manifests, pipeline run cards) funnels through here
so the module has ONE visual language instead of twelve ad-hoc ones.

Themes:
* ``ninja`` (default) — electric-blue-on-dark energy: box banners,
  status glyphs, ASCII bars. Matches the owner's ninja "vrede peace"
  termux theme.
* ``plain`` — no decoration, still structured (logs, files).
* ``minimal`` — compact single-line friendly.

Nothing here is decoration-only: bars, journey maps and tables carry
real numbers. ``Theme.render_*`` helpers never raise — worst case they
return the plain-text fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "Theme", "THEMES", "theme", "banner", "card", "kv", "bar",
    "sparkbar", "journey_map", "status_line", "section",
]

# ── glyphs ───────────────────────────────────────────────────────────────────

_OK = "✓"
_FAIL = "✗"
_WARN = "!"
_ARROW = "→"
_BULLET = "•"


@dataclass
class Theme:
    """A named output theme."""
    name: str
    banner_char: str = "═"
    corner_tl: str = "╔"
    corner_tr: str = "╗"
    corner_bl: str = "╚"
    corner_br: str = "╝"
    edge: str = "║"
    bullet: str = _BULLET
    ok: str = _OK
    fail: str = _FAIL
    warn: str = _WARN
    arrow: str = _ARROW
    bar_full: str = "█"
    bar_empty: str = "░"
    pad: int = 1

    # ── primitives ──────────────────────────────────────────────────────
    def banner(self, title: str, subtitle: str = "",
               width: int = 56) -> str:
        """Full-width box banner: title (+ optional subtitle)."""
        try:
            inner = width - 2
            top = self.corner_tl + self.banner_char * inner + self.corner_tr
            bot = self.corner_bl + self.banner_char * inner + self.corner_br
            lines = [top]
            t = str(title)[:inner].center(inner)
            lines.append(f"{self.edge}{t}{self.edge}")
            if subtitle:
                s = str(subtitle)[:inner].center(inner)
                lines.append(f"{self.edge}{s}{self.edge}")
            lines.append(bot)
            return "\n".join(lines)
        except Exception:  # noqa: BLE001 — never break output
            return str(title)

    def section(self, title: str) -> str:
        """A section header line."""
        try:
            return f"\n── {title} " + "─" * max(2, 46 - len(title))
        except Exception:  # noqa: BLE001
            return str(title)

    def kv(self, pairs: Sequence[tuple[str, Any]],
           indent: str = "  ") -> str:
        """Aligned key: value table."""
        try:
            rows = [(str(k), str(v)) for k, v in pairs if v not in ("", None)]
            if not rows:
                return ""
            w = max(len(k) for k, _ in rows)
            return "\n".join(
                f"{indent}{k.ljust(w)} : {v}" for k, v in rows)
        except Exception:  # noqa: BLE001
            return ""

    def card(self, title: str, pairs: Sequence[tuple[str, Any]],
             footer: str = "") -> str:
        """Banner + kv table + optional footer: the standard report card."""
        parts = [self.banner(title)]
        body = self.kv(pairs)
        if body:
            parts.append(body)
        if footer:
            parts.append(f"  {footer}")
        return "\n".join(parts)

    def bar(self, frac: float, width: int = 20) -> str:
        """ASCII progress/level bar for a 0..1 fraction."""
        try:
            f = max(0.0, min(1.0, float(frac)))
            n = int(round(f * width))
            return self.bar_full * n + self.bar_empty * (width - n)
        except Exception:  # noqa: BLE001
            return ""

    def sparkbar(self, values: Sequence[float],
                 width: int = 24) -> str:
        """Mini bar chart of a value series (e.g. an energy curve)."""
        try:
            vals = [max(0.0, min(1.0, float(v))) for v in values]
            if not vals:
                return ""
            # downsample/upsample to width
            out = []
            for i in range(width):
                pos = i * (len(vals) - 1) / max(1, width - 1)
                lo, hi = int(pos), min(len(vals) - 1, int(pos) + 1)
                frac = pos - lo
                v = vals[lo] * (1 - frac) + vals[hi] * frac
                out.append(self.bar_full if v >= 0.75 else
                           "▓" if v >= 0.5 else "▒" if v >= 0.25 else "░")
            return "".join(out)
        except Exception:  # noqa: BLE001
            return ""

    def status_line(self, ok: bool | None, label: str,
                    detail: str = "") -> str:
        """One honest status line: ✓/✗/! + label + detail."""
        glyph = self.ok if ok is True else self.fail if ok is False else self.warn
        line = f"[{glyph}] {label}"
        if detail:
            line += f" — {detail}"
        return line

    def journey_map(self, stops: Sequence[str]) -> str:
        """ASCII route map: 8A → 9A → 10A → 8B."""
        try:
            return f"  {f' {self.arrow} '.join(str(s) for s in stops)}"
        except Exception:  # noqa: BLE001
            return " → ".join(str(s) for s in stops)


THEMES: dict[str, Theme] = {
    "ninja": Theme("ninja"),
    "plain": Theme("plain", banner_char="-", corner_tl="+", corner_tr="+",
                   corner_bl="+", corner_br="+", edge="|", bullet="-",
                   bar_full="#", bar_empty="-"),
    "minimal": Theme("minimal", banner_char=" ", corner_tl=" ", corner_tr=" ",
                     corner_bl=" ", corner_br=" ", edge=" ", bullet="-",
                     bar_full="=", bar_empty="."),
}

_current: Theme = THEMES["ninja"]


def theme(name: str = "") -> Theme:
    """Get (and optionally switch) the active theme. Never raises."""
    global _current
    if name and name in THEMES:
        _current = THEMES[name]
    return _current


# ── module-level shortcuts on the active theme ───────────────────────────────

def banner(title: str, subtitle: str = "", width: int = 56) -> str:
    return _current.banner(title, subtitle, width)


def section(title: str) -> str:
    return _current.section(title)


def kv(pairs: Sequence[tuple[str, Any]], indent: str = "  ") -> str:
    return _current.kv(pairs, indent)


def card(title: str, pairs: Sequence[tuple[str, Any]],
         footer: str = "") -> str:
    return _current.card(title, pairs, footer)


def bar(frac: float, width: int = 20) -> str:
    return _current.bar(frac, width)


def sparkbar(values: Sequence[float], width: int = 24) -> str:
    return _current.sparkbar(values, width)


def status_line(ok: bool | None, label: str, detail: str = "") -> str:
    return _current.status_line(ok, label, detail)


def journey_map(stops: Sequence[str]) -> str:
    return _current.journey_map(stops)
