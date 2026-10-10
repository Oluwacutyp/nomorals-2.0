"""Console widgets: progress bars, sparklines, live screens, message cards.

All widgets are pure-stdlib ANSI. They emit nothing when color/tty is
unavailable (plain-text fallbacks), and never use black backgrounds or red.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .palette import (
    BOLD,
    BRIGHT_CYAN,
    BRIGHT_WHITE,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    STEALTH_AMBER,
    STEALTH_CYAN,
    STEALTH_TEXT,
    SUBTLE,
    TITLE,
    WARN,
    _BLOCK_FRACS,
    paint,
    strip_ansi,
    supports_color,
    truncate_visible,
    visible_width,
)

_SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Platform icons for the rich message display (emoji, not colored text).
PLATFORM_ICONS = {
    "telegram": "✈️",
    "telegram-bot": "🤖",
    "local": "💻",
    "discord": "🎮",
    "whatsapp": "💬",
}

_UP_ARROW = "\033[A"
_CLEAR_LINE = "\033[2K\r"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
# Alternate screen buffer: the watch dashboard gets its own screen,
# fully isolated from log lines on the main screen. On exit the main
# screen (and cursor) is restored exactly as it was — no bleed-through,
# no residual corruption. Termux-safe (standard xterm sequence).
_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"
_HOME = "\033[H"
_CLEAR_BELOW = "\033[J"


def sparkline(
    values: Iterable[float],
    *,
    width: int = 24,
    color: bool | None = None,
    min_label: bool = False,
    baseline: bool = False,
    warn_at: float | None = None,
    crit_at: float | None = None,
) -> str:
    """Tiny inline bar chart, e.g. ``▁▂▄▇█``. Empty input renders dim dashes.

    ``min_label=True`` appends ``min/max`` values. ``baseline=True``
    normalizes against the data's min (flat lines stay flat, like
    terminal-charts' ±1% floor). ``warn_at``/``crit_at`` paint values at
    or above the thresholds amber/magenta instead of cyan.
    """
    vals = [max(0.0, float(v)) for v in values]
    if not vals:
        return paint("─" * width, DIM, color=color)
    lo = min(vals)
    peak = max(vals) or 1.0
    span = (peak - lo) if baseline and peak > lo else peak
    span = span or 1.0
    n = len(_SPARK_CHARS) - 1
    chars = [
        _SPARK_CHARS[min(n, int((v - (lo if baseline else 0.0)) / span * n))]
        for v in vals
    ]
    # Resample to width.
    if len(chars) > width:
        step = len(chars) / width
        chars = [chars[int(i * step)] for i in range(width)]
    elif len(chars) < width:
        chars = chars + [_SPARK_CHARS[0]] * (width - len(chars))
    # Per-char threshold coloring.
    if warn_at is not None or crit_at is not None:
        sample = vals[:width] + [0.0] * max(0, width - len(vals))
        parts = []
        for ch, v in zip(chars, sample):
            if crit_at is not None and v >= crit_at:
                parts.append(paint(ch, WARN + BOLD, color=color))
            elif warn_at is not None and v >= warn_at:
                parts.append(paint(ch, WARN, color=color))
            else:
                parts.append(paint(ch, CYAN, color=color))
        body = "".join(parts)
    else:
        body = paint("".join(chars), CYAN, color=color)
    if min_label:
        body += paint(f" {lo:g}/{peak:g}", DIM, color=color)
    return body


def gauge(
    value: float,
    *,
    width: int = 18,
    color: bool | None = None,
    bar_color: str = GREEN,
    warn_at: float = 0.75,
    crit_at: float = 0.9,
    label: str = "",
) -> str:
    """btop-style meter with 1/8-cell sub-precision: ``████▍░░░░ 62%``.

    ``value`` is 0..1 (clamped). Colors shift to amber/magenta past the
    thresholds — severity by color AND the number, never color alone.
    """
    frac = max(0.0, min(1.0, float(value)))
    cells = frac * width
    full = int(cells)
    part = int(round((cells - full) * 8))
    if part == 8:
        full += 1
        part = 0
    bar = "█" * full
    if full < width:
        bar += _BLOCK_FRACS[part] if part else "░"
        bar += "░" * (width - full - 1)
    if frac >= crit_at:
        bc = MAGENTA + BOLD
    elif frac >= warn_at:
        bc = WARN
    else:
        bc = bar_color
    pct = f"{frac * 100:5.1f}%"
    head = f"{paint(label + ' ', CYAN, color=color)}" if label else ""
    return (
        f"{head}{paint(bar, bc, color=color)} "
        f"{paint(pct, BOLD if frac >= warn_at else SUBTLE, color=color)}"
    )


def columns_chart(
    values: Iterable[float],
    *,
    height: int = 6,
    width: int | None = None,
    color: bool | None = None,
    bar_color: str = CYAN,
    labels: list[str] | None = None,
) -> list[str]:
    """Vertical bar chart (chartli ``columns`` idiom), one string per row.

    Returns ``height`` rows plus an optional label row.
    """
    vals = [max(0.0, float(v)) for v in values]
    if not vals:
        return [paint("(no data)", DIM, color=color)]
    if width is not None and len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    peak = max(vals) or 1.0
    cols = [min(height, int(round(v / peak * height))) for v in vals]
    rows: list[str] = []
    for row in range(height, 0, -1):
        line = "".join(
            paint("█", bar_color, color=color)
            if c >= row
            else " "
            for c in cols
        )
        rows.append(line)
    if labels:
        lab = "".join(
            (str(labels[i])[:1] if i < len(labels) else " ") for i in range(len(vals))
        )
        rows.append(paint(lab, DIM, color=color))
    return rows


def braille_chart(
    values: Iterable[float],
    *,
    width: int = 36,
    height: int = 5,
    color: bool | None = None,
    line_color: str = CYAN,
    min_label: bool = True,
) -> list[str]:
    """Braille line chart (terminal-charts idiom): 2×4 dots per cell.

    Consecutive points are joined vertically so it reads as a line;
    the scale fits the data with headroom so a flat day looks flat.
    """
    vals = [float(v) for v in values]
    if not vals:
        return [paint("(no data)", DIM, color=color)]
    # Resample: average when too many points, interpolate when too few.
    steps = width * 2
    if len(vals) > steps:
        chunk = len(vals) / steps
        vals = [
            sum(vals[int(i * chunk): int((i + 1) * chunk)]) / max(1, int((i + 1) * chunk) - int(i * chunk))
            for i in range(steps)
        ]
    elif len(vals) < steps:
        out: list[float] = []
        for i in range(steps):
            pos = i / max(1, steps - 1) * (len(vals) - 1)
            lo_i = int(pos)
            hi_i = min(len(vals) - 1, lo_i + 1)
            f = pos - lo_i
            out.append(vals[lo_i] * (1 - f) + vals[hi_i] * f)
        vals = out
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = span * 0.15
    lo -= pad
    span += 2 * pad
    rows = height  # braille rows; each holds 4 dot-rows
    dot_rows = rows * 4

    def level(v: float) -> int:
        return max(0, min(dot_rows - 1, int((v - lo) / span * dot_rows)))

    lvls = [level(v) for v in vals]
    # Braille dot map: dots 1-8 → bit positions.
    lines: list[str] = []
    for r in range(rows):
        chars: list[str] = []
        for cx in range(width):
            pair = lvls[cx * 2: cx * 2 + 2]
            # Join consecutive points vertically (line, not dots).
            if len(pair) == 2:
                lo_l, hi_l = min(pair), max(pair)
            else:
                lo_l = hi_l = pair[0] if pair else 0
            dots = 0
            for dx, lvl in enumerate(pair):
                for dr in range(4):
                    dot_row = r * 4 + dr  # 0 = top
                    if lo_l <= (dot_rows - 1 - dot_row) <= hi_l or lvl == (dot_rows - 1 - dot_row):
                        bit = [0, 3, 1, 4, 2, 5, 6, 7][dx * 4 + dr]
                        dots |= 1 << bit
            # Also fill the vertical span between the pair (line join).
            for dr in range(4):
                dot_row = r * 4 + dr
                lvl_here = dot_rows - 1 - dot_row
                if lo_l < lvl_here < hi_l:
                    bit = [0, 3, 1, 4, 2, 5, 6, 7][dr]
                    dots |= 1 << bit
            chars.append(chr(0x2800 + dots))
        lines.append(paint("".join(chars), line_color, color=color))
    if min_label:
        lines.append(
            paint(f"min {min(vals):g} · max {max(vals):g}", DIM, color=color)
        )
    return lines


def heatmap(
    rows: list[list[float]],
    *,
    color: bool | None = None,
    ramp: str = " ░▒▓█",
    labels: list[str] | None = None,
) -> list[str]:
    """Density heatmap (chartli ``heatmap`` idiom): one row per series."""
    if not rows or not any(rows):
        return [paint("(no data)", DIM, color=color)]
    flat = [v for row in rows for v in row]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    n = len(ramp) - 1
    out: list[str] = []
    for i, row in enumerate(rows):
        cells = "".join(ramp[min(n, int((v - lo) / span * n))] for v in row)
        prefix = f"{paint(str(labels[i])[:10], SUBTLE, color=color)} " if labels and i < len(labels) else ""
        out.append(prefix + paint(cells, CYAN, color=color))
    return out


def rule_caption(label: str, width: int = 58, *, color: bool | None = None) -> str:
    """Inline panel caption (btop idiom): ``┤ label ├────`` (exactly ``width``)."""
    label = f" {label} "
    fill = max(2, width - len(label) - 2)
    return (
        paint("┤", SUBTLE, color=color)
        + paint(label, TITLE + BOLD, color=color)
        + paint("├" + "─" * fill, SUBTLE, color=color)
    )


def table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    color: bool | None = None,
    box: str = "rounded",
    header_color: str = CYAN,
    max_width: int | None = None,
) -> list[str]:
    """Rich-Table-style box table, stdlib-only.

    ``box`` is ``rounded`` | ``single`` | ``double`` | ``heavy``.
    Column widths fit the content; rows are truncated to ``max_width``.
    """
    from .palette import (
        BOX_DOUBLE, BOX_HEAVY, BOX_ROUNDED, BOX_SINGLE, pad_visible,
    )

    glyphs = {"rounded": BOX_ROUNDED, "single": BOX_SINGLE,
              "double": BOX_DOUBLE, "heavy": BOX_HEAVY}.get(box, BOX_ROUNDED)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [visible_width(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], visible_width(cell))
    if max_width is not None:
        total = sum(widths) + 3 * len(widths) + 1
        if total > max_width:  # shrink widest columns first
            budget = max_width - 3 * len(widths) - 1
            while sum(widths) > budget and max(widths) > 4:
                i = widths.index(max(widths))
                widths[i] -= 1

    def border(left: str, mid: str, right: str, fill: str) -> str:
        return paint(
            left + mid.join(fill * (w + 2) for w in widths) + right,
            SUBTLE, color=color,
        )

    lines = [border(glyphs["tl"], glyphs["tt"], glyphs["tr"], glyphs["h"])]
    head = (
        paint(glyphs["v"], SUBTLE, color=color)
        + "".join(
            " " + paint(pad_visible(h, w), header_color + BOLD, color=color) + " "
            + paint(glyphs["v"], SUBTLE, color=color)
            for h, w in zip(headers, widths)
        )
    )
    lines.append(head)
    lines.append(border(glyphs["lt"], glyphs["cross"], glyphs["rt"], glyphs["h"]))
    for r in str_rows:
        cells = "".join(
            " " + pad_visible(r[i] if i < len(r) else "", w) + " "
            + paint(glyphs["v"], SUBTLE, color=color)
            for i, w in enumerate(widths)
        )
        lines.append(paint(glyphs["v"], SUBTLE, color=color) + cells)
    lines.append(border(glyphs["bl"], glyphs["bt"], glyphs["br"], glyphs["h"]))
    return lines


def panel(
    body: str | list[str],
    title: str = "",
    *,
    width: int | None = None,
    color: bool | None = None,
    box: str = "rounded",
    border_color: str = SUBTLE,
) -> list[str]:
    """Rich-Panel-style titled box, stdlib-only."""
    from .palette import (
        BOX_DOUBLE, BOX_HEAVY, BOX_ROUNDED, BOX_SINGLE,
        truncate_visible, visible_width,
    )

    glyphs = {"rounded": BOX_ROUNDED, "single": BOX_SINGLE,
              "double": BOX_DOUBLE, "heavy": BOX_HEAVY}.get(box, BOX_ROUNDED)
    lines_in = body.split("\n") if isinstance(body, str) else list(body)
    w = width or max((visible_width(strip_ansi(ln)) for ln in lines_in), default=0)
    w = max(4, w)
    out = []
    if title:
        cap = f" {title} "
        fill = max(0, w - len(cap))
        out.append(
            paint(glyphs["tl"] + glyphs["h"], border_color, color=color)
            + paint(cap, TITLE + BOLD, color=color)
            + paint(glyphs["h"] * fill + glyphs["tr"], border_color, color=color)
        )
    else:
        out.append(paint(glyphs["tl"] + glyphs["h"] * (w + 2) + glyphs["tr"],
                         border_color, color=color))
    for ln in lines_in:
        ln = truncate_visible(ln, w)
        pad = " " * max(0, w - visible_width(ln))
        out.append(
            paint(glyphs["v"] + " ", border_color, color=color)
            + ln + pad
            + paint(" " + glyphs["v"], border_color, color=color)
        )
    out.append(paint(glyphs["bl"] + glyphs["h"] * (w + 2) + glyphs["br"],
                     border_color, color=color))
    return out


class Spinner:
    """Animated status spinner (Rich ``console.status`` idiom).

    Usage::

        with Spinner("thinking…"):
            do_work()
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self,
        text: str = "",
        *,
        color: bool | None = None,
        out: Any = None,
        interval: float = 0.08,
    ) -> None:
        self.text = text
        self.color = supports_color() if color is None else color
        self.out = out or sys.stderr
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(self.interval):
            frame = self.FRAMES[i % len(self.FRAMES)]
            line = f"\r{paint(frame, CYAN, color=self.color)} {paint(self.text, SUBTLE, color=self.color)}"
            try:
                self.out.write(line)
                self.out.flush()
            except Exception:  # noqa: BLE001 - best effort
                break
            i += 1

    def __enter__(self) -> "Spinner":
        if self.color:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            self.out.write(f"{self.text}…\n")
            self.out.flush()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.out.write("\r" + " " * (len(strip_ansi(self.text)) + 4) + "\r")
            self.out.flush()
        except Exception:  # noqa: BLE001 - best effort
            pass


def status(text: str = "", **kw: Any) -> Spinner:
    """``with status("working…"):`` — the Rich ``console.status`` idiom."""
    return Spinner(text, **kw)


def _fmt_num(n: float) -> str:
    """tqdm-style unit scaling: 1500 → ``1.5k``, 2.3e6 → ``2.3M``."""
    n = float(n)
    for unit in ("", "k", "M", "G", "T"):
        if abs(n) < 1000 or unit == "T":
            if unit:
                return f"{n:.1f}{unit}"
            return f"{n:.0f}" if n == int(n) else f"{n:.1f}"
        n /= 1000
    return f"{n:.1f}T"


class ProgressBar:
    """Thread-safe progress bar with smoothed ETA. Pure ANSI, Termux-safe.

    tqdm-grade behavior, stdlib-only:

    - EMA-smoothed rate → stable ETA (no jitter from bursty updates)
    - unit scaling (``1.5k``, ``2.3M``) via ``unit`` / ``unit_scale``
    - adaptive width (fills the terminal) with 1/8-cell sub-precision fill
    - update throttling (``min_interval``) so fast loops don't flood the tty
    - ``bar_format`` template with ``{desc} {bar} {pct} {eta} {rate} {n}/{total}``
    - postfix stats dict (``set_postfix(loss=0.02)``)
    - indeterminate (spinner) mode when ``total`` is None
    - usable as a context manager and as ``ProgressBar.track(iterable)``

    Usage::

        bar = ProgressBar("downloading", total=100)
        for i in ...:
            bar.update(i)
        bar.done()
    """

    #: Spinner frames for indeterminate mode (console.status idiom).
    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self,
        label: str,
        total: float | None,
        *,
        width: int | None = None,
        color: bool | None = None,
        out: Any = None,
        show_eta: bool = True,
        unit: str = "",
        unit_scale: bool = False,
        min_interval: float = 0.1,
        bar_format: str | None = None,
        postfix: dict[str, Any] | None = None,
    ) -> None:
        self.label = label
        self.total = None if total is None else max(1e-9, float(total))
        self.width = width  # resolved lazily (adaptive) in _render
        self.color = supports_color() if color is None else color
        self.out = out or sys.stderr
        self.show_eta = show_eta
        self.unit = unit
        self.unit_scale = unit_scale
        self.min_interval = max(0.0, float(min_interval))
        self.bar_format = (
            bar_format
            or "{desc} {bar} {pct} {n}/{total} [{elapsed}<{eta}, {rate}{unit}/s]{postfix}"
        )
        self._postfix: dict[str, Any] = dict(postfix or {})
        self._lock = threading.Lock()
        self._done = False
        self._start = time.monotonic()
        self._last_draw = 0.0
        self._last_len = 0
        self._n = 0.0
        self._rate = 0.0  # EMA-smoothed items/sec
        self._spin_i = 0

    # ── public API ──

    def set_postfix(self, **kw: Any) -> None:
        """Attach ``key=value`` stats shown after the bar (tqdm idiom)."""
        with self._lock:
            self._postfix.update(kw)

    def update(self, done: float, n: float = 1) -> None:  # noqa: ARG002 - kept for tqdm parity
        """Set absolute progress to ``done`` (throttled redraw)."""
        with self._lock:
            if self._done:
                return
            now = time.monotonic()
            # EMA rate from the delta since the last update.
            dt = now - self._last_draw if self._last_draw else max(1e-6, now - self._start)
            if dt > 0:
                inst = max(0.0, float(done) - self._n) / dt
                self._rate = inst if self._rate == 0 else 0.3 * inst + 0.7 * self._rate
            self._n = max(0.0, float(done))
            if now - self._last_draw < self.min_interval and not self._finished():
                return
            self._draw_locked(now)

    def advance(self, n: float = 1) -> None:
        """Advance progress by ``n`` (tqdm ``update(n)`` idiom)."""
        # No lock here: update() takes it (Lock is not reentrant).
        self.update(self._n + n)

    def done(self, suffix: str = "done") -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            line = self._render(self.total if self.total else self._n,
                                time.monotonic())
            pad = " " * max(0, self._last_len - len(strip_ansi(line)))
            self.out.write(f"\r{line}{pad} {paint(suffix, GREEN, color=self.color)}\n")
            self.out.flush()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._done:
            self.done()

    @classmethod
    def track(
        cls,
        iterable: Iterable[Any],
        label: str = "",
        total: float | None = None,
        **kw: Any,
    ) -> Iterable[Any]:
        """Wrap an iterable with a progress bar (tqdm idiom)."""
        items = list(iterable) if total is None else iterable
        n_total = total if total is not None else (
            len(items) if hasattr(items, "__len__") else None
        )
        bar = cls(label or "working", n_total, **kw)
        try:
            for i, item in enumerate(items, 1):
                yield item
                bar.update(i)
        finally:
            bar.done()

    # ── internals ──

    def _finished(self) -> bool:
        return self.total is not None and self._n >= self.total

    def _draw_locked(self, now: float) -> None:
        line = self._render(self._n, now)
        pad = " " * max(0, self._last_len - len(strip_ansi(line)))
        self.out.write(f"\r{line}{pad}")
        self.out.flush()
        self._last_len = len(strip_ansi(line))
        self._last_draw = now

    def _fmt_n(self, n: float) -> str:
        if self.unit_scale:
            return f"{_fmt_num(n)}{self.unit}"
        if n == int(n):
            return f"{int(n)}{self.unit}"
        return f"{n:.1f}{self.unit}"

    def _render(self, done: float, now: float) -> str:
        c = self.color
        elapsed = now - self._start
        width = self.width or min(40, max(20, (shutil.get_terminal_size((80, 24)).columns - 40)))

        if self.total is None:
            # Indeterminate: bouncing spinner + elapsed + rate.
            self._spin_i += 1
            frame = self.SPINNER_FRAMES[self._spin_i % len(self.SPINNER_FRAMES)]
            rate = f"{self._fmt_rate()}" if self._rate else "—"
            line = (
                f"{paint(self.label, CYAN, color=c)} "
                f"{paint(frame, GREEN, color=c)} "
                f"{paint(self._fmt_n(done), BOLD, color=c)} "
                f"{paint(f'[{elapsed:4.0f}s, {rate}{self.unit}/s]', DIM, color=c)}"
            )
            return line + self._render_postfix(c)

        frac = min(1.0, max(0.0, done / self.total))
        # 1/8-cell sub-precision fill (btop meter idiom).
        cells = frac * width
        full = int(cells)
        part = int(round((cells - full) * 8))
        if part == 8:
            full += 1
            part = 0
        bar = "█" * full
        if full < width:
            bar += _BLOCK_FRACS[part] if part else "░"
            bar += "░" * (width - full - 1)
        pct = f"{frac * 100:5.1f}%"
        eta = "—"
        if self.show_eta and frac > 0.005 and frac < 1.0 and self._rate > 0:
            remain = (self.total - done) / self._rate
            eta = f"{remain:4.0f}s"

        def _p(text: str, code: str) -> str:
            return paint(text, code, color=c)

        fields = {
            "desc": _p(self.label, CYAN),
            "bar": _p(bar, GREEN),
            "pct": _p(pct, BOLD),
            "elapsed": f"{elapsed:4.0f}s",
            "eta": eta,
            "rate": self._fmt_rate(),
            "unit": self.unit,
            "n": self._fmt_n(done),
            "total": self._fmt_n(self.total),
            "postfix": self._render_postfix(c),
        }
        try:
            return self.bar_format.format(**fields)
        except (KeyError, IndexError, ValueError):
            return f"{fields['desc']} [{fields['bar']}] {fields['pct']}"

    def _fmt_rate(self) -> str:
        if self._rate <= 0:
            return "—"
        if self.unit_scale:
            return _fmt_num(self._rate)
        return f"{self._rate:.1f}"

    def _render_postfix(self, c: bool | None) -> str:
        if not self._postfix:
            return ""
        inner = ", ".join(f"{k}={v}" for k, v in self._postfix.items())
        return " " + paint(f"[{inner}]", DIM, color=c)


class LiveScreen:
    """In-place redrawing screen for watch-mode dashboards.

    Usage::

        with LiveScreen(interval=2.0) as screen:
            while screen.tick():
                screen.draw(render_dashboard(snapshot()))
    """

    def __init__(self, interval: float = 2.0, *, color: bool | None = None, out: Any = None):
        self.interval = max(0.5, float(interval))
        self.color = supports_color() if color is None else color
        self.out = out or sys.stdout
        self._stop = False

    def __enter__(self) -> "LiveScreen":
        if self.color:
            # Alternate screen: isolated buffer, restored on exit.
            self.out.write(_ALT_SCREEN_ON + _HIDE_CURSOR)
            self.out.flush()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.color:
            self.out.write(_SHOW_CURSOR + _ALT_SCREEN_OFF)
            self.out.flush()
        self._stop = True

    def tick(self) -> bool:
        """Sleep until the next frame; False when interrupted."""
        if self._stop:
            return False
        try:
            time.sleep(self.interval)
        except KeyboardInterrupt:
            return False
        return not self._stop

    def draw(self, text: str) -> None:
        if not self.color:
            self.out.write(text + "\n")
            self.out.flush()
            return
        # One atomic write: home, full frame, clear below. Any log line
        # that slipped in between frames is wiped by the next redraw,
        # and the alternate screen keeps the main terminal untouched.
        self.out.write(_HOME + text + _CLEAR_BELOW)
        self.out.flush()

    def stop(self) -> None:
        self._stop = True


def format_message_card(
    *,
    platform: str,
    sender: str,
    text: str,
    chat_title: str = "",
    timestamp: float | None = None,
    incoming: bool = True,
    color: bool | None = None,
) -> str:
    """Rich one-block rendering of a chat message for the console mirror.

    Example::

        ✈️ telegram · 12:04:33
        ┌─ Mary (@chfjdhx) in xauusd_sentinel_signal
        │ /game stats
    """
    icon = PLATFORM_ICONS.get((platform or "").lower(), "💭")
    ts = time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))
    head = f"{icon} {paint(platform, CYAN, color=color)} · {paint(ts, DIM, color=color)}"
    who = paint(sender or "?", BOLD, color=color)
    where = f" in {paint(chat_title, SUBTLE, color=color)}" if chat_title else ""
    direction = "→" if incoming else "←"
    body_lines = (text or "").splitlines() or [""]
    body = "\n".join(f"{paint('│', DIM, color=color)} {ln}" for ln in body_lines[:6])
    if len(body_lines) > 6:
        body += f"\n{paint('│', DIM, color=color)} {paint('…', DIM, color=color)}"
    return (
        f"{head}\n"
        f"{paint('┌─', DIM, color=color)} {direction} {who}{where}\n"
        f"{body}"
    )


__all__ = [
    "ProgressBar",
    "LiveScreen",
    "GodScreen",
    "MessageEvent",
    "MessageFeed",
    "WatchHub",
    "Spinner",
    "AVATAR",
    "sparkline",
    "barchart",
    "gauge",
    "columns_chart",
    "braille_chart",
    "heatmap",
    "table",
    "panel",
    "rule_caption",
    "status",
    "gradient_text",
    "format_message_card",
    "format_feed_line",
    "PLATFORM_ICONS",
    "WATCH_VIEWS",
    "WATCH_VIEW_KEYS",
]


# ═══════════════════════════════════════════════════════════════════════════
# God-tier watch mode: message feed, watch hub, split-pane live screen.
# ═══════════════════════════════════════════════════════════════════════════

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class MessageEvent:
    """One inbound/outbound chat message, for the watch-mode feed."""

    platform: str = "?"
    sender: str = "?"
    text: str = ""
    chat_title: str = ""
    timestamp: float = 0.0
    incoming: bool = True

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class MessageFeed:
    """Thread-safe ring buffer of recent message events.

    The gateway's console mirror pushes here; the watch screen drains
    from here. Never blocks, never raises.
    """

    def __init__(self, capacity: int = 50) -> None:
        self._capacity = max(10, capacity)
        self._lock = threading.Lock()
        self._events: list[MessageEvent] = []
        self._unread = 0

    def push(self, event: MessageEvent) -> None:
        try:
            with self._lock:
                # No fuzzy deduplication here: at-most-once delivery is
                # guaranteed by the adapters (they skip replayed Telegram
                # message ids at the source). The feed records what it is
                # given, exactly once per push.
                self._events.append(event)
                if len(self._events) > self._capacity:
                    del self._events[: len(self._events) - self._capacity]
                self._unread += 1
        except Exception:  # noqa: BLE001 - feed must never break the caller
            pass

    def recent(self, n: int) -> list[MessageEvent]:
        with self._lock:
            return list(self._events[-max(0, n):])

    def search(self, pattern: str, n: int = 50) -> list[MessageEvent]:
        """Regex search over buffered events, newest first (k9s ``/`` idiom)."""
        import re

        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        with self._lock:
            hits = [
                e for e in reversed(self._events)
                if rx.search(e.text or "") or rx.search(e.sender or "")
                or rx.search(e.chat_title or "")
            ]
        return hits[: max(0, n)]

    def by_platform(self, platform: str, n: int = 50) -> list[MessageEvent]:
        """Newest-first events for one platform."""
        want = (platform or "").lower()
        with self._lock:
            hits = [e for e in reversed(self._events)
                    if (e.platform or "").lower() == want]
        return hits[: max(0, n)]

    def clear(self) -> None:
        """Drop all buffered events and reset the unread counter."""
        with self._lock:
            self._events.clear()
            self._unread = 0

    @property
    def unread(self) -> int:
        with self._lock:
            return self._unread

    def mark_read(self) -> int:
        """Reset the unread counter; returns how many were unread."""
        with self._lock:
            n = self._unread
            self._unread = 0
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class WatchHub:
    """Coordinates watch mode with the console message mirror.

    When a ``dashboard --watch`` session is active, the gateway mirror
    must NOT print (it would flash over the dashboard). Instead it
    pushes into the shared feed, and the watch screen renders the feed
    in its own pane. Thread-safe; the mirror runs on gateway threads.
    """

    _lock = threading.Lock()
    _active = False
    _feed = MessageFeed()

    @classmethod
    def set_active(cls, active: bool) -> None:
        with cls._lock:
            cls._active = bool(active)
            if active:
                cls._feed.mark_read()

    @classmethod
    def is_active(cls) -> bool:
        with cls._lock:
            return cls._active

    @classmethod
    def feed(cls) -> MessageFeed:
        return cls._feed

    @classmethod
    def stats(cls) -> dict[str, Any]:
        """Feed stats for the watch header."""
        feed = cls._feed
        with feed._lock:  # noqa: SLF001 - same-module coordination
            plats: dict[str, int] = {}
            for e in feed._events:
                plats[(e.platform or "?").lower()] = (
                    plats.get((e.platform or "?").lower(), 0) + 1
                )
            return {
                "active": cls._active,
                "buffered": len(feed._events),
                "unread": feed._unread,
                "platforms": plats,
            }


def format_feed_line(event: MessageEvent, *, color: bool | None = None) -> str:
    """Compact one-line feed rendering: ``✈️ 15:32 Mary: /game stats``."""
    icon = PLATFORM_ICONS.get((event.platform or "").lower(), "💭")
    ts = time.strftime("%H:%M", time.localtime(event.timestamp))
    arrow = "→" if event.incoming else "←"
    sender = (event.sender or "?")[:24]
    text = " ".join((event.text or "").split())
    if len(text) > 90:
        text = text[:87] + "…"
    where = f" @{event.chat_title}" if event.chat_title else ""
    line = (
        f"{icon} {paint(ts, DIM, color=color)} "
        f"{paint(arrow, CYAN, color=color)} "
        f"{paint(sender, BOLD, color=color)}"
        f"{paint(where, SUBTLE, color=color)}: {text}"
    )
    return line


def barchart(
    items: list[tuple[str, float]],
    *,
    width: int = 18,
    color: bool | None = None,
    bar_color: str = CYAN,
) -> list[str]:
    """Horizontal bar chart lines: ``[('mafia', 12), ...]``.

    Returns one string per item: ``label  ████████  12``.
    Empty input returns a single dim placeholder line.
    """
    if not items:
        return [paint("(no data)", DIM, color=color)]
    peak = max((v for _, v in items), default=0) or 1.0
    label_w = max(len(str(label)) for label, _ in items)
    lines: list[str] = []
    for label, value in items:
        frac = max(0.0, min(1.0, float(value) / peak))
        filled = int(width * frac)
        bar = "█" * filled + "░" * (width - filled)
        num = f"{value:g}"
        lines.append(
            f"  {paint(str(label).ljust(label_w), SUBTLE, color=color)} "
            f"{paint(bar, bar_color, color=color)} "
            f"{paint(num, BOLD, color=color)}"
        )
    return lines


def gradient_text(
    text: str,
    start: int,
    end: int,
    *,
    color: bool | None = None,
) -> str:
    """Per-character 256-color gradient from ``start`` to ``end``.

    ``start``/``end`` are 256-color palette indexes (e.g. 51 → 201 for
    cyan→magenta). Termux-safe; plain text when color is off.
    """
    use_color = supports_color() if color is None else color
    if not use_color or not text:
        return text
    n = len(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        frac = i / max(1, n - 1)
        code = int(round(start + (end - start) * frac))
        code = max(0, min(255, code))
        out.append(f"\033[38;5;{code}m{ch}")
    out.append("\033[0m")
    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════
# Watch-mode terminal ownership: real single-keypress input + output guard.
# ═══════════════════════════════════════════════════════════════════════════

class _KeyReader:
    """Single-keypress input for watch mode.

    Uses termios raw mode when available (Linux/Termux) so ``1/2/3/4/d/q``
    fire on a bare keypress — no Enter needed. Falls back to canonical
    line input (key + Enter) when raw mode is unavailable. Always
    restores the terminal on exit.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old: Any = None
        try:
            import termios

            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            self._fd = fd
        except Exception:  # noqa: BLE001 - not a tty / no termios
            self._fd = None

    @property
    def raw(self) -> bool:
        """True when single-keypress mode is available."""
        return self._fd is not None

    def __enter__(self) -> "_KeyReader":
        if self._fd is not None:
            try:
                import tty

                tty.setraw(self._fd)
            except Exception:  # noqa: BLE001 - fall back to canonical
                self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None and self._old is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    def get(self, timeout: float) -> str | None:
        """Wait up to ``timeout`` seconds for one keypress.

        Returns the key character (lowercased), ``"esc"`` for escape
        sequences, or None on timeout / EOF / error.
        """
        if self._fd is not None:
            return self._get_raw(timeout)
        return self._get_canonical(timeout)

    def _get_raw(self, timeout: float) -> str | None:
        try:
            ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout))
        except Exception:  # noqa: BLE001
            return None
        if not ready:
            return None
        try:
            data = os.read(self._fd, 16)
        except OSError:
            return None
        if not data:
            return None  # EOF
        ch = data[:1].decode("utf-8", "replace")
        if ch == "\x1b":
            return "esc"  # arrow keys etc. — swallow the whole sequence
        if ch == "\x03":
            return "q"  # Ctrl-C arrives as ETX in raw mode
        return ch.lower() or None

    def _get_canonical(self, timeout: float) -> str | None:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
        except Exception:  # noqa: BLE001
            return None
        if not ready:
            return None
        try:
            line = sys.stdin.readline()
        except Exception:  # noqa: BLE001
            return None
        if not line:
            return None  # EOF
        return line.strip().lower()[:1] or None


class _WatchMuteFilter(logging.Filter):
    """Drops log records while the watch screen owns the terminal.

    Attached to every logging handler except DebugHub's capture handler,
    so the debug view keeps receiving telemetry while nothing prints.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not WatchHub.is_active()


class _ScreenGuard:
    """Gives the watch dashboard exclusive ownership of the terminal.

    The alternate-screen buffer alone proved insufficient on the owner's
    phone: gateway threads write to the same fd, so log lines landed
    inside the alt-screen buffer mid-frame. Two stronger layers:

    1. **Logging mute** — every logging handler except DebugHub's capture
       handler gets a mute filter while the guard is active. Logging
       produces zero terminal output, but telemetry still flows to the
       debug view (key ``d``).
    2. **fd redirect** — fds 1 and 2 are redirected to a spill file; the
       dashboard writes through a private duplicate of the original
       stdout fd. Any stray ``print()`` or C-level write lands in the
       spill file, never on screen.

    On exit everything is restored: fds, logging filters, cursor.
    When ``isolate`` is False (tests pass their own stream) only the
    logging mute applies — no fd games.
    """

    def __init__(self, *, isolate: bool = True, out: Any = None) -> None:
        self._isolate = isolate
        self._filter = _WatchMuteFilter()
        self._muted: list[logging.Handler] = []
        self._fd_out: int | None = None
        self._fd_err: int | None = None
        self._spill: Any = None
        self.tty: Any = out if out is not None else sys.stdout

    # ── logging mute ──

    def _mute_logging(self) -> None:
        seen: set[int] = set()
        handlers: list[logging.Handler] = list(logging.root.handlers)
        for lg in logging.Logger.manager.loggerDict.values():
            if isinstance(lg, logging.Logger):
                handlers.extend(lg.handlers)
        for handler in handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            if getattr(handler, "_devon_debug_capture", False):
                continue  # DebugHub keeps capturing for the debug view
            try:
                handler.addFilter(self._filter)
                self._muted.append(handler)
            except Exception:  # noqa: BLE001 - best effort
                pass

    def _unmute_logging(self) -> None:
        for handler in self._muted:
            try:
                handler.removeFilter(self._filter)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        self._muted.clear()

    # ── lifecycle ──

    def __enter__(self) -> "_ScreenGuard":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        self._mute_logging()
        if not self._isolate:
            return self
        try:
            if not os.isatty(1):
                return self
        except Exception:  # noqa: BLE001
            return self
        try:
            self._fd_out = os.dup(1)
            self._fd_err = os.dup(2)
            spill_path = os.path.join(tempfile.gettempdir(),
                                      "devon-watch-spill.log")
            self._spill = open(spill_path, "ab", buffering=0)
            os.dup2(self._spill.fileno(), 1)
            os.dup2(self._spill.fileno(), 2)
            # Private line-buffered handle to the real terminal — the
            # dashboard's only way out while fds 1/2 point at the spill.
            self.tty = os.fdopen(os.dup(self._fd_out), "w", buffering=1)
        except Exception:  # noqa: BLE001 - isolation is best-effort
            self.tty = sys.stdout
            self._fd_out = self._fd_err = None
            self._spill = None
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.tty.flush()
        except Exception:  # noqa: BLE001
            pass
        if self._fd_out is not None:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.dup2(self._fd_out, 1)
                os.dup2(self._fd_err, 2)
            except Exception:  # noqa: BLE001
                pass
            for fh in (self.tty, self._spill):
                try:
                    if fh is not None:
                        fh.close()
                except Exception:  # noqa: BLE001
                    pass
            for fd in (self._fd_out, self._fd_err):
                try:
                    if fd is not None:
                        os.close(fd)
                except Exception:  # noqa: BLE001
                    pass
            self.tty = sys.stdout
            self._fd_out = self._fd_err = None
            self._spill = None
        self._unmute_logging()


# View ids for the watch screen's keyboard switching.
WATCH_VIEWS = ("status", "games", "jobs", "brain", "debug")
WATCH_VIEW_KEYS = {
    "1": "status",
    "2": "games",
    "3": "jobs",
    "4": "brain",
    "d": "debug",
}
#: Hotkey shown in the header tabs, per view.
_VIEW_HOTKEY = {"status": "1", "games": "2", "jobs": "3", "brain": "4",
                "debug": "d"}
#: Extra drill-down views: no single-key hotkey (the 5-view contract is
#: pinned by tests), reachable via `:view <name>` and render_view().
WATCH_EXTRA_VIEWS = ("adapters", "health", "top")

#: Colon-commands available in GodScreen command mode (k9s ``:`` idiom).
_GOD_CMDS = ("theme", "view", "interval", "clear", "quit", "help")

#: Miniature ninja avatar for the header — the terminal can't show the
#: owner's real ninja avatar, so this stands in next to the DEVON title.
AVATAR = "🥷"

class GodScreen:
    """Full-screen live console: typographic header, view pane, message feed.

    Layout (terminal-size aware, every line width-truncated)::

        DEVON · live ⠋                    ⏱ 2m 30s  📨 49  🧠 groq
        universal agent os · no-morals 2.0
        [1] status  [2] games  [3] jobs  [4] brain  [d] debug
        ─────────────────────────────────────────────────────────
        ┌─ <view> ──────────────────────────────────────────────┐
        │ view content (status / games / jobs / brain / debug)   │
        └───────────────────────────────────────────────────────┘
        ┌─ messages (live) ─────────────────────────────────────┐
        │ compact feed lines                                      │
        └───────────────────────────────────────────────────────┘
        status bar · key hints

    Keys work on a bare keypress (termios raw mode; canonical fallback):
    ``1/2/3/4/d`` switch views, ``q``/Esc/Ctrl-C quits. While active the
    :class:`_ScreenGuard` owns the terminal: logging is muted (except the
    debug telemetry capture) and fds 1/2 are redirected to a spill file,
    so no log line can ever corrupt the frame.
    """

    HEADER_H = 4   # 4-row typographic header + separator
    STATUS_H = 2

    def __init__(
        self,
        interval: float = 2.0,
        *,
        snapshot: Callable[[], dict[str, Any]] | None = None,
        color: bool | None = None,
        out: Any = None,
    ) -> None:
        self.interval = max(0.5, float(interval))
        self._snapshot = snapshot or (lambda: {})
        self.color = supports_color() if color is None else color
        self.out = out or sys.stdout
        # fd isolation only when writing to the real stdout (not tests).
        self._isolate = out is None
        self._stop = False
        self._view = "status"
        self._frame = 0
        self._last_size: tuple[int, int] | None = None  # resize → full clear
        self._stdin_ok = bool(
            getattr(sys.stdin, "isatty", lambda: False)()
        )
        # k9s-style interaction state
        self._help = False            # "?" overlay
        self._cmd_buf: str | None = None   # ":" command mode buffer
        self._filter_buf: str | None = None  # "/" feed-filter buffer
        self._feed_filter = ""        # active regex filter on the feed
        self._feed_offset = 0         # j/k scrollback offset (0 = live tail)
        self._notice = ""             # one-frame notice line (command results)

    # ── public ──

    @property
    def view(self) -> str:
        return self._view

    def stop(self) -> None:
        self._stop = True

    def run(self) -> str:
        """Blocking watch loop. Returns an exit message for the console."""
        if not self.color:
            return paint(
                "watch mode needs a real terminal — "
                "type 'dashboard' for a static snapshot instead.",
                DIM,
            )
        WatchHub.set_active(True)
        from .debug import DebugHub

        DebugHub.install()
        guard = _ScreenGuard(isolate=self._isolate, out=self.out)
        tty_out = self.out
        try:
            with guard:
                tty_out = guard.tty
                # Alternate screen buffer: isolated screen, restored on
                # exit. (The guard's fd redirect + log mute are the real
                # corruption fix; this keeps the main screen pristine.)
                tty_out.write(_ALT_SCREEN_ON + _HIDE_CURSOR)
                tty_out.flush()
                try:
                    with _KeyReader() as keys:
                        while not self._stop:
                            self._render_frame(tty_out)
                            if not self._wait_key(keys):
                                break
                except KeyboardInterrupt:
                    self._stop = True  # Ctrl-C is a documented way out
                finally:
                    # Exit the alt screen BEFORE the guard closes the
                    # tty handle — otherwise this write is swallowed.
                    try:
                        tty_out.write(_SHOW_CURSOR + _ALT_SCREEN_OFF)
                        tty_out.flush()
                    except Exception:  # noqa: BLE001 - best-effort
                        pass
        finally:
            WatchHub.set_active(False)
        return paint("exited live dashboard", DIM)

    # ── internals ──

    def _wait_key(self, keys: "_KeyReader | None" = None) -> bool:
        """Wait up to ``interval`` for a keypress. False = quit requested."""
        if keys is None:  # pragma: no cover - tests patch this method
            try:
                time.sleep(self.interval)
            except KeyboardInterrupt:
                return False
            return not self._stop
        if not self._stdin_ok:
            try:
                time.sleep(self.interval)
            except KeyboardInterrupt:
                return False
            return not self._stop
        # Help overlay: any key dismisses (q/Esc still quit).
        if self._help:
            key = keys.get(self.interval * 4)
            if key is None:
                return not self._stop
            self._help = False
            if key in ("q", "esc", "\x03"):
                return False
            return not self._stop
        # Command / filter mode: mini line editor until Enter/Esc.
        if self._cmd_buf is not None or self._filter_buf is not None:
            return self._read_line_mode(keys)
        try:
            key = keys.get(self.interval)
        except KeyboardInterrupt:
            return False
        if key is None:  # timeout → refresh
            return not self._stop
        return self._handle_nav_key(key)

    def _handle_nav_key(self, key: str) -> bool:
        """Top-level watch-mode keys. False = quit requested."""
        if key in ("q", "esc", "\x03"):
            return False
        if key in WATCH_VIEW_KEYS:
            self._view = WATCH_VIEW_KEYS[key]
            self._feed_offset = 0
            return not self._stop
        if key == "?":
            self._help = True
            return not self._stop
        if key == ":":
            self._cmd_buf = ""
            return not self._stop
        if key == "/":
            self._filter_buf = ""
            return not self._stop
        if key == "t":  # cycle theme live
            self._cycle_theme()
            return not self._stop
        if key in ("+", "="):
            self.interval = min(10.0, self.interval + 0.5)
            self._notice = f"refresh every {self.interval:.1f}s"
            return not self._stop
        if key in ("-", "_"):
            self.interval = max(0.5, self.interval - 0.5)
            self._notice = f"refresh every {self.interval:.1f}s"
            return not self._stop
        if key == "j":  # feed scrollback: older
            self._feed_offset += 3
            return not self._stop
        if key == "k":  # feed scrollback: newer
            self._feed_offset = max(0, self._feed_offset - 3)
            return not self._stop
        if key == "G":  # jump back to live tail
            self._feed_offset = 0
            return not self._stop
        return not self._stop

    def _read_line_mode(self, keys: "_KeyReader") -> bool:
        """One tick of the ``:``/``/`` mini line editor."""
        is_cmd = self._cmd_buf is not None
        try:
            key = keys.get(0.2)
        except KeyboardInterrupt:
            self._cmd_buf = self._filter_buf = None
            return not self._stop
        if key is None:
            return not self._stop  # keep rendering the prompt row
        buf = self._cmd_buf if is_cmd else self._filter_buf
        assert buf is not None
        if key in ("\r", "\n"):
            if is_cmd:
                self._run_god_command(buf.strip())
                self._cmd_buf = None
            else:
                self._feed_filter = buf.strip()
                self._feed_offset = 0
                self._filter_buf = None
                if self._feed_filter:
                    self._notice = f"feed filter: /{self._feed_filter}/"
            return not self._stop
        if key == "esc":
            self._cmd_buf = self._filter_buf = None
            return not self._stop
        if key in ("\x7f", "\x08"):  # backspace
            buf = buf[:-1]
        elif len(key) == 1 and key.isprintable():
            buf += key
        if is_cmd:
            self._cmd_buf = buf
        else:
            self._filter_buf = buf
        return not self._stop

    def _run_god_command(self, cmd: str) -> bool:
        """Execute a ``:`` command. Returns False when it requests quit."""
        parts = cmd.split()
        if not parts:
            return True
        name, args = parts[0].lower(), parts[1:]
        if name in ("q", "quit", "exit"):
            self._stop = True
            return False
        if name == "view" and args:
            want = args[0].lower()
            if want in WATCH_VIEWS + WATCH_EXTRA_VIEWS:
                self._view = want
                self._feed_offset = 0
                self._notice = f"view → {want}"
            else:
                self._notice = f"unknown view '{want}'"
            return True
        if name == "theme":
            if args:
                self._set_theme(args[0].lower())
            else:
                from .themes import theme_name as _tn
                self._notice = f"theme: {_tn()}"
            return True
        if name == "interval" and args:
            try:
                self.interval = max(0.5, min(10.0, float(args[0])))
                self._notice = f"refresh every {self.interval:.1f}s"
            except ValueError:
                self._notice = "interval: need a number"
            return True
        if name == "clear":
            WatchHub.feed().clear()
            self._feed_filter = ""
            self._feed_offset = 0
            self._notice = "feed cleared"
            return True
        if name == "help":
            self._help = True
            return True
        self._notice = f"unknown :{name} — try :help"
        return True

    def _cycle_theme(self) -> None:
        """Cycle NM_CONSOLE_THEME to the next theme (``t`` key)."""
        import os

        from .themes import list_themes, theme_name as _tn

        names = list_themes()
        try:
            nxt = names[(names.index(_tn()) + 1) % len(names)]
        except ValueError:
            nxt = names[0]
        os.environ["NM_CONSOLE_THEME"] = nxt
        self._notice = f"theme → {nxt}"

    def _set_theme(self, name: str) -> None:
        import os

        from .themes import list_themes

        if name in list_themes():
            os.environ["NM_CONSOLE_THEME"] = name
            self._notice = f"theme → {name}"
        else:
            self._notice = f"unknown theme '{name}'"

    def _filtered_feed(self, feed: "MessageFeed") -> list["MessageEvent"]:
        """Feed events with the active ``/`` regex filter applied."""
        if not self._feed_filter:
            return feed.recent(500)
        return list(reversed(feed.search(self._feed_filter, n=500)))

    def _render_help_overlay(
        self, lines: list[str], width: int, height: int, c: bool | None,
        _t: Any,
    ) -> list[str]:
        """``?`` key overlay: every key, grouped (htop discoverability)."""
        from .palette import BOX_ROUNDED

        b = BOX_ROUNDED
        body = [
            ("views", "1 status · 2 games · 3 jobs · 4 brain · d debug "
                      "(extra: :view adapters|health|top)"),
            ("command", ": opens command mode — theme <name> · view <name> · "
                       "interval <secs> · clear · quit · help"),
            ("filter", "/ filters the message feed (regex) · Esc clears"),
            ("scroll", "j / k scroll feed · G back to live tail"),
            ("tweak", "t cycle theme · + / − refresh interval"),
            ("quit", "q / Esc / Ctrl-C"),
        ]
        w = min(64, width - 4)
        box: list[str] = []
        top = b["tl"] + b["h"] * (w + 2) + b["tr"]
        box.append(paint(top, STEALTH_CYAN, color=c))
        title = "⌨ keys"
        box.append(
            paint(b["v"] + " ", STEALTH_CYAN, color=c)
            + paint(title, STEALTH_CYAN + BOLD, color=c)
            + " " * max(0, w - len(title))
            + paint(" " + b["v"], STEALTH_CYAN, color=c)
        )
        for label, desc in body:
            row = f"{paint(label, WARN, color=c)}  {paint(desc, STEALTH_TEXT, color=c)}"
            row_w = len(label) + 2 + len(desc)
            box.append(
                paint(b["v"] + " ", STEALTH_CYAN, color=c)
                + row + " " * max(0, w - row_w)
                + paint(" " + b["v"], STEALTH_CYAN, color=c)
            )
        box.append(paint(b["bl"] + b["h"] * (w + 2) + b["br"],
                         STEALTH_CYAN, color=c))
        box.append(paint("  any key dismisses", DIM, color=c))
        # Overlay the box over the middle of the content region,
        # keeping header + footer pinned.
        head, foot = lines[:1], lines[-1:]
        mid_h = max(1, height - len(head) - len(foot) - len(box))
        pad_top = mid_h // 2
        mid = [""] * pad_top + [_t("  " + ln) for ln in box]
        while len(mid) < height - len(head) - len(foot):
            mid.append("")
        return head + mid[: height - len(head) - len(foot)] + foot

    def _snap(self) -> dict[str, Any]:
        try:
            snap = self._snapshot()
            return snap if isinstance(snap, dict) else {}
        except Exception:  # noqa: BLE001 - dashboard is best-effort
            return {}

    # ── frame assembly: ONE paint path (Grok layout discipline) ──

    #: Short platform tags for the feed rows.
    _PLAT_SHORT = {
        "telegram": "TG",
        "telegram-bot": "BOT",
        "local": "LOC",
        "discord": "DC",
        "whatsapp": "WA",
    }

    #: Adapter short names for the traffic row.
    _ADAPTER_SHORT = {
        "telegram": "tg",
        "telegram-bot": "bot",
        "local": "local",
    }

    def _ninja_mark(self) -> str:
        """Devon's face in the TUI: one ANSI glyph, cyan eye.

        The full ninja avatar lives on the Telegram bot PFP; the terminal
        gets this single mark beside the title — never a multi-row figure
        that fights the layout.
        """
        c = self.color
        return (
            paint("\ufe5d", SUBTLE, color=c)
            + paint("\u25c9", STEALTH_CYAN + BOLD, color=c)
            + paint("\ufe5e", SUBTLE, color=c)
        )

    def _fill_rule(self, left: str, right: str, width: int) -> str:
        """Join ``left``/``right`` with a ─ filler to exactly ``width``."""
        gap = max(0, width - visible_width(left) - visible_width(right))
        return left + paint("\u2500" * gap, STEALTH_CYAN, color=self.color) + right

    def _traffic_line(self, snap: dict[str, Any]) -> str:
        """Row 1 (status view): adapters, autonomy, arena, traffic."""
        c = self.color
        adapters = snap.get("adapters") or {}
        order = ["telegram", "telegram-bot", "local"]
        names = order + [n for n in adapters if n not in order]
        parts: list[str] = []
        for name in names:
            info = adapters.get(name) or {}
            short = self._ADAPTER_SHORT.get(name, str(name)[:8])
            running = bool(info.get("running", True))
            dot = (paint("\u25cf", STEALTH_CYAN, color=c) if running
                   else paint("\u25cb", DIM, color=c))
            rin = info.get("received", "\u2014")
            sout = info.get("sent", "\u2014")
            parts.append(
                f"{dot} {paint(short, STEALTH_TEXT, color=c)} "
                f"{paint(f'{rin}/{sout}', DIM, color=c)}"
            )
        extras = snap.get("extras") or {}
        autonomy = str(extras.get("autonomy", "\u2014")).upper()
        arena = str(extras.get("arena", "\u2014")).upper()
        parts.append(paint(f"autonomy {autonomy}",
                           STEALTH_CYAN if autonomy == "ON" else DIM, color=c))
        parts.append(paint(f"arena {arena}",
                           STEALTH_CYAN if arena == "ON" else DIM, color=c))
        traffic = snap.get("traffic") or {}
        msgs = traffic.get("messages", 0)
        errs = traffic.get("errors", 0)
        parts.append(paint(f"msgs {msgs}", STEALTH_TEXT, color=c))
        parts.append(paint(f"err {errs}",
                           STEALTH_AMBER + BOLD if errs else DIM, color=c))
        return paint("\u2502 ", STEALTH_CYAN, color=c) + "  ".join(parts)

    def _traffic_line_compact(self, snap: dict[str, Any]) -> str:
        """Narrow-terminal traffic row: ``tg\u25cf bot\u25cf loc\u25cb``."""
        c = self.color
        adapters = snap.get("adapters") or {}
        order = ["telegram", "telegram-bot", "local"]
        names = order + [n for n in adapters if n not in order]
        bits = []
        for name in names:
            info = adapters.get(name) or {}
            short = self._ADAPTER_SHORT.get(name, str(name)[:8])
            if bool(info.get("running", True)):
                bits.append(paint(f"{short}\u25cf", STEALTH_CYAN, color=c))
            else:
                bits.append(paint(f"{short}\u25cb", DIM, color=c))
        return paint("\u2502 ", STEALTH_CYAN, color=c) + " ".join(bits)

    def _scheduler_line(self, snap: dict[str, Any]) -> str:
        """Row 2 (status view): next scheduled jobs, ``name HH:MM``."""
        from .dashboard import _fmt_ts

        c = self.color
        sched = snap.get("scheduler") or {}
        jobs = [j for j in (sched.get("jobs") or [])
                if j.get("enabled", True)][:3]
        bits = []
        for job in jobs:
            name = str(job.get("name") or job.get("id") or "?")
            ts = _fmt_ts(job.get("next_run"))
            bits.append(f"{paint(name, STEALTH_TEXT, color=c)} "
                        f"{paint(ts, STEALTH_CYAN, color=c)}")
        body = " \u00b7 ".join(bits) if bits else paint("no jobs", DIM, color=c)
        return (paint("\u2502 ", STEALTH_CYAN, color=c)
                + paint("next: ", DIM, color=c) + body)

    def _feed_line(self, event: "MessageEvent") -> str:
        """One compact feed row: ``\u2502 17:04  TG  Ade  preview\u2026``."""
        from .palette import visible_width

        c = self.color
        ts = time.strftime("%H:%M", time.localtime(event.timestamp or time.time()))
        plat = self._PLAT_SHORT.get((event.platform or "").lower(), "??")
        sender = (event.sender or "?")[:12]
        text = " ".join((event.text or "").split())
        # Truncate by VISIBLE width (emoji = 2 cells), not char count.
        # Prefix is ~22 cells; leave room so the full line fits narrow terms.
        if visible_width(text) > 48:
            from .palette import truncate_visible as _tv
            text = _tv(text, 48)
        where = f" @{event.chat_title}" if event.chat_title else ""
        return (
            paint("\u2502 ", STEALTH_CYAN, color=c)
            + paint(ts, DIM, color=c) + "  "
            + paint(plat, STEALTH_CYAN, color=c) + "  "
            + paint(sender, STEALTH_TEXT + BOLD, color=c) + "  "
            + paint(text, STEALTH_TEXT, color=c)
            + paint(where[:20], SUBTLE, color=c)
        )

    def _render_frame(self, out: Any = None) -> None:
        """Render one frame — the ONLY paint path.

        Fixed-region layout (Grok discipline)::

            \u250c\u2500 DEVON \ufe5d\u25c9\ufe5e live \u00b7 2m30s \u00b7 no-morals 2.0 \u2500\u2500 [1][2][3][4][d][q]
            \u2502 \u25cf tg 68/0  \u25cf bot 1/0  \u25cf local 0/0  autonomy ON  msgs 70 \u00b7 err 0
            \u2502 next: rooms 14:16 \u00b7 watchers 14:16 \u00b7 research 16:50
            \u251c\u2500 messages \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            \u2502 17:04  TG  Ade     Jobs\u2026  preview truncated\u2026
            \u2502 (quiet \u2014 new messages appear here)
            \u2514\u2500 1 status \u00b7 2 games \u00b7 3 jobs \u00b7 4 brain \u00b7 d debug \u00b7 q quit \u2500\u2500

        Enforced: ONE status block (never drawn twice), logs muted by the
        :class:`_ScreenGuard`, fixed rows, full clear + redraw on resize,
        no partial overwrites, every line width-truncated.
        """
        from . import dashboard as _d

        out = out if out is not None else self.out
        snap = self._snap()
        cols, rows = shutil.get_terminal_size((80, 24))
        width = max(40, cols)
        height = max(16, rows)

        # Resize → clear + full redraw once. No partial overwrites.
        size = (cols, rows)
        if size != getattr(self, "_last_size", None):
            self._last_size = size
            try:
                out.write("\033[2J")
                out.flush()
            except Exception:  # noqa: BLE001 - best-effort
                pass

        c = self.color

        def _t(line: str) -> str:
            return truncate_visible(line, width)

        spin = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        uptime = _fmt_uptime(float(snap.get("uptime_s", 0) or 0))

        # ── row 0: header ──
        head_left = (
            paint("\u250c\u2500 ", STEALTH_CYAN, color=c)
            + paint("DEVON", STEALTH_CYAN + BOLD, color=c)
            + " " + self._ninja_mark() + " "
            + paint(f"live \u00b7 {uptime} \u00b7 no-morals 2.0", DIM, color=c)
            + " "
        )
        tabs = "".join(
            paint(f"[{_VIEW_HOTKEY[v]}]",
                  STEALTH_CYAN + BOLD if v == self._view else DIM,
                  color=c)
            for v in WATCH_VIEWS
        )
        lines = [_t(self._fill_rule(head_left, tabs, width))]

        feed = WatchHub.feed()
        unread = feed.unread
        feed.mark_read()

        # A prompt/notice row renders below the content only when active;
        # reserve its row so the frame stays pinned to the terminal height.
        prompt_rows = 1 if (
            self._cmd_buf is not None
            or self._filter_buf is not None
            or self._notice
        ) else 0

        if self._view == "status":
            # ── rows 1-2: traffic + scheduler (the ONE status block) ──
            if width < 60:
                lines.append(_t(self._traffic_line_compact(snap)))
            else:
                lines.append(_t(self._traffic_line(snap)))
            lines.append(_t(self._scheduler_line(snap)))
            # ── separator ──
            sep_label = "messages"
            if self._feed_filter:
                sep_label += "  /%s/" % self._feed_filter
            if self._feed_offset:
                sep_label += "  \u2195 -%d" % self._feed_offset
            sep_left = (paint("\u251c\u2500 ", STEALTH_CYAN, color=c)
                        + paint(sep_label, STEALTH_CYAN + BOLD, color=c) + " ")
            lines.append(_t(self._fill_rule(sep_left, "", width)))
            # ── feed takes the rest (filter + scrollback aware) ──
            feed_h = max(3, height - len(lines) - 1 - prompt_rows)  # minus footer
            events = self._filtered_feed(feed)
            if self._feed_offset:
                events = events[: max(0, len(events) - self._feed_offset)]
            shown = events[-feed_h:] if events else []
            for ev in shown:
                lines.append(_t(self._feed_line(ev)))
            for _ in range(feed_h - len(shown) - (0 if shown else 1)):
                lines.append(_t(paint("\u2502", SUBTLE, color=c)))
            if not shown:
                hint = ("(no matches \u2014 Esc clears the filter)"
                        if self._feed_filter else
                        "(quiet \u2014 new messages appear here)")
                lines.append(_t(paint("\u2502 ", STEALTH_CYAN, color=c)
                                + paint(hint, DIM, color=c)))
        else:
            # ── other views: label + content, no status duplication ──
            sep_left = (paint("\u251c\u2500 ", STEALTH_CYAN, color=c)
                        + paint(self._view, STEALTH_CYAN + BOLD, color=c) + " ")
            lines.append(_t(self._fill_rule(sep_left, "", width)))
            content_h = max(3, height - len(lines) - 1 - prompt_rows)  # minus footer
            view_text = _d.render_view(snap, self._view, color=c, bare=True)
            view_lines = view_text.splitlines()[:content_h]
            for ln in view_lines:
                lines.append(_t(paint("\u2502 ", STEALTH_CYAN, color=c) + ln))
            for _ in range(content_h - len(view_lines)):
                lines.append(_t(paint("\u2502", SUBTLE, color=c)))

        # ── help overlay replaces the content region ──
        if self._help:
            lines = self._render_help_overlay(lines, width, height, c, _t)

        # ── command / filter prompt row ──
        if self._cmd_buf is not None:
            prompt = (paint(":", STEALTH_CYAN + BOLD, color=c)
                      + paint(self._cmd_buf, BRIGHT_WHITE, color=c)
                      + paint("\u2588", STEALTH_CYAN, color=c))
            lines.append(_t(prompt))
        elif self._filter_buf is not None:
            prompt = (paint("/", WARN + BOLD, color=c)
                      + paint(self._filter_buf, BRIGHT_WHITE, color=c)
                      + paint("\u2588", WARN, color=c))
            lines.append(_t(prompt))
        elif self._notice:
            lines.append(_t(paint("\u2502 ", STEALTH_CYAN, color=c)
                            + paint(self._notice, WARN, color=c)))
            self._notice = ""

        # ── footer (1 row, pinned; htop's permanent key bar idiom) ──
        # Short on purpose: the full key list lives in the "?" overlay.
        # (The footer must keep "q quit" visible on 80-col terminals.)
        foot_left = (paint("\u2514\u2500 ", STEALTH_CYAN, color=c)
                     + paint("1 status \u00b7 2 games \u00b7 3 jobs \u00b7 4 brain \u00b7 "
                             "d debug \u00b7 : cmd \u00b7 q quit",
                             DIM, color=c))
        foot_right = ""
        if unread:
            foot_right += " " + paint(f"\U0001f4ec {unread} new",
                                      STEALTH_AMBER + BOLD, color=c)
        lines.append(_t(self._fill_rule(foot_left, foot_right, width)))

        # Pin to exactly the terminal height; every line fits the width,
        # so nothing wraps and the layout cannot garble.
        while len(lines) < height:
            lines.append("")
        frame = "\n".join(_t(ln) for ln in lines[:height])
        out.write(_HOME + frame + _CLEAR_BELOW)
        out.flush()


def _fmt_uptime(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
