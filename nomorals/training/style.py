"""God-tier presentation for the training module: themed cards, tables,
sparklines, and progress bars.

Everything here is dependency-free and renders to plain text.  Color is
auto-disabled when stdout is not a TTY (logs stay clean) and can be forced
off with ``Theme(color=False)`` — the "ninja" theme is the default: dark,
electric-blue accents, box-drawing borders.

    from nomorals.training.style import Theme, card, table, sparkline

    theme = Theme()
    print(card(theme, "PROMOTION GATE", [("run", "personal-v3"),
                                        ("score", "0.8124"),
                                        ("verdict", theme.ok("PROMOTED")]),
               width=44))
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Theme",
    "card",
    "table",
    "sparkline",
    "progress_bar",
    "render_run_card",
    "render_gate_report",
    "render_policy_card",
    "render_eval_report",
]

# ── theme ───────────────────────────────────────────────────────────────────

_RESET = "\033[0m"


@dataclass
class Theme:
    """One visual identity for training output.

    ``color`` defaults to auto (TTY on stdout).  ``accent`` picks the
    palette: ``"ninja"`` (electric blue on dark), ``"plain"`` (no accents
    even with color on), ``"ember"`` (warm amber).
    """

    name: str = "ninja"
    color: bool | None = None
    accent: str = "ninja"
    _codes: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.color is None:
            try:
                self.color = bool(sys.stdout.isatty()) and os.environ.get("NO_COLOR", "") == ""
            except Exception:  # noqa: BLE001
                self.color = False
        palettes = {
            "ninja": {
                "title": "\033[1;96m", "key": "\033[36m", "val": "\033[97m",
                "ok": "\033[1;92m", "warn": "\033[1;93m", "bad": "\033[1;91m",
                "dim": "\033[2m", "bar": "\033[96m",
            },
            "ember": {
                "title": "\033[1;33m", "key": "\033[33m", "val": "\033[97m",
                "ok": "\033[1;92m", "warn": "\033[1;93m", "bad": "\033[1;91m",
                "dim": "\033[2m", "bar": "\033[33m",
            },
            "plain": {},
        }
        self._codes = palettes.get(self.accent, palettes["ninja"]) if self.color else {}

    def paint(self, text: str, role: str) -> str:
        code = self._codes.get(role, "")
        return f"{code}{text}{_RESET}" if code else text

    def ok(self, text: str) -> str:
        return self.paint(str(text), "ok")

    def warn(self, text: str) -> str:
        return self.paint(str(text), "warn")

    def bad(self, text: str) -> str:
        return self.paint(str(text), "bad")

    def dim(self, text: str) -> str:
        return self.paint(str(text), "dim")


# ── primitives ──────────────────────────────────────────────────────────────

def _visible(text: str) -> int:
    """Printable width, ignoring ANSI escapes."""
    out = 0
    in_escape = False
    for ch in text:
        if ch == "\033":
            in_escape = True
            continue
        if in_escape:
            if ch == "m":
                in_escape = False
            continue
        out += 1
    return out


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible(text))


def card(
    theme: Theme,
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    width: int = 52,
    footer: str = "",
) -> str:
    """A bordered key/value card.  Values may already carry theme paint."""
    inner = max(10, width - 4)
    top = "╭" + "─" * (inner + 2) + "╮"
    bottom = "╰" + "─" * (inner + 2) + "╯"
    lines = [theme.paint(top, "dim"), theme.paint("│ ", "dim")
             + theme.paint(title.upper(), "title")
             + " " * max(0, inner - _visible(title)) + theme.paint(" │", "dim")]
    lines.append(theme.paint("├" + "─" * (inner + 2) + "┤", "dim"))
    for key, value in rows:
        label = theme.paint(key, "key") + ":"
        body = _pad(label + " " + str(value), inner)
        lines.append(theme.paint("│ ", "dim") + body + theme.paint(" │", "dim"))
    if footer:
        lines.append(theme.paint("├" + "─" * (inner + 2) + "┤", "dim"))
        lines.append(theme.paint("│ ", "dim") + _pad(theme.dim(footer), inner)
                     + theme.paint(" │", "dim"))
    lines.append(theme.paint(bottom, "dim"))
    return "\n".join(lines)


def table(
    theme: Theme,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    align_right: Sequence[int] | None = None,
) -> str:
    """A minimal column table with a header rule."""
    align_right = set(align_right or ())
    cells = [[str(c) for c in row] for row in rows]
    widths = [max([_visible(h)] + [_visible(r[i]) for r in cells] or [0])
              for i, h in enumerate(headers)]
    head = "  ".join(
        (_pad(h, w) if i not in align_right else _pad(h, w)[::-1].rjust(w)[::-1] or _pad(h, w))
        for i, (h, w) in enumerate(zip(headers, widths)))
    out = [theme.paint(head, "title"), theme.dim("─" * _visible(head))]
    for row in cells:
        out.append("  ".join(_pad(c, w) for c, w in zip(row, widths)))
    return "\n".join(out)


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], *, width: int = 24) -> str:
    """A loss-history sparkline.  Falls back gracefully on empty/NaN input."""
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not clean:
        return "·" * min(width, 8)
    if len(clean) == 1:
        return _SPARK[3] * min(width, 8)
    lo, hi = min(clean), max(clean)
    span = (hi - lo) or 1.0
    step = max(1, len(clean) // width)
    sampled = clean[::step][:width]
    return "".join(_SPARK[min(7, int((v - lo) / span * 7))] for v in sampled)


def progress_bar(theme: Theme, fraction: float, *, width: int = 24) -> str:
    """``██████░░░░`` style bar, 0.0–1.0."""
    frac = max(0.0, min(1.0, float(fraction)))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return theme.paint(bar, "bar") + f" {frac * 100:5.1f}%"


# ── domain cards ────────────────────────────────────────────────────────────

def render_run_card(theme: Theme, run: Mapping[str, Any]) -> str:
    """One card for a training run row (registry-shaped dict)."""
    status = str(run.get("status", "?"))
    status_painted = (
        theme.ok(status.upper()) if status in {"promoted", "done"}
        else theme.bad(status.upper()) if status in {"failed", "rejected"}
        else theme.warn(status.upper()))
    metrics = run.get("metrics") or {}
    rows = [
        ("run", str(run.get("name") or run.get("id", "?"))),
        ("status", status_painted),
        ("backend", str(run.get("backend", "?"))),
        ("base", str(run.get("base_model", "?") or "—")),
    ]
    for key in ("score", "eval_loss", "perplexity", "golden_score"):
        if key in metrics and metrics[key] is not None:
            rows.append((key, f"{metrics[key]}"))
    hist = metrics.get("train_loss_history") or []
    if hist:
        rows.append(("loss", sparkline(hist)))
    if run.get("error"):
        rows.append(("error", theme.bad(str(run["error"])[:60])))
    return card(theme, "training run", rows, footer=str(run.get("output_path", "")))


def render_gate_report(theme: Theme, report: Mapping[str, Any]) -> str:
    """The promotion-gate verdict as a card."""
    verdict = bool(report.get("passed"))
    rows = [
        ("challenger", str(report.get("challenger", "?"))),
        ("champion", str(report.get("champion", "?") or "— none —")),
        ("challenger score", f"{report.get('challenger_score', 0):.6f}"),
        ("champion score", f"{report.get('champion_score', 0):.6f}"
         if report.get("champion_score") is not None else "—"),
        ("margin", f"{report.get('margin', 0):+.6f}"),
        ("min gain", f"{report.get('min_gain', 0):.6f}"),
        ("verdict", theme.ok("PASS — promote") if verdict else theme.bad("FAIL — reject")),
    ]
    for reason in report.get("reasons", []) or []:
        rows.append(("·", theme.dim(str(reason))))
    return card(theme, "promotion gate", rows)


def render_policy_card(theme: Theme, decision: Mapping[str, Any]) -> str:
    """A retraining-policy decision as a card."""
    should = bool(decision.get("should_retrain"))
    rows = [
        ("retrain", theme.ok("YES") if should else theme.dim("no")),
        ("new examples", str(decision.get("new_examples", 0))),
    ]
    elapsed = decision.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
        rows.append(("since last run", f"{elapsed / 3600:.1f}h"))
    drift = decision.get("drift_score")
    if drift is not None:
        rows.append(("drift (PSI)", f"{drift:.3f}"))
    for reason in decision.get("reasons", []) or []:
        rows.append(("·", str(reason)))
    return card(theme, "retraining policy", rows)


def render_eval_report(theme: Theme, report: Mapping[str, Any]) -> str:
    """The full evaluation battery as one card."""
    rows: list[tuple[str, str]] = []
    if report.get("eval_loss") is not None:
        rows.append(("eval loss", f"{report['eval_loss']:.4f}"))
    if report.get("perplexity") is not None:
        rows.append(("perplexity", f"{report['perplexity']:.2f}"))
    if report.get("golden_score") is not None:
        g = report["golden_score"]
        rows.append(("golden", theme.ok(f"{g:.2%}") if g >= 0.8
                     else theme.warn(f"{g:.2%}") if g >= 0.5 else theme.bad(f"{g:.2%}")))
    if report.get("judge_score") is not None:
        rows.append(("judge", f"{report['judge_score']:.3f}"))
    if report.get("win_rate") is not None:
        rows.append(("pairwise win-rate", f"{report['win_rate']:.1%}"))
    cats = report.get("by_category") or {}
    for category, score in cats.items():
        rows.append((f"  {category}", f"{score:.2%}"))
    failures = report.get("failures") or []
    for failure in failures[:3]:
        rows.append(("✗", theme.bad(str(failure)[:58])))
    if len(failures) > 3:
        rows.append(("", theme.dim(f"… +{len(failures) - 3} more")))
    return card(theme, "evaluation", rows)


def format_seconds(seconds: float) -> str:
    """90.5 → '1m 30s', 3700 → '1h 1m'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m {int(seconds % 60)}s"
    return f"{minutes // 60}h {minutes % 60}m"
