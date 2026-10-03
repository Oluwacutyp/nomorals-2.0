"""Chart generation for the data-science workspace.

Renders matplotlib charts to PNG bytes, then hands them to the caller
(usually the CLI) to store as artifacts with provenance. Matplotlib uses
the non-interactive Agg backend — no display needed.

Supported kinds: ``line``, ``bar``, ``scatter``, ``hist``.
"""

from __future__ import annotations

import io
from typing import Any

from .errors import PlotError

__all__ = ["PLOT_KINDS", "render_plot"]

PLOT_KINDS = ("line", "bar", "scatter", "hist")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotError(
            "matplotlib is required for plotting "
            "(pip install matplotlib)") from exc
    return plt


def render_plot(frame: Any, kind: str, *, x: str = "", y: str = "",
                title: str = "") -> bytes:
    """Render ``frame`` as a PNG chart. Returns PNG bytes.

    Raises :exc:`PlotError` for unknown kinds, missing columns, or
    render failures. Fail fast — never an empty/broken image.
    """
    if kind not in PLOT_KINDS:
        raise PlotError(f"unknown plot kind {kind!r}; kinds: {PLOT_KINDS}")
    plt = _mpl()
    cols = [str(c) for c in frame.columns]
    if kind in ("line", "bar", "scatter"):
        if not x or not y:
            raise PlotError(f"{kind} plot needs --x and --y columns")
        for col in (x, y):
            if col not in cols:
                raise PlotError(f"column {col!r} not in {cols}")
    if kind == "hist":
        if not x:
            raise PlotError("hist plot needs --x column")
        if x not in cols:
            raise PlotError(f"column {x!r} not in {cols}")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if kind == "line":
            ax.plot(frame[x], frame[y])
        elif kind == "bar":
            ax.bar(frame[x].astype(str), frame[y])
        elif kind == "scatter":
            ax.scatter(frame[x], frame[y])
        elif kind == "hist":
            ax.hist(frame[x].dropna())
        ax.set_xlabel(x)
        if y:
            ax.set_ylabel(y)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        data = buf.getvalue()
    except PlotError:
        raise
    except Exception as exc:
        raise PlotError(f"render failed: {exc}") from exc
    finally:
        plt.close(fig)
    if not data:
        raise PlotError("render produced empty PNG")
    return data
