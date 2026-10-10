"""Chart generation for the data-science workspace.

Renders matplotlib charts to PNG bytes, then hands them to the caller
(usually the CLI) to store as artifacts with provenance. Matplotlib uses
the non-interactive Agg backend — no display needed.

Supported kinds: ``line``, ``bar``, ``barh``, ``scatter``, ``hist``,
``kde``, ``box``, ``violin``, ``heatmap``, ``area``, ``pie``.

Styling lives in the :data:`THEMES` presets (``light``/``dark``/``ninja``/
``minimal``) — the render path itself is style-agnostic and just applies
whichever preset is requested. The ``ninja`` theme uses electric-blue
accents on a dark background.
"""

from __future__ import annotations

import io
from typing import Any

from .errors import PlotError

__all__ = ["PLOT_KINDS", "THEMES", "render_plot"]

PLOT_KINDS = (
    "line", "bar", "barh", "scatter", "hist", "kde",
    "box", "violin", "heatmap", "area", "pie",
)

# Style presets — composable, never hardcoded into the renderers.
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#dddddd",
        "grid.alpha": 0.9,
    },
    "dark": {
        "figure.facecolor": "#1a1d23",
        "axes.facecolor": "#1a1d23",
        "axes.edgecolor": "#8b949e",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#30363d",
        "grid.alpha": 0.9,
    },
    "ninja": {
        "figure.facecolor": "#05070d",
        "axes.facecolor": "#05070d",
        "axes.edgecolor": "#00b4ff",
        "axes.labelcolor": "#c9f1ff",
        "text.color": "#e8f7ff",
        "xtick.color": "#7fd4ff",
        "ytick.color": "#7fd4ff",
        "grid.color": "#0e2a3d",
        "grid.alpha": 1.0,
    },
    "minimal": {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": "#444444",
        "text.color": "#444444",
        "xtick.color": "#777777",
        "ytick.color": "#777777",
        "grid.color": "#eeeeee",
        "grid.alpha": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
}

# Kinds that need an (x, y) pair; area allows a bare y (x = index).
_XY_KINDS = ("line", "bar", "barh", "scatter")
# Kinds that draw on a single column.
_SINGLE_KINDS = ("hist", "kde")
# Kinds with (optional category x, numeric y).
_GROUP_KINDS = ("box", "violin")


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


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise PlotError(
            "numpy is required for plotting (pip install numpy)") from exc
    return np


def _check_col(cols: list[str], col: str, kind: str) -> None:
    if col not in cols:
        raise PlotError(f"column {col!r} not in {cols} (kind={kind})")


def _top_categories(frame: Any, x: str, y: str, top_n: int):
    """Return (labels, values) capped at ``top_n`` categories, ordered by
    value descending so the chart stays readable."""
    import pandas as pd
    s = frame[[x, y]].copy()
    s[x] = s[x].astype(str)
    grouped = s.groupby(x, as_index=False)[y].sum(numeric_only=True)
    grouped = grouped.sort_values(y, ascending=False).head(max(top_n, 1))
    return grouped[x].tolist(), grouped[y].tolist()


def _render_line(ax, frame, x, y, trend):
    ax.plot(frame[x], frame[y])
    if trend:
        _add_trend(ax, frame[x], frame[y])


def _add_trend(ax, xs, ys) -> None:
    """Best-fit line (paper-figure-codegen's scatter+regression pattern)."""
    np = _np()
    import pandas as pd
    data = pd.DataFrame({"x": xs, "y": ys}).dropna()
    data = data[pd.to_numeric(data["x"], errors="coerce").notna()
                & pd.to_numeric(data["y"], errors="coerce").notna()]
    if len(data) < 2:
        return
    xv = pd.to_numeric(data["x"]).to_numpy(dtype=float)
    yv = pd.to_numeric(data["y"]).to_numpy(dtype=float)
    if np.ptp(xv) == 0:
        return
    slope, intercept = np.polyfit(xv, yv, 1)
    line_x = np.array([xv.min(), xv.max()])
    ax.plot(line_x, slope * line_x + intercept, linestyle="--",
            linewidth=1.5, label=f"trend (slope={slope:.3g})")
    ax.legend()


def _render_bar(ax, frame, x, y, top_n, horizontal=False):
    labels, values = _top_categories(frame, x, y, top_n)
    if horizontal:
        ax.barh(labels, values)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values)
        if len(labels) > 8:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")


def _render_scatter(ax, frame, x, y, trend):
    ax.scatter(frame[x], frame[y], alpha=0.7)
    if trend:
        _add_trend(ax, frame[x], frame[y])


def _render_hist(ax, frame, x, bins):
    vals = frame[x].dropna()
    try:
        vals = vals.astype(float)
    except (ValueError, TypeError):
        pass
    ax.hist(vals, bins=bins, edgecolor="black", linewidth=0.5)


def _render_kde(ax, frame, x, bins):
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise PlotError(
            "kde plots need scipy (pip install scipy)") from exc
    np = _np()
    vals = frame[x].dropna()
    try:
        vals = vals.astype(float).to_numpy()
    except (ValueError, TypeError) as exc:
        raise PlotError(f"kde needs a numeric column, got {x!r}") from exc
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2 or np.ptp(vals) == 0:
        raise PlotError(f"kde needs ≥2 distinct values in {x!r}")
    kde = gaussian_kde(vals)
    grid = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(grid, kde(grid))
    ax.fill_between(grid, kde(grid), alpha=0.25)


def _render_group(ax, frame, kind, x, y, top_n):
    """box / violin: numeric y, optionally grouped by category x."""
    np = _np()
    yv = frame[y].dropna()
    try:
        yv = yv.astype(float)
    except (ValueError, TypeError) as exc:
        raise PlotError(f"{kind} needs a numeric y column, got {y!r}") from exc
    if not x:
        data = [yv.to_numpy()]
        labels = [y]
    else:
        cats = frame[x].astype(str).fillna("∅").unique().tolist()[:max(top_n, 1)]
        data, labels = [], []
        for cat in cats:
            grp = yv[frame[x].astype(str).fillna("∅") == cat].to_numpy()
            if len(grp):
                data.append(grp)
                labels.append(cat)
        if not data:
            raise PlotError(f"no plottable groups for x={x!r}, y={y!r}")
    if kind == "box":
        ax.boxplot(data, labels=labels, patch_artist=True)
    else:
        parts = ax.violinplot(data, showmeans=True)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0,
                           ha="right" if len(labels) > 6 else "center")


def _render_heatmap(ax, frame, x):
    np = _np()
    cols = [c for c in (x.split(",") if x else []) if c.strip()]
    num = frame.select_dtypes(include="number")
    if cols:
        missing = [c for c in cols if c not in num.columns]
        if missing:
            raise PlotError(f"heatmap columns not numeric/found: {missing}")
        num = num[cols]
    if num.shape[1] < 2:
        raise PlotError("heatmap needs ≥2 numeric columns")
    corr = num.corr(numeric_only=True).to_numpy()
    labels = [str(c) for c in num.columns]
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)


def _render_area(ax, frame, x, y):
    np = _np()
    ycols = [c.strip() for c in y.split(",") if c.strip()]
    xs = frame[x].to_numpy() if x else np.arange(len(frame))
    stacked = [frame[c].fillna(0).to_numpy(dtype=float) for c in ycols]
    ax.stackplot(xs, *stacked, labels=ycols, alpha=0.8)
    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0)


def _render_pie(ax, frame, x, top_n):
    vc = frame[x].astype(str).fillna("∅").value_counts().head(max(top_n, 1))
    if vc.empty:
        raise PlotError(f"pie: column {x!r} has no values")
    ax.pie(vc.to_numpy(), labels=vc.index.tolist(), autopct="%1.1f%%",
           startangle=90)
    ax.axis("equal")


def render_plot(frame: Any, kind: str, *, x: str = "", y: str = "",
                title: str = "", theme: str = "light",
                figsize: tuple[float, float] = (10, 6), dpi: int = 100,
                bins: int = 30, grid: bool = True, top_n: int = 20,
                trend: bool = False) -> bytes:
    """Render ``frame`` as a PNG chart. Returns PNG bytes.

    ``theme`` is one of :data:`THEMES` (style presets, never hardcoded
    looks). ``bins`` controls hist resolution, ``top_n`` caps categorical
    charts so they stay readable, ``trend`` adds a best-fit line to
    line/scatter plots.

    Raises :exc:`PlotError` for unknown kinds, missing columns, or
    render failures. Fail fast — never an empty/broken image.
    """
    if kind not in PLOT_KINDS:
        raise PlotError(f"unknown plot kind {kind!r}; kinds: {PLOT_KINDS}")
    if theme not in THEMES:
        raise PlotError(f"unknown theme {theme!r}; themes: {tuple(THEMES)}")
    plt = _mpl()
    cols = [str(c) for c in frame.columns]

    if kind in _XY_KINDS:
        if not x or not y:
            raise PlotError(f"{kind} plot needs --x and --y columns")
        _check_col(cols, x, kind)
        _check_col(cols, y, kind)
    elif kind in _SINGLE_KINDS:
        if not x:
            raise PlotError(f"{kind} plot needs --x column")
        _check_col(cols, x, kind)
    elif kind in _GROUP_KINDS:
        if not y:
            raise PlotError(f"{kind} plot needs --y column")
        _check_col(cols, y, kind)
        if x:
            _check_col(cols, x, kind)
    elif kind == "area":
        if not y:
            raise PlotError("area plot needs --y column(s)")
        for col in [c.strip() for c in y.split(",") if c.strip()]:
            _check_col(cols, col, kind)
        if x:
            _check_col(cols, x, kind)
    elif kind == "pie":
        if y:
            raise PlotError("pie plot takes only --x (category column)")
        if not x:
            raise PlotError("pie plot needs --x column")
        _check_col(cols, x, kind)
    elif kind == "heatmap":
        pass  # validated inside the renderer

    with plt.rc_context(THEMES[theme]):
        fig, ax = plt.subplots(figsize=figsize)
        try:
            if kind == "line":
                _render_line(ax, frame, x, y, trend)
            elif kind == "bar":
                _render_bar(ax, frame, x, y, top_n)
            elif kind == "barh":
                _render_bar(ax, frame, x, y, top_n, horizontal=True)
            elif kind == "scatter":
                _render_scatter(ax, frame, x, y, trend)
            elif kind == "hist":
                _render_hist(ax, frame, x, bins)
            elif kind == "kde":
                _render_kde(ax, frame, x, bins)
            elif kind in _GROUP_KINDS:
                _render_group(ax, frame, kind, x, y, top_n)
            elif kind == "heatmap":
                _render_heatmap(ax, frame, x)
            elif kind == "area":
                _render_area(ax, frame, x, y)
            elif kind == "pie":
                _render_pie(ax, frame, x, top_n)
            ax.set_xlabel(x)
            if y and kind != "pie":
                ax.set_ylabel(y)
            if title:
                ax.set_title(title)
            if grid and kind not in ("pie", "heatmap"):
                ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.5)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
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
