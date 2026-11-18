"""Retail price history — CamelCamelCamel, Devon-native.

"Watch this on Amazon, tell me under $50" → a persistent price
watch → a scheduler beat journals one price point per series per
check → a drop below target (or a new low) pages the owner once,
and "price history of X" draws the chart from her own journal.

The three-series distinction is real in the data model and the
chart: Amazon watches follow ``amazon`` (the list price),
``tp_new`` (third-party new) or ``tp_used`` (third-party used);
other marketplaces have one series, named after the marketplace.
The fetcher reports what it actually observed — a series with no
observation this check is named, never invented.

Alert discipline follows the travel price-watcher idiom
(:mod:`nomorals.travel.watchers`): one alert per drop level
(``alerted_minor`` — a fresh low), a target crossing fires once,
back-in-stock fires on the out→in transition only, and a
cooldown in the alert log stops flapping repeats. A second tick
below target stays silent.

History begins the day watching begins: the chart carries the
true start date (first recorded point), never a backfilled one.
Marketplace coverage follows her shopping sources
(``integrations/naija_shopping.py`` SITES) — seeded here, never
fixed at CamelCamelCamel's 8.

The price fetcher is injected (``fetcher(product_ref,
marketplace) -> list[observation] | None``): the organ never
imports the L5 shopping engine — the tool layer
(:mod:`nomorals.tools.price_history`) and the scheduler tick
supply it. A check with no fetcher records nothing and says so.

Scheduler seam: ``ensure_price_history_job(scheduler)`` registers
a daily 07:00 cron with action ``PRICE_HISTORY_ACTION``; the host
calls :func:`run_price_history_tick_once` when that action fires.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PriceWatch",
    "PricePoint",
    "PriceAlert",
    "ChartResult",
    "PriceHistoryStore",
    "MARKETPLACES",
    "AMAZON_SERIES",
    "SERIES_LABELS",
    "PRICE_HISTORY_ACTION",
    "PRICE_HISTORY_JOB",
    "PRICE_HISTORY_SPEC",
    "parse_price_request",
    "trend_stats",
    "format_minor",
    "format_alert",
    "format_watches",
    "format_history",
    "render_price_chart",
    "run_price_history_tick_once",
    "ensure_price_history_job",
    "check_all",
]

#: Scheduler action name the host dispatches to the tick.
PRICE_HISTORY_ACTION = "price_history_check"

#: Scheduler job name (idempotent by name).
PRICE_HISTORY_JOB = "price history tick"

#: Cron: daily price check at 07:00 — the travel-watcher idiom.
PRICE_HISTORY_SPEC = "0 7 * * *"

_DEFAULT_DB = "~/.nomorals/commerce/price_history.db"

#: Marketplace coverage — seeded from her shopping sources
#: (``integrations/naija_shopping.py`` SITES), never fixed at
#: CamelCamelCamel's 8. ``currency`` is the marketplace's own.
MARKETPLACES: dict[str, dict[str, str]] = {
    "amazon": {"name": "Amazon", "currency": "USD"},
    "jumia": {"name": "Jumia Nigeria", "currency": "NGN"},
    "konga": {"name": "Konga", "currency": "NGN"},
    "jiji": {"name": "Jiji Nigeria", "currency": "NGN"},
    "kara": {"name": "Kara Nigeria", "currency": "NGN"},
    "slot": {"name": "SLOT", "currency": "NGN"},
    "temu": {"name": "Temu", "currency": "USD"},
    "aliexpress": {"name": "AliExpress", "currency": "USD"},
    "ebay": {"name": "eBay", "currency": "USD"},
    "banggood": {"name": "Banggood", "currency": "USD"},
}

#: The three Amazon price series (CamelCamelCamel's distinction).
AMAZON_SERIES = ("amazon", "tp_new", "tp_used")

SERIES_LABELS = {
    "amazon": "Amazon",
    "tp_new": "3rd-party new",
    "tp_used": "3rd-party used",
}

#: One line colour per series, shared by both chart rungs so the
#: pixel proof means the same thing on either.
SERIES_COLORS = {
    "amazon": "#1f77b4",
    "tp_new": "#ff7f0e",
    "tp_used": "#2ca02c",
}
_FALLBACK_COLORS = ("#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf")


def _obs_minor(obs: dict[str, Any]) -> int:
    """One observation's price in minor units.

    Observations carry ``price_minor`` (minor units) or ``price``
    (major units); anything else reads as no price.
    """
    if "price_minor" in obs:
        try:
            return max(0, int(obs["price_minor"]))
        except (TypeError, ValueError):
            return 0
    price = obs.get("price")
    if isinstance(price, (int, float)):
        return _to_minor(price, "")
    return 0


def _series_color(series: str, index: int = 0) -> str:
    if series in SERIES_COLORS:
        return SERIES_COLORS[series]
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def _series_label(series: str, marketplace: str = "") -> str:
    if series in SERIES_LABELS:
        return SERIES_LABELS[series]
    if marketplace and series == marketplace:
        return MARKETPLACES.get(marketplace, {}).get("name", series)
    return series


# ── money (minor units, like the travel watcher) ──────────────────────

def _to_minor(amount: float, currency: str) -> int:
    """Major units → minor (kobo/cents). Never negative."""
    try:
        return max(0, int(round(float(amount) * 100)))
    except (TypeError, ValueError):
        return 0


def _fmt_minor(minor: int, currency: str = "") -> str:
    """₦450,000 / $49.99 — currency-aware, no invented symbols."""
    cur = (currency or "").upper()
    if cur == "USD":
        return f"${minor / 100:,.2f}"
    if cur == "NGN":
        return f"₦{minor / 100:,.0f}"
    if cur:
        return f"{minor / 100:,.2f} {cur}"
    return f"{minor / 100:,.2f}"


def format_minor(minor: int, currency: str = "") -> str:
    """Public money formatter: ₦450,000 / $49.99."""
    return _fmt_minor(minor, currency)


def _short_minor(minor: int, currency: str = "") -> str:
    """₦450k / $1.2k — compact form for alert lines."""
    cur = (currency or "").upper()
    sym = "$" if cur == "USD" else "₦" if cur == "NGN" else ""
    n = minor / 100
    if n >= 1000:
        return f"{sym}{n / 1000:,.1f}k".replace(".0k", "k")
    if cur == "USD":
        return f"${n:,.2f}"
    return f"{sym}{n:,.0f}"


@dataclass
class PriceWatch:
    id: str
    product_ref: str
    marketplace: str
    series: str = "amazon"
    target_minor: int = 0  # 0 = no target, alert on any drop ≥ threshold
    threshold_pct: float = 10.0
    currency: str = ""
    last_minor: int = 0
    last_currency: str = ""
    alerted_minor: int = 0  # price level already alerted at
    last_in_stock: bool | None = None
    active: bool = True
    created_at: float = 0.0
    last_checked: float = 0.0
    cooldown_h: float = 24.0  # min hours between repeat stock alerts

    @property
    def label(self) -> str:
        market = MARKETPLACES.get(self.marketplace, {}).get(
            "name", self.marketplace)
        return f"{self.product_ref} ({market})"


@dataclass
class PricePoint:
    ts: float
    series: str
    price_minor: int
    currency: str = ""
    in_stock: bool = True


@dataclass
class PriceAlert:
    watch_id: str
    product_ref: str
    marketplace: str
    series: str
    previous_minor: int
    current_minor: int
    currency: str = ""
    kind: str = "drop"  # drop | stock
    target_minor: int = 0
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = format_alert(self)


@dataclass
class ChartResult:
    png: bytes
    sha256: str
    width: int
    height: int
    series: list[str]
    n_points: int
    started_at: str  # YYYY-MM-DD of the first recorded point
    rung: str  # matplotlib | pil
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256, "width": self.width,
            "height": self.height, "series": list(self.series),
            "n_points": self.n_points, "started_at": self.started_at,
            "rung": self.rung, "title": self.title,
            "bytes": len(self.png),
        }

# ── NL parsing ────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_IATA_RE = re.compile(r"\b([A-Z]{3})\b")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MARKET_RE = re.compile(
    r"\b(amazon|jumia|konga|jiji|kara|slot|temu|aliexpress|ebay|banggood"
    r"|payporte)\b", re.I)
_TARGET_RE = re.compile(
    r"(?:under|below|max|less than)\s*(₦|\$|N)?\s*([\d,.]+)\s*(k|m)?\b",
    re.I)
_SERIES_NEW_RE = re.compile(
    r"(?:third[ -]?party|3rd[ -]?party)\s+new\b", re.I)
_SERIES_USED_RE = re.compile(
    r"(?:third[ -]?party|3rd[ -]?party)\s+used\b", re.I)
_HISTORY_RE = re.compile(
    r"\bprice histor(?:y|ies)\b|\bhow (?:has|have) the price\b"
    r"|\bshow(?: me)? (?:the )?price (?:chart|graph|history)\b", re.I)
_LEAD_RE = re.compile(
    r"^(?:please\s+)?(?:watch|track|show(?: me)?|get(?: me)?|check)\b\s*",
    re.I)
_STRIP_LEAD = re.compile(
    r"^(?:the price of|price of|this|the|a|an|my)(?:\s+|$)", re.I)
_STRIP_SERIES = re.compile(
    r"^(?:third[ -]?party|3rd[ -]?party)\s+(?:new|used)\s+"
    r"(?:price of\s+)?", re.I)

_URL_HOSTS = (
    ("amazon.", "amazon"),
    ("jumia.", "jumia"),
    ("konga.", "konga"),
    ("jiji.", "jiji"),
    ("kara.", "kara"),
    ("slot.", "slot"),
    ("temu.", "temu"),
    ("aliexpress.", "aliexpress"),
    ("ebay.", "ebay"),
    ("banggood.", "banggood"),
)


def _marketplace_from_url(url: str) -> str:
    low = url.lower()
    for host, slug in _URL_HOSTS:
        if host in low:
            return slug
    return ""


def _marketplace_currency(marketplace: str) -> str:
    return MARKETPLACES.get(marketplace, {}).get("currency", "")


def _parse_target(text: str, marketplace: str
                  ) -> tuple[int, str]:
    """'under ₦200k' / 'under $50' → (minor, currency). (0, '') if none."""
    tm = _TARGET_RE.search(text or "")
    if tm is None:
        return 0, ""
    sym = tm.group(1) or ""
    mult = 1000 if (tm.group(3) or "").lower() == "k" else \
        1000000 if (tm.group(3) or "").lower() == "m" else 1
    try:
        amount = float(tm.group(2).replace(",", "")) * mult
    except ValueError:
        return 0, ""
    if sym == "₦" or sym == "N":
        currency = "NGN"
    elif sym == "$":
        currency = "USD"
    else:
        currency = _marketplace_currency(marketplace)
    return _to_minor(amount, currency), currency


def _clean_product(raw: str) -> str:
    """'this iPhone 15 on Amazon' → 'iPhone 15' (best effort)."""
    text = (raw or "").strip().strip("\"'“”")
    text = _LEAD_RE.sub("", text)
    cut = len(text)
    for sep in (" on ", " under ", " below ", " for ", " tell me",
                " at ", " in "):
        idx = text.lower().find(sep)
        if 0 <= idx < cut:
            cut = idx
    text = text[:cut].strip().strip("\"'“”")
    text = _STRIP_LEAD.sub("", text).strip()
    text = _STRIP_SERIES.sub("", text).strip()
    return text


def _flight_shaped(text: str) -> bool:
    """Two IATA codes + a date = the flight watcher's shape, not ours.

    The product hook runs before the flight hook; a flight-shaped
    line must fall through to it untouched.
    """
    codes = _IATA_RE.findall((text or "").upper())
    return len(codes) >= 2 and bool(_DATE_RE.search(text or ""))


def parse_price_request(text: str) -> dict[str, Any] | None:
    """Parse a retail price-watch / price-history request.

    Returns ``{"kind": "watch"|"history", ...}`` or None when the
    text is not a retail price request. Never raises.
    """
    raw = (text or "").strip()
    if not raw or raw.startswith("/"):
        return None
    low = raw.lower()
    url_m = _URL_RE.search(raw)
    url = url_m.group(0).rstrip(".,)") if url_m else ""
    market_m = _MARKET_RE.search(raw)
    marketplace = (market_m.group(1).lower() if market_m else ""
                   or _marketplace_from_url(url))
    if _flight_shaped(raw):
        return None

    if _HISTORY_RE.search(raw):
        if "price" not in low and not url:
            return None
        product = ""
        if url:
            product = url
        else:
            m = re.search(r"(?:of|for)\s+(.+)$", raw, re.I)
            if m:
                product = _clean_product(m.group(1))
        return {"kind": "history", "product_ref": product,
                "marketplace": marketplace}

    if not re.search(r"\b(watch|track)\b", low):
        return None
    if ("price" not in low and not marketplace and not url
            and "under" not in low and "below" not in low):
        return None
    product = url
    if not product:
        product = _clean_product(raw)
    target_minor, currency = _parse_target(raw, marketplace)
    if _SERIES_NEW_RE.search(raw):
        series = "tp_new"
    elif _SERIES_USED_RE.search(raw):
        series = "tp_used"
    elif marketplace in MARKETPLACES:
        series = "amazon" if marketplace == "amazon" else marketplace
    else:
        series = "amazon"
    return {"kind": "watch", "product_ref": product,
            "marketplace": marketplace, "series": series,
            "target_minor": target_minor, "currency": currency}


# ── formatting + stats ────────────────────────────────────────────────

def format_alert(alert: PriceAlert) -> str:
    prev = _short_minor(alert.previous_minor, alert.currency)
    curr = _short_minor(alert.current_minor, alert.currency)
    market = MARKETPLACES.get(alert.marketplace, {}).get(
        "name", alert.marketplace)
    label = SERIES_LABELS.get(alert.series, alert.series)
    if alert.kind == "stock":
        lines = [
            f"📦 {alert.product_ref} is back in stock on {market} "
            f"({label}) — {curr}.",
        ]
        close = "Want it? Grab it before it sells out again."
    else:
        drop = 0.0
        if alert.previous_minor:
            drop = ((alert.previous_minor - alert.current_minor)
                    / alert.previous_minor * 100)
        lines = [
            f"📉 {alert.product_ref} dropped on {market} ({label}): "
            f"{prev} → {curr} ({drop:.0f}% off).",
        ]
        close = "Want it? Buy before the price moves again."
    if alert.target_minor:
        lines.append(
            f"target was ≤ {_short_minor(alert.target_minor, alert.currency)} ✅")
    lines.append(close)
    return "\n".join(lines)


def format_watches(watches: list[PriceWatch]) -> str:
    if not watches:
        return ("No active price watches. Try 'watch this on Amazon, "
                "tell me under $50'.")
    lines = ["👀 Active price watches:"]
    for w in watches:
        bit = f"• {w.label}"
        if w.last_minor:
            bit += f" — {_short_minor(w.last_minor, w.last_currency)}"
        else:
            bit += " — not checked yet"
        if w.target_minor:
            bit += f" (target ≤ {_short_minor(w.target_minor, w.currency)})"
        lines.append(bit)
    return "\n".join(lines)


def trend_stats(points: list[PricePoint]) -> dict[str, Any]:
    """Min/avg/max + day change from price history. Pure."""
    vals = [p.price_minor for p in points if p.price_minor > 0]
    if not vals:
        return {}
    stats: dict[str, Any] = {
        "n": len(vals),
        "low": min(vals),
        "high": max(vals),
        "avg": sum(vals) / len(vals),
        "latest": vals[-1],
    }
    if len(vals) >= 2:
        prev = vals[-2]
        stats["day_change_pct"] = (
            (vals[-1] - prev) / prev * 100 if prev else 0.0)
    return stats


def format_history(watch: PriceWatch, points: list[PricePoint],
                   *, chart_path: str = "") -> str:
    """Text summary for a price-history answer (chart rides beside it)."""
    stats = trend_stats(points)
    started = ""
    if points:
        started = time.strftime("%Y-%m-%d", time.gmtime(points[0].ts))
    lines = [f"📈 Price history — {watch.label}:"]
    if stats:
        cur = watch.last_currency or watch.currency
        lines.append(
            f"low {_fmt_minor(stats['low'], cur)} · "
            f"avg {_fmt_minor(int(stats['avg']), cur)} · "
            f"high {_fmt_minor(stats['high'], cur)} · "
            f"now {_fmt_minor(stats['latest'], cur)} "
            f"({stats['n']} checks)")
        if "day_change_pct" in stats:
            d = stats["day_change_pct"]
            lines.append(f"moved {d:+.1f}% since the last check")
    else:
        lines.append("no price points yet — the first check is pending.")
    if started:
        lines.append(f"history starts {started} — the day watching "
                     f"began, nothing before it.")
    if chart_path:
        lines.append(f"chart: {chart_path}")
    return "\n".join(lines)

# ── chart ─────────────────────────────────────────────────────────────

_CHART_W, _CHART_H = 1000, 600


def _chart_series(points: list[PricePoint]) -> dict[str, list[PricePoint]]:
    """Points grouped by series, each oldest first, empty series dropped."""
    grouped: dict[str, list[PricePoint]] = {}
    for p in sorted(points, key=lambda p: p.ts):
        grouped.setdefault(p.series, []).append(p)
    return {s: ps for s, ps in grouped.items() if ps}


def _hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _render_chart_matplotlib(
        points: list[PricePoint], *, title: str,
        target_minor: int, currency: str,
        started_at: str) -> ChartResult:
    import datetime

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = _chart_series(points)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    try:
        for idx, (series, ps) in enumerate(sorted(grouped.items())):
            dates = [datetime.datetime.fromtimestamp(p.ts) for p in ps]
            prices = [p.price_minor / 100 for p in ps]
            ax.plot(dates, prices, color=_series_color(series, idx),
                    marker="o", markersize=3,
                    label=_series_label(series))
        if target_minor:
            ax.axhline(target_minor / 100, color="#d62728",
                       linestyle="--", linewidth=1.5,
                       label=f"target {_fmt_minor(target_minor, currency)}")
        ax.set_title(f"{title} — since {started_at}")
        ax.set_ylabel(f"price ({currency or 'minor'})")
        ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        png = buf.getvalue()
    finally:
        plt.close(fig)
    if not png:
        raise ValueError("matplotlib render produced empty PNG")
    return ChartResult(
        png=png, sha256=hashlib.sha256(png).hexdigest(),
        width=_CHART_W, height=_CHART_H,
        series=sorted(grouped), n_points=len(points),
        started_at=started_at, rung="matplotlib", title=title)


def _render_chart_pil(
        points: list[PricePoint], *, title: str,
        target_minor: int, currency: str,
        started_at: str) -> ChartResult:
    from PIL import Image, ImageDraw, ImageFont

    grouped = _chart_series(points)
    img = Image.new("RGB", (_CHART_W, _CHART_H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception as exc:  # noqa: BLE001 - default font is a fallback
        raise ValueError(f"no font for chart: {exc}") from exc
    left, right, top, bottom = 80, 20, 50, 60
    plot_w = _CHART_W - left - right
    plot_h = _CHART_H - top - bottom
    all_prices = [p.price_minor for ps in grouped.values() for p in ps]
    lo = min(all_prices)
    hi = max(all_prices)
    if target_minor:
        lo = min(lo, target_minor)
        hi = max(hi, target_minor)
    span = max(hi - lo, 1)
    lo -= int(span * 0.05)
    hi += int(span * 0.05)
    span = hi - lo

    def _x(i: int, n: int) -> float:
        return left + (plot_w * i / max(n - 1, 1))

    def _y(minor: int) -> float:
        return top + plot_h - (plot_h * (minor - lo) / span)

    draw.text((left, 10), f"{title} — since {started_at}",
              fill="black", font=font)
    for row in range(5):
        gy = top + plot_h * row / 4
        val = hi - span * row / 4
        draw.line([(left, gy), (_CHART_W - right, gy)],
                  fill="#dddddd", width=1)
        draw.text((8, gy - 6), _fmt_minor(int(val), currency),
                  fill="#555555", font=font)
    n = max(len(ps) for ps in grouped.values())
    if target_minor:
        ty = _y(target_minor)
        for x0 in range(left, _CHART_W - right, 12):
            draw.line([(x0, ty), (min(x0 + 6, _CHART_W - right), ty)],
                      fill="#d62728", width=2)
        draw.text((_CHART_W - right - 180, ty - 18),
                  f"target {_fmt_minor(target_minor, currency)}",
                  fill="#d62728", font=font)
    for idx, (series, ps) in enumerate(sorted(grouped.items())):
        color = _hex_to_rgb(_series_color(series, idx))
        pts = [(_x(i, len(ps)), _y(p.price_minor))
               for i, p in enumerate(ps)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        for x0, y0 in pts:
            draw.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], fill=color)
        draw.text((left + idx * 180, _CHART_H - 30),
                  _series_label(series), fill=color, font=font)
    draw.text((left, _CHART_H - 16),
              f"{n} points · {started_at} → today",
              fill="#555555", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    if not png:
        raise ValueError("PIL render produced empty PNG")
    return ChartResult(
        png=png, sha256=hashlib.sha256(png).hexdigest(),
        width=_CHART_W, height=_CHART_H,
        series=sorted(grouped), n_points=len(points),
        started_at=started_at, rung="pil", title=title)


def render_price_chart(points: list[PricePoint], *, title: str,
                       target_minor: int = 0, currency: str = "",
                       started_at: str = "") -> ChartResult:
    """Draw the price-history chart from journalled points.

    Matplotlib first (multi-series lines, target dashed line); PIL
    where matplotlib is absent. The receipt names the rung that
    served. ``started_at`` is the true start date — the first
    recorded point — and is printed on the chart itself. Raises
    ValueError when there is nothing to draw or no renderer works.
    """
    if not points:
        raise ValueError("no price points to chart")
    start = started_at or time.strftime(
        "%Y-%m-%d", time.gmtime(min(p.ts for p in points)))
    try:
        return _render_chart_matplotlib(
            points, title=title, target_minor=target_minor,
            currency=currency, started_at=start)
    except ImportError:
        _log.debug("matplotlib absent for price chart; PIL fallback")
    return _render_chart_pil(
        points, title=title, target_minor=target_minor,
        currency=currency, started_at=start)

# ── persistence ───────────────────────────────────────────────────────

_WATCHES_DDL = """CREATE TABLE IF NOT EXISTS price_watches (
                   id TEXT PRIMARY KEY, product_ref TEXT,
                   marketplace TEXT, series TEXT,
                   target_minor INTEGER DEFAULT 0,
                   threshold_pct REAL DEFAULT 10.0,
                   currency TEXT DEFAULT '',
                   last_minor INTEGER DEFAULT 0,
                   last_currency TEXT DEFAULT '',
                   alerted_minor INTEGER DEFAULT 0,
                   last_in_stock INTEGER DEFAULT -1,
                   active INTEGER DEFAULT 1,
                   created_at REAL, last_checked REAL DEFAULT 0,
                   cooldown_h REAL DEFAULT 24.0)"""
_POINTS_DDL = """CREATE TABLE IF NOT EXISTS price_points (
                   watch_id TEXT NOT NULL, ts REAL NOT NULL,
                   series TEXT NOT NULL, price_minor INTEGER NOT NULL,
                   currency TEXT DEFAULT '',
                   in_stock INTEGER NOT NULL DEFAULT 1)"""
_ALERT_LOG_DDL = """CREATE TABLE IF NOT EXISTS price_alert_log (
                   watch_id TEXT NOT NULL, at REAL NOT NULL,
                   kind TEXT NOT NULL, minor INTEGER NOT NULL)"""
_DDL = ";\n".join((_WATCHES_DDL, _POINTS_DDL, _ALERT_LOG_DDL))


class PriceHistoryStore:
    """Persistent retail price watches + their journalled history.

    When a document store is resolved (Mongo program P5 — an
    explicit ``store`` or the boot ambient), the same API runs over
    the seam instead (registered in storage.sidecars as
    commerce-price-history): watches are documents in
    ``commerce-price-history__price_watches`` by their text id; the
    PK-less price_points / price_alert_log children ride put_raw
    with caller-shaped ids (``<watch_id>:<n>``) allocated by one
    atomic counter per table on the meta document. The file's
    existing rows are imported once, through the same mapping, at
    first construction.
    """

    def __init__(
        self,
        db_path: str = "",
        *,
        fetcher: Callable[[str, str], list[dict[str, Any]] | None] | None = None,
        sender: Callable[..., bool] | None = None,
        now: Callable[[], float] | None = None,
        store: Any = None,
    ) -> None:
        import os
        path = db_path or os.path.expanduser(_DEFAULT_DB)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = None
        self._port: Any = None
        self._fetcher = fetcher
        self._sender = sender
        self._now = now or time.time
        from ..storage.sidecars import sidecar_port

        port = sidecar_port("commerce-price-history", _DDL,
                            ("price_watches",), path, store=store)
        if port is not None:
            imported = {"price_points": 0, "price_alert_log": 0}

            def write_watches(rows: list[dict[str, Any]]) -> int:
                table = port.tables["price_watches"]
                for row in rows:
                    table.put(row)
                return len(rows)

            def _child_writer(table_name: str):
                def write(rows: list[dict[str, Any]]) -> int:
                    ordinals: dict[str, int] = {}
                    for row in rows:
                        watch_id = str(row["watch_id"])
                        ordinal = ordinals.get(watch_id, 0)
                        ordinals[watch_id] = ordinal + 1
                        port.put_raw(table_name,
                                     f"{watch_id}:{ordinal}", dict(row))
                    imported[table_name] = len(rows)
                    return len(rows)
                return write

            def seed_counters(_counts: dict[str, int]) -> None:
                doc = port.store.get(port.meta_collection,
                                     "counters") or {}
                merged = dict(doc)
                for table_name in ("price_points", "price_alert_log"):
                    if int(merged.get(table_name) or 0) \
                            < imported[table_name]:
                        merged[table_name] = imported[table_name]
                if merged != doc:
                    port.store.put(port.meta_collection, "counters",
                                   merged)

            port.ensure_imported(
                {"price_watches": write_watches,
                 "price_points": _child_writer("price_points"),
                 "price_alert_log": _child_writer("price_alert_log")},
                after_import=seed_counters)
            self._port = port
            return
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_WATCHES_DDL)
        self._db.execute(_POINTS_DDL)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pp_watch "
            "ON price_points (watch_id, ts)"
        )
        self._db.execute(_ALERT_LOG_DDL)
        self._db.commit()

    # ── CRUD ──────────────────────────────────────────────────────────

    def watch(
        self,
        product_ref: str,
        marketplace: str,
        *,
        series: str = "",
        target_minor: int = 0,
        threshold_pct: float = 10.0,
        currency: str = "",
        cooldown_hours: float = 24.0,
    ) -> PriceWatch:
        product_ref = (product_ref or "").strip()
        if not product_ref:
            raise ValueError("product_ref must not be empty")
        marketplace = ((marketplace or "amazon").strip().lower())
        if marketplace not in MARKETPLACES:
            raise ValueError(
                f"unknown marketplace {marketplace!r}; "
                f"known: {sorted(MARKETPLACES)}")
        if not series:
            series = ("amazon" if marketplace == "amazon"
                      else marketplace)
        if series not in AMAZON_SERIES and series != marketplace:
            raise ValueError(
                f"series must be one of {list(AMAZON_SERIES)} for "
                f"Amazon, or the marketplace slug ({marketplace!r})")
        w = PriceWatch(
            id="pwatch_" + uuid.uuid4().hex[:8],
            product_ref=product_ref, marketplace=marketplace,
            series=series, target_minor=max(0, int(target_minor or 0)),
            threshold_pct=float(threshold_pct or 10.0),
            currency=currency or _marketplace_currency(marketplace),
            cooldown_h=float(cooldown_hours or 24.0),
            created_at=self._now(),
        )
        if self._port is not None:
            self._port.tables["price_watches"].put({
                "id": w.id, "product_ref": w.product_ref,
                "marketplace": w.marketplace, "series": w.series,
                "target_minor": w.target_minor,
                "threshold_pct": w.threshold_pct,
                "currency": w.currency,
                "cooldown_h": w.cooldown_h,
                "created_at": w.created_at,
            })
            return w
        self._db.execute(
            """INSERT INTO price_watches (id, product_ref, marketplace,
               series, target_minor, threshold_pct, currency,
               cooldown_h, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (w.id, w.product_ref, w.marketplace, w.series,
             w.target_minor, w.threshold_pct, w.currency,
             w.cooldown_h, w.created_at),
        )
        self._db.commit()
        return w

    def unwatch(self, watch_id: str) -> bool:
        if self._port is not None:
            return self._port.tables["price_watches"].update(
                watch_id, {"active": 0})
        cur = self._db.execute(
            "UPDATE price_watches SET active = 0 WHERE id = ?",
            (watch_id,))
        self._db.commit()
        return cur.rowcount > 0

    def list_watches(self, *, active_only: bool = True
                     ) -> list[PriceWatch]:
        if self._port is not None:
            table = self._port.tables["price_watches"]
            rows = (table.find({"active": 1}) if active_only
                    else table.all())
            rows.sort(key=lambda r: (r["created_at"] or 0))
            return [self._row_to_watch(r) for r in rows]
        q = "SELECT * FROM price_watches"
        if active_only:
            q += " WHERE active = 1"
        q += " ORDER BY created_at"
        return [self._row_to_watch(r)
                for r in self._db.execute(q).fetchall()]

    def get(self, watch_id: str) -> PriceWatch | None:
        if self._port is not None:
            r = self._port.tables["price_watches"].get(watch_id)
            return self._row_to_watch(r) if r else None
        r = self._db.execute(
            "SELECT * FROM price_watches WHERE id = ?",
            (watch_id,)).fetchone()
        return self._row_to_watch(r) if r else None

    def find_by_product(self, query: str, marketplace: str = ""
                        ) -> list[PriceWatch]:
        """Watches whose product ref contains the query (either way)."""
        q = (query or "").strip().lower()
        if not q:
            return []
        market = (marketplace or "").strip().lower()
        out = []
        for w in self.list_watches():
            if market and w.marketplace != market:
                continue
            ref = w.product_ref.lower()
            if q in ref or ref in q:
                out.append(w)
        return out

    @staticmethod
    def _row_to_watch(r: Any) -> PriceWatch:
        cols = set(r.keys())
        stock = r["last_in_stock"]
        return PriceWatch(
            id=r["id"], product_ref=r["product_ref"],
            marketplace=r["marketplace"], series=r["series"],
            target_minor=r["target_minor"],
            threshold_pct=r["threshold_pct"],
            currency=r["currency"] or "",
            last_minor=r["last_minor"],
            last_currency=r["last_currency"] or "",
            alerted_minor=r["alerted_minor"],
            last_in_stock=(None if stock in (None, -1)
                           else bool(stock)),
            active=bool(r["active"]),
            created_at=r["created_at"], last_checked=r["last_checked"],
            cooldown_h=(r["cooldown_h"] if "cooldown_h" in cols
                        else 24.0),
        )

    # ── history ───────────────────────────────────────────────────────

    def _child_docs(self, table: str,
                    watch_id: str) -> list[dict[str, Any]]:
        assert self._port is not None
        docs = self._port.find_raw(table, {"watch_id": watch_id})
        docs.sort(key=lambda d: int(str(d["_id"]).rsplit(":", 1)[1]))
        return docs

    def _record_point(self, watch_id: str, point: PricePoint) -> None:
        if self._port is not None:
            ordinal = self._port.next_id("price_points")
            self._port.put_raw("price_points", f"{watch_id}:{ordinal}", {
                "watch_id": watch_id, "ts": point.ts,
                "series": point.series,
                "price_minor": int(point.price_minor),
                "currency": point.currency or "",
                "in_stock": int(bool(point.in_stock))})
            cutoff = point.ts - 120 * 86400
            for doc in self._child_docs("price_points", watch_id):
                if doc["ts"] < cutoff:
                    self._port.store.delete(
                        "commerce-price-history__price_points",
                        doc["_id"])
            return
        self._db.execute(
            "INSERT INTO price_points (watch_id, ts, series, "
            "price_minor, currency, in_stock) VALUES (?, ?, ?, ?, ?, ?)",
            (watch_id, point.ts, point.series, int(point.price_minor),
             point.currency or "", int(bool(point.in_stock))))
        self._db.execute(
            "DELETE FROM price_points WHERE watch_id = ?"
            " AND ts < ?", (watch_id, point.ts - 120 * 86400))

    def history(self, watch_id: str, *, days: int = 60
                ) -> list[PricePoint]:
        """Price points for a watch, oldest first. Never raises."""
        try:
            cutoff = self._now() - days * 86400
            if self._port is not None:
                docs = [d for d in self._child_docs("price_points",
                                                    watch_id)
                        if d["ts"] >= cutoff]
                docs.sort(key=lambda d: d["ts"])
                return [PricePoint(
                    ts=d["ts"], series=d["series"],
                    price_minor=d["price_minor"],
                    currency=d["currency"] or "",
                    in_stock=bool(d["in_stock"]))
                    for d in docs]
            rows = self._db.execute(
                "SELECT ts, series, price_minor, currency, in_stock"
                " FROM price_points"
                " WHERE watch_id = ? AND ts >= ? ORDER BY ts",
                (watch_id, cutoff)).fetchall()
        except Exception:  # noqa: BLE001 - history never raises
            _log.debug("price history read failed", exc_info=True)
            return []
        return [PricePoint(
            ts=r["ts"], series=r["series"],
            price_minor=r["price_minor"], currency=r["currency"] or "",
            in_stock=bool(r["in_stock"])) for r in rows]

    def chart_for(self, watch_id: str, *, days: int = 60
                  ) -> ChartResult | None:
        """Chart artifact from the watch's own journal, or None."""
        w = self.get(watch_id)
        if w is None:
            return None
        points = self.history(watch_id, days=days)
        if not points:
            return None
        try:
            return render_price_chart(
                points, title=w.product_ref,
                target_minor=w.target_minor,
                currency=w.last_currency or w.currency)
        except Exception:  # noqa: BLE001 - a failed chart is None
            _log.debug("price chart failed", exc_info=True)
            return None

    # ── checking ──────────────────────────────────────────────────────

    def _recent_alert(self, watch_id: str, kind: str, now: float,
                      cooldown_h: float) -> bool:
        if cooldown_h <= 0:
            return False
        try:
            if self._port is not None:
                ats = [d["at"] for d in self._child_docs(
                    "price_alert_log", watch_id)
                    if d["kind"] == kind]
                latest = max(ats) if ats else None
                return bool(latest and now - latest < cooldown_h * 3600)
            row = self._db.execute(
                "SELECT MAX(at) AS at FROM price_alert_log"
                " WHERE watch_id = ? AND kind = ?",
                (watch_id, kind)).fetchone()
        except Exception:  # noqa: BLE001 - cooldowns fail open
            _log.debug("price alert cooldown read failed", exc_info=True)
            return False
        return bool(row and row["at"]
                    and now - row["at"] < cooldown_h * 3600)

    def _log_alert(self, watch_id: str, kind: str, minor: int,
                   now: float) -> None:
        try:
            if self._port is not None:
                ordinal = self._port.next_id("price_alert_log")
                self._port.put_raw(
                    "price_alert_log", f"{watch_id}:{ordinal}", {
                        "watch_id": watch_id, "at": now, "kind": kind,
                        "minor": int(minor)})
                return
            self._db.execute(
                "INSERT INTO price_alert_log (watch_id, at, kind, minor)"
                " VALUES (?, ?, ?, ?)", (watch_id, now, kind, int(minor)))
        except Exception:  # noqa: BLE001 - alert log is best effort
            _log.debug("price alert log failed", exc_info=True)

    def _update_seen(self, watch: PriceWatch, current_minor: int,
                     currency: str, in_stock: bool | None) -> None:
        fields: dict[str, Any] = {
            "last_minor": int(current_minor),
            "last_currency": currency or "",
        }
        if in_stock is not None:
            fields["last_in_stock"] = int(bool(in_stock))
        if self._port is not None:
            self._port.tables["price_watches"].update(watch.id, fields)
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._db.execute(
            f"UPDATE price_watches SET {sets} WHERE id = ?",
            (*fields.values(), watch.id))

    def check(self, watch: PriceWatch,
              fetcher: Callable[[str, str], list[dict[str, Any]] | None] | None = None
              ) -> dict[str, Any]:
        """Run one price check. Never raises.

        Returns ``{"ok", "watch_id", "alert", "points",
        "observed_series", "fetch_error"}``. Every observed series
        is journalled; the alert is evaluated on the watch's
        followed series only.
        """
        now = self._now()
        fresh = self.get(watch.id)
        if fresh is not None:
            watch = fresh
        fetch = fetcher if fetcher is not None else self._fetcher
        if self._port is not None:
            self._port.tables["price_watches"].update(
                watch.id, {"last_checked": now})
        else:
            self._db.execute(
                "UPDATE price_watches SET last_checked = ? WHERE id = ?",
                (now, watch.id))
        if fetch is None:
            if self._port is None:
                self._db.commit()
            return {"ok": False, "watch_id": watch.id, "alert": None,
                    "points": 0, "observed_series": [],
                    "fetch_error": "no price source wired — pass a "
                    "fetcher (the price_watch tool supplies one)"}
        try:
            observations = fetch(watch.product_ref, watch.marketplace)
        except Exception as exc:  # noqa: BLE001 - fetch failure is a report
            _log.debug("price fetch failed", exc_info=True)
            if self._port is None:
                self._db.commit()
            return {"ok": False, "watch_id": watch.id, "alert": None,
                    "points": 0, "observed_series": [],
                    "fetch_error": f"price fetch failed: {exc}"}
        if not observations:
            if self._port is None:
                self._db.commit()
            return {"ok": False, "watch_id": watch.id, "alert": None,
                    "points": 0, "observed_series": [],
                    "fetch_error": "no price observed for this product "
                    "on this marketplace"}
        points = 0
        observed: list[str] = []
        followed: dict[str, Any] | None = None
        for obs in observations:
            series = str(obs.get("series") or watch.series)
            price = _obs_minor(obs)
            if price <= 0:
                continue
            currency = str(obs.get("currency") or watch.currency or "")
            in_stock = obs.get("in_stock", True)
            in_stock = bool(in_stock) if in_stock is not None else True
            self._record_point(watch.id, PricePoint(
                ts=now, series=series, price_minor=price,
                currency=currency, in_stock=in_stock))
            points += 1
            observed.append(series)
            if series == watch.series:
                followed = {"price_minor": price, "currency": currency,
                            "in_stock": in_stock}
        if followed is None:
            if self._port is None:
                self._db.commit()
            return {"ok": True, "watch_id": watch.id, "alert": None,
                    "points": points, "observed_series": observed,
                    "fetch_error": f"followed series {watch.series!r} "
                    "was not observed this check — no alert evaluated"}
        current = followed["price_minor"]
        currency = followed["currency"]
        in_stock = followed["in_stock"]
        prev_minor = watch.last_minor
        prev_stock = watch.last_in_stock
        self._update_seen(watch, current, currency, in_stock)
        alert: PriceAlert | None = None
        if prev_minor:
            alert = self._drop_alert(watch, prev_minor, current, currency,
                                     now)
        if alert is None and prev_stock is False and in_stock:
            alert = self._stock_alert(watch, current, currency, now)
        if self._port is None:
            self._db.commit()
        if alert is not None and self._sender is not None:
            with contextlib.suppress(Exception):
                self._sender(alert.text)
        return {"ok": True, "watch_id": watch.id, "alert": alert,
                "points": points, "observed_series": observed,
                "fetch_error": ""}

    def _drop_alert(self, watch: PriceWatch, prev_minor: int,
                    current_minor: int, currency: str, now: float
                    ) -> PriceAlert | None:
        if current_minor >= prev_minor:
            return None
        drop_pct = (prev_minor - current_minor) / prev_minor * 100
        new_low = (not watch.alerted_minor
                   or current_minor < watch.alerted_minor)
        threshold_ok = drop_pct >= watch.threshold_pct
        target_crossed = (bool(watch.target_minor)
                          and current_minor <= watch.target_minor
                          and prev_minor > watch.target_minor)
        target_ok = (not watch.target_minor
                     or current_minor <= watch.target_minor)
        if not ((threshold_ok or target_crossed) and new_low
                and target_ok):
            return None
        # New lows are the alert currency (one alert per drop level);
        # a second tick at the same level stays silent.
        alert = PriceAlert(
            watch_id=watch.id, product_ref=watch.product_ref,
            marketplace=watch.marketplace, series=watch.series,
            previous_minor=prev_minor, current_minor=current_minor,
            currency=currency or watch.currency,
            kind="drop", target_minor=watch.target_minor)
        if self._port is not None:
            self._port.tables["price_watches"].update(
                watch.id, {"alerted_minor": current_minor})
        else:
            self._db.execute(
                "UPDATE price_watches SET alerted_minor = ? WHERE id = ?",
                (current_minor, watch.id))
        self._log_alert(watch.id, "drop", current_minor, now)
        return alert

    def _stock_alert(self, watch: PriceWatch, current_minor: int,
                     currency: str, now: float) -> PriceAlert | None:
        if self._recent_alert(watch.id, "stock", now, watch.cooldown_h):
            return None
        alert = PriceAlert(
            watch_id=watch.id, product_ref=watch.product_ref,
            marketplace=watch.marketplace, series=watch.series,
            previous_minor=watch.last_minor or current_minor,
            current_minor=current_minor, currency=currency,
            kind="stock", target_minor=watch.target_minor)
        self._log_alert(watch.id, "stock", current_minor, now)
        return alert

    def check_all(
        self,
        fetcher: Callable[[str, str], list[dict[str, Any]] | None] | None = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for w in self.list_watches():
            try:
                results.append(self.check(w, fetcher))
            except Exception as exc:  # noqa: BLE001 - one bad watch never kills the beat
                _log.debug("price watch check failed", exc_info=True)
                results.append({"ok": False, "watch_id": w.id,
                                "alert": None, "points": 0,
                                "observed_series": [],
                                "fetch_error": f"check failed: {exc}"})
        return results


def check_all(store: PriceHistoryStore | None = None, **kwargs: Any
              ) -> list[dict[str, Any]]:
    """Host entry point: run every active watch once."""
    s = store or PriceHistoryStore()
    return s.check_all(kwargs.get("fetcher"))


def run_price_history_tick_once(
        *, context: Any = None, notifier: Any = None,
        fetcher: Callable[[str, str], list[dict[str, Any]] | None] | None = None,
        now: float | None = None,
        home: Any = None) -> dict[str, Any]:
    """One price-history tick: check every active watch, publish
    what fired. Never raises. A quiet tick (no watches) makes no
    network call at all — the fetcher is only built when a watch
    exists.
    """
    now_f = float(now if now is not None else time.time())
    try:
        home_path = _resolve_home(home, context)
        store = PriceHistoryStore(
            _db_path_for(home_path), fetcher=fetcher,
            now=lambda: now_f)
        results = store.check_all(fetcher)
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "checked": 0, "alerts": 0,
                "ran_at": now_f, "error": f"tick failed: {exc}"}
    fired = 0
    for result in results:
        alert = result.get("alert")
        if alert is None:
            continue
        fired += 1
        if notifier is None:
            continue
        with contextlib.suppress(Exception):
            notifier.publish(
                "price_history",
                title=f"price alert: {alert.product_ref}",
                body=alert.text)
    return {"ok": True, "checked": len(results), "alerts": fired,
            "ran_at": now_f, "error": ""}


def _resolve_home(home: Any, context: Any) -> str:
    if home:
        return str(home)
    settings = getattr(context, "settings", None) if context else None
    h = getattr(settings, "home", None) if settings else None
    return str(h or "~/.nomorals")


def _db_path_for(home: str) -> str:
    import os
    return os.path.join(os.path.expanduser(home), "commerce",
                        "price_history.db")


def ensure_price_history_job(scheduler: Any, *,
                             spec: str = PRICE_HISTORY_SPEC
                             ) -> Any | None:
    """Register the daily price-history job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"),
                                          job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != PRICE_HISTORY_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            PRICE_HISTORY_JOB, spec, "tool",
            {"tool": "__price_history_tick__", "args": {}}, routine=True)
    except Exception:  # noqa: BLE001 - job registration is best effort
        _log.debug("price history job not registered", exc_info=True)
        return None
