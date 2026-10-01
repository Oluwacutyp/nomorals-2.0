"""FinancialExpert: Devon's conversational trading brain (Prompt 07).

Wraps the Sentinel.py engine (via
:mod:`nomorals.integrations.sentinel_bridge`) in plain-language analysis:
regime briefs, backtest verdicts, directional signals, and side-by-side
comparisons. Everything is research/education — never financial advice.

All Sentinel imports happen lazily inside the bridge; importing this module
must stay cheap (no pandas at module top level).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..integrations import sentinel_bridge as bridge

__all__ = [
    "FinancialExpert",
    "AnalysisReport",
    "BacktestSummary",
    "Signal",
    "Comparison",
    "DISCLAIMER",
]

_log = get_logger(__name__)

DISCLAIMER = (
    "For research and education only — not financial advice. "
    "Past performance does not predict future results."
)

_REGIME_WORDS = {
    "trend": "trending — price is moving with direction and momentum",
    "range": "ranging — price is bouncing between support and resistance",
    "squeeze": "squeezing — volatility is compressed, a breakout is brewing",
    "panic": "panicking — fear is driving sharp, disorderly moves",
}


def _regime_plain(label: str) -> str:
    low = (label or "").strip().lower()
    for key, words in _REGIME_WORDS.items():
        if key in low:
            return words
    return f"in a {label.strip() or 'neutral'} state"


def _top_strategies(ranked: Any, n: int = 3) -> list[dict[str, Any]]:
    """Top-n strategies from the engine's ranked DataFrame, JSON-safe."""
    out: list[dict[str, Any]] = []
    try:
        frame = ranked.head(n) if hasattr(ranked, "head") else ranked
        for name, row in frame.iterrows():
            get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
            out.append({
                "name": str(name),
                "score": round(float(get("score", 0.0) or 0.0), 3),
                "sharpe": round(float(get("sharpe", 0.0) or 0.0), 2),
                "hit_rate": round(float(get("hit_rate", 0.0) or 0.0), 3),
                "turnover": round(float(get("turnover", 0.0) or 0.0), 3),
            })
    except Exception:  # noqa: BLE001 - best effort on odd frames
        pass
    return out


@dataclass
class AnalysisReport:
    id: str = ""
    symbol: str = ""
    market: str = "crypto"
    timeframe: str = "1h"
    bars: int = 0
    regime_label: str = ""
    regime_plain: str = ""
    bias: float = 0.0
    agreement: float = 0.0
    position_now: float = 0.0
    approved: bool = False
    size_fraction: float = 0.0
    stop_distance_pct: float = 0.0
    strategies: list[dict[str, Any]] = field(default_factory=list)
    n_strategies: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "symbol": self.symbol, "market": self.market,
            "timeframe": self.timeframe, "bars": self.bars,
            "regime_label": self.regime_label, "regime_plain": self.regime_plain,
            "bias": self.bias, "agreement": self.agreement,
            "position_now": self.position_now, "approved": self.approved,
            "size_fraction": self.size_fraction,
            "stop_distance_pct": self.stop_distance_pct,
            "strategies": self.strategies, "n_strategies": self.n_strategies,
            "created_at": self.created_at, "disclaimer": DISCLAIMER,
        }

    def summary_text(self) -> str:
        lines = [
            f"{self.symbol} [{self.market} {self.timeframe}] — "
            f"{self.regime_plain} (regime: {self.regime_label.strip()}).",
            f"Engine bias {self.bias:+.2f} with {self.agreement:.0%} strategy "
            f"agreement; current stance "
            f"{'LONG' if self.position_now > 0 else 'SHORT' if self.position_now < 0 else 'FLAT'}"
            f"{' (ML-approved)' if self.approved else ''}.",
        ]
        if self.strategies:
            lines.append("Top strategies for this regime:")
            for i, s in enumerate(self.strategies, 1):
                lines.append(
                    f"  {i}. {s['name']} — score {s['score']}, "
                    f"sharpe {s['sharpe']}, hit-rate {s['hit_rate']:.0%}")
        if self.size_fraction:
            lines.append(
                f"Suggested risk size ~{self.size_fraction:.1%} of equity, "
                f"stop ~{self.stop_distance_pct:.1%} away.")
        lines.append(DISCLAIMER)
        return "\n".join(lines)


@dataclass
class BacktestSummary:
    id: str = ""
    symbol: str = ""
    market: str = "crypto"
    profile: str = "default"
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    passes: bool = False
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "symbol": self.symbol, "market": self.market,
            "profile": self.profile, "metrics": self.metrics,
            "verdict": self.verdict, "passes": self.passes,
            "created_at": self.created_at, "disclaimer": DISCLAIMER,
        }

    def summary_text(self) -> str:
        m = self.metrics
        lines = [
            f"Backtest {self.symbol} [{self.market}, {self.profile}]:",
            f"  return {m.get('total_return', 0):+.1%} | CAGR "
            f"{m.get('cagr', 0):+.1%} | sharpe {m.get('sharpe', 0):.2f} | "
            f"sortino {m.get('sortino', 0):.2f}",
            f"  max drawdown {m.get('max_dd', 0):.1%} | profit factor "
            f"{m.get('profit_factor', 0):.2f} | win rate "
            f"{m.get('win_rate', 0):.0%} | trades {m.get('trades', 0)}",
            f"Verdict: {self.verdict}",
            DISCLAIMER,
        ]
        return "\n".join(lines)


@dataclass
class Signal:
    id: str = ""
    symbol: str = ""
    market: str = "crypto"
    direction: str = "flat"          # bullish | bearish | flat
    confidence: float = 0.0
    invalidation: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "symbol": self.symbol, "market": self.market,
            "direction": self.direction, "confidence": self.confidence,
            "invalidation": self.invalidation,
            "created_at": self.created_at, "disclaimer": DISCLAIMER,
        }

    def summary_text(self) -> str:
        return (
            f"Signal {self.symbol} [{self.market}]: {self.direction.upper()} "
            f"(confidence {self.confidence:.0%}).\n"
            f"{self.invalidation}\n{DISCLAIMER}"
        )


@dataclass
class Comparison:
    symbols: list[str] = field(default_factory=list)
    market: str = "crypto"
    rows: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols, "market": self.market,
            "rows": self.rows, "created_at": self.created_at,
            "disclaimer": DISCLAIMER,
        }

    def summary_text(self) -> str:
        lines = [f"Comparison [{self.market}]:"]
        for r in self.rows:
            lines.append(
                f"  {r['symbol']:14s} {r['regime_label'].strip():10s} "
                f"bias {r['bias']:+.2f} stance {r['stance']:5s} "
                f"top: {r['top_strategy']}")
        lines.append(DISCLAIMER)
        return "\n".join(lines)


class FinancialExpert:
    """The conversational trading brain. Paper-first; live is gated elsewhere."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.settings = getattr(context, "settings", None)

    # ── internals ─────────────────────────────────────────────────────
    def _bar(self) -> dict[str, float]:
        trading = getattr(self.settings, "trading", None)
        get = (lambda name, default: getattr(trading, name, default)
               if trading is not None else default)
        return {
            "min_sharpe": float(get("min_sharpe", 1.0)),
            "max_drawdown_pct": float(get("max_drawdown_pct", 20.0)),
            "min_profit_factor": float(get("min_profit_factor", 1.3)),
        }

    def _room_log(self, text: str) -> None:
        """Best-effort log to the current Prompt-05 room, if any."""
        try:
            rooms = getattr(self.context, "rooms", None)
            room = getattr(self.context, "current_room", None)
            if rooms is not None and room is not None:
                log = getattr(rooms, "append_log", None)
                if callable(log):
                    log(room, text)
        except Exception:  # noqa: BLE001 - logging never breaks analysis
            pass

    # ── public API ────────────────────────────────────────────────────
    def analyze(self, symbol: str, market: str = "crypto",
                timeframe: str = "1h", bars: int = 2000) -> AnalysisReport:
        engine = bridge.get_engine(market)
        df = bridge.load_data(symbol, market, timeframe, bars)
        report = engine.scan(df, symbol=symbol)
        try:
            n_bars = int(getattr(report, "bars", 0) or 0)
        except (TypeError, ValueError):
            n_bars = 0
        if not n_bars:
            try:
                n_bars = len(df)
            except TypeError:
                n_bars = 0
        out = AnalysisReport(
            id=new_id("ta"), symbol=symbol, market=market,
            timeframe=timeframe, bars=n_bars,
            regime_label=str(getattr(report, "regime_label", "")),
            regime_plain=_regime_plain(
                str(getattr(report, "regime_label", ""))),
            bias=float(getattr(report, "bias", 0.0)),
            agreement=float(getattr(report, "agreement", 0.0)),
            position_now=float(getattr(report, "position_now", 0.0)),
            approved=bool(getattr(report, "approved", False)),
            size_fraction=float(getattr(report, "size_fraction", 0.0)),
            stop_distance_pct=float(
                getattr(report, "stop_distance", 0.0)) * 100.0,
            strategies=_top_strategies(getattr(report, "ranked", None)),
            n_strategies=int(getattr(report, "n_strategies", 0)),
            created_at=time.time(),
        )
        self._room_log(f"analyze {symbol}: {out.regime_label.strip()} "
                       f"bias={out.bias:+.2f}")
        _log.info("financial_expert.analyze %s -> %s", symbol,
                  out.regime_label.strip())
        return out

    def backtest(self, symbol: str, market: str = "crypto",
                 strategy: str | None = None,
                 profile: str = "default") -> BacktestSummary:
        engine = bridge.get_engine(market, profile)
        df = bridge.load_data(symbol, market, "1h", 2000)
        raw = engine.backtest(df)
        metrics = {
            k: raw.get(k) for k in (
                "final_equity", "total_return", "cagr", "sharpe", "sortino",
                "max_dd", "profit_factor", "win_rate", "trades", "turnover",
                "exposure")
        }
        bar = self._bar()
        checks = [
            (f"sharpe ≥ {bar['min_sharpe']}",
             float(metrics.get("sharpe") or 0) >= bar["min_sharpe"]),
            (f"max drawdown ≤ {bar['max_drawdown_pct']}%",
             abs(float(metrics.get("max_dd") or 0)) * 100
             <= bar["max_drawdown_pct"]),
            (f"profit factor ≥ {bar['min_profit_factor']}",
             float(metrics.get("profit_factor") or 0)
             >= bar["min_profit_factor"]),
        ]
        passes = all(ok for _, ok in checks)
        verdict = (
            f"{'PASSES' if passes else 'FAILS'} the bar: "
            + ", ".join(f"{name} {'✓' if ok else '✗'}"
                        for name, ok in checks))
        if strategy:
            verdict += f" (strategy filter {strategy!r} not applied — " \
                       f"engine backtests the fused strategy mix)"
        out = BacktestSummary(
            id=new_id("tb"), symbol=symbol, market=market, profile=profile,
            metrics={k: (round(float(v), 4) if isinstance(v, (int, float))
                         else v) for k, v in metrics.items()},
            verdict=verdict, passes=passes, created_at=time.time(),
        )
        self._room_log(f"backtest {symbol}: {verdict}")
        _log.info("financial_expert.backtest %s passes=%s", symbol, passes)
        return out

    def signal(self, symbol: str, market: str = "crypto") -> Signal:
        rep = self.analyze(symbol, market=market, bars=600)
        pos = rep.position_now
        direction = ("bullish" if pos > 0 else
                     "bearish" if pos < 0 else "flat")
        confidence = min(0.95, abs(rep.bias) * 0.5 + rep.agreement * 0.5)
        # Invalidation from the stop distance: price from the last scan.
        invalidation = (
            f"{direction.capitalize()} while the regime holds; "
            f"invalid below a {rep.stop_distance_pct:.1f}% adverse move "
            f"(engine stop distance)."
            if direction != "flat" else
            "No directional edge right now — agreement too low or "
            "ML gate not approving. Waiting is the position.")
        return Signal(
            id=new_id("ts"), symbol=symbol, market=market,
            direction=direction, confidence=round(confidence, 2),
            invalidation=invalidation, created_at=time.time(),
        )

    def compare(self, symbols: list[str],
                market: str = "crypto") -> Comparison:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                rep = self.analyze(symbol, market=market, bars=600)
                pos = rep.position_now
                rows.append({
                    "symbol": symbol,
                    "regime_label": rep.regime_label,
                    "bias": round(rep.bias, 3),
                    "stance": ("LONG" if pos > 0 else
                               "SHORT" if pos < 0 else "FLAT"),
                    "top_strategy": (rep.strategies[0]["name"]
                                     if rep.strategies else "—"),
                    "ok": True,
                })
            except Exception as exc:  # noqa: BLE001 - one bad symbol ≠ no table
                rows.append({"symbol": symbol, "ok": False,
                             "error": str(exc)[:160]})
        return Comparison(symbols=list(symbols), market=market, rows=rows,
                          created_at=time.time())
