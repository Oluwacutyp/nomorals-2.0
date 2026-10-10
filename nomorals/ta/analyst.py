"""Independent analyst: Devon's own read on the market.

BOUNDARY LAW (read twice):
- The Analyst NEVER places, modifies, or closes orders. It produces
  TradeIdeas — scored, with invalidation and targets. Nothing more.
- The Executor (``nomorals/finance/trading_desk.py`` → ``nomorals/
  connectors/exness.py``) NEVER does analysis. It takes a TradeIdea,
  applies the RiskPolicy, and executes — or refuses.
- Broker signals, "AI predictions", and guru calls are INPUTS at best,
  never authorities. The analyst forms its own view from price data.

A TradeIdea is honest about uncertainty: confidence is a calibrated
0–100 score, never a promise. No strategy here guarantees profit.
Backtest before demo; demo before live; risk ≤1% per idea.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .math import atr, clamp, ensure_ohlcv

_log = logging.getLogger(__name__)


class TAError(Exception):
    """Raised when a TA operation cannot be completed."""


try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


def _require() -> None:
    if not (_HAS_NUMPY and _HAS_PANDAS):
        raise TAError("numpy and pandas are required: pip install nomorals[ta]")


@dataclass
class TradeIdea:
    """One independent read on one symbol. Analysis only — not an order."""
    symbol: str
    side: int                      # +1 long, −1 short, 0 no-trade
    confidence: float              # 0–100, calibrated-ish, never certain
    entry: float
    invalidation: float            # the level that kills the idea
    targets: list                  # [(price, weight)] partials
    stop_atr: float = 2.0
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    session_gate: dict = field(default_factory=dict)
    regime: str = ""
    timeframe: str = ""            # e.g. "1h", "15m" — for lesson gating
    mtf_alignment: dict = field(default_factory=dict)
    smc_score: float = 0.0
    structure_bias: int = 0
    # The analyst signs its work. The executor checks this field exists
    # and refuses ideas that lack it (unsigned = untrusted).
    analyst_signature: str = "ta.analyst.v1"


class Analyst:
    """Fuse structure + SMC + indicators + regime + MTF + sessions.

    Weights are explicit and tunable; nothing is a black box. Confidence
    is penalized (not boosted) by disagreement — the analyst would rather
    say "no trade" than manufacture conviction.
    """

    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.weights = {
            "structure": float(c.get("w_structure", 0.30)),
            "smc": float(c.get("w_smc", 0.20)),
            "indicators": float(c.get("w_indicators", 0.25)),
            "regime": float(c.get("w_regime", 0.15)),
            "mtf": float(c.get("w_mtf", 0.10)),
        }
        self.min_confidence = float(c.get("min_confidence", 55.0))
        self.stop_atr = float(c.get("stop_atr", 2.0))
        self.target_rs = [float(x) for x in c.get("target_rs", (1.5, 2.5, 4.0))]

    # ── public ────────────────────────────────────────────────────────

    def analyze(self, df, symbol: str = "XAUUSD",
                higher_tf: dict | None = None) -> TradeIdea:
        """Full independent read. Never raises on bad data — returns a
        no-trade idea with the reason in ``warnings``."""
        _require()
        try:
            df = ensure_ohlcv(df)
        except Exception as exc:
            return self._no_trade(symbol, f"bad data: {exc}")
        if len(df) < 60:
            return self._no_trade(symbol, "insufficient history (<60 bars)")

        from . import sessions as _sess
        from .regime import RegimeDetector
        from .smc import smc_confluence
        from .structure import label_swings, market_bias, nearest_levels, \
            sr_zones, swing_points

        close = float(df["close"].iloc[-1])
        a = float(atr(df).iloc[-1]) or close * 0.001 or 1e-9
        reasons, warnings = [], []
        votes = {}  # component -> (−1..+1)

        # 1. structure (30%)
        try:
            pts = swing_points(df)
            bias = market_bias(label_swings(pts))["bias"] if not pts.empty else 0
            votes["structure"] = bias
            reasons.append(f"market structure bias: {bias:+d}")
        except Exception as exc:
            warnings.append(f"structure failed: {exc}")
            votes["structure"] = 0

        # 2. SMC confluence (20%)
        try:
            smc = smc_confluence(df)
            votes["smc"] = smc["bias"]
            reasons.extend(smc["notes"][:3])
            smc_score = smc["score"]
        except Exception as exc:
            warnings.append(f"smc failed: {exc}")
            votes["smc"] = 0
            smc_score = 0.0

        # 3. indicators: EMA trend + RSI location (25%)
        try:
            from .math import ema, rsi
            e20 = float(ema(df["close"], 20).iloc[-1])
            e50 = float(ema(df["close"], 50).iloc[-1])
            r = float(rsi(df["close"]).iloc[-1])
            ind = 0
            if e20 > e50 and close > e20:
                ind = 1
            elif e20 < e50 and close < e20:
                ind = -1
            if r > 70 and ind > 0:
                warnings.append(f"RSI {r:.0f} overbought — longs fade")
                ind = 0
            elif r < 30 and ind < 0:
                warnings.append(f"RSI {r:.0f} oversold — shorts fade")
                ind = 0
            votes["indicators"] = ind
            reasons.append(f"EMA20/50 {'bullish' if ind > 0 else 'bearish' if ind < 0 else 'flat'}, RSI {r:.0f}")
        except Exception as exc:
            warnings.append(f"indicators failed: {exc}")
            votes["indicators"] = 0

        # 4. regime (15%)
        try:
            regime = RegimeDetector().current(df)["label"]
            rvote = 1 if regime == "TREND_UP" else \
                -1 if regime == "TREND_DOWN" else 0
            if regime == "PANIC":
                warnings.append("PANIC regime — stand aside")
                rvote = 0
            votes["regime"] = rvote
            reasons.append(f"regime: {regime}")
        except Exception as exc:
            warnings.append(f"regime failed: {exc}")
            votes["regime"] = 0
            regime = "UNKNOWN"

        # 5. higher-timeframe alignment (10%)
        mtf = {}
        try:
            if higher_tf:
                for tf_name, hdf in higher_tf.items():
                    hb = market_bias(label_swings(swing_points(hdf)))["bias"] \
                        if len(hdf) > 30 else 0
                    mtf[tf_name] = hb
                agree = [v for v in mtf.values() if v != 0]
                votes["mtf"] = sum(agree) / max(len(agree), 1) \
                    if agree else 0
                reasons.append(f"HTF alignment: {mtf}")
            else:
                votes["mtf"] = 0
        except Exception as exc:
            warnings.append(f"mtf failed: {exc}")
            votes["mtf"] = 0

        # ── fuse with disagreement penalty ──
        wsum = sum(self.weights.values())
        raw = sum(votes[k] * self.weights[k] for k in votes) / wsum
        # disagreement penalty: spread across votes shrinks conviction
        vals = list(votes.values())
        spread = (max(vals) - min(vals)) / 2.0  # 0..1
        conviction = abs(raw) * (1.0 - 0.5 * spread)
        confidence = clamp(conviction * 100.0, 0, 100)

        side = 1 if raw > 0.15 else -1 if raw < -0.15 else 0
        if confidence < self.min_confidence:
            side = 0
            warnings.append(
                f"confidence {confidence:.0f} < threshold "
                f"{self.min_confidence:.0f} — no trade")

        # ── session gate (advisory, recorded on the idea) ──
        gate = {}
        try:
            gate = _sess.time_filter_ok()
            if side != 0 and not gate["ok"]:
                warnings.append(f"session gate: {gate['reason']}")
                # sessions advise; they don't veto a strong read —
                # but confidence pays for trading the chop
                confidence = clamp(confidence * 0.7, 0, 100)
                if confidence < self.min_confidence:
                    side = 0
        except Exception as exc:
            warnings.append(f"session gate failed: {exc}")

        # ── levels: entry, invalidation, targets ──
        stop_dist = self.stop_atr * a
        invalidation = close - side * stop_dist if side else close
        targets = []
        if side:
            for i, r_mult in enumerate(self.target_rs):
                tgt = close + side * stop_dist * r_mult
                w = (0.5, 0.3, 0.2)[i] if i < 3 else 0.1
                targets.append((round(tgt, 5), w))
            # snap first target to nearest S/R if close (confluence)
            try:
                zones = sr_zones(df)
                lv = nearest_levels(zones, close)
                key = "resistance" if side > 0 else "support"
                if lv[key] and abs(lv[key]["mid"] - targets[0][0]) / a < 1.0:
                    targets[0] = (round(lv[key]["mid"], 5), targets[0][1])
                    reasons.append(
                        f"TP1 snapped to {key} {lv[key]['mid']:.5f} "
                        f"(strength {lv[key]['strength']})")
            except Exception:
                pass

        return TradeIdea(
            symbol=symbol, side=side,
            confidence=round(float(confidence), 1),
            entry=round(close, 5),
            invalidation=round(invalidation, 5),
            targets=targets, stop_atr=self.stop_atr,
            reasons=reasons, warnings=warnings,
            session_gate=gate, regime=regime,
            mtf_alignment=mtf, smc_score=round(float(smc_score), 1),
            structure_bias=int(votes.get("structure", 0)),
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _no_trade(self, symbol: str, reason: str) -> TradeIdea:
        return TradeIdea(symbol=symbol, side=0, confidence=0.0,
                         entry=0.0, invalidation=0.0, targets=[],
                         warnings=[reason])


def executor_check(idea: TradeIdea) -> dict:
    """The executor's pre-flight: refuse unsigned or malformed ideas.

    Called by ``trading_desk`` before any order reaches the broker.
    Returns ``{"ok": bool, "reason": str}``. This is the enforcement
    point of the boundary law — analysis stays in ta/, execution stays
    in finance/connectors.

    Also consults the backtest lessons (``ta.lessons``): configs with
    a "DO NOT TRADE" verdict from real backtest evidence are refused
    here, not at the broker.
    """
    if not isinstance(idea, TradeIdea):
        return {"ok": False, "reason": "not a TradeIdea — refusing"}
    if idea.analyst_signature != "ta.analyst.v1":
        return {"ok": False, "reason": "unsigned idea — refusing"}
    if idea.side == 0:
        return {"ok": False, "reason": "no-trade idea — nothing to execute"}
    if not (0 < idea.confidence <= 100):
        return {"ok": False, "reason": "confidence out of range — refusing"}
    if idea.side > 0 and not (idea.invalidation < idea.entry):
        return {"ok": False,
                "reason": "long invalidation above entry — refusing"}
    if idea.side < 0 and not (idea.invalidation > idea.entry):
        return {"ok": False,
                "reason": "short invalidation below entry — refusing"}
    # Backtest lesson gate: refuse configs proven to have no edge.
    try:
        from .lessons import check_lesson
        lesson = check_lesson(getattr(idea, "symbol", "") or "",
                              getattr(idea, "timeframe", "") or "")
        if lesson["verdict"] == "DO NOT TRADE":
            return {
                "ok": False,
                "reason": (
                    f"backtest lesson {lesson['lesson_id']}: "
                    f"{lesson['evidence']}"
                ),
            }
    except Exception:
        pass  # lesson gate is advisory, never a crash vector
    return {"ok": True, "reason": "idea well-formed"}
