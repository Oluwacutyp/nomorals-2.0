"""Committee analysis pipeline: bars in, trade-ready read-out.

The native replacement for Sentinel's QuantumEngine scan, built from the
ported parts: clean → regime detect → run the canonical strategy zoo →
fuse the committee → rank strategies → meta-gate approval → risk sizing
and ATR stop ladder.

``analyze(df, profile)`` returns a plain dict with everything
``FinancialExpert`` needs; it raises ``ValueError`` on unusable data and
never touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.logging_setup import get_logger
from .data import clean_ohlcv
from .math import atr, ensure_ohlcv
from .meta import MetaGate, labeled_matrix, sklearn_available
from .regime import RegimeDetector
from .risk import PROFILES, RiskManager
from .signals import fuse_all
from .strategies import (STRATEGIES, list_strategies, rank_strategies,
                         run_zoo)

_log = get_logger(__name__)

__all__ = ["analyze", "committee_position", "PROFILES",
           "regime_kind_weights", "regime_vote_alignment"]


def regime_kind_weights(regime: dict) -> dict:
    """Per-kind committee weights adapted to the detected regime.

    Trend/momentum strategies earn their weight in trends, mean-reversion
    and reversal strategies in ranges, breakout/squeeze strategies when
    volatility is compressing. ``fuse_all`` normalizes these, so only the
    ratios matter. Keeps a floor on every kind — regimes are probabilistic
    and the committee should never go fully deaf to dissent.
    """
    p_trend = float(regime.get("p_trend", 0.0))
    p_range = float(regime.get("p_range", 0.0))
    p_squeeze = float(regime.get("p_squeeze", 0.0))
    return {
        "trend": 0.4 + 1.6 * p_trend,
        "momentum": 0.4 + 1.4 * p_trend,
        "breakout": 0.4 + 0.8 * p_trend + 1.2 * p_squeeze,
        "squeeze": 0.4 + 1.4 * p_squeeze,
        "meanrev": 0.4 + 1.6 * p_range,
        "reversal": 0.4 + 1.2 * p_range,
        "confluence": 1.0,
    }


_STRATEGY_KIND_LOOKUP = {name: cls.kind for name, cls in STRATEGIES.items()}


def regime_vote_alignment(frames: dict, regime_label: str,
                          p_trend: float) -> dict:
    """Discount counter-regime vote directions in strong trends.

    Returns adjusted frames (copies): in a ``TREND_UP``/``TREND_DOWN``
    regime, votes pointing against the trend have their confidence scaled
    by ``max(0.2, 1 - p_trend)`` — trends persist, so counter-trend
    signals (pullback flips, early fades, mid-trend divergences) should
    whisper, not shout. Range/squeeze/panic regimes are untouched: there
    the committee votes at full weight. Signals of 0 (abstain) are never
    touched.
    """
    direction = {"TREND_UP": 1.0, "TREND_DOWN": -1.0}.get(
        (regime_label or "").upper())
    if direction is None:
        return frames
    factor = max(0.2, 1.0 - float(p_trend))
    out = {}
    for name, frame in frames.items():
        d = np.sign(frame["signal"].to_numpy(dtype=float))
        scale = np.where(d == 0.0, 1.0,
                         np.where(d == direction, 1.0, factor))
        adj = frame.copy()
        adj["confidence"] = frame["confidence"] * scale
        out[name] = adj
    return out


def committee_position(df: pd.DataFrame, names: list[str] | None = None,
                       min_agreement: float = 0.0) -> pd.Series:
    """Fused committee position series for a set of strategy names."""
    df = ensure_ohlcv(df)
    names = names or list_strategies()
    frames = run_zoo(names, df)
    if not frames:
        return pd.Series(0.0, index=df.index, name="position")
    return fuse_all(frames, min_agreement=min_agreement)["position"]


def _meta_approval(vote: pd.Series, agreement: pd.Series,
                   regime_frame: pd.DataFrame, close: pd.Series,
                   threshold: float = 0.55,
                   min_agreement: float = 0.50,
                   enter_threshold: float = 0.10) -> tuple[bool, str]:
    """ML veto when sklearn is present, rule fallback otherwise.

    Returns (approved, method) where method is "meta-gate" or "rule".
    """
    if sklearn_available():
        try:
            X, y, mask = labeled_matrix(vote, agreement, regime_frame, close)
            gate = MetaGate(threshold=threshold).fit(X[mask], y[mask])
            if gate.model is not None:
                ok = bool(gate.approve(X.iloc[[-1]])[0])
                return ok, "meta-gate"
        except Exception as e:
            _log.warning("meta-gate failed, falling back to rule-based approval: %s", e)
    # Rule fallback: the committee agrees AND the vote clears the cost gate.
    ok = bool(agreement.iloc[-1] >= min_agreement
              and abs(vote.iloc[-1]) >= enter_threshold)
    return ok, "rule"


def analyze(df: pd.DataFrame, profile: str = "default",
            strategies: list[str] | None = None) -> dict:
    """Full committee analysis of OHLCV bars.

    ``profile`` is default|aggressive|conservative (risk sizing + gates).
    Returns a dict with: regime_label, regime (snapshot), bias, agreement,
    position_now, approved, approval_method, size_fraction, stop_distance_pct,
    stops, entry, atr, ranked (DataFrame), n_strategies, bars, atr_pct,
    enter_threshold.
    """
    profile = (profile or "default").strip().lower()
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    cfg = PROFILES[profile]
    df = clean_ohlcv(df)
    if len(df) < 30:
        raise ValueError(f"need at least 30 bars, got {len(df)}")

    detector = RegimeDetector()
    regime_frame = detector.fit(df)
    last = regime_frame.iloc[-1]
    regime_label = str(last["label"])
    regime = {
        "label": regime_label,
        "code": int(last["regime"]),
        "p_trend": float(last["p_trend"]),
        "p_range": float(last["p_range"]),
        "p_squeeze": float(last["p_squeeze"]),
        "p_panic": float(last["p_panic"]),
        "trend_r2": float(last["trend_r2"]),
        "vol_ratio": float(last["vol_ratio"]),
    }

    names = strategies or list_strategies()
    frames = run_zoo(names, df)
    frames = regime_vote_alignment(frames, regime_label,
                                   float(last["p_trend"]))
    close = df["close"].astype(float)
    a = atr(df, 14)
    atr_last = float(a.iloc[-1])
    entry = float(close.iloc[-1])
    atr_pct = atr_last / (entry + 1e-12)

    if not frames:
        rm = RiskManager(cfg)
        return {
            "regime_label": regime_label, "regime": regime,
            "bias": 0.0, "agreement": 0.0, "position_now": 0.0,
            "approved": False, "approval_method": "none",
            "size_fraction": 0.0, "stop_distance_pct": 0.0,
            "stops": rm.stop_levels(1, entry, atr_last),
            "entry": entry, "atr": atr_last, "atr_pct": atr_pct,
            "ranked": rank_strategies({}, close), "n_strategies": 0,
            "bars": len(df), "enter_threshold": 0.0,
            "strategies": [],
        }

    fused = fuse_all(frames, cost_bps=float(cfg["cost_bps"]),
                     atr_pct=atr_pct,
                     min_agreement=float(cfg["min_agreement"]),
                     kind_weights=regime_kind_weights(regime),
                     lookup=_STRATEGY_KIND_LOOKUP)
    blend = fused["blend"]
    vote = blend["vote"]
    agreement = blend["agreement"]
    position = fused["position"]
    bias = float(vote.iloc[-1])
    agr = float(agreement.iloc[-1])
    position_now = float(position.iloc[-1])
    enter_thr = float(fused["enter_threshold"])

    ranked = rank_strategies(frames, close)
    approved, method = _meta_approval(vote, agreement, regime_frame, close,
                                     min_agreement=float(cfg["min_agreement"]),
                                     enter_threshold=enter_thr)

    rm = RiskManager(cfg)
    side = int(np.sign(position_now)) or int(np.sign(bias)) or 1
    stops = rm.stop_levels(side, entry, atr_last)
    sizing = rm.size_position(100_000.0, entry, atr_last)
    stop_distance_pct = abs(entry - stops["stop"]) / (entry + 1e-12) * 100.0

    return {
        "regime_label": regime_label, "regime": regime,
        "bias": round(bias, 4), "agreement": round(agr, 4),
        "position_now": position_now,
        "approved": approved, "approval_method": method,
        "size_fraction": float(sizing["fraction"]),
        "stop_distance_pct": float(stop_distance_pct),
        "stops": {k: float(v) for k, v in stops.items()},
        "entry": entry, "atr": atr_last, "atr_pct": atr_pct,
        "ranked": ranked, "n_strategies": len(frames),
        "bars": len(df), "enter_threshold": enter_thr,
        "strategies": list(frames),
        # NOTE: stop_distance_pct is a percent number (1.03 == 1.03%).
    }
