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


# ── lazy optional deps ──────────────────────────────────────────────────
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


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise TAError("numpy is required for this operation: pip install nomorals[ta]")


def _require_pandas() -> None:
    if not _HAS_PANDAS:
        raise TAError("pandas is required for this operation: pip install nomorals[ta]")


from ..core.logging_setup import get_logger
from .data import clean_ohlcv
from .math import atr, ensure_ohlcv
from .meta import MetaGate, labeled_matrix, sklearn_available
from .regime import RegimeDetector, regime_playbook
from .risk import PROFILES, RiskManager, position_size
from .signals import explain_vote, fuse_all, vote_quality
from .strategies import (STRATEGIES, list_strategies, rank_strategies,
                         run_zoo)

_log = get_logger(__name__)

__all__ = ["analyze", "committee_position", "PROFILES",
           "regime_kind_weights", "regime_vote_alignment",
           # ── sweep additions: presentation + trade plans + MTF ──
           "render_report", "trade_plan", "analyze_mtf",
           "explain_committee"]


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
        d = _np.sign(frame["signal"].to_numpy(dtype=float))
        scale = _np.where(d == 0.0, 1.0,
                         _np.where(d == direction, 1.0, factor))
        adj = frame.copy()
        adj["confidence"] = frame["confidence"] * scale
        out[name] = adj
    return out


def committee_position(df: _pd.DataFrame, names: list[str] | None = None,
                       min_agreement: float = 0.0) -> _pd.Series:
    """Fused committee position series for a set of strategy names."""
    df = ensure_ohlcv(df)
    names = names or list_strategies()
    frames = run_zoo(names, df)
    if not frames:
        return _pd.Series(0.0, index=df.index, name="position")
    return fuse_all(frames, min_agreement=min_agreement)["position"]


def _meta_approval(vote: _pd.Series, agreement: _pd.Series,
                   regime_frame: _pd.DataFrame, close: _pd.Series,
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


def analyze(df: _pd.DataFrame, profile: str = "default",
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
    side = int(_np.sign(position_now)) or int(_np.sign(bias)) or 1
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
        # Sweep additions (kept private-ish; render via render_report):
        "_frames": frames,
        "_blend": blend,
        "_kind_weights": regime_kind_weights(regime),
    }


# ── sweep additions ──────────────────────────────────────────────────────

def explain_committee(result: dict, top: int = 8) -> _pd.DataFrame:
    """Per-strategy contribution table from an ``analyze()`` result."""
    frames = result.get("_frames") or {}
    blend = result.get("_blend")
    if not frames or blend is None or blend.empty:
        return _pd.DataFrame()
    table = explain_vote(frames, blend,
                         kind_weights=result.get("_kind_weights"),
                         lookup=_STRATEGY_KIND_LOOKUP)
    return table.head(max(1, int(top)))


def trade_plan(df: _pd.DataFrame, profile: str = "default",
               equity: float = 100_000.0,
               strategies: list[str] | None = None) -> dict:
    """One dict with everything needed to place the trade.

    Entry, side, size (shares + fraction), ATR stop ladder, dollar risk,
    invalidation (regime flip level), and the committee's reasoning.
    """
    res = analyze(df, profile=profile, strategies=strategies)
    side = int(_np.sign(res["position_now"])) or int(_np.sign(
        res["bias"])) or 0
    entry = res["entry"]
    atr_v = res["atr"]
    sizing = position_size(equity, entry, atr_v, profile=profile, side=side)
    risk_dollars = (sizing["fraction"] * equity
                    * res["stop_distance_pct"] / 100.0)
    playbook = regime_playbook().get(res["regime_label"], {})
    return {
        "side": side,
        "side_name": "LONG" if side > 0 else "SHORT" if side < 0 else "FLAT",
        "entry": entry,
        "shares": sizing["shares"],
        "size_fraction": sizing["fraction"],
        "notional": sizing["notional"],
        "stops": sizing["stops"],
        "stop_distance_pct": res["stop_distance_pct"],
        "risk_dollars": float(risk_dollars),
        "risk_pct_equity": float(risk_dollars / max(equity, 1e-12) * 100.0),
        "approved": res["approved"],
        "approval_method": res["approval_method"],
        "regime": res["regime_label"],
        "bias": res["bias"],
        "agreement": res["agreement"],
        "playbook": playbook,
        "profile": sizing["profile"],
    }


def analyze_mtf(df: _pd.DataFrame, rules: tuple = ("4h", "1D"),
                profile: str = "default",
                strategies: list[str] | None = None) -> dict:
    """Multi-timeframe analysis: higher TF gives permission, base gives timing.

    Elder's Triple Screen, natively: the base timeframe is analyzed as
    usual, and each higher timeframe contributes its regime. If a higher
    TF regime opposes the base bias, the bias is discounted (not vetoed
    outright — the higher TF is slower, not smarter).
    """
    from .data import multi_timeframe

    base = analyze(df, profile=profile, strategies=strategies)
    mtf = multi_timeframe(df, rules=rules)
    detector = RegimeDetector()
    tf_regimes = {}
    for rule, frame in mtf.items():
        if rule == "base" or len(frame) < 30:
            continue
        try:
            tf_regimes[rule] = detector.current(frame)["label"]
        except Exception as e:
            _log.debug("mtf regime for %s failed: %s", rule, e)
            continue
    bias = base["bias"]
    base_label = base["regime_label"]
    discounts = []
    for rule, label in tf_regimes.items():
        opposed = ((label == "TREND_UP" and bias < 0)
                   or (label == "TREND_DOWN" and bias > 0)
                   or (label == "PANIC" and bias != 0))
        if opposed:
            discounts.append(rule)
    discount = 0.5 ** len(discounts)
    base["tf_regimes"] = tf_regimes
    base["tf_discounts"] = discounts
    base["bias_mtf"] = round(bias * discount, 4)
    base["position_now_mtf"] = base["position_now"] * discount
    base["mtf_aligned"] = not discounts
    return base


def _meter(x: float, width: int = 21) -> str:
    """Unicode bias meter: ──────●────── style, ● marks the value."""
    x = max(-1.0, min(1.0, float(x)))
    pos = int(round((x + 1.0) / 2.0 * (width - 1)))
    bar = ["─"] * width
    bar[width // 2] = "┼"
    bar[pos] = "●"
    return "".join(bar)


def _bias_word(bias: float) -> str:
    a = abs(bias)
    if a < 0.05:
        return "NEUTRAL"
    word = "BULLISH" if bias > 0 else "BEARISH"
    strength = "slightly " if a < 0.25 else "" if a < 0.6 else "strongly "
    return f"{strength}{word}".upper()


def render_report(result: dict, theme: str = "rich",
                  symbol: str = "") -> str:
    """The committee briefing, made to be read in chat.

    Regime banner, unicode bias meter, per-strategy contribution table,
    trade-plan box, risk readout. ``theme``: "rich" (box-drawing) or
    "plain" (ASCII). Numbers are rounded for humans; the raw dict keeps
    full precision.
    """
    rich = theme != "plain"
    H, V = ("═", "║") if rich else ("=", "|")
    top = "╔" + H * 64 + "╗" if rich else "+" + "=" * 64 + "+"
    mid = "╠" + H * 64 + "╣" if rich else "+" + "=" * 64 + "+"
    bot = "╚" + H * 64 + "╝" if rich else "+" + "=" * 64 + "+"

    def row(label: str, val: str) -> str:
        return f"{V} {label:<30} {val:>30} {V}"

    title = f"DEVON TA BRIEFING{f' — {symbol}' if symbol else ''}"
    L = [top, row(title[:60], f"{result.get('bars', 0)} bars")]
    L.append(mid)

    # ── regime banner ──
    reg = result.get("regime", {})
    L.append(row("REGIME", result.get("regime_label", "?")))
    L.append(row("P(trend) / P(range)",
                 f"{reg.get('p_trend', 0):.0%} / {reg.get('p_range', 0):.0%}"))
    L.append(row("P(squeeze) / P(panic)",
                 f"{reg.get('p_squeeze', 0):.0%} / {reg.get('p_panic', 0):.0%}"))
    pb = regime_playbook().get(result.get("regime_label", ""), {})
    if pb:
        note = pb.get("note", "")
        L.append(row("Playbook",
                     f"favor {', '.join(pb.get('favor', []))}"[:30]))
        # Wrap the note across full-width lines instead of truncating.
        words, line = note.split(), ""
        for w_ in words:
            if len(line) + len(w_) + 1 > 60:
                L.append(f"{V} {line:<62} {V}")
                line = w_
            else:
                line = (line + " " + w_).strip()
        if line:
            L.append(f"{V} {line:<62} {V}")
    L.append(mid)

    # ── bias meter ──
    bias = result.get("bias", 0.0)
    L.append(row("COMMITTEE BIAS", _bias_word(bias)))
    L.append(row(" ", _meter(bias)))
    L.append(row("Vote / agreement",
                 f"{bias:+.3f} / {result.get('agreement', 0):.0%}"))
    L.append(row("Position now",
                 f"{result.get('position_now', 0):+.0f}  "
                 f"(enter ≥ {result.get('enter_threshold', 0):.2f})"))
    appr = "APPROVED" if result.get("approved") else "NOT APPROVED"
    L.append(row("Signal", f"{appr} [{result.get('approval_method')}]"))
    if "bias_mtf" in result:
        L.append(row("MTF bias",
                     f"{result['bias_mtf']:+.3f} "
                     f"{'(aligned)' if result.get('mtf_aligned') else '(discounted: ' + ','.join(result.get('tf_discounts', [])) + ')'}"))
    L.append(mid)

    # ── committee table ──
    L.append(row("TOP CONTRIBUTORS", "signal × weight → vote"))
    table = explain_committee(result, top=6)
    if not table.empty:
        for name, r in table.iterrows():
            arrow = "▲" if r["signal"] > 0 else "▼" if r["signal"] < 0 else "·"
            if not rich:
                arrow = "^" if r["signal"] > 0 else "v" if r["signal"] < 0 else "-"
            L.append(row(f" {arrow} {name}"[:30],
                         f"{r['signal']:+.0f}  conf {r['confidence']:.2f}  "
                         f"w {r['contribution']:+.3f}"[:30]))
    else:
        L.append(row(" (no active strategies)", ""))
    L.append(mid)

    # ── trade plan box ──
    stops = result.get("stops", {}) or {}
    side = int(_np.sign(result.get("position_now", 0))) or int(
        _np.sign(bias)) or 1
    side_name = "LONG" if side > 0 else "SHORT"
    approved = bool(result.get("approved"))
    plan_head = (f"{side_name} @ {result.get('entry', 0):,.4g}"
                 if approved else f"{side_name} @ {result.get('entry', 0):,.4g} "
                 "(if approved)")
    L.append(row("TRADE PLAN", plan_head[:30]))
    if stops:
        L.append(row("Stop", f"{stops.get('stop', 0):,.4g}"))
        L.append(row("Breakeven trigger",
                     f"{stops.get('breakeven_trigger', 0):,.4g}"))
        L.append(row("Targets",
                     f"{stops.get('target_1', 0):,.4g} / "
                     f"{stops.get('target_2', 0):,.4g}"))
    L.append(row("Size (fraction)", f"{result.get('size_fraction', 0):.1%}"))
    L.append(row("Stop distance", f"{result.get('stop_distance_pct', 0):.2f}%"))
    L.append(row("ATR", f"{result.get('atr', 0):,.4g} "
                        f"({result.get('atr_pct', 0) * 100:.2f}%)"))
    ranked = result.get("ranked")
    if ranked is not None and len(ranked):
        best = ranked.index[0]
        L.append(row("Best strategy (in-sample)",
                     f"{best} [{ranked.iloc[0]['sharpe']:+.2f}]"[:30]))
    L.append(bot)
    return "\n".join(L)
