"""Signal fusion: turn a committee of strategy frames into one tradable vote.

Ported from ``sentinel/signals/fusion.py`` (user's own Sentinel.py bot).

Every strategy emits a frame with ``signal`` (-1/0/+1), ``confidence``
(0..1) and ``gate`` (0..1 exposure throttle). Fusion weights each vote by
confidence × gate, optionally re-weights by strategy kind and by recent
per-strategy scores, derives the entry threshold from costs (never a magic
number), and applies hysteresis + an agreement filter so the position series
doesn't flicker.
"""

from __future__ import annotations


from .math import effective_n, softmax


# ── lazy optional deps ──────────────────────────────────────────────────
# numpy/pandas are optional. The package imports without them; functions
# that need them raise TAError with a clear install hint.

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



__all__ = [
    "KINDS",
    "fuse_equal",
    "fuse_weighted",
    "adaptive_kind_weights",
    "kind_of",
    "cost_aware_threshold",
    "hysteresis_position",
    "disagreement_filter",
    "fuse_all",
    # ── sweep additions ──
    "strategy_correlation",
    "fuse_diversified",
    "fuse_stacked",
    "explain_vote",
    "vote_quality",
    "min_hold_position",
]

KINDS = ("trend", "meanrev", "breakout", "momentum", "squeeze", "reversal",
         "confluence")


def fuse_equal(frames: dict) -> _pd.DataFrame:
    """Confidence × gate weighted vote across ``{name: signal-frame}``."""
    if not frames:
        return _pd.DataFrame(columns=["vote", "agreement", "n"])
    names = list(frames)
    idx = frames[names[0]].index
    votes = _np.zeros(len(idx))
    mass = _np.zeros(len(idx))
    longs = _np.zeros(len(idx))
    for s in frames.values():
        w = s["confidence"].to_numpy(dtype=float) * s["gate"].to_numpy(dtype=float)
        d = _np.sign(s["signal"].to_numpy(dtype=float))
        votes += d * w
        mass += w
        longs += (d > 0).astype(float)
    vote = votes / (mass + 1e-12)
    agreement = _np.abs(longs / max(1, len(frames)) - 0.5) * 2.0
    return _pd.DataFrame(
        {"vote": vote, "agreement": agreement, "n": float(len(frames))}, index=idx
    )


def adaptive_kind_weights(recent_scores: dict | None,
                          prior: float = 1.0) -> dict:
    """Softmax over recent per-kind scores; falls back to uniform."""
    base = {k: float(prior) for k in KINDS}
    if recent_scores:
        for k, v in recent_scores.items():
            if k in base:
                base[k] = float(prior) + float(v)
    w = softmax(base)
    tot = sum(w.values()) or 1.0
    return {k: v / tot * len(KINDS) for k, v in w.items()}


def kind_of(name: str, lookup: dict | None = None) -> str:
    """Strategy kind for weighting: explicit lookup, else name sniffing."""
    if lookup and name in lookup:
        return str(lookup[name])
    low = name.lower()
    for k in KINDS:
        if k in low:
            return k
    return "confluence"


def fuse_weighted(frames: dict, kind_weights: dict | None = None,
                  lookup: dict | None = None,
                  scores: dict | None = None) -> _pd.DataFrame:
    """Vote with per-kind and per-strategy weighting on top of confidence."""
    if not frames:
        return _pd.DataFrame(columns=["vote", "agreement", "n"])
    kw = adaptive_kind_weights(
        None if kind_weights is None else {k: 0.0 for k in KINDS})
    if kind_weights:
        tot = sum(max(0.0, float(v)) for v in kind_weights.values()) or 1.0
        kw = {k: max(0.0, float(kind_weights.get(k, 1.0))) / tot * len(KINDS)
              for k in KINDS}
    sw = None
    if scores:
        sw = softmax({n: float(scores.get(n, 0.0)) for n in frames})
        m = sum(sw.values()) / max(1, len(sw))
        sw = {n: v / (m + 1e-12) for n, v in sw.items()}
    names = list(frames)
    idx = frames[names[0]].index
    votes = _np.zeros(len(idx))
    mass = _np.zeros(len(idx))
    longs = _np.zeros(len(idx))
    for n, s in frames.items():
        w = s["confidence"].to_numpy(dtype=float) * s["gate"].to_numpy(dtype=float)
        w = w * kw.get(kind_of(n, lookup), 1.0)
        if sw:
            w = w * sw.get(n, 1.0)
        d = _np.sign(s["signal"].to_numpy(dtype=float))
        votes += d * w
        mass += w
        longs += (d > 0).astype(float)
    vote = votes / (mass + 1e-12)
    agreement = _np.abs(longs / max(1, len(frames)) - 0.5) * 2.0
    return _pd.DataFrame(
        {"vote": vote, "agreement": agreement, "n": float(len(frames))}, index=idx
    )


def cost_aware_threshold(cost_bps: float, atr_pct: float,
                         k: float = 1.0) -> float:
    """Minimum vote magnitude worth trading, derived from costs.

    Needs ``edge >= 2× costs`` in ATR units — never a magic number.
    """
    edge_need = (float(cost_bps) / 1e4) / (max(1e-6, float(atr_pct)) + 1e-9)
    return float(_np.clip(k * edge_need * 2.0, 0.02, 0.6))


def hysteresis_position(vote: _pd.Series, enter: float,
                        exit: float = 0.03) -> _pd.Series:
    """Sticky position series: enter past ``enter``, exit below ``exit``.

    Reversals need a full ``enter``-sized opposing vote; plain decay exits
    below ``exit``. Deterministic.
    """
    v = _np.asarray(vote, dtype=float)
    out = _np.zeros(len(v))
    state = 0.0
    for i in range(len(v)):
        if state == 0.0 and abs(v[i]) >= enter:
            state = float(_np.sign(v[i]))
        elif state > 0 and v[i] < -enter:
            state = -1.0
        elif state < 0 and v[i] > enter:
            state = 1.0
        elif state != 0.0 and abs(v[i]) < exit:
            state = 0.0
        out[i] = state
    return _pd.Series(out, index=vote.index, name="position")


def disagreement_filter(blend: _pd.DataFrame,
                        min_agreement: float = 0.55) -> _pd.Series:
    """Binary mask: 1 where the committee agrees enough to be trusted."""
    agr = blend["agreement"].to_numpy(dtype=float)
    return _pd.Series((agr >= min_agreement).astype(float), index=blend.index,
                     name="agree_mask")


def fuse_all(frames: dict, kind_weights: dict | None = None,
             lookup: dict | None = None, scores: dict | None = None,
             cost_bps: float = 10.0, atr_pct: float = 0.01,
             min_agreement: float = 0.55) -> dict:
    """Full committee pipeline: weighted vote → cost gate → hysteresis.

    Returns ``{"blend", "position", "enter_threshold", "n_strategies"}``.
    """
    blend = fuse_weighted(frames, kind_weights, lookup, scores)
    if blend.empty:
        return {"blend": blend,
                "position": blend.get("vote", _pd.Series(dtype=float)),
                "enter_threshold": cost_aware_threshold(cost_bps, atr_pct),
                "n_strategies": len(frames)}
    enter = cost_aware_threshold(cost_bps, atr_pct)
    pos = hysteresis_position(blend["vote"], enter)
    mask = disagreement_filter(blend, min_agreement)
    pos = pos * mask
    return {"blend": blend, "position": pos.rename("position"),
            "enter_threshold": enter, "n_strategies": len(frames)}


# ── sweep additions: diversification, stacking, explanation ──────────────

def strategy_correlation(frames: dict) -> _pd.DataFrame:
    """Pairwise correlation of confidence-weighted strategy votes.

    Correlated strategies double-count the same bet — this matrix is the
    input to the diversification penalty.
    """
    if not frames:
        return _pd.DataFrame()
    names = list(frames)
    votes = _np.column_stack([
        _np.sign(frames[n]["signal"].to_numpy(dtype=float))
        * frames[n]["confidence"].to_numpy(dtype=float)
        for n in names
    ])
    with _np.errstate(invalid="ignore", divide="ignore"):
        corr = _np.corrcoef(votes, rowvar=False)
    corr = _np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    _np.fill_diagonal(corr, 1.0)
    return _pd.DataFrame(corr, index=names, columns=names)


def fuse_diversified(frames: dict, kind_weights: dict | None = None,
                     lookup: dict | None = None,
                     scores: dict | None = None,
                     div_strength: float = 1.0) -> _pd.DataFrame:
    """Vote penalized by strategy correlation (risk-parity over strategies).

    Each strategy's weight is multiplied by ``1 / (1 + mean|corr|)`` —
    a committee of clones collapses toward one vote, while genuinely
    independent strategies keep their weight. ``div_strength`` scales the
    penalty (0 = plain weighted vote).
    """
    if not frames:
        return _pd.DataFrame(columns=["vote", "agreement", "n"])
    corr = strategy_correlation(frames)
    names = list(frames)
    div = {}
    for n in names:
        others = [c for c in names if c != n]
        mean_corr = float(_np.abs(
            corr.loc[n, others]).mean()) if others else 0.0
        div[n] = 1.0 / (1.0 + float(div_strength) * mean_corr)
    # Fold the diversification factor into per-strategy scores.
    adj_scores = {n: float((scores or {}).get(n, 0.0))
                  + _np.log(max(div[n], 1e-9)) for n in names}
    return fuse_weighted(frames, kind_weights, lookup, adj_scores)


def fuse_stacked(frames: dict, close: _pd.Series, horizon: int = 5,
                 kind_weights: dict | None = None,
                 lookup: dict | None = None) -> _pd.DataFrame:
    """Logistic stacking over strategy votes (sklearn-optional).

    Trains P(next-``horizon``-bar return > 0 | committee votes) with a
    TimeSeriesSplit; the vote becomes the stacked probability recentered
    to [-1, 1]. Falls back to ``fuse_weighted`` when sklearn is missing
    or the fit fails — never a hard dependency.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return fuse_weighted(frames, kind_weights, lookup)
    if not frames:
        return _pd.DataFrame(columns=["vote", "agreement", "n"])
    names = list(frames)
    idx = frames[names[0]].index
    X = _np.column_stack([
        _np.sign(frames[n]["signal"].to_numpy(dtype=float))
        * frames[n]["confidence"].to_numpy(dtype=float)
        * frames[n]["gate"].to_numpy(dtype=float)
        for n in names
    ])
    fwd = (close.reindex(idx).astype(float).shift(-horizon)
           / (close.reindex(idx).astype(float) + 1e-12) - 1.0)
    y = (fwd.fillna(0.0).to_numpy() > 0).astype(int)
    usable = _np.arange(len(idx)) < len(idx) - horizon
    if usable.sum() < 50 or len(_np.unique(y[usable])) < 2:
        return fuse_weighted(frames, kind_weights, lookup)
    try:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=500, C=1.0))
        # Fit on the first 70% (time-ordered), predict the rest OOF-style.
        cut = int(usable.sum() * 0.7)
        order = _np.where(usable)[0]
        clf.fit(X[order[:cut]], y[order[:cut]])
        proba = _np.full(len(idx), 0.5)
        proba[order[cut:]] = clf.predict_proba(X[order[cut:]])[:, 1]
        proba[:order[cut][0] if len(order[cut]) else 0] = 0.5
    except Exception:
        return fuse_weighted(frames, kind_weights, lookup)
    vote = _pd.Series(2.0 * proba - 1.0, index=idx, name="vote")
    base = fuse_weighted(frames, kind_weights, lookup)
    out = base.copy()
    out["vote"] = vote
    out["stack_proba"] = proba
    return out


def explain_vote(frames: dict, blend: _pd.DataFrame,
                 kind_weights: dict | None = None,
                 lookup: dict | None = None,
                 at: int = -1) -> _pd.DataFrame:
    """Per-strategy contribution table at bar ``at`` — the committee, shown.

    Columns: signal, confidence, gate, kind, weight, contribution
    (weight × signed vote, i.e. how much this strategy moved the needle).
    Sorted by |contribution|, best first.
    """
    if not frames or blend.empty:
        return _pd.DataFrame()
    kw = adaptive_kind_weights(
        None if kind_weights is None else {k: 0.0 for k in KINDS})
    if kind_weights:
        tot = sum(max(0.0, float(v)) for v in kind_weights.values()) or 1.0
        kw = {k: max(0.0, float(kind_weights.get(k, 1.0))) / tot * len(KINDS)
              for k in KINDS}
    rows = []
    for n, s in frames.items():
        sig = float(_np.sign(s["signal"].iloc[at]))
        conf = float(s["confidence"].iloc[at])
        gate = float(s["gate"].iloc[at])
        w = conf * gate * kw.get(kind_of(n, lookup), 1.0)
        rows.append({"strategy": n, "kind": kind_of(n, lookup),
                     "signal": sig, "confidence": round(conf, 3),
                     "gate": round(gate, 3),
                     "weight": round(w, 4),
                     "contribution": round(sig * w, 4)})
    df = _pd.DataFrame(rows).set_index("strategy")
    return df.reindex(
        df["contribution"].abs().sort_values(ascending=False).index)


def vote_quality(blend: _pd.DataFrame) -> dict:
    """How much is this vote worth? Agreement, concentration, breadth."""
    if blend.empty:
        return {"agreement": 0.0, "effective_n": 0.0, "hhi": 0.0}
    agr = blend["agreement"].to_numpy(dtype=float)
    vote = blend["vote"].to_numpy(dtype=float)
    mag = _np.abs(vote)
    tot = mag.sum()
    if tot <= 1e-12:
        hhi, eff = 0.0, 0.0
    else:
        p = mag / tot
        hhi = float((p ** 2).sum())
        eff = float(1.0 / (hhi + 1e-12))
    return {
        "agreement": float(agr[-1]),
        "mean_agreement": float(agr.mean()),
        "effective_n": eff,          # independent bets in the vote mass
        "hhi": hhi,                 # concentration (1 = one bar decides)
        "vote_now": float(vote[-1]),
    }


def min_hold_position(position: _pd.Series,
                      min_bars: int = 3) -> _pd.Series:
    """Enforce a minimum holding period — kills churn from flicker.

    A new position must survive ``min_bars`` before it may flip or exit.
    """
    v = _np.asarray(position, dtype=float)
    out = _np.zeros(len(v))
    state = 0.0
    held = 0
    for i in range(len(v)):
        want = v[i]
        if want != state:
            if held >= min_bars or state == 0.0:
                state, held = want, 0
            # else: hold the current state — the flip was noise
        else:
            held += 1
        out[i] = state
    return _pd.Series(out, index=position.index, name="position")
