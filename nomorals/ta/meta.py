"""Meta-labeling gate: a supervised veto over the committee's vote.

Ported from ``sentinel/ml/meta.py`` (user's own Sentinel.py bot).

``labeled_matrix`` builds (X, y, mask) from the fused vote, agreement and
regime features: the label is "did sign(vote) win over the horizon?".
``MetaGate`` fits a calibrated gradient-boosting classifier on it and
approves only the bars the model expects to win.

sklearn is imported lazily and is optional: when it is missing (or fitting
fails), the gate degrades to ``predict_proba → 0.5`` and callers should use
the rule-based fallback in :mod:`nomorals.ta.pipeline`.
"""

from __future__ import annotations

import logging


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




__all__ = ["labeled_matrix", "MetaGate", "sklearn_available",
           # ── sweep additions: AFML-canonical labeling + validation ──
           "triple_barrier_labels", "meta_labels_from_barriers",
           "purged_kfold_splits", "bet_size_from_prob", "oof_predict_proba",
           "permutation_importance"]

_log = logging.getLogger(__name__)

_REGIME_COLS = ("vol_ratio", "range_pos", "width_q", "vol_q", "trend_r2",
                "efficiency", "p_trend", "p_range", "p_squeeze", "p_panic")


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def labeled_matrix(vote: _pd.Series, agreement: _pd.Series,
                   regime: _pd.DataFrame, close: _pd.Series,
                   horizon: int = 10, min_vote: float = 0.1):
    """Build (X, y, mask): features at t, label = did sign(vote) win?"""
    idx = vote.index
    fwd = close.shift(-horizon).astype(float) / (close.astype(float) + 1e-12) - 1.0
    v = vote.reindex(idx).fillna(0.0)
    a = agreement.reindex(idx).fillna(0.0)
    X = _pd.DataFrame(index=idx)
    X["vote"] = v
    X["agreement"] = a
    X["vote_x_agree"] = v * a
    for c in _REGIME_COLS:
        if c in regime.columns:
            X[c] = regime[c].reindex(idx).fillna(0.0)
    direction = _np.sign(v.to_numpy(dtype=float))
    fwd_a = fwd.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    won = (direction * fwd_a) > 0
    active = (_np.abs(v.to_numpy(dtype=float)) >= min_vote) & (
        _np.arange(len(idx)) < len(idx) - horizon)
    y = _pd.Series(_np.where(won & active, 1, 0), index=idx)
    return X.fillna(0.0), y, _pd.Series(active, index=idx)


class MetaGate:
    """Calibrated gradient-boosting veto. Graceful fallback if sklearn is missing."""

    def __init__(self, threshold: float = 0.55, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state
        self.model = None
        self.n_features = 0
        self.train_pos_rate = 0.5

    def _build(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
            random_state=self.random_state)
        clf = CalibratedClassifierCV(estimator=base, cv=3)
        return make_pipeline(StandardScaler(), clf)

    def fit(self, X, y, sample_weight=None) -> "MetaGate":
        Xa = _np.asarray(X, dtype=float)
        ya = _np.asarray(y, dtype=int)
        self.n_features = Xa.shape[1]
        self.train_pos_rate = float(ya.mean()) if len(ya) else 0.5
        if len(_np.unique(ya)) < 2:
            self.model = None
            return self
        try:
            self.model = self._build()
            self.model.fit(Xa, ya)
        except Exception as e:
            _log.debug("meta model fit failed, disabling: %s", e)
            self.model = None
        return self

    def predict_proba(self, X) -> _np.ndarray:
        Xa = _np.asarray(X, dtype=float)
        if self.model is None or Xa.shape[1] != self.n_features \
                or self.n_features == 0:
            return _np.full(len(Xa), 0.5)
        try:
            return self.model.predict_proba(Xa)[:, 1]
        except Exception as e:
            _log.debug("meta model predict failed, using 0.5: %s", e)
            return _np.full(len(Xa), 0.5)

    def approve(self, X, threshold: float | None = None) -> _np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        return self.predict_proba(X) >= thr

    def save(self, path) -> None:
        try:
            import joblib

            joblib.dump({"model": self.model, "thr": self.threshold,
                         "nf": self.n_features, "pos": self.train_pos_rate},
                        path)
        except Exception as e:
            _log.warning("could not save meta model to %s: %s", path, e)

    def load(self, path) -> bool:
        try:
            import joblib

            d = joblib.load(path)
            self.model, self.threshold = d["model"], d["thr"]
            self.n_features, self.train_pos_rate = d["nf"], d["pos"]
            return True
        except Exception as e:
            _log.debug("meta model load failed: %s", e)
            return False

    def summary(self) -> dict:
        """Inspectable state: what the gate learned (or that it didn't)."""
        return {
            "fitted": self.model is not None,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "train_pos_rate": self.train_pos_rate,
            "fallback": self.model is None,
        }


# ── sweep additions ──────────────────────────────────────────────────────
# The old labeled_matrix used a naive forward-return label. The AFML
# canon (López de Prado Ch. 3/4/10) is: triple-barrier labels on the real
# high/low path → meta-label (did the primary side win?) → secondary
# classifier → bet size from P(win) → purged/embargoed CV throughout.

def triple_barrier_labels(df: _pd.DataFrame, side: _pd.Series,
                          pt_mult: float = 2.0, sl_mult: float = 1.0,
                          max_horizon: int = 20,
                          atr_period: int = 14) -> _pd.DataFrame:
    """AFML triple-barrier labels on the actual OHLC path.

    For each bar with ``side`` ±1: profit-take at ``pt_mult``×ATR,
    stop-loss at ``sl_mult``×ATR, vertical barrier at ``max_horizon``
    bars. Barriers are checked against the bar's high/low (the real
    path, not closes). Returns ``bin`` ∈ {−1, 0, +1} — which barrier was
    touched first — plus ``t1`` (bars to resolution) and ``ret``.
    """
    from .math import atr as _atr_fn
    from .math import ensure_ohlcv as _ensure

    df = _ensure(df)
    s = _np.sign(side.reindex(df.index).fillna(0.0).to_numpy(dtype=float))
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    a = _atr_fn(df, max(2, int(atr_period))).to_numpy() + 1e-12
    n = len(df)
    hz = max(1, int(max_horizon))
    ptm, slm = float(pt_mult), float(sl_mult)
    bins = _np.zeros(n)
    t1 = _np.full(n, hz)
    rets = _np.zeros(n)
    for i in range(n):
        if s[i] == 0 or i >= n - 1:
            continue
        entry = c[i]
        pt = entry + s[i] * ptm * a[i]
        sl = entry - s[i] * slm * a[i]
        end = min(n, i + 1 + hz)
        resolved = False
        for j in range(i + 1, end):
            if s[i] > 0:
                hit_pt = h[j] >= pt
                hit_sl = l[j] <= sl
            else:
                hit_pt = l[j] <= pt
                hit_sl = h[j] >= sl
            if hit_pt and hit_sl:
                # Same-bar ambiguity: the conservative call is the loss.
                bins[i] = -1.0
                rets[i] = -slm * a[i] / (entry + 1e-12)
                t1[i] = j - i
                resolved = True
                break
            if hit_pt:
                bins[i] = 1.0
                rets[i] = ptm * a[i] / (entry + 1e-12)
                t1[i] = j - i
                resolved = True
                break
            if hit_sl:
                bins[i] = -1.0
                rets[i] = -slm * a[i] / (entry + 1e-12)
                t1[i] = j - i
                resolved = True
                break
        if not resolved:
            bins[i] = 0.0  # vertical barrier: timeout
            rets[i] = s[i] * (c[end - 1] - entry) / (entry + 1e-12)
            t1[i] = end - 1 - i
    return _pd.DataFrame({"bin": bins, "t1": t1, "ret": rets},
                         index=df.index)


def meta_labels_from_barriers(tb: _pd.DataFrame) -> _pd.Series:
    """Meta-label: 1 if the primary side's trade won (bin == +1)."""
    return (_pd.Series(tb["bin"].to_numpy(dtype=float),
                       index=tb.index) == 1.0).astype(int).rename("meta_y")


def purged_kfold_splits(n: int, n_splits: int = 5, embargo: int = 0,
                        label_spans: _np.ndarray | None = None):
    """Purged K-fold CV for overlapping financial labels (AFML Ch. 7).

    Yields ``(train_idx, test_idx)``. Any training sample whose label
    span ``[i, i + span)`` overlaps a test fold (plus ``embargo`` bars)
    is purged from training — no leakage through label overlap.
    ``label_spans[i]`` = bars from i until that label resolves.
    """
    n = int(n)
    n_splits = max(2, int(n_splits))
    embargo = max(0, int(embargo))
    fold = n // n_splits
    spans = (_np.asarray(label_spans, dtype=int) if label_spans is not None
             else _np.zeros(n, dtype=int))
    for k in range(n_splits):
        test_start = k * fold
        test_end = n if k == n_splits - 1 else (k + 1) * fold
        test_idx = _np.arange(test_start, test_end)
        train_idx = _np.concatenate(
            [_np.arange(0, test_start), _np.arange(test_end, n)])
        if len(train_idx):
            # Purge: drop train samples whose label window touches the test
            # fold (extended by the embargo on both sides).
            lo = test_start - embargo
            hi = test_end + embargo
            overlap = ((train_idx < hi) & (train_idx + spans[train_idx] > lo))
            train_idx = train_idx[~overlap]
        yield train_idx, test_idx


def bet_size_from_prob(prob: float | _np.ndarray, max_size: float = 1.0,
                       power: float = 1.0) -> float | _np.ndarray:
    """De Prado bet sizing: size grows with P(win) (AFML 10.4).

    Maps probability → ``[-max_size, +max_size]`` via the centered
    sigmoid-ish curve ``sign(2p−1)·|2p−1|^power``. A coin flip (p=0.5)
    sizes zero; p→1 sizes full. ``power > 1`` concentrates on high
    conviction; ``power < 1`` spreads more evenly.
    """
    p = _np.asarray(prob, dtype=float)
    m = _np.sign(2.0 * p - 1.0) * _np.abs(2.0 * p - 1.0) ** float(power)
    out = _np.clip(m, -1.0, 1.0) * float(max_size)
    return float(out) if out.ndim == 0 else out


def oof_predict_proba(model_fn, X, y, n_splits: int = 5,
                      embargo: int = 0,
                      label_spans: _np.ndarray | None = None) -> _np.ndarray:
    """Out-of-fold predicted probabilities under purged K-fold.

    ``model_fn`` is a zero-arg callable returning a fresh unfitted
    classifier with ``predict_proba``. Never trust in-sample meta
    probabilities — this is the honest version.
    """
    Xa = _np.asarray(X, dtype=float)
    ya = _np.asarray(y, dtype=int)
    oof = _np.full(len(Xa), 0.5)
    for tr, te in purged_kfold_splits(len(Xa), n_splits, embargo,
                                      label_spans):
        if len(tr) < 10 or len(te) == 0 or len(_np.unique(ya[tr])) < 2:
            continue
        try:
            model = model_fn()
            model.fit(Xa[tr], ya[tr])
            oof[te] = model.predict_proba(Xa[te])[:, 1]
        except Exception as e:
            _log.debug("oof fold failed: %s", e)
            continue
    return oof


def permutation_importance(predict_fn, X, y, n_repeats: int = 5,
                           seed: int = 42,
                           metric: str = "accuracy") -> dict:
    """Model-agnostic permutation importance (no sklearn needed).

    ``predict_fn(X)`` → predicted probabilities or labels. Shuffles each
    column ``n_repeats`` times and measures the score drop — works for
    the MetaGate, a stacked fuser, anything.
    """
    Xa = _np.asarray(X, dtype=float)
    ya = _np.asarray(y, dtype=int)
    rng = _np.random.default_rng(seed)

    def _score(pred):
        pred = _np.asarray(pred)
        if pred.ndim > 1:
            pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
        labels = (pred >= 0.5).astype(int) if pred.max() <= 1.0 \
            and pred.min() >= 0.0 else pred.astype(int)
        if metric == "logloss":
            p = _np.clip(pred if pred.ndim == 1 else pred, 1e-9, 1 - 1e-9)
            return float(-(ya * _np.log(p) + (1 - ya) * _np.log(1 - p)).mean())
        return float((labels == ya).mean())

    base = _score(predict_fn(Xa))
    cols = list(X.columns) if hasattr(X, "columns") else [
        f"f{i}" for i in range(Xa.shape[1])]
    imp = {}
    for j, name in enumerate(cols):
        drops = []
        for _ in range(max(1, int(n_repeats))):
            Xs = Xa.copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            drops.append(base - _score(predict_fn(Xs)))
        imp[name] = float(_np.mean(drops))
    # Higher = more important. For logloss, drops are negative-is-better;
    # keep the raw sign and document it.
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
