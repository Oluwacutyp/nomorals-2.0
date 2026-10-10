"""Adaptive market-regime detector.

Ported from ``sentinel/core/regime.py`` (user's own Sentinel.py bot). Five
states — TREND_UP / TREND_DOWN / RANGE / SQUEEZE / PANIC — with thresholds
calibrated from rolling quantiles of each feature's own history, so the
detector auto-tunes to any symbol or timeframe. Hysteresis prevents label
flicker.
"""

from __future__ import annotations


from .math import atr, ensure_ohlcv, rolling_quantile


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



__all__ = ["LABELS", "CODES", "RegimeDetector", "HMMRegimeDetector",
           "transition_matrix", "expected_durations", "persistence_score",
           "regime_playbook"]

LABELS = ("TREND_UP", "TREND_DOWN", "RANGE", "SQUEEZE", "PANIC")
CODES = {name: i for i, name in enumerate(LABELS)}


def _r2_trend(close: _pd.Series, window: int) -> _pd.Series:
    """Signed R² of a rolling log-price linear fit (trend strength × sign)."""
    x = _np.arange(window, dtype=float)

    def _r2(v):
        if len(v) < 8:
            return 0.0
        y = _np.asarray(v, dtype=float)
        if float(_np.std(y)) <= 1e-12:
            return 0.0
        xx = x[-len(y):]
        a, b = _np.polyfit(xx, y, 1)
        pred = a * xx + b
        ss_res = float(_np.sum((y - pred) ** 2))
        ss_tot = float(_np.sum((y - y.mean()) ** 2)) + 1e-12
        return float(max(0.0, 1.0 - ss_res / ss_tot)) * float(_np.sign(a) or 1.0)

    return (
        close.rolling(window, min_periods=max(8, window // 3))
        .apply(_r2, raw=True)
        .fillna(0.0)
    )


def _efficiency(close: _pd.Series, window: int) -> _pd.Series:
    """Kaufman-style efficiency ratio, signed by net direction."""
    net = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=5).sum()
    er = (net / (path + 1e-12)).fillna(0.0)
    direction = _np.sign(close.diff(window).fillna(0.0))
    return er * direction


class RegimeDetector:
    """Five-state regime engine with quantile-calibrated thresholds."""

    def __init__(self, lookback: int = 200, feature_window: int = 60,
                 cal_window: int = 500, hysteresis: int = 3):
        self.lookback = lookback
        self.feature_window = feature_window
        self.cal_window = cal_window
        self.hysteresis = hysteresis

    def features(self, df: _pd.DataFrame) -> _pd.DataFrame:
        df = ensure_ohlcv(df)
        close = df["close"]
        logc = _np.log(close.clip(lower=1e-9))
        w = self.feature_window
        f = _pd.DataFrame(index=df.index)
        f["trend_r2"] = _r2_trend(logc, w)
        f["efficiency"] = _efficiency(close, w)
        rets = close.pct_change().fillna(0.0)
        f["vol"] = rets.rolling(w, min_periods=10).std().bfill().ffill().fillna(1e-9)
        f["vol_median"] = (
            f["vol"].rolling(self.lookback, min_periods=20).median().bfill().ffill()
        )
        f["vol_ratio"] = (f["vol"] / (f["vol_median"] + 1e-12)).fillna(1.0)
        a = atr(df, 14)
        f["atr_ratio"] = (
            a / (a.rolling(self.lookback, min_periods=20).mean().bfill().ffill() + 1e-12)
        ).fillna(1.0)
        hi = close.rolling(w, min_periods=5).max()
        lo = close.rolling(w, min_periods=5).min()
        f["range_pos"] = ((close - lo) / ((hi - lo) + 1e-12)).fillna(0.5)
        width = (hi - lo) / (close.rolling(w, min_periods=5).mean() + 1e-12)
        f["width"] = width.fillna(0.0)
        f["width_q"] = (
            f["width"].rolling(self.cal_window, min_periods=30).rank(pct=True).fillna(0.5)
        )
        f["vol_q"] = (
            f["vol_ratio"].rolling(self.cal_window, min_periods=30).rank(pct=True).fillna(0.5)
        )
        return f.fillna(0.0)

    def _raw_label(self, f: _pd.DataFrame, sq: float, pq: float,
                   tq: float) -> _pd.Series:
        tr = f["trend_r2"].to_numpy()
        er = f["efficiency"].to_numpy()
        wq = f["width_q"].to_numpy()
        vq = f["vol_q"].to_numpy()
        labels = _np.full(len(f), CODES["RANGE"], dtype=int)
        trend = (_np.abs(tr) > tq) & (_np.abs(er) > 0.25)
        labels[trend & (tr > 0)] = CODES["TREND_UP"]
        labels[trend & (tr < 0)] = CODES["TREND_DOWN"]
        labels[(wq < sq) & (~trend)] = CODES["SQUEEZE"]
        labels[vq > pq] = CODES["PANIC"]
        return _pd.Series(labels, index=f.index)

    def fit(self, df: _pd.DataFrame, squeeze_q: float = 0.15,
            panic_q: float = 0.95, trend_q: float = 0.45) -> _pd.DataFrame:
        """Full regime frame: features + hysteresis-smoothed label + probs."""
        f = self.features(df)
        raw = self._raw_label(f, squeeze_q, panic_q, trend_q)
        lab = raw.to_numpy().copy()
        h = max(1, self.hysteresis)
        for i in range(1, len(lab)):
            if lab[i] != lab[i - 1]:
                j = i
                while j < len(lab) and lab[j] == lab[i]:
                    j += 1
                if j - i < h:
                    lab[i:j] = lab[i - 1]
        out = f.copy()
        out["regime"] = lab
        out["label"] = [LABELS[int(c)] for c in lab]
        tr = _np.abs(f["trend_r2"].to_numpy())
        out["p_trend"] = _np.clip((tr - trend_q) / (1 - trend_q + 1e-9), 0, 1)
        out["p_squeeze"] = _np.clip(
            (squeeze_q - f["width_q"].to_numpy()) / (squeeze_q + 1e-9), 0, 1)
        out["p_panic"] = _np.clip(
            (f["vol_q"].to_numpy() - panic_q) / (1 - panic_q + 1e-9), 0, 1)
        out["p_range"] = _np.clip(
            1 - out[["p_trend", "p_squeeze", "p_panic"]].max(axis=1), 0, 1)
        return out

    def current(self, df: _pd.DataFrame, **kw) -> dict:
        """Regime snapshot at the last bar, with a lookback label mix."""
        frame = self.fit(df, **kw)
        last = frame.iloc[-1]
        counts = (
            frame["label"].iloc[-self.lookback:].value_counts(normalize=True).to_dict()
        )
        return {
            "label": str(last["label"]),
            "code": int(last["regime"]),
            "p_trend": float(last["p_trend"]),
            "p_range": float(last["p_range"]),
            "p_squeeze": float(last["p_squeeze"]),
            "p_panic": float(last["p_panic"]),
            "trend_r2": float(last["trend_r2"]),
            "vol_ratio": float(last["vol_ratio"]),
            "mix_lookback": {k: float(v) for k, v in counts.items()},
        }

    def transition_alert(self, df: _pd.DataFrame, **kw) -> dict:
        """Did the regime just change? Flags squeeze→expansion and panic onset."""
        frame = self.fit(df, **kw)
        if len(frame) < 10:
            return {"transition": False}
        prev = str(frame["label"].iloc[-6])
        now = str(frame["label"].iloc[-1])
        return {
            "transition": prev != now,
            "from": prev,
            "to": now,
            "squeeze_to_expansion": prev == "SQUEEZE"
            and now in ("TREND_UP", "TREND_DOWN", "PANIC"),
            "panic_onset": now == "PANIC" and prev != "PANIC",
        }


# ── sweep additions: HMM detector, transition analytics, playbook ───────

def transition_matrix(labels) -> _pd.DataFrame:
    """Empirical P(next | current) over a label series."""
    lab = _pd.Series(labels).reset_index(drop=True)
    states = sorted(lab.unique())
    mat = _pd.DataFrame(0.0, index=states, columns=states)
    for a, b in zip(lab.iloc[:-1], lab.iloc[1:]):
        mat.loc[a, b] += 1.0
    return mat.div(mat.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)


def expected_durations(labels) -> dict:
    """Mean bars spent per visit, per regime (from the transition matrix)."""
    mat = transition_matrix(labels)
    out = {}
    for s in mat.index:
        p_stay = float(mat.loc[s, s])
        out[str(s)] = float(1.0 / (1.0 - p_stay)) if p_stay < 1.0 else float(
            (labels == s).sum())
    return out


def persistence_score(labels) -> float:
    """Fraction of bars where the regime did not change (stability)."""
    lab = _pd.Series(labels).reset_index(drop=True)
    if len(lab) < 2:
        return 1.0
    return float((lab.iloc[1:].to_numpy() == lab.iloc[:-1].to_numpy()).mean())


def regime_playbook() -> dict:
    """Regime → action table: favored/avoided strategy kinds + exposure.

    The actionability the detector was missing. Grounded in the standard
    practitioner mapping (trend systems in trends, fades in ranges,
    breakouts out of squeezes, risk-off in panic).
    """
    return {
        "TREND_UP": {
            "favor": ["trend", "momentum", "breakout"],
            "avoid": ["meanrev", "reversal"],
            "exposure_mult": 1.0,
            "note": "Ride with the trend; fade counter-trend signals.",
        },
        "TREND_DOWN": {
            "favor": ["trend", "momentum", "breakout"],
            "avoid": ["meanrev", "reversal"],
            "exposure_mult": 1.0,
            "note": "Ride with the trend; fade counter-trend signals.",
        },
        "RANGE": {
            "favor": ["meanrev", "reversal", "confluence"],
            "avoid": ["trend", "breakout"],
            "exposure_mult": 0.8,
            "note": "Fade extremes; trend systems chop here.",
        },
        "SQUEEZE": {
            "favor": ["breakout", "squeeze", "confluence"],
            "avoid": ["meanrev"],
            "exposure_mult": 0.6,
            "note": "Energy compressing — size for the expansion, not the chop.",
        },
        "PANIC": {
            "favor": ["confluence"],
            "avoid": ["trend", "momentum", "breakout", "meanrev"],
            "exposure_mult": 0.3,
            "note": "Risk-off: correlations go to 1, systems break. Small or flat.",
        },
    }


def hmm_available() -> bool:
    try:
        import hmmlearn  # noqa: F401
        return True
    except ImportError:
        return False


class HMMRegimeDetector:
    """4-state Gaussian HMM regime detector (vol × trend decomposition).

    Mined from the hmmlearn-based market-regime repos: states are learned
    from (log returns, rolling vol, momentum) via Baum-Welch and decoded
    with Viterbi, then mapped onto interpretable labels:

    - low vol + uptrend   → TREND_UP
    - low vol + downtrend → TREND_DOWN
    - high vol            → PANIC (down) / TREND_UP volatile (up)
    - near-zero drift     → RANGE / SQUEEZE by volatility level

    Requires ``hmmlearn`` (optional); without it, ``fit`` falls back to
    the rule-based :class:`RegimeDetector` and reports ``method="rule"``.
    """

    def __init__(self, n_states: int = 4, window: int = 20,
                 n_iter: int = 100, random_state: int = 42):
        self.n_states = max(2, int(n_states))
        self.window = max(5, int(window))
        self.n_iter = max(10, int(n_iter))
        self.random_state = int(random_state)
        self.model = None
        self.state_labels: dict[int, str] = {}

    def _features(self, df: _pd.DataFrame) -> _np.ndarray:
        df = ensure_ohlcv(df)
        c = df["close"].astype(float)
        lr = _np.log(c / c.shift(1)).fillna(0.0)
        vol = lr.rolling(self.window, min_periods=5).std().bfill().fillna(1e-9)
        mom = lr.rolling(self.window, min_periods=5).mean().bfill().fillna(0.0)
        X = _np.column_stack([lr.to_numpy(), vol.to_numpy(), mom.to_numpy()])
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-12
        return (X - mu) / sd

    def _map_states(self, X: _np.ndarray, states: _np.ndarray) -> dict:
        mapping = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                mapping[s] = "RANGE"
                continue
            m_ret = float(X[mask, 0].mean())
            m_vol = float(X[mask, 1].mean())
            m_mom = float(X[mask, 2].mean())
            if m_vol > 0.75:
                mapping[s] = "PANIC" if m_mom <= 0 else "TREND_UP"
            elif abs(m_mom) > 0.35 or abs(m_ret) > 0.25:
                mapping[s] = "TREND_UP" if m_mom >= 0 else "TREND_DOWN"
            elif m_vol < -0.5:
                mapping[s] = "SQUEEZE"
            else:
                mapping[s] = "RANGE"
        return mapping

    def fit(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """State label per bar + ``method`` used ('hmm' or 'rule')."""
        df = ensure_ohlcv(df)
        if not hmm_available():
            frame = RegimeDetector().fit(df)
            frame["method"] = "rule"
            return frame[["label", "method"]]
        from hmmlearn.hmm import GaussianHMM

        X = self._features(df)
        try:
            self.model = GaussianHMM(
                n_components=self.n_states, covariance_type="diag",
                n_iter=self.n_iter, random_state=self.random_state)
            states = self.model.fit_predict(X)
        except Exception:
            frame = RegimeDetector().fit(df)
            frame["method"] = "rule"
            return frame[["label", "method"]]
        self.state_labels = self._map_states(X, states)
        labels = [self.state_labels[int(s)] for s in states]
        # Viterbi flicker: 3-bar hysteresis, same as the rule detector.
        lab = _np.array([CODES[l] for l in labels])
        for i in range(1, len(lab)):
            if lab[i] != lab[i - 1]:
                j = i
                while j < len(lab) and lab[j] == lab[i]:
                    j += 1
                if j - i < 3:
                    lab[i:j] = lab[i - 1]
        labels = [LABELS[int(c)] for c in lab]
        out = _pd.DataFrame({"label": labels,
                             "state": [int(s) for s in states],
                             "method": "hmm"}, index=df.index)
        return out

    def current(self, df: _pd.DataFrame) -> dict:
        frame = self.fit(df)
        last = frame.iloc[-1]
        return {
            "label": str(last["label"]),
            "method": str(last["method"]),
            "persistence": persistence_score(frame["label"]),
            "expected_durations": expected_durations(frame["label"]),
        }
