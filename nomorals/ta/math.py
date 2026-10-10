"""Core TA math: OHLCV validation, averages, oscillators, performance stats.

Ported from ``sentinel/utils/helpers.py`` (user's own Sentinel.py bot) with
one addition: a textbook Wilder RSI (the original only computed RSI inline
inside generated code). numpy + pandas only; no NaNs leak past warmup.
"""

from __future__ import annotations

import math as _math
import random


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
    "OHLCV",
    "ensure_ohlcv",
    "ema",
    "sma",
    "rsi",
    "true_range",
    "atr",
    "rolling_zscore",
    "rolling_quantile",
    "log_returns",
    "sharpe",
    "sortino",
    "max_drawdown",
    "profit_factor",
    "clamp",
    "softmax",
    "resample_ohlcv",
    "seed_all",
    # ── sweep additions: full risk-adjusted stat zoo ──
    "cagr",
    "calmar",
    "sterling",
    "ulcer_index",
    "ulcer_performance_index",
    "value_at_risk",
    "expected_shortfall",
    "omega_ratio",
    "tail_ratio",
    "probabilistic_sharpe",
    "deflated_sharpe",
    "autocorr_adjusted_sharpe",
    "information_ratio",
    "parkinson_vol",
    "garman_klass_vol",
    "yang_zhang_vol",
    "rolling_sharpe",
    "max_drawdown_duration",
    "skew",
    "kurtosis",
    "monte_carlo_shuffle",
    "effective_n",
]

OHLCV = ("open", "high", "low", "close", "volume")


def ensure_ohlcv(df: _pd.DataFrame) -> _pd.DataFrame:
    """Validate a frame, sort the index, coerce dtypes, drop empty rows."""
    if df is None or len(df) == 0:
        raise ValueError("empty dataframe")
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype(float)
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype(float).fillna(0.0)
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) == 0:
        raise ValueError("empty dataframe after cleaning")
    return out


def ema(s: _pd.Series, span: int) -> _pd.Series:
    """Exponential moving average (Wilder/recursive, ``adjust=False``)."""
    return s.astype(float).ewm(span=max(2, int(span)), adjust=False).mean()


def sma(s: _pd.Series, window: int) -> _pd.Series:
    """Simple moving average (min_periods=1 so the head is usable)."""
    return s.astype(float).rolling(max(2, int(window)), min_periods=1).mean()


def rsi(close: _pd.Series, period: int = 14) -> _pd.Series:
    """Wilder RSI in [0, 100].

    Recursive Wilder smoothing (``ewm(alpha=1/period, adjust=False)``), seeded
    on the first observation. A perfectly flat series yields 50 (neutral);
    pure-up yields 100, pure-down yields 0. Warmup is backfilled.
    """
    period = max(2, int(period))
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100.0 - 100.0 / (1.0 + rs)
    flat = (avg_gain < 1e-12) & (avg_loss < 1e-12)
    out = out.mask(flat, 50.0)
    return out.bfill().fillna(50.0).rename(f"rsi_{period}")


def true_range(df: _pd.DataFrame) -> _pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    return _pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df: _pd.DataFrame, n: int = 14) -> _pd.Series:
    """Average true range (Wilder smoothing). Never NaN, floored at 1e-9."""
    tr = true_range(df)
    return (
        tr.ewm(alpha=1.0 / max(2, int(n)), adjust=False)
        .mean()
        .bfill()
        .fillna(1e-9)
        .rename(f"atr_{n}")
    )


def rolling_zscore(s: _pd.Series, window: int) -> _pd.Series:
    """Distance from the rolling mean in rolling standard deviations."""
    m = s.rolling(window, min_periods=max(2, window // 4)).mean()
    sd = (
        s.rolling(window, min_periods=max(2, window // 4))
        .std()
        .bfill()
        .ffill()
        .fillna(1e-9)
    )
    return ((s - m) / (sd + 1e-12)).fillna(0.0)


def rolling_quantile(s: _pd.Series, q: float, window: int) -> _pd.Series:
    """Rolling q-quantile, forward/back filled (adaptive thresholds)."""
    return (
        s.rolling(window, min_periods=max(5, window // 5))
        .quantile(q)
        .bfill()
        .ffill()
    )


def log_returns(close: _pd.Series) -> _pd.Series:
    """Log returns, first value 0."""
    c = close.astype(float)
    return _np.log(c / c.shift(1)).fillna(0.0)


def sharpe(returns: _pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio (0 when undefined)."""
    r = _np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(_np.mean(r) / sd * _math.sqrt(periods))


def sortino(returns: _pd.Series, periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only; 0 when undefined)."""
    r = _np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    dn = r[r < 0]
    if len(dn) < 2 or float(_np.std(dn)) <= 1e-12:
        return 0.0
    return float(_np.mean(r) / float(_np.std(dn)) * _math.sqrt(periods))


def max_drawdown(equity: _pd.Series) -> dict:
    """Max drawdown (negative fraction) plus peak/trough bar indices."""
    eq = _np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return {"max_dd": 0.0, "peak": 0, "trough": 0}
    peak = _np.maximum.accumulate(eq)
    dd = _np.where(peak > 0, (eq - peak) / peak, 0.0)
    trough = int(_np.argmin(dd))
    peak_i = int(_np.argmax(eq[: trough + 1])) if trough else 0
    return {"max_dd": float(dd[trough]), "peak": peak_i, "trough": trough}


def profit_factor(returns: _pd.Series) -> float:
    """Gross profit / gross loss (inf when no losses, 0 when nothing)."""
    r = _np.asarray(returns, dtype=float)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def softmax(d: dict) -> dict:
    """Softmax over a {key: score} mapping (numerically safe)."""
    if not d:
        return {}
    keys = list(d.keys())
    v = _np.array([float(d[k]) for k in keys], dtype=float)
    v = v - float(_np.max(v))
    e = _np.exp(_np.clip(v, -50, 50))
    s = float(e.sum()) or 1.0
    return {k: float(e[i] / s) for i, k in enumerate(keys)}


def resample_ohlcv(df: _pd.DataFrame, rule: str) -> _pd.DataFrame:
    """Resample bars to a higher timeframe (e.g. '4h', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    out = ensure_ohlcv(df).resample(rule).agg(agg).dropna(subset=["close"])
    return out


def seed_all(seed: int = 42) -> None:
    """Seed stdlib random + numpy (deterministic synthetic data / tests)."""
    random.seed(seed)
    _np.random.seed(seed % (2 ** 32 - 1))


# ── risk-adjusted performance zoo ────────────────────────────────────────
# Mined from empyrical / QuantStats / Bailey & López de Prado. All take
# simple (non-log) per-period returns unless noted; all return 0.0 (never
# inf/nan) when undefined.

def _as_returns(returns) -> _np.ndarray:
    return _np.asarray(returns, dtype=float)


def cagr(equity, periods: int = 252) -> float:
    """Compound annual growth rate from an equity curve."""
    eq = _np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return 0.0
    years = len(eq) / max(1, periods)
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def calmar(returns, periods: int = 252) -> float:
    """CAGR / |max drawdown| — return per unit of worst pain."""
    r = _as_returns(returns)
    if len(r) < 3:
        return 0.0
    eq = _np.cumprod(1.0 + r)
    dd = abs(float(max_drawdown(_pd.Series(eq))["max_dd"]))
    if dd <= 1e-12:
        return 0.0
    return cagr(eq, periods) / dd


def sterling(returns, periods: int = 252) -> float:
    """CAGR / average annual max drawdown (Sterling ratio, de-trended)."""
    r = _as_returns(returns)
    if len(r) < 3:
        return 0.0
    eq = _np.cumprod(1.0 + r)
    dd = abs(float(max_drawdown(_pd.Series(eq))["max_dd"]))
    years = max(1.0, len(r) / periods)
    avg_dd = dd / years + 0.10  # Sterling's +10% convention avoids div/0
    return cagr(eq, periods) / avg_dd


def ulcer_index(equity) -> float:
    """Ulcer Index: RMS of drawdown percentages (Peter Martin).

    Measures the depth *and* duration of pain — what Sharpe misses.
    """
    eq = _np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = _np.maximum.accumulate(eq)
    dd_pct = _np.where(peak > 0, 100.0 * (eq - peak) / peak, 0.0)
    return float(_np.sqrt(_np.mean(dd_pct ** 2)))


def ulcer_performance_index(returns, periods: int = 252) -> float:
    """Excess return / Ulcer Index — Sharpe for people who hate drawdowns."""
    r = _as_returns(returns)
    if len(r) < 3:
        return 0.0
    ui = ulcer_index(_np.cumprod(1.0 + r))
    if ui <= 1e-12:
        return 0.0
    return float(_np.mean(r) * periods / ui)


def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical VaR: the ``alpha``-quantile loss (positive number)."""
    r = _as_returns(returns)
    if len(r) < 5:
        return 0.0
    return float(-_np.quantile(r, _np.clip(alpha, 0.001, 0.5)))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall: mean loss beyond VaR (positive number)."""
    r = _as_returns(returns)
    if len(r) < 5:
        return 0.0
    var = -value_at_risk(r, alpha)
    tail = r[r <= var]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Probability-weighted gains / losses vs ``threshold`` (Keating/Shadwick)."""
    r = _as_returns(returns)
    if len(r) < 3:
        return 0.0
    gains = float(_np.sum(_np.maximum(r - threshold, 0.0)))
    losses = float(_np.sum(_np.maximum(threshold - r, 0.0)))
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def tail_ratio(returns) -> float:
    """95th percentile / |5th percentile| — asymmetry of the tails."""
    r = _as_returns(returns)
    if len(r) < 10:
        return 0.0
    up = float(_np.quantile(r, 0.95))
    dn = abs(float(_np.quantile(r, 0.05)))
    if dn <= 1e-12:
        return float("inf") if up > 0 else 0.0
    return up / dn


def skew(returns) -> float:
    """Sample skewness (0 when undefined)."""
    r = _as_returns(returns)
    if len(r) < 4:
        return 0.0
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(_np.mean(((r - r.mean()) / sd) ** 3))


def kurtosis(returns) -> float:
    """Pearson kurtosis (normal == 3.0; 0 when undefined)."""
    r = _as_returns(returns)
    if len(r) < 5:
        return 0.0
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(_np.mean(((r - r.mean()) / sd) ** 4))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))


def probabilistic_sharpe(returns, benchmark_sr: float = 0.0,
                         periods: int = 252) -> float:
    """PSR: P(true Sharpe > ``benchmark_sr``) — Bailey & López de Prado.

    Accounts for sample length, skewness and fat tails. The honest answer
    to "is this backtest skill or luck".
    """
    r = _as_returns(returns)
    n = len(r)
    if n < 10:
        return 0.0
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    sr = float(_np.mean(r) / sd * _math.sqrt(periods))
    g3 = skew(r)
    g4 = kurtosis(r)  # Pearson (normal == 3)
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 1e-12:
        return 0.0
    z = (sr - benchmark_sr) * _math.sqrt(n - 1) / _math.sqrt(denom)
    return float(_np.clip(_norm_cdf(z), 0.0, 1.0))


def deflated_sharpe(returns, n_trials: int, periods: int = 252) -> float:
    """DSR: PSR corrected for running ``n_trials`` strategies.

    The expected Sharpe of the best of ``n_trials`` lucky strategies
    becomes the benchmark. Use the number of configurations you actually
    tried — honesty in, honesty out.
    """
    r = _as_returns(returns)
    n = len(r)
    if n < 10 or n_trials < 1:
        return 0.0
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    sr = float(_np.mean(r) / sd * _math.sqrt(periods))
    g3 = skew(r)
    g4 = kurtosis(r)
    # Bailey & López de Prado (2014): expected Sharpe of the best of K
    # lucky trials under the null, via the Gumbel approximation.
    gamma = 0.5772156649  # Euler-Mascheroni
    sr0 = ((1.0 - gamma) * _norm_ppf(1.0 - 1.0 / n_trials)
           + gamma * _norm_ppf(1.0 - 1.0 / (n_trials * _math.e)))
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 1e-12:
        return 0.0
    # Null variance of SR estimate (Lo 2002 form with n-1).
    var_sr0 = (1.0 - g3 * sr0 + ((g4 - 1.0) / 4.0) * sr0 * sr0) / (n - 1)
    if var_sr0 <= 1e-12:
        return 0.0
    z = (sr - sr0) / _math.sqrt(var_sr0)
    return float(_np.clip(_norm_cdf(z), 0.0, 1.0))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation) — no scipy needed."""
    p = _np.clip(p, 1e-12, 1.0 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = _math.sqrt(-2 * _math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = _math.sqrt(-2 * _math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                             + b[4]) * r + 1)


def autocorr_adjusted_sharpe(returns, periods: int = 252,
                             max_lag: int = 10) -> float:
    """Sharpe with Lo (2002) autocorrelation annualization correction.

    Smoothed/autocorrelated returns inflate naive Sharpe; this deflates
    the annualization factor by the autocorrelation structure.
    """
    r = _as_returns(returns)
    n = len(r)
    if n < max_lag + 5:
        return sharpe(_pd.Series(r), periods)
    sd = float(_np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    q = periods
    rho = [_np.corrcoef(r[:-k], r[k:])[0, 1] if k < n - 1 else 0.0
           for k in range(1, min(max_lag, n - 2) + 1)]
    rho = [0.0 if not _np.isfinite(x) else float(x) for x in rho]
    factor = q + 2.0 * sum((q - k) * rho[k - 1] for k in range(1, len(rho) + 1))
    factor = max(factor, 1.0)
    return float(_np.mean(r) / sd * _math.sqrt(factor))


def information_ratio(returns, benchmark) -> float:
    """Mean active return / tracking error (annualized inside is caller's)."""
    r = _as_returns(returns)
    b = _as_returns(benchmark)
    n = min(len(r), len(b))
    if n < 3:
        return 0.0
    active = r[:n] - b[:n]
    te = float(_np.std(active, ddof=1))
    if te <= 1e-12:
        return 0.0
    return float(_np.mean(active) / te)


# ── range-based volatility estimators ────────────────────────────────────
# More efficient than close-to-close: they use the full bar path.

def parkinson_vol(df, window: int = 20) -> _pd.Series:
    """Parkinson (1980): uses high/low — ~5x more efficient than CC."""
    df = ensure_ohlcv(df)
    hl = _np.log(df["high"].astype(float) / df["low"].astype(float).clip(lower=1e-12))
    var = (hl ** 2 / (4.0 * _math.log(2.0))).rolling(
        max(2, int(window)), min_periods=2).mean()
    return _np.sqrt(var).bfill().fillna(0.0).rename(f"parkinson_{window}")


def garman_klass_vol(df, window: int = 20) -> _pd.Series:
    """Garman-Klass (1980): OHLC estimator, ~7x more efficient than CC."""
    df = ensure_ohlcv(df)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    term = (0.5 * _np.log(h / l.clip(lower=1e-12)) ** 2
            - (2.0 * _math.log(2.0) - 1.0)
            * _np.log(c / o.clip(lower=1e-12)) ** 2)
    var = term.rolling(max(2, int(window)), min_periods=2).mean().clip(lower=0.0)
    return _np.sqrt(var).bfill().fillna(0.0).rename(f"garman_klass_{window}")


def yang_zhang_vol(df, window: int = 20) -> _pd.Series:
    """Yang-Zhang (2000): drift-independent, minimum-variance estimator."""
    df = ensure_ohlcv(df)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    w = max(2, int(window))
    co = _np.log(c / o.clip(lower=1e-12))
    oc = _np.log(o / c.shift(1).clip(lower=1e-12))
    oc = oc.fillna(0.0)
    rs = _np.log(h / c.clip(lower=1e-12)) * _np.log(h / o.clip(lower=1e-12)) \
        + _np.log(l / c.clip(lower=1e-12)) * _np.log(l / o.clip(lower=1e-12))
    k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
    var = (oc.rolling(w, min_periods=2).var()
           + k * co.rolling(w, min_periods=2).var()
           + (1.0 - k) * rs.rolling(w, min_periods=2).mean()).clip(lower=0.0)
    return _np.sqrt(var).bfill().fillna(0.0).rename(f"yang_zhang_{window}")


# ── rolling / diagnostic helpers ─────────────────────────────────────────

def rolling_sharpe(returns, window: int = 63,
                   periods: int = 252) -> _pd.Series:
    """Rolling annualized Sharpe — regime-stability of the edge."""
    r = _pd.Series(_as_returns(returns))
    m = r.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = r.rolling(window, min_periods=max(5, window // 3)).std(ddof=1)
    out = m / (sd + 1e-12) * _math.sqrt(periods)
    return out.fillna(0.0).rename(f"rolling_sharpe_{window}")


def max_drawdown_duration(equity) -> int:
    """Longest underwater stretch in bars."""
    eq = _np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0
    peak = _np.maximum.accumulate(eq)
    under = eq < peak
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def monte_carlo_shuffle(returns, n_sims: int = 2000,
                        seed: int = 42) -> dict:
    """Reshuffle returns ``n_sims`` times: distribution of Sharpe/maxDD.

    Destroys timing/sequence — if the real Sharpe sits far outside the
    reshuffled distribution, the edge lives in *sequencing*, not luck.
    """
    r = _as_returns(returns)
    rng = _np.random.default_rng(seed)
    sharpes, dds = [], []
    for _ in range(max(1, int(n_sims))):
        rs = rng.permutation(r)
        sd = float(_np.std(rs, ddof=1))
        sharpes.append(float(_np.mean(rs) / (sd + 1e-12)))
        eq = _np.cumprod(1.0 + rs)
        dds.append(abs(float(max_drawdown(_pd.Series(eq))["max_dd"])))
    sharpes = _np.array(sharpes)
    dds = _np.array(dds)
    return {
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_p95": float(_np.quantile(sharpes, 0.95)),
        "sharpe_p5": float(_np.quantile(sharpes, 0.05)),
        "maxdd_mean": float(dds.mean()),
        "maxdd_p95": float(_np.quantile(dds, 0.95)),
        "n_sims": int(n_sims),
    }


def effective_n(weights) -> float:
    """Participation ratio: 1/sum(w²) — how many *independent* bets.

    A 9-strategy committee where 3 strategies carry all the weight has an
    effective N near 3, not 9. Used by signal fusion and vote quality.
    """
    w = _np.asarray(list(weights.values()) if isinstance(weights, dict)
                   else weights, dtype=float)
    w = _np.abs(w)
    tot = w.sum()
    if tot <= 1e-12:
        return 0.0
    w = w / tot
    return float(1.0 / (_np.sum(w ** 2) + 1e-12))

