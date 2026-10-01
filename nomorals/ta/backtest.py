"""Backtest engines: event-driven (realistic) + vectorized (fast research).

Ported from ``sentinel/backtest/engine.py`` (user's own Sentinel.py bot),
plus a small embargoed walk-forward harness over the canonical strategy zoo.

The event engine is a bar-by-bar broker simulator: spread, ATR slippage,
fees, funding, execution latency, leverage cap. The vector engine is the
fast approximation for research loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import split_embargo
from .math import atr, ensure_ohlcv, max_drawdown, profit_factor, sharpe, sortino
from .signals import fuse_all
from .strategies import run_zoo

__all__ = ["EventBacktester", "VectorBacktester", "walk_forward",
           "periods_per_year"]


def periods_per_year(index: pd.Index) -> int:
    """Infer annualization from bar spacing (fallback: 252)."""
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 5:
            dt = (index[-1] - index[0]).total_seconds() / max(1, len(index) - 1)
            if dt <= 0:
                return 252
            return max(1, int(round(365.25 * 86400 / dt)))
    except Exception:  # noqa: E103 - documented 252 fallback
        pass
    return 252


class EventBacktester:
    """Bar-by-bar broker simulator with spread, slippage, fees, funding."""

    def __init__(self, cash: float = 100_000.0, fee_bps: float = 5.0,
                 slippage_atr: float = 0.05, spread_bps: float = 2.0,
                 funding_8h_bps: float = 0.0, latency_bars: int = 1,
                 max_leverage: float = 1.0):
        self.cash = cash
        self.fee_bps = fee_bps
        self.slippage_atr = slippage_atr
        self.spread_bps = spread_bps
        self.funding_8h_bps = funding_8h_bps
        self.latency_bars = latency_bars
        self.max_leverage = max_leverage

    def _funding_per_bar(self, df: pd.DataFrame) -> float:
        if not self.funding_8h_bps:
            return 0.0
        try:
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > 5:
                dt_h = ((idx[-1] - idx[0]).total_seconds()
                        / max(1, len(idx) - 1) / 3600.0)
                if dt_h > 0:
                    return (self.funding_8h_bps / 1e4) * (dt_h / 8.0)
        except Exception:  # noqa: E103 - documented 0.0 fallback
            pass
        return 0.0

    def run(self, df: pd.DataFrame, position: pd.Series,
            fraction: float = 1.0) -> dict:
        """Simulate ``position`` (-1..1 target exposure) bar by bar."""
        df = ensure_ohlcv(df)
        pos = (position.reindex(df.index).fillna(0.0).clip(-1, 1)
               .to_numpy(dtype=float))
        if self.latency_bars > 0:
            pos = np.roll(pos, self.latency_bars)
            pos[: self.latency_bars] = 0.0
        pos = np.clip(pos * fraction, -self.max_leverage, self.max_leverage)
        close = df["close"].to_numpy(dtype=float)
        a = atr(df).to_numpy(dtype=float)
        spread = close * (self.spread_bps / 1e4)
        slip = a * self.slippage_atr
        fee = self.fee_bps / 1e4
        fpb = self._funding_per_bar(df)

        cash, units = float(self.cash), 0.0
        entry_px, entry_units, entry_cost = None, 0.0, 0.0
        trades, eq, expos, traded = [], [], [], 0.0
        for i in range(len(df)):
            px = close[i]
            if fpb and units != 0:
                cash -= abs(units * px) * fpb
            equity = cash + units * px
            tgt_units = equity * pos[i] / px if px > 0 else 0.0
            prev_sign = float(np.sign(units))
            if abs(tgt_units - units) > 1e-12:
                side = float(np.sign(tgt_units - units))
                trade_px = px + side * (spread[i] + slip[i])
                delta = tgt_units - units
                cost = abs(delta * trade_px) * fee
                cash -= delta * trade_px + cost
                traded += abs(delta * trade_px)
                units = tgt_units
                new_sign = float(np.sign(units))
                if prev_sign == 0.0 and new_sign != 0.0:
                    entry_px, entry_units, entry_cost = trade_px, units, cost
                elif prev_sign != 0.0 and new_sign == 0.0:
                    pnl = (trade_px - entry_px) * entry_units - entry_cost - cost
                    trades.append({"pnl": float(pnl), "bars": 0,
                                   "side": prev_sign})
                    entry_px, entry_units, entry_cost = None, 0.0, 0.0
                elif prev_sign != 0.0 and new_sign != 0.0 \
                        and new_sign != prev_sign:
                    pnl = (trade_px - entry_px) * entry_units \
                        - entry_cost - cost / 2
                    trades.append({"pnl": float(pnl), "bars": 0,
                                   "side": prev_sign})
                    entry_px, entry_units, entry_cost = trade_px, units, cost / 2
            equity = cash + units * px
            eq.append(equity)
            expos.append(abs(units * px) / (abs(equity) + 1e-12))
        if units != 0.0 and entry_px is not None:
            pnl = (close[-1] - entry_px) * entry_units - entry_cost
            trades.append({"pnl": float(pnl), "bars": 0,
                           "side": float(np.sign(units))})
        return self._metrics(df, np.array(eq), trades, traded, expos)

    def _metrics(self, df, eq, trades, traded, expos) -> dict:
        eq_s = pd.Series(eq, index=df.index)
        rets = eq_s.pct_change().fillna(0.0)
        ppy = periods_per_year(df.index)
        years = max(1e-9, len(df) / ppy)
        cagr = float((eq[-1] / self.cash) ** (1 / years) - 1) if eq[-1] > 0 else -1.0
        pnls = np.array([t["pnl"] for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        return {
            "final_equity": float(eq[-1]),
            "total_return": float(eq[-1] / self.cash - 1),
            "cagr": cagr,
            "sharpe": sharpe(rets, ppy),
            "sortino": sortino(rets, ppy),
            "max_dd": float(max_drawdown(eq_s)["max_dd"]),
            "profit_factor": profit_factor(rets),
            "win_rate": float(np.mean(pnls > 0)) if len(pnls) else 0.0,
            "expectancy": float(np.mean(pnls) / self.cash) if len(pnls) else 0.0,
            "trades": int(len(pnls)),
            "avg_win": float(wins.mean() / self.cash) if len(wins) else 0.0,
            "turnover": float(traded / (self.cash + 1e-12)),
            "exposure": float(np.mean(expos)) if expos else 0.0,
            "equity": eq_s,
            "trade_pnls": pnls,
        }


class VectorBacktester:
    """Fast vectorized approximation for research and optimization loops."""

    def __init__(self, fee_bps: float = 5.0, latency_bars: int = 1):
        self.fee_bps = fee_bps
        self.latency_bars = latency_bars

    def run(self, df: pd.DataFrame, position: pd.Series) -> dict:
        df = ensure_ohlcv(df)
        rets = df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
        pos = (position.reindex(df.index).fillna(0.0).clip(-1, 1)
               .to_numpy(dtype=float))
        pos = np.roll(pos, self.latency_bars)
        pos[: self.latency_bars] = 0.0
        turnover = np.abs(np.diff(pos, prepend=0.0))
        costs = turnover * (self.fee_bps / 1e4)
        strat = pos * rets - costs
        eq = 100_000.0 * np.exp(np.cumsum(np.log1p(np.clip(strat, -0.99, None))))
        eq_s = pd.Series(eq, index=df.index)
        ppy = periods_per_year(df.index)
        return {
            "total_return": float(eq[-1] / eq[0] - 1),
            "sharpe": sharpe(pd.Series(strat), ppy),
            "sortino": sortino(pd.Series(strat), ppy),
            "max_dd": float(max_drawdown(eq_s)["max_dd"]),
            "turnover": float(turnover.mean()),
            "exposure": float(np.mean(np.abs(pos))),
            "equity": eq_s,
        }


def _committee_position(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Fused committee position for a set of strategy names."""
    frames = run_zoo(names, df)
    if not frames:
        return pd.Series(0.0, index=df.index, name="position")
    fused = fuse_all(frames, min_agreement=0.0)
    return fused["position"]


def walk_forward(df: pd.DataFrame, names: list[str] | None = None,
                 n_splits: int = 4, embargo_bars: int = 20,
                 backtester: EventBacktester | None = None) -> dict:
    """Embargoed walk-forward: run the committee on each test fold.

    Splits ``df`` into ``n_splits`` chronological folds with an embargo gap
    (no look-ahead), runs the event backtester on each test fold, and
    aggregates. Rule-based strategies need no fitting, so this measures
    stability across regimes rather than tuning generalization.
    """
    from .strategies import list_strategies

    df = ensure_ohlcv(df)
    names = names or list_strategies()
    bt = backtester or EventBacktester()
    n = len(df)
    fold = n // (n_splits + 1)
    folds = []
    for k in range(n_splits):
        test_start = (k + 1) * fold + embargo_bars
        test_end = min(n, (k + 2) * fold)
        if test_end - test_start < 30:
            continue
        seg = df.iloc[test_start:test_end]
        pos = _committee_position(seg, names)
        res = bt.run(seg, pos)
        folds.append({
            "fold": k,
            "bars": len(seg),
            "total_return": res["total_return"],
            "sharpe": res["sharpe"],
            "max_dd": res["max_dd"],
            "trades": res["trades"],
        })
    if not folds:
        return {"folds": [], "n_folds": 0}
    rets = np.array([f["total_return"] for f in folds])
    sharpes = np.array([f["sharpe"] for f in folds])
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "mean_sharpe": float(sharpes.mean()),
        "worst_fold_return": float(rets.min()),
        "positive_folds": int((rets > 0).sum()),
    }
