"""Backtest engines: event-driven (realistic) + vectorized (fast research).

Ported from ``sentinel/backtest/engine.py`` (user's own Sentinel.py bot),
plus a small embargoed walk-forward harness over the canonical strategy zoo.

The event engine is a bar-by-bar broker simulator: spread, ATR slippage,
fees, funding, execution latency, leverage cap. The vector engine is the
fast approximation for research loops.
"""

from __future__ import annotations


from .data import split_embargo
from .math import (atr, calmar, cagr, deflated_sharpe, ensure_ohlcv,
                   expected_shortfall, max_drawdown, max_drawdown_duration,
                   omega_ratio, probabilistic_sharpe, profit_factor, sharpe,
                   sortino, tail_ratio, ulcer_index, value_at_risk)
from .signals import fuse_all
from .strategies import run_zoo


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



__all__ = ["EventBacktester", "VectorBacktester", "walk_forward",
           "periods_per_year", "tear_sheet", "monte_carlo_trades",
           "risk_of_ruin", "compare", "expectancy_by_side"]


def periods_per_year(index: _pd.Index) -> int:
    """Infer annualization from bar spacing (fallback: 252)."""
    try:
        if isinstance(index, _pd.DatetimeIndex) and len(index) > 5:
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
                 max_leverage: float = 1.0, stop_on_touch: bool = False,
                 stop_atr_mult: float = 2.0):
        self.cash = cash
        self.fee_bps = fee_bps
        self.slippage_atr = slippage_atr
        self.spread_bps = spread_bps
        self.funding_8h_bps = funding_8h_bps
        self.latency_bars = latency_bars
        self.max_leverage = max_leverage
        # Intrabar stop semantics (backtesting.py's lesson): when a
        # position is open, a stop ``stop_atr_mult`` ATRs away is checked
        # against the bar's high/low — a touch fills at the stop instead
        # of pretending the close was reachable. Conservative, honest.
        self.stop_on_touch = stop_on_touch
        self.stop_atr_mult = stop_atr_mult

    def _funding_per_bar(self, df: _pd.DataFrame) -> float:
        if not self.funding_8h_bps:
            return 0.0
        try:
            idx = df.index
            if isinstance(idx, _pd.DatetimeIndex) and len(idx) > 5:
                dt_h = ((idx[-1] - idx[0]).total_seconds()
                        / max(1, len(idx) - 1) / 3600.0)
                if dt_h > 0:
                    return (self.funding_8h_bps / 1e4) * (dt_h / 8.0)
        except Exception:  # noqa: E103 - documented 0.0 fallback
            pass
        return 0.0

    def run(self, df: _pd.DataFrame, position: _pd.Series,
            fraction: float = 1.0) -> dict:
        """Simulate ``position`` (-1..1 target exposure) bar by bar."""
        df = ensure_ohlcv(df)
        pos = (position.reindex(df.index).fillna(0.0).clip(-1, 1)
               .to_numpy(dtype=float))
        if self.latency_bars > 0:
            pos = _np.roll(pos, self.latency_bars)
            pos[: self.latency_bars] = 0.0
        pos = _np.clip(pos * fraction, -self.max_leverage, self.max_leverage)
        close = df["close"].to_numpy(dtype=float)
        a = atr(df).to_numpy(dtype=float)
        spread = close * (self.spread_bps / 1e4)
        slip = a * self.slippage_atr
        fee = self.fee_bps / 1e4
        fpb = self._funding_per_bar(df)

        cash, units = float(self.cash), 0.0
        entry_px, entry_units, entry_cost = None, 0.0, 0.0
        entry_i, mae, mfe = 0, 0.0, 0.0
        idx = df.index
        hi = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        a_arr = a  # ATR per bar, for the intrabar stop
        trades, eq, expos, traded = [], [], [], 0.0

        def _close_trade(exit_px: float, exit_cost: float, i: int,
                         exit_side: float, why: str):
            pnl = (exit_px - entry_px) * entry_units - entry_cost - exit_cost
            bars_held = i - entry_i
            denom = abs(entry_px * entry_units) + 1e-12
            trades.append({
                "entry_time": idx[entry_i], "exit_time": idx[i],
                "side": float(_np.sign(entry_units)),
                "entry_px": float(entry_px), "exit_px": float(exit_px),
                "pnl": float(pnl), "return_pct": float(pnl / denom * 100.0),
                "bars": int(bars_held),
                "mae_pct": float(mae * 100.0), "mfe_pct": float(mfe * 100.0),
                "exit": why,
            })

        for i in range(len(df)):
            px = close[i]
            if fpb and units != 0:
                cash -= abs(units * px) * fpb
            # ── intrabar stop touch: fill at the stop, not the close ──
            stopped = False
            if self.stop_on_touch and units != 0.0 and entry_px is not None:
                sgn = float(_np.sign(units))
                stop_px = (entry_px - sgn * self.stop_atr_mult * a_arr[i])
                touched = (lo[i] <= stop_px) if sgn > 0 else (hi[i] >= stop_px)
                if touched:
                    cost = abs(units * stop_px) * fee
                    cash += units * stop_px - cost
                    traded += abs(units * stop_px)
                    _close_trade(stop_px, cost, i, sgn, "stop")
                    entry_px, entry_units, entry_cost = None, 0.0, 0.0
                    units = 0.0
                    stopped = True
            # ── MAE/MFE excursion tracking while a trade is open ──
            if units != 0.0 and entry_px is not None and not stopped:
                sgn = float(_np.sign(units))
                if sgn > 0:
                    mae = min(mae, (lo[i] - entry_px) / (entry_px + 1e-12))
                    mfe = max(mfe, (hi[i] - entry_px) / (entry_px + 1e-12))
                else:
                    mae = min(mae, (entry_px - hi[i]) / (entry_px + 1e-12))
                    mfe = max(mfe, (entry_px - lo[i]) / (entry_px + 1e-12))
            equity = cash + units * px
            tgt_units = equity * pos[i] / px if px > 0 else 0.0
            prev_sign = float(_np.sign(units))
            if not stopped and abs(tgt_units - units) > 1e-12:
                side = float(_np.sign(tgt_units - units))
                trade_px = px + side * (spread[i] + slip[i])
                delta = tgt_units - units
                cost = abs(delta * trade_px) * fee
                cash -= delta * trade_px + cost
                traded += abs(delta * trade_px)
                units = tgt_units
                new_sign = float(_np.sign(units))
                if prev_sign == 0.0 and new_sign != 0.0:
                    entry_px, entry_units, entry_cost = trade_px, units, cost
                    entry_i, mae, mfe = i, 0.0, 0.0
                elif prev_sign != 0.0 and new_sign == 0.0:
                    _close_trade(trade_px, cost, i, prev_sign, "signal")
                    entry_px, entry_units, entry_cost = None, 0.0, 0.0
                elif prev_sign != 0.0 and new_sign != 0.0 \
                        and new_sign != prev_sign:
                    _close_trade(trade_px, cost / 2, i, prev_sign, "reverse")
                    entry_px, entry_units, entry_cost = trade_px, units, cost / 2
                    entry_i, mae, mfe = i, 0.0, 0.0
            equity = cash + units * px
            eq.append(equity)
            expos.append(abs(units * px) / (abs(equity) + 1e-12))
        if units != 0.0 and entry_px is not None:
            _close_trade(close[-1], 0.0, len(df) - 1, float(_np.sign(units)),
                         "open")
        return self._metrics(df, _np.array(eq), trades, traded, expos)

    def _metrics(self, df, eq, trades, traded, expos) -> dict:
        eq_s = _pd.Series(eq, index=df.index)
        rets = eq_s.pct_change().fillna(0.0)
        ppy = periods_per_year(df.index)
        years = max(1e-9, len(df) / ppy)
        cagr_v = float((eq[-1] / self.cash) ** (1 / years) - 1) \
            if eq[-1] > 0 else -1.0
        pnls = _np.array([t["pnl"] for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        ledger = _pd.DataFrame(trades)
        r_arr = rets.to_numpy(dtype=float)
        return {
            "final_equity": float(eq[-1]),
            "total_return": float(eq[-1] / self.cash - 1),
            "cagr": cagr_v,
            "sharpe": sharpe(rets, ppy),
            "sortino": sortino(rets, ppy),
            "calmar": calmar(r_arr, ppy),
            "max_dd": float(max_drawdown(eq_s)["max_dd"]),
            "max_dd_bars": max_drawdown_duration(eq),
            "ulcer": ulcer_index(eq),
            "profit_factor": profit_factor(rets),
            "omega": omega_ratio(r_arr),
            "tail_ratio": tail_ratio(r_arr),
            "var_95": value_at_risk(r_arr),
            "cvar_95": expected_shortfall(r_arr),
            # Trust layer (backtest-forensics lesson): is it skill or luck?
            "psr": probabilistic_sharpe(r_arr, 0.0, ppy),
            "dsr_20": deflated_sharpe(r_arr, 20, ppy),
            "win_rate": float(_np.mean(pnls > 0)) if len(pnls) else 0.0,
            "expectancy": float(_np.mean(pnls) / self.cash) if len(pnls) else 0.0,
            "expectancy_r": float(_np.mean(pnls) / (abs(losses).mean() + 1e-12))
            if len(losses) else 0.0,
            "trades": int(len(pnls)),
            "avg_win": float(wins.mean() / self.cash) if len(wins) else 0.0,
            "avg_loss": float(losses.mean() / self.cash) if len(losses) else 0.0,
            "avg_bars_held": float(ledger["bars"].mean()) if len(ledger) else 0.0,
            "avg_mae_pct": float(ledger["mae_pct"].mean()) if len(ledger) else 0.0,
            "avg_mfe_pct": float(ledger["mfe_pct"].mean()) if len(ledger) else 0.0,
            "turnover": float(traded / (self.cash + 1e-12)),
            "exposure": float(_np.mean(expos)) if expos else 0.0,
            "equity": eq_s,
            "trade_pnls": pnls,
            "ledger": ledger,
        }


class VectorBacktester:
    """Fast vectorized approximation for research and optimization loops."""

    def __init__(self, fee_bps: float = 5.0, latency_bars: int = 1):
        self.fee_bps = fee_bps
        self.latency_bars = latency_bars

    def run(self, df: _pd.DataFrame, position: _pd.Series) -> dict:
        df = ensure_ohlcv(df)
        rets = df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
        pos = (position.reindex(df.index).fillna(0.0).clip(-1, 1)
               .to_numpy(dtype=float))
        pos = _np.roll(pos, self.latency_bars)
        pos[: self.latency_bars] = 0.0
        turnover = _np.abs(_np.diff(pos, prepend=0.0))
        costs = turnover * (self.fee_bps / 1e4)
        strat = pos * rets - costs
        eq = 100_000.0 * _np.exp(_np.cumsum(_np.log1p(_np.clip(strat, -0.99, None))))
        eq_s = _pd.Series(eq, index=df.index)
        ppy = periods_per_year(df.index)
        return {
            "total_return": float(eq[-1] / eq[0] - 1),
            "sharpe": sharpe(_pd.Series(strat), ppy),
            "sortino": sortino(_pd.Series(strat), ppy),
            "max_dd": float(max_drawdown(eq_s)["max_dd"]),
            "turnover": float(turnover.mean()),
            "exposure": float(_np.mean(_np.abs(pos))),
            "equity": eq_s,
        }


def _committee_position(df: _pd.DataFrame, names: list[str]) -> _pd.Series:
    """Fused committee position for a set of strategy names."""
    frames = run_zoo(names, df)
    if not frames:
        return _pd.Series(0.0, index=df.index, name="position")
    fused = fuse_all(frames, min_agreement=0.0)
    return fused["position"]


def walk_forward(df: _pd.DataFrame, names: list[str] | None = None,
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
    rets = _np.array([f["total_return"] for f in folds])
    sharpes = _np.array([f["sharpe"] for f in folds])
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "mean_sharpe": float(sharpes.mean()),
        "worst_fold_return": float(rets.min()),
        "positive_folds": int((rets > 0).sum()),
    }


# ── sweep additions: trust layer + presentation ──────────────────────────

def _fmt(x: float, digits: int = 2) -> str:
    if x != x or x in (float("inf"), float("-inf")):  # nan / inf
        return "   n/a"
    return f"{x:>{7}.{digits}f}"


def tear_sheet(res: dict, title: str = "BACKTEST", theme: str = "rich") -> str:
    """QuantStats-style tear sheet as plain text — chat-native, not HTML.

    One scannable page: returns, risk-adjusted ratios, the trust layer
    (PSR/DSR), drawdown pain, and trade anatomy incl. MAE/MFE. ``theme``
    is "rich" (box-drawing) or "plain" (ASCII).
    """
    rich = theme != "plain"
    H, V = ("═", "║") if rich else ("=", "|")
    top = "╔" + H * 62 + "╗" if rich else "+" + "=" * 62 + "+"
    mid = "╠" + H * 62 + "╣" if rich else "+" + "=" * 62 + "+"
    bot = "╚" + H * 62 + "╝" if rich else "+" + "=" * 62 + "+"

    def row(label: str, val: str) -> str:
        return f"{V} {label:<34} {val:>24} {V}"

    def verdict() -> str:
        psr, dsr, n = res.get("psr", 0), res.get("dsr_20", 0), res.get("trades", 0)
        if n < 10:
            return "n/a — fewer than 10 trades"
        if dsr >= 0.95:
            return "LIKELY REAL — survives trial deflation"
        if psr >= 0.95:
            return "PROMISING — PSR strong, DSR weak (few trials?)"
        if psr >= 0.80:
            return "MARGINAL — needs more data / fewer trials"
        return "LIKELY LUCK — do not trade this"

    L = [top, row(title[:60], f"{res.get('trades', 0)} trades")]
    L.append(mid)
    L.append(row("Total return", f"{res.get('total_return', 0) * 100:+.2f}%"))
    L.append(row("CAGR", f"{res.get('cagr', 0) * 100:+.2f}%"))
    L.append(row("Final equity", f"${res.get('final_equity', 0):,.0f}"))
    L.append(mid)
    L.append(row("Sharpe", _fmt(res.get("sharpe", 0))))
    L.append(row("Sortino", _fmt(res.get("sortino", 0))))
    L.append(row("Calmar", _fmt(res.get("calmar", 0))))
    L.append(row("Omega", _fmt(res.get("omega", 0))))
    L.append(row("Tail ratio", _fmt(res.get("tail_ratio", 0))))
    L.append(mid)
    L.append(row("PSR  P(SR>0)", f"{res.get('psr', 0):.1%}"))
    L.append(row("DSR  (20 trials)", f"{res.get('dsr_20', 0):.1%}"))
    L.append(row("Verdict", verdict()[:24]))
    L.append(mid)
    L.append(row("Max drawdown", f"{res.get('max_dd', 0) * 100:.2f}%"))
    L.append(row("Max DD duration", f"{res.get('max_dd_bars', 0)} bars"))
    L.append(row("Ulcer index", _fmt(res.get("ulcer", 0))))
    L.append(row("Daily VaR 95%", f"{res.get('var_95', 0) * 100:.2f}%"))
    L.append(row("Daily CVaR 95%", f"{res.get('cvar_95', 0) * 100:.2f}%"))
    L.append(mid)
    L.append(row("Win rate", f"{res.get('win_rate', 0):.1%}"))
    L.append(row("Profit factor", _fmt(res.get("profit_factor", 0))))
    L.append(row("Expectancy (R)", _fmt(res.get("expectancy_r", 0))))
    L.append(row("Avg bars held", _fmt(res.get("avg_bars_held", 0), 1)))
    L.append(row("Avg MAE / MFE",
                 f"{res.get('avg_mae_pct', 0):.2f}% / "
                 f"{res.get('avg_mfe_pct', 0):+.2f}%"))
    L.append(row("Exposure", f"{res.get('exposure', 0):.1%}"))
    L.append(bot)
    return "\n".join(L)


def monte_carlo_trades(res: dict, n_sims: int = 2000,
                       seed: int = 42) -> dict:
    """Reshuffle the trade ledger: is the Sharpe timing or luck?

    Permutes trade order (destroys sequencing, keeps the return
    distribution) and compares the real Sharpe against the reshuffled
    distribution — the poor quant's PBO.
    """
    from .math import monte_carlo_shuffle

    ledger = res.get("ledger")
    if ledger is None or len(ledger) < 10:
        return {"n_trades": 0, "note": "need >= 10 trades"}
    rets = (ledger["pnl"].to_numpy(dtype=float)
            / (abs(ledger["pnl"]).mean() + 1e-12))
    mc = monte_carlo_shuffle(rets, n_sims=n_sims, seed=seed)
    real_sharpe = res.get("sharpe", 0.0)
    # Sharpe of the reshuffled trade-return series, scaled comparably.
    mc["real_sharpe"] = float(real_sharpe)
    mc["beats_shuffled_pct"] = float(
        100.0 * (real_sharpe > mc["sharpe_p95"]))
    mc["n_trades"] = int(len(ledger))
    return mc


def risk_of_ruin(win_rate: float, payoff: float,
                 risk_frac: float) -> float:
    """Probability of ruin before doubling (naive gambler's formula).

    ``payoff`` = avg win / avg loss, ``risk_frac`` = fraction risked/trade.
    """
    p = min(0.99, max(0.01, float(win_rate)))
    b = max(0.1, float(payoff))
    f = min(0.5, max(1e-4, float(risk_frac)))
    q = 1.0 - p
    # Fraction of a "doubling goal" in risk units.
    n = max(1, int(round(_np.log(2.0) / _np.log(1.0 + f * b))))
    if p <= q / b:
        return 1.0
    r = (q / (p * b)) if p * b > 0 else 1.0
    return float(_np.clip((r ** n), 0.0, 1.0))


def expectancy_by_side(res: dict) -> dict:
    """Expectancy split by long vs short from the trade ledger."""
    ledger = res.get("ledger")
    if ledger is None or ledger.empty:
        return {}
    out = {}
    for side, grp in ledger.groupby("side"):
        name = "long" if side > 0 else "short"
        out[name] = {
            "trades": int(len(grp)),
            "win_rate": float((grp["pnl"] > 0).mean()),
            "expectancy": float(grp["pnl"].mean()),
            "avg_mae_pct": float(grp["mae_pct"].mean()),
            "avg_mfe_pct": float(grp["mfe_pct"].mean()),
        }
    return out


def compare(results: dict[str, dict]) -> _pd.DataFrame:
    """Side-by-side comparison table for named backtest results."""
    rows = []
    for name, r in results.items():
        rows.append({
            "name": name,
            "return_%": r.get("total_return", 0) * 100,
            "sharpe": r.get("sharpe", 0),
            "sortino": r.get("sortino", 0),
            "calmar": r.get("calmar", 0),
            "max_dd_%": r.get("max_dd", 0) * 100,
            "psr": r.get("psr", 0),
            "dsr_20": r.get("dsr_20", 0),
            "win_rate": r.get("win_rate", 0),
            "trades": r.get("trades", 0),
            "expectancy_r": r.get("expectancy_r", 0),
        })
    df = _pd.DataFrame(rows).set_index("name")
    return df.sort_values("sharpe", ascending=False).round(3)
