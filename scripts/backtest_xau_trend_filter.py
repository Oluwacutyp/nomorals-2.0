#!/usr/bin/env python3
"""XAUUSD trend-filter backtest — Devon's TA with HTF trend gate.

Variants:
  A (baseline): no filter
  B: skip SHORTS when 4h is in STRONG uptrend (close > EMA200 & ADX > 25)
  C: skip SHORTS when 4h is in ANY uptrend (close > EMA200) — longs only

Phase 1: walk-forward Analyst signals (expensive, done once).
Phase 2: apply each filter variant + simulate trades (cheap).

Method: walk-forward, 500-bar window, Analyst signals, enter at next open.
1% risk/trade, stop at invalidation, partials 1.5R/2.5R/4R (50/30/20).
Costs: $0.35 spread + $0.10 slippage per round trip, scaled by size.
No lookahead: HTF filter uses completed 4h bars only (shifted by 1).
Session gate uses each bar's own timestamp.
"""

import sys
import time
import json
import os

sys.path.insert(0, os.path.expanduser("~/workspace/devon"))

import numpy as np
import pandas as pd

from nomorals.ta.analyst import Analyst
from nomorals.ta.math import ema, resample_ohlcv
from nomorals.ta.indicators import adx
import nomorals.ta.sessions as sess_mod

_orig_gate = sess_mod.time_filter_ok
LOG = os.path.expanduser("~/workspace/trend_bt.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def collect_signals(df: pd.DataFrame, window: int = 500) -> list:
    """Walk-forward Analyst. Returns list of dicts with signal details."""
    analyst = Analyst()
    n = len(df)
    signals = []
    t0 = time.time()
    for i in range(window, n - 1):
        if (i - window) % 1000 == 0:
            log(f"bar {i-window}/{n-window-1} ({time.time()-t0:.0f}s)")
        bar_time = df.index[i]
        sess_mod.time_filter_ok = (
            lambda ts=None, allowed=("london", "overlap", "new_york"),
            _bt=bar_time: _orig_gate(ts=_bt, allowed=allowed)
        )
        w = df.iloc[i - window:i]
        try:
            idea = analyst.analyze(w, symbol="XAUUSD")
        except Exception:
            continue
        finally:
            sess_mod.time_filter_ok = _orig_gate
        if idea.side == 0:
            continue
        signals.append({
            "bar": i,
            "side": idea.side,
            "confidence": idea.confidence,
            "entry_ref": idea.entry,
            "stop_dist": abs(idea.entry - idea.invalidation),
            "regime": idea.regime,
        })
    log(f"collected {len(signals)} signals in {time.time()-t0:.0f}s")
    return signals


def simulate(signals: list, df: pd.DataFrame, filt: callable,
             start_cash: float = 100_000.0) -> dict:
    equity = start_cash
    peak = start_cash
    max_dd = 0.0
    trades = []
    n = len(df)

    for s in signals:
        if not filt(s):
            continue
        i = s["bar"]
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        entry_px = float(df["open"].iloc[entry_idx])
        stop_dist = s["stop_dist"]
        if stop_dist <= 0:
            continue
        side = s["side"]
        stop_px = entry_px - side * stop_dist
        tgts = [(entry_px + side * stop_dist * rm, wt, rm)
                for rm, wt in zip((1.5, 2.5, 4.0), (0.5, 0.3, 0.2))]

        risk_dollars = 0.01 * equity
        contracts = risk_dollars / (stop_dist * 100.0)  # GC=F $100/pt
        cost = 0.45 * contracts

        pnl = -cost
        remaining = 1.0
        r_total = 0.0
        open_tgts = [list(t) for t in tgts]
        exited = False
        end_idx = min(entry_idx + 200, n - 1)

        for j in range(entry_idx, end_idx):
            h = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])
            if side == 1:
                if lo <= stop_px:
                    pnl -= risk_dollars * remaining
                    r_total -= remaining
                    remaining = 0.0
                    exited = True
                    break
                for t in [x for x in open_tgts if h >= x[0]]:
                    pnl += risk_dollars * t[1] * t[2]
                    r_total += t[1] * t[2]
                    remaining -= t[1]
                open_tgts = [x for x in open_tgts if h < x[0]]
            else:
                if h >= stop_px:
                    pnl -= risk_dollars * remaining
                    r_total -= remaining
                    remaining = 0.0
                    exited = True
                    break
                for t in [x for x in open_tgts if lo <= x[0]]:
                    pnl += risk_dollars * t[1] * t[2]
                    r_total += t[1] * t[2]
                    remaining -= t[1]
                open_tgts = [x for x in open_tgts if lo > x[0]]
            if remaining <= 1e-9:
                exited = True
                break

        if not exited and remaining > 1e-9:
            final_px = float(df["close"].iloc[end_idx - 1])
            r_exit = (final_px - entry_px) * side / stop_dist
            pnl += risk_dollars * remaining * r_exit
            r_total += remaining * r_exit

        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        trades.append((pnl, r_total))

    pnls = np.array([t[0] for t in trades]) if trades else np.array([0.0])
    n_tr = len(trades)
    wins = int(np.sum(pnls > 0))
    gp = float(pnls[pnls > 0].sum()) if wins else 0.0
    gl = float(-pnls[pnls < 0].sum()) if n_tr - wins else 0.0
    rets = pnls / start_cash
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(60) \
        if n_tr > 1 else 0.0
    return {
        "trades": n_tr,
        "win_rate": round(100 * wins / n_tr, 1) if n_tr else 0.0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "return_pct": round(100 * (equity - start_cash) / start_cash, 2),
        "max_dd_pct": round(100 * max_dd, 2),
        "sharpe": round(float(sharpe), 2),
        "final_equity": round(equity, 2),
    }


def main():
    if os.path.exists(LOG):
        os.remove(LOG)
    log("Fetching GC=F 1h (2y)...")
    import yfinance as yf
    df = yf.download("GC=F", period="2y", interval="1h",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    log(f"{len(df)} bars: {df.index[0]} -> {df.index[-1]}")

    log("4h trend filter...")
    h4 = resample_ohlcv(df, "4h")
    e200 = ema(h4["close"], 200)
    ax = adx(h4, 14)
    sb_h4 = (h4["close"] > e200) & (ax["adx"] > 25)
    b_h4 = h4["close"] > e200
    sb = sb_h4.shift(1).reindex(df.index, method="ffill") \
        .fillna(False).values
    b = b_h4.shift(1).reindex(df.index, method="ffill") \
        .fillna(False).values
    log(f"strong_bull: {sb.mean()*100:.1f}% of bars, "
        f"bull: {b.mean()*100:.1f}%")

    log("Phase 1: collecting signals...")
    signals = collect_signals(df)
    for s in signals:
        s["strong_bull"] = bool(sb[s["bar"]])
        s["bull"] = bool(b[s["bar"]])

    # persist signals for reproducibility
    sig_path = os.path.expanduser("~/workspace/trend_bt_signals.json")
    with open(sig_path, "w") as f:
        json.dump(signals, f)
    log(f"signals saved to {sig_path}")

    log("Phase 2: simulating variants...")
    results = {
        "A": simulate(signals, df, lambda s: True),
        "B": simulate(signals, df,
                      lambda s: not (s["side"] == -1 and s["strong_bull"])),
        "C": simulate(signals, df,
                      lambda s: not (s["side"] == -1 and s["bull"])),
    }
    bh = (float(df["close"].iloc[-1]) / float(df["open"].iloc[501]) - 1) * 100
    for v in results:
        results[v]["buy_hold_pct"] = round(bh, 2)

    log("=" * 72)
    labels = {"A": "A: baseline (no filter)",
              "B": "B: no shorts in STRONG bull",
              "C": "C: longs only in ANY bull"}
    log(f"{'Variant':<30}{'Trades':>8}{'WR%':>8}{'PF':>8}"
        f"{'Return%':>10}{'MaxDD%':>9}{'Sharpe':>8}")
    log("-" * 72)
    for v in ("A", "B", "C"):
        r = results[v]
        log(f"{labels[v]:<30}{r['trades']:>8}{r['win_rate']:>8.1f}"
            f"{r['profit_factor']:>8.2f}{r['return_pct']:>10.2f}"
            f"{r['max_dd_pct']:>9.2f}{r['sharpe']:>8.2f}")
    log("-" * 72)
    log(f"Buy & hold: {bh:+.2f}%   |   v1 baseline: -6.53% (123 trades)")
    log("=" * 72)

    res_path = os.path.expanduser("~/workspace/trend_bt_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"results -> {res_path}")
    log("DONE")


if __name__ == "__main__":
    main()
