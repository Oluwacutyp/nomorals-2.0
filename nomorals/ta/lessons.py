"""Hard lessons from honest backtests.

This module exists so Devon never repeats proven mistakes. Every lesson
below comes from a real backtest with real numbers — not theory, not
guru courses, not broker marketing.

Sources:
- ``vendor/sentinel/Sentinel_PHASE_BCD_REPORT.md`` (2026-09-07)
- Devon's own TA walk-forward backtest (2026-10-10): XAUUSD 1h, 2yr,
  11,454 bars, no lookahead, realistic costs.

The cardinal rule: **backtest results are evidence. Treat them as such.**
"""

from __future__ import annotations


__all__ = ["LESSONS", "check_lesson", "lesson_for"]


LESSONS: tuple[dict, ...] = (
    {
        "id": "xau-mr-no-edge",
        "symbol": "XAUUSD",
        "timeframes": ("1h", "15m"),
        "verdict": "DO NOT TRADE",
        "evidence": (
            "XAUUSD 1h (730d): -23.23%, PF 0.76, 46 trades, "
            "hit 20% emergency stop ~day 46. "
            "XAUUSD 15m (60d): -19.93%, PF 0.27, 10 trades, "
            "hit 20% emergency stop ~day 3."
        ),
        "why": (
            "Mean-reversion-dominant signal mix (37/46 trades on 1h) in a "
            "2024-2026 gold tape produces a coin-flip win rate (52.2%) while "
            "paying full spread. The problem is trade SELECTION, not stop "
            "distance or confluence layers. New layers cannot rescue a "
            "negative-expectancy base."
        ),
        "what_was_tried": (
            "Macro Trinity voting (DXY/US10Y/correlation), ADR stop floor, "
            "Asian range breakout — trimmed losses slightly (-25.10% → -23.23%) "
            "but did not create an edge. The report states: 'I do not believe "
            "they can [fix it], because the problem is not stops or confluence.'"
        ),
        "conditions_to_revisit": (
            "Raise min-score gate / kill mean-reversion on H1 in ranging regimes; "
            "session filter (losses concentrate in NewYork -$2,161 on 1h); "
            "sizing guardrail max_lot 0.1 (~0.8% risk/trade makes -25% into ~-9%). "
            "Each change needs a 4-scenario A/B before shipping."
        ),
    },
    {
        "id": "xau-15m-worse",
        "symbol": "XAUUSD",
        "timeframes": ("15m",),
        "verdict": "DO NOT TRADE",
        "evidence": (
            "XAUUSD 15m (60d): -19.93%, PF 0.27, WR 30.0%, 10 trades. "
            "The 15m 25%-WR mean-reversion cluster is the bleed."
        ),
        "why": (
            "Lower timeframes amplify noise without adding signal. "
            "Mean-reversion on 15m XAU is the worst config tested."
        ),
        "what_was_tried": "Same layers as 1h. Trinity voted neutral, ADR floor never bound.",
        "conditions_to_revisit": "None identified. Avoid until a fundamentally different approach is backtested.",
    },
    {
        "id": "btc-1h-thin-edge",
        "symbol": "BTCUSD",
        "timeframes": ("1h",),
        "verdict": "TRADE WITH CAUTION — data collection only",
        "evidence": (
            "BTCUSD 1h (730d): +16.54%, PF 1.06, 573 trades, WR 44.0%, "
            "MaxDD 13.92%. The ONLY green configuration."
        ),
        "why": (
            "PF 1.06 means the edge is roughly the size of the friction. "
            "Spread/swap/latency in live trading shave a chunk. "
            "This is a data-collecting config: run it, let ML learn, "
            "do NOT scale it yet."
        ),
        "what_was_tried": (
            "BTC microstructure sentiment layer (funding/OI/taker/Fear&Greed) "
            "could NOT be validated — Binance API blocked from sandbox (HTTP 451). "
            "The +16.54% came from the base strategy engine alone."
        ),
        "conditions_to_revisit": (
            "Re-run with sentiment layer ON (needs Binance access). "
            "Scale only after ML champion promotes out-of-sample."
        ),
    },
    {
        "id": "btc-15m-no-edge",
        "symbol": "BTCUSD",
        "timeframes": ("15m",),
        "verdict": "DO NOT TRADE",
        "evidence": (
            "BTCUSD 15m (60d): -17.38%, PF 0.73, WR 38.5%, 169 trades."
        ),
        "why": (
            "Same noise problem as XAU 15m. The 1h edge does not transfer down."
        ),
        "what_was_tried": "Base engine only (sentiment OFF).",
        "conditions_to_revisit": "None. The 1h timeframe holds the edge, if any.",
    },
    {
        "id": "devon-ta-xau-no-edge",
        "symbol": "XAUUSD",
        "timeframes": ("1h",),
        "verdict": "DO NOT TRADE",
        "evidence": (
            "Devon's own TA backtest (2026-10-10): XAUUSD 1h, 2 years, "
            "11,454 bars (GC=F futures). 123 trades: -6.53% return, "
            "PF 0.85, WR 36.6%, MaxDD 8.32%, Sharpe -0.54. "
            "Buy & hold over same period: +56.59%. "
            "Walk-forward, no lookahead, $0.35 spread + $0.10 slippage, "
            "1% risk/trade, 500-bar rolling window."
        ),
        "why": (
            "Devon's TA (structure + SMC + adaptive + regime + sessions + "
            "analyst fusion) does not have an edge on XAUUSD 1h. It "
            "outperforms Sentinel's -23.23% (better risk management: 8.3% "
            "vs 29.7% max DD) but still loses money while buy-and-hold "
            "gains 57%. The system takes too many counter-trend signals "
            "in a strong bull market. 36.6% win rate with PF 0.85 means "
            "the losers outweigh the winners."
        ),
        "what_was_tried": (
            "Full TA pipeline: fractal swings, S/R zones, order blocks, "
            "FVG, liquidity sweeps, volatility-breathing parameters, "
            "5-state regime detector, session killzones, analyst signal "
            "fusion with disagreement penalty. Min confidence 55."
        ),
        "conditions_to_revisit": (
            "Backtest on range-bound XAU periods (not just bull market). "
            "Try higher confidence threshold (>70). Add trend filter: "
            "no shorts when HTF is strongly bullish. Each change needs "
            "a fresh walk-forward backtest before the block lifts."
        ),
    },
    {
        "id": "sizing-is-survival",
        "symbol": "*",
        "timeframes": ("*",),
        "verdict": "RULE",
        "evidence": (
            "At $10k, 0.5 lots ≈ 4% risk/trade on XAU. A 20% drawdown is "
            "only 5 consecutive full stops away. With max_lot 0.1 (~0.8% "
            "risk/trade), a -25% two-year outcome becomes roughly -9% "
            "with identical signal quality."
        ),
        "why": (
            "Sizing does not create an edge — it makes the LACK of an edge "
            "survivable while data is collected. This is the biggest single "
            "lever and the most ignored."
        ),
        "what_was_tried": "N/A — sizing analysis, not a strategy change.",
        "conditions_to_revisit": "Always apply. No strategy ships without a sizing guardrail.",
    },
    {
        "id": "no-in-sample-tuning",
        "symbol": "*",
        "timeframes": ("*",),
        "verdict": "RULE",
        "evidence": (
            "Nothing in the Sentinel backtest was tuned in-sample. All "
            "thresholds are research-published values. Every layer is "
            "disclosed ON/OFF in the report header."
        ),
        "why": (
            "Adjusting parameters until the backtest turns green is how "
            "backtests lie. The report states: 'The one thing I will NOT do "
            "is keep adjusting until XAU turns green — that is how backtests lie.'"
        ),
        "what_was_tried": "N/A — methodology rule.",
        "conditions_to_revisit": (
            "Out-of-sample windows (last 120 days) checked separately. "
            "4-scenario A/B for every change. Never tune to flatter."
        ),
    },
)


def lesson_for(symbol: str, timeframe: str = "") -> dict | None:
    """Get the applicable lesson for a symbol/timeframe, or None.

    Returns the most specific matching lesson (symbol+timeframe match
    beats symbol-only, which beats universal rules).
    """
    sym = (symbol or "").upper()
    tf = (timeframe or "").lower()
    # Strip common suffixes: XAUUSDm -> XAUUSD
    for suffix in ("M", "M+", "."):
        if sym.endswith(suffix) and len(sym) > 6:
            sym = sym[: -len(suffix)]
    best: dict | None = None
    best_score = -1
    for lesson in LESSONS:
        lsym = lesson["symbol"]
        if lsym != "*" and lsym != sym:
            # Also match XAU prefix (XAUUSD matches XAU)
            if not (lsym == "XAUUSD" and sym.startswith("XAU")):
                if not (lsym == "BTCUSD" and sym.startswith("BTC")):
                    continue
        score = 1
        tfs = lesson["timeframes"]
        if "*" not in tfs:
            if tf and tf not in tfs:
                continue
            score = 2  # timeframe-specific match
        if score > best_score:
            best_score = score
            best = lesson
    return best


def check_lesson(symbol: str, timeframe: str = "") -> dict:
    """Pre-trade lesson check. Returns verdict + evidence.

    Call this before allowing ANY trade. If the verdict is
    "DO NOT TRADE", the trade should be blocked with the evidence
    shown to the user.
    """
    lesson = lesson_for(symbol, timeframe)
    if lesson is None:
        return {
            "verdict": "NO DATA",
            "symbol": symbol,
            "timeframe": timeframe,
            "message": (
                f"No backtest lesson on file for {symbol} {timeframe}. "
                "Proceed with standard risk controls, but this config "
                "has not been validated."
            ),
        }
    return {
        "verdict": lesson["verdict"],
        "lesson_id": lesson["id"],
        "symbol": symbol,
        "timeframe": timeframe,
        "evidence": lesson["evidence"],
        "why": lesson["why"],
        "conditions_to_revisit": lesson["conditions_to_revisit"],
    }
