"""Devon's native technical-analysis kit.

Ported from the genuinely good math inside the user's own Sentinel.py bot
(``sentinel/utils/helpers.py``, ``backtest/engine.py``, ``risk/manager.py``,
``signals/fusion.py``, ``core/regime.py``, ``core/adaptive.py``,
``strategies/base.py``, ``data/feed.py``, ``ml/meta.py``) — the same
battle-tested algorithms, now first-class Devon modules with no submodule
dependency and no codegen filler.

Deliberately NOT ported: the ``generated_*`` indicator/strategy/pattern zoos
(~33k lines of codegen permutations with fantasy names like
``m_rsi_stoch_halo`` / ``StratRailAlpha`` — near-duplicate parameter
permutations with no evidence of edge), the QuantumEngine glue over that zoo,
the ccxt live broker, and the old CLI. Quality over quantity.

Sweep upgrades (mined from finta, TA-Lib, talipp, backtesting.py, vectorbt,
QuantStats, López de Prado AFML, hmmlearn regime repos, Bulkowski):
``patterns`` (30 candlestick patterns with confluence scoring),
streaming O(1) indicators, the full risk-adjusted stat zoo (PSR/DSR/Calmar/
Omega/CVaR/...), range-based vol estimators, MAE/MFE trade ledgers,
text tear sheets, diversified/stacked fusion, 7 new strategies, HMM regime
detection, live position tracking, triple-barrier meta-labels, purged CV,
keyless Kraken/Bybit feeds with disk cache, and a chat-native briefing
renderer.
"""

from . import (analyst, backtest, data, feeds, indicators, math, meta,
               patterns, pipeline, regime, risk, sessions, signals, smc,
               strategies, structure)

__all__ = [
    "analyst",
    "backtest",
    "data",
    "feeds",
    "indicators",
    "math",
    "meta",
    "patterns",
    "pipeline",
    "regime",
    "risk",
    "sessions",
    "signals",
    "smc",
    "strategies",
    "structure",
]
