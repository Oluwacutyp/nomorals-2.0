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
"""

from . import backtest, data, feeds, indicators, math, meta, pipeline, regime, risk, signals, strategies

__all__ = [
    "backtest",
    "data",
    "feeds",
    "indicators",
    "math",
    "meta",
    "pipeline",
    "regime",
    "risk",
    "signals",
    "strategies",
]
