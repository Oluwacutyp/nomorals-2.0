"""finance — agent tools for market analysis via FinancialExpert.

Wraps the FinancialExpert agent module so LLM agents and chat have a
finance tool surface (previously only the ``nm finance`` CLI existed).
Uses dynamic import to respect layering (tools L4 must not statically
import agents L5).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["register"]


def _expert_class() -> Any:
    """Dynamically load FinancialExpert (layering-safe)."""
    mod = importlib.import_module("nomorals.agents.financial_expert")
    return mod.FinancialExpert


def register(registry: Any) -> None:
    @registry.register(
        "finance_price",
        description=(
            "Get the current live price for a symbol — USE THIS for every "
            "price query (do NOT fetch price API URLs directly; many "
            "well-known endpoints like CoinDesk v1 are retired). "
            "Args: symbol (e.g. BTC, XAUUSD, AAPL), market (crypto|forex|stock, default crypto). "
            "Built-in fallback chain: Binance → CoinGecko (crypto), "
            "Yahoo → Stooq (stocks), Frankfurter (forex)."
        ),
        capability="network",
        parameters={
            "symbol": "str — trading symbol (e.g. BTC, XAUUSD)",
            "market": "str — crypto|forex|stock (default crypto)",
        },
    )
    def finance_price(context: Any, symbol: str, market: str = "crypto") -> dict[str, Any]:
        import importlib
        md = importlib.import_module("nomorals.integrations.market_data")
        return md.quote(symbol, market=market)

    @registry.register(
        "finance_analyze",
        description=(
            "Analyze a symbol: regime, indicators, trade idea. "
            "Args: symbol (e.g. BTC/USDT), market (crypto|stock|forex), "
            "timeframe (e.g. 1h, 1d)."
        ),
        capability="network",
        parameters={
            "symbol": "str — trading symbol",
            "market": "str — crypto|stock|forex (default crypto)",
            "timeframe": "str — bar timeframe (default 1h)",
        },
    )
    def finance_analyze(context: Any, symbol: str, market: str = "crypto",
                        timeframe: str = "1h") -> dict[str, Any]:
        expert = _expert_class()(context)
        report = expert.analyze(symbol, market=market, timeframe=timeframe)
        return report.to_dict() if hasattr(report, "to_dict") else {"result": str(report)}

    @registry.register(
        "finance_signal",
        description=(
            "Get a trading signal for a symbol (long/short/neutral with "
            "confidence). Args: symbol, market, timeframe."
        ),
        capability="network",
        parameters={
            "symbol": "str — crypto|stock|forex (default crypto)",
            "timeframe": "str — bar timeframe (default 1h)",
        },
    )
    def finance_signal(context: Any, symbol: str, market: str = "crypto",
                       timeframe: str = "1h") -> dict[str, Any]:
        expert = _expert_class()(context)
        sig = expert.signal(symbol, market=market, timeframe=timeframe)
        return sig.to_dict() if hasattr(sig, "to_dict") else {"result": str(sig)}

    @registry.register(
        "finance_backtest",
        description=(
            "Backtest a strategy on historical data. Args: symbol, market, "
            "timeframe, strategy name."
        ),
        capability="network",
        parameters={
            "symbol": "str — trading symbol",
            "market": "str — crypto|stock|forex (default crypto)",
            "timeframe": "str — bar timeframe (default 1h)",
            "strategy": "str — strategy name (default trend)",
        },
    )
    def finance_backtest(context: Any, symbol: str, market: str = "crypto",
                         timeframe: str = "1h",
                         strategy: str = "trend") -> dict[str, Any]:
        expert = _expert_class()(context)
        summary = expert.backtest(symbol, market=market, timeframe=timeframe,
                                  strategy=strategy)
        return summary.to_dict() if hasattr(summary, "to_dict") else {"result": str(summary)}

    # Expense tracking + conversational budgeting (Naira-first). Lives in
    # nomorals/finance/; registered here alongside the market tools.
    from ..finance.tools import register as _register_finance_tools

    _register_finance_tools(registry)
