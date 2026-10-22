"""finance — agent tools for market analysis via FinancialExpert.

Wraps the FinancialExpert agent module so LLM agents and chat have a
finance tool surface (previously only the ``nm finance`` CLI existed).
Uses dynamic import to respect layering (tools L4 must not statically
import agents L5).
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Any

__all__ = ["register"]


def _expert_class() -> Any:
    """Dynamically load FinancialExpert (layering-safe)."""
    mod = importlib.import_module("nomorals.agents.financial_expert")
    return mod.FinancialExpert


def _wants_context(fn: Any) -> bool:
    """True when ``fn`` declares a required leading ``context`` param.

    The finance tool family (this module's market tools and the whole
    ``nomorals.finance.tools`` package) is written context-first:
    ``def finance_alert(context, action=..., ...)``. But registry
    dispatch (``ToolRegistry._call_inner``) invokes ``spec.fn(**kwargs)``
    and NEVER supplies a context argument — the registry's documented
    convention (see ``tools/autonomy.py``) is that a dispatched tool
    closes over ``registry.context`` instead of receiving it. A
    context-first function registered raw therefore dies on every call
    with ``TypeError: ... missing 1 required positional argument:
    'context'`` — which is exactly how the scheduled "finance alerts"
    and "finance digest" jobs failed on every run (Pass #75).
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return bool(positional and positional[0].name == "context"
                and positional[0].default is inspect.Parameter.empty)


class _ContextBoundRegistry:
    """Registry view binding the registry's context into context-first
    tool functions at registration time.

    Wraps each registered function that declares a required leading
    ``context`` parameter in ``functools.partial(fn, registry.context)``
    — the registration-time equivalent of the closure-over-context
    pattern the scheduler bridge tools (``pulse``, ``briefing``, the
    ``__*_`` tools) are written with. Declared parameter metadata is
    untouched: the family's ``parameters=`` dicts never listed
    ``context`` (it is dispatcher-supplied), so a bound function's
    effective signature now matches its declared schema. Functions
    without a leading context parameter register unchanged.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.context = getattr(inner, "context", None)

    def _bind(self, fn: Any) -> Any:
        if fn is not None and _wants_context(fn):
            return functools.partial(fn, self.context)
        return fn

    def register(self, name: str, fn: Any = None, **meta: Any) -> Any:
        if fn is not None:
            return self._inner.register(name, self._bind(fn), **meta)
        decorator = self._inner.register(name, **meta)

        def _registering(func: Any) -> Any:
            return decorator(self._bind(func))

        return _registering

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def register(registry: Any) -> None:
    # Bind the app context into the context-first finance family (this
    # module's tools + the nomorals.finance.tools package below) — see
    # _ContextBoundRegistry. Without this, every registry dispatch of a
    # finance tool raised TypeError and the scheduled finance jobs
    # failed on every run.
    registry = _ContextBoundRegistry(registry)
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
