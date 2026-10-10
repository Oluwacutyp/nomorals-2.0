"""Research cost table — deliberately import-light.

This module exists to break the import cycle between
``nomorals.research.pipeline`` (which needs ``nomorals.llm`` for brain
calls) and ``nomorals.llm.router`` (which needs the flat per-call USD
cost for metering).  Both sides import :data:`COST_TABLE` from here;
this module imports nothing from the rest of nomorals so it can never
re-introduce the cycle.

Keep it boring: constants only, no imports beyond ``__future__``.
"""

from __future__ import annotations

__all__ = ["COST_TABLE"]

COST_TABLE: dict[str, float] = {
    "web_search": 0.001,   # one web_search call
    "web_fetch": 0.002,    # one web_fetch call (full page read)
    "llm_call": 0.0008,    # one decompose / synthesize / clarify LLM call
}
