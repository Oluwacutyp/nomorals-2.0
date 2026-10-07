"""God-tier agentic orchestration loop for owner chat.

A ReAct-style loop (think → act → observe) that dynamically decides which
tools to use, in what order, and changes strategy based on results.

Unlike the rigid intent → handler → response pipeline in agent_loop.py,
this loop lets the model choose tools step by step, recover from failures
by trying different approaches, and chain unexpected tool combinations.

Designed to work with weaker models (Groq 70B, local 3.8B) via highly
structured prompts and constrained action formats — no clever few-shot
chains that only work on frontier models.

Usage as an alternative mode in owner chat::

    from nomorals.agents.orchestration import run_agentic
    result = run_agentic(message, llm=llm, registry=registry, context=ctx)
"""

from .bridge import agentic_mode_enabled, maybe_run_agentic
from .context import LoopMemory
from .loop import AgenticLoop, LoopResult, run_agentic
from .tools import ToolAdapter

__all__ = [
    "AgenticLoop",
    "LoopMemory",
    "LoopResult",
    "ToolAdapter",
    "agentic_mode_enabled",
    "maybe_run_agentic",
    "run_agentic",
]
