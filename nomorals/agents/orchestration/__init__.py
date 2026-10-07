"""God-tier agentic orchestration loop for owner chat.

A ReAct-style loop (think → act → observe) that dynamically decides which
tools to use, in what order, and changes strategy based on results.

Unlike the rigid intent → handler → response pipeline in agent_loop.py,
this loop lets the model choose tools step by step, dispatch independent
calls in parallel, recover from failures by trying different approaches,
and chain unexpected tool combinations. Tool selection is
relevance-ranked so every registered tool stays reachable, and a loop
that pauses with ``ask`` resumes its plan on the user's next message
instead of starting fresh.

Prompts are highly structured (strict JSON action format) so the loop
stays robust on smaller models — structure is reliability, not a
capability ceiling. Maximum capability always.

Usage as an alternative mode in owner chat::

    from nomorals.agents.orchestration import run_agentic
    result = run_agentic(message, llm=llm, registry=registry)
    # when result.asked_user: persist result.memory_snapshot, then on the
    # user's reply: run_agentic(reply, llm=llm, registry=registry,
    #                           resume_from=result.memory_snapshot)
"""

from .bridge import agentic_mode_enabled, is_model_available, maybe_run_agentic
from .context import LoopMemory
from .loop import AgenticLoop, LoopResult, run_agentic
from .planner_bridge import PlannerBridge, model_steps_to_plan, observation_from_result
from .tools import ToolAdapter

__all__ = [
    "AgenticLoop",
    "LoopMemory",
    "LoopResult",
    "PlannerBridge",
    "ToolAdapter",
    "agentic_mode_enabled",
    "is_model_available",
    "maybe_run_agentic",
    "model_steps_to_plan",
    "observation_from_result",
    "run_agentic",
]
