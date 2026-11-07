"""Brain eval pack: the measuring stick for the models Devon's brain directs.

Imported into the repo from the standalone pack built 2026-10-22
(``~/workspace/codebeast-eval/`` — day log 2026-10-22, "Brain eval pack
BUILT + PROVEN"). 110 cases in three suites:

* **capability** (49) — arithmetic, logic traps, real coding (scored by
  RUNNING the code in an isolated subprocess), strict
  instruction-following, checkable facts. Deterministic scoring only.
* **refusal / freedom** (30) — does the model engage, without refusing
  or lecturing, on what its owner legitimately asks of his own model:
  security study, one-sided opinions, adult-but-legal creative writing,
  Devon-domain tasks. Scored COMPLIED / HEDGED / REFUSED from the
  transparent patterns in ``data/refusal_patterns.json``.
* **tools** (31) — fluency in Devon's ACTUAL tool-call dialect: the
  production ``_SYSTEM_TEMPLATE`` and ``parse_tool_calls`` from
  :mod:`nomorals.agents.partner.tool_loop`, listings rendered live from
  the real :class:`~nomorals.tools.registry.ToolRegistry`.

What changed in the move in-repo (the standalone pack's named drift
weakness, removed): the tools suite no longer carries a frozen copy of
the system template or the tool listings. Both are read live at run
time — the template imported from the tool loop, the listing lines
rendered by the live registry — and every run journal records the
template's hash, so a contract change shows up in the record instead of
silently invalidating old scores. Case data (prompts, checkers,
expectations) is unchanged from the 2026-10-22 pack except that
``data/listings.json`` now maps each listing key to tool NAMES only.

Runs go through Devon's own router machinery
(:class:`~nomorals.llm.router.LLMRouter`): pick a rung — ``groq``
(her Groq provider), ``chain`` (her configured chain, exactly as
production builds it), ``local`` (a llama.cpp server — the rung a
locally served model, e.g. the codebeast GGUF, is graded through) — and
every run journals to the durable eval store (see :mod:`.store`):
per-case verdict, latency, provider/model, pack version. Scores from
different models are comparable run-over-run; that ladder is the point
(Groq baseline first, then each candidate model as it lands).

Surfaces: ``nm braineval`` (CLI) and the ``brain_eval`` registry tool,
both thin shells over :class:`.engine.BrainEvalEngine`.
"""

from __future__ import annotations

#: Pack version — bump when case data or scoring changes meaningfully.
#: 1.0 = the 2026-10-22 standalone pack. 1.1 = in-repo import: same
#: cases and scoring, live template/listings instead of frozen copies.
PACK_VERSION = "1.1"

from .engine import BrainEvalEngine  # noqa: E402

__all__ = ["PACK_VERSION", "BrainEvalEngine"]
