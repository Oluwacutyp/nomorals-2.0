"""Brain eval tool: the ``brain_eval`` surface over BrainEvalEngine.

Grades a MODEL RUNG — never "the brain": Devon's brain is the
directing layer above all models; this tool exercises one rung
(Groq-hosted, her configured chain, or a locally served model) through
her router machinery against the 110-case eval pack (capability /
freedom / Devon tool fluency) and journals the run for comparison.

Actions: run | last | compare | status. A full run makes one router
call per case (110) — expect it to take a while on a real rung; use
``limit`` for a smoke slice. ``mock`` rung proves the machinery only
and is never a baseline.
"""

from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "brain_eval",
        description=(
            "Run or read the brain eval pack: grade a model rung "
            "(action='run', rung groq|chain|local|mock, optional model, "
            "suites, limit) against the 110 capability/freedom/tool-"
            "fluency cases through Devon's own router, journaled for "
            "comparison. action='last' returns the latest scorecard, "
            "'compare' the last two runs side by side, 'status' the "
            "pack + store state. Groq is the recorded baseline rung; "
            "local grades a llama.cpp server; mock is a machinery "
            "smoke, never a baseline."
        ),
        capability=Capability.NET_OUT,
        result_protocol="dict_ok",
    )
    def brain_eval(action: str = "status", *, rung: str = "groq",
                   model: str = "", base_url: str = "",
                   suites: str = "", limit: int = 0,
                   run_a: str = "", run_b: str = "") -> dict[str, Any]:
        from ..agents.braineval import BrainEvalEngine
        from ..agents.braineval.report import render_run

        engine = BrainEvalEngine()
        if action == "status":
            return {"ok": True, "text": engine.render_status(),
                    **engine.status()}
        if action == "run":
            wanted = [s.strip() for s in suites.split(",") if s.strip()] or None
            run = engine.run(rung, model=model, base_url=base_url,
                             suites=wanted, limit=int(limit or 0))
            if run.get("error"):
                return {"ok": False, "error": run["error"]}
            return {"ok": True, "run_id": run["run_id"],
                    "status": run["status"], "provider": run["provider"],
                    "model": run["model"], "aggregate": run["aggregate"],
                    "text": render_run(run)}
        if action == "last":
            run = engine.last()
            if run is None:
                return {"ok": False,
                        "error": "no eval runs journaled yet"}
            return {"ok": True, "run_id": run["run_id"],
                    "text": render_run(run)}
        if action == "compare":
            pair = engine.compare(run_a, run_b)
            if "error" in pair:
                return {"ok": False, "error": pair["error"]}
            from ..agents.braineval.report import render_compare
            return {"ok": True, "text": render_compare(pair["a"], pair["b"])}
        return {"ok": False, "error": f"unknown brain_eval action {action!r}"}
