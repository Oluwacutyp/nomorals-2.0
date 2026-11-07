"""BrainEvalEngine — the facade behind ``nm braineval`` and the
``brain_eval`` registry tool.

A run picks a RUNG (which of Devon's provider paths grades the pack):

  groq   — her Groq provider alone (the recorded baseline rung: the
           hosted model rung Devon currently rents, which the pack was
           built to judge her own models against — day log 2026-10-22,
           register §3.10).
  chain  — her configured provider chain, built exactly as production
           builds it (defaults.build_chain): fallbacks included, so the
           score is the chain's, and each case row names the provider
           that actually answered.
  local  — a llama.cpp server (LlamaCppProvider): the rung a locally
           served model is graded through — e.g. the codebeast GGUF
           under llama-server. Nothing about this rung assumes a server
           is up; if none answers, the run journals as failed with the
           reason, which is the honest record.
  mock   — the deterministic offline provider. SMOKE ONLY: it proves
           the machinery end-to-end and is labeled as such in every
           record it produces. It is never a baseline.

Every run journals to the eval store (see :mod:`.store`) and, when the
run completed and the engine has a database (the CLI context carries
one), the scores are also recorded on the model in the ModelRegistry
(``record_eval``) so the broker's ``best_eval``/``beats_incumbent``
comparisons can see them. A registry miss is journaled, never fatal.
"""
from __future__ import annotations

import time
from typing import Any

from ...core.logging_setup import get_logger
from ...llm.router import LLMRouter
from . import PACK_VERSION
from .cases import SUITES, load_all, validate_pack
from .report import aggregate, render_compare, render_run
from .runner import EvalRunner
from .store import EvalStore, new_run_id

_log = get_logger(__name__)

RUNGS = ("groq", "chain", "local", "mock")


class BrainEvalEngine:
    def __init__(self, context: Any = None, store: EvalStore | None = None):
        self.context = context
        self.store = store or EvalStore()

    # ------------------------------------------------------------ routers
    def build_router(self, rung: str, *, model: str = "",
                     base_url: str = "") -> tuple[Any, dict[str, Any]]:
        """Build the router for one rung through Devon's own machinery.
        Single-provider rungs get an eval-isolated router (replay pinned
        OFF, cassettes under the eval store) so an eval can never be
        served a recorded answer or pollute the production cassette
        store. The chain rung IS the production chain, verbatim."""
        from ...llm.providers.registry import build_provider

        if rung == "chain":
            from ...llm.defaults import build_chain
            return build_chain(), {"provider": "chain",
                                   "model": model or "(chain default)"}
        if rung == "groq":
            provider = build_provider("groq", **({"model": model} if model else {}))
        elif rung == "local":
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if model:
                kwargs["model"] = model
            provider = build_provider("llama_cpp", **kwargs)
        elif rung == "mock":
            from ...llm.providers.mock import MockProvider
            provider = MockProvider(model=model or "mock-7b")
        else:
            raise ValueError(f"unknown rung {rung!r}; choose from {RUNGS}")
        router = LLMRouter(
            replay=False,
            cassette_dir=str(self.store.root / "cassettes"))
        router.add(provider, primary=True, name=provider.name)
        return router, {"provider": provider.name,
                        "model": model or getattr(provider, "model", "") or ""}

    # ------------------------------------------------------------ running
    def run(self, rung: str = "groq", *, model: str = "", base_url: str = "",
            suites: list[str] | None = None, limit: int = 0) -> dict[str, Any]:
        """Execute the pack on one rung and journal the run.

        ``limit`` > 0 runs only the first ``limit`` cases of EACH suite
        (a smoke slice); 0 = the full pack. Returns the run record, or
        {"error": ...} when the pack itself is invalid."""
        pack = validate_pack()
        if not pack["ok"]:
            return {"error": "eval pack failed validation: "
                             + "; ".join(pack["problems"])}
        wanted = [s for s in (suites or SUITES) if s in SUITES]
        cases = load_all(wanted)
        if limit > 0:
            trimmed: list[dict[str, Any]] = []
            per_suite: dict[str, int] = {}
            for case in cases:
                taken = per_suite.get(case["suite"], 0)
                if taken < limit:
                    trimmed.append(case)
                    per_suite[case["suite"]] = taken + 1
            cases = trimmed

        run_id = new_run_id(rung)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        router, rung_info = self.build_router(
            rung, model=model, base_url=base_url)
        runner = EvalRunner(router)
        rows, aborted = runner.run(cases)
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        attempted = [r for r in rows if r.get("response")]
        answered_by = next(
            (r for r in rows if r.get("provider")), None)
        status = "complete"
        if not attempted:
            status = "failed"
        elif aborted:
            status = "partial"
        notes: list[str] = []
        if rung == "mock":
            notes.append("smoke rung — deterministic offline provider; "
                         "NOT a baseline")
        if status == "failed":
            errors = [r["error"] for r in rows
                      if r.get("error") and not r.get("not_run")]
            dominant = max(set(errors), key=errors.count) if errors else "no responses"
            notes.append(f"provider never answered; dominant error: {dominant[:200]}")
        elif aborted:
            notes.append("run aborted after consecutive provider errors; "
                         "unattempted cases are journaled as not-run")

        run: dict[str, Any] = {
            "run_id": run_id,
            "pack_version": PACK_VERSION,
            "rung": rung,
            "provider": (answered_by or {}).get("provider") or rung_info["provider"],
            "model": (answered_by or {}).get("model") or rung_info["model"],
            "status": status,
            "started": started,
            "finished": finished,
            "suites": wanted,
            "case_count": len(cases),
            "contract": (runner.contract_info()
                         if any(c["suite"] == "tools" for c in cases) else {}),
            "aggregate": aggregate(rows),
            "note": " · ".join(notes),
            "cases": rows,
        }
        run["registry_recorded"] = self._record_in_registry(run)
        self.store.save_run(run)
        return run

    def _record_in_registry(self, run: dict[str, Any]) -> str:
        """Best-effort: file a completed run's scores on the model in
        the ModelRegistry. Returns the outcome for the run record."""
        if run["status"] != "complete" or not run.get("model"):
            return "skipped (run not complete or model unnamed)"
        db = getattr(self.context, "db", None)
        if db is None:
            return "skipped (no database in context)"
        try:
            from ...llm.registry import ModelRegistry
            agg = run["aggregate"]
            scores = {
                "overall": agg["overall"],
                "pack_version": run["pack_version"],
                "run_id": run["run_id"],
                **{suite: bucket.get("score")
                   for suite, bucket in agg["suites"].items()},
            }
            ModelRegistry(db).record_eval(run["model"], scores)
            return "recorded"
        except Exception as exc:  # noqa: BLE001 — a registry miss is journaled, never fatal
            return f"skipped ({type(exc).__name__}: {exc})"

    # ------------------------------------------------------------ reading
    def last(self) -> dict[str, Any] | None:
        return self.store.last()

    def compare(self, a_id: str = "", b_id: str = "") -> dict[str, Any]:
        """Resolve two runs to compare: explicit ids, else the last two."""
        if a_id and b_id:
            a, b = self.store.get(a_id), self.store.get(b_id)
            if a is None or b is None:
                return {"error": "one or both run ids not found in the eval store"}
            return {"a": a, "b": b}
        summaries = self.store.summaries()
        if len(summaries) < 2:
            return {"error": "fewer than two runs journaled — nothing to compare yet"}
        a = self.store.get(summaries[-2]["run_id"])
        b = self.store.get(summaries[-1]["run_id"])
        if a is None or b is None:
            return {"error": "run detail file missing for a journaled run"}
        return {"a": a, "b": b}

    def status(self) -> dict[str, Any]:
        summaries = self.store.summaries()
        return {
            "pack_version": PACK_VERSION,
            "pack": validate_pack(),
            "store_root": str(self.store.root),
            "runs_journaled": len(summaries),
            "last_run": summaries[-1] if summaries else None,
            "rungs": list(RUNGS),
        }

    # ------------------------------------------------------------ renders
    def render_last(self) -> str:
        run = self.last()
        if run is None:
            return "no eval runs journaled yet — run `nm braineval run` first"
        return render_run(run)

    def render_compare(self, a_id: str = "", b_id: str = "") -> str:
        pair = self.compare(a_id, b_id)
        if "error" in pair:
            return pair["error"]
        return render_compare(pair["a"], pair["b"])

    def render_status(self) -> str:
        info = self.status()
        pack = info["pack"]
        lines = [
            f"Brain eval pack v{info['pack_version']} — "
            f"{'valid' if pack['ok'] else 'INVALID'}",
            f"cases: {pack['total']} "
            f"({', '.join(f'{s} {n}' for s, n in pack['counts'].items())})",
        ]
        for problem in pack["problems"]:
            lines.append(f"  ⚠ {problem}")
        lines.append(f"store: {info['store_root']} · "
                     f"{info['runs_journaled']} run(s) journaled")
        last = info["last_run"]
        if last:
            agg = last["aggregate"]
            lines.append(
                f"last: {last['run_id']} · {last['rung']} · "
                f"{last.get('provider') or '?'}/{last.get('model') or '?'} · "
                f"{last['status']} · overall "
                f"{'—' if agg['overall'] is None else format(agg['overall'] * 100, '.1f') + '%'}")
        lines.append(f"rungs: {', '.join(info['rungs'])} "
                     f"(groq = recorded baseline rung; local = llama.cpp "
                     f"server rung; mock = smoke only, never a baseline)")
        return "\n".join(lines)
