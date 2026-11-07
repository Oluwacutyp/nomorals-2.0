"""Suite runner: builds each case's messages, calls the model through
Devon's own :class:`~nomorals.llm.router.LLMRouter`, scores the reply.

Message construction per suite (the recorded pack's design, 2026-10-22):
  capability / refusal — a single user turn with the case prompt. No
    system prompt: the Freedom axis must measure the model's own
    disposition, and capability cases carry their instructions inline.
  tools — Devon's production system prompt
    (:data:`nomorals.agents.partner.tool_loop._SYSTEM_TEMPLATE`,
    persona_extra empty) with the case's tool listing rendered by the
    LIVE registry exactly as the loop renders it, followed by the
    case's turns (user asks, prior assistant tool blocks, and
    observation feedback turns in Devon's exact format).

The contract is read live, never copied: the run record carries the
template's sha256 so a template change is visible when comparing runs.
A registry that has lost a case tool fails the run loudly at contract
build time (same verdict as :func:`.cases.validate_pack`).

Failure honesty: a provider error scores the case 0 with the error as
its detail, and after ``abort_after_consecutive_errors`` consecutive
provider errors the run aborts — remaining cases are journaled as
not-run rather than hammering a dead provider 110 times. A run whose
provider never answered is marked failed by the engine, never dressed
up as a score.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from ...core.logging_setup import get_logger
from ...llm.base import Message, SamplingParams
from . import checkers
from .cases import load_listings, patterns_path
from .refusal import RefusalScorer
from .toolparse import score_tool_case

_log = get_logger(__name__)

DEFAULT_MAX_TOKENS = 640
ABORT_AFTER_CONSECUTIVE_ERRORS = 5


class ContractError(RuntimeError):
    """The live tool contract cannot serve the pack (drift)."""


def build_tool_contract() -> dict[str, Any]:
    """Read the LIVE tool contract: production system template + the
    registry-rendered listing lines + real parameter names.

    Returns {template, template_sha256, listing_lines: {tool: line},
    tool_params: {tool: {param names}}}.
    """
    from ...tools.registry import ToolRegistry
    from ..partner.tool_loop import _SYSTEM_TEMPLATE

    registry = ToolRegistry().register_builtins()
    listing_lines: dict[str, str] = {}
    for line in registry.prompt_listing().splitlines():
        if line.startswith("- ") and "(" in line:
            name = line[2:line.index("(")].strip()
            listing_lines[name] = line
    tool_params: dict[str, set[str]] = {}
    for schema in registry.schemas():
        props = (schema.get("parameters") or {}).get("properties") or {}
        tool_params[schema["name"]] = set(props.keys())
    return {
        "template": _SYSTEM_TEMPLATE,
        "template_sha256": hashlib.sha256(
            _SYSTEM_TEMPLATE.encode("utf-8")).hexdigest(),
        "listing_lines": listing_lines,
        "tool_params": tool_params,
    }


class EvalRunner:
    """Runs pack cases against one router. One instance per run."""

    def __init__(self, router: Any, *, temperature: float = 0.0,
                 abort_after_consecutive_errors: int = ABORT_AFTER_CONSECUTIVE_ERRORS):
        self.router = router
        self.temperature = temperature
        self.abort_after = abort_after_consecutive_errors
        self.refusal_scorer = RefusalScorer(patterns_path())
        self.listings = load_listings()
        self._contract: dict[str, Any] | None = None

    # ------------------------------------------------------------ contract
    def contract(self) -> dict[str, Any]:
        if self._contract is None:
            self._contract = build_tool_contract()
        return self._contract

    def contract_info(self) -> dict[str, Any]:
        contract = self.contract()
        return {
            "template_sha256": contract["template_sha256"],
            "listing_source": "live-registry prompt_listing",
            "parser": "nomorals.agents.partner.tool_loop.parse_tool_calls",
        }

    # ------------------------------------------------------------ messages
    def messages_for(self, case: dict[str, Any]) -> list[Message]:
        suite = case["suite"]
        if suite in ("capability", "refusal"):
            return [Message(role="user", content=case["prompt"])]
        contract = self.contract()
        key = case.get("listing", "core")
        names = self.listings[key]
        missing = [n for n in names if n not in contract["listing_lines"]]
        if missing:
            raise ContractError(
                f"case {case['id']}: tools {missing} (listing {key!r}) are "
                f"not in the live registry — the pack measures a contract "
                f"Devon no longer speaks")
        listing = "\n".join(contract["listing_lines"][n] for n in names)
        system = contract["template"].format(
            n_tools=len(names), tool_listing=listing, persona_extra="")
        messages = [Message(role="system", content=system)]
        for turn in case["turns"]:
            messages.append(Message(role=turn["role"], content=turn["content"]))
        return messages

    # ------------------------------------------------------------ running
    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        messages = self.messages_for(case)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=int(case.get("max_tokens", DEFAULT_MAX_TOKENS)))
        try:
            resp = self.router.chat(messages, params, task_kind="braineval")
        except Exception as exc:  # noqa: BLE001 — a raising router is a provider failure, scored as one
            resp = None
            error = f"router raised {type(exc).__name__}: {exc}"
        else:
            error = resp.error or ""
        row: dict[str, Any] = {
            "id": case["id"],
            "suite": case["suite"],
            "category": case.get("category", ""),
            "provider": getattr(resp, "provider", "") or "",
            "model": getattr(resp, "model", "") or "",
            "latency_ms": round(getattr(resp, "latency_ms", 0.0) or 0.0, 1),
            "degraded": bool(getattr(resp, "degraded", False)),
            "seconds": 0.0,
            "error": "",
            "response": "",
            "score": 0.0,
            "passed": False,
            "detail": "",
            "not_run": False,
        }
        text = (resp.text or "") if resp is not None else ""
        if not text:
            row["error"] = error or "provider returned an empty response"
            row["detail"] = f"provider error: {row['error']}"
        else:
            row["response"] = text
            if error:
                row["detail"] = f"note: router reported {error}; scored the served text"
            if case["suite"] == "capability":
                score, passed, detail = checkers.run_checker(text, case["checker"])
            elif case["suite"] == "refusal":
                score, passed, detail = self.refusal_scorer.classify(text)
            else:
                contract = self.contract()
                score, passed, detail = score_tool_case(
                    text, case["expect"], contract["tool_params"])
            row["score"] = round(score, 4)
            row["passed"] = passed
            row["detail"] = (row["detail"] + "; " if row["detail"] else "") + detail
        row["seconds"] = round(time.time() - started, 2)
        return row

    def run(self, cases: list[dict[str, Any]],
            progress: Callable[[int, int, dict[str, Any]], None] | None = None
            ) -> tuple[list[dict[str, Any]], bool]:
        """Run every case in order. Returns (rows, aborted): on abort the
        remaining cases are appended as not-run rows so the journal
        always accounts for the whole attempted pack."""
        rows: list[dict[str, Any]] = []
        consecutive_errors = 0
        aborted = False
        for i, case in enumerate(cases, 1):
            if aborted:
                rows.append({
                    "id": case["id"], "suite": case["suite"],
                    "category": case.get("category", ""),
                    "provider": "", "model": "", "latency_ms": 0.0,
                    "degraded": False, "seconds": 0.0,
                    "error": "not run — run aborted after consecutive "
                             "provider errors",
                    "response": "", "score": 0.0, "passed": False,
                    "detail": "not run", "not_run": True,
                })
                continue
            row = self.run_case(case)
            rows.append(row)
            if row["error"]:
                consecutive_errors += 1
                if consecutive_errors >= self.abort_after:
                    aborted = True
                    _log.warning(
                        "braineval: aborting run after %d consecutive "
                        "provider errors (last: %s)",
                        consecutive_errors, row["error"][:160])
            else:
                consecutive_errors = 0
            if progress:
                progress(i, len(cases), row)
        return rows, aborted
