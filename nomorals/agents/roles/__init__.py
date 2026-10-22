"""Specialized agents.

Each role is small on purpose: the intelligence lives in the tools and the model,
the role declares *what it is allowed to do* and *how it turns a task into tool
calls*. Capability narrowing is the interesting part — a critic agent genuinely
cannot write to disk, which is what makes its verdict trustworthy.

One exception, by design: the ``coding`` role is a bridge, not an engine.
:class:`MissionCodingAgent` drives the real
:class:`nomorals.agents.coding.CodingAgent` loop (plan → draft → verify →
repair → gates), so a mission coding step gets exactly the same machinery
as every other coding path. :class:`SnippetWriterAgent` is the retired
single-shot writer, kept for direct lightweight use and deliberately
unregistered — see pass #61 (W3).
"""

from __future__ import annotations

from typing import Any

from ...llm.brain import brain_for
from ..base import Agent
from ...core.tasks import Task
from ...core.policy import Capability
from ...search.adaptive import adaptive_result_limit
# Prompt 02: the role specs live in ..role_specs (this name is a package, so
# the spec module could not be called roles.py); re-exported here so the
# whole swarm API is importable from nomorals.agents.roles.
from ..role_specs import (
    RoleRegistry,
    RoleSpec,
    SwarmAgent,
    check_spec_call,
    default_registry,
    record_tool_denial,
)


class RoleAgent(Agent):
    """Base for tool-driven roles.

    A Prompt-02 :class:`RoleSpec` may be bound via ``role_spec=`` (or
    :func:`build_agent(..., role_spec=...)`).  When bound, :meth:`_call_tool`
    enforces the spec's allowlist, read-only flag, and path guards *before*
    the registry call — the same rule the swarm layer applies, so the
    existing agents are held to the locked contract on every path.
    Unbound agents behave exactly as before.
    """

    def __init__(self, *, role_spec: RoleSpec | None = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.role_spec = role_spec

    def _tools(self) -> Any:
        return getattr(self.context, "tools", None) if self.context else None

    def _call_tool(self, name: str, **kwargs: Any) -> Any:
        tools = self._tools()
        if tools is None:
            raise RuntimeError(f"agent {self.name} has no tool registry")
        spec = getattr(self, "role_spec", None)
        if spec is not None:
            get_spec = getattr(tools, "get", None)
            tool_spec = get_spec(name) if callable(get_spec) else None
            denied = check_spec_call(
                spec, name, kwargs,
                getattr(tool_spec, "capability", "") or "")
            if denied is not None:
                record_tool_denial(self.context, spec.name, name,
                                   denied.reason, str(denied))
                raise denied
        # Idempotency: when a runner has scoped this unit of work (a
        # mission step, a plan step, one inbound chat message — see
        # nomorals.core.idempotency_keys), derive the registry key from
        # that scope so a retry of the same step replays this call's
        # recorded result instead of re-applying its effect. No scope →
        # unkeyed, exactly as before. An explicit caller-supplied key
        # always wins.
        from ...core.idempotency_keys import scoped_idempotency_key

        call_kwargs = dict(kwargs)
        explicit = call_kwargs.pop("idempotency_key", None)
        key = explicit or scoped_idempotency_key(name, call_kwargs)
        if key:
            call_kwargs["idempotency_key"] = key
        outcome = tools.call(name, actor=self.name,
                             capabilities=self.capabilities, **call_kwargs)
        if not outcome.ok:
            # This message is what mission reports (and chat) show —
            # strip terminal escapes subprocess tools leak into errors.
            from ...core.text import strip_ansi
            raise RuntimeError(
                f"tool {name} failed: "
                f"{strip_ansi(str(outcome.error))[:500]}")
        return outcome.value

    def _task_goal(self, task_input: Any) -> str:
        if isinstance(task_input, Task):
            return str(task_input.payload.get("goal") or task_input.name)
        if isinstance(task_input, dict):
            return str(task_input.get("goal") or task_input.get("query") or "")
        return str(task_input or "")


class ResearchAgent(RoleAgent):
    role = "research"
    required_capabilities = (
        Capability.NET_OUT, Capability.NET_BROWSER, Capability.NET_DOWNLOAD,
        Capability.FS_READ, Capability.FS_WRITE, Capability.MEM_READ,
        Capability.MEM_WRITE, Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def _fetch_failures(self) -> dict[str, str]:
        """Run-scoped ledger of URLs whose fetch already failed.

        Keyed by the current idempotency scope (one mission step / one
        unit of work) inside ``context.extras`` so a step RETRY sees
        the same ledger the failed attempt wrote: a dead URL is never
        fetched identically twice in a run — it is skipped, or retried
        only through a different route.  Falls back to an instance
        dict when the context carries no extras mapping.
        """
        extras = getattr(self.context, "extras", None) \
            if self.context is not None else None
        if isinstance(extras, dict):
            from ...core.idempotency_keys import current_idempotency_scope
            scope = current_idempotency_scope() or "unscoped"
            ledgers = extras.setdefault("research_fetch_failures", {})
            if isinstance(ledgers, dict):
                ledger = ledgers.setdefault(scope, {})
                if isinstance(ledger, dict):
                    return ledger
        local = getattr(self, "_fetch_failure_local", None)
        if local is None:
            local = self._fetch_failure_local = {}
        return local

    def _fetch_via_browser(self, url: str) -> dict[str, Any] | None:
        """Alternate route for a page ``web_fetch`` could not read:
        the rendered browser session.  Returns a page dict or None.
        Never raises — this is a fallback, not a second way to die."""
        try:
            self._call_tool("browser", action="open", url=url,
                            session="research")
            out = self._call_tool("browser", action="text",
                                  session="research", max_chars=6000)
        except Exception:  # noqa: BLE001 - fallback failure = skip
            return None
        text = ""
        if isinstance(out, dict):
            text = str(out.get("text") or "")
        elif isinstance(out, str):
            text = out
        if not text.strip():
            return None
        return {"url": url, "title": "", "text": text[:4000]}

    def work(self, task_input: Any) -> Any:
        goal = self._task_goal(task_input)
        self.check_cancelled()
        # adaptive breadth: the goal's own phrasing sets the result count
        search = self._call_tool("web_search", query=goal,
                                 max_results=adaptive_result_limit(goal))
        from ...core.http import unwrap_redirect_url
        from ...core.text import strip_ansi
        failures = self._fetch_failures()
        pages: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()
        for hit in (search.get("results") or []):
            if len(pages) >= 4:
                break
            self.check_cancelled()
            raw_url = hit.get("url")
            if not raw_url:
                continue
            # Search results can carry click-tracking URLs — the real
            # destination is what gets fetched (and ledgered).
            url = unwrap_redirect_url(str(raw_url))
            if not url or url in seen:
                continue
            seen.add(url)
            if url in failures:
                # Already failed in this run (e.g. a step retry): never
                # fetch it identically again — record the skip and move
                # on to sources that still have a chance.
                skipped.append({"url": url,
                                "reason": failures[url]})
                continue
            try:
                fetched = self._call_tool("web_fetch", url=url)
            except Exception as exc:  # noqa: BLE001 - one bad source
                # must never kill the step: robots vetoes, dead hosts
                # and bot walls are per-source facts, not step facts.
                reason = strip_ansi(str(exc))[:300]
                page = self._fetch_via_browser(url)
                if page is not None:
                    page["title"] = str(hit.get("title") or "")
                    pages.append(page)
                    continue
                failures[url] = reason
                skipped.append({"url": url, "reason": reason})
                continue
            if isinstance(fetched, dict) and fetched.get("text"):
                pages.append({"url": url,
                              "title": fetched.get("title", ""),
                              "text": fetched["text"][:4000]})
            else:
                reason = "fetch returned no text"
                failures[url] = reason
                skipped.append({"url": url, "reason": reason})
        if not pages:
            # A research step that retrieved nothing has not done its
            # job — report failure (the mission runner settles a
            # top-level "error" as a failed step), never a hollow
            # success with an empty summary.  The skips ride along so
            # the report names WHY each source was unusable.
            detail = ""
            if skipped:
                detail = "; skipped: " + "; ".join(
                    f"{s['url']} — {s['reason']}" for s in skipped[:4])
            return {"goal": goal, "sources": [], "skipped": skipped,
                    "error": f"no sources retrieved for research step: "
                             f"{goal}{detail}"}
        summary = self._summarize(goal, pages)
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is not None:
            for page in pages:
                memory.remember(
                    f"Source for {goal[:60]!r}: {page['title'] or page['url']}",
                    kind="episode", importance=0.4, source="research",
                    metadata={"url": page["url"]},
                )
        result: dict[str, Any] = {
            "goal": goal,
            "sources": [{"url": p["url"], "title": p["title"]}
                        for p in pages],
            "summary": summary,
        }
        if skipped:
            # Journaled, not fatal: the step succeeded on the sources
            # that answered; these were skipped and why.
            result["skipped"] = skipped
        return result

    def _summarize(self, goal: str, pages: list[dict[str, Any]]) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            return "\n".join(f"- {p['title'] or p['url']}" for p in pages)
        from ...llm.base import Message, SamplingParams

        corpus = "\n\n".join(f"[{p['url']}]\n{p['text'][:2500]}" for p in pages)
        response = brain_for(self.context).chat(
            [
                Message.system("You synthesize research. Cite the source URL inline."),
                Message.user(f"Question: {goal}\n\nSources:\n{corpus}"),
            ],
            SamplingParams(temperature=0.3, max_tokens=1500),
        task_kind="chat")
        if not response.ok:
            return "\n".join(f"- {p['url']}" for p in pages)
        self.budget.charge_tokens(response.usage.total_tokens)
        return response.text


#: Mission-context budget for the real coding loop: the chat caller's
#: iteration budget (5) at the chat NL caller's timeout (300s). Mission
#: steps are long-running autonomous units of work, not chat turns, so
#: they deliberately get the generous end of the existing callers'
#: ranges (chat /code: 5×120s, projects: 3×60s). A step's payload may
#: override either via "max_iterations" / "timeout". Token spend has
#: NO mission default (pass #65): a payload "max_tokens" opts the
#: step into the loop's spend cap; absent, the step runs uncapped.
_MISSION_CODING_MAX_ITERATIONS = 5
_MISSION_CODING_TIMEOUT_S = 300.0


class MissionCodingAgent(RoleAgent):
    """The ``coding`` role: a bridge to the real CodingAgent loop.

    Mission (and swarm-delegated) coding steps execute through
    :class:`nomorals.agents.coding.CodingAgent` — plan → draft →
    verify → repair → lint/review gates — never through a single-shot
    generator. This class only translates: the step's goal/spec
    becomes the loop's task, a step-carried ``accept`` command becomes
    the run's accept check, and the loop's ``CodingResult`` becomes
    the role-result dict the mission runner settles:

    - loop green → an evidence dict (files, the loop's own per-step
      status from pass #60, verification mode, review verdict) with NO
      top-level ``error``;
    - loop red → the same evidence plus a top-level ``error`` naming
      the loop's reason (and its failed step, when it has one) — the
      runner settles that shape as a FAILED step. A coding step is
      done only when the real loop says it is done.

    ``plan_mode`` is forced off: a mission step is pre-authorized work
    executing autonomously — a plan-approval pause would strand the
    step with nothing done and nobody able to approve it. The step's
    payload reaches the bridge via :meth:`bind_step`, which the
    mission runner calls when the agent it built provides it.
    """

    role = "coding"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_CODE,
        Capability.EXEC_SHELL, Capability.MEM_READ, Capability.MEM_WRITE,
        Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._step_payload: dict[str, Any] = {}

    def bind_step(self, step: Any) -> None:
        """Receive the mission step being executed (runner hook).

        The mission runner calls this when the built agent provides
        it, so the step's payload — its ``accept`` command and budget
        overrides — reaches the loop. Accepts a PlanStep, a persisted
        plan-entry dict, or anything carrying a ``payload`` mapping.
        """
        payload = getattr(step, "payload", None)
        if payload is None and isinstance(step, dict):
            payload = step.get("payload") or step.get("policy")
        self._step_payload = dict(payload) if isinstance(payload, dict) else {}

    def work(self, task_input: Any) -> Any:
        goal, spec = self._goal_and_spec(task_input)
        # The bound step's payload is the base spec; what the caller
        # passed explicitly for THIS run wins over it.
        merged = {**self._step_payload, **spec}
        if not goal:
            goal = str(merged.get("goal") or merged.get("task") or "")
        if not goal.strip():
            return {"error": "coding step has no goal/task to execute"}
        self.check_cancelled()
        accept = str(merged.get("accept") or merged.get("accept_command") or "")
        root = merged.get("root") or None
        try:
            max_iterations = int(merged.get("max_iterations")
                                 or _MISSION_CODING_MAX_ITERATIONS)
            timeout = float(merged.get("timeout")
                            or _MISSION_CODING_TIMEOUT_S)
        except (TypeError, ValueError):
            return {"goal": goal,
                    "error": "coding step carries a non-numeric "
                             "max_iterations/timeout budget"}
        # Pass #65 (W7): a step may bound its own token spend with a
        # payload "max_tokens" — forwarded to the loop's optional cap.
        # There is deliberately NO mission default: absent means the
        # step runs uncapped (metered, like every run).
        raw_cap = merged.get("max_tokens")
        if raw_cap is None:
            max_tokens = None
        else:
            try:
                max_tokens = int(raw_cap)
            except (TypeError, ValueError):
                return {"goal": goal,
                        "error": "coding step carries a non-numeric "
                                 "max_tokens budget"}
        from ..coding import CodingAgent as _LoopCodingAgent

        # Pass #66: forward the role's scope into the loop.  The
        # carrier is the bound RoleSpec — the attribute the swarm
        # delegation sets on this agent — and the scoped view is the
        # same RoleEnforcingRegistry the swarm layer builds over
        # context.tools, so the loop's governed surfaces (explore
        # reads, edits, verify, lint, accept) answer to the step's
        # allowlist instead of the full registry.  Unbound (a plain
        # mission step): no spec exists to forward — the loop is
        # constructed exactly as before, with no tools kwarg at all.
        ctor_kwargs: dict[str, Any] = {}
        spec = getattr(self, "role_spec", None)
        inner_tools = (getattr(self.context, "tools", None)
                       if self.context is not None else None)
        if spec is not None and inner_tools is not None:
            from ..role_specs import RoleEnforcingRegistry

            ctor_kwargs["tools"] = RoleEnforcingRegistry(
                inner_tools, spec, self.context)
        if root:
            ctor_kwargs["root"] = str(root)
        try:
            loop = _LoopCodingAgent(self.context, **ctor_kwargs)
            result = loop.run(
                goal,
                accept=accept,
                max_iterations=max_iterations,
                timeout=timeout,
                plan_mode=False,
                max_tokens=max_tokens,
                # Pass #66 (W10): mission steps are the long-running
                # autonomous surface — verify runs in the background
                # (lint concurrent, cap derived from this step's
                # timeout by the loop) instead of blocking on it.
                background_tests=True,
            )
        except Exception as exc:  # noqa: BLE001 - a step failure is a result
            return {"goal": goal, "engine": "coding_agent",
                    "error": f"coding loop could not run: "
                             f"{type(exc).__name__}: {exc}"}
        return self._settle(goal, result)

    @staticmethod
    def _goal_and_spec(task_input: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(task_input, Task):
            spec = dict(task_input.payload or {})
            return (str(spec.get("goal") or spec.get("task")
                        or task_input.name or ""), spec)
        if isinstance(task_input, dict):
            spec = dict(task_input)
            return (str(spec.get("goal") or spec.get("task") or ""), spec)
        return (str(task_input or ""), {})

    @staticmethod
    def _settle(goal: str, result: Any) -> dict[str, Any]:
        steps = list(getattr(result, "steps", None) or [])
        out: dict[str, Any] = {
            "goal": goal,
            "engine": "coding_agent",
            "ok": bool(result.ok),
            "files": list(result.files or []),
            "iterations": int(result.iterations or 0),
            "verification": str(result.verification or ""),
            "steps": steps,
            "replans": int(result.replans or 0),
            "review": dict(result.review or {}),
            "output": str(result.output or "")[-2000:],
            "seconds": round(float(result.seconds or 0.0), 2),
            # Pass #65 (W7): the run's metered token usage rides the
            # settlement as evidence, like the review verdict does.
            "usage": dict(getattr(result, "usage", None) or {}),
            # Pass #66: scope refusals the loop recorded ride along
            # too — a step settled by its own allowlist says so here.
            "scope_denials": list(
                getattr(result, "scope_denials", None) or []),
        }
        # Pass #69: the loop's step account, pre-rendered by the one
        # shared renderer — the owner's results view
        # (missions.progress.payload_text) leads with it, so a coding
        # step's per-step outcome is visible in /mission results
        # instead of only the loop's raw output tail.  Absent when
        # the run carried no steps (renderer returns "").
        from ..coding import render_step_account

        account = render_step_account(result)
        if account:
            out["step_account"] = account
        if result.ok:
            return out
        reason = str(result.error or "").strip()
        if not reason and getattr(result, "needs_approval", False):
            reason = ("coding run paused for plan approval, which a "
                      "mission step cannot answer")
        if not reason:
            failed = [s for s in steps
                      if isinstance(s, dict) and s.get("status") == "failed"]
            if failed:
                step = failed[0]
                reason = (f"coding step {step.get('id', '?')} "
                          f"({step.get('path', '?')}) failed: "
                          f"{step.get('reason') or 'no reason recorded'}")
            else:
                reason = "coding run finished without verifying green"
        out["error"] = reason
        return out


class SnippetWriterAgent(RoleAgent):
    """Lightweight single-shot snippet writer — NOT the CodingAgent.

    Generates ONE file from a goal and, by default, runs it. No plan,
    no tests, no lint, no review, no rollback: that is exactly why it
    exists (a quick throwaway snippet) and exactly why it must never
    be the engine behind a mission or swarm coding step — those route
    to :class:`MissionCodingAgent`, which drives the real loop. This
    class kept the old shadow's behavior verbatim when it was demoted
    (pass #61, W3); it is deliberately absent from :data:`ROLES`, so
    no registry, route, or log line can mistake it for the CodingAgent
    again.
    """

    role = "snippet"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_CODE,
        Capability.EXEC_SHELL, Capability.MEM_READ, Capability.MEM_WRITE,
        Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        goal = self._task_goal(task_input)
        self.check_cancelled()
        spec = task_input.payload if isinstance(task_input, Task) else {}
        language = spec.get("language", "python")
        code = self._generate(goal, language)
        if not code.strip():
            # No model output (or no model at all): writing and
            # "running" an empty file would settle as a hollow success.
            return {"goal": goal, "language": language,
                    "error": f"code generation produced no code for: {goal}"}
        if not spec.get("execute", True):
            return {"goal": goal, "language": language, "code": code, "executed": False}

        path = spec.get("path") or f"generated/{_slug(goal)}.{_extension(language)}"
        self._call_tool("fs_write", path=path, content=code)
        executed = self._call_tool("shell_run", command=f"python {path}", timeout=60)
        return {
            "goal": goal, "language": language, "path": path, "code": code,
            "executed": True, "exit_code": executed.get("exit_code"),
            "stdout": (executed.get("stdout") or "")[:2000],
            "stderr": (executed.get("stderr") or "")[:1000],
        }

    def _generate(self, goal: str, language: str) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            # No model: no code. work() turns the empty result into an
            # honest failure instead of shipping a placeholder comment
            # as if it were the deliverable.
            return ""
        from ...llm.base import Message, SamplingParams

        response = brain_for(self.context).chat(
            [
                Message.system(f"Write complete, runnable {language}. No placeholders. No commentary outside the code."),
                Message.user(goal),
            ],
            SamplingParams(temperature=0.2, max_tokens=4096),
        task_kind="chat")
        self.budget.charge_tokens(response.usage.total_tokens)
        return _strip_code_fence(response.text) if response.ok else ""


class VisionAgent(RoleAgent):
    role = "vision"
    required_capabilities = (Capability.FS_READ, Capability.NET_DOWNLOAD, Capability.MODEL_CALL, Capability.MEM_WRITE)

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {"path": task_input}
        if not spec.get("path") and not spec.get("url"):
            return {"error": "vision step has no image 'path' or 'url' to describe"}
        self.check_cancelled()
        return self._call_tool(
            "vision_describe",
            path=spec.get("path", ""),
            url=spec.get("url", ""),
            prompt=spec.get("prompt", ""),
        )


class DataCollectionAgent(RoleAgent):
    role = "data_collection"
    required_capabilities = (
        Capability.NET_OUT, Capability.NET_DOWNLOAD, Capability.FS_READ,
        Capability.FS_WRITE, Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        goal = self._task_goal(task_input)
        self.check_cancelled()
        return self._call_tool(
            "data_harvest",
            topic=goal,
            prompts=spec.get("prompts") or [goal],
            limit=int(spec.get("limit", 10)),
        )


class TrainingAgent(RoleAgent):
    role = "training"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_SHELL,
        Capability.TRAIN_RUN, Capability.MODEL_DOWNLOAD, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        if not spec.get("dataset"):
            return {"error": "training step has no 'dataset' to train on"}
        self.check_cancelled()
        return self._call_tool(
            "training_run",
            dataset=spec.get("dataset", ""),
            base_model=spec.get("base_model", ""),
            backend=spec.get("backend", "native"),
            epochs=int(spec.get("epochs", 1)),
        )


class SocialAgent(RoleAgent):
    role = "social"
    required_capabilities = (
        Capability.SOCIAL_POST, Capability.SOCIAL_READ, Capability.NET_OUT,
        Capability.FS_READ, Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        missing = [field for field in ("platform", "content")
                   if not spec.get(field)]
        if missing:
            return {"error": "social step is missing required input: "
                             + ", ".join(missing)}
        self.check_cancelled()
        return self._call_tool(
            "social_post",
            platform=spec.get("platform", ""),
            content=spec.get("content", ""),
            scheduled_at=spec.get("scheduled_at"),
        )


class ExecutionAgent(RoleAgent):
    role = "execution"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_CODE,
        Capability.EXEC_SHELL, Capability.NET_OUT, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        command = spec.get("command")
        if command:
            self.check_cancelled()
            return self._call_tool("shell_run", command=command, timeout=float(spec.get("timeout", 60)))
        # No command means this step cannot do anything — an execution
        # agent runs shell commands, nothing else. Report the failure;
        # returning a success-shaped echo here is what let hollow
        # missions "complete" without executing a single step.
        return {"goal": self._task_goal(task_input),
                "error": "no action specified: execution step received "
                         "no 'command' to run"}


class MemoryAgent(RoleAgent):
    role = "memory"
    required_capabilities = (
        Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_WRITE, Capability.MODEL_CALL,
    )

    def work(self, task_input: Any) -> Any:
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is None:
            return {"error": "no memory subsystem"}
        spec = task_input.payload if isinstance(task_input, Task) else {}
        action = spec.get("action", "consolidate")
        self.check_cancelled()
        if action == "recall":
            return {"results": memory.recall(spec.get("query", ""), limit=int(spec.get("limit", 8)))}
        if action == "remember":
            return {"id": memory.remember(spec.get("content", ""), kind=spec.get("kind", "episode"))}
        # Additive: the destructive consolidate() deletes memories, so it
        # is explicit/human-invoked only — a role task never deletes.
        return memory.consolidate_additive()


class CriticAgent(RoleAgent):
    """Verifies other agents' work. Read-only on purpose.

    A critic that could edit files would just fix what it was supposed to be
    judging, and the verdict would be worthless.
    """

    role = "critic"
    required_capabilities = (Capability.MEM_READ, Capability.DB_READ, Capability.FS_READ, Capability.MODEL_CALL)

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {"output": task_input}
        goal = spec.get("goal", "")
        output = spec.get("output", "")
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            # A verdict that was never produced must not be fabricated:
            # a made-up 0.5 score would read as a real critique.
            return {"error": "no model available to critique"}
        from ...llm.base import Message, SamplingParams

        response = brain_for(self.context).chat(
            [
                Message.system(
                    "You are a strict reviewer. List concrete defects. "
                    'Reply as JSON: {"score": 0.0-1.0, "issues": ["..."], "verdict": "..."}'
                ),
                Message.user(f"Goal: {goal}\n\nDeliverable:\n{str(output)[:6000]}"),
            ],
            SamplingParams(temperature=0.1, max_tokens=1024, json_mode=True),
        task_kind="chat")
        self.budget.charge_tokens(response.usage.total_tokens)
        if not response.ok:
            return {"error": "critique unavailable: the model call did not succeed"}
        from .orchestrator_helpers import parse_json_loose

        data = parse_json_loose(response.text) or {}
        return {
            "score": float(data.get("score", 0.5)),
            "issues": list(data.get("issues") or [])[:8],
            "verdict": str(data.get("verdict", "")),
        }


class ReflectionAgent(RoleAgent):
    role = "reflection"
    required_capabilities = (Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_READ, Capability.MODEL_CALL)

    def work(self, task_input: Any) -> Any:
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is None:
            return {"error": "no memory subsystem"}
        self.check_cancelled()
        # Additive: reflection distils lessons without deleting the
        # episodes they came from — "forget only on explicit command".
        return memory.consolidate_additive()


#: role name → class, for dynamic spawning.
ROLES: dict[str, type[Agent]] = {
    cls.role: cls
    for cls in (
        ResearchAgent, MissionCodingAgent, VisionAgent, DataCollectionAgent, TrainingAgent,
        SocialAgent, ExecutionAgent, MemoryAgent, CriticAgent, ReflectionAgent,
    )
}


def build_agent(role: str, **kwargs: Any) -> Agent:
    """Instantiate a role by name."""
    try:
        cls = ROLES[role]
    except KeyError:
        return ExecutionAgent(**kwargs)
    return cls(**kwargs)


def _slug(text: str, limit: int = 32) -> str:
    from ...core.text import slugify

    # canonical: nomorals.core.text.slugify (old impl crashed on None;
    # the shared one returns the fallback instead)
    return slugify(text, limit=limit, fallback="task")


def _extension(language: str) -> str:
    return {
        "python": "py", "javascript": "js", "typescript": "ts", "bash": "sh",
        "shell": "sh", "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
    }.get(language.lower(), "txt")


def _strip_code_fence(text: str) -> str:
    import re

    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n```$", text, re.DOTALL)
    return match.group(1) if match else text


__all__ = [
    "ROLES",
    "CriticAgent",
    "DataCollectionAgent",
    "ExecutionAgent",
    "MemoryAgent",
    "MissionCodingAgent",
    "ReflectionAgent",
    "ResearchAgent",
    "RoleAgent",
    "SnippetWriterAgent",
    "SocialAgent",
    "TrainingAgent",
    "VisionAgent",
    "build_agent",
]
