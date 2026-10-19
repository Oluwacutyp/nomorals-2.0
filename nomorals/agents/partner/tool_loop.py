"""Brain-driven tool-calling loop — the spine of the rebuild.

The problem: the brain generated text but could never *use* the system's
capabilities. 282 tools sat behind a registry the LLM couldn't reach.
Everything was command-triggered, not capability-driven.

The fix: a genuine ReAct-style loop. The LLM sees the full tool registry
(names, descriptions, schemas), selects tools from plain language,
executes them, reflects on results, and iterates until done.

Design notes (mined gold):
- ReAct (Thought/Action/Observation): text-based, works on EVERY provider.
  No provider-specific function-calling API — Groq, HF, local GGUF all
  speak text. This is deliberate, not a limitation.
- OpenAI function calling: structured JSON tool calls, iterate until done.
- Anthropic tool use: the model decides when it's finished; results feed
  back as conversation turns.
- The trash-build gold: even the simplest agent loops get one thing right
  — a tight parse/execute/feedback cycle with no magic. We keep it tight.

NO regex/keyword intent routing anywhere in this module. The LLM reads
plain language and picks tools. "It either stays dynamic or should not
exist."

Gating is enforced at the EXECUTION layer (registry.call's capability
enforcement), not by prompt instructions. A restricted actor sees a
filtered tool list AND gets denied at call time if they reach for more.
"""

from __future__ import annotations

import contextvars
import difflib
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

#: Max characters of a tool result shown to the model in one observation.
_OBSERVATION_LIMIT = 4000
#: Bounded remainder store: full results behind truncated observations.
#: Keyed ``rem-<hex>``; oldest entries evicted past the cap so a long run
#: cannot grow memory without bound.
_REMAINDER_LIMIT = 64
_REMAINDER_STORE: dict[str, str] = {}
_REMAINDER_LOCK = threading.Lock()

#: A timeout never means "the tool stopped" — the tool may still finish
#: and change the world. Observations and results carry this marker so no
#: caller blind-replays a timed-out call.
OUTCOME_UNKNOWN_MARKER = "outcome_unknown"


def store_remainder(full_text: str) -> str:
    """Retain a full result behind a truncated observation; returns its key."""
    key = f"rem-{uuid.uuid4().hex[:16]}"
    with _REMAINDER_LOCK:
        _REMAINDER_STORE[key] = full_text
        while len(_REMAINDER_STORE) > _REMAINDER_LIMIT:
            _REMAINDER_STORE.pop(next(iter(_REMAINDER_STORE)))
    return key


def retrieve_remainder(key: str) -> str | None:
    """Fetch a full result previously stored by :func:`store_remainder`."""
    with _REMAINDER_LOCK:
        return _REMAINDER_STORE.get(key)

#: The model emits tool calls as fenced JSON blocks. One or more per turn.
_TOOL_FENCE_RE = re.compile(
    r"```tool\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
#: Fallback: bare JSON object with a "tool" key. Matched with a
#: balanced-brace scanner (not a regex) so nested args survive.
_BARE_TOOL_START_RE = re.compile(
    r"\{\s*\"tool\"\s*:\s*\"[^\"]+\"\s*,", re.DOTALL)


@dataclass
class ToolCall:
    """One parsed tool invocation from the model."""
    name: str
    args: dict[str, Any]
    raw: str = ""


@dataclass
class ToolLoopTrace:
    """One loop iteration, for audit and debugging."""
    iteration: int
    model_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ToolLoopResult:
    """What the loop produced."""
    answer: str
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    trace: list[ToolLoopTrace] = field(default_factory=list)
    #: True when the loop hit a budget or a failure and the answer is partial.
    degraded: bool = False
    degrade_note: str = ""
    model: str = ""
    ok: bool = True
    error: str = ""
    #: Correlation ids joining this turn to its request (Batch 2).
    request_id: str = ""
    run_id: str = ""


@dataclass
class ToolLoopEvent:
    """One lifecycle event, for live UIs and telemetry.

    ``kind`` is one of: ``turn_start``, ``turn_end``, ``tool_start``,
    ``tool_end``, ``stalled``, ``final``. Mirrors OpenAI Agents SDK
    lifecycle hooks / run_streamed(): the loop is observable while it
    runs, not just when it finishes.
    """
    kind: str
    iteration: int = 0
    tool_name: str = ""
    detail: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "iteration": self.iteration,
                "tool": self.tool_name, "detail": self.detail, "at": self.at}


#: How many times the same (tool, args) may repeat back-to-back before the
#: loop calls it a stall and degrades honestly instead of burning budget.
_STALL_REPEATS = 3


def _call_signature(call: "ToolCall") -> str:
    """Stable hash of (tool name, canonical args) for stall detection."""
    try:
        canon = json.dumps(call.args, sort_keys=True, default=str)
    except Exception:
        canon = str(call.args)
    return hashlib.sha256(f"{call.name}\n{canon}".encode()).hexdigest()[:16]


def _scan_balanced(text: str, start: int) -> int:
    """Index just past the closing brace matching ``text[start] == '{'``.

    String-aware: braces inside "..." (with \\ escapes) don't count.
    Returns -1 when unbalanced.
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from model text. Never raises.

    Accepts fenced ```tool blocks (preferred) and bare {"tool": ...}
    JSON objects (fallback, balanced-brace scanned so nested args
    survive). Malformed JSON is skipped with the text preserved — the
    model sees the parse failure as feedback.
    """
    calls: list[ToolCall] = []
    seen_spans: list[tuple[int, int]] = []

    def _try_block(raw: str, span: tuple[int, int]) -> None:
        # Skip spans already captured by the fence matcher.
        if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
            return
        raw = raw.strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        name = data.get("tool")
        if not isinstance(name, str) or not name.strip():
            return
        args = data.get("args", {})
        if not isinstance(args, dict):
            return
        calls.append(ToolCall(name=name.strip(), args=args, raw=raw))
        seen_spans.append(span)

    text = text or ""
    for m in _TOOL_FENCE_RE.finditer(text):
        _try_block(m.group(1), m.span())
    for m in _BARE_TOOL_START_RE.finditer(text):
        end = _scan_balanced(text, m.start())
        if end == -1:
            continue
        _try_block(text[m.start():end], (m.start(), end))
    return calls


def _strip_tool_blocks(text: str) -> str:
    """Remove tool-call blocks, leaving the model's prose (final answer)."""
    text = _TOOL_FENCE_RE.sub("", text or "")
    # Only strip bare blocks that parsed as tool calls (balanced scan).
    out: list[str] = []
    pos = 0
    for m in _BARE_TOOL_START_RE.finditer(text):
        end = _scan_balanced(text, m.start())
        if end == -1:
            continue
        raw = text[m.start():end]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("tool"), str):
            out.append(text[pos:m.start()])
            pos = end
    out.append(text[pos:])
    return "".join(out).strip()


_SYSTEM_TEMPLATE = """\
You are Devon, a capable personal AI with real tools at your disposal. You \
understand plain language and use your tools to get things done — you never \
pretend to have done something you haven't, and you never describe a tool \
result you didn't actually get.

You have {n_tools} tools available:

{tool_listing}

How to work:
- Think step by step about what the user wants.
- When you need a tool, emit a fenced block like this:

```tool
{{"tool": "tool_name", "args": {{"param": "value"}}}}
```

- You may emit multiple tool calls in one turn; they run in order.
- After each turn, you receive the tool results as observations. Reflect on \
them, then either call more tools or answer.
- When you are done and have what you need, reply directly to the user in \
your own voice. No tool block means your text IS the final answer.
- If a tool fails, say what failed honestly — never fake success, never \
invent a result.
- Keep args exactly as the schema describes. Unknown tools or bad args \
waste everyone's time.
{persona_extra}\
"""


class ToolCallingLoop:
    """The brain's tool-calling loop. One instance per brain; run() per turn.

    ``llm`` is the Brain (nomorals/llm/brain.py) — anything with a
    ``chat(messages, ...)`` method returning an LLMResponse works.
    ``registry`` is the ToolRegistry.
    """

    def __init__(
        self,
        llm: Any,
        registry: Any,
        *,
        max_iterations: int = 8,
        tool_timeout_s: float = 90.0,
        total_timeout_s: float = 300.0,
        max_tool_calls_per_turn: int = 5,
        on_event: Callable[[ToolLoopEvent], None] | None = None,
        parallel_per_turn: bool = False,
        isolated_tools: Any | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_iterations = max(1, int(max_iterations))
        self.tool_timeout_s = max(1.0, float(tool_timeout_s))
        self.total_timeout_s = max(10.0, float(total_timeout_s))
        self.max_tool_calls_per_turn = max(1, int(max_tool_calls_per_turn))
        #: Lifecycle hook — called with a ToolLoopEvent for turn/tool/final
        #: transitions. Never allowed to break the loop.
        self.on_event = on_event
        #: When True, a turn's tool calls execute concurrently in the pool
        #: (smolagents ToolCallingAgent style) instead of sequentially.
        self.parallel_per_turn = bool(parallel_per_turn)
        self._pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="tool-loop")
        #: Tools that run in an ISOLATED CHILD PROCESS (see
        #: :mod:`nomorals.core.process_isolation`) instead of the
        #: thread pool, so a timeout SIGKILLs them — forcible
        #: termination with a KNOWN outcome (dead), never unknown.
        #: A tool is isolated when its name is here OR its spec
        #: metadata sets ``process_isolated=True``. Thread-based
        #: tools keep cooperative cancellation + unknown-outcome
        #: recording (unchanged).
        self.isolated_tools: frozenset[str] = frozenset(
            isolated_tools or ())
        #: Timed-out calls whose tool may still finish: call signature →
        #: record. :meth:`reconcile_unknown` surfaces late completions so
        #: the loop never blind-replays them.
        self._unknown_outcomes: dict[str, dict[str, Any]] = {}
        self._unknown_lock = threading.Lock()

    def unknown_outcomes(self) -> list[dict[str, Any]]:
        """Timed-out calls still unresolved (outcome unknown)."""
        with self._unknown_lock:
            return [dict(v) for v in self._unknown_outcomes.values()]

    def reconcile_unknown(self) -> list[dict[str, Any]]:
        """Collect late completions of timed-out calls.

        Returns one record per newly finished call (with its result
        text); still-running calls stay in :meth:`unknown_outcomes`.
        """
        done: list[dict[str, Any]] = []
        with self._unknown_lock:
            for sig, rec in list(self._unknown_outcomes.items()):
                fut = rec.get("_future")
                try:
                    finished = bool(fut is not None and fut.done())
                except Exception:  # noqa: BLE001 - a dead future is not done
                    finished = False
                if not finished:
                    continue
                try:
                    rec["result"] = str(fut.result(timeout=0))
                    rec["status"] = "completed_late"
                except Exception as exc:  # noqa: BLE001 - late failure is data
                    rec["result"] = f"[late error] {exc}"
                    rec["status"] = "failed_late"
                rec.pop("_future", None)
                done.append({k: v for k, v in rec.items()
                             if not k.startswith("_")})
                del self._unknown_outcomes[sig]
        return done

    def _register_unknown(self, call: ToolCall, fut: Any) -> None:
        with self._unknown_lock:
            self._unknown_outcomes[_call_signature(call)] = {
                "tool": call.name,
                "args": dict(call.args),
                "status": OUTCOME_UNKNOWN_MARKER,
                "started_at": time.time(),
                "_future": fut,
            }

    def _unknown_message(self, call: ToolCall) -> str:
        return (
            f"[tool error] '{call.name}' timed out after "
            f"{self.tool_timeout_s:.0f}s. Outcome UNKNOWN "
            f"({OUTCOME_UNKNOWN_MARKER}): "
            f"the tool may still finish — do not retry blindly; "
            f"reconcile via unknown_outcomes() before re-issuing."
        )

    def _emit(self, kind: str, iteration: int = 0, tool_name: str = "",
              detail: str = "") -> None:
        """Fire a lifecycle event. Never raises."""
        if self.on_event is None:
            return
        try:
            self.on_event(ToolLoopEvent(
                kind=kind, iteration=iteration, tool_name=tool_name,
                detail=detail))
        except Exception:  # noqa: BLE001 — telemetry never breaks the loop
            _log.debug("tool loop event hook failed", exc_info=True)

    def _tool_listing(
        self,
        capabilities: Any = None,
        query: str = "",
    ) -> str:
        """Tool list for the prompt, filtered to what the actor may use.

        Ranked by relevance to the query (top 40) instead of dumping all
        400+ tools — saves ~20k tokens per turn. Falls back to the full
        listing when ranking is unavailable.
        """
        try:
            ranked = self.registry.ranked_listing(
                query, capabilities=capabilities, limit=40)
            if ranked.strip():
                return ranked
        except (AttributeError, TypeError):
            pass
        try:
            listing = self.registry.prompt_listing(capabilities=capabilities)
        except TypeError:
            listing = self.registry.prompt_listing()
        if not listing.strip():
            return "(no tools available to you in this context)"
        return listing

    def _system_prompt(
        self,
        *,
        capabilities: Any = None,
        persona_extra: str = "",
        query: str = "",
    ) -> str:
        listing = self._tool_listing(capabilities, query)
        n = listing.count("\n- ") + (1 if listing.strip() else 0)
        return _SYSTEM_TEMPLATE.format(
            n_tools=n,
            tool_listing=listing,
            persona_extra=(persona_extra.rstrip() + "\n"
                           if persona_extra.strip() else ""),
        )

    def _submit(self, fn: Callable[..., Any], *args: Any,
                **kwargs: Any) -> Any:
        """Submit ``fn`` to the pool inside the submitter's context.

        contextvars (request/run/mission correlation ids) do not cross
        ThreadPoolExecutor boundaries on their own — without this copy,
        every ``tool.call`` bus event from the loop carries empty ids
        and replay/timeline cannot join the call to its request.
        """
        return self._pool.submit(
            contextvars.copy_context().run, fn, *args, **kwargs)

    def _invoke(
        self,
        call: ToolCall,
        *,
        actor: str,
        capabilities: Any = None,
        loop_ctx: dict[str, Any] | None = None,
    ) -> str:
        """Run one tool call synchronously. Never raises — failures are text.

        This is the pool-free core: :meth:`_execute` wraps it in a timeout,
        and parallel per-turn mode submits it directly (no nested pools,
        which would deadlock the 4-worker executor).
        """
        spec = None
        try:
            spec = self.registry.get(call.name)
        except Exception:
            pass
        if spec is None:
            names = []
            try:
                names = self.registry.names()
            except Exception:
                pass
            hint = ""
            # Honest, useful: suggest the closest name when it's a near-miss.
            close = difflib.get_close_matches(call.name, names, n=1, cutoff=0.7)
            if close:
                hint = f" Did you mean '{close[0]}'?"
            return (f"[tool error] unknown tool '{call.name}'.{hint} "
                    f"Use only tools from the list.")

        # Loop context (platform, chat, owner identity) rides on the
        # registry's context so tools can read it without signature changes.
        ctx = getattr(self.registry, "context", None)
        if ctx is not None and loop_ctx:
            try:
                setattr(ctx, "loop_ctx", dict(loop_ctx))
            except Exception:
                pass

        # Social gate: code-enforced, not prompt-only. For non-owner actors,
        # social-adjacent tools go through the grant check BEFORE the
        # registry. The registry's capability enforcement is the second
        # layer; this is the social-specific first layer.
        #
        # Group admins get admin tools in groups where they hold status:
        # the sender's role is resolved per-group (cached 60s), and the
        # grant reflects it. Owner bypass is unchanged.
        if actor != "owner" and (
            call.name.startswith(
                ("social_", "telegram_", "tgbot_", "whatsapp_", "game_",
                 "memory_")
            ) or call.name in ("games",)
        ):
            try:
                from ..social_gate import check_tool_call, grant_for
                from ..group_roles import resolve_group_role, ROLE_ADMIN
                ctx = loop_ctx or {}
                chat_kind = ctx.get("chat_kind", "dm")
                group_role = "member"
                # Only resolve roles for group chats — DMs don't have roles
                if chat_kind == "group":
                    platform = ctx.get("platform", "")
                    chat_key = ctx.get("chat_key", "")
                    sender_id = ctx.get("sender_id", "")
                    if platform and chat_key and sender_id:
                        try:
                            from ...tools.social import get_gateway
                            gateway = get_gateway()
                            adapter = None
                            if gateway is not None:
                                adapters = getattr(gateway, "adapters", {})
                                adapter = adapters.get(platform)
                            if adapter is not None:
                                group_role = resolve_group_role(
                                    platform, adapter, chat_key, sender_id)
                        except Exception:
                            pass  # fail closed → member
                grant = grant_for(is_owner=False, chat_kind=chat_kind,
                                  group_role=group_role)
                allowed, reason = check_tool_call(call.name, grant=grant)
                if not allowed:
                    return f"[denied] {reason}"
            except Exception:  # noqa: BLE001 — gate failure = deny
                return "[denied] social gate unavailable"

        def _run() -> Any:
            kwargs: dict[str, Any] = {"capabilities": capabilities} \
                if capabilities is not None else {}
            return self.registry.call(
                call.name, actor=actor, **kwargs, **call.args)

        # Outcome is Ok/Err — unwrap honestly.
        try:
            outcome = _run()
        except Exception as exc:  # noqa: BLE001
            self._note_tool_weakness(call.name, f"crashed: {exc}")
            return f"[tool error] '{call.name}' crashed: {exc}"
        try:
            if hasattr(outcome, "ok") and outcome.ok:
                value = outcome.value if hasattr(outcome, "value") else outcome
                text = self._render_value(value)
                return f"[tool ok] '{call.name}' returned:\n{text}"
            err = getattr(outcome, "error", outcome)
            msg = getattr(err, "message", str(err)) if err is not None else "unknown error"
            denied = "denied" in str(msg).lower() or "capability" in str(msg).lower()
            tag = "denied" if denied else "failed"
            if not denied:
                self._note_tool_weakness(call.name, msg)
            return (f"[tool {tag}] '{call.name}': {msg} — "
                    f"{'you lack permission for this tool.' if denied else 'adapt or report honestly.'}")
        except Exception as exc:  # noqa: BLE001
            return f"[tool error] '{call.name}' result unreadable: {exc}"

    def _execute(
        self,
        call: ToolCall,
        *,
        actor: str,
        capabilities: Any = None,
        loop_ctx: dict[str, Any] | None = None,
    ) -> str:
        """Run one tool call with timeout. Never raises — failures are text.

        Process-isolated tools (name in ``isolated_tools`` or spec
        metadata ``process_isolated``) run in a forked child: a
        timeout SIGKILLs the child, so the outcome is KNOWN (dead) —
        no unknown-outcome record, no still-running tool. Thread
        tools keep the cooperative path (cancel is best-effort, the
        outcome stays UNKNOWN and is recorded for reconciliation).
        """
        if self._is_isolated(call.name):
            return self._execute_isolated(
                call, actor=actor, capabilities=capabilities,
                loop_ctx=loop_ctx)
        fut = self._submit(
            self._invoke, call, actor=actor, capabilities=capabilities,
            loop_ctx=loop_ctx)
        try:
            return fut.result(timeout=self.tool_timeout_s)
        except FuturesTimeout:
            self._note_tool_weakness(call.name, "timeout")
            try:
                fut.cancel()  # no-op once running: the tool may still finish
            except Exception:  # noqa: BLE001 - cancellation is best-effort
                pass
            self._register_unknown(call, fut)
            return self._unknown_message(call)
        except Exception as exc:  # noqa: BLE001 — execution must not kill the loop
            self._note_tool_weakness(call.name, f"crashed: {exc}")
            return f"[tool error] '{call.name}' crashed: {exc}"

    def _is_isolated(self, name: str) -> bool:
        """True when this tool runs in a child process (forcible kill)."""
        if name in self.isolated_tools:
            return True
        try:
            spec = self.registry.get(name)
        except Exception:  # noqa: BLE001 - unknown tool = not isolated
            return False
        metadata = getattr(spec, "metadata", None) or {}
        return bool(metadata.get("process_isolated"))

    def _execute_isolated(self, call: ToolCall, *, actor: str,
                          capabilities: Any = None,
                          loop_ctx: dict[str, Any] | None = None) -> str:
        """Run one tool call in a forked child process. Never raises."""
        from ...core.process_isolation import run_isolated

        result = run_isolated(
            self._invoke, call, actor=actor, capabilities=capabilities,
            loop_ctx=loop_ctx, timeout_s=self.tool_timeout_s)
        if result.status == "ok":
            return str(result.value)
        if result.status == "error":
            self._note_tool_weakness(call.name, f"crashed: {result.error}")
            return f"[tool error] '{call.name}' crashed: {result.error}"
        if result.status == "timeout-killed":
            self._note_tool_weakness(call.name, "timeout-killed")
            return (f"[tool error] '{call.name}' timed out after "
                    f"{self.tool_timeout_s:.0f}s and its process was "
                    f"KILLED (forcible termination — outcome KNOWN: "
                    f"dead, no side effects after the kill). Do NOT "
                    f"re-issue blindly: check state first.")
        if result.status == "crashed":
            self._note_tool_weakness(call.name, f"crashed: {result.error}")
            return (f"[tool error] '{call.name}' process crashed: "
                    f"{result.error}")
        # not-supported (no fork): fall back to the cooperative thread
        # path — same unknown-outcome semantics as any thread tool.
        fut = self._submit(
            self._invoke, call, actor=actor, capabilities=capabilities,
            loop_ctx=loop_ctx)
        try:
            return fut.result(timeout=self.tool_timeout_s)
        except FuturesTimeout:
            self._note_tool_weakness(call.name, "timeout")
            try:
                fut.cancel()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            self._register_unknown(call, fut)
            return self._unknown_message(call)
        except Exception as exc:  # noqa: BLE001
            self._note_tool_weakness(call.name, f"crashed: {exc}")
            return f"[tool error] '{call.name}' crashed: {exc}"

    def _execute_turn(
        self,
        calls: list[ToolCall],
        *,
        actor: str,
        capabilities: Any = None,
        loop_ctx: dict[str, Any] | None = None,
        iteration: int = 0,
    ) -> list[str]:
        """Execute one turn's calls, observations in original call order.

        Sequential by default (order preserved, tool B sees tool A's
        result next turn). With ``parallel_per_turn=True`` the calls run
        concurrently in the pool — smolagents ToolCallingAgent style —
        which suits independent lookups.
        """
        if not self.parallel_per_turn or len(calls) < 2:
            out: list[str] = []
            for call in calls:
                self._emit("tool_start", iteration=iteration,
                           tool_name=call.name)
                obs = self._execute(
                    call, actor=actor, capabilities=capabilities,
                    loop_ctx=loop_ctx)
                self._emit("tool_end", iteration=iteration,
                           tool_name=call.name,
                           detail=obs[:120].replace("\n", " "))
                out.append(obs)
            return out
        # Parallel: submit _invoke directly (never _execute — that would
        # nest pool submissions and deadlock the 4-worker executor).
        pending = [
            (call, self._submit(
                self._invoke, call, actor=actor,
                capabilities=capabilities, loop_ctx=loop_ctx))
            for call in calls
        ]
        for call, _ in pending:
            self._emit("tool_start", iteration=iteration, tool_name=call.name)
        # ONE absolute deadline for the whole parallel turn — not a fresh
        # timeout per future, which could consume N × timeout overall.
        deadline = time.monotonic() + self.tool_timeout_s
        out = []
        for call, fut in pending:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                obs = fut.result(timeout=remaining)
            except FuturesTimeout:
                self._note_tool_weakness(call.name, "timeout")
                try:
                    fut.cancel()
                except Exception:  # noqa: BLE001 - best-effort
                    pass
                self._register_unknown(call, fut)
                obs = self._unknown_message(call)
            except Exception as exc:  # noqa: BLE001
                self._note_tool_weakness(call.name, f"crashed: {exc}")
                obs = f"[tool error] '{call.name}' crashed: {exc}"
            self._emit("tool_end", iteration=iteration, tool_name=call.name,
                       detail=str(obs)[:120].replace("\n", " "))
            out.append(obs)
        return out

    def _note_tool_weakness(self, tool_name: str, error: str) -> None:
        """Feed tool failures into weakness detection. Never breaks the loop.

        The ledger lives in the app database (the registry context's db),
        not a sidecar — a tool that keeps failing is exactly what the
        weakness system clusters on.
        """
        try:
            from ...autonomy.weakness import report_weakness
            import time as _t
            db = getattr(getattr(self.registry, "context", None),
                         "db", None)
            if db is None:
                return
            report_weakness(db, "tool_failure", tool_name,
                            {"error": str(error)[:300], "ts": _t.time()})
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _render_value(value: Any, _depth: int = 0) -> str:
        """Render a tool result for the model. Bounded, never raises.

        Truncation is NEVER silent and NEVER produces fragmentary JSON:

        * dicts over the limit become a valid-JSON envelope with
          ``_complete: false``, the key list, a per-key preview, and a
          ``_remainder`` handle whose full text
          :func:`retrieve_remainder` returns;
        * strings over the limit keep their head plus an explicit
          ``[…truncated …]`` marker naming the remainder handle;
        * lists keep the ``… (N more)`` tail they already had.
        """
        if _depth > 3:
            return "…"
        if value is None:
            return "(no output)"
        if isinstance(value, str):
            if len(value) <= _OBSERVATION_LIMIT:
                return value or "(empty)"
            key = store_remainder(value)
            return (
                f"{value[:_OBSERVATION_LIMIT]}\n"
                f"[…truncated: showing {_OBSERVATION_LIMIT:,} of "
                f"{len(value):,} chars — complete=false — "
                f"retrieve remainder {key} for the full result]"
            )
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            try:
                text = json.dumps(value, indent=1, default=str,
                                  ensure_ascii=False)
            except Exception:
                return ToolCallingLoop._render_value(
                    str(value), _depth + 1)
            if len(text) <= _OBSERVATION_LIMIT:
                return text
            key = store_remainder(text)
            preview: dict[str, Any] = {}
            for k, v in value.items():
                rendered = ToolCallingLoop._render_value(v, _depth + 1)
                preview[str(k)] = (
                    rendered[:200] + "…" if len(rendered) > 200
                    else rendered)
                if len(json.dumps(preview, default=str,
                                  ensure_ascii=False)) > 1500:
                    break
            envelope = {
                "_complete": False,
                "_truncated": True,
                "_total_chars": len(text),
                "_remainder": key,
                "_keys": sorted(str(k) for k in value),
                "_preview": preview,
            }
            return json.dumps(envelope, indent=1, default=str,
                              ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            items = [ToolCallingLoop._render_value(v, _depth + 1)
                     for v in list(value)[:20]]
            tail = f"\n… ({len(value) - 20} more)" if len(value) > 20 else ""
            return "\n".join(items) + tail
        try:
            return ToolCallingLoop._render_value(str(value), _depth + 1)
        except Exception:
            return "(unprintable result)"

    @staticmethod
    def retrieve_remainder(key: str) -> str | None:
        """Fetch a full result stored behind a truncated observation."""
        return retrieve_remainder(key)

    def run(
        self,
        goal: str,
        *,
        context_lines: Sequence[str] = (),
        actor: str = "owner",
        capabilities: Any = None,
        persona_extra: str = "",
        loop_ctx: dict[str, Any] | None = None,
        history: Sequence[Any] = (),
        request_id: str = "",
        run_id: str = "",
    ) -> ToolLoopResult:
        """Run the loop until the model answers or budgets run out.

        ``goal`` is the user's message. ``context_lines`` is background
        (memories, chat profile, continuity). ``actor`` is "owner" or
        "outsider" — enforced at execution. ``loop_ctx`` carries platform,
        chat key, and owner identity to tools that read it.
        ``request_id`` / ``run_id`` join this turn to the caller's
        request (correlation); generated when omitted.
        """
        from ...core.correlation import (
            correlation_context,
            new_request_id,
            new_run_id,
        )
        from ...llm.base import Message

        request_id = request_id or new_request_id()
        run_id = run_id or new_run_id()
        with correlation_context(request_id=request_id, run_id=run_id):
            return self._run_inner(
                goal, context_lines=context_lines, actor=actor,
                capabilities=capabilities, persona_extra=persona_extra,
                loop_ctx=loop_ctx, history=history,
                request_id=request_id, run_id=run_id)

    def _run_inner(
        self,
        goal: str,
        *,
        context_lines: Sequence[str] = (),
        actor: str = "owner",
        capabilities: Any = None,
        persona_extra: str = "",
        loop_ctx: dict[str, Any] | None = None,
        history: Sequence[Any] = (),
        request_id: str = "",
        run_id: str = "",
    ) -> ToolLoopResult:
        from ...llm.base import Message

        started = time.time()
        trace: list[ToolLoopTrace] = []
        tools_used: list[str] = []

        system = self._system_prompt(
            capabilities=capabilities, persona_extra=persona_extra,
            query=goal)

        user_block = goal
        if context_lines:
            user_block = ("Context:\n" + "\n".join(context_lines)
                          + "\n\n" + goal)

        messages: list[Any] = [Message.system(system)]
        for h in history:
            messages.append(h)
        messages.append(Message.user(user_block))

        degraded = False
        degrade_note = ""
        model_name = ""
        #: Stall detection: the same (tool, args) repeated back-to-back means
        #: the model is stuck. Durable-agent practice: stop burning budget
        #: and degrade honestly instead.
        recent_sigs: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            elapsed = time.time() - started
            if elapsed >= self.total_timeout_s:
                degraded, degrade_note = True, (
                    f"total budget ({self.total_timeout_s:.0f}s) exhausted "
                    f"after {iteration - 1} iterations")
                break

            self._emit("turn_start", iteration=iteration)
            step_started = time.perf_counter()
            try:
                resp = self.llm.chat(
                    messages, task_kind="tool_loop",
                    timeout_s=min(120.0, self.total_timeout_s - elapsed))
            except Exception as exc:  # noqa: BLE001
                return ToolLoopResult(
                    answer="", ok=False,
                    error=f"model call failed on iteration {iteration}: {exc}",
                    iterations=iteration - 1, trace=trace,
                    tools_used=tools_used, request_id=request_id, run_id=run_id)

            model_name = getattr(resp, "model", "") or model_name
            if not getattr(resp, "ok", False):
                err = getattr(resp, "error", "unknown model error")
                # One honest retry on transient-looking failures happens
                # inside the router; here we surface it.
                return ToolLoopResult(
                    answer="", ok=False,
                    error=f"model failed on iteration {iteration}: {err}",
                    iterations=iteration - 1, trace=trace,
                    tools_used=tools_used, degraded=True,
                    degrade_note=f"model error: {err}",
                    request_id=request_id, run_id=run_id)

            text = getattr(resp, "text", "") or ""
            all_calls = parse_tool_calls(text)
            calls = all_calls[:self.max_tool_calls_per_turn]
            dropped = len(all_calls) - len(calls)

            step = ToolLoopTrace(
                iteration=iteration, model_text=text[:2000],
                tool_calls=calls,
                latency_ms=(time.perf_counter() - step_started) * 1000.0)

            if not calls:
                # No tool calls: the model's text IS the final answer.
                trace.append(step)
                answer = _strip_tool_blocks(text).strip()
                result = ToolLoopResult(
                    answer=answer or "(empty response)",
                    tools_used=tools_used, iterations=iteration,
                    trace=trace, degraded=degraded,
                    degrade_note=degrade_note, model=model_name)
                self._emit("final", iteration=iteration,
                           detail=f"answer, {len(answer)} chars")
                return result

            # Stall detection: identical calls back-to-back = stuck model.
            sigs = [_call_signature(c) for c in calls]
            recent_sigs = (recent_sigs + sigs)[-_STALL_REPEATS:]
            if (len(recent_sigs) == _STALL_REPEATS
                    and len(set(recent_sigs)) == 1):
                stall = calls[-1]
                self._emit("stalled", iteration=iteration,
                          tool_name=stall.name,
                          detail=f"same call repeated {_STALL_REPEATS}x")
                trace.append(step)
                result = ToolLoopResult(
                    answer=("I got stuck repeating the same tool call "
                            f"('{stall.name}') — stopping instead of "
                            "burning more budget."),
                    tools_used=tools_used, iterations=iteration,
                    trace=trace, degraded=True,
                    degrade_note=(f"stall: '{stall.name}' repeated "
                                  f"{_STALL_REPEATS}x consecutively"),
                    model=model_name, ok=False,
                    error="stall: repeated identical tool call",
                    request_id=request_id, run_id=run_id)
                self._emit("final", iteration=iteration, detail="stalled")
                return result

            # Execute tools, feed observations back.
            messages.append(Message.assistant(text))
            observations = self._execute_turn(
                calls, actor=actor, capabilities=capabilities,
                loop_ctx=loop_ctx, iteration=iteration)
            for call, obs in zip(calls, observations):
                tools_used.append(call.name)
                step.observations.append(obs[:500])
            self._emit("turn_end", iteration=iteration,
                       detail=f"{len(calls)} tool(s)")
            trace.append(step)

            obs_block = "\n\n".join(
                f"Observation {i + 1}:\n{o}"
                for i, o in enumerate(observations))
            if dropped > 0:
                obs_block += (
                    f"\n\nNote: {dropped} tool call(s) were dropped "
                    f"(limit {self.max_tool_calls_per_turn} per turn) — "
                    f"re-emit the important ones next turn if still needed.")
            messages.append(Message.user(
                f"Tool observations (reflect, then continue or answer):\n\n"
                f"{obs_block}"))

        # Budget exhausted without a final answer.
        result = ToolLoopResult(
            answer="(ran out of iterations before finishing)",
            tools_used=tools_used, iterations=self.max_iterations,
            trace=trace, degraded=True,
            degrade_note=degrade_note or (
                f"iteration budget ({self.max_iterations}) exhausted; "
                f"used: {', '.join(tools_used) or 'none'}"),
            model=model_name, ok=False,
            error="iteration budget exhausted",
            request_id=request_id, run_id=run_id)
        self._emit("final", iteration=self.max_iterations,
                   detail="budget exhausted")
        return result

    def audit_reachability(
        self, capabilities: Any = None,
    ) -> dict[str, Any]:
        """Every registered tool must be reachable from the loop.

        Returns {total, listed, missing[]} — missing tools are reported,
        never silently dropped.
        """
        try:
            names = set(self.registry.names())
        except Exception as exc:
            return {"total": 0, "listed": 0, "missing": [],
                    "error": f"registry unreadable: {exc}"}
        try:
            schemas = self.registry.schemas(capabilities=capabilities)
            listed = {s.get("name", "") for s in schemas}
        except TypeError:
            schemas = self.registry.schemas()
            listed = {s.get("name", "") for s in schemas}
        except Exception as exc:
            return {"total": len(names), "listed": 0, "missing": sorted(names),
                    "error": f"schemas unreadable: {exc}"}
        missing = sorted(names - listed)
        return {"total": len(names), "listed": len(listed),
                "missing": missing}


def render_transcript(result: ToolLoopResult, *, max_obs_chars: int = 300) -> str:
    """Render a run's trace as human-readable markdown.

    The model's thinking is the narrative; tool calls are compact lines
    with truncated observations. Designed for chat display — this is what
    the user sees when they ask "what did you do?".
    """
    from ..render import ICONS, banner, bar, bullets, kv, truncate

    lines = [banner("Tool loop", ICONS["tool"])]
    head = {
        "iterations": result.iterations,
        "tools used": ", ".join(result.tools_used) or "none",
        "model": result.model or "—",
        "status": ("degraded — " + result.degrade_note) if result.degraded
                  else ("failed — " + result.error if not result.ok else "ok"),
    }
    lines.append(kv(head.items()))
    for step in result.trace:
        lines.append("")
        lines.append(f"**Turn {step.iteration}** "
                     f"({step.latency_ms:.0f} ms)")
        thought = _strip_tool_blocks(step.model_text).strip()
        if thought:
            lines.append(truncate(thought, 500))
        for call in step.tool_calls:
            arg_preview = truncate(
                ", ".join(f"{k}={truncate(str(v), 40)}"
                          for k, v in call.args.items()), 90)
            lines.append(f"`{call.name}`({arg_preview})")
        for obs in step.observations:
            icon = (ICONS["fail"] if "[tool error]" in obs
                    or "[tool failed]" in obs else ICONS["ok"])
            lines.append(f"{icon} {truncate(obs, max_obs_chars)}")
    lines.append("")
    lines.append(f"**Answer:** {truncate(result.answer, 800)}")
    return "\n".join(lines)
