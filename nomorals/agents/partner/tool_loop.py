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

import difflib
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

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
        """Run one tool call with timeout. Never raises — failures are text."""
        fut = self._pool.submit(
            self._invoke, call, actor=actor, capabilities=capabilities,
            loop_ctx=loop_ctx)
        try:
            return fut.result(timeout=self.tool_timeout_s)
        except FuturesTimeout:
            self._note_tool_weakness(call.name, "timeout")
            return (f"[tool error] '{call.name}' timed out after "
                    f"{self.tool_timeout_s:.0f}s.")
        except Exception as exc:  # noqa: BLE001 — execution must not kill the loop
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
            (call, self._pool.submit(
                self._invoke, call, actor=actor,
                capabilities=capabilities, loop_ctx=loop_ctx))
            for call in calls
        ]
        for call, _ in pending:
            self._emit("tool_start", iteration=iteration, tool_name=call.name)
        out = []
        for call, fut in pending:
            try:
                obs = fut.result(timeout=self.tool_timeout_s)
            except FuturesTimeout:
                self._note_tool_weakness(call.name, "timeout")
                obs = (f"[tool error] '{call.name}' timed out after "
                       f"{self.tool_timeout_s:.0f}s.")
            except Exception as exc:  # noqa: BLE001
                self._note_tool_weakness(call.name, f"crashed: {exc}")
                obs = f"[tool error] '{call.name}' crashed: {exc}"
            self._emit("tool_end", iteration=iteration, tool_name=call.name,
                       detail=str(obs)[:120].replace("\n", " "))
            out.append(obs)
        return out

    def _note_tool_weakness(self, tool_name: str, error: str) -> None:
        """Feed tool failures into weakness detection. Never breaks the loop."""
        try:
            from ...autonomy.weakness import report_weakness
            from ...storage.db import Database
            import time as _t
            # Workspace dir from loop_ctx if available.
            ws = None
            try:
                ws = getattr(self, "_workspace_dir", None)
            except Exception:  # noqa: BLE001
                pass
            if not ws:
                return
            db = Database(ws)
            report_weakness(db, "tool_failure", tool_name,
                            {"error": str(error)[:300], "ts": _t.time()})
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _render_value(value: Any, _depth: int = 0) -> str:
        """Render a tool result for the model. Bounded, never raises."""
        if _depth > 3:
            return "…"
        if value is None:
            return "(no output)"
        if isinstance(value, str):
            return value[:4000] if len(value) > 4000 else value or "(empty)"
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            try:
                text = json.dumps(value, indent=1, default=str)
            except Exception:
                return str(value)[:4000]
            return text[:4000]
        if isinstance(value, (list, tuple)):
            items = [ToolCallingLoop._render_value(v, _depth + 1)
                     for v in list(value)[:20]]
            tail = f"\n… ({len(value) - 20} more)" if len(value) > 20 else ""
            return "\n".join(items) + tail
        try:
            return str(value)[:4000]
        except Exception:
            return "(unprintable result)"

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
    ) -> ToolLoopResult:
        """Run the loop until the model answers or budgets run out.

        ``goal`` is the user's message. ``context_lines`` is background
        (memories, chat profile, continuity). ``actor`` is "owner" or
        "outsider" — enforced at execution. ``loop_ctx`` carries platform,
        chat key, and owner identity to tools that read it.
        """
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
                    tools_used=tools_used)

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
                    degrade_note=f"model error: {err}")

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
                    error="stall: repeated identical tool call")
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
            error="iteration budget exhausted")
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
