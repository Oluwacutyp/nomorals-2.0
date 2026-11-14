"""Journal trace/replay view (LangFuse shape, mined — never a hosted dep).

Devon's journal already records the events — conversations, messages,
tool calls, audit entries, cognition stages, skill uses. What was
missing is the *view*: open one request and see the whole chain, then
re-run it against the current code and diff the outputs. This module is
that view. It is strictly read-only over the journal: it never writes a
row, never changes what gets journaled, and never adds a store.

A **trace** is one conversation (``conversations.id``). Its **spans** are
the journal rows that belong to it, in time order:

* ``message`` spans — from ``messages``, joined exactly on
  ``conversation_id``. Full content is recorded, so these carry real
  inputs (user) and outputs (assistant), plus the model + token count
  where the writer recorded them. Latency and cost are NOT recorded per
  message — they are listed under ``missing``, never invented.
* ``tool`` spans — from ``tool_calls``, joined by time window (the
  conversation's ``[created_at, updated_at]``). The journal records only
  an ``args_digest`` and a ``result_digest`` (which the writer never
  fills in) — the full arguments and results are NOT in the journal, so
  they are listed under ``missing`` with the reason, never guessed.
  A tool span also carries its policy decision, status, duration and
  error, which the journal does record.
* ``audit`` / ``cognition`` / ``skill`` spans — from ``audit_log`` /
  ``cognition_log`` / ``skill_uses``, joined by the same time window.

Spans joined by time window are marked ``join: "time-window"``; message
spans are marked ``join: "conversation"``. A trace is split into **turns**
at each user message: a turn is one request and everything Devon did
for it until the next request.

**Replay** re-runs a trace's recorded inputs through the current code
and diffs the outputs. The runners are injected (the brain is injected
by the layers above; this module never imports it):

* model spans — the recorded prompt (the messages up to that assistant
  reply) goes through ``model_runner``; the new text is compared with
  the recorded text: ``same`` / ``changed`` / ``diverged``.
* tool spans — re-runnable only when the caller supplies the arguments
  (``tool_args``, keyed by span id: the journal holds digests only) AND
  ``allow_tools`` is set AND the tool's recorded capability is read-only
  (``fs.read`` / ``mem.read`` / ``db.read`` / ``social.read``). Anything
  else is ``skipped`` with the precise reason — a tool that would write
  is reported as ``would write (capability)`` and is never executed.
  A re-run tool has no recorded output to compare against (the journal
  never stores one), so a clean re-run reports ``replayed``.
* everything else (user messages, audit, cognition, skill spans) is
  ``skipped`` — input or observation only, nothing to re-run.

Replay is bounded: at most ``max_spans`` spans (default 50, the rest
reported as truncated) and ``span_timeout_s`` per span (default 30s).
Replay never writes to the journal or any live store.

LangFuse / AgentOps / LangSmith / Helicone are the mined shape, not a
dependency: no hosted service, no new pip package, no telemetry leaves
the box. See ~/workspace/agents-list-verify/AGENTS_MAP.md,
"Observability Frameworks".
"""

from __future__ import annotations

import concurrent.futures
import html as _html
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability
from ..storage.conversations import ConversationLog
from ..storage.rowstore import attached_rows

__all__ = [
    "READ_ONLY_CAPABILITIES",
    "ReplayResult",
    "Span",
    "Trace",
    "TraceBuilder",
    "TraceNotFound",
    "Turn",
    "get_model_runner",
    "get_tool_runner",
    "replay_trace",
    "set_model_runner",
    "set_tool_runner",
    "trace_page",
    "write_trace_page",
]

#: Tool capabilities that cannot write anywhere. Everything else is
#: treated as a write for replay purposes and is never executed.
READ_ONLY_CAPABILITIES: frozenset[str] = frozenset({
    Capability.FS_READ,
    Capability.MEM_READ,
    Capability.DB_READ,
    Capability.SOCIAL_READ,
})

#: How much content a tool result carries (the model sees the trace
#: through the registry tools; a trace is not a dump).
TOOL_CONTENT_CHARS = 2000

#: How much content the HTML page carries per span.
PAGE_CONTENT_CHARS = 2000

#: How much content the CLI text view carries per span.
CLI_CONTENT_CHARS = 500

#: A trace's spans, newest-first cap for the list view is the caller's.
DEFAULT_LIST_LIMIT = 10

#: Replay bounds.
DEFAULT_MAX_SPANS = 50
DEFAULT_SPAN_TIMEOUT_S = 30.0

#: Journal tables a trace reads (fixed names, never user input).
_TOOL_TABLE = "tool_calls"
_AUDIT_TABLE = "audit_log"
_COGNITION_TABLE = "cognition_log"
_SKILL_TABLE = "skill_uses"

#: ``(table, timestamp column)`` for the time-window join.
_WINDOW_TABLES: tuple[tuple[str, str], ...] = (
    (_TOOL_TABLE, "created_at"),
    (_AUDIT_TABLE, "ts"),
    (_COGNITION_TABLE, "ts"),
    (_SKILL_TABLE, "used_at"),
)

#: HTML page policy: the page is self-contained (inline CSS only, no
#: scripts, no external resources, no images), so the policy allows
#: inline styles and nothing else.
TRACE_CSP = "default-src 'none'; style-src 'unsafe-inline'"

_PAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# ── runner seams (models_provision pattern: module attributes so tests
# and the CLI can replace them; importing this module runs nothing) ──

_model_runner: Callable[[list[dict[str, str]]], str] | None = None
_tool_runner: Callable[[str, dict[str, Any]], Any] | None = None


def set_model_runner(fn: Callable[[list[dict[str, str]]], str] | None) -> None:
    """Attach the model runner replay uses (``None`` detaches it)."""
    global _model_runner
    _model_runner = fn


def set_tool_runner(fn: Callable[[str, dict[str, Any]], Any] | None) -> None:
    """Attach the tool runner replay uses (``None`` detaches it)."""
    global _tool_runner
    _tool_runner = fn


def get_model_runner() -> Callable[[list[dict[str, str]]], str] | None:
    """The attached model runner, if any."""
    return _model_runner


def get_tool_runner() -> Callable[[str, dict[str, Any]], Any] | None:
    """The attached tool runner, if any."""
    return _tool_runner


class TraceNotFound(ToolError):
    """No conversation with that id exists in the journal."""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _json_or_none(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """(possibly truncated text, whether it was cut)."""
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


# ── spans / turns / traces ─────────────────────────────────────────────


@dataclass
class Span:
    """One journal row inside a trace."""

    id: str
    kind: str  # message | tool | audit | cognition | skill
    title: str
    ts: float
    duration_ms: float | None = None
    status: str = ""
    join: str = "time-window"  # conversation | time-window
    input: str | None = None
    output: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def to_dict(self, *, truncate: int = TOOL_CONTENT_CHARS) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "ts": self.ts,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "join": self.join,
            "missing": list(self.missing),
            "detail": self.detail,
        }
        if self.input is not None:
            text, cut = _truncate(self.input, truncate)
            data["input"] = text
            data["input_truncated"] = cut
        if self.output is not None:
            text, cut = _truncate(self.output, truncate)
            data["output"] = text
            data["output_truncated"] = cut
        return data


@dataclass
class Turn:
    """One user request and everything Devon did for it."""

    index: int
    start_ts: float
    end_ts: float
    user_text: str
    span_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "user_text": self.user_text,
            "span_ids": list(self.span_ids),
        }


@dataclass
class Trace:
    """One conversation reconstructed as a trace."""

    conversation_id: str
    title: str
    agent: str
    channel: str
    created_at: float
    updated_at: float
    spans: list[Span] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, truncate: int = TOOL_CONTENT_CHARS) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "agent": self.agent,
            "channel": self.channel,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stats": self.stats,
            "turns": [t.to_dict() for t in self.turns],
            "spans": [s.to_dict(truncate=truncate) for s in self.spans],
        }


# ── builder ────────────────────────────────────────────────────────────


class TraceBuilder:
    """Reconstruct traces from the existing journal. Read-only."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.log = ConversationLog(db)

    # ── reading (seam-aware: RowTable when attached, else SQL) ──

    def _rows(self, table: str, ts_col: str, lo: float,
              hi: float) -> list[dict[str, Any]]:
        """Rows of ``table`` with ``ts_col`` inside ``[lo, hi]``, oldest first."""
        seam = attached_rows(self.db, table)
        if seam is not None:
            rows = seam.find({ts_col: {"$gte": lo, "$lte": hi}})
            rows.sort(key=lambda r: (_num(r.get(ts_col)), str(r.get("id", ""))))
            return rows
        return self.db.query(
            f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} <= ? "
            f"ORDER BY {ts_col}, id",
            (lo, hi),
        )

    def _count(self, table: str, col: str, value: Any) -> int:
        seam = attached_rows(self.db, table)
        if seam is not None:
            return len(seam.find({col: value}))
        row = self.db.query_one(
            f"SELECT COUNT(*) AS n FROM {table} WHERE {col} = ?", (value,))
        return int((row or {}).get("n") or 0)

    # ── list ──

    def list_traces(self, *, limit: int = DEFAULT_LIST_LIMIT,
                    channel: str = "", agent: str = "",
                    since: float = 0.0) -> list[dict[str, Any]]:
        """Recent conversations, newest first, with span counts."""
        rows = self.log.conversations()
        rows.sort(key=lambda r: (_num(r.get("updated_at")),
                                 _num(r.get("created_at")),
                                 str(r.get("id", ""))), reverse=True)
        out: list[dict[str, Any]] = []
        for row in rows:
            if channel and (row.get("channel") or "") != channel:
                continue
            if agent and (row.get("agent") or "") != agent:
                continue
            if since and _num(row.get("updated_at")) < since:
                continue
            conv_id = str(row.get("id", ""))
            out.append({
                "conversation_id": conv_id,
                "title": _text(row.get("title")),
                "agent": _text(row.get("agent")),
                "channel": _text(row.get("channel")),
                "created_at": _num(row.get("created_at")),
                "updated_at": _num(row.get("updated_at")),
                "tokens": int(_num(row.get("tokens"))),
                "messages": self._count("messages", "conversation_id", conv_id),
            })
            if len(out) >= max(1, limit):
                break
        return out

    # ── build ──

    def build(self, conversation_id: str) -> Trace:
        """Reconstruct one conversation as a trace."""
        conv = self.log.get_conversation(conversation_id)
        if conv is None:
            raise TraceNotFound(
                f"no conversation {conversation_id!r} in the journal")
        created = _num(conv.get("created_at"))
        updated = _num(conv.get("updated_at")) or created
        messages = self.log.messages_for(conversation_id)
        if messages:
            first_ts = _num(messages[0].get("created_at"))
            last_ts = _num(messages[-1].get("created_at"))
            lo = min(created, first_ts) if created else first_ts
            hi = max(updated, last_ts)
        else:
            lo, hi = created, updated

        spans: list[Span] = []
        for row in messages:
            role = _text(row.get("role")) or "message"
            content = _text(row.get("content"))
            model = _text(row.get("model"))
            tokens = int(_num(row.get("tokens")))
            detail: dict[str, Any] = {"role": role}
            if model:
                detail["model"] = model
            if tokens:
                detail["tokens"] = tokens
            missing = ["latency_ms", "cost"]
            spans.append(Span(
                id=str(row.get("id", "")),
                kind="message",
                title=f"{role}: {content[:60]}" if content else role,
                ts=_num(row.get("created_at")),
                status=role,
                join="conversation",
                input=content if role == "user" else None,
                output=content if role != "user" else None,
                detail=detail,
                missing=missing,
            ))
        spans.extend(self._tool_spans(lo, hi))
        spans.extend(self._audit_spans(lo, hi))
        spans.extend(self._cognition_spans(lo, hi))
        spans.extend(self._skill_spans(lo, hi))
        # Time order; ties break message-first so a turn's user message
        # precedes the tool calls it caused.
        kind_order = {"message": 0, "tool": 1, "audit": 2, "cognition": 3,
                      "skill": 4}
        spans.sort(key=lambda s: (s.ts, kind_order.get(s.kind, 9), s.id))

        turns = self._turns(spans, hi)
        return Trace(
            conversation_id=conversation_id,
            title=_text(conv.get("title")),
            agent=_text(conv.get("agent")),
            channel=_text(conv.get("channel")),
            created_at=created,
            updated_at=updated,
            spans=spans,
            turns=turns,
            stats=self._stats(spans, messages, len(turns)),
        )

    def _tool_spans(self, lo: float, hi: float) -> list[Span]:
        out: list[Span] = []
        for row in self._rows(_TOOL_TABLE, "created_at", lo, hi):
            tool = _text(row.get("tool")) or "?"
            status = _text(row.get("status")) or "pending"
            out.append(Span(
                id=str(row.get("id", "")),
                kind="tool",
                title=f"tool {tool}",
                ts=_num(row.get("created_at")),
                duration_ms=_num(row.get("duration_ms")),
                status=status,
                join="time-window",
                detail={
                    "tool": tool,
                    "actor": _text(row.get("actor")),
                    "capability": _text(row.get("capability")),
                    "decision": _text(row.get("decision")),
                    "args_digest": _text(row.get("args_digest")),
                    "error": _text(row.get("error")),
                },
                missing=["args", "result"],
            ))
        return out

    def _audit_spans(self, lo: float, hi: float) -> list[Span]:
        out: list[Span] = []
        for row in self._rows(_AUDIT_TABLE, "ts", lo, hi):
            action = _text(row.get("action")) or "?"
            detail = _json_or_none(row.get("detail")) or {}
            out.append(Span(
                id=str(row.get("id", "")),
                kind="audit",
                title=f"audit {action}",
                ts=_num(row.get("ts")),
                status=_text(row.get("decision")),
                join="time-window",
                detail={
                    "action": action,
                    "actor": _text(row.get("actor")),
                    "capability": _text(row.get("capability")),
                    "decision": _text(row.get("decision")),
                    "detail": detail if isinstance(detail, dict) else
                    {"value": detail},
                },
            ))
        return out

    def _cognition_spans(self, lo: float, hi: float) -> list[Span]:
        out: list[Span] = []
        for row in self._rows(_COGNITION_TABLE, "ts", lo, hi):
            stages = _json_or_none(row.get("stages")) or {}
            seconds = _num(row.get("seconds"))
            out.append(Span(
                id=str(row.get("id", "")),
                kind="cognition",
                title="cognition",
                ts=_num(row.get("ts")),
                duration_ms=round(seconds * 1000, 3) if seconds else None,
                join="time-window",
                detail={
                    "stages": stages if isinstance(stages, dict) else
                    {"value": stages},
                    "note": _text(row.get("note")),
                },
            ))
        return out

    def _skill_spans(self, lo: float, hi: float) -> list[Span]:
        out: list[Span] = []
        for row in self._rows(_SKILL_TABLE, "used_at", lo, hi):
            skill = _text(row.get("skill_id")) or "?"
            ok = bool(row.get("success"))
            out.append(Span(
                id=str(row.get("id", "")),
                kind="skill",
                title=f"skill {skill}",
                ts=_num(row.get("used_at")),
                status="ok" if ok else "failed",
                join="time-window",
                detail={
                    "skill_id": skill,
                    "task": _text(row.get("task")),
                    "outcome": _text(row.get("outcome")),
                    "context": _text(row.get("context")),
                },
            ))
        return out

    def _turns(self, spans: list[Span], end_ts: float) -> list[Turn]:
        turns: list[Turn] = []
        current: Turn | None = None
        for span in spans:
            if span.kind == "message" and span.detail.get("role") == "user":
                if current is not None:
                    current.end_ts = span.ts
                current = Turn(
                    index=len(turns),
                    start_ts=span.ts,
                    end_ts=end_ts,
                    user_text=span.output or span.input or "",
                    span_ids=[],
                )
                turns.append(current)
            if current is None:
                current = Turn(index=0, start_ts=span.ts, end_ts=end_ts,
                               user_text="", span_ids=[])
                turns.append(current)
            current.span_ids.append(span.id)
        if current is not None:
            current.end_ts = end_ts
        return turns

    def _stats(self, spans: list[Span],
               messages: list[dict[str, Any]],
               n_turns: int) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        tool_status: dict[str, int] = {}
        tool_ms = 0.0
        for span in spans:
            by_kind[span.kind] = by_kind.get(span.kind, 0) + 1
            if span.kind == "tool":
                tool_status[span.status] = tool_status.get(span.status, 0) + 1
                tool_ms += span.duration_ms or 0.0
        tokens = sum(int(_num(m.get("tokens"))) for m in messages)
        return {
            "spans": len(spans),
            "by_kind": by_kind,
            "tool_status": tool_status,
            "total_tool_ms": round(tool_ms, 3),
            "total_tokens": tokens,
            "turns": n_turns,
        }


# ── replay ─────────────────────────────────────────────────────────────


@dataclass
class ReplayResult:
    """What replaying one trace produced."""

    conversation_id: str
    verdicts: dict[str, int]
    results: list[dict[str, Any]]
    truncated: bool
    spans_considered: int
    spans_total: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "verdicts": self.verdicts,
            "results": self.results,
            "truncated": self.truncated,
            "spans_considered": self.spans_considered,
            "spans_total": self.spans_total,
            "seconds": round(self.seconds, 3),
        }


def _run_with_timeout(fn: Callable[[], Any],
                      timeout_s: float) -> tuple[Any, str | None]:
    """Run ``fn`` with a per-span timeout. Returns (value, error-or-None)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=max(0.1, timeout_s)), None
        except concurrent.futures.TimeoutError:
            return None, f"span timed out after {timeout_s:g}s"
        except Exception as exc:  # noqa: BLE001 - a runner failure is a verdict
            return None, f"{type(exc).__name__}: {exc}"


def replay_trace(
    trace: Trace,
    *,
    model_runner: Callable[[list[dict[str, str]]], str] | None = None,
    tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
    tool_args: dict[str, dict[str, Any]] | None = None,
    allow_tools: bool = False,
    max_spans: int = DEFAULT_MAX_SPANS,
    span_timeout_s: float = DEFAULT_SPAN_TIMEOUT_S,
) -> ReplayResult:
    """Re-run a trace's recorded inputs through the current code.

    ``model_runner`` / ``tool_runner`` default to the module seams
    (``set_model_runner`` / ``set_tool_runner``). Nothing here writes to
    the journal or any live store: model spans generate text, tool spans
    run only when read-only + args supplied + ``allow_tools``.
    """
    started = time.perf_counter()
    model_runner = model_runner if model_runner is not None else _model_runner
    tool_runner = tool_runner if tool_runner is not None else _tool_runner
    tool_args = tool_args or {}
    spans = trace.spans
    truncated = len(spans) > max(1, max_spans)
    considered = spans[:max(1, max_spans)]

    # The recorded prompt at each assistant message: every message span
    # before it, in order.
    prompt_at: dict[str, list[dict[str, str]]] = {}
    prompt: list[dict[str, str]] = []
    for span in spans:
        if span.kind == "message":
            role = str(span.detail.get("role") or "message")
            text = span.output or span.input or ""
            if role == "assistant":
                prompt_at[span.id] = list(prompt)
            prompt.append({"role": role, "content": text})

    verdicts: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for span in considered:
        verdict, reason, recorded, replayed = _replay_span(
            span, prompt_at.get(span.id, []),
            model_runner=model_runner, tool_runner=tool_runner,
            tool_args=tool_args, allow_tools=allow_tools,
            span_timeout_s=span_timeout_s,
        )
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        entry: dict[str, Any] = {
            "span_id": span.id, "kind": span.kind, "title": span.title,
            "verdict": verdict, "reason": reason,
        }
        if recorded is not None:
            text, cut = _truncate(recorded, TOOL_CONTENT_CHARS)
            entry["recorded"] = text
            entry["recorded_truncated"] = cut
        if replayed is not None:
            text, cut = _truncate(replayed, TOOL_CONTENT_CHARS)
            entry["replayed"] = text
            entry["replayed_truncated"] = cut
        results.append(entry)
    return ReplayResult(
        conversation_id=trace.conversation_id,
        verdicts=verdicts,
        results=results,
        truncated=truncated,
        spans_considered=len(considered),
        spans_total=len(spans),
        seconds=time.perf_counter() - started,
    )


def _replay_span(
    span: Span,
    prompt: list[dict[str, str]],
    *,
    model_runner: Callable[[list[dict[str, str]]], str] | None,
    tool_runner: Callable[[str, dict[str, Any]], Any] | None,
    tool_args: dict[str, dict[str, Any]],
    allow_tools: bool,
    span_timeout_s: float,
) -> tuple[str, str, str | None, str | None]:
    """One span's replay. Returns (verdict, reason, recorded, replayed)."""
    recorded = span.output if span.kind == "message" else None
    if span.kind == "message":
        if span.detail.get("role") == "user":
            return ("skipped", "input, not re-run", recorded, None)
        if model_runner is None:
            return ("skipped", "no model runner attached", recorded, None)
        value, error = _run_with_timeout(
            lambda: model_runner(prompt), span_timeout_s)
        if error is not None:
            return ("diverged", error, recorded, None)
        new_text = _text(value)
        if not new_text.strip():
            return ("diverged", "model returned empty text", recorded, "")
        if new_text.strip() == (recorded or "").strip():
            return ("same", "replayed text matches the recorded reply",
                    recorded, new_text)
        return ("changed", "replayed text differs from the recorded reply",
                recorded, new_text)
    if span.kind == "tool":
        tool = str(span.detail.get("tool") or "?")
        capability = str(span.detail.get("capability") or "")
        if not allow_tools:
            return ("skipped",
                    "tool replay needs --allow-tools (default: model spans only)",
                    None, None)
        if span.id not in tool_args:
            return ("skipped",
                    "args recorded as digest only — supply --tool-args to re-run",
                    None, None)
        if capability not in READ_ONLY_CAPABILITIES:
            return ("skipped",
                    f"would write (capability {capability or 'unknown'}) — "
                    "never executed",
                    None, None)
        if tool_runner is None:
            return ("skipped", "no tool runner attached", None, None)
        args = tool_args[span.id]
        value, error = _run_with_timeout(
            lambda: tool_runner(tool, args), span_timeout_s)
        if error is not None:
            return ("diverged", error, None, None)
        # The journal never stores a tool result, so there is no
        # recorded output to compare against — a clean re-run reports
        # `replayed`, not `same`.
        try:
            replayed = json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            replayed = _text(value)
        return ("replayed", "re-ran cleanly; no recorded output to compare",
                None, replayed)
    return ("skipped", "observation only — nothing to re-run", None, None)


# ── HTML view ──────────────────────────────────────────────────────────


def trace_page(trace: Trace) -> str:
    """One trace → a standalone HTML page (nested span tree).

    Self-contained: inline CSS only, no scripts, no external resources.
    Every journal string is escaped, so a ``</script>`` inside a recorded
    message cannot break out of anything — there is no script to break
    out of.
    """
    esc = _html.escape
    stats = trace.stats
    lines: list[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang=\"en\">")
    lines.append("<head>")
    lines.append("<meta charset=\"utf-8\">")
    lines.append(f"<meta http-equiv=\"Content-Security-Policy\" "
                 f"content=\"{TRACE_CSP}\">")
    lines.append(f"<title>trace {esc(trace.conversation_id)}</title>")
    lines.append("<style>body{font-family:system-ui,sans-serif;"
                 "max-width:900px;margin:2em auto;padding:0 1em;"
                 "color:#1a1a1a}h1{font-size:1.3em}h2{font-size:1.1em;"
                 "margin-top:1.5em}.meta{color:#555;font-size:.9em}"
                 ".span{border:1px solid #ddd;border-radius:6px;"
                 "padding:.6em .8em;margin:.5em 0}"
                 ".kind{display:inline-block;font-size:.75em;"
                 "font-weight:700;padding:.1em .5em;border-radius:4px;"
                 "background:#eee;margin-right:.5em}"
                 ".kind-message{background:#dbeafe}.kind-tool{background:#fef3c7}"
                 ".kind-audit{background:#e5e7eb}.kind-cognition{background:#ede9fe}"
                 ".kind-skill{background:#dcfce7}"
                 ".ok{color:#166534}.err{color:#991b1b}.muted{color:#666;"
                 "font-size:.85em}pre{white-space:pre-wrap;word-break:break-word;"
                 "background:#f8f8f8;padding:.5em;border-radius:4px;"
                 "font-size:.9em}.missing{color:#92400e;font-size:.85em}"
                 ".turn{border-left:3px solid #2563eb;padding-left:.8em;"
                 "margin:1.2em 0}</style>")
    lines.append("</head><body>")
    lines.append(f"<h1>trace: {esc(trace.title or trace.conversation_id)}</h1>")
    lines.append(
        f"<p class=\"meta\">conversation {esc(trace.conversation_id)} · "
        f"{esc(trace.channel or '?')} · agent {esc(trace.agent or '?')} · "
        f"{stats.get('spans', 0)} spans · "
        f"{stats.get('turns', 0)} turns · "
        f"{stats.get('total_tokens', 0)} tokens · "
        f"{stats.get('total_tool_ms', 0)} ms of tool time</p>")
    lines.append(
        "<p class=\"muted\">Tool and audit spans are joined by time window "
        "— the journal records no request id on those rows. Message spans "
        "are joined exactly on the conversation. Fields the journal never "
        "records are listed as missing, never filled in.</p>")
    by_id = {s.id: s for s in trace.spans}
    for turn in trace.turns:
        lines.append("<div class=\"turn\">")
        lines.append(f"<h2>turn {turn.index + 1}</h2>")
        if turn.user_text:
            text, cut = _truncate(turn.user_text, PAGE_CONTENT_CHARS)
            lines.append(f"<pre>{esc(text)}</pre>")
            if cut:
                lines.append("<p class=\"muted\">(user text truncated)</p>")
        for span_id in turn.span_ids:
            span = by_id.get(span_id)
            if span is None:
                continue
            lines.append("<div class=\"span\">")
            lines.append(
                f"<span class=\"kind kind-{esc(span.kind)}\">"
                f"{esc(span.kind)}</span>"
                f"<strong>{esc(span.title)}</strong> "
                f"<span class=\"muted\">{time.strftime('%H:%M:%S', time.gmtime(span.ts))}"
                f"{f' · {span.duration_ms:g} ms' if span.duration_ms is not None else ''}"
                f"{f' · {esc(span.status)}' if span.status else ''}"
                f"{' · time-window join' if span.join == 'time-window' else ''}"
                f"</span>")
            body = span.output if span.output is not None else span.input
            if body:
                text, cut = _truncate(body, PAGE_CONTENT_CHARS)
                lines.append(f"<pre>{esc(text)}</pre>")
                if cut:
                    lines.append("<p class=\"muted\">(truncated)</p>")
            if span.detail:
                lines.append(
                    "<p class=\"muted\">"
                    + esc(json.dumps(span.detail, sort_keys=True, default=str)[:800])
                    + "</p>")
            if span.missing:
                lines.append(
                    "<p class=\"missing\">missing from the journal: "
                    + esc(", ".join(span.missing))
                    + (" — tool arguments are recorded as a digest only"
                       if span.kind == "tool" else "")
                    + "</p>")
            lines.append("</div>")
        lines.append("</div>")
    lines.append("</body></html>")
    return "\n".join(lines)


def write_trace_page(html_text: str, name: str, *,
                     out_dir: str | Path | None = None) -> Path:
    """Write a generated trace page to disk and return its path.

    ``name`` is a bare page name (letters, digits, ``-``/``_`` — no
    separators, no extension); the file lands as ``<name>.html`` under
    ``~/.nomorals/traces/`` unless ``out_dir`` is given.
    """
    if not _PAGE_NAME_RE.match(name):
        raise ValueError(
            "page name must be a bare name (letters, digits, -/_): "
            + repr(name))
    base = (Path(out_dir) if out_dir is not None
            else Path.home() / ".nomorals" / "traces")
    base.mkdir(parents=True, exist_ok=True)
    path = base / (name + ".html")
    path.write_text(html_text, encoding="utf-8")
    return path


# ── registry tools ─────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """Attach the trace tools to a registry."""
    from ..core.policy import Capability as _Cap

    def _db() -> Any:
        context = getattr(registry, "context", None)
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            raise ToolError("no database attached to this context")
        return db

    @registry.register(
        "trace_list",
        description="List recent journal traces (one trace = one "
                    "conversation): id, title, channel, message count.",
        capability=_Cap.DB_READ,
    )
    def trace_list(limit: int = DEFAULT_LIST_LIMIT, channel: str = "",
                   agent: str = "") -> dict[str, Any]:
        builder = TraceBuilder(_db())
        traces = builder.list_traces(limit=int(limit), channel=channel,
                                     agent=agent)
        return {"count": len(traces), "traces": traces}

    @registry.register(
        "trace_show",
        description="Show one journal trace: the whole chain of spans "
                    "(messages, tool calls, audit, cognition, skill uses) "
                    "for a conversation, in time order, with turns.",
        capability=_Cap.DB_READ,
    )
    def trace_show(conversation_id: str) -> dict[str, Any]:
        builder = TraceBuilder(_db())
        return builder.build(conversation_id).to_dict()

    @registry.register(
        "trace_replay",
        description="Replay one journal trace against the current code: "
                    "model spans re-run through the live model and are "
                    "diffed (same/changed/diverged); tool spans re-run only "
                    "when args are supplied, the tool is read-only, and "
                    "allow_tools is set — a tool that would write is never "
                    "executed.",
        capability=_Cap.MODEL_CALL,
    )
    def trace_replay(conversation_id: str,
                     max_spans: int = DEFAULT_MAX_SPANS,
                     span_timeout_s: float = DEFAULT_SPAN_TIMEOUT_S,
                     allow_tools: bool = False,
                     tool_args: dict[str, dict[str, Any]] | None = None
                     ) -> dict[str, Any]:
        builder = TraceBuilder(_db())
        trace = builder.build(conversation_id)
        return replay_trace(
            trace, tool_args=tool_args, allow_tools=bool(allow_tools),
            max_spans=int(max_spans), span_timeout_s=float(span_timeout_s),
        ).to_dict()
