"""``nm trace`` — journal trace/replay view from the CLI.

One trace is one conversation from the journal, reconstructed as the
whole chain of spans (messages, tool calls, audit, cognition, skill
uses) in time order, split into turns. Replay re-runs the recorded
inputs through the current code and diffs the outputs. See
nomorals/tools/trace_view.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...tools.trace_view import (
    CLI_CONTENT_CHARS,
    TraceBuilder,
    TraceNotFound,
    replay_trace,
    set_model_runner,
    trace_page,
    write_trace_page,
)
from ..emit import _emit


def _db_path(context: Any) -> str:
    settings = getattr(context, "settings", None)
    path = getattr(settings, "db_path", None) if settings is not None else None
    if path:
        return str(path)
    from ...core.config import load_settings

    return str(load_settings().db_path)


def _open_db(context: Any) -> Any:
    from ...storage.db import Database

    return Database(_db_path(context))


def _brain_model_runner(context: Any, timeout_s: float):
    """The live model as a replay runner (deterministic: temperature 0)."""
    from ...llm.base import Message, SamplingParams
    from ...llm.brain import brain_for

    brain = brain_for(context)

    def run(prompt: list[dict[str, str]]) -> str:
        messages: list[Message] = []
        for item in prompt:
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            if role == "system":
                messages.append(Message.system(content))
            elif role == "assistant":
                messages.append(Message.assistant(content))
            else:
                messages.append(Message.user(content))
        response = brain.chat(
            messages,
            SamplingParams(temperature=0.0, max_tokens=1024),
            task_kind="trace-replay",
            timeout_s=float(timeout_s),
        )
        if not response.ok or not (response.text or "").strip():
            raise RuntimeError(
                f"brain replay failed ({response.failure_class or 'unknown'}): "
                f"{response.error or 'no error detail'}")
        return str(response.text)

    return run


def _load_tool_args(raw: str) -> dict[str, dict[str, Any]]:
    """--tool-args: a JSON object, or a path to a JSON file holding one."""
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        data = json.loads(text)
    else:
        data = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--tool-args must be a JSON object keyed by span id")
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"--tool-args[{key!r}] must be an object of arguments")
        out[str(key)] = value
    return out


def _cmd_trace(args: argparse.Namespace, context: Any) -> int:
    """Route `nm trace` to list / show / replay."""
    action = getattr(args, "trace_action", "") or ""
    try:
        if action == "list":
            return _trace_list(args, context)
        if action == "show":
            return _trace_show(args, context)
        if action == "replay":
            return _trace_replay(args, context)
    except TraceNotFound as exc:
        print(f"trace: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"trace: {exc}", file=sys.stderr)
        return 2
    print("usage: nm trace {list|show|replay} ...", file=sys.stderr)
    return 2


def _trace_list(args: argparse.Namespace, context: Any) -> int:
    db = _open_db(context)
    try:
        builder = TraceBuilder(db)
        traces = builder.list_traces(
            limit=int(getattr(args, "limit", 10) or 10),
            channel=str(getattr(args, "channel", "") or ""),
            agent=str(getattr(args, "agent", "") or ""),
        )
    finally:
        db.close()
    return _emit_list(args, traces)


def _emit_list(args: argparse.Namespace,
               traces: list[dict[str, Any]]) -> int:
    lines = ["traces — one per conversation, newest first"]
    for row in traces:
        lines.append(
            f"• {row['conversation_id']} — {row['title'] or '(untitled)'} "
            f"[{row['channel'] or '?'}] {row['messages']} messages")
    _emit(args, {"count": len(traces), "traces": traces},
          "\n".join(lines) if traces else "no traces in the journal yet")
    return 0


def _trace_show(args: argparse.Namespace, context: Any) -> int:
    conv_id = str(getattr(args, "conversation_id", "") or "")
    if not conv_id:
        print("nm trace show: a conversation id is required", file=sys.stderr)
        return 2
    db = _open_db(context)
    try:
        trace = TraceBuilder(db).build(conv_id)
    finally:
        db.close()
    html_arg = getattr(args, "html", None)
    if html_arg is not None:
        path = write_trace_page(
            trace_page(trace), conv_id.replace(":", "_").replace("/", "_"),
            out_dir=html_arg if html_arg else None)
        print(f"trace page: {path}")
    return _emit_show(args, trace)


def _emit_show(args: argparse.Namespace, trace: Any) -> int:
    stats = trace.stats
    lines = [
        f"trace {trace.conversation_id} — {trace.title or '(untitled)'}",
        f"{trace.channel or '?'} · agent {trace.agent or '?'} · "
        f"{stats.get('spans', 0)} spans · {stats.get('turns', 0)} turns · "
        f"{stats.get('total_tokens', 0)} tokens · "
        f"{stats.get('total_tool_ms', 0)} ms of tool time",
    ]
    by_id = {s.id: s for s in trace.spans}
    for turn in trace.turns:
        lines.append(f"\n— turn {turn.index + 1} —")
        for span_id in turn.span_ids:
            span = by_id.get(span_id)
            if span is None:
                continue
            head = f"[{span.kind}] {span.title}"
            if span.duration_ms is not None:
                head += f" ({span.duration_ms:g} ms)"
            if span.status:
                head += f" [{span.status}]"
            if span.join == "time-window":
                head += " (time-window join)"
            lines.append(head)
            body = span.output if span.output is not None else span.input
            if body:
                text = body if len(body) <= CLI_CONTENT_CHARS else \
                    body[:CLI_CONTENT_CHARS] + "…"
                lines.append("    " + text.replace("\n", "\n    "))
            if span.missing:
                lines.append("    missing from the journal: "
                             + ", ".join(span.missing))
    _emit(args, trace.to_dict(), "\n".join(lines))
    return 0


def _trace_replay(args: argparse.Namespace, context: Any) -> int:
    conv_id = str(getattr(args, "conversation_id", "") or "")
    if not conv_id:
        print("nm trace replay: a conversation id is required",
              file=sys.stderr)
        return 2
    timeout_s = float(getattr(args, "timeout", 30.0) or 30.0)
    tool_args = _load_tool_args(str(getattr(args, "tool_args", "") or ""))
    allow_tools = bool(getattr(args, "allow_tools", False))
    # The brain is the model runner — unless a runner is already
    # attached (tests attach a fake), in which case it stands.
    from ...tools.trace_view import get_model_runner

    if get_model_runner() is None:
        set_model_runner(_brain_model_runner(context, timeout_s))
    db = _open_db(context)
    try:
        trace = TraceBuilder(db).build(conv_id)
    finally:
        db.close()
    result = replay_trace(
        trace, tool_args=tool_args, allow_tools=allow_tools,
        max_spans=int(getattr(args, "max_spans", 50) or 50),
        span_timeout_s=timeout_s,
    )
    return _emit_replay(args, result)


def _emit_replay(args: argparse.Namespace, result: Any) -> int:
    lines = [f"replay {result.conversation_id} — "
             f"{result.spans_considered}/{result.spans_total} spans"
             + (" (truncated)" if result.truncated else "")
             + f" in {result.seconds:.1f}s"]
    for verdict, count in sorted(result.verdicts.items()):
        lines.append(f"  {verdict}: {count}")
    for entry in result.results:
        line = f"• [{entry['kind']}] {entry['title']}: {entry['verdict']}"
        if entry.get("reason"):
            line += f" — {entry['reason']}"
        lines.append(line)
    _emit(args, result.to_dict(), "\n".join(lines))
    return 0
