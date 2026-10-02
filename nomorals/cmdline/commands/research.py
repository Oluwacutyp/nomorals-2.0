"""``nm research-loop`` / ``kg`` / ``cookies`` / ``structure``."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any
from ..emit import _emit



def _cmd_research_loop(args: argparse.Namespace, context: Any) -> int:
    """`nm research-loop <action>` — the always-on research loop CLI.

    Mirrors the ``research_loop`` tool's actions (tick|run|status|ensure|
    enable|disable|topics|set_topics) against the real loop module, so the
    console and the tool/agent surface stay in lockstep.
    """
    from ...agents import research_loop as rlmod

    action = (getattr(args, "action", "status") or "status").strip().lower()

    if action == "status":
        data = rlmod.status(context)
        job = data.get("job") or {}
        gates = data.get("gates") or {}
        last = data.get("last_run") or {}
        last_line = "never"
        if last:
            ts = last.get("started_at", 0) or 0
            last_line = (f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))} "
                         f"({'ok' if last.get('ok') else 'deferred: ' + str(last.get('skipped_reason') or last.get('error') or '')})"
                         if ts else "recorded")
        lines = [
            "research loop:",
            f"  job: {'scheduled' if job.get('scheduled') else 'not scheduled'}"
            + (f", every {job.get('interval_hours')}h" if job.get("scheduled") else "")
            + ("" if job.get("enabled", True) else " (disabled)"),
            f"  gates: feature={'on' if gates.get('feature_research') else 'off'}, "
            f"proactive={'on' if gates.get('proactive_master') else 'off'}, "
            f"quiet_hours={'yes' if gates.get('quiet_hours') else 'no'}",
            f"  last run: {last_line}",
            f"  pending proposals: {data.get('pending_proposals', 0)}",
            f"  signals: {rlmod.signals_summary(data.get('signals') or {})}",
            f"  topics: {', '.join(data.get('topics') or []) or 'none set'}",
        ]
        _emit(args, data, "\n".join(lines))
        return 0

    if action == "tick":
        max_topics = int(getattr(args, "max_topics", 0) or 0)
        loop = rlmod.ResearchLoop(
            context, max_topics=max_topics or rlmod._MAX_TOPICS_PER_TICK)
        result = loop.tick()
        ok = bool(result.get("ok"))
        topics = ", ".join(result.get("topics") or [])
        text = (f"tick {'ok' if ok else 'deferred'}"
                + (f" — {result.get('skipped_reason')}" if result.get("skipped_reason") else "")
                + (f" — topics: {topics}" if topics else "")
                + f" — findings {result.get('findings_count', 0)}, "
                  f"proposals {result.get('proposals_created', 0)}")
        _emit(args, result, text)
        return 0

    if action == "run":
        topic = (getattr(args, "topic", "") or "").strip()
        if not topic:
            print("research-loop run needs a topic — nm research-loop run \"<topic>\"",
                  file=sys.stderr)
            return 2
        try:
            result = rlmod.run_topic(context, topic)
        except Exception as exc:  # noqa: BLE001
            print(f"research-loop run: {exc}", file=sys.stderr)
            return 1
        _emit(args, result,
              f"run {'deferred' if result.get('deferred') else 'ok'}: {topic}"
              + (f" — {result.get('skipped_reason')}" if result.get("skipped_reason") else "")
              + f" — proposals {result.get('proposals_created', 0)}")
        return 0

    if action == "ensure":
        result = rlmod.ensure_research_job(context)
        _emit(args, result,
              f"research loop job: {'already scheduled' if result.get('already_scheduled') else 'scheduled'} "
              f"(every {result.get('interval_hours')}h)")
        return 0

    if action in ("enable", "disable"):
        from ...agents.scheduler import Scheduler

        sched = Scheduler(context)
        jobs = [j for j in sched.list_jobs()
                if j.get("name") == rlmod.RESEARCH_LOOP_JOB]
        if not jobs:
            info = rlmod.ensure_research_job(context)
            jobs = [j for j in sched.list_jobs()
                    if j.get("name") == rlmod.RESEARCH_LOOP_JOB]
            if not jobs:
                print(f"research-loop {action}: job registration failed",
                      file=sys.stderr)
                return 1
        row = sched.set_enabled(jobs[0]["id"], action == "enable")
        _emit(args, {"enabled": action == "enable", "job": row},
              f"research loop job {'enabled' if action == 'enable' else 'disabled'}")
        return 0

    if action == "topics":
        topics = rlmod.owner_topics(context)
        _emit(args, {"topics": topics},
              "research topics:\n" + "\n".join(f"  · {t}" for t in topics)
              if topics else "no research topics set — nm research-loop set_topics --topics \"a, b\"")
        return 0

    if action == "set_topics":
        raw = (getattr(args, "topics", "") or "").strip()
        topics = [t.strip() for t in raw.split(",") if t.strip()]
        if not topics:
            print("research-loop set_topics needs --topics \"a, b, c\"",
                  file=sys.stderr)
            return 2
        saved = rlmod.set_owner_topics(context, topics)
        _emit(args, {"topics": saved},
              f"research topics set ({len(saved)}): " + ", ".join(saved))
        return 0

    print(f"research-loop: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_kg(args: argparse.Namespace, context: Any) -> int:
    """Knowledge graph maintenance from the operator's seat."""
    from ...agents.kg import KnowledgeGraph

    g = KnowledgeGraph(context.db)
    action = getattr(args, "action", "stats") or "stats"
    try:
        limit = int(getattr(args, "limit", 10) or 10)
    except (TypeError, ValueError):
        limit = 10
    if action == "stats":
        st = g.stats()
        _emit(args, st, f"kg: {st['nodes']} nodes, {st['edges']} edges "
              f"({len(st.get('by_type', {}))} types)")
        return 0
    if action == "consolidate":
        out = g.consolidate()
        _emit(args, out, f"kg: consolidated — merged "
              f"{out.get('merged', 0)} duplicate node(s)")
        return 0
    if action == "communities":
        comms = g.communities(limit=limit)
        lines = [f"kg: {len(comms)} communities"]
        for i, c in enumerate(comms, 1):
            lines.append(f"  {i}. size {c.get('size', 0)} — "
                         f"{len(c.get('members') or [])} member ids, "
                         f"types {sorted((c.get('types') or {}).keys())}")
        _emit(args, comms, "\n".join(lines))
        return 0
    if action == "top":
        rows = g.top(n=limit)
        lines = [f"kg: top {len(rows)} hubs by degree"]
        lines.extend(f"  {r.get('label', '')} — {r.get('degree', 0)} "
                     f"({r.get('type', 'entity')})" for r in rows)
        _emit(args, rows, "\n".join(lines))
        return 0
    if action == "decay":
        out = g.decay()
        _emit(args, out, f"kg: decay — lowered confidence on "
              f"{out.get('updated', 0)} node(s); "
              f"{out.get('stale', 0)} now stale")
        return 0
    print(f"kg: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_cookies(args: argparse.Namespace, context: Any) -> int:
    """Cookie lab CLI: parse/classify a Set-Cookie blob, or ingest it
    into the knowledge graph."""
    from ...core.cookies import CookieLab

    lab = CookieLab()
    text = " ".join(getattr(args, "text", []) or []).strip()
    if not text:
        print('usage: nm cookies "<cookie text>"  |  '
              'nm cookies ingest "<cookie text>"', file=sys.stderr)
        return 2
    action = "parse"
    if text.lower().startswith(("ingest ", "ingest\n")):
        action = "ingest"
        text = text.split(None, 1)[1].strip()
    if action == "ingest":
        from nomorals.agents.kg import KnowledgeGraph
        out = lab.ingest(context, text,
                        source=getattr(args, "source", "") or "cookies",
                        graph=KnowledgeGraph(context.db))
        _emit(args, out, f"cookies: ingested: {out.get('nodes', 0)} "
              f"graph node(s) — services: "
              f"{', '.join(out.get('services') or []) or 'none'}")
        return 0
    rep = lab.report(text)
    sec = rep.get("security", {})
    lines = [f"cookies: {rep['count']} parsed — "
             f"services: {', '.join(rep.get('services') or []) or 'none'}"]
    for c in rep["cookies"]:
        flags = ",".join(f if v in ("", True, None) else f"{f}={v}"
                         for f, v in (c.get("flags") or {}).items())
        via = f" [{c['decode_via']}]" if c.get("decode_via") else ""
        lines.append(f"  {c['name']} — {c.get('kind', 'unknown')} "
                     f"({c.get('service') or 'no service'})"
                     f"{(' flags: ' + flags) if flags else ''}{via}")
    lines.append(
        f"security: {len(sec.get('plaintext_auth') or [])} plaintext auth "
        f"cookie(s), {sec.get('with_httponly', 0)} HttpOnly, "
        f"{sec.get('with_secure', 0)} Secure")
    if sec.get("plaintext_auth"):
        lines.append("  plaintext: " + ", ".join(sec["plaintext_auth"]))
    _emit(args, rep, "\n".join(lines))
    return 0


def _cmd_structure(args: argparse.Namespace, context: Any) -> int:
    """Prompt-architect CLI: turn a raw objective into a structured brief."""
    from ...agents.structuring import structure_text

    objective = (getattr(args, "objective", "") or "").strip()
    if not objective:
        print('usage: nm structure "<objective text>"', file=sys.stderr)
        return 2
    brief = structure_text(
        context, objective,
        for_=getattr(args, "for_", "mission") or "mission",
        polish=bool(getattr(args, "polish", False)))
    _emit(args, brief, brief.get("brief") or "")
    return 0
