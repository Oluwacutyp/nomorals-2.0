"""``nm memory`` — memory records and actions."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_memory(args: argparse.Namespace, context: Any) -> int:
    memory = context.memory
    # Prompt 11 subcommands take precedence over the legacy flat flags
    if getattr(args, "memory_action", None):
        return _cmd_memory_action(args, context)
    if args.remember:
        record_id = memory.remember(args.remember, source="cli")
        _emit(args, {"id": record_id}, f"remembered as {record_id}")
        return 0
    if args.consolidate:
        report = memory.consolidate()
        _emit(args, report, f"consolidated: {report}")
        return 0
    if args.stats or not args.query:
        stats = memory.stats_snapshot()
        _emit(args, stats, json.dumps(stats, indent=2, default=str))
        return 0
    result = memory.recall(args.query, limit=args.limit)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0
    if not result.records:
        print("no matches")
        return 0
    for record in result.records:
        print(f"{record.score:.3f} [{record.kind}] {record.content[:160]}")
    return 0


def _cmd_memory_action(args: argparse.Namespace, context: Any) -> int:
    """Prompt 11 subcommands: show/list/edit/forget/forget-kind/export/private."""
    from ...memory.persona import (MemoryCurator, PeopleGraph, PersonaGuide,
                                 UserModel, export_model)

    memory = context.memory
    action = args.memory_action

    if action == "show":
        model = UserModel.rebuild(memory)
        guide = PersonaGuide.build(model)
        if args.json:
            print(json.dumps({"model": model.to_dict(),
                              "guide": guide.to_dict()},
                             indent=2, default=str))
            return 0
        print("═══ how Devon sees you ═══")
        if model.identity:
            print("\n— identity —")
            for key, attr in model.identity.items():
                print(f"  {key}: {attr.value} "
                      f"(confidence {attr.confidence:.2f})")
        if model.preferences:
            print("\n— preferences —")
            for pref in model.preferences[:10]:
                flag = "" if pref.actionable else " [low confidence]"
                print(f"  • {pref.value[:100]} "
                      f"(confidence {pref.confidence:.2f}){flag}")
        if model.interests:
            print("\n— interests —")
            for interest in model.interests[:8]:
                print(f"  • {interest.topic} "
                      f"(heat {interest.heat:.2f}, {interest.mentions}×)")
        if model.routines:
            print("\n— routines —")
            for routine in model.routines[:6]:
                print(f"  • {routine.description} "
                      f"[{routine.status}, {routine.supporting_episodes}×"
                      + (f", {routine.time_hint}" if routine.time_hint else "")
                      + "]")
        if model.goals_in_flight:
            print("\n— goals in flight —")
            for goal in model.goals_in_flight[:6]:
                print(f"  • {goal.value[:100]}")
        people = PeopleGraph.build(memory).people
        if people:
            print("\n— people —")
            for key in sorted(people)[:10]:
                entry = people[key]
                print(f"  • {entry.name}"
                      + (f" ({entry.role})" if entry.role else "")
                      + f" — mentioned {entry.mention_count}×")
        print("\n— persona guide (injected into Devon's context) —")
        for line in guide.lines:
            print(f"  {line}")
        return 0

    if action == "list":
        result = memory.recall(args.query or "", limit=args.limit,
                               kind=args.kind or "",
                               include_private=args.include_private)
        if args.json:
            print(json.dumps(
                [{"id": r.id, "kind": r.kind, "content": r.content,
                  "private": bool((r.metadata or {}).get("private")),
                  "created_at": r.created_at} for r in result.records],
                indent=2, default=str))
            return 0
        if not result.records:
            print("no matches")
            return 0
        for record in result.records:
            priv = " [private]" if (record.metadata or {}).get("private") \
                else ""
            print(f"{record.id} [{record.kind}]{priv} "
                  f"{record.content[:140]}")
        return 0

    if action == "edit":
        n = memory.update(args.record_id, content=args.text)
        print("updated" if n else "not found")
        return 0 if n else 1

    if action == "forget":
        curator = MemoryCurator(memory)
        curator._archive(args.record_id, reason="cli-forget")
        n = memory.forget(args.record_id)
        print("forgotten (archived, 30-day undo via curator)"
              if n else "not found")
        return 0 if n else 1

    if action == "forget-kind":
        if not args.yes:
            print(f"this will forget ALL '{args.kind}' records. "
                  f"re-run with --yes to confirm.", file=sys.stderr)
            return 2
        result = memory.recall("", limit=5000, kind=args.kind,
                               include_private=True)
        curator = MemoryCurator(memory)
        n = 0
        for record in result.records:
            curator._archive(record.id, reason="cli-forget-kind")
            n += memory.forget(record.id)
        print(f"forgotten {n} records of kind '{args.kind}' (archived)")
        return 0

    if action == "private":
        n = memory.mark_private(args.record_id)
        print("marked private — excluded from proactive recall and training"
              if n else "not found")
        return 0 if n else 1

    if action == "public":
        n = memory.mark_public(args.record_id)
        print("private flag cleared" if n else "not found")
        return 0 if n else 1

    if action == "export":
        dump = export_model(memory)
        print(json.dumps(dump, indent=2, default=str))
        return 0

    if action == "rebuild":
        model = UserModel.rebuild(memory)
        print(f"rebuilt user model from {model.record_count} records "
              f"({len(model.preferences)} preferences, "
              f"{len(model.interests)} interests, "
              f"{len(model.routines)} routines)")
        return 0

    if action == "curate":
        report = MemoryCurator(memory).scheduled_pass()
        _emit(args, report, f"curation: {report}")
        return 0

    if action == "consolidate":
        from ...memory.cadence import consolidate_now
        report = consolidate_now(memory)
        _emit(args, report,
              f"consolidated (additive): {report.get('summaries', 0)} facts "
              f"from {report.get('episodes', 0)} episodes "
              f"in {report.get('seconds', 0)}s")
        return 0 if report.get("ok") else 1

    if action == "health":
        report = memory.health()
        _emit(args, report, json.dumps(report, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if action == "repair":
        report = memory.repair_embeddings(dry_run=args.dry_run)
        _emit(args, report, json.dumps(report, indent=2, default=str))
        return 0

    if action == "timeline":
        from ...memory.deep_recall import timeline
        items = timeline(memory, args.topic, limit=args.limit)
        if args.json:
            print(json.dumps([r.to_dict() for r in items],
                             indent=2, default=str))
            return 0
        if not items:
            print(f"nothing on '{args.topic}' yet")
            return 0
        import datetime as _dt
        for r in items:
            when = _dt.datetime.fromtimestamp(
                r.created_at).strftime("%Y-%m-%d")
            flag = " (superseded)" if (r.metadata or {}).get(
                "superseded_by") else ""
            print(f"{when} [{r.kind}]{flag} {r.content[:140]}")
        return 0

    if action == "deep":
        from ...memory.deep_recall import recall_deep
        result = recall_deep(memory, args.query, limit=args.limit,
                             scope=args.scope)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
            return 0
        print(f"deep recall ({result.hops} hops"
              + (f", via: {', '.join(result.expansion_terms)}"
                 if result.expansion_terms else "") + ")")
        for r in result.records:
            print(f"{r.score:.3f} [{r.kind}] {r.content[:140]}")
        return 0

    if action == "contradictions":
        from ...memory.contradictions import detect_for
        targets: list[tuple[str, list]] = []
        if args.record_id:
            targets.append((args.record_id,
                            detect_for(memory, args.record_id)))
        else:
            seen: set[str] = set()
            for kind in ("fact", "preference"):
                for r in memory.recall("", limit=30,
                                       kind=kind).records:
                    if r.id in seen:
                        continue
                    seen.add(r.id)
                    hits = detect_for(memory, r.id)
                    if hits:
                        targets.append((r.id, hits))
                    if len(targets) >= 8:
                        break
        payload = [{"record_id": rid,
                    "contradictions": [c.to_dict() for c in hits]}
                   for rid, hits in targets if hits]
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        if not payload:
            print("no contradictions detected")
            return 0
        for entry in payload:
            for c in entry["contradictions"]:
                print(f"[{c['strategy']} {c['confidence']:.2f}] {c['note']}")
        return 0

    if action == "scopes":
        from ...memory.scopes import scopes_summary
        counts = scopes_summary(memory)
        _emit(args, counts,
              "\n".join(f"{name}: {n}" for name, n in counts.items())
              or "no memory spaces yet — everything is global")
        return 0

    if action == "schedule":
        from ...memory.cadence import status
        report = status(memory)
        _emit(args, report, json.dumps(report, indent=2, default=str))
        return 0

    if action == "backup":
        from ...memory.backup import backup_to
        import tempfile as _tf
        dest = args.dest or _tf.mkdtemp(prefix="nm-mem-backup-")
        report = backup_to(memory, dest, label="cli")
        _emit(args, report,
              f"backup {'ok' if report.get('ok') else 'FAILED'}: "
              f"{report.get('backup_dir', report.get('error'))}")
        return 0 if report.get("ok") else 1

    if action == "import":
        from ...memory.backup import import_records
        try:
            with open(args.path, encoding="utf-8") as fh:
                bundle = json.load(fh)
        except OSError as exc:
            print(f"cannot read {args.path}: {exc}", file=sys.stderr)
            return 1
        records = bundle.get("records", bundle) if isinstance(
            bundle, dict) else bundle
        report = import_records(memory, records,
                                dry_run=args.dry_run, source="cli-import")
        _emit(args, report,
              f"import: {report.get('imported', 0)} imported, "
              f"{report.get('skipped_duplicate', 0)} duplicates skipped, "
              f"{report.get('failed', 0)} failed")
        return 0 if report.get("ok") else 1

    print(f"unknown memory action: {action}", file=sys.stderr)
    return 2
