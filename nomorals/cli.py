"""Command-line interface: ``nm`` / ``python -m nomorals``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .compat import feature_report, report_as_text
from .core.logging_setup import setup_logging
from .version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nm", description="NoMorals Core — self-hosted multi-agent AI substrate."
    )
    parser.add_argument("--version", action="version", version=f"nomorals {__version__}")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="show environment capabilities and health")
    sub.add_parser("config", help="print the effective configuration")

    models = sub.add_parser("models", help="inspect the model registry and catalog")
    models.add_argument("--catalog", action="store_true", help="list the curated catalog")
    models.add_argument("--search", default="")
    models.add_argument("--kind", default="")
    models.add_argument("--max-params", type=int, default=0)
    models.add_argument("--activate", default="", help="activate a registered model by name")

    tools = sub.add_parser("tools", help="list available tools and their capabilities")
    tools.add_argument("--schema", action="store_true", help="emit full JSON schemas")

    memory = sub.add_parser("memory", help="inspect and query memory")
    memory.add_argument("--query", default="")
    memory.add_argument("--limit", type=int, default=8)
    memory.add_argument("--consolidate", action="store_true")
    memory.add_argument("--stats", action="store_true")
    memory.add_argument("--remember", default="")

    agent = sub.add_parser("run", help="run a goal through the orchestrator")
    agent.add_argument("goal", nargs="+")
    agent.add_argument("--max-steps", type=int, default=8)
    agent.add_argument("--no-reflect", action="store_true")

    ask = sub.add_parser("ask", help="single-turn chat with the active model")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--system", default="")
    ask.add_argument("--max-tokens", type=int, default=1024)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--model", default="", help="hot-swap to this provider for this call")

    backup = sub.add_parser("backup", help="create, list, verify, or restore backups")
    backup.add_argument("--create", action="store_true")
    backup.add_argument("--list", action="store_true")
    backup.add_argument("--verify", action="store_true")
    backup.add_argument("--restore", default="")
    backup.add_argument("--push", action="store_true", help="push the latest backup to git")

    sub.add_parser("serve", help="start the HTTP API server")
    queue = sub.add_parser("queue", help="inspect the durable work queue")
    queue.add_argument("--topic", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging(args.log_level, force=True)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI is the last line of defence
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    else:
        print(text)


def _dispatch(args: argparse.Namespace) -> int:
    from .core.config import load_settings

    settings = load_settings(args.config)

    if args.command == "doctor":
        return _cmd_doctor(args, settings)
    if args.command == "config":
        _emit(args, settings.to_dict(), _render_config(settings))
        return 0

    from .agents.context import build_context

    with build_context(settings) as context:
        if args.command == "models":
            return _cmd_models(args, context)
        if args.command == "tools":
            return _cmd_tools(args, context)
        if args.command == "memory":
            return _cmd_memory(args, context)
        if args.command == "run":
            return _cmd_run(args, context)
        if args.command == "ask":
            return _cmd_ask(args, context)
        if args.command == "backup":
            return _cmd_backup(args, context)
        if args.command == "serve":
            return _cmd_serve(args, context)
        if args.command == "queue":
            return _cmd_queue(args, context)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


# ── commands ───────────────────────────────────────────────────────────────────


def _cmd_doctor(args: argparse.Namespace, settings: Any) -> int:
    report = feature_report()
    from .storage.db import Database

    db = Database(settings.db_path)
    summary = db.migrate()
    health = {
        "version": __version__,
        "profile": settings.profile,
        "database": {
            "path": str(settings.db_path),
            "schema_version": summary.version,
            "tables": len(db.tables()),
            "integrity": db.integrity_check(),
        },
        "cpu_count": __import__("os").cpu_count(),
    }
    db.close()
    payload = {**health, **report}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"NoMorals Core {__version__}  profile={settings.profile}")
    print(f"database: {health['database']['path']}")
    print(f"  schema v{summary.version}, {health['database']['tables']} tables, "
          f"integrity {health['database']['integrity']}")
    print()
    print(report_as_text(report))
    return 0


def _render_config(settings: Any) -> str:
    data = settings.to_dict()
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key} = {sub_value!r}")
        else:
            lines.append(f"{key} = {value!r}")
    return "\n".join(lines)


def _cmd_models(args: argparse.Namespace, context: Any) -> int:
    from .llm.registry import ModelRegistry, search_catalog

    registry = ModelRegistry(context.db)
    if args.activate:
        record = registry.activate(args.activate)
        _emit(args, {"activated": record.name}, f"activated {record.name}")
        return 0
    if args.catalog or args.search or args.kind:
        entries = search_catalog(
            args.search,
            kind=args.kind,
            max_params=args.max_params or None,
        )
        payload = [
            {
                "repo_id": e.repo_id, "family": e.family, "kind": e.kind,
                "params": e.params, "context": e.context_length,
                "size_gb": e.size_hint_gb, "license": e.license, "notes": e.notes,
            }
            for e in entries
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        for entry in entries:
            print(f"{entry.repo_id}")
            print(f"    {entry.kind:<9} {entry.params/1e9:>5.1f}B  ctx {entry.context_length:>6}  "
                  f"~{entry.size_hint_gb}GB  {entry.license}")
            print(f"    {entry.notes}")
        return 0
    stats = registry.stats()
    rows = registry.list(limit=50)
    if args.json:
        print(json.dumps({"stats": stats, "models": [r.__dict__ for r in rows]}, indent=2, default=str))
        return 0
    print(f"active: {stats['active'] or '(none)'}   registered: {stats['total']}   "
          f"finetunes: {stats['finetunes']}")
    for record in rows:
        mark = "*" if record.active else " "
        print(f" {mark} {record.name:<52} {record.kind:<10} {record.source}")
    return 0


def _cmd_tools(args: argparse.Namespace, context: Any) -> int:
    tools = context.tools.register_builtins()
    if args.schema:
        print(json.dumps(tools.schemas(), indent=2))
        return 0
    for schema in tools.schemas():
        params = ", ".join(schema["parameters"])
        print(f"{schema['name']:<20} [{schema['capability'] or '-'}]  {schema['description']}")
        if params:
            print(f"{'':<20} ({params})")
    return 0


def _cmd_memory(args: argparse.Namespace, context: Any) -> int:
    memory = context.memory
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


def _cmd_run(args: argparse.Namespace, context: Any) -> int:
    from .agents.orchestrator import MasterOrchestrator

    goal = " ".join(args.goal)
    orchestrator = MasterOrchestrator(context, max_steps=args.max_steps)
    result = orchestrator.run(goal, reflect=not args.no_reflect)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"plan ({len(result.plan.steps)} steps): {result.plan.rationale or 'n/a'}")
    for step in result.plan.steps:
        print(f"  - {step.name} [{step.role}/{step.kind.value}] deps={step.depends_on}")
    print(f"\nreport: {result.report.to_dict()}")
    print(f"\n{result.answer}")
    if result.lessons:
        print("\nlessons:")
        for lesson in result.lessons:
            print(f"  - {lesson}")
    return 0 if result.ok else 1


def _cmd_ask(args: argparse.Namespace, context: Any) -> int:
    from .llm.base import Message, SamplingParams

    if args.model:
        context.router.set_active(args.model)
    messages = ([Message.system(args.system)] if args.system else []) + [
        Message.user(" ".join(args.prompt))
    ]
    response = context.router.chat(
        messages, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, default=str))
    else:
        print(response.text or f"[no output: {response.error}]")
    return 0 if response.ok else 1


def _cmd_backup(args: argparse.Namespace, context: Any) -> int:
    from .storage.backup import BackupManager

    manager = BackupManager(
        context.db,
        context.settings.backup_dir,
        keep=context.settings.backup.keep,
        compress=context.settings.backup.compress,
        git_repo=context.settings.backup.git_repo,
    )
    if args.create:
        info = manager.create(label="cli")
        manager.rotate()
        _emit(args, info.to_dict(), f"created {info.name} ({info.size} bytes)")
        return 0
    if args.verify:
        problems = manager.verify()
        _emit(args, {"problems": problems}, "\n".join(problems) if problems else "backup verified clean")
        return 1 if problems else 0
    if args.restore:
        path = manager.restore(args.restore)
        _emit(args, {"restored": str(path)}, f"restored to {path}")
        return 0
    if args.push:
        result = manager.push_to_git()
        _emit(args, result, f"push: {result}")
        return 0 if result.get("pushed") else 1
    entries = manager.list()
    payload = [e.to_dict() for e in entries]
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not entries:
        print("no backups")
        return 0
    for entry in entries:
        print(f"{entry.name}  {entry.size:>10} bytes  schema v{entry.schema_version}  {entry.label}")
    return 0


def _cmd_serve(args: argparse.Namespace, context: Any) -> int:
    from .api.server import serve

    return serve(
        context,
        host=context.settings.api.host,
        port=context.settings.api.port,
    )


def _cmd_queue(args: argparse.Namespace, context: Any) -> int:
    from .storage.queue import WorkQueue

    queue = WorkQueue(context.db)
    payload = {"pending": queue.pending(args.topic or None), "topics": queue.topics()}
    _emit(args, payload, json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
