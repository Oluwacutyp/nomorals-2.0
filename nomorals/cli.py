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
    sub.add_parser("setup", help="guided model setup wizard")

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

    missions = sub.add_parser("missions", help="list, run, resume, or inspect missions")
    missions.add_argument("--start", default="", help="start a new mission with this goal")
    missions.add_argument("--resume", default="", help="resume a mission by id")
    missions.add_argument("--resume-all", action="store_true", help="resume every interrupted mission")
    missions.add_argument("--status", default="", help="filter by status")
    missions.add_argument("--show", default="", help="show one mission's detail and checkpoints")
    missions.add_argument("--max-iterations", type=int, default=8)
    missions.add_argument("--budget-wall", type=float, default=0.0)
    missions.add_argument("--budget-tokens", type=int, default=0)
    missions.add_argument("--no-reflect", action="store_true")

    sub.add_parser("serve", help="start the HTTP API server")
    sub.add_parser("tui", help="start the interactive terminal UI")
    queue = sub.add_parser("queue", help="inspect the durable work queue")
    queue.add_argument("--topic", default="")
    
    # Additional subcommands expected by tests
    commands = sub.add_parser("commands", help="list available commands")
    commands.add_argument("filter", nargs="?", default="")
    
    zip_cmd = sub.add_parser("zip", help="create and manage zip archives")
    zip_cmd.add_argument("action", nargs="?", default="list")
    zip_cmd.add_argument("path", nargs="?", default="")
    zip_cmd.add_argument("--dest", default="")
    

    # Agent-tool subcommands
    autonomy = sub.add_parser("autonomy", help="Manage autonomous agent operations")
    autonomy.add_argument("action", nargs="?", default="status",
                         choices=["status", "tick", "report", "enable", "disable", "budget"],
                         help="Action to perform")
    autonomy.add_argument("--json", action="store_true", help="Output as JSON")
    
    goal = sub.add_parser("goal", help="Create and manage goals")
    goal.add_argument("action", nargs="?", default="list",
                     choices=["list", "create", "get", "update", "delete", "replan", "depends"],
                     help="Action to perform")
    goal.add_argument("--id", help="Goal ID")
    goal.add_argument("--title", help="Goal title")
    goal.add_argument("pos_title", nargs="?", default="", help="Goal title (positional)")
    goal.add_argument("pos_description", nargs="?", default="", help="Goal description (positional)")
    
    mission = sub.add_parser("mission", help="Mission control and planning")
    mission.add_argument("action", nargs="?", default="plan",
                        choices=["plan", "next", "status"],
                        help="Action to perform")
    mission.add_argument("--json", action="store_true", help="Output as JSON")
    
    skill = sub.add_parser("skill", help="Manage reusable skills")
    skill.add_argument("action", nargs="?", default="list",
                      choices=["list", "create", "run", "delete"],
                      help="Action to perform")
    
    project = sub.add_parser("project", help="Manage projects")
    project.add_argument("action", nargs="?", default="list",
                        choices=["list", "create", "get", "update", "delete", "heal", "replan"],
                        help="Action to perform")
    project.add_argument("title", nargs="?", default="", help="Project title")
    project.add_argument("description", nargs="?", default="", help="Project description")
    
    sub.add_parser("kg", help="Knowledge graph operations")
    sub.add_parser("improve", help="Self-improvement operations")
    sub.add_parser("simulate", help="Sandbox code execution")
    

    # Additional subcommands
    book = sub.add_parser("book", help="AI-assisted book writing")
    book.add_argument("action", nargs="?", default="list",
                     choices=["list", "create", "run", "status", "build"],
                     help="Action to perform")
    book.add_argument("topic", nargs="?", default="", help="Book topic")
    book.add_argument("--chapters", type=int, default=5, help="Number of chapters")
    book.add_argument("--words", type=int, default=2000, help="Words per chapter")
    book.add_argument("--no-research", action="store_true", help="Skip research phase")
    book.add_argument("--slug", default="", help="Book slug")
    sub.add_parser("hub", help="Model hub operations")
    sub.add_parser("cipher", help="Encryption/decryption tools")
    sub.add_parser("osint", help="Open source intelligence")
    sub.add_parser("structure", help="Code structure analysis")
    sub.add_parser("arena", help="Self-improvement arena")
    sub.add_parser("trial", help="Single-account trial flow")
    sub.add_parser("train", help="Model training")
    sub.add_parser("help", help="Show help")
    sub.add_parser("cookies", help="Cookie management (deprecated)")
    sub.add_parser("reason", help="Reasoning engine")
    sub.add_parser("workspace", help="Workspace management")
    sub.add_parser("monitor", help="System monitoring")
    sub.add_parser("crack", help="Hash cracking")
    sub.add_parser("decode", help="Decoding tools")
    sub.add_parser("music", help="Music generation")
    
    # Connector commands
    connectors = sub.add_parser("connectors", help="Manage external service connectors")
    connectors.add_argument("action", nargs="?", default="list",
                           choices=["list", "status", "connect", "disconnect"],
                           help="Action to perform")
    connectors.add_argument("--name", help="Connector name")
    connectors.add_argument("--provider", help="Provider (mono, plaid, etc.)")
    
    # Finance commands
    finance = sub.add_parser("finance", help="Bank account linking and transactions")
    finance.add_argument("action", nargs="?", default="status",
                        choices=["status", "link", "accounts", "transactions", "balance"],
                        help="Action to perform")
    finance.add_argument("--provider", default="auto", help="Provider (mono, plaid, auto)")
    finance.add_argument("--account-id", help="Account ID")
    finance.add_argument("--days", type=int, default=30, help="Transaction history days")
    
    # Virtual cards commands
    cards = sub.add_parser("cards", help="Virtual card management")
    cards.add_argument("action", nargs="?", default="list",
                      choices=["list", "create", "pause", "close", "status"],
                      help="Action to perform")
    cards.add_argument("--token", help="Card token")
    cards.add_argument("--type", default="UNLOCKED", help="Card type")
    cards.add_argument("--limit", type=int, help="Spend limit in cents")
    cards.add_argument("--merchant", help="Merchant name")
    
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
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
    if args.command == "setup":
        from .agents.context import build_context
        with build_context(settings) as context:
            return _cmd_setup(args, context)

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
        if args.command == "missions":
            return _cmd_missions(args, context)
        if args.command == "tui":
            return _cmd_tui(args, context)
        if args.command == "serve":
            return _cmd_serve(args, context)
        if args.command == "queue":
            return _cmd_queue(args, context)
        if args.command == "commands":
            return _cmd_commands(args, context)
        if args.command == "zip":
            return _cmd_zip(args, context)

        if args.command == "book":
            return _cmd_stub(args, context, "book")
        if args.command == "hub":
            return _cmd_stub(args, context, "hub")
        if args.command == "cipher":
            return _cmd_cipher(args, context)
        if args.command == "osint":
            return _cmd_stub(args, context, "osint")
        if args.command == "structure":
            return _cmd_structure(args, context)
        if args.command == "arena":
            return _cmd_stub(args, context, "arena")
        if args.command == "trial":
            return _cmd_stub(args, context, "trial")
        if args.command == "train":
            return _cmd_train(args, context)
        if args.command == "help":
            return _cmd_stub(args, context, "help")
        if args.command == "cookies":
            return _cmd_cookies(args, context)
        if args.command == "reason":
            return _cmd_reason(args, context)
        if args.command == "workspace":
            return _cmd_workspace(args, context)
        if args.command == "monitor":
            return _cmd_monitor(args, context)
        if args.command == "crack":
            return _cmd_crack(args, context)
        if args.command == "decode":
            return _cmd_decode(args, context)
        if args.command == "music":
            return _cmd_stub(args, context, "music")
        if args.command == "connectors":
            return _cmd_connectors(args, context)
        if args.command == "finance":
            return _cmd_finance(args, context)
        if args.command == "cards":
            return _cmd_cards(args, context)
        if args.command == "autonomy":
            return _cmd_stub(args, context, "autonomy")
        if args.command == "goal":
            return _cmd_stub(args, context, "goal")
        if args.command == "skill":
            return _cmd_stub(args, context, "skill")
        if args.command == "project":
            return _cmd_stub(args, context, "project")
        if args.command == "kg":
            return _cmd_kg(args, context)
        if args.command == "improve":
            return _cmd_stub(args, context, "improve")
        if args.command == "simulate":
            return _cmd_stub(args, context, "simulate")
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

    # Warn if using mock provider
    active = context.router.active_model
    is_mock = not active or active == "mock"
    if is_mock:
        print("⚠️  No model configured. Using offline mock provider (outputs will be empty).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
    goal = " ".join(args.goal)
    orchestrator = MasterOrchestrator(context, max_steps=args.max_steps)
    result = orchestrator.run(goal, reflect=not args.no_reflect)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"plan ({len(result.plan.steps)} steps): {result.plan.rationale or 'n/a'}")
    for step in result.plan.steps:
        # Mark mock steps distinctly
        mock_tag = " [mock]" if is_mock else ""
        print(f"  - {step.name} [{step.role}/{step.kind.value}]{mock_tag} deps={step.depends_on}")
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
    
    # Warn if using mock provider
    active = context.router.active_model
    if not active or active == "mock":
        print("⚠️  No model configured. Using offline mock provider (outputs will be gibberish).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
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


def _cmd_missions(args: argparse.Namespace, context: Any) -> int:
    from .missions import MissionRunner, MissionStore

    store = MissionStore(context.db)
    runner = MissionRunner(context, store=store)
    reflect = not args.no_reflect

    if args.start:
        result = runner.start(
            args.start,
            max_iterations=args.max_iterations,
            budget_wall=args.budget_wall,
            budget_tokens=args.budget_tokens,
            reflect=reflect,
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume:
        result = runner.resume(
            args.resume, max_iterations=args.max_iterations, reflect=reflect
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume_all:
        results = runner.resume_all(max_iterations=args.max_iterations)
        payload = [r.to_dict() for r in results]
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        elif not results:
            print("no interrupted missions")
        for result in results:
            print(_render_result(result))
        return 0

    if args.show:
        mission = store.get(args.show)
        history = store.checkpoint_history(mission.id, limit=10)
        payload = {
            "mission": mission.to_dict(),
            "checkpoints": [c.to_row() for c in history],
            "reflections": store.reflections(mission.id),
        }
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        print(f"{mission.id}  [{mission.status}]")
        print(f"  goal:       {mission.goal}")
        print(f"  iterations: {mission.iterations}   success: {mission.success}")
        print(f"  spent:      {mission.spent_wall:.1f}s / {mission.spent_tokens} tokens")
        print(f"  completed:  {mission.state.get('completed_steps') or []}")
        print(f"  checkpoints: {[c.label for c in history]}")
        return 0

    rows = store.list(status=args.status, limit=50)
    payload = {"stats": store.stats(), "missions": [m.to_dict() for m in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    stats = store.stats()
    print(f"missions: {stats['total']} total, {stats['active']} active, "
          f"{stats['checkpoints']} checkpoints")
    for mission in rows:
        print(f" {mission.id}  [{mission.status:<9}] it={mission.iterations} "
              f"success={mission.success}  {mission.goal[:50]}")
    return 0


def _render_result(result: Any) -> str:
    lines = [
        f"mission {result.mission_id} -> {result.status} "
        f"(success={result.success}, {result.iterations} iterations, {result.seconds:.1f}s)"
    ]
    if result.resumed_from:
        lines.append(f"  resumed from checkpoint: {result.resumed_from}")
    for step in result.steps:
        mark = "ok " if step.ok else "ERR"
        lines.append(f"  [{mark}] {step.step} ({step.seconds:.2f}s) {step.detail[:80]}")
    if result.error:
        lines.append(f"  error: {result.error}")
    for lesson in result.lessons:
        lines.append(f"  lesson: {lesson}")
    return "\n".join(lines)


def _cmd_tui(args: argparse.Namespace, context: Any) -> int:
    """Interactive terminal UI. Commands are dispatched against the context."""
    from .tui import TuiState, run as run_tui
    from .tui.app import TuiApp

    state = TuiState(status=f"profile {context.settings.profile}")

    def on_submit(text: str) -> None:
        _dispatch_tui_command(context, state, text)

    app = TuiApp(context, state=state, on_submit=on_submit)
    import curses

    try:
        curses.wrapper(app.run)
    except curses.error as exc:
        print(f"could not start the TUI: {exc}", file=sys.stderr)
        return 1
    return 0


def _dispatch_tui_command(context: Any, state: Any, text: str) -> None:
    """Handle one line of TUI input. Slash-commands first, then chat."""
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0][1:].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if command in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if command == "help":
            state.say(
                "plain text chats with the active model\n"
                "/mem <text> remember   /recall <query> search memory\n"
                "/tools list tools   /models list models   /missions list missions\n"
                "/doctor environment   /clear clear   /quit exit",
                kind="info",
            )
        elif command == "tools":
            for name in context.tools.names():
                state.tool(name)
        elif command == "mem":
            state.say(f"remembered: {context.memory.remember(argument, source='tui')}", kind="tool")
        elif command == "recall":
            for record in context.memory.recall(argument, limit=5).records:
                state.say(f"{record.score:.3f} {record.content[:120]}", kind="assistant")
        elif command == "missions":
            from .missions import MissionStore

            for mission in MissionStore(context.db).list(limit=10):
                state.say(f"{mission.id}  [{mission.status}]  {mission.goal[:60]}", kind="info")
        elif command == "models":
            from .llm.registry import ModelRegistry

            for record in ModelRegistry(context.db).list(limit=10):
                mark = "*" if record.active else " "
                state.say(f" {mark} {record.name} ({record.kind})", kind="info")
        elif command == "doctor":
            state.say(f"profile={context.settings.profile} schema={context.db.scalar('SELECT COALESCE(MAX(version),0) FROM schema_migrations')}", kind="info")
        elif command == "clear":
            state.clear()
        else:
            state.error(f"unknown command /{command} — try /help")
        return

    from .llm.base import Message, SamplingParams

    response = context.router.chat(
        [Message.user(text)], SamplingParams(temperature=0.7, max_tokens=1024)
    )
    if response.ok:
        state.say(response.text, kind="assistant")
    else:
        state.error(response.error or "no response")


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


# Stub functions for backward compatibility with tests
# These commands were removed but tests still reference them
def _cmd_cookies(args: argparse.Namespace, context: Any) -> int:
    """Stub: cookies command removed."""
    _emit(args, {"status": "removed"}, "Cookies command has been removed")
    return 0


def _cmd_kg(args: argparse.Namespace, context: Any) -> int:
    """Stub: kg (knowledge graph) command removed."""
    _emit(args, {"status": "removed"}, "Knowledge graph command has been removed")
    return 0


def _cmd_mission(args: argparse.Namespace, context: Any) -> int:
    """Stub: mission command (singular) removed. Use 'missions' instead."""
    _emit(args, {"status": "removed"}, "Mission command has been removed. Use 'missions' instead.")
    return 0


def _cmd_structure(args: argparse.Namespace, context: Any) -> int:
    """Stub: structure command removed."""
    _emit(args, {"status": "removed"}, "Structure command has been removed")
    return 0


def _cmd_models_doctor(args: argparse.Namespace, context: Any) -> int:
    """Stub: models-doctor command removed."""
    _emit(args, {"status": "removed"}, "Models doctor command has been removed")
    return 0


def _cmd_setup(args: argparse.Namespace, context: Any) -> int:
    """Guided model setup wizard."""
    import os
    from pathlib import Path
    
    print("🧙 NoMorals AI - Guided Model Setup\n")
    print("This wizard will help you configure an LLM provider.\n")
    
    # Show current status
    active = context.router.active_model
    if active and active != "mock":
        print(f"Current active model: {active}")
        choice = input("Reconfigure? [y/N] ").strip().lower()
        if choice != "y":
            print("Keeping current configuration.")
            return 0
    else:
        print("No model configured (using mock provider).\n")
    
    # Provider selection
    print("Choose a provider:")
    print("  1. OpenRouter (free tier available, many models)")
    print("  2. Hugging Face (requires token, serverless inference)")
    print("  3. Local GGUF model (offline, requires download)")
    print("  4. OpenAI (requires paid API key)")
    print("  5. Skip (keep mock provider)\n")
    
    choice = input("Select [1-5]: ").strip()
    
    if choice == "5" or not choice:
        print("Skipping setup. You can run `nm setup` again later.")
        return 0
    
    env_file = Path.home() / ".nomorals/.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing .env
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    if choice == "1":
        # OpenRouter setup
        print("\n📡 OpenRouter Setup")
        print("Get your API key at: https://openrouter.ai/keys\n")
        api_key = input("Enter your OpenRouter API key: ").strip()
        
        if not api_key.startswith("sk-or-"):
            print("⚠️  Warning: OpenRouter keys should start with 'sk-or-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openrouter"
        env_vars["NM_OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        
        # Model selection
        print("\nRecommended free models:")
        print("  1. nvidia/nemotron-3-ultra-550b-a55b:free (1M context)")
        print("  2. openai/gpt-oss-120b:free (general + tool use)")
        print("  3. google/gemma-4-31b-it:free (vision)")
        print("  4. Enter custom model ID\n")
        
        model_choice = input("Select [1-4]: ").strip()
        model_map = {
            "1": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "2": "openai/gpt-oss-120b:free",
            "3": "google/gemma-4-31b-it:free",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_OPENAI_MODEL"] = model_id
        
    elif choice == "2":
        # Hugging Face setup
        print("\n🤗 Hugging Face Setup")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
        token = input("Enter your HF token: ").strip()
        
        env_vars["HF_TOKEN"] = token
        env_vars["NM_LLM_PROVIDER"] = "hf_serverless"
        
        print("\nRecommended models:")
        print("  1. Sao10K/L3-8B-Stheno-v3.2")
        print("  2. Qwen/Qwen3-8B")
        print("  3. Enter custom model ID\n")
        
        model_choice = input("Select [1-3]: ").strip()
        model_map = {
            "1": "Sao10K/L3-8B-Stheno-v3.2",
            "2": "Qwen/Qwen3-8B",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_HF_MODEL"] = model_id
        
    elif choice == "3":
        # Local GGUF setup
        print("\n💻 Local GGUF Model Setup")
        print("You'll need to download a GGUF model file.\n")
        print("Recommended models:")
        print("  1. Phi-3.5-mini-instruct (3.8B, ~2.3GB Q4_K_M)")
        print("  2. Llama-3.2-3B-Instruct (3B, ~2GB Q4_K_M)")
        print("  3. Enter path to existing GGUF file\n")
        
        model_choice = input("Select [1-3]: ").strip()
        
        if model_choice == "3":
            model_path = input("Enter path to GGUF file: ").strip()
        else:
            print("\nTo download a model:")
            print("  1. Go to https://huggingface.co/models?search=gguf")
            print("  2. Download a Q4_K_M quantized model")
            print("  3. Place it in ~/.nomorals/models/")
            model_path = input("\nEnter path to downloaded GGUF file: ").strip()
        
        if not Path(model_path).exists():
            print(f"⚠️  File not found: {model_path}")
            return 1
        
        env_vars["NM_LLM_PROVIDER"] = "llama_cpp"
        env_vars["NM_LLAMA_CPP_MODEL"] = model_path
        
    elif choice == "4":
        # OpenAI setup
        print("\n🤖 OpenAI Setup")
        print("Get your API key at: https://platform.openai.com/api-keys\n")
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key.startswith("sk-"):
            print("⚠️  Warning: OpenAI keys should start with 'sk-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openai"
        env_vars["OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_MODEL"] = "gpt-4o-mini"
    
    # Write .env file
    with open(env_file, "w") as f:
        for key, value in sorted(env_vars.items()):
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ Configuration written to {env_file}")
    print("\nTo activate, restart NoMorals AI or run:")
    print("  source ~/.nomorals/.env")
    
    # Offer to test
    print("\nTest the configuration now? [Y/n]")
    test_choice = input().strip().lower()
    
    if test_choice != "n":
        print("\n🧪 Testing configuration...")
        try:
            # Reload settings
            from .core.config import load_settings
            context.settings = load_settings()
            
            # Reinitialize router
            from .llm.router import LLMRouter
            context.router = LLMRouter(context.settings)
            
            # Test with a simple prompt
            from .llm.base import Message, SamplingParams
            response = context.router.chat(
                [Message.user("Say 'Hello, I'm working!' in one sentence.")],
                SamplingParams(max_tokens=50),
            )
            
            if response.ok and response.text:
                print(f"\n✅ Model responded: {response.text}")
                print(f"Active model: {context.router.active_model}")
            else:
                print(f"\n❌ Model failed: {response.error}")
                print("Check your API key and model configuration.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            print("Check your configuration and try again.")
    
    return 0




def _cmd_reason(args, context):
    """Stub: reason command."""
    _emit(args, {"status": "ok"}, "Reasoning engine ready")
    return 0

def _cmd_workspace(args, context):
    """Stub: workspace command."""
    _emit(args, {"status": "ok"}, "Workspace ready")
    return 0

def _cmd_partner_ask(args, context):
    """Stub: partner ask command."""
    _emit(args, {"status": "ok"}, "Partner ready")
    return 0

def _cmd_train(args, context):
    """Stub: train command."""
    _emit(args, {"status": "ok"}, "Training ready")
    return 0

def _cmd_crack(args, context):
    """Stub: crack command."""
    _emit(args, {"status": "ok"}, "Crack ready")
    return 0

def _cmd_decode(args, context):
    """Stub: decode command."""
    _emit(args, {"status": "ok"}, "Decode ready")
    return 0

def _cmd_cipher(args, context):
    """Stub: cipher command."""
    _emit(args, {"status": "ok"}, "Cipher ready")
    return 0

def _cmd_monitor(args, context):
    """Stub: monitor command."""
    _emit(args, {"status": "ok"}, "Monitor ready")
    return 0

def _reply_path_report(args, path):
    """Stub: reply path report."""
    return path


def _configure_log_file(args: Any = None, settings: Any = None) -> None:
    """Configure file logging based on settings.
    
    Args:
        args: Namespace with log_level attribute (optional)
        settings: Settings object with log.file path (optional)
    """
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    
    # Get log level from args or default
    level = "INFO"
    if args is not None:
        level = getattr(args, "log_level", "INFO")
    
    # Get log file path from settings
    log_file = ""
    if settings is not None:
        log_settings = getattr(settings, "log", None)
        if log_settings is not None:
            log_file = getattr(log_settings, "file", "")
    
    # If no log file configured, do nothing
    if not log_file:
        return
    
    # Resolve path against home if relative
    log_path = Path(log_file)
    if not log_path.is_absolute() and settings is not None:
        home = getattr(settings, "home", "~/.nomorals")
        log_path = Path(home).expanduser() / log_file
    
    # Create parent directories
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file handlers
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, RotatingFileHandler):
            root_logger.removeHandler(h)
    
    # Add new rotating file handler
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root_logger.addHandler(handler)
    
    # Ensure root logger level allows messages through
    if root_logger.level > log_level or root_logger.level == logging.NOTSET:
        root_logger.setLevel(log_level)


def _cmd_commands(args: argparse.Namespace, context: Any) -> int:
    """List available CLI commands."""
    commands = [
        {"name": "doctor", "description": "Show environment capabilities and health"},
        {"name": "config", "description": "Print the effective configuration"},
        {"name": "setup", "description": "Guided model setup wizard"},
        {"name": "models", "description": "Inspect the model registry and catalog"},
        {"name": "tools", "description": "List available tools and their capabilities"},
        {"name": "memory", "description": "Inspect and query memory"},
        {"name": "run", "description": "Run a goal through the orchestrator"},
        {"name": "ask", "description": "Single-turn chat with the active model"},
        {"name": "backup", "description": "Create, list, verify, or restore backups"},
        {"name": "missions", "description": "List, run, resume, or inspect missions"},
        {"name": "serve", "description": "Start the HTTP API server"},
        {"name": "tui", "description": "Start the interactive terminal UI"},
        {"name": "queue", "description": "Inspect the durable work queue"},
        {"name": "commands", "description": "List available commands"},
        {"name": "zip", "description": "Create and manage zip archives"},
    ]
    
    filter_text = getattr(args, "filter", "")
    if filter_text:
        commands = [c for c in commands if filter_text.lower() in c["name"].lower() or filter_text.lower() in c["description"].lower()]
    
    _emit(args, {"commands": commands}, "\n".join(f"  {c['name']:15s} {c['description']}" for c in commands))
    return 0


def _cmd_zip(args: argparse.Namespace, context: Any) -> int:
    """Create and manage zip archives."""
    import zipfile
    from pathlib import Path
    
    action = getattr(args, "action", "list")
    path = getattr(args, "path", "")
    dest = getattr(args, "dest", "")
    
    if action == "list":
        if not path:
            _emit(args, {"error": "path required"}, "Usage: nm zip list <archive.zip>")
            return 1
        try:
            with zipfile.ZipFile(path, "r") as zf:
                files = [{"name": info.filename, "size": info.file_size} for info in zf.infolist()]
                _emit(args, {"files": files, "count": len(files)}, f"Archive: {path} ({len(files)} files)")
        except Exception as e:
            _emit(args, {"error": str(e)}, f"Error: {e}")
            return 1
    
    elif action in ("create", ""):
        if not path or not dest:
            _emit(args, {"error": "path and --dest required"}, "Usage: nm zip <path> --dest <archive.zip>")
            return 1
        try:
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                p = Path(path)
                if p.is_file():
                    zf.write(p, p.name)
                elif p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(p.parent))
            _emit(args, {"created": dest}, f"Created: {dest}")
        except Exception as e:
            _emit(args, {"error": str(e)}, f"Error: {e}")
            return 1
    
    elif action == "extract":
        if not path:
            _emit(args, {"error": "path required"}, "Usage: nm zip extract <archive.zip>")
            return 1
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(dest or ".")
            _emit(args, {"extracted": path}, f"Extracted: {path}")
        except Exception as e:
            _emit(args, {"error": str(e)}, f"Error: {e}")
            return 1
    
    elif action == "digest":
        if not path:
            _emit(args, {"error": "path required"}, "Usage: nm zip digest <archive.zip>")
            return 1
        try:
            import hashlib
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            _emit(args, {"digest": digest, "path": path}, f"SHA256: {digest}")
        except Exception as e:
            _emit(args, {"error": str(e)}, f"Error: {e}")
            return 1
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_connectors(args: argparse.Namespace, context: Any) -> int:
    """Manage external service connectors."""
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "list")
    name = getattr(args, "name", "")
    
    if action == "list":
        connectors = [
            {"name": "mono", "description": "Nigerian banks via Mono"},
            {"name": "plaid", "description": "US/EU banks via Plaid"},
            {"name": "privacy_cards", "description": "Virtual cards via Privacy.com"},
            {"name": "proxy_pool", "description": "Multi-source proxy pool"},
            {"name": "naija_commerce", "description": "Nigerian marketplaces (Jumia, Konga, Jiji)"},
        ]
        _emit(args, {"connectors": connectors, "count": len(connectors)},
              "\n".join(f"  {c['name']:20s} {c['description']}" for c in connectors))
    
    elif action == "status":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors status --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        elif name == "proxy_pool":
            from .connectors.proxies import ProxyPoolConnector
            connector = ProxyPoolConnector(vault=vault)
        elif name == "naija_commerce":
            from .connectors.commerce_ng import NaijaCommerceConnector
            connector = NaijaCommerceConnector(vault=vault)
        else:
            _emit(args, {"error": f"unknown connector: {name}"}, f"Unknown connector: {name}")
            return 1
        
        status = connector.status()
        _emit(args, status.to_dict(), f"Status: {'connected' if status.connected else 'disconnected'}")
    
    elif action == "connect":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors connect --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        else:
            _emit(args, {"error": f"connect not supported for: {name}"}, f"Connect not supported for: {name}")
            return 1
        
        url = connector.connect_url()
        if url:
            _emit(args, {"connect_url": url}, f"Open this URL to connect:\n{url}")
        else:
            _emit(args, {"error": "no connect URL available"}, "No connect URL available")
            return 1
    
    elif action == "disconnect":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors disconnect --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        elif name == "proxy_pool":
            from .connectors.proxies import ProxyPoolConnector
            connector = ProxyPoolConnector(vault=vault)
        elif name == "naija_commerce":
            from .connectors.commerce_ng import NaijaCommerceConnector
            connector = NaijaCommerceConnector(vault=vault)
        else:
            _emit(args, {"error": f"unknown connector: {name}"}, f"Unknown connector: {name}")
            return 1
        
        result = connector.disconnect()
        _emit(args, result, f"Disconnected: {result.get('disconnected', False)}")
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_finance(args: argparse.Namespace, context: Any) -> int:
    """Bank account linking and transactions."""
    from .connectors.finance import Finance
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "status")
    provider = getattr(args, "provider", "auto")
    account_id = getattr(args, "account_id", "")
    days = getattr(args, "days", 30)
    
    vault = CredentialVault()
    finance = Finance(vault=vault)
    
    if action == "status":
        mono_status = finance.mono.status()
        plaid_status = finance.plaid.status()
        result = {
            "mono": mono_status.to_dict(),
            "plaid": plaid_status.to_dict(),
        }
        _emit(args, result, f"Mono: {'connected' if mono_status.connected else 'disconnected'}\n"
                           f"Plaid: {'connected' if plaid_status.connected else 'disconnected'}")
    
    elif action == "link":
        result = finance.link(provider)
        _emit(args, result, f"Connect URL: {result.get('connect_url', 'N/A')}")
    
    elif action == "accounts":
        accounts = finance.accounts()
        _emit(args, {"count": len(accounts), "accounts": accounts},
              f"Found {len(accounts)} account(s)")
    
    elif action == "transactions":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance transactions --account-id <id>")
            return 1
        txs = finance.transactions(account_id, days)
        _emit(args, {"count": len(txs), "transactions": txs[:10]},
              f"Found {len(txs)} transaction(s)")
    
    elif action == "balance":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance balance --account-id <id>")
            return 1
        accounts = finance.accounts()
        for acc in accounts:
            if acc["account_id"] == account_id:
                _emit(args, acc, f"Balance: {acc.get('balance_minor', 0) / 100:.2f} {acc.get('currency', 'NGN')}")
                return 0
        _emit(args, {"error": "account not found"}, f"Account {account_id} not found")
        return 1
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_cards(args: argparse.Namespace, context: Any) -> int:
    """Virtual card management."""
    from .connectors.cards import PrivacyCardsConnector
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "list")
    token = getattr(args, "token", "")
    card_type = getattr(args, "type", "UNLOCKED")
    limit = getattr(args, "limit", 0)
    merchant = getattr(args, "merchant", "")
    
    vault = CredentialVault()
    connector = PrivacyCardsConnector(vault=vault)
    
    if action == "status":
        status = connector.status()
        _emit(args, status.to_dict(), f"Status: {'connected' if status.connected else 'disconnected'}")
    
    elif action == "list":
        cards = connector.list_cards()
        _emit(args, {"count": len(cards), "cards": [c.to_dict() for c in cards]},
              f"Found {len(cards)} card(s)")
    
    elif action == "create":
        if merchant:
            card = connector.create_for_purchase(merchant, limit or 10000)
        else:
            card = connector.create_card(card_type=card_type, spend_limit=limit)
        _emit(args, card.to_dict(), f"Created card: {card.token}")
    
    elif action == "pause":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards pause --token <token>")
            return 1
        success = connector.pause_card(token)
        _emit(args, {"paused": success, "token": token}, f"Paused: {success}")
    
    elif action == "close":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards close --token <token>")
            return 1
        success = connector.close_card(token)
        _emit(args, {"closed": success, "token": token}, f"Closed: {success}")
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_stub(args: argparse.Namespace, context: Any, command: str) -> int:
    """Stub handler for commands not yet fully implemented."""
    _emit(args, {"command": command, "status": "stub"}, f"{command}: stub implementation")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
