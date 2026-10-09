"""``nm`` entry point: ``main()``, dispatch, session attach, log config."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.logging_setup import RedactionFilter, get_logger, setup_logging
from .commands.agent import _cmd_ask, _cmd_run
from .commands.autonomy import _cmd_autonomy
from .commands.backup import _cmd_backup
from .commands.benchmark import _cmd_benchmark
from .commands.bet import _cmd_bet
from .commands.book import _cmd_book
from .commands.account import _cmd_account
from .commands.books import _cmd_books
from .commands.briefing import _cmd_briefing
from .commands.builders import _cmd_build
from .commands.captcha import _cmd_captcha
from .commands.cards import _cmd_cards
from .commands.browse import _cmd_browse
from .commands.code import _cmd_code
from .commands.doc import _cmd_doc
from .commands.data import _cmd_data
from .commands.datasci import _cmd_datasci
from .commands.doctor import _cmd_doctor, _cmd_models, _cmd_setup
from .commands.exec import _cmd_apps, _cmd_exec
from .commands.finance import _cmd_finance
from .commands.games import _cmd_arena, _cmd_simulate, _cmd_skill, _cmd_trial
from .commands.goal import _cmd_goal
from .commands.idea import _cmd_idea
from .commands.golden import _cmd_golden
from .commands.hub import _cmd_hub
from .commands.imggen import cmd_imggen
from .commands.shorts import cmd_shorts
from .commands.improve import _cmd_improve
from .commands.inbox import _cmd_inbox
from .commands.media import _cmd_media, _cmd_studio
from .commands.memory import _cmd_memory
from .commands.meta import _cmd_commands, _cmd_connectors, _cmd_deliver, _cmd_help, _cmd_zip
from .commands.mind import _cmd_mind
from .commands.mission import _cmd_mission
from .commands.missions import _cmd_missions
from .commands.models import _cmd_model_broker
from .commands.money import _cmd_money
from .commands.music import _cmd_music
from .commands.native import _cmd_native
from .commands.owner import _cmd_owner
from .commands.partner import _cmd_reason, _cmd_workspace
from .commands.power import _cmd_power
from .commands.project import _cmd_project
from .commands.plugin import _cmd_plugin
from .commands.queue import _cmd_queue
from .commands.recover import _cmd_recover
from .commands.repo import _cmd_repo
from .commands.research import _cmd_cookies, _cmd_kg, _cmd_research_loop, _cmd_structure
from .commands.room import _cmd_room
from .commands.security import (
    _cmd_cipher,
    _cmd_crack,
    _cmd_decode,
    _cmd_monitor,
    _cmd_osint,
    _cmd_watch,
)
from .commands.search import _cmd_search
from .commands.serve import _cmd_serve
from .commands.skills import _cmd_skill_pkg
from .commands.snapshot import _cmd_snapshot
from .commands.stream import _cmd_stream
from .commands.status import _cmd_status
from .commands.session import _cmd_session
from .commands.swarm import _cmd_swarm
from .commands.timeline import _cmd_timeline
from .commands.tools import _cmd_tools
from .commands.trade import _cmd_trade
from .commands.train import _cmd_train
from .commands.trigger import _cmd_trigger
from .commands.schedule import _cmd_schedule
from .commands.db import _cmd_db
from .commands.tui import _cmd_tui
from .commands.update import _cmd_update
from .commands.vision import _cmd_vision
from .commands.voice import _cmd_voice
from .commands.weather import _cmd_weather
from .commands.wisdom import _cmd_wisdom
from .commands.mesh import _cmd_mesh
from .commands.sync import _cmd_sync
from .emit import _emit
from .parser import CLI_ALIASES, _parser

_log = get_logger(__name__)



def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    setup_logging(args.log_level, force=True)
    # SIGTERM (supervisor stop, `kill`, Termux session end) must shut the
    # bot down the same way Ctrl-C does: context teardown, bus drain,
    # gateway stop, final status beacon. Without this the process died
    # instantly and `nm status` later reported a stale death as "went stale"
    # instead of a clean stop.
    from ..core.shutdown import install_sigterm_as_interrupt

    install_sigterm_as_interrupt()
    try:
        return _dispatch(args)
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E106 - deliberate top-level shutdown; exit 130
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI is the last line of defence
        _log.exception(
            "command dispatch failed: %s", getattr(args, "command", "?")
        )
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _attach_cli_session(context: Any) -> None:
    """Best-effort: create-or-reuse the CLI's os.Session and stash it.

    Delegates to :class:`nomorals.os.session_bridge.SessionBridge` so the
    CLI and chat surfaces share one session-resolution path. The session is
    stashed on ``context.extras["os_session"]`` for commands to use. Never
    raises and never changes any command's behavior — any failure here is
    logged at debug and skipped.
    """
    try:
        from ..os.session_bridge import SessionBridge

        db = getattr(context, "db", None)
        bridge = SessionBridge(db=db)
        session = bridge.session_for_cli(principal="owner")
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            extras["os_session"] = session
        else:  # exotic contexts without an extras dict
            context.os_session = session
    except Exception:  # noqa: BLE001 - session attach must never break the CLI
        _log.debug("CLI os.Session attach skipped", exc_info=True)


def _attach_timeline(context: Any) -> None:
    """Best-effort: persist bus events to the H2 event timeline for this run.

    Without this, ``nm timeline`` and ``nm mission replay`` only ever saw
    what tests persisted — the production serving path never called
    ``Timeline.attach()``. Attaching here (L7 entry point) keeps L5/L6
    decoupled: missions emit on ``global_bus`` and this persists them into
    the same database file ``nm timeline`` reads (``context.db.path``).

    Delivery is inline when the bus dispatcher isn't running (see
    ``EventBus.publish``), so no event is lost on quick CLI exit, and
    ``Timeline.record`` commits immediately, so a later ``nm timeline``
    process sees the events. Never raises and never changes any command's
    behavior — any failure here is logged at debug and skipped.
    """
    try:
        from ..core.events import global_bus
        from ..os.timeline import Timeline

        db = getattr(context, "db", None)
        db_path = getattr(db, "path", None)
        if not db_path:
            return  # exotic context without a database file: nothing to persist to
        tl = Timeline(db_path)
        tl.attach(global_bus)
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            extras["timeline"] = tl
        else:  # exotic contexts without an extras dict
            context.timeline = tl
    except Exception:  # noqa: BLE001 - timeline attach must never break the CLI
        _log.debug("CLI timeline attach skipped", exc_info=True)


def _canonical_command(name: str) -> str:
    """Resolve a CLI alias to its canonical command name (``st`` → ``status``).

    argparse leaves the typed alias in ``args.command``; dispatch works on
    canonical names only, so this runs first inside ``_dispatch``.
    """
    for canonical, aliases in CLI_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name


def _dispatch(args: argparse.Namespace) -> int:
    from ..core.config import load_settings

    settings = load_settings(args.config)
    # The log.file / NM_LOG_FILE setting was silently ignored — the
    # RotatingFileHandler was never attached, so file logging never
    # happened despite being configured and documented.
    _configure_log_file(args, settings)
    args.command = _canonical_command(args.command)

    if args.command == "doctor":
        return _cmd_doctor(args, settings)
    # Settings-only commands: they manage the live state itself (snapshots,
    # recovery, self-update), so they must NOT hold the database open the
    # way build_context would — restore's running/dirty detection and the
    # transactional swap depend on seeing the system honestly.
    if args.command == "snapshot":
        return _cmd_snapshot(args, settings)
    if args.command == "recover":
        return _cmd_recover(args, settings)
    if args.command == "update":
        return _cmd_update(args, settings)
    if args.command == "config":
        _emit(args, settings.to_dict(), _render_config(settings))
        return 0
    if args.command == "setup":
        from ..agents.context import build_context
        with build_context(settings) as context:
            return _cmd_setup(args, context)

    from ..agents.context import build_context

    with build_context(settings) as context:
        _attach_cli_session(context)  # best-effort os.Session for this run
        _attach_timeline(context)  # best-effort: persist bus events to the timeline
        if args.command == "models":
            if getattr(args, "model_action", ""):
                return _cmd_model_broker(args, context)
            return _cmd_models(args, context)
        if args.command == "data":
            return _cmd_data(args, context)
        if args.command == "datasci":
            return _cmd_datasci(args, context)
        if args.command == "tools":
            return _cmd_tools(args, context)
        if args.command == "memory":
            return _cmd_memory(args, context)
        if args.command == "owner":
            return _cmd_owner(args, context)
        if args.command == "power":
            return _cmd_power(args, context)
        if args.command == "run":
            return _cmd_run(args, context)
        if args.command == "ask":
            return _cmd_ask(args, context)
        if args.command == "backup":
            return _cmd_backup(args, context)
        if args.command == "golden":
            return _cmd_golden(args, context)
        if args.command == "missions":
            return _cmd_missions(args, context)
        if args.command == "timeline":
            return _cmd_timeline(args, context)
        if args.command == "tui":
            return _cmd_tui(args, context)
        if args.command == "serve":
            return _cmd_serve(args, context)
        if args.command == "stream":
            return _cmd_stream(args, context)
        if args.command == "queue":
            return _cmd_queue(args, context)
        if args.command == "commands":
            return _cmd_commands(args, context)
        if args.command == "zip":
            return _cmd_zip(args, context)
        if args.command == "deliver":
            return _cmd_deliver(args, context)
        if args.command == "status":
            return _cmd_status(args, context)
        if args.command == "session":
            return _cmd_session(args, context)
        if args.command == "mind":
            return _cmd_mind(args, context)

        if args.command == "book":
            return _cmd_book(args, context)
        if args.command == "books":
            return _cmd_books(args, context)
        if args.command == "hub":
            return _cmd_hub(args, context)
        if args.command == "cipher":
            return _cmd_cipher(args, context)
        if args.command == "osint":
            return _cmd_osint(args, context)
        if args.command == "structure":
            return _cmd_structure(args, context)
        if args.command == "money":
            return _cmd_money(args, context)
        if args.command == "arena":
            return _cmd_arena(args, context)
        if args.command == "trial":
            return _cmd_trial(args, context)
        if args.command == "train":
            return _cmd_train(args, context)
        if args.command == "help":
            return _cmd_help(args, context)
        if args.command == "cookies":
            return _cmd_cookies(args, context)
        if args.command == "reason":
            return _cmd_reason(args, context)
        if args.command == "workspace":
            return _cmd_workspace(args, context)
        if args.command == "monitor":
            return _cmd_monitor(args, context)
        if args.command == "watch":
            return _cmd_watch(args, context)
        if args.command == "crack":
            return _cmd_crack(args, context)
        if args.command == "decode":
            return _cmd_decode(args, context)
        if args.command == "music":
            return _cmd_music(args, context)
        if args.command == "exec":
            return _cmd_exec(args, context)
        if args.command == "apps":
            return _cmd_apps(args, context)
        if args.command == "connectors":
            return _cmd_connectors(args, context)
        if args.command == "finance":
            return _cmd_finance(args, context)
        if args.command == "improve":
            return _cmd_improve(args, context)
        if args.command == "trade":
            return _cmd_trade(args, context)
        if args.command == "swarm":
            return _cmd_swarm(args, context)
        if args.command == "native":
            return _cmd_native(args, context)
        if args.command == "cards":
            return _cmd_cards(args, context)
        if args.command == "autonomy":
            return _cmd_autonomy(args, context)
        if args.command == "goal":
            return _cmd_goal(args, context)
        if args.command == "idea":
            return _cmd_idea(args, context)
        if args.command == "skill":
            # New executable-skill verbs go to the skills package; the
            # legacy knowledge-library verbs stay on the games handler.
            if args.action in {"list", "install", "enable", "disable",
                               "run", "benchmark", "library"}:
                return _cmd_skill_pkg(args, context)
            return _cmd_skill(args, context)
        if args.command == "project":
            return _cmd_project(args, context)
        if args.command == "plugin":
            return _cmd_plugin(args, context)
        if args.command == "mission":
            return _cmd_mission(args, context)
        if args.command == "kg":
            return _cmd_kg(args, context)
        if args.command == "simulate":
            return _cmd_simulate(args, context)
        if args.command == "research-loop":
            return _cmd_research_loop(args, context)
        if args.command == "code":
            return _cmd_code(args, context)
        if args.command == "doc":
            return _cmd_doc(args, context)
        if args.command == "wisdom":
            return _cmd_wisdom(args, context)
        if args.command == "mesh":
            return _cmd_mesh(args, context)
        if args.command == "sync":
            return _cmd_sync(args, context)

        if args.command == "trigger":
            return _cmd_trigger(args, context)
        if args.command == "schedule":
            return _cmd_schedule(args, context)
        if args.command == "db":
            return _cmd_db(args, context)
        if args.command == "search":
            return _cmd_search(args, context)
        if args.command == "browse":
            return _cmd_browse(args, context)
        if args.command == "repo":
            return _cmd_repo(args, context)
        if args.command == "media":
            return _cmd_media(args, context)
        if args.command == "studio":
            return _cmd_studio(args, context)
        if args.command == "voice":
            return _cmd_voice(args, context)
        if args.command == "captcha":
            return _cmd_captcha(args, context)
        if args.command == "account":
            return _cmd_account(args, context)
        if args.command == "bet":
            return _cmd_bet(args, context)
        if args.command == "build":
            return _cmd_build(args, context)
        if args.command == "weather":
            return _cmd_weather(args, context)
        if args.command == "vision":
            return _cmd_vision(args, context)
        if args.command == "imggen":
            return cmd_imggen(args, context)
        if args.command == "shorts":
            return cmd_shorts(args, context)
        if args.command == "inbox":
            return _cmd_inbox(args, context)
        if args.command == "room":
            return _cmd_room(args, context)
        if args.command == "briefing":
            return _cmd_briefing(args, context)
        if args.command == "benchmark":
            return _cmd_benchmark(args, context)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


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


def _configure_log_file(args: Any = None, settings: Any = None) -> None:
    """Configure file logging based on settings.

    Args:
        args: Namespace with log_level attribute (optional)
        settings: Settings object with log.file path (optional)
    """
    import logging
    from logging.handlers import RotatingFileHandler

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
    handler.addFilter(RedactionFilter())  # keep tokens/passwords out of the log file
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root_logger.addHandler(handler)

    # Ensure root logger level allows messages through
    if root_logger.level > log_level or root_logger.level == logging.NOTSET:
        root_logger.setLevel(log_level)
