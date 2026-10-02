"""``nm`` entry point: ``main()``, dispatch, session attach, log config."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence
from pathlib import Path
from ..core.logging_setup import setup_logging
from .parser import CLI_ALIASES, _parser
from .emit import _emit
from .commands.data import _cmd_data
from .commands.native import _cmd_native
from .commands.timeline import _cmd_timeline
from .commands.doctor import _cmd_doctor, _cmd_models, _cmd_setup
from .commands.models import _cmd_model_broker
from .commands.tools import _cmd_tools
from .commands.owner import _cmd_owner
from .commands.power import _cmd_power
from .commands.memory import _cmd_memory
from .commands.code import _cmd_code
from .commands.media import _cmd_media, _cmd_studio
from .commands.captcha import _cmd_captcha
from .commands.weather import _cmd_weather
from .commands.bet import _cmd_bet
from .commands.voice import _cmd_voice
from .commands.vision import _cmd_vision
from .commands.improve import _cmd_improve
from .commands.trade import _cmd_trade
from .commands.swarm import _cmd_swarm
from .commands.room import _cmd_room
from .commands.benchmark import _cmd_benchmark
from .commands.briefing import _cmd_briefing
from .commands.inbox import _cmd_inbox
from .commands.agent import _cmd_run, _cmd_ask
from .commands.backup import _cmd_backup
from .commands.snapshot import _cmd_snapshot
from .commands.recover import _cmd_recover
from .commands.update import _cmd_update
from .commands.golden import _cmd_golden
from .commands.missions import _cmd_missions
from .commands.tui import _cmd_tui
from .commands.serve import _cmd_serve
from .commands.queue import _cmd_queue
from .commands.status import _cmd_status
from .commands.mind import _cmd_mind
from .commands.hub import _cmd_hub
from .commands.book import _cmd_book
from .commands.games import _cmd_arena, _cmd_trial, _cmd_skill, _cmd_simulate
from .commands.skills import _cmd_skill_pkg
from .commands.research import _cmd_research_loop, _cmd_kg, _cmd_cookies, _cmd_structure
from .commands.partner import _cmd_reason, _cmd_workspace
from .commands.train import _cmd_train
from .commands.security import _cmd_crack, _cmd_decode, _cmd_osint, _cmd_cipher, _cmd_monitor, _cmd_watch
from .commands.money import _cmd_money
from .commands.meta import _cmd_commands, _cmd_deliver, _cmd_zip, _cmd_connectors, _cmd_help
from .commands.finance import _cmd_finance
from .commands.cards import _cmd_cards
from .commands.autonomy import _cmd_autonomy
from .commands.music import _cmd_music
from .commands.exec import _cmd_exec, _cmd_apps
from .commands.project import _cmd_project
from .commands.mission import _cmd_mission
from .commands.goal import _cmd_goal

from ..core.logging_setup import get_logger

_log = get_logger(__name__)



def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    setup_logging(args.log_level, force=True)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E106 - deliberate top-level shutdown; exit 130
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI is the last line of defence
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _attach_cli_session(context: Any) -> None:
    """Best-effort: create-or-reuse the CLI's os.Session and stash it.

    Reuses the latest active ``cli``/``owner`` session when one exists,
    otherwise creates a fresh one via
    :class:`nomorals.os.session.SessionStore`. The session is stashed on
    ``context.extras["os_session"]`` for commands to use. Never raises and
    never changes any command's behavior — any failure here is logged at
    debug and skipped.
    """
    try:
        from ..os.session import SessionStore

        db = getattr(context, "db", None)
        store = SessionStore(db=db) if db is not None else SessionStore()
        session = None
        try:
            active = [s for s in store.list_active()
                      if getattr(s, "frontend", "") == "cli"
                      and getattr(s, "principal", "") == "owner"]
            if active:
                session = active[-1]
        except Exception:  # noqa: BLE001 - reuse is optional; fall to create
            _log.debug("listing active os sessions failed", exc_info=True)
        if session is None:
            session = store.create(frontend="cli", principal="owner")
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            extras["os_session"] = session
        else:  # exotic contexts without an extras dict
            setattr(context, "os_session", session)
    except Exception:  # noqa: BLE001 - session attach must never break the CLI
        _log.debug("CLI os.Session attach skipped", exc_info=True)


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
        if args.command == "models":
            if getattr(args, "model_action", ""):
                return _cmd_model_broker(args, context)
            return _cmd_models(args, context)
        if args.command == "data":
            return _cmd_data(args, context)
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
        if args.command == "mind":
            return _cmd_mind(args, context)

        if args.command == "book":
            return _cmd_book(args, context)
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
        if args.command == "skill":
            # New executable-skill verbs go to the skills package; the
            # legacy knowledge-library verbs stay on the games handler.
            if args.action in {"list", "install", "enable", "disable",
                               "run", "benchmark", "library"}:
                return _cmd_skill_pkg(args, context)
            return _cmd_skill(args, context)
        if args.command == "project":
            return _cmd_project(args, context)
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
        if args.command == "media":
            return _cmd_media(args, context)
        if args.command == "studio":
            return _cmd_studio(args, context)
        if args.command == "voice":
            return _cmd_voice(args, context)
        if args.command == "captcha":
            return _cmd_captcha(args, context)
        if args.command == "bet":
            return _cmd_bet(args, context)
        if args.command == "weather":
            return _cmd_weather(args, context)
        if args.command == "vision":
            return _cmd_vision(args, context)
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
