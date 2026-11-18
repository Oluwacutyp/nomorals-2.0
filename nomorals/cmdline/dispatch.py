"""``nm`` entry point: ``main()``, dispatch, session attach, log config."""

from __future__ import annotations

import argparse
import contextlib
import io
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from ..core.logging_setup import RedactionFilter, get_logger, setup_logging
from . import style as _style
from .commands.agent import _cmd_agent, _cmd_ask, _cmd_run
from .commands.autonomy import _cmd_autonomy
from .commands.av import _cmd_av
from .commands.airports import _cmd_airports
from .commands.compute import _cmd_compute
from .commands.evidence import _cmd_evidence
from .commands.devask import _cmd_devask
from .commands.consensus import _cmd_consensus
from .commands.scholar import _cmd_scholar
from .commands.paper_graph import _cmd_paper_graph
from .commands.paper_reader import _cmd_readpaper
from .commands.soundtrack import _cmd_soundtrack
from .commands.soundscape import _cmd_soundscape
from .commands.flight import _cmd_flight
from .commands.breach import _cmd_breach
from .commands.burn import _cmd_burn
from .commands.backup import _cmd_backup
from .commands.benchmark import _cmd_benchmark
from .commands.bet import _cmd_bet
from .commands.braineval import _cmd_braineval
from .commands.sandbox import _cmd_sandbox
from .commands.apk import _cmd_apk
from .commands.re import _cmd_re
from .commands.repair import _cmd_repair
from .commands.desk import _cmd_desk
from .commands.decisions import _cmd_decisions
from .commands.focus import _cmd_focus
from .commands.workflow import _cmd_workflow
from .commands.plan import _cmd_plan
from .commands.museum import _cmd_museum
from .commands.community import _cmd_community
from .commands.groups import _cmd_groups
from .commands.digest import _cmd_digest
from .commands.markets import _cmd_markets
from .commands.pump import _cmd_pump
from .commands.book import _cmd_book
from .commands.account import _cmd_account
from .commands.books import _cmd_books
from .commands.briefing import _cmd_briefing
from .commands.notify import _cmd_notify
from .commands.offline import _cmd_offline
from .commands.outcomes import _cmd_outcomes
from .commands.audio import _cmd_audio
from .commands.builders import _cmd_build
from .commands.captcha import _cmd_captcha
from .commands.fonts import _cmd_fonts
from .commands.cards import _cmd_cards
from .commands.browse import _cmd_browse
from .commands.code import _cmd_code
from .commands.graph import _cmd_graph
from .commands.trace import _cmd_trace
from .commands.prompt import _cmd_prompt
from .commands.doc import _cmd_doc
from .commands.data import _cmd_data
from .commands.analytics import _cmd_analytics
from .commands.datasci import _cmd_datasci
from .commands.doctor import _cmd_doctor, _cmd_models, _cmd_setup
from .commands.brain import _cmd_brain
from .commands.exec import _cmd_apps, _cmd_exec
from .commands.finance import _cmd_finance
from .commands.games import _cmd_arena, _cmd_simulate, _cmd_skill, _cmd_trial
from .commands.goal import _cmd_goal
from .commands.goals import _cmd_goals
from .commands.idea import _cmd_idea
from .commands.golden import _cmd_golden
from .commands.hub import _cmd_hub
from .commands.imggen import cmd_imggen
from .commands.shorts import cmd_shorts
from .commands.video import cmd_video
from .commands.board import cmd_board
from .commands.director import cmd_director
from .commands.improve import _cmd_improve
from .commands.inbox import _cmd_inbox
from .commands.learn import _cmd_learn
from .commands.legal import _cmd_legal
from .commands.immune import _cmd_immune
from .commands.impossible import _cmd_impossible
from .commands.media import _cmd_media, _cmd_studio, _cmd_trendops
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
from .commands.maps import _cmd_maps
from .commands.proxy import _cmd_proxy
from .commands.sources import _cmd_sources
from .commands.project import _cmd_project
from .commands.pulse import _cmd_pulse
from .commands.plugin import _cmd_plugin
from .commands.queue import _cmd_queue
from .commands.sent import _cmd_sent
from .commands.recover import _cmd_recover
from .commands.repo import _cmd_repo
from .commands.research import _cmd_arena_research, _cmd_cookies, _cmd_kg, _cmd_research_loop, _cmd_structure
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
from .commands.chat import _cmd_chat
from .commands.completion import _cmd_completion
from .commands.bench import _cmd_bench
from .commands.consent import _cmd_consent
from .commands.organs import _cmd_organs
from .commands.swarm import _cmd_swarm
from .commands.timeline import _cmd_timeline
from .commands.tools import _cmd_tools
from .commands.trade import _cmd_trade
from .commands.train import _cmd_train
from .commands.render import _cmd_render
from .commands.trigger import _cmd_trigger
from .commands.schedule import _cmd_schedule
from .commands.db import _cmd_db
from .commands.tui import _cmd_tui
from .commands.update import _cmd_update
from .commands.vision import _cmd_vision
from .commands.voice import _cmd_voice
from .commands.weather import _cmd_weather
from .commands.why import _cmd_why
from .commands.wisdom import _cmd_wisdom
from .commands.mesh import _cmd_mesh
from .commands.sync import _cmd_sync
from .commands.failures import _cmd_failures
from .commands.menu import _cmd_menu
from .emit import _emit
from .parser import CLI_ALIASES, _parser

_log = get_logger(__name__)


def _ensure_console_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 (errors='replace') when possible.

    Python on Windows defaults console stdout to the console code page
    (usually cp1252). CLI output contains characters cp1252 cannot encode
    (e.g. the rightwards arrow U+2192 in ``--help``), and argparse's
    ``print_help`` then raises ``UnicodeEncodeError`` — the ``--help``
    traceback seen on EC2. Reconfiguring up front makes every CLI write
    safe regardless of console encoding; ``errors='replace'`` means an
    unencodable char degrades to ``?`` instead of crashing the command.
    Best-effort: any failure here is swallowed — a console we cannot
    reconfigure must never break the CLI itself.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - console setup must never break CLI
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """The ``nm`` entry point.

    Exit-code contract (stable, scriptable):

    * ``0`` — the command ran and succeeded.
    * ``1`` — the command ran and failed (exception or handler error).
    * ``2`` — usage error: bad arguments, unknown command.
    * ``130`` — interrupted (Ctrl-C / SIGTERM).
    """
    # BEFORE parsing: argparse prints --help during parse_args, and that
    # print crashes on a cp1252 console if help contains non-cp1252 chars.
    _ensure_console_utf8()
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    if getattr(args, "no_color", False):
        _style.set_no_color(True)
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


# Commands that must never run inside the GodConsole: they take over the
# terminal, block the console loop, or restart the bot process. The
# refusal names the command and the reason (see _terminal_refusal).
TERMINAL_COMMANDS = ("run", "serve", "tui")


def _terminal_refusal(argv: list[str]) -> str | None:
    """Why ``argv`` must not run inside the GodConsole, or None if safe.

    ``run`` starts an orchestrator goal and blocks until it finishes;
    ``serve`` starts the HTTP server and blocks forever; ``stream``
    defaults to ``serve``; ``tui`` nests a curses screen inside the
    console's raw-mode screen; ``update`` prompts on stdin (which the
    console owns) and then pulls + restarts the bot. The safe siblings
    (``stream status``, ``update --check-only``) are allowed through.
    """
    if not argv:
        return None
    first = argv[0].lower()
    if first in TERMINAL_COMMANDS:
        reasons = {
            "run": "it runs an agent goal to completion and blocks the console",
            "serve": "it starts the HTTP server and blocks forever",
            "tui": "it launches the curses TUI nested inside the console",
        }
        return (
            f"'{first}' cannot be run inside the console: "
            f"{reasons[first]}. Run it from a real terminal instead."
        )
    if first == "stream" and (len(argv) < 2 or argv[1].lower() != "status"):
        return (
            "'stream' cannot be run inside the console: it starts the "
            "streaming server and blocks forever. 'stream status' is fine, "
            "and 'stream serve' belongs in a real terminal."
        )
    if first == "update" and "--check-only" not in argv:
        return (
            "'update' cannot be run inside the console: it asks for "
            "confirmation on stdin and then pulls and restarts the bot. "
            "'update --check-only' is fine."
        )
    return None


def run_command_string(cmd: str) -> str:
    """Run one ``nm ...`` command line and return its output as text.

    The GodConsole renders a string, so this is the string->string
    adapter over the print-based cmdline dispatch: shlex-split, parse
    with the real parser, run ``_dispatch`` with stdout/stderr
    captured, and return the captured text (plus an exit-code line
    when the command failed). Deliberately does NOT call ``main()``:
    ``main()`` reconfigures logging (``force=True``) and installs a
    SIGTERM handler, both wrong inside the already-running bot
    process. Never raises: parse failures, handler exceptions and
    interrupts all come back as text.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return ""
    try:
        argv = shlex.split(cmd.strip())
    except ValueError as exc:
        return f"could not parse command: {exc}"
    if argv and argv[0].lower() == "nm":
        argv = argv[1:]
    refusal = _terminal_refusal(argv)
    if refusal is not None:
        return refusal
    if not argv:
        return ""
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = _parser().parse_args(argv)
            rc = _dispatch(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        text = (out.getvalue() + err.getvalue()).strip()
        if code == 0:
            return text
        return f"{text}\n(exit code: {code})".strip() if text else f"(exit code: {code})"
    except KeyboardInterrupt:
        return "interrupted"
    except Exception as exc:  # noqa: BLE001 - adapter must never raise into the console
        return f"error: {type(exc).__name__}: {exc}"
    text = (out.getvalue() + err.getvalue()).strip()
    if rc != 0:
        text = f"{text}\n(exit code: {rc})".strip() if text else f"(exit code: {rc})"
    return text


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


class _CommandSpec(NamedTuple):
    """One registered ``nm`` command.

    ``handler`` takes ``(args, context)`` when ``needs_context`` is true,
    ``(args, settings)`` otherwise. Settings-only commands manage the live
    state itself (snapshots, recovery, self-update), so they must NOT hold
    the database open the way ``build_context`` would.
    """

    handler: Callable[[Any, Any], int]
    needs_context: bool = True


def _dispatch_models(args: argparse.Namespace, context: Any) -> int:
    """``nm models`` — registry/catalog verbs, or the broker actions."""
    if getattr(args, "model_action", ""):
        return _cmd_model_broker(args, context)
    return _cmd_models(args, context)


def _dispatch_skill(args: argparse.Namespace, context: Any) -> int:
    """``nm skill`` — executable-skill verbs go to the skills package; the
    legacy knowledge-library verbs stay on the games handler."""
    if args.action in {"list", "install", "enable", "disable",
                       "run", "benchmark", "library"}:
        return _cmd_skill_pkg(args, context)
    return _cmd_skill(args, context)


def _cmd_config(args: argparse.Namespace, settings: Any) -> int:
    """``nm config`` — print the effective configuration (settings only)."""
    _emit(args, settings.to_dict(), _render_config(settings))
    return 0


# The dispatch registry: canonical command name → spec. This is the single
# source of truth for "what commands exist" — ``_dispatch`` looks handlers
# up here instead of walking a ~100-branch if-chain, so adding a command is
# one table entry, never a dispatcher edit. ``register_command()`` below is
# the plugin-facing way to extend it at runtime (explicit, not import magic).
_COMMANDS: dict[str, _CommandSpec] = {
    # settings-only: they manage live state, never hold the DB open
    "doctor": _CommandSpec(_cmd_doctor, needs_context=False),
    "snapshot": _CommandSpec(_cmd_snapshot, needs_context=False),
    "recover": _CommandSpec(_cmd_recover, needs_context=False),
    "update": _CommandSpec(_cmd_update, needs_context=False),
    "repair": _CommandSpec(_cmd_repair, needs_context=False),
    "config": _CommandSpec(_cmd_config, needs_context=False),
    "completion": _CommandSpec(_cmd_completion, needs_context=False),
    "consent": _CommandSpec(_cmd_consent, needs_context=False),
    "bench": _CommandSpec(_cmd_bench, needs_context=False),
    "organs": _CommandSpec(_cmd_organs, needs_context=False),
    # everything else runs inside build_context()
    "setup": _CommandSpec(_cmd_setup),
    "models": _CommandSpec(_dispatch_models),
    "data": _CommandSpec(_cmd_data),
    "brain": _CommandSpec(_cmd_brain),
    "datasci": _CommandSpec(_cmd_datasci),
    "tools": _CommandSpec(_cmd_tools),
    "memory": _CommandSpec(_cmd_memory),
    "owner": _CommandSpec(_cmd_owner),
    "power": _CommandSpec(_cmd_power),
    "proxy": _CommandSpec(_cmd_proxy),
    "sources": _CommandSpec(_cmd_sources),
    "maps": _CommandSpec(_cmd_maps),
    "pulse": _CommandSpec(_cmd_pulse),
    "run": _CommandSpec(_cmd_run),
    "ask": _CommandSpec(_cmd_ask),
    "agent": _CommandSpec(_cmd_agent),
    "backup": _CommandSpec(_cmd_backup),
    "golden": _CommandSpec(_cmd_golden),
    "missions": _CommandSpec(_cmd_missions),
    "timeline": _CommandSpec(_cmd_timeline),
    "tui": _CommandSpec(_cmd_tui),
    "serve": _CommandSpec(_cmd_serve),
    "stream": _CommandSpec(_cmd_stream),
    "queue": _CommandSpec(_cmd_queue),
    "sent": _CommandSpec(_cmd_sent),
    "commands": _CommandSpec(_cmd_commands),
    "zip": _CommandSpec(_cmd_zip),
    "deliver": _CommandSpec(_cmd_deliver),
    "status": _CommandSpec(_cmd_status),
    "session": _CommandSpec(_cmd_session),
    "chat": _CommandSpec(_cmd_chat),
    "mind": _CommandSpec(_cmd_mind),
    "book": _CommandSpec(_cmd_book),
    "books": _CommandSpec(_cmd_books),
    "hub": _CommandSpec(_cmd_hub),
    "cipher": _CommandSpec(_cmd_cipher),
    "osint": _CommandSpec(_cmd_osint),
    "structure": _CommandSpec(_cmd_structure),
    "money": _CommandSpec(_cmd_money),
    "arena": _CommandSpec(_cmd_arena),
    "trial": _CommandSpec(_cmd_trial),
    "train": _CommandSpec(_cmd_train),
    "render": _CommandSpec(_cmd_render),
    "help": _CommandSpec(_cmd_help),
    "cookies": _CommandSpec(_cmd_cookies),
    "reason": _CommandSpec(_cmd_reason),
    "workspace": _CommandSpec(_cmd_workspace),
    "monitor": _CommandSpec(_cmd_monitor),
    "watch": _CommandSpec(_cmd_watch),
    "crack": _CommandSpec(_cmd_crack),
    "decode": _CommandSpec(_cmd_decode),
    "music": _CommandSpec(_cmd_music),
    "exec": _CommandSpec(_cmd_exec),
    "apps": _CommandSpec(_cmd_apps),
    "connectors": _CommandSpec(_cmd_connectors),
    "finance": _CommandSpec(_cmd_finance),
    "improve": _CommandSpec(_cmd_improve),
    "trade": _CommandSpec(_cmd_trade),
    "swarm": _CommandSpec(_cmd_swarm),
    "native": _CommandSpec(_cmd_native),
    "cards": _CommandSpec(_cmd_cards),
    "autonomy": _CommandSpec(_cmd_autonomy),
    "goal": _CommandSpec(_cmd_goal),
    "goals": _CommandSpec(_cmd_goals),
    "idea": _CommandSpec(_cmd_idea),
    "skill": _CommandSpec(_dispatch_skill),
    "project": _CommandSpec(_cmd_project),
    "plugin": _CommandSpec(_cmd_plugin),
    "mission": _CommandSpec(_cmd_mission),
    "kg": _CommandSpec(_cmd_kg),
    "simulate": _CommandSpec(_cmd_simulate),
    "research-loop": _CommandSpec(_cmd_research_loop),
    "arena-research": _CommandSpec(_cmd_arena_research),
    "code": _CommandSpec(_cmd_code),
    "graph": _CommandSpec(_cmd_graph),
    "trace": _CommandSpec(_cmd_trace),
    "prompt": _CommandSpec(_cmd_prompt),
    "doc": _CommandSpec(_cmd_doc),
    "wisdom": _CommandSpec(_cmd_wisdom),
    "mesh": _CommandSpec(_cmd_mesh),
    "sync": _CommandSpec(_cmd_sync),
    "failures": _CommandSpec(_cmd_failures, needs_context=False),
    "menu": _CommandSpec(_cmd_menu),
    "trigger": _CommandSpec(_cmd_trigger),
    "schedule": _CommandSpec(_cmd_schedule),
    "db": _CommandSpec(_cmd_db),
    "analytics": _CommandSpec(_cmd_analytics),
    "search": _CommandSpec(_cmd_search),
    "browse": _CommandSpec(_cmd_browse),
    "repo": _CommandSpec(_cmd_repo),
    "media": _CommandSpec(_cmd_media),
    "trendops": _CommandSpec(_cmd_trendops),
    "studio": _CommandSpec(_cmd_studio),
    "voice": _CommandSpec(_cmd_voice),
    "captcha": _CommandSpec(_cmd_captcha),
    "fonts": _CommandSpec(_cmd_fonts),
    "account": _CommandSpec(_cmd_account),
    "bet": _CommandSpec(_cmd_bet),
    "braineval": _CommandSpec(_cmd_braineval),
    "sandbox": _CommandSpec(_cmd_sandbox),
    "apk": _CommandSpec(_cmd_apk),
    "re": _CommandSpec(_cmd_re),
    "desk": _CommandSpec(_cmd_desk),
    "decisions": _CommandSpec(_cmd_decisions),
    "focus": _CommandSpec(_cmd_focus),
    "workflow": _CommandSpec(_cmd_workflow),
    "plan": _CommandSpec(_cmd_plan),
    "museum": _CommandSpec(_cmd_museum),
    "community": _CommandSpec(_cmd_community),
    "groups": _CommandSpec(_cmd_groups),
    "digest": _CommandSpec(_cmd_digest),
    "markets": _CommandSpec(_cmd_markets),
    "pump": _CommandSpec(_cmd_pump),
    "build": _CommandSpec(_cmd_build),
    "weather": _CommandSpec(_cmd_weather),
    "why": _CommandSpec(_cmd_why),
    "vision": _CommandSpec(_cmd_vision),
    "imggen": _CommandSpec(cmd_imggen),
    "shorts": _CommandSpec(cmd_shorts),
    "video": _CommandSpec(cmd_video),
    "board": _CommandSpec(cmd_board),
    "director": _CommandSpec(cmd_director),
    "inbox": _CommandSpec(_cmd_inbox),
    "learn": _CommandSpec(_cmd_learn),
    "legal": _CommandSpec(_cmd_legal),
    "immune": _CommandSpec(_cmd_immune),
    "impossible": _CommandSpec(_cmd_impossible),
    "audio": _CommandSpec(_cmd_audio),
    "room": _CommandSpec(_cmd_room),
    "briefing": _CommandSpec(_cmd_briefing),
    "outcomes": _CommandSpec(_cmd_outcomes),
    "notify": _CommandSpec(_cmd_notify),
    "offline": _CommandSpec(_cmd_offline),
    "av": _CommandSpec(_cmd_av),
    "airports": _CommandSpec(_cmd_airports),
    "compute": _CommandSpec(_cmd_compute),
    "evidence": _CommandSpec(_cmd_evidence),
    "devask": _CommandSpec(_cmd_devask),
    "consensus": _CommandSpec(_cmd_consensus),
    "scholar": _CommandSpec(_cmd_scholar),
    "papergraph": _CommandSpec(_cmd_paper_graph),
    "readpaper": _CommandSpec(_cmd_readpaper),
    "soundtrack": _CommandSpec(_cmd_soundtrack),
    "soundscape": _CommandSpec(_cmd_soundscape),
    "flight": _CommandSpec(_cmd_flight),
    "breach": _CommandSpec(_cmd_breach),
    "burn": _CommandSpec(_cmd_burn),
    "benchmark": _CommandSpec(_cmd_benchmark),
}


def register_command(name: str, handler: Callable[[Any, Any], int] | None = None, *,
                     needs_context: bool = True) -> Callable[[Any, Any], int]:
    """Register a new ``nm`` command handler at runtime (plugin API).

    ``name`` must be the canonical command name (no aliases — those live in
    ``CLI_ALIASES`` in ``parser.py``). Raises :class:`ValueError` if the name
    is already registered or the handler is not callable. Works direct or as
    a decorator factory::

        register_command("mything", _cmd_mything)

        @register_command("mything")
        def _cmd_mything(args, context): ...
    """
    def _register(fn: Callable[[Any, Any], int]) -> Callable[[Any, Any], int]:
        if not name or not isinstance(name, str):
            raise ValueError("command name must be a non-empty string")
        if not callable(fn):
            raise ValueError(f"handler for {name!r} is not callable")
        if name in _COMMANDS:
            raise ValueError(f"command {name!r} is already registered")
        _COMMANDS[name] = _CommandSpec(fn, needs_context)
        return fn

    if handler is None:
        return _register
    return _register(handler)


def _suggest_command(name: str) -> list[str]:
    """Closest command names to *name* (canonicals + aliases), for typos."""
    from difflib import get_close_matches

    candidates: list[str] = []
    for cand in list(_COMMANDS) + list(CLI_ALIASES):
        if cand not in candidates:
            candidates.append(cand)
    for aliases in CLI_ALIASES.values():
        for alias in aliases:
            if alias not in candidates:
                candidates.append(alias)
    return get_close_matches(name, candidates, n=3, cutoff=0.6)


def _dispatch(args: argparse.Namespace) -> int:
    from ..core.config import load_settings

    settings = load_settings(args.config)
    # The log.file / NM_LOG_FILE setting was silently ignored — the
    # RotatingFileHandler was never attached, so file logging never
    # happened despite being configured and documented.
    _configure_log_file(args, settings)
    args.command = _canonical_command(args.command)

    spec = _COMMANDS.get(args.command)
    if spec is None:
        suggestions = _suggest_command(args.command)
        lines = [f"unknown command: {args.command}"]
        if suggestions:
            did = ", ".join(_style.accent(s) for s in suggestions)
            lines.append(f"did you mean: {did}?")
        lines.append("run `nm help cli` for the full command list.")
        print("\n".join(lines), file=sys.stderr)
        return 2

    if not spec.needs_context:
        return spec.handler(args, settings)

    from ..agents.context import build_context

    with build_context(settings) as context:
        _attach_cli_session(context)  # best-effort os.Session for this run
        _attach_timeline(context)  # best-effort: persist bus events to the timeline
        return spec.handler(args, context)


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
