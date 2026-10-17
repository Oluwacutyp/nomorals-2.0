#!/usr/bin/env python3
"""Long-running chat bot entrypoint.

Wires the partner runtime to whichever chat adapters are enabled in settings
and blocks forever. The intended shape on a small always-on host (Serv00,
a cheap VPS) is:

    NM_LLM_PROVIDER=hf_serverless          # brain = HF Inference API, no local model
    NM_HF_MODEL=<your HF repo>             # e.g. <user>/codebeast-3.8b
    # Optional brain fallbacks (auto-used when HF errors or credits run out):
    # NM_LLM_FALLBACK_CHAIN=groq,openrouter
    # NM_GROQ_API_KEY=...  NM_GROQ_MODEL=openai/gpt-oss-120b
    # NM_OPENROUTER_API_KEY=...  NM_OPENROUTER_MODEL=qwen/qwen3-8b:free
    HF_TOKEN=...                           # huggingface.co/settings/tokens
    NM_PARTNER_PLATFORMS=telegram-bot   # or: telegram,telegram-bot for +userbot
    NM_CHAT_TELEGRAM_BOT_ENABLED=true
    NM_CHAT_TELEGRAM_BOT_TOKEN=...         # from @BotFather
    NM_CHAT_TELEGRAM_BOT_CHATS=...         # owner chat id(s)

    # Optional userbot (your personal account via Telethon — needs
    # `pip install telethon` and a one-time interactive login):
    # NM_CHAT_TELEGRAM_ENABLED=true
    # NM_CHAT_TELEGRAM_API_ID=...          # from my.telegram.org
    # NM_CHAT_TELEGRAM_API_HASH=...
    # NM_CHAT_TELEGRAM_SESSION=~/.devon-telegram.session

    # Optional WhatsApp (your own account via the Node bridge — needs
    # `cd bridge && npm install` and a one-time QR scan):
    # NM_CHAT_WHATSAPP_ENABLED=true        # also add whatsapp to NM_PARTNER_PLATFORMS
    # and start the bridge: bash scripts/serv00/start_whatsapp.sh

The Telegram *bot* adapter long-polls ``getUpdates`` — no webhook, no inbound
ports, no public IP needed — so it works behind any NAT/shared host.

Configuration is 100% environment variables; never put tokens in this file.
See ``docs/SERV00_DEPLOY.md`` for the full Serv00 walkthrough.
"""

from __future__ import annotations

import collections
import logging
import os
import signal
import sys
import threading
import time


def _ensure_ssl_cert_file() -> None:
    """Point every HTTPS client in this process at certifi's CA bundle.

    Runs at import time, before any provider / adapter module is imported,
    so libraries that resolve their CA store from the environment
    (httpx, requests, urllib's default context) all see the same bundle.
    An explicitly set SSL_CERT_FILE is never overridden. When certifi is
    unavailable, nothing is changed and clients fall back to the platform
    store. Verification is never disabled here.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    try:
        cafile = certifi.where()
    except Exception:  # noqa: BLE001 - broken certifi: leave env alone
        return
    if cafile and os.path.isfile(cafile):
        os.environ["SSL_CERT_FILE"] = cafile


_ensure_ssl_cert_file()

# Make the repo importable without pip install: repo root is the parent of
# this script's directory (scripts/run_chat_bot.py -> <repo>/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _MessageHistory:
    """Ring buffer of inbound timestamps for the dashboard sparkline.

    Thread-safe. Buckets into per-minute counts for the last 30 minutes.
    """

    def __init__(self, minutes: int = 30) -> None:
        self.minutes = minutes
        self._lock = threading.Lock()
        self._stamps: collections.deque[float] = collections.deque(maxlen=4096)

    def record(self) -> None:
        with self._lock:
            self._stamps.append(time.time())

    def per_minute(self) -> list[int]:
        now = time.time()
        buckets = [0] * self.minutes
        with self._lock:
            stamps = list(self._stamps)
        for ts in stamps:
            age_min = int((now - ts) // 60)
            if 0 <= age_min < self.minutes:
                buckets[self.minutes - 1 - age_min] += 1
        return buckets


def _build_snapshot_provider(runtime, context, boot_mono, history: _MessageHistory):
    """Assemble the live-status snapshot for the console dashboard."""

    def _snapshot():
        snap: dict = {}
        snap["uptime_s"] = time.monotonic() - boot_mono
        # ── adapters ──
        adapters: dict = {}
        try:
            status = runtime.gateway.status()
            for name, info in status.items():
                if name.startswith("_"):
                    continue
                adapters[name] = {
                    "running": bool(info.get("running_in_session", True)),
                    "received": info.get("received", "—"),
                    "sent": info.get("sent", "—"),
                }
        except Exception:  # noqa: BLE001 - dashboard is best-effort
            pass
        snap["adapters"] = adapters
        # ── traffic ──
        try:
            with runtime._stats_lock:
                snap["traffic"] = dict(runtime.stats)
        except Exception:  # noqa: BLE001
            snap["traffic"] = getattr(runtime, "stats", {})
        # ── scheduler ──
        try:
            sched = getattr(runtime, "_scheduler", None)
            jobs = []
            running = False
            if sched is not None:
                running = bool(sched.running())
                for job in sched.list_jobs():
                    jobs.append(
                        {
                            "name": job.get("name") or job.get("id"),
                            "spec": job.get("spec") or job.get("schedule") or "",
                            "enabled": job.get("enabled", True),
                            "next_run": job.get("next_run"),
                        }
                    )
            snap["scheduler"] = {"running": running, "jobs": jobs}
        except Exception:  # noqa: BLE001
            snap["scheduler"] = {}
        # ── games ──
        try:
            row = context.db.query_one("SELECT COUNT(*) AS n FROM game_players")
            players = (row or {}).get("n", "—")
        except Exception:  # noqa: BLE001
            players = "—"
        snap["games"] = {"players": players}
        # ── activity sparkline ──
        try:
            snap["history"] = history.per_minute()
        except Exception:  # noqa: BLE001
            snap["history"] = []
        # ── brain (LLM router health) ──
        try:
            router = getattr(getattr(runtime, "_brain", None), "router", None) or getattr(
                runtime, "_router", None
            )
            if router is not None and hasattr(router, "stats_snapshot"):
                snap["llm"] = router.stats_snapshot()
            else:
                snap["llm"] = {}
        except Exception:  # noqa: BLE001
            snap["llm"] = {}
        # ── theme ──
        snap["theme"] = os.environ.get("NM_CONSOLE_THEME", "ocean")
        # ── extras ──
        extras: dict = {}
        try:
            extras["autonomy"] = "on" if getattr(runtime, "_autonomy", None) else "off"
        except Exception:  # noqa: BLE001
            pass
        try:
            extras["arena"] = "on" if getattr(runtime, "_arena", None) else "off"
        except Exception:  # noqa: BLE001
            pass
        snap["extras"] = extras
        return snap

    return _snapshot


def _print_banner(started, *, color=True):
    from nomorals.console.banner import render_banner
    from nomorals.console.themes import get_theme

    try:
        version = ""
        try:
            from nomorals import __version__  # type: ignore

            version = str(__version__)
        except Exception:  # noqa: BLE001 - version is cosmetic
            pass
        print(render_banner([str(s) for s in started], version=version, theme=get_theme(), color=color), flush=True)
    except Exception:  # noqa: BLE001 - banner must never break boot
        print("  DEVON · chat gateway live", flush=True)


def _install_console_mirror(runtime, history: _MessageHistory) -> None:
    """Rich inbound message cards on the local terminal + activity history.

    When a ``dashboard --watch`` session is active, messages are queued
    into the watch feed instead of printing — printing would flash over
    the live dashboard. Best-effort: any failure disables the mirror.
    """
    try:
        from nomorals.console.palette import supports_color
        from nomorals.console.widgets import (
            MessageEvent,
            WatchHub,
            format_message_card,
        )

        gateway = getattr(runtime, "gateway", None)
        if gateway is None:
            return
        color_ok = supports_color()

        def _event_from(message: object) -> MessageEvent:
            chat = getattr(message, "chat", None)
            return MessageEvent(
                platform=str(getattr(chat, "platform", "?") or "?"),
                sender=str(getattr(message, "sender", "?") or "?"),
                text=str(getattr(message, "text", "") or "")[:400],
                chat_title=str(getattr(chat, "title", "") or ""),
                timestamp=float(getattr(message, "ts", 0) or 0) or 0.0,
                incoming=bool(getattr(message, "incoming", True)),
            )

        def _mirror(message: object) -> None:
            history.record()
            try:
                event = _event_from(message)
            except Exception:  # noqa: BLE001 - mirror is cosmetic
                return
            # Watch mode owns the screen: queue, don't print.
            if WatchHub.is_active():
                WatchHub.feed().push(event)
                return
            if not color_ok:
                return
            try:
                card = format_message_card(
                    platform=event.platform,
                    sender=event.sender,
                    text=event.text,
                    chat_title=event.chat_title,
                    timestamp=event.timestamp or None,
                    incoming=event.incoming,
                    color=True,
                )
                print(f"\n{card}", flush=True)
            except Exception:  # noqa: BLE001 - mirror is cosmetic
                pass

        gateway.console_mirror = _mirror
    except Exception:  # noqa: BLE001 - mirror is optional
        pass


#: Main-thread shutdown wait cadence. ``threading.Event.wait()`` with NO
#: timeout blocks in ``WaitForSingleObject`` on Windows and the CPython
#: SIGINT handler never gets to run — Ctrl+C is deferred forever and the
#: bot can only be killed from a second window. Waking every 0.5s lets
#: the main thread run bytecode, process pending signals, and exit
#: promptly. (``time.sleep()`` is interruptible on Windows; an
#: indefinite ``Event.wait()`` is not.)
_SHUTDOWN_POLL_S = 0.5


def _install_signal_handlers(stop: threading.Event, log: logging.Logger) -> None:
    """Install SIGINT/SIGTERM (+SIGBREAK on Windows) shutdown handlers.

    Handlers do exactly one thing — set ``stop``. All real work (gateway
    stop, adapter close, context teardown) happens on the main thread
    in :func:`main`, never inside a signal handler. Installed BEFORE
    the runtime starts so Ctrl+C works from the first second.

    Safe to call from any thread: ``signal.signal`` raises ValueError
    off the main thread, which is swallowed (the bot then still stops
    via the wait loop + a second-window kill, same as before).
    """
    def _on_signal(signum: int, _frame: object) -> None:
        log.info("shutdown signal received (signal %s)", signum)
        stop.set()

    for _name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        _sig = getattr(signal, _name, None)
        if _sig is None:
            continue  # SIGBREAK is Windows-only; SIGTERM missing on some platforms
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError, RuntimeError) as exc:
            log.debug("shutdown handler for %s not installed: %s", _name, exc)


def _wait_for_shutdown(stop: threading.Event) -> None:
    """Block until ``stop`` is set, waking every :data:`_SHUTDOWN_POLL_S`.

    The short timeout is the Ctrl+C fix on Windows (see
    :data:`_SHUTDOWN_POLL_S`): an indefinite ``stop.wait()`` never
    returns to the interpreter, so a pending SIGINT handler never runs.
    """
    while not stop.is_set():
        stop.wait(timeout=_SHUTDOWN_POLL_S)


def _telegram_bot_start_message(started: list, gateway: object | None) -> str | None:
    """Warning when the telegram-bot adapter did not start.

    Reports the ACTUAL preflight failure reason (e.g. SSL certificate
    error), not a generic "set the token" — the token may already be set
    and the owner would otherwise chase a token that's already there.
    Returns None when telegram-bot started fine (no warning needed).
    """
    if "telegram-bot" in [str(s).lower() for s in started]:
        return None
    errors = getattr(gateway, "preflight_errors", None) or {}
    reason = errors.get("telegram-bot", "")
    if reason:
        return f"telegram-bot adapter did NOT start: {reason}"
    return (
        "telegram-bot adapter did NOT start — set "
        "NM_CHAT_TELEGRAM_BOT_ENABLED=true and "
        "NM_CHAT_TELEGRAM_BOT_TOKEN before launching"
    )


def main() -> int:
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.core.logging_setup import setup_logging

    # Colored, compact logs — no black backgrounds, no red text.
    setup_logging(level=os.environ.get("NM_LOG_LEVEL", "INFO"), color=True)
    log = logging.getLogger("run_chat_bot")

    from nomorals.agents.startup_progress import (
        StartupProgress, verify_stores)

    progress = StartupProgress()
    progress.start()
    progress.stage("starting")
    settings = None
    try:
        settings = load_settings()
        progress.stage("config loaded")
        # Ping FIRST — the owner DM hears "starting" before the heavy
        # context build, never after minutes of silence.
        progress.ping(settings, "Devon is starting… (config loaded)")
        boot_mono = time.monotonic()

        stop = threading.Event()

        # Handlers BEFORE the runtime starts, so Ctrl+C / Ctrl+Break work
        # from the first second — and the wait loop below wakes every
        # _SHUTDOWN_POLL_S so the main thread actually processes them
        # (an indefinite Event.wait() swallows Ctrl+C on Windows).
        _install_signal_handlers(stop, log)

        with build_context(settings) as context:
            # One OS Session per chat, across every platform (session_bridge):
            # the gateway attaches message.meta["os_session_id"] on inbound so
            # memory, persona, and missions key off the session, not the platform.
            from nomorals.os.session_bridge import SessionBridge

            bridge = SessionBridge(db=context.db)
            runtime = PartnerRuntime(context, session_bridge=bridge)
            started = runtime.start()
            progress.stage("adapters up")
            # Stores verified BEFORE ready: a store that cannot even
            # be read is a boot failure, reported honestly (DM +
            # console) instead of surfacing as weird behaviour later.
            _problems = verify_stores(context)
            if _problems:
                raise RuntimeError("stores failed verification: "
                                   + "; ".join(_problems))
            progress.stage("stores verified")
            _print_banner(started)
            log.info("adapters started: %s", started)
            msg = _telegram_bot_start_message(started, getattr(runtime, "gateway", None))
            if msg:
                log.warning("%s", msg)
            # Console-only commands (dashboard, status, jobs, …) for the local
            # terminal adapter. Anything else flows to the brain untouched.
            # Also: rich inbound mirror + activity history for the dashboard.
            history = _MessageHistory()
            _install_console_mirror(runtime, history)
            try:
                # Imported here (not at top of main): console commands are
                # optional — if the console package ever fails to import
                # (e.g. a Unix-only dep on Windows), the bot must still run.
                from nomorals.console import ConsoleCommands

                snapshot = _build_snapshot_provider(runtime, context, boot_mono, history)
                commands = ConsoleCommands(snapshot)
                local = None
                try:
                    local = runtime.gateway.adapters.get("local")
                except Exception:  # noqa: BLE001 - gateway internals are best-effort
                    local = None
                if local is not None:
                    local.command_hook = commands.handle
            except Exception:  # noqa: BLE001 - console commands are optional
                log.debug("console commands unavailable", exc_info=True)
            progress.stage("ready")
            # Ready ping — the owner hears the boot finished, not just
            # that it started. Best-effort: a failed ping never blocks.
            progress.ping(settings, "Devon is ready.")
            progress.stop()
            try:
                _wait_for_shutdown(stop)
            finally:
                try:
                    runtime.stop()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    pass
    except Exception as exc:  # noqa: BLE001 - honest failure, then raise
        progress.fail(settings, exc)
        raise
    print("Devon stopped.", flush=True)
    log.info("Devon stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
