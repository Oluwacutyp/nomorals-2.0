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

import logging
import os
import signal
import sys
import threading
import time

# Make the repo importable without pip install: repo root is the parent of
# this script's directory (scripts/run_chat_bot.py -> <repo>/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_snapshot_provider(runtime, context, boot_mono):
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
            snap["games"] = {"players": (row or {}).get("n", "—")}
        except Exception:  # noqa: BLE001
            snap["games"] = {}
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
    from nomorals.console.palette import (
        ACCENT,
        BOLD,
        BRIGHT_WHITE,
        CYAN,
        DIM,
        GREEN,
        TITLE,
        paint,
    )

    bar = paint("─" * 46, DIM)
    print(bar)
    print(f"  {paint('D E V O N', TITLE + BOLD)} {paint('· chat gateway live', DIM)}")
    print(bar)
    for name in started:
        print(f"  {paint('●', GREEN)} {paint(name, BRIGHT_WHITE)} {paint('connected', DIM)}")
    print(f"  {paint('console commands:', ACCENT)} {paint('dashboard · status · jobs · clear · help', CYAN)}")
    print(bar, flush=True)


def main() -> int:
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.console import ConsoleCommands
    from nomorals.core.config import load_settings
    from nomorals.core.logging_setup import setup_logging

    # Colored, compact logs — no black backgrounds, no red text.
    setup_logging(level=os.environ.get("NM_LOG_LEVEL", "INFO"), color=True)
    log = logging.getLogger("run_chat_bot")

    settings = load_settings()
    boot_mono = time.monotonic()

    stop = threading.Event()

    def _on_signal(*_args: object) -> None:
        log.info("shutdown signal received")
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    with build_context(settings) as context:
        # One OS Session per chat, across every platform (session_bridge):
        # the gateway attaches message.meta["os_session_id"] on inbound so
        # memory, persona, and missions key off the session, not the platform.
        from nomorals.os.session_bridge import SessionBridge

        bridge = SessionBridge(db=context.db)
        runtime = PartnerRuntime(context, session_bridge=bridge)
        started = runtime.start()
        _print_banner(started)
        log.info("adapters started: %s", started)
        if "telegram-bot" not in [str(s).lower() for s in started]:
            log.warning(
                "telegram-bot adapter did NOT start — set "
                "NM_CHAT_TELEGRAM_BOT_ENABLED=true and "
                "NM_CHAT_TELEGRAM_BOT_TOKEN before launching"
            )
        # Console-only commands (dashboard, status, jobs, …) for the local
        # terminal adapter. Anything else flows to the brain untouched.
        try:
            snapshot = _build_snapshot_provider(runtime, context, boot_mono)
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
        try:
            stop.wait()
        finally:
            try:
                runtime.stop()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass
    log.info("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
