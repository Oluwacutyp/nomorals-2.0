#!/usr/bin/env python3
"""Long-running chat bot entrypoint.

Wires the partner runtime to whichever chat adapters are enabled in settings
and blocks forever. The intended shape on a small always-on host (Serv00,
a cheap VPS) is:

    NM_LLM_PROVIDER=hf_serverless          # brain = HF Inference API, no local model
    NM_HF_MODEL=<your HF repo>             # e.g. <user>/codebeast-3.8b
    HF_TOKEN=...                           # huggingface.co/settings/tokens
    NM_PARTNER_PLATFORMS=telegram-bot
    NM_CHAT_TELEGRAM_BOT_ENABLED=true
    NM_CHAT_TELEGRAM_BOT_TOKEN=...         # from @BotFather
    NM_CHAT_TELEGRAM_BOT_CHATS=...         # owner chat id(s)

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

# Make the repo importable without pip install: repo root is the parent of
# this script's directory (scripts/run_chat_bot.py -> <repo>/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_chat_bot")

    settings = load_settings()

    stop = threading.Event()

    def _on_signal(*_args: object) -> None:
        log.info("shutdown signal received")
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    with build_context(settings) as context:
        runtime = PartnerRuntime(context)
        started = runtime.start()
        print(f"adapters started: {started}", flush=True)
        log.info("adapters started: %s", started)
        if "telegram-bot" not in [str(s).lower() for s in started]:
            log.warning(
                "telegram-bot adapter did NOT start — set "
                "NM_CHAT_TELEGRAM_BOT_ENABLED=true and "
                "NM_CHAT_TELEGRAM_BOT_TOKEN before launching"
            )
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
