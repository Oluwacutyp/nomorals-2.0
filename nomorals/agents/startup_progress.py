"""Startup progress + ping-first (infra spec §8.4).

Devon must never sit silent for minutes while starting:

- ``starting…`` prints the moment the process starts, before any
  heavy work.
- Per-stage progress: ``starting…`` → ``config loaded`` → ``adapters
  up`` → ``stores verified`` → ``ready``. Every stage goes to the
  console; the ping + ready stages also go to the owner DM.
- Ping first: the owner DM gets "Devon is starting…" BEFORE the
  heavy work (context build), via a minimal Telegram Bot API call
  that imports nothing heavy. Best-effort — a failed ping is logged
  and startup continues; the ping must never block the boot.
- "Still working" heartbeat: while a stage runs long, the console
  gets ``still working — <stage>…`` every ``heartbeat_s`` seconds
  (default 30) so a slow stage never looks hung.
- Honest failure: an exception at any stage sends the REAL cause
  (never generic, never empty, secrets redacted) to the owner DM +
  console, then re-raises. Startup never fails silently.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Callable

from ..core.errors import exc_str, redact_secrets

_log = logging.getLogger("nomorals.agents.startup_progress")

STAGES = ("starting", "config loaded", "adapters up",
          "stores verified", "ready")
DEFAULT_HEARTBEAT_S = 30.0
_PING_TIMEOUT_S = 10.0


def ping_owner_dm(settings: Any, text: str) -> bool:
    """Send one line to the owner DM via the Telegram Bot API.

    Reads the bot token + owner chat from settings (falling back to
    the env vars the settings loader reads). Returns True when the
    Bot API accepted the message, False otherwise — never raises.
    The token is never logged.
    """
    try:
        token = ""
        owner_chats = ""
        partner = getattr(settings, "partner", None)
        if partner is not None:
            owner_chats = str(getattr(partner, "owner_chats", "") or "")
        # The token comes from settings FIRST: load_settings reads the
        # .env FILE into the settings tree without exporting it to
        # os.environ, so an env-only read misses the token on exactly
        # the file-based installs this ping exists for. The env var is
        # the fallback (settings doubles in tests, env-only setups).
        chat = getattr(settings, "chat", None)
        if chat is not None:
            token = str(getattr(chat, "telegram_bot_token", "") or "").strip()
        if not token:
            token = (os.environ.get("NM_CHAT_TELEGRAM_BOT_TOKEN", "")
                     or "").strip()
        if not token:
            return False
        if not owner_chats.strip():
            owner_chats = os.environ.get("NM_PARTNER_OWNER_CHATS", "") or ""
        chat_id = ""
        for entry in owner_chats.split(","):
            entry = entry.strip()
            if not entry:
                continue
            # "telegram:123" or a bare chat id.
            chat_id = entry.rpartition(":")[2].strip() or entry
            if chat_id:
                break
        if not chat_id:
            return False
        payload = json.dumps({
            "chat_id": chat_id, "text": text,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=_PING_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        return bool(body.get("ok"))
    except Exception as exc:  # noqa: BLE001 - ping is best-effort
        _log.warning("startup ping failed: %s", exc_str(exc))
        return False


def verify_stores(context: Any) -> list[str]:
    """Check the stores boot depends on. Returns problem strings (empty = ok).

    Corrupt store files are NOT a problem — the stores degrade to
    defaults by design (corrupt → fresh, never crash). A problem is a
    store that cannot even be READ (permissions, missing home).
    """
    problems: list[str] = []
    try:
        from ..partner.chat_modes import get_chat_settings
        get_chat_settings("startup-probe")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        problems.append(f"chat-modes store unreadable: {exc_str(exc)}")
    try:
        from ..partner.style_overrides import get_overrides
        get_overrides("startup-probe")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        problems.append(f"style-overrides store unreadable: {exc_str(exc)}")
    db = getattr(context, "db", None)
    if db is not None:
        try:
            db.query("SELECT 1")
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            problems.append(f"database unreachable: {exc_str(exc)}")
    return problems


class StartupProgress:
    """Per-stage startup progress + heartbeat + honest failure."""

    def __init__(self, *, sink: Callable[[str], None] | None = None,
                 heartbeat_s: float = DEFAULT_HEARTBEAT_S) -> None:
        self._sink = sink or (lambda line: print(line, flush=True))
        self._heartbeat_s = max(1.0, heartbeat_s)
        self.stages: list[str] = []
        self._current = "starting"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _emit(self, line: str) -> None:
        try:
            self._sink(line)
        except Exception:  # noqa: BLE001 - progress must never break boot
            pass

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(timeout=self._heartbeat_s):
            self._emit(f"still working — {self._current}…")

    def start(self) -> None:
        """Begin the heartbeat; call before the first stage."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._heartbeat_loop,
                                        name="startup-heartbeat",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def stage(self, name: str) -> None:
        """Record + report one stage."""
        self._current = name
        self.stages.append(name)
        self._emit(f"Devon: {name}…")

    def ping(self, settings: Any, text: str) -> bool:
        """Best-effort owner-DM ping. Returns True when delivered."""
        delivered = ping_owner_dm(settings, text)
        if not delivered:
            _log.info("startup ping not delivered (no token/chat or "
                      "unreachable) — continuing boot")
        return delivered

    def fail(self, settings: Any, exc: BaseException) -> str:
        """Report an honest startup failure (DM + console) and stop.

        Returns the failure message. The caller re-raises the original
        exception — this never swallows it.
        """
        self.stop()
        cause = redact_secrets(exc_str(exc) or type(exc).__name__)
        message = (f"Devon failed to start at stage "
                   f"'{self._current}': {cause}. Fix it and relaunch — "
                   f"nothing is running.")
        _log.error("startup failed at %s: %s", self._current, cause)
        self._emit(f"Devon failed to start: {cause}")
        try:
            ping_owner_dm(settings, f"⚠️ {message}")
        except Exception:  # noqa: BLE001 - failure report is best-effort
            pass
        return message
