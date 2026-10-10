"""``nm chat`` — inspect the conversation surfaces (chat adapters).

Verbs (the gateway must NOT be running — these are static checks and
durable state, not live control):

    nm chat platforms        capability matrix: what each platform can do
    nm chat doctor           per-platform setup checks (deps, env, bridge)
    nm chat outbox           WhatsApp outbox: queued sends from bridge outages
    nm chat outbox clear     drop everything still queued

The live loop is ``scripts/run_chat_bot.py`` (adapters) and the partner
runtime (brain); ``nm chat`` is the read side: what exists, what is
configured, what is waiting to be delivered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
from typing import Any

from ..emit import _emit

#: Static capability matrix — read off the adapter implementations in
#: nomorals/social/chat/. Keep this in sync when an adapter gains one.
CAPABILITY_MATRIX: list[dict[str, Any]] = [
    {
        "platform": "telegram",
        "label": "userbot (your own account, Telethon/MTProto)",
        "typing": True, "recording": True, "media": True,
        "voice_notes": True, "buttons": False, "groups": True,
        "threads": True, "receipts": False, "history": True, "outbox": False,
    },
    {
        "platform": "telegram-bot",
        "label": "BotFather bot (Bot API, long-polling)",
        "typing": True, "recording": True, "media": True,
        "voice_notes": True, "buttons": True, "groups": True,
        "threads": True, "receipts": False, "history": False, "outbox": False,
    },
    {
        "platform": "discord",
        "label": "your own account (discord.py user client)",
        "typing": True, "recording": False, "media": True,
        "voice_notes": False, "buttons": False, "groups": True,
        "threads": True, "receipts": False, "history": True, "outbox": False,
    },
    {
        "platform": "whatsapp",
        "label": "your own account (Node/Baileys bridge)",
        "typing": True, "recording": True, "media": True,
        "voice_notes": True, "buttons": False, "groups": True,
        "threads": False, "receipts": True, "history": True, "outbox": True,
    },
    {
        "platform": "sms",
        "label": "Twilio SMS fallback (opt-in, costs money)",
        "typing": False, "recording": False, "media": False,
        "voice_notes": False, "buttons": False, "groups": False,
        "threads": False, "receipts": False, "history": False, "outbox": False,
    },
    {
        "platform": "local",
        "label": "console (development)",
        "typing": False, "recording": False, "media": True,
        "voice_notes": False, "buttons": False, "groups": False,
        "threads": False, "receipts": False, "history": False, "outbox": False,
    },
    {
        "platform": "webhook",
        "label": "generic inbound webhook",
        "typing": False, "recording": False, "media": False,
        "voice_notes": False, "buttons": False, "groups": True,
        "threads": False, "receipts": False, "history": False, "outbox": False,
    },
]

_CAPABILITY_COLS = [
    ("typing", "typing"), ("recording", "rec"),
    ("media", "media"), ("voice_notes", "voice"),
    ("buttons", "btn"), ("groups", "grp"),
    ("threads", "thr"), ("receipts", "rcpt"),
    ("history", "hist"), ("outbox", "outbox"),
]


def _platforms_text() -> str:
    header = f"{'platform':13s} " + " ".join(f"{short:>6s}" for _, short in _CAPABILITY_COLS)
    lines = [header]
    for row in CAPABILITY_MATRIX:
        cells = " ".join(
            f"{('y' if row[key] else '-'):>6s}" for key, _ in _CAPABILITY_COLS)
        lines.append(f"{row['platform']:13s} {cells}")
    lines.append("")
    for row in CAPABILITY_MATRIX:
        lines.append(f"  {row['platform']:11s} {row['label']}")
    return "\n".join(lines)


def _check_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _doctor(settings: Any) -> list[dict[str, Any]]:
    """Per-platform setup checks. Each row: platform, ok, detail."""
    rows: list[dict[str, Any]] = []
    chat = getattr(settings, "chat", None) if settings is not None else None
    env = os.environ.get

    def _row(platform: str, ok: bool, detail: str) -> None:
        rows.append({"platform": platform, "ok": bool(ok), "detail": detail})

    # telegram userbot
    if chat is not None and getattr(chat, "telegram_enabled", False):
        if not _check_module("telethon"):
            _row("telegram", False, "enabled but telethon not installed")
        elif not getattr(chat, "telegram_api_id", ""):
            _row("telegram", False, "enabled but NM_CHAT_TELEGRAM_API_ID missing")
        else:
            _row("telegram", True, "enabled, telethon present")
    else:
        _row("telegram", True, "disabled (NM_CHAT_TELEGRAM_ENABLED)")

    # telegram bot
    if chat is not None and getattr(chat, "telegram_bot_enabled", False):
        token = str(getattr(chat, "telegram_bot_token", "") or "")
        if ":" not in token:
            _row("telegram-bot", False, "NM_CHAT_TELEGRAM_BOT_TOKEN looks wrong (no ':')")
        else:
            _row("telegram-bot", True, "enabled")
    else:
        _row("telegram-bot", True, "disabled (NM_CHAT_TELEGRAM_BOT_ENABLED)")

    # discord
    if chat is not None and getattr(chat, "discord_enabled", False):
        if not _check_module("discord"):
            _row("discord", False, "enabled but discord.py not installed")
        else:
            token = str(getattr(chat, "discord_token", "") or "")
            _row("discord", bool(token),
                 "enabled, discord.py present" if token else
                 "enabled but NM_CHAT_DISCORD_TOKEN missing")
    else:
        _row("discord", True, "disabled (NM_CHAT_DISCORD_ENABLED)")

    # whatsapp
    if chat is not None and getattr(chat, "whatsapp_enabled", False):
        host = str(getattr(chat, "whatsapp_host", "127.0.0.1") or "127.0.0.1")
        port = int(getattr(chat, "whatsapp_port", 8787) or 8787)
        node = shutil.which("node") is not None
        bridge = os.path.isfile(
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
                         "bridge", "whatsapp-bridge.mjs"))
        reachable = False
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            reachable = True
        except OSError:
            pass
        detail = (f"node={'yes' if node else 'NO'}, bridge={'yes' if bridge else 'NO'}, "
                  f"bridge@ {host}:{port}={'up' if reachable else 'DOWN'}")
        _row("whatsapp", node and bridge and reachable, detail)
    else:
        _row("whatsapp", True, "disabled (NM_CHAT_WHATSAPP_ENABLED)")

    # sms
    if chat is not None and getattr(chat, "sms_enabled", False):
        number = str(getattr(chat, "sms_from_number", "") or "")
        token = str(getattr(chat, "sms_auth_token", "") or "")
        if not number:
            _row("sms", False, "enabled but NM_CHAT_SMS_FROM_NUMBER missing")
        else:
            _row("sms", True, "enabled" + ("" if token else
                 " — WARNING: no NM_CHAT_SMS_AUTH_TOKEN, webhooks accepted unsigned"))
    else:
        _row("sms", True, "disabled (NM_CHAT_SMS_ENABLED)")

    # owner gating (the hard constraint)
    partner = getattr(settings, "partner", None) if settings is not None else None
    owner_chats = str(getattr(partner, "owner_chats", "") or "").strip() if partner else ""
    restricted = getattr(partner, "gate_restricted_chats", True) if partner else True
    _row("gating", bool(owner_chats) and restricted,
         f"owner_chats={'set' if owner_chats else 'EMPTY — nobody is gated as owner'}, "
         f"restricted_gating={'on' if restricted else 'OFF — kill-switch engaged'}")
    _ = env  # env is read via settings; kept for future direct-env checks
    return rows


def _doctor_text(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        mark = "ok " if row["ok"] else "FAIL"
        lines.append(f"[{mark}] {row['platform']:13s} {row['detail']}")
    return "\n".join(lines)


def _outbox_path(settings: Any) -> str:
    chat = getattr(settings, "chat", None) if settings is not None else None
    raw = str(getattr(chat, "whatsapp_outbox_dir", "data/chat/outbox")
              if chat is not None else "data/chat/outbox")
    resolve = getattr(settings, "resolve", None) if settings is not None else None
    try:
        return str(resolve(raw)) if resolve else raw
    except Exception:  # noqa: BLE001
        return raw


def _read_outbox(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001
        return []


def _cmd_chat(args: argparse.Namespace, context: Any) -> int:
    """``nm chat platforms|doctor|outbox [clear]`` — read side of the chat layer."""
    settings = getattr(context, "settings", None)
    task = " ".join(args.task).strip() if args.task else "platforms"
    verb = task.split()[0].lower() if task else "platforms"
    rest = task.split()[1:] if task else []

    if verb == "platforms":
        _emit(args, {"platforms": CAPABILITY_MATRIX}, _platforms_text())
        return 0

    if verb == "doctor":
        rows = _doctor(settings)
        _emit(args, {"checks": rows}, _doctor_text(rows))
        return 0 if all(r["ok"] for r in rows) else 1

    if verb == "outbox":
        path = os.path.join(_outbox_path(settings), "whatsapp.jsonl")
        if rest and rest[0] == "clear":
            try:
                if os.path.isfile(path):
                    os.remove(path)
                print("whatsapp outbox cleared", flush=True)
            except OSError as exc:
                print(f"could not clear outbox: {exc}", flush=True)
                return 1
            return 0
        entries = _read_outbox(path)
        if not entries:
            print("whatsapp outbox: empty (nothing queued)", flush=True)
            return 0
        lines = [f"whatsapp outbox: {len(entries)} queued (oldest first)"]
        import time as _time
        for entry in entries[:50]:
            age = _time.time() - float(entry.get("enqueued_at", _time.time()))
            kind = str(entry.get("kind") or "?")
            cid = str(entry.get("chat_id") or "?")
            text = str(entry.get("text") or entry.get("path") or "")
            lines.append(f"  [{age / 60:.0f}m] {kind:5s} -> {cid}: {text[:70]}")
        if len(entries) > 50:
            lines.append(f"  ... and {len(entries) - 50} more")
        text = "\n".join(lines)
        _emit(args, {"pending": len(entries),
                     "entries": entries[:50]}, text)
        return 0

    print("usage: nm chat platforms|doctor|outbox [clear]", flush=True)
    return 2
