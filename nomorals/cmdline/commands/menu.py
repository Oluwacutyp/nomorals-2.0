"""``nm menu`` — Telegram "/" picker sync from the command registry."""
from __future__ import annotations

import json
import sys
from typing import Any

_BOT_NAMES = {"gated": "telegram-bot", "personal": "telegram-personal"}


def _cmd_menu(args: Any, context: Any) -> int:
    """Route ``nm menu sync``.

    Builds each selected bot's adapter from settings and pushes its
    registry-derived "/" list via setMyCommands (the same sync the
    adapter runs at startup). ``--dry-run`` builds and validates the
    list without calling the API. A bot with no token / disabled is an
    honest refusal naming the bot, never a silent skip.
    """
    words = list(getattr(args, "task", None) or [])
    if not words or words[0] != "sync":
        print("usage: nm menu sync [--bot gated|personal|both] [--dry-run] [--json]",
              file=sys.stderr)
        return 2
    bot_sel = str(getattr(args, "bot", "both") or "both")
    if bot_sel not in ("gated", "personal", "both"):
        print(f"menu: unknown bot {bot_sel!r} (gated|personal|both)",
              file=sys.stderr)
        return 2
    dry_run = bool(getattr(args, "dry_run", False))
    as_json = bool(getattr(args, "json", False))
    selected = (["telegram-bot", "telegram-personal"] if bot_sel == "both"
                else [_BOT_NAMES[bot_sel]])
    settings = getattr(context, "settings", None)
    from ...social.chat import build_adapter
    from ...social.chat.personal_commands import sync_bot_commands

    results: dict[str, Any] = {}
    rc = 0
    for name in selected:
        try:
            adapter = build_adapter(settings, name)
        except Exception as exc:  # noqa: BLE001 - bad token/config is a refusal
            results[name] = {
                "ok": False, "bot": name, "pushed": 0, "skipped": 0,
                "problems": [f"cannot build {name}: {exc}"],
            }
            rc = 1
            continue
        if adapter is None:
            results[name] = {
                "ok": False, "bot": name, "pushed": 0, "skipped": 0,
                "problems": [f"{name} is not enabled or has no token"],
            }
            rc = 1
            continue
        out = sync_bot_commands(adapter, dry_run=dry_run)
        results[name] = out
        if not out.get("ok"):
            rc = 1
    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return rc
    for name in selected:
        out = results[name]
        if out.get("ok"):
            count = len(out.get("commands") or [])
            skipped = out.get("skipped", 0)
            if dry_run:
                print(f"{name}: dry run — would push {count} commands "
                      f"({skipped} past the BotFather cap stay in /list)")
            else:
                print(f"{name}: pushed {count} commands "
                      f"({skipped} past the BotFather cap stay in /list)")
        else:
            print(f"{name}: refused — "
                  f"{'; '.join(out.get('problems') or ['?'])}")
    return rc
