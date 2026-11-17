"""``nm burn`` — burn-after-reading notes from the console.

The console twin of ``/burn`` (Privnote build-new): the same
``memory/ephemeral_notes.py`` organ, the same words, the same
state. A note is sealed under a fresh random key; the link carries
the key and never lives in the store. The first read shows the
text and destroys it — a second read finds nothing.

* ``nm burn seal <text...> [--expire 1h|1d|7d|30d|never] [--password WORD]`` — seal a note, print the link
* ``nm burn read <code> [--password WORD]`` — read once (exit 1 when the note is gone, expired, or the code/password is wrong)
* ``nm burn list`` — live notes as metadata only, never contents
* ``nm burn purge <note-id>`` — delete an unread note early
* ``nm burn tick`` — run one expiry sweep now (the hourly job, on demand)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["_cmd_burn"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _when(epoch: float | None) -> str:
    if epoch is None:
        return "never"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
    except Exception:  # noqa: BLE001 - a bad timestamp is data
        return str(epoch)


def _take_flag(rest: list[str], flag: str) -> tuple[list[str], str | None]:
    """Pull a trailing ``--flag VALUE`` out of the word list."""
    out: list[str] = []
    value: str | None = None
    i = 0
    while i < len(rest):
        if rest[i] == flag and i + 1 < len(rest):
            value = rest[i + 1]
            i += 2
            continue
        out.append(rest[i])
        i += 1
    return out, value


def _cmd_burn(args: argparse.Namespace, context: Any) -> int:
    """``nm burn seal|read|list|purge|tick``."""
    from ...memory.ephemeral_notes import (
        BurnNoteStore,
        list_notes,
        parse_expiry,
        purge_note,
        read_note,
        run_burn_note_tick_once,
        seal_note,
    )

    home = _home_path(context)
    home_s = str(home) if home else None
    first = (getattr(args, "burn_action", None) or "").strip().lower()
    rest = list(getattr(args, "burn_rest", None) or [])
    as_json = bool(getattr(args, "json", False))

    if first == "tick":
        out = run_burn_note_tick_once(context)
        if as_json:
            print(json.dumps(out, indent=2))
        else:
            print(f"swept {out.get('swept', 0)} expired note(s); "
                  f"{out.get('live', 0)} live.")
        return 0 if out.get("ok") else 1

    if home_s is None:
        return _usage_error("nm burn needs a home (settings.home)")
    store = BurnNoteStore(home_s)

    if first == "seal":
        rest, expire = _take_flag(rest, "--expire")
        rest, password = _take_flag(rest, "--password")
        text = " ".join(rest).strip()
        if not text:
            return _usage_error(
                "usage: nm burn seal <text...> "
                "[--expire 1h|1d|7d|30d|never] [--password WORD]")
        try:
            seconds = parse_expiry(expire if expire else None)
        except ValueError as exc:
            return _usage_error(str(exc))
        try:
            sealed = seal_note(store, text, expires_in_s=seconds,
                               password=password or None)
        except ValueError as exc:
            return _usage_error(str(exc))
        if as_json:
            print(json.dumps(sealed, indent=2))
        else:
            print("sealed — read once with:")
            print(f"  nm burn read {sealed['link']}")
            print(f"  or in chat: /burn read {sealed['link']}")
            print(f"expires: {_when(sealed['expires_at'])}"
                  + (" · password-gated" if sealed["has_password"]
                     else ""))
            print("anyone holding the link can read it once — "
                  "the link is the key, so keep it off public chats.")
        return 0

    if first == "read":
        rest, password = _take_flag(rest, "--password")
        if not rest:
            return _usage_error("usage: nm burn read <code> "
                                "[--password WORD]")
        out = read_note(store, rest[0], password=password or None)
        if as_json:
            print(json.dumps(out, indent=2))
        elif out.get("ok"):
            print(out["text"])
            print("— note burned: it is gone now.")
        else:
            print(f"cannot read: {out.get('error')}", file=sys.stderr)
        return 0 if out.get("ok") else 1

    if first == "list":
        notes = list_notes(store)
        if as_json:
            print(json.dumps({"notes": notes}, indent=2))
        elif not notes:
            print("no live burn notes.")
        else:
            for meta in notes:
                print(f"{meta['note_id']} · created "
                      f"{_when(meta['created_at'])} · expires "
                      f"{_when(meta['expires_at'])}"
                      + (" · password-gated" if meta["has_password"]
                         else ""))
        return 0

    if first == "purge":
        if not rest:
            return _usage_error("usage: nm burn purge <note-id>")
        purged = purge_note(store, rest[0])
        if as_json:
            print(json.dumps({"ok": purged, "note_id": rest[0]}))
        elif purged:
            print(f"purged {rest[0]}.")
        else:
            print(f"no live note {rest[0]}.", file=sys.stderr)
        return 0 if purged else 1

    return _usage_error(
        "usage: nm burn seal <text...> | read <code> | list | "
        "purge <note-id> | tick")
