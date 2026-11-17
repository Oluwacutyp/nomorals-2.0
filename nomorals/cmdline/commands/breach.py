"""``nm breach`` — breach exposure from the console.

    nm breach ada@example.com
    nm breach password hunter2 [--json]
    nm breach watch add ada@example.com
    nm breach watch list | status | check
    nm breach watch remove ada@example.com

The same organ chat /breach uses: HIBP v3 for accounts (key-gated,
NM_OSINT_HIBP_KEY), Pwned Passwords k-anonymity for passwords (free —
only 5 hash characters ever leave the box). A password given on the
command line is checked and forgotten: it is never stored, never
logged, never returned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_breach"]


def _key(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return ""
    osint = getattr(settings, "osint", None)
    if osint is None:
        return ""
    return str(getattr(osint, "hibp_key", "") or "")


def _home(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _cmd_breach(args: argparse.Namespace, context: Any) -> int:
    from ...security.breach import (
        BreachWatchStore,
        check_account,
        check_password,
        format_breach_reply,
        run_breach_watch_tick_once,
    )

    target = " ".join(args.target or []).strip()
    as_json = bool(getattr(args, "json", False))
    if not target:
        print("breach: give me an email, a domain, or "
              "`password <password>` — nm breach ada@example.com",
              file=sys.stderr)
        return 2
    try:
        if target.lower().startswith("password "):
            result = check_password(target[9:].strip())
            if as_json:
                print(json.dumps(result, indent=2))
            else:
                print(format_breach_reply(result))
            return 0 if result.get("ok") else 1
        if target.lower().startswith("watch"):
            home = _home(context)
            if home is None:
                print("breach: no home directory", file=sys.stderr)
                return 1
            path = home / "data" / "breach_watch.json"
            store = BreachWatchStore(path)
            rest = target[5:].strip()
            wparts = rest.split(None, 1)
            wverb = wparts[0].lower() if wparts else "list"
            warg = wparts[1].strip() if len(wparts) > 1 else ""
            if wverb == "add":
                if not warg or "@" not in warg:
                    print("breach: watch add needs an email address",
                          file=sys.stderr)
                    return 2
                ok = store.add(warg)
                print(f"{warg.lower()} is now watched."
                      if ok else f"{warg.lower()} is already watched.")
                return 0
            if wverb == "remove":
                ok = store.remove(warg)
                print(f"no longer watching {warg.lower()}."
                      if ok else f"{warg.lower()} was not watched.")
                return 0
            if wverb == "list":
                watched = store.list_watched()
                if as_json:
                    print(json.dumps({"watched": watched}, indent=2))
                elif watched:
                    print("\n".join(watched))
                else:
                    print("no watched addresses yet.")
                return 0
            if wverb == "status":
                watched = store.list_watched()
                if as_json:
                    print(json.dumps({
                        "watched": watched,
                        "last_check": store.last_check(),
                        "seen": {addr: store.seen_names(addr)
                                 for addr in watched}}, indent=2))
                else:
                    print(f"watched: {len(watched)} address(es)")
                    for addr in watched:
                        print(f"  {addr} — "
                              f"{len(store.seen_names(addr))} known breach(es)")
                    last = store.last_check()
                    print(f"last check: {last if last else 'never'}")
                return 0
            if wverb == "check":
                result = run_breach_watch_tick_once(
                    context=context, notifier=None, state_path=path)
                if as_json:
                    print(json.dumps(result, indent=2))
                elif not result.get("ok"):
                    print(f"breach watch check failed: "
                          f"{result.get('error') or 'unknown error'}",
                          file=sys.stderr)
                elif result.get("new"):
                    print("new breach(es) on watched addresses:")
                    for name in result["new"]:
                        print(f"  • {name}")
                else:
                    print(f"watch check done — "
                          f"{len(result.get('checked') or [])} address(es) "
                          "checked, nothing new.")
                return 0 if result.get("ok") else 1
            print("breach: watch add|remove|list|status|check",
                  file=sys.stderr)
            return 2
        result = check_account(target, key=_key(context))
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            print(format_breach_reply(result))
        return 0 if result.get("ok") else 1
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        print(f"breach: {exc}", file=sys.stderr)
        return 1
