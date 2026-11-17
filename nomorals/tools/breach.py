"""Breach tools: ``breach_password`` + ``breach_watch``.

Thin surfaces over :mod:`nomorals.security.breach` — the organ owns
the k-anonymity password check, the HIBP account check and the
watcher. The email/domain check already exists as ``breach_check``
in ``tools/osint_people``; these two are the halves it never had.

Privacy, stated once: the password check sends only the first 5
characters of the password's SHA-1 to the range service. The
password itself and the full hash are never stored, never logged
and never returned — the tool audit records only an args digest,
never the argument values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _hibp_key_from_context(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return ""
    osint = getattr(settings, "osint", None)
    if osint is None:
        return ""
    return str(getattr(osint, "hibp_key", "") or "")


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _watch_path(context: Any) -> Path | None:
    home = _home_path(context)
    if home is None:
        return None
    return home / "data" / "breach_watch.json"


def register(registry: Any) -> None:
    """Attach the ``breach_password`` + ``breach_watch`` tools."""
    context = registry.context

    @registry.register(
        "breach_password",
        description=(
            "Has this password appeared in a known breach? Pwned "
            "Passwords k-anonymity: only the first 5 characters of the "
            "password's SHA-1 ever leave this box; the password itself "
            "and the full hash are never stored or logged. Free, no key. "
            "password=<the password to check>. Returns ok, pwned, count."
        ),
        capability=Capability.NET_OUT,
    )
    def breach_password_tool(password: str = "") -> dict[str, Any]:
        from ..security.breach import check_password

        if not password or not str(password).strip():
            raise ValueError("breach_password needs a password")
        return check_password(str(password))

    @registry.register(
        "breach_watch",
        description=(
            "Watch the owner's own addresses for new breaches: add or "
            "remove a watched address, list them, show the watch "
            "status, or run a check now. A new breach naming a watched "
            "address pages the owner once. action=<add|remove|list|"
            "status|check>; address=<email> for add/remove."
        ),
        capability=Capability.NET_OUT,
    )
    def breach_watch_tool(action: str = "list",
                          address: str = "") -> dict[str, Any]:
        from ..security.breach import (
            BreachWatchStore,
            run_breach_watch_tick_once,
        )

        action = (action or "list").strip().lower()
        path = _watch_path(context)
        if path is None:
            raise ValueError("breach_watch needs a home directory")
        store = BreachWatchStore(path)
        if action == "add":
            target = (address or "").strip().lower()
            if not target or "@" not in target:
                raise ValueError("breach_watch add needs an email address")
            return {"ok": store.add(target), "action": "add",
                    "address": target, "watched": store.list_watched()}
        if action == "remove":
            target = (address or "").strip().lower()
            return {"ok": store.remove(target), "action": "remove",
                    "address": target, "watched": store.list_watched()}
        if action == "list":
            return {"ok": True, "action": "list",
                    "watched": store.list_watched(),
                    "last_check": store.last_check()}
        if action == "status":
            watched = store.list_watched()
            return {"ok": True, "action": "status", "watched": watched,
                    "last_check": store.last_check(),
                    "seen": {addr: store.seen_names(addr)
                             for addr in watched}}
        if action == "check":
            # One tick, right now — the same code the daily job runs.
            result = run_breach_watch_tick_once(
                context=context, notifier=None, state_path=path)
            result["action"] = "check"
            return result
        raise ValueError(
            "breach_watch action must be add|remove|list|status|check")
