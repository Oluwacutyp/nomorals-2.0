"""Community tool registry: the closed allowlist of tools a community-facing
agent may call.

ALLOWLIST (the whole thing — nothing else is registered here):

* mini-app tools (capability ``community.miniapp``):
    ``miniapp_new`` · ``miniapp_list`` · ``miniapp_show`` · ``miniapp_vote``
    · ``miniapp_close`` · ``miniapp_expense`` · ``miniapp_settle``
    · ``miniapp_rsvp`` · ``miniapp_nudge`` · ``miniapp_rm``
* pure utilities (capability ``community.fun``):
    ``roll`` (dice, e.g. ``2d6``) · ``coin`` (coin flip)

EXPLICITLY EXCLUDED: memory tools, vault/account tools, private
connectors, exec/shell, filesystem tools, network tools, model calls,
social post/DM tools, database tools. Every tool here is pure Python
over group-scoped mini-app state or randomness — there is no path from
this registry to owner data.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..tools.registry import ToolRegistry
from . import miniapps
from .policy import COMMUNITY_CAPABILITIES

__all__ = ["community_registry", "COMMUNITY_TOOL_NAMES"]


# ── mini-app tools (thin, explicit-context wrappers) ────────────────────────
# Each takes group_key / user explicitly: no ambient owner context leaks in.


def _store(data_dir: Any = None) -> miniapps.MiniAppStore:
    return miniapps.MiniAppStore(data_dir=data_dir) if data_dir else miniapps.MiniAppStore()


def miniapp_new(kind: str, group_key: str, title: str,
                options: list[str] | None = None, date: str = "",
                data_dir: Any = None) -> dict[str, Any]:
    """Create a group mini-app. Returns its id and a chat-ready render."""
    app = miniapps.create_miniapp(kind, group_key, title,
                                  options=options or [], date=date)
    _store(data_dir).put(app)
    return {"id": app.id, "render": miniapps.render_miniapp(app)}


def miniapp_list(group_key: str, data_dir: Any = None) -> list[dict[str, Any]]:
    """List mini-apps in a group."""
    return [{"id": a.id, "kind": a.kind, "title": a.title}
            for a in _store(data_dir).load(group_key)]


def miniapp_show(app_id: str, group_key: str, data_dir: Any = None) -> str:
    """Render one mini-app as chat text."""
    app = _store(data_dir).get(group_key, app_id)
    return miniapps.render_miniapp(app) if app else f"No mini-app {app_id!r} here."


def _act(app_id: str, group_key: str, user_id: str, user_name: str,
         action: str, args: list[str] | None, data_dir: Any = None) -> str:
    store = _store(data_dir)
    app = store.get(group_key, app_id)
    if app is None:
        return f"No mini-app {app_id!r} here."
    updated, reply = miniapps.apply_action(app, user_id, user_name, action, args or [])
    store.put(updated)
    return reply


def miniapp_vote(app_id: str, group_key: str, user_id: str, user_name: str,
                 option: int, data_dir: Any = None) -> str:
    """Vote in a poll (1-based option number; changeable until closed)."""
    return _act(app_id, group_key, user_id, user_name, "vote", [str(option)], data_dir)


def miniapp_close(app_id: str, group_key: str, user_id: str, user_name: str,
                  data_dir: Any = None) -> str:
    """Close a poll and show final results."""
    return _act(app_id, group_key, user_id, user_name, "close", [], data_dir)


def miniapp_expense(app_id: str, group_key: str, user_id: str, user_name: str,
                    amount: str, what: str, for_names: str = "",
                    data_dir: Any = None) -> str:
    """Record a shared expense. ``for_names``: comma-separated, empty = split with payer."""
    args = [amount, what + (f" for {for_names}" if for_names else "")]
    return _act(app_id, group_key, user_id, user_name, "expense", args, data_dir)


def miniapp_settle(app_id: str, group_key: str, user_id: str, user_name: str,
                   from_name: str, to_name: str, amount: str,
                   data_dir: Any = None) -> str:
    """Record a settlement payment between two people."""
    return _act(app_id, group_key, user_id, user_name, "settle",
                [from_name, to_name, amount], data_dir)


def miniapp_rsvp(app_id: str, group_key: str, user_id: str, user_name: str,
                 response: str, data_dir: Any = None) -> str:
    """RSVP yes/no/maybe to an event."""
    return _act(app_id, group_key, user_id, user_name, response, [], data_dir)


def miniapp_nudge(app_id: str, group_key: str, user_id: str, user_name: str,
                  data_dir: Any = None) -> str:
    """List who hasn't RSVP'd yet."""
    return _act(app_id, group_key, user_id, user_name, "nudge", [], data_dir)


def miniapp_rm(app_id: str, group_key: str, data_dir: Any = None) -> str:
    """Delete a mini-app from the group."""
    return "🗑️ Removed." if _store(data_dir).remove(group_key, app_id) else f"No mini-app {app_id!r} here."


# ── pure utilities ──────────────────────────────────────────────────────────


def roll(spec: str = "1d6") -> str:
    """Roll dice: 'd6', '2d6', '1d20+3'. Pure randomness, no state."""
    m = re.fullmatch(r"\s*(\d{1,3})?[dD](\d{1,4})\s*([+-]\s*\d{1,4})?\s*", spec or "")
    if not m:
        return f"Can't parse {spec!r} — try like 2d6 or 1d20+3."
    n = int(m.group(1) or 1)
    sides = int(m.group(2))
    mod = int(m.group(3).replace(" ", "")) if m.group(3) else 0
    if not (1 <= n <= 100 and 2 <= sides <= 1000):
        return "Keep it sane: up to 100 dice, up to d1000."
    rolls = [random.randint(1, sides) for _ in range(n)]
    total = sum(rolls) + mod
    detail = "+".join(map(str, rolls))
    if mod:
        detail += f"{mod:+d}"
    return f"🎲 {spec.strip()} → {detail} = **{total}**"


def coin() -> str:
    """Flip a coin."""
    return "🪙 " + random.choice(["Heads", "Tails"])


COMMUNITY_TOOL_NAMES: tuple[str, ...] = (
    "miniapp_new", "miniapp_list", "miniapp_show", "miniapp_vote",
    "miniapp_close", "miniapp_expense", "miniapp_settle", "miniapp_rsvp",
    "miniapp_nudge", "miniapp_rm", "roll", "coin",
)


def community_registry(context: Any = None) -> ToolRegistry:
    """Build the community tool registry — the allowlist, nothing more."""
    reg = ToolRegistry(context=context)
    miniapp_fns = {
        "miniapp_new": (miniapp_new, "Create a group mini-app (poll, expenses, or rsvp)."),
        "miniapp_list": (miniapp_list, "List the mini-apps in a group."),
        "miniapp_show": (miniapp_show, "Show one mini-app as chat text."),
        "miniapp_vote": (miniapp_vote, "Vote in a poll (1-based option number)."),
        "miniapp_close": (miniapp_close, "Close a poll and show final results."),
        "miniapp_expense": (miniapp_expense, "Record a shared expense in an expenses mini-app."),
        "miniapp_settle": (miniapp_settle, "Record a settlement payment between two people."),
        "miniapp_rsvp": (miniapp_rsvp, "RSVP yes/no/maybe to an event."),
        "miniapp_nudge": (miniapp_nudge, "List who hasn't RSVP'd yet."),
        "miniapp_rm": (miniapp_rm, "Delete a mini-app from the group."),
    }
    for name, (fn, desc) in miniapp_fns.items():
        reg.register(name, fn, description=desc, capability="community.miniapp")
    reg.register("roll", roll, description="Roll dice like 2d6 or 1d20+3.", capability="community.fun")
    reg.register("coin", coin, description="Flip a coin.", capability="community.fun")
    return reg
