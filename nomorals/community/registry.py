"""Community tool registry: the closed allowlist of tools a community-facing
agent may call.

ALLOWLIST (the whole thing — nothing else is registered here):

* mini-app tools (capability ``community.miniapp``):
    ``miniapp_new`` · ``miniapp_list`` · ``miniapp_show`` · ``miniapp_vote``
    · ``miniapp_close`` · ``miniapp_expense`` · ``miniapp_settle``
    · ``miniapp_rsvp`` · ``miniapp_nudge`` · ``miniapp_rm``
* pure utilities (capability ``community.fun``):
    ``roll`` (Avrae-style dice: ``2d20k1``, ``4d6!``, ``d20 adv``,
    ``4dF``, ``2d8+1d6 # damage``) · ``coin`` (N flips, streaks) ·
    ``pick`` (random choice) · ``shuffle`` (random order)

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


def _rng(rng: Any = None) -> Any:
    """Injectable RNG (tests seed it); defaults to the global random."""
    return rng if rng is not None else random


# Dice grammar (Avrae/Sidekick-flavoured, chat-native):
#   2d6  1d20+3  2d8+1d6        composite terms
#   4d6k3  2d20kh1  3d6dl1      keep / drop highest / lowest
#   4d6!                       exploding dice (re-roll max, chain)
#   d20 adv / d20 dis           advantage / disadvantage (2d20, keep hi/lo)
#   4dF                        Fate dice (-/blank/+)
#   2d6 # damage               trailing description
_DICE_TERM = re.compile(r"""
    (?P<count>\d{1,3})? [dD]
    (?P<sides>\d{1,4}|[Ff])
    (?P<explode>!)?
    (?P<keepdrop>(?:k|kh|kl|d|dh|dl)\d{1,3})?
""", re.VERBOSE)
_DICE_SPLIT = re.compile(r"(?=[+-])")
_RNG_INJECT = "_rng"  # kwarg name tools accept for a seeded RNG


def _eval_term(term: str, rng: Any) -> tuple[list[int], list[int], int, str] | str:
    """One dice term → (all_rolls, kept_rolls, sides, kind) or error string."""
    m = _DICE_TERM.fullmatch(term.strip())
    if not m:
        return f"can't parse {term!r}"
    count = int(m.group("count") or 1)
    sides_raw = m.group("sides")
    explode = bool(m.group("explode"))
    kd = (m.group("keepdrop") or "").lower()
    if not (1 <= count <= 100):
        return "keep it sane: 1–100 dice per term"
    if sides_raw.upper() == "F":
        if kd or explode:
            return "Fate dice don't take keep/drop or exploding"
        rolls = [rng.choice((-1, 0, 1)) for _ in range(count)]
        kept = list(rolls)
        glyph = {1: "+", 0: "·", -1: "-"}
        detail = "[" + " ".join(glyph[r] for r in rolls) + "]"
        return rolls, kept, "F", detail
    sides = int(sides_raw)
    if not (2 <= sides <= 1000):
        return "keep it sane: d2–d1000"

    def one() -> int:
        return rng.randint(1, sides)

    rolls = [one() for _ in range(count)]
    # exploding: max rolls re-roll and chain (Roll20-style)
    if explode:
        i = 0
        while i < len(rolls):
            if rolls[i] == sides and len(rolls) < 500:
                rolls.append(one())
            i += 1
    kept = list(rolls)
    note = ""
    if kd:
        op, k = kd[0], int(kd[1:])
        k = min(k, len(kept))
        order = sorted(range(len(kept)), key=lambda i: kept[i])
        if op == "k":  # kh / k → keep highest k
            keep_idx = set(order[-k:]) if k else set()
            note = f"kh{k}"
        elif op == "d":  # dl / dh → drop highest/lowest k
            if kd.startswith("dh"):
                drop_idx = set(order[-k:])
                note = f"dh{k}"
            else:
                drop_idx = set(order[:k])
                note = f"dl{k}"
            keep_idx = set(range(len(kept))) - drop_idx
        kept = [kept[i] for i in sorted(keep_idx)]
        if not kept:
            kept = [0]
    return rolls, kept, sides, note


def _crit_note(sides: Any, kept: list[int]) -> str:
    if sides == 20 and len(kept) == 1:
        if kept[0] == 20:
            return " — **CRIT!** 🎯"
        if kept[0] == 1:
            return " — **fumble** 💀"
    return ""


def roll(spec: str = "1d6", **kwargs: Any) -> str:
    """Roll dice, Avrae-style.

    ``2d6`` · ``1d20+3`` · ``2d8+1d6 # damage`` (composite) ·
    ``4d6k3`` / ``2d20kh1`` / ``3d6dl1`` (keep/drop) ·
    ``4d6!`` (exploding) · ``d20 adv`` / ``d20 dis`` ·
    ``4dF`` (Fate). Pure randomness, no state.
    """
    rng = _rng(kwargs.get(_RNG_INJECT))
    raw = (spec or "").strip() or "1d6"
    desc = ""
    if "#" in raw:
        raw, _, desc = raw.partition("#")
        raw, desc = raw.strip(), desc.strip()[:60]
    adv = dis = False
    m = re.search(r"\b(adv|dis|advantage|disadvantage)\b", raw, re.I)
    if m:
        adv = m.group(1).lower().startswith("adv")
        dis = not adv
        raw = (raw[:m.start()] + raw[m.end():]).strip()
    if not raw:
        raw = "1d6"
    total = 0
    parts: list[str] = []
    crit_notes: list[str] = []
    for chunk in _DICE_SPLIT.split(raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        sign = 1
        if chunk[0] == "+":
            chunk = chunk[1:].strip()
        elif chunk[0] == "-":
            sign = -1
            chunk = chunk[1:].strip()
        if "d" in chunk.lower() or "D" in chunk:
            res = _eval_term(chunk, rng)
            if isinstance(res, str):
                return f"🎲 {res} — try like 2d6, 4d6k3, 2d8+1d6, or d20 adv."
            rolls, kept, sides, detail = res
            subtotal = sign * sum(kept)
            total += subtotal
            if sides == "F":
                parts.append(f"{'-' if sign < 0 else ''}{chunk}→{detail} ({subtotal:+d})")
            else:
                show = "+".join(map(str, rolls))
                if len(kept) != len(rolls):
                    show = f"[{show} keep {','.join(map(str, kept))}]"
                if detail:
                    show += f" ({detail})"
                parts.append(f"{'-' if sign < 0 else ''}{chunk}→{show}")
            crit_notes.append(_crit_note(sides, kept))
        else:
            try:
                total += sign * int(chunk)
                parts.append(f"{'-' if sign < 0 else '+'}{chunk}")
            except ValueError:
                return f"🎲 can't parse {chunk!r} in {spec!r}."
    if (adv or dis):
        # advantage/disadvantage: roll the whole expression twice, keep
        # the better (adv) or worse (dis) total — Avrae's behaviour.
        r1 = roll(raw, **{_RNG_INJECT: rng})
        r2 = roll(raw, **{_RNG_INJECT: rng})
        t1, t2 = _last_total(r1), _last_total(r2)
        if t1 is not None and t2 is not None:
            best = max(t1, t2) if adv else min(t1, t2)
            tag = "advantage" if adv else "disadvantage"
            label = f" ({desc})" if desc else ""
            return (f"🎲 {spec.strip()}{label} [{tag}]: {t1} vs {t2} → "
                    f"**{best}**")
    label = f" ({desc})" if desc else ""
    detail = " ".join(parts)
    crit = "".join(crit_notes)
    return f"🎲 {spec.strip()}{label} → {detail} = **{total}**{crit}"


def _last_total(rendered: str) -> int | None:
    m = re.search(r"\*\*(-?\d+)\*\*", rendered)
    return int(m.group(1)) if m else None


def coin(n: int = 1, **kwargs: Any) -> str:
    """Flip a coin — or N coins, with streaks called out."""
    rng = _rng(kwargs.get(_RNG_INJECT))
    try:
        n = max(1, min(50, int(n)))
    except (TypeError, ValueError):
        n = 1
    flips = [rng.choice(["Heads", "Tails"]) for _ in range(n)]
    if n == 1:
        return "🪙 " + flips[0]
    heads = flips.count("Heads")
    streak = 1
    for f in flips[1:]:
        if f == flips[0]:
            streak += 1
        else:
            break
    tail = f" — opens with {streak}× {flips[0]}" if streak >= 3 else ""
    return (f"🪙 {n} flips: {heads} Heads, {n - heads} Tails{tail}\n"
            f"   {' '.join('H' if f == 'Heads' else 'T' for f in flips)}")


def pick(choices: str, **kwargs: Any) -> str:
    """Pick one at random from a comma-separated list."""
    rng = _rng(kwargs.get(_RNG_INJECT))
    items = [c.strip() for c in (choices or "").split(",") if c.strip()]
    if not items:
        return 'Give me options: pick "pizza, sushi, tacos"'
    if len(items) == 1:
        return f"🎯 {items[0]} (only one option — bold choice)"
    return f"🎯 {rng.choice(items)}"


def shuffle(items: str, **kwargs: Any) -> str:
    """Random order for a comma-separated list (turn order, playlists…)."""
    rng = _rng(kwargs.get(_RNG_INJECT))
    lst = [c.strip() for c in (items or "").split(",") if c.strip()]
    if len(lst) < 2:
        return 'Give me at least 2 items: shuffle "a, b, c"'
    order = list(lst)
    rng.shuffle(order)
    return "🔀 " + " → ".join(order)


COMMUNITY_TOOL_NAMES: tuple[str, ...] = (
    "miniapp_new", "miniapp_list", "miniapp_show", "miniapp_vote",
    "miniapp_close", "miniapp_expense", "miniapp_settle", "miniapp_rsvp",
    "miniapp_nudge", "miniapp_rm",
    "roll", "coin", "pick", "shuffle",
)


def community_registry(context: Any = None) -> ToolRegistry:
    """Build the community tool registry — the allowlist, nothing more."""
    reg = ToolRegistry(context=context)
    miniapp_fns = {
        "miniapp_new": (miniapp_new, "Create a group mini-app (poll, quiz, expenses, or rsvp)."),
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
    reg.register("roll", roll,
                 description="Roll dice Avrae-style: 2d6, 4d6k3, 2d8+1d6 # dmg, 4d6!, d20 adv, 4dF.",
                 capability="community.fun")
    reg.register("coin", coin,
                 description="Flip a coin (or N coins).",
                 capability="community.fun")
    reg.register("pick", pick,
                 description='Pick one at random: pick "pizza, sushi, tacos".',
                 capability="community.fun")
    reg.register("shuffle", shuffle,
                 description='Random order: shuffle "a, b, c".',
                 capability="community.fun")
    return reg
