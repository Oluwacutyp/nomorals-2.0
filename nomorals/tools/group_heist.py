"""Group-heist tool: ``group_heist`` — the brain's hands on the
group crime layer: case the board, scout a target, start/join/run
a crew heist, rob a bank solo, and run the bunker.

One tool, action-dispatched, over
:mod:`nomorals.social.chat.group_surface`; the organs own the
math — the crew's combined gear against the target's installed
defenses, the odds printed before the run — this module is only
the surface.

``action=targets`` answers the casing board for one group
(``group=<name or key>``): treasury, district banks, open street
scores, vault count. ``action=scout`` reads one member's odds
against a target (``target=treasury|bank <district>|vault
<member>|convoy|hoard <id>``). ``action=start`` opens a crew
heist on a target; ``action=join`` joins the open lobby;
``action=go`` runs it. ``action=rob`` is the lone-wolf score
(``target=bank <district>``). ``action=bunker`` runs the hideout
(``bunker=buy 1|2|3|stash <n|all>|take <n|all>|hide|status``).

The actor is always the owner (``partner.owner_sender_ids``):
the brain never spends another member's money, never jails
another member's crew.  A group the brain cannot see is answered
honestly as nothing on record, never invented.
"""

from __future__ import annotations

import random
import time
from typing import Any

from ..core.policy import Capability
from ..social.chat.typography import sparks

__all__ = ["register"]


def _surface(context: Any) -> Any:
    from ..social.chat.group_surface import default_surface

    settings = getattr(context, "settings", None)
    ws_dir = getattr(settings, "workspace_dir", ".") or "."
    return default_surface(ws_dir)


def _find_groups(surface: Any, needle: str) -> list[str]:
    """Group keys whose title or key contains ``needle``."""
    needle = (needle or "").strip().lower()
    if not needle:
        return []
    matches = []
    for key in sorted(getattr(surface.store, "data", None) or {}):
        entry = surface.store.group(key)
        title = str(entry.get("title") or entry.get("name") or "")
        if needle in title.lower() or needle in str(key).lower():
            matches.append(str(key))
    return matches


def _owner_id(context: Any) -> str:
    """The owner's sender id — the tool's only actor."""
    settings = getattr(context, "settings", None)
    raw = ""
    if settings is not None and hasattr(settings, "get"):
        raw = str(settings.get("partner.owner_sender_ids") or "")
    for part in raw.replace(",", " ").split():
        if part.strip():
            return part.strip()
    return ""


def _resolve_member(surface: Any, key: str,
                    needle: str) -> list[str]:
    """Member ids in the group matching an id, @username or name
    substring — the vault target lookup for the tool."""
    needle = (needle or "").strip().lstrip("@").lower()
    if not needle:
        return []
    members = (surface.store.group(key).get("members") or {})
    found = []
    for mid, record in members.items():
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "")
        username = str(record.get("username") or "")
        if needle in (str(mid).lower(), username.lower()) \
                or needle in name.lower():
            found.append(str(mid))
    return found


def _parse_target(surface: Any, key: str, text: str,
                  ) -> tuple[str, str] | tuple[None, str]:
    """(kind, target_id) for the tool's ``target`` words, or
    (None, refusal)."""
    words = str(text or "").split()
    if not words:
        return None, "name a target: treasury, bank <district>, vault <member>, convoy <id>, hoard <id>."
    kind = words[0].lower()
    if kind == "treasury":
        return "treasury", ""
    if kind == "bank" and len(words) >= 2:
        return "bank", words[1].lower()
    if kind == "vault":
        if len(words) < 2:
            return None, "name the member whose vault: vault <member>."
        found = _resolve_member(surface, key, words[1])
        if not found:
            return None, f"no member in this group matches {words[1]!r}."
        if len(found) > 1:
            return None, f"more than one member matches {words[1]!r} — name one."
        return "vault", found[0]
    if kind in ("convoy", "hoard") and len(words) >= 2:
        return kind, words[1].lower()
    return None, ("name a target: treasury, bank <district>, vault "
                  "<member>, convoy <id>, hoard <id>.")


def _refusal(out: dict[str, Any]) -> str:
    """A surface refusal in plain words."""
    reason = str(out.get("reason") or "")
    if reason == "mode":
        return ("the streets are quiet here — conflict mode is "
                f"{out.get('mode') or 'off'}, and that needs "
                f"{out.get('need') or 'crime'}.")
    if reason == "off":
        return "the group economy is off here."
    if reason == "frozen":
        return "the conflict state is frozen — it needs an admin's eyes first."
    if reason == "jailed":
        return "you're in jail — bail out first."
    if reason == "hospital":
        return "you're in hospital — a medkit discharges you."
    if reason == "cooldown":
        return (f"too soon — the heat dies down in "
                f"~{max(1, int(out.get('wait_s') or 0) // 60)} min.")
    if reason == "self":
        return "not against yourself."
    if reason == "target":
        return "no such target here."
    if reason == "district":
        return "which bank? name a district."
    if reason == "tier":
        return ("the big scores are gang work — they need a gang "
                "of 5+ behind the crew.")
    if reason == "gone":
        return "that score is already gone."
    if reason == "expired":
        return "that score's window closed."
    if reason == "not_found":
        return "no heist is forming right now."
    if reason == "full":
        return "that crew is full (5 ride at most)."
    if reason == "already" or reason == "already_crew":
        return "you're already on this crew."
    if reason == "crew_small":
        return "a heist needs a crew of 2–5."
    if reason == "target_hidden":
        return "they're lying low in their bunker."
    if reason == "cap":
        return "the earn cap refuses that take — try again later."
    if reason == "treasury":
        return "the treasury can't cover that take right now."
    if reason == "none":
        return "you own no bunker yet — buy one first."
    if reason == "sparks":
        return "your wallet can't cover that."
    if reason == "gems":
        return "you don't hold the gems that needs."
    if reason == "empty":
        return "nothing there to take."
    if reason == "hidden":
        return "you're already lying low."
    return "that doesn't work as written."


def _render_targets(out: dict[str, Any]) -> str:
    treasury = out.get("treasury") or {}
    lines = [f"treasury: {sparks(int(treasury.get('available') or 0))} free · "
             f"defense {int(treasury.get('defense') or 0)}"]
    for bank in out.get("banks") or []:
        line = (f"{bank.get('name')} bank — defense "
                f"{int(bank.get('defense') or 0)}")
        if int(bank.get("hq_defense") or 0):
            line += f" · HQ +{int(bank['hq_defense'])}"
        lines.append(line)
    for score in out.get("scores") or []:
        lines.append(f"{score.get('name')} — "
                     f"{sparks(int(score.get('prize') or 0))} · "
                     f"defense {int(score.get('defense') or 0)} · "
                     f"{score.get('kind')} {score.get('id')}")
    if not (out.get("scores") or []):
        lines.append("no street score is open right now.")
    lines.append(f"member vaults: {int(out.get('vaults') or 0)} hold "
                 f"{sparks(int(out.get('vault_total') or 0))} in total.")
    if out.get("lobby") is not None:
        lines.append("a crew is forming.")
    return "\n".join(lines)


def _render_scout(kind: str, target_id: str,
                  out: dict[str, Any]) -> str:
    where = {"treasury": "the group treasury"}.get(
        kind, out.get("name") or target_id)
    lines = [f"{where} — solo read:",
             f"your offense {int(out.get('member_offense') or 0)} · "
             f"defense {int(out.get('defense') or 0)} · "
             f"odds {int(round(float(out.get('odds') or 0) * 100))}% · "
             f"offense for a coin flip "
             f"{int(out.get('needed_offense') or 0)}"]
    for note in out.get("notes") or []:
        lines.append(f"· {note}")
    if out.get("heat_stars"):
        lines.append(f"police heat on you: "
                     f"{float(out['heat_stars']):.0f}★.")
    return "\n".join(lines)


def _render_bunker(bunker: dict[str, Any] | None,
                   hidden: bool = False) -> str:
    if not bunker:
        return ("no bunker yet — 1 bolt-hole (200 sparks, holds "
                "500, hide 30 min), 2 safehouse (600 sparks, holds "
                "2,000, hide 2h), 3 bunker (1,500 sparks + 2 gems, "
                "holds 6,000, hide 8h).")
    text = (f"{bunker.get('name')} — holds "
            f"{sparks(int(bunker.get('sparks') or 0))} of "
            f"{sparks(int(bunker.get('cap') or 0))}.")
    if hidden:
        text += " Lying low — the police can't find you."
    return text


def register(registry: Any) -> None:
    """Attach the ``group_heist`` tool."""
    context = registry.context

    @registry.register(
        "group_heist",
        description=(
            "Group heists, solo bank robbery and bunkers: case the "
            "board (targets), scout one member's odds against a "
            "target (scout), open a crew heist (start), join the "
            "open lobby (join), run it (go), rob a district bank "
            "solo (rob), or run the hideout (bunker). The actor is "
            "always the owner; a group the brain cannot see is "
            "answered as nothing on record."
        ),
        capability=Capability.SOCIAL_BULK,
    )
    def group_heist_tool(
        action: str = "targets",
        group: str = "",
        target: str = "",
        bunker: str = "",
    ) -> dict[str, Any]:
        surface = _surface(context)
        owner = _owner_id(context)
        if not owner:
            return {"ok": False,
                    "error": "no owner sender id configured."}
        action = str(action or "targets").strip().lower()
        if action not in ("targets", "scout", "start", "join",
                          "go", "rob", "bunker"):
            return {"ok": False,
                    "error": f"unknown action {action!r}"}
        matches = _find_groups(surface, group)
        if not matches:
            return {"ok": True,
                    "text": (f"no record of a group matching "
                             f"{group!r}.")}
        if len(matches) > 1:
            names = ", ".join(
                str(surface.store.group(key).get("title") or key)
                for key in matches)
            return {"ok": True,
                    "text": (f"more than one group matches "
                             f"{group!r}: {names} — name one.")}
        key = matches[0]
        now = time.time()
        if action == "targets":
            out = surface.heist_targets(key, now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True, "text": _render_targets(out),
                    "targets": out}
        if action == "scout":
            kind, target_id = _parse_target(surface, key, target)
            if kind is None:
                return {"ok": True, "text": target_id}
            out = surface.heist_scout(key, owner, kind, target_id,
                                      now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True,
                    "text": _render_scout(kind, target_id, out),
                    "scout": out}
        if action == "start":
            kind, target_id = _parse_target(surface, key, target)
            if kind is None:
                return {"ok": True, "text": target_id}
            out = surface.heist_start(key, owner, kind, target_id,
                                      now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            lobby = out["heist"]
            crew = ", ".join(str(m) for m in
                             lobby.get("crew") or [])
            return {"ok": True,
                    "text": (f"crew forming on "
                             f"{kind}:{target_id or ''} — crew: "
                             f"{crew}. /heist join brings riders, "
                             f"/heist go runs it."),
                    "heist": lobby}
        if action == "join":
            out = surface.heist_join(key, owner, now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True,
                    "text": "you're on the crew.",
                    "heist": out["heist"]}
        if action == "go":
            out = surface.heist_go(key, owner,
                                   rng=random.Random(), now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            outcome = str(out.get("outcome") or "")
            take = sparks(int(out.get("take") or 0))
            if outcome == "success":
                text = f"the crew cracks it — {take} split."
            elif outcome == "partial":
                text = f"the crew gets in but runs early — {take} grabbed."
            elif outcome == "decoy":
                text = "a decoy vault full of confetti — the real money never moved."
            elif outcome == "empty":
                text = "the crew gets in — nothing worth taking."
            else:
                text = "bust — the whole crew is jailed for 45 minutes."
            return {"ok": True, "text": text, "result": out}
        if action == "rob":
            words = str(target or "").split()
            if not words or words[0].lower() != "bank" \
                    or len(words) < 2:
                return {"ok": True,
                        "text": "usage: rob target 'bank <district>'."}
            out = surface.rob_bank(key, owner, words[1].lower(),
                                   rng=random.Random(), now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            outcome = str(out.get("outcome") or "")
            if outcome == "success":
                text = (f"got away — "
                        f"{sparks(int(out.get('take') or 0))} lifted "
                        f"from the {words[1].lower()} bank, solo.")
            elif outcome == "empty":
                text = "you got in — the bank holds nothing worth taking."
            else:
                text = "caught robbing the bank — 30 minutes in jail."
            return {"ok": True, "text": text, "result": out}
        words = str(bunker or "").split()
        head = words[0].lower() if words else "status"
        if head == "buy" and len(words) >= 2:
            try:
                tier = int(words[1])
            except (TypeError, ValueError):
                return {"ok": True,
                        "text": "usage: bunker 'buy 1|2|3'."}
            out = surface.bunker_buy(key, owner, tier, now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True,
                    "text": _render_bunker(out["bunker"]),
                    "bunker": out["bunker"]}
        if head in ("stash", "take") and len(words) >= 2:
            if str(words[1]).lower() == "all":
                if head == "stash":
                    amount = int(surface.economy.balance(key,
                                                         owner))
                else:
                    held = surface.conflict.bunker_of(key, owner)
                    amount = int((held or {}).get("sparks") or 0)
            else:
                try:
                    amount = int(words[1])
                except (TypeError, ValueError):
                    return {"ok": True,
                            "text": f"usage: bunker '{head} <amount|all>'."}
            call = (surface.bunker_stash if head == "stash"
                    else surface.bunker_take)
            out = call(key, owner, amount, now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True,
                    "text": _render_bunker(
                        surface.conflict.bunker_of(key, owner)),
                    "bunker": surface.conflict.bunker_of(key,
                                                         owner)}
        if head == "hide":
            out = surface.bunker_hide(key, owner, now=now)
            if not out.get("ok"):
                return {"ok": True, "text": _refusal(out)}
            return {"ok": True,
                    "text": _render_bunker(
                        surface.conflict.bunker_of(key, owner),
                        hidden=True),
                    "bunker": surface.conflict.bunker_of(key,
                                                         owner)}
        out = surface.bunker_status(key, owner, now=now)
        if not out.get("ok"):
            return {"ok": True, "text": _refusal(out)}
        return {"ok": True,
                "text": _render_bunker(out.get("bunker"),
                                       hidden=bool(out.get("hidden"))),
                "bunker": out.get("bunker")}
