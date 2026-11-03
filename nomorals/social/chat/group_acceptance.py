"""Join acceptance & admission rules (DESIGN §28).

Admission to a group is a rules system, not a door.  This organ
holds the per-group configuration shape and the PURE evaluation:
given a candidate's facts (what the platform actually reported
about them) it returns a verdict — admit, admit restricted, queue
for an admin, decline, kick — with the exact criteria that failed
and the line that tells the member what to fix.  The group surface
owns the state (restricted records, the request queue, the raid
posture, the /glog trail); the runtime owns execution through the
adapters' admin surfaces.

Honesty rules, stated once and held everywhere:

* Account age is NOT a criterion.  The Bot API does not expose
  it; an id-magnitude proxy would be a guess dressed as a fact.
* A criterion the platform cannot verify (the profile-photo
  lookup failed) is UNKNOWN, not failed: in request mode the
  candidate queues for a human; in open mode they are admitted
  and the gap is journaled — never restricted on a maybe.
* Bots are refused only when the group says so, and an admin can
  whitelist a specific bot (utility bots are legitimate members).

Raid posture (the defense interplay): a join burst tightens the
effective configuration — open behaves as request (clean joins
are held restricted for an admin's tap), auto-approve turns off,
and the restricted grace shrinks.  One config surface, tightened
by a state flag — never a second system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Every toggleable criterion, in evaluation order.
CRITERIA = ("username", "photo", "firstname", "lastname",
            "hygiene", "banned", "bots")

#: Enforcement actions a failure can take (globally or per
#: criterion).  ``queue`` is not an enforcement action — it is
#: what request mode does with anything that isn't a clean pass
#: or an outright decline.
ACTIONS = ("decline", "restricted", "kick")

ACCEPTANCE_DEFAULTS: dict[str, Any] = {
    "mode": "open",               # open | request | invite
    "criteria": {
        # ALL criteria ship OFF: acceptance is a configured
        # system (§28's first sentence — "group acceptance can
        # be configured").  Until a group turns a criterion on,
        # joins take the exact path they always took, and the
        # raid bookkeeping never engages.  (Defaults that quietly
        # changed every group's door would be a live surprise,
        # however well-meant.)
        "username": False,        # must have a username
        "photo": False,           # must have a profile photo
        "firstname": False,       # first name present, not
                                  # emoji-only
        "lastname": False,        # last name present
        "hygiene": False,         # names free of links/@handles/
                                  # filter words
        "banned": False,          # not previously banned here
        "bots": False,            # refuse bot accounts unless
                                  # whitelisted
    },
    "action": "restricted",       # global enforcement action
    "overrides": {},              # criterion -> action
    "grace_minutes": 60,          # restricted grace before a
                                  # kick-action escalates
    "auto_approve": True,         # request mode: clean requests
                                  # approve themselves
    "bot_whitelist": [],          # user ids or @usernames
    "raid_burst": 8,              # joins inside the window…
    "raid_window_s": 120,         # …that read as a raid
    "raid_cooldown_s": 600,       # posture holds this long
    "raid_grace_minutes": 5,      # shrunken grace during a raid
}

#: What the member is told to fix, per failed criterion.  Plain
#: words, the exact step where the platform has one.
FIX_LINES = {
    "username": "set a username (Telegram Settings → Username)",
    "photo": "add a profile photo",
    "firstname": "set a first name (letters, not only emoji)",
    "lastname": "set a last name (Telegram Settings → Edit profile)",
    "hygiene": "change your display name — it carries a link, "
               "an @handle, or a filtered word",
    "banned": "this account was banned from this group before",
    "bots": "bot accounts are not admitted here",
    "raid_hold": "the group is under a join rush — an admin "
                 "will release you shortly",
}

_LINK_RE = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)",
                      re.IGNORECASE)


def _emoji_only(text: str) -> bool:
    """True when a name carries no letters or digits at all —
    only emoji, symbols, spaces and punctuation."""
    return not any(ch.isalnum() for ch in str(text or ""))


def name_hygiene_failures(name: str, username: str,
                          filters: list[str]) -> list[str]:
    """Why a display name fails hygiene (empty list = clean).
    Rides the group's existing word-filter list (§28: the same
    machinery, applied to names)."""
    out: list[str] = []
    hay = f"{name or ''} {username or ''}".strip()
    if _LINK_RE.search(hay):
        out.append("link in name")
    if "@" in str(name or ""):
        out.append("@handle in name")
    lowered = hay.lower()
    for word in filters or []:
        w = str(word or "").strip().lower()
        if w and w in lowered:
            out.append(f"filtered word: {w}")
            break
    return out


@dataclass
class Evaluation:
    """One candidate judged against one group's rules."""

    verdict: str = "admit"        # admit | restricted | queue |
                                  # decline | kick
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    action: str = ""              # the enforcement action chosen
                                  # when the verdict enforces

    @property
    def clean(self) -> bool:
        return not self.missing and not self.unknown


def effective_cfg(cfg: dict[str, Any], *, raid: bool) -> dict[str, Any]:
    """The configuration as enforced right now: during a raid,
    request posture + no auto-approve + shrunken grace."""
    out = dict(cfg or {})
    if raid:
        out["auto_approve"] = False
        out["grace_minutes"] = min(
            int(out.get("grace_minutes") or 60),
            int(out.get("raid_grace_minutes") or 5))
        out["_raid"] = True
    return out


def evaluate(cfg: dict[str, Any], facts: dict[str, Any]) -> Evaluation:
    """Judge one candidate.  ``facts`` keys: user_id, username,
    first_name, last_name, name (combined fallback), is_bot,
    previously_banned (bool), ban_reason, whitelisted (bool).
    ``cfg`` is the group's acceptance block (already
    raid-effective where that applies).  The profile-photo fact
    arrives separately because it needs a platform lookup:
    ``facts['has_photo']`` is True/False, or None when the
    platform could not say."""
    criteria = dict(cfg.get("criteria") or {})
    overrides = dict(cfg.get("overrides") or {})
    global_action = str(cfg.get("action") or "restricted")
    if global_action not in ACTIONS:
        global_action = "restricted"

    failures: list[tuple[str, str]] = []   # (criterion, reason)
    unknown: list[str] = []

    username = str(facts.get("username") or "")
    first = str(facts.get("first_name") or "")
    last = str(facts.get("last_name") or "")
    combined = str(facts.get("name") or "")
    if not first and combined:
        parts = combined.split(None, 1)
        first = parts[0]
        if not last and len(parts) > 1:
            last = parts[1]

    if criteria.get("username") and not username:
        failures.append(("username", FIX_LINES["username"]))
    if criteria.get("photo"):
        has_photo = facts.get("has_photo", None)
        if has_photo is False:
            failures.append(("photo", FIX_LINES["photo"]))
        elif has_photo is None:
            unknown.append("photo")
    if criteria.get("firstname") and (not first.strip()
                                      or _emoji_only(first)):
        failures.append(("firstname", FIX_LINES["firstname"]))
    if criteria.get("lastname") and not last.strip():
        failures.append(("lastname", FIX_LINES["lastname"]))
    if criteria.get("hygiene"):
        bad = name_hygiene_failures(
            combined or f"{first} {last}".strip(), username,
            list(facts.get("filter_words") or []))
        if bad:
            failures.append(("hygiene",
                             f"{FIX_LINES['hygiene']} "
                             f"({'; '.join(bad)})"))
    if criteria.get("banned") and facts.get("previously_banned"):
        reason = FIX_LINES["banned"]
        if facts.get("ban_reason"):
            reason += f" ({facts['ban_reason']})"
        failures.append(("banned", reason))
    if criteria.get("bots") and facts.get("is_bot") \
            and not facts.get("whitelisted"):
        failures.append(("bots", FIX_LINES["bots"]))

    ev = Evaluation(unknown=unknown)
    if not failures:
        if unknown:
            ev.verdict = "queue"
            ev.reasons = [
                "couldn't verify: "
                + ", ".join(unknown)
                + " — a human decides"]
        return ev

    ev.missing = [key for key, _reason in failures]
    ev.reasons = [reason for _key, reason in failures]
    # The enforcement action: the STRONGEST per-criterion choice
    # among the failures (decline > kick > restricted), each
    # criterion falling back to the group's global action.
    rank = {"restricted": 1, "kick": 2, "decline": 3}
    chosen = "restricted"
    for key, _reason in failures:
        action = str(overrides.get(key) or global_action)
        if action not in ACTIONS:
            action = global_action
        if rank[action] > rank[chosen]:
            chosen = action
    ev.action = chosen
    ev.verdict = chosen
    return ev


def fix_it_text(reasons: list[str], *,
                grace_minutes: int | None = None) -> str:
    """The one message a restricted member gets (§28): exactly
    what to fix, and the button that re-checks.  Rendered by the
    caller through the typography organ; this is the wording."""
    lines = ["before you can take part here, please fix:"]
    for reason in reasons:
        lines.append(f"• {reason}")
    if grace_minutes:
        lines.append(
            f"you've got {grace_minutes} minutes before you're "
            f"removed — fix it, then tap “check me again”.")
    else:
        lines.append("fix it, then tap “check me again”.")
    return "\n".join(lines)
