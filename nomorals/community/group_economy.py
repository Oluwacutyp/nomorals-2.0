"""Group economy: points for real activity, inside one group chat.

Points ("sparks") are **game points only**. They have no monetary
value, are not redeemable for cash, are not transferable outside the
group, and are not a financial asset of any kind. The module says so
in every user-facing string, and the one place real money can appear
— a real prize — is gated behind explicit owner funding (see
:class:`PrizePool`).

Earning:
    Real activity earns: a helpful answer, hosting/attending an event,
    inviting someone who joins, a daily streak, completing a quest.
    Amounts live in :data:`QUESTS` / :data:`EARN_TABLE` as data, not
    scattered magic numbers.

Spending:
    The :data:`STATUS_SHOP` sells status, not power: a shout-out, a
    pinned message, a custom sticker/emoji, a DJ request, a
    party-theme vote, a nickname change. Status costs nothing to
    produce and is worth more than a fake sword — so the shop sells
    exactly that, and nothing pay-to-win.

Ledger hygiene:
    Append-only. Balances are always recomputed from the ledger, so
    every balance is reconstructible. Reversals never delete: a
    reversal is a new entry that references the original, and the
    original stays in the ledger with ``reversed_by`` set.

Transfers + treasury:
    Members can tip each other; the group treasury collects
    contributions and funds group-wide grants (party prizes, shout-out
    pools). Treasury moves are ledger entries too.

Seasons:
    Quarterly reset. Balances zero out via explicit closing entries
    (never deletion); top members carry over badges that survive the
    reset and show on their statement forever.

Anti-abuse:
    One account per verified member (unverified members earn nothing),
    per-period earn caps, and flags for alt farming (many accounts,
    one human-shaped pattern) and collusion (circular tipping rings).

Mining loop — honestly:
    Members can "mine" sparks by claiming. The claim rate decays with
    repeated claims (``MINING_DECAY``), the visible "hashrate" is just
    the member's engagement score (real activity, nothing else), and
    mining mints game points only. No real-world value is created or
    promised, ever.

Isolation: stdlib + ``nomorals.core`` only. No owner data, no
connectors, no accounts. Participants are group-local ids.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.group_economy")

__all__ = [
    "POINTS_NAME", "CURRENCY_DISCLAIMER", "EARN_TABLE", "QUESTS", "STATUS_SHOP",
    "EARN_CAPS", "MINING_BASE_RATE", "MINING_DECAY", "MINING_CLAIM_COOLDOWN_S",
    "LedgerEntry", "GroupEconomy", "PrizePool", "season_key", "current_season",
    "format_statement",
]

#: The unit. Always "sparks" in user-facing text — never "coins" (which
#: reads as currency) and never anything that sounds like money.
POINTS_NAME = "sparks"

#: The honesty line, reused by every user-facing render in this module.
CURRENCY_DISCLAIMER = (
    "Sparks are game points only. They have no monetary value, "
    "are not redeemable for cash, and are not a financial asset."
)

#: Fixed earn amounts per real activity. Data, not code paths.
EARN_TABLE: dict[str, int] = {
    "helpful_answer": 15,   # a member's answer marked helpful
    "event_host": 40,       # hosted a group event / party
    "event_attend": 10,     # showed up (RSVP yes + attended)
    "invite_joined": 25,    # invited someone who actually joined
    "streak_day": 5,        # one more day on a daily streak
    "trivia_win": 10,       # party trivia correct answer (base)
    "quest": 0,             # quests pay their own table amount (see QUESTS)
    "mining_claim": 0,      # mining pays the decaying claim rate
}

#: Per-member earn caps per period. Caps count real earns (quests,
#: activity, mining) — not transfers in, not treasury grants.
EARN_CAPS: dict[str, int] = {"daily": 200, "weekly": 800}

#: Quest table: id → (name, period, activity kind, count needed, reward).
#: ``period`` is "daily" or "weekly"; a quest is claimable once per period.
QUESTS: dict[str, dict[str, Any]] = {
    "helpful_3": {"name": "Helpful Hands", "period": "daily",
                  "kind": "helpful_answer", "need": 3, "reward": 30,
                  "blurb": "3 helpful answers in a day"},
    "invite_1": {"name": "Bring a Friend", "period": "weekly",
                 "kind": "invite_joined", "need": 1, "reward": 25,
                 "blurb": "Invite someone who joins this week"},
    "streak_7": {"name": "Week Warrior", "period": "weekly",
                 "kind": "streak_day", "need": 7, "reward": 50,
                 "blurb": "Active 7 days in a week"},
    "event_2": {"name": "Show Up", "period": "weekly",
                "kind": "event_attend", "need": 2, "reward": 40,
                "blurb": "Attend 2 events in a week"},
    "quest_master": {"name": "Quest Master", "period": "weekly",
                     "kind": "quest", "need": 3, "reward": 100,
                     "blurb": "Finish 3 quests in a week"},
}

#: Status shop: id → (name, cost, blurb). Status, not power.
STATUS_SHOP: dict[str, dict[str, Any]] = {
    "shout_out": {"name": "Shout-out", "cost": 30,
                  "blurb": "The bot shouts you out in the group"},
    "pinned_message": {"name": "Pinned Message", "cost": 60,
                       "blurb": "Your message pinned for 24h"},
    "custom_sticker": {"name": "Custom Sticker", "cost": 80,
                       "blurb": "A sticker made from your words"},
    "custom_emoji": {"name": "Custom Emoji", "cost": 80,
                     "blurb": "A group emoji in your honour"},
    "dj_request": {"name": "DJ Request", "cost": 50,
                   "blurb": "Your track jumps the party queue"},
    "party_theme_vote": {"name": "Party Theme Vote", "cost": 40,
                         "blurb": "Your theme pick counts double next party"},
    "nickname_change": {"name": "Nickname Change", "cost": 20,
                        "blurb": "A new group nickname for a week"},
}

#: Mining: first claim of the day pays this; each further claim that day
#: pays ``rate * MINING_DECAY ** claims_today`` (floor 1).
MINING_BASE_RATE = 10
MINING_DECAY = 0.85
MINING_CLAIM_COOLDOWN_S = 60.0  # seconds between claims — no claim spam

#: Anti-abuse: flags when a member's circular tipping volume passes this
#: in a rolling 24h window, or when too many accounts share one fingerprint.
COLLUSION_VOLUME_24H = 500
ALT_FARMING_ACCOUNTS = 3

_DATA_DIR = Path.home() / ".devon" / "community" / "group_economy"


def season_key(ts: float | None = None) -> str:
    """Quarterly season key, e.g. ``"2026-Q4"``."""
    t = time.localtime(ts if ts is not None else time.time())
    return f"{t.tm_year}-Q{(t.tm_mon - 1) // 3 + 1}"


def current_season(ts: float | None = None) -> str:
    return season_key(ts)


def _period_key(period: str, ts: float) -> str:
    t = time.localtime(ts)
    if period == "daily":
        return time.strftime("%Y-%m-%d", t)
    # weekly: ISO year-week — stable across month boundaries
    return f"{t.tm_year}-W{time.strftime('%V', t)}"


# ── ledger ───────────────────────────────────────────────────────────────


@dataclass
class LedgerEntry:
    """One append-only ledger line. Nothing here is ever deleted."""

    id: str = ""
    group_id: str = ""
    member: str = ""  # "" = the group treasury
    kind: str = ""  # earn | spend | transfer_out | transfer_in | treasury_in |
    # treasury_out | reversal | season_close | badge | mining
    amount: int = 0  # signed: + earns, - spends
    reason: str = ""
    ref: str = ""  # quest id / entry id / party id that caused this
    season: str = ""
    ts: float = 0.0
    reversed_by: str = ""  # reversal entry id, when this entry was reversed
    reverses: str = ""  # original entry id, when this entry IS a reversal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        return cls(
            id=str(data.get("id", "")),
            group_id=str(data.get("group_id", "")),
            member=str(data.get("member", "")),
            kind=str(data.get("kind", "")),
            amount=int(data.get("amount") or 0),
            reason=str(data.get("reason", "")),
            ref=str(data.get("ref", "")),
            season=str(data.get("season", "")),
            ts=float(data.get("ts") or 0.0),
            reversed_by=str(data.get("reversed_by", "")),
            reverses=str(data.get("reverses", "")),
        )


class GroupEconomy:
    """Group ledger + quests + shop + treasury + seasons + mining.

    All state is one JSON file per group under
    ``~/.devon/community/group_economy/``: ``ledger``, ``activities``,
    ``badges``, ``flags``, ``verified``, ``prizes``, ``mining``.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR

    # ── persistence ──
    def _path(self, group_id: str, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                       for c in str(group_id)).strip("_") or "group"
        return self.data_dir / safe / f"{name}.json"

    def _load_json(self, group_id: str, name: str, default: Any) -> Any:
        try:
            return json.loads(self._path(group_id, name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _save_json(self, group_id: str, name: str, data: Any) -> None:
        try:
            path = self._path(group_id, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            _log.warning("group economy save failed %s/%s: %s", group_id, name, exc)

    def _ledger(self, group_id: str) -> list[LedgerEntry]:
        out: list[LedgerEntry] = []
        for item in self._load_json(group_id, "ledger", []) or []:
            try:
                e = LedgerEntry.from_dict(item)
                if e.id:
                    out.append(e)
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    def _save_ledger(self, group_id: str, entries: list[LedgerEntry]) -> None:
        self._save_json(group_id, "ledger", [e.to_dict() for e in entries])

    def _append(self, group_id: str, entry: LedgerEntry) -> LedgerEntry:
        entries = self._ledger(group_id)
        entries.append(entry)
        self._save_ledger(group_id, entries)
        return entry

    def _new_entry(self, group_id: str, member: str, kind: str, amount: int,
                   reason: str, ref: str = "", ts: float | None = None,
                   reverses: str = "") -> LedgerEntry:
        now = ts if ts is not None else time.time()
        return LedgerEntry(
            id="led_" + new_short_id(), group_id=group_id, member=member,
            kind=kind, amount=int(amount), reason=(reason or "")[:300],
            ref=(ref or "")[:120], season=season_key(now), ts=now,
            reverses=reverses)

    # ── balances: always recomputed from the ledger ──
    def balance(self, group_id: str, member: str) -> int:
        """Balance reconstructed from the ledger — the only source of truth."""
        return sum(e.amount for e in self._ledger(group_id)
                   if e.member == member)

    def balances(self, group_id: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._ledger(group_id):
            out[e.member] = out.get(e.member, 0) + e.amount
        return {m: b for m, b in out.items() if b != 0}

    def verify(self, group_id: str) -> dict[str, Any]:
        """Recompute every balance from scratch; report the ledger size.

        There is no second source of truth to disagree with — verification
        is "recompute and show your work", which is what makes a balance
        disputable in the member's favour.
        """
        entries = self._ledger(group_id)
        return {"entries": len(entries),
                "balances": self.balances(group_id),
                "reversed": sum(1 for e in entries if e.reversed_by),
                "reconstructible": True}

    def statement(self, group_id: str, member: str) -> dict[str, Any]:
        """Per-member statement: entries + balance + badges + disclaimer."""
        entries = [e for e in self._ledger(group_id) if e.member == member]
        return {
            "member": member,
            "balance": sum(e.amount for e in entries),
            "entries": [e.to_dict() for e in sorted(entries, key=lambda e: e.ts)],
            "badges": self.badges(group_id, member),
            "disclaimer": CURRENCY_DISCLAIMER,
        }

    # ── verification (one account per verified member) ──
    def verify_member(self, group_id: str, member: str,
                      fingerprint: str = "") -> bool:
        """Mark a member verified. ``fingerprint`` is a group-local
        device/join fingerprint the host supplies — used only for
        alt-farming detection, never stored outside the group file."""
        verified = self._load_json(group_id, "verified", {}) or {}
        verified[str(member)] = {"fingerprint": hashlib.sha256(
            str(fingerprint or member).encode()).hexdigest()[:16],
            "at": time.time()}
        self._save_json(group_id, "verified", verified)
        self._check_alt_farming(group_id)
        return True

    def is_verified(self, group_id: str, member: str) -> bool:
        verified = self._load_json(group_id, "verified", {}) or {}
        return str(member) in verified

    # ── earning ──
    def _earned_in_period(self, group_id: str, member: str,
                          period: str, now: float) -> int:
        """Earns (earn/mining/quest kinds) in this daily/weekly period."""
        key = _period_key(period, now)
        total = 0
        for e in self._ledger(group_id):
            if e.member != member or e.amount <= 0:
                continue
            if e.kind not in ("earn", "mining"):
                continue
            if _period_key(period, e.ts) == key:
                total += e.amount
        return total

    def award(self, group_id: str, member: str, amount: int, reason: str,
              ref: str = "", *, kind: str = "earn",
              now: float | None = None) -> LedgerEntry | None:
        """Credit points for real activity. Returns None when blocked.

        Blocked when: unverified member, non-positive amount, or either
        earn cap (daily / weekly) would be exceeded. Caps are enforced
        here, in the one place all earns flow through.
        """
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        if not self.is_verified(group_id, member):
            return None
        for period, cap in EARN_CAPS.items():
            if self._earned_in_period(group_id, member, period, now) + amount > cap:
                self._flag(group_id, member, f"earn_cap_{period}",
                           f"earn of {amount} would exceed {period} cap {cap}")
                return None
        return self._append(group_id, self._new_entry(
            group_id, member, kind, amount, reason, ref=ref, ts=now))

    def record_activity(self, group_id: str, member: str, kind: str,
                        now: float | None = None) -> LedgerEntry | None:
        """Record real activity AND pay its earn-table amount, if any.

        Activity is the raw material for quests, streaks, and mining
        hashrate. Unknown kinds are recorded but pay nothing.
        """
        now = now if now is not None else time.time()
        activities = self._load_json(group_id, "activities", []) or []
        activities.append({"member": member, "kind": kind, "ts": now})
        self._save_json(group_id, "activities", activities[-5000:])
        amount = EARN_TABLE.get(kind, 0)
        if amount > 0:
            return self.award(group_id, member, amount,
                              f"activity: {kind}", ref=kind, now=now)
        return None

    def activity_count(self, group_id: str, member: str, kind: str,
                       period: str, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        key = _period_key(period, now)
        return sum(1 for a in (self._load_json(group_id, "activities", []) or [])
                   if a.get("member") == member and a.get("kind") == kind
                   and _period_key(period, float(a.get("ts") or 0)) == key)

    # ── quests ──
    def quest_progress(self, group_id: str, member: str,
                       now: float | None = None) -> dict[str, dict[str, Any]]:
        now = now if now is not None else time.time()
        claims = self._load_json(group_id, "quest_claims", {}) or {}
        out: dict[str, dict[str, Any]] = {}
        for qid, q in QUESTS.items():
            have = self.activity_count(group_id, member, q["kind"],
                                       q["period"], now)
            if q["kind"] == "quest":
                have = sum(1 for e in self._ledger(group_id)
                           if e.member == member and e.kind == "earn"
                           and e.ref.startswith("quest:")
                           and _period_key(q["period"], e.ts) == _period_key(q["period"], now))
            claimed = claims.get(f"{member}:{qid}:{_period_key(q['period'], now)}", False)
            out[qid] = {"name": q["name"], "blurb": q["blurb"],
                        "have": have, "need": q["need"],
                        "reward": q["reward"], "complete": have >= q["need"],
                        "claimed": bool(claimed)}
        return out

    def claim_quest(self, group_id: str, member: str, quest_id: str,
                    now: float | None = None) -> LedgerEntry | None:
        """Claim a completed quest's reward. Once per period, never twice."""
        now = now if now is not None else time.time()
        q = QUESTS.get(quest_id)
        if q is None:
            return None
        prog = self.quest_progress(group_id, member, now).get(quest_id) or {}
        if not prog.get("complete") or prog.get("claimed"):
            return None
        entry = self.award(group_id, member, int(q["reward"]),
                           f"quest: {q['name']}", ref=f"quest:{quest_id}", now=now)
        if entry is None:
            return None  # cap hit — the quest stays claimable next period
        claims = self._load_json(group_id, "quest_claims", {}) or {}
        claims[f"{member}:{quest_id}:{_period_key(q['period'], now)}"] = True
        self._save_json(group_id, "quest_claims", claims)
        return entry

    # ── spending: the status shop ──
    def buy_status(self, group_id: str, member: str, item_id: str,
                   now: float | None = None) -> dict[str, Any] | None:
        """Buy a status item. Returns the grant record or None.

        The grant is what the chat layer acts on (shout-out, pin, …).
        Nothing here confers gameplay power — status only.
        """
        now = now if now is not None else time.time()
        item = STATUS_SHOP.get(item_id)
        if item is None:
            return None
        if self.balance(group_id, member) < int(item["cost"]):
            return None
        entry = self._append(group_id, self._new_entry(
            group_id, member, "spend", -int(item["cost"]),
            f"status shop: {item['name']}", ref=f"shop:{item_id}", ts=now))
        return {"item": item_id, "name": item["name"], "cost": item["cost"],
                "member": member, "ledger_id": entry.id,
                "granted_at": now, "disclaimer": CURRENCY_DISCLAIMER}

    # ── transfers + treasury ──
    def transfer(self, group_id: str, sender: str, recipient: str, amount: int,
                 note: str = "", now: float | None = None) -> bool:
        """Tip another member. Both sides are ledger entries; never a delete."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or sender == recipient or not sender or not recipient:
            return False
        if self.balance(group_id, sender) < amount:
            return False
        self._append(group_id, self._new_entry(
            group_id, sender, "transfer_out", -amount,
            f"tip to {recipient}: {note}"[:300], ref=f"to:{recipient}", ts=now))
        self._append(group_id, self._new_entry(
            group_id, recipient, "transfer_in", amount,
            f"tip from {sender}: {note}"[:300], ref=f"from:{sender}", ts=now))
        self._check_collusion(group_id, now)
        return True

    def treasury_balance(self, group_id: str) -> int:
        return self.balance(group_id, "")

    def contribute_treasury(self, group_id: str, member: str, amount: int,
                            now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or self.balance(group_id, member) < amount:
            return False
        self._append(group_id, self._new_entry(
            group_id, member, "treasury_in", -amount,
            "contributed to group treasury", ref="treasury", ts=now))
        self._append(group_id, self._new_entry(
            group_id, "", "treasury_in", amount,
            f"contribution from {member}", ref=f"from:{member}", ts=now))
        return True

    def treasury_grant(self, group_id: str, recipient: str, amount: int,
                       reason: str = "", now: float | None = None) -> bool:
        """Pay out of the treasury (party prize, shout-out pool, …)."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or self.treasury_balance(group_id) < amount:
            return False
        self._append(group_id, self._new_entry(
            group_id, "", "treasury_out", -amount,
            f"treasury grant to {recipient}: {reason}"[:300],
            ref=f"to:{recipient}", ts=now))
        self._append(group_id, self._new_entry(
            group_id, recipient, "treasury_out", amount,
            f"treasury grant: {reason}"[:300], ref="treasury", ts=now))
        return True

    # ── reversals: never delete ──
    def reverse(self, group_id: str, entry_id: str, reason: str,
                now: float | None = None) -> LedgerEntry | None:
        """Reverse an entry. The original stays; a reversal entry offsets it.

        Returns the reversal entry, or None when the entry doesn't exist
        or was already reversed. Reversing a reversal is refused.
        """
        now = now if now is not None else time.time()
        entries = self._ledger(group_id)
        target = next((e for e in entries if e.id == entry_id), None)
        if target is None or target.reversed_by or target.kind == "reversal":
            return None
        reversal = self._new_entry(
            group_id, target.member, "reversal", -target.amount,
            f"reversal: {reason}"[:300], ref=target.id, ts=now,
            reverses=target.id)
        target.reversed_by = reversal.id
        entries.append(reversal)
        self._save_ledger(group_id, entries)
        return reversal

    # ── seasons ──
    def badges(self, group_id: str, member: str = "") -> list[dict[str, Any]]:
        all_badges = self._load_json(group_id, "badges", []) or []
        if member:
            return [b for b in all_badges if b.get("member") == member]
        return list(all_badges)

    def close_season(self, group_id: str, season: str = "",
                     now: float | None = None) -> dict[str, Any]:
        """Quarterly reset. Balances zero via closing entries; badges carry over.

        Top 3 members by balance earn season badges that survive the
        reset. Nothing is deleted — the closing entries are part of the
        permanent ledger, so any past balance stays reconstructible.
        """
        now = now if now is not None else time.time()
        season = season or season_key(now)
        entries = self._ledger(group_id)
        balances = self.balances(group_id)
        ranked = sorted(((m, b) for m, b in balances.items() if m),
                        key=lambda kv: -kv[1])
        badges = self._load_json(group_id, "badges", []) or []
        medals = ["Season Champion", "Season Runner-up", "Season Third"]
        for i, (member, _) in enumerate(ranked[:3]):
            badges.append({"member": member, "badge": medals[i],
                           "season": season, "at": now})
        self._save_json(group_id, "badges", badges)
        closed: list[dict[str, Any]] = []
        for member, bal in list(balances.items()):
            if bal == 0:
                continue
            closing = self._new_entry(
                group_id, member, "season_close", -bal,
                f"season {season} close: balance reset", ref=season, ts=now)
            entries.append(closing)
            closed.append({"member": member, "closed": bal})
        self._save_ledger(group_id, entries)
        return {"season": season, "closed": closed,
                "badges": badges[-len(ranked[:3]):] if ranked else [],
                "disclaimer": CURRENCY_DISCLAIMER}

    # ── anti-abuse ──
    def flags(self, group_id: str) -> list[dict[str, Any]]:
        return list(self._load_json(group_id, "flags", []) or [])

    def _flag(self, group_id: str, member: str, kind: str, detail: str) -> None:
        flags = self._load_json(group_id, "flags", []) or []
        flags.append({"member": member, "kind": kind, "detail": detail[:300],
                      "at": time.time()})
        self._save_json(group_id, "flags", flags[-500:])

    def _check_alt_farming(self, group_id: str) -> None:
        verified = self._load_json(group_id, "verified", {}) or {}
        by_fp: dict[str, list[str]] = {}
        for member, rec in verified.items():
            by_fp.setdefault(str(rec.get("fingerprint", "")), []).append(member)
        for fp, members in by_fp.items():
            if len(members) >= ALT_FARMING_ACCOUNTS and fp:
                for m in members:
                    self._flag(group_id, m, "alt_farming",
                               f"{len(members)} verified accounts share one fingerprint")

    def _check_collusion(self, group_id: str, now: float) -> None:
        """Flag circular tipping: A→B and B→A within 24h above the volume bar."""
        entries = [e for e in self._ledger(group_id)
                   if e.kind in ("transfer_out", "transfer_in")
                   and now - e.ts <= 86400]
        pair_volume: dict[tuple[str, str], int] = {}
        for e in entries:
            if e.kind == "transfer_out" and e.ref.startswith("to:"):
                other = e.ref[3:]
                key = tuple(sorted((e.member, other)))
                pair_volume[key] = pair_volume.get(key, 0) + abs(e.amount)
        for (a, b), vol in pair_volume.items():
            if vol >= COLLUSION_VOLUME_24H:
                self._flag(group_id, a, "collusion",
                           f"circular tipping with {b}: {vol} {POINTS_NAME} in 24h")
                self._flag(group_id, b, "collusion",
                           f"circular tipping with {a}: {vol} {POINTS_NAME} in 24h")

    # ── mining loop (honest) ──
    def hashrate(self, group_id: str, member: str,
                 now: float | None = None) -> dict[str, Any]:
        """Visible hashrate = engagement. Nothing else. No hardware, no luck.

        Engagement = real activity in the last 7 days (helpful answers,
        events, invites, quests). The number is shown to the member next
        to their claim rate so the loop is legible.
        """
        now = now if now is not None else time.time()
        week_ago = now - 7 * 86400
        activities = self._load_json(group_id, "activities", []) or []
        recent = [a for a in activities
                  if a.get("member") == member and float(a.get("ts") or 0) >= week_ago]
        weights = {"helpful_answer": 3, "event_host": 5, "event_attend": 2,
                   "invite_joined": 4, "quest": 4, "streak_day": 1}
        score = sum(weights.get(str(a.get("kind")), 1) for a in recent)
        return {"member": member, "hashrate": score,
                "activity_7d": len(recent),
                "note": "Hashrate is engagement: real activity in the last 7 days. "
                        "It boosts nothing except the visible number."}

    def claim_rate(self, group_id: str, member: str,
                   now: float | None = None) -> int:
        """Decaying claim rate: each claim today pays less than the last."""
        now = now if now is not None else time.time()
        day = _period_key("daily", now)
        claims_today = sum(1 for e in self._ledger(group_id)
                           if e.member == member and e.kind == "mining"
                           and _period_key("daily", e.ts) == day)
        return max(1, round(MINING_BASE_RATE * (MINING_DECAY ** claims_today)))

    def mining_claim(self, group_id: str, member: str,
                     now: float | None = None) -> LedgerEntry | None:
        """Claim mined points at the decaying rate. Cooldown + caps apply.

        Mining mints game points only. The rate decays within the day so
        grinding claims is pointless; the hashrate display is engagement,
        not a multiplier.
        """
        now = now if now is not None else time.time()
        mining = self._load_json(group_id, "mining", {}) or {}
        last = float((mining.get(member) or {}).get("last_claim") or 0)
        if now - last < MINING_CLAIM_COOLDOWN_S:
            return None
        rate = self.claim_rate(group_id, member, now)
        entry = self.award(group_id, member, rate,
                           "mining claim (game points only)",
                           ref="mining", kind="mining", now=now)
        if entry is None:
            return None  # cap hit or unverified
        mining[member] = {"last_claim": now}
        self._save_json(group_id, "mining", mining)
        return entry


class PrizePool:
    """Real prizes — the ONE place real value can appear, and only with
    explicit owner funding.

    A prize is created unfunded. It becomes redeemable only when the
    owner funds it explicitly (``funded_by_owner=True`` plus a written
    ``owner_approval`` record: who approved, what, when). Until then,
    :meth:`redeem` refuses honestly. Points themselves are never a
    prize and never convert to anything — a prize is a separate,
    owner-paid thing a member can win, e.g. "owner buys the winner a
    pizza". The economy never promises, implies, or prices a prize.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR

    def _path(self, group_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                       for c in str(group_id)).strip("_") or "group"
        return self.data_dir / safe / "prizes.json"

    def _load(self, group_id: str) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self._path(group_id).read_text(encoding="utf-8"))
            return list(raw) if isinstance(raw, list) else []
        except (OSError, ValueError):
            return []

    def _save(self, group_id: str, prizes: list[dict[str, Any]]) -> None:
        try:
            path = self._path(group_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(prizes, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            _log.warning("prize save failed for %s: %s", group_id, exc)

    def create_prize(self, group_id: str, description: str,
                     funded_by_owner: bool = False,
                     owner_approval: str = "") -> dict[str, Any]:
        """Create a prize. Unfunded by default — redeem refuses until funded."""
        prize = {"id": "prz_" + new_short_id(),
                 "group_id": group_id,
                 "description": (description or "").strip()[:300],
                 "funded_by_owner": bool(funded_by_owner and owner_approval.strip()),
                 "owner_approval": (owner_approval or "").strip()[:300],
                 "winner": "", "redeemed_at": 0.0,
                 "created_at": time.time()}
        prizes = self._load(group_id)
        prizes.append(prize)
        self._save(group_id, prizes)
        return prize

    def redeem(self, group_id: str, prize_id: str, winner: str) -> dict[str, Any] | None:
        """Redeem a prize to a winner. Refuses unless owner-funded."""
        prizes = self._load(group_id)
        for prize in prizes:
            if prize["id"] == prize_id or prize["id"].startswith(prize_id):
                if prize.get("winner"):
                    return None  # already claimed
                if not prize.get("funded_by_owner") or not prize.get("owner_approval"):
                    return None  # not funded — no prize exists to redeem
                prize["winner"] = winner
                prize["redeemed_at"] = time.time()
                self._save(group_id, prizes)
                return prize
        return None

    def list_prizes(self, group_id: str) -> list[dict[str, Any]]:
        return self._load(group_id)


def format_statement(statement: dict[str, Any]) -> str:
    """One member's statement as chat text, disclaimer included."""
    lines = [f"💰 **{statement.get('member')}** — "
             f"{statement.get('balance', 0)} {POINTS_NAME}"]
    for b in statement.get("badges", []):
        lines.append(f"  🏅 {b.get('badge')} ({b.get('season')})")
    lines.append(f"  _{CURRENCY_DISCLAIMER}_")
    return "\n".join(lines)
