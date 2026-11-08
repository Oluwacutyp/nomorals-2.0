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

import contextlib
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

try:
    import fcntl
except ImportError:  # Windows (the EC2 host) — msvcrt instead
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # POSIX — fcntl instead
    msvcrt = None  # type: ignore[assignment]

_log = get_logger("nomorals.community.group_economy")

__all__ = [
    "POINTS_NAME", "CURRENCY_DISCLAIMER", "EARN_TABLE", "QUESTS", "STATUS_SHOP",
    "BANK_PREFIX", "JACKPOT_MEMBER", "DAILY_BASE", "DAILY_STREAK_STEP",
    "DAILY_STREAK_CAP", "GROUP_QUESTS", "SLOTS_REELS", "SLOTS_TRIPLE",
    "SLOTS_PAIR_PAYS",
    "EARN_CAPS", "MINING_BASE_RATE", "MINING_DECAY", "MINING_CLAIM_COOLDOWN_S",
    "LedgerEntry", "GroupEconomy", "GroupEconomyError", "PrizePool",
    "EconomyPersistenceError", "BankBlacklistedError",
    "HEIST_RULES_VERSION", "BLACKLIST_FIRST_H", "BLACKLIST_REPEAT_H",
    "season_key", "current_season",
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

#: Bank vaults + jackpot live under pseudo-members, the treasury's
#: own trick ("" is the treasury) generalized: "bank:<id>" is one
#: member's vault (safe from robbery — rob takes wallet only) and
#: "jackpot:" is the shared pot gambling losses feed.  Pseudo
#: members never appear in holder statistics or leaderboards.
BANK_PREFIX = "bank:"
JACKPOT_MEMBER = "jackpot:"

#: The daily claim (Septorch-loop staple): once per day, streaks
#: grow the amount modestly — showing up is the faucet's front door.
DAILY_BASE = 25
DAILY_STREAK_STEP = 2
DAILY_STREAK_CAP = 25

#: Group quests: COLLECTIVE goals over the whole group's activity
#: in the period (DESIGN §5).  Completion mints the bonus into the
#: treasury — the country rewarding itself for being alive.
GROUP_QUESTS: dict[str, dict[str, Any]] = {
    "chat_week": {"name": "Loud Week", "measure": "messages",
                  "period": "weekly", "need": 500, "bonus": 200,
                  "blurb": "500 messages group-wide this week"},
    "earn_week": {"name": "Working Country", "measure": "earned",
                  "period": "weekly", "need": 2000, "bonus": 250,
                  "blurb": "2,000 sparks earned group-wide this week"},
    "show_up_week": {"name": "Full House", "measure": "attends",
                     "period": "weekly", "need": 10, "bonus": 150,
                     "blurb": "10 event attendances this week"},
}

#: Slots: weighted reels + three-of-a-kind multipliers.  Expected
#: return is below the bet by design — gambling is a SINK with a
#: jackpot dream attached, not a faucet.
SLOTS_REELS: list[tuple[str, int]] = [
    ("🍒", 30), ("🍋", 25), ("🔔", 20), ("⭐", 15), ("💎", 7),
    ("7️⃣", 3)]
SLOTS_TRIPLE: dict[str, int] = {
    "🍒": 3, "🍋": 4, "🔔": 6, "⭐": 10, "💎": 25, "7️⃣": 77}
SLOTS_PAIR_PAYS = 1  # any pair returns the bet (push)

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


class GroupEconomyError(RuntimeError):
    """The group's ledger exists but cannot be read.  Raised,
    never swallowed into an empty balance: a corrupt ledger
    freezes the economy (see GroupSurface) rather than losing
    the group's money history to the next append."""


class EconomyPersistenceError(GroupEconomyError):
    """A state write failed (disk full, replace refused, …).

    Raised to the caller — never logged-and-forgotten — so no
    command can report a success the disk never took (doc 3,
    finding B). The operation that raised it did NOT commit: the
    pre-operation file is intact because every save is a
    write-temp-then-replace."""


class BankBlacklistedError(GroupEconomyError):
    """A credit into a blacklisted member's bank vault, refused
    AT the bank operation (doc 4 §C).

    The blacklist binds the vault, not the chat command: every
    route that can land sparks in a member's vault funnels
    through the paired-move machinery, and the machinery itself
    raises this. Carries the case reference and the lift time
    so the surface can state the terms instead of mumbling
    "deposit failed"."""

    def __init__(self, member: str, case_id: str,
                 until: float) -> None:
        super().__init__(
            f"bank deposits for {member} are blacklisted under "
            f"case {case_id} until {until}")
        self.member = member
        self.case_id = case_id
        self.until = float(until)


#: The consequence rules a heist case is settled under, stamped
#: on the case at open (doc 4 §A: group + RULES VERSION ride the
#: record). A later rules change never rewrites what an old
#: case says it was settled by.
HEIST_RULES_VERSION = "heist-rules-1"

#: Bank-deposit blacklist windows (owner's ruling on doc 4,
#: 2026-11-08): 48h from the bust for a first conviction, 96h
#: on a repeat bust — then the blacklist lifts on its own.
#: No fee, no pardon step, no permanent list.
BLACKLIST_FIRST_H = 48
BLACKLIST_REPEAT_H = 96


# ── cross-process writer lock (finding C) ─────────────────────────
#
# One lock file per group directory serializes every writer —
# across GroupEconomy instances AND across processes. A bare
# threading.Lock cannot do that (the audit's lost-update probe used
# two instances), and atomic file replacement alone does not make
# a read-modify-write transaction atomic.

_fallback_locks: dict[str, threading.Lock] = {}
_fallback_locks_guard = threading.Lock()


def _acquire_file_lock(fd: int, lock_path: Path) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:  # neither primitive: process-local fallback (named, not silent)
        key = str(lock_path)
        with _fallback_locks_guard:
            lock = _fallback_locks.setdefault(key, threading.Lock())
        lock.acquire()
        _fallback_held[threading.get_ident()] = (key, lock)


def _release_file_lock(fd: int, lock_path: Path) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        held = _fallback_held.pop(threading.get_ident(), None)
        if held is not None:
            held[1].release()


_fallback_held: dict[int, tuple[str, threading.Lock]] = {}


@contextlib.contextmanager
def _locked_path(lock_path: Path):
    """Hold the OS file lock at ``lock_path`` for the block."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _acquire_file_lock(fd, lock_path)
        try:
            yield
        finally:
            _release_file_lock(fd, lock_path)
    finally:
        os.close(fd)


@dataclass
class LedgerEntry:
    """One append-only ledger line. Nothing here is ever deleted."""

    id: str = ""
    group_id: str = ""
    member: str = ""  # "" = the group treasury
    kind: str = ""  # earn | spend | transfer_out | transfer_in | treasury_in |
    # treasury_out | reversal | season_close | badge | mining | wager |
    # convert
    amount: int = 0  # signed: + earns, - spends
    reason: str = ""
    ref: str = ""  # quest id / entry id / party id that caused this
    season: str = ""
    ts: float = 0.0
    reversed_by: str = ""  # reversal entry id, when this entry was reversed
    reverses: str = ""  # original entry id, when this entry IS a reversal
    #: Operation identity (finding D): every leg of one logical
    #: operation carries the same caller-supplied ``op_id``. A retry
    #: with the same op_id finds these legs and returns the original
    #: receipt instead of moving the money twice. Empty = the
    #: caller supplied no identity (legacy behavior).
    op_id: str = ""

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
            op_id=str(data.get("op_id", "")),
        )


class GroupEconomy:
    """Group ledger + quests + shop + treasury + seasons + mining.

    All state is one JSON file per group under
    ``~/.devon/community/group_economy/``: ``ledger``, ``activities``,
    ``badges``, ``flags``, ``verified``, ``prizes``, ``mining``.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        #: Per-thread stack of groups this instance currently holds
        #: the writer lock for — the lock is reentrant within a
        #: thread so a public operation can wrap its whole body
        #: while the primitives it calls lock again underneath.
        self._lock_local = threading.local()

    # ── persistence ──
    def _group_dir(self, group_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                       for c in str(group_id)).strip("_") or "group"
        return self.data_dir / safe

    def _path(self, group_id: str, name: str) -> Path:
        return self._group_dir(group_id) / f"{name}.json"

    @contextlib.contextmanager
    def _locked(self, group_id: str):
        """Serialize all writers of one group's state (finding C).

        The balance check and every leg it authorizes run inside
        one hold, so two instances/processes can no longer
        read-modify-write over each other. The lock file carries
        no state; it only names the group directory's turn.
        """
        stack: list[str] = getattr(self._lock_local, "stack", None) \
            or []
        self._lock_local.stack = stack
        if group_id in stack:
            stack.append(group_id)
            try:
                yield
            finally:
                stack.pop()
            return
        with _locked_path(self._group_dir(group_id) / "economy.lock"):
            stack.append(group_id)
            try:
                yield
            finally:
                stack.pop()

    def _load_json(self, group_id: str, name: str, default: Any) -> Any:
        try:
            return json.loads(self._path(group_id, name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _save_json(self, group_id: str, name: str, data: Any) -> None:
        """Persist one state file — or raise (finding B).

        A failed save is an :class:`EconomyPersistenceError` at the
        caller, never a warning line under a success report. The
        write is temp-then-replace, so a failure leaves the previous
        good file untouched: the operation did not commit."""
        try:
            path = self._path(group_id, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise EconomyPersistenceError(
                f"group economy save failed for {group_id}/{name}: "
                f"{exc} — the operation was not committed") from exc

    def _ledger(self, group_id: str) -> list[LedgerEntry]:
        # The ledger is the money: an existing-but-unreadable
        # ledger file is NOT an empty ledger.  Reading it as empty
        # would let the next append overwrite the group's whole
        # history with a single entry — unreadable raises, and the
        # caller freezes economy writes instead (DESIGN §5
        # integrity).  A missing file is a fresh group: empty.
        path = self._path(group_id, "ledger")
        raw: Any = []
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GroupEconomyError(
                    f"group ledger for {group_id} is unreadable "
                    f"({type(exc).__name__}: {exc}) — refusing to "
                    f"treat it as empty") from exc
            if not isinstance(raw, list):
                raise GroupEconomyError(
                    f"group ledger for {group_id} is not a list — "
                    f"refusing to treat it as empty")
        out: list[LedgerEntry] = []
        for item in raw or []:
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
        self._append_many(group_id, [entry])
        return entry

    def _append_many(self, group_id: str,
                     new_entries: list[LedgerEntry]) -> list[LedgerEntry]:
        """Commit a whole operation's legs in ONE save (finding D).

        The ledger is loaded once, every leg appended in memory,
        and the file replaced once — under the group writer lock.
        A crash or write failure anywhere leaves either the whole
        operation or none of it; the old per-leg appends could
        strand half a transfer.
        """
        with self._locked(group_id):
            entries = self._ledger(group_id)
            entries.extend(new_entries)
            self._save_ledger(group_id, entries)
        return new_entries

    def _new_entry(self, group_id: str, member: str, kind: str, amount: int,
                   reason: str, ref: str = "", ts: float | None = None,
                   reverses: str = "", op_id: str = "") -> LedgerEntry:
        now = ts if ts is not None else time.time()
        return LedgerEntry(
            id="led_" + new_short_id(), group_id=group_id, member=member,
            kind=kind, amount=int(amount), reason=(reason or "")[:300],
            ref=(ref or "")[:120], season=season_key(now), ts=now,
            reverses=reverses, op_id=op_id)

    def _receipt(self, group_id: str, op_id: str,
                 expected_legs: int) -> list[LedgerEntry]:
        """The recorded legs of a previous operation (finding D).

        Empty when this op_id never ran. A PARTIAL set — a legacy
        crash between legs — raises: it must be reconciled by a
        human, never completed silently or reported as whole.
        """
        if not op_id:
            return []
        found = [e for e in self._ledger(group_id) if e.op_id == op_id]
        if not found:
            return []
        if len(found) != expected_legs:
            raise GroupEconomyError(
                f"operation {op_id!r} in group {group_id} is "
                f"partially recorded ({len(found)} of "
                f"{expected_legs} legs) — refusing to repeat or "
                "complete it silently; reconcile the ledger first")
        return found

    def _resolve_op_id(self, group_id: str, op_id: str,
                       shape: list[tuple[str, str, int, str]],
                       expected_legs: int) -> str:
        """Resolve a caller-supplied operation identity against
        the ledger (finding D, completed for platform-shaped
        identities).

        ``shape`` is the (member, kind, amount, ref) legs the
        operation is about to append. Three verdicts:

        - the id is unused → it stands;
        - the id's recorded legs MATCH the shape → a retry of
          the same operation: the id stands, and the caller's
          receipt check answers it without moving money again;
        - the id's recorded legs DIFFER → the identity was
          reused for a different operation. Platform message ids
          make this real, not theoretical: every button tap on
          one card shares the card's message id, and an edited
          message keeps its id under new text. Answering with
          the old receipt would report money that never moved
          (a false success); refusing would break a legitimate
          second tap. The operation takes the first free
          suffixed id (``<id>#2``, ``<id>#3``, …), executes
          honestly, and its own retries dedupe against the
          suffix.

        A PARTIAL recorded set under any candidate raises, as
        :meth:`_receipt` does: half an operation is reconciled
        by a human, never routed around. Callers hold the group
        writer lock, so resolve → receipt → append is one
        atomic decision."""
        if not op_id:
            return ""
        candidate = op_id
        suffix = 1
        while True:
            found = self._receipt(group_id, candidate,
                                  expected_legs)
            if not found:
                return candidate
            have = [(e.member, e.kind, e.amount, e.ref)
                    for e in found]
            if have == list(shape):
                return candidate
            suffix += 1
            candidate = f"{op_id}#{suffix}"

    def _post_commit(self, fn: Any, *args: Any) -> None:
        """Run post-commit bookkeeping (abuse checks).

        The operation has already committed. A persistence failure
        here must NOT be reported as the operation failing — that
        lie would invite a duplicate retry of money that already
        moved. It is logged at error level instead: visible and
        named, in the one place the distinction is documented.
        """
        try:
            fn(*args)
        except EconomyPersistenceError as exc:
            _log.error("post-commit bookkeeping failed: %s", exc)

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
        with self._locked(group_id):
            verified = self._load_json(group_id, "verified", {}) or {}
            verified[str(member)] = {"fingerprint": hashlib.sha256(
                str(fingerprint or member).encode()).hexdigest()[:16],
                "at": time.time()}
            self._save_json(group_id, "verified", verified)
            # The verification has committed above; the alt-farming
            # check's flag bookkeeping is post-commit.
            self._post_commit(self._check_alt_farming, group_id)
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
              now: float | None = None,
              op_id: str = "") -> LedgerEntry | None:
        """Credit points for real activity. Returns None when blocked.

        Blocked when: unverified member, non-positive amount, or either
        earn cap (daily / weekly) would be exceeded. Caps are enforced
        here, in the one place all earns flow through. With ``op_id``,
        a retry returns the ORIGINAL entry — the receipt — before any
        cap check: the award already happened, so it is never paid
        twice and never re-judged.
        """
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return found[0]
            if not self.is_verified(group_id, member):
                return None
            for period, cap in EARN_CAPS.items():
                if self._earned_in_period(group_id, member, period, now) + amount > cap:
                    self._flag(group_id, member, f"earn_cap_{period}",
                               f"earn of {amount} would exceed {period} cap {cap}")
                    return None
            return self._append(group_id, self._new_entry(
                group_id, member, kind, amount, reason, ref=ref, ts=now,
                op_id=op_id))

    # ── wagering (games): the member's own money, both ways ──
    # Wager movements are NOT earns: no caps, no faucet.  A debit is
    # balance-checked and atomic (insufficient -> None, nothing
    # moves); a credit is a stake returning or winnings paid.  Both
    # journal as kind "wager" so statements show the game's legs.

    def wager_debit(self, group_id: str, member: str, amount: int,
                    reason: str, ref: str = "",
                    now: float | None = None,
                    op_id: str = "") -> LedgerEntry | None:
        """Escrow ``amount`` from the member's wallet for a game
        wager.  Returns None — moving nothing — when the balance
        cannot cover it."""
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return found[0]
            if self.balance(group_id, member) < amount:
                return None
            return self._append(group_id, self._new_entry(
                group_id, member, "wager", -amount, reason, ref=ref,
                ts=now, op_id=op_id))

    def wager_credit(self, group_id: str, member: str, amount: int,
                     reason: str, ref: str = "",
                     now: float | None = None,
                     op_id: str = "") -> LedgerEntry | None:
        """Return a stake / pay winnings to the member's wallet."""
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return found[0]
            return self._append(group_id, self._new_entry(
                group_id, member, "wager", amount, reason, ref=ref,
                ts=now, op_id=op_id))

    # ── conversion (the §25 exchange): the member's own money ──
    # Conversion legs journal as kind "convert" under one shared
    # ref per conversion (convert:<id>:<side>).  Like wager legs
    # they are NOT earns — no caps, no faucet — but conversion-IN
    # is issuance: issuance_state counts positive convert legs
    # as earned and negative ones as sunk, so the monetary
    # policy sees the whole money supply (§25's amendment to
    # the policy's kind lists, named here rather than hidden).

    def convert_credit(self, group_id: str, member: str,
                       amount: int, reason: str, ref: str = "",
                       now: float | None = None,
                       op_id: str = "") -> LedgerEntry | None:
        """Credit converted sparks.  ``member`` may be the
        treasury ("") — the territory skim leg lands there."""
        amount = int(amount)
        if amount <= 0:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return found[0]
            return self._append(group_id, self._new_entry(
                group_id, member, "convert", amount, reason, ref=ref,
                ts=now, op_id=op_id))

    def convert_debit(self, group_id: str, member: str,
                      amount: int, reason: str, ref: str = "",
                      now: float | None = None,
                      op_id: str = "") -> LedgerEntry | None:
        """Debit sparks for a reverse conversion.  Balance-
        checked: returns None — moving nothing — when the
        wallet cannot cover it."""
        amount = int(amount)
        if amount <= 0 or not member:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return found[0]
            if self.balance(group_id, member) < amount:
                return None
            return self._append(group_id, self._new_entry(
                group_id, member, "convert", -amount, reason, ref=ref,
                ts=now, op_id=op_id))

    def record_activity(self, group_id: str, member: str, kind: str,
                        now: float | None = None) -> LedgerEntry | None:
        """Record real activity AND pay its earn-table amount, if any.

        Activity is the raw material for quests, streaks, and mining
        hashrate. Unknown kinds are recorded but pay nothing.
        """
        now = now if now is not None else time.time()
        with self._locked(group_id):
            activities = self._load_json(group_id, "activities", []) or []
            activities.append({"member": member, "kind": kind, "ts": now})
            # The record lands BEFORE the award: if this save fails
            # the error propagates and no award is reported for an
            # activity that was never recorded.
            self._save_json(group_id, "activities", activities[-5000:])
            amount = EARN_TABLE.get(kind, 0)
            if amount > 0:
                # Monetary policy: when issuance runs hot, the faucet
                # halves — visibly (issuance_state is on /treasury),
                # never silently.
                if self.issuance_state(group_id, now=now)["throttled"]:
                    amount = max(1, amount // 2)
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
        """Claim a completed quest's reward. Once per period, never twice.

        The award and the claim record live in two files, so the
        pair cannot commit in one save. The LEDGER is therefore the
        authority on "was this quest paid": if the award committed
        but the claim-record save failed (the caller saw the error),
        a retry finds the paid award below, finishes the claim
        record, and returns the original entry — never a second
        payment.
        """
        now = now if now is not None else time.time()
        q = QUESTS.get(quest_id)
        if q is None:
            return None
        with self._locked(group_id):
            prog = self.quest_progress(group_id, member, now).get(quest_id) or {}
            if not prog.get("complete") or prog.get("claimed"):
                return None
            period = _period_key(q["period"], now)
            paid = next(
                (e for e in self._ledger(group_id)
                 if e.member == member and e.kind == "earn"
                 and e.ref == f"quest:{quest_id}"
                 and _period_key(q["period"], e.ts) == period),
                None)
            claims = self._load_json(group_id, "quest_claims", {}) or {}
            claim_key = f"{member}:{quest_id}:{period}"
            if paid is not None:
                claims[claim_key] = True
                self._save_json(group_id, "quest_claims", claims)
                return paid
            entry = self.award(group_id, member, int(q["reward"]),
                               f"quest: {q['name']}", ref=f"quest:{quest_id}",
                               now=now)
            if entry is None:
                return None  # cap hit — the quest stays claimable next period
            claims[claim_key] = True
            self._save_json(group_id, "quest_claims", claims)
            return entry

    # ── spending: the status shop ──
    def buy_status(self, group_id: str, member: str, item_id: str,
                   now: float | None = None,
                   op_id: str = "") -> dict[str, Any] | None:
        """Buy a status item. Returns the grant record or None.

        The grant is what the chat layer acts on (shout-out, pin, …).
        Nothing here confers gameplay power — status only. With
        ``op_id``, a retry returns the original grant (same
        ``ledger_id``) instead of charging twice.
        """
        now = now if now is not None else time.time()
        item = STATUS_SHOP.get(item_id)
        if item is None:
            return None
        with self._locked(group_id):
            if op_id:
                found = self._receipt(group_id, op_id, 1)
                if found:
                    return {"item": item_id, "name": item["name"],
                            "cost": item["cost"], "member": member,
                            "ledger_id": found[0].id,
                            "granted_at": found[0].ts,
                            "disclaimer": CURRENCY_DISCLAIMER}
            if self.balance(group_id, member) < int(item["cost"]):
                return None
            entry = self._append(group_id, self._new_entry(
                group_id, member, "spend", -int(item["cost"]),
                f"status shop: {item['name']}", ref=f"shop:{item_id}",
                ts=now, op_id=op_id))
            return {"item": item_id, "name": item["name"],
                    "cost": item["cost"], "member": member,
                    "ledger_id": entry.id, "granted_at": now,
                    "disclaimer": CURRENCY_DISCLAIMER}

    # ── transfers + treasury ──
    def transfer(self, group_id: str, sender: str, recipient: str, amount: int,
                 note: str = "", now: float | None = None,
                 op_id: str = "") -> bool:
        """Tip another member. Both sides are ledger entries; never a delete.

        The two legs commit in ONE save (finding D): a crash can no
        longer leave the debit without the credit. With ``op_id``, a
        retry reports the original operation's success without
        moving the money again.
        """
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or sender == recipient or not sender or not recipient:
            return False
        with self._locked(group_id):
            if op_id:
                op_id = self._resolve_op_id(
                    group_id, op_id,
                    [(sender, "transfer_out", -amount,
                      f"to:{recipient}"),
                     (recipient, "transfer_in", amount,
                      f"from:{sender}")], 2)
                if self._receipt(group_id, op_id, 2):
                    return True
            if self.balance(group_id, sender) < amount:
                return False
            self._append_many(group_id, [
                self._new_entry(
                    group_id, sender, "transfer_out", -amount,
                    f"tip to {recipient}: {note}"[:300],
                    ref=f"to:{recipient}", ts=now, op_id=op_id),
                self._new_entry(
                    group_id, recipient, "transfer_in", amount,
                    f"tip from {sender}: {note}"[:300],
                    ref=f"from:{sender}", ts=now, op_id=op_id),
            ])
            self._post_commit(self._check_collusion, group_id, now)
            return True

    def transfer_taxed(self, group_id: str, sender: str, recipient: str,
                       gross: int, fee: int, note: str = "",
                       now: float | None = None,
                       op_id: str = "") -> bool:
        """A transfer with its treasury tax, as ONE operation.

        All four legs — the fee out of the sender, the fee into the
        treasury, the net out of the sender, the net into the
        recipient — commit in a single save, balance-checked
        against the gross. (Before this, the surface ran the fee as
        a separate contribution and the transfer after it: a failure
        between them took the fee and delivered nothing.) With
        ``fee == 0`` this is exactly :meth:`transfer`'s two legs.
        """
        now = now if now is not None else time.time()
        gross = int(gross)
        fee = int(fee)
        if gross <= 0 or fee < 0 or fee >= gross \
                or sender == recipient or not sender or not recipient:
            return False
        net = gross - fee
        legs = 4 if fee else 2
        with self._locked(group_id):
            if op_id:
                shape: list[tuple[str, str, int, str]] = []
                if fee:
                    shape += [
                        (sender, "treasury_in", -fee, "treasury"),
                        ("", "treasury_in", fee, f"from:{sender}")]
                shape += [
                    (sender, "transfer_out", -net,
                     f"to:{recipient}"),
                    (recipient, "transfer_in", net,
                     f"from:{sender}")]
                op_id = self._resolve_op_id(group_id, op_id,
                                            shape, legs)
                if self._receipt(group_id, op_id, legs):
                    return True
            if self.balance(group_id, sender) < gross:
                return False
            entries: list[LedgerEntry] = []
            if fee:
                entries += [
                    self._new_entry(
                        group_id, sender, "treasury_in", -fee,
                        "contributed to group treasury", ref="treasury",
                        ts=now, op_id=op_id),
                    self._new_entry(
                        group_id, "", "treasury_in", fee,
                        f"contribution from {sender}",
                        ref=f"from:{sender}", ts=now, op_id=op_id),
                ]
            entries += [
                self._new_entry(
                    group_id, sender, "transfer_out", -net,
                    f"tip to {recipient}: {note}"[:300],
                    ref=f"to:{recipient}", ts=now, op_id=op_id),
                self._new_entry(
                    group_id, recipient, "transfer_in", net,
                    f"tip from {sender}: {note}"[:300],
                    ref=f"from:{sender}", ts=now, op_id=op_id),
            ]
            self._append_many(group_id, entries)
            self._post_commit(self._check_collusion, group_id, now)
            return True

    def treasury_balance(self, group_id: str) -> int:
        return self.balance(group_id, "")

    def contribute_treasury(self, group_id: str, member: str, amount: int,
                            now: float | None = None,
                            op_id: str = "") -> bool:
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0:
            return False
        with self._locked(group_id):
            if op_id and self._receipt(group_id, op_id, 2):
                return True
            if self.balance(group_id, member) < amount:
                return False
            self._append_many(group_id, [
                self._new_entry(
                    group_id, member, "treasury_in", -amount,
                    "contributed to group treasury", ref="treasury",
                    ts=now, op_id=op_id),
                self._new_entry(
                    group_id, "", "treasury_in", amount,
                    f"contribution from {member}", ref=f"from:{member}",
                    ts=now, op_id=op_id),
            ])
            return True

    def treasury_grant(self, group_id: str, recipient: str, amount: int,
                       reason: str = "", now: float | None = None,
                       op_id: str = "") -> bool:
        """Pay out of the treasury (party prize, shout-out pool, …)."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0:
            return False
        with self._locked(group_id):
            if op_id and self._receipt(group_id, op_id, 2):
                return True
            if self.treasury_balance(group_id) < amount:
                return False
            self._append_many(group_id, [
                self._new_entry(
                    group_id, "", "treasury_out", -amount,
                    f"treasury grant to {recipient}: {reason}"[:300],
                    ref=f"to:{recipient}", ts=now, op_id=op_id),
                self._new_entry(
                    group_id, recipient, "treasury_out", amount,
                    f"treasury grant: {reason}"[:300], ref="treasury",
                    ts=now, op_id=op_id),
            ])
            return True

    # ── bank vaults / jackpot / gambling / robbery ──
    # The Septorch loop, on this ledger's mechanics: every move is
    # paired legs between ledger members (real or pseudo), so the
    # group's total is conserved except at named issuance points
    # (awards, quest bonuses, gambling payouts) — all journaled.

    @staticmethod
    def is_pseudo_member(member: str) -> bool:
        return member == "" or member == JACKPOT_MEMBER \
            or str(member).startswith(BANK_PREFIX)

    def bank_member(self, member: str) -> str:
        return BANK_PREFIX + str(member)

    def bank_balance(self, group_id: str, member: str) -> int:
        return self.balance(group_id, self.bank_member(member))

    def total_balance(self, group_id: str, member: str) -> int:
        return self.balance(group_id, member) \
            + self.bank_balance(group_id, member)

    def _move(self, group_id: str, src: str, dst: str, amount: int,
              kind: str, reason: str,
              now: float | None = None, op_id: str = "") -> bool:
        """Paired-leg move between any two ledger members — one
        commit for both legs (finding D), receipt-idempotent under
        ``op_id``."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or src == dst:
            return False
        with self._locked(group_id):
            if op_id:
                op_id = self._resolve_op_id(
                    group_id, op_id,
                    [(src, kind, -amount, f"to:{dst}"),
                     (dst, kind, amount, f"from:{src}")], 2)
                if self._receipt(group_id, op_id, 2):
                    return True
            if dst.startswith(BANK_PREFIX):
                # THE bank-deposit blacklist binds here, at the
                # operation itself (doc 4 §C) — not in any chat
                # command. A credit whose destination is a
                # member's bank vault IS a deposit, whoever
                # sends it: the member, another member paying
                # into their vault, or any helper moving sparks
                # vault-ward. While the member's conviction
                # stands, the move is refused and nothing moves.
                # Debits OUT of a vault (withdrawals, mortgage
                # instalments, fines, vault raids) are not
                # deposits and are never touched by this check;
                # credits to a WALLET are not deposits either —
                # the blacklist binds the vault, not the purse.
                vault_owner = dst[len(BANK_PREFIX):]
                if vault_owner and not self.is_pseudo_member(
                        vault_owner):
                    active = self.bank_blacklist(
                        group_id, vault_owner, now=now)
                    if active is not None:
                        raise BankBlacklistedError(
                            vault_owner, active["case_id"],
                            active["until"])
            if self.balance(group_id, src) < amount:
                return False
            self._append_many(group_id, [
                self._new_entry(
                    group_id, src, kind, -amount, reason[:300],
                    ref=f"to:{dst}", ts=now, op_id=op_id),
                self._new_entry(
                    group_id, dst, kind, amount, reason[:300],
                    ref=f"from:{src}", ts=now, op_id=op_id),
            ])
            return True

    def bank_deposit(self, group_id: str, member: str, amount: int,
                     now: float | None = None,
                     op_id: str = "") -> bool:
        """Wallet → the member's own vault.  Vaulted sparks are
        safe from robbery and count toward their total wealth."""
        return self._move(group_id, member, self.bank_member(member),
                          amount, "bank_deposit",
                          "deposited to bank vault", now, op_id=op_id)

    def bank_withdraw(self, group_id: str, member: str, amount: int,
                      now: float | None = None,
                      op_id: str = "") -> bool:
        return self._move(group_id, self.bank_member(member), member,
                          amount, "bank_withdraw",
                          "withdrew from bank vault", now, op_id=op_id)

    # ── heist case book + the bank-deposit blacklist ──────────
    # Doc 4 §A/§C/§E: a resolved bank-heist bust opens ONE
    # durable case in the bank's own book (cases.json, beside
    # the ledger, under the same writer lock and the same
    # raise-never-swallow persistence discipline). The case is
    # the conviction: it carries the offenders + roles, the
    # evidence, each offender's blacklist window (fixed at
    # open: 48h, 96h on a repeat), the arrest record and the
    # seizure manifest as settlement records them, and an
    # explicit settlement state — pending → committed, or
    # recovery_required when a step is still owed. The
    # blacklist is DERIVED from the book: there is no second
    # list to drift out of sync with the cases.

    def _load_cases(self, group_id: str) -> dict[str, Any]:
        # The case book is a conviction record: an existing-but-
        # unreadable book RAISES (the ledger's discipline) — a
        # blacklist must never evaporate because a file failed
        # to parse. A missing file is a group with no cases.
        path = self._path(group_id, "cases")
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GroupEconomyError(
                    f"heist case book for {group_id} is unreadable "
                    f"({type(exc).__name__}: {exc}) — refusing to "
                    f"treat it as empty") from exc
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("cases"), dict):
                raise GroupEconomyError(
                    f"heist case book for {group_id} is malformed "
                    f"— refusing to treat it as empty")
            return raw
        return {"cases": {}}

    def _save_cases(self, group_id: str,
                    book: dict[str, Any]) -> None:
        self._save_json(group_id, "cases", book)

    def open_heist_case(self, group_id: str, *, heist_id: str,
                        target: dict[str, Any], crew: list[str],
                        leader: str, evidence: dict[str, Any],
                        now: float | None = None
                        ) -> tuple[dict[str, Any], bool]:
        """Open the case for one resolved bust — idempotently.

        The case id derives from the heist identity, so opening
        the same bust twice (a retried settlement, a restarted
        process) returns the SAME case instead of convicting
        twice. Each offender's blacklist window is fixed here,
        at open: 48h for a first conviction, 96h when any prior
        case in this group already names them. Returns
        (case, created)."""
        now = now if now is not None else time.time()
        case_id = f"case-{heist_id}"
        with self._locked(group_id):
            book = self._load_cases(group_id)
            existing = book["cases"].get(case_id)
            if existing is not None:
                return existing, False
            prior_offenders = {
                member
                for other in book["cases"].values()
                for member in (other.get("offenders") or {})}
            offenders: dict[str, Any] = {}
            for member in crew:
                repeat = member in prior_offenders
                hours = BLACKLIST_REPEAT_H if repeat \
                    else BLACKLIST_FIRST_H
                offenders[member] = {
                    "role": "leader" if member == leader
                            else "crew",
                    "jailed_until": 0.0,
                    "blacklist_hours": hours,
                    "repeat": repeat,
                    "blacklist_until": now + hours * 3600.0,
                    "seized": [],
                    "seizure_done": False,
                }
            case = {
                "id": case_id,
                "group_id": group_id,
                "heist_id": str(heist_id),
                "rules_version": HEIST_RULES_VERSION,
                "target": {"kind": str(target.get("kind") or ""),
                           "id": str(target.get("id") or "")},
                "outcome": "bust",
                "evidence": {
                    "odds": evidence.get("odds"),
                    "offense": evidence.get("offense"),
                    "defense": evidence.get("defense"),
                    "notes": list(evidence.get("notes") or []),
                    "take": int(evidence.get("take") or 0),
                },
                "opened_at": now,
                "state": "pending",
                "offenders": offenders,
                "settled_at": 0.0,
            }
            book["cases"][case_id] = case
            self._save_cases(group_id, book)
            return case, True

    def heist_case(self, group_id: str,
                   case_id: str) -> dict[str, Any] | None:
        with self._locked(group_id):
            return self._load_cases(group_id)["cases"].get(
                case_id)

    def case_for_heist(self, group_id: str,
                       heist_id: str) -> dict[str, Any] | None:
        with self._locked(group_id):
            for case in self._load_cases(group_id)[
                    "cases"].values():
                if case.get("heist_id") == str(heist_id):
                    return case
            return None

    def update_heist_case(self, group_id: str, case_id: str,
                          mutator: Any) -> dict[str, Any] | None:
        """One locked read-modify-write of a case record — the
        settlement's step ledger. The mutator edits the case
        dict in place; the whole book is re-saved under the
        group writer lock, so a crash lands either the step or
        none of it, and a retry re-reads what actually
        committed."""
        with self._locked(group_id):
            book = self._load_cases(group_id)
            case = book["cases"].get(case_id)
            if case is None:
                return None
            mutator(case)
            self._save_cases(group_id, book)
            return case

    def bank_blacklist(self, group_id: str, member: str, *,
                       now: float | None = None
                       ) -> dict[str, Any] | None:
        """The member's ACTIVE deposit blacklist, derived from
        the case book: the unexpired window with the latest
        lift time, or None. Expiry is automatic — a window in
        the past is not a blacklist, it is history (see
        :meth:`bank_blacklist_lifted`)."""
        now = now if now is not None else time.time()
        best: dict[str, Any] | None = None
        with self._locked(group_id):
            cases = self._load_cases(group_id)["cases"].values()
            for case in cases:
                slot = (case.get("offenders") or {}).get(member)
                if not slot:
                    continue
                until = float(slot.get("blacklist_until") or 0)
                if until > now and (best is None
                                    or until > best["until"]):
                    best = {"case_id": str(case["id"]),
                            "until": until,
                            "hours": int(
                                slot.get("blacklist_hours") or 0),
                            "repeat": bool(slot.get("repeat"))}
        return best

    def bank_blacklist_lifted(self, group_id: str, member: str, *,
                              now: float | None = None
                              ) -> dict[str, Any] | None:
        """The member's most recent LIFTED blacklist (window in
        the past), or None — bank status states the lift and
        its case reference instead of going silent the moment
        the window closes."""
        now = now if now is not None else time.time()
        best: dict[str, Any] | None = None
        with self._locked(group_id):
            cases = self._load_cases(group_id)["cases"].values()
            for case in cases:
                slot = (case.get("offenders") or {}).get(member)
                if not slot:
                    continue
                until = float(slot.get("blacklist_until") or 0)
                if 0 < until <= now and (best is None
                                         or until > best["until"]):
                    best = {"case_id": str(case["id"]),
                            "until": until,
                            "hours": int(
                                slot.get("blacklist_hours") or 0),
                            "repeat": bool(slot.get("repeat"))}
        return best

    def jackpot_balance(self, group_id: str) -> int:
        return self.balance(group_id, JACKPOT_MEMBER)

    def gamble_slots(self, group_id: str, member: str, bet: int,
                     rng: Any, now: float | None = None) -> dict[str, Any]:
        """One slots pull, settled on the ledger.

        The bet leaves the wallet; a loss is split between the
        treasury and the jackpot pot (conserved — the burn leg
        equals the two credit legs); a win is paid at the table
        multiplier; triple 7s ALSO empties the jackpot pot."""
        now = now if now is not None else time.time()
        bet = int(bet)
        with self._locked(group_id):
            if bet <= 0 or self.balance(group_id, member) < bet:
                return {"ok": False, "reason": "funds"}
            # The whole spin settles in ONE commit (finding D): bet,
            # payout / house legs, and the jackpot sweep are computed
            # first and appended together.
            symbols = [rng.choices(
                [s for s, _w in SLOTS_REELS],
                weights=[w for _s, w in SLOTS_REELS])[0] for _ in range(3)]
            mult = 0
            if symbols[0] == symbols[1] == symbols[2]:
                mult = SLOTS_TRIPLE[symbols[0]]
            elif len(set(symbols)) == 2:
                mult = SLOTS_PAIR_PAYS
            payout = bet * mult
            jackpot = 0
            legs = [self._new_entry(
                group_id, member, "gamble", -bet, "slots bet",
                ref="slots", ts=now)]
            if payout > 0:
                legs.append(self._new_entry(
                    group_id, member, "gamble", payout,
                    f"slots win {''.join(symbols)} ×{mult}",
                    ref="slots", ts=now))
            else:
                to_treasury = bet // 2
                to_jackpot = bet - to_treasury
                if to_treasury:
                    legs.append(self._new_entry(
                        group_id, "", "treasury_in", to_treasury,
                        "slots house share", ref="slots", ts=now))
                if to_jackpot:
                    legs.append(self._new_entry(
                        group_id, JACKPOT_MEMBER, "jackpot_in",
                        to_jackpot, "slots jackpot feed",
                        ref="slots", ts=now))
            if symbols == ["7️⃣", "7️⃣", "7️⃣"]:
                pot = self.jackpot_balance(group_id)
                if pot > 0:
                    legs += [
                        self._new_entry(
                            group_id, JACKPOT_MEMBER, "jackpot_win",
                            -pot, "JACKPOT — triple 7s",
                            ref=f"to:{member}", ts=now),
                        self._new_entry(
                            group_id, member, "jackpot_win", pot,
                            "JACKPOT — triple 7s", ref="from:jackpot:",
                            ts=now),
                    ]
                    jackpot = pot
            self._append_many(group_id, legs)
        return {"ok": True, "symbols": symbols, "multiplier": mult,
                "payout": payout, "jackpot": jackpot,
                "balance": self.balance(group_id, member)}

    def gamble_coinflip(self, group_id: str, member: str, bet: int,
                        choice: str, rng: Any,
                        now: float | None = None) -> dict[str, Any]:
        """Coinflip against the house: a win pays 1.9× (5% edge);
        a loss splits treasury/jackpot like slots."""
        now = now if now is not None else time.time()
        bet = int(bet)
        with self._locked(group_id):
            if bet <= 0 or self.balance(group_id, member) < bet:
                return {"ok": False, "reason": "funds"}
            flip = "heads" if rng.random() < 0.5 else "tails"
            won = (choice or "").lower().startswith(flip[0])
            payout = int(bet * 1.9) if won else 0
            # One commit for the whole flip (finding D).
            legs = [self._new_entry(
                group_id, member, "gamble", -bet, "coinflip bet",
                ref="coinflip", ts=now)]
            if won:
                legs.append(self._new_entry(
                    group_id, member, "gamble", payout,
                    f"coinflip win ({flip})", ref="coinflip", ts=now))
            else:
                to_treasury = bet // 2
                to_jackpot = bet - to_treasury
                if to_treasury:
                    legs.append(self._new_entry(
                        group_id, "", "treasury_in", to_treasury,
                        "coinflip house share", ref="coinflip", ts=now))
                if to_jackpot:
                    legs.append(self._new_entry(
                        group_id, JACKPOT_MEMBER, "jackpot_in",
                        to_jackpot, "coinflip jackpot feed",
                        ref="coinflip", ts=now))
            self._append_many(group_id, legs)
        return {"ok": True, "flip": flip, "won": won,
                "payout": payout,
                "balance": self.balance(group_id, member)}

    def rob_attempt(self, group_id: str, actor: str, target: str,
                    rng: Any, now: float | None = None) -> dict[str, Any]:
        """Rob the target's WALLET (banked sparks are safe).

        Success moves 10–25% of the target's wallet (capped);
        failure is reported for the surface to jail or fine.
        The money legs live here; consequences (jail) are state
        the surface owns."""
        now = now if now is not None else time.time()
        with self._locked(group_id):
            if actor == target or not actor or not target:
                return {"ok": False, "reason": "target"}
            wallet = self.balance(group_id, target)
            if wallet <= 0:
                return {"ok": False, "reason": "empty"}
            if rng.random() < 0.45:
                amount = max(1, min(200, int(wallet * rng.uniform(0.10, 0.25))))
                moved = self._move(group_id, target, actor, amount,
                                   "robbery", f"robbed by {actor}", now)
                return {"ok": True, "success": bool(moved),
                        "amount": amount if moved else 0}
            return {"ok": True, "success": False, "amount": 0}

    # ── group quests / monetary policy / seasons ──

    def group_quest_progress(self, group_id: str,
                             now: float | None = None) -> dict[str, Any]:
        """Collective progress over the whole group's activity."""
        now = now if now is not None else time.time()
        activities = self._load_json(group_id, "activities", []) or []
        entries = self._ledger(group_id)
        out: dict[str, Any] = {}
        for quest_id, quest in GROUP_QUESTS.items():
            key = _period_key(quest["period"], now)
            if quest["measure"] == "messages":
                have = sum(
                    1 for a in activities
                    if a.get("kind") == "message"
                    and _period_key(quest["period"],
                                    float(a.get("ts") or 0)) == key)
            elif quest["measure"] == "attends":
                have = sum(
                    1 for a in activities
                    if a.get("kind") == "event_attend"
                    and _period_key(quest["period"],
                                    float(a.get("ts") or 0)) == key)
            else:  # earned
                have = sum(
                    e.amount for e in entries
                    if e.kind in ("earn", "mining") and e.amount > 0
                    and not self.is_pseudo_member(e.member)
                    and _period_key(quest["period"], e.ts) == key)
            out[quest_id] = {**quest, "have": int(have),
                             "complete": have >= quest["need"]}
        return out

    def treasury_bonus(self, group_id: str, amount: int, reason: str,
                       now: float | None = None) -> bool:
        """Mint a bonus into the treasury (group-quest completion).
        Issuance, journaled as itself — never disguised."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0:
            return False
        self._append(group_id, self._new_entry(
            group_id, "", "treasury_in", amount, reason[:300],
            ref="bonus", ts=now))
        return True

    def issuance_state(self, group_id: str, *, days: int = 7,
                       now: float | None = None) -> dict[str, Any]:
        """Monetary policy (DESIGN §5): faucet issuance vs sinks
        over the window.  When issuance runs hot — earned more
        than 3× the sunk total, at real volume — faucets throttle
        (activity pays halve) until sinks catch up.  Visible on
        /treasury; never a hidden nerf."""
        now = now if now is not None else time.time()
        since = now - days * 86400
        earned = sunk = 0
        for e in self._ledger(group_id):
            if e.ts < since or self.is_pseudo_member(e.member):
                continue
            if e.kind in ("earn", "mining", "convert") and e.amount > 0:
                # "convert" (§25): sparks a member converted IN
                # from games coins are issuance exactly like a
                # faucet's — the policy counts the whole supply.
                earned += e.amount
            elif e.amount < 0 and e.kind in (
                    "spend", "treasury_in", "gamble", "bank_deposit",
                    # "convert" (§25): sparks converted back OUT
                    # to games coins leave circulation — a sink.
                    "convert",
                    # real estate (§20): every member-side estate
                    # spend is a sink the policy must see — buys,
                    # market fees, mortgage repayments, insurance
                    # premiums, repair bills, management.
                    "estate_buy", "estate_fee", "estate_repay",
                    "estate_insure", "estate_repair",
                    "estate_manager"):
                sunk += -e.amount
        throttled = earned >= 1000 and earned > 3 * max(1, sunk)
        return {"earned": earned, "sunk": sunk, "days": days,
                "throttled": throttled}

    def season_top(self, group_id: str, season: str, *,
                   limit: int = 3) -> list[tuple[str, int]]:
        totals: dict[str, int] = {}
        for e in self._ledger(group_id):
            if e.season != season or e.amount <= 0:
                continue
            if e.kind not in ("earn", "mining"):
                continue
            if self.is_pseudo_member(e.member):
                continue
            totals[e.member] = totals.get(e.member, 0) + e.amount
        return sorted(totals.items(), key=lambda kv: -kv[1])[:limit]

    def archive_season(self, group_id: str, season: str) -> int:
        """Snapshot one season's ledger entries to the archive
        file (rollover keeps the proof, not just the totals)."""
        with self._locked(group_id):
            rows = [e.to_dict() for e in self._ledger(group_id)
                    if e.season == season]
            self._save_json(group_id, f"archive-{season}", rows)
            return len(rows)

    # ── reversals: never delete ──
    def reverse(self, group_id: str, entry_id: str, reason: str,
                now: float | None = None) -> LedgerEntry | None:
        """Reverse an entry. The original stays; a reversal entry offsets it.

        Returns the reversal entry, or None when the entry doesn't exist
        or was already reversed. Reversing a reversal is refused.
        """
        now = now if now is not None else time.time()
        with self._locked(group_id):
            entries = self._ledger(group_id)
            target = next((e for e in entries if e.id == entry_id), None)
            if target is None or target.reversed_by \
                    or target.kind == "reversal":
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
        with self._locked(group_id):
            entries = self._ledger(group_id)
            balances = self.balances(group_id)
            ranked = sorted(((m, b) for m, b in balances.items() if m),
                            key=lambda kv: -kv[1])
            badges = self._load_json(group_id, "badges", []) or []
            medals = ["Season Champion", "Season Runner-up",
                      "Season Third"]
            for i, (member, _) in enumerate(ranked[:3]):
                # Badge dedupe: a retried close must not mint the
                # same season badge twice (finding D).
                if not any(b.get("member") == member
                           and b.get("badge") == medals[i]
                           and b.get("season") == season
                           for b in badges):
                    badges.append({"member": member, "badge": medals[i],
                                   "season": season, "at": now})
            self._save_json(group_id, "badges", badges)
            closed: list[dict[str, Any]] = []
            for member, bal in list(balances.items()):
                if bal == 0:
                    continue
                closing = self._new_entry(
                    group_id, member, "season_close", -bal,
                    f"season {season} close: balance reset",
                    ref=season, ts=now)
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
        with self._locked(group_id):
            # The ledger is the cooldown authority (finding D): the
            # award IS the cooldown record, so the pair commits as
            # one save; the mining.json timestamp write remains for
            # display, as post-commit bookkeeping — and the fallback
            # authority for members with no ledger history.
            last_ledger = max(
                (e.ts for e in self._ledger(group_id)
                 if e.member == member and e.kind == "mining"),
                default=0.0)
            mining = self._load_json(group_id, "mining", {}) or {}
            last = last_ledger or float(
                (mining.get(member) or {}).get("last_claim") or 0)
            if now - last < MINING_CLAIM_COOLDOWN_S:
                return None
            rate = self.claim_rate(group_id, member, now)
            entry = self.award(group_id, member, rate,
                               "mining claim (game points only)",
                               ref="mining", kind="mining", now=now)
            if entry is None:
                return None  # cap hit or unverified
            mining[member] = {"last_claim": now}
            self._post_commit(self._save_json, group_id, "mining",
                              mining)
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
        """Persist prizes — or raise (finding B, same contract as
        the economy ledger): a failed save never reports success."""
        try:
            path = self._path(group_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(prizes, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise EconomyPersistenceError(
                f"prize save failed for {group_id}: {exc} — the "
                "operation was not committed") from exc

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
        # The group writer lock serializes this against every other
        # writer of the group's state (finding C).
        with _locked_path(self._path(group_id).parent / "economy.lock"):
            prizes = self._load(group_id)
            prizes.append(prize)
            self._save(group_id, prizes)
        return prize

    def redeem(self, group_id: str, prize_id: str, winner: str) -> dict[str, Any] | None:
        """Redeem a prize to a winner. Refuses unless owner-funded."""
        with _locked_path(self._path(group_id).parent / "economy.lock"):
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
