"""Math as earning: lessons, quizzes, teaching, and problem creation.

How a member earns in Math Class:
  * complete a lesson + pass its quiz → reward-ladder payout;
  * teach others — post an explanation of a solution; each upvote
    from another member pays a teaching bonus;
  * create problems — a problem other players *fail* pays its creator
    a bonus (hard problems are valuable; trivial ones pay nothing,
    because nobody fails them).

Every earn is an auditable ledger entry. Balances are always recomputed
from the ledger, so any balance is reconstructible. Reversals never
delete: a reversal is a new entry that references the original, and the
original stays with ``reversed_by`` set — same rule as
:mod:`nomorals.community.group_economy`.

Coins are game currency: they buy status inside the group only. They
are never an investment, never redeemable for cash, never a promise
of value, and never a financial asset of any kind.

Guardrails, enforced here:
  * one economy account per verified member (unverified earns nothing);
  * per-period earn caps (daily / weekly);
  * decaying rate on repeat farming (same lesson/quiz again pays less);
  * alt-account detection is record-only: same device/IP fingerprint
    on 3+ accounts raises a flag for a human to review. It never
    auto-bans, never auto-docks, never blocks — flags are evidence,
    not verdicts.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .reward_ladder import FIRST_CLEAR_CHEST, reward_for

_log = get_logger(__name__)

__all__ = [
    "MATH_COINS_DISCLAIMER", "MATH_EARN_CAPS", "TEACH_UPVOTE_BONUS",
    "TEACH_UPVOTE_CAP", "CREATOR_BASE_BONUS", "CREATOR_DECAY",
    "ALT_FLAG_THRESHOLD", "MathLedgerEntry", "MathEconomy",
]

#: The honesty line, reused by every user-facing render in this module.
MATH_COINS_DISCLAIMER = (
    "Math coins are game currency only. They buy status inside the group, "
    "and are never an investment, never redeemable for cash, "
    "never a promise of value, and never a financial asset."
)

#: Per-member earn caps per period for math earning. Caps count real
#: earns (lessons, quizzes, teaching, creator bonuses) — not reversals.
MATH_EARN_CAPS: dict[str, int] = {"daily": 500, "weekly": 2000}

#: Teaching: each upvote on your explanation pays this, up to the cap
#: per explanation. (coins, xp)
TEACH_UPVOTE_BONUS: tuple[int, int] = (5, 10)
TEACH_UPVOTE_CAP = 10

#: Creator bonus: a solver failing your problem pays this base amount,
#: decaying per previous fail on the same problem.
CREATOR_BASE_BONUS: tuple[int, int] = (8, 15)
CREATOR_DECAY = 0.8

#: Alt-account detection: this many accounts sharing one fingerprint
#: raises a flag. Record-only — never an auto-ban.
ALT_FLAG_THRESHOLD = 3


def _period_key(period: str, ts: float) -> str:
    t = time.localtime(ts)
    if period == "daily":
        return time.strftime("%Y-%m-%d", t)
    return f"{t.tm_year}-W{time.strftime('%V', t)}"


def _hash_fingerprint(raw: str) -> str:
    return hashlib.sha256(str(raw or "").encode()).hexdigest()[:16]


@dataclass
class MathLedgerEntry:
    """One append-only ledger line. Nothing here is ever deleted."""

    id: str = ""
    player: str = ""
    kind: str = ""  # earn | reversal | badge | title
    amount_coins: int = 0
    amount_xp: int = 0
    reason: str = ""
    ref: str = ""  # lesson id / problem id / explanation id
    difficulty: str = ""
    ts: float = 0.0
    reversed_by: str = ""
    reverses: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MathLedgerEntry":
        return cls(
            id=str(data.get("id", "")),
            player=str(data.get("player", "")),
            kind=str(data.get("kind", "")),
            amount_coins=int(data.get("amount_coins") or 0),
            amount_xp=int(data.get("amount_xp") or 0),
            reason=str(data.get("reason", "")),
            ref=str(data.get("ref", "")),
            difficulty=str(data.get("difficulty", "")),
            ts=float(data.get("ts") or 0.0),
            reversed_by=str(data.get("reversed_by", "")),
            reverses=str(data.get("reverses", "")),
        )


class MathEconomy:
    """Math earning over an append-only ledger.

    ``credit_fn`` (optional) is called on every earn as
    ``credit_fn(player, coins, xp, reason)`` — the hook where a live
    wallet (e.g. :class:`nomorals.games.players.PlayerStore` via
    ``add_coins``/``add_xp``) gets credited. The ledger stays the
    source of truth either way: balances recompute from entries.
    """

    def __init__(self, credit_fn: Callable[..., Any] | None = None,
                 db: Any = None) -> None:
        self.credit_fn = credit_fn
        self.db = db
        self._entries: list[MathLedgerEntry] = []
        self._verified: dict[str, dict[str, Any]] = {}
        self._fingerprints: dict[str, list[str]] = {}  # fp -> players
        self._flags: list[dict[str, Any]] = []
        self._lessons: dict[str, dict[str, Any]] = {}  # lesson_id -> meta
        self._lesson_clears: dict[tuple[str, str], int] = {}  # (player, lesson) -> n
        self._explanations: dict[str, dict[str, Any]] = {}
        self._problems: dict[str, dict[str, Any]] = {}  # problem_id -> meta
        self._seq = 0

    # ── verification: one account per verified member ──
    def verify_member(self, player: str, fingerprint: str = "") -> bool:
        """Mark a member verified. ``fingerprint`` is a group-local
        device/IP fingerprint the host supplies — hashed, used only
        for alt-account detection, never stored raw."""
        fp = _hash_fingerprint(fingerprint or player)
        self._verified[str(player)] = {"fingerprint": fp, "at": time.time()}
        holders = self._fingerprints.setdefault(fp, [])
        if player not in holders:
            holders.append(player)
        if len(holders) >= ALT_FLAG_THRESHOLD and fp:
            for holder in holders:
                self._flag(holder, "alt_accounts",
                            f"{len(holders)} verified accounts share one "
                            f"device/IP fingerprint — record only, no action taken")
        return True

    def is_verified(self, player: str) -> bool:
        return str(player) in self._verified

    def flags(self, player: str = "") -> list[dict[str, Any]]:
        if player:
            return [f for f in self._flags if f.get("player") == player]
        return list(self._flags)

    def _flag(self, player: str, kind: str, detail: str) -> None:
        self._flags.append({"player": player, "kind": kind,
                            "detail": detail[:300], "at": time.time(),
                            "action_taken": "none — record only, never auto-ban"})

    # ── ledger ──
    def _new_entry(self, player: str, kind: str, coins: int, xp: int,
                   reason: str, ref: str = "", difficulty: str = "",
                   ts: float | None = None,
                   reverses: str = "") -> MathLedgerEntry:
        self._seq += 1
        return MathLedgerEntry(
            id=f"mled_{self._seq}", player=player, kind=kind,
            amount_coins=int(coins), amount_xp=int(xp),
            reason=(reason or "")[:300], ref=(ref or "")[:120],
            difficulty=difficulty, ts=ts if ts is not None else time.time(),
            reverses=reverses)

    def _earned_in_period(self, player: str, period: str, now: float) -> int:
        key = _period_key(period, now)
        return sum(e.amount_coins for e in self._entries
                   if e.player == player and e.amount_coins > 0
                   and e.kind == "earn"
                   and _period_key(period, e.ts) == key)

    def credit(self, player: str, coins: int, xp: int, reason: str,
               ref: str = "", difficulty: str = "",
               now: float | None = None) -> MathLedgerEntry | None:
        """Credit an earn. Returns None when blocked (unverified,
        non-positive, or either earn cap would be exceeded)."""
        now = now if now is not None else time.time()
        coins, xp = int(coins), int(xp)
        if coins <= 0 and xp <= 0:
            return None
        if not self.is_verified(player):
            return None
        for period, cap in MATH_EARN_CAPS.items():
            if (self._earned_in_period(player, period, now) + max(coins, 0)
                    > cap):
                self._flag(player, f"earn_cap_{period}",
                           f"earn of {coins}c would exceed {period} cap {cap}")
                return None
        entry = self._new_entry(player, "earn", coins, xp, reason,
                                ref=ref, difficulty=difficulty, ts=now)
        self._entries.append(entry)
        if self.credit_fn is not None:
            try:
                self.credit_fn(player, coins, xp, reason)
            except Exception as exc:  # noqa: BLE001
                _log.warning("math credit hook failed for %s: %s", player, exc)
        if self.db is not None:
            try:
                from .economy import record_ledger
                record_ledger(self.db, player, "source", coins,
                              f"math:{ref or reason[:40]}")
            except Exception:  # noqa: BLE001
                pass
        return entry

    def balance(self, player: str) -> dict[str, int]:
        """Balance recomputed from the ledger — the only source of truth."""
        coins = sum(e.amount_coins for e in self._entries if e.player == player)
        xp = sum(e.amount_xp for e in self._entries if e.player == player)
        return {"coins": coins, "xp": xp}

    def balances(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for e in self._entries:
            row = out.setdefault(e.player, {"coins": 0, "xp": 0})
            row["coins"] += e.amount_coins
            row["xp"] += e.amount_xp
        return out

    def verify(self) -> dict[str, Any]:
        """Recompute every balance from scratch; report ledger size."""
        return {"entries": len(self._entries),
                "balances": self.balances(),
                "reversed": sum(1 for e in self._entries if e.reversed_by),
                "reconstructible": True}

    def statement(self, player: str) -> dict[str, Any]:
        entries = [e for e in self._entries if e.player == player]
        return {"player": player,
                "balance": self.balance(player),
                "entries": [e.to_dict() for e in sorted(entries, key=lambda e: e.ts)],
                "flags": self.flags(player),
                "disclaimer": MATH_COINS_DISCLAIMER}

    def reverse(self, entry_id: str, reason: str,
                now: float | None = None) -> MathLedgerEntry | None:
        """Reverse an entry. The original stays; a reversal entry
        offsets it. Never deletes. Returns None when the entry doesn't
        exist, was already reversed, or is itself a reversal."""
        now = now if now is not None else time.time()
        target = next((e for e in self._entries if e.id == entry_id), None)
        if target is None or target.reversed_by or target.kind == "reversal":
            return None
        reversal = self._new_entry(
            target.player, "reversal", -target.amount_coins,
            -target.amount_xp, f"reversal: {reason}", ref=target.id,
            difficulty=target.difficulty, ts=now, reverses=target.id)
        target.reversed_by = reversal.id
        self._entries.append(reversal)
        return reversal

    # ── lessons + quizzes ──
    def register_lesson(self, lesson_id: str, tier: str, title: str,
                        blurb: str = "") -> None:
        self._lessons[lesson_id] = {"tier": tier, "title": title,
                                    "blurb": blurb}

    def complete_lesson(self, player: str, lesson_id: str, passed: bool,
                        streak_days: int = 0,
                        now: float | None = None) -> dict[str, Any]:
        """Complete a lesson + pass its quiz → earn. A failed quiz
        earns nothing. Repeats pay the farming-decayed rate; the
        first clear of a lesson pays the tier's bonus chest."""
        now = now if now is not None else time.time()
        lesson = self._lessons.get(lesson_id)
        if lesson is None:
            return {"ok": False, "error": "unknown_lesson"}
        if not passed:
            return {"ok": False, "error": "quiz_failed",
                    "message": "Quiz not passed — no reward. Study the worked "
                               "solution and retry."}
        clears = self._lesson_clears.get((player, lesson_id), 0)
        reward = reward_for(lesson["tier"], streak_days=streak_days,
                            first_clear=(clears == 0),
                            repeat_count=clears)
        entry = self.credit(player, reward["coins"], reward["xp"],
                            f"lesson: {lesson['title']}", ref=lesson_id,
                            difficulty=lesson["tier"], now=now)
        if entry is None:
            return {"ok": False, "error": "blocked",
                    "message": "Earn blocked (unverified member or earn cap)."}
        self._lesson_clears[(player, lesson_id)] = clears + 1
        return {"ok": True, "lesson": lesson_id, "reward": reward,
                "ledger_id": entry.id, "repeat": clears,
                "disclaimer": MATH_COINS_DISCLAIMER}

    # ── teaching ──
    def explain_solution(self, player: str, problem_id: str,
                         explanation: str) -> dict[str, Any]:
        """Post an explanation of a problem's solution. Earns when
        other members upvote it — not on posting, so spam pays nothing."""
        expl_id = f"expl_{len(self._explanations) + 1}_{_hash_fingerprint(player + problem_id)}"
        self._explanations[expl_id] = {
            "id": expl_id, "author": player, "problem_id": problem_id,
            "text": (explanation or "")[:2000], "upvotes": 0,
            "voters": [], "at": time.time()}
        return {"ok": True, "explanation_id": expl_id,
                "message": "Explanation posted — upvotes from other members earn bonuses."}

    def upvote_explanation(self, expl_id: str, voter: str,
                           now: float | None = None) -> dict[str, Any]:
        """Upvote someone else's explanation. Pays the author the
        teaching bonus, up to the per-explanation cap. Self-upvotes
        and repeat votes pay nothing."""
        now = now if now is not None else time.time()
        expl = self._explanations.get(expl_id)
        if expl is None:
            return {"ok": False, "error": "unknown_explanation"}
        if voter == expl["author"] or voter in expl["voters"]:
            return {"ok": False, "error": "invalid_vote",
                    "message": "Self-upvotes and repeat votes pay nothing."}
        expl["voters"].append(voter)
        if expl["upvotes"] >= TEACH_UPVOTE_CAP:
            expl["upvotes"] += 1
            return {"ok": True, "paid": False,
                    "message": "Upvote counted; this explanation already hit its bonus cap."}
        expl["upvotes"] += 1
        coins, xp = TEACH_UPVOTE_BONUS
        entry = self.credit(expl["author"], coins, xp,
                            "teaching bonus: explanation upvoted",
                            ref=expl_id, now=now)
        if entry is None:
            return {"ok": False, "error": "blocked",
                    "message": "Teaching bonus blocked (unverified or earn cap)."}
        return {"ok": True, "paid": True, "coins": coins, "xp": xp,
                "upvotes": expl["upvotes"], "ledger_id": entry.id}

    # ── problem creation ──
    def register_created_problem(self, player: str, problem_id: str,
                                 tier: str) -> dict[str, Any]:
        self._problems[problem_id] = {"creator": player, "tier": tier,
                                      "fails": 0, "solves": 0}
        return {"ok": True, "problem_id": problem_id, "creator": player,
                "message": "Problem registered — each solver who fails it earns you a creator bonus."}

    def record_solver_outcome(self, problem_id: str, solver: str,
                              solved: bool,
                              now: float | None = None) -> dict[str, Any]:
        """Record a solver's outcome on a player-created problem. A
        failure pays the creator a decaying bonus; a solve pays the
        creator nothing (the problem was beaten)."""
        now = now if now is not None else time.time()
        meta = self._problems.get(problem_id)
        if meta is None:
            return {"ok": False, "error": "unknown_problem"}
        creator = meta["creator"]
        if solved:
            meta["solves"] += 1
            return {"ok": True, "paid": False, "message": "Solved — no creator bonus."}
        if solver == creator:
            return {"ok": False, "error": "self_play",
                    "message": "Your own attempts on your problem pay no creator bonus."}
        fails = meta["fails"]
        meta["fails"] += 1
        base_coins, base_xp = CREATOR_BASE_BONUS
        decay = max(0.10, CREATOR_DECAY ** fails)
        coins, xp = int(round(base_coins * decay)), int(round(base_xp * decay))
        entry = self.credit(creator, coins, xp,
                            "creator bonus: solver failed your problem",
                            ref=problem_id, difficulty=meta["tier"], now=now)
        if entry is None:
            return {"ok": False, "error": "blocked",
                    "message": "Creator bonus blocked (unverified or earn cap)."}
        return {"ok": True, "paid": True, "coins": coins, "xp": xp,
                "fails": meta["fails"], "ledger_id": entry.id}


def first_clear_chest_note(tier: str) -> str:
    """Human line for a tier's first-clear chest."""
    chest = FIRST_CLEAR_CHEST.get(tier, {})
    return (f"First {tier} clear: bonus chest "
            f"+{chest.get('coins', 0)} coins, +{chest.get('xp', 0)} XP.")
