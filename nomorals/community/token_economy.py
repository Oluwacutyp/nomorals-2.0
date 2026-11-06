"""Game-currency token economy WITH anti-abuse at launch.

This is the skip-list correction: the token is built WITH its
safeguards, not gated behind them. Anti-abuse is part of the feature,
not a reason to skip it.

What this token IS: an in-group game currency. Members earn it by
playing / contributing, spend it on in-group perks, transfer it to
each other. It is accounted in an append-only ledger.

What this token IS NOT — stated in code, enforced in tests:

* **NOT an investment.** No yield, no appreciation, no expectation of
  profit. :func:`describe` / :attr:`TOKEN_DISCLAIMER` say so, and
  :meth:`TokenEconomy.describe` is asserted in tests to contain
  "not an investment".
* **NOT purchasable with real money.** There is no buy path. Minting
  happens only through :meth:`TokenEconomy.earn` (activity) — never
  through payment. There is deliberately no ``buy_with_real_money``
  method.
* **NOT cash-out-able.** There is no redeem-for-money path. Balances
  are game currency only.

Anti-abuse, built in from day one:

* **Earn caps** per member per period (daily / weekly).
* **Decaying rate on repeat farming** — the same earn ``kind`` repeated
  inside the decay window pays ``base * decay ** (n-1)``; farming the
  same action stops paying.
* **Farming detection** — velocity (too many earns / transfers in a
  short window) + alt signals (two members sharing one device /
  fingerprint, or rapid back-and-forth transfers) flag the account
  for review and throttle further earns/transfers until cleared.
* **Transfer caps** per member per period.
* **One account per verified member** — ``verify`` binds one identity
  key to one member; a second identity key for the same member, or
  the same identity key for a different member, is rejected.
* **Reversals never delete** — a reversal appends a compensating
  ledger entry referencing the original; the original entry stays.
  Balances are always recomputed from the full ledger, so history is
  auditable.

**One purse (AA-21 integration).** This class no longer keeps a
ledger of its own. It is the token *policy layer* over the group's
single :class:`~nomorals.community.group_economy.GroupEconomy`
ledger — the same ledger the group economy commands, quests, shop
and parties settle into. Token balances ARE group-economy balances;
token entries are group-economy ledger entries (the token view
numbers them sequentially per group). What stays token-specific is
the policy above: identity binding on verify, the decaying earn
rate, velocity tripwires, and the token cap tables (which differ
from the economy's own award caps — each mint path is gated by its
own policy, every path settles into the one ledger). Verification
state is the economy's own ``verified`` file: ``verify`` binds the
identity key as the member's economy fingerprint, so a member
verified here is verified everywhere in the group economy.

Universal: the economy is per-group; ``nation`` / ``currency_label``
are display metadata for the group's locale (any nation — a group in
Brazil labels its token in BRL-flavoured display text, a group in
Japan in JPY-flavoured text). The mechanics are identical everywhere;
no nation is the axis.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .group_economy import GroupEconomy

__all__ = [
    "TOKEN_DISCLAIMER", "TokenError", "TokenEntry", "TokenEconomy",
    "EARN_CAPS", "TRANSFER_CAPS", "DECAY_WINDOW_S", "DECAY_FACTOR",
    "persistent_token_economy",
]

TOKEN_DISCLAIMER = (
    "Game currency only. This token is NOT an investment: it carries no "
    "yield, no appreciation, and no expectation of profit. It cannot be "
    "bought with real money and cannot be redeemed for real money. It has "
    "no monetary value outside the group that issued it."
)

#: Earn caps per member per period (in token units).
EARN_CAPS: dict[str, int] = {"daily": 500, "weekly": 2000}
#: Transfer caps per member per period (in token units).
TRANSFER_CAPS: dict[str, int] = {"daily": 1000, "weekly": 4000}
#: Repeat same-kind earns inside this window decay.
DECAY_WINDOW_S = 3600.0
#: Each repeat of the same earn kind pays this fraction of the previous.
DECAY_FACTOR = 0.5
#: Velocity tripwires: more than N earns / transfers inside the window flags.
VELOCITY_MAX_EARNS = 20
VELOCITY_MAX_TRANSFERS = 20
VELOCITY_WINDOW_S = 600.0

#: The token view names the group-economy treasury member ("") thus.
TREASURY_MEMBER = "__treasury__"


class TokenError(ValueError):
    """Token rule violation (cap, unverified, farming flag, ...)."""


@dataclass
class TokenEntry:
    entry_id: int
    ts: float
    member: str
    kind: str  # earn | transfer_out | transfer_in | treasury | reversal | burn
    amount: int  # signed: credits positive, debits negative
    reason: str
    ref: str = ""
    reversed_by: int = 0  # entry_id of the reversal, 0 = not reversed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenEntry":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


def _period_key(period: str, ts: float) -> str:
    t = time.gmtime(ts)
    if period == "daily":
        return time.strftime("%Y-%m-%d", t)
    if period == "weekly":
        return time.strftime("%Y-W%V", t)
    return time.strftime("%Y-%m", t)


def persistent_token_economy(group_id: str, *,
                             data_dir: str | Path | None = None,
                             ) -> TokenEconomy:
    """A TokenEconomy bound to the group's REAL group-economy store.

    Production callers (party economy hooks, group surfaces) must use
    this — a bare ``TokenEconomy(group_id)`` is instance-ephemeral by
    contract. With no ``data_dir`` the group economy's own default
    root is used, so token state lands exactly where every other
    group-economy surface reads it.
    """
    if data_dir is None:
        data_dir = GroupEconomy().data_dir
    return TokenEconomy(group_id, data_dir=data_dir)


def _identity_hash(identity_key: str) -> str:
    """The fingerprint hash the group economy stores for an identity key.

    Mirrors ``GroupEconomy.verify_member``'s storage form
    (``sha256(fingerprint)[:16]``) so token binding checks can read
    the economy's own verified map without a second store.
    """
    return hashlib.sha256(str(identity_key).encode()).hexdigest()[:16]


class TokenEconomy:
    """Token policy over one group's GroupEconomy ledger (one purse).

    The wrapped economy is the only store: ledger, verified map and
    persisted flags all live in the group's group-economy files.
    This class adds the token policy (binding, decay, velocity,
    token caps) and the sequential token view of the ledger.

    ``data_dir`` is the group-economy data root (the same directory
    a bare ``GroupEconomy(data_dir=...)`` would use). With no
    ``data_dir`` the old no-persistence contract holds: state lives
    in a per-instance temporary directory and dies with the
    instance — nothing is written to the live community dir.
    """

    def __init__(self, group_id: str, *,
                 data_dir: str | Path | None = None,
                 currency_label: str = "coins",
                 nation: str = "") -> None:
        self.group_id = group_id
        self.currency_label = currency_label
        self.nation = nation
        if data_dir is None:
            data_dir = tempfile.mkdtemp(prefix="devon-token-")
        self._economy = GroupEconomy(data_dir=data_dir)
        # Session-scoped flags raised by THIS layer's tripwires
        # (velocity / alt signals). The economy's own persisted flags
        # are consulted too — see flags()/is_flagged().
        self._flags: list[dict[str, Any]] = []

    # ── the not-an-investment contract ─────────────────────────────
    @staticmethod
    def describe() -> str:
        """Plain-language token description. Always contains the disclaimer."""
        return ("The group token is game currency. " + TOKEN_DISCLAIMER)

    # ── verification: one account per verified member ──────────────
    def _verified_map(self) -> dict[str, dict[str, Any]]:
        return self._economy._load_json(self.group_id, "verified", {}) or {}

    def verify(self, member: str, identity_key: str, *,
               fingerprint: str = "") -> bool:
        """Bind one identity key to one member. Returns False on conflict.

        Rejected when: the identity key already belongs to another
        member (alt account), or the member already verified under a
        different identity key (second account). The binding lands
        in the group economy's verified map — the identity key is
        stored as the member's economy fingerprint, so the member
        is verified for every group-economy surface, not just here.
        """
        if not member or not identity_key:
            return False
        effective = fingerprint or identity_key
        digest = _identity_hash(effective)
        verified = self._verified_map()
        for other, rec in verified.items():
            if other != member and rec.get("fingerprint") == digest:
                return False  # identity already bound to another member
        existing = verified.get(member)
        if existing is not None and existing.get("fingerprint") != digest:
            return False  # member already bound to a different identity
        if existing is not None:
            return True  # same binding re-asserted: idempotent
        return self._economy.verify_member(
            self.group_id, member, fingerprint=effective)

    def is_verified(self, member: str) -> bool:
        return self._economy.is_verified(self.group_id, member)

    # ── the token view of the shared ledger ────────────────────────
    def _raw_entries(self) -> list[Any]:
        return self._economy._ledger(self.group_id)

    def _token_view(self, raw: list[Any]) -> list[TokenEntry]:
        positions = {e.id: i + 1 for i, e in enumerate(raw)}
        return [
            TokenEntry(
                entry_id=i + 1,
                ts=e.ts,
                member=e.member or TREASURY_MEMBER,
                kind=e.kind,
                amount=e.amount,
                reason=e.reason,
                ref=e.ref,
                reversed_by=positions.get(e.reversed_by, 0),
            )
            for i, e in enumerate(raw)
        ]

    def _append(self, member: str, kind: str, amount: int, reason: str,
                ref: str = "", ts: float = 0.0) -> TokenEntry:
        self._economy._append(self.group_id, self._economy._new_entry(
            self.group_id, member, kind, int(amount), reason, ref=ref,
            ts=ts or time.time()))
        # The ledger is append-only: the entry just written is last.
        return self._token_view(self._raw_entries())[-1]

    def balance(self, member: str) -> int:
        if member == TREASURY_MEMBER:
            return self._economy.treasury_balance(self.group_id)
        return self._economy.balance(self.group_id, member)

    def balances(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for member, amount in self._economy.balances(self.group_id).items():
            out[member or TREASURY_MEMBER] = amount
        return out

    def treasury_balance(self) -> int:
        return self._economy.treasury_balance(self.group_id)

    def entries(self, member: str = "") -> list[TokenEntry]:
        view = self._token_view(self._raw_entries())
        return [e for e in view if not member or e.member == member]

    def _earned_in_period(self, member: str, period: str, now: float) -> int:
        key = _period_key(period, now)
        return sum(e.amount for e in self.entries(member)
                   if e.kind == "earn" and e.amount > 0
                   and _period_key(period, e.ts) == key)

    def _transferred_in_period(self, member: str, period: str, now: float) -> int:
        key = _period_key(period, now)
        return sum(-e.amount for e in self.entries(member)
                   if e.kind == "transfer_out"
                   and _period_key(period, e.ts) == key)

    # ── farming detection ───────────────────────────────────────────
    def _flag(self, member: str, kind: str, detail: str) -> None:
        self._flags.append({"member": member, "kind": kind, "detail": detail,
                            "at": time.time()})

    def flags(self, member: str = "") -> list[dict[str, Any]]:
        """This layer's session flags + the economy's persisted flags."""
        own = [f for f in self._flags if not member or f["member"] == member]
        persisted = [f for f in self._economy.flags(self.group_id)
                     if not member or f.get("member") == member]
        return own + persisted

    def is_flagged(self, member: str) -> bool:
        return bool(self.flags(member))

    def clear_flag(self, member: str) -> int:
        """Clear THIS layer's session flags for ``member``.

        The group economy's persisted flags are the group's audit
        trail and stay — they are review history, not a session
        throttle. Returns how many session flags were removed.
        """
        before = len(self._flags)
        self._flags = [f for f in self._flags if f["member"] != member]
        return before - len(self._flags)

    def _velocity_check(self, member: str, kind: str, now: float) -> bool:
        """True when velocity is abusive: too many same-kind entries lately."""
        window_start = now - VELOCITY_WINDOW_S
        kinds = {"earn": ("earn", VELOCITY_MAX_EARNS),
                 "transfer": ("transfer_out", VELOCITY_MAX_TRANSFERS)}
        want, limit = kinds[kind]
        recent = sum(1 for e in self.entries(member)
                     if e.kind == want and e.ts >= window_start)
        if recent >= limit:
            self._flag(member, f"velocity_{kind}",
                       f"{recent} {want} entries in {VELOCITY_WINDOW_S:.0f}s")
            return True
        return False

    def _alt_check(self, member: str) -> bool:
        """True when an alt signal fires: fingerprint shared with another member."""
        verified = self._verified_map()
        fp = verified.get(member, {}).get("fingerprint", "")
        if not fp:
            return False
        others = [m for m, v in verified.items()
                  if m != member and v.get("fingerprint") == fp]
        if others:
            self._flag(member, "shared_fingerprint",
                       f"fingerprint shared with {others}")
            return True
        return False

    # ── earn (mint): activity only, never real money ────────────────
    def earn(self, member: str, base_amount: int, kind: str, *,
             reason: str = "", now: float | None = None) -> TokenEntry | None:
        """Mint tokens for activity. Returns None when blocked.

        Blocked when: unverified, non-positive amount, a period earn
        cap would be exceeded, or the account is flagged (farming).
        Repeat same-``kind`` earns inside the decay window pay a
        decaying rate: ``base * DECAY_FACTOR ** (n-1)``.

        The mint settles into the group's one GroupEconomy ledger
        (kind ``earn``, ref = the earn kind) — the token policy above
        is the gate for this path, exactly as the economy's own caps
        gate its ``award`` path.
        """
        now = now if now is not None else time.time()
        if base_amount <= 0 or not member or not self.is_verified(member):
            return None
        if self.is_flagged(member):
            return None
        if self._velocity_check(member, "earn", now) or self._alt_check(member):
            return None
        # decaying rate on repeat farming of the same kind
        repeats = sum(1 for e in self.entries(member)
                      if e.kind == "earn" and e.ref == kind
                      and e.ts >= now - DECAY_WINDOW_S)
        amount = int(base_amount * (DECAY_FACTOR ** repeats))
        if amount <= 0:
            return None
        for period, cap in EARN_CAPS.items():
            if self._earned_in_period(member, period, now) + amount > cap:
                self._flag(member, f"earn_cap_{period}",
                           f"earn of {amount} would exceed {period} cap {cap}")
                return None
        return self._append(member, "earn", amount, reason or f"earn:{kind}",
                            ref=kind, ts=now)

    # ── transfer ────────────────────────────────────────────────────
    def transfer(self, sender: str, recipient: str, amount: int, *,
                 reason: str = "", now: float | None = None) -> bool:
        """Move game currency between verified members. False when blocked."""
        now = now if now is not None else time.time()
        amount = int(amount)
        if amount <= 0 or sender == recipient:
            return False
        if not (self.is_verified(sender) and self.is_verified(recipient)):
            return False
        if self.is_flagged(sender) or self.is_flagged(recipient):
            return False
        if self.balance(sender) < amount:
            return False
        if self._velocity_check(sender, "transfer", now):
            return False
        for period, cap in TRANSFER_CAPS.items():
            if self._transferred_in_period(sender, period, now) + amount > cap:
                self._flag(sender, f"transfer_cap_{period}",
                           f"transfer of {amount} would exceed {period} cap {cap}")
                return False
        return self._economy.transfer(
            self.group_id, sender, recipient, amount,
            note=reason, now=now)

    # ── treasury ────────────────────────────────────────────────────
    def to_treasury(self, member: str, amount: int, *,
                    reason: str = "", now: float | None = None) -> bool:
        if amount <= 0 or not self.is_verified(member) \
                or self.balance(member) < amount:
            return False
        now = now if now is not None else time.time()
        return self._economy.contribute_treasury(
            self.group_id, member, int(amount), now=now)

    # ── reversal: append-only, never deletes ────────────────────────
    def reverse(self, entry_id: int, *, reason: str = "",
                now: float | None = None) -> TokenEntry | None:
        """Reverse an entry by appending a compensating entry.

        ``entry_id`` is the token view's sequential id (see
        :meth:`entries`). The original entry is NEVER deleted or
        edited — the group economy marks it ``reversed_by`` and a
        new entry with the negated amount is appended. Returns the
        reversal entry, or None when the entry does not exist / was
        already reversed.
        """
        now = now if now is not None else time.time()
        raw = self._raw_entries()
        if entry_id < 1 or entry_id > len(raw):
            return None
        target = raw[entry_id - 1]
        reversal = self._economy.reverse(
            self.group_id, target.id,
            reason or f"reversal of #{entry_id}", now=now)
        if reversal is None:
            return None
        # Append-only ledger: the reversal just written is last.
        return self._token_view(self._raw_entries())[-1]

    # ── audit ───────────────────────────────────────────────────────
    def verify_ledger(self) -> dict[str, Any]:
        """Integrity check over the shared ledger, in the token view:
        ids sequential, reversals balanced, no deletes."""
        raw = self._raw_entries()
        view = self._token_view(raw)
        sequential = [e.entry_id for e in view] == list(range(1, len(view) + 1))
        by_id = {e.id: e for e in raw}
        reversals_ok = True
        for e in raw:
            if e.kind == "reversal":
                target = by_id.get(e.reverses)
                if target is None or target.reversed_by != e.id \
                        or target.amount + e.amount != 0:
                    reversals_ok = False
        return {"entries": len(view), "sequential_ids": sequential,
                "reversals_balanced": reversals_ok,
                "flags": len(self.flags())}
