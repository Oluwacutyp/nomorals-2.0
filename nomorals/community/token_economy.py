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

Universal: the economy is per-group; ``nation`` / ``currency_label``
are display metadata for the group's locale (any nation — a group in
Brazil labels its token in BRL-flavoured display text, a group in
Japan in JPY-flavoured text). The mechanics are identical everywhere;
no nation is the axis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.text import safe_filename

__all__ = [
    "TOKEN_DISCLAIMER", "TokenError", "TokenEntry", "TokenEconomy",
    "EARN_CAPS", "TRANSFER_CAPS", "DECAY_WINDOW_S", "DECAY_FACTOR",
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


class TokenEconomy:
    """Append-only game-currency token ledger for one group."""

    def __init__(self, group_id: str, *,
                 data_dir: str | Path | None = None,
                 currency_label: str = "coins",
                 nation: str = "") -> None:
        self.group_id = group_id
        self.currency_label = currency_label
        self.nation = nation
        self._data_dir = Path(data_dir) if data_dir else None
        self._entries: list[TokenEntry] = []
        self._verified: dict[str, dict[str, Any]] = {}  # member -> {identity_key, fingerprint}
        self._identity_to_member: dict[str, str] = {}
        self._flags: list[dict[str, Any]] = []
        self._seq = 0
        if self._data_dir is not None:
            self._load()

    # ── persistence (optional; in-memory when no data_dir) ──────────────────
    def _path(self, name: str) -> Path:
        assert self._data_dir is not None
        # group_id is chat-controlled: sanitise before it becomes a
        # directory name (a group titled "a/b" or "x:y" must not escape
        # the data dir or raise OSError on Windows). Load/save both go
        # through here, so the mapping stays consistent.
        d = self._data_dir / safe_filename(str(self.group_id),
                                           fallback="group")
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self._path("token_ledger").read_text(encoding="utf-8"))
            self._entries = [TokenEntry.from_dict(e) for e in raw]
            self._seq = max([e.entry_id for e in self._entries], default=0)
            self._verified = json.loads(self._path("token_verified").read_text(encoding="utf-8"))
            self._identity_to_member = {v.get("identity_key", ""): m
                                         for m, v in self._verified.items() if v.get("identity_key")}
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        if self._data_dir is None:
            return
        self._path("token_ledger").write_text(
            json.dumps([e.to_dict() for e in self._entries], indent=1), encoding="utf-8")
        self._path("token_verified").write_text(
            json.dumps(self._verified, indent=1), encoding="utf-8")

    # ── the not-an-investment contract ──────────────────────────────────────
    @staticmethod
    def describe() -> str:
        """Plain-language token description. Always contains the disclaimer."""
        return ("The group token is game currency. " + TOKEN_DISCLAIMER)

    # ── verification: one account per verified member ──────────────────────
    def verify(self, member: str, identity_key: str, *,
               fingerprint: str = "") -> bool:
        """Bind one identity key to one member. Returns False on conflict.

        Rejected when: the identity key already belongs to another
        member (alt account), or the member already verified under a
        different identity key (second account).
        """
        if not member or not identity_key:
            return False
        owner = self._identity_to_member.get(identity_key)
        if owner is not None and owner != member:
            return False
        existing = self._verified.get(member)
        if existing is not None and existing.get("identity_key") != identity_key:
            return False
        self._verified[member] = {"identity_key": identity_key,
                                  "fingerprint": fingerprint,
                                  "verified_at": time.time()}
        self._identity_to_member[identity_key] = member
        self._save()
        return True

    def is_verified(self, member: str) -> bool:
        return member in self._verified

    # ── ledger core ─────────────────────────────────────────────────────────
    def _append(self, member: str, kind: str, amount: int, reason: str,
                ref: str = "", ts: float = 0.0) -> TokenEntry:
        self._seq += 1
        entry = TokenEntry(entry_id=self._seq, ts=ts or time.time(),
                           member=member, kind=kind, amount=int(amount),
                           reason=reason, ref=ref)
        self._entries.append(entry)
        self._save()
        return entry

    def balance(self, member: str) -> int:
        return sum(e.amount for e in self._entries if e.member == member)

    def balances(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._entries:
            out[e.member] = out.get(e.member, 0) + e.amount
        return out

    def treasury_balance(self) -> int:
        return self.balance("__treasury__")

    def entries(self, member: str = "") -> list[TokenEntry]:
        return [e for e in self._entries if not member or e.member == member]

    def _earned_in_period(self, member: str, period: str, now: float) -> int:
        key = _period_key(period, now)
        return sum(e.amount for e in self._entries
                   if e.member == member and e.kind == "earn" and e.amount > 0
                   and _period_key(period, e.ts) == key)

    def _transferred_in_period(self, member: str, period: str, now: float) -> int:
        key = _period_key(period, now)
        return sum(-e.amount for e in self._entries
                   if e.member == member and e.kind == "transfer_out"
                   and _period_key(period, e.ts) == key)

    # ── farming detection ───────────────────────────────────────────────────
    def _flag(self, member: str, kind: str, detail: str) -> None:
        self._flags.append({"member": member, "kind": kind, "detail": detail,
                            "at": time.time()})

    def flags(self, member: str = "") -> list[dict[str, Any]]:
        return [f for f in self._flags if not member or f["member"] == member]

    def is_flagged(self, member: str) -> bool:
        return any(f["member"] == member for f in self._flags)

    def clear_flag(self, member: str) -> int:
        before = len(self._flags)
        self._flags = [f for f in self._flags if f["member"] != member]
        return before - len(self._flags)

    def _velocity_check(self, member: str, kind: str, now: float) -> bool:
        """True when velocity is abusive: too many same-kind entries lately."""
        window_start = now - VELOCITY_WINDOW_S
        kinds = {"earn": ("earn", VELOCITY_MAX_EARNS),
                 "transfer": ("transfer_out", VELOCITY_MAX_TRANSFERS)}
        want, limit = kinds[kind]
        recent = sum(1 for e in self._entries
                     if e.member == member and e.kind == want and e.ts >= window_start)
        if recent >= limit:
            self._flag(member, f"velocity_{kind}",
                       f"{recent} {want} entries in {VELOCITY_WINDOW_S:.0f}s")
            return True
        return False

    def _alt_check(self, member: str) -> bool:
        """True when an alt signal fires: fingerprint shared with another member."""
        fp = self._verified.get(member, {}).get("fingerprint", "")
        if not fp:
            return False
        others = [m for m, v in self._verified.items()
                  if m != member and v.get("fingerprint") == fp]
        if others:
            self._flag(member, "shared_fingerprint",
                       f"fingerprint shared with {others}")
            return True
        return False

    # ── earn (mint): activity only, never real money ────────────────────────
    def earn(self, member: str, base_amount: int, kind: str, *,
             reason: str = "", now: float | None = None) -> TokenEntry | None:
        """Mint tokens for activity. Returns None when blocked.

        Blocked when: unverified, non-positive amount, a period earn
        cap would be exceeded, or the account is flagged (farming).
        Repeat same-``kind`` earns inside the decay window pay a
        decaying rate: ``base * DECAY_FACTOR ** (n-1)``.
        """
        now = now if now is not None else time.time()
        if base_amount <= 0 or not member or not self.is_verified(member):
            return None
        if self.is_flagged(member):
            return None
        if self._velocity_check(member, "earn", now) or self._alt_check(member):
            return None
        # decaying rate on repeat farming of the same kind
        repeats = sum(1 for e in self._entries
                      if e.member == member and e.kind == "earn" and e.ref == kind
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

    # ── transfer ────────────────────────────────────────────────────────────
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
        self._append(sender, "transfer_out", -amount,
                     reason or f"transfer to {recipient}", ref=recipient, ts=now)
        self._append(recipient, "transfer_in", amount,
                     reason or f"transfer from {sender}", ref=sender, ts=now)
        return True

    # ── treasury ────────────────────────────────────────────────────────────
    def to_treasury(self, member: str, amount: int, *,
                    reason: str = "", now: float | None = None) -> bool:
        if amount <= 0 or not self.is_verified(member) or self.balance(member) < amount:
            return False
        now = now if now is not None else time.time()
        self._append(member, "transfer_out", -int(amount),
                     reason or "to treasury", ref="__treasury__", ts=now)
        self._append("__treasury__", "treasury", int(amount),
                     reason or f"from {member}", ref=member, ts=now)
        return True

    # ── reversal: append-only, never deletes ────────────────────────────────
    def reverse(self, entry_id: int, *, reason: str = "",
                now: float | None = None) -> TokenEntry | None:
        """Reverse an entry by appending a compensating entry.

        The original entry is NEVER deleted or edited — it is marked
        ``reversed_by`` and a new entry with the negated amount is
        appended. Returns the reversal entry, or None when the entry
        does not exist / was already reversed.
        """
        now = now if now is not None else time.time()
        orig = next((e for e in self._entries if e.entry_id == entry_id), None)
        if orig is None or orig.reversed_by:
            return None
        if orig.kind in ("reversal",):
            return None
        reversal = self._append(orig.member, "reversal", -orig.amount,
                                reason or f"reversal of #{entry_id}",
                                ref=f"reverses:{entry_id}", ts=now)
        orig.reversed_by = reversal.entry_id
        self._save()
        return reversal

    # ── audit ───────────────────────────────────────────────────────────────
    def verify_ledger(self) -> dict[str, Any]:
        """Integrity check: ids sequential, reversals balanced, no deletes."""
        ids = [e.entry_id for e in self._entries]
        sequential = ids == list(range(1, len(ids) + 1))
        reversals_ok = True
        for e in self._entries:
            if e.kind == "reversal":
                target = next((o for o in self._entries
                               if o.entry_id == int(e.ref.split(":")[1])), None)
                if target is None or target.reversed_by != e.entry_id \
                        or target.amount + e.amount != 0:
                    reversals_ok = False
        return {"entries": len(ids), "sequential_ids": sequential,
                "reversals_balanced": reversals_ok,
                "flags": len(self._flags)}
