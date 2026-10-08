"""Agent payment mandates — money architecture, not policy (#69).

A mandate is the agent's standing authority to move money: who it acts for,
what scope, per-transaction and per-day caps, expiry, revocability. Every
money-moving dispatch checks the active mandate BEFORE executing — this is
structural, not advisory.

Credential isolation (Coinbase Agentic Wallets pattern): a mandate references
vault-held credentials by ID (``credential_ref``). The model never touches
raw keys — the mandate record must never contain a secret.

Revocation is instant: ``revoke()`` / ``revoke_all("owner")`` ("stop all
spending") kills authority immediately; the next dispatch check fails.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .budgets import finance_paths
from .ledger import Ledger, format_naira

_log = logging.getLogger(__name__)

__all__ = [
    "MandateError",
    "PaymentMandate",
    "MandateStore",
    "MandateCheck",
    "issue_mandate",
    "require_mandate",
    "check_mandate",
    "daily_transfer_spend",
]

#: Scope that covers every money-moving operation.
SCOPE_ALL = "all"

#: Scope for bank transfers (Paystack ``POST /transfer``).
SCOPE_TRANSFER = "transfer"


class MandateError(Exception):
    """No usable mandate — money must not move."""


@dataclass
class PaymentMandate:
    """Standing authority for the agent to move money."""

    id: str
    agent_id: str          # which agent holds this mandate
    principal: str         # who the agent acts for ("owner")
    scope: str             # "transfer" | "all" | ...
    cap_per_txn: int       # kobo — max per single transaction
    cap_per_day: int       # kobo — max total per calendar day
    expires_at: float      # epoch seconds
    revocable: bool = True
    credential_ref: str = ""  # vault key reference, NEVER a raw secret
    created_at: float = 0.0
    revoked_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    def covers(self, scope: str) -> bool:
        return self.scope == SCOPE_ALL or self.scope == scope


@dataclass
class MandateCheck:
    """Result of validating a mandate against a planned spend."""

    ok: bool
    mandate: PaymentMandate | None = None
    reason: str = ""
    daily_spent_kobo: int = 0
    daily_remaining_kobo: int = 0


def _looks_like_secret(value: str) -> bool:
    """Heuristic: refuse to store anything that smells like a raw key."""
    v = (value or "").strip()
    if not v:
        return False
    # sk_live_/sk_test_, long hex/base64 blobs, "bearer " prefixes
    if v.startswith(("sk_live_", "sk_test_", "rk_live_", "rk_test_")):
        return True
    if v.lower().startswith("bearer "):
        return True
    if len(v) >= 32 and all(c in "0123456789abcdefABCDEF" for c in v):
        return True
    return False


class MandateStore:
    """Persistent mandate registry.

    JSON at ``~/.nomorals/finance/mandates.json`` (mirrors RecipientStore).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            _, budgets_path = finance_paths(None)
            path = Path(budgets_path).parent / "mandates.json"
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] | None = None

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._load(), indent=2), encoding="utf-8")

    def _put(self, mandate: PaymentMandate) -> None:
        self._load()[mandate.id] = mandate.to_dict()
        self._save()

    # ── issue / revoke ───────────────────────────────────────────
    def issue(
        self,
        *,
        principal: str = "owner",
        scope: str = SCOPE_TRANSFER,
        cap_per_txn: int,
        cap_per_day: int,
        ttl_days: float = 30.0,
        agent_id: str = "devon",
        credential_ref: str = "",
        revocable: bool = True,
    ) -> PaymentMandate:
        """Create a mandate. Owner-only — the chat layer gates this."""
        if cap_per_txn <= 0 or cap_per_day <= 0:
            raise MandateError("caps must be positive (kobo)")
        if cap_per_txn > cap_per_day:
            raise MandateError("per-transaction cap cannot exceed the daily cap")
        if _looks_like_secret(credential_ref):
            raise MandateError(
                "credential_ref must be a vault reference, never a raw secret")
        now = time.time()
        mandate = PaymentMandate(
            id="mand_" + uuid.uuid4().hex[:12],
            agent_id=agent_id,
            principal=(principal or "owner").strip(),
            scope=(scope or SCOPE_TRANSFER).strip(),
            cap_per_txn=int(cap_per_txn),
            cap_per_day=int(cap_per_day),
            expires_at=now + float(ttl_days) * 86400,
            revocable=bool(revocable),
            credential_ref=(credential_ref or "").strip(),
            created_at=now,
        )
        self._put(mandate)
        _log.info("mandate issued: %s scope=%s cap=%s/day=%s",
                  mandate.id, mandate.scope,
                  format_naira(mandate.cap_per_txn),
                  format_naira(mandate.cap_per_day))
        return mandate

    def get(self, mandate_id: str) -> PaymentMandate | None:
        data = self._load().get(mandate_id or "")
        if not data:
            return None
        try:
            return PaymentMandate(**data)
        except TypeError:
            return None

    def active(self, principal: str = "owner",
               scope: str = SCOPE_TRANSFER) -> PaymentMandate | None:
        """The currently usable mandate for (principal, scope), if any."""
        now = time.time()
        best: PaymentMandate | None = None
        for data in self._load().values():
            m = self.get(data.get("id", ""))
            if m is None:
                continue
            if m.principal != principal or not m.covers(scope):
                continue
            if m.revoked or m.expires_at <= now:
                continue
            if best is None or m.created_at > best.created_at:
                best = m
        return best

    def list(self, principal: str = "") -> list[PaymentMandate]:
        out = []
        for data in self._load().values():
            m = self.get(data.get("id", ""))
            if m is None:
                continue
            if principal and m.principal != principal:
                continue
            out.append(m)
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def revoke(self, mandate_id: str) -> bool:
        """Revoke one mandate. Instant — the next dispatch check fails."""
        m = self.get(mandate_id)
        if m is None or m.revoked:
            return False
        if not m.revocable:
            raise MandateError(f"mandate {mandate_id} is irrevocable")
        m.revoked_at = time.time()
        self._put(m)
        _log.warning("mandate revoked: %s", mandate_id)
        return True

    def revoke_all(self, principal: str = "owner") -> int:
        """Kill every revocable mandate for a principal. Instant."""
        count = 0
        for m in self.list(principal):
            if not m.revoked and m.revocable:
                m.revoked_at = time.time()
                self._put(m)
                count += 1
        _log.warning("revoked %d mandate(s) for %s", count, principal)
        return count


def daily_transfer_spend(ledger: Ledger, *, now: float | None = None) -> int:
    """Kobo moved out as transfers since local midnight. Never raises."""
    try:
        now = now if now is not None else time.time()
        midnight = now - (now % 86400)
        txns = ledger.transactions(since=midnight, kind="spend",
                                   category="transfer")
        return sum(max(0, int(t.amount_kobo)) for t in txns)
    except Exception:  # noqa: BLE001 - caps must fail safe, not loud
        _log.debug("daily transfer spend unreadable", exc_info=True)
        return 0


def check_mandate(
    store: MandateStore,
    principal: str,
    scope: str,
    amount_kobo: int,
    *,
    ledger: Ledger | None = None,
    now: float | None = None,
) -> MandateCheck:
    """Validate authority for a planned spend. Pure check, no side effects."""
    now = now if now is not None else time.time()
    m = store.active(principal, scope)
    if m is None:
        # Distinguish "never had one" from "had one but it's dead" —
        # the message tells the owner exactly what to do.
        any_mandates = [x for x in store.list(principal) if x.covers(scope)]
        if any_mandates:
            latest = any_mandates[0]
            if latest.revoked:
                reason = (f"mandate {latest.id} was revoked — issue a new "
                          f"one with /mandate issue")
            else:
                reason = (f"mandate {latest.id} expired — issue a new one "
                          f"with /mandate issue")
        else:
            reason = ("no payment mandate for this scope — the owner must "
                      "issue one with /mandate issue before money can move")
        return MandateCheck(ok=False, reason=reason)
    if amount_kobo > m.cap_per_txn:
        return MandateCheck(
            ok=False, mandate=m,
            reason=(f"{format_naira(amount_kobo)} exceeds the per-transaction "
                    f"cap of {format_naira(m.cap_per_txn)} "
                    f"(mandate {m.id})"))
    spent = daily_transfer_spend(ledger, now=now) if ledger is not None else 0
    if spent + amount_kobo > m.cap_per_day:
        return MandateCheck(
            ok=False, mandate=m, daily_spent_kobo=spent,
            reason=(f"{format_naira(amount_kobo)} would exceed the daily cap "
                    f"of {format_naira(m.cap_per_day)} "
                    f"({format_naira(spent)} already spent today; "
                    f"mandate {m.id})"))
    return MandateCheck(ok=True, mandate=m, daily_spent_kobo=spent,
                        daily_remaining_kobo=m.cap_per_day - spent - amount_kobo)


def require_mandate(
    store: MandateStore | None,
    principal: str,
    scope: str,
    amount_kobo: int,
    *,
    ledger: Ledger | None = None,
) -> PaymentMandate:
    """Dispatch-path gate. Raises MandateError when money must not move.

    Call this at the top of every money-moving dispatch — it is structural,
    not advisory. ``store=None`` resolves the default store.
    """
    store = store or MandateStore()
    result = check_mandate(store, principal, scope, amount_kobo, ledger=ledger)
    if not result.ok:
        raise MandateError(result.reason)
    assert result.mandate is not None
    return result.mandate


def issue_mandate(
    store: MandateStore | None = None,
    *,
    principal: str = "owner",
    scope: str = SCOPE_TRANSFER,
    cap_per_txn: int,
    cap_per_day: int,
    ttl_days: float = 30.0,
    agent_id: str = "devon",
    credential_ref: str = "",
) -> PaymentMandate:
    """Convenience wrapper over ``MandateStore.issue``."""
    return (store or MandateStore()).issue(
        principal=principal, scope=scope, cap_per_txn=cap_per_txn,
        cap_per_day=cap_per_day, ttl_days=ttl_days, agent_id=agent_id,
        credential_ref=credential_ref)
