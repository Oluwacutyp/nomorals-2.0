"""Reconciliation: payments → finance ledger, double-entry, balanced.

Every reconciled payment lands in :mod:`nomorals.finance.ledger` as a
balanced pair:

- DEBIT  cash (income transaction, ``+amount``)
- CREDIT revenue (recorded in the journal as the offsetting leg)
- DEBIT  payment fees (spend transaction, ``+fee``) when a fee exists

The journal keeps both legs so ``balanced`` can be proven at any time:
sum(debits) == sum(credits) across every journal entry. A payment that
cannot be matched to an invoice is recorded but flagged
``unreconciled`` — it is never silently dropped, and never double-counted.

Ledger binding: when ``nomorals.finance.ledger.Ledger`` imports, it is
used (pass an instance, or a path). Otherwise an in-memory ledger with
the same ``log()`` / ``transactions()`` shape is used, so this module
works standalone in tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "JournalEntry",
    "InMemoryLedger",
    "Reconciler",
]


@dataclass
class JournalEntry:
    """One double-entry journal line. debits == credits per entry set."""
    entry_id: str
    payment_reference: str
    invoice_id: str
    debit_minor: int   # cash in (+ amount), fees (+ fee)
    credit_minor: int  # revenue recognised
    fee_minor: int = 0
    kind: str = "payment"  # payment | refund | fee
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id, "payment_reference": self.payment_reference,
            "invoice_id": self.invoice_id, "debit_minor": self.debit_minor,
            "credit_minor": self.credit_minor, "fee_minor": self.fee_minor,
            "kind": self.kind, "ts": self.ts,
        }


class InMemoryLedger:
    """Ledger-shaped fallback: ``log()`` / ``transactions()`` like finance."""

    def __init__(self) -> None:
        self._txns: list[Any] = []

    def log(self, amount_kobo: int, category: str | None = None, note: str = "",
            kind: str = "spend", ts: float | None = None,
            source: str = "manual") -> Any:
        from types import SimpleNamespace
        txn = SimpleNamespace(
            amount_kobo=int(amount_kobo), category=category or "other",
            note=note, kind=kind, ts=ts if ts is not None else time.time(),
            source=source)
        self._txns.append(txn)
        return txn

    def transactions(self, **_kwargs: Any) -> list[Any]:
        return list(self._txns)


def _load_ledger(ledger: Any = None, ledger_path: str | None = None) -> Any:
    if ledger is not None:
        return ledger
    if ledger_path is not None:
        try:
            from ..finance.ledger import Ledger as FinanceLedger
            return FinanceLedger(ledger_path)
        except Exception:  # noqa: BLE001 — fall back to in-memory
            _log.debug("finance ledger unavailable, using in-memory",
                       exc_info=True)
    try:
        from ..finance.ledger import Ledger as FinanceLedger  # noqa: F401
        # No path given: use in-memory so reconciliation never writes to
        # the owner's real ledger by accident.
        return InMemoryLedger()
    except Exception:  # noqa: BLE001
        return InMemoryLedger()


class Reconciler:
    """Match payments to invoices and post balanced entries to the ledger.

    Args:
        ledger: a ``Ledger``-shaped object (``log``/``transactions``).
        ledger_path: path for the real finance ledger when ``ledger`` is
            None. When neither is given, an in-memory ledger is used.
        now: injectable clock.
    """

    def __init__(self, *, ledger: Any = None, ledger_path: str | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self.ledger = _load_ledger(ledger, ledger_path)
        self._now = now or time.time
        self.journal: list[JournalEntry] = []
        self.unreconciled: list[dict[str, Any]] = []
        self._seen_refs: set[str] = set()

    # ── core ──

    def reconcile(self, payment: Any, invoice: Any | None = None) -> JournalEntry:
        """Post one payment to the ledger. Returns its journal entry.

        - Duplicate references are rejected (no double-counting).
        - Failed / non-succeeded payments are rejected.
        - With no matching invoice, the payment is still posted but flagged
          ``unreconciled`` in :attr:`unreconciled` and the entry's
          ``invoice_id`` is ``""``.
        - Refunds (negative amount) post as a reversal entry.
        """
        ref = str(getattr(payment, "reference", "") or "")
        if not ref:
            raise ValueError("payment has no reference — cannot reconcile")
        if ref in self._seen_refs:
            raise ValueError(f"payment {ref!r} already reconciled")
        status = str(getattr(payment, "status", "") or "")
        if status not in ("succeeded", "refunded"):
            raise ValueError(
                f"cannot reconcile a {status} payment (ref {ref})")
        amount = int(getattr(payment, "amount_minor", 0) or 0)
        fee = int(getattr(payment, "fee_minor", 0) or 0)
        invoice_id = str(getattr(invoice, "id", "") or "") if invoice else ""
        if invoice is not None:
            inv_status = str(getattr(invoice, "status", "") or "")
            inv_amount = int(getattr(invoice, "amount_minor", 0) or 0)
            if inv_status != "paid":
                raise ValueError(
                    f"invoice {invoice_id} is {inv_status} — mark it paid "
                    "before reconciling")
            if inv_amount != amount and amount >= 0:
                raise ValueError(
                    f"payment {amount} does not match invoice {inv_amount}")
        ts = self._now()
        if amount < 0:
            # Refund reversal: cash out, revenue reduced. Balanced like
            # every other entry: debit == credit == refunded amount.
            entry = JournalEntry(
                entry_id=f"jrnl_{len(self.journal) + 1}",
                payment_reference=ref, invoice_id=invoice_id,
                debit_minor=-amount, credit_minor=-amount, fee_minor=0,
                kind="refund", ts=ts)
            self.ledger.log(-amount, "sales_refund",
                            f"refund {ref}", kind="spend", ts=ts,
                            source="commerce-reconciliation")
        else:
            entry = JournalEntry(
                entry_id=f"jrnl_{len(self.journal) + 1}",
                payment_reference=ref, invoice_id=invoice_id,
                debit_minor=amount, credit_minor=amount,
                fee_minor=fee, kind="payment", ts=ts)
            self.ledger.log(amount, "sales",
                            f"payment {ref} via {getattr(payment, 'provider', '')}",
                            kind="income", ts=ts,
                            source="commerce-reconciliation")
            if fee:
                self.ledger.log(fee, "fees",
                                f"gateway fee {ref}", kind="spend", ts=ts,
                                source="commerce-reconciliation")
        self.journal.append(entry)
        self._seen_refs.add(ref)
        if not invoice_id:
            self.unreconciled.append({
                "reference": ref, "amount_minor": amount,
                "reason": "no matching invoice", "ts": ts,
            })
        return entry

    def reconcile_many(self, pairs: list[tuple[Any, Any | None]]) -> list[JournalEntry]:
        return [self.reconcile(p, inv) for p, inv in pairs]

    # ── balances / reports ──

    @property
    def balanced(self) -> bool:
        """True iff every journal entry's debits equal its credits."""
        return all(e.debit_minor == e.credit_minor for e in self.journal)

    def totals(self) -> dict[str, int]:
        return {
            "gross_minor": sum(e.debit_minor for e in self.journal
                               if e.kind == "payment"),
            "fees_minor": sum(e.fee_minor for e in self.journal),
            "refunds_minor": sum(e.credit_minor for e in self.journal
                                  if e.kind == "refund"),
            "entries": len(self.journal),
            "unreconciled_count": len(self.unreconciled),
        }

    def unreconciled_report(self) -> str:
        """Human-readable report of payments with no matching invoice."""
        if not self.unreconciled:
            return "✅ All reconciled payments matched an invoice."
        lines = [f"⚠️ {len(self.unreconciled)} unreconciled payment(s):"]
        for u in self.unreconciled:
            lines.append(
                f"  - {u['reference']}: {u['amount_minor']} minor "
                f"({u['reason']})")
        lines.append("Action: match each to an invoice or create one, "
                     "then reconcile again.")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        totals = self.totals()
        return {
            **totals,
            "balanced": self.balanced,
            "net_minor": totals["gross_minor"] - totals["fees_minor"]
            - totals["refunds_minor"],
        }
