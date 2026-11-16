"""Durable commerce store: the eight dormant organs' records as one JSON file.

Layout: ``<home>/data/commerce.json`` (the /store surface points it at
Devon's home; in-memory when no path is given)::

    {"products": [<Catalog product dict>, ...],
     "payments": [<Payment.to_dict()>, ...],
     "invoices": [<Invoice.to_dict()>, ...],
     "subscriptions": [<Subscription.to_dict()>, ...],
     "disputes": [<Dispute.to_dict()>, ...],
     "tickets": [<Ticket.to_dict()>, ...],
     "shipments": [{"id", "carrier", "waybill", "order_ref", ...}, ...],
     "journal": [<JournalEntry.to_dict()>, ...],
     "unreconciled": [{"reference", "amount_minor", "reason", "ts"}, ...]}

The persistence shape mirrors
:class:`nomorals.shows.store.ShowStore`: tolerant load (a missing file
starts empty, a bad row is skipped), atomic tmp+replace save after
every mutation. A file that is not JSON at all is QUARANTINED (renamed
to ``commerce.json.corrupt-<epoch>``) and the load RAISES — a torn or
hand-edited money store must never silently start empty and lose every
payment. The organs themselves (``payments.py``, ``catalog.py``, ...)
stay pure: they never touch this file. This store rehydrates them
from the recorded dicts and writes them back.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["CommerceStore", "default_commerce_path"]


def default_commerce_path(home: Path | str | None = None) -> Path:
    """The durable commerce file under a home (``<home>/data/commerce.json``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "commerce.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_SECTIONS: tuple[str, ...] = (
    "products", "payments", "invoices", "subscriptions", "disputes",
    "tickets", "shipments", "journal", "unreconciled",
)


def _is_row(section: str, row: dict[str, Any]) -> bool:
    """A stored row is well-formed enough to rehydrate, else skipped."""
    required: dict[str, tuple[str, ...]] = {
        "products": ("id", "name", "price_minor"),
        "payments": ("id", "reference", "amount_minor"),
        "invoices": ("id", "customer", "amount_minor"),
        "subscriptions": ("id", "customer", "amount_minor"),
        "disputes": ("id", "payment_reference", "amount_minor"),
        "tickets": ("id", "message"),
        "shipments": ("id", "waybill"),
    }
    keys = required.get(section, ("id",))
    return all(row.get(key) not in (None, "") for key in keys)


class CommerceStore:
    """Commerce records, durable JSON, one atomic file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._data: dict[str, list[dict[str, Any]]] = {
            section: [] for section in _SECTIONS}
        if self.path is not None:
            self._load()

    # ── load / save ────────────────────────────────────────────────

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            bad = self.path.with_name(
                f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self.path, bad)
            except OSError:
                bad = self.path
            raise ValueError(
                f"commerce store is corrupt — quarantined to {bad}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"commerce store must hold an object, found "
                f"{type(raw).__name__}")
        for section in _SECTIONS:
            rows = raw.get(section, []) or []
            kept = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # Journal/unreconciled rows key on reference, not id.
                if section in ("journal", "unreconciled"):
                    if row.get("payment_reference") or row.get("reference"):
                        kept.append(dict(row))
                    continue
                if _is_row(section, row):
                    kept.append(dict(row))
            self._data[section] = kept

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path,
            json.dumps({section: list(rows)
                        for section, rows in self._data.items()},
                       indent=2, sort_keys=True))

    def _rows(self, section: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self._data[section]]

    def _set(self, section: str, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._data[section] = [dict(row) for row in rows]
            self._save()

    # ── catalog ────────────────────────────────────────────────────

    def catalog(self) -> Any:
        """A :class:`Catalog` over the stored products."""
        from .catalog import Catalog

        return Catalog(self._rows("products"))

    def save_catalog(self, catalog: Any) -> None:
        self._set("products", catalog.list())

    # ── payments ───────────────────────────────────────────────────

    def gateway(self, *, http: Any = None, keys: Any = None) -> Any:
        """A :class:`PaymentGateway` rehydrated from the stored records."""
        from .payments import Dispute, Invoice, Payment, PaymentGateway, Subscription

        gw = PaymentGateway(keys=keys, http=http)
        gw.payments = [Payment(**row) for row in self._rows("payments")]
        gw.invoices = [Invoice(**row) for row in self._rows("invoices")]
        gw.subscriptions = [Subscription(**row)
                            for row in self._rows("subscriptions")]
        gw.disputes = [Dispute(**row) for row in self._rows("disputes")]
        return gw

    def save_gateway(self, gateway: Any) -> None:
        with self._lock:
            self._data["payments"] = [p.to_dict()
                                      for p in gateway.payments]
            self._data["invoices"] = [i.to_dict()
                                      for i in gateway.invoices]
            self._data["subscriptions"] = [s.to_dict()
                                           for s in gateway.subscriptions]
            self._data["disputes"] = [d.to_dict()
                                      for d in gateway.disputes]
            self._save()

    # ── support ────────────────────────────────────────────────────

    def inbox(self) -> Any:
        """A :class:`SupportInbox` rehydrated from the stored tickets."""
        from .support import SupportInbox, Ticket

        inbox = SupportInbox()
        inbox.tickets = [Ticket(**row) for row in self._rows("tickets")]
        return inbox

    def save_inbox(self, inbox: Any) -> None:
        self._set("tickets", [t.to_dict() for t in inbox.tickets])

    # ── shipments ──────────────────────────────────────────────────

    def record_shipment(self, shipment: dict[str, Any]) -> dict[str, Any]:
        """Store a shipment row (carrier + waybill + order ref)."""
        row = dict(shipment)
        if not str(row.get("id") or "").strip():
            raise ValueError("shipment needs an id")
        if not str(row.get("waybill") or "").strip():
            raise ValueError("shipment needs a waybill")
        with self._lock:
            rows = [r for r in self._data["shipments"]
                    if str(r.get("id")) != str(row["id"])]
            rows.append(row)
            self._data["shipments"] = rows
            self._save()
            return dict(row)

    def list_shipments(self) -> list[dict[str, Any]]:
        return self._rows("shipments")

    def find_shipment(self, carrier: str, waybill: str) -> dict[str, Any] | None:
        for row in self._data["shipments"]:
            if (str(row.get("carrier", "")).lower()
                    == (carrier or "").strip().lower()
                    and str(row.get("waybill", "")).strip()
                    == (waybill or "").strip()):
                return dict(row)
        return None

    # ── reconciliation journal ─────────────────────────────────────

    def reconciler(self, *, ledger: Any = None,
                   ledger_path: str | Path | None = None) -> Any:
        """A :class:`Reconciler` with the stored journal restored.

        The reconciler's own ``journal`` / ``unreconciled`` / ``_seen_refs``
        are rebuilt from the stored rows, so a duplicate reference is
        still rejected after a restart and entries keep their ids.
        """
        from .reconciliation import JournalEntry, Reconciler

        rec = Reconciler(ledger=ledger, ledger_path=ledger_path)
        rec.journal = [JournalEntry(**row)
                       for row in self._rows("journal")]
        rec.unreconciled = self._rows("unreconciled")
        rec._seen_refs = {str(e.payment_reference) for e in rec.journal}
        return rec

    def save_reconciler(self, reconciler: Any) -> None:
        with self._lock:
            self._data["journal"] = [e.to_dict()
                                     for e in reconciler.journal]
            self._data["unreconciled"] = [dict(u)
                                          for u in reconciler.unreconciled]
            self._save()
