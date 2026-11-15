"""Vehicle / asset maintenance — service intervals, reminders, auditable costs.

MORE_TO_ADD gap 7. NEW module; it does not rebuild any existing organ
(``nomorals/finance/`` owns money movement/budgets; this module owns the
*asset* side: what was serviced, when, at what reading, for how much).

Universal: assets are kind-tagged (``vehicle`` / ``generator`` / ``ac`` /
``laptop`` / ``appliance`` / ``other``), the currency is a parameter
(Nigeria default ``₦`` — never a boundary), and every pack/country is
just metadata on the asset. No locale is special-cased.

Money: integer minor units (kobo/cents) everywhere, mirroring
``nomorals/finance/ledger.py`` — float drift is impossible by
construction. ``cost_minor`` is an ``int``; ``format_cost`` renders it
against the asset's currency symbol.

Auditability: every cost lives in :attr:`AssetLedger.records` as a
:class:`ServiceRecord` with asset id, date, odometer reading, vendor and
receipt ref. Nothing is ever deleted or overwritten — corrections are
new records (use ``adjustment`` in the description).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "Asset",
    "ServiceRecord",
    "Reminder",
    "AssetLedger",
    "ASSET_KINDS",
    "DEFAULT_CURRENCY",
]

DEFAULT_CURRENCY = "₦"

#: Extensible asset kinds. ``other`` is the escape hatch — new kinds never
#: require a code change, just a kind string.
ASSET_KINDS = ("vehicle", "generator", "ac", "laptop", "appliance", "other")


def _parse_date(value: Any) -> date | None:
    """ISO date (or ``None``) from a stored value; junk fails honestly."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"bad date {value!r}") from None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass
class Asset:
    """One maintainable asset.

    ``interval_km`` / ``interval_days`` are the service schedule: a service
    is due when EITHER threshold is crossed (whichever comes first).
    Either may be ``None`` (e.g. a laptop has no odometer → time-only).
    """

    name: str
    kind: str = "other"
    currency: str = DEFAULT_CURRENCY
    pack: str = "NG"
    purchase_date: date | None = None
    odometer_km: float = 0.0
    interval_km: float | None = None
    interval_days: int | None = None
    last_service_date: date | None = None
    last_service_km: float = 0.0
    notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if self.kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind {self.kind!r}; use one of {ASSET_KINDS} or 'other'")
        if self.interval_km is not None and self.interval_km <= 0:
            raise ValueError("interval_km must be positive")
        if self.interval_days is not None and self.interval_days <= 0:
            raise ValueError("interval_days must be positive")
        if self.odometer_km < 0:
            raise ValueError("odometer_km must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON shape (dates as ISO strings, ``None`` stays ``None``)."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "currency": self.currency,
            "pack": self.pack,
            "purchase_date": self.purchase_date.isoformat() if self.purchase_date else None,
            "odometer_km": self.odometer_km,
            "interval_km": self.interval_km,
            "interval_days": self.interval_days,
            "last_service_date": self.last_service_date.isoformat() if self.last_service_date else None,
            "last_service_km": self.last_service_km,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Asset:
        """Rebuild from :meth:`to_dict`; a missing/ill-typed name fails honestly."""
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"bad asset row: no name in {row!r}")
        return cls(
            id=str(row["id"]) if row.get("id") else uuid.uuid4().hex[:12],
            name=name,
            kind=str(row.get("kind", "other")),
            currency=str(row.get("currency", DEFAULT_CURRENCY)),
            pack=str(row.get("pack", "NG")),
            purchase_date=_parse_date(row.get("purchase_date")),
            odometer_km=float(row.get("odometer_km", 0.0)),
            interval_km=_as_float(row.get("interval_km")),
            interval_days=_as_int(row.get("interval_days")),
            last_service_date=_parse_date(row.get("last_service_date")),
            last_service_km=float(row.get("last_service_km", 0.0)),
            notes=str(row.get("notes", "")),
        )


@dataclass(frozen=True)
class ServiceRecord:
    """One auditable service event — the ledger entry for a cost."""

    asset_id: str
    service_date: date
    description: str
    cost_minor: int = 0
    odometer_km: float | None = None
    vendor: str = ""
    receipt_ref: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if self.cost_minor < 0:
            raise ValueError("cost_minor must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON shape (date as an ISO string)."""
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "service_date": self.service_date.isoformat(),
            "description": self.description,
            "cost_minor": self.cost_minor,
            "odometer_km": self.odometer_km,
            "vendor": self.vendor,
            "receipt_ref": self.receipt_ref,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ServiceRecord:
        """Rebuild from :meth:`to_dict`; a bad/missing date fails honestly."""
        service_date = _parse_date(row.get("service_date"))
        if service_date is None:
            raise ValueError(f"bad service record row: no service_date in {row!r}")
        return cls(
            id=str(row["id"]) if row.get("id") else uuid.uuid4().hex[:12],
            asset_id=str(row["asset_id"]),
            service_date=service_date,
            description=str(row.get("description", "")),
            cost_minor=int(row.get("cost_minor", 0)),
            odometer_km=_as_float(row.get("odometer_km")),
            vendor=str(row.get("vendor", "")),
            receipt_ref=str(row.get("receipt_ref", "")),
        )


@dataclass(frozen=True)
class Reminder:
    """A due / upcoming service reminder for one asset."""

    asset_id: str
    asset_name: str
    reason: str  # "km" | "time" | "km+time"
    overdue: bool
    detail: str
    due_km: float | None = None
    due_date: date | None = None


class AssetLedger:
    """Registry of assets + their full service/cost history.

    JSON-backed when a ``path`` is given (the /assets surface stores it
    at ``<home>/data/assets.json``); in-memory when it is not. The
    persistence shape mirrors :class:`nomorals.decisions.journal.DecisionJournal`:
    tolerant load (a missing/corrupt file starts empty, a bad row is
    skipped), atomic tmp+replace save after every mutation.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.assets: dict[str, Asset] = {}
        self.records: list[ServiceRecord] = []
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for row in raw.get("assets") or []:
            try:
                asset = Asset.from_dict(row)
                self.assets[asset.id] = asset
            except (TypeError, ValueError):
                continue
        for row in raw.get("records") or []:
            try:
                record = ServiceRecord.from_dict(row)
            except (TypeError, ValueError, KeyError):
                continue
            if record.asset_id not in self.assets:
                continue
            self.records.append(record)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"assets": [a.to_dict() for a in self.assets.values()],
                     "records": [r.to_dict() for r in self.records]},
                    fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            _log.debug("assets._save failed", exc_info=True)

    # -- assets ---------------------------------------------------------
    def add_asset(self, asset: Asset) -> Asset:
        if asset.id in self.assets:
            raise ValueError(f"asset {asset.id!r} already registered")
        self.assets[asset.id] = asset
        self._save()
        return asset

    def get_asset(self, asset_id: str) -> Asset:
        try:
            return self.assets[asset_id]
        except KeyError:
            raise ValueError(f"unknown asset {asset_id!r}") from None

    # -- service logging --------------------------------------------------
    def log_service(self, record: ServiceRecord) -> ServiceRecord:
        """Append a service record; updates the asset's last-service state.

        The record is appended, never mutated — the ledger is the audit
        trail. ``odometer_km`` on the record also advances the asset's
        odometer when it moves forward. Validation runs BEFORE the
        append, so a refused record leaves no trace (in memory or on
        disk).
        """
        asset = self.get_asset(record.asset_id)
        if (record.odometer_km is not None
                and record.odometer_km < asset.last_service_km):
            raise ValueError("odometer reading went backwards")
        self.records.append(record)
        asset.last_service_date = record.service_date
        if record.odometer_km is not None:
            asset.last_service_km = record.odometer_km
            asset.odometer_km = max(asset.odometer_km, record.odometer_km)
        self._save()
        return record

    # -- reminders --------------------------------------------------------
    def due_reminders(
        self,
        as_of: date,
        *,
        current_km: dict[str, float] | None = None,
        upcoming_days: int = 14,
    ) -> list[Reminder]:
        """Reminders for services due now or within ``upcoming_days``.

        Fires when the km threshold OR the time threshold is crossed
        (whichever first). ``current_km`` maps asset id → present reading;
        defaults to the asset's stored odometer. A reminder fires on
        *either* interval — never requires both.
        """
        current_km = current_km or {}
        out: list[Reminder] = []
        for asset in self.assets.values():
            km_now = current_km.get(asset.id, asset.odometer_km)
            km_due = (
                asset.interval_km is not None
                and (km_now - asset.last_service_km) >= asset.interval_km
            )
            km_soon = (
                asset.interval_km is not None
                and not km_due
                and (km_now - asset.last_service_km)
                >= asset.interval_km * 0.9
            )
            days_since: int | None = None
            if asset.last_service_date is not None:
                days_since = (as_of - asset.last_service_date).days
            time_due = (
                asset.interval_days is not None
                and days_since is not None
                and days_since >= asset.interval_days
            )
            time_soon = (
                asset.interval_days is not None
                and days_since is not None
                and not time_due
                and days_since >= asset.interval_days - upcoming_days
            )
            if km_due and time_due:
                reason, overdue = "km+time", True
            elif km_due:
                reason, overdue = "km", True
            elif time_due:
                reason, overdue = "time", True
            elif km_soon or time_soon:
                reason = "km" if km_soon and not time_soon else ("time" if time_soon and not km_soon else "km+time")
                overdue = False
            else:
                continue
            due_km = (
                asset.last_service_km + asset.interval_km
                if asset.interval_km is not None
                else None
            )
            out.append(
                Reminder(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    reason=reason,
                    overdue=overdue,
                    detail=(
                        f"{asset.name}: service {'OVERDUE' if overdue else 'due soon'} "
                        f"({reason}; {km_now:.0f} km, last service "
                        f"{asset.last_service_date})"
                    ),
                    due_km=due_km,
                    due_date=asset.last_service_date,
                )
            )
        return out

    # -- costs / audit ------------------------------------------------------
    def total_cost(self, asset_id: str, *, currency: str | None = None) -> int:
        """Total minor-unit cost for one asset (all records, auditable)."""
        self.get_asset(asset_id)
        return sum(r.cost_minor for r in self.records if r.asset_id == asset_id)

    def total_cost_all(self) -> dict[str, int]:
        """Total minor-unit cost per asset id."""
        totals: dict[str, int] = {}
        for r in self.records:
            totals[r.asset_id] = totals.get(r.asset_id, 0) + r.cost_minor
        return totals

    def export_ledger(self) -> list[dict[str, Any]]:
        """Full audit export: every cost record as plain dicts."""
        return [
            {
                "id": r.id,
                "asset_id": r.asset_id,
                "asset_name": self.assets[r.asset_id].name if r.asset_id in self.assets else "?",
                "date": r.service_date.isoformat(),
                "description": r.description,
                "cost_minor": r.cost_minor,
                "odometer_km": r.odometer_km,
                "vendor": r.vendor,
                "receipt_ref": r.receipt_ref,
            }
            for r in self.records
        ]

    @staticmethod
    def format_cost(cost_minor: int, currency: str = DEFAULT_CURRENCY) -> str:
        """Render integer minor units as ``₦12,500.00``-style text."""
        return f"{currency}{cost_minor / 100:,.2f}"
