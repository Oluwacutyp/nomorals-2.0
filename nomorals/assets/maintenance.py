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

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

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
    """Registry of assets + their full service/cost history."""

    def __init__(self) -> None:
        self.assets: dict[str, Asset] = {}
        self.records: list[ServiceRecord] = []

    # -- assets ---------------------------------------------------------
    def add_asset(self, asset: Asset) -> Asset:
        if asset.id in self.assets:
            raise ValueError(f"asset {asset.id!r} already registered")
        self.assets[asset.id] = asset
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
        odometer when it moves forward.
        """
        asset = self.get_asset(record.asset_id)
        self.records.append(record)
        asset.last_service_date = record.service_date
        if record.odometer_km is not None:
            if record.odometer_km < asset.last_service_km:
                raise ValueError("odometer reading went backwards")
            asset.last_service_km = record.odometer_km
            asset.odometer_km = max(asset.odometer_km, record.odometer_km)
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
