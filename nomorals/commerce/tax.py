"""Tax / VAT: Nigeria FIRS 7.5% VAT + client jurisdictions table.

Nigeria: Value Added Tax is 7.5% (FIRS, since the 2019 Finance Act
raised it from 5%). ``compute_tax(amount, "NG")`` applies exactly that.

The jurisdiction table is a *descriptor* per jurisdiction — name, rate,
tax name, and notes — for the places Devon's clients sell into. Rates
change; the table carries ``verify`` guidance and
:func:`jurisdictions` lets operators override a rate at runtime instead
of trusting a stale constant. ``compute_tax`` uses integer math only
(basis points), so there is no float drift on money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Jurisdiction",
    "JURISDICTIONS",
    "jurisdictions",
    "compute_tax",
    "NIGERIA_VAT_BPS",
]

#: Nigeria VAT in basis points: 7.5% = 750 bps.
NIGERIA_VAT_BPS = 750


@dataclass(frozen=True)
class Jurisdiction:
    """Descriptor for one tax jurisdiction."""
    code: str        # ISO country code
    name: str
    tax_name: str    # "VAT" | "GST" | "Sales tax" | ...
    rate_bps: int    # rate in basis points (750 = 7.5%)
    notes: str = ""


JURISDICTIONS: dict[str, Jurisdiction] = {
    "NG": Jurisdiction("NG", "Nigeria", "VAT", 750,
                       "FIRS VAT 7.5% (Finance Act 2019, up from 5%)."),
    "GH": Jurisdiction("GH", "Ghana", "VAT", 1500,
                       "Ghana VAT 15% (incl. levies in practice — "
                       "confirm with GRA for the client's exact mix)."),
    "KE": Jurisdiction("KE", "Kenya", "VAT", 1600,
                       "Kenya VAT 16%."),
    "ZA": Jurisdiction("ZA", "South Africa", "VAT", 1500,
                       "South Africa VAT 15%."),
    "GB": Jurisdiction("GB", "United Kingdom", "VAT", 2000,
                       "UK standard VAT 20%."),
    "US": Jurisdiction("US", "United States", "Sales tax", 0,
                       "No federal VAT — sales tax varies by state/county "
                       "(0–10%+). Rate 0 here means 'not applicable at "
                       "federal level'; pass an override for a real state."),
    "CA": Jurisdiction("CA", "Canada", "GST/HST", 500,
                       "Federal GST 5%; HST/PST extra by province — "
                       "override for the province."),
    "IN": Jurisdiction("IN", "India", "GST", 1800,
                       "GST slabs vary (0/5/12/18/28%); 18% is the common "
                       "standard slab — override per product."),
}


def jurisdictions(overrides: dict[str, int] | None = None) -> dict[str, Jurisdiction]:
    """Jurisdiction table, with optional per-code rate overrides (bps).

    Operators override a rate when the law changes instead of waiting
    for a code update: ``jurisdictions({"NG": 750})``.
    """
    table = dict(JURISDICTIONS)
    for code, bps in (overrides or {}).items():
        key = (code or "").strip().upper()
        if key in table and isinstance(bps, int) and bps >= 0:
            old = table[key]
            table[key] = Jurisdiction(old.code, old.name, old.tax_name,
                                       bps, old.notes + " (operator override)")
        elif key:
            table[key] = Jurisdiction(key, key, "VAT", bps,
                                       "Operator-added jurisdiction.")
    return table


def compute_tax(amount_minor: int, jurisdiction: str = "NG", *,
                overrides: dict[str, int] | None = None) -> dict[str, Any]:
    """Compute tax on ``amount_minor`` (net, minor units).

    Returns ``{jurisdiction, tax_name, rate_bps, tax_minor,
    total_minor, net_minor}``. Integer math: ``tax = amount * bps // 10000``.
    Raises ValueError on unknown jurisdiction / negative amount.
    """
    if amount_minor < 0:
        raise ValueError("amount_minor must be non-negative")
    code = (jurisdiction or "").strip().upper()
    table = jurisdictions(overrides)
    if code not in table:
        raise ValueError(
            f"unknown jurisdiction {jurisdiction!r} — known: "
            f"{sorted(table)} (or pass an override)")
    j = table[code]
    tax = (int(amount_minor) * j.rate_bps) // 10_000
    return {
        "jurisdiction": j.code,
        "jurisdiction_name": j.name,
        "tax_name": j.tax_name,
        "rate_bps": j.rate_bps,
        "rate_pct": j.rate_bps / 100,
        "net_minor": int(amount_minor),
        "tax_minor": tax,
        "total_minor": int(amount_minor) + tax,
        "notes": j.notes,
    }
