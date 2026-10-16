"""Travel visa layer: visa requirements per (origin, destination, passport).

A visa rule is a descriptor, not a guess: for a given origin,
destination, and passport nationality the layer answers
``visa_needed()`` and ``documents_required()`` from an explicit
curated table plus caller-registered rules. Anything not in the
table is honestly ``unknown`` — the layer NEVER invents a visa rule,
a stay length, or a document list for a route it does not know.

The built-in table is deliberately small (well-established rules
only) and every entry carries a ``notes`` field telling the caller
to verify with the embassy/consulate before travelling, because
visa rules change. Callers with a fuller source should register
their own rules via :class:`VisaTable.register` — the table is data,
not a claim of completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "VisaRequirement",
    "VisaTable",
    "VISA_TABLE",
    "visa_needed",
    "documents_required",
    "lookup",
    "UNKNOWN_REASON",
]

UNKNOWN_REASON = (
    "no visa rule in the table for this route — check the destination "
    "embassy / consulate / official immigration site before travelling; "
    "this layer never guesses visa rules."
)

VERIFY_NOTE = "Verify with the destination embassy/consulate before travelling."


@dataclass(frozen=True)
class VisaRequirement:
    """One visa rule descriptor for (origin, destination, passport)."""

    origin: str = ""        # where the traveller departs from (ISO code)
    destination: str = ""   # where they travel to (ISO code)
    passport: str = ""      # passport nationality (ISO code)
    status: str = ""        # visa_free | visa_on_arrival | evisa |
                            # eta_required | esta_required | visa_required
    stay_days: int | None = None  # max stay under this rule, if known
    documents: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "destination": self.destination,
                "passport": self.passport, "status": self.status,
                "stay_days": self.stay_days,
                "documents": list(self.documents), "notes": self.notes}


def _req(origin: str, destination: str, passport: str, status: str,
         stay_days: int | None, documents: tuple[str, ...],
         notes: str) -> VisaRequirement:
    return VisaRequirement(origin=origin, destination=destination,
                           passport=passport, status=status,
                           stay_days=stay_days, documents=documents,
                           notes=notes)


#: Built-in curated rules. Small on purpose: every entry is a
#: well-established rule; anything else is ``unknown``.
#: Codes are ISO 3166-1 alpha-2 (NG, GH, US, GB, KE, …).
VISA_TABLE: dict[tuple[str, str, str], VisaRequirement] = {
    ("NG", "NG", "NG"): _req("NG", "NG", "NG", "visa_free", None, (),
                            "Domestic travel — no visa. " + VERIFY_NOTE),
    ("US", "US", "US"): _req("US", "US", "US", "visa_free", None, (),
                            "Domestic travel — no visa. " + VERIFY_NOTE),
    ("GB", "GB", "GB"): _req("GB", "GB", "GB", "visa_free", None, (),
                            "Domestic travel — no visa. " + VERIFY_NOTE),
    # ECOWAS: Nigerian passport holders travel visa-free within ECOWAS.
    ("NG", "GH", "NG"): _req("NG", "GH", "NG", "visa_free", 90,
                            ("valid passport", "return/onward ticket"),
                            "ECOWAS free movement. " + VERIFY_NOTE),
    ("NG", "BJ", "NG"): _req("NG", "BJ", "NG", "visa_free", 90,
                            ("valid passport", "return/onward ticket"),
                            "ECOWAS free movement. " + VERIFY_NOTE),
    ("NG", "TG", "NG"): _req("NG", "TG", "NG", "visa_free", 90,
                            ("valid passport", "return/onward ticket"),
                            "ECOWAS free movement. " + VERIFY_NOTE),
    ("NG", "SN", "NG"): _req("NG", "SN", "NG", "visa_free", 90,
                            ("valid passport", "return/onward ticket"),
                            "ECOWAS free movement. " + VERIFY_NOTE),
    # Nigerian passport -> UK / US: a visa is required before travel.
    ("NG", "GB", "NG"): _req("NG", "GB", "NG", "visa_required", None,
                            ("valid passport", "completed visa application",
                             "supporting documents per visa category",
                             "proof of funds / ties"),
                            "Apply before travelling; category rules vary. "
                            + VERIFY_NOTE),
    ("NG", "US", "NG"): _req("NG", "US", "NG", "visa_required", None,
                            ("valid passport", "completed DS-160",
                             "visa interview appointment",
                             "supporting documents per visa category"),
                            "Apply before travelling; category rules vary. "
                            + VERIFY_NOTE),
    # Kenya: visitors need a Kenya eTA before travel (replaced visa-on-arrival).
    ("NG", "KE", "NG"): _req("NG", "KE", "NG", "eta_required", 90,
                            ("valid passport (6+ months)",
                             "approved Kenya eTA",
                             "return/onward ticket"),
                            "Apply for the eTA before travelling. "
                            + VERIFY_NOTE),
    ("US", "KE", "US"): _req("US", "KE", "US", "eta_required", 90,
                            ("valid passport (6+ months)",
                             "approved Kenya eTA",
                             "return/onward ticket"),
                            "Apply for the eTA before travelling. "
                            + VERIFY_NOTE),
    # UK: US passport holders need an ETA before travel.
    ("US", "GB", "US"): _req("US", "GB", "US", "eta_required", 180,
                            ("valid passport", "approved UK ETA"),
                            "ETA required before travel (not a visa). "
                            + VERIFY_NOTE),
    # US: UK passport holders travel under the Visa Waiver Program (ESTA).
    ("GB", "US", "GB"): _req("GB", "US", "GB", "esta_required", 90,
                            ("valid e-passport", "approved ESTA",
                             "return/onward ticket"),
                            "ESTA required before travel (not a visa). "
                            + VERIFY_NOTE),
    # US: US passport holders travel visa-free to the UK's neighbour
    # routes commonly asked about — kept to the well-known pair below.
    ("US", "CA", "US"): _req("US", "CA", "US", "visa_free", 180,
                            ("valid passport",),
                            "US citizens are visa-exempt for Canada. "
                            + VERIFY_NOTE),
    ("CA", "US", "US"): _req("CA", "US", "US", "visa_free", None,
                            ("valid passport",),
                            "US citizens returning home — no visa. "
                            + VERIFY_NOTE),
}


def _key(origin: str, destination: str, passport: str) -> tuple[str, str, str]:
    return ((origin or "").strip().upper(),
            (destination or "").strip().upper(),
            (passport or "").strip().upper())


class VisaTable:
    """A visa rule table: the built-in curated rules plus caller-
    registered ones. Lookup is exact-key; there is no fuzzy fallback
    and no invented rule. Never raises."""

    def __init__(self,
                 rules: dict[tuple[str, str, str], VisaRequirement] | None = None) -> None:
        self._rules: dict[tuple[str, str, str], VisaRequirement] = dict(
            VISA_TABLE if rules is None else rules)

    def register(self, rule: VisaRequirement) -> bool:
        """Add/override one rule. Returns False for a malformed rule."""
        try:
            if not isinstance(rule, VisaRequirement):
                return False
            key = _key(rule.origin, rule.destination, rule.passport)
            if not all(key) or not rule.status:
                return False
            self._rules[key] = VisaRequirement(
                origin=key[0], destination=key[1], passport=key[2],
                status=rule.status.strip().lower(),
                stay_days=rule.stay_days,
                documents=tuple(rule.documents or []),
                notes=str(rule.notes or ""))
            return True
        except Exception:  # noqa: BLE001
            return False

    def lookup(self, origin: str, destination: str,
               passport: str) -> VisaRequirement | None:
        """Exact-key lookup. None when the route is not in the table —
        the caller must report that as unknown, never invent."""
        try:
            return self._rules.get(_key(origin, destination, passport))
        except Exception:  # noqa: BLE001
            return None

    def known_routes(self) -> list[tuple[str, str, str]]:
        return sorted(self._rules)


#: Shared default table instance (built-in rules only).
DEFAULT_TABLE = VisaTable()


def lookup(origin: str, destination: str, passport: str,
           *, table: VisaTable | None = None) -> dict[str, Any]:
    """Full descriptor for a route. ``known: False`` + ``status:
    "unknown"`` when the route is not in the table — never a guess."""
    try:
        rule = (table or DEFAULT_TABLE).lookup(origin, destination, passport)
        if rule is None:
            return {"known": False, "status": "unknown",
                    "visa_needed": None, "stay_days": None,
                    "documents": [], "notes": UNKNOWN_REASON,
                    "origin": (origin or "").strip().upper(),
                    "destination": (destination or "").strip().upper(),
                    "passport": (passport or "").strip().upper()}
        return {"known": True, "status": rule.status,
                "visa_needed": rule.status in (
                    "visa_required", "evisa", "eta_required",
                    "esta_required", "visa_on_arrival"),
                "stay_days": rule.stay_days,
                "documents": list(rule.documents),
                "notes": rule.notes,
                "origin": rule.origin, "destination": rule.destination,
                "passport": rule.passport}
    except Exception:  # noqa: BLE001
        _log.debug("visa lookup failed", exc_info=True)
        return {"known": False, "status": "unknown", "visa_needed": None,
                "stay_days": None, "documents": [], "notes": UNKNOWN_REASON,
                "origin": "", "destination": "", "passport": ""}


def visa_needed(origin: str, destination: str, passport: str,
                *, table: VisaTable | None = None) -> dict[str, Any]:
    """Does this route need a visa / ETA / ESTA / eVisa?

    Returns ``{"needed": bool | None, "status": ..., "known": ...}``.
    ``needed`` is None when the route is unknown — the honest answer,
    never a fabricated yes/no. Never raises.
    """
    result = lookup(origin, destination, passport, table=table)
    return {"needed": result["visa_needed"],
            "status": result["status"], "known": result["known"],
            "stay_days": result["stay_days"],
            "notes": result["notes"]}


def documents_required(origin: str, destination: str, passport: str,
                       *, table: VisaTable | None = None) -> dict[str, Any]:
    """Documents for a route. Unknown routes return an empty list with
    ``known: False`` — never an invented document list. Never raises."""
    result = lookup(origin, destination, passport, table=table)
    return {"documents": list(result["documents"]),
            "known": result["known"], "status": result["status"],
            "notes": result["notes"]}
