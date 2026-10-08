"""Property intelligence: rental scam detection, passports, valuations."""

from nomorals.property.scam import (
    AREA_NORMS,
    ILLEGAL_FEE_TERMS,
    ScamFlag,
    ScamReport,
    ScamStore,
    check_listing,
    control_scamcheck,
)

__all__ = [
    "AREA_NORMS",
    "ILLEGAL_FEE_TERMS",
    "ScamFlag",
    "ScamReport",
    "ScamStore",
    "check_listing",
    "control_scamcheck",
]
