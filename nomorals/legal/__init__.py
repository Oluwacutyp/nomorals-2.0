"""Legal information surface — never legal advice."""

from .contracts import (
    DISCLAIMER,
    CONTRACT_TYPES,
    Finding,
    Review,
    control_contract,
    detect_contract_type,
    extract_clauses,
    format_review,
    information_only_check,
    review_contract,
)

__all__ = [
    "DISCLAIMER",
    "CONTRACT_TYPES",
    "Finding",
    "Review",
    "control_contract",
    "detect_contract_type",
    "extract_clauses",
    "format_review",
    "information_only_check",
    "review_contract",
]
