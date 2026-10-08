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

from .aid import (
    LANGUAGES,
    LANGUAGE_NAMES,
    Answer,
    answer_legal_question,
    bill_answer,
    control_legal,
    corpus_search,
    format_answer,
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
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "Answer",
    "answer_legal_question",
    "bill_answer",
    "control_legal",
    "corpus_search",
    "format_answer",
]
