"""Registry of every autonomous action class Devon may take.

Every outward action Devon takes on the owner's behalf belongs to one
of :data:`ACTION_CLASSES`.  The registry is the single source of truth
for that vocabulary:

- :data:`ACTION_CLASSES` — the canonical extended class names.
- :data:`ACTION_CLASS_TO_LEDGER_TYPE` — how each class maps onto the
  five base types the :class:`RepresentationLedger
  <nomorals.cognition.representation.RepresentationLedger>` understands
  (``negotiate`` / ``purchase`` / ``apply`` / ``message`` / ``post``).
  The mapping is *narrowing only*: the extended class name is always
  preserved in the ledger record's metadata by
  :mod:`nomorals.cognition.ledger_ext`, so no information is lost when
  an extended class is stored under a base type.
- :func:`record_all` / :func:`record_all_classes` — helpers that route
  every recording through the ledger extension, so no action class can
  silently bypass the ledger.

Layer rule: this module imports only stdlib.  Recording helpers take
a ledger object and import the ledger extension lazily, so there is
no import cycle (``ledger_ext`` imports this registry at top level).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "ACTION_CLASSES",
    "ACTION_CLASS_TO_LEDGER_TYPE",
    "ACTION_ALIASES",
    "normalize_action_class",
    "ledger_type_for",
    "record_action",
    "record_all",
    "record_all_classes",
]

#: Every autonomous action class.  ``buy``/``purchase``, ``send``/
#: ``message`` and ``post``/``publish`` are kept as distinct classes
#: (call sites use whichever verb reads naturally); they map onto the
#: same ledger base type.
ACTION_CLASSES = (
    "negotiate",
    "buy",
    "purchase",
    "apply",
    "post",
    "publish",
    "send",
    "message",
    "schedule",
    "trade",
    "transfer",
    "book",
    "comment",
    "reply",
    "invite",
    "subscribe",
    "cancel",
    "refund",
    "claim",
    "bid",
    "offer",
    "hire",
    "share",
)

#: Extended class -> RepresentationLedger base type.  Covers every
#: entry of ACTION_CLASSES; :func:`ledger_type_for` raises on anything
#: outside the registry so a new class cannot silently fall through.
ACTION_CLASS_TO_LEDGER_TYPE = {
    "negotiate": "negotiate",
    # Money movement lands under "purchase".
    "buy": "purchase",
    "purchase": "purchase",
    "trade": "purchase",
    "transfer": "purchase",
    "book": "purchase",
    "bid": "purchase",
    "offer": "purchase",
    "refund": "purchase",
    "cancel": "purchase",
    # Applications / enrolment land under "apply".
    "apply": "apply",
    "claim": "apply",
    "subscribe": "apply",
    "hire": "apply",
    # Public output lands under "post".
    "post": "post",
    "publish": "post",
    "comment": "post",
    "reply": "post",
    "share": "post",
    # Directed communication (incl. scheduling one) lands under "message".
    "send": "message",
    "message": "message",
    "schedule": "message",
    "invite": "message",
}

#: Non-canonical spellings accepted by :func:`normalize_action_class`.
ACTION_ALIASES = {
    "dm": "message",
    "email": "message",
    "mail": "message",
    "text": "message",
    "notify": "message",
    "remind": "schedule",
    "calendar": "schedule",
    "purchase_order": "purchase",
    "checkout": "buy",
    "tweet": "post",
    "tweet_post": "post",
}


def normalize_action_class(action_class: str) -> str:
    """Normalize a caller-supplied class name to a canonical class.

    Lowercases, strips whitespace, resolves :data:`ACTION_ALIASES`.
    Raises :class:`ValueError` for anything outside the registry.
    """
    if not isinstance(action_class, str):
        raise ValueError(f"action_class must be str, got {type(action_class).__name__}")
    key = action_class.strip().lower().replace("-", "_").replace(" ", "_")
    key = ACTION_ALIASES.get(key, key)
    if key not in ACTION_CLASSES:
        raise ValueError(
            f"unknown action_class {action_class!r}; use one of {ACTION_CLASSES}")
    return key


def ledger_type_for(action_class: str) -> str:
    """Base ledger type for an (alias-tolerant) action class."""
    return ACTION_CLASS_TO_LEDGER_TYPE[normalize_action_class(action_class)]


def _record_one(ledger: Any, spec: Mapping[str, Any]) -> Any:
    """Record one action through the ledger extension (or a bare ledger).

    ``ledger`` may be an
    :class:`ExtendedRepresentationLedger
    <nomorals.cognition.ledger_ext.ExtendedRepresentationLedger>` (preferred
    — stakes / counterparty / prediction metadata are preserved) or a
    plain :class:`RepresentationLedger
    <nomorals.cognition.representation.RepresentationLedger>` (falls back
    to ``log_action`` under the mapped base type).  ``spec`` carries
    ``action_class`` plus any ``record_autonomous_action`` keyword
    arguments (``stakes``, ``counterparty``, ``outcome``,
    ``owner_feedback``, ``description``, ``metadata``, ...).
    """
    spec = dict(spec)
    action_class = normalize_action_class(spec.pop("action_class", ""))
    record_fn = getattr(ledger, "record_autonomous_action", None)
    if record_fn is not None:
        return record_fn(action_class, **spec)
    # Bare ledger fallback: map to the base type, keep the extended
    # class in metadata so the record stays attributable.
    from .representation import OUTCOMES

    outcome = spec.pop("outcome", "unknown") or "unknown"
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; use {OUTCOMES}")
    description = spec.pop("description", "") or f"{action_class} action"
    metadata = dict(spec.pop("metadata", None) or {})
    metadata.setdefault("action_class", action_class)
    for key in ("stakes", "counterparty", "predicted_confidence"):
        if key in spec:
            metadata.setdefault(key, spec.pop(key))
    owner_feedback = spec.pop("owner_feedback", None)
    rec = ledger.log_action(
        ledger_type_for(action_class), description,
        outcome=outcome, metadata=metadata)
    if owner_feedback is not None and getattr(rec, "id", ""):
        ledger.record_outcome(rec.id, outcome, owner_feedback=owner_feedback)
        refreshed = ledger.get(rec.id)
        if refreshed is not None:
            return refreshed
    return rec


def record_action(ledger: Any, action_class: str, **kwargs: Any) -> Any:
    """Record a single autonomous action through the ledger.

    Thin wrapper over :func:`_record_one` for the one-action call sites.
    """
    return _record_one(ledger, {"action_class": action_class, **kwargs})


def record_all(ledger: Any, items: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Record every action in ``items`` through the ledger.

    Each item is a mapping with at least ``action_class``.  Returns the
    ledger records in input order.  A bad ``action_class`` raises
    :class:`ValueError` before anything is recorded for that item;
    items are recorded in order so a caller can see the prefix that
    succeeded.
    """
    return [_record_one(ledger, item) for item in items]


def record_all_classes(
    ledger: Any,
    *,
    description_template: str = "Coverage probe: {action_class}",
    outcome: str = "unknown",
    **kwargs: Any,
) -> list[Any]:
    """Record one action for every class in :data:`ACTION_CLASSES`.

    Coverage helper: proves the ledger accepts every registered class
    (used by the cognition test-suite and by smoke checks after a
    registry change).  Returns one record per class, in
    :data:`ACTION_CLASSES` order.
    """
    return record_all(
        ledger,
        [
            {
                "action_class": cls,
                "description": description_template.format(action_class=cls),
                "outcome": outcome,
                **kwargs,
            }
            for cls in ACTION_CLASSES
        ],
    )
