"""Preference feedback — signals in, ranking out.

Weighting mirrors :mod:`nomorals.cognition.representation`: owner
approval is ``+1``, owner disapproval is ``-1`` weighted harder
(``DISAPPROVE_WEIGHT = 1.5``) — disapproval hits harder than approval
helps.

* :func:`record_signal` — one approve / disapprove on one item.
* :func:`suggest` — rank *supplied* candidates by learned weights.
  Never invents candidates: the output is a re-ordering (with scores)
  of exactly the input list.  Unknown domain → ``[]`` (honest empty,
  not a guessed ranking).
* :func:`learn_from_representation` — consume RepresentationLedger-
  style entries (dicts / objects with ``action`` / ``domain`` /
  ``owner_feedback``) and record them as signals.  Duck-typed: this
  module never imports the representation ledger.

Only the standard library is imported.
"""

from __future__ import annotations

from typing import Any

from .adapters import get_adapter
from .store import DOMAIN_DESCRIPTORS, PreferenceStore, extract_axis_values

__all__ = [
    "APPROVE_WEIGHT",
    "DISAPPROVE_WEIGHT",
    "record_signal",
    "suggest",
    "suggest_ranked",
    "learn_from_representation",
    "score_item",
]

#: Approve moves a weight by +1; disapprove by -1.5 (harder).
APPROVE_WEIGHT = 1.0
DISAPPROVE_WEIGHT = 1.5


def _feedback_to_bool(value: Any) -> bool | None:
    """owner_feedback (+1 / -1 / 'up' / 'down' / True / False) → bool / None."""
    if value in (1, "1", "+1", "up", "approve", "approved", True):
        return True
    if value in (-1, "-1", "down", "disapprove", "disapproved", False):
        return False
    return None


def _item_of(entry: Any) -> Any:
    """Best item dict for a representation-style entry, or {}."""
    if isinstance(entry, dict):
        for key in ("item", "metadata", "attributes"):
            value = entry.get(key)
            if isinstance(value, dict) and value:
                return value
        # A flat dict carrying axis keys itself (e.g. {"energy": "high"}).
        return entry
    for attr in ("item", "metadata"):
        value = getattr(entry, attr, None)
        if isinstance(value, dict) and value:
            return value
    return {}


def _domain_of(entry: Any) -> str | None:
    if isinstance(entry, dict):
        for key in ("domain", "action_type", "action"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("domain")
            if isinstance(value, str) and value:
                return value
        return None
    for attr in ("domain", "action_type", "action"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value:
            return value
    metadata = getattr(entry, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("domain")
        if isinstance(value, str) and value:
            return value
    return None


def _feedback_of(entry: Any) -> Any:
    if isinstance(entry, dict):
        for key in ("owner_feedback", "feedback", "approve", "approved"):
            if key in entry:
                return entry[key]
        return None
    for attr in ("owner_feedback", "feedback", "approve"):
        if hasattr(entry, attr):
            return getattr(entry, attr)
    return None


def record_signal(store: PreferenceStore, domain: str, item: Any,
                  approve: bool, source: str = "representation") -> dict[str, Any]:
    """Record one approve (``True``) / disapprove (``False``) signal.

    Updates the learned weight of every (axis, value) the item carries
    on the domain's descriptor: approve ``+1.0``, disapprove ``-1.5``.
    Unknown domain → no weights touched, ``{"recorded": False}``.
    Items carrying no descriptor axes → logged, no weights touched.
    Returns ``{"recorded", "domain", "approve", "delta", "updates"}``.
    """
    if domain not in DOMAIN_DESCRIPTORS:
        return {"recorded": False, "domain": domain, "approve": bool(approve),
                "delta": 0.0, "updates": {}, "reason": "unknown domain"}
    adapter = get_adapter(domain)
    values = (adapter.extract(item) if adapter is not None
              else extract_axis_values(item, domain))
    delta = APPROVE_WEIGHT if approve else -DISAPPROVE_WEIGHT
    updates: dict[str, dict[str, float]] = {}
    for axis, axis_values in values.items():
        for value in axis_values:
            new = store.update_weight(domain, axis, value, delta)
            updates.setdefault(axis, {})[value] = new
    store.log_signal(domain, item, bool(approve), delta, source)
    return {"recorded": True, "domain": domain, "approve": bool(approve),
            "delta": delta, "updates": updates}


def score_item(store: PreferenceStore, domain: str, item: Any) -> float:
    """Score one item by summing learned weights of its axis values.

    Unknown domain → 0.0.  Unseen values contribute 0.0 — a fresh
    store scores everything 0 and preserves the caller's order.
    """
    if domain not in DOMAIN_DESCRIPTORS:
        return 0.0
    adapter = get_adapter(domain)
    values = (adapter.extract(item) if adapter is not None
              else extract_axis_values(item, domain))
    weights = store.get_weights(domain)
    total = 0.0
    for axis, axis_values in values.items():
        bucket = weights.get(axis, {})
        for value in axis_values:
            total += float(bucket.get(value, 0.0))
    return total


def suggest(store: PreferenceStore, domain: str,
            candidates: list[Any] | None) -> list[dict[str, Any]]:
    """Rank supplied candidates by learned weights.

    Returns ``[{"item": candidate, "candidate": candidate, "score": float}]``
    in ranked order (highest score first, stable for ties so a fresh
    store preserves input order).  Never invents candidates: the
    output contains exactly the input items, re-ordered.  Unknown
    domain or no candidates → ``[]``.
    """
    if not candidates or domain not in DOMAIN_DESCRIPTORS:
        return []
    scored = [
        {"item": candidate, "candidate": candidate,
         "score": score_item(store, domain, candidate)}
        for candidate in candidates
    ]
    # Stable sort: ties keep the caller's order.
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored


# Alias with a more explicit name.
suggest_ranked = suggest


def learn_from_representation(store: PreferenceStore,
                              ledger_entries: list[Any] | Any | None) -> dict[str, Any]:
    """Consume RepresentationLedger-style entries as preference signals.

    Each entry needs a domain (``domain`` / ``action_type`` / ``action``,
    or ``metadata.domain``) and an ``owner_feedback`` of +1 (approve)
    or -1 (disapprove).  The item is taken from ``item`` / ``metadata``
    / the entry itself when it carries descriptor axes.  Entries with
    no feedback, an unknown domain, or no usable item are skipped and
    counted — never guessed.

    Accepts a list of dicts / objects, a single entry, or a ledger-like
    object exposing ``recent()`` / ``entries`` / ``actions``.  Returns
    ``{"recorded", "skipped", "details"}``.
    """
    if ledger_entries is None:
        return {"recorded": 0, "skipped": 0, "details": []}
    entries: list[Any] = []
    if isinstance(ledger_entries, list):
        entries = ledger_entries
    elif isinstance(ledger_entries, dict):
        entries = [ledger_entries]
    elif hasattr(ledger_entries, "recent") and callable(ledger_entries.recent):
        try:
            entries = list(ledger_entries.recent(limit=1000))
        except TypeError:
            entries = list(ledger_entries.recent())
    elif hasattr(ledger_entries, "entries"):
        entries = list(getattr(ledger_entries, "entries") or [])
    elif hasattr(ledger_entries, "actions"):
        entries = list(getattr(ledger_entries, "actions") or [])
    else:
        entries = [ledger_entries]

    recorded = 0
    skipped = 0
    details: list[dict[str, Any]] = []
    for entry in entries:
        domain = _domain_of(entry)
        approve = _feedback_to_bool(_feedback_of(entry))
        item = _item_of(entry)
        if domain is None or domain not in DOMAIN_DESCRIPTORS or approve is None:
            skipped += 1
            details.append({"recorded": False, "domain": domain,
                            "reason": "missing domain / feedback / unknown domain"})
            continue
        result = record_signal(store, domain, item, approve,
                               source="representation")
        if result.get("recorded") and result.get("updates"):
            recorded += 1
        else:
            # Logged but carried no descriptor axes — counted as
            # skipped for learning (nothing was learned from it).
            skipped += 1
        details.append({"recorded": bool(result.get("recorded") and result.get("updates")),
                        "domain": domain, "approve": approve,
                        "updates": result.get("updates", {})})
    return {"recorded": recorded, "skipped": skipped, "details": details}
