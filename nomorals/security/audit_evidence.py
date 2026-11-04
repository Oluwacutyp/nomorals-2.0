"""Evidence packs: what makes an audit finding *proven*.

A probe that comes back with only a count proves little.  Every
finding in this class carries a full evidence pack, captured at
audit time:

- the exact request sent (method, URL, headers) — reproducible
  by anyone;
- the response status + headers;
- the FULL response schema — every field name and its type;
- the source's own record count, where the backend states one
  (its pagination headers / totals — the source testifying
  against itself);
- a bounded sample of REAL records.

Masking law: values in a stored pack are masked at rest —
always, for every target.  Identifier-shaped values keep their
head and tail (a passport number shows as a real number's
head/tail, a date of birth as a real date's shape) — enough to
prove the record is genuine, never enough to use it.  The
*delivered* pack (the one handed to the owner in the report) is
unmasked only when the target is in the owner's own-asset
registry — his systems, his data, his call.  What is persisted
(the audit journal) is the masked form regardless: the stored
third-party sample is evidence of shape, not a copy of the
database.
"""

from __future__ import annotations

import re
import time
from typing import Any

__all__ = [
    "DEFAULT_SAMPLE_CAP",
    "OWN_ASSET_SAMPLE_CAP",
    "mask_value",
    "mask_record",
    "schema_of",
    "field_sensitivity",
    "build_pack",
    "pack_for_storage",
    "SENSITIVE_FIELD_TIERS",
]

#: Third-party targets: the sample is proof, not extraction —
#: this many rows is conclusive and no more are fetched into the
#: pack.  Enforced in the fetch layer of the scanner: a
#: third-party collection read never lands more rows than this
#: in a pack.
DEFAULT_SAMPLE_CAP = 5

#: The owner's own assets: his declaration, his data — the pack
#: may carry a working sample three orders larger.  (Transport
#: byte caps in the fetcher still apply: they are sanity, not
#: policy.)
OWN_ASSET_SAMPLE_CAP = 1000


# ── field sensitivity ────────────────────────────────────────
#: Field-name → sensitivity tier.  Tier 3 = identity/financial
#: core (the Lovable headlines: passport numbers, payment
#: details).  Tier 2 = personal data.  Tier 1 = contact-adjacent.
SENSITIVE_FIELD_TIERS: tuple[tuple[int, str], ...] = (
    (3, r"passport|ssn|social_security|card_?number|credit_?card|"
        r"cvv|cvc|iban|bank_?account|payment|password|passwd|"
        r"secret|private_?key"),
    (2, r"date_?of_?birth|\bdob\b|birth|address|phone|mobile|"
        r"salary|income|health|medical|diagnosis|location|"
        r"latitude|longitude|gps"),
    (1, r"email|e-?mail|full_?name|first_?name|last_?name|"
        r"username|user_?name|profile"),
)

_TIER_RX = tuple((tier, re.compile(rx, re.IGNORECASE))
                 for tier, rx in SENSITIVE_FIELD_TIERS)


def field_sensitivity(schema: dict[str, str]) -> tuple[int, list[str]]:
    """The highest sensitivity tier among a schema's fields,
    and the field names that reached it.  (0, []) for a schema
    with no personal-data shape at all."""
    best = 0
    matched: list[str] = []
    for name in schema:
        for tier, rx in _TIER_RX:
            if rx.search(str(name)):
                if tier > best:
                    best, matched = tier, [str(name)]
                elif tier == best:
                    matched.append(str(name))
                break
    return best, matched


# ── masking ──────────────────────────────────────────────────
_DATE_RX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def mask_value(value: Any) -> Any:
    """Mask one sample value for storage/display.  Strings keep
    a head and a tail sized to prove shape without carrying the
    value; dates keep their year and day (the shape of a real
    DOB) with the month masked; short strings mask fully;
    numbers/bools/None pass (a count or a flag is not an
    identity)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    m = _DATE_RX.match(text)
    if m:
        return f"{m.group(1)}-••-{m.group(3)}"
    if len(text) <= 4:
        return "•" * len(text)
    if len(text) <= 8:
        return f"{text[:1]}•••{text[-1:]}"
    head = text[:4]
    tail = text[-2:]
    return f"{head}••••{tail}"


def mask_record(record: dict[str, Any]) -> dict[str, Any]:
    """One sample row, masked.  Nested dicts/lists mask
    recursively; keys are never masked (the schema is the
    evidence's skeleton — field names are not personal data)."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            out[key] = mask_record(value)
        elif isinstance(value, list):
            out[key] = [mask_record(v) if isinstance(v, dict)
                        else mask_value(v) for v in value[:5]]
        else:
            out[key] = mask_value(value)
    return out


# ── schema ───────────────────────────────────────────────────
def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "str"


def schema_of(records: list[dict[str, Any]]) -> dict[str, str]:
    """Field name → type, unioned across the sample rows.  A
    field seen with two types reads ``int|str`` — the schema is
    what the source actually serves, warts included."""
    schema: dict[str, set[str]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            schema.setdefault(str(key), set()).add(_type_name(value))
    return {k: "|".join(sorted(v)) for k, v in sorted(schema.items())}


# ── the pack ─────────────────────────────────────────────────
def build_pack(*, check: str, target: str,
               request: dict[str, Any],
               response: dict[str, Any],
               schema: dict[str, str] | None = None,
               total_count: int | None = None,
               sample: list[dict[str, Any]] | None = None,
               own_asset: bool = False,
               notes: str = "") -> dict[str, Any]:
    """Assemble one evidence pack (the delivered form).  The
    sample rides raw here; :func:`pack_for_storage` produces
    the at-rest twin.  Callers must never persist this dict
    directly for a target with a sample."""
    rows = [r for r in (sample or []) if isinstance(r, dict)]
    return {
        "check": check,
        "target": target,
        "captured_at": time.time(),
        "request": {
            "method": str(request.get("method") or "GET"),
            "url": str(request.get("url") or target),
            "headers": dict(request.get("headers") or {}),
        },
        "response": {
            "status": response.get("status"),
            "headers": dict(response.get("headers") or {}),
            "content_type": str(response.get("content_type") or ""),
        },
        "schema": dict(schema or {}),
        "total_count": total_count,
        "sample": rows,
        "sample_masked": [mask_record(r) for r in rows],
        "own_asset": bool(own_asset),
        "notes": str(notes or ""),
    }


def pack_for_storage(pack: dict[str, Any]) -> dict[str, Any]:
    """The at-rest twin of a pack: the raw sample is replaced by
    its masked form.  This is the ONLY form the audit journal
    stores — for own assets too (the delivered report is the
    owner's copy; the journal is evidence of shape)."""
    stored = dict(pack)
    stored["sample"] = [mask_record(r) for r in pack.get("sample") or []]
    stored["sample_at_rest"] = "masked"
    return stored
