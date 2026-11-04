"""Secret-pattern battery shared by the security-audit organs.

One battery, two consumers: the exposure scanner mines page
source and JS bundles with it, the repo auditor mines working
trees and git history with it.  A pattern here is a *shape*, not
a verdict — every hit carries the pattern name and a redacted
form of the value, and the calling organ decides what the hit
means in context (a Stripe *test* key in a README is not a
leaked live key; a ``service_role`` JWT in a shipped bundle is).

Redaction law (the whole audit class): a secret's full value is
never written into a report, a journal, or a chat card.  Reports
carry :func:`redact_secret` output — head + tail, enough for the
owner of the secret to recognise it, useless to anyone else.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SecretHit",
    "SECRET_PATTERNS",
    "scan_text_for_secrets",
    "redact_secret",
    "jwt_payload",
    "shannon_entropy",
    "entropy_hits",
    "PLACEHOLDER_MARKERS",
]


@dataclass
class SecretHit:
    """One pattern hit: where, what shape, redacted value."""

    pattern: str
    value: str          # the raw matched secret (never persisted raw
                        # by callers — they store ``redacted``)
    offset: int = 0

    @property
    def redacted(self) -> str:
        return redact_secret(self.value)


def redact_secret(value: str) -> str:
    """Head/tail redaction: first 4 + last 4 for long values,
    fully masked for short ones.  The shape (provider prefix
    like ``sk_live_``) survives because the pattern match starts
    at the prefix."""
    v = str(value or "")
    if len(v) <= 8:
        return "•" * len(v)
    return f"{v[:4]}…{v[-4:]}"


#: Values that are documentation, not secrets.  A hit whose value
#: contains one of these markers (case-insensitive) is dropped by
#: the assignment-shaped patterns (the rigid provider patterns
#: keep their hits — an ``AKIA…`` string is a real key shape
#: wherever it sits, and the report can say "example in docs").
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "example", "changeme", "change-me", "your_", "your-", "xxxx",
    "placeholder", "dummy", "sample", "test_key", "insert_", "<",
    "${", "todo", "redacted", "•",
)


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(m in low for m in PLACEHOLDER_MARKERS)


#: (name, regex, group) — group is the capture group holding the
#: secret value (0 = whole match).  Provider patterns are rigid
#: (their alphabets/lengths are the provider's own); the generic
#: assignment pattern is the wide net behind them.
SECRET_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", 0),
    ("aws_secret_key",
     r"(?i)aws_secret_access_key\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})", 1),
    ("gcp_service_account",
     r'"type"\s*:\s*"service_account"', 0),
    ("private_key_block",
     r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----", 0),
    ("stripe_live_key", r"sk_live_[0-9a-zA-Z]{16,}", 0),
    ("stripe_restricted_key", r"rk_live_[0-9a-zA-Z]{16,}", 0),
    ("github_token", r"ghp_[A-Za-z0-9]{36,}", 0),
    ("github_fine_grained", r"github_pat_[A-Za-z0-9_]{40,}", 0),
    ("github_oauth", r"gho_[A-Za-z0-9]{36,}", 0),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}", 0),
    ("sendgrid_key", r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}", 0),
    ("mailgun_key", r"key-[0-9a-z]{32}", 0),
    ("firebase_api_key", r"AIza[0-9A-Za-z_-]{35}", 0),
    ("twilio_key", r"SK[0-9a-f]{32}", 0),
    ("jwt", r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}", 0),
    ("url_embedded_credentials",
     r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]{3,}@", 0),
    ("generic_secret_assignment",
     r"(?i)(?:api[_-]?key|apikey|client[_-]?secret|auth[_-]?token|"
     r"access[_-]?token|secret|passwd|password)\s*[:=]\s*"
     r"[\"']([A-Za-z0-9_\-./+=]{12,})[\"']", 1),
)

_COMPILED: tuple[tuple[str, "re.Pattern[str]", int], ...] = tuple(
    (name, re.compile(rx), group)
    for name, rx, group in SECRET_PATTERNS)


def scan_text_for_secrets(text: str, *, base_offset: int = 0) -> list[SecretHit]:
    """Run the battery over ``text``.  Rigid provider patterns
    report every hit; the generic assignment pattern drops
    placeholder values (docs examples are not findings)."""
    hits: list[SecretHit] = []
    if not text:
        return hits
    seen: set[tuple[str, int]] = set()
    for name, rx, group in _COMPILED:
        for m in rx.finditer(text):
            try:
                value = m.group(group) or ""
            except IndexError:
                value = m.group(0)
            if not value:
                continue
            if name == "generic_secret_assignment" \
                    and _is_placeholder(value):
                continue
            key = (name, base_offset + m.start())
            if key in seen:
                continue
            seen.add(key)
            hits.append(SecretHit(pattern=name, value=value,
                                  offset=base_offset + m.start()))
    return hits


def jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT's payload (no signature check — the audit
    reads claims, it never trusts them).  {} when undecodable.
    A Supabase ``service_role`` claim is the difference between a
    public-by-design anon key and a master key in the bundle."""
    try:
        parts = str(token).split(".")
        if len(parts) != 3:
            return {}
        raw = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - an undecodable token is
        return {}             # still a token hit, just claimless


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


_ASSIGNMENT_RX = re.compile(
    r"(?i)([A-Za-z0-9_]*(?:key|secret|token|passwd|password|cred)"
    r"[A-Za-z0-9_]*)\s*[:=]\s*[\"']([A-Za-z0-9_\-./+=]{20,})[\"']")

#: High-entropy threshold for an assigned string literal to count
#: as a probable secret the rigid patterns missed.
ENTROPY_THRESHOLD = 4.5


def entropy_hits(text: str, *, base_offset: int = 0) -> list[SecretHit]:
    """Assigned string literals whose Shannon entropy says
    'machine-generated secret' even though no provider pattern
    matched.  Restricted to key-shaped assignment names so prose
    and hashes in data files do not flood the report."""
    hits: list[SecretHit] = []
    for m in _ASSIGNMENT_RX.finditer(text or ""):
        value = m.group(2)
        if _is_placeholder(value):
            continue
        if shannon_entropy(value) >= ENTROPY_THRESHOLD:
            hits.append(SecretHit(
                pattern="high_entropy_assignment", value=value,
                offset=base_offset + m.start()))
    return hits
