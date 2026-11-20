"""Response fingerprinting and differential comparison.

A detector that compares against a string is a detector that breaks on
the next redesign. This measures a baseline instead: status, length band,
word count, header set, reflection offsets, timing. A probe is
INTERESTING only when it departs from the baseline on a weighted score.

Two entry points matter:

* `fingerprint()` — the measurement.
* `compare()` — weighted delta with the axes named, so a finding can say
  *why* rather than "different".

Reflection search is offset-aware: a payload echoed at the same offset as
the baseline's placeholder is weak evidence, at a new offset is strong.
"""
from __future__ import annotations

import hashlib
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

__all__ = ["Fingerprint", "Delta", "fingerprint", "compare", "similarity",
           "login_delta"]

_TOKEN = re.compile(r"[A-Za-z0-9_\-]{4,}")
_SQL_ERROR = re.compile(
    r"SQL syntax|mysql_fetch|ORA-\d{5}|PostgreSQL.{0,20}ERROR|"
    r"SQLite.{0,20}error|Unclosed quotation|ODBC.{0,20}Driver", re.I)
_STACK = re.compile(r"Traceback \(most recent call last\)|"
                    r"at [\w.$]+ \([\w./]+:\d+:\d+\)|"
                    r"#\d+ /\S+\.php\(\d+\)", re.I)


@dataclass
class Fingerprint:
    status: int = 0
    length: int = 0
    words: int = 0
    headers: frozenset[str] = frozenset()
    location: str = ""
    cookies: frozenset[str] = frozenset()
    elapsed_ms: float = 0.0
    digest: str = ""
    body: str = field(default="", repr=False)
    reflected: list[int] = field(default_factory=list)
    sql_error: bool = False
    stack: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "length": self.length,
                "words": self.words, "elapsed_ms": round(self.elapsed_ms),
                "digest": self.digest, "reflected": self.reflected[:8],
                "sql_error": self.sql_error, "stack": self.stack,
                "headers": sorted(self.headers)[:20],
                "location": self.location,
                "cookies": sorted(self.cookies)[:20]}


def fingerprint(status: int, body: str, headers: Any = None,
                elapsed_ms: float = 0.0,
                markers: Iterable[str] = (),
                location: str = "", cookies: Iterable[str] = (),
                case_insensitive_markers: bool = False,
                length: int | None = None) -> Fingerprint:
    """Measure a response. `markers` are searched for reflection
    offsets (`reflected` holds the first offset of each marker found);
    pass `case_insensitive_markers` when the marker is a fail string
    the server may echo in any case (the login verdict needs that).
    `location` / `cookies` are the redirect target and the cookie
    names — the login verdict's axes. `length` overrides the measured
    length (the attacker counts raw bytes, not decoded characters)."""
    text = body or ""
    if case_insensitive_markers:
        lowered = text.lower()
        found = [lowered.find(m.lower()) for m in markers
                 if m and m.lower() in lowered]
    else:
        found = [text.find(m) for m in markers if m and m in text]
    return Fingerprint(
        status=status, length=len(text) if length is None else length,
        words=len(_TOKEN.findall(text)),
        headers=frozenset(k.lower() for k in
                          (headers.keys() if hasattr(headers, "keys") else [])),
        location=location,
        cookies=frozenset(m.lower() for m in cookies),
        elapsed_ms=elapsed_ms,
        digest=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        body=text, reflected=found,
        sql_error=bool(_SQL_ERROR.search(text)),
        stack=bool(_STACK.search(text)),
    )


@dataclass
class Delta:
    score: int = 0
    axes: dict[str, int] = field(default_factory=dict)
    verdict: str = "same"

    @property
    def interesting(self) -> bool:
        return self.score >= 3

    def as_dict(self) -> dict[str, Any]:
        return {"score": self.score, "axes": self.axes, "verdict": self.verdict}


def similarity(a: Fingerprint, b: Fingerprint) -> float:
    """0..1 token-set overlap, cheap and good enough to spot a soft 404."""
    sa = set(_TOKEN.findall(a.body))
    sb = set(_TOKEN.findall(b.body))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def compare(base: Fingerprint, probe: Fingerprint,
            baseline_times: Optional[list[float]] = None) -> Delta:
    """Weighted departure from a measured baseline."""
    axes: dict[str, int] = {}
    if probe.status != base.status:
        axes["status"] = 3
    if base.length and abs(probe.length - base.length) > max(64,
                                                             base.length * 0.20):
        axes["length"] = 2
    if probe.digest != base.digest:
        axes["digest"] = 1
    if probe.headers - base.headers:
        axes["headers"] = 1
    if probe.reflected and len(probe.reflected) > len(base.reflected):
        axes["reflection"] = 3
    if probe.sql_error and not base.sql_error:
        axes["sql_error"] = 5
    if probe.stack and not base.stack:
        axes["stack"] = 4
    if baseline_times and len(baseline_times) >= 3:
        mean = statistics.mean(baseline_times)
        sd = statistics.pstdev(baseline_times) or 1.0
        if probe.elapsed_ms > mean + (4 * sd) and probe.elapsed_ms > mean * 1.8:
            axes["timing"] = 3
    if similarity(base, probe) > 0.95 and "digest" in axes:
        axes.pop("digest")           # same page, noise only
    score = sum(axes.values())
    verdict = ("strong" if score >= 6 else "interesting" if score >= 3
               else "same")
    return Delta(score=score, axes=axes, verdict=verdict)


def login_delta(base: Fingerprint, probe: Fingerprint) -> int:
    """The login verdict on the unified Fingerprint: how far an
    attempt's response departs from the known-failure shape. Two axes
    (score >= 2) make the attempt a candidate. This is the attacker's
    baseline-differential rule, moved here so there is ONE fingerprint
    primitive and ONE verdict for the same question.

    The axes are the login question, not the probe question, so the
    calibration differs from compare() on purpose: the length band is
    48 bytes / 15% (a login page's failure body is short and stable),
    and the fail-string axis reads `reflected` — the attempt's
    fingerprint is built with markers=[fail_string], so `reflected`
    is non-empty exactly when the fail string is present. A changed
    redirect target or a new cookie is one axis each.
    """
    score = 0
    if probe.status != base.status:
        score += 1
    if abs(probe.length - base.length) > max(48, base.length * 0.15):
        score += 1
    if probe.location != base.location:
        score += 1
    if probe.cookies - base.cookies:
        score += 1
    if base.reflected and not probe.reflected:
        score += 1
    return score
