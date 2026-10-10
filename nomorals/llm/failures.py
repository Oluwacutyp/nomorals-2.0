"""Typed provider-failure taxonomy and per-class recovery policy.

The router used to treat every provider failure the same way: substring
checks for "429" and exponential cooldown for everything else.  That is
wrong in three concrete ways:

* a **context overflow** (the *prompt* was too big) parked a healthy
  provider for 30s+ and burned the whole failover chain, when the right
  recovery is to shrink the context and retry — the provider is fine;
* an **auth/config** failure (bad key, retired model name) is not
  transient — re-probing every 30s just burns quota and hides the real
  message ("your HF key is invalid");
* a **rate limit** wants a longer cooldown and an honest "backing off"
  note, not the same treatment as a 500.

:class:`classify_failure` turns an error string (and, when available, the
exception) into a :class:`FailureInfo` carrying the failure class, whether
it is retryable, and the :class:`RecoveryPolicy` the router and the brain
apply: which cooldown, whether to fail over, whether to shrink the
context before retrying, and the plain-language hint the owner sees.

The router consults this in ``_note_failure``; the brain consults it in
``explain_failure`` and in its context-overflow retry; the cognition
trajectory store records the class so failure clusters group by *kind*
of breakage instead of by raw message text.
"""

from __future__ import annotations

import enum
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FailureClass",
    "FailureInfo",
    "RecoveryPolicy",
    "RECOVERY",
    "RetryBudget",
    "backoff_delay",
    "classify_failure",
    "is_retryable",
    "parse_retry_after",
    "retry_after_s",
    "should_failover",
]


class FailureClass(str, enum.Enum):
    """What kind of provider failure this is.  Drives recovery."""

    #: 429 / quota exhausted — back off, the quota window refills.
    RATE_LIMITED = "rate_limited"
    #: 401/403 — the key is wrong or lacks permission.  Not transient:
    #: re-probing is pointless until the operator fixes the credential.
    AUTH = "auth"
    #: The prompt exceeded the model's context window.  The provider is
    #: healthy; the *caller* must shrink the context and retry.  Never
    #: parks the provider.
    CONTEXT_OVERFLOW = "context_overflow"
    #: The call timed out — the provider may be overloaded.
    TIMEOUT = "timeout"
    #: Connection refused / reset / DNS — the endpoint is unreachable.
    NETWORK = "network"
    #: 5xx — the provider's side is broken.
    SERVER = "server"
    #: 404 / not hosted / bad model name — a config error, not transient.
    CONFIG = "config"
    #: The router's daily spend cap was reached — stop spending, not a
    #: provider problem at all.
    BUDGET = "budget"
    #: Anything else.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryPolicy:
    """What to do about one failure class.  A strategy, not flags."""

    failure_class: FailureClass
    #: May this be retried (possibly after recovery)?
    retryable: bool
    #: Continue down the failover chain?
    failover: bool
    #: Router cooldown for the failed provider, seconds.  0 means "do not
    #: park the provider" — used for caller-side failures.
    cooldown_s: float
    #: Retry with a shrunken context (truncate/summarize) before failing over?
    reduce_context: bool
    #: Plain-language fix, shown to the operator.
    hint: str


#: The recovery strategy per failure class.
RECOVERY: dict[FailureClass, RecoveryPolicy] = {
    FailureClass.RATE_LIMITED: RecoveryPolicy(
        FailureClass.RATE_LIMITED, retryable=True, failover=True,
        cooldown_s=120.0, reduce_context=False,
        hint="rate/quota limit — the router backs off automatically; check plan quota"),
    FailureClass.AUTH: RecoveryPolicy(
        FailureClass.AUTH, retryable=False, failover=True,
        cooldown_s=600.0, reduce_context=False,
        hint="the provider rejected the credentials — check the API key is set and valid"),
    FailureClass.CONTEXT_OVERFLOW: RecoveryPolicy(
        FailureClass.CONTEXT_OVERFLOW, retryable=True, failover=True,
        cooldown_s=0.0, reduce_context=True,
        hint="the prompt exceeded the model's context window — the brain shrinks the context and retries"),
    FailureClass.TIMEOUT: RecoveryPolicy(
        FailureClass.TIMEOUT, retryable=True, failover=True,
        cooldown_s=30.0, reduce_context=False,
        hint="the provider timed out — it may be overloaded; the next call fails over"),
    FailureClass.NETWORK: RecoveryPolicy(
        FailureClass.NETWORK, retryable=True, failover=True,
        cooldown_s=60.0, reduce_context=False,
        hint="the provider endpoint is unreachable — check the network or whether a local server is running"),
    FailureClass.SERVER: RecoveryPolicy(
        FailureClass.SERVER, retryable=True, failover=True,
        cooldown_s=60.0, reduce_context=False,
        hint="the provider errored on its side — usually transient; failover covers it"),
    FailureClass.CONFIG: RecoveryPolicy(
        FailureClass.CONFIG, retryable=False, failover=True,
        cooldown_s=600.0, reduce_context=False,
        hint="the model name or endpoint is wrong — check the configured model id"),
    FailureClass.BUDGET: RecoveryPolicy(
        FailureClass.BUDGET, retryable=False, failover=False,
        cooldown_s=0.0, reduce_context=False,
        hint="daily LLM spend cap reached — raise the budget or wait for the next UTC day"),
    FailureClass.UNKNOWN: RecoveryPolicy(
        FailureClass.UNKNOWN, retryable=True, failover=True,
        cooldown_s=30.0, reduce_context=False,
        hint="unclassified failure — the router fails over automatically"),
}


# ── patterns (ordered: first hit wins) ───────────────────────────────────────
# Each entry: (FailureClass, compiled regex).  Checked in order, so the
# specific classes (auth, context) precede the generic ones (server).

_PATTERNS: tuple[tuple[FailureClass, "re.Pattern[str]"], ...] = (
    (FailureClass.AUTH, re.compile(
        r"\b401\b|unauthorized|invalid[_\s-]?api[_\s-]?key|incorrect[_\s-]?api[_\s-]?key|"
        r"authentication[_\s-]?fail|bad[_\s-]?credentials|"
        r"\b403\b|forbidden|permission[_\s-]?denied|access[_\s-]?denied", re.I)),
    (FailureClass.RATE_LIMITED, re.compile(
        r"\b429\b|rate[_\s-]?limit|ratelimit|too[_\s-]?many[_\s-]?requests|"
        r"quota[_\s-]?exceeded|quota|tokens[_\s-]?per[_\s-]?minute|tpm[_\s-]?limit", re.I)),
    (FailureClass.CONTEXT_OVERFLOW, re.compile(
        r"context[_\s-]?length|context[_\s-]?window|maximum[_\s-]?context|"
        r"too[_\s-]?many[_\s-]?tokens|token[_\s-]?limit|"
        r"reduce[_\s-]?the[_\s-]?length|input[_\s-]?too[_\s-]?long|"
        r"prompt[_\s-]?too[_\s-]?long|exceeds[_\s-]?the[_\s-]?maximum|"
        r"model[’']?s[_\s-]?maximum[_\s-]?context", re.I)),
    (FailureClass.CONFIG, re.compile(
        r"\b404\b|not[_\s-]?found|not[_\s-]?hosted|no[_\s-]?such[_\s-]?model|"
        r"model[_\s-]?not[_\s-]?exist|does[_\s-]?not[_\s-]?exist|"
        r"invalid[_\s-]?model|unknown[_\s-]?model", re.I)),
    (FailureClass.TIMEOUT, re.compile(
        r"timed[_\s-]?out|timeout|deadline[_\s-]?exceeded|"
        r"read[_\s-]?timed[_\s-]?out|connect[_\s-]?timeout", re.I)),
    (FailureClass.NETWORK, re.compile(
        r"connection[_\s-]?refused|connection[_\s-]?reset|econnrefused|econnreset|"
        r"network[_\s-]?unreachable|name[_\s-]?resolution|dns|"
        r"nodename[_\s-]?nor[_\s-]?servname|temporary[_\s-]?failure[_\s-]?in[_\s-]?name|"
        r"connection[_\s-]?aborted|broken[_\s-]?pipe|no[_\s-]?route[_\s-]?to[_\s-]?host", re.I)),
    (FailureClass.SERVER, re.compile(
        r"\b50[0234]\b|internal[_\s-]?server[_\s-]?error|bad[_\s-]?gateway|"
        r"service[_\s-]?unavailable|gateway[_\s-]?timeout|overloaded", re.I)),
)

#: Provider error types that carry a semantic class regardless of message.
_EXCEPTION_CLASSES: tuple[tuple[type, FailureClass], ...] = ()


def _register_exception_classes() -> None:
    """Bind framework error types to failure classes (lazy: core.errors is
    cheap, but this module must stay importable even before it loads)."""
    global _EXCEPTION_CLASSES
    if _EXCEPTION_CLASSES:
        return
    try:
        from ..core.errors import ContextOverflow, ProviderUnavailable, RateLimited
    except Exception:  # noqa: BLE001 — taxonomy still works on text alone
        return
    _EXCEPTION_CLASSES = (
        (RateLimited, FailureClass.RATE_LIMITED),
        (ContextOverflow, FailureClass.CONTEXT_OVERFLOW),
        (ProviderUnavailable, FailureClass.NETWORK),
    )


_RE_RETRY_AFTER = re.compile(
    r"retry[_\s-]?after[:=\s]+(\d+(?:\.\d+)?)", re.I)


def parse_retry_after(text: str) -> float | None:
    """Seconds from a Retry-After hint in the error text, else None."""
    m = _RE_RETRY_AFTER.search(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(value, 3600.0))


@dataclass(frozen=True)
class FailureInfo:
    """The classification of one provider failure."""

    failure_class: FailureClass
    #: Original error text (trimmed).
    error: str
    retryable: bool
    retry_after_s: float | None
    policy: RecoveryPolicy = field(repr=False)

    @property
    def hint(self) -> str:
        return self.policy.hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.failure_class.value,
            "retryable": self.retryable,
            "retry_after_s": self.retry_after_s,
            "cooldown_s": self.policy.cooldown_s,
            "reduce_context": self.policy.reduce_context,
            "hint": self.policy.hint,
        }


def retry_after_s(error: str | None) -> float | None:
    """Seconds the server asked us to wait (Retry-After), else None.

    The server knows more than we do about when it will be ready — always
    prefer this over the computed backoff when present.
    """
    return parse_retry_after(error)


def backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
    jitter: str = "full",
) -> float:
    """Exponential backoff with jitter for retry attempt ``attempt`` (0-based).

    ``jitter="full"`` is the AWS-recommended default: ``random(0, min(base *
    2**attempt, cap))`` — without it, every client retries on the same beat
    and the thundering herd takes the recovering service back down.
    ``"equal"`` splits the difference (half fixed, half jittered);
    ``"none"`` is deterministic (tests, not production).
    """
    exp = min(base * (2.0 ** max(0, attempt)), cap)
    mode = (jitter or "full").lower()
    if mode == "none":
        return round(exp, 3)
    if mode == "equal":
        return round(exp / 2.0 + random.uniform(0, exp / 2.0), 3)
    return round(random.uniform(0, exp), 3)


def is_retryable(failure_class: FailureClass | str) -> bool:
    """May this failure class be retried (possibly after recovery)?

    Single source of truth: the ``RECOVERY`` table.  AUTH/CONFIG are not
    retryable — re-probing a dead credential is quota arson.
    """
    fc = failure_class if isinstance(failure_class, FailureClass) \
        else FailureClass(str(failure_class).lower())
    policy = RECOVERY.get(fc)
    return bool(policy.retryable) if policy is not None else True


def should_failover(failure_class: FailureClass | str) -> bool:
    """Continue down the failover chain for this failure class?

    Mirrors Portkey's ``on_status_codes`` insight: a terminal error (bad
    key, dead model id) fails the *call* fast instead of burning the whole
    chain.  CONTEXT_OVERFLOW still fails over (a bigger window downstream
    may serve it) but the caller's shrink-and-retry owns the real fix.
    """
    fc = failure_class if isinstance(failure_class, FailureClass) \
        else FailureClass(str(failure_class).lower())
    policy = RECOVERY.get(fc)
    return bool(policy.failover) if policy is not None else True


class RetryBudget:
    """Wall-clock budget for a retry loop (tenaz-style total_timeout).

    Retries are bounded by attempts AND by time — a provider that fails
    fast must not spin 20 attempts in 3 seconds and call it "resilient".
    """

    def __init__(self, deadline_s: float = 60.0) -> None:
        self.deadline_s = max(0.0, float(deadline_s))
        self.started = time.monotonic()
        self.attempts = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline_s - self.elapsed)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def next_delay(self, *, base: float = 1.0, cap: float = 60.0,
                   jitter: str = "full") -> float:
        """Backoff for the next attempt, clamped to the remaining budget."""
        self.attempts += 1
        delay = backoff_delay(self.attempts - 1, base=base, cap=cap,
                              jitter=jitter)
        return min(delay, self.remaining)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadline_s": self.deadline_s,
            "elapsed_s": round(self.elapsed, 3),
            "remaining_s": round(self.remaining, 3),
            "attempts": self.attempts,
            "exhausted": self.exhausted,
        }


def classify_failure(error: str | None, exc: BaseException | None = None) -> FailureInfo:
    """Classify one provider failure.  Never raises.

    ``exc`` (when given) is checked first: framework error types carry a
    semantic class regardless of their message text.  Otherwise the
    ordered pattern table decides on the message text.
    """
    _register_exception_classes()
    text = (error or "").strip()
    for exc_type, fc in _EXCEPTION_CLASSES:
        if exc is not None and isinstance(exc, exc_type):
            policy = RECOVERY[fc]
            return FailureInfo(
                failure_class=fc, error=text[:400], retryable=policy.retryable,
                retry_after_s=parse_retry_after(text), policy=policy)
    low_text = text
    for fc, pattern in _PATTERNS:
        if pattern.search(low_text):
            policy = RECOVERY[fc]
            return FailureInfo(
                failure_class=fc, error=text[:400], retryable=policy.retryable,
                retry_after_s=parse_retry_after(text), policy=policy)
    policy = RECOVERY[FailureClass.UNKNOWN]
    return FailureInfo(
        failure_class=FailureClass.UNKNOWN, error=text[:400],
        retryable=policy.retryable, retry_after_s=parse_retry_after(text),
        policy=policy)
