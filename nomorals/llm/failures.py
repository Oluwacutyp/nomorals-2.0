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
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FailureClass",
    "FailureInfo",
    "RecoveryPolicy",
    "RECOVERY",
    "classify_failure",
    "parse_retry_after",
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
