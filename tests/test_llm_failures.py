"""Failure taxonomy: classification matrix + recovery policies. Offline."""
from __future__ import annotations

import pytest

from nomorals.llm.failures import (
    FailureClass,
    classify_failure,
    parse_retry_after,
    RECOVERY,
)


@pytest.mark.parametrize("text,expected", [
    ("ProviderError: 429 Too Many Requests", FailureClass.RATE_LIMITED),
    ("rate limit exceeded for model", FailureClass.RATE_LIMITED),
    ("RateLimitError: You exceeded your current quota", FailureClass.RATE_LIMITED),
    ("401 Unauthorized: invalid api key", FailureClass.AUTH),
    ("Invalid API key provided", FailureClass.AUTH),
    ("403 Forbidden", FailureClass.AUTH),
    ("authentication failed: bad credentials", FailureClass.AUTH),
    ("400: This model's maximum context length is 8192 tokens. Reduce the length",
     FailureClass.CONTEXT_OVERFLOW),
    ("context_length_exceeded: too many tokens", FailureClass.CONTEXT_OVERFLOW),
    ("input too long: prompt exceeds the model's maximum context window",
     FailureClass.CONTEXT_OVERFLOW),
    ("request timed out after 120s", FailureClass.TIMEOUT),
    ("TimeoutError: timed out", FailureClass.TIMEOUT),
    ("connection refused on localhost:8080", FailureClass.NETWORK),
    ("ConnectionResetError: connection reset by peer", FailureClass.NETWORK),
    ("NameResolutionError: temporary failure in name resolution", FailureClass.NETWORK),
    ("503 Service Unavailable", FailureClass.SERVER),
    ("500 internal server error", FailureClass.SERVER),
    ("overloaded: the model is overloaded", FailureClass.SERVER),
    ("404: model not found", FailureClass.CONFIG),
    ("the model 'foo/bar' is not hosted", FailureClass.CONFIG),
    ("NoSuchModel: no such model", FailureClass.CONFIG),
    ("weird new failure nobody has seen", FailureClass.UNKNOWN),
    ("", FailureClass.UNKNOWN),
    (None, FailureClass.UNKNOWN),
])
def test_classify_failure_matrix(text, expected):
    info = classify_failure(text)
    assert info.failure_class is expected, text
    assert info.policy.failure_class is expected
    assert info.hint  # every class has a human hint


def test_context_overflow_is_retryable_without_parking():
    policy = RECOVERY[FailureClass.CONTEXT_OVERFLOW]
    assert policy.retryable is True
    assert policy.cooldown_s == 0.0  # never park a healthy provider
    assert policy.reduce_context is True
    assert policy.failover is True  # a bigger-window provider may serve it


def test_auth_is_not_retryable_but_parks_long():
    policy = RECOVERY[FailureClass.AUTH]
    assert policy.retryable is False
    assert policy.cooldown_s >= 300.0


def test_rate_limited_parks_longer_than_transient():
    assert RECOVERY[FailureClass.RATE_LIMITED].cooldown_s >= \
        RECOVERY[FailureClass.TIMEOUT].cooldown_s


def test_retry_after_parsed():
    assert parse_retry_after("429 slow down; retry-after: 45") == 45.0
    assert parse_retry_after("RateLimited retry_after=120") == 120.0
    assert parse_retry_after("no hint here") is None
    assert parse_retry_after("") is None


def test_info_carries_retry_after():
    info = classify_failure("429 too many requests; retry-after: 30")
    assert info.retry_after_s == 30.0
    assert info.to_dict()["class"] == "rate_limited"


def test_exception_types_classify():
    from nomorals.core.errors import ContextOverflow, RateLimited
    assert classify_failure("boom", RateLimited("x")).failure_class is FailureClass.RATE_LIMITED
    assert classify_failure("boom", ContextOverflow("x")).failure_class is FailureClass.CONTEXT_OVERFLOW


def test_never_raises():
    classify_failure(None, None)
    classify_failure("", ValueError("x"))
    parse_retry_after(None)
