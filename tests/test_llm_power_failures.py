"""Tail-90 pins: nomorals/llm/power.py census sites.

The module's contract is explicit: ``power_state`` is "cheap and
never raises", computed over arbitrary duck-typed context objects.
Every site is a defensive guard inside that probe, held as an
exclusion:

42 — cache-read crash -> pass, the probe runs anyway. The cache
is a 60s performance garnish; a context whose attribute access
raises for ``_power_state_cache`` simply gets a fresh probe and
the same verdict the probe produces.
48 — cache-write crash -> pass, the probed state is still
returned, byte-identical to the cached path's value. A frozen /
slots context loses only the caching, never the answer.
59 — ``settings.offline`` read crash -> pass, the probe
continues to the router evidence. Tension, stated plainly: if
the owner's offline flag is set but unreadable (the settings
object's own attribute access raises), a healthy router can
still yield "full". Held: the trigger requires a settings
object whose attribute access itself raises — every other read
in the probe is equally suspect on such an object — and the
flag's normal path (readable True) wins over the router, as
the control pins.
71 — ``stats_snapshot()`` crash -> "degraded". Degraded is the
module's designed uncertainty value — the identical return for
"provider configured but unverified" and for >=3 consecutive
failures — meaning "attempt the model, expect fallback". It is
NOT the success value ("full"), and no consumer treats it as
health: model_usable() admits it precisely because fallback is
expected.
109 — ``invalidate`` cache-clear crash -> pass. Best-effort
invalidation; a context that refuses the write keeps its cached
state only until the 60s TTL expires — the staleness bound is
the documented cache design.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.agency import failures as agency_failures
from nomorals.llm import power as P


def _healthy_router():
    return SimpleNamespace(stats_snapshot=lambda: {
        "active": "hf_serverless", "chain": ["hf_serverless"],
        "health": {}})


def _not_recorded(before: int) -> None:
    assert agency_failures.failure_count() == before


# ── 42: cache read crash -> probe still answers ─────────────────

class _CacheReadHostile:
    def __getattr__(self, name):
        if name == "_power_state_cache":
            raise RuntimeError("cache read boom")
        raise AttributeError(name)


def test_42_cache_read_crash_falls_through_to_probe():
    before = agency_failures.failure_count()
    assert P.power_state(_CacheReadHostile()) == "offline"
    _not_recorded(before)


# ── 48: cache write crash -> state still returned ───────────────

class _FrozenCtx:
    settings = None
    router = None

    def __setattr__(self, name, value):
        if name == "_power_state_cache":
            raise RuntimeError("frozen")
        super().__setattr__(name, value)


def test_48_cache_write_crash_still_returns_state():
    before = agency_failures.failure_count()
    assert P.power_state(_FrozenCtx()) == "offline"
    # control: a writable context gets the same value, cached
    ctl = SimpleNamespace(settings=None, router=None)
    assert P.power_state(ctl) == "offline"
    assert getattr(ctl, "_power_state_cache", None) is not None
    _not_recorded(before)


# ── 59: offline-flag read crash -> router evidence decides ──────

class _RaisingSettings:
    @property
    def offline(self):
        raise RuntimeError("settings boom")


def test_59_offline_read_crash_probe_continues():
    before = agency_failures.failure_count()
    ctx = SimpleNamespace(settings=_RaisingSettings(),
                          router=_healthy_router())
    assert P.power_state(ctx) == "full"
    # control: a readable offline=True wins over a healthy router
    ctl = SimpleNamespace(settings=SimpleNamespace(offline=True),
                          router=_healthy_router())
    assert P.power_state(ctl) == "offline"
    _not_recorded(before)


# ── 71: snapshot crash -> degraded, the uncertainty value ───────

def _boom(*a, **k):
    raise RuntimeError("snapshot boom")


def test_71_snapshot_crash_returns_degraded():
    before = agency_failures.failure_count()
    ctx = SimpleNamespace(settings=None,
                          router=SimpleNamespace(stats_snapshot=_boom))
    assert P.power_state(ctx) == "degraded"
    # degraded = attempt with fallback expected, never "full"
    assert P.model_usable(ctx) is True
    # controls: healthy router -> full; no snapshot fn -> the
    # same degraded value (configured-but-unverified family)
    assert P.power_state(SimpleNamespace(
        settings=None, router=_healthy_router())) == "full"
    assert P.power_state(SimpleNamespace(
        settings=None, router=SimpleNamespace())) == "degraded"
    _not_recorded(before)


# ── 109: invalidate crash -> swallowed, TTL bounds staleness ────

def test_109_invalidate_frozen_context_never_raises():
    before = agency_failures.failure_count()
    frozen = _FrozenCtx()
    object.__setattr__(frozen, "_power_state_cache",
                       ("full", time.monotonic()))
    P.invalidate(frozen)  # must not raise
    # control: a writable context is actually cleared
    plain = SimpleNamespace()
    plain._power_state_cache = ("full", time.monotonic())
    P.invalidate(plain)
    assert plain._power_state_cache is None
    _not_recorded(before)
