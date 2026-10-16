"""Remaining Tier 0 wiring: VCPU runtime adapter, power gate, search
spine, persisted organ bus, and per-promotion rollback.

Every test here exercises only NEW files plus the pre-existing modules
they wrap — nothing existing was modified except ``__init__.py``
exports.
"""

from __future__ import annotations

import pytest

from nomorals.agents.runtime_vcpu import RuntimeVCPUAdapter
from nomorals.autonomy.power_aware import PowerAwareGate
from nomorals.organs_bus import PersistedBus
from nomorals.power.monitor import PowerStatus
from nomorals.search import spine as spine_mod
from nomorals.search.base import SourceAdapter
from nomorals.search.model import SearchResult
from nomorals.search.spine import SearchSpine
from nomorals.tier0_rollback import (
    clear_registry,
    promotion_status,
    promotions,
    record_promotion,
    rollback,
)
from nomorals.workspace.vcpu import VcpuStatus


# ── runtime → VCPU farm ──────────────────────────────────────────────────────

def test_adapter_uses_farm_and_respects_termux_envelope():
    with RuntimeVCPUAdapter(profile="termux") as adapter:
        assert adapter.using_farm is True
        assert adapter.backend == "vcpu-farm"
        # Termux envelope caps at 3 VCPUs (profile ENVELOPES["termux"]).
        assert adapter.max_workers <= 3
        assert adapter.worker_count <= 3
        assert adapter.worker_count >= 1
        future = adapter.submit(lambda: 6 * 7, affinity="cpu")
        assert future.result(timeout=10) == 42
        # Affinity-aware dispatch: io + balanced also route.
        assert adapter.submit(lambda: "io", affinity="io").result(10) == "io"
        assert adapter.submit(lambda: "b", affinity="balanced").result(10) == "b"


def test_adapter_clamps_max_workers_into_envelope():
    # Asking for 10 workers on Termux must NOT build a 10-VCPU farm.
    with RuntimeVCPUAdapter(profile="termux", max_workers=10) as adapter:
        assert adapter.max_workers == 3
        assert adapter.worker_count == 3


def test_adapter_rejects_unknown_affinity():
    with RuntimeVCPUAdapter(profile="termux") as adapter:
        with pytest.raises(ValueError):
            adapter.submit(lambda: 1, affinity="gpu")


def test_adapter_pause_resume_on_farm():
    with RuntimeVCPUAdapter(profile="termux") as adapter:
        assert adapter.pause() >= 1
        statuses = {v.status for v in adapter._farm.vcpus()}
        assert statuses == {VcpuStatus.PAUSED}
        future = adapter.submit(lambda: "after-resume")
        assert adapter.resume() >= 1
        assert future.result(timeout=10) == "after-resume"


def test_adapter_fallback_is_honest_never_fake_farm():
    def _boom(**_kwargs):
        raise RuntimeError("no farm here")

    with RuntimeVCPUAdapter(profile="termux", farm_factory=_boom) as adapter:
        assert adapter.using_farm is False
        assert adapter.backend == "thread-pool-fallback"
        assert "no farm here" in adapter.fallback_reason
        # Work still completes on the fallback — honestly labelled.
        assert adapter.submit(lambda: 99).result(timeout=10) == 99
        name, future = adapter.submit_named(lambda: 1)
        assert name == "fallback"
        assert future.result(timeout=10) == 1
        # Pause is unsupported on fallback: reports 0, does not hang.
        assert adapter.pause() == 0


# ── power-aware gate ─────────────────────────────────────────────────────────

def _heavy_status() -> PowerStatus:
    return PowerStatus(
        battery_pct=8.0, thermal_state="hot",
        throttled=True, ok=False,
        reasons=["battery critical (8%)"],
        degradation_level=3, degradation_name="critical",
        allowed_tiers=["critical"],
    )


def _ok_status() -> PowerStatus:
    return PowerStatus(
        battery_pct=82.0, thermal_state="nominal",
        throttled=False, ok=True, reasons=[],
        degradation_level=0, degradation_name="full",
        allowed_tiers=["critical", "important", "background", "bulk"],
    )


class _FakeMonitor:
    def __init__(self, status: PowerStatus) -> None:
        self._status = status

    def status(self) -> PowerStatus:
        return self._status


def test_heavy_shed_blocks_background_ticks():
    gate = PowerAwareGate(monitor=_FakeMonitor(_heavy_status()))
    assert gate.heavy_shed() is True
    allowed, _reason, shed = gate.should_tick("research", tier="background")
    assert allowed is False
    assert shed == "heavy"
    allowed, _reason, shed = gate.should_tick("memory", tier="bulk")
    assert allowed is False
    # Critical-tier organs still tick under heavy shed (safety path).
    allowed, _reason, shed = gate.should_tick("watchdog", tier="critical")
    assert allowed is True
    assert shed == "heavy"


def test_gate_tick_never_calls_fn_when_blocked():
    gate = PowerAwareGate(monitor=_FakeMonitor(_heavy_status()))
    calls: list[str] = []

    def _tick() -> str:
        calls.append("ran")
        return "done"

    out = gate.gate_tick("research", _tick, tier="background")
    assert out["skipped"] is True
    assert calls == []  # invariant: no organ tick under heavy shed
    out = gate.gate_tick("watchdog", _tick, tier="critical")
    assert out["skipped"] is False
    assert out["result"] == "done"
    assert calls == ["ran"]


def test_gate_allows_when_power_ok():
    gate = PowerAwareGate(monitor=_FakeMonitor(_ok_status()))
    assert gate.heavy_shed() is False
    allowed, _reason, shed = gate.should_tick("research", tier="background")
    assert allowed is True
    assert shed == "none"


def test_gate_rejects_unknown_tier():
    gate = PowerAwareGate(monitor=_FakeMonitor(_ok_status()))
    with pytest.raises(ValueError):
        gate.should_tick("research", tier="ultra")


def test_gate_fails_open_with_no_monitor():
    # Default sampler-less monitor: ok=True, all tiers allowed.
    gate = PowerAwareGate()
    allowed, _reason, _shed = gate.should_tick("research")
    assert allowed is True


# ── search spine ─────────────────────────────────────────────────────────────

class _FakeAdapter(SourceAdapter):
    name = "memory"
    result_type = "memory"

    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    def search(self, query, *, limit, since=None, before=None):
        return list(self._hits)


def _hit(title: str, snippet: str, source: str, raw: float) -> SearchResult:
    return SearchResult(query="q", title=title, snippet=snippet,
                        source=source, type="memory", raw_score=raw)


def test_spine_dedupes_by_content_hash_and_deterministic_order():
    spine = SearchSpine()
    dup_a = _hit("Same Title", "same snippet text", "memory", 9.0)
    dup_b = _hit("Same Title", "same snippet text", "wisdom", 1.0)
    keep = _hit("Other", "different text here", "memory", 5.0)
    adapters = {
        "memory": _FakeAdapter([dup_a, keep]),
        "wisdom": _FakeAdapter([dup_b]),
    }
    first = spine.search("q", sources=["memory", "wisdom"],
                         adapters=adapters, limit=10)
    assert first.deduped >= 1  # the cross-source duplicate folded away
    titles = [h.title for h in first.hits]
    assert titles.count("Same Title") == 1
    # Deterministic: a second identical run yields the identical order.
    second = spine.search("q", sources=["memory", "wisdom"],
                          adapters=adapters, limit=10)
    assert [h.title for h in second.hits] == titles
    assert [h.score for h in second.hits] == [h.score for h in first.hits]


def test_spine_pipeline_legacy_dedupes_and_ranks():
    spine = SearchSpine()
    hits = [
        [_hit("A", "x", "memory", 3.0), _hit("A", "x", "memory", 3.0)],
        [_hit("B", "y", "wisdom", 1.0)],
    ]
    ranked, dropped = spine.pipeline(hits, "q", fusion="legacy")
    assert dropped == 1
    assert [h.title for h in ranked] == ["A", "B"]


def test_spine_bm25_rerank_prefers_title_match():
    spine = SearchSpine()
    hits = [
        _hit("unrelated doc", "alpha beta gamma", "memory", 0.0),
        _hit("alpha doc", "nothing here", "memory", 0.0),
    ]
    ranked = spine.bm25_rerank(hits, "alpha")
    assert ranked[0].title == "alpha doc"


def test_spine_facades_delegate_to_spine_and_originals_survive():
    from nomorals.search.sources import MemoryAdapter

    inner = _FakeAdapter([
        _hit("Dup", "same", "memory", 2.0),
        _hit("Dup", "same", "memory", 2.0),
        _hit("Solo", "only", "memory", 1.0),
    ])
    facade = spine_mod._SpineFacadeAdapter(inner)
    assert facade.probe() is None  # probe is the original's
    out = facade.search("q", limit=10)
    assert [h.title for h in out].count("Dup") == 1  # spine deduped
    # The original adapter class is untouched (facade wraps, never replaces).
    assert MemoryAdapter.name == "memory"
    assert MemoryAdapter is not spine_mod._SpineFacadeAdapter


def test_spine_facade_factories_construct():
    # Factories build facades around the real adapters without touching them.
    assert spine_mod.memory_hybrid_adapter(context=None).name == "memory"
    assert spine_mod.wisdom_hybrid_adapter(context=None).name == "wisdom"
    assert spine_mod.books_library_adapter(context=None).name == "books"


# ── persisted organ bus ──────────────────────────────────────────────────────

def test_bus_drains_exactly_once(tmp_path):
    bus = PersistedBus(tmp_path / "bus.jsonl")
    bus.publish({"kind": "tick", "dst": "research", "payload": {"n": 1}})
    bus.publish({"kind": "tick", "dst": "research", "payload": {"n": 2}})
    bus.publish({"kind": "tick", "dst": "wisdom", "payload": {"n": 3}})
    assert bus.pending_count() == 3
    first = bus.drain(dst="research")
    assert [e["payload"]["n"] for e in first] == [1, 2]
    assert bus.drain(dst="research") == []  # exactly once
    assert bus.pending_count(dst="wisdom") == 1
    # Durable: a fresh instance on the same files sees nothing redelivered.
    bus2 = PersistedBus(tmp_path / "bus.jsonl")
    assert bus2.drain(dst="research") == []
    assert bus2.pending_count(dst="wisdom") == 1
    assert [e["payload"]["n"] for e in bus2.drain(dst="wisdom")] == [3]


def test_bus_publish_survives_reopen(tmp_path):
    bus = PersistedBus(tmp_path / "bus.jsonl")
    bus.publish({"kind": "ping", "dst": "memory"})
    bus.close()
    reopened = PersistedBus(tmp_path / "bus.jsonl")
    assert reopened.pending_count() == 1


def test_bus_dead_letters_poison_after_max_attempts(tmp_path):
    bus = PersistedBus(tmp_path / "bus.jsonl", max_attempts=2)
    bus.publish({"kind": "boom", "dst": "research", "payload": {"n": 1}})
    bus.publish({"kind": "ok", "dst": "research", "payload": {"n": 2}})

    def _handler(event: dict) -> None:
        if event["kind"] == "boom":
            raise ValueError("poison")

    first = bus.drain_and_handle(_handler, dst="research")
    assert first == {"delivered": 1, "dead_lettered": 0, "retried": 1}
    second = bus.drain_and_handle(_handler, dst="research")
    assert second == {"delivered": 0, "dead_lettered": 1, "retried": 0}
    # Poison is gone from the pending queue and visible in review.
    assert bus.pending_count() == 0
    assert bus.drain() == []
    dead = bus.dead_letter_review()
    assert len(dead) == 1
    assert dead[0]["event"]["kind"] == "boom"
    assert "poison" in dead[0]["reason"]
    assert "2/2" in dead[0]["reason"]


def test_bus_explicit_dead_letter(tmp_path):
    bus = PersistedBus(tmp_path / "bus.jsonl")
    event = bus.publish({"kind": "bad", "dst": "research"})
    bus.dead_letter(event, "manual quarantine")
    assert bus.pending_count() == 0
    review = bus.dead_letter_review()
    assert len(review) == 1
    assert review[0]["reason"] == "manual quarantine"
    # Re-dead-lettering the same id is a no-op (first reason stands).
    bus.dead_letter(event, "second reason")
    assert len(bus.dead_letter_review()) == 1


# ── Tier 0 rollback ─────────────────────────────────────────────────────────

def test_promotion_rollback_is_idempotent():
    clear_registry()
    calls: list[str] = []
    record_promotion("test-promotion-x", lambda: calls.append("rolled"))
    assert promotion_status("test-promotion-x")["status"] == "active"
    rollback("test-promotion-x")
    assert promotion_status("test-promotion-x")["status"] == "rolled_back"
    rollback("test-promotion-x")  # second call: no-op, fn runs once
    assert calls == ["rolled"]


def test_promotion_rollback_unknown_name():
    clear_registry()
    with pytest.raises(KeyError):
        rollback("no-such-promotion")


def test_every_tier0_promotion_registered_and_rollback_works():
    # Earlier tests call clear_registry(); re-register all four here,
    # exactly as module import does.
    from nomorals.agents.runtime_vcpu import (
        register_promotion as runtime_register,
    )
    from nomorals.autonomy.power_aware import (
        register_promotion as gate_register,
    )
    from nomorals.organs_bus import register_promotion as bus_register
    from nomorals.search.spine import register_promotion as spine_register

    runtime_register()
    gate_register()
    bus_register()
    spine_register()

    names = {p["name"] for p in promotions()}
    assert {"runtime_vcpu", "power_aware_gate",
            "search_spine", "organs_bus"} <= names
    # Real rollback path: rolling back the spine promotion flips its flag,
    # then re-registering restores it for the rest of the suite.
    assert spine_mod.promotion_active() is True
    rollback("search_spine")
    assert spine_mod.promotion_active() is False
    spine_register()
    assert spine_mod.promotion_active() is True
