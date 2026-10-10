"""Sweep tests for the upgraded power module.

Covers: error taxonomy, jittered backoff policies, energy ledger,
local (psutil/sysfs) sampler, hysteresis, drain history, status cards,
task constraints, anti-starvation escalation, non-preemptible tasks,
deferred-task inspection, cancel, and energy-ordered batches.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import dataclass

from nomorals.core.events import global_bus
from nomorals.power import (
    CONSTRAINT_KEYS,
    BackoffPolicy,
    EnergyLedger,
    InvalidPowerSpecError,
    PowerAwareScheduler,
    PowerBudgetExhaustedError,
    PowerConstrainedError,
    PowerError,
    PowerMonitor,
    PowerSamplerUnavailableError,
    PowerStatus,
    TaskDeferredError,
    local_sampler,
)
from nomorals.storage.db import Database


def _db() -> Database:
    from nomorals.storage.migrations import MIGRATIONS
    from nomorals.storage.schema import MigrationRunner
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


@dataclass
class _FakeSample:
    battery_percent: float | None = None
    thermal_state: str | None = None
    battery_status: str = ""
    idle: bool | None = None


class _FakeManager:
    """Deterministic ResourceManager stand-in."""

    def __init__(self, battery=None, thermal=None, throttled=False, ok=True,
                 reasons=None, degradation_level=0, degradation_name="full",
                 allowed_tiers=None, budgets=None, charging=None,
                 battery_status="", drain_pct_per_h=None, idle=None):
        self._sample = _FakeSample(battery, thermal, battery_status, idle)
        self._advisory = {
            "throttled": throttled, "ok": ok, "reasons": reasons or [],
            "degradation_level": degradation_level,
            "degradation": {
                "level": degradation_level, "name": degradation_name,
                "allowed_tiers": allowed_tiers or
                ["critical", "important", "background", "bulk"],
            },
            "battery_forecast": {
                k: v for k, v in {
                    "charging": charging,
                    "drain_pct_per_h": drain_pct_per_h,
                }.items() if v is not None
            },
        }
        self.budgets = budgets

    def sample(self):
        return self._sample

    def consult(self, mission=None):
        return dict(self._advisory)


class _ZeroRng:
    """Deterministic RNG: always the lowest possible delay."""

    def uniform(self, a, b):
        return a


class _MaxRng:
    """Deterministic RNG: always the highest possible delay."""

    def uniform(self, a, b):
        return b


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class ErrorTaxonomyTests(unittest.TestCase):
    def test_base_carries_code_and_flag(self):
        e = PowerError("boom")
        self.assertEqual(e.code, "power_error")
        self.assertFalse(e.retryable)
        d = e.to_dict()
        self.assertEqual(d["code"], "power_error")
        self.assertEqual(d["message"], "boom")

    def test_retryable_family(self):
        for cls, code in (
            (TaskDeferredError, "task_deferred"),
            (PowerConstrainedError, "power_constrained"),
            (PowerBudgetExhaustedError, "budget_exhausted"),
            (PowerSamplerUnavailableError, "sampler_unavailable"),
        ):
            e = cls("x", retry_after_s=12.0)
            self.assertTrue(e.retryable, cls)
            self.assertEqual(e.code, code, cls)
            self.assertEqual(e.retry_after_s, 12.0)

    def test_invalid_spec_is_terminal(self):
        e = InvalidPowerSpecError("bad class")
        self.assertFalse(e.retryable)
        self.assertEqual(e.code, "invalid_power_spec")

    def test_task_deferred_carries_reason(self):
        e = TaskDeferredError(reason="tier_shed", retry_after_s=30.0,
                              job_id="j1")
        self.assertEqual(e.reason, "tier_shed")
        self.assertEqual(e.job_id, "j1")
        self.assertIn("tier_shed", e.details["reason"])


# ---------------------------------------------------------------------------
# backoff
# ---------------------------------------------------------------------------

class BackoffPolicyTests(unittest.TestCase):
    def test_full_jitter_bounds_and_growth(self):
        p = BackoffPolicy(base=10.0, cap=1000.0, strategy="full",
                          rng=_MaxRng())
        self.assertEqual(p.next_delay(0), 10.0)
        self.assertEqual(p.next_delay(1), 20.0)
        self.assertEqual(p.next_delay(2), 40.0)

    def test_cap_respected(self):
        p = BackoffPolicy(base=10.0, cap=25.0, strategy="full",
                          rng=_MaxRng())
        self.assertEqual(p.next_delay(10), 25.0)

    def test_full_jitter_minimum_is_zero(self):
        p = BackoffPolicy(base=10.0, cap=1000.0, rng=_ZeroRng())
        self.assertEqual(p.next_delay(3), 0.0)

    def test_equal_jitter_bounds(self):
        p = BackoffPolicy(base=10.0, cap=1000.0, strategy="equal",
                          rng=_MaxRng())
        # cap_delay/2 + cap_delay/2 == cap_delay
        self.assertEqual(p.next_delay(2), 40.0)
        p2 = BackoffPolicy(base=10.0, cap=1000.0, strategy="equal",
                           rng=_ZeroRng())
        self.assertEqual(p2.next_delay(2), 20.0)

    def test_none_is_deterministic_exponential(self):
        p = BackoffPolicy(base=5.0, cap=100.0, strategy="none")
        self.assertEqual(p.next_delay(0), 5.0)
        self.assertEqual(p.next_delay(3), 40.0)
        self.assertEqual(p.next_delay(9), 100.0)

    def test_decorrelated_is_per_key_stateful(self):
        p = BackoffPolicy(base=10.0, cap=1000.0, strategy="decorrelated",
                          rng=_MaxRng())
        d1 = p.next_delay(0, key="a")  # uniform(10, 30) -> 30
        d2 = p.next_delay(0, key="a")  # uniform(10, 90) -> 90
        self.assertEqual(d1, 30.0)
        self.assertEqual(d2, 90.0)
        # other key starts fresh
        self.assertEqual(p.next_delay(0, key="b"), 30.0)
        p.reset(key="a")
        self.assertEqual(p.next_delay(0, key="a"), 30.0)

    def test_bad_strategy_rejected(self):
        with self.assertRaises(ValueError):
            BackoffPolicy(strategy="sometimes")

    def test_to_dict(self):
        d = BackoffPolicy(base=7, cap=70, strategy="equal").to_dict()
        self.assertEqual(d, {"base": 7.0, "cap": 70.0, "strategy": "equal"})


# ---------------------------------------------------------------------------
# energy ledger
# ---------------------------------------------------------------------------

class EnergyLedgerTests(unittest.TestCase):
    def test_unknown_type_returns_none(self):
        led = EnergyLedger()
        self.assertIsNone(led.estimated_wh("never_seen"))
        self.assertIsNone(led.avg_duration_s("never_seen"))
        self.assertIsNone(led.batch_wh([("never_seen", 0)]))

    def test_learned_estimate(self):
        led = EnergyLedger()
        led.record("train", 3600.0, mem_mb=2000.0)   # 1h
        led.record("train", 1800.0, mem_mb=2000.0)   # 0.5h
        avg = led.avg_duration_s("train")
        self.assertAlmostEqual(avg, 2700.0)
        wh = led.estimated_wh("train", mem_mb=2000.0)
        # (8 + 0.001*2000) W * 0.75 h = 10 * 0.75
        self.assertAlmostEqual(wh, 7.5)

    def test_batch_wh_sums_known(self):
        led = EnergyLedger()
        led.record("a", 3600.0)
        led.record("b", 1800.0)
        total = led.batch_wh([("a", 0.0), ("b", 0.0)])
        self.assertAlmostEqual(total, 8.0 * 1.0 + 8.0 * 0.5)

    def test_summary_shape(self):
        led = EnergyLedger()
        led.record("x", 60.0, mem_mb=512)
        s = led.summary()["x"]
        self.assertEqual(s["runs"], 1)
        self.assertAlmostEqual(s["avg_duration_s"], 60.0)
        self.assertGreater(s["estimated_wh_per_run"], 0)

    def test_bad_input_never_raises(self):
        led = EnergyLedger()
        led.record("", -5)
        led.record("t", None)  # type: ignore[arg-type]
        self.assertIsNone(led.estimated_wh(""))


# ---------------------------------------------------------------------------
# local sampler
# ---------------------------------------------------------------------------

class _FakePsutilBattery:
    def __init__(self, percent, plugged, secsleft):
        self.percent = percent
        self.power_plugged = plugged
        self.secsleft = secsleft


class _FakePsutil:
    def __init__(self, battery=None, temps=None):
        self._battery = battery
        self._temps = temps or {}

    def sensors_battery(self):
        return self._battery

    def sensors_temperatures(self):
        return self._temps


class _FakeTemp:
    def __init__(self, current, high=None, critical=None):
        self.current = current
        self.high = high
        self.critical = critical


class LocalSamplerTests(unittest.TestCase):
    def _factory(self, **kw):
        return local_sampler(**kw)

    def test_psutil_battery_and_thermal(self):
        ps = _FakePsutil(
            battery=_FakePsutilBattery(82.0, False, 7200),
            temps={"coretemp": [_FakeTemp(55.0, high=100.0,
                                         critical=105.0)]})
        sampler = self._factory(psutil_mod=ps)()
        s = sampler.sample()
        self.assertAlmostEqual(s.battery_percent, 82.0)
        self.assertEqual(s.thermal_state, "nominal")
        adv = sampler.consult()
        self.assertTrue(adv["ok"])
        self.assertFalse(adv["throttled"])
        self.assertEqual(adv["degradation_level"], 0)

    def test_critical_battery_not_ok(self):
        ps = _FakePsutil(battery=_FakePsutilBattery(8.0, False, 900))
        sampler = self._factory(psutil_mod=ps)()
        sampler.sample()
        adv = sampler.consult()
        self.assertFalse(adv["ok"])
        self.assertTrue(adv["throttled"])
        self.assertEqual(adv["degradation_level"], 3)
        self.assertEqual(adv["degradation"]["allowed_tiers"], ["critical"])

    def test_charging_clears_battery_pressure(self):
        ps = _FakePsutil(battery=_FakePsutilBattery(8.0, True, -1))
        sampler = self._factory(psutil_mod=ps)()
        sampler.sample()
        adv = sampler.consult()
        self.assertTrue(adv["ok"])
        self.assertFalse(adv["throttled"])
        self.assertTrue(adv["battery_forecast"]["charging"])

    def test_hot_thermal_throttles(self):
        ps = _FakePsutil(
            battery=_FakePsutilBattery(80.0, False, 9000),
            temps={"coretemp": [_FakeTemp(106.0, high=100.0,
                                         critical=105.0)]})
        sampler = self._factory(psutil_mod=ps)()
        s = sampler.sample()
        self.assertEqual(s.thermal_state, "hot")
        adv = sampler.consult()
        self.assertTrue(adv["throttled"])
        self.assertIn("thermal hot", adv["reasons"])

    def test_warm_thermal_is_advisory_only(self):
        ps = _FakePsutil(
            battery=_FakePsutilBattery(80.0, False, 9000),
            temps={"coretemp": [_FakeTemp(75.0, high=100.0,
                                         critical=105.0)]})
        sampler = self._factory(psutil_mod=ps)()
        sampler.sample()
        adv = sampler.consult()
        self.assertFalse(adv["throttled"])
        self.assertTrue(adv["ok"])

    def test_sysfs_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            bat = os.path.join(root, "BAT0")
            ac = os.path.join(root, "ACAD")
            os.makedirs(bat)
            os.makedirs(ac)
            open(os.path.join(bat, "type"), "w").write("Battery")
            open(os.path.join(bat, "capacity"), "w").write("42\n")
            open(os.path.join(bat, "status"), "w").write("Discharging\n")
            open(os.path.join(ac, "type"), "w").write("Mains")
            open(os.path.join(ac, "online"), "w").write("0\n")
            sampler = self._factory(sysfs_root=root)()
            s = sampler.sample()
            self.assertAlmostEqual(s.battery_percent, 42.0)
            self.assertEqual(s.battery_status, "Discharging")
            adv = sampler.consult()
            self.assertTrue(adv["ok"])  # 42% is fine

    def test_sysfs_charging_detected(self):
        with tempfile.TemporaryDirectory() as root:
            bat = os.path.join(root, "BAT1")
            os.makedirs(bat)
            open(os.path.join(bat, "type"), "w").write("Battery")
            open(os.path.join(bat, "capacity"), "w").write("10\n")
            open(os.path.join(bat, "status"), "w").write("Charging\n")
            sampler = self._factory(sysfs_root=root)()
            sampler.sample()
            adv = sampler.consult()
            self.assertTrue(adv["ok"])  # charging clears low-battery pressure

    def test_no_sources_degrades_to_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            sampler = self._factory(sysfs_root=root)()
            s = sampler.sample()
            self.assertIsNone(s.battery_percent)
            self.assertIsNone(s.thermal_state)
            adv = sampler.consult()
            self.assertTrue(adv["ok"])
            self.assertFalse(adv["throttled"])


# ---------------------------------------------------------------------------
# monitor: hysteresis, history, cards, transitions
# ---------------------------------------------------------------------------

class MonitorHysteresisTests(unittest.TestCase):
    def _flapping(self, stability):
        # alternates throttled True/False each call
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return _FakeManager(throttled=calls["n"] % 2 == 0,
                                reasons=["blip"] if calls["n"] % 2 == 0 else [])
        return PowerMonitor(sampler=factory, stability_reads=stability)

    def test_single_blip_does_not_flip(self):
        m = self._flapping(stability=3)
        first = m.status()
        self.assertFalse(first.throttled)  # damped init: first raw wins
        # second call: raw=True, damped=False, pending=1 < 3 -> stays False
        second = m.status()
        self.assertFalse(second.throttled)
        self.assertTrue(any("stable reads" in r for r in second.reasons))

    def test_sustained_change_flips(self):
        m = PowerMonitor(
            sampler=lambda: _FakeManager(throttled=True, reasons=["hot"]),
            stability_reads=2)
        m.status()  # raw=True, damped=None -> True immediately
        # now flip raw to False twice
        m._sampler = lambda: _FakeManager(throttled=False)
        s1 = m.status()
        self.assertTrue(s1.throttled)   # pending=1 < 2
        s2 = m.status()
        self.assertFalse(s2.throttled)  # pending=2 -> flips

    def test_stability_one_preserves_old_behavior(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(throttled=True),
                         stability_reads=1)
        self.assertTrue(m.status().throttled)


class MonitorHistoryTests(unittest.TestCase):
    def test_drain_rate_from_history(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(battery=80.0))
        now = time.time()
        # 10% over 1h of discharging
        m._history.append((now - 3600, 80.0, False))
        m._history.append((now, 70.0, False))
        self.assertAlmostEqual(m._history_drain_rate(), 10.0)

    def test_drain_rate_needs_span(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(battery=80.0))
        now = time.time()
        m._history.append((now - 30, 80.0, False))
        m._history.append((now, 70.0, False))
        self.assertIsNone(m._history_drain_rate())

    def test_charging_samples_ignored(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(battery=80.0))
        now = time.time()
        m._history.append((now - 3600, 50.0, True))
        m._history.append((now, 70.0, True))
        self.assertIsNone(m._history_drain_rate())

    def test_can_sustain(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(battery=80.0))
        now = time.time()
        m._history.append((now - 3600, 80.0, False))
        m._history.append((now, 70.0, False))  # 10%/h
        m.drain_rate_pct_per_h = lambda: 10.0
        self.assertTrue(m.can_sustain(30, 10.0))    # 5% <= 10%
        self.assertFalse(m.can_sustain(120, 10.0))  # 20% > 10%

    def test_can_sustain_unknown_is_optimistic(self):
        m = PowerMonitor(sampler=lambda: _FakeManager())
        m.drain_rate_pct_per_h = lambda: None
        self.assertTrue(m.can_sustain(600, 1.0))

    def test_status_picks_up_local_forecast(self):
        m = PowerMonitor(sampler=lambda: _FakeManager(battery=60.0))
        now = time.time()
        m._history.append((now - 3600, 70.0, False))
        m._history.append((now, 60.0, False))
        st = m.status()  # sampler has no forecast; history fills it in
        self.assertAlmostEqual(st.drain_rate_pct_per_h, 10.0, places=2)
        self.assertAlmostEqual(st.time_to_empty_min, 360.0, places=0)


class StatusCardTests(unittest.TestCase):
    def test_format_renders_card(self):
        st = PowerStatus(
            battery_pct=82.0, thermal_state="nominal", throttled=False,
            ok=True, reasons=[], charging=True,
            drain_rate_pct_per_h=4.0, time_to_empty_min=1200.0,
            degradation_level=0, degradation_name="full",
            allowed_tiers=["critical", "important", "background", "bulk"])
        card = st.format()
        self.assertIn("82%", card)
        self.assertIn("nominal", card)
        self.assertIn("all classes flowing", card)
        self.assertIn("critical✓", card)

    def test_format_shows_shed_tiers(self):
        st = PowerStatus(
            battery_pct=10.0, thermal_state="hot", throttled=True, ok=False,
            reasons=["battery critical"], charging=False,
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"])
        card = st.format()
        self.assertIn("bulk✗", card)
        self.assertIn("HEAVY+MEDIUM DEFERRED", card)
        self.assertIn("battery critical", card)


class TransitionEventTests(unittest.TestCase):
    def test_degradation_and_charging_events(self):
        seen: list = []
        sub = global_bus.subscribe("power.*", lambda e: seen.append(e),
                                   sync=True)
        try:
            states = [
                _FakeManager(degradation_level=0, charging=False),
                _FakeManager(degradation_level=2, degradation_name="low",
                             allowed_tiers=["critical", "important"],
                             charging=True),
            ]
            calls = {"n": 0}

            def factory():
                m = states[min(calls["n"], 1)]
                calls["n"] += 1
                return m

            mon = PowerMonitor(sampler=factory)
            mon.status()  # baseline, no events
            mon.status()  # level 0->2 and charging False->True
            topics = {e.topic for e in seen}
            self.assertIn("power.degradation.changed", topics)
            self.assertIn("power.charging.changed", topics)
            deg = next(e for e in seen
                       if e.topic == "power.degradation.changed")
            self.assertEqual(deg.data["from_level"], 0)
            self.assertEqual(deg.data["to_level"], 2)
        finally:
            global_bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------

class _ZeroRngPolicy(BackoffPolicy):
    def __init__(self, **kw):
        super().__init__(rng=_ZeroRng(), **kw)


class SchedulerSweepTests(unittest.TestCase):
    def _sched(self, rng_zero=True, **mgr_kw):
        monitor = PowerMonitor(sampler=lambda: _FakeManager(**mgr_kw))
        kw: dict = {"monitor": monitor}
        if rng_zero:
            kw["backoff_rng"] = _ZeroRng()
        return PowerAwareScheduler(_db(), **kw)

    # -- constraints ----------------------------------------------------
    def test_requires_charging_defers_when_unplugged(self):
        s = self._sched(charging=False)
        s.dispatch("backup", {}, power_class="light",
                   constraints={"requires_charging": True})
        self.assertEqual(s.poll(), [])
        waiting = s.deferred_tasks()
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0]["defer_reason"], "constraint_unmet")

    def test_requires_charging_flows_when_plugged(self):
        s = self._sched(charging=True)
        s.dispatch("backup", {}, power_class="light",
                   constraints={"requires_charging": True})
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)

    def test_requires_battery_not_low(self):
        s = self._sched(battery=10.0, charging=False)
        s.dispatch("sync", {}, power_class="light",
                   constraints={"requires_battery_not_low": True})
        self.assertEqual(s.poll(), [])
        s2 = self._sched(battery=80.0, charging=False)
        s2.dispatch("sync", {}, power_class="light",
                    constraints={"requires_battery_not_low": True})
        self.assertEqual(len(s2.poll()), 1)

    def test_unknown_constraint_rejected(self):
        s = self._sched()
        with self.assertRaises(InvalidPowerSpecError):
            s.dispatch("t", {}, constraints={"requires_wifi": True})

    def test_constraint_keys_exported(self):
        self.assertIn("requires_charging", CONSTRAINT_KEYS)

    # -- jittered deferral bookkeeping ----------------------------------
    def test_deferral_counts_and_backoff_grows(self):
        s = self._sched(battery=5.0, throttled=True)  # heavy gated
        s.dispatch("train", {}, power_class="heavy")
        self.assertEqual(s.poll(), [])  # heavy not even in plan: no deferral
        # tier shed path (light class, bulk tier at level 3)
        s2 = PowerAwareScheduler(
            _db(),
            monitor=PowerMonitor(sampler=lambda: _FakeManager(
                degradation_level=3, degradation_name="critical",
                allowed_tiers=["critical"])),
            backoff_rng=_ZeroRng())
        s2.dispatch("job", {}, power_class="light", tier="bulk")
        self.assertEqual(s2.poll(), [])
        waiting = s2.deferred_tasks()
        self.assertEqual(waiting[0]["deferred"], 1)
        self.assertEqual(waiting[0]["defer_reason"], "tier_shed")

    def test_backoff_policy_wired(self):
        s = self._sched()
        self.assertIsInstance(s.deferral_backoff, BackoffPolicy)
        self.assertIsInstance(s.budget_backoff, BackoffPolicy)
        self.assertEqual(s.deferral_backoff.base, 30.0)
        self.assertEqual(s.budget_backoff.base, 60.0)

    # -- anti-starvation escalation --------------------------------------
    def test_escalation_after_max_deferrals(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        seen = []
        sub = global_bus.subscribe("power.task.escalated",
                                   lambda e: seen.append(e), sync=True)
        try:
            s.dispatch("job", {}, power_class="light", tier="bulk",
                       max_deferrals=1)
            self.assertEqual(s.poll(), [])          # deferred ×1
            tasks = s.poll()                        # escalated → admitted
            self.assertEqual(len(tasks), 1)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].data["deferred"], 1)
        finally:
            global_bus.unsubscribe(sub)

    def test_overdue_task_escalates(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        s.dispatch("job", {}, power_class="light", tier="bulk",
                   deadline=time.time() - 1)
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)

    # -- non-preemptible --------------------------------------------------
    def test_non_preemptible_survives_tier_shed(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        s.dispatch("vip", {}, power_class="light", tier="bulk",
                   preemptible=False)
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0].preemptible)

    def test_preemptible_default_sheds(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        s.dispatch("job", {}, power_class="light", tier="bulk")
        self.assertEqual(s.poll(), [])

    # -- force ------------------------------------------------------------
    def test_force_overrides_all_gating(self):
        s = self._sched(battery=5.0, throttled=True)
        s.dispatch("train", {}, power_class="heavy")
        tasks = s.poll(force=True)
        self.assertEqual(len(tasks), 1)

    # -- deferred inspection / cancel --------------------------------------
    def test_deferred_tasks_lists_waiting(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        s.dispatch("a", {}, power_class="light", tier="bulk", priority=5)
        s.dispatch("b", {}, power_class="light", tier="bulk", priority=1)
        s.poll()
        waiting = s.deferred_tasks()
        self.assertEqual(len(waiting), 2)
        self.assertEqual(waiting[0]["priority"], 5)  # sorted by priority

    def test_cancel_drops_task(self):
        mon = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        s = PowerAwareScheduler(_db(), monitor=mon, backoff_rng=_ZeroRng())
        jid = s.dispatch("job", {}, power_class="light", tier="bulk")
        s.poll()
        self.assertEqual(len(s.deferred_tasks()), 1)
        s.cancel(jid)
        self.assertEqual(s.deferred_tasks(), [])
        self.assertEqual(s.poll(), [])

    def test_cancel_unknown_raises_power_error(self):
        s = self._sched()
        with self.assertRaises(PowerError):
            s.cancel("nope")

    def test_invalid_power_class_raises_structured_error(self):
        s = self._sched()
        with self.assertRaises(InvalidPowerSpecError) as cm:
            s.dispatch("t", {}, power_class="ultra")
        self.assertEqual(cm.exception.code, "invalid_power_spec")
        self.assertFalse(cm.exception.retryable)

    def test_invalid_tier_raises_structured_error(self):
        s = self._sched()
        with self.assertRaises(InvalidPowerSpecError):
            s.dispatch("t", {}, tier="mega")

    # -- energy ------------------------------------------------------------
    def test_complete_feeds_energy_ledger(self):
        s = self._sched()
        jid = s.dispatch("train", {}, power_class="light", mem_mb=512)
        s.poll()
        s.complete(jid)
        summary = s.energy.summary()
        self.assertIn("train", summary)
        self.assertEqual(summary["train"]["runs"], 1)

    def test_batch_ordered_cheapest_first(self):
        s = self._sched()
        s.energy.record("heavy_train", 7200.0, mem_mb=4000.0)
        s.energy.record("quick_ping", 2.0, mem_mb=50.0)
        s.dispatch("heavy_train", {}, power_class="light")
        s.dispatch("quick_ping", {}, power_class="light")
        tasks = s.poll(batch=2)
        self.assertEqual([t.task_type for t in tasks],
                         ["quick_ping", "heavy_train"])

    def test_tier_outranks_energy(self):
        s = self._sched()
        s.energy.record("cheap", 1.0)
        s.energy.record("pricey", 99999.0)
        s.dispatch("pricey", {}, power_class="light", tier="critical")
        s.dispatch("cheap", {}, power_class="light", tier="bulk")
        tasks = s.poll(batch=2)
        self.assertEqual(tasks[0].task_type, "pricey")

    # -- summaries ----------------------------------------------------------
    def test_summary_renders(self):
        s = self._sched(battery=5.0, throttled=True)
        s.dispatch("train", {}, power_class="heavy", tier="bulk")
        out = s.summary()
        self.assertIn("power scheduler", out)
        self.assertIn("heavy: 1", out)
        self.assertIn("deferred", out)

    def test_next_window_hint(self):
        s = self._sched(battery=5.0, throttled=True)
        hint = s.next_window_hint()
        self.assertIn("5%", hint)
        s2 = self._sched(battery=80.0)
        self.assertIn("flowing now", s2.next_window_hint())

    def test_stats_includes_new_sections(self):
        s = self._sched()
        st = s.stats()
        self.assertIn("energy", st)
        self.assertIn("backoff", st)
        self.assertEqual(st["backoff"]["deferral"]["strategy"], "full")

    def test_dispatch_roundtrip_new_fields(self):
        s = self._sched(charging=True)
        jid = s.dispatch("t", {"x": 1}, power_class="light",
                         constraints={"requires_charging": True},
                         preemptible=False, max_deferrals=3,
                         deadline=time.time() + 600)
        tasks = s.poll()
        self.assertEqual(len(tasks), 1)
        t = tasks[0]
        self.assertEqual(t.job_id, jid)
        self.assertEqual(t.constraints, {"requires_charging": True})
        self.assertFalse(t.preemptible)
        self.assertEqual(t.max_deferrals, 3)
        self.assertFalse(t.overdue)
        self.assertFalse(t.escalated)
        d = t.to_dict()
        self.assertEqual(d["constraints"], {"requires_charging": True})


if __name__ == "__main__":
    unittest.main()
