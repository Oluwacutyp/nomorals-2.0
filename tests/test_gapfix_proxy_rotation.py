"""Lane 54 — proxy rotation pool over the owner's proxy list.

``ProxyManager`` answers "which proxy is active"; ``RotationPool``
answers "which proxy next": concurrent probing, EWMA latency plus a
failure penalty, quarantine with exponential backoff, sticky hand-out.
A dead proxy never raises — it is a data point.

Audit tests for the pasted draft's suspect points:
- probe mutations happen under the lock even though probes run on a
  thread pool (hammer test: probe/acquire/report concurrently).
- the sticky lookup cannot KeyError (sticky URL always a pool key).
- the score-weighted pick is deterministic under a seeded RNG.
"""

from __future__ import annotations

import random
import threading
import time
import types
import unittest
from unittest import mock

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.tools import proxy_rotation as PR
from nomorals.tools.proxy_rotation import ProxyHealth, RotationPool
from nomorals.tools.registry import ToolRegistry

_FAST = "http://198.51.100.7:3128"
_SLOW = "http://203.0.113.9:8080"
_CRED = "http://alice:secretpass@203.0.113.9:8080"
_DEAD = "http://192.0.2.55:8888"


class _FakeResp:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.status = 200 if ok else 502
        self.text = "198.51.100.7"


class _FakeHttpClient:
    """Scripted per-proxy behaviour: {proxy_url: ok | Exception}."""

    behavior: dict[str, object] = {}

    def __init__(self, *, timeout: float = 8.0, proxy_url: str = "",
                 **_kw: object) -> None:
        self.proxy_url = proxy_url

    def get(self, url: str, *, timeout: float = 8.0) -> _FakeResp:
        outcome = self.behavior.get(self.proxy_url, True)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResp(ok=bool(outcome))


def _pool(urls, **kw) -> RotationPool:
    return RotationPool(urls, probe_url="https://api.ipify.org", **kw)


def _ctx(known: str = "") -> types.SimpleNamespace:
    proxy = types.SimpleNamespace(
        probe_url="https://api.ipify.org", probe_timeout=1.0,
        known=known, active="", url="")
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(proxy=proxy), db=None)


class TestProxyHealthScore(unittest.TestCase):
    def test_score_is_latency_plus_failure_penalty(self) -> None:
        h = ProxyHealth(url=_FAST, latency_ms=120.0, fail_streak=2)
        self.assertEqual(h.score, 120.0 + 2 * 2_500.0)

    def test_quarantine_property(self) -> None:
        h = ProxyHealth(url=_FAST, quarantined_until=time.monotonic() + 60)
        self.assertTrue(h.quarantined)
        h.quarantined_until = time.monotonic() - 1
        self.assertFalse(h.quarantined)

    def test_as_dict_masks_credentials(self) -> None:
        d = ProxyHealth(url=_CRED).as_dict()
        self.assertNotIn("secretpass", str(d))
        self.assertEqual(d["proxy"], "http://alice:***@203.0.113.9:8080")


class TestProbe(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = mock.patch.object(PR, "HttpClient", _FakeHttpClient)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        _FakeHttpClient.behavior = {}

    def test_dead_proxy_is_a_data_point_not_an_exception(self) -> None:
        _FakeHttpClient.behavior = {_DEAD: ConnectionError("boom")}
        pool = _pool([_FAST, _DEAD])
        rows = pool.probe_all()
        self.assertEqual(len(rows), 2)
        dead = next(r for r in rows if r.url == _DEAD)
        self.assertFalse(dead.ok)
        self.assertEqual(dead.fail_streak, 1)
        # worst-last: the dead row sorts after the healthy one
        self.assertEqual(rows[-1].url, _DEAD)
        live = next(r for r in rows if r.url == _FAST)
        self.assertTrue(live.ok)
        self.assertEqual(live.fail_streak, 0)

    def test_unparseable_url_skipped(self) -> None:
        pool = _pool([_FAST, "   ", _SLOW])
        self.assertEqual(len(pool), 2)

    def test_empty_pool_probes_to_nothing(self) -> None:
        pool = _pool([])
        self.assertEqual(pool.probe_all(), [])
        self.assertEqual(pool.acquire(), None)
        self.assertEqual(pool.snapshot()["size"], 0)


class TestScoringAndQuarantine(unittest.TestCase):
    def test_flaky_fast_loses_to_slow_reliable(self) -> None:
        pool = _pool([_FAST, _SLOW], rng=random.Random(0))
        pool.report(_FAST, ok=True, latency_ms=50.0)
        pool.report(_FAST, ok=False)
        pool.report(_FAST, ok=False)
        pool.report(_SLOW, ok=True, latency_ms=400.0)
        rows = pool.snapshot()["rows"]
        # snapshot sorts by score: slow-reliable first
        self.assertEqual(rows[0]["proxy"], _SLOW)
        self.assertEqual(rows[-1]["proxy"], _FAST)
        self.assertGreater(rows[-1]["latency_ms"], 0)

    def test_ewma_moves_toward_new_latency(self) -> None:
        pool = _pool([_FAST])
        pool.report(_FAST, ok=True, latency_ms=100.0)
        pool.report(_FAST, ok=True, latency_ms=200.0)
        h = pool._health[_FAST]
        # 0 -> 0.7*0 + 0.3*100 = 30, then 0.7*30 + 0.3*200 = 81
        self.assertAlmostEqual(h.latency_ms, 81.0, places=6)

    def test_quarantine_after_max_streak_with_backoff_growth(self) -> None:
        pool = _pool([_FAST], max_fail_streak=3)
        pool.report(_FAST, ok=False)
        pool.report(_FAST, ok=False)
        self.assertFalse(pool._health[_FAST].quarantined)
        pool.report(_FAST, ok=False)
        h = pool._health[_FAST]
        self.assertTrue(h.quarantined)
        remaining = h.quarantined_until - time.monotonic()
        self.assertGreater(remaining, 50.0)
        self.assertLessEqual(remaining, 60.0)
        pool.report(_FAST, ok=False)  # streak 4 -> 120s
        remaining = h.quarantined_until - time.monotonic()
        self.assertGreater(remaining, 110.0)
        self.assertLessEqual(remaining, 120.0)

    def test_quarantine_cap(self) -> None:
        pool = _pool([_FAST], max_fail_streak=3)
        for _ in range(9):
            pool.report(_FAST, ok=False)
        h = pool._health[_FAST]
        remaining = h.quarantined_until - time.monotonic()
        self.assertGreater(remaining, 1700.0)
        self.assertLessEqual(remaining, 1800.0)

    def test_quarantined_proxy_not_acquired_then_returns_after_expiry(self) -> None:
        pool = _pool([_FAST], max_fail_streak=3)
        for _ in range(3):
            pool.report(_FAST, ok=False)
        self.assertIsNone(pool.acquire())
        pool._health[_FAST].quarantined_until = time.monotonic() - 1
        self.assertEqual(pool.acquire(), _FAST)

    def test_success_clears_streak(self) -> None:
        pool = _pool([_FAST])
        pool.report(_FAST, ok=False)
        pool.report(_FAST, ok=False)
        pool.report(_FAST, ok=True, latency_ms=10.0)
        self.assertEqual(pool._health[_FAST].fail_streak, 0)
        self.assertTrue(pool._health[_FAST].ok)

    def test_report_unknown_url_is_silent(self) -> None:
        pool = _pool([_FAST])
        pool.report("http://203.0.113.99:9999", ok=False)  # no raise


class TestAcquire(unittest.TestCase):
    def test_sticky_holds_one_proxy_then_repicks(self) -> None:
        pool = _pool([_FAST, _SLOW], sticky=3, rng=random.Random(7))
        pool.report(_FAST, ok=True, latency_ms=10.0)
        pool.report(_SLOW, ok=True, latency_ms=500.0)
        # Replicate the seeded draw sequence the pool will consume:
        # head is score-ordered [fast, slow]; one draw per fresh pick.
        expect = random.Random(7)
        first = expect.choice([_FAST, _SLOW])
        second = expect.choice([_FAST, _SLOW])
        self.assertEqual(pool.acquire(), first)
        self.assertEqual(pool.acquire(), first)
        self.assertEqual(pool.acquire(), first)
        self.assertEqual(pool.acquire(), second)
        self.assertEqual(pool._health[first].uses, 3)

    def test_exclude_respected(self) -> None:
        pool = _pool([_FAST, _SLOW], rng=random.Random(0))
        pool.report(_FAST, ok=True, latency_ms=10.0)
        pool.report(_SLOW, ok=True, latency_ms=20.0)
        self.assertEqual(pool.acquire(exclude=_FAST), _SLOW)
        self.assertEqual(pool.acquire(exclude=_FAST), _SLOW)
        # excluding the only live proxy (other quarantined) -> direct
        pool2 = _pool([_FAST, _SLOW], max_fail_streak=1)
        pool2.report(_SLOW, ok=False)
        self.assertIsNone(pool2.acquire(exclude=_FAST))

    def test_sticky_cleared_when_quarantined(self) -> None:
        pool = _pool([_FAST, _SLOW], sticky=10, max_fail_streak=1)
        pool.report(_FAST, ok=True, latency_ms=10.0)
        pool.report(_SLOW, ok=True, latency_ms=20.0)
        first = pool.acquire()
        self.assertIn(first, (_FAST, _SLOW))
        other = _SLOW if first == _FAST else _FAST
        pool.report(first, ok=False)
        self.assertIsNone(pool._sticky_url)
        self.assertEqual(pool.acquire(), other)


class TestConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = mock.patch.object(PR, "HttpClient", _FakeHttpClient)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        _FakeHttpClient.behavior = {}

    def test_probe_acquire_report_under_contention(self) -> None:
        pool = _pool([_FAST, _SLOW, _DEAD], sticky=2,
                     rng=random.Random(1))
        _FakeHttpClient.behavior = {_DEAD: ConnectionError("down")}
        errors: list[BaseException] = []

        def probe_loop() -> None:
            try:
                for _ in range(20):
                    pool.probe_all()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def acquire_loop() -> None:
            try:
                for _ in range(200):
                    pool.acquire()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def report_loop() -> None:
            try:
                for _ in range(200):
                    pool.report(_FAST, ok=True, latency_ms=15.0)
                    pool.report(_SLOW, ok=False)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=f) for f in
                   (probe_loop, acquire_loop, report_loop)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        snap = pool.snapshot()
        for row in snap["rows"]:
            self.assertGreaterEqual(row["uses"], 0)
            self.assertGreaterEqual(row["fail_streak"], 0)
            self.assertGreaterEqual(row["latency_ms"], 0)


class TestToolRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = mock.patch.object(PR, "HttpClient", _FakeHttpClient)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        _FakeHttpClient.behavior = {}

    def _registry(self, known: str = "") -> ToolRegistry:
        reg = ToolRegistry(context=_ctx(known))
        PR.register(reg)
        return reg

    def test_both_tools_registered(self) -> None:
        reg = self._registry()
        self.assertIn("proxy_rotate_probe", reg.names())
        self.assertIn("proxy_rotate_pick", reg.names())

    def test_probe_tool_scores_known_list(self) -> None:
        reg = self._registry(f"{_FAST},{_DEAD}")
        _FakeHttpClient.behavior = {_DEAD: ConnectionError("down")}
        out = reg.call("proxy_rotate_probe")
        self.assertTrue(out.ok, out)
        self.assertEqual(out.value["size"], 2)
        self.assertEqual(out.value["healthy"], 1)
        self.assertEqual(out.value["quarantined"], 0)
        self.assertNotIn("secretpass", str(out.value))

    def test_probe_tool_empty_inventory(self) -> None:
        reg = self._registry()
        out = reg.call("proxy_rotate_probe")
        self.assertTrue(out.ok, out)
        self.assertIn("no proxies known", out.value["note"])

    def test_pick_tool_returns_next_proxy(self) -> None:
        reg = self._registry(f"{_FAST},{_SLOW}")
        out = reg.call("proxy_rotate_pick")
        self.assertTrue(out.ok, out)
        self.assertIn(out.value["proxy"], (_FAST, _SLOW))
        self.assertEqual(out.value["pool"], 2)

    def test_pick_tool_empty_inventory_goes_direct(self) -> None:
        reg = self._registry()
        out = reg.call("proxy_rotate_pick")
        self.assertTrue(out.ok, out)
        self.assertIsNone(out.value["proxy"])
        self.assertEqual(out.value["pool"], 0)

    def test_pick_masks_credentials(self) -> None:
        reg = self._registry(_CRED)
        out = reg.call("proxy_rotate_pick")
        self.assertTrue(out.ok, out)
        self.assertNotIn("secretpass", str(out.value))
        self.assertEqual(out.value["proxy"],
                         "http://alice:***@203.0.113.9:8080")


class _FakeTools:
    def __init__(self, values: dict[str, dict]) -> None:
        self._values = values

    def call(self, name: str, **_kwargs):
        return types.SimpleNamespace(ok=True, value=self._values[name])


def _runtime(values: dict[str, dict]) -> RuntimeSystemMixin:
    rt = RuntimeSystemMixin()
    rt.context = types.SimpleNamespace(tools=_FakeTools(values))
    return rt


class TestChatVerbs(unittest.TestCase):
    def test_rotate_probe_verb(self) -> None:
        rt = _runtime({"proxy_rotate_probe": {
            "size": 2, "healthy": 1, "quarantined": 0,
            "rows": [
                {"proxy": _FAST, "ok": True, "fail_streak": 0,
                 "latency_ms": 42, "uses": 3, "quarantined_for": 0},
                {"proxy": _DEAD, "ok": False, "fail_streak": 2,
                 "latency_ms": 8000, "uses": 0, "quarantined_for": 0},
            ]}})
        out = rt._control_proxy("rotate-probe")
        self.assertIn("1 healthy", out)
        self.assertIn(_FAST, out)
        self.assertIn(_DEAD, out)

    def test_rotate_probe_verb_empty(self) -> None:
        rt = _runtime({"proxy_rotate_probe": {
            "note": "no proxies known — add to NM_PROXY_KNOWN or /proxy set"}})
        self.assertIn("no proxies known",
                      rt._control_proxy("rotate-probe"))

    def test_rotate_pick_verb(self) -> None:
        rt = _runtime({"proxy_rotate_pick": {"proxy": _FAST, "pool": 2}})
        out = rt._control_proxy("rotate-pick")
        self.assertIn(_FAST, out)
        self.assertIn("/proxy set", out)

    def test_rotate_pick_verb_direct(self) -> None:
        rt = _runtime({"proxy_rotate_pick": {"proxy": None, "pool": 0}})
        self.assertIn("direct", rt._control_proxy("rotate-pick"))


if __name__ == "__main__":
    unittest.main()
