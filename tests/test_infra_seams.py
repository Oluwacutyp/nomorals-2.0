"""Infrastructure audit seams (Batch 1): failing-first tests.

Each test pins one audit seam from INFRASTRUCTURE_NEXT:
  B — observation truncation: valid JSON + completeness flag + remainder.
  D — tiny numbers never render as zero without an approximation label.
  C — timeout means outcome UNKNOWN, never a blind replay; parallel
      waits share one absolute turn deadline.
  E — idempotency bound to (tool, key, actor, args digest), in-flight
      reservation, restart-safe via db.
  A — effective delivery mode is visible; degraded durability is
      reported on the result, not only in a log.
"""

import json
import threading
import time
import unittest

from nomorals.agents.partner.tool_loop import ToolCallingLoop, ToolCall
from nomorals.core.tool_render import render_tool_value
from nomorals.social.chat.base import ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _loop(**kw):
    return ToolCallingLoop(llm=None, registry=ToolRegistry(), **kw)


class TestSeamBTruncation(unittest.TestCase):
    def test_big_dict_observation_is_valid_json_with_completeness_flag(self):
        loop = _loop()
        big = {"payload": "x" * 5000, "other": "y" * 5000}
        text = loop._render_value(big)
        parsed = json.loads(text)  # must not raise: no mid-JSON cut
        self.assertIs(parsed.get("_complete"), False)
        self.assertIn("_remainder", parsed)
        self.assertEqual(sorted(parsed["_keys"]), ["other", "payload"])
        # Remainder retrievable and complete.
        rest = ToolCallingLoop.retrieve_remainder(parsed["_remainder"])
        self.assertIsNotNone(rest)
        full = json.loads(rest)
        self.assertEqual(full, big)

    def test_small_dict_observation_stays_complete(self):
        loop = _loop()
        text = loop._render_value({"a": 1, "b": "two"})
        parsed = json.loads(text)
        self.assertEqual(parsed, {"a": 1, "b": "two"})

    def test_long_string_observation_marks_truncation_and_remainder(self):
        loop = _loop()
        value = "z" * 9000
        text = loop._render_value(value)
        self.assertIn("truncated", text.lower())
        self.assertIn("complete", text.lower())
        # A remainder handle is embedded; the full string is retrievable.
        key = [tok for tok in text.split() if tok.startswith("rem-")]
        self.assertTrue(key, text[-300:])
        self.assertEqual(ToolCallingLoop.retrieve_remainder(key[0]), value)

    def test_unicode_and_multiline_survive_truncation(self):
        loop = _loop()
        value = {"lines": ["héllo\nwörld 🎉" * 400, "tail"]}
        text = loop._render_value(value)
        parsed = json.loads(text)
        self.assertIs(parsed.get("_complete"), False)
        rest = ToolCallingLoop.retrieve_remainder(parsed["_remainder"])
        self.assertIn("🎉", rest)


class TestSeamDNumericPrecision(unittest.TestCase):
    def test_tiny_number_never_zero(self):
        out = render_tool_value({"measurement": 0.000001})
        self.assertNotIn("measurement: 0\n", out + "\n")
        self.assertNotEqual(out.strip(), "measurement: 0")
        self.assertIn("1", out)  # 1e-06 keeps the magnitude visible

    def test_small_and_normal_numbers(self):
        self.assertIn("0.005", render_tool_value({"v": 0.005}))
        self.assertIn("12.34", render_tool_value({"v": 12.34}))
        self.assertIn("0", render_tool_value({"v": 0.0}))
        self.assertIn("100", render_tool_value({"v": 100.0}))

    def test_tiny_number_in_list(self):
        out = render_tool_value([0.0000005, 2.5])
        self.assertIn("5", out)
        self.assertNotIn("• 0\n", out + "\n")


class TestSeamCBackgroundJobs(unittest.TestCase):
    def test_slow_tool_parks_and_late_result_is_harvested(self):
        # No blunt timeout: a slow tool parks as a background job with
        # heartbeats; its late result is harvested, never faked, never
        # dropped, and never blind-replayed.
        finished: list[str] = []

        def slow():
            time.sleep(2.5)
            finished.append("done")
            return "late-result"

        reg = ToolRegistry()
        reg.register("slow", slow, description="slow")
        loop = ToolCallingLoop(llm=None, registry=reg, park_after_s=0.2)
        obs = loop._execute(ToolCall("slow", {}), actor="owner")
        self.assertIn("parked as a background job", obs)
        self.assertIn("Do NOT re-issue", obs)
        # Nothing is registered as unknown-timeout anymore — the job is
        # parked and visible in the registry with heartbeats.
        self.assertEqual(loop.unknown_outcomes(), [])
        self.assertTrue(
            loop._jobs.list(status="running", labels={"owner": "loop"}))
        # The tool really did finish later: the harvest surfaces it.
        time.sleep(3.0)
        harvested = loop._jobs.collect_ready(labels={"owner": "loop"})
        self.assertEqual(len(harvested), 1)
        self.assertEqual(harvested[0].status, "done")
        self.assertIn("late-result", str(harvested[0].result))
        self.assertEqual(finished, ["done"])

    def test_parallel_turn_uses_one_absolute_deadline(self):
        started = time.monotonic()

        def slow():
            time.sleep(5)
            return "x"

        reg = ToolRegistry()
        reg.register("slow", slow, description="slow")
        loop = ToolCallingLoop(llm=None, registry=reg, tool_timeout_s=1.0,
                               parallel_per_turn=True)
        t0 = time.monotonic()
        out = loop._execute_turn(
            [ToolCall("slow", {}), ToolCall("slow", {}), ToolCall("slow", {})],
            actor="owner")
        elapsed = time.monotonic() - t0
        # Past the shared deadline the calls park (background jobs with
        # heartbeats) instead of timing out — honest, never UNKNOWN.
        self.assertTrue(all("parked as a background job" in o for o in out))
        self.assertTrue(all("UNKNOWN" not in o for o in out))
        # Three fresh per-call waits would take ~3s; one shared deadline
        # takes ~1s (plus scheduling slack).
        self.assertLess(elapsed, 2.5, f"parallel wait took {elapsed:.2f}s")
        self.assertGreater(time.monotonic() - started, 0)


class TestSeamEIdempotency(unittest.TestCase):
    def _reg(self, db=None):
        from types import SimpleNamespace

        reg = ToolRegistry(context=SimpleNamespace(db=db), enforce=False)
        calls: list[dict] = []
        reg.calls_log = calls

        def effect(amount=0):
            calls.append({"amount": amount})
            return {"applied": amount, "n": len(calls)}

        reg.register("effect", effect, description="effect")
        return reg, calls

    def test_conflicting_args_rejected_not_replayed(self):
        reg, calls = self._reg()
        first = reg.call("effect", actor="alice", idempotency_key="k1", amount=10)
        self.assertTrue(first.ok)
        clash = reg.call("effect", actor="alice", idempotency_key="k1", amount=99)
        self.assertFalse(clash.ok)
        self.assertIn("conflict", str(clash.error).lower())
        self.assertEqual(len(calls), 1)

    def test_other_actor_cannot_replay_result(self):
        reg, calls = self._reg()
        reg.call("effect", actor="alice", idempotency_key="k1", amount=10)
        other = reg.call("effect", actor="mallory", idempotency_key="k1",
                         amount=10)
        self.assertFalse(other.ok)
        self.assertEqual(len(calls), 1)

    def test_concurrent_duplicates_execute_once(self):
        reg, calls = self._reg()
        barrier = threading.Barrier(8)
        outcomes = []

        def worker():
            barrier.wait()
            outcomes.append(reg.call("effect", actor="alice",
                                     idempotency_key="k1", amount=10))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(o.ok for o in outcomes))
        self.assertEqual(
            {json.dumps(o.value, sort_keys=True, default=str)
             for o in outcomes},
            {json.dumps(outcomes[0].value, sort_keys=True, default=str)})

    def test_restart_safe_via_db(self):
        db = Database(":memory:")
        db.migrate()
        reg1, calls1 = self._reg(db)
        first = reg1.call("effect", actor="alice", idempotency_key="k1",
                          amount=10)
        self.assertTrue(first.ok)
        # New registry instance, same db = simulated restart.
        reg2, calls2 = self._reg(db)
        again = reg2.call("effect", actor="alice", idempotency_key="k1",
                          amount=10)
        self.assertTrue(again.ok)
        self.assertEqual(again.value, first.value)
        self.assertEqual(len(calls2), 0)


class TestSeamADeliveryVisibility(unittest.TestCase):
    def test_gateway_delivery_mode_visible(self):
        gw = ChatGateway({"local": _LocalAdapter()}, db=None)
        self.assertEqual(gw.delivery_mode, "direct")
        self.assertEqual(gw.status()["_delivery_mode"], "direct")

    def test_degraded_durability_reported_on_result(self):
        gw = ChatGateway({"local": _LocalAdapter()}, db=None,
                         durable_outbox=True)
        self.assertEqual(gw.delivery_mode, "durable-requested-unavailable")
        res = gw.send("local", ChatRef(platform="local", chat_id="1"),
                      "hello", durable=True)
        self.assertTrue(res.ok)
        self.assertTrue(res.delivery_degraded)


class _LocalAdapter:
    name = "local"
    supported_kinds = ()

    def health(self):
        return {"ok": True}

    def send(self, chat, text, **kw):
        return SendResult(ok=True, platform="local", message_id="m-1")


if __name__ == "__main__":
    unittest.main()
