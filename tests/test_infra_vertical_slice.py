"""Batch 2 vertical slice: one request, identifiable end to end.

request → tool produces overflow data → full artifact saved → preview
rendered → process interrupted before send → restart → result delivered
with receipt → user opens full details → replay explains the run.
Plus a partial-failure variant (one tool fails, the rest succeed).
"""

import json
import time
import unittest

from nomorals.agents.partner.tool_loop import ToolCallingLoop, ToolCall
from nomorals.core.correlation import correlation_context, current_correlation
from nomorals.core.events import global_bus
from nomorals.core.result_contract import ExecutionResult
from nomorals.core.tool_render import render_tool_value
from nomorals.os.replay import replay_mission
from nomorals.os.timeline import Timeline
from nomorals.social.chat.base import ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


class _Adapter:
    name = "local"
    supported_kinds = ()

    def __init__(self) -> None:
        self.sent: list[str] = []

    def health(self):
        return {"ok": True}

    def send(self, chat, text, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform="local",
                          message_id=f"m-{len(self.sent)}")


class TestVerticalSlice(unittest.TestCase):
    def setUp(self):
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.db = Database(f"{self._tmp.name}/slice.db")
        self.db.migrate()
        self.timeline = Timeline()
        self.timeline.attach(global_bus, sync=True)
        self.addCleanup(self.timeline.detach, global_bus)
        self.addCleanup(self._tmp.cleanup)

    def _gateway(self, adapter):
        # durable_drain_sync=False: send() enqueues + returns a receipt
        # handle WITHOUT delivering — the kill-before-delivery point.
        return ChatGateway({"local": adapter}, db=self.db,
                           durable_outbox=True, durable_drain_sync=False)

    def test_full_slice_kill_before_delivery_restart_delivers(self):
        request_id = "req-slice-1"
        run_id = "run-slice-1"
        chat = ChatRef(platform="local", chat_id="1")
        adapter = _Adapter()

        # 1. Tool produces enough data to overflow chat.
        reg = ToolRegistry(enforce=False)

        def big_report():
            return {"rows": [{"i": i, "note": "x" * 300} for i in range(40)]}

        reg.register("big_report", big_report, description="big")
        with correlation_context(request_id=request_id, run_id=run_id,
                                 mission_id="mission-slice-1"):
            outcome = reg.call("big_report", actor="owner")
            self.assertTrue(outcome.ok)
            corr = current_correlation()
        self.assertEqual(corr["request_id"], request_id)

        # 2. Model-facing observation: bounded, valid, remainder retrievable.
        loop = ToolCallingLoop(llm=None, registry=reg)
        obs = loop._render_value(outcome.value)
        parsed = json.loads(obs)
        self.assertIs(parsed.get("_complete"), False)
        full = json.loads(ToolCallingLoop.retrieve_remainder(
            parsed["_remainder"]))
        self.assertEqual(len(full["rows"]), 40)

        # 3. Full artifact saved; preview rendered for chat.
        store = ArtifactStore(self.db, BlobStore(self.db, self._tmp.name + "/blobs"))
        art = store.put_json(full, type="report", creator="big_report",
                             mission_id="mission-slice-1",
                             metadata={"request_id": request_id,
                                       "run_id": run_id})
        self.assertEqual(len(store.read_text(art.id)) > 0, True)
        preview = render_tool_value(
            {"report": art.uri, "rows": len(full["rows"]),
             "status": "ready"})
        self.assertIn(art.uri, preview)

        # 4. Enqueue delivery, then "kill" before send (gateway dropped,
        #    never drained). Restart = fresh gateway + outbox, same db.
        gw1 = self._gateway(adapter)
        res1 = gw1.send("local", chat, preview, durable=True)
        self.assertTrue(res1.ok)
        self.assertEqual(adapter.sent, [])  # nothing delivered yet
        del gw1

        gw2 = self._gateway(adapter)
        delivered = gw2.outbox.deliver_chat(
            chat.key, "local", gw2._deliver_outbox_payload)
        self.assertTrue(all(o["ok"] for o in delivered))
        self.assertEqual(len(adapter.sent), 1)
        self.assertIn(art.uri, adapter.sent[0])

        # 5. Receipt exists; result carries the correlated ids.
        receipts = gw2.outbox.receipts(chat_key=chat.key)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["state"], "sent")
        result = ExecutionResult.succeeded(
            summary="report delivered", data={"rows": 40},
            artifacts=[art.uri], request_id=request_id, run_id=run_id,
            delivery_state="sent")
        self.assertEqual(result.request_id, request_id)
        self.assertEqual(result.delivery_state, "sent")

        # 6. User opens full details from the artifact, not the preview.
        reopened = json.loads(store.read_text(art.id))
        self.assertEqual(len(reopened["rows"]), 40)

        # 7. Replay explains the run: the tool.call event is on the timeline.
        replay = replay_mission(self.timeline, "mission-slice-1")
        self.assertTrue(any(c["tool"] == "big_report"
                            for c in replay.tool_calls),
                        f"tool_calls={replay.tool_calls!r}")

    def test_partial_failure_variant(self):
        reg = ToolRegistry(enforce=False)

        def good():
            return {"n": 3}

        def bad():
            raise RuntimeError("boom")

        reg.register("good", good, description="good")
        reg.register("bad", bad, description="bad")
        loop = ToolCallingLoop(llm=None, registry=reg)
        obs_good = loop._invoke(ToolCall("good", {}), actor="owner")
        obs_bad = loop._invoke(ToolCall("bad", {}), actor="owner")
        self.assertIn("[tool ok]", obs_good)
        self.assertIn("boom", obs_bad)
        result = ExecutionResult.partial(
            summary="1 of 2 tools succeeded", data={"succeeded": ["good"],
                                                   "failed": ["bad"]},
            recovery=["inspect bad-tool failure", "retry remaining"])
        self.assertEqual(result.status, "partial")
        self.assertIn("retry remaining", result.recovery)


if __name__ == "__main__":
    unittest.main()
