"""Gapfix R3-69: Wisdom Tier 2/3 organs wired into the keeper.

Failing-first: every test here must fail on the unmodified tree
(no keeper.compare_positions / lineage / practice_memory, no
/wisdom lineage verb, no recall subverb, no completeness in status)
and pass after the wiring.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.agents.partner.runtime_wisdom import RuntimeWisdomMixin
from nomorals.cmdline.commands.wisdom import _cmd_wisdom
from nomorals.wisdom import WisdomChatManager, WisdomKeeper
from nomorals.wisdom.practice_memory import MissingSourceError


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-tiers-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


class _FakeNotifier:
    def __init__(self):
        self.sent = []

    def publish(self, *args, **kwargs):
        self.sent.append((args, kwargs))


class FakeRuntime(RuntimeWisdomMixin):
    """Test double carrying the real mixin methods."""

    def __init__(self, ctx, manager):
        self.context = ctx
        self._manager = manager
        self.sent: list[str] = []

    def _ref_from_key(self, key):
        plat, _, cid = key.partition(":")
        return SimpleNamespace(platform=plat or "local", chat_id=cid,
                               key=key)

    def _send_long_checked(self, platform, chat, text):
        self.sent.append(text)
        return ""

    def _wisdom_chat_manager(self):
        return self._manager


class FakeClock:
    """Injectable clock: records sleeps, never actually waits."""

    def __init__(self, now=1700000000.0):
        self._now = now
        self.ticks: list[float] = []

    def sleep(self, seconds):
        self.ticks.append(float(seconds))

    def now(self):
        return self._now


def _runtime(ctx):
    return FakeRuntime(ctx, WisdomChatManager(ctx, clock=FakeClock(),
                                              notifier=_FakeNotifier()))


class KeeperTierTests(unittest.TestCase):
    def test_compare_positions_known_topic(self):
        ctx, _ = _ctx()
        d = WisdomKeeper(ctx).compare_positions("self")
        trads = [p.tradition for p in d.positions]
        self.assertEqual(len(d.positions), 3)
        self.assertIn("Advaita Vedanta", trads)
        self.assertIn("Theravada Buddhism", trads)
        self.assertIn("Ifa / Yoruba", trads)
        self.assertIn("not averaged", d.to_dict()["note"])
        # Positions stay side by side: no merged/synthesized claim.
        self.assertTrue(all(p.basis and p.sources for p in d.positions))

    def test_compare_positions_unknown_topic_raises(self):
        ctx, _ = _ctx()
        with self.assertRaises(ValueError) as cm:
            WisdomKeeper(ctx).compare_positions("no-such-topic")
        self.assertIn("self", str(cm.exception))

    def test_disagreement_topics(self):
        ctx, _ = _ctx()
        self.assertEqual(WisdomKeeper(ctx).disagreement_topics(),
                         ["divine", "liberation", "self"])

    def test_lineage_known_name(self):
        ctx, _ = _ctx()
        out = WisdomKeeper(ctx).lineage("Buddha")
        self.assertTrue(out, "lineage should find the Buddha node")
        names = [entry["node"]["name"] for entry in out]
        self.assertTrue(any("Buddha" in n for n in names))
        # Every neighbour edge carries its source — unsourced lineage
        # is never stored, so every edge here must name one.
        for entry in out:
            for nb in entry["neighbours"]:
                self.assertTrue(nb["edge"]["source"])

    def test_lineage_nodes_lists_timeline_graph(self):
        ctx, _ = _ctx()
        nodes = WisdomKeeper(ctx).lineage_nodes()
        self.assertEqual(len(nodes), 52)
        self.assertIn("Siddhartha Gautama, the Buddha",
                      [n["name"] for n in nodes])

    def test_practice_memory_round_trip(self):
        ctx, tmp = _ctx()
        k = WisdomKeeper(ctx)
        mem = k.practice_memory
        item = mem.add_item("What is ori?", "pack: nigeria",
                            source={"tradition": "Yoruba",
                                    "work": "Nigeria pack"}, now=0.0)
        self.assertEqual([i.id for i in mem.due_items(now=0.0)],
                         [item.id])
        mem.review(item.id, 5, now=0.0)
        self.assertEqual(mem.get(item.id).interval_days, 1)
        self.assertEqual(mem.stats(now=0.0)["items"], 1)
        # Durable: the store lives under the wisdom root, next to the
        # practice log — a fresh keeper on the same home sees it.
        import os
        self.assertTrue(os.path.exists(
            os.path.join(tmp, "wisdom", "practice_memory.json")))
        k2 = WisdomKeeper(ctx)
        self.assertEqual(k2.practice_memory.get(item.id).prompt,
                         "What is ori?")

    def test_practice_memory_requires_source(self):
        ctx, _ = _ctx()
        mem = WisdomKeeper(ctx).practice_memory
        with self.assertRaises(MissingSourceError):
            mem.add_item("What is ori?", "pack: nigeria",
                         source={"tradition": "Yoruba"})

    def test_status_carries_completeness(self):
        ctx, _ = _ctx()
        report = WisdomKeeper(ctx).status()["completeness"]
        self.assertTrue(report["all_done"], report["items"])
        labels = [i["item"] for i in report["items"]]
        self.assertIn("disagreement map", labels)
        self.assertIn("lineage graph", labels)
        self.assertIn("practice memory + spaced repetition", labels)


class ChatTierTests(unittest.TestCase):
    def test_chat_compare_self_shows_map(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        rt._control_wisdom("compare self", chat_key="telegram:1")
        text = "\n".join(rt.sent)
        self.assertIn("Advaita Vedanta", text)
        self.assertIn("Positions mapped, not averaged", text)

    def test_chat_compare_unknown_topic_unchanged(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        rt._control_wisdom("compare zxq-no-such-topic",
                           chat_key="telegram:1")
        text = "\n".join(rt.sent)
        self.assertIn("timeline context", text)
        self.assertNotIn("Positions mapped, not averaged", text)

    def test_chat_lineage(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        rt._control_wisdom("lineage Buddha", chat_key="telegram:1")
        text = "\n".join(rt.sent)
        self.assertIn("Buddha", text)
        self.assertIn("tradition_sequence", text)

    def test_chat_lineage_no_arg_lists_nodes(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        rt._control_wisdom("lineage", chat_key="telegram:1")
        text = "\n".join(rt.sent)
        self.assertIn("Siddhartha Gautama, the Buddha", text)

    def test_chat_practice_recall_flow(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        reply = rt._control_wisdom(
            "practice recall add What is ori? | pack: nigeria | Yoruba | "
            "Nigeria pack", chat_key="telegram:1")
        self.assertIn("What is ori?", reply)
        reply = rt._control_wisdom("practice recall due",
                                   chat_key="telegram:1")
        self.assertIn("What is ori?", reply)
        # The due list prints the item id; review it at quality 5.
        item_id = reply.split("`")[1]
        reply = rt._control_wisdom(f"practice recall review {item_id} 5",
                                   chat_key="telegram:1")
        self.assertIn("due in 1d", reply)
        reply = rt._control_wisdom("practice recall stats",
                                   chat_key="telegram:1")
        self.assertIn("1 item", reply)

    def test_chat_status_names_completeness(self):
        ctx, _ = _ctx()
        rt = _runtime(ctx)
        reply = rt._control_wisdom("status", chat_key="telegram:1")
        self.assertIn("completeness:", reply)


class CliTierTests(unittest.TestCase):
    def _args(self, task, json=False):
        return SimpleNamespace(task=task, json=json)

    def test_cli_compare_self_shows_map(self):
        ctx, _ = _ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_wisdom(self._args(["compare", "self"]), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Advaita Vedanta", buf.getvalue())

    def test_cli_lineage(self):
        ctx, _ = _ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_wisdom(self._args(["lineage", "Buddha"]), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Buddha", buf.getvalue())

    def test_cli_practice_recall_stats(self):
        ctx, _ = _ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_wisdom(
                self._args(["practice", "recall", "stats"]), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("0 item", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
