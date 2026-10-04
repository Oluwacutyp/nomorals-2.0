"""R8: router failover paths (nomorals.llm.router).

Covers the previously untested critical failover behavior: embed failover,
vision capability gating, failover bookkeeping (degraded/failed_providers/
fallback_note), the all-cooling-down path, probe(), and repair hooks.
All offline with mock providers.
"""

from __future__ import annotations

import unittest

from nomorals.llm.base import Message
from nomorals.core.errors import ModelError, ProviderUnavailable
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


class _BoomChat(MockProvider):
    def chat(self, messages, params=None, **kw):
        raise RuntimeError("chat boom")


class _BoomEmbed(MockProvider):
    def embed(self, texts, **kw):
        raise RuntimeError("embed boom")


class _SickHealth(MockProvider):
    def health(self):
        raise RuntimeError("health check exploded")


def _router(**kw):
    return LLMRouter(cooldown_seconds=60.0, failure_threshold=2, **kw)


class EmbedFailoverTests(unittest.TestCase):
    def test_embed_failover_to_next_provider(self):
        r = _router()
        r.add(_BoomEmbed(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        vectors = r.embed(["hello"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), r.get("good").dimensions)

    def test_embed_no_capable_provider(self):
        r = _router()

        class _NoEmbed(MockProvider):
            @property
            def capabilities(self):
                return {"chat"}

        r.add(_NoEmbed(model="x"), name="x")
        with self.assertRaises(ModelError):
            r.embed(["hello"])

    def test_embed_all_fail(self):
        r = _router()
        r.add(_BoomEmbed(model="b1"), name="b1")
        r.add(_BoomEmbed(model="b2"), name="b2")
        with self.assertRaises(ProviderUnavailable):
            r.embed(["hello"])

    def test_embed_skips_cooling_down(self):
        r = _router()
        bad = _BoomEmbed(model="bad")
        r.add(bad, name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r.embed(["x"])  # bad fails (1), good serves
        r.embed(["x"])  # bad fails (2) → over threshold → 60s cooldown
        before = r._health["bad"].failures
        self.assertEqual(before, 2)
        # bad is cooling down: served straight by good, no new failure logged
        vectors = r.embed(["x"])
        self.assertEqual(len(vectors[0]), r.get("good").dimensions)
        self.assertEqual(r._health["bad"].failures, before)


class VisionGatingTests(unittest.TestCase):
    def test_describe_image_requires_vision_capability(self):
        r = _router()
        r.add(MockProvider(model="m"), name="m")  # mock has no "vision" cap
        with self.assertRaises(ModelError) as ctx:
            r.describe_image(b"bytes", "what is this")
        self.assertIn("vision", str(ctx.exception))


class FailoverBookkeepingTests(unittest.TestCase):
    def test_failover_marks_degraded_with_chain(self):
        r = _router()
        r.add(_BoomChat(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        resp = r.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertTrue(resp.degraded)
        self.assertEqual(resp.failed_providers, ["bad"])
        self.assertIn("bad failed", resp.fallback_note)
        self.assertIn("served by good", resp.fallback_note)
        self.assertEqual(r.stats["failovers"], 1)

    def test_all_cooling_down(self):
        r = _router()
        r.add(_BoomChat(model="a"), name="a")
        r.add(_BoomChat(model="b"), name="b")
        r.chat([Message.user("hi")])  # failure 1 (no cooldown yet)
        r.chat([Message.user("hi")])  # failure 2 → both cooling down
        resp = r.chat([Message.user("hi")])
        self.assertFalse(resp.ok)
        self.assertIn("cooling down", resp.error)

    def test_total_failure_reports_chain(self):
        r = _router()
        r.add(_BoomChat(model="a"), name="a", primary=True)
        resp = r.chat([Message.user("hi")])
        self.assertFalse(resp.ok)
        self.assertEqual(resp.failed_providers, ["a"])
        self.assertIn("no provider served this call", resp.fallback_note)
        self.assertEqual(r.stats["failures"], 1)


class ProbeTests(unittest.TestCase):
    def test_probe_reports_per_provider(self):
        r = _router()
        r.add(MockProvider(model="ok"), name="ok")
        r.add(_SickHealth(model="sick"), name="sick")
        self.assertEqual(r.probe(), {"ok": True, "sick": False})


class RepairHookTests(unittest.TestCase):
    def test_dict_repair_hook_called_on_failure(self):
        fired = []
        r = _router(repair_hooks={"fix": lambda: fired.append(1)})
        r.add(_BoomChat(model="a"), name="a")
        r.chat([Message.user("hi")])
        self.assertEqual(fired, [1])
        self.assertEqual(r.stats["repairs"], 1)
        # per-hook cooldown: a second immediate failure does not refire it
        r.chat([Message.user("hi")])
        self.assertEqual(fired, [1])

    def test_list_repair_hook_called(self):
        seen = []
        r = _router(repair_hooks=[lambda name, error: seen.append((name, error))])
        r.add(_BoomChat(model="a"), name="a")
        r.chat([Message.user("hi")])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "a")
        self.assertIn("boom", seen[0][1])

    def test_repair_hook_exception_swallowed(self):
        def _bad():
            raise RuntimeError("hook boom")

        r = _router(repair_hooks={"bad": _bad})
        r.add(_BoomChat(model="a"), name="a")
        r.chat([Message.user("hi")])  # hook raises — dispatch must survive
        r.chat([Message.user("hi")])
        self.assertEqual(r.stats["failures"], 2)


if __name__ == "__main__":
    unittest.main()
