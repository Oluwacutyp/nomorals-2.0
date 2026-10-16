"""T4.1: LLM cassettes — record / replay through the router.

Offline with MockProvider. Each test gets a fresh tmp cassette dir, so
nothing touches ~/.nomorals/cassettes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.llm.cassette import (CassetteStore, canonical_key,
                                   response_from_record)
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


class _VisionMock(MockProvider):
    @property
    def capabilities(self):
        return {"chat", "complete", "embed", "vision"}

    def describe_image(self, image, prompt="", params=None, **kw):
        self.calls.append([image])
        return LLMResponse(text=f"saw {len(image)} bytes", model=self.model)


class CassetteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-cassette-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        # Pin the cassette env off unless the test sets it: recording is
        # default-on, replay is opt-in — tests set each explicitly.
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        os.environ.pop("NM_LLM_REPLAY", None)
        os.environ.pop("NM_LLM_CASSETTE", None)

    def _router(self, **kw):
        return LLMRouter(cassette_dir=str(self.tmp / "cassettes"), **kw)

    def _add(self, router, model="mock-7b", name="mock", cls=MockProvider):
        provider = cls(model=model)
        router.add(provider, name=name, primary=True)
        return provider

    def test_record_then_replay_skips_provider(self):
        r = self._router()
        provider = self._add(r)
        msgs = [Message.user("hello")]
        first = r.chat(msgs)
        self.assertTrue(first.ok)
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue((self.tmp / "cassettes" / "cassettes.jsonl").exists())
        # Second identical call with replay on: provider untouched.
        r2 = self._router(replay=True)
        provider2 = self._add(r2)
        second = r2.chat(msgs)
        self.assertEqual(second.text, first.text)
        self.assertEqual(len(provider2.calls), 0)
        self.assertTrue(second.raw.get("replayed_from_cassette"))
        self.assertEqual(second.route_trace, [])
        self.assertEqual(second.provider, first.provider)
        self.assertEqual(second.model, first.model)

    def test_replay_off_by_default_calls_provider_twice(self):
        r = self._router()
        provider = self._add(r)
        msgs = [Message.user("hello")]
        r.chat(msgs)
        r.chat(msgs)
        self.assertEqual(len(provider.calls), 2)

    def test_recording_disabled_writes_nothing(self):
        with mock.patch.dict(os.environ, {"NM_LLM_CASSETTE": "0"}):
            r = self._router()
            provider = self._add(r)
            r.chat([Message.user("hello")])
            self.assertEqual(len(provider.calls), 1)
            self.assertFalse((self.tmp / "cassettes").exists())

    def test_key_excludes_provider_and_model(self):
        # Record under provider A / model A; a router with ONLY provider
        # B / model B replays it — the key is provider-independent.
        ra = self._router()
        pa = self._add(ra, model="model-a", name="provider-a")
        first = ra.chat([Message.user("same prompt")])
        rb = self._router(replay=True)
        pb = self._add(rb, model="model-b", name="provider-b")
        second = rb.chat([Message.user("same prompt")])
        self.assertEqual(second.text, first.text)
        self.assertEqual(len(pb.calls), 0)

    def test_task_kind_is_part_of_key(self):
        r = self._router(replay=True)
        provider = self._add(r)
        r.chat([Message.user("same")], task_kind="code")
        r.chat([Message.user("same")], task_kind="chat")
        self.assertEqual(len(provider.calls), 2)

    def test_failures_are_not_recorded(self):
        provider = MockProvider(model="mock-7b", failure_rate=1.0)
        r = self._router()
        r.add(provider, name="mock", primary=True)
        resp = r.chat([Message.user("hello")])
        self.assertFalse(resp.ok)
        self.assertFalse((self.tmp / "cassettes" / "cassettes.jsonl").exists())

    def test_first_write_wins(self):
        store = CassetteStore(self.tmp / "cassettes")
        key = canonical_key("chat", "", [{"role": "user", "content": "x"}],
                            None)
        first = LLMResponse(text="one", model="m", provider="p")
        second = LLMResponse(text="two", model="m", provider="p")
        self.assertTrue(store.record(key, first))
        self.assertFalse(store.record(key, second))
        self.assertEqual(response_from_record(store.lookup(key)).text,
                         "one")

    def test_vision_key_uses_image_hash(self):
        # Record with replay off, then replay with a fresh router.
        r0 = self._router()
        p0 = self._add(r0, cls=_VisionMock)
        img = b"fake-image-bytes"
        r0.describe_image(img, "what is this?")
        self.assertEqual(len(p0.calls), 1)
        r = self._router(replay=True)
        provider = self._add(r, cls=_VisionMock)
        first = r.describe_image(img, "what is this?")
        self.assertEqual(len(provider.calls), 0)
        self.assertTrue(first.raw.get("replayed_from_cassette"))
        r.describe_image(b"other-bytes", "what is this?")
        self.assertEqual(len(provider.calls), 1)

    def test_complete_round_trip(self):
        r0 = self._router()
        p0 = self._add(r0)
        first = r0.complete("once upon a time")
        self.assertEqual(len(p0.calls), 1)
        r = self._router(replay=True)
        provider = self._add(r)
        second = r.complete("once upon a time")
        self.assertEqual(second.text, first.text)
        self.assertEqual(len(provider.calls), 0)

    def test_params_are_part_of_key(self):
        r = self._router(replay=True)
        provider = self._add(r)
        r.chat([Message.user("same")],
               params=SamplingParams(temperature=0.1))
        r.chat([Message.user("same")],
               params=SamplingParams(temperature=0.9))
        self.assertEqual(len(provider.calls), 2)

    def test_replay_mode_is_read_only(self):
        # Replay ON + fresh dir: the miss falls through to the provider
        # AND nothing is recorded — otherwise a second identical call
        # in the same run would replay the first call's recording.
        r = self._router(replay=True)
        provider = self._add(r)
        r.chat([Message.user("no record under replay")])
        r.chat([Message.user("no record under replay")])
        self.assertEqual(len(provider.calls), 2)
        self.assertFalse((self.tmp / "cassettes").exists())

    def test_replay_via_env(self):
        # Record with replay OFF (replay mode is read-only), then replay
        # with NM_LLM_REPLAY=1.
        r = self._router()
        provider = self._add(r)
        first = r.chat([Message.user("env replay")])
        self.assertEqual(len(provider.calls), 1)
        with mock.patch.dict(os.environ, {"NM_LLM_REPLAY": "1"}):
            r2 = self._router()
            provider2 = self._add(r2)
            second = r2.chat([Message.user("env replay")])
            self.assertEqual(second.text, first.text)
            self.assertEqual(len(provider2.calls), 0)
