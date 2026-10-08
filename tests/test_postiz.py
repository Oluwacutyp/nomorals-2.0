"""Build-map #42: Postiz publishing backend + per-platform tone.

All offline: HTTP is mocked at the HttpClient boundary, so the assertions
are about payload shape, retry behaviour, fail-closed config, per-channel
error surfacing, and tone adaptation — not about a live Postiz instance.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.social import SocialManager
from nomorals.social.adapters import postiz
from nomorals.social.base import Account, PostResult
from nomorals.social.tone import (
    ChannelSpec,
    Publisher,
    adapt_tone,
)


class FakeResponse:
    def __init__(self, status: int, body: str = "", headers: dict | None = None):
        self.status = status
        self.body = body.encode()
        self.headers = headers or {}
        self.url = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.body.decode()


class FakeClient:
    """Programmable HttpClient stand-in. Records every call."""

    def __init__(self, script: list[FakeResponse]):
        self.script = list(script)
        self.calls: list[tuple] = []

    def _next(self) -> FakeResponse:
        if not self.script:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self.script.pop(0)

    def post_json(self, url, payload, **kw):
        self.calls.append(("POST", url, payload))
        return self._next()

    def request(self, method, url, **kw):
        self.calls.append((method, url))
        return self._next()

    def post_multipart(self, url, files=None, **kw):
        self.calls.append(("MULTIPART", url, [f[1] for f in (files or [])]))
        return self._next()


def _adapter(**kw) -> postiz.Adapter:
    return postiz.Adapter(base_url="http://postiz.local:5000",
                          api_key="test-key", **kw)


def _account(**kw) -> Account:
    return Account(platform="postiz", handle="postiz-main",
                   credentials="literal:test-key", **kw)


class PostizPayloadTests(unittest.TestCase):
    def test_post_now_payload_shape(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p1"}))])
        with patch.object(postiz, "HttpClient", return_value=client):
            result = _adapter().post(
                _account(), "hello world", integration_ids=["int-1", "int-2"])
        self.assertTrue(result.ok, result.error)
        method, url, payload = client.calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/api/public/v1/posts"), url)
        self.assertEqual(payload["type"], "now")
        self.assertFalse(payload["shortLink"])
        self.assertEqual(payload["tags"], [])
        self.assertEqual(len(payload["posts"]), 2)
        first = payload["posts"][0]
        self.assertEqual(first["integration"], {"id": "int-1"})
        self.assertEqual(first["value"][0]["content"], "hello world")
        self.assertIn("group", first)
        # both posts share one group id
        self.assertEqual(payload["posts"][0]["group"], payload["posts"][1]["group"])
        self.assertEqual(result.external_id, "p1")

    def test_schedule_at_datetime_mapping(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p2"}))])
        when = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
        with patch.object(postiz, "HttpClient", return_value=client):
            result = _adapter().post(
                _account(), "scheduled!", integration_ids=["int-1"],
                schedule_at=when)
        self.assertTrue(result.ok, result.error)
        payload = client.calls[0][2]
        self.assertEqual(payload["type"], "schedule")
        self.assertEqual(payload["date"], "2026-10-20T12:00:00+00:00")

    def test_schedule_at_iso_string(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p3"}))])
        with patch.object(postiz, "HttpClient", return_value=client):
            _adapter().post(_account(), "x", integration_ids=["int-1"],
                            schedule_at="2026-11-01T09:30:00")
        payload = client.calls[0][2]
        self.assertEqual(payload["type"], "schedule")
        self.assertIn("2026-11-01T09:30:00", payload["date"])

    def test_settings_per_integration(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p4"}))])
        settings = {"yt-1": {"title": "My video", "type": "public"}}
        with patch.object(postiz, "HttpClient", return_value=client):
            _adapter().post(_account(), "video!", integration_ids=["yt-1"],
                            settings=settings)
        payload = client.calls[0][2]
        self.assertEqual(payload["posts"][0]["settings"],
                         {"title": "My video", "type": "public"})

    def test_integration_id_from_account_limits(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p5"}))])
        account = _account(limits={"integration_id": "lim-9"})
        with patch.object(postiz, "HttpClient", return_value=client):
            result = _adapter().post(account, "via limits")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(client.calls[0][2]["posts"][0]["integration"],
                         {"id": "lim-9"})


class PostizFailureTests(unittest.TestCase):
    def test_missing_config_fails_closed(self):
        adapter = postiz.Adapter()  # no env in test
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POSTIZ_URL", None)
            os.environ.pop("POSTIZ_API_KEY", None)
            result = adapter.post(Account(platform="postiz", handle="h"),
                                  "never sent")
        self.assertFalse(result.ok)
        self.assertIn("not configured", result.error)
        self.assertIn("Nothing was posted", result.error)

    def test_missing_integration_fails_closed(self):
        client = FakeClient([])
        with patch.object(postiz, "HttpClient", return_value=client):
            result = _adapter().post(_account(), "no channel")
        self.assertFalse(result.ok)
        self.assertIn("no Postiz integration", result.error)
        self.assertEqual(client.calls, [])  # no HTTP at all

    def test_retry_on_429_then_success(self):
        client = FakeClient([
            FakeResponse(429, "slow down"),
            FakeResponse(200, json.dumps({"id": "p6"})),
        ])
        with patch.object(postiz, "HttpClient", return_value=client), \
             patch.object(postiz.time, "sleep", return_value=None):
            result = _adapter().post(_account(), "retry me",
                                     integration_ids=["int-1"])
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len([c for c in client.calls if c[0] == "POST"]), 2)

    def test_gives_up_after_max_attempts(self):
        client = FakeClient([FakeResponse(503, "down")] * 3)
        with patch.object(postiz, "HttpClient", return_value=client), \
             patch.object(postiz.time, "sleep", return_value=None):
            result = _adapter(max_attempts=3).post(
                _account(), "unlucky", integration_ids=["int-1"])
        self.assertFalse(result.ok)
        self.assertIn("503", result.error)
        self.assertEqual(result.status_code, 503)

    def test_per_channel_errors_surfaced(self):
        body = json.dumps({
            "id": "p7",
            "posts": [
                {"integration": {"id": "ok-1"}},
                {"integration": {"id": "bad-1"}, "error": "token revoked"},
            ],
        })
        client = FakeClient([FakeResponse(200, body)])
        with patch.object(postiz, "HttpClient", return_value=client):
            result = _adapter().post(
                _account(), "mixed", integration_ids=["ok-1", "bad-1"])
        self.assertFalse(result.ok)
        self.assertIn("token revoked", result.error)
        channels = {c["integration_id"]: c
                    for c in result.metrics["channels"]}
        self.assertTrue(channels["ok-1"]["ok"])
        self.assertFalse(channels["bad-1"]["ok"])


class PostizMediaTests(unittest.TestCase):
    def test_upload_before_post_two_step(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"\x89PNG fake")
        tmp.close()
        try:
            client = FakeClient([
                FakeResponse(200, json.dumps({"id": "f1", "path": "/u/f1.png"})),
                FakeResponse(200, json.dumps({"id": "p8"})),
            ])
            with patch.object(postiz, "HttpClient", return_value=client):
                result = _adapter().post(
                    _account(), "with media", integration_ids=["int-1"],
                    media_paths=[tmp.name])
            self.assertTrue(result.ok, result.error)
            self.assertEqual(client.calls[0][0], "MULTIPART")
            self.assertTrue(client.calls[0][1].endswith("/api/public/v1/upload"))
            payload = client.calls[1][2]
            images = payload["posts"][0]["value"][0]["image"]
            self.assertEqual(images, [{"id": "f1", "path": "/u/f1.png"}])
        finally:
            os.unlink(tmp.name)


class ToneTests(unittest.TestCase):
    def test_x_trim_rule_fallback(self):
        long_text = "word " * 100
        out = adapt_tone(long_text, "x")
        self.assertLessEqual(len(out), 280)
        self.assertTrue(out.endswith("…"))

    def test_linkedin_strips_hashtags(self):
        out = adapt_tone("Great launch day! #startup #hiring", "linkedin")
        self.assertNotIn("#", out)
        self.assertIn("Great launch day!", out)

    def test_threads_keeps_hashtags(self):
        out = adapt_tone("shipping today #buildinpublic", "threads")
        self.assertIn("#buildinpublic", out)

    def test_empty_passthrough(self):
        self.assertEqual(adapt_tone("", "x"), "")
        self.assertEqual(adapt_tone("   ", "linkedin"), "")

    def test_llm_fn_used_when_given(self):
        out = adapt_tone("hello", "linkedin",
                         llm_fn=lambda p: "FORMAL: hello")
        self.assertEqual(out, "FORMAL: hello")

    def test_llm_failure_falls_back_to_rules(self):
        def boom(prompt):
            raise RuntimeError("model down")
        out = adapt_tone("hello #tag", "linkedin", llm_fn=boom)
        self.assertNotIn("#", out)  # rule fallback applied


class PublisherTests(unittest.TestCase):
    def test_publish_adapted_one_call_per_channel_variants(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p9"}))])
        channels = [ChannelSpec("li-1", "linkedin"), ChannelSpec("x-1", "x")]
        with patch.object(postiz, "HttpClient", return_value=client):
            pub = Publisher(_adapter(), _account())
            results = pub.publish_adapted(
                "Big news! #launch", channels,
                llm_fn=lambda p: "REWRITTEN")
        self.assertEqual(set(results), {"li-1", "x-1"})
        payload = client.calls[0][2]
        variants = {p["integration"]["id"]: p["value"][0]["content"]
                    for p in payload["posts"]}
        self.assertEqual(variants["li-1"], "REWRITTEN")
        self.assertEqual(variants["x-1"], "REWRITTEN")
        # per-channel results carry the adapted text
        self.assertEqual(results["li-1"].metrics["adapted_text"], "REWRITTEN")
        self.assertEqual(results["li-1"].platform, "postiz:linkedin")

    def test_publish_adapted_empty_channels_rejected(self):
        pub = Publisher(_adapter(), _account())
        with self.assertRaises(ValueError):
            pub.publish_adapted("x", [])


class ManagerWiringTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def _manager(self):
        ctx = build_context(Settings(home=self.home))
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        return SocialManager(ctx)

    def test_register_postiz_when_env_set(self):
        with patch.dict(os.environ, {"POSTIZ_URL": "http://p.local:5000",
                                     "POSTIZ_API_KEY": "k"}):
            mgr = self._manager().register_builtins()
        self.assertIn("postiz", mgr.adapters)

    def test_no_postiz_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POSTIZ_URL", None)
            mgr = self._manager().register_builtins()
        self.assertNotIn("postiz", mgr.adapters)
        self.assertIn("mastodon", mgr.adapters)  # existing behaviour kept

    def test_publish_through_postiz_account(self):
        client = FakeClient([FakeResponse(200, json.dumps({"id": "p10"}))])
        with patch.dict(os.environ, {"POSTIZ_URL": "http://p.local:5000"}), \
             patch.object(postiz, "HttpClient", return_value=client):
            mgr = self._manager().register_builtins()
            mgr.connect("postiz", "main", credentials="env:POSTIZ_API_KEY",
                        limits={"integration_id": "int-1"})
            with patch.dict(os.environ, {"POSTIZ_API_KEY": "k"}):
                outcome = mgr.publish("hello via postiz", platforms=["postiz"],
                                      parallel=False)
        self.assertTrue(outcome.posted, [r.error for r in outcome.results])


if __name__ == "__main__":
    unittest.main()
