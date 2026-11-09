"""WS3 P3 — platform truth: his real surfaces as draft platforms.

Pins:
(a) the LinkedIn adapter is registered by register_builtins and posts
    through the native-connector seam with the connector's real call
    shape (post(text, confirmed=True)); it does NOT override metrics
    (the connector exposes no metrics operation);
(b) the X adapter's paid-tier refusal shape — no token and a 403 both
    read as tier/connection refusals with the manual step, never a
    generic error, never retryable; X metrics ARE real (public_metrics);
(c) the weekly pipeline's default platform set is exactly
    (linkedin, x, telegram, whatsapp-status), linkedin first, and the
    fan-out applies adapt_tone per platform — one queued draft per
    platform per text, carrying that platform's own text;
(d) WA-status / TG-channel draft posting routes through the
    personal-surface seam and refuses IN WORDS when there is no live
    surface, no paired account, or the owner's standing stop;
(e) metrics-capability truth per adapter: the metrics() override is
    present iff genuinely implemented.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from nomorals.social import content_pipeline
from nomorals.social.adapters.linkedin import LinkedInAdapter
from nomorals.social.adapters.personal import (
    TelegramChannelAdapter,
    WhatsAppStatusAdapter,
)
from nomorals.social.adapters.x import XAdapter
from nomorals.social.base import ERROR_AUTH, Account, PlatformAdapter
from nomorals.social.content_pipeline import DEFAULT_PLATFORMS, weekly_content
from nomorals.social.drafts import DraftQueue
from nomorals.social.manager import SocialManager
from nomorals.social.pairing import (
    read_registry_snapshot,
    registry_path_for,
    select_account,
)
from nomorals.storage.db import Database


def _manager(tmp_path):
    db = Database(":memory:")
    ctx = SimpleNamespace(
        db=db, tools=None, extras={},
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    return SocialManager(ctx).register_builtins()


# ── (a) LinkedIn ─────────────────────────────────────────────────────


class FakeConnector:
    """The connector seam, faked at the builder: records the exact
    call shape the adapter drives."""

    def __init__(self, result=None, exc=None):
        self.calls: list = []
        self._result = result if result is not None else {"id": "urn:li:share:777"}
        self._exc = exc

    def post(self, text, *, confirmed=False):
        self.calls.append((text, confirmed))
        if self._exc is not None:
            raise self._exc
        return self._result

    def test_connection(self):
        return True


class FakeConnectorError(Exception):
    def __init__(self, message, status_code=0):
        super().__init__(message)
        self.status_code = status_code


def test_linkedin_registered_in_builtins(tmp_path):
    social = _manager(tmp_path)
    adapter = social.adapters["linkedin"]
    assert isinstance(adapter, LinkedInAdapter)
    assert adapter.max_chars == 3000


def test_linkedin_posts_through_connector_with_real_call_shape():
    fake = FakeConnector()
    adapter = LinkedInAdapter(connector_builder=lambda: fake)
    account = Account(platform="linkedin", handle="pelumi")
    result = adapter.post(account, "  Shipping the weekly build notes.  ")
    assert result.ok
    # The connector's real call shape: stripped text, confirmed=True
    # (the #43 attestation — authorization happened upstream).
    assert fake.calls == [("Shipping the weekly build notes.", True)]
    assert result.external_id == "urn:li:share:777"
    assert result.url == ("https://www.linkedin.com/feed/update/"
                          "urn:li:share:777/")


def test_linkedin_not_connected_words_pass_through():
    fake = FakeConnector(exc=FakeConnectorError(
        "linkedin is not connected — run "
        "`nm connectors connect --name linkedin` first"))
    adapter = LinkedInAdapter(connector_builder=lambda: fake)
    result = adapter.post(Account(platform="linkedin", handle="p"), "hi")
    assert not result.ok
    assert result.error_code == ERROR_AUTH
    assert "not connected" in result.error


def test_linkedin_api_error_keeps_status_and_words():
    fake = FakeConnector(exc=FakeConnectorError(
        "linkedin token rejected (HTTP 401) — reconnect", status_code=401))
    adapter = LinkedInAdapter(connector_builder=lambda: fake)
    result = adapter.post(Account(platform="linkedin", handle="p"), "hi")
    assert not result.ok
    assert result.status_code == 401
    assert result.error_code == ERROR_AUTH
    assert "reconnect" in result.error


def test_linkedin_unwired_refuses_in_words():
    adapter = LinkedInAdapter(connector_builder=None)
    result = adapter.post(Account(platform="linkedin", handle="p"), "hi")
    assert not result.ok
    assert "not wired in this process" in result.error
    assert not result.retryable


def test_linkedin_metrics_not_overridden():
    # The connector exposes no metrics operation (no engagement call,
    # no r_member_social scope) — overriding metrics() would make the
    # P1 spine believe numbers exist. It must stay the base default.
    assert type(LinkedInAdapter()).metrics is PlatformAdapter.metrics


# ── (b) X ────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    @property
    def ok(self):
        return 200 <= self.status < 300

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)


def _x_account():
    return Account(platform="x", handle="@peace",
                   credentials="env:X_ACCESS_TOKEN")


def test_x_no_token_is_a_tier_refusal_not_a_generic_error(monkeypatch):
    monkeypatch.delenv("X_ACCESS_TOKEN", raising=False)
    result = XAdapter().post(_x_account(), "hello")
    assert not result.ok
    assert result.error_code == ERROR_AUTH
    assert not result.retryable
    assert "no X user access token" in result.error
    assert "Manual step" in result.error
    assert "PAID tier" in result.error


def test_x_posts_text_with_user_token(monkeypatch):
    monkeypatch.setenv("X_ACCESS_TOKEN", "tok-123")
    http = FakeHttp(FakeResponse(201, {"data": {"id": "42", "text": "hi"}}))
    result = XAdapter(http=http).post(_x_account(), "hi")
    assert result.ok
    assert result.external_id == "42"
    assert result.url == "https://x.com/peace/status/42"
    method, url, kwargs = http.calls[0]
    assert method == "POST" and url == "https://api.x.com/2/tweets"
    assert kwargs["headers"]["Authorization"] == "Bearer tok-123"
    assert kwargs["json"] == {"text": "hi"}


def test_x_403_carries_the_free_tier_hint(monkeypatch):
    monkeypatch.setenv("X_ACCESS_TOKEN", "tok-123")
    http = FakeHttp(FakeResponse(403, {"detail": "Forbidden"}))
    result = XAdapter(http=http).post(_x_account(), "hi")
    assert not result.ok
    assert "free tier" in result.error
    assert "Basic/Pro" in result.error
    assert result.status_code == 403


def test_x_metrics_maps_public_metrics(monkeypatch):
    monkeypatch.setenv("X_ACCESS_TOKEN", "tok-123")
    http = FakeHttp(FakeResponse(200, {"data": {"public_metrics": {
        "like_count": 5, "reply_count": 2, "retweet_count": 3,
        "quote_count": 1, "impression_count": 99}}}))
    out = XAdapter(http=http).metrics(_x_account(), "42")
    assert out == {"likes": 5, "replies": 2, "reposts": 3,
                   "quotes": 1, "impressions": 99}
    assert type(XAdapter()).metrics is not PlatformAdapter.metrics


def test_x_metrics_best_effort_empty(monkeypatch):
    monkeypatch.delenv("X_ACCESS_TOKEN", raising=False)
    assert XAdapter().metrics(_x_account(), "42") == {}


# ── (c) weekly pipeline defaults + fan-out ───────────────────────────


def test_weekly_default_platform_set_exact():
    assert DEFAULT_PLATFORMS == ("linkedin", "x", "telegram",
                                 "whatsapp-status")
    default = inspect.signature(weekly_content).parameters["platforms"].default
    assert tuple(default) == DEFAULT_PLATFORMS
    assert default[0] == "linkedin"


_LONG_SOURCE = (
    "#gold #xauusd #trading " + "Gold held the level and pushed "
    "through the session high while the dollar softened and buyers "
    "defended every dip into the zone with real size behind it " * 3)


def test_weekly_fanout_applies_adapt_tone_per_platform(tmp_path,
                                                       monkeypatch):
    # Controlled source texts through the pipeline's own drafting
    # seam; llm_fn=None so adapt_tone takes its deterministic
    # rule-based path (per-profile hashtag strip + hard trim).
    texts = [_LONG_SOURCE] + [
        f"Short note {i} about the desk." for i in range(4)]
    monkeypatch.setattr(content_pipeline, "_generate_drafts",
                        lambda *a, **k: list(texts))
    queue = DraftQueue(db_path=str(tmp_path / "drafts.db"))
    result = weekly_content("gold trading", queue=queue, findings=[])
    assert len(result.drafts) == 5
    first = next(d for d in result.drafts
                 if d.content.startswith("#gold"))
    # One source text → per-platform texts + one queued draft each.
    assert set(first.platform_texts) == set(DEFAULT_PLATFORMS)
    assert set(first.draft_ids) == set(DEFAULT_PLATFORMS)
    assert first.draft_id == first.draft_ids["linkedin"]
    # adapt_tone really shaped each surface's text:
    assert len(first.platform_texts["x"]) <= 280
    assert "#" not in first.platform_texts["linkedin"]
    assert len(first.platform_texts["whatsapp-status"]) <= 700
    # The queued drafts carry exactly those texts, one platform each.
    for platform, draft_id in first.draft_ids.items():
        queued = queue.get(draft_id)
        assert queued is not None
        assert queued.platforms == [platform]
        assert queued.content == first.platform_texts[platform]
        assert queued.metadata["platform"] == platform
        assert queued.metadata["fanout_group"].startswith("ws3fan:")
    # Every source draft fanned out the same way.
    for d in result.drafts:
        assert set(d.draft_ids) == set(DEFAULT_PLATFORMS)


# ── (d) personal surfaces ────────────────────────────────────────────


class FakeChatAdapter:
    def __init__(self):
        self.status_calls: list = []
        self.send_calls: list = []

    def send_status(self, text="", *, media_path="", caption=""):
        self.status_calls.append((text, media_path, caption))
        return {"ok": True, "message_id": "wa-msg-1"}

    def send(self, chat, text, **kwargs):
        self.send_calls.append((chat, text))
        return SimpleNamespace(ok=True, message_id="99", error="")


class FakeRec:
    id = "acct-1"
    platform = "whatsapp"


class FakeSurface:
    def __init__(self, chat=None, rec=None, stopped=False, error=""):
        self._chat = chat
        self._rec = rec
        self._stopped = stopped
        self._error = error

    def chat_adapter(self, platform):
        return self._chat

    def registry_error(self):
        return self._error

    def paired_account(self, platform):
        return self._rec

    def owner_stopped(self, platform):
        return self._stopped


def _wa_account():
    return Account(platform="whatsapp-status", handle="owner")


def _tg_account(handle="@mychannel"):
    return Account(platform="telegram", handle=handle)


def test_whatsapp_status_posts_through_the_surface_seam():
    chat = FakeChatAdapter()
    adapter = WhatsAppStatusAdapter(
        surface=FakeSurface(chat=chat, rec=FakeRec()))
    result = adapter.post(_wa_account(), "Desk update: gold held.")
    assert result.ok
    assert result.external_id == "wa-msg-1"
    assert chat.status_calls == [("Desk update: gold held.", "", "")]


def test_whatsapp_status_refuses_when_no_live_surface():
    adapter = WhatsAppStatusAdapter(
        surface=FakeSurface(chat=None, rec=FakeRec()))
    result = adapter.post(_wa_account(), "hi")
    assert not result.ok
    assert "no live whatsapp surface in this process" in result.error


def test_whatsapp_status_refuses_when_unpaired():
    adapter = WhatsAppStatusAdapter(
        surface=FakeSurface(chat=FakeChatAdapter(), rec=None))
    result = adapter.post(_wa_account(), "hi")
    assert not result.ok
    assert "no paired whatsapp account" in result.error
    assert "/pair one first" in result.error


def test_whatsapp_status_refuses_when_owner_stopped():
    chat = FakeChatAdapter()
    adapter = WhatsAppStatusAdapter(
        surface=FakeSurface(chat=chat, rec=FakeRec(), stopped=True))
    result = adapter.post(_wa_account(), "hi")
    assert not result.ok
    assert "standing stop" in result.error
    assert chat.status_calls == []  # the stop governs: nothing posted


def test_whatsapp_status_refuses_when_registry_unreadable():
    adapter = WhatsAppStatusAdapter(surface=FakeSurface(
        chat=FakeChatAdapter(), rec=FakeRec(),
        error="the paired-account registry could not be read (bad json)"))
    result = adapter.post(_wa_account(), "hi")
    assert not result.ok
    assert "cannot be verified" in result.error


def test_telegram_channel_posts_through_the_surface_seam():
    chat = FakeChatAdapter()
    adapter = TelegramChannelAdapter(
        surface=FakeSurface(chat=chat, rec=FakeRec()))
    result = adapter.post(_tg_account(), "Channel post: week ahead.")
    assert result.ok
    assert result.external_id == "99"
    assert result.url == "https://t.me/mychannel/99"
    ref, text = chat.send_calls[0]
    assert ref.chat_id == "@mychannel" and ref.kind == "channel"
    assert text == "Channel post: week ahead."


def test_telegram_refuses_without_a_channel_target():
    adapter = TelegramChannelAdapter(
        surface=FakeSurface(chat=FakeChatAdapter(), rec=FakeRec()))
    result = adapter.post(_tg_account(handle="peace"), "hi")
    assert not result.ok
    assert "names no channel" in result.error


def test_telegram_refuses_when_owner_stopped():
    chat = FakeChatAdapter()
    adapter = TelegramChannelAdapter(
        surface=FakeSurface(chat=chat, rec=FakeRec(), stopped=True))
    result = adapter.post(_tg_account(), "hi")
    assert not result.ok
    assert "standing stop" in result.error
    assert chat.send_calls == []


def test_registry_snapshot_reads_pairing_truth(tmp_path):
    path = registry_path_for(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "accounts": [
            {"id": "wa-1", "platform": "whatsapp",
             "identity": "2348000000000", "status": "linked",
             "created_at": 2.0},
            {"id": "wa-0", "platform": "whatsapp",
             "identity": "2348111111111", "status": "registered",
             "created_at": 1.0},
        ],
        "owner_stopped": {"whatsapp": 1700000000.0},
    }), encoding="utf-8")
    snap = read_registry_snapshot(path)
    assert snap.error == ""
    assert "whatsapp" in snap.owner_stopped
    # Selection is the shared routing order: linked beats earliest.
    assert select_account(snap.records("whatsapp")).id == "wa-1"


def test_registry_snapshot_missing_file_is_empty_not_error(tmp_path):
    snap = read_registry_snapshot(registry_path_for(tmp_path))
    assert snap.error == ""
    assert snap.accounts == ()


def test_registry_snapshot_corrupt_file_fails_closed(tmp_path):
    path = registry_path_for(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    snap = read_registry_snapshot(path)
    assert snap.error != ""
    assert snap.accounts == ()


# ── (e) metrics-capability truth table ───────────────────────────────


def test_metrics_capability_truth_per_adapter(tmp_path):
    social = _manager(tmp_path)
    expected = {
        "mastodon": True,        # real favourites/reblogs read
        "bluesky": False,        # no override — spine names it unavailable
        "linkedin": False,       # connector exposes no metrics operation
        "x": True,               # real public_metrics read
        "whatsapp-status": False,
        "telegram": False,
    }
    for name, overridden in expected.items():
        adapter = social.adapters[name]
        assert (type(adapter).metrics is not PlatformAdapter.metrics) \
            is overridden, name
    for name in ("linkedin", "x", "whatsapp-status", "telegram"):
        assert name in social.adapters
