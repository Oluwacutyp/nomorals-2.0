"""Tests for nomorals.media.contentops.publish.

All HTTP is mocked — no network, no real credentials. The YouTube
resumable flow is driven through init → chunk → finalize with scripted
responses, including the resume-from-offset path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.youtube import YouTubeError
from nomorals.media.contentops.publish import (
    CapabilityUnavailable,
    MetaPublisher,
    PublishLedger,
    TikTokPublisher,
    XPublisher,
    YouTubePublisher,
    adapt_description,
    adapt_title,
    format_hashtags,
)
from nomorals.storage.db import Database


# ── fakes ──────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return json.dumps(self._payload) if self._payload is not None else ""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted HTTP stand-in. Routes match (method, url substring);
    values may be FakeResponse or a queue of them, or a callable."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: list[tuple[str, str, Any]] = []

    def route(self, method: str, path: str, response: Any) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append({"method": method.upper(), "url": url, **kw})
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                if callable(resp):
                    return resp(method.upper(), url, kw)
                if isinstance(resp, list):
                    if not resp:
                        raise AssertionError(
                            f"route queue exhausted for {method} {rp}")
                    return resp.pop(0)
                return resp
        raise AssertionError(f"no mock route for {method} {url}")

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, **kw)

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch("GET", url, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload=payload, **kw)


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _yt(http: FakeHttp | None = None) -> tuple[YouTubePublisher, FakeHttp]:
    http = http or FakeHttp()
    pub = YouTubePublisher(_vault(), http=http)
    # Vault tokens with a far-future access token: no refresh is attempted.
    pub._store_google_tokens(
        "chan", "cid",
        {"refresh_token": "rt", "access_token": "at", "expires_in": 99999},
    )
    return pub, http


def _video_file(tmp_path: Path, size: int = 25) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(bytes(i % 256 for i in range(size)))
    assert p.stat().st_size == size
    return p


# ── platform adaptation ────────────────────────────────────────────────

def test_adapt_title_youtube_truncates_to_100():
    long = "x" * 150
    out = adapt_title("youtube", long)
    assert len(out) == 100
    assert out.endswith("…")
    assert adapt_title("youtube", "short") == "short"


def test_adapt_description_x_hard_limit_280():
    out = adapt_description("x", "hello world " * 30, ["Tag1", "tag2"])
    assert len(out) <= 280
    assert "#Tag1" in out and "#tag2" in out


def test_format_hashtags_dedupes_and_strips():
    assert format_hashtags(["AI", "#ai", "music!", ""]) == "#AI #music"


def test_adapt_description_instagram_hashtag_block():
    out = adapt_description("instagram", "my reel", ["reels", "ai"])
    assert out.startswith("my reel\n.\n.\n.\n")
    assert "#reels #ai" in out


# ── YouTube resumable upload ───────────────────────────────────────────

def _route_yt_upload(http: FakeHttp, *, video_id: str = "vid123") -> None:
    http.route(
        "POST", "/upload/youtube/v3/videos?uploadType=resumable",
        FakeResponse(200, {}, headers={
            "Location": "https://www.googleapis.com/upload/session/abc"}),
    )
    http.route(
        "PUT", "/upload/session/abc",
        [
            FakeResponse(308, None),
            FakeResponse(308, None),
            FakeResponse(200, {"id": video_id,
                               "snippet": {"title": "t"}}),
        ],
    )


def test_youtube_resumable_flow_init_chunk_finalize(tmp_path):
    pub, http = _yt()
    path = _video_file(tmp_path, 25)
    _route_yt_upload(http)
    result = pub.publish(path, "my title", confirmed=True, chunk_size=10)
    assert result["video_id"] == "vid123"
    assert result["quota_total"] == 1600
    assert result["url"] == "https://www.youtube.com/watch?v=vid123"

    puts = [c for c in http.calls if c["method"] == "PUT"]
    assert len(puts) == 3
    ranges = [c["headers"]["Content-Range"] for c in puts]
    assert ranges == ["bytes 0-9/25", "bytes 10-19/25", "bytes 20-24/25"]

    init = next(c for c in http.calls if "uploadType=resumable" in c["url"])
    assert init["headers"]["X-Upload-Content-Length"] == "25"
    assert init["headers"]["X-Upload-Content-Type"] == "video/mp4"
    meta = json.loads(init["data"])
    assert meta["snippet"]["title"] == "my title"
    assert meta["status"]["privacyStatus"] == "private"


def test_youtube_publish_scheduling_requires_private(tmp_path):
    pub, _ = _yt()
    path = _video_file(tmp_path)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    with pytest.raises(YouTubeError, match="requires privacy='private'"):
        pub.publish(path, "t", privacy="public", publish_at=future,
                    confirmed=True)


def test_youtube_publish_at_sets_publish_at_in_metadata(tmp_path):
    pub, http = _yt()
    path = _video_file(tmp_path, 25)
    _route_yt_upload(http)
    future = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    result = pub.publish(path, "sched", publish_at=future, confirmed=True,
                         chunk_size=10)
    assert result["publish_at"] == "2030-01-02T03:04:05Z"
    assert result["privacy"] == "private"
    init = next(c for c in http.calls if "uploadType=resumable" in c["url"])
    meta = json.loads(init["data"])
    assert meta["status"]["publishAt"] == "2030-01-02T03:04:05Z"


def test_youtube_resume_after_transport_failure(tmp_path):
    pub, http = _yt()
    path = _video_file(tmp_path, 25)
    http.route(
        "POST", "/upload/youtube/v3/videos?uploadType=resumable",
        FakeResponse(200, {}, headers={
            "Location": "https://www.googleapis.com/upload/session/abc"}),
    )

    calls = {"n": 0}

    def put_handler(method, url, kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("dropped")
        headers = kw.get("headers", {})
        if headers.get("Content-Range") == "bytes */25":
            return FakeResponse(308, None,
                                headers={"Range": "bytes=0-9"})
        # resumed chunk then final chunk
        return (FakeResponse(308, None) if calls["n"] < 4
                else FakeResponse(200, {"id": "vid9"}))

    http.route("PUT", "/upload/session/abc", put_handler)
    video = pub.upload_chunks(
        "https://www.googleapis.com/upload/session/abc",
        path, 25, "video/mp4", chunk_size=10)
    assert video["id"] == "vid9"


def test_youtube_thumbnail_and_playlist_quota(tmp_path):
    pub, http = _yt()
    path = _video_file(tmp_path, 25)
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8fakejpeg")
    _route_yt_upload(http, video_id="vid7")
    http.route("POST", "/youtube/v3/thumbnails/set",
               FakeResponse(200, {"items": [{"id": "vid7"}]}))
    http.route("POST", "/youtube/v3/playlistItems",
               FakeResponse(200, {"id": "pli1"}))
    result = pub.publish(path, "t", confirmed=True, chunk_size=10,
                         thumbnail_path=thumb, playlist_id="PL123")
    assert result["quota_total"] == 1600 + 50 + 50
    assert result["quota_spent"]["thumbnails.set"] == 50
    assert result["quota_spent"]["playlistItems.insert"] == 50
    assert result["thumbnail_set"] is True
    assert result["playlist_id"] == "PL123"


def test_youtube_publish_writes_ledger(tmp_path):
    pub, http = _yt()
    path = _video_file(tmp_path, 25)
    _route_yt_upload(http)
    ledger = PublishLedger(tmp_path / "ledger.json")
    result = pub.publish(path, "ledger me", confirmed=True, chunk_size=10,
                         ledger=ledger)
    entry = ledger.get(result["ledger_id"])
    assert entry["platform"] == "youtube"
    assert entry["platform_video_id"] == "vid123"
    assert entry["quota_spent"] == 1600
    assert entry["status"] == "uploaded"


def test_youtube_is_shorts_candidate():
    assert YouTubePublisher.is_shorts_candidate(30, 1080, 1920) is True
    assert YouTubePublisher.is_shorts_candidate(61, 1080, 1920) is False
    assert YouTubePublisher.is_shorts_candidate(30, 1920, 1080) is False


def test_youtube_publish_refuses_without_confirmation(tmp_path):
    from nomorals.connectors.base import ConnectorError
    pub, _ = _yt()
    path = _video_file(tmp_path)
    with pytest.raises(ConnectorError, match="confirmation"):
        pub.publish(path, "t")


# ── ledger ─────────────────────────────────────────────────────────────

def test_ledger_round_trip_and_summary(tmp_path):
    ledger = PublishLedger(tmp_path / "ledger.json")
    e1 = ledger.record(platform="youtube", file="a.mp4", title="A",
                       platform_video_id="v1", quota_spent=1600)
    ledger.record(platform="tiktok", file="b.mp4", title="B",
                  status="processing")
    ledger.update(e1["id"], status="uploaded", notes="ok")
    assert ledger.get(e1["id"])["notes"] == "ok"

    again = PublishLedger(tmp_path / "ledger.json")
    assert again.summary() == {
        "total": 2,
        "by_platform": {"youtube": 1, "tiktok": 1},
        "by_status": {"uploaded": 1, "processing": 1},
        "quota_spent_total": 1600,
    }
    assert again.list(platform="youtube")[0]["id"] == e1["id"]


def test_ledger_rejects_bad_status(tmp_path):
    from nomorals.media.contentops.publish.ledger import PublishLedgerError
    ledger = PublishLedger(tmp_path / "ledger.json")
    with pytest.raises(PublishLedgerError, match="invalid status"):
        ledger.record(platform="x", file="a.mp4", title="A",
                      status="teleported")


def test_ledger_refuses_corrupt_file(tmp_path):
    from nomorals.media.contentops.publish.ledger import PublishLedgerError
    bad = tmp_path / "ledger.json"
    bad.write_text("{not json")
    with pytest.raises(PublishLedgerError, match="unreadable"):
        PublishLedger(bad)


# ── capability-gated publishers ──────────────────────────────────────

def test_tiktok_without_keys_raises_honest_error(tmp_path, monkeypatch):
    for var in ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET",
                "TIKTOK_ACCESS_TOKEN", "TIKTOK_REFRESH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    pub = TikTokPublisher(http=FakeHttp())
    path = _video_file(tmp_path)
    with pytest.raises(CapabilityUnavailable) as exc:
        pub.post_video(path, caption="hi")
    assert "developers.tiktok.com" in exc.value.manual_step
    assert "audit" in exc.value.manual_step.lower()
    # no HTTP was attempted
    assert pub.http.calls == []


def test_tiktok_unaudited_error_surfaces_audit_step(tmp_path, monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    http = FakeHttp()
    http.route("POST", "/post/publish/creator_info/query/",
               FakeResponse(200, {"error": {"code": "ok", "message": ""},
                                  "data": {"privacy_level_options":
                                           ["SELF_ONLY"]}}))
    http.route(
        "POST", "/post/publish/video/init/",
        FakeResponse(200, {
            "error": {"code": "unaudited_client_can_only_post_to_private_accounts",
                      "message": "not audited"},
            "data": {}}))
    pub = TikTokPublisher(http=http)
    path = _video_file(tmp_path)
    with pytest.raises(CapabilityUnavailable) as exc:
        pub.post_video(path, caption="hi")
    assert "audit" in exc.value.manual_step.lower()


def test_tiktok_full_direct_post_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    http = FakeHttp()
    http.route("POST", "/post/publish/creator_info/query/",
               FakeResponse(200, {"error": {"code": "ok", "message": ""},
                                  "data": {"privacy_level_options":
                                           ["PUBLIC_TO_EVERYONE",
                                            "SELF_ONLY"]}}))
    http.route("POST", "/post/publish/video/init/",
               FakeResponse(200, {"error": {"code": "ok", "message": ""},
                                  "data": {"post_id": "p1",
                                           "upload_url":
                                           "https://upload.tiktok/u1"}}))
    http.route("PUT", "https://upload.tiktok/u1",
               [FakeResponse(200, {}), FakeResponse(200, {}),
                FakeResponse(200, {})])
    http.route("POST", "/post/publish/video/publish/",
               FakeResponse(200, {"error": {"code": "ok", "message": ""},
                                  "data": {"publish_id": "pub9"}}))
    pub = TikTokPublisher(http=http)
    path = _video_file(tmp_path, 25)
    ledger = PublishLedger(tmp_path / "ledger.json")
    result = pub.post_video(path, caption="hello", tags=["ai"],
                            chunk_size=10, ledger=ledger)
    assert result["post_id"] == "p1"
    assert result["publish_id"] == "pub9"
    assert result["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert "#ai" in result["caption"]
    puts = [c for c in http.calls if c["method"] == "PUT"]
    assert [c["headers"]["Content-Range"] for c in puts] == [
        "bytes 0-9/25", "bytes 10-19/25", "bytes 20-24/25"]
    assert ledger.list(platform="tiktok")[0]["status"] == "processing"


def test_x_without_token_raises_honest_error(tmp_path, monkeypatch):
    monkeypatch.delenv("X_ACCESS_TOKEN", raising=False)
    pub = XPublisher(http=FakeHttp())
    path = _video_file(tmp_path)
    with pytest.raises(CapabilityUnavailable) as exc:
        pub.post_video(path, "hello")
    assert "developer.x.com" in exc.value.manual_step
    assert "PAID" in exc.value.manual_step
    assert pub.http.calls == []


def test_x_full_chunked_upload_and_tweet(tmp_path, monkeypatch):
    monkeypatch.setenv("X_ACCESS_TOKEN", "tok")
    http = FakeHttp()
    http.route("POST", "/2/media/upload/initialize",
               FakeResponse(200, {"data": {"id": "m1"}}))
    http.route("POST", "/2/media/upload/m1/append",
               [FakeResponse(204, None), FakeResponse(204, None),
                FakeResponse(204, None)])
    http.route("POST", "/2/media/upload/m1/finalize",
               FakeResponse(200, {"data": {
                   "id": "m1",
                   "processing_info": {"state": "succeeded"}}}))
    http.route("POST", "/2/tweets",
               FakeResponse(200, {"data": {"id": "t123", "text": "hi"}}))
    pub = XPublisher(http=http)
    path = _video_file(tmp_path, 25)

    # chunk_size small to force 3 chunks; bypass the real class chunk size
    orig = XPublisher.upload_video
    pub2_calls = {}

    def upload_spy(self, file_path, **kw):
        pub2_calls.update(kw)
        return orig(self, file_path, chunk_size=10, **kw)

    import types
    pub.upload_video = types.MethodType(upload_spy, pub)
    result = pub.post_video(path, "hello world " * 40, tags=["ai"])
    assert result["tweet_id"] == "t123"
    assert result["media_id"] == "m1"
    assert len(result["text"]) <= 280
    appends = [c for c in http.calls if "/append" in c["url"]]
    assert len(appends) == 3
    init = next(c for c in http.calls if "/initialize" in c["url"])
    assert init["payload"]["media_category"] == "tweet_video"
    assert init["payload"]["total_bytes"] == 25


def test_x_refuses_too_long_video(tmp_path, monkeypatch):
    monkeypatch.setenv("X_ACCESS_TOKEN", "tok")
    pub = XPublisher(http=FakeHttp())
    path = _video_file(tmp_path)
    with pytest.raises(CapabilityUnavailable, match="140s"):
        pub.upload_video(path, duration_s=200)


def test_meta_reel_requires_public_url(tmp_path):
    pub = MetaPublisher(_vault(), http=FakeHttp())
    with pytest.raises(CapabilityUnavailable) as exc:
        pub.publish_reel("/tmp/local.mp4", caption="hi", confirmed=True)
    assert "public HTTPS URL" in exc.value.manual_step
    assert pub.http.calls == []


def test_meta_reel_full_flow(tmp_path):
    http = FakeHttp()
    pub = MetaPublisher(_vault(), http=http)
    pub._store_credential("iguser", "token123", credential_type="oauth_token",
                          metadata={"ig_user_id": "12345"})
    http.route("POST", "/12345/media", FakeResponse(200, {"id": "c1"}))
    http.route("GET", "/c1", [FakeResponse(200, {"status_code": "IN_PROGRESS"}),
                              FakeResponse(200, {"status_code": "FINISHED"})])
    http.route("POST", "/12345/media_publish",
               FakeResponse(200, {"id": "media999"}))
    http.route("GET", "/media999",
               FakeResponse(200, {"permalink": "https://ig.test/p/x"}))
    result = pub.publish_reel("https://cdn.example.com/v.mp4",
                              caption="my reel", tags=["reels"],
                              confirmed=True, poll_interval_s=0.01)
    assert result["media_id"] == "media999"
    assert result["permalink"] == "https://ig.test/p/x"
    container = next(c for c in http.calls
                     if c["method"] == "POST" and "/12345/media" in c["url"]
                     and "media_publish" not in c["url"])
    assert container["payload"]["media_type"] == "REELS"
    assert "#reels" in container["payload"]["caption"]


def test_meta_facebook_video_needs_page_token(tmp_path):
    pub = MetaPublisher(_vault(), http=FakeHttp())
    with pytest.raises(CapabilityUnavailable) as exc:
        pub.publish_facebook_video(
            "999", video_url="https://cdn.example.com/v.mp4",
            confirmed=True)
    assert "me/accounts" in exc.value.manual_step
    assert "pages_manage_posts" in exc.value.manual_step
    assert pub.http.calls == []
