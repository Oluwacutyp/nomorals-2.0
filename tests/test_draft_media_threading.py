"""WS3 P7 remainder — draft media threading into execute_post.

Before this pass a draft could carry media (``metadata["media"]`` —
the GS desk landing attaches its PNG exactly this way) and the
publish path below the manager already carried media end to end,
but execute_post never loaded the entries and never passed them:
every approved media draft posted text-only, silently.

Pins:
(a) a media draft executes with its media reaching a recording
    adapter in draft order, and the post row's record (media_paths
    + offered/carried metrics) matches what the adapter received;
(b) a no-media draft executes byte-identically to before — publish
    is called with NO media_paths kwarg;
(c) a platform whose adapter carries no media posts the text and
    the record + owner-facing note SAY the media was not carried;
(d) a single-carry platform (first-item only) records the rest as
    not carried, by count;
(e) an unresolvable media entry fails the execution BEFORE any
    post: no adapter is called, the draft is not marked posted,
    the failure names the media (never an internal id);
(f) the adapter-shape fixes: Mastodon uploads media_paths into
    media_ids; Bluesky uploads image blobs into an images embed
    (non-image media is not attached and the record shows it).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.base import Account, PlatformAdapter, PostResult
from nomorals.social.drafts import (
    POSTED,
    SCHEDULED,
    DraftQueue,
    execute_post,
    handle_draft_callback,
    media_notes,
    post_scheduled_draft,
)
from nomorals.social.manager import SocialManager
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-bytes"
JPG_BYTES = b"\xff\xd8\xff\xe0fake-jpg-bytes"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42fake-video"


# ── recording fake adapters ────────────────────────────────────────────────


class RecordingAdapter(PlatformAdapter):
    """Carries everything it is offered and reports attachments —
    the postiz-shaped contract."""

    max_chars = 5000

    def __init__(self, name: str) -> None:
        self.name = name
        self.supports_media = True
        self.calls: list[dict] = []

    def post(self, account, content, **kwargs):
        media = [Path(p) for p in (kwargs.get("media_paths") or [])]
        self.calls.append({
            "content": content,
            "names": [p.name for p in media],
            "blobs": [p.read_bytes() for p in media],
            "reply_to": kwargs.get("reply_to") or "",
        })
        return PostResult(
            platform=self.name, ok=True,
            external_id=f"ext-{len(self.calls)}",
            metrics={"media_attached": len(media)} if media else {})


class TextOnlyAdapter(PlatformAdapter):
    """The x/linkedin shape: accepts the call, carries no media,
    reports nothing attached."""

    max_chars = 5000
    supports_media = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    def post(self, account, content, **kwargs):
        self.calls.append({
            "content": content,
            "media_paths": list(kwargs.get("media_paths") or []),
        })
        return PostResult(platform=self.name, ok=True, external_id="ext-t")


class FirstOnlyAdapter(PlatformAdapter):
    """The personal-surface shape: a send takes ONE media item."""

    max_chars = 5000
    supports_media = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    def post(self, account, content, **kwargs):
        media = list(kwargs.get("media_paths") or [])
        self.calls.append({"content": content, "used": media[:1]})
        return PostResult(
            platform=self.name, ok=True, external_id="ext-1",
            metrics={"media_attached": 1} if media else {})


# ── fixtures ───────────────────────────────────────────────────────────────


def _manager(tmp_path, adapters):
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(db=db, tools=None, extras={},
                          settings=SimpleNamespace(
                              workspace_dir=str(tmp_path)))
    manager = SocialManager(
        ctx, adapters={a.name: a for a in adapters},
        enforce_policy=False)
    for adapter in adapters:
        manager.connect(adapter.name, f"@owner-{adapter.name}")
    return manager


def _store(tmp_path, db=None):
    db = db or Database(":memory:")
    db.migrate()
    return ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))


def _queue(tmp_path):
    return DraftQueue(db_path=tmp_path / "drafts.db")


def _media_draft(queue, store, items, *, platforms):
    """items: list of (bytes, mime, name) → a draft whose metadata
    media entries point at freshly stored artifacts, desk-style."""
    entries = []
    for data, mime, name in items:
        art = store.put(data, type="image", creator="test",
                        mime=mime, metadata={"filename": name})
        entries.append({"artifact": art.id, "mime": mime,
                        "name": name})
    draft = queue.create_draft(
        "the desk card for today", platforms=platforms,
        metadata={"media": entries})
    return draft


# ── (a) media reaches the adapter, in order, record matches ───────────────


def test_media_reaches_adapter_in_order_and_record_matches(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    rec = RecordingAdapter("mastodon")
    manager = _manager(tmp_path, [rec])
    draft = _media_draft(
        queue, store,
        [(PNG_BYTES, "image/png", "desk.png"),
         (JPG_BYTES, "image/jpeg", "chart.jpg")],
        platforms=["mastodon"])
    queue.propose(draft.id)
    queue.approve(draft.id)

    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue, artifact_store=store)

    assert len(outcome.posted) == 1
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["names"] == ["desk.png", "chart.jpg"]  # draft order
    assert call["blobs"] == [PNG_BYTES, JPG_BYTES]     # real bytes
    # the execution record (the social_posts row) matches the wire
    rows = manager.posts.find(platform="mastodon")
    assert len(rows) == 1
    row = rows[0]
    assert [Path(p).name for p in row["media_paths"]] == [
        "desk.png", "chart.jpg"]
    metrics = row["metrics"]
    assert metrics["media_offered"] == 2
    assert metrics["media_carried"] == 2
    assert metrics["media_names"] == ["desk.png", "chart.jpg"]
    # and the draft's own lifecycle completed
    assert queue.get(draft.id).status == POSTED
    # owner-facing note: carried, named platform
    note = media_notes(outcome)
    assert "media went out on mastodon" in note


def test_store_derived_from_manager_db_when_not_passed(tmp_path):
    """No explicit store: execute_post builds the canonical store
    over the manager's (file-backed) db and still resolves."""
    db = Database(tmp_path / "ctx.db")
    db.migrate()
    store = ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))
    queue = _queue(tmp_path)
    rec = RecordingAdapter("mastodon")
    ctx = SimpleNamespace(db=db, tools=None, extras={},
                          settings=SimpleNamespace(
                              workspace_dir=str(tmp_path)))
    manager = SocialManager(ctx, adapters={"mastodon": rec},
                            enforce_policy=False)
    manager.connect("mastodon", "@owner")
    draft = _media_draft(queue, store,
                         [(PNG_BYTES, "image/png", "desk.png")],
                         platforms=["mastodon"])
    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue)
    assert len(outcome.posted) == 1
    assert rec.calls[0]["blobs"] == [PNG_BYTES]


# ── (b) media-less drafts: byte-identical call shape ───────────────────────


def test_no_media_draft_publish_call_shape_unchanged(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    rec = RecordingAdapter("mastodon")
    manager = _manager(tmp_path, [rec])
    draft = queue.create_draft("plain text post",
                                platforms=["mastodon"])
    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue, artifact_store=store)
    assert len(outcome.posted) == 1
    assert rec.calls[0]["names"] == []
    rows = manager.posts.find(platform="mastodon")
    assert rows[0]["media_paths"] == []
    assert "media_offered" not in (rows[0]["metrics"] or {})
    assert media_notes(outcome) == ""


# ── (c) a no-media platform: text posts, the gap is NAMED ──────────────────


def test_text_only_platform_records_and_names_uncarried_media(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    plain = TextOnlyAdapter("x")
    manager = _manager(tmp_path, [plain])
    draft = _media_draft(queue, store,
                         [(PNG_BYTES, "image/png", "desk.png")],
                         platforms=["x"])
    outcome = execute_post(manager, draft.content, platforms=["x"],
                           draft_id=draft.id, queue=queue,
                           artifact_store=store)
    # the post went out (text), the adapter WAS offered the media…
    assert len(outcome.posted) == 1
    assert len(plain.calls[0]["media_paths"]) == 1
    # …and the record says it carried none of it
    row = manager.posts.find(platform="x")[0]
    assert row["metrics"]["media_offered"] == 1
    assert row["metrics"]["media_carried"] == 0
    note = media_notes(outcome)
    assert "x posted the text without the media" in note
    assert "0 of 1" in note


def test_callback_reply_names_carried_and_uncarried_media(tmp_path):
    """The approve-tap path (handle_draft_callback → execute_post)
    in its production shape: the manager's db is file-backed and
    the draft's artifacts live in that same db, so execute_post
    derives the store from the manager — no explicit store."""
    db = Database(tmp_path / "ctx.db")
    db.migrate()
    store = ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))
    queue = _queue(tmp_path)
    plain = TextOnlyAdapter("x")
    rec = RecordingAdapter("mastodon")
    ctx = SimpleNamespace(db=db, tools=None, extras={},
                          settings=SimpleNamespace(
                              workspace_dir=str(tmp_path)))
    manager = SocialManager(
        ctx, adapters={"x": plain, "mastodon": rec},
        enforce_policy=False)
    manager.connect("x", "@owner-x")
    manager.connect("mastodon", "@owner-mastodon")
    draft = _media_draft(queue, store,
                         [(PNG_BYTES, "image/png", "desk.png")],
                         platforms=["x", "mastodon"])
    queue.propose(draft.id)
    reply = handle_draft_callback(
        queue, f"draft:post:{draft.id}", manager=manager)
    assert "posted to 2 platform(s)." in reply
    assert "media went out on mastodon" in reply
    assert "x posted the text without the media" in reply
    assert rec.calls[0]["names"] == ["desk.png"]
    assert queue.get(draft.id).status == POSTED


def test_scheduled_post_carries_media_and_note(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    rec = RecordingAdapter("mastodon")
    plain = TextOnlyAdapter("x")
    manager = _manager(tmp_path, [rec, plain])
    draft = _media_draft(queue, store,
                         [(PNG_BYTES, "image/png", "desk.png")],
                         platforms=["mastodon", "x"])
    queue.propose(draft.id)
    queue.schedule(draft.id, 1791482400.0)
    assert queue.get(draft.id).status == SCHEDULED
    # post_scheduled_draft uses execute_post without an explicit
    # store; give the manager a context-carried store instead.
    manager.context.artifact_store = store
    result = post_scheduled_draft(draft.id, queue=queue,
                                  manager=manager)
    assert result["ok"] is True
    assert result["posted"] == 2
    assert rec.calls[0]["names"] == ["desk.png"]
    assert "media went out on mastodon" in result["media_note"]
    assert "x posted the text without the media" in result[
        "media_note"]


# ── (d) first-only platform: the remainder is counted, not dropped ─────────


def test_first_only_platform_records_partial_carry(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    first = FirstOnlyAdapter("telegram")
    manager = _manager(tmp_path, [first])
    draft = _media_draft(
        queue, store,
        [(PNG_BYTES, "image/png", "one.png"),
         (JPG_BYTES, "image/jpeg", "two.jpg"),
         (MP4_BYTES, "video/mp4", "clip.mp4")],
        platforms=["telegram"])
    outcome = execute_post(manager, draft.content,
                           platforms=["telegram"], draft_id=draft.id,
                           queue=queue, artifact_store=store)
    assert len(outcome.posted) == 1
    row = manager.posts.find(platform="telegram")[0]
    assert row["metrics"]["media_offered"] == 3
    assert row["metrics"]["media_carried"] == 1
    note = media_notes(outcome)
    assert "telegram carried 1 of 3 media items" in note


# ── (e) unresolvable media fails BEFORE posting ─────────────────────────────


def test_missing_artifact_fails_before_posting(tmp_path):
    queue = _queue(tmp_path)
    store = _store(tmp_path)
    rec = RecordingAdapter("mastodon")
    manager = _manager(tmp_path, [rec])
    draft = queue.create_draft(
        "card with a vanished image", platforms=["mastodon"],
        metadata={"media": [{"artifact": "art-gone-123",
                             "mime": "image/png",
                             "name": "desk.png"}]})
    queue.propose(draft.id)
    queue.approve(draft.id)
    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue, artifact_store=store)
    # nothing posted, anywhere
    assert rec.calls == []
    assert outcome.posted == []
    assert len(outcome.failed) == 1
    failure = outcome.failed[0]
    assert failure.platform == "mastodon"
    assert failure.error_code == "media"
    # the failure names the media by its display name — and never
    # leaks the artifact id into the owner-facing record
    assert "desk.png" in failure.error
    assert "no longer in the artifact store" in failure.error
    assert "art-gone-123" not in failure.error
    # the draft was NOT marked posted — no masquerade
    assert queue.get(draft.id).status != POSTED
    assert manager.posts.find(platform="mastodon") == []


def test_missing_file_entry_fails_before_posting(tmp_path):
    queue = _queue(tmp_path)
    rec = RecordingAdapter("mastodon")
    manager = _manager(tmp_path, [rec])
    gone = tmp_path / "gone.png"
    draft = queue.create_draft(
        "path media", platforms=["mastodon"],
        metadata={"media": [{"path": str(gone), "mime": "image/png",
                             "name": "gone.png"}]})
    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue, artifact_store=_store(tmp_path))
    assert rec.calls == []
    assert outcome.posted == []
    assert "gone.png" in outcome.failed[0].error
    assert queue.get(draft.id).status != POSTED


def test_path_entry_media_posts(tmp_path):
    queue = _queue(tmp_path)
    rec = RecordingAdapter("mastodon")
    manager = _manager(tmp_path, [rec])
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG_BYTES)
    draft = queue.create_draft(
        "path media", platforms=["mastodon"],
        metadata={"media": [{"path": str(photo), "mime": "image/png",
                             "name": "photo.png"}]})
    outcome = execute_post(manager, draft.content,
                           platforms=["mastodon"], draft_id=draft.id,
                           queue=queue, artifact_store=_store(tmp_path))
    assert len(outcome.posted) == 1
    assert rec.calls[0]["names"] == ["photo.png"]
    assert rec.calls[0]["blobs"] == [PNG_BYTES]


# ── (f) the adapter shapes: mastodon uploads, bluesky embeds ───────────────


class _FakeResponse:
    def __init__(self, payload, *, ok=True, status=200):
        self._payload = payload
        self.ok = ok
        self.status = status
        self.text = json.dumps(payload)
        self.headers = {}


class _MastodonHttp:
    """Records the adapter's real HTTP calls; answers uploads and
    the status post."""

    def __init__(self):
        self.multipart_calls: list[dict] = []
        self.json_calls: list[dict] = []

    def post_multipart(self, url, files=None, **kw):
        self.multipart_calls.append({"url": url, "files": files})
        return _FakeResponse({"id": "m-42"})

    def post_json(self, url, payload, **kw):
        self.json_calls.append({"url": url, "payload": payload})
        return _FakeResponse({"id": "status-1",
                              "url": "https://example.social/@me/1"})


def test_mastodon_uploads_media_paths_into_media_ids(tmp_path):
    from nomorals.social.adapters.mastodon import Adapter

    http = _MastodonHttp()
    adapter = Adapter()
    adapter._client = lambda account: http  # noqa: SLF001 — test seam
    photo = tmp_path / "desk.png"
    photo.write_bytes(PNG_BYTES)
    account = Account(platform="mastodon", handle="@me@example.social",
                      credentials="env:WHATEVER")
    result = adapter.post(account, "hello", media_paths=[str(photo)])
    assert result.ok
    assert len(http.multipart_calls) == 1
    assert http.multipart_calls[0]["url"].endswith("/api/v1/media")
    assert http.multipart_calls[0]["files"][0][0] == "file"
    payload = http.json_calls[0]["payload"]
    assert payload["media_ids[]"] == ["m-42"]
    assert result.metrics["media_attached"] == 1


def test_mastodon_upload_failure_fails_the_post(tmp_path):
    from nomorals.social.adapters.mastodon import Adapter

    class _BadHttp(_MastodonHttp):
        def post_multipart(self, url, files=None, **kw):
            return _FakeResponse({"error": "boom"}, ok=False,
                                 status=422)

    adapter = Adapter()
    adapter._client = lambda account: _BadHttp()  # noqa: SLF001
    photo = tmp_path / "desk.png"
    photo.write_bytes(PNG_BYTES)
    account = Account(platform="mastodon", handle="@me@example.social")
    result = adapter.post(account, "hello", media_paths=[str(photo)])
    assert not result.ok
    assert result.error_code == "media"
    assert "desk.png" in result.error


class _BlueskyHttp:
    """Answers uploadBlob; records raw bodies by content type."""

    def __init__(self):
        self.uploads: list[dict] = []

    def request(self, method, url, data=None, headers=None, **kw):
        self.uploads.append({"url": url, "data": data,
                             "headers": headers or {}})
        return _FakeResponse(
            {"blob": {"$type": "blob",
                      "ref": {"$link": "bafkrei-test"},
                      "mimeType": (headers or {}).get(
                          "Content-Type", ""),
                      "size": len(data or b"")}})


def _bluesky_adapter(monkeypatch, http):
    from nomorals.social.adapters import bluesky as bsky

    adapter = bsky.Adapter()
    monkeypatch.setattr(
        adapter, "_login",
        lambda account: {"accessJwt": "jwt", "did": "did:plc:test"})
    monkeypatch.setattr(bsky, "HttpClient", lambda **kw: http)
    created: list[dict] = []

    def _create(account, session, record):
        created.append(record)
        return {"uri": "at://did:plc:test/app.bsky.feed.post/abc",
                "cid": "cid-1"}

    monkeypatch.setattr(adapter, "_create_post", _create)
    return adapter, created


def test_bluesky_embeds_uploaded_image_blobs(tmp_path, monkeypatch):
    http = _BlueskyHttp()
    adapter, created = _bluesky_adapter(monkeypatch, http)
    photo = tmp_path / "desk.png"
    photo.write_bytes(PNG_BYTES)
    account = Account(platform="bluesky", handle="me.bsky.social")
    result = adapter.post(account, "hello bluesky",
                          media_paths=[str(photo)])
    assert result.ok
    assert len(http.uploads) == 1
    assert http.uploads[0]["data"] == PNG_BYTES
    assert http.uploads[0]["headers"]["Content-Type"] == "image/png"
    embed = created[0]["embed"]
    assert embed["$type"] == "app.bsky.embed.images"
    assert embed["images"][0]["image"]["ref"]["$link"] == (
        "bafkrei-test")
    assert result.metrics["media_attached"] == 1


def test_bluesky_non_image_media_is_not_attached(tmp_path, monkeypatch):
    http = _BlueskyHttp()
    adapter, created = _bluesky_adapter(monkeypatch, http)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(MP4_BYTES)
    account = Account(platform="bluesky", handle="me.bsky.social")
    result = adapter.post(account, "a video draft",
                          media_paths=[str(clip)])
    assert result.ok
    assert http.uploads == []          # nothing uploaded
    assert "embed" not in created[0]   # nothing attached
    assert result.metrics["media_attached"] == 0
