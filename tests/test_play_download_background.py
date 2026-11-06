"""Background downloads for /play + /download (owner decision 2026-10-19):
long downloads move to background jobs and get sent once done.

Before this build, the chat handlers downloaded SYNCHRONOUSLY:
``_play_send_with_fallthrough`` / ``_play_send_in_chat_result`` /
``_control_download`` blocked the chat pump thread for the whole
multi-minute download.  Now the handler submits a job to the shared
JobRegistry (``nomorals/core/jobs.py``); the existing synchronous
body runs on the job's worker thread and sends the file itself when
the download lands.  Failed workers and heartbeat-silent
(presumed-dead) jobs are surfaced by ``_harvest_play_downloads`` —
exactly once, honest UNKNOWN, never a fake failure.

Presentation overhaul (owner, 2026-11-05): the message flow is ONE
message edited in place.  ``_play_download_bg`` sends the ack
itself (no queue counts, no "[job …]" id), the worker edits that
same message to the final state — the delivered item's line on
success, the honest failure on failure — and a success posts NO
follow-up text at all.  The handler returns "" (the kind family is
MEDIA_DELIVERED: the ack already posted).  Jobs are found by their
labels now — the ack no longer carries an id to parse.
"""

import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest


class _FakeGateway:
    """Records sends/files/edits; plays the adapter for edit verbs."""

    def __init__(self):
        self.sent = []    # (platform, chat, text)
        self.files = []   # (platform, target, path, caption)
        self.edits = []   # (message_id, text)
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def send_file(self, platform, target, path, caption=""):
        self.files.append((platform, target, path, caption))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id="f1")

    def _adapter_for(self, platform):
        return self

    def edit_message(self, chat, message_id, text, *, buttons=None):
        self.edits.append((str(message_id), text))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway()
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _item(kind="soundcloud", path="https://soundcloud.com/a/b", title="Song"):
    return {"kind": kind, "path": path, "title": title, "artist": "A"}


def _collect(chat_key, title="Song", timeout=10.0):
    from nomorals.core.jobs import shared_registry

    snaps = shared_registry().list(
        labels={"owner": "play", "chat": chat_key, "title": title})
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


def _sent_texts(rt):
    return [t for _p, _c, t in rt.gateway.sent]


# ── 1: the handler returns without blocking ───────────────────────────


def test_handler_returns_immediately_while_download_blocks():
    rt = _handler()
    chat = _chat("bg-1")
    started = threading.Event()
    release = threading.Event()

    engine = mock.Mock()

    def _blocking_download(item):
        started.set()
        release.wait(10)
        return {"ok": True, "path": "/tmp/song.mp3", "title": "Song"}

    engine.download.side_effect = _blocking_download
    engine.queue.return_value = []

    t0 = time.time()
    out = rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: rt._play_send_in_chat_result(engine, _item(), chat),
        title="Song")
    elapsed = time.time() - t0
    assert elapsed < 2.0, \
        f"handler blocked for {elapsed:.1f}s on a parked download"
    # The ack already posted — the handler's own return is empty.
    assert out == ""
    assert _sent_texts(rt) == ["⏳ Song"]
    assert started.wait(5), "job worker never started the download"
    # Nothing delivered yet — the download is still parked.
    assert rt.gateway.files == []
    release.set()
    job = _collect(chat.key)
    assert job.status == "done"
    res = job.result
    assert res["delivered"] is True and res["sent"] is True
    assert res["ack_id"] == "m1"
    assert rt.gateway.files == [
        ("telegram", "telegram:bg-1", "/tmp/song.mp3", "🎵 Song — A")]
    # The ack was edited to the final state — the delivered line.
    assert rt.gateway.edits == [("m1", "🎵 Song — A")]


# ── 2: completion: file + ack edit to the right chat, no follow-up ────


def test_completed_job_sends_file_and_edits_ack():
    rt = _handler()
    chat = _chat("bg-2")
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/song.mp3", "title": "Song"}
    engine.queue.return_value = []
    out = rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: rt._play_send_in_chat_result(engine, _item(), chat),
        title="Song")
    assert out == ""
    job = _collect(chat.key)
    assert job.result["delivered"] is True
    assert rt.gateway.files == [
        ("telegram", "telegram:bg-2", "/tmp/song.mp3", "🎵 Song — A")]
    # No meta-narration: after the ack, NO further text is SENT —
    # the completion is the ack's edit plus the file landing.
    assert _sent_texts(rt) == ["⏳ Song"]
    assert rt.gateway.edits == [("m1", "🎵 Song — A")]


# ── 3: honest failure — wording + queue hygiene preserved in the job ──


def test_failed_download_reports_honestly_and_drops_queue_row():
    rt = _handler()
    chat = _chat("bg-3")
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": False, "reason": "This video is DRM protected"}
    item = _item()
    engine.queue.return_value = [
        {"index": 0, "kind": item["kind"], "path": item["path"],
         "title": item["title"]}]
    removed = []
    engine.remove.side_effect = lambda idx: removed.append(idx)
    rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: rt._play_send_in_chat_result(engine, item, chat),
        title="Song")
    job = _collect(chat.key)
    assert job.result["sent"] is False
    assert rt.gateway.files == []
    # The failure lands as the ack's edit — once, honestly worded.
    assert _sent_texts(rt) == ["⏳ Song"]
    assert len(rt.gateway.edits) == 1
    edit_text = rt.gateway.edits[0][1]
    assert "couldn't download" in edit_text
    assert "couldn't send" not in edit_text
    assert removed == [0], \
        f"stranded queue row not dropped in background path: {removed}"


# ── 4: fall-through across candidates still works inside the job ──────


def test_fallthrough_still_tries_next_candidate_in_background():
    rt = _handler()
    chat = _chat("bg-4")
    sc = _item("soundcloud", "https://soundcloud.com/a/b", "Song")
    yt = _item("youtube", "https://www.youtube.com/watch?v=abcdefghijk",
               "Song")
    engine = mock.Mock()
    engine.download.side_effect = [
        {"ok": False, "reason": "This video is DRM protected"},
        {"ok": True, "path": "/tmp/yt.mp3", "title": "Song"}]
    engine.queue.return_value = []
    attempts = ["soundcloud-search — ok (1)", "youtube-search — ok (1)"]
    rt._play_download_bg(
        chat, chat.key, "⏳ Song",
        lambda: rt._play_send_with_fallthrough(
            engine, sc, [yt], attempts, chat),
        title="Song")
    job = _collect(chat.key)
    assert job.status == "done"
    assert len(rt.gateway.files) == 1
    assert rt.gateway.files[0][2] == "/tmp/yt.mp3"
    # The tried-list is journal-only now: it appears in NO surface
    # text (ack, edits, captions).
    surface = _sent_texts(rt) + [t for _m, t in rt.gateway.edits] \
        + [c for _p, _t, _path, c in rt.gateway.files]
    assert all("sources tried" not in t for t in surface), surface
    # Progress was reported between candidates (heartbeat visibility).
    progress = job.snapshot()["last_progress"]
    assert progress and "trying" in progress, \
        f"no per-candidate progress note recorded: {progress!r}"


# ── 5: pick-list consumed only when the file actually went out ────────


def _pick_rt_and_token(chat, *, download_ok):
    from nomorals.media.picklist import PickCache

    rt = _handler()
    rt._ref_from_key = lambda key: chat
    cand = SimpleNamespace(
        path_or_url="https://soundcloud.com/a/b", kind="soundcloud",
        title="Song", artist="A", duration=1.0)
    token = PickCache.store([cand], chat.key)
    assert token
    engine = mock.Mock()
    engine._enqueue.return_value = {
        "kind": "soundcloud", "path": cand.path_or_url, "title": "Song"}
    if download_ok:
        engine.download.return_value = {
            "ok": True, "path": "/tmp/song.mp3", "title": "Song"}
    else:
        engine.download.return_value = {
            "ok": False, "reason": "This video is DRM protected"}
    engine.queue.return_value = [
        {"index": 0, "kind": "soundcloud", "path": cand.path_or_url,
         "title": "Song"}]
    engine.remove.return_value = None
    return rt, token, engine


def test_pick_consumed_on_successful_background_send():
    from nomorals.media.picklist import PickCache

    chat = _chat("bg-5")
    rt, token, engine = _pick_rt_and_token(chat, download_ok=True)
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        out = rt._play_pick(f"{token} 1", chat.key)
    assert out == ""
    job = _collect(chat.key)
    assert job.result["sent"] is True
    assert PickCache.get(token) is None, \
        "pick-list must be consumed once the picked track was sent"


def test_pick_not_consumed_when_background_download_fails():
    from nomorals.media.picklist import PickCache

    chat = _chat("bg-6")
    rt, token, engine = _pick_rt_and_token(chat, download_ok=False)
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        rt._play_pick(f"{token} 1", chat.key)
    job = _collect(chat.key)
    assert job.result["sent"] is False
    assert PickCache.get(token) is not None, \
        "a failed pick must leave the list alive for another try"


# ── 6: presumed-dead surfaces via harvest — honest UNKNOWN, once ──────


def test_presumed_dead_job_surfaces_unknown_exactly_once():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("bg-7")
    release = threading.Event()
    # Long heartbeat interval: the heartbeat thread won't refresh
    # last_heartbeat_at during the test, so the backdate below holds.
    job = shared_registry().submit(
        lambda: release.wait(30), tool="play-download",
        labels={"owner": "play", "chat": chat.key, "title": "Song"},
        heartbeat_interval_s=60.0)
    with job.lock:
        job.last_heartbeat_at = time.time() - 400.0
    rt._harvest_play_downloads(chat.key, chat)
    sends = _sent_texts(rt)
    assert len(sends) == 1, f"expected one UNKNOWN note, got {sends}"
    assert "went quiet" in sends[0] and "outcome unknown" in sends[0]
    assert "Song" in sends[0]
    # Exactly once: a second harvest must not re-report the same job.
    rt._harvest_play_downloads(chat.key, chat)
    assert len(_sent_texts(rt)) == 1
    release.set()
    shared_registry().collect(job.id, timeout=5)


def test_harvest_ignores_jobs_the_body_already_delivered():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("bg-8")
    job = shared_registry().submit(
        lambda: {"ok": True, "delivered": True, "sent": True,
                 "title": "Song", "text": "🎵 Song"},
        tool="play-download",
        labels={"owner": "play", "chat": chat.key, "title": "Song"})
    shared_registry().collect(job.id, timeout=5)
    rt._harvest_play_downloads(chat.key, chat)
    assert rt.gateway.sent == []


def test_harvest_reports_crashed_worker():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("bg-9")

    def _boom():
        raise RuntimeError("worker exploded")

    job = shared_registry().submit(
        _boom, tool="play-download",
        labels={"owner": "play", "chat": chat.key, "title": "Song"})
    shared_registry().collect(job.id, timeout=5)
    rt._harvest_play_downloads(chat.key, chat)
    sends = _sent_texts(rt)
    assert len(sends) == 1 and "broke" in sends[0], sends
    assert "worker exploded" in sends[0]


def test_harvest_edits_the_ack_when_the_body_could_not():
    # done-but-undelivered with a surviving ack id: the outcome
    # REPLACES the ack (edit), it does not post as a new message.
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("bg-12")
    job = shared_registry().submit(
        lambda: {"ok": True, "delivered": False, "sent": False,
                 "title": "Song", "ack_id": "m1",
                 "text": "couldn't download the audio: DRM"},
        tool="play-download",
        labels={"owner": "play", "chat": chat.key, "title": "Song"})
    shared_registry().collect(job.id, timeout=5)
    rt._harvest_play_downloads(chat.key, chat)
    assert rt.gateway.edits == [
        ("m1", "couldn't download the audio: DRM")]
    assert rt.gateway.sent == []


# ── 7: /download goes to the background too ───────────────────────────


def test_control_download_returns_ack_and_sends_when_done(tmp_path):
    rt = _handler()
    chat = _chat("bg-10")
    rt._ref_from_key = lambda key: chat
    target = tmp_path / "clip.mp3"
    target.write_bytes(b"x" * 2048)
    report = SimpleNamespace(
        ok=True, path=str(target), title="Clip", size_bytes=2048,
        stages=[], hint="")
    with mock.patch("nomorals.media.downloader.MediaDownloader") as MD:
        MD.return_value.download.return_value = report
        out = rt._control_download(
            "https://example.com/clip.mp3 audio", chat_key=chat.key)
    assert out == ""
    assert _sent_texts(rt) == ["⏳ https://example.com/clip.mp3"]
    job = _collect(chat.key, title="https://example.com/clip.mp3")
    assert job.result["delivered"] is True
    assert len(rt.gateway.files) == 1
    assert rt.gateway.files[0][2] == str(target)
    # Final state replaces the ack: the item's own line — no size
    # receipt, no "sent to this chat".
    assert rt.gateway.edits == [("m1", "🎵 Clip")]
    surface = _sent_texts(rt) + [t for _m, t in rt.gateway.edits]
    assert all("sent to this chat" not in t for t in surface)


# ── 8: registry failure falls back to the synchronous body ────────────


def test_submit_failure_falls_back_to_synchronous_send():
    rt = _handler()
    chat = _chat("bg-11")
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/song.mp3", "title": "Song"}
    engine.queue.return_value = []
    with mock.patch("nomorals.core.jobs.shared_registry",
                    side_effect=RuntimeError("registry down")):
        text = rt._play_download_bg(
            chat, chat.key, "⏳ Song",
            lambda: rt._play_send_in_chat_result(
                engine, _item(), chat),
            title="Song")
    # Sync fallback returns the body's own outcome line — the file
    # send itself is the delivery, no narration around it.
    assert text == "🎵 Song — A"
    assert len(rt.gateway.files) == 1
