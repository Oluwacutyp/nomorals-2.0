"""Background downloads for /play + /download (owner decision 2026-10-19):
long downloads move to background jobs and get sent once done.

Before this build, the chat handlers downloaded SYNCHRONOUSLY:
``_play_send_with_fallthrough`` / ``_play_send_in_chat_result`` /
``_control_download`` blocked the chat pump thread for the whole
multi-minute download.  Now the handler submits a job to the shared
JobRegistry (``nomorals/core/jobs.py``) and returns an immediate ack;
the existing synchronous body runs on the job's worker thread,
sends the file itself when the download lands, and posts its final
text (tried-list / "sent" / honest "couldn't download") as a
follow-up.  Failed workers and heartbeat-silent (presumed-dead)
jobs are surfaced by ``_harvest_play_downloads`` — exactly once,
honest UNKNOWN, never a fake failure.

Failing-first: ``_play_download_bg`` / ``_harvest_play_downloads``
did not exist before this build.
"""

import re
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest


def _handler():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt.gateway.send.return_value = mock.Mock(ok=True)
    rt.gateway.send_file.return_value = mock.Mock(ok=True)
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _item(kind="soundcloud", path="https://soundcloud.com/a/b", title="Song"):
    return {"kind": kind, "path": path, "title": title, "artist": "A"}


def _job_id_from_ack(ack: str) -> str:
    m = re.search(r"\[job (job[0-9a-f]+)\]", ack or "")
    assert m, f"ack carries no job id: {ack!r}"
    return m.group(1)


def _collect(chat_key, job_id, timeout=10.0):
    from nomorals.core.jobs import shared_registry

    job = shared_registry().collect(job_id, timeout=timeout)
    assert job is not None and job.done(), \
        f"job {job_id} never finished: {job.status if job else None}"
    return job


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

    out = "queued 1 (queue 1):\n  - Song"
    t0 = time.time()
    ack = rt._play_download_bg(
        chat, chat.key, out,
        lambda: rt._play_send_in_chat_result(engine, _item(), out, chat),
        title="Song")
    elapsed = time.time() - t0
    assert elapsed < 2.0, \
        f"handler blocked for {elapsed:.1f}s on a parked download"
    assert "downloading in the background" in ack
    assert out in ack
    job_id = _job_id_from_ack(ack)
    assert started.wait(5), "job worker never started the download"
    # Nothing delivered yet — the download is still parked.
    rt.gateway.send_file.assert_not_called()
    release.set()
    job = _collect(chat.key, job_id)
    assert job.status == "done"
    res = job.result
    assert res["delivered"] is True and res["sent"] is True
    rt.gateway.send_file.assert_called_once_with(
        "telegram", "telegram:bg-1", "/tmp/song.mp3",
        caption="🎵 Song")


# ── 2: completion delivers the file + outcome text to the right chat ──


def test_completed_job_sends_file_and_followup_to_originating_chat():
    rt = _handler()
    chat = _chat("bg-2")
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/song.mp3", "title": "Song"}
    engine.queue.return_value = []
    out = "queued 1 (queue 1):\n  - Song"
    ack = rt._play_download_bg(
        chat, chat.key, out,
        lambda: rt._play_send_in_chat_result(engine, _item(), out, chat),
        title="Song")
    job = _collect(chat.key, _job_id_from_ack(ack))
    assert job.result["delivered"] is True
    rt.gateway.send_file.assert_called_once_with(
        "telegram", "telegram:bg-2", "/tmp/song.mp3", caption="🎵 Song")
    followups = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert any("▶ sent" in t and "Song" in t for t in followups), \
        f"no sent-confirmation follow-up: {followups}"
    # The follow-up must not repeat the queue line the ack already showed.
    assert all("queued 1" not in t for t in followups)


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
    out = "queued 1 (queue 1):\n  - Song"
    ack = rt._play_download_bg(
        chat, chat.key, out,
        lambda: rt._play_send_in_chat_result(engine, item, out, chat),
        title="Song")
    job = _collect(chat.key, _job_id_from_ack(ack))
    assert job.result["sent"] is False
    rt.gateway.send_file.assert_not_called()
    followups = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert any("couldn't download" in t for t in followups), followups
    assert all("couldn't send" not in t for t in followups)
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
    out = "queued 1 (queue 1):\n  - Song"
    ack = rt._play_download_bg(
        chat, chat.key, out,
        lambda: (rt._play_send_with_fallthrough(
            engine, sc, [yt], attempts, out, chat), True),
        title="Song")
    job = _collect(chat.key, _job_id_from_ack(ack))
    assert job.status == "done"
    rt.gateway.send_file.assert_called_once()
    assert rt.gateway.send_file.call_args.args[2] == "/tmp/yt.mp3"
    followups = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert any("sources tried (2)" in t for t in followups), followups
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
        ack = rt._play_pick(f"{token} 1", chat.key)
    assert "downloading in the background" in ack
    job = _collect(chat.key, _job_id_from_ack(ack))
    assert job.result["sent"] is True
    assert PickCache.get(token) is None, \
        "pick-list must be consumed once the picked track was sent"


def test_pick_not_consumed_when_background_download_fails():
    from nomorals.media.picklist import PickCache

    chat = _chat("bg-6")
    rt, token, engine = _pick_rt_and_token(chat, download_ok=False)
    with mock.patch("nomorals.media.playback.PlaybackEngine",
                    return_value=engine):
        ack = rt._play_pick(f"{token} 1", chat.key)
    job = _collect(chat.key, _job_id_from_ack(ack))
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
    sends = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert len(sends) == 1, f"expected one UNKNOWN note, got {sends}"
    assert "went quiet" in sends[0] and "outcome unknown" in sends[0]
    assert "Song" in sends[0]
    # Exactly once: a second harvest must not re-report the same job.
    rt._harvest_play_downloads(chat.key, chat)
    assert rt.gateway.send.call_count == 1
    release.set()
    shared_registry().collect(job.id, timeout=5)


def test_harvest_ignores_jobs_the_body_already_delivered():
    from nomorals.core.jobs import shared_registry

    rt = _handler()
    chat = _chat("bg-8")
    job = shared_registry().submit(
        lambda: {"ok": True, "delivered": True, "sent": True,
                 "title": "Song", "text": "▶ sent"},
        tool="play-download",
        labels={"owner": "play", "chat": chat.key, "title": "Song"})
    shared_registry().collect(job.id, timeout=5)
    rt._harvest_play_downloads(chat.key, chat)
    rt.gateway.send.assert_not_called()


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
    sends = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert len(sends) == 1 and "broke" in sends[0], sends
    assert "worker exploded" in sends[0]


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
        ack = rt._control_download(
            "https://example.com/clip.mp3 audio", chat_key=chat.key)
    assert "downloading" in ack and "background" in ack
    job = _collect(chat.key, _job_id_from_ack(ack))
    assert job.result["delivered"] is True
    rt.gateway.send_file.assert_called_once()
    assert rt.gateway.send_file.call_args.args[2] == str(target)
    followups = [c.args[2] for c in rt.gateway.send.call_args_list]
    assert any("downloaded" in t and "sent to this chat" in t
               for t in followups), followups


# ── 8: registry failure falls back to the synchronous body ────────────


def test_submit_failure_falls_back_to_synchronous_send():
    rt = _handler()
    chat = _chat("bg-11")
    engine = mock.Mock()
    engine.download.return_value = {
        "ok": True, "path": "/tmp/song.mp3", "title": "Song"}
    engine.queue.return_value = []
    out = "queued 1 (queue 1):\n  - Song"
    with mock.patch("nomorals.core.jobs.shared_registry",
                    side_effect=RuntimeError("registry down")):
        text = rt._play_download_bg(
            chat, chat.key, out,
            lambda: rt._play_send_in_chat_result(
                engine, _item(), out, chat),
            title="Song")
    assert "▶ sent" in text and "downloading in the background" not in text
    rt.gateway.send_file.assert_called_once()
