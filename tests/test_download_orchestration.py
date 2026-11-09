"""Play & Downloads Phase 2 — per-type acceptance contracts and the
shared download orchestrator (doc 7 §4, doc 8 P1).

The contract core is pure: a landed file is checked against what its
type must be BEFORE anything is delivered — container/extension
sanity, size floor, parseability where an in-repo parser exists.
A failed acceptance is a fact with a reason, never a delivery.

The orchestrator loop runs candidates on the Phase-1 progress
stream: retrieve → verify → process, moving to the next candidate
on a rejection with the reason journaled into the event stream.
Delivery itself stays with the caller (Phase-1 law: only the layer
that handed the file over may mark ``delivered``).
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_FRAME = b"\xff\xfb\x90\x00"
MP3_BODY = _FRAME * 2048                                    # 8192 B
HTML_BODY = (b"<!DOCTYPE html><html><head><title>404 Not Found"
             b"</title></head><body>gone</body></html>") * 8
MP4_BODY = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096
WAV_BODY = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " \
    + b"\x00" * 64
MZ_BODY = b"MZ" + b"\x00" * 2048


def _write(tmp_path: Path, name: str, body: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(body)
    return str(p)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _png(path: Path) -> str:
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (16, 12), (200, 30, 30))
    img.save(str(path), format="PNG")
    return str(path)


def _pdf(tmp_path: Path) -> str:
    from nomorals.core.pdf import pdf_write

    path, _n = pdf_write(str(tmp_path / "doc.pdf"),
                         "Hello document\n\nA real paragraph of text.")
    return path


# ── sniff + kind inference ────────────────────────────────────────────

def test_sniff_recognises_the_families(tmp_path):
    from nomorals.media.download_orchestrator import sniff_format

    assert sniff_format(_write(tmp_path, "a.mp3", MP3_BODY)) == "mp3"
    assert sniff_format(_write(tmp_path, "a.html", HTML_BODY)) == "html"
    assert sniff_format(_write(tmp_path, "a.mp4", MP4_BODY)) == "mp4"
    assert sniff_format(_write(tmp_path, "a.wav", WAV_BODY)) == "wav"
    assert sniff_format(_write(tmp_path, "a.zip",
                               _zip_bytes({"x.txt": b"hi"}))) == "zip"
    assert sniff_format(_write(tmp_path, "a.pdf",
                               b"%PDF-1.4\n" + b"0" * 64)) == "pdf"
    assert sniff_format(_write(tmp_path, "a.exe", MZ_BODY)) == "pe"
    assert sniff_format(_write(tmp_path, "a.bin", b"")) == "empty"
    assert sniff_format(str(tmp_path / "missing")) == "missing"


def test_infer_kind_from_extension_and_content(tmp_path):
    from nomorals.media.download_orchestrator import infer_kind

    assert infer_kind("x.mp3") == "audio"
    assert infer_kind("x.mkv") == "video"
    assert infer_kind("x.epub") == "book"
    assert infer_kind("x.pdf") == "document"
    assert infer_kind("x.zip") == "archive"
    assert infer_kind("x.apk") == "software"
    assert infer_kind("x.bin", audio_only=True) == "audio"
    png = _png(tmp_path / "shot.bin")
    assert infer_kind(png) == "image"


# ── per-type contracts ────────────────────────────────────────────────

def test_song_accepts_real_audio_and_records_format(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    acc = accept_artifact(_write(tmp_path, "s.mp3", MP3_BODY), "song")
    assert acc.ok, acc.reason
    assert acc.facts["format"] == "mp3"
    assert acc.facts["size_bytes"] == len(MP3_BODY)


def test_song_rejects_an_html_error_page(tmp_path):
    """The classic wrong-media failure: an error page renamed .mp3
    is positively identified as HTML and refused BEFORE delivery."""
    from nomorals.media.download_orchestrator import accept_artifact

    acc = accept_artifact(_write(tmp_path, "s.mp3", HTML_BODY), "song")
    assert not acc.ok
    assert "HTML" in acc.reason


def test_every_binary_kind_rejects_zero_bytes(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    empty = _write(tmp_path, "e.bin", b"")
    for kind in ("song", "video", "image", "document", "book",
                 "archive", "software", "web_capture", "file"):
        acc = accept_artifact(empty, kind)
        assert not acc.ok, kind
        assert "zero" in acc.reason.lower(), (kind, acc.reason)


def test_song_rejects_preview_length_against_expected_duration(tmp_path):
    """A 2-second file for a 200-second track is a rejected clip —
    the full-track verdict rides inside the song contract."""
    ffmpeg = pytest.importorskip("shutil").which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg absent")
    import subprocess

    out = tmp_path / "clip.mp3"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", str(out)],
        check=True, capture_output=True, timeout=60)
    from nomorals.media.download_orchestrator import accept_artifact

    acc = accept_artifact(str(out), "song", expected_duration=200.0)
    assert not acc.ok
    assert "preview" in acc.reason or "clip" in acc.reason


def test_video_contract(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    assert accept_artifact(
        _write(tmp_path, "v.mp4", MP4_BODY), "video").ok
    acc = accept_artifact(_write(tmp_path, "v.mp4", HTML_BODY), "video")
    assert not acc.ok and "HTML" in acc.reason


def test_image_contract_reads_dimensions(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    acc = accept_artifact(_png(tmp_path / "p.png"), "image")
    assert acc.ok, acc.reason
    assert acc.facts["width"] == 16 and acc.facts["height"] == 12
    acc = accept_artifact(
        _write(tmp_path, "p.png", _zip_bytes({"x": b"y"})), "image")
    assert not acc.ok


def test_document_contract_parses_pdf_and_counts_pages(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    acc = accept_artifact(_pdf(tmp_path), "document")
    assert acc.ok, acc.reason
    assert acc.facts["format"] == "pdf"
    assert acc.facts.get("pages", 0) >= 1
    acc = accept_artifact(_write(tmp_path, "d.pdf", MP3_BODY), "document")
    assert not acc.ok


def test_book_contract_accepts_epub_rejects_audio(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    epub = _zip_bytes({
        "mimetype": b"application/epub+zip",
        "META-INF/container.xml": b"<container/>",
        "OEBPS/ch1.xhtml": b"<html>chapter</html>"})
    assert accept_artifact(_write(tmp_path, "b.epub", epub), "book").ok
    acc = accept_artifact(_write(tmp_path, "b.epub", MP3_BODY), "book")
    assert not acc.ok


def test_archive_contract_checks_structure_and_corruption(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    good = _zip_bytes({"a.txt": b"hello", "b/c.txt": b"world"})
    acc = accept_artifact(_write(tmp_path, "a.zip", good), "archive")
    assert acc.ok, acc.reason
    assert acc.facts["members"] == 2
    bad = b"PK\x03\x04" + b"\xff" * 128
    acc = accept_artifact(_write(tmp_path, "a.zip", bad), "archive")
    assert not acc.ok and "corrupt" in acc.reason.lower()
    tar = _tar_bytes({"only.txt": b"data"})
    acc = accept_artifact(_write(tmp_path, "a.tar", tar), "archive")
    assert acc.ok and acc.facts["members"] == 1


def test_software_contract_apk_needs_a_manifest(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    apk = _zip_bytes({"AndroidManifest.xml": b"\x03\x00\x08\x00",
                      "classes.dex": b"dex\n"})
    assert accept_artifact(_write(tmp_path, "a.apk", apk), "software").ok
    plain = _zip_bytes({"readme.txt": b"not an app"})
    acc = accept_artifact(_write(tmp_path, "a.apk", plain), "software")
    assert not acc.ok and "AndroidManifest" in acc.reason
    assert accept_artifact(
        _write(tmp_path, "a.exe", MZ_BODY), "software").ok


def test_web_capture_contract_matches_the_expected_format(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    png = _png(tmp_path / "shot.png")
    assert accept_artifact(png, "web_capture",
                           expected_format="png").ok
    acc = accept_artifact(_pdf(tmp_path), "web_capture",
                          expected_format="png")
    assert not acc.ok


def test_generic_file_contract_is_existence_plus_bytes(tmp_path):
    from nomorals.media.download_orchestrator import accept_artifact

    assert accept_artifact(
        _write(tmp_path, "page.html", HTML_BODY), "file").ok
    acc = accept_artifact(str(tmp_path / "nope.bin"), "file")
    assert not acc.ok and "missing" in acc.reason.lower()


# ── the orchestrator loop ─────────────────────────────────────────────

def _tracker():
    from nomorals.core.progress_events import ProgressTracker

    return ProgressTracker(op_id="op-test", title="t")


def test_loop_moves_to_next_candidate_on_rejection(tmp_path):
    from nomorals.media.download_orchestrator import DownloadOrchestrator

    bad = _write(tmp_path, "bad.mp3", HTML_BODY)
    good = _write(tmp_path, "good.mp3", MP3_BODY)
    tracker = _tracker()
    orch = DownloadOrchestrator(tracker=tracker)
    seen = []

    def retrieve(cand):
        seen.append(cand)
        return (bad if cand == "first" else good), ""

    result = orch.run(["first", "second"], retrieve, kind="song",
                      label_of=lambda c: c)
    assert result.ok and result.path == good
    assert seen == ["first", "second"]
    assert len(result.attempts) == 2
    assert not result.attempts[0].ok
    assert "HTML" in result.attempts[0].reason
    assert result.attempts[1].ok
    # The rejection reason is journaled in the event stream itself.
    notes = [e.note for e in tracker.events]
    assert any("HTML" in n for n in notes), notes
    stages = [e.stage for e in tracker.events]
    assert "verifying" in stages
    assert tracker.snapshot()["source_index"] == 2


def test_loop_reports_total_failure_with_every_reason(tmp_path):
    from nomorals.media.download_orchestrator import DownloadOrchestrator

    orch = DownloadOrchestrator(tracker=_tracker())

    def retrieve(cand):
        return None, f"{cand} is down"

    result = orch.run(["a", "b"], retrieve, kind="song")
    assert not result.ok and result.path == ""
    assert [a.reason for a in result.attempts] == ["a is down", "b is down"]
    story = result.story()
    assert "a is down" in story and "b is down" in story


def test_loop_process_steps_run_after_acceptance_with_stage(tmp_path):
    from nomorals.core.progress_events import PROCESSING
    from nomorals.media.download_orchestrator import DownloadOrchestrator

    good = _write(tmp_path, "good.mp3", MP3_BODY)
    tracker = _tracker()
    orch = DownloadOrchestrator(tracker=tracker)
    staged = _write(tmp_path, "staged.mp3", MP3_BODY)

    result = orch.run(
        ["only"], lambda c: (good, ""), kind="song",
        process_steps=[("tagging the file", lambda p: staged)])
    assert result.ok and result.path == staged
    proc_events = [e for e in tracker.events if e.stage == PROCESSING]
    assert proc_events and "tagging" in proc_events[0].note


def test_loop_rejected_candidate_never_reaches_processing(tmp_path):
    from nomorals.media.download_orchestrator import DownloadOrchestrator

    bad = _write(tmp_path, "bad.mp3", HTML_BODY)
    orch = DownloadOrchestrator(tracker=_tracker())
    called = []
    result = orch.run(
        ["only"], lambda c: (bad, ""), kind="song",
        process_steps=[("tagging", lambda p: called.append(p) or p)])
    assert not result.ok and called == []


# ── wiring: /download produce refuses a wrong artifact ───────────────

def _handler(tmp_path):
    from tests.test_play_progress_demonstrator import _handler as build

    return build(tmp_path)


def _chat(platform="telegram", chat_id="12345"):
    chat = mock.Mock()
    chat.platform = platform
    chat.chat_id = chat_id
    chat.key = f"{platform}:{chat_id}"
    return chat


def test_download_produce_refuses_an_html_page_as_audio(tmp_path):
    from nomorals.media import downloader as dl_mod

    rt = _handler(tmp_path)
    chat = _chat()
    landed = _write(tmp_path, "song.mp3", HTML_BODY)
    report = SimpleNamespace(ok=True, path=landed, title="Song",
                             url="https://x.test/song.mp3")
    with mock.patch.object(dl_mod.MediaDownloader, "download",
                           return_value=report):
        text, sent = rt._download_produce(
            "https://x.test/song.mp3", True, chat)
    assert sent is False
    assert rt.gateway.files == []
    assert "HTML" in text


def test_download_produce_refuses_a_zero_byte_video(tmp_path):
    from nomorals.media import downloader as dl_mod

    rt = _handler(tmp_path)
    chat = _chat()
    landed = _write(tmp_path, "clip.mp4", b"")
    report = SimpleNamespace(ok=True, path=landed, title="Clip",
                             url="https://x.test/clip.mp4")
    with mock.patch.object(dl_mod.MediaDownloader, "download",
                           return_value=report):
        text, sent = rt._download_produce(
            "https://x.test/clip.mp4", False, chat)
    assert sent is False
    assert rt.gateway.files == []
    assert "zero" in text.lower()


def test_download_produce_sends_a_valid_video(tmp_path):
    from nomorals.media import downloader as dl_mod

    rt = _handler(tmp_path)
    chat = _chat()
    landed = _write(tmp_path, "clip.mp4", MP4_BODY)
    report = SimpleNamespace(ok=True, path=landed, title="Clip",
                             url="https://x.test/clip.mp4")
    with mock.patch.object(dl_mod.MediaDownloader, "download",
                           return_value=report):
        text, sent = rt._download_produce(
            "https://x.test/clip.mp4", False, chat)
    assert sent is True
    assert len(rt.gateway.files) == 1


# ── wiring: the browser download queue accepts before "done" ─────────

def test_queue_marks_zero_byte_finished_file_failed(tmp_path):
    from nomorals.browser.downloads import JOB_DONE, JOB_FAILED, DownloadQueue

    dest = tmp_path / "file.bin"
    queue = DownloadQueue(
        tmp_path / "q.json",
        fetcher=lambda job: {"ok": True, "path": str(dest),
                             "size_bytes": 0,
                             "sha256": "0" * 64, "backend": "fake"})
    dest.write_bytes(b"")
    job = queue.enqueue("https://x.test/file.bin", dest)
    done = queue.run_job(job.job_id)
    assert done.status == JOB_FAILED
    assert done.status != JOB_DONE
    assert "zero" in done.error.lower()


def test_queue_archive_kind_checks_structure(tmp_path):
    from nomorals.browser.downloads import JOB_DONE, JOB_FAILED, DownloadQueue

    good = _zip_bytes({"a.txt": b"hello"})
    dest = tmp_path / "a.zip"
    dest.write_bytes(good)
    import hashlib

    queue = DownloadQueue(
        tmp_path / "q.json",
        fetcher=lambda job: {"ok": True, "path": str(dest),
                             "size_bytes": len(good),
                             "sha256": hashlib.sha256(good).hexdigest(),
                             "backend": "fake"})
    job = queue.enqueue("https://x.test/a.zip", dest, kind="archive")
    assert queue.run_job(job.job_id).status == JOB_DONE

    dest2 = tmp_path / "b.zip"
    dest2.write_bytes(b"just some text, not a zip at all")
    queue2 = DownloadQueue(
        tmp_path / "q2.json",
        fetcher=lambda job: {"ok": True, "path": str(dest2),
                             "size_bytes": dest2.stat().st_size,
                             "sha256": "0" * 64, "backend": "fake"})
    job2 = queue2.enqueue("https://x.test/b.zip", dest2, kind="archive")
    done2 = queue2.run_job(job2.job_id)
    assert done2.status == JOB_FAILED


# ── wiring: web captures are accepted before they are sent ───────────

class _RecordingGateway:
    def __init__(self):
        self.files = []

    def send_file(self, platform, target, path, caption="", kind=None,
                  **kw):
        self.files.append((path, kind))
        return SimpleNamespace(ok=True)


def _group_handler(tmp_path):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    rt = RuntimeGroupMixin.__new__(RuntimeGroupMixin)
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = _RecordingGateway()
    return rt


def _message():
    return SimpleNamespace(chat=SimpleNamespace(
        platform="telegram", key="telegram:1"))


def test_webshot_deliver_sends_a_real_png(tmp_path):
    rt = _group_handler(tmp_path)
    png_bytes = Path(_png(tmp_path / "cap.png")).read_bytes()
    path = rt._webshot_deliver(_message(), png_bytes,
                                caption="shot", pdf=False)
    assert path.endswith(".png")
    assert len(rt.gateway.files) == 1


def test_webshot_deliver_refuses_non_png_bytes(tmp_path):
    import pytest as _pytest

    rt = _group_handler(tmp_path)
    with _pytest.raises(ValueError, match="capture rejected"):
        rt._webshot_deliver(_message(), HTML_BODY,
                            caption="shot", pdf=False)
    assert rt.gateway.files == []


def test_webrec_deliver_refuses_a_non_video_file(tmp_path):
    import pytest as _pytest

    rt = _group_handler(tmp_path)
    bogus = tmp_path / "rec.mp4"
    bogus.write_bytes(b"this is a text file wearing a .mp4 name")
    with _pytest.raises(ValueError, match="recording rejected"):
        rt._webrec_deliver(_message(), str(bogus),
                           caption="rec", name="rec.mp4")
    assert rt.gateway.files == []
    good = tmp_path / "rec2.mp4"
    good.write_bytes(MP4_BODY)
    assert rt._webrec_deliver(_message(), str(good),
                              caption="rec", name="rec.mp4")


# ── wiring: the registry journey verifies through the contract ──────

def test_registry_journey_skips_an_html_candidate(tmp_path):
    from nomorals.music_sources.live import _LiveJourney

    bad = _write(tmp_path, "bad.mp3", HTML_BODY)
    good = _write(tmp_path, "good.mp3", MP3_BODY)

    def cand(kind, path):
        return SimpleNamespace(
            kind=kind, path_or_url=path, title="Song", artist="A",
            duration=0.0, preview_only=False, downloadable=True,
            source_name=kind)

    journey = _LiveJourney(SimpleNamespace())
    journey._candidates = [cand("url", "https://bad.test/s.mp3"),
                           cand("url", "https://good.test/s.mp3")]

    class _Engine:
        def download(self, item):
            path = bad if "bad" in item["path"] else good
            return {"ok": True, "path": path,
                    "title": "Song", "artist": "A"}

    journey._engine = _Engine()
    source = SimpleNamespace(id="url", licence="unknown",
                             source_class="fetch")
    result = journey.download(source, "Song")
    assert result.downloaded is True
    assert result.payload["path"] == good
    assert result.payload["verify_state"] in ("verified", "unverified")
