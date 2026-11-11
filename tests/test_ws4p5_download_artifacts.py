"""WS4 P5 remainder — downloads register as WS1 artifacts.

The exploration loop cites exemplar media by WS1 artifact
reference (feed.py verifies every citation through the artifact
store), but ``media/downloader.py`` never registered what it
landed — everything Devon downloaded was invisible to the sweeps.
These pins close that gap at the downloader's completion seam:

* a completed download registers exactly one artifact carrying the
  producer idiom's provenance (origin / filename / source URL /
  content hash / mime / size, creator + provenance source);
* the artifact resolves through the SAME query shape the WS4 feed
  consumes (``claims_as_findings(..., artifacts=store)``) and its
  bytes read back identical;
* a failed/partial download registers nothing;
* a store failure never changes the download's result — it is
  journaled in the downloader's own failure ledger instead;
* re-downloading identical bytes dedups within the download
  origin (the store's producer rule: first registration wins);
* download outputs are byte-identical with and without a store.

Offline throughout: the kit stage is driven by an in-memory
transport double (the test_progress_events pattern), the direct
stage is mocked down, and every store/download path lives under
the test's tmp dir.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.media import downloader_failures
from nomorals.media.download_kit import ResumeDownloader
from nomorals.media.downloader import MediaDownloader
from nomorals.media.explore import ExploreStore
from nomorals.media.explore.feed import claims_as_findings
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

_BODY = bytes(range(256)) * 10  # 2560 deterministic bytes
_URL = "https://cdn.test/song.mp3"
_DAY = 86400.0


class _Server:
    """In-memory HTTP-ish server speaking the kit's transport seam."""

    def __init__(self, body=_BODY, *, fail_after=None):
        self.body = body
        self.fail_after = fail_after

    def head(self, url):
        return {"content_length": len(self.body), "etag": "v1"}

    def stream(self, url, start, end):
        data = self.body[start:]
        sent = 0
        for i in range(0, len(data), 1000):
            chunk = data[i:i + 1000]
            if self.fail_after is not None \
                    and sent + len(chunk) > self.fail_after:
                keep = self.fail_after - sent
                if keep > 0:
                    yield chunk[:keep]
                raise ConnectionError("connection reset")
            sent += len(chunk)
            yield chunk


def _artifact_store(tmp_path) -> ArtifactStore:
    db = Database(":memory:")
    db.migrate()
    return ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))


def _ctx(tmp_path, db=None):
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp_path), db=db)


def _downloader(ctx, server, **kw) -> MediaDownloader:
    kit = ResumeDownloader(head=server.head,
                           get_range_stream=server.stream)
    return MediaDownloader(
        ctx, kit_downloader=kit,
        ledger=SimpleNamespace(append=lambda **kwargs: True), **kw)


def _land(dl: MediaDownloader, url: str, dest: Path):
    """One download with the direct stage mocked down, so the kit
    stage is the one that lands (or fails to land) the file."""
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("yt-dlp path down")):
        return dl.download(url, dest_dir=dest)


def _download_rows(store: ArtifactStore):
    return [a for a in store.search(creator="media_downloader",
                                    limit=100)
            if (a.metadata or {}).get("origin") == "download"]


# ── registration ────────────────────────────────────────────────


def test_completed_download_registers_one_artifact(tmp_path):
    store = _artifact_store(tmp_path)
    dl = _downloader(_ctx(tmp_path), _Server(), artifact_store=store)
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()

    rows = _download_rows(store)
    assert len(rows) == 1
    art = rows[0]
    digest = hashlib.sha256(_BODY).hexdigest()
    assert art.type == "audio"
    assert art.mime == "audio/mpeg"
    assert art.size == len(_BODY)
    assert art.content_hash == digest
    meta = art.metadata
    assert meta["origin"] == "download"
    assert meta["filename"] == "song.mp3"
    assert meta["name"] == "song.mp3"
    assert meta["title"] == (report.title or "song.mp3")
    assert meta["source_url"] == _URL
    assert meta["source_path"] == report.path
    assert meta["host"] == "cdn.test"
    assert meta["size_bytes"] == len(_BODY)
    assert meta["sha256"] == digest
    assert meta["stage"] == "kit"
    assert meta["ts"] > 0
    assert art.provenance.source_type == "tool"
    assert art.provenance.source_id == "media_downloader"
    assert art.provenance.source_timestamp > 0
    # The artifact's bytes are the downloaded bytes, and the landed
    # file itself is untouched at the report's path.
    assert store.read(art.id) == _BODY
    assert Path(report.path).read_bytes() == _BODY


def test_context_built_store_registers_without_injection(tmp_path):
    """Production wiring: no injected store — the downloader builds
    the runtime's construction over the context's database."""
    db = Database(tmp_path / "ctx.db")
    db.migrate()
    dl = _downloader(_ctx(tmp_path, db=db), _Server())
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()

    db2 = Database(tmp_path / "ctx.db")
    store = ArtifactStore(db2, BlobStore(db2, tmp_path / "blobs"))
    rows = _download_rows(store)
    assert len(rows) == 1
    assert store.read(rows[0].id) == _BODY


def test_no_store_registers_nothing_and_journals_nothing(tmp_path):
    """No database anywhere = nowhere to register: the download is
    untouched and the absence is NOT journaled as a failure."""
    downloader_failures.clear_failures()
    dl = _downloader(_ctx(tmp_path, db=None), _Server())
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()
    sites = {f.site for f in downloader_failures.recent_failures(100)}
    assert "_register_download_artifact" not in sites
    assert "_artifact_store" not in sites


# ── the exploration feed sees the download ──────────────────────


def _claim(store, *, key, at, subject, subject_kind, backing):
    store.record_claim({
        "claim_id": f"claim:{key}",
        "claim_key": key,
        "kind": "trend",
        "subject": subject,
        "subject_kind": subject_kind,
        "text": f"{key}: sustained movement in the field",
        "source": "explore-trends",
        "observed_at": at,
        "scope": key,
        "confidence_basis": "seen in 2 sweeps; classes video + web",
        "basis": {"sweeps": ["sweep-1", "sweep-2"],
                  "source_classes": ["video", "web"]},
        "freshness_s": 3 * _DAY,
        "published_at": None,
        "contradicts": "",
        "entities": [],
        "backing": backing,
    })


def test_download_is_citable_through_the_exploration_feed(tmp_path):
    store = _artifact_store(tmp_path)
    dl = _downloader(_ctx(tmp_path), _Server(), artifact_store=store)
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()
    art = _download_rows(store)[0]

    now = time.time()
    explore = ExploreStore()
    explore.note_candidate(
        "channel", "Clip Channel", via="sweep:youtube_search",
        platform="youtube", beat="clips",
        evidence={"title": "Clip One", "url": "https://youtu.be/c1",
                  "instrument": "youtube_search",
                  "artifact_id": art.id},
        sweep_id="sweep-1", now=now - 700)
    key = explore.candidates()[0]["key"]
    _claim(explore, key="clip channel rising", at=now - 600,
           subject=key, subject_kind="channel",
           backing={"candidate_key": key, "observations": []})

    findings = claims_as_findings(explore, now=now, artifacts=store)
    assert len(findings) == 1
    finding = findings[0]
    assert finding["artifact_id"] == art.id
    assert finding["artifact_uri"] == f"artifact://{art.id}"
    assert finding["exemplars"][0]["artifact_id"] == art.id
    # …and the citation resolves to the downloaded bytes.
    assert store.read(finding["artifact_id"]) == _BODY


# ── failures register nothing; store failures never bite ───────


def test_failed_download_registers_nothing(tmp_path):
    store = _artifact_store(tmp_path)
    server = _Server(fail_after=0)  # stream dies on the first chunk
    dl = _downloader(_ctx(tmp_path), server, artifact_store=store,
                     max_proxies=0)
    report = _land(dl, _URL, tmp_path / "landed")
    assert not report.ok
    assert store.search(limit=10) == []
    assert store.stats()["total"] == 0


class _BrokenStore:
    """Store double whose write seam raises (the read seam works)."""

    def find_by_content_hash(self, digest):
        return []

    def put_file(self, *args, **kwargs):
        raise RuntimeError("artifact store is down")


def test_store_failure_leaves_download_untouched_and_is_journaled(
        tmp_path):
    downloader_failures.clear_failures()
    control = _downloader(_ctx(tmp_path, db=None), _Server())
    control_report = _land(control, _URL, tmp_path / "control")
    assert control_report.ok, control_report.summary()

    dl = _downloader(_ctx(tmp_path), _Server(),
                     artifact_store=_BrokenStore())
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()
    # Byte-identical outcome: the registration failure changed
    # nothing the caller can see.
    assert report.summary() == control_report.summary()
    assert report.size_bytes == control_report.size_bytes
    assert Path(report.path).read_bytes() == _BODY
    sites = {f.site for f in downloader_failures.recent_failures(100)}
    assert "_register_download_artifact" in sites


# ── dedup ───────────────────────────────────────────────────────


def test_identical_redownload_dedups_within_download_origin(
        tmp_path):
    store = _artifact_store(tmp_path)
    dl = _downloader(_ctx(tmp_path), _Server(), artifact_store=store)
    first = _land(dl, _URL, tmp_path / "one")
    assert first.ok, first.summary()
    second = _land(dl, "https://mirror.test/song.mp3", tmp_path / "two")
    assert second.ok, second.summary()
    assert len(_download_rows(store)) == 1
    assert store.stats()["total"] == 1


def test_same_bytes_under_another_origin_do_not_block(tmp_path):
    """Dedup is per-origin (the producer rule): bytes Devon already
    holds as a chat-inbound artifact still earn the download its
    own record — and the download's re-downloads still dedup."""
    store = _artifact_store(tmp_path)
    store.put(_BODY, type="file", creator="chat_inbound",
              metadata={"origin": "chat_inbound",
                        "filename": "song.mp3"})
    dl = _downloader(_ctx(tmp_path), _Server(), artifact_store=store)
    report = _land(dl, _URL, tmp_path / "landed")
    assert report.ok, report.summary()
    assert len(_download_rows(store)) == 1
    assert store.stats()["total"] == 2
    again = _land(dl, _URL, tmp_path / "landed2")
    assert again.ok, again.summary()
    assert len(_download_rows(store)) == 1
    assert store.stats()["total"] == 2
