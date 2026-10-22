"""Pass #87 — successful downloads teach the registry.

Pass #86 wired the media/download source registry live (outcomes,
persistence, candidate discovery + health-loop promotion) but its
report's first weakness was explicit: a promoted candidate joins with
``extraction_rule=""``, so its downloads still flow through the
yt-dlp generic / direct structural fallbacks — promotion granted
trust and ordering, not a learned extractor rule.

Survey truth this suite pins:

* The registry's rule representation is per-source and frozen:
  ``SourceDescriptor.extraction_rule`` (one string) +
  ``quality_map`` (label → yt-dlp format), hand-written for seeds.
  It cannot hold stage / extractor / evidence counts, so learning
  extends the registry minimally with a separate LEARNED slot per
  source key; the descriptor's hand-written facts are never touched
  and outrank the learned slot wherever both exist.
* The kit's ``ExtractorRegistry`` is a derived view
  (``descriptors_for_registry``) — it now derives from the registry's
  EFFECTIVE facts (hand-written first, learned second), so a learned
  rule is visible through the view with no kit-side fork.
* At the downloader's ``_succeed`` seam the facts actually exercised
  are: the winning stage (direct | kit | proxy | browser — the
  chain's real vocabulary), ``report.extractor`` (the backend that
  landed the file: ``yt_dlp`` / ``yt-dlp-cli`` / ``direct`` from
  ``tools.media.download``, ``download-kit``, ``browser-session`` —
  the per-site yt-dlp extractor key is NOT propagated by the direct
  stage, so it is never invented here), and — for the stages that
  receive one — the quality label → format translation the
  descriptor performed in ``download()``.
* Rule degradation rides #86's health vocabulary: ``STAND_DOWN_AFTER``
  consecutive failures through a learned rule stop it being
  preferred; ``DISABLE_AFTER`` clears the slot.

All fetches are stubbed — no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


# ── helpers ───────────────────────────────────────────────────────────


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ctx(tmp_path, db=None):
    return SimpleNamespace(
        db=db,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _promoted(reg, host="freshbeats.example.org"):
    """A grown source the way #86 grows one: candidate → health loop."""
    assert reg.note_candidate(f"https://{host}/song", via="test") is True
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1, summary
    desc = reg.for_domain(host)
    assert desc is not None and desc.extraction_rule == ""
    return desc


def _landed(extractor="yt_dlp"):
    def _dl(url, dest, **kw):
        p = Path(dest) / "song.mp3"
        p.write_bytes(b"ID3" + b"y" * 32)
        return {"url": url, "path": str(p),
                "bytes": p.stat().st_size, "title": "Song",
                "extractor": extractor, "seconds": 0.1}
    return _dl


def _kit_fake(calls, ok=True):
    import hashlib

    from nomorals.media.download_kit import ResumeReport

    def _download(self, url, dest, **kw):
        calls.append(url)
        if not ok:
            return ResumeReport(error="fetch failed: boom (stubbed)")
        data = b"ID3" + b"x" * 64
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        return ResumeReport(ok=True, path=str(dest),
                            size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest())
    return _download


# ── (1) learning writes the exercised facts into a learned slot ──────


def test_learn_rule_is_visible_through_the_derived_view(tmp_path):
    from nomorals.media.download_kit import ExtractorRegistry

    reg = _seeded()
    desc = _promoted(reg)
    url = "https://freshbeats.example.org/track/123"
    # before learning: only the yt-dlp generic structural fallback
    before = ExtractorRegistry(registry=reg).best_for(url)
    assert before is not None and before.name == "ytdlp-generic"

    assert reg.learn_rule(
        desc, stage="kit", extractor="download-kit",
        extraction_rule="direct") is True

    view = ExtractorRegistry(registry=reg)
    after = view.best_for(url)
    assert after is not None and after.name == "freshbeats.example.org"
    assert after.extraction_rule == "direct"
    assert after.downloadable is True


def test_learned_quality_map_fills_only_the_empty_slot(tmp_path):
    from nomorals.media.download_kit import ExtractorRegistry

    reg = _seeded()
    desc = _promoted(reg)
    url = "https://freshbeats.example.org/watch/9"
    generic = ExtractorRegistry(registry=reg).best_for(url)
    assert generic.format_for("480p") == ""  # generic map has no 480p

    reg.learn_rule(desc, stage="direct", extractor="yt_dlp",
                   extraction_rule="ytdlp", quality_label="480p",
                   format_spec="best[height<=480]")
    view_desc = ExtractorRegistry(registry=reg).best_for(url)
    assert view_desc.name == "freshbeats.example.org"
    assert view_desc.format_for("480p") == "best[height<=480]"
    rule = reg.learned_rule(desc)
    assert rule.stage == "direct" and rule.extractor == "yt_dlp"
    assert rule.successes == 1 and rule.consecutive_failures == 0
    assert rule.is_active


def test_seed_rule_outranks_learned_and_is_never_overwritten(tmp_path):
    from nomorals.media.download_kit import ExtractorRegistry

    reg = _seeded()
    yt = reg.for_url("https://www.youtube.com/watch?v=abc")
    assert yt.extraction_rule == "ytdlp"
    # learning records evidence even for a seed source…
    assert reg.learn_rule(yt, stage="kit", extractor="download-kit",
                          extraction_rule="direct",
                          quality_label="720p",
                          format_spec="best[height<=720]") is True
    assert reg.learned_rule(yt) is not None
    # …but the seed's hand-written facts stay exactly in charge
    assert yt.extraction_rule == "ytdlp"
    view_desc = ExtractorRegistry(registry=reg).best_for(
        "https://www.youtube.com/watch?v=abc")
    assert view_desc.name == "youtube"
    assert view_desc.extraction_rule == "ytdlp"
    assert view_desc.format_for("720p") == "best[height<=720]"
    assert reg.effective_extraction_rule(yt) == "ytdlp"


def test_learning_refuses_candidates_and_unknown_sources(tmp_path):
    from nomorals.media.source_registry import (
        Licence, SourceDescriptor, TrustTier)

    reg = _seeded()
    assert reg.note_candidate("https://cand.example.org/x",
                              via="test") is True
    cand = reg.candidates()[0]
    assert reg.learn_rule(cand, stage="direct", extractor="yt_dlp",
                          extraction_rule="ytdlp") is False
    ghost = SourceDescriptor(
        nation="NG", language="en", domain="ghost.example.org",
        name="Ghost", kinds=("music",), licence=Licence.UNKNOWN,
        trust_tier=TrustTier.UNVERIFIED)
    assert reg.learn_rule(ghost, stage="direct", extractor="yt_dlp",
                          extraction_rule="ytdlp") is False
    assert reg.learn_rule(cand, stage="", extractor="",
                          extraction_rule="") is False


# ── (4) evidence discipline: degrade → stand down → clear ────────────


def test_rule_degrades_and_clears_on_health_thresholds(tmp_path):
    from nomorals.media.source_registry import (
        DISABLE_AFTER, STAND_DOWN_AFTER)

    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    assert reg.learned_for_url(
        "https://freshbeats.example.org/track/1") is not None

    for _ in range(STAND_DOWN_AFTER - 1):
        assert reg.note_rule_failure(desc) == "active"
    assert reg.learned_for_url(
        "https://freshbeats.example.org/track/1") is not None
    # the STAND_DOWN_AFTER-th consecutive failure stops it being
    # preferred — the slot (and its evidence) is still on record
    assert reg.note_rule_failure(desc) == "stood-down"
    assert reg.learned_for_url(
        "https://freshbeats.example.org/track/1") is None
    rule = reg.learned_rule(desc)
    assert rule is not None and rule.status == "stood-down"
    assert rule.consecutive_failures == STAND_DOWN_AFTER
    # further failures clear the slot entirely at DISABLE_AFTER
    state = ""
    for _ in range(DISABLE_AFTER - STAND_DOWN_AFTER):
        state = reg.note_rule_failure(desc)
    assert state == "cleared"
    assert reg.learned_rule(desc) is None
    assert reg.note_rule_failure(desc) == ""


def test_success_through_rule_resets_the_failure_streak(tmp_path):
    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    reg.note_rule_failure(desc)
    reg.note_rule_failure(desc)
    rule = reg.learned_rule(desc)
    assert rule.consecutive_failures == 2 and rule.failures == 2
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    rule = reg.learned_rule(desc)
    assert rule.consecutive_failures == 0 and rule.successes == 2
    # new evidence supersedes: a different winning stage replaces
    reg.learn_rule(desc, stage="browser", extractor="browser-session",
                   extraction_rule="scrape")
    rule = reg.learned_rule(desc)
    assert rule.stage == "browser" and rule.successes == 1
    assert rule.consecutive_failures == 0


# ── (5) persistence + status ──────────────────────────────────────────


def test_learned_rules_persist_across_restart(tmp_path):
    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="direct", extractor="yt_dlp",
                   extraction_rule="ytdlp", quality_label="720p",
                   format_spec="best[height<=720]")
    reg.note_rule_failure(desc)

    reg2 = _seeded(store=kv)
    desc2 = reg2.for_domain("freshbeats.example.org")
    assert desc2 is not None
    rule = reg2.learned_rule(desc2)
    assert rule is not None, reg2.status()
    assert rule.stage == "direct" and rule.extractor == "yt_dlp"
    assert rule.quality_label == "720p"
    assert rule.format_spec == "best[height<=720]"
    assert rule.successes == 1 and rule.failures == 1
    assert rule.consecutive_failures == 1 and rule.is_active


def test_v1_persisted_state_still_loads(tmp_path):
    import json as _json

    from nomorals.media.source_registry import _PERSIST_KEY

    kv, _db = _kv(tmp_path)
    reg = _seeded(store=kv)
    yt = reg.for_url("https://www.youtube.com/watch?v=abc")
    reg.record_outcome(yt, ok=True, latency_s=1.0)
    # rewind the stored payload to the #86 shape (version 1, no
    # learned slot) — it must load exactly as #86 left it
    raw = kv.get(_PERSIST_KEY)
    payload = _json.loads(raw) if isinstance(raw, str) else dict(raw)
    assert payload["version"] == 2  # #87 writes the grown shape
    payload["version"] = 1
    payload.pop("learned", None)
    kv.set_raw(_PERSIST_KEY, _json.dumps(payload), "json")

    reg2 = _seeded(store=kv)
    yt2 = reg2.for_url("https://www.youtube.com/watch?v=abc")
    assert reg2.stats_for(yt2)["ok"] == 1
    assert reg2.learned_rule(yt2) is None


def test_status_shows_learned_rules_distinctly_from_seed_rules(tmp_path):
    from nomorals.media.source_registry import render_status_text

    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct", quality_label="audio",
                   format_spec="bestaudio/best")
    st = reg.status()
    assert st["learned_count"] == 1, st
    src = [s for s in st["sources"]
           if s["domain"] == "freshbeats.example.org"][0]
    assert src["extraction_rule"] == ""  # no hand-written (seed) rule
    learned = src["learned_rule"]
    assert learned is not None
    assert learned["stage"] == "kit"
    assert learned["extractor"] == "download-kit"
    assert learned["extraction_rule"] == "direct"
    assert learned["quality_label"] == "audio"
    assert learned["format_spec"] == "bestaudio/best"
    assert learned["successes"] == 1 and learned["status"] == "active"
    yt = [s for s in st["sources"] if s["domain"] == "youtube.com"][0]
    assert yt["extraction_rule"] == "ytdlp" and yt["learned_rule"] is None

    text = render_status_text(st)
    assert "learned rule" in text, text
    assert "freshbeats.example.org" in text
    assert "stage=kit" in text and "download-kit" in text
    assert "audio→bestaudio/best" in text


# ── (1)+(3) downloader integration: learn, then take the learned path ─


def test_successful_download_teaches_the_registry(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    desc = _promoted(reg)
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg)
    with mock.patch("nomorals.tools.media.download",
                    side_effect=_landed()):
        report = dl.download("https://freshbeats.example.org/watch/9",
                             dest_dir=tmp_path, format_spec="720p")
    assert report.ok, report.summary()
    rule = reg.learned_rule(desc)
    assert rule is not None, reg.status()
    assert rule.stage == "direct" and rule.extractor == "yt_dlp"
    assert rule.extraction_rule == "ytdlp"
    # the translation that landed: the generic descriptor's 720p map,
    # now recorded against THIS source
    assert rule.quality_label == "720p"
    assert rule.format_spec == "best[height<=720]"


def test_second_download_takes_the_learned_kit_path_first(tmp_path):
    from nomorals.media.download_kit import ExtractorRegistry
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    desc = _promoted(reg)
    url = "https://freshbeats.example.org/files/track.mp3"
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _kit_fake(kit_calls)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        first = dl.download(url, dest_dir=tmp_path)
    assert first.ok and first.tried()[0] == "direct", first.summary()
    rule = reg.learned_rule(desc)
    assert rule is not None and rule.stage == "kit"
    assert rule.extraction_rule == "direct"
    # the learned descriptor now leads the derived view for this URL
    view_desc = ExtractorRegistry(registry=reg).best_for(url)
    assert view_desc.name == "freshbeats.example.org"

    # second download: the learned kit path goes FIRST — the direct
    # stage is never even attempted when the kit lands it
    direct_calls: list = []

    def _direct_spy(url, dest, **kw):
        direct_calls.append(url)
        raise RuntimeError("must not be reached")

    kit_calls.clear()
    with mock.patch("nomorals.tools.media.download",
                    side_effect=_direct_spy), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _kit_fake(kit_calls)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        second = dl.download(url, dest_dir=tmp_path)
    assert second.ok, second.summary()
    assert kit_calls and not direct_calls
    assert second.tried()[0] == "kit", second.stages


def test_learned_format_translation_serves_the_next_download(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    _promoted(reg)
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg)
    url = "https://freshbeats.example.org/watch/9"
    with mock.patch("nomorals.tools.media.download",
                    side_effect=_landed()):
        assert dl.download(url, dest_dir=tmp_path,
                           format_spec="720p").ok
    seen: dict = {}

    def capture(url, dest, **kw):
        seen.update(kw)
        return _landed()(url, dest, **kw)

    with mock.patch("nomorals.tools.media.download",
                    side_effect=capture):
        report = dl.download(url, dest_dir=tmp_path, format_spec="720p")
    assert report.ok, report.summary()
    assert seen.get("format_spec") == "best[height<=720]"
    # and the descriptor that translated it is the source's own
    # learned one, ahead of the generic fallback
    from nomorals.media.download_kit import ExtractorRegistry
    assert ExtractorRegistry(registry=reg).best_for(url).name == \
        "freshbeats.example.org"


def test_failed_learned_path_falls_back_in_the_same_attempt(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _kit_fake(kit_calls, ok=False)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://freshbeats.example.org/f/t.mp3",
                             dest_dir=tmp_path)
    # no regression: the whole generic chain still ran in this attempt
    assert not report.ok
    assert kit_calls, "learned kit path was not attempted first"
    assert report.tried()[0] == "kit", report.stages
    assert "direct" in report.tried(), report.stages
    # the failure was recorded ONCE against the rule
    rule = reg.learned_rule(desc)
    assert rule is not None and rule.consecutive_failures == 1
    assert rule.failures == 1


def test_stood_down_rule_is_no_longer_preferred(tmp_path):
    from nomorals.media.downloader import MediaDownloader
    from nomorals.media.source_registry import STAND_DOWN_AFTER

    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    for _ in range(STAND_DOWN_AFTER):
        reg.note_rule_failure(desc)
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _kit_fake(kit_calls, ok=False)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://freshbeats.example.org/f/t.mp3",
                             dest_dir=tmp_path)
    assert not report.ok
    # the chain starts from the generic head again — kit is only the
    # canonical direct-media rung now, after direct
    assert report.tried()[0] == "direct", report.stages


def test_learned_rule_never_bypasses_the_descriptor_gate(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    spotify = reg.for_url("https://open.spotify.com/track/abc")
    reg.learn_rule(spotify, stage="direct", extractor="yt_dlp",
                   extraction_rule="ytdlp")
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    with mock.patch("nomorals.tools.media.download") as md, \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://open.spotify.com/track/abc",
                             dest_dir=tmp_path)
    assert md.call_count == 0
    assert not report.ok
    assert "download" in (report.likely_cause + report.hint).lower()


def test_learned_rule_never_bypasses_disk_preflight(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    reg = _seeded()
    desc = _promoted(reg)
    reg.learn_rule(desc, stage="kit", extractor="download-kit",
                   extraction_rule="direct")
    dl = MediaDownloader(_ctx(tmp_path), source_registry=reg,
                         direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download") as md, \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _kit_fake(kit_calls)), \
            mock.patch("nomorals.media.download_kit.shutil.disk_usage",
                       return_value=(10**12, 10**12, 0)):
        report = dl.download("https://freshbeats.example.org/f/t.mp3",
                             dest_dir=tmp_path)
    assert md.call_count == 0 and not kit_calls
    assert not report.ok
    assert "disk" in report.summary().lower()
