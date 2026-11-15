"""Source-retry bound (owner flag 2026-11-15): one source must not be
hammered while the other sources sit unused.

The live message was ``source 1 · attempt 22`` on a `/play` whose first
source kept failing. Two compounding causes, both fixed here:

* a chain-stage switch counted as an "attempt" (five switches per
  download), so a handful of failing downloads read as 22 tries;
* the registry journey ran all candidates under ONE source number
  (`manage_sources=False`), so the run never visibly moved on.

The bound: a source gets its opening try plus up to 2 retries
(`MAX_ATTEMPTS_PER_SOURCE = 3`); after that the run moves to the next
source, and a repeat download for the same exhausted source refuses
fast instead of running another full chain. A route switch resets the
transfer measurements without counting as an attempt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from nomorals.core.progress_events import (
    MAX_ATTEMPTS_PER_SOURCE,
    ProgressTracker,
    bind_tracker,
)
from nomorals.media.downloader import MediaDownloader
from nomorals.media.resolver import ResolvedAudio


def _tracker() -> ProgressTracker:
    return ProgressTracker(op_id="retry-bound", title="Abefeholics")


def _cand(kind, url):
    return ResolvedAudio(ok=True, path_or_url=url, title="Song",
                         artist="Artist", duration=0.0, kind=kind)


class _FakeEngine:
    """PlaybackEngine seam: download(item) from a per-URL table."""

    def __init__(self, results):
        self._results = results

    def download(self, item):
        return self._results[item["path"]]


# ── the counter: route switches are not attempts ──────────────────────


def test_route_switch_resets_measurements_without_counting():
    tr = _tracker()
    tr.begin_source("youtube")
    tr.report_bytes(512, 1024)
    assert tr.attempt == 1
    tr.begin_route("route: kit")
    assert tr.attempt == 1
    snap = tr.snapshot()
    assert snap["bytes_received"] == 0
    assert snap["bytes_expected"] is None
    assert snap["note"] == "route: kit"


def test_begin_attempt_still_counts_a_real_retry():
    tr = _tracker()
    tr.begin_source("youtube")
    tr.begin_attempt("retrying the direct route")
    assert tr.attempt == 2


def test_new_source_restarts_the_attempt_count():
    tr = _tracker()
    tr.begin_source("youtube")
    tr.begin_attempt("retrying the direct route")
    assert tr.attempt == 2
    tr.begin_source("soundcloud")
    assert tr.attempt == 1
    assert tr.source_index == 2


# ── one failing download costs at most the bound ──────────────────────


def test_one_failing_download_costs_one_attempt_when_no_retry():
    # No direct retry is possible here (the unroutable URL fails as a
    # site error, not a network failure): the only attempts a download
    # may spend are real retries, so a five-stage chain costs one.
    tr = _tracker()
    with bind_tracker(tr):
        dl = MediaDownloader(context=None, direct_retries=1)
        report = dl.download("https://example.invalid/song.mp3",
                             audio_only=True, timeout=30.0)
    assert not report.ok
    assert tr.attempt == 1
    assert tr.attempt <= MAX_ATTEMPTS_PER_SOURCE


def test_direct_retries_count_but_never_pass_the_bound():
    tr = _tracker()
    with bind_tracker(tr):
        dl = MediaDownloader(context=None, direct_retries=3)
        with mock.patch("nomorals.tools.media.download",
                        side_effect=ConnectionError("conn reset")):
            report = dl.download("https://example.invalid/song.mp3",
                                 audio_only=True, timeout=30.0)
    assert not report.ok
    # Opening try + 2 retries: the bound, never more — even though
    # the chain still runs every stage (no route was removed).
    assert tr.attempt == MAX_ATTEMPTS_PER_SOURCE
    assert [s.stage for s in report.stages].count("direct") >= 1


def test_second_download_for_the_same_exhausted_source_refuses_fast():
    tr = _tracker()
    with bind_tracker(tr):
        dl = MediaDownloader(context=None, direct_retries=3)
        with mock.patch("nomorals.tools.media.download",
                        side_effect=ConnectionError("conn reset")):
            first = dl.download("https://example.invalid/song.mp3",
                                audio_only=True, timeout=30.0)
            assert not first.ok
            assert tr.attempt == MAX_ATTEMPTS_PER_SOURCE
            # Same tracker, same source, no begin_source between the
            # calls: the source already burned its tries.
            second = dl.download("https://example.invalid/song.mp3",
                                 audio_only=True, timeout=30.0)
    assert not second.ok
    assert tr.attempt == MAX_ATTEMPTS_PER_SOURCE
    assert "already used its 3 attempts" in (second.likely_cause or "")
    # The refusal ran no chain: no fresh direct attempt was spent.
    assert not [s for s in second.stages if s.stage == "direct"]


# ── the journey advances the source per candidate ─────────────────────


def test_registry_journey_advances_the_source_per_candidate(tmp_path):
    from nomorals.music_sources.live import live_collaborators
    from nomorals.music_sources.registry import by_id

    cands = [_cand("soundcloud", f"https://soundcloud.example/t{i}")
             for i in range(3)]
    engine = _FakeEngine({c.path_or_url: {"ok": False, "reason": "down"}
                          for c in cands})
    resolver = mock.Mock()
    resolver.search_candidates.return_value = list(cands)
    _search_fn, download_fn = live_collaborators(SimpleNamespace())
    tr = _tracker()
    with mock.patch("nomorals.media.resolver.SourceResolver",
                    lambda ctx: resolver), \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       lambda ctx: engine), \
            bind_tracker(tr):
        result = download_fn(by_id("soundcloud"), "Song")
    assert not result.downloaded
    # One source per candidate: the run visibly uses all the sources,
    # each starting its own attempt count.
    assert tr.source_index == 3
    assert tr.attempt == 1
    # The failure names the sources tried, once.
    assert "no route delivered the track" in result.detail
    assert result.detail.count("soundcloud") == 3


# ── a failing source is marked down ───────────────────────────────────


def test_three_consecutive_failures_stand_the_source_down(tmp_path):
    from nomorals.media.source_registry import _seed
    from nomorals.music_sources import registry_bridge as bridge
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    reg = _seed(store=KVStore(db))
    # The music rows are not in the seed: they fold in through the
    # bridge (the same fold play_via_registry runs before recording).
    assert bridge.ensure_music_sources(reg) > 0
    desc = bridge.music_source_desc(reg, "soundcloud")
    assert desc is not None
    assert bridge.is_source_usable(reg, "soundcloud")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="network_failure")
    assert not bridge.is_source_usable(reg, "soundcloud")
    # A success re-activates it: stood down is not dead.
    reg.record_outcome(desc, True)
    assert bridge.is_source_usable(reg, "soundcloud")
