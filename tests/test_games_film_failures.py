"""Tail-29 pins: games/film.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions. Every handler below is the
module's documented never-raises contract in action; each pin drives the
handler into its failure path and pins the exact fallback value plus the
fact that the agents failures ledger is unchanged (snapshot before/after,
tail-8 method). film.py calls no record function anywhere (grep-proven),
so the ledger must not move.

Grounds in brief (full case in SURVEY.md): 194/229 return honest failure
tuples that analyze_vod turns into available=False + the reason note —
never a success; 275 drops one moment's unparseable reads, and the
all-empty case is caught by analyze_vod's empty-check into an honest
unavailable breakdown (no false clean); 411 returns available=False with
a note naming the missing form coach; 516 returns an honest False (the
save-failure visibility gap is at the two CALLERS, which discard the
bool — out-of-batch, not this site); 537/561/629 return the documented
no-data values under the store's degrade-to-empty contract, and the
owner-facing "no ..." renderings are a display-layer distinction the
store API cannot express (display-pass note, tail-19/26 family — NOT a
tail-21 1201 convert, because unlike 1201 there is no sibling block in
the same function already surfacing failures); 606 returns 0 for a
failed ingest and its only caller discards the count, so nothing is
rendered from it.

The pins file itself contains no exception handlers, so it adds zero
census sites.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import failures as ag_fail
from nomorals.games import film as F


def _ledger():
    return list(ag_fail.recent_failures())


class _RaisingDB:
    def execute(self, *a, **k):
        raise RuntimeError("db down (execute)")

    def commit(self, *a, **k):
        raise RuntimeError("db down (commit)")


def _store(cls, db):
    s = object.__new__(cls)
    s._db = db
    return s


def test_scene_extraction_failure_is_honest():
    """194: ffmpeg failure -> (None, failure reason), ledger unchanged."""
    p = Path(tempfile.mkdtemp()) / "v.mp4"
    p.write_bytes(b"x")
    before = _ledger()
    orig_which, orig_run = F.shutil.which, F.subprocess.run
    try:
        F.shutil.which = lambda *a, **k: "/usr/bin/ffmpeg"
        F.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("ffmpeg down"))
        assert F._scenes_for(str(p)) == (
            None, "scene extraction failed for this video.")
    finally:
        F.shutil.which, F.subprocess.run = orig_which, orig_run
    assert _ledger() == before


def test_uniform_fallback_failure_is_honest():
    """229: frame pull failure -> (None, failure reason), ledger unchanged."""
    before = _ledger()
    orig_which, orig_run = F.shutil.which, F.subprocess.run
    try:
        F.shutil.which = lambda *a, **k: "/usr/bin/ffmpeg"
        F.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("ffmpeg down"))
        assert F._uniform_moments("v.mp4", tempfile.mkdtemp(), 5) == (
            None, "couldn't extract any frames from this video.")
    finally:
        F.shutil.which, F.subprocess.run = orig_which, orig_run
    assert _ledger() == before


def test_moment_parse_failure_drops_only_that_moment():
    """275: unparseable moment text -> ([], []), ledger unchanged."""
    before = _ledger()
    assert F._parse_moment_lines(12345) == ([], [])
    assert _ledger() == before


def test_workout_import_failure_is_unavailable():
    """411: form-coach import failure -> available=False + named note."""
    before = _ledger()
    key = "nomorals.health.form"
    had = key in sys.modules
    saved = sys.modules.get(key)
    try:
        sys.modules[key] = None  # next import raises ImportError
        bd = F._analyze_workout_vod("v.mp4", "squat", "bid1", None)
        assert bd.available is False and "form coach" in bd.note
    finally:
        if had:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)
    assert _ledger() == before


def test_store_save_failure_returns_false():
    """516: save on a dead DB -> False (honest bool), ledger unchanged."""
    before = _ledger()
    s = _store(F.FilmStore, _RaisingDB())
    bd = F.FilmBreakdown(id="t1", game="valorant", video="v.mp4",
                         mistakes=[], highlights=[], coaching="",
                         created_at=123.0)
    assert s.save(bd) is False
    assert _ledger() == before


def test_store_list_failure_is_empty():
    """537: list on a dead DB -> [] (documented empty), ledger unchanged."""
    before = _ledger()
    assert _store(F.FilmStore, _RaisingDB()).list() == []
    assert _ledger() == before


def test_store_get_failure_is_none():
    """561: get on a dead DB -> None (documented miss), ledger unchanged."""
    before = _ledger()
    assert _store(F.FilmStore, _RaisingDB()).get("nope") is None
    assert _ledger() == before


def test_fingerprint_ingest_failure_returns_zero():
    """606: ingest on a dead DB -> 0, ledger unchanged."""
    before = _ledger()
    bd = F.FilmBreakdown(id="b1", game="valorant", video="v.mp4",
                         available=True,
                         mistakes=[F.PlayNote(
                             kind="mistake",
                             text="you always overextend at minute 8")])
    assert _store(F.FingerprintStore, _RaisingDB()).ingest("u1", bd) == 0
    assert _ledger() == before


def test_fingerprint_read_failure_is_empty():
    """629: fingerprint on a dead DB -> [] (documented empty), unchanged."""
    before = _ledger()
    assert _store(F.FingerprintStore, _RaisingDB()).fingerprint("u1") == []
    assert _ledger() == before
