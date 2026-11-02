"""Tail-99 pins: nomorals/books/serial.py census sites.

The serial orchestrator's law is printed in its header: drift
is "told, not blocking", a story decides its own shape, and
missing pieces degrade the measurement — never the story.
Three sites are that law's machinery and are held as
exclusions; the fourth is pinned AND flagged (below):

52 — ``_read_chapter_text()``: any read failure -> "" ("missing
file, honest empty"). The empty string is a 0-word text: it can
seed no voiceprint and fake no drift score, and both callers
(``write_next`` chapter capture, reference building) treat
emptiness as absence.
152 — ``_reference()``: a stored reference voiceprint that can
no longer be parsed -> rebuild from the story's own first
chapters and save the fresh print. The corrupt value is by
definition unreadable; nothing readable is lost, and the
rebuilt print is genuinely fingerprinted from real chapter
text (pinned by round-trip below).
311 — ``continue_past_ending()``: a reference chapter that
cannot be read is skipped; the reference is fingerprinted
from the chapters that did read. If none read, the reference
is None and every continued chapter reports ``drift: None`` —
the gap is on the face of the result.

81 — ``_load_meta()``: a corrupt serial.json -> a fresh default
meta. ⚠ FLAGGED — DATA-LOSS SHAPE, targeted-fix candidate, NOT
a batch convert (protocol: save/persist swallows read back
later leave the batch). The load alone loses nothing, but
every SerialManager flow pairs the load with ``_save_meta``
(start_serial, start_novel, _reference, write_next), so the
first save after a corruption permanently overwrites whatever
the file still held — drift history, wisdom motifs, the
``for:`` field — with the fresh default. The pin below proves
the mechanism exactly as it stands today; the fix (quarantine
the corrupt file / refuse to overwrite an unparseable meta)
belongs to the targeted-fix pass.

NOT-recording is pinned with the snapshot method: serial.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.books.serial as SER
from nomorals.books.voiceprint import Voiceprint, fingerprint


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _manager(root: Path) -> SER.SerialManager:
    mgr = object.__new__(SER.SerialManager)
    mgr.fiction = SimpleNamespace(story_dir=lambda slug: root / slug)
    return mgr


# ── 52: unreadable chapter -> honest empty ──────────────────────

def test_52_unreadable_chapter_is_empty(tmp_path):
    before = _snap()
    assert SER._read_chapter_text(tmp_path / "nope.md") == ""
    assert SER._read_chapter_text(tmp_path) == ""  # a directory
    # control: a real chapter reads, title line stripped
    ch = tmp_path / "ch.md"
    ch.write_text("# My Title\n\nline one\nline two\n")
    assert SER._read_chapter_text(ch) == "line one\nline two"
    assert _snap() == before


# ── 81: corrupt serial.json -> fresh default (FLAGGED) ──────────

def test_81_corrupt_meta_loads_fresh_and_save_overwrites(tmp_path):
    before = _snap()
    mgr = _manager(tmp_path)
    (tmp_path / "s1").mkdir(parents=True)
    meta_file = tmp_path / "s1" / "serial.json"
    # a truncated write: unparseable, but the drift entry is
    # visibly still in the bytes before any save happens
    meta_file.write_text(
        '{"slug": "s1", "drift_log": [{"mark": "KEEPME"}], "wis')
    loaded = mgr._load_meta("s1")
    assert loaded == {"slug": "s1", "drift_log": [],
                      "wisdom_motifs": []}
    # the flagged mechanism: the next save makes the loss durable
    mgr._save_meta("s1", loaded)
    assert "KEEPME" not in meta_file.read_text()
    # control: a parseable meta loads whole, nothing dropped
    meta_file.write_text(json.dumps(
        {"slug": "s1", "drift_log": [{"mark": "KEEPME"}], "extra": 3}))
    assert mgr._load_meta("s1")["drift_log"] == [{"mark": "KEEPME"}]
    assert mgr._load_meta("s1")["extra"] == 3
    assert _snap() == before


# ── 152: corrupt stored voiceprint -> rebuilt from chapters ─────

def test_152_corrupt_reference_voiceprint_is_rebuilt(tmp_path):
    before = _snap()
    mgr = _manager(tmp_path)
    (tmp_path / "s1").mkdir(parents=True)
    meta_file = tmp_path / "s1" / "serial.json"
    meta_file.write_text(json.dumps(
        {"slug": "s1", "reference_voiceprint": ["corrupt"]}))
    texts = ["alpha beta gamma delta epsilon zeta eta theta iota"]
    ref = mgr._reference("s1", texts)
    assert ref.words == fingerprint(texts[0]).words == 9
    # the rebuilt print was saved and round-trips
    saved = mgr._load_meta("s1")
    assert Voiceprint.from_dict(
        saved["reference_voiceprint"]).words == 9
    # control: a valid stored print is returned as-is, file untouched
    good = fingerprint("one two three four five")
    meta_file.write_text(json.dumps(
        {"slug": "s1", "reference_voiceprint": good.to_dict()}))
    bytes_before = meta_file.read_text()
    ref = mgr._reference("s1", ["totally different words here now"])
    assert ref.words == 5
    assert meta_file.read_text() == bytes_before
    assert _snap() == before


# ── 311: unreadable reference chapters are skipped ──────────────

def _continue_manager(chapter_text) -> SER.SerialManager:
    mgr = object.__new__(SER.SerialManager)
    story = SimpleNamespace(slug="s1", title="T", current_chapter=9)
    mgr.reader = SimpleNamespace(
        _load=lambda slug: story, chapter_text=chapter_text)
    mgr.continuer = SimpleNamespace(
        prepare=lambda slug: {"bible": 1},
        continue_story=lambda slug, n, direction, from_chapter: {
            "chapters": [{"n": 10, "text":
                          "red orange yellow green blue indigo violet"}],
            "continued_from_chapter": 9, "bible": {"bible": 1}})
    return mgr


def test_311_reference_skips_unreadable_chapters():
    before = _snap()

    def partial(slug, num):
        if num == 2:
            raise RuntimeError("chapter gone")
        return {1: "red orange yellow green blue indigo violet",
                3: "black white gray silver gold copper iron"}[num]

    out = _continue_manager(partial).continue_past_ending("s1", 1)
    # reference built from chapters 1+3 -> drift is really scored
    assert out["chapters"][0]["drift"] is not None
    # all reference chapters unreadable -> no reference, and the
    # result says so on its face: drift None, never a fake score
    out = _continue_manager(_raise).continue_past_ending("s1", 1)
    assert out["chapters"][0]["drift"] is None
    assert _snap() == before
