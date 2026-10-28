"""Tail-26 pins: audiobook.py failure values + NOT-recording.

Verdict: 1 convert / 8 exclusions. Site 630 (epub_to_audiobook's
history save failing while the build still returned ok True with a
clean success note) is the convert: the audio IS ready, so ok True
stands, but the note now says the history row was lost and gives the
master path — otherwise /audiobook status would deny the book ("no
audiobooks produced yet") and check by id could not find it, with
nothing in the reply pointing at the files. Site 713 (save's own
except -> False) stays an exclusion: the bool IS the failure signal;
the fix belongs at the caller (630), tail-23 638 shape.

Every other handler is documented never-raises (module docstring:
"refuses rather than producing fake audio or wrong metadata"); each
pin drives the handler into its except path and pins the exact
fallback value plus the fact that the agents failures ledger is
unchanged (snapshot before/after, tail-8 method). audiobook.py calls
no record function anywhere (grep-proven), so the ledger must not
move.

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.audio import audiobook as A


def _ledger():
    return list(ag_fail.recent_failures())


def _wav(tmp):
    p = Path(tmp) / "tone.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 24000)
    return p


def _epub(tmp):
    p = Path(tmp) / "book.epub"
    p.write_bytes(b"PK fake")
    return p


def _sections_doc():
    return SimpleNamespace(
        sections=[SimpleNamespace(heading="Intro", text="word " * 200,
                                  level=1)],
        metadata={})


def test_store_rules_lookup_failure_refuses():
    """142: rule lookup failure -> ok False, never a fabricated card."""
    before = _ledger()
    rules = dict(A.DISCLOSURE_RULES)
    rules.pop("version", None)
    with mock.patch.object(A, "DISCLOSURE_RULES", rules):
        r = A.store_rules("spotify")
    assert r.get("ok") is False
    assert A.store_rules("spotify").get("ok") is True
    assert A.store_rules("no-such-store").get("ok") is False
    assert _ledger() == before


def test_split_chapters_fallback_failure_returns_empty():
    """242: full_text failure -> [] (docstring: 'returns [] when the
    book cannot be parsed'). The build then fails honestly on empty
    chapters ('could not read any chapters'), never fake audio."""
    before = _ledger()
    import nomorals.documents.model as model_mod
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    epub = _epub(tmp)
    empty_doc = SimpleNamespace(sections=[], metadata={})
    with mock.patch.object(model_mod, "full_text",
                           side_effect=RuntimeError("model down")):
        assert A.split_chapters(str(epub),
                                parse_fn=lambda p: empty_doc) == []
    with mock.patch.object(model_mod, "full_text",
                           return_value="word " * 6000):
        assert len(A.split_chapters(
            str(epub), parse_fn=lambda p: empty_doc)) >= 2
    assert _ledger() == before


def test_book_title_author_failure_uses_stem():
    """274: parse failure -> (filename stem, '') — real path data,
    honestly empty author, never a fabricated author."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    epub = _epub(tmp)

    def _boom(p):
        raise RuntimeError("parse down")

    assert A._book_title_author(str(epub), parse_fn=_boom) == ("book", "")
    doc = SimpleNamespace(metadata={"title": "Real Title",
                                    "author": "Ada"})
    assert A._book_title_author(
        str(epub), parse_fn=lambda p: doc) == ("Real Title", "Ada")
    assert _ledger() == before


def test_audio_duration_failure_returns_zero():
    """330: ffprobe failure -> 0.0, byte-identical to the no-ffprobe
    branch value (tail-10 shape). The note's '0.0h' rendering on a
    real-length book is a display-layer wart, not a handler lie."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    wav = _wav(tmp)
    with mock.patch.object(A.shutil, "which", return_value=None):
        assert A._audio_duration(wav) == 0.0
    with mock.patch.object(A.subprocess, "run",
                           side_effect=RuntimeError("ffprobe down")):
        assert A._audio_duration(wav) == 0.0
    with mock.patch.object(A.subprocess, "run",
                           return_value=SimpleNamespace(stdout="123.45\n")):
        assert A._audio_duration(wav) == 123.45
    assert _ledger() == before


def test_epub_save_failure_warned_not_silent():
    """630 CONVERT: failed history save -> ok True (the audio IS
    ready) AND the note warns 'not saved' with the master path.
    Control: working save -> ok True, no warning. store=None ->
    ok True, no warning (no save attempted, no failure)."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    wav = _wav(tmp)
    epub = _epub(tmp)

    class _FailStore:
        def __init__(self):
            self.saved = []

        def save(self, book):
            self.saved.append(book)
            return False

    class _OkStore:
        def __init__(self):
            self.saved = []

        def save(self, book):
            self.saved.append(book)
            return True

    def _run(store):
        with mock.patch.object(A, "master_lufs",
                               side_effect=lambda p, **k: p), \
                mock.patch.object(A, "check_compliance",
                                  return_value={"ok": True,
                                                "verdict": "PASS"}), \
                mock.patch.object(A, "chapter_pacing",
                                  return_value={"ok": True,
                                                "verdict": "PASS",
                                                "note": "ok"}), \
                mock.patch.object(A, "export_mp3", return_value=""), \
                mock.patch("nomorals.media_edit.videos.concat",
                           return_value={"output": str(wav)}), \
                mock.patch("nomorals.audio.edit._normalize_wav",
                           side_effect=lambda p, *a, **k: p):
            return A.epub_to_audiobook(
                str(epub), {"narrator": "ref"}, ["spotify"],
                tts_fn=lambda text, ref, lang="en": str(wav),
                parse_fn=lambda p: _sections_doc(),
                out_dir=str(Path(tmp) / "out"), store=store)

    fs = _FailStore()
    res = _run(fs)
    assert res.get("ok") is True
    assert len(fs.saved) == 1
    assert "not saved" in res.get("note", "")
    assert "master file:" in res.get("note", "")
    os_store = _OkStore()
    res2 = _run(os_store)
    assert res2.get("ok") is True
    assert len(os_store.saved) == 1
    assert "not saved" not in res2.get("note", "")
    res3 = _run(None)
    assert res3.get("ok") is True
    assert "not saved" not in res3.get("note", "")
    assert _ledger() == before


def test_default_db_mkdir_failure_still_returns_path():
    """663: mkdir failure -> path returned anyway; the sqlite
    connect then fails and init logs + degrades to None (methods
    handle None). Benign pass-through, downstream handles it."""
    before = _ledger()
    with mock.patch.object(Path, "mkdir", side_effect=OSError("ro")):
        assert A._default_db().endswith("audiobooks.db")
    assert A._default_db().endswith("audiobooks.db")
    assert _ledger() == before


class _RaisingDB:
    def execute(self, *a, **k):
        raise RuntimeError("db down")

    def commit(self):
        raise RuntimeError("db down")


def _book(wav):
    return A.Audiobook(book_id="ab_test1", title="T", author="A",
                       chapters=[], master_path=str(wav),
                       duration_s=60.0, disclosure={},
                       target_stores=["spotify"], created_at=1.0)


def test_store_save_failure_returns_false():
    """701: save failure -> False, the bool failure signal the 630
    caller now checks. Control: real DB -> True."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    wav = _wav(tmp)
    store = A.AudiobookStore(str(Path(tmp) / "a.db"))
    assert store.save(_book(wav)) is True
    store._db = _RaisingDB()
    assert store.save(_book(wav)) is False
    assert _ledger() == before


def test_store_list_failure_returns_empty():
    """713: list failure -> [], byte-identical to the db-None branch
    value (tail-10 shape)."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    wav = _wav(tmp)
    store = A.AudiobookStore(str(Path(tmp) / "b.db"))
    store.save(_book(wav))
    assert len(store.list()) == 1
    store._db = _RaisingDB()
    assert store.list() == []
    assert _ledger() == before


def test_store_get_failure_returns_none():
    """724: get failure -> None, same as the db-None branch and the
    no-row value (tail-10 shape)."""
    before = _ledger()
    tmp = tempfile.mkdtemp(prefix="ab-pin-")
    wav = _wav(tmp)
    store = A.AudiobookStore(str(Path(tmp) / "c.db"))
    store.save(_book(wav))
    assert (store.get("ab_test1") or {}).get("book_id") == "ab_test1"
    assert store.get("nope") is None
    store._db = _RaisingDB()
    assert store.get("ab_test1") is None
    assert _ledger() == before
