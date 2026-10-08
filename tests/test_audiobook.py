"""Offline tests for build-map #106 — EPUB → audiobook + disclosure."""

import io
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.audio.audiobook import (  # noqa: E402
    DISCLOSURE_RULES,
    KNOWN_STORES,
    AudiobookStore,
    control_audiobook,
    epub_to_audiobook,
    master_lufs,
    split_chapters,
)


def _make_epub(path, chapters):
    """Minimal EPUB with one xhtml file per chapter."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        items = "".join(
            f'<item id="c{i}" href="c{i}.xhtml" '
            f'media-type="application/xhtml+xml"/>' for i, _ in enumerate(chapters))
        spine = "".join(f'<itemref idref="c{i}"/>' for i, _ in enumerate(chapters))
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package version="3.0" '
                   'xmlns="http://www.idpf.org/2007/opf" unique-identifier="b">'
                   f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:title>Test Book</dc:title><dc:creator>Test Author</dc:creator>'
                   '</metadata><manifest>' + items +
                   '</manifest><spine>' + spine + '</spine></package>')
        for i, (title, body) in enumerate(chapters):
            z.writestr(f"OEBPS/c{i}.xhtml",
                       '<?xml version="1.0"?><html '
                       'xmlns="http://www.w3.org/1999/xhtml"><body>'
                       f'<h1>{title}</h1><p>{body}</p></body></html>')


def _wav(path, secs=1.0, freq=440.0):
    import wave
    import struct
    import math
    n = int(24000 * secs)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        for i in range(n):
            v = int(12000 * math.sin(2 * math.pi * freq * i / 24000))
            w.writeframes(struct.pack("<h", v))
    return path


def test_chapter_split_from_headings():
    tmp = tempfile.mkdtemp()
    epub = os.path.join(tmp, "book.epub")
    _make_epub(epub, [("Chapter One", "Once upon a time."),
                      ("Chapter Two", "The story continues."),
                      ("Chapter Three", "The end.")])
    chs = split_chapters(epub)
    assert len(chs) >= 2, f"expected chapters, got {len(chs)}"
    assert chs[0].title, "chapter needs a title"
    assert all(c.text.strip() for c in chs), "chapters need text"


def test_chapter_split_missing_file():
    assert split_chapters("/nonexistent/book.epub") == []


def test_chapter_split_garbage():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "junk.epub")
    with open(p, "wb") as f:
        f.write(b"not an epub at all")
    assert split_chapters(p) == []


def test_disclosure_config_versioned():
    assert DISCLOSURE_RULES.get("version"), "rules must be versioned"
    for store in ("acx", "spotify", "kobo"):
        assert store in DISCLOSURE_RULES, f"{store} rules missing"
        cfg = DISCLOSURE_RULES[store]
        assert cfg.get("requires_disclosure") is True
        assert cfg.get("disclosure_text"), "disclosure text required"
        assert cfg.get("field"), "metadata field required"


def test_known_stores():
    assert set(KNOWN_STORES) == {"acx", "spotify", "kobo"}


def test_epub_to_audiobook_mock_tts():
    tmp = tempfile.mkdtemp()
    epub = os.path.join(tmp, "book.epub")
    _make_epub(epub, [("One", "Hello world."), ("Two", "Goodbye world.")])
    voice = _wav(os.path.join(tmp, "voice.wav"))
    out_dir = os.path.join(tmp, "out")

    def mock_tts(text, ref, lang="en"):
        p = os.path.join(tmp, f"ch_{abs(hash(text)) % 9999}.wav")
        return _wav(p, secs=0.5)

    res = epub_to_audiobook(epub, {"narrator": voice},
                            ["spotify", "kobo"], tts_fn=mock_tts,
                            out_dir=out_dir)
    if shutil_which_ffmpeg() is None:
        assert res["ok"] is False, "no ffmpeg → honest failure, not fake"
        assert "ffmpeg" in res["reason"].lower() or "master" in res["reason"].lower()
        return
    assert res["ok"] is True, f"failed: {res.get('reason')}"
    book = res["audiobook"]
    assert len(book.chapters) >= 2
    assert book.master_path and os.path.exists(book.master_path)
    assert set(book.disclosure.keys()) == {"spotify", "kobo"}
    assert book.disclosure["spotify"]["rules_version"] == DISCLOSURE_RULES["version"]
    assert "note" in res and "LUFS" in res["note"]


def shutil_which_ffmpeg():
    import shutil
    return shutil.which("ffmpeg")


def test_epub_to_audiobook_no_narrator_refuses():
    tmp = tempfile.mkdtemp()
    epub = os.path.join(tmp, "book.epub")
    _make_epub(epub, [("One", "Hello.")])
    res = epub_to_audiobook(epub, None, ["spotify"],
                            tts_fn=lambda t, r, lang="en": None)
    assert res["ok"] is False
    assert "narrator" in res["reason"].lower()


def test_epub_to_audiobook_missing_epub():
    res = epub_to_audiobook("/nope.epub", {"narrator": "x.wav"})
    assert res["ok"] is False


def test_epub_to_audiobook_tts_failure_refuses():
    tmp = tempfile.mkdtemp()
    epub = os.path.join(tmp, "book.epub")
    _make_epub(epub, [("One", "Hello.")])
    voice = _wav(os.path.join(tmp, "voice.wav"))
    res = epub_to_audiobook(epub, {"narrator": voice}, ["spotify"],
                            tts_fn=lambda t, r, lang="en": None)
    assert res["ok"] is False, "TTS failure must refuse, not fake"


def test_epub_to_audiobook_never_raises():
    res = epub_to_audiobook(None, None, None)
    assert res["ok"] is False
    res2 = epub_to_audiobook("", {"narrator": ""}, ["bogus-store"])
    assert res2["ok"] is False


def test_unknown_store_defaults():
    tmp = tempfile.mkdtemp()
    epub = os.path.join(tmp, "book.epub")
    _make_epub(epub, [("One", "Hello.")])
    voice = _wav(os.path.join(tmp, "voice.wav"))
    res = epub_to_audiobook(epub, {"narrator": voice}, ["bogus"],
                            tts_fn=lambda t, r, lang="en": _wav(
                                os.path.join(tmp, "x.wav"), 0.3))
    if res["ok"]:
        # unknown store dropped, sensible default applied
        assert "spotify" in res["audiobook"].target_stores


def test_master_lufs_missing_ffmpeg_or_file():
    assert master_lufs("/nonexistent.wav") is None
    tmp = tempfile.mkdtemp()
    w = _wav(os.path.join(tmp, "a.wav"))
    out = master_lufs(w)
    if shutil_which_ffmpeg() is None:
        assert out is None
    else:
        assert out and os.path.exists(out)


def test_store_roundtrip():
    tmp = tempfile.mkdtemp()
    s = AudiobookStore(db_path=os.path.join(tmp, "ab.db"))
    from nomorals.audio.audiobook import Audiobook
    book = Audiobook(book_id="ab_test1", title="T", author="A",
                     master_path="/x.wav", duration_s=60.0,
                     target_stores=["spotify"], created_at=1.0)
    assert s.save(book) is True
    books = s.list()
    assert any(b["book_id"] == "ab_test1" for b in books)
    got = s.get("ab_test1")
    assert got and got["title"] == "T"
    assert s.get("nope") is None


def test_chat_status_empty():
    out = control_audiobook("status")
    assert isinstance(out, str)


def test_chat_usage():
    out = control_audiobook("")
    assert "audiobook" in out.lower()


def test_chat_make_missing_file():
    out = control_audiobook("make /nonexistent.epub narrator=x.wav")
    assert "couldn't" in out.lower() or "no epub" in out.lower()


def test_chat_make_parses_stores_and_voices():
    # garbage EPub → honest failure, but parsing must not raise
    out = control_audiobook("make /nonexistent.epub narrator=v.wav acx")
    assert isinstance(out, str)


def test_chat_never_raises():
    for tail in [None, "", "make", "make   ", "bogus command xyz"]:
        out = control_audiobook(tail)
        assert isinstance(out, str) and out


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
