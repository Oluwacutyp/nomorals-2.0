"""Tail-55 pins for nomorals/workspace/inbox.py —
0 converts / 8 exclusions (236, 249, 294, 540, 629, 676, 893, 1304).

8 tests for 8 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

inbox.py calls no failures-ledger record function anywhere
(grep-proven: no `record_` call in the file), so NOT-recording
is pinned against the module's own observability surface: the
inbox DB state (inbox_items / inbox_history tables) snapshot
before/after each trigger — no failure row written, no item
error set, no table mutated beyond the site's intended
operation.

Verdict notes (full grounds in the tail-55 SURVEY.md):
- 236: html_to_text parser-feed failure -> partial text; the
  parts collected before the failure are genuine extracted
  text, the title is extracted independently, and stock
  HTMLParser.feed does not raise on malformed HTML.
- 249: default_fetch decode failure -> ""; bytes.decode with
  errors="replace" cannot raise on real bytes, and "" is the
  same value a genuinely empty response yields.
- 294: parse_when non-ISO ValueError -> pass is the designed
  branch into the remaining format parsers; every returned
  value is genuinely computed by them.
- 540: _add_column ALTER failure -> pass; the anticipated
  failure is the duplicate-column case of an idempotent
  migration (the normal second-startup path).
- 629: _inbox_size per-file stat failure -> file skipped; a
  file vanishing mid-scan has no knowable size, and the cap
  verdict is re-measured on every registration.
- 676: intake rmdir failure -> source dir left in place; the
  anticipated cause is a non-empty dir (subdirs are never
  moved by design), and registration is unaffected.
- 893: purge_done unlink failure -> pass; `files` counts only
  successful unlinks and `rows` counts rows actually deleted,
  so both returned counts are literally true.
- 1304: sweep directive-probe failure -> d=None; only the
  ordering heuristic is lost — process() re-runs the probe
  per item and fails the item honestly if it raises there.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import nomorals.workspace.inbox as ib
from nomorals.storage.db import Database
from nomorals.workspace.inbox import Inbox


def _snap(inbox):
    return {
        "items": inbox.db.query("SELECT * FROM inbox_items ORDER BY id"),
        "history": inbox.db.query("SELECT * FROM inbox_history ORDER BY id"),
    }


# ── 236: html_to_text feed failure -> partial text ─────────────────────

def test_236_html_to_text_feed_failure_returns_partial_text(tmp_path):
    inbox = Inbox(tmp_path / "root")
    before = _snap(inbox)
    feed_log = []

    class RaisingExtractor(ib._TextExtractor):
        def feed(self, html):
            feed_log.append(html)
            self.handle_data("partial words")
            raise RuntimeError("feed boom")

    html = "<title>My Title</title><p>partial words and more text</p>"
    with mock.patch.object(ib, "_TextExtractor", RaisingExtractor):
        title, body = ib.html_to_text(html)
    assert feed_log == [html[:500_000]]
    assert title == "My Title"
    assert body == "partial words"
    # control: the healthy extractor yields the full text
    title2, body2 = ib.html_to_text(html)
    assert title2 == "My Title"
    assert "partial words and more text" in body2
    assert _snap(inbox) == before


# ── 249: default_fetch decode failure -> "" ────────────────────────────

def test_249_default_fetch_decode_failure_returns_empty(tmp_path):
    inbox = Inbox(tmp_path / "root")
    before = _snap(inbox)
    decode_log = []

    class BadBytes(bytes):
        def decode(self, *a, **k):
            decode_log.append((a, k))
            raise UnicodeDecodeError("utf-8", b"x", 0, 1, "boom")

    class Resp:
        def __init__(self, raw):
            self._raw = raw

        def read(self, n=-1):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return Resp(BadBytes(b"\xff\xfe garbage"))

    with mock.patch.object(ib.urllib.request, "urlopen", fake_urlopen):
        out = ib.default_fetch("https://example.test/page")
    assert out == ""
    assert calls == ["https://example.test/page"]
    assert decode_log  # the decode was actually attempted

    def ok_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return Resp(b"plain body text")

    with mock.patch.object(ib.urllib.request, "urlopen", ok_urlopen):
        assert ib.default_fetch("https://example.test/plain") == "plain body text"

    def html_urlopen(req, timeout=None):
        return Resp(b"<html><head><title>T</title></head>"
                    b"<body>hello body</body></html>")

    with mock.patch.object(ib.urllib.request, "urlopen", html_urlopen):
        # (the extractor also surfaces the <title> text as a body part)
        assert ib.default_fetch("https://example.test/html") == "T\n\nT hello body"
    assert _snap(inbox) == before


# ── 294: parse_when non-ISO -> fall through to other formats ───────────

def test_294_parse_when_non_iso_falls_through_to_other_formats(tmp_path):
    inbox = Inbox(tmp_path / "root")
    before = _snap(inbox)
    calls = []

    class LoggingDatetime(datetime):
        @classmethod
        def fromisoformat(cls, s):
            calls.append(s)
            return super().fromisoformat(s)

    now = datetime(2026, 10, 29, 12, 0, 0).timestamp()  # a Thursday
    base = datetime.fromtimestamp(now)
    assert base.weekday() == 3
    expected_fri = (base + timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0).timestamp()
    with mock.patch.object(ib, "datetime", LoggingDatetime):
        # non-ISO: fromisoformat attempted, ValueError swallowed,
        # the weekday parser computes the genuine value
        assert ib.parse_when("friday 9am", now) == expected_fri
        assert calls == ["friday 9am"]
        calls.clear()
        # control: a real ISO string parses through the guarded branch
        iso = "2027-01-15 09:00"
        assert ib.parse_when(iso, now) == datetime.fromisoformat(iso).timestamp()
        assert calls == [iso]
        calls.clear()
        # garbage genuinely parses as nothing -> None
        assert ib.parse_when("sometime eventually", now) is None
        assert calls == ["sometime eventually"]
    assert _snap(inbox) == before


# ── 540: _add_column duplicate-column ALTER failure -> pass ────────────

def test_540_add_column_duplicate_alter_failure_passes(tmp_path):
    sql_log = []

    class LoggingDB(Database):
        def execute(self, sql, params=()):
            sql_log.append(sql)
            return super().execute(sql, params)

    root = tmp_path / "root"
    root.mkdir()
    db = LoggingDB(root / "inbox.db")
    inbox = Inbox(root, db=db)
    sql_log.clear()
    before = _snap(inbox)
    # 'priority' already exists (added by _ensure_schema): the ALTER is
    # attempted, raises duplicate-column, and is swallowed.
    assert inbox._add_column("inbox_items", "priority",
                             "INTEGER NOT NULL DEFAULT 0") is None
    assert any("ALTER TABLE inbox_items ADD COLUMN priority" in s
               for s in sql_log)
    assert _snap(inbox) == before
    # control: a genuinely new column IS added by the same path
    sql_log.clear()
    inbox._add_column("inbox_items", "probe_col", "TEXT DEFAULT ''")
    cols = [r["name"] for r in db.query("PRAGMA table_info(inbox_items)")]
    assert "probe_col" in cols
    assert any("ADD COLUMN probe_col" in s for s in sql_log)


# ── 629: _inbox_size per-file stat failure -> file skipped ─────────────

def test_629_inbox_size_stat_failure_skips_vanished_file(tmp_path):
    inbox = Inbox(tmp_path / "root")
    a = inbox.dir / "a.txt"
    a.write_bytes(b"12345")
    b = inbox.dir / "b.txt"
    b.write_bytes(b"1234567")
    before = _snap(inbox)
    total = inbox._inbox_size()  # control: both files counted
    orig_stat = Path.stat
    counts = {}

    def flaky(self, **kw):
        key = str(self)
        counts[key] = counts.get(key, 0) + 1
        if key == str(b) and counts[key] >= 2:
            raise OSError("vanished mid-scan")
        return orig_stat(self, **kw)

    with mock.patch.object(Path, "stat", flaky):
        total2 = inbox._inbox_size()
    # b.txt was statted by is_file (1st) and by the guarded sum (2nd,
    # raised) — the guarded operation was actually attempted
    assert counts.get(str(b), 0) >= 2
    assert total2 == total - 7
    assert _snap(inbox) == before


# ── 676: intake rmdir failure -> source dir left, items registered ─────

def test_676_intake_rmdir_failure_leaves_nonempty_source_dir(tmp_path):
    inbox = Inbox(tmp_path / "root")
    src = tmp_path / "src"
    src.mkdir()
    (src / "note.txt").write_text("hello inbox\n", encoding="utf-8")
    (src / "sub").mkdir()
    (src / "sub" / "inner.txt").write_text("nested\n", encoding="utf-8")
    rmdir_log = []
    orig_rmdir = Path.rmdir

    def logged_rmdir(self):
        rmdir_log.append(str(self))
        return orig_rmdir(self)  # raises OSError: dir not empty

    before_hist = _snap(inbox)["history"]
    with mock.patch.object(Path, "rmdir", logged_rmdir):
        item = inbox.intake(src)
    assert str(src) in rmdir_log  # the rmdir was actually attempted
    # ...and it failed (a successful rmdir would have removed the dir),
    # so the dir and its never-moved subdir remain in place
    assert src.exists() and (src / "sub" / "inner.txt").exists()
    assert item.name == "src__note.txt"  # dir intake namespaces names
    after = _snap(inbox)
    assert len(after["items"]) == 1
    assert after["items"][0]["error"] == ""
    assert after["history"] == before_hist
    # control: a flat source dir IS removed by the same path
    src2 = tmp_path / "src2"
    src2.mkdir()
    (src2 / "only.txt").write_text("x\n", encoding="utf-8")
    with mock.patch.object(Path, "rmdir", logged_rmdir):
        inbox.intake(src2)
    assert str(src2) in rmdir_log
    assert not src2.exists()


# ── 893: purge_done unlink failure -> counts stay literally true ───────

def test_893_purge_done_unlink_failure_counts_stay_true(tmp_path):
    inbox = Inbox(tmp_path / "root")
    old = time.time() - 40 * 86400
    paths = {}
    for name in ("old1.txt", "old2.txt"):
        p = inbox.dir / name
        p.write_bytes(b"data")
        it = inbox._insert(name, str(p), None, kind="file", status="done")
        inbox.db.execute("UPDATE inbox_items SET updated_at=? WHERE id=?",
                         (old, it.id))
        paths[name] = p
    before_hist = _snap(inbox)["history"]
    unlink_log = []
    orig_unlink = Path.unlink
    bad = str(paths["old2.txt"])

    def flaky_unlink(self, **kw):
        unlink_log.append(str(self))
        if str(self) == bad:
            raise OSError("already gone")
        return orig_unlink(self, **kw)

    with mock.patch.object(Path, "unlink", flaky_unlink):
        res = inbox.purge_done(30.0)
    # both unlinks attempted; only the successful one is counted, and
    # both rows were actually deleted — both counts literally true
    assert res == {"rows": 2, "files": 1}
    assert bad in unlink_log
    assert str(paths["old1.txt"]) in unlink_log
    assert not paths["old1.txt"].exists()  # control: healthy unlink worked
    assert paths["old2.txt"].exists()      # failed unlink: file remains
    assert inbox.db.query("SELECT id FROM inbox_items") == []
    assert _snap(inbox)["history"] == before_hist


# ── 1304: sweep directive-probe failure -> sweep completes ─────────────

def test_1304_sweep_directive_probe_failure_sweep_completes(tmp_path):
    inbox = Inbox(tmp_path / "root")
    for name, text in (("note1.txt", "just a note\n"),
                       ("note2.txt", "another note\n")):
        p = inbox.dir / name
        p.write_text(text, encoding="utf-8")
        inbox.intake(p)
    items = {it.name: it for it in inbox.list_items()}
    note1 = items["note1.txt"]
    assert note1.kind == "note"
    orig = inbox._directive_for
    raised = []
    state = {"done": False}

    def flaky(item):
        if item.id == note1.id and not state["done"]:
            state["done"] = True
            raised.append(item.id)
            raise RuntimeError("directive probe boom")
        return orig(item)

    inbox._directive_for = flaky
    report = inbox.sweep()
    assert report["swept"] is True
    assert raised == [note1.id]  # the sweep's probe was actually attempted
    after = {it.name: it for it in inbox.list_items()}
    # the probe failure cost only the ordering hint: process() re-ran
    # the probe per item and the note was classified and finished
    assert after["note1.txt"].status == "done"
    assert after["note1.txt"].error == ""
    assert after["note2.txt"].status == "done"
    hist = _snap(inbox)["history"]
    assert hist and all(h["outcome"] == "processed" for h in hist)
