"""Tail-59 pins for nomorals/books/forge.py —
1 convert (395, by parent amendment ruling) / 6 exclusions.

11 tests: 7 site tests (one per site) + an all-valid control
for 395 + one rendering pin per traced list_books consumer
(CLI, partner surface, book_list tool). Each exclusion test
pins one site with a control half proving the guarded
operation was actually ATTEMPTED, asserting via the operation's
own call log / double. The file contains no exception handlers,
so it adds zero census sites.

For the six exclusions, NOT-recording is pinned against the
books package ledger (nomorals/books/failures.py):
recent_failures() snapshot before/after every trigger. Site
395 is the deliberate exception (amendment): forge now imports
record_failure from the package ledger, exactly as its siblings
library.py / sources.py do, and its pin asserts the recording.

Verdict notes (full grounds in the tail-59 SURVEY.md):
- 69: clean_title model_usable failure -> _usable=False -> the
  documented offline heuristic title; a real title, literally
  derived from the topic, no model/absence claim.
- 103: _model_title chat failure -> next prompt, then "" ->
  clean_title's heuristic fallback; "" is the genuine
  no-model-title value.
- 236: forge.plan_book model_usable failure -> None, the
  documented no-plan value ("model isn't usable / response is
  bad"); callers fall back to separate calls.
- 290: forge.plan_book outer failure (chat/parse) -> None; the
  handler's own comment names the fallback (separate calls).
- 321: infer_chapter_count chat failure -> pass -> genre/topic
  heuristic count; a real count from the documented fallback.
- 395: CONVERTED (parent amendment ruling) — list_books
  corrupt book.json -> that book is still skipped ("a corrupt
  book must not hide the rest": the valid books list exactly as
  before), but the skip now binds the error, logs a warning and
  records to the books ledger (forge.list_books), and the
  unreadable count surfaces via forge.last_list_unreadable to
  the CLI, the partner surface and the book_list tool, so no
  surface renders a false "no books yet" over books on disk.
  Missing-json directories stay skipped silently, uncounted.
- 443: create() outline plan failure -> planned=None ->
  clean_title + seed/make_outline fallback; the book is really
  created with a real outline. (On the base tree
  outline.plan_book does not exist, so this handler fires on
  every explicit-chapters create via ImportError — a genuine
  out-of-batch defect documented in the SURVEY; the pin
  installs the attribute double to exercise the handler as
  written.)
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.books.forge as F
import nomorals.books.outline as outline_mod
from nomorals.books.failures import recent_failures
from nomorals.books.model import Book


def _snap():
    return list(recent_failures(200))


class _Resp:
    ok = True

    def __init__(self, t):
        self.text = t


def test_69_clean_title_usable_failure_falls_back_to_heuristic():
    before = _snap()
    calls = []

    def bad_usable(ctx):
        calls.append(ctx)
        raise RuntimeError("usable boom")

    ctx = SimpleNamespace(router=object())
    with mock.patch("nomorals.llm.power.model_usable", bad_usable):
        out = F.clean_title("a guide to urban gardening", ctx)
    assert out == "Urban Gardening"
    assert len(calls) == 1

    class FakeBrain:
        def __init__(self, router=None):
            self.router = router

        def chat(self, msgs, params, task_kind=None):
            return _Resp("A Model Title")

    with mock.patch("nomorals.llm.power.model_usable", lambda c: True), \
            mock.patch("nomorals.llm.brain.Brain", FakeBrain):
        assert F.clean_title("a guide to urban gardening", ctx) == "A Model Title"
    assert _snap() == before


def test_103_model_title_chat_failure_returns_empty():
    before = _snap()
    chat_calls = []

    class RaisingBrain:
        def __init__(self, router=None):
            pass

        def chat(self, msgs, params, task_kind=None):
            chat_calls.append(1)
            raise RuntimeError("chat boom")

    with mock.patch("nomorals.llm.brain.Brain", RaisingBrain):
        assert F._model_title("urban gardening", object()) == ""
    assert len(chat_calls) == 2  # both prompts attempted

    class FakeBrain:
        def __init__(self, router=None):
            self.router = router

        def chat(self, msgs, params, task_kind=None):
            return _Resp("A Model Title")

    with mock.patch("nomorals.llm.brain.Brain", FakeBrain):
        assert F._model_title("urban gardening", object()) == "A Model Title"
    assert _snap() == before


def test_236_plan_book_usable_failure_returns_none():
    before = _snap()
    calls = []

    def bad_usable(ctx):
        calls.append(1)
        raise RuntimeError("usable boom")

    with mock.patch("nomorals.llm.power.model_usable", bad_usable):
        assert F.plan_book("topic", "", SimpleNamespace(router=object())) is None
    assert len(calls) == 1
    assert _snap() == before


def test_290_plan_book_chat_failure_returns_none():
    before = _snap()
    chat_calls = []

    class RaisingBrainFor:
        def chat(self, msgs, params, task_kind=None):
            chat_calls.append(1)
            raise RuntimeError("plan boom")

    payload = json.dumps({"title": "Trust Again", "chapters": [
        {"title": "One", "beats": ["a"]}, {"title": "Two", "beats": ["b"]},
        {"title": "Three", "beats": ["c"]}]})

    class GoodBrainFor:
        def chat(self, msgs, params, task_kind=None):
            return _Resp(payload)

    ctx = SimpleNamespace(router=object())
    with mock.patch("nomorals.llm.power.model_usable", lambda c: True), \
            mock.patch.object(F, "brain_for", lambda c: RaisingBrainFor()):
        assert F.plan_book("trust", "", ctx) is None
    assert len(chat_calls) == 1
    with mock.patch("nomorals.llm.power.model_usable", lambda c: True), \
            mock.patch.object(F, "brain_for", lambda c: GoodBrainFor()):
        plan = F.plan_book("trust", "", ctx)
    assert plan is not None and plan["title"] == "Trust Again"
    assert len(plan["chapters"]) == 3
    assert _snap() == before


def test_321_infer_chapter_count_chat_failure_uses_heuristic():
    before = _snap()
    chat_calls = []

    class RaisingBrainFor:
        def chat(self, msgs, params, task_kind=None):
            chat_calls.append(1)
            raise RuntimeError("infer boom")

    class NineBrain:
        def chat(self, msgs, params, task_kind=None):
            return _Resp("9")

    ctx = SimpleNamespace(router=object())
    with mock.patch.object(F, "brain_for", lambda c: RaisingBrainFor()):
        n = F.infer_chapter_count("a novel about the sea", ctx)
    assert isinstance(n, int) and 12 <= n <= 24  # novel genre band
    assert len(chat_calls) == 1
    with mock.patch.object(F, "brain_for", lambda c: NineBrain()):
        assert F.infer_chapter_count("anything", ctx) == 9
    assert _snap() == before


def test_395_list_books_corrupt_book_skipped_recorded_counted(caplog):
    """CONVERTED site (parent amendment ruling) — the skip
    contract survives (valid book still listed), but the skip is
    observable: warning logged, books ledger gains exactly one
    forge.list_books entry, and the count surfaces on the
    instance. Missing-json dirs stay silent and uncounted."""
    before = _snap()
    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    forge = F.BookForge(ctx)
    good = Book(topic="gardening", slug="good-book", title="Good Book")
    forge.save(good)
    bad_dir = Path(tmp) / "books" / "bad-book"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "book.json").write_text("{not json", encoding="utf-8")
    (Path(tmp) / "books" / "no-json-dir").mkdir(parents=True, exist_ok=True)
    loads_calls = []
    real_loads = json.loads

    def logging_loads(s, *a, **k):
        loads_calls.append(s[:20])
        return real_loads(s, *a, **k)

    with caplog.at_level("WARNING", logger="nomorals.books.forge"), \
            mock.patch("json.loads", logging_loads):
        books = forge.list_books()
    assert [b["slug"] for b in books] == ["good-book"]  # contract preserved
    assert len(loads_calls) == 2  # both files' reads attempted
    assert forge.last_list_unreadable == 1  # no-json dir NOT counted
    assert any("skipping unreadable book" in r.message
               for r in caplog.records)
    after = _snap()
    new = after[len(before):]
    assert len(new) == 1 and new[0].site == "forge.list_books"
    # control: fix the file -> both listed, count resets, no record
    (bad_dir / "book.json").write_text(json.dumps(good.to_dict()),
                                       encoding="utf-8")
    before2 = _snap()
    assert len(forge.list_books()) == 2
    assert forge.last_list_unreadable == 0
    assert _snap() == before2


def test_395_list_books_all_valid_count_zero():
    """395 control: an all-valid store lists both books with a
    zero unreadable count and records nothing."""
    before = _snap()
    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    forge = F.BookForge(ctx)
    forge.save(Book(topic="gardening", slug="good-book", title="Good Book"))
    forge.save(Book(topic="the sea", slug="sea-book", title="Sea Book"))
    books = forge.list_books()
    assert sorted(b["slug"] for b in books) == ["good-book", "sea-book"]
    assert forge.last_list_unreadable == 0
    assert _snap() == before


def _corrupt_only_ctx(n=1):
    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    forge = F.BookForge(ctx)
    for i in range(n):
        d = Path(tmp) / "books" / f"bad-book-{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "book.json").write_text("{not json", encoding="utf-8")
    return ctx


def test_395_cli_list_renders_unreadable_not_absence(capsys):
    """395 consumer pin — `nm book list`: a corrupt-only store
    renders the honest unreadable line, never 'no books yet';
    an empty store keeps the absence line."""
    import argparse

    from nomorals.cmdline.commands.book import _cmd_book

    _cmd_book(argparse.Namespace(action="list", json=False),
              _corrupt_only_ctx(2))
    out = capsys.readouterr().out
    assert "no readable books" in out and "2 book(s)" in out
    assert "no books yet" not in out
    tmp = tempfile.mkdtemp()
    empty = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    _cmd_book(argparse.Namespace(action="list", json=False), empty)
    assert "no books yet" in capsys.readouterr().out


def test_395_partner_list_renders_unreadable_not_absence():
    """395 consumer pin — partner /book surface: same honest
    distinction for both the list and status verbs."""
    from nomorals.agents.partner.runtime_search import RuntimeSearchMixin

    m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
    m.context = _corrupt_only_ctx(1)
    out = m._control_book("list", "k")
    assert "no readable books" in out and "1 book(s)" in out
    assert "no books yet" not in out
    out = m._control_book("status", "k")
    assert "no readable books" in out
    assert "no books on disk yet" not in out
    tmp = tempfile.mkdtemp()
    m2 = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
    m2.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp))
    assert m2._control_book("list", "k").startswith("no books yet.")
    assert m2._control_book("status", "k") == "no books on disk yet."


def test_395_book_list_tool_includes_unreadable_count():
    """395 consumer pin — structured book_list tool result
    carries the unreadable count alongside 'books'."""
    import nomorals.books.tools as book_tools

    class FakeRegistry:
        def __init__(self, context):
            self.context = context
            self.fns = {}

        def register(self, name, **kw):
            def deco(fn):
                self.fns[name] = fn
                return fn
            return deco

    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    forge = F.BookForge(ctx)
    forge.save(Book(topic="gardening", slug="good-book", title="Good Book"))
    bad_dir = Path(tmp) / "books" / "bad-book"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "book.json").write_text("{not json", encoding="utf-8")
    reg = FakeRegistry(ctx)
    book_tools.register(reg)
    res = reg.fns["book_list"]()
    assert [b["slug"] for b in res["books"]] == ["good-book"]
    assert res["unreadable"] == 1


def test_443_create_plan_failure_falls_back_to_outline():
    before = _snap()
    plan_calls = []

    def bad_plan(topic, notes, context, genre=""):
        plan_calls.append(1)
        raise RuntimeError("plan boom")

    tmp = tempfile.mkdtemp()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
    with mock.patch.object(outline_mod, "plan_book", bad_plan, create=True):
        book = F.BookForge(ctx).create("the history of paper clips",
                                       chapters=5, research=False)
    assert len(plan_calls) == 1
    assert len(book.chapters) == 5 and bool(book.title)

    def good_plan(topic, notes, context, genre=""):
        return {"title": "Planned Title",
                "chapters": [{"title": f"P{i}", "beats": ["x"]}
                             for i in range(5)]}

    tmp2 = tempfile.mkdtemp()
    ctx2 = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp2))
    with mock.patch.object(outline_mod, "plan_book", good_plan, create=True):
        book2 = F.BookForge(ctx2).create("the history of paper clips",
                                         chapters=5, research=False)
    assert book2.title == "Planned Title"
    assert book2.chapters[0].title == "P0"
    assert _snap() == before
