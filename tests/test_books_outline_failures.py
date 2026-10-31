"""Failure pins for nomorals/books/outline.py (tail-68).

Every site is an adjudicated EXCLUSION: the module's documented design
is "model-first, template/heuristic floor" — a model failure returns
None exactly like genuine model absence (no context / no router /
unusable answer), and the caller builds a real deterministic outline
from the floor. The failure value never masquerades as model output:
None is not a chapter list, and the floor's chapters are genuine
topic-specific content. Each pin proves the guarded operation was
attempted (spy), the floor behavior that follows, and that nothing
was recorded on the books failure ledger.
"""
from types import SimpleNamespace

from nomorals.books import failures as book_failures
from nomorals.books import outline as om
from nomorals.books.model import Book, Chapter

CTX = SimpleNamespace(router=object())
OUTLINE_JSON = ('[{"title": "Alpha", "beats": ["a1"]},'
                ' {"title": "Beta", "beats": ["b1"]},'
                ' {"title": "Gamma", "beats": ["g1"]}]')


class FakeBrain:
    def __init__(self, text="", exc=None):
        self.text, self.exc, self.calls = text, exc, 0

    def chat(self, messages, params, task_kind=None):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(text=self.text, ok=True)


# 263 — _model_outline: the model_usable probe raises -> swallowed;
# the probe is an advisory pre-check (its sibling model paths do not
# probe at all) and the real call proceeds, itself fully guarded.
def test_263_usable_probe_failure_still_uses_model(monkeypatch):
    before = book_failures.failure_count()
    probe_calls = []

    def raising_usable(ctx):
        probe_calls.append(ctx)
        raise RuntimeError("power probe boom")

    brain = FakeBrain(text=OUTLINE_JSON)
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain)
    monkeypatch.setattr("nomorals.llm.power.model_usable", raising_usable)
    book = Book(topic="sourdough baking", slug="p263")
    book._context = CTX
    chapters = om._model_outline(book, n_chapters=3)
    assert len(probe_calls) == 1  # the probe was really attempted
    assert brain.calls == 1  # and the model call still happened
    assert chapters is not None and len(chapters) == 3
    assert book_failures.failure_count() == before


# 310 — _model_outline: the chat itself raises -> None, the documented
# "caller falls back to the template" contract; make_outline builds
# the full template outline (identical to the no-model path).
def test_310_chat_failure_returns_none_and_template_floor(monkeypatch):
    before = book_failures.failure_count()
    brain = FakeBrain(exc=RuntimeError("chat boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain)
    monkeypatch.setattr("nomorals.llm.power.model_usable",
                        lambda ctx: True)
    book = Book(topic="sourdough baking", slug="p310")
    book._context = CTX
    assert om._model_outline(book, n_chapters=3) is None
    assert brain.calls == 1  # the chat was really attempted

    brain2 = FakeBrain(exc=RuntimeError("chat boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain2)
    book2 = Book(topic="sourdough baking", slug="p310b")
    chapters = om.make_outline(book2, n_chapters=5, context=CTX)
    assert len(chapters) == 5  # template floor honors the count
    assert book2.status == "planned"
    assert book_failures.failure_count() == before


# 377 — beat_sheet_outline: the genre engine's beat_note raises ->
# swallowed; the chapter keeps the sheet's own beat names/notes and
# coverage, only the additive flavor note is missing.
def test_377_beat_note_failure_keeps_sheet_beats(monkeypatch):
    before = book_failures.failure_count()

    class BoomEngine:
        def __init__(self):
            self.calls = 0

        def beat_note(self, ch_no, n):
            self.calls += 1
            raise RuntimeError("engine boom")

    boom = BoomEngine()
    monkeypatch.setattr("nomorals.books.fiction.engine_for",
                        lambda genre: boom)
    book = Book(topic="a mystery novel", slug="p377")
    chs = om.beat_sheet_outline(book, n_chapters=6, genre="mystery")
    assert boom.calls == 6  # the note was attempted per chapter
    assert len(chs) == 6
    assert all(c.title and c.coverage.startswith("__beat__:")
               for c in chs)
    assert all(c.beats and not any("Serve it in" in b for b in c.beats)
               for c in chs)

    class OkEngine:
        def beat_note(self, ch_no, n):
            return "Story beat: x — y. Serve it in mystery terms."

    monkeypatch.setattr("nomorals.books.fiction.engine_for",
                        lambda genre: OkEngine())
    chs2 = om.beat_sheet_outline(book, n_chapters=6, genre="mystery")
    assert any("Serve it in" in b for c in chs2 for b in c.beats)
    assert book_failures.failure_count() == before


# 429 — _model_seed: the chat raises -> None; seed_outline plants the
# template seed (the same 3-chapter opening arc as the no-model path).
def test_429_seed_chat_failure_template_seed(monkeypatch):
    before = book_failures.failure_count()
    brain = FakeBrain(exc=RuntimeError("seed boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain)
    book = Book(topic="sourdough baking", slug="p429")
    book._context = CTX
    assert om._model_seed(book) is None
    assert brain.calls == 1  # the chat was really attempted

    brain2 = FakeBrain(exc=RuntimeError("seed boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain2)
    book2 = Book(topic="sourdough baking", slug="p429b")
    seed = om.seed_outline(book2, context=CTX)
    assert len(seed) == 3
    assert seed[0].title.startswith("What ")
    assert seed[1].title == "Foundations: The Core Ideas"
    assert seed[2].title.startswith("Deep Dive:")
    assert book_failures.failure_count() == before


# 443 — _parse_chapters_json: json.loads raises -> None. A broken
# answer belongs to the same designed outcome class as a non-JSON or
# empty answer (both return None from this function's earlier checks).
def test_443_unparseable_answer_returns_none(monkeypatch):
    before = book_failures.failure_count()
    seen = []

    class FakeJson:
        @staticmethod
        def loads(s, **k):
            seen.append(s)
            raise ValueError("bad json")

        dumps = staticmethod(om.json.dumps)

    monkeypatch.setattr(om, "json", FakeJson)
    out = om._parse_chapters_json('[{"title": "X", "beats": ["a"]}]',
                                  want=1)
    assert out is None
    assert len(seen) == 1  # the parse was really attempted
    monkeypatch.undo()
    good = om._parse_chapters_json('[{"title": "X", "beats": ["a"]}]',
                                   want=1)
    assert good is not None and len(good) == 1  # honest control
    assert book_failures.failure_count() == before


# 561 — _model_continuation: the chat raises -> None; assess_continuation
# decides from the heuristic coverage floor instead.
def test_561_continuation_failure_heuristic_floor(monkeypatch):
    before = book_failures.failure_count()
    brain = FakeBrain(exc=RuntimeError("continuation boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain)
    book = Book(topic="sourdough baking", slug="p561", organic=True)
    book.coverage = ["yeast care"]
    book.chapters = [Chapter(number=1, title="Intro", status="written",
                             coverage="__intro__")]
    book._context = CTX
    assert om._model_continuation(book) is None
    assert brain.calls == 1  # the chat was really attempted

    brain2 = FakeBrain(exc=RuntimeError("continuation boom"))
    monkeypatch.setattr(om, "brain_for", lambda ctx: brain2)
    decision = om.assess_continuation(book, context=CTX)
    assert decision["complete"] is False
    assert decision["added"] == ["Deep Dive: Yeast Care"]
    assert decision["reason"] == "still to cover: yeast care"
    assert book_failures.failure_count() == before
