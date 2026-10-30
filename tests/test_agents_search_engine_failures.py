"""Tail-65 pins for nomorals/agents/search/engine.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's own call log. The file contains
no exception handlers, so it adds zero census sites.

engine.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the agents package ledger (nomorals/agents/failures.py):
recent_failures() snapshot before/after every trigger.
(The module's bind+log guards — fetch, decompose, leads,
journal, cache-put — are not census sites and stay as-is.)

Verdict notes (full grounds in the tail-65 SURVEY.md):
- 167: a corrupt PDF degrades to an empty-text page
  (chars 0) — byte-identical to a genuinely text-less PDF
  and treated the same by design: run()'s chars>0 filter
  keeps it out of the summary and citation slots.
- 206: a dead crawl frontier page is skipped, exactly the
  absence read() itself returns for a dead page.
- 229: a crashing stats snapshot answers "no model" —
  the designed offline path; the report honestly labels
  model_summary False on that path.
- 330: trust.feedback is a learning side-channel inside a
  block the code itself scopes as a display bonus; the
  read outcome and report are unaffected.
- 453: history() degrades to [] — the empty-store value;
  the sole consumer renders "no research runs yet"
  (directives-114 / identity_bank-list family).
- 482: a sick cache answers a miss, and the fallback
  performs the real operation — a live fetch.
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as agent_fail
from nomorals.agents.search.engine import SearchEngine


def _snap():
    return list(agent_fail.recent_failures(200))


class Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class EngDB:
    def __init__(self, fail_on=""):
        self.fail_on = fail_on
        self.calls = []

    def query_one(self, *a):
        self.calls.append("query_one")
        if self.fail_on == "query_one":
            raise RuntimeError("db read dead")
        return None

    def query(self, *a):
        self.calls.append("query")
        if self.fail_on == "query":
            raise RuntimeError("db read dead")
        return []

    def execute(self, *a):
        self.calls.append("execute")

    def transaction(self):
        return Tx()


def _engine(db=None, **ctxkw):
    ctx = SimpleNamespace(db=db if db is not None else EngDB(), **ctxkw)
    return SearchEngine(ctx)


def test_167_corrupt_pdf_reads_as_empty_page():
    before = _snap()
    eng = _engine()
    eng._respect_robots = False

    class Resp:
        ok = True
        status = 200
        content_type = "application/pdf"
        body = b"%PDF-1.4 fake"
        text = ""
        url = "https://example.com/report.pdf"

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, url):
            self.calls.append(url)
            return Resp()

    eng._client = Client()
    calls = []

    def bad_pdf(body):
        calls.append(1)
        raise RuntimeError("corrupt pdf")

    with mock.patch("nomorals.core.pdf.read_pdf_text", bad_pdf):
        page = eng.read("https://example.com/report.pdf")
    assert calls == [1]
    assert eng._client.calls == ["https://example.com/report.pdf"]
    assert page is not None and page["pdf"] is True
    assert page["text"] == "" and page["chars"] == 0
    assert page["title"] == "report.pdf"
    assert _snap() == before


def test_206_crawl_dead_page_is_skipped():
    before = _snap()
    eng = _engine()

    class DeadClient:
        def __init__(self):
            self.calls = []

        def get(self, url):
            self.calls.append(url)
            raise RuntimeError("net dead")

    eng._client = DeadClient()
    assert eng.crawl("https://example.com/") == []
    assert eng._client.calls == ["https://example.com/"]
    assert _snap() == before


def test_229_model_available_snapshot_failure_is_false():
    before = _snap()
    calls = []

    def bad_snap():
        calls.append(1)
        raise RuntimeError("stats dead")

    eng = _engine(router=SimpleNamespace(stats_snapshot=bad_snap))
    assert eng._model_available() is False
    assert calls == [1]
    assert _snap() == before


def test_330_trust_feedback_failure_keeps_report():
    before = _snap()

    class FakeTrust:
        instances = []

        def __init__(self, context, store=None):
            self.feedback_calls = []
            FakeTrust.instances.append(self)

        def annotate(self, top):
            pass

        def feedback(self, url, was_ok):
            self.feedback_calls.append((url, was_ok))
            raise RuntimeError("trust dead")

    eng = _engine(db=None, router=None)
    eng.search = lambda query, max_results=None, freshness="": [
        {"url": "https://example.com/alpha",
         "title": "Alpha research findings",
         "snippet": "alpha research snippet with plenty of substance"}]
    page = {"url": "https://example.com/alpha",
            "title": "Alpha research findings",
            "text": "alpha body text", "domain": "example.com",
            "chars": 15, "pdf": False, "links": []}
    eng.read = lambda url, max_chars=40000: page
    with mock.patch("nomorals.agents.search.engine.SourceTrust", FakeTrust):
        report = eng.run("alpha research", mode="quick", pages=3)
    assert FakeTrust.instances[0].feedback_calls == [
        ("https://example.com/alpha", True)]
    assert report["pages_read"] == ["https://example.com/alpha"]
    assert _snap() == before


def test_453_history_query_failure_returns_empty():
    before = _snap()
    db = EngDB(fail_on="query")
    eng = _engine(db=db)
    assert eng.history() == []
    assert db.calls == ["query"]
    assert _snap() == before


def test_482_cache_get_failure_is_a_miss_and_refetches():
    before = _snap()
    db = EngDB(fail_on="query_one")
    eng = _engine(db=db)
    assert eng._cache_get("https://example.com/") is None
    assert db.calls == ["query_one"]

    eng._respect_robots = False

    class Resp:
        ok = True
        status = 200
        content_type = "text/html"
        body = b""
        text = ("<html><head><title>Hi there page</title></head>"
                "<body>body words here</body></html>")
        url = "https://example.com/"

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, url):
            self.calls.append(url)
            return Resp()

    eng._client = Client()
    page = eng.read("https://example.com/")
    assert eng._client.calls == ["https://example.com/"]
    assert page is not None and page["title"] == "Hi there page"
    assert _snap() == before
