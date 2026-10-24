"""Pass #138: send tail failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its tail
region returned safe defaults on failure with no trace:

- ``spam_check_template``: a failed check returned score 0 with
  verdict "unknown" — and the ``/send spamcheck`` caller blesses any
  score under 30 as "clean enough for bulk", so an unchecked template
  went to bulk with a clean bill. A fail-open gate, twin of #129's
  ``allowed`` and #134's ``_warmup_ok``.
- ``status``: a failed read returned its half-built dict, so zero
  spend and no providers read as "nothing spent / nothing
  configured" — the same shape as #133's ``campaign_report``.
- ``get_engine``: a failed singleton quietly retried the IDENTICAL
  constructor (same method, not another method, so not #126's probe
  class) and handed back a non-singleton engine with the cause lost.

All three now record into the shared marketing ledger
(``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged.

Deliberately NOT in scope here (no new evidence): the 3 debug-honest
sites (``render_template``, ``__init__`` DB handler, ``process`` —
each ``_log.debug(..., exc_info=True)``, proven by walking Call
keywords), the 4 pinned exclusions from #137 (``parse_dsn``,
``_dns_txt`` socket/parse-continues, ``deliverability_check``
inner), and the 3 ``__init__``-silent sites (converted in #139).
"""

import ast
import tempfile
from pathlib import Path

import pytest

import nomorals.marketing.send as send_mod
import nomorals.tools.error_scan as es
from nomorals.marketing.failures import clear_failures, recent_failures
from nomorals.marketing.send import SendEngine, control_send
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


@pytest.fixture(autouse=True)
def _clean_engine_singleton():
    old = send_mod._ENGINE
    send_mod._ENGINE = None
    yield
    send_mod._ENGINE = old


def _sites():
    return [f.site for f in recent_failures(1000)]


class _BoomDB:
    """DB handle whose execute/commit always explode."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom-db")

    def commit(self, *args, **kwargs):
        raise RuntimeError("boom-commit")


def _boom_engine():
    e = SendEngine.__new__(SendEngine)
    e._db = _BoomDB()
    e._tracker = None
    e._callables = {}
    return e


def _eng():
    e = SendEngine(db_path=tempfile.mktemp(suffix=".db"))
    e.register_provider("mock", kind="mock", rate_per_minute=1000,
                        sender=lambda to, subj, body: True)
    return e


# ── spam_check_template: an unchecked template must not read as clean ──

def test_spam_check_template_raise_records_but_still_returns_unknown(monkeypatch):
    e = _eng()

    def _boom(*args, **kwargs):
        raise RuntimeError("boom-get")

    monkeypatch.setattr(e, "get_template", _boom)
    res = e.spam_check_template("x")
    assert res == {"score": 0, "verdict": "unknown", "issues": []}
    assert _sites() == ["send.SendEngine.spam_check_template"]


def test_spam_check_template_boom_db_no_double_record():
    # get_template records at its own source site and returns None, so
    # spam_check takes the honest "template not found" path and its own
    # except never fires: exactly one record, at the source.
    e = _boom_engine()
    res = e.spam_check_template("x")
    assert res["verdict"] == "template not found"
    assert _sites() == ["send.SendEngine.get_template"]


def test_spam_check_template_benign_paths_record_nothing():
    e = _eng()
    e.save_template("hello",
                    "Hello world, this is a perfectly normal message body "
                    "with no tricks in it at all, just a friendly note.",
                    subject="Hello")
    res = e.spam_check_template("hello")
    assert res["verdict"] == "clean"
    assert e.spam_check_template("nope")["verdict"] == "template not found"
    assert _sites() == []


def test_spamcheck_caller_blesses_unchecked_template(monkeypatch):
    # End-to-end through the real caller: a check that never ran
    # returns score 0, and the caller prints "clean enough for bulk"
    # on score < 30 regardless of verdict. The record is what makes
    # this visible instead of silent.
    e = _eng()

    def _boom(*args, **kwargs):
        raise RuntimeError("boom-get")

    monkeypatch.setattr(e, "get_template", _boom)
    out = control_send("spamcheck x", engine=e)
    assert "0/100 — unknown" in out
    assert "clean enough for bulk" in out
    assert _sites() == ["send.SendEngine.spam_check_template"]


# ── status: a half-built dict is not a status ──

class _PartialDB:
    """Delegates to the real connection, except the send_log query."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, params=()):
        if "send_log" in sql:
            raise RuntimeError("boom-send_log")
        return self._real.execute(sql, params)

    def commit(self):
        return self._real.commit()


def test_status_partial_db_failure_records_but_still_returns_partial():
    e = _eng()
    e.save_template("hi", "Hello there, friend.", subject="Hi")
    assert e.enqueue("a@example.com", "hi", {}, "mock") != ""
    real_db = e._db
    e._db = _PartialDB(real_db)
    st = e.status()
    # Partial, not all-zero: the queued count survived, the rest never
    # got built — exactly the shape that used to read as a clean bill.
    assert st["queued"] == 1
    assert st["sent_24h"] == 0
    assert st["providers"] == []
    assert _sites() == ["send.SendEngine.status"]


def test_status_benign_path_records_nothing():
    e = _eng()
    st = e.status()
    assert [p["name"] for p in st["providers"]] == ["mock"]
    assert _sites() == []


def test_status_no_db_returns_zeros_no_record():
    e = SendEngine.__new__(SendEngine)
    e._db = None
    st = e.status()
    assert st == {"queued": 0, "sent_24h": 0, "failed": 0, "dead": 0,
                  "blocklisted": 0, "suppressed": 0, "spent_24h_kobo": 0,
                  "providers": []}
    assert _sites() == []


# ── get_engine: a retried constructor is not a singleton ──

def test_get_engine_transient_failure_records_and_returns_fresh(monkeypatch):
    calls = {"n": 0}

    class _Flaky:
        def __init__(self, db_path=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom-construct")
            self.db_path = db_path

    monkeypatch.setattr(send_mod, "SendEngine", _Flaky)
    res = send_mod.get_engine("some.db")
    assert isinstance(res, _Flaky)
    assert _sites() == ["send.get_engine"]
    # The singleton break is real, not theoretical: the fallback object
    # was never stored, so the next call starts over from _ENGINE None.
    assert send_mod._ENGINE is None


def test_get_engine_persistent_failure_still_raises_after_record(monkeypatch):
    # Documents current behaviour, unchanged by this pass: the except
    # retries the same failing constructor, so a deterministic failure
    # still propagates out of a "Never raises" function — with a record
    # left behind now, where before there was nothing.
    class _AlwaysBoom:
        def __init__(self, db_path=""):
            raise RuntimeError("boom-always")

    monkeypatch.setattr(send_mod, "SendEngine", _AlwaysBoom)
    with pytest.raises(RuntimeError):
        send_mod.get_engine("x.db")
    assert _sites() == ["send.get_engine"]


def test_get_engine_singleton_benign_path_records_nothing(tmp_path):
    eng1 = send_mod.get_engine(str(tmp_path / "eng.db"))
    assert eng1 is send_mod._ENGINE
    assert send_mod.get_engine(str(tmp_path / "eng.db")) is eng1
    assert _sites() == []


# ── structural + scan pins ──

def test_ast_tail_sites_record():
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    expected = {"spam_check_template": "send.SendEngine.spam_check_template",
                "status": "send.SendEngine.status",
                "get_engine": "send.get_engine"}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name in expected:
            handlers = [n for n in ast.walk(node)
                        if isinstance(n, ast.ExceptHandler)]
            assert len(handlers) == 1, node.name
            h = handlers[0]
            assert h.name == "e", node.name
            calls = [n for n in ast.walk(h)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "record_failure"]
            assert len(calls) == 1, node.name
            assert calls[0].args[0].value == expected[node.name]
            found[node.name] = True
    assert found == {k: True for k in expected}


def _enclosing_func(line):
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    best = "<module>"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                best = node.name
    return best


def test_scan_send_remainder_is_exactly_7(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 10 remainder after #138 - 3 __init__ converts in #139
    # = 7 remainder: 3 debug-honest (render_template, __init__ DB
    # handler, process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 0 untouched sites: send.py is closed except for
    # pinned judgments.
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 7
    funcs = sorted(_enclosing_func(f.line) for f in rep.findings)
    assert funcs == sorted(["render_template", "parse_dsn", "_dns_txt",
                            "_dns_txt", "deliverability_check", "__init__",
                            "process"])


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
