"""Tail-49 pins for nomorals/research/pipeline.py — 0 converts /
8 exclusions, 0 flagged.

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the guarded path
was actually exercised, asserting via the double's own call log
that the failing operation itself was attempted. The file
contains no exception handlers, so it adds zero census sites.

pipeline.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure import or call in the module),
so NOT-recording is pinned with the tail-8/tail-43 snapshot
method: the agents recent_failures() ledger before/after each
trigger must be unchanged.

Verdict notes (full grounds in the tail-49 SURVEY.md):
- 387: _browser_fetch_fallback finally-block browser close
  failure -> pass; the returned page text was already decided
  by the text call, the close result changes nothing.
- 402: _research_workers profile-kind detection failure ->
  pass and the profile-default worker count stands; detection
  is documented to never break research, and the returned
  count is a real capped count either way.
- 426: _learned_domain_score db failure -> None, the
  documented value for a missing table / unseen domain; the
  rerank consumer never filters, it only demotes on evidence.
- 455: _domain_runs db failure -> 0, the genuine unseen-domain
  value; the consumer's weak-domain gate needs >= 3 runs, so
  0 can never demote.
- 482: _emit_progress callback failure -> pass; progress is
  an optional side-channel and its failure changes no return
  value or consumer decision.
- 744: _send_to_chats ChatRef import failure -> ChatRef None
  and the platform falls back to split(':')[0], which is
  exactly ChatRef.parse's platform for any well-formed key;
  the gateway send result still decides sent/failures.
- 1111: _verify_cited_sentences_stats citations import
  failure -> (raw, 0, 0); verification is advisory, the
  stats honestly report 0 checked, and citation mapping
  downstream still strips invented labels.
- 1282: _corroborated_count citations failure -> 0, the
  genuine no-corroboration value for an advisory audit stat.
"""

import sys
import types
from types import SimpleNamespace

import nomorals.core.profiles as profiles
import nomorals.research.pipeline as P
from nomorals.agents import failures as ag_fail


def _snap():
    return list(ag_fail.recent_failures())


def _finding(url="https://a.example/x",
             title="Outlier pays $30 per hour",
             snippet="Outlier pays $30 per hour for AI training gigs.",
             detail=""):
    return P.ResearchFinding(job_id="j", title=title, url=url,
                             snippet=snippet, detail=detail)


class _LogModule(types.ModuleType):
    """sys.modules stand-in whose missing attribute access is logged."""

    def __init__(self, name, log):
        super().__init__(name)
        self._log = log

    def __getattr__(self, name):
        self._log.append(name)
        raise AttributeError(name)


def test_387_browser_close_failure_keeps_text():
    before = _snap()
    calls = []

    class Reg:
        def __init__(self, fail_close):
            self.fail_close = fail_close

        def call(self, name, actor=None, action=None, **kw):
            calls.append((name, action))
            if action == "close" and self.fail_close:
                raise RuntimeError("close boom")
            if action == "text":
                return SimpleNamespace(ok=True, value={"text": "hello page"})
            return SimpleNamespace(ok=True, value={})

    out = P._browser_fetch_fallback(Reg(True), "https://x.example", 100)
    assert out == "hello page"
    assert ("browser", "close") in calls  # the close itself was attempted
    calls.clear()
    out2 = P._browser_fetch_fallback(Reg(False), "https://x.example", 100)
    assert out2 == "hello page" and ("browser", "close") in calls
    assert _snap() == before


def test_402_profile_detection_failure_keeps_default_workers(monkeypatch):
    before = _snap()
    kind_calls = []
    monkeypatch.setattr(profiles, "profile_value",
                        lambda name, default=None: default)

    def bad_kind():
        kind_calls.append("kind")
        raise RuntimeError("profile boom")

    monkeypatch.setattr(profiles, "get_profile_kind", bad_kind)
    assert P._research_workers(10) == 3
    assert kind_calls == ["kind"]  # detection itself was attempted
    monkeypatch.setattr(profiles, "get_profile_kind", lambda: "termux")
    assert P._research_workers(10) == 2  # control: termux cap applies
    monkeypatch.setattr(profiles, "get_profile_kind", lambda: "workstation")
    assert P._research_workers(10) == 3  # control: no cap off-termux
    assert _snap() == before


def test_426_learned_domain_score_db_failure_is_none():
    before = _snap()
    db_calls = []

    class DB:
        def query_one(self, sql, params=()):
            db_calls.append(sql)
            raise RuntimeError("no source_quality table")

    assert P._learned_domain_score(DB(), "https://www.example.com/x") is None
    assert db_calls and "source_quality" in db_calls[0]  # query attempted

    class DBOk:
        def query_one(self, sql, params=()):
            return {"runs": 4, "worth_runs": 1}

    assert P._learned_domain_score(DBOk(), "https://example.com/x") == 0.25
    assert _snap() == before


def test_455_domain_runs_db_failure_is_zero():
    before = _snap()
    db_calls = []

    class DB:
        def query_one(self, sql, params=()):
            db_calls.append(sql)
            raise RuntimeError("db boom")

    assert P._domain_runs(DB(), "https://example.com/x") == 0
    assert db_calls and "source_quality" in db_calls[0]  # query attempted

    class DBOk:
        def query_one(self, sql, params=()):
            return {"runs": 5}

    assert P._domain_runs(DBOk(), "https://example.com/x") == 5
    assert _snap() == before


def test_482_emit_progress_callback_failure_is_dropped():
    before = _snap()
    prog_calls = []

    def bad_progress(phase, item):
        prog_calls.append((phase, item))
        raise RuntimeError("ui boom")

    assert P._emit_progress(bad_progress, "search", "q1") is None
    assert prog_calls == [("search", "q1")]  # the callback was attempted
    got = []
    P._emit_progress(lambda ph, it: got.append((ph, it)), "fetch", "u")
    assert got == [("fetch", "u")]  # control: healthy callback receives
    assert _snap() == before


def test_744_chatref_import_failure_falls_back_to_split_platform(monkeypatch):
    before = _snap()
    sent_calls = []

    class GW:
        owner_chats = ["telegram:123"]

        def send(self, platform, chat_key, text):
            sent_calls.append((platform, chat_key, text))
            return SimpleNamespace(ok=True, error=None)

    rctx = P.ResearchContext(db=None, registry=None, gateway=GW())
    import_log = []
    monkeypatch.setitem(
        sys.modules, "nomorals.social.chat.gateway",
        _LogModule("nomorals.social.chat.gateway", import_log))
    sent, failures = P._send_to_chats(rctx, "hi")
    assert sent == ["telegram:123"] and failures == []
    assert "ChatRef" in import_log  # the import itself was attempted
    assert sent_calls[-1][0] == "telegram"  # split(':')[0] fallback
    monkeypatch.undo()
    sent_calls.clear()
    sent2, failures2 = P._send_to_chats(rctx, "hi")
    assert sent2 == ["telegram:123"] and failures2 == []
    assert sent_calls[-1][0] == "telegram"  # control: parse agrees
    assert _snap() == before


def test_1111_citations_import_failure_passes_text_through(monkeypatch):
    before = _snap()
    f1 = _finding(detail="Outlier pays $30 per hour.")
    raw = "Outlier pays $30 per hour. [S1]"
    cit_log = []
    monkeypatch.setitem(
        sys.modules, "nomorals.research.citations",
        _LogModule("nomorals.research.citations", cit_log))
    out = P._verify_cited_sentences_stats(raw, [f1])
    assert out == (raw, 0, 0)
    assert "sentence_supported" in cit_log  # the import was attempted
    monkeypatch.undo()
    text, checked, stripped = P._verify_cited_sentences_stats(raw, [f1])
    assert (checked, stripped) == (1, 0) and text == raw  # control
    assert _snap() == before


def test_1282_corroboration_failure_is_zero(monkeypatch):
    before = _snap()
    fa = _finding(url="https://a.example/x")
    fb = _finding(url="https://b.example/y")
    cit_log = []
    monkeypatch.setitem(
        sys.modules, "nomorals.research.citations",
        _LogModule("nomorals.research.citations", cit_log))
    assert P._corroborated_count([fa, fb], [fa, fb]) == 0
    assert "corroboration_map" in cit_log  # the import was attempted
    monkeypatch.undo()
    # control: two same-topic sources on different domains corroborate
    assert P._corroborated_count([fa, fb], [fa, fb]) == 2
    assert _snap() == before
