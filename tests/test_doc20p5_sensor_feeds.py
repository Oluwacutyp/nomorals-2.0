"""Doc 20 P5 — sensor feeds: the immune system gains inbound senses.

Pins, end to end:

* the failure-ledger feed groups the app DB's ``source='tool'``
  failure rows by (tool, signature) over a trailing window: below
  the threshold (≥5 in 3600s by default) nothing opens; at the
  threshold exactly ONE case opens, its event carrying the full
  recurrence (feed key, count, span, every failure id) and its
  evidence citing every id in a summary item plus detailed row
  citations;
* a re-evaluation in the same window bucket with no new failures
  is a no-op (no case spam, no evidence spam); further failures
  extend the open case with an update item citing ONLY the new
  ids; a SEALED case is never extended — a fresh ``#2`` case
  opens beside it;
* fingerprint-less rows fall back to a ``nofp:`` signature from
  the normalized error text; rows outside the window never count;
  a row's effective time is ``ts``, falling back to
  ``created_at``;
* an unreadable or absent ledger reads UNAVAILABLE with its
  reason — zero cases, never an exception (the /immune feeds
  surface states it in words);
* the env overrides move the threshold and malformed /
  non-positive overrides fall back to the defaults;
* the FP seat is the owner's verb: ``/immune mark-fp`` lands
  ``record_false_positive`` on the record — an ACTIVE antibody
  demotes to probation, a probation antibody's stint restarts —
  and the verdict journals as a ``false_positive`` observation;
* drift days and fever windows journal to the durable
  observation log (``drift_day`` / ``fever_started`` /
  ``fever_ended``), which survives a fresh construction (the
  restart simulation), skips torn lines, and never raises on
  append — a broken path degrades with the reason recorded;
* lazy evaluation rides the status / casefile / feeds reads;
  feeds create NO antibodies and NO enforcement receipts;
* the P1 healing ticket's immune-state evidence carries the
  observation journal (records + log) inside its already-sourced
  evidence field — the sources ∪ absent partition stays exact;
* the default observation path resolves to
  ``~/.nomorals/data/immune-observations.jsonl`` with NO env
  override set (the resolver smoke the design demands).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.immune import wiring
from nomorals.immune.chat import control_immune, get_immune_state, reset_immune_state
from nomorals.immune.feeds import ObservationLog, evaluate_failure_feed, feed_policy

NOW = 1_700_000_000.0

#: The failures table in its migrated shape (base DDL + the
#: attribution columns the registry's ledger writer fills:
#: family, fingerprint, lesson, ts, skill).
FAILURES_DDL = """
CREATE TABLE failures (
    id TEXT PRIMARY KEY,
    error TEXT NOT NULL DEFAULT '',
    error_type TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolution TEXT NOT NULL DEFAULT '',
    times_seen INTEGER NOT NULL DEFAULT 1,
    root_cause TEXT NOT NULL DEFAULT '',
    family TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL DEFAULT '',
    skill TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL DEFAULT 0
)
"""


def _failures_db(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "app.db")
    conn.execute(FAILURES_DDL)
    conn.commit()
    return conn


def _plant_failure(conn, fid, *, tool="web_fetch", fingerprint="fp-1",
                   error="connection refused", ts=NOW - 60.0,
                   created_at=None, source="tool"):
    """One ledger row in the registry writer's own shape."""
    conn.execute(
        "INSERT INTO failures (id, source, summary, error, family, "
        "lesson, ts, fingerprint, skill, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (fid, source, f"{tool} failed {error}", error, "tool", "",
         ts, fingerprint, tool, ts if created_at is None else created_at))
    conn.commit()


def _plant_group(conn, prefix, count, **kwargs):
    base_ts = kwargs.pop("ts", NOW - 300.0)
    for i in range(count):
        _plant_failure(conn, f"{prefix}-{i}", ts=base_ts + i * 10,
                       **kwargs)


def _plant_recent(conn, prefix, count, **kwargs):
    """Plant rows inside the trailing window at REAL wall time —
    the chat surface's lazy evaluation reads with time.time()."""
    _plant_group(conn, prefix, count, ts=time.time() - 300.0, **kwargs)


@pytest.fixture()
def immune_env(tmp_path, monkeypatch):
    """Isolate every shared immune store on tmp files."""
    monkeypatch.setenv(
        wiring.CORPUS_STORE_ENV_VAR, str(tmp_path / "immune-selfcorpus.json"))
    monkeypatch.setenv(
        wiring.ANTIBODY_STORE_ENV_VAR, str(tmp_path / "immune-antibodies.json"))
    monkeypatch.setenv(
        wiring.STORE_ENV_VAR, str(tmp_path / "immune-baseline.json"))
    obs_path = tmp_path / "immune-observations.jsonl"
    monkeypatch.setenv(wiring.OBSERVATIONS_LOG_ENV_VAR, str(obs_path))
    wiring.reset_shared_corpus_stores()
    wiring.reset_shared_antibody_stores()
    wiring.reset_shared_monitors()
    wiring.reset_shared_observation_logs()
    key = f"p5-{time.time_ns()}"
    reset_immune_state(key)
    yield key, tmp_path, obs_path
    reset_immune_state(key)
    wiring.reset_shared_corpus_stores()
    wiring.reset_shared_antibody_stores()
    wiring.reset_shared_monitors()
    wiring.reset_shared_observation_logs()


def _feed_cases(state):
    return [c for c in state.casefiles
            if isinstance(c.event, dict)
            and c.event.get("kind") == "tool_failure_recurrence"]


# ── the failure-ledger feed ─────────────────────────────────────

class TestFailureFeedThreshold:

    def test_below_threshold_opens_no_case(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 4)
        casefiles: list = []
        report = evaluate_failure_feed(conn, casefiles, now=NOW)
        assert report["available"] is True
        assert report["rows"] == 4
        assert report["groups_over_threshold"] == 0
        assert report["opened"] == []
        assert casefiles == []

    def test_at_threshold_opens_one_case_with_citations(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 5, tool="web_fetch", fingerprint="fp-9",
                     error="connection refused")
        casefiles: list = []
        report = evaluate_failure_feed(conn, casefiles, now=NOW)
        assert report["opened"] and len(report["opened"]) == 1
        assert len(casefiles) == 1
        case = casefiles[0]
        event = case.event
        assert event["kind"] == "tool_failure_recurrence"
        assert event["source"] == "failure_ledger_feed"
        assert event["tool"] == "web_fetch"
        assert event["signature"] == "fp-9"
        assert event["count"] == 5
        assert event["first_ts"] <= event["last_ts"]
        assert event["failure_ids"] == [f"f-{i}" for i in range(5)]
        assert event["feed_key"].startswith(
            "tool_failure:web_fetch:fp-9:")
        summary = [e for e in case.evidence
                   if e.get("kind") == "failure_ledger_summary"]
        assert len(summary) == 1
        assert summary[0]["failure_ids"] == [f"f-{i}" for i in range(5)]
        details = [e for e in case.evidence
                   if e.get("kind") == "failure_ledger_row"]
        assert len(details) == 5
        assert {d["failure_id"] for d in details} == {
            f"f-{i}" for i in range(5)}

    def test_groups_are_per_tool_and_signature(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "a", 5, tool="web_fetch", fingerprint="fp-1")
        _plant_group(conn, "b", 5, tool="mailer", fingerprint="fp-2")
        _plant_group(conn, "c", 3, tool="web_fetch", fingerprint="fp-3")
        casefiles: list = []
        report = evaluate_failure_feed(conn, casefiles, now=NOW)
        assert report["groups"] == 3
        assert report["groups_over_threshold"] == 2
        assert len(casefiles) == 2
        assert {c.event["tool"] for c in casefiles} == {
            "web_fetch", "mailer"}


class TestFailureFeedIdempotence:

    def test_reevaluation_same_bucket_is_noop(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 5)
        casefiles: list = []
        evaluate_failure_feed(conn, casefiles, now=NOW)
        evidence_len = len(casefiles[0].evidence)
        report = evaluate_failure_feed(conn, casefiles, now=NOW + 30.0)
        assert report["opened"] == [] and report["extended"] == []
        assert len(casefiles) == 1
        assert len(casefiles[0].evidence) == evidence_len

    def test_extension_cites_only_new_ids(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 5)
        casefiles: list = []
        evaluate_failure_feed(conn, casefiles, now=NOW)
        _plant_failure(conn, "f-5", ts=NOW - 20.0)
        _plant_failure(conn, "f-6", ts=NOW - 10.0)
        report = evaluate_failure_feed(conn, casefiles, now=NOW + 30.0)
        assert report["opened"] == []
        assert len(report["extended"]) == 1
        assert len(casefiles) == 1
        updates = [e for e in casefiles[0].evidence
                   if e.get("kind") == "failure_ledger_update"]
        assert len(updates) == 1
        assert updates[0]["new_failure_ids"] == ["f-5", "f-6"]
        assert updates[0]["count"] == 7

    def test_sealed_case_never_extended_fresh_case_opens(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 5)
        casefiles: list = []
        evaluate_failure_feed(conn, casefiles, now=NOW)
        first = casefiles[0]
        first.seal()
        evidence_len = len(first.evidence)
        _plant_failure(conn, "f-5", ts=NOW - 5.0)
        report = evaluate_failure_feed(conn, casefiles, now=NOW + 30.0)
        assert len(report["opened"]) == 1
        assert report["opened"][0].endswith("#2")
        assert len(casefiles) == 2
        assert len(first.evidence) == evidence_len  # untouched
        assert casefiles[1].event["feed_key"].endswith("#2")


class TestFailureFeedSignaturesAndWindow:

    def test_fingerprintless_rows_fall_back_to_nofp(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "n", 5, fingerprint="",
                     error="TLS handshake timeout")
        _plant_group(conn, "m", 5, fingerprint="",
                     error="dns resolution failed")
        casefiles: list = []
        evaluate_failure_feed(conn, casefiles, now=NOW)
        assert len(casefiles) == 2
        signatures = {c.event["signature"] for c in casefiles}
        assert all(s.startswith("nofp:") for s in signatures)
        assert len(signatures) == 2  # different errors, different sigs

    def test_rows_outside_window_never_count(self, tmp_path):
        conn = _failures_db(tmp_path)
        _plant_group(conn, "old", 5, ts=NOW - 7200.0)
        _plant_group(conn, "new", 4)
        casefiles: list = []
        report = evaluate_failure_feed(conn, casefiles, now=NOW)
        assert report["rows"] == 4
        assert casefiles == []

    def test_effective_ts_falls_back_to_created_at(self, tmp_path):
        conn = _failures_db(tmp_path)
        for i in range(5):
            _plant_failure(conn, f"f-{i}", ts=0, created_at=NOW - 100.0 + i)
        casefiles: list = []
        report = evaluate_failure_feed(conn, casefiles, now=NOW)
        assert report["rows"] == 5
        assert len(casefiles) == 1


class TestFailureFeedHonesty:

    def test_unavailable_without_db(self):
        casefiles: list = []
        report = evaluate_failure_feed(None, casefiles, now=NOW)
        assert report["available"] is False
        assert "database" in report["reason"]
        assert casefiles == []

    def test_unavailable_with_unreadable_ledger(self):
        class _BrokenDb:
            def execute(self, sql, params=()):
                raise RuntimeError("no such table: failures")

        casefiles: list = []
        report = evaluate_failure_feed(_BrokenDb(), casefiles, now=NOW)
        assert report["available"] is False
        assert "unreadable" in report["reason"]
        assert casefiles == []

    def test_env_override_moves_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NM_IMMUNE_FEED_MIN_FAILURES", "3")
        assert feed_policy() == (3, 3600.0)
        conn = _failures_db(tmp_path)
        _plant_group(conn, "f", 3)
        casefiles: list = []
        evaluate_failure_feed(conn, casefiles, now=NOW)
        assert len(casefiles) == 1

    def test_malformed_env_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("NM_IMMUNE_FEED_MIN_FAILURES", "abc")
        monkeypatch.setenv("NM_IMMUNE_FEED_WINDOW_S", "-5")
        assert feed_policy() == (5, 3600.0)
        monkeypatch.setenv("NM_IMMUNE_FEED_MIN_FAILURES", "0")
        assert feed_policy() == (5, 3600.0)


# ── the observation journal ─────────────────────────────────────

class TestObservationLog:

    def test_records_survive_fresh_construction(self, tmp_path):
        path = tmp_path / "obs.jsonl"
        log = wiring.shared_observation_log(path)
        assert log.append("drift_day", {"date": "2026-11-10"}) is True
        assert log.append("fever_started", {"reason": "probe"}) is True
        wiring.reset_shared_observation_logs()
        fresh = wiring.shared_observation_log(path)
        assert fresh is not log
        assert fresh.count() == 2
        assert fresh.count("drift_day") == 1
        assert fresh.records("fever_started")[0]["reason"] == "probe"

    def test_torn_lines_are_skipped(self, tmp_path):
        path = tmp_path / "obs.jsonl"
        log = ObservationLog(path)
        log.append("drift_day", {"date": "2026-11-10"})
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("this is not json\n")
            fh.write('{"kind": "fever_started", "ts": 1.0, "rea')
        assert log.count() == 1
        assert log.append("fever_ended", {"resolved": True}) is True
        assert log.count() == 2

    def test_append_never_raises_and_degrades(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        log = ObservationLog(blocker / "obs.jsonl")
        assert log.append("drift_day", {"date": "2026-11-10"}) is False
        assert log.degraded_reason != ""
        assert "unwritable" in log.degraded_reason

    def test_default_observations_path_smoke(self, monkeypatch):
        monkeypatch.delenv(wiring.OBSERVATIONS_LOG_ENV_VAR, raising=False)
        path = wiring.default_observations_path()
        assert path == Path.home() / ".nomorals" / "data" / \
            "immune-observations.jsonl"
        log = wiring.shared_observation_log()
        assert log.path == path
        wiring.reset_shared_observation_logs()


# ── the chat surface: feeds, mark-fp, journal hooks ─────────────

class TestChatSurface:

    def test_status_and_casefile_reads_evaluate_lazily(
            self, immune_env, tmp_path):
        key, _tmp, _obs = immune_env
        conn = _failures_db(tmp_path)
        _plant_recent(conn, "f", 5)
        context = SimpleNamespace(db=conn)
        state = get_immune_state(key)
        assert _feed_cases(state) == []
        out = control_immune("status", context=context, chat_key=key)
        assert "immune status" in out
        assert len(_feed_cases(state)) == 1

    def test_feeds_verb_renders_counts_and_unavailable(
            self, immune_env, tmp_path):
        key, _tmp, _obs = immune_env
        out = control_immune("feeds", context=None, chat_key=key)
        assert "UNAVAILABLE" in out
        assert "no feed cases opened" in out
        conn = _failures_db(tmp_path)
        _plant_recent(conn, "f", 5)
        context = SimpleNamespace(db=conn)
        out = control_immune("feeds", context=context, chat_key=key)
        assert "UNAVAILABLE" not in out
        assert "5 tool failure row(s)" in out
        assert "opened 1 case(s)" in out
        out = control_immune("feeds", context=context, chat_key=key)
        assert "opened 0 case(s)" in out

    def test_feeds_create_no_antibodies_or_receipts(
            self, immune_env, tmp_path):
        key, _tmp, _obs = immune_env
        conn = _failures_db(tmp_path)
        _plant_recent(conn, "f", 6)
        context = SimpleNamespace(db=conn)
        state = get_immune_state(key)
        antibodies_before = len(state.store.all())
        receipts_before = len(state.receipts)
        control_immune("feeds", context=context, chat_key=key)
        control_immune("status", context=context, chat_key=key)
        assert len(state.store.all()) == antibodies_before
        assert len(state.receipts) == receipts_before
        assert len(_feed_cases(state)) == 1

    def test_drift_and_fever_observations_journal(self, immune_env):
        key, _tmp, _obs = immune_env
        control_immune("drift add 100 5 2 10", chat_key=key)
        control_immune("fever start probe-fever 120", chat_key=key)
        control_immune("fever end", chat_key=key)
        state = get_immune_state(key)
        assert state.observations.count("drift_day") == 1
        assert state.observations.count("fever_started") == 1
        ended = state.observations.records("fever_ended")
        assert len(ended) == 1
        assert ended[0]["reason"] == "probe-fever"


class TestMarkFp:

    def _add_active(self, key, pattern="zxq-marker-alpha"):
        out = control_immune(
            f"antibodies add {pattern} owner planted pattern",
            chat_key=key)
        assert "antibody accepted" in out
        state = get_immune_state(key)
        entry = state.store.all()[0]
        assert entry["status"] == "active"
        return state, entry

    def test_mark_fp_demotes_active_antibody(self, immune_env):
        key, _tmp, _obs = immune_env
        state, entry = self._add_active(key)
        out = control_immune(
            f"mark-fp {entry['id'][:8]} fired on a benign paste",
            chat_key=key)
        assert "demoted to probation" in out
        record = state.store.get(entry["id"])
        assert record["status"] == "probation"
        assert record["false_positive_count"] == 1
        fps = state.observations.records("false_positive")
        assert len(fps) == 1
        assert fps[0]["antibody_id"] == entry["id"]
        assert fps[0]["status_before"] == "active"

    def test_mark_fp_restarts_probation_stint(self, immune_env):
        key, _tmp, _obs = immune_env
        state = get_immune_state(key)
        candidate = state.thymus.propose(
            "zxq-marker-beta", "virus", "learned in suite")
        accepted = state.thymus.accept(candidate)
        state.store.add_learned(accepted)
        state.store.match("content with zxq-marker-beta inside")
        state.store.match("more zxq-marker-beta content")
        before = state.store.get(accepted.id)
        assert before["status"] == "probation"
        assert before["probation_matches"] == 2
        out = control_immune(f"mark-fp {accepted.id[:8]}", chat_key=key)
        assert "stint restarted" in out
        after = state.store.get(accepted.id)
        assert after["probation_matches"] == 0
        assert after["false_positive_count"] == 1

    def test_mark_fp_unknown_id(self, immune_env):
        key, _tmp, _obs = immune_env
        out = control_immune("mark-fp deadbeef", chat_key=key)
        assert "no antibody" in out


# ── P1 coherence ────────────────────────────────────────────────

class TestHealingTicketCoherence:

    def test_state_ticket_evidence_carries_observations(self, immune_env):
        from nomorals.agents.healing_ticket import HEALING_FIELDS, healing_ticket_for

        key, _tmp, obs_path = immune_env
        control_immune("drift add 100 5 2 10", chat_key=key)
        state = get_immune_state(key)
        ticket = healing_ticket_for(
            f"state:{key}", immune_states={key: state})
        assert ticket["found"]
        sourced = set(ticket["sources"])
        absent = set(ticket["absent"])
        assert sourced | absent == set(HEALING_FIELDS)
        assert not (sourced & absent)
        observations = ticket["evidence"]["observations"]
        assert observations["records"] == 1
        assert observations["log"] == str(obs_path)
