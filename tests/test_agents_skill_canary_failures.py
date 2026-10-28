"""Tail-25 pins: skill_canary.py failure values + NOT-recording.

Verdict: 1 convert / 8 exclusions. Site 332 (evaluate's decision UPDATE
failing still returned ok True + "promoted"/"reverted" while the run row
stayed "running") is the convert: the returned dict now reports the
record failure honestly, and settle_canaries skips non-ok outcomes
(required companion guard — without it, an ok-False outcome fell through
to the revert branch and marked the edit regressed; the guard also
closes the "no running canary" -> revert hole). The notification value
here is the decision itself: on record failure nothing is committed and
nothing is marked, so the run is retried next settle instead of being
acted on under a false record.

Every other handler is documented never-raises (or best-effort); each
pin drives the handler into its except path and pins the exact fallback
value plus the fact that the agents failures ledger is unchanged
(snapshot before/after, tail-8 method). skill_canary.py calls no record
function anywhere (grep-proven), so the ledger must not move.

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

import time
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.agents import skill_canary as SC


def _ledger():
    return list(ag_fail.recent_failures())


class _FakeDB:
    """In-memory stand-in for the canary tables, with raisable halves."""

    def __init__(self):
        self.runs = []
        self.obs = []
        self.versions = []
        self.edits = []
        self.executed = []
        self.query_raises = False
        self.query_one_raises = False
        self.execute_raises = False

    def query(self, sql, params=()):
        if self.query_raises:
            raise RuntimeError("db down (query)")
        if "canary_observations" in sql:
            agg = {}
            for c, v, s in self.obs:
                if c != params[0]:
                    continue
                ok, n = agg.get(v, (0, 0))
                agg[v] = (ok + (1 if s else 0), n + 1)
            return [{"version_hash": v, "success": 1, "n": ok}
                    for v, (ok, n) in agg.items() if ok] + \
                   [{"version_hash": v, "success": 0, "n": n - ok}
                    for v, (ok, n) in agg.items() if n - ok]
        if "skill_versions" in sql:
            rows = [r for r in self.versions if r["skill_name"] == params[0]]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return rows[:params[1]]
        if "canary_runs" in sql:
            rows = list(self.runs)
            if params:
                rows = [r for r in rows if r["skill_name"] == params[0]]
            rows.sort(key=lambda r: r["started_at"], reverse=True)
            return rows[:params[-1]]
        return []

    def query_one(self, sql, params=()):
        if self.query_one_raises:
            raise RuntimeError("db down (query_one)")
        if "skill_versions" in sql and "ORDER BY created_at DESC LIMIT 1" in sql:
            rows = [r for r in self.versions if r["skill_name"] == params[0]]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return rows[0] if rows else None
        if "skill_versions" in sql:
            for r in self.versions:
                if r["skill_name"] == params[0] and r["version_hash"] == params[1]:
                    return r
            return None
        if "canary_runs" in sql:
            rows = [r for r in self.runs
                    if r["skill_name"] == params[0] and r["status"] == "running"]
            rows.sort(key=lambda r: r["started_at"], reverse=True)
            return rows[0] if rows else None
        if "skill_edits" in sql:
            for e in self.edits:
                if e["skill_name"] == params[0] and e["after_hash"] == params[1]:
                    return e
            return None
        return None

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if self.execute_raises:
            raise RuntimeError("db down (execute)")
        if sql.startswith("INSERT INTO skill_versions"):
            self.versions.append({"id": params[0], "skill_name": params[1],
                                  "version_hash": params[2], "parent_hash": params[3],
                                  "body": params[4], "source": params[5],
                                  "created_at": params[6]})
        elif sql.startswith("INSERT INTO canary_runs"):
            self.runs.append({"id": params[0], "skill_name": params[1],
                              "canary_hash": params[2], "baseline_hash": params[3],
                              "fraction": params[4], "status": "running",
                              "decision": "", "decision_detail": None,
                              "started_at": params[5], "decided_at": 0})
        elif sql.startswith("INSERT INTO canary_observations"):
            self.obs.append((params[1], params[2], bool(params[3])))
        elif sql.startswith("UPDATE canary_runs"):
            for r in self.runs:
                if r["id"] == params[4]:
                    r["status"], r["decision"], r["decision_detail"], r["decided_at"] = params[:4]
        return None


def _ctx(db):
    return SimpleNamespace(db=db)


def _seed_run(db, skill="sk", canary_n=20, baseline_n=20, canary_ok=20, baseline_ok=10):
    db.runs.append({"id": "run1", "skill_name": skill, "canary_hash": "CH",
                    "baseline_hash": "BH", "fraction": 0.2, "status": "running",
                    "decision": "", "decision_detail": None,
                    "started_at": time.time() - 100, "decided_at": 0})
    for _ in range(canary_ok):
        db.obs.append(("run1", "CH", True))
    for _ in range(canary_n - canary_ok):
        db.obs.append(("run1", "CH", False))
    for _ in range(baseline_ok):
        db.obs.append(("run1", "BH", True))
    for _ in range(baseline_n - baseline_ok):
        db.obs.append(("run1", "BH", False))
    return db.runs[-1]


def test_from_row_corrupt_detail_returns_empty():
    """80: corrupt decision_detail JSON -> {} (control: valid -> parsed,
    None -> {}). The decision/status survive; only the detail numbers
    are dropped — missing, never fabricated."""
    before = _ledger()
    row = {"id": "r", "skill_name": "s", "canary_hash": "c", "baseline_hash": "b",
           "fraction": 0.2, "status": "running", "decision": "",
           "decision_detail": "{bad json", "started_at": 0, "decided_at": 0}
    assert SC.CanaryRun.from_row(row).decision_detail == {}
    row["decision_detail"] = '{"canary": {"ok": 3}}'
    assert SC.CanaryRun.from_row(row).decision_detail == {"canary": {"ok": 3}}
    row["decision_detail"] = None
    assert SC.CanaryRun.from_row(row).decision_detail == {}
    assert _ledger() == before


def test_latest_hash_db_failure_returns_empty():
    """115: query failure -> "" (control: latest row wins, empty -> "").
    "" is the documented no-versions value. Chain-linkage note: if the
    read fails but a later INSERT succeeds, the new version's parent
    is "" — but nothing walks parent_hash links (only the schema and
    this module reference it), so no consumer is misled today."""
    before = _ledger()
    db = _FakeDB()
    db.query_one_raises = True
    assert SC.CanaryRollout(_ctx(db)).latest_hash("sk") == ""
    db = _FakeDB()
    db.versions.append({"skill_name": "sk", "version_hash": "H2", "created_at": 2})
    db.versions.append({"skill_name": "sk", "version_hash": "H1", "created_at": 1})
    assert SC.CanaryRollout(_ctx(db)).latest_hash("sk") == "H2"
    assert SC.CanaryRollout(_ctx(_FakeDB())).latest_hash("sk") == ""
    assert _ledger() == before


def test_get_body_db_failure_returns_none():
    """140: query failure -> None (control: stored body, missing -> None).
    None is the documented miss value: restore_version fails (no restore
    happens) and resolve_body falls back to the live baseline. The
    restore error text ("no such version") misattributes a DB failure,
    but the outcome is safe — nothing is restored on a failure."""
    before = _ledger()
    db = _FakeDB()
    db.query_one_raises = True
    assert SC.CanaryRollout(_ctx(db)).get_body("sk", "H") is None
    db = _FakeDB()
    db.versions.append({"skill_name": "sk", "version_hash": "H", "body": "BODY", "created_at": 1})
    assert SC.CanaryRollout(_ctx(db)).get_body("sk", "H") == "BODY"
    assert SC.CanaryRollout(_ctx(db)).get_body("sk", "ZZ") is None
    assert _ledger() == before


def test_versions_db_failure_returns_empty():
    """151: query failure -> [] (control: rows -> list). Display-only;
    [] is the no-versions value and no action follows from it."""
    before = _ledger()
    db = _FakeDB()
    db.query_raises = True
    assert SC.CanaryRollout(_ctx(db)).versions("sk") == []
    db = _FakeDB()
    db.versions.append({"skill_name": "sk", "version_hash": "H", "created_at": 1})
    assert len(SC.CanaryRollout(_ctx(db)).versions("sk")) == 1
    assert _ledger() == before


def test_active_db_failure_returns_none():
    """206: query failure -> None (control: running row, none -> None).
    None is the documented no-active-run value: routing falls back to
    baseline, observe drops (best-effort contract), evaluate reports
    "no running canary" — safe outcomes, no decision made on a failure."""
    before = _ledger()
    db = _FakeDB()
    db.query_one_raises = True
    assert SC.CanaryRollout(_ctx(db)).active("sk") is None
    db = _FakeDB()
    _seed_run(db)
    assert SC.CanaryRollout(_ctx(db)).active("sk") is not None
    assert SC.CanaryRollout(_ctx(_FakeDB())).active("sk") is None
    assert _ledger() == before


def test_counts_db_failure_returns_zeros():
    """255: query failure -> zero counts (control: real counts). The
    zeros cannot cause a wrong promotion: decide() returns
    "inconclusive" on empty canary data, so an age-ready run with no
    readable observations keeps running instead of promoting."""
    before = _ledger()
    db = _FakeDB()
    db.query_raises = True
    run = SC.CanaryRun("run1", "sk", "CH", "BH", 0.2)
    assert SC.CanaryRollout(_ctx(db))._counts(run) == {"canary": (0, 0), "baseline": (0, 0)}
    db = _FakeDB()
    _seed_run(db, canary_n=3, baseline_n=2, canary_ok=2, baseline_ok=1)
    assert SC.CanaryRollout(_ctx(db))._counts(run) == {"canary": (2, 3), "baseline": (1, 2)}
    assert _ledger() == before


def test_proof_check_failure_refuses_promotion():
    """315: proof_check raising -> proven False -> revert (fail-closed).
    Control: proof ok -> promote. Untested versions are refused, never
    promoted — the failure direction is the safe one, by design."""
    before = _ledger()
    db = _FakeDB()
    _seed_run(db)

    def _boom(skill, vhash):
        raise RuntimeError("proof down")

    assert SC.CanaryRollout(_ctx(db)).evaluate("sk", proof_check=_boom)["decision"] == "revert"
    db = _FakeDB()
    _seed_run(db)
    assert SC.CanaryRollout(_ctx(db)).evaluate("sk", proof_check=lambda s, v: True)["decision"] == "promote"
    assert _ledger() == before


def test_evaluate_update_failure_reports_not_confirmed():
    """332 CONVERT: failed decision UPDATE -> ok False + error string,
    row stays "running", nothing committed or marked. Control: working
    UPDATE -> promote + row promoted. The settle guard (companion pin
    below) is what makes this safe: without it, settle fell through
    to revert on a non-decision outcome."""
    before = _ledger()
    db = _FakeDB()
    row = _seed_run(db)
    rollout = SC.CanaryRollout(_ctx(db))
    db.execute_raises = True
    res = rollout.evaluate("sk")
    assert res.get("ok") is False
    assert res.get("error") == "couldn't record canary decision — try again"
    assert row["status"] == "running"
    db = _FakeDB()
    row = _seed_run(db)
    res = SC.CanaryRollout(_ctx(db)).evaluate("sk")
    assert res["status"] == "promoted" and row["status"] == "promoted"
    assert _ledger() == before


def test_settle_skips_non_ok_outcomes():
    """Settle companion guard: an ok-False evaluate outcome is skipped —
    no commit, no regressed marking, run not settled. Control: a promote
    outcome commits the body and marks the edit applied (proves the
    stub harness exercises the branches, so the skip is not vacuous)."""
    before = _ledger()
    import nomorals.agents.skill_evolution as SE

    class _StubRollout:
        def __init__(self, context, outcome, body):
            self._outcome = outcome
            self._body = body

        def history(self, limit=50):
            return [SC.CanaryRun("run1", "sk", "CH", "BH", 0.2)]

        def evaluate(self, skill_name):
            return self._outcome

        def get_body(self, skill_name, vhash):
            return self._body

    def _run_settle(outcome, body):
        db = _FakeDB()
        db.edits.append({"id": "e1", "skill_name": "sk", "after_hash": "CH", "status": "canary"})
        loop = SE.SkillEvolutionLoop.__new__(SE.SkillEvolutionLoop)
        loop.db = db
        loop.context = _ctx(db)
        committed = []
        loop._commit_body = lambda skill, b: committed.append((skill, b))
        with mock.patch.object(SC, "CanaryRollout",
                               lambda context: _StubRollout(context, outcome, body)):
            settled = loop.settle_canaries()
        return db, committed, settled

    db, committed, settled = _run_settle({"ok": False, "error": "no running canary"}, None)
    assert committed == []
    assert settled == []
    assert not [s for s in db.executed if "status='regressed'" in s]
    assert not [s for s in db.executed if "status='applied'" in s]
    db, committed, settled = _run_settle(
        {"ok": True, "decision": "promote", "canary": {"ok": 20, "n": 20},
         "baseline": {"ok": 10, "n": 20}}, "CANARY-BODY")
    assert committed == [("sk", "CANARY-BODY")]
    assert [s for s in db.executed if "status='applied'" in s]
    assert settled and settled[0]["decision"] == "promote"
    assert _ledger() == before


def test_history_db_failure_returns_empty():
    """349: query failure -> [] (control: rows -> runs). Display-only;
    [] is the no-runs value and no action follows from it."""
    before = _ledger()
    db = _FakeDB()
    db.query_raises = True
    assert SC.CanaryRollout(_ctx(db)).history("sk") == []
    db = _FakeDB()
    _seed_run(db)
    assert len(SC.CanaryRollout(_ctx(db)).history("sk")) == 1
    assert _ledger() == before
