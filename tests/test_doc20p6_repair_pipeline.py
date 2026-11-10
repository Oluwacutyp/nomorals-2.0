"""Doc 20 P6 — the repair pipeline, rungs 7–10.

The acceptance shapes are the doc's own §9 battery, run end-to-end
on a miniature fixture repo whose organ carries a REAL defect (its
test fails against the shipped code, its runtime probe raises):

* a fixing candidate flows proposed → suite → canary → gate →
  promoted, every transition journaled on the incident, and the
  journal's verified-fix table learns the repair;
* PROOF 1 — a candidate that fixes the defect but regresses a
  sibling behaviour is caught by the canary probes and rolled back
  AUTOMATICALLY, with the rollback on the ticket record;
* PROOF 2 — the write-path audit: production writes exist only
  inside the guarded promotion path, promotion re-derives the
  ticket trail from the journal and refuses (writing nothing) when
  the trail is forged or incomplete, and a repair with no incident
  is refused by construction;
* PROOF 3 — a candidate that fails the sandbox suite NEVER reaches
  canary (no canary record, no canary worktree, production clean).

Plus the gate shapes (consequential parks for the owner; approve
promotes; reject discards; a post-promotion monitor failure rolls
the promotion back; the retained rollback restores a promoted run),
the frozen-file refusal, the security suite step, and the doc-9
scoping pin: the pipeline imports and writes NOTHING mission-shaped.
"""

from __future__ import annotations

import ast
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents import repair_pipeline as rp
from nomorals.agents.repair_pipeline import (
    PatchCandidate,
    RepairPipeline,
    RepairRefused,
    assess_patch,
)
from nomorals.core.diff import unified_diff
from nomorals.core.incidents import IncidentJournal
from nomorals.os.snapshots import SnapshotManager

WIDGET_DEFECTIVE = (
    '"""Fixture organ with a real defect."""\n\n\n'
    "def compute():\n"
    '    raise ValueError("widget miscalibrated")\n\n\n'
    "def calibrate():\n"
    '    return "ok"\n'
)
WIDGET_FIXED = WIDGET_DEFECTIVE.replace(
    '    raise ValueError("widget miscalibrated")\n',
    "    return 42\n")
WIDGET_REGRESSED = WIDGET_FIXED.replace(
    '    return "ok"\n', '    return "broken"\n')
TEST_WIDGET = (
    "from fixture_pkg.widget import compute\n\n\n"
    "def test_compute():\n"
    "    assert compute() == 42\n"
)
PROBE = (sys.executable, "-c",
         "from fixture_pkg.widget import compute, calibrate\n"
         "assert compute() == 42\n"
         "assert calibrate() == 'ok'\n")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _fixture_repo(base: Path) -> Path:
    repo = base / "repo"
    (repo / "fixture_pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "fixture_pkg" / "__init__.py").write_text(
        '__version__ = "0.1"\n', encoding="utf-8")
    (repo / "fixture_pkg" / "widget.py").write_text(
        WIDGET_DEFECTIVE, encoding="utf-8")
    (repo / "tests" / "test_widget.py").write_text(
        TEST_WIDGET, encoding="utf-8")
    (repo / "conftest.py").write_text("", encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fixture base")
    return repo


def _raise_defect() -> None:
    raise ValueError("widget miscalibrated")


def _journal(base: Path) -> tuple[IncidentJournal, int]:
    journal = IncidentJournal(path=str(base / "error_journal.db"))
    try:
        _raise_defect()
    except ValueError as exc:
        incident = journal.record_incident(exc, subsystem="fixture/widget")
    return journal, incident.id


def _snapshots(base: Path) -> tuple[SnapshotManager, Path]:
    home = base / "home"
    home.mkdir(parents=True, exist_ok=True)
    db_path = base / "live.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    return SnapshotManager(home=home, db_path=db_path), db_path


def _health_factory(state: dict):
    def factory(root, db_path):
        def _import():
            proc = subprocess.run(
                [sys.executable, "-c", "import fixture_pkg"],
                cwd=str(root), capture_output=True, text=True, timeout=60,
                env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"})
            return proc.returncode == 0, (
                "fixture package imports" if proc.returncode == 0
                else proc.stderr[-200:])

        def _db():
            if db_path is None:
                return True, "no database for this profile"
            if state.get("monitor_fail"):
                return False, "live database failed its check (fixture)"
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute("PRAGMA integrity_check").fetchone()
            finally:
                conn.close()
            return (row and row[0] == "ok"), "integrity: " + str(row)

        _import.__name__ = "fixture_import"
        _db.__name__ = "fixture_db"
        return [_import, _db]

    return factory


def _pipeline(base: Path, repo: Path, journal: IncidentJournal,
              snapshots, db_path, state: dict = None) -> RepairPipeline:
    return RepairPipeline(
        journal=journal, repo_dir=repo, work_root=base / "repair-runs",
        snapshots=snapshots,
        health_factory=_health_factory(state if state is not None else {}),
        db_path=db_path)


def _diff(new_text: str, old_text: str = WIDGET_DEFECTIVE,
          path: str = "fixture_pkg/widget.py") -> str:
    return unified_diff(old_text, new_text, path=path)


def _strategies(pipe: RepairPipeline, state) -> list:
    return [r["strategy"] for r in
            pipe._recoveries_for(state.incident_id)
            if (r.get("detail") or {}).get("run") == state.run_id]


@pytest.fixture()
def world(tmp_path: Path):
    repo = _fixture_repo(tmp_path)
    journal, incident_id = _journal(tmp_path)
    snapshots, db_path = _snapshots(tmp_path)
    return SimpleNamespace(
        base=tmp_path, repo=repo, journal=journal,
        incident_id=incident_id, snapshots=snapshots, db_path=db_path,
        head=_git(repo, "rev-parse", "HEAD"))


# ── the happy path: rungs 7–10 end to end ─────────────────────────


def test_happy_path_promotes_and_journals_every_rung(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    candidate = PatchCandidate(
        diff_text=_diff(WIDGET_FIXED), producer="test",
        rationale="compute returns the calibrated value",
        incident_id=world.incident_id, probes=(PROBE,))
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_PROMOTED

    # The production tree really changed, in exactly one commit.
    text = (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8")
    assert "return 42" in text
    assert _git(world.repo, "rev-list", "--count", "HEAD") == "2"
    assert state.run_id in _git(world.repo, "log", "-1", "--format=%B")
    assert _git(world.repo, "status", "--porcelain") == ""

    # Every rung transition is on the incident's journal trail, in
    # order, and only the promotion claims verified.
    strategies = _strategies(pipe, state)
    assert strategies[0] == rp.STRATEGY_PROPOSED
    assert strategies[-1] == rp.STRATEGY_PROMOTE
    assert rp.STRATEGY_CANARY in strategies
    assert rp.STRATEGY_GATE in strategies
    suite_records = [r for r in pipe._recoveries_for(world.incident_id)
                     if r["strategy"] == rp.STRATEGY_SUITE]
    assert len(suite_records) == 3  # tests + security + property
    assert all((r["detail"] or {}).get("ok") for r in suite_records)
    for rec in pipe._recoveries_for(world.incident_id):
        if rec["strategy"] == rp.STRATEGY_PROMOTE:
            assert rec["verified"] is True
            detail = rec["detail"]
            assert detail["from_sha"] == world.head
            assert detail["snapshot"]
        elif rec["strategy"].startswith("repair_"):
            assert rec["verified"] is False

    # The journal learned the fix (mined rule 3: verified only).
    try:
        _raise_defect()
    except ValueError as exc:
        known = world.journal.known_fix(exc, "fixture/widget")
    assert known is not None
    assert known["strategy"] == rp.STRATEGY_PROMOTE

    # P1's healing ticket projects the run with NO ticket changes:
    # the incident reads recovered and every rung record is a cited
    # healing-run ticket of its own.
    from nomorals.agents.healing_ticket import healing_ticket_for
    ticket = healing_ticket_for(
        "incident:" + str(world.incident_id), journal=world.journal)
    assert ticket["lifecycle"]["standing"] == "recovered"
    seen = set()
    for ref in ticket["related_refs"]:
        if ref.startswith("heal:"):
            sub = healing_ticket_for(ref, journal=world.journal)
            seen.add((sub["purpose"] or {}).get("strategy"))
    assert rp.STRATEGY_PROMOTE in seen
    assert rp.STRATEGY_SUITE in seen
    assert rp.STRATEGY_CANARY in seen

    # Scratch worktrees are gone; the run is reconstructable cold.
    assert not (world.base / "repair-runs" / state.run_id
                / "sandbox").exists()
    assert not (world.base / "repair-runs" / state.run_id
                / "canary").exists()
    cold = RepairPipeline(
        journal=IncidentJournal(
            path=str(world.base / "error_journal.db")),
        repo_dir=world.repo, work_root=world.base / "repair-runs")
    runs = cold.list_runs()
    assert [r.run_id for r in runs] == [state.run_id]
    assert runs[0].status == rp.STATUS_PROMOTED


def test_rerun_of_same_candidate_is_idempotent(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    candidate = PatchCandidate(
        diff_text=_diff(WIDGET_FIXED), producer="test",
        incident_id=world.incident_id, probes=(PROBE,))
    first = pipe.run(candidate)
    count = len(pipe._recoveries_for(world.incident_id))
    second = pipe.run(candidate)
    assert second.run_id == first.run_id
    assert second.status == rp.STATUS_PROMOTED
    assert len(pipe._recoveries_for(world.incident_id)) == count
    assert _git(world.repo, "rev-list", "--count", "HEAD") == "2"


# ── PROOF 1: canary regression → automatic rollback ───────────────


def test_canary_regression_rolls_back_automatically(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    candidate = PatchCandidate(
        diff_text=_diff(WIDGET_REGRESSED), producer="test",
        rationale="fixes compute, silently breaks calibrate",
        incident_id=world.incident_id, probes=(PROBE,))
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_CANARY_ROLLED_BACK

    # Production was never written.
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert _git(world.repo, "rev-parse", "HEAD") == world.head
    assert _git(world.repo, "status", "--porcelain") == ""

    # The ticket record SHOWS the automatic rollback, verified.
    strategies = _strategies(pipe, state)
    assert strategies[-1] == rp.STRATEGY_CANARY_ROLLBACK
    canary = state.detail_of(rp.STRATEGY_CANARY)
    assert canary["ok"] is False
    rollback = state.detail_of(rp.STRATEGY_CANARY_ROLLBACK)
    assert rollback["ok"] is True
    assert "tree restored" in rollback["restore"]
    assert not (world.base / "repair-runs" / state.run_id
                / "canary").exists()

    # And the journal did NOT learn this as a fix.
    try:
        _raise_defect()
    except ValueError as exc:
        assert world.journal.known_fix(exc, "fixture/widget") is None


# ── PROOF 3: suite failure never reaches canary ───────────────────


def test_suite_failure_never_reaches_canary(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    still_broken = WIDGET_DEFECTIVE.replace(
        '"""Fixture organ with a real defect."""',
        '"""Fixture organ, retitled but still defective."""')
    candidate = PatchCandidate(
        diff_text=_diff(still_broken), producer="test",
        incident_id=world.incident_id, probes=(PROBE,))
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_SUITE_FAILED

    strategies = _strategies(pipe, state)
    assert rp.STRATEGY_CANARY not in strategies
    assert rp.STRATEGY_CANARY_ROLLBACK not in strategies
    assert rp.STRATEGY_GATE not in strategies
    assert rp.STRATEGY_PROMOTE not in strategies
    failed = [r for r in pipe._recoveries_for(world.incident_id)
              if r["strategy"] == rp.STRATEGY_SUITE
              and not (r["detail"] or {}).get("ok")]
    assert failed and failed[0]["detail"]["kind"] == "tests"
    run_dir = world.base / "repair-runs" / state.run_id
    assert not (run_dir / "canary").exists()
    assert not (run_dir / "sandbox").exists()
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert _git(world.repo, "rev-parse", "HEAD") == world.head


def test_security_step_blocks_a_secret_in_the_patch(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    # Assembled from fragments so no full-format secret literal
    # ever sits in this source file (house rule for fixtures).
    planted = "AK" + "IA" + "1234567890ABCDEF"
    poisoned = WIDGET_FIXED.replace(
        "    return 42\n",
        "    return 42\n\n\nKEY = " + repr(planted) + "\n")
    candidate = PatchCandidate(
        diff_text=_diff(poisoned), producer="test",
        incident_id=world.incident_id, probes=(PROBE,))
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_SUITE_FAILED
    failed = [r for r in pipe._recoveries_for(world.incident_id)
              if r["strategy"] == rp.STRATEGY_SUITE
              and not (r["detail"] or {}).get("ok")]
    assert failed and failed[0]["detail"]["step"] == "security:added-lines"
    assert rp.STRATEGY_CANARY not in _strategies(pipe, state)


# ── PROOF 2: the write-path audit ─────────────────────────────────


def test_write_path_audit_structure():
    """Every production-write primitive in the pipeline module sits
    inside the guarded promotion path or the tree-restore helper;
    worktree applies are the only other apply_patch call site, and
    promotion's first act is the trail guard."""
    tree = ast.parse(Path(rp.__file__).read_text(encoding="utf-8"))
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}
    apply_sites = []
    commit_sites = []
    for name, fn in functions.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Name) and node.func.id == "apply_patch":
                apply_sites.append(name)
            if isinstance(node, ast.Call):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and sub.value == "commit":
                        commit_sites.append(name)
    assert sorted(set(apply_sites)) == ["_apply_to_worktree", "_promote"]
    assert set(commit_sites) == {"_promote"}
    promote = functions["_promote"]
    first = promote.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert first.value.func.attr == "_require_trail"
    # The worktree helper is never handed the production tree.
    source = Path(rp.__file__).read_text(encoding="utf-8")
    assert "_apply_to_worktree(\n            diff_text, self.repo_dir)" \
        not in source
    assert "self.repo_dir)" in source  # sanity: the check is not vacuous


def test_promotion_refuses_a_forged_trail(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    # Forge a run whose ticket shows only a proposal — no suite, no
    # canary — then demand the production write directly.
    import json as _json

    from nomorals.core.incidents import RecoveryRecord
    chash = rp.candidate_hash_of(_diff(WIDGET_FIXED))
    run_id = "rp-" + str(world.incident_id) + "-" + chash[:10]
    incident = world.journal.get_incident(world.incident_id)
    world.journal.record_recovery(RecoveryRecord(
        incident_id=world.incident_id, signature=incident.signature,
        strategy=rp.STRATEGY_PROPOSED,
        detail=_json.dumps({"run": run_id, "candidate": chash,
                            "status": rp.STATUS_PROPOSED, "ok": True,
                            "files": ["fixture_pkg/widget.py"]})))
    run_dir = world.base / "repair-runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "candidate.diff").write_text(
        _diff(WIDGET_FIXED), encoding="utf-8")
    state = pipe._load_run(run_id)
    assert state is not None
    with pytest.raises(RepairRefused):
        pipe._promote(state, by="owner")
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert _git(world.repo, "rev-parse", "HEAD") == world.head


def test_repair_without_an_incident_is_refused(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    with pytest.raises(RepairRefused):
        pipe.run(PatchCandidate(diff_text=_diff(WIDGET_FIXED),
                                producer="test", incident_id=None))
    with pytest.raises(RepairRefused):
        pipe.run(PatchCandidate(diff_text=_diff(WIDGET_FIXED),
                                producer="test", incident_id=99999))
    assert _git(world.repo, "rev-parse", "HEAD") == world.head


def test_frozen_file_candidate_refused_before_any_write(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    candidate = PatchCandidate(
        diff_text=unified_diff("a\n", "b\n",
                               path="nomorals/memory/persona.py"),
        producer="test", incident_id=world.incident_id)
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_REFUSED
    assert _strategies(pipe, state) == [rp.STRATEGY_REFUSED]
    run_dir = world.base / "repair-runs" / state.run_id
    assert not (run_dir / "sandbox").exists()
    assert not (run_dir / "canary").exists()
    assert _git(world.repo, "rev-parse", "HEAD") == world.head


# ── the gate: park, approve, reject ───────────────────────────────


_CONFIG_CREATION = (
    "--- /dev/null\n"
    "+++ b/fixture_pkg/config.py\n"
    "@@ -0,0 +1 @@\n"
    "+SETTING = 1\n"
)


def _consequential_candidate(incident_id: int) -> PatchCandidate:
    diff = _diff(WIDGET_FIXED) + _CONFIG_CREATION
    return PatchCandidate(
        diff_text=diff, producer="test",
        rationale="fix plus a new config module (sensitive path)",
        incident_id=incident_id, probes=(PROBE,))


def test_consequential_patch_parks_then_owner_approves(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    state = pipe.run(_consequential_candidate(world.incident_id))
    assert state.status == rp.STATUS_AWAITING
    gate = state.detail_of(rp.STRATEGY_GATE)
    assert gate["decision"] == "owner-required"
    assert "sensitive-path" in gate["assessment"]["risks"]
    # Parked means NOT written.
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert not (world.repo / "fixture_pkg" / "config.py").exists()
    assert _git(world.repo, "rev-parse", "HEAD") == world.head

    promoted = pipe.approve(state.run_id, by="owner")
    assert promoted.status == rp.STATUS_PROMOTED
    assert "return 42" in (world.repo / "fixture_pkg"
                           / "widget.py").read_text(encoding="utf-8")
    assert (world.repo / "fixture_pkg" / "config.py").exists()
    with pytest.raises(RepairRefused):
        pipe.approve(state.run_id, by="owner")


def test_owner_reject_discards_the_candidate(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    state = pipe.run(_consequential_candidate(world.incident_id))
    assert state.status == rp.STATUS_AWAITING
    rejected = pipe.reject(state.run_id, by="owner", reason="not now")
    assert rejected.status == rp.STATUS_REJECTED
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert not (world.base / "repair-runs" / state.run_id
                / "canary").exists()
    assert _git(world.repo, "rev-parse", "HEAD") == world.head


# ── rung 10: monitor rollback + the retained rollback ─────────────


def test_promotion_monitor_failure_rolls_back(world):
    state_flag: dict = {}
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path, state_flag)
    state = pipe.run(_consequential_candidate(world.incident_id))
    assert state.status == rp.STATUS_AWAITING
    state_flag["monitor_fail"] = True
    rolled = pipe.approve(state.run_id, by="owner")
    assert rolled.status == rp.STATUS_PROMOTION_ROLLED_BACK
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert not (world.repo / "fixture_pkg" / "config.py").exists()
    assert _git(world.repo, "rev-parse", "HEAD") == world.head
    assert _git(world.repo, "status", "--porcelain") == ""
    rollback = rolled.detail_of(rp.STRATEGY_ROLLBACK)
    assert rollback["phase"] == "promotion"
    assert rollback["stage"] == "monitor"
    assert rollback["snapshot"]
    assert rollback["ok"] is True
    assert world.snapshots.latest() is not None


def test_retained_rollback_restores_a_promoted_run(world):
    pipe = _pipeline(world.base, world.repo, world.journal,
                     world.snapshots, world.db_path)
    candidate = PatchCandidate(
        diff_text=_diff(WIDGET_FIXED), producer="test",
        incident_id=world.incident_id, probes=(PROBE,))
    state = pipe.run(candidate)
    assert state.status == rp.STATUS_PROMOTED
    rolled = pipe.rollback(state.run_id, by="owner", reason="drill")
    assert rolled.status == rp.STATUS_ROLLED_BACK
    assert (world.repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert _git(world.repo, "rev-parse", "HEAD") == world.head
    assert _git(world.repo, "status", "--porcelain") == ""
    with pytest.raises(RepairRefused):
        pipe.rollback(state.run_id, by="owner")


# ── assessment mirror + doc-9 scoping pins ────────────────────────


def test_assessment_matches_the_coding_agents_metrics():
    from nomorals.agents.coding import _diff_metrics
    battery = [
        _diff(WIDGET_FIXED),
        _diff(WIDGET_DEFECTIVE.replace("def compute():",
                                       "def compute_v2():")),
        _diff(WIDGET_FIXED) + _CONFIG_CREATION,
        unified_diff("x\n" * 60, "", path="fixture_pkg/big.py"),
        unified_diff("", "a\n", path="a/one.py")
        + unified_diff("", "b\n", path="b/two.py")
        + unified_diff("", "c\n", path="c/three.py")
        + unified_diff("", "d\n", path="d/four.py")
        + unified_diff("", "e\n", path="e/five.py"),
    ]
    for diff in battery:
        mine = assess_patch(diff)
        theirs = _diff_metrics(diff)
        hard = {"sensitive-path", "public-api-change", "file-deletion",
                "heavy-deletion"}
        expected = bool(hard.intersection(theirs["risks"])) or (
            theirs["file_count"] >= 5
            or (theirs["added"] + theirs["deleted"]) >= 400)
        assert mine["consequential"] == expected, diff
        assert mine["file_count"] == theirs["file_count"]
        assert mine["added"] == theirs["added"]
        assert mine["deleted"] == theirs["deleted"]


def test_pipeline_never_touches_mission_repair():
    """Doc-9 bind, pinned: rungs 7–10 repair CODE only.  Widening
    repair into missions would ride MissionRunner.self_heal /
    MissionStore, where doc 9's two narrow fixes are the stated
    prerequisite — this module must not go near them."""
    source = Path(rp.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    names = set()
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    assert not any("missions" in name or "mission_state" in name
                   for name in imported)
    # The shipped ladder organs are never imported either: rungs
    # 1–6 keep their exact P2 behaviour, unentangled.
    assert not any("error_system" in name or "selfheal" in name
                   or name.endswith("tools.registry")
                   for name in imported)
    assert "MissionRunner" not in names
    assert "self_heal" not in attrs


# ── the console front door ────────────────────────────────────────


def _fake_settings(base: Path):
    home = base / "clihome"
    (home / "data").mkdir(parents=True)
    return SimpleNamespace(
        home=str(home), home_path=home, db_path=base / "cli.db",
        blob_dir=None, storage=None, to_dict=lambda: {})


def test_cli_repair_surface(tmp_path: Path):
    from nomorals.cmdline.commands.repair import _cmd_repair
    settings = _fake_settings(tmp_path)
    repo = _fixture_repo(tmp_path)
    journal = IncidentJournal(
        path=str(tmp_path / "clihome" / "data" / "error_journal.db"))
    try:
        _raise_defect()
    except ValueError as exc:
        incident = journal.record_incident(exc, subsystem="fixture/widget")
    journal.close()

    def args(**kw):
        base = dict(task=[], json=False, incident=None, patch="",
                    probe=[], producer="", repo=str(repo), config=None)
        base.update(kw)
        return SimpleNamespace(**base)

    assert _cmd_repair(args(), settings) == 2
    assert _cmd_repair(args(task=["runs"]), settings) == 0

    patch_file = tmp_path / "fix.diff"
    patch_file.write_text(_diff(WIDGET_FIXED), encoding="utf-8")
    probe = (sys.executable + " -c \"from fixture_pkg.widget import "
             "compute; assert compute() == 42\"")
    rc = _cmd_repair(args(
        task=["run"], incident=incident.id, patch=str(patch_file),
        probe=[probe]), settings)
    # The console pipeline runs the REAL default health factory for
    # the canary; the fixture tree is not a Devon tree, so the
    # canary honestly fails and rolls back — the wiring is proven
    # end-to-end and the production fixture is untouched.
    assert rc == 1
    assert (repo / "fixture_pkg" / "widget.py").read_text(
        encoding="utf-8") == WIDGET_DEFECTIVE
    assert _cmd_repair(args(task=["runs"]), settings) == 0
    journal2 = IncidentJournal(
        path=str(tmp_path / "clihome" / "data" / "error_journal.db"))
    pipe = RepairPipeline(
        journal=journal2, repo_dir=repo,
        work_root=tmp_path / "clihome" / "data" / "repair-runs")
    runs = pipe.list_runs()
    assert len(runs) == 1
    assert runs[0].status == rp.STATUS_CANARY_ROLLED_BACK
