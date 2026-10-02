"""Ship gate: arena builds → evolution proposals → repo commits.

Covers nomorals/agents/arena/ship.py (promote_build, verify_proposal_files,
ship_queue, approve_ship, deny_ship) and the new-file edit support in
EvolutionAgent.apply (nomorals/agents/evolution.py).

Uses Database(":memory:")+migrate() for state and throwaway `git init`
repos for the apply/commit paths.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents.arena.ship import (
    approve_ship,
    deny_ship,
    promote_build,
    ship_queue,
    verify_proposal_files,
)
from nomorals.agents.evolution import EvolutionAgent, EvolutionProposal
from nomorals.agents.power import PowerMode
from nomorals.core.errors import ToolError
from nomorals.storage.db import Database


# ── fixtures ──────────────────────────────────────────────────────────────

CLEAN_MODULE = '"""A tiny shipped widget."""\n\n\ndef widget():\n    return 42\n'
HELPER_MODULE = '"""Helper."""\n\n\ndef helper():\n    return "ok"\n'
SWALLOWED = "def run():\n    try:\n        do_thing()\n    except:\n        pass\n"
SYNTAX_BROKEN = "def broken(:\n    pass\n"


class _FakePartner:
    power_default_on = True


class _FakeEvolution:
    work_branch = ""
    main_branch = "main"
    push_on_publish = False


class _FakeSettings:
    partner = _FakePartner()
    evolution = _FakeEvolution()


class FakeCtx:
    """Enough context for EvolutionAgent + the ship functions."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.settings = _FakeSettings()
        self.extras: dict = {}
        self.router = None
        # power mode active without touching the auto-activation machinery
        pm = PowerMode(self)
        pm._active = True
        self.extras["power"] = pm


def make_ctx() -> tuple[Database, FakeCtx]:
    db = Database(":memory:")
    db.migrate()
    return db, FakeCtx(db)


def git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                   capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"],
                   cwd=repo, check=True, capture_output=True)


def git_log(repo: Path) -> str:
    out = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True)
    return out.stdout


def make_build_dir(files: dict[str, str]) -> Path:
    """A fake arena build dir: {relative path: content}."""
    tmp = Path(tempfile.mkdtemp(prefix="arena-build-"))
    for rel, content in files.items():
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return tmp


def insert_build(db: Database, build_id: str, build_dir: Path, *,
                 name: str = "widget", status: str = "pending",
                 syntax: str = "ok", purpose: str = "a test widget",
                 topic: str = "widgets") -> None:
    specs = []
    for path in sorted(build_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(build_dir).as_posix()
            specs.append({"path": rel,
                          "chars": len(path.read_text(encoding="utf-8"))})
    db.execute(
        "INSERT INTO arena_builds "
        "(id, name, purpose, topic, files, dir, status, syntax, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (build_id, name, purpose, topic, json.dumps(specs), str(build_dir),
         status, syntax, time.time()),
    )


def build_row(db: Database, build_id: str) -> dict | None:
    return db.query_one("SELECT * FROM arena_builds WHERE id = ?",
                        (build_id,))


class ShipTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.db, self.ctx = make_ctx()
        self._tmp: list[Path] = []

    def tearDown(self) -> None:
        for path in self._tmp:
            shutil.rmtree(path, ignore_errors=True)

    def track(self, path: Path) -> Path:
        self._tmp.append(path)
        return path


# ── promote_build ─────────────────────────────────────────────────────────

class PromoteTests(ShipTestBase):
    def test_happy_path(self) -> None:
        build_dir = self.track(make_build_dir({
            "widget.py": CLEAN_MODULE,
            "sub/helper.py": HELPER_MODULE,
        }))
        insert_build(self.db, "build123", build_dir)
        pid = promote_build(self.db, self.ctx, "build123", "/tmp/repo")
        self.assertTrue(pid.startswith("evo-arena-build123-"), pid)

        prop = EvolutionAgent(self.ctx)._load(pid)
        self.assertIsNotNone(prop)
        assert prop is not None
        paths = sorted(e["path"] for e in prop.edits)
        self.assertEqual(paths, ["arena_builds/widget/sub/helper.py",
                                 "arena_builds/widget/widget.py"])
        self.assertTrue(all(e["old"] == "" for e in prop.edits))
        self.assertIn("[arena]", prop.instruction)
        self.assertIn("a test widget", prop.instruction)
        self.assertEqual(prop.status, "planned")
        # every shipped path lives under arena_builds/, never under nomorals/
        self.assertTrue(all(p.startswith("arena_builds/") for p in paths))
        self.assertFalse(any(p.startswith("nomorals/") for p in paths))

        row = build_row(self.db, "build123")
        assert row is not None
        self.assertEqual(row["status"], "promoted")

    def test_rejects_non_pending(self) -> None:
        build_dir = self.track(make_build_dir({"widget.py": CLEAN_MODULE}))
        insert_build(self.db, "b-approved", build_dir, status="approved")
        with self.assertRaises(ToolError) as cm:
            promote_build(self.db, self.ctx, "b-approved", "/tmp/repo")
        self.assertIn("not 'pending'", str(cm.exception))

    def test_rejects_bad_syntax(self) -> None:
        build_dir = self.track(make_build_dir({"widget.py": CLEAN_MODULE}))
        insert_build(self.db, "b-badsyn", build_dir,
                     syntax="widget.py: SyntaxError: bad")
        with self.assertRaises(ToolError) as cm:
            promote_build(self.db, self.ctx, "b-badsyn", "/tmp/repo")
        self.assertIn("syntax gate", str(cm.exception))

    def test_rejects_missing_dir(self) -> None:
        build_dir = self.track(make_build_dir({"widget.py": CLEAN_MODULE}))
        insert_build(self.db, "b-gone", build_dir)
        shutil.rmtree(build_dir)
        with self.assertRaises(ToolError) as cm:
            promote_build(self.db, self.ctx, "b-gone", "/tmp/repo")
        self.assertIn("missing on disk", str(cm.exception))

    def test_rejects_unknown_and_empty_id(self) -> None:
        with self.assertRaises(ToolError):
            promote_build(self.db, self.ctx, "nope", "/tmp/repo")
        with self.assertRaises(ToolError):
            promote_build(self.db, self.ctx, "  ", "/tmp/repo")

    def test_rejects_non_directory_build_dir(self) -> None:
        build_dir = self.track(make_build_dir({"ok.py": CLEAN_MODULE}))
        # a build row whose dir points at a file, not a directory
        insert_build(self.db, "b-file", build_dir / "ok.py")
        with self.assertRaises(ToolError) as cm:
            promote_build(self.db, self.ctx, "b-file", "/tmp/repo")
        self.assertIn("missing on disk", str(cm.exception))


# ── verify_proposal_files ─────────────────────────────────────────────────

class VerifyGateTests(unittest.TestCase):
    def test_clean_code_passes(self) -> None:
        ok, report = verify_proposal_files(
            [{"path": "arena_builds/w/widget.py", "old": "",
              "new": CLEAN_MODULE}], "/tmp/repo")
        self.assertTrue(ok, report)
        self.assertIn("passed", report)

    def test_syntax_error_fails(self) -> None:
        ok, report = verify_proposal_files(
            [{"path": "arena_builds/w/broken.py", "old": "",
              "new": SYNTAX_BROKEN}], "/tmp/repo")
        self.assertFalse(ok)
        self.assertIn("syntax error", report)

    def test_swallowed_exception_fails(self) -> None:
        ok, report = verify_proposal_files(
            [{"path": "arena_builds/w/swallow.py", "old": "",
              "new": SWALLOWED}], "/tmp/repo")
        self.assertFalse(ok, report)
        # bare except → E101 (error); the gate fails on error severity
        self.assertIn("/error]", report)

    def test_path_escape_fails(self) -> None:
        ok, report = verify_proposal_files(
            [{"path": "../evil.py", "old": "", "new": CLEAN_MODULE}],
            "/tmp/repo")
        self.assertFalse(ok)
        self.assertIn("escapes", report)

    def test_replacement_edits_are_not_gated(self) -> None:
        # fragments of existing files are not whole modules — neither the
        # compile gate nor error_scan applies to them.
        ok, report = verify_proposal_files(
            [{"path": "nomorals/x.py", "old": "X = 1", "new": "X = 2"}],
            "/tmp/repo")
        self.assertTrue(ok, report)

    def test_non_python_files_pass_through(self) -> None:
        ok, report = verify_proposal_files(
            [{"path": "arena_builds/w/README.md", "old": "",
              "new": "# widget\n\nsome docs\n"}], "/tmp/repo")
        self.assertTrue(ok, report)

    def test_empty_edits_pass(self) -> None:
        ok, _ = verify_proposal_files([], "/tmp/repo")
        self.assertTrue(ok)


# ── deny_ship / ship_queue ────────────────────────────────────────────────

class DenyTests(ShipTestBase):
    def test_deny_records_reason(self) -> None:
        agent = EvolutionAgent(self.ctx)
        prop = EvolutionProposal(id="evo-arena-test-1",
                                 instruction="[arena] x (topic: y)",
                                 edits=[])
        agent._save(prop)
        self.assertTrue(deny_ship(self.db, self.ctx, "evo-arena-test-1",
                                 "not good enough"))
        loaded = agent._load("evo-arena-test-1")
        assert loaded is not None
        self.assertEqual(loaded.status, "rejected")
        self.assertIn("not good enough", loaded.verify_result)

    def test_deny_missing_returns_false(self) -> None:
        self.assertFalse(deny_ship(self.db, self.ctx, "evo-nope", "r"))
        self.assertFalse(deny_ship(self.db, self.ctx, "", "r"))


class ShipQueueTests(ShipTestBase):
    def test_lists_only_arena_proposals(self) -> None:
        build_dir = self.track(make_build_dir({"widget.py": CLEAN_MODULE}))
        insert_build(self.db, "bq1", build_dir)
        arena_pid = promote_build(self.db, self.ctx, "bq1", "/tmp/repo")

        agent = EvolutionAgent(self.ctx)
        # a non-arena proposal must not appear
        agent._save(EvolutionProposal(id="evo-999",
                                      instruction="fix the thing",
                                      edits=[]))
        # a rejected arena proposal must not appear
        agent._save(EvolutionProposal(id="evo-arena-old-1",
                                      instruction="[arena] old (topic: z)",
                                      edits=[], status="rejected"))

        queue = ship_queue(self.db, self.ctx)
        ids = [q["id"] for q in queue]
        self.assertEqual(ids, [arena_pid])
        entry = queue[0]
        self.assertEqual(entry["status"], "planned")
        self.assertIn("[arena]", entry["instruction"])
        self.assertEqual(entry["edits"], ["arena_builds/widget/widget.py"])
        self.assertGreater(entry["created_at"], 0)

    def test_empty_queue(self) -> None:
        self.assertEqual(ship_queue(self.db, self.ctx), [])


# ── approve_ship (real git repo) ──────────────────────────────────────────

class ApproveShipTests(ShipTestBase):
    def _repo(self) -> Path:
        repo = self.track(Path(tempfile.mkdtemp(prefix="ship-repo-")))
        git_init(repo)
        return repo

    def _promote(self, build_id: str = "as1",
                 files: dict[str, str] | None = None) -> str:
        build_dir = self.track(make_build_dir(
            files or {"widget.py": CLEAN_MODULE}))
        insert_build(self.db, build_id, build_dir)
        return promote_build(self.db, self.ctx, build_id, "/tmp/repo")

    def test_apply_writes_files_and_commits(self) -> None:
        repo = self._repo()
        pid = self._promote()
        result = approve_ship(self.db, self.ctx, pid, repo,
                              commit=True, full_suite=False)
        self.assertTrue(result.get("applied"), result)
        self.assertEqual(result.get("status"), "applied")
        landed = repo / "arena_builds" / "widget" / "widget.py"
        self.assertTrue(landed.is_file())
        self.assertEqual(landed.read_text(encoding="utf-8"), CLEAN_MODULE)
        # nothing landed under nomorals/
        self.assertFalse((repo / "nomorals").exists())
        # commit exists
        log = git_log(repo)
        self.assertIn("evolve:", log)
        self.assertTrue(result.get("commit"))
        # and the proposal row knows it
        loaded = EvolutionAgent(self.ctx)._load(pid)
        assert loaded is not None
        self.assertEqual(loaded.status, "applied")
        self.assertEqual(loaded.commit, result["commit"])

    def test_apply_rejects_bad_proposal_upfront(self) -> None:
        repo = self._repo()
        build_dir = self.track(make_build_dir({"bad.py": SWALLOWED}))
        insert_build(self.db, "as-bad", build_dir)
        pid = promote_build(self.db, self.ctx, "as-bad", "/tmp/repo")
        result = approve_ship(self.db, self.ctx, pid, repo,
                              commit=True, full_suite=False)
        self.assertFalse(result.get("applied"))
        self.assertEqual(result.get("status"), "rejected")
        self.assertIn("ship gate failed", result.get("reason", ""))
        # nothing written to the repo
        self.assertFalse((repo / "arena_builds").exists())
        loaded = EvolutionAgent(self.ctx)._load(pid)
        assert loaded is not None
        self.assertEqual(loaded.status, "rejected")

    def test_apply_missing_proposal_raises(self) -> None:
        repo = self._repo()
        with self.assertRaises(ToolError):
            approve_ship(self.db, self.ctx, "evo-nope", repo)

    def test_dirty_tree_surfaces_toolerror(self) -> None:
        repo = self._repo()
        pid = self._promote("as-clean")
        first = approve_ship(self.db, self.ctx, pid, repo,
                             commit=True, full_suite=False)
        self.assertTrue(first.get("applied"))
        # dirty the tree with an uncommitted file
        (repo / "dirty.txt").write_text("uncommitted", encoding="utf-8")
        pid2 = self._promote("as-clean2")
        with self.assertRaises(ToolError) as cm:
            approve_ship(self.db, self.ctx, pid2, repo,
                         commit=True, full_suite=False)
        self.assertIn("working tree is dirty", str(cm.exception))


# ── evolution new-file edit support ───────────────────────────────────────

class EvolutionNewFileTests(ShipTestBase):
    def _repo(self) -> Path:
        repo = self.track(Path(tempfile.mkdtemp(prefix="evo-repo-")))
        git_init(repo)
        return repo

    def test_new_file_edit_writes_file(self) -> None:
        repo = self._repo()
        agent = EvolutionAgent(self.ctx, repo_root=repo)
        prop = EvolutionProposal(
            id="evo-newfile-1", instruction="[arena] new file test",
            edits=[{"path": "arena_builds/w/mod.py", "old": "",
                    "new": "X = 1\n"}])
        agent._save(prop)
        out = agent.apply("evo-newfile-1", verify=False, commit=False)
        self.assertTrue(out["applied"])
        landed = repo / "arena_builds" / "w" / "mod.py"
        self.assertTrue(landed.is_file())
        self.assertEqual(landed.read_text(encoding="utf-8"), "X = 1\n")
        self.assertEqual(agent._load("evo-newfile-1").status, "applied")

    def test_missing_target_with_old_rejects(self) -> None:
        repo = self._repo()
        agent = EvolutionAgent(self.ctx, repo_root=repo)
        prop = EvolutionProposal(
            id="evo-missing-1", instruction="bad plan",
            edits=[{"path": "nope.py", "old": "x = 1", "new": "x = 2"}])
        agent._save(prop)
        with self.assertRaises(ToolError) as cm:
            agent.apply("evo-missing-1", verify=False, commit=False)
        self.assertIn("edit target missing: nope.py", str(cm.exception))
        loaded = agent._load("evo-missing-1")
        assert loaded is not None
        self.assertEqual(loaded.status, "rejected")
        self.assertEqual(loaded.verify_result, "edit target missing: nope.py")

    def test_replacement_edit_still_works(self) -> None:
        repo = self._repo()
        (repo / "a.py").write_text("X = 1\n", encoding="utf-8")
        # the file must be committed first — apply needs a clean tree
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo,
                       check=True, capture_output=True)
        agent = EvolutionAgent(self.ctx, repo_root=repo)
        prop = EvolutionProposal(
            id="evo-replace-1", instruction="replace",
            edits=[{"path": "a.py", "old": "X = 1", "new": "X = 2"}])
        agent._save(prop)
        out = agent.apply("evo-replace-1", verify=False, commit=False)
        self.assertTrue(out["applied"])
        self.assertEqual((repo / "a.py").read_text(encoding="utf-8"),
                         "X = 2\n")


# ── scores/sample integration contracts ───────────────────────────────────
# The /arena scores and /arena sample verbs are thin wrappers over the
# arena scoring/sampling modules; these tests pin the contract the verbs
# rely on (return shapes, db=None dry-run behavior).

class ScoreSampleContractTests(ShipTestBase):
    def test_category_scores_reads_migration_61_table(self) -> None:
        from nomorals.agents.arena.scoring import (
            category_scores,
            record_score,
        )

        self.assertEqual(category_scores(self.db), {})
        sid = record_score(
            self.db, topic="t1", category="AI",
            research_usefulness=0.8, build_compiled=True,
            tests_passed=True, edit_precision=0.9, latency_s=12.5)
        self.assertTrue(sid)
        record_score(self.db, topic="t2", category="ai",
                     research_usefulness=0.4, latency_s=7.5)
        table = category_scores(self.db)
        self.assertEqual(set(table), {"ai"})
        slot = table["ai"]
        self.assertEqual(slot["runs"], 2)
        self.assertIsNotNone(slot["avg"])
        self.assertGreater(slot["avg"], 0.0)
        self.assertAlmostEqual(slot["avg_latency"], 10.0)

    def test_sample_challenge_dry_run(self) -> None:
        from nomorals.agents.arena.sampling import sample_challenge

        cat, topic, entry = sample_challenge(
            db=None, profile={"ai": 3.0}, anti_repeat=0)
        self.assertTrue(cat)
        self.assertTrue(topic)
        self.assertIsInstance(entry, dict)
        for key in ("d", "kind", "verify"):
            self.assertIn(key, entry)


if __name__ == "__main__":
    unittest.main()
