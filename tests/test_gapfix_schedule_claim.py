"""R3-60 guard: the schedule claim book stays a seam component.

Audit §60 proposed wiring ``ScheduleClaimStore`` into the live
scheduler or deleting it. Both are wrong — see
``~/workspace/gapfix60-verify/GAPFIX60_SURVEY.md`` for the full
evidence. In short: the store is the Mongo P4 seam port of the claim
book, proven on both rungs by ``tests/test_mongo_p4.py``; the live
``Scheduler`` claims in-process by design; wiring it in would fork
the ``schedule_jobs`` record between two stores while seven native
readers/writers keep writing SQL; deleting it would break the P4
suite and destroy the primitive the wholesale scheduler domain port
needs. This file pins the disposition so no future lane re-proposes
either move.

Failing-first for this lane: the disposition-text test below fails on
the unmodified tree (the recorded seven-module list carries two stale
paths — ``agents/autonomy/presence.py`` and
``agents/scheduler/facade.py`` do not exist). The no-fork test is
proven separately by injection: it fails when the forbidden import is
added to ``agents/scheduler.py`` and passes once reverted.
"""

import ast
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _imported_modules(path: Path) -> set[str]:
    """Every module an AST import in *path* names (absolute + relative)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            for alias in node.names:
                names.add(alias.name)
    return names


class NoForkTests(unittest.TestCase):
    def test_live_scheduler_never_imports_the_seam_claim_store(self):
        # The no-fork rule, from the module docstring and
        # carry.PORT_RESIDUE: the live Scheduler keeps its native SQL
        # path in P4; a claim-only seam wiring would fork the record
        # between two stores. If this fails, someone wired it.
        names = _imported_modules(_REPO / "nomorals/agents/scheduler.py")
        for name in names:
            self.assertNotEqual(
                name.rsplit(".", 1)[-1], "schedule_claim",
                f"agents/scheduler.py imports the seam claim store: {name}")
        self.assertNotIn("ScheduleClaimStore", names)

    def test_claim_book_disposition_is_recorded(self):
        # The disposition itself is the verdict: ported to the seam,
        # live path stays native, fork reason on record, wholesale
        # port a later pass.
        from nomorals.storage.carry import PORT_RESIDUE

        jobs_text = PORT_RESIDUE["schedule_jobs"]
        lowered = jobs_text.lower()
        self.assertIn("claim", lowered)
        self.assertIn("storage/schedule_claim.py", jobs_text)
        self.assertIn("native", lowered)
        self.assertIn("fork", lowered)
        # schedule_runs carries the same disposition by reference, not
        # by repeating the text.
        runs_text = PORT_RESIDUE["schedule_runs"]
        self.assertIn("schedule_jobs", runs_text)
        self.assertIn("claim", runs_text.lower())
        # The seven native readers/writers the disposition names must
        # be real paths — two of them were stale (agents/ prefix) and
        # are corrected in this lane.
        for rel in ("agents/job_reports.py", "agents/motivation.py",
                    "agents/outcome_steward.py", "agents/skill_evolution.py",
                    "autonomy/presence.py", "cmdline/commands/schedule.py",
                    "scheduler/facade.py"):
            self.assertIn(rel, jobs_text, rel)
            self.assertTrue((_REPO / "nomorals" / rel).is_file(), rel)
        for stale in ("agents/autonomy/presence.py",
                      "agents/scheduler/facade.py"):
            self.assertNotIn(stale, jobs_text, stale)

    def test_claim_store_is_importable_with_its_claim_api(self):
        # The store is live seam infrastructure, not dead code: the
        # export exists and the claim API is intact. Semantics are
        # owned by the P4 suite (tests/test_mongo_p4.py), not duplicated
        # here.
        from nomorals.storage import ScheduleClaimStore

        for method in ("claim", "complete", "release", "runs_for",
                       "due_jobs", "get_job", "put_job"):
            self.assertTrue(callable(getattr(ScheduleClaimStore, method)),
                            method)
        self.assertTrue(callable(ScheduleClaimStore.attached))


if __name__ == "__main__":
    unittest.main()
