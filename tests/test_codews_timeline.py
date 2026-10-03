"""Timeline wiring for the codews organ (Wave K).

Attaches a :class:`~nomorals.os.timeline.Timeline` to the process bus and
asserts workspace-open, patch-apply, test-run, and build-run events land
in it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.codews import (
    CodeWorkspace,
    apply_patch,
    run_build,
    run_tests,
)
from nomorals.core.events import global_bus
from nomorals.os.timeline import Timeline

GIT = shutil.which("git")

MODIFY_DIFF = """\
--- a/hello.txt
+++ b/hello.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
"""


def _git(*args, cwd):
    proc = subprocess.run([GIT, *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc


class CodewsTimelineTestCase(unittest.TestCase):
    def setUp(self):
        if not GIT:
            self.skipTest("git binary not found on PATH")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = Path(self.tmp) / "repo"
        self.repo.mkdir()
        _git("init", "-q", cwd=self.repo)
        _git("config", "user.email", "test@example.com", cwd=self.repo)
        _git("config", "user.name", "Test", cwd=self.repo)
        (self.repo / "hello.txt").write_text("line1\nline2\nline3\n")
        _git("add", ".", cwd=self.repo)
        _git("commit", "-q", "-m", "initial commit", cwd=self.repo)

        self.timeline = Timeline()  # in-memory
        self.addCleanup(self.timeline.close)
        self.timeline.attach(global_bus, sync=True)
        self.addCleanup(self.timeline.detach, global_bus)

    def _events(self, topic):
        return self.timeline.query(topic=topic, limit=50)

    def test_workspace_opened(self):
        ws = CodeWorkspace(self.repo, mission_id="m9")
        rows = self._events("codews.workspace.opened")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["root"], str(ws.root))
        self.assertEqual(data["mission_id"], "m9")
        self.assertEqual(rows[0]["source"], "nomorals.codews.workspace")

    def test_patch_applied_emits_only_for_real_apply(self):
        # dry run: telemetry only, no event.
        apply_patch(MODIFY_DIFF, dry_run=True, root=self.repo)
        self.assertEqual(self._events("codews.patch.applied"), [])

        results = apply_patch(MODIFY_DIFF, dry_run=False, root=self.repo)
        self.assertTrue(all(r["ok"] for r in results))
        rows = self._events("codews.patch.applied")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["root"], str(self.repo))
        self.assertEqual(data["files"], ["hello.txt"])
        self.assertEqual(data["patched"], 1)
        self.assertEqual(data["failed"], 0)

    def test_tests_run_pass_and_fail(self):
        tdir = Path(self.tmp) / "passing"
        tdir.mkdir()
        (tdir / "test_t.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        res = run_tests(tdir)
        self.assertTrue(res["ok"])
        rows = self._events("codews.tests.run")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["root"], str(tdir))
        self.assertTrue(data["ok"])
        self.assertGreaterEqual(data["passed"], 1)
        self.assertEqual(data["failed"], 0)

        fdir = Path(self.tmp) / "failing"
        fdir.mkdir()
        (fdir / "test_t.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_bad(self):\n"
            "        self.assertTrue(False)\n",
            encoding="utf-8",
        )
        res = run_tests(fdir)
        self.assertFalse(res["ok"])
        rows = self._events("codews.tests.run")
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["data"]["ok"])

    def test_build_run(self):
        bdir = Path(self.tmp) / "mk"
        bdir.mkdir()
        (bdir / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
        res = run_build(bdir)
        self.assertTrue(res["ok"])
        rows = self._events("codews.build.run")
        self.assertEqual(len(rows), 1)
        data = rows[0]["data"]
        self.assertEqual(data["root"], str(bdir))
        self.assertEqual(data["target"], "all")
        self.assertTrue(data["ok"])

    def test_no_timeline_works_fine(self):
        # Fail-open telemetry: workspace ops must work with no subscriber.
        self.timeline.detach(global_bus)
        ws = CodeWorkspace(self.repo)
        self.assertIn("branch", ws.status())


if __name__ == "__main__":
    unittest.main()
