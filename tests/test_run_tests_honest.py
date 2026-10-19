"""Failing-first tests: run_tests must be honest about long suites.

Devon self-reported "pytest never finished, hit the 120-second timeout, so
the tests are either hanging or taking too long". The diagnosis was wrong:
the full suite is ~815 files / ~85 min, so a 120s cap GUARANTEES a timeout
that looks like a hang. These tests pin the honest behavior:

* a timeout says the suite was STILL RUNNING (not that it hung);
* the full suite runs in chunks (chunk="i/n" runs one slice);
* with no paths, run_tests targets the changed files, not the full suite;
* a full-suite run names the chunking escape hatch instead of dying silently.
"""

from __future__ import annotations

import subprocess
import textwrap
import unittest
from pathlib import Path

from nomorals.tools import pytest_runner as pr


def _write_test(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                   check=True, timeout=30)


class TimeoutHonestyTests(unittest.TestCase):
    def test_timeout_note_says_still_running_not_hung(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_test(root / "tests" / "test_slow.py", """
                import time
                import unittest

                class SlowTests(unittest.TestCase):
                    def test_sleeps(self) -> None:
                        time.sleep(30)
            """)
            res = pr._run_pytest(root, ["tests"], timeout=0.5)
            self.assertFalse(res["ok"])
            note = str(res.get("note") or "")
            self.assertIn("still running", note.lower())
            self.assertNotIn("timed out after", note.lower())

    def test_chunk_runs_one_slice(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("test_a", "test_b", "test_c", "test_d"):
                _write_test(root / "tests" / f"{name}.py", f"""
                    import unittest

                    class {name.title().replace('_', '')}Tests(unittest.TestCase):
                        def test_passes(self) -> None:
                            self.assertTrue(True)
                """)
            first = pr.run_tests(paths=["tests"], repo=str(root), chunk="1/2")
            second = pr.run_tests(paths=["tests"], repo=str(root), chunk="2/2")
            self.assertTrue(first["ok"], first)
            self.assertTrue(second["ok"], second)
            self.assertEqual(first.get("chunk"), "1/2")
            self.assertEqual(second.get("chunk"), "2/2")
            seen = set(first.get("selected") or []) | set(second.get("selected") or [])
            self.assertEqual(len(seen), 4, seen)
            self.assertFalse(set(first.get("selected") or [])
                             & set(second.get("selected") or []))

    def test_full_suite_note_names_chunking(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_test(root / "tests" / "test_a.py", """
                import unittest

                class ATests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            res = pr.run_tests(paths=["tests"], repo=str(root))
            note = str(res.get("note") or "")
            self.assertIn("chunk", note.lower())

    def test_string_paths_accepted(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_test(root / "tests" / "test_a.py", """
                import unittest

                class ATests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            # cmdline/commands/repo.py calls run_tests(root, ...) with a
            # bare string: previously list(str) split it into characters.
            res = pr.run_tests(paths=str(root / "tests"), repo=str(root))  # type: ignore[arg-type]
            self.assertTrue(res["ok"], res)

    def test_default_is_targeted_not_full_suite(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git("init", cwd=root)
            _git("config", "user.email", "t@t", cwd=root)
            _git("config", "user.name", "t", cwd=root)
            (root / "foo.py").write_text("X = 1\n", encoding="utf-8")
            _write_test(root / "tests" / "test_foo.py", """
                import unittest

                class FooTests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            _write_test(root / "tests" / "test_other.py", """
                import unittest

                class OtherTests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            _git("add", "-A", cwd=root)
            _git("commit", "-m", "init", cwd=root)
            (root / "foo.py").write_text("X = 2\n", encoding="utf-8")
            res = pr.run_tests(repo=str(root))
            self.assertTrue(res["ok"], res)
            selected = list(res.get("selected") or [])
            self.assertEqual(selected, ["tests/test_foo.py"], selected)

    def test_default_no_changes_is_honest_skip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git("init", cwd=root)
            _git("config", "user.email", "t@t", cwd=root)
            _git("config", "user.name", "t", cwd=root)
            _write_test(root / "tests" / "test_a.py", """
                import unittest

                class ATests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            _git("add", "-A", cwd=root)
            _git("commit", "-m", "init", cwd=root)
            res = pr.run_tests(repo=str(root))
            self.assertTrue(res["ok"], res)
            self.assertEqual(res.get("passed"), 0)
            self.assertIn("nothing ran", str(res.get("note") or "").lower())

    def test_registry_wrapper_passes_timeout_and_chunk(self) -> None:
        captured: dict[str, object] = {}

        class FakeRegistry:
            def register(self, name: str, **_kw: object):  # noqa: ANN001, ANN202
                def deco(fn):  # noqa: ANN001, ANN202
                    captured[name] = fn
                    return fn

                return deco

        pr.register(FakeRegistry())  # type: ignore[arg-type]
        fn = captured["run_tests"]
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_test(root / "tests" / "test_a.py", """
                import unittest

                class ATests(unittest.TestCase):
                    def test_passes(self) -> None:
                        self.assertTrue(True)
            """)
            res = fn(paths=["tests"], repo=str(root), timeout=60, chunk="1/1")
            self.assertTrue(res["ok"], res)
            self.assertEqual(res.get("chunk"), "1/1")


if __name__ == "__main__":
    unittest.main()
