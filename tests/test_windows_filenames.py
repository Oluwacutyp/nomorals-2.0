"""Windows-checkout guard: no tracked filename may be illegal on Windows.

Regression: the repo once contained a tracked file named
``:memory:.migrate.lock`` — created by the migration lock when a
``Database`` was opened on ``Path(":memory:")`` (``Path != str``, so the
in-memory check missed it). ``:`` is illegal in Windows filenames, so
``git clone`` on Windows failed the ENTIRE checkout with
``error: invalid path ':memory:.migrate.lock'`` / ``fatal: unable to
checkout working tree`` — leaving ``scripts/`` and everything else
missing. It was never user error; it was this file.

This test fails if ANY tracked path contains a Windows-illegal name, so
the breakage can never come back.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Characters illegal in a Windows filename component. "/" is the path
# separator and is checked per-component, so it is not in this set.
ILLEGAL_CHARS = set('<>:"|?*')
# Bare names Windows reserves regardless of extension (COM1..COM9, LPT1..LPT9).
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def illegal_components(rel_path: str) -> list[str]:
    """Return the offending components of a repo-relative path, if any."""
    bad = []
    for comp in rel_path.replace("\\", "/").split("/"):
        if not comp or comp in (".", ".."):
            continue
        if any(ch in comp for ch in ILLEGAL_CHARS):
            bad.append(comp)
        elif comp != comp.rstrip(" ."):
            bad.append(comp)  # trailing space / dot: illegal on Windows
        elif comp.split(".")[0].upper() in RESERVED:
            bad.append(comp)
    return bad


def tracked_files() -> list[str]:
    """Tracked paths via git; falls back to a tree walk if git is absent."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        found = []
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in filenames:
                found.append(str(Path(dirpath, name).relative_to(REPO_ROOT)))
        return found


def test_no_windows_illegal_tracked_filenames() -> None:
    bad: dict[str, list[str]] = {}
    for rel in tracked_files():
        offenders = illegal_components(rel)
        if offenders:
            bad[rel] = offenders
    assert not bad, (
        "Windows-illegal filenames in the repo (a Windows clone cannot "
        f"check out these files at all): {bad}"
    )


def test_no_memory_migrate_lock_file() -> None:
    assert ":memory:.migrate.lock" not in tracked_files()
    assert not (REPO_ROOT / ":memory:.migrate.lock").exists()


def test_memory_database_never_creates_lock_file(tmp_path, monkeypatch) -> None:
    """Database(Path(':memory:')) is in-memory and migrate() creates no lock file.

    Regression for the exact chain that produced the illegal file:
    open_database() wraps its argument in Path() first, so a ":memory:"
    database arrived at Database.__init__ as Path(":memory:"), and
    Path(":memory:") == ":memory:" is False — the old check missed it.
    """
    from nomorals.storage.db import Database, open_database

    monkeypatch.chdir(tmp_path)

    db = Database(Path(":memory:"))
    assert db.path is None
    assert db.is_memory
    db.migrate()
    db.close()
    assert list(tmp_path.iterdir()) == []

    db2, _recovered, _backup = open_database(":memory:")
    assert db2.path is None
    assert db2.is_memory
    db2.close()
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / ":memory:.migrate.lock").exists()


def test_migration_lock_skips_in_memory_and_unsafe_names(tmp_path, monkeypatch) -> None:
    """_migration_lock yields without touching the filesystem for :memory:."""
    from nomorals.storage.schema import MigrationRunner

    monkeypatch.chdir(tmp_path)

    class FakeDb:
        def __init__(self, path):
            self.path = path

        def execute(self, *args, **kwargs):
            return None

        def query(self, *args, **kwargs):
            return []

    for in_memory_path in (":memory:", Path(":memory:")):
        runner = MigrationRunner.__new__(MigrationRunner)
        runner.db = FakeDb(in_memory_path)
        with runner._migration_lock():
            pass
        assert list(tmp_path.iterdir()) == []

    # A path whose derived lock filename would contain ':' is also skipped.
    runner = MigrationRunner.__new__(MigrationRunner)
    runner.db = FakeDb(Path("we:ird.db"))
    with runner._migration_lock():
        pass
    assert list(tmp_path.iterdir()) == []
