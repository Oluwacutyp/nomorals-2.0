"""Windows import safety for the console package (termios crash).

Regression: ``scripts/run_chat_bot.py`` died on Windows with
``ModuleNotFoundError: No module named 'termios'`` because
``nomorals/console/godconsole.py`` imported the Unix-only ``termios`` /
``tty`` modules unconditionally, and ``nomorals/console/__init__.py``
imported godconsole eagerly. The chat bot never opens the fullscreen UI,
so the console package must import cleanly with those modules absent.

Import simulation runs in a CLEAN SUBPROCESS (a meta-path finder that
blocks ``termios`` / ``tty``), because evicting modules from
``sys.modules`` in-process leaves the parent package's submodule
attribute stale and makes two import statements resolve to different
module objects.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GODCONSOLE_SRC = REPO_ROOT / "nomorals" / "console" / "godconsole.py"
RUN_CHAT_BOT = REPO_ROOT / "scripts" / "run_chat_bot.py"

_BLOCKER_PREAMBLE = """
import sys

class _BlockUnixTerminalModules:
    \"\"\"Pretend termios/tty do not exist, as on Windows.\"\"\"
    def find_spec(self, name, path=None, target=None):
        if name in ("termios", "tty"):
            raise ImportError(f"blocked {name} (simulated Windows)")
        return None

sys.meta_path.insert(0, _BlockUnixTerminalModules())
"""


def _run_blocked(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER_PREAMBLE + code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_godconsole_imports_without_termios():
    proc = _run_blocked(
        "import nomorals.console.godconsole as gc\n"
        "assert gc.termios is None\n"
        "assert gc.tty is None\n"
        "assert hasattr(gc, 'GodConsole')\n"
        "assert hasattr(gc, 'InputLine')\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_console_package_imports_without_touching_godconsole():
    proc = _run_blocked(
        "import sys\n"
        "import nomorals.console as console\n"
        "assert hasattr(console, 'ConsoleCommands')\n"
        "assert 'nomorals.console.godconsole' not in sys.modules, "
        "'godconsole must stay lazy when only ConsoleCommands is needed'\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_lazy_godconsole_still_resolves_without_termios():
    proc = _run_blocked(
        "from nomorals.console import EventBuffer, GodConsole, InputLine\n"
        "assert GodConsole is not None\n"
        "assert InputLine is not None\n"
        "assert EventBuffer is not None\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_run_chat_bot_module_imports_without_termios():
    proc = _run_blocked(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location("
        f"'run_chat_bot', r'{RUN_CHAT_BOT}')\n"
        "assert spec is not None and spec.loader is not None\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)  # must not raise ImportError\n"
        "assert hasattr(module, 'main')\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_no_unguarded_unix_imports_in_godconsole():
    src = GODCONSOLE_SRC.read_text(encoding="utf-8")
    for mod in ("termios", "tty"):
        # No bare top-level ``import <mod>`` — it must live in a try block.
        assert not re.search(rf"(?m)^import {mod}\s*$", src), (
            f"unguarded `import {mod}` in godconsole.py — "
            "it must be try/except ImportError with a None fallback"
        )
        assert f"{mod} = None" in src, f"missing `{mod} = None` fallback"
    assert "except ImportError" in src


def _godconsole_module():
    # Single lookup: importing the submodule two different ways in one
    # test can resolve to different module objects when a parent
    # package defines __getattr__ (PEP 562) and sys.modules was ever
    # manipulated. sys.modules is the one true identity.
    import importlib

    importlib.import_module("nomorals.console.godconsole")
    return sys.modules["nomorals.console.godconsole"]


def test_wait_key_windows_path_uses_msvcrt():
    gc = _godconsole_module()
    console = gc.GodConsole.__new__(gc.GodConsole)
    calls = {"kbhit": 0}

    class _FakeMsvcrt:
        def kbhit(self):
            calls["kbhit"] += 1
            return True

        def getwch(self):
            return "\r"

    real_msvcrt = gc.msvcrt
    try:
        gc.msvcrt = _FakeMsvcrt()
        assert console._wait_key(1.0) == "\n"
        assert calls["kbhit"] >= 1
    finally:
        gc.msvcrt = real_msvcrt


def test_wait_key_windows_timeout_returns_none():
    gc = _godconsole_module()
    console = gc.GodConsole.__new__(gc.GodConsole)

    class _FakeMsvcrt:
        def kbhit(self):
            return False

        def getwch(self):  # pragma: no cover - never called
            raise AssertionError("getwch must not be called without kbhit")

    real_msvcrt = gc.msvcrt
    try:
        gc.msvcrt = _FakeMsvcrt()
        assert console._wait_key(0.05) is None
    finally:
        gc.msvcrt = real_msvcrt
