"""Windows import safety for the sandbox tools (``resource`` crash).

Regression: on Windows EC2 the bot logged
``tool module 'sandbox_code' failed to register: ModuleNotFoundError: No
module named 'resource'`` (same for ``'shell'``) because
``nomorals/tools/shell.py`` imported the Unix-only ``resource`` module at
top level, and ``sandbox_code`` imports from ``shell``. The tools must
REGISTER on Windows even when rlimits are unavailable.

Import simulation runs in a CLEAN SUBPROCESS (a meta-path finder that
blocks ``resource``), mirroring ``test_windows_console_import.py``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_BLOCKER_PREAMBLE = """
import sys

class _BlockResource:
    \"\"\"Pretend resource does not exist, as on Windows.\"\"\"
    def find_spec(self, name, path=None, target=None):
        if name == "resource":
            raise ImportError("blocked resource (simulated Windows)")
        return None

sys.meta_path.insert(0, _BlockResource())
"""


def _run_blocked(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER_PREAMBLE + code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


_STUB = """
class _Registry:
    def __init__(self):
        self.context = None
        self.tools = {}

    def register(self, name, description="", capability=None, **_kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco
"""


def test_shell_imports_without_resource():
    proc = _run_blocked(
        "import nomorals.tools.shell as shell\n"
        "assert shell.resource is None\n"
        "shell._set(0, 1, 1)\n"  # no-op, must not raise
        "shell.SandboxLimits().preexec()\n"  # rlimits skipped, setsid only
    )
    assert proc.returncode == 0, proc.stderr


def test_shell_and_sandbox_code_register_without_resource():
    proc = _run_blocked(
        _STUB
        + "import nomorals.tools.shell as shell\n"
        + "import nomorals.tools.sandbox_code as sandbox_code\n"
        + "reg = _Registry()\n"
        + "shell.register(reg)\n"
        + "assert 'shell_run' in reg.tools\n"
        + "sandbox_code.register(reg)\n"
        + "assert 'code' in reg.tools\n"
    )
    assert proc.returncode == 0, proc.stderr
