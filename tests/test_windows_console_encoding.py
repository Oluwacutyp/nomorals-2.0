"""Windows console encoding: CLI must never die on a cp1252 console.

Regression (owner's EC2 screenshot): ``python -m nomorals --help`` died with::

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

because help text contained the rightwards arrow U+2192 and Python on
Windows defaults console stdout to cp1252. The deploy script then printed
``OK: python -m nomorals --help works`` DESPITE the traceback (exit 1),
because verify checked output, not the exit code. A second failure in the
same run: ``nm doctor`` writes a benign INFO line to stderr, and the
deploy script's ``2>&1 | Select-Object`` turned that stderr INFO into a
PowerShell NativeCommandError.

Fix, three layers (all asserted here):
1. Python: ``_ensure_console_utf8()`` reconfigures stdout/stderr to
   UTF-8 with errors='replace' before argparse prints anything, and the
   argparse help strings themselves are ASCII (no U+2192 in help).
2. Deploy script: forces UTF-8 (``$env:PYTHONIOENCODING``,
   ``[Console]::OutputEncoding``, ``chcp 65001``) before running Python.
3. Deploy verify: checks the native exit code after every verify command
   (output is not success), and runs native exes via Start-Process
   (Invoke-Native) instead of PowerShell's & operator. In PS 5.1 ANY
   stderr line from a native exe becomes a NativeCommandError even with
   `> file 2> file` redirection, so & + redirection cannot be used here.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY = REPO_ROOT / "scripts" / "deploy_devon_ec2_rdp.ps1"


def _all_help_text() -> str:
    from nomorals.cmdline.parser import _parser

    parser = _parser()
    texts = [parser.format_help()]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                texts.append(sub.format_help())
    return "\n".join(texts)


def test_cli_help_has_no_rightwards_arrow() -> None:
    """The exact char that crashed --help on EC2 must not be in help text."""
    assert "\u2192" not in _all_help_text()


def test_cli_help_encodes_in_cp1252() -> None:
    """Full --help output (top + every subparser) must encode in cp1252."""
    _all_help_text().encode("cp1252")


def test_cp1252_stdout_simulation() -> None:
    """Writing help to a simulated cp1252 console must not raise."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    stream.write(_all_help_text())
    stream.flush()
    assert buf.getvalue()


def test_ensure_console_utf8_reconfigures(monkeypatch: pytest.MonkeyPatch) -> None:
    from nomorals.cmdline.dispatch import _ensure_console_utf8

    calls: list[tuple[str, str]] = []

    class _FakeStream:
        def reconfigure(self, encoding: str, errors: str) -> None:
            calls.append((encoding, errors))

    fake = _FakeStream()
    monkeypatch.setattr(sys, "stdout", fake)
    monkeypatch.setattr(sys, "stderr", fake)
    _ensure_console_utf8()
    assert ("utf-8", "replace") in calls
    assert len(calls) == 2, "both stdout and stderr must be reconfigured"


def test_ensure_console_utf8_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A console that cannot be reconfigured must never break the CLI."""
    from nomorals.cmdline.dispatch import _ensure_console_utf8

    class _BrokenStream:
        def reconfigure(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("no reconfigure here")

    monkeypatch.setattr(sys, "stdout", _BrokenStream())
    monkeypatch.setattr(sys, "stderr", object())
    _ensure_console_utf8()  # must not raise


def test_deploy_script_sets_utf8_encoding() -> None:
    src = DEPLOY.read_text(encoding="utf-8")
    assert 'PYTHONIOENCODING' in src
    assert 'OutputEncoding' in src
    assert 'chcp 65001' in src


def test_deploy_verify_checks_exit_code() -> None:
    """Verify must fail on nonzero exit - output is not success."""
    src = DEPLOY.read_text(encoding="utf-8")
    # Exit code comes from Start-Process (.ExitCode via Invoke-Native),
    # not $LASTEXITCODE: the & operator is banned here (NativeCommandError).
    assert "Invoke-Native" in src
    assert "ExitCode" in src
    assert 'VERIFY FAILED' in src
    # --help verify must throw when the exit code is nonzero, not print OK.
    assert "--help exited" in src


def test_deploy_verify_uses_start_process_not_ampersand() -> None:
    """No bare `& $venvPy ...` native call may remain in the deploy script.

    In PS 5.1 ANY stderr from a native exe under & becomes a
    NativeCommandError, even with `> file 2> file` redirection (owner's
    EC2 screenshot: `nm doctor` died at line 246 exactly that way).
    All native verify calls must go through Invoke-Native (Start-Process),
    which redirects at the process level and never emits the error record.
    Comment lines may discuss the banned pattern; only code lines count.
    """
    src = DEPLOY.read_text(encoding="utf-8")
    code_lines = [
        line for line in src.splitlines() if not line.strip().startswith("#")
    ]
    bad_amp = [line for line in code_lines if "& $venvPy" in line]
    assert not bad_amp, "bare & $venvPy lines in deploy script:\n" + "\n".join(bad_amp)
    bad_redirect = [
        line
        for line in code_lines
        if "2>" in line and ("$venvPy" in line or "doctor" in line.lower())
    ]
    assert not bad_redirect, "native redirect lines in deploy script:\n" + "\n".join(bad_redirect)
    assert "Start-Process" in src
    assert "RedirectStandardError" in src, "stderr must still be captured to a file, never silenced"


def test_deploy_verify_never_pipes_stderr_into_select_object() -> None:
    """No code line may pipe 2>&1 into Select-Object (NativeCommandError).

    A comment may *mention* the pattern (it documents why it is banned);
    only executable lines are checked.
    """
    src = DEPLOY.read_text(encoding="utf-8")
    bad = [
        line
        for line in src.splitlines()
        if "2>&1" in line and not line.strip().startswith("#")
    ]
    assert not bad, "executable 2>&1 lines in deploy script:\n" + "\n".join(bad)


def test_installer_verification_uses_start_process() -> None:
    """Installer verification must use Start-Process, not & + 2>&1.

    Same PS 5.1 NativeCommandError trap as the deploy script: the
    installer's Invoke-Verification ran `--help` and `doctor` via
    `& $venvPy ... 2>&1 | Select-Object`, so a benign stderr INFO
    became an error record. Preflight `python --version` capture is
    covered by the same rule; winget/nvidia-smi `2>$null` (stderr
    discarded, try/catch, worked) and pip (live output, no redirect)
    are intentionally left alone.
    """
    installer = (REPO_ROOT / "scripts" / "install_devon_windows.ps1").read_text(
        encoding="utf-8"
    )
    code_lines = [
        line
        for line in installer.splitlines()
        if not line.strip().startswith("#")
    ]
    bad = [
        line
        for line in code_lines
        if "2>&1" in line and ("$venvPy" in line or "python" in line)
    ]
    assert not bad, "executable 2>&1 native lines in installer:\n" + "\n".join(bad)
    assert "Start-Process" in installer
    assert "Invoke-Native" in installer
