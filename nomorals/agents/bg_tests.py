"""Phase D: background test runs.

Long suites run in a background subprocess while the agent does useful
work (lint, diff prep) instead of blocking.  Poll with backoff, surface
live progress (passed/failed/error counts streamed from the runner), and
kill the whole process tree when the cap is exceeded — the session never
hangs.  Killing reuses ``shell.py``'s ``_kill_tree``.
"""

from __future__ import annotations

import codecs
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.shell import _kill_tree

_log = get_logger(__name__)

#: Default cap for a background run: 15 minutes (spec).
DEFAULT_CAP_SECONDS = 900.0

_TAIL_CHARS = 8000

# pytest -q / unittest dot-stream progress characters.
_DOT_PASS = "."
_DOT_FAIL = "F"
_DOT_ERROR = "E"
_DOT_SKIP = "s"

# pytest -v line verdicts.
_VERBOSE_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
# unittest -v line verdicts, e.g. "test_a (mod.T.test_a) ... ok".
_UNITTEST_VERBOSE_RE = re.compile(r"\.\.\.\s+(ok|FAIL|ERROR|skipped)\s*$")
# Final summary counters, e.g. "3 passed, 1 failed in 2.34s".
_SUMMARY_RE = re.compile(r"(\d+)\s+(passed|failed|error|skipped)", re.I)


class BackgroundTestRun:
    """A test suite running in a background subprocess.

    Usage::

        run = BackgroundTestRun(argv, cwd=root, cap_seconds=900)
        lint_res = lint(changed, repo=root)          # useful work meanwhile
        final = run.wait(on_progress=log_progress)   # poll with backoff
    """

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path,
        cap_seconds: float = DEFAULT_CAP_SECONDS,
        label: str = "tests",
    ) -> None:
        self.argv = list(argv)
        self.cwd = str(cwd)
        self.cap_seconds = cap_seconds
        self.label = label
        self._t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # own process group: _kill_tree is safe
        )
        self._output: list[str] = []
        self._passed = 0
        self._failed = 0
        self._errors = 0
        self._skipped = 0
        self._timed_out = False
        self._killed = False
        self._exit_code: int | None = None
        self._done = False
        # Incremental decoder: _drain reads the fd directly (os.read never
        # blocks after select; TextIOWrapper.read(n) would block for n
        # chars), so multibyte characters split across chunks still decode
        # cleanly.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # Line reassembly: a verdict line can arrive split across chunks
        # (unittest -v prints "test_x ... " before the test and "ok" after),
        # so only complete lines are tallied.
        self._pending = ""
        _log.info("background %s started (pid %s, cap %.0fs): %s",
                  label, self._proc.pid, cap_seconds,
                  " ".join(self.argv[:6]))

    # ── progress ──────────────────────────────────────────────────────
    def _drain(self) -> None:
        """Non-blocking read of whatever the child has printed."""
        stream = self._proc.stdout
        if stream is None or self._done:
            return
        try:
            fd = stream.fileno()
            while select.select([fd], [], [], 0)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                text = self._decoder.decode(chunk)
                if text:
                    self._feed(text)
        except (OSError, ValueError):
            pass

    def _feed(self, text: str) -> None:
        """Reassemble lines across chunks; tally each complete line."""
        self._output.append(text)
        self._pending += text
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        for line in lines:
            self._tally_line(line)

    def _tally_line(self, line: str) -> None:
        m = _VERBOSE_RE.match(line.strip())
        if m:
            verdict = m.group(1)
            if verdict == "PASSED":
                self._passed += 1
            elif verdict == "FAILED":
                self._failed += 1
            elif verdict == "ERROR":
                self._errors += 1
            else:
                self._skipped += 1
            return
        m = _UNITTEST_VERBOSE_RE.search(line)
        if m:
            verdict = m.group(1)
            if verdict == "ok":
                self._passed += 1
            elif verdict == "FAIL":
                self._failed += 1
            elif verdict == "ERROR":
                self._errors += 1
            else:
                self._skipped += 1
            return
        # dot-stream (pytest -q / unittest): tally progress characters from
        # the line's leading token only.  pytest appends a "[100%]"
        # trailer to the dot line, so the whole-line check would miss it;
        # the token check keeps tracebacks (e.g. 'File "..."') and summary
        # text out of the tally.
        stripped = line.strip()
        if stripped:
            token = stripped.split()[0]
            rest = stripped[len(token):].lstrip()
            if (token and all(c in ".*FEsxX" for c in token)
                    and (not rest or rest.startswith("["))):
                self._passed += token.count(_DOT_PASS)
                self._failed += token.count(_DOT_FAIL)
                self._errors += token.count(_DOT_ERROR)
                self._skipped += token.count(_DOT_SKIP)

    def _reap(self) -> None:
        if self._done:
            return
        rc = self._proc.poll()
        if rc is None:
            return
        # process ended: drain whatever is left, then close the pipe.
        try:
            rest = self._proc.communicate(timeout=5)[0] or ""
        except Exception:  # noqa: BLE001 — best-effort drain
            rest = ""
        tail = self._decoder.decode(b"", final=True)
        if tail:
            rest = tail + rest
        if rest or self._pending:
            self._feed(rest + ("\n" if self._pending else ""))
            self._pending = ""
        self._exit_code = rc
        self._done = True

    # ── public ────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        self._drain()
        self._reap()
        return not self._done

    def poll(self) -> dict[str, Any]:
        """Check status once.  Kills the run if it exceeded the cap."""
        self._drain()
        self._reap()
        elapsed = time.perf_counter() - self._t0
        if not self._done and elapsed > self.cap_seconds:
            self.kill(reason=f"exceeded cap of {self.cap_seconds:.0f}s")
            self._timed_out = True
            self._reap()
        return self.status()

    def status(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._t0
        tail = "".join(self._output)[-_TAIL_CHARS:]
        return {
            "label": self.label,
            "running": not self._done,
            "passed": self._passed,
            "failed": self._failed,
            "errors": self._errors,
            "skipped": self._skipped,
            "seconds": round(elapsed, 2),
            "timed_out": self._timed_out,
            "killed": self._killed,
            "exit_code": self._exit_code,
            "output_tail": tail,
        }

    def kill(self, reason: str = "killed") -> dict[str, Any]:
        """Kill the whole process tree; return partial results."""
        if not self._done and not self._killed:
            _log.warning("background %s %s — killing process tree",
                         self.label, reason)
            _kill_tree(self._proc)
            self._killed = True
            self._reap()
        return self.status()

    def wait(
        self,
        *,
        poll_interval: float = 0.5,
        max_interval: float = 5.0,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll with backoff until the run finishes; return the final
        structured result (same shape as ``run_tests``)."""
        interval = poll_interval
        last = (-1, -1, -1)
        while True:
            st = self.poll()
            counts = (st["passed"], st["failed"], st["errors"])
            if on_progress is not None and counts != last:
                last = counts
                try:
                    on_progress(st)
                except Exception:  # noqa: BLE001 — progress never sinks
                    pass
            if not st["running"]:
                break
            # wake promptly at the cap deadline instead of oversleeping
            # the backoff, so the kill lands on time.
            remaining = self.cap_seconds - (time.perf_counter() - self._t0)
            time.sleep(min(interval, max(0.1, remaining)))
            interval = min(interval * 1.5, max_interval)
        return self.result()

    def result(self) -> dict[str, Any]:
        """Final structured result, shaped like ``run_tests`` output."""
        st = self.status()
        output = "".join(self._output)
        # findall yields (count, word) pairs — invert into word -> count.
        counts = {word.lower(): int(num)
                  for num, word in _SUMMARY_RE.findall(output)}
        passed = counts.get("passed", self._passed)
        failed = int(counts.get("failed", self._failed))
        errors = int(counts.get("error", self._errors))
        failed_items: list[dict[str, Any]] = []
        for m in re.finditer(r"^(FAILED|ERROR)\s+(\S+)", output, re.M):
            failed_items.append({
                "test_id": m.group(2),
                "file": m.group(2).split("::")[0],
                "error_snippet": st["output_tail"][-800:],
            })
        ok = (self._exit_code in (0, 5)) and not self._timed_out
        return {
            "ok": ok,
            "passed": passed,
            "failed": failed_items,
            "errors": errors,
            "seconds": st["seconds"],
            "selected": [self.label],
            "runner": "background",
            "timed_out": self._timed_out,
            "killed": self._killed,
            "partial": self._timed_out or self._killed,
            "note": ("killed after exceeding the cap — partial results"
                     if (self._timed_out or self._killed) else None),
        }


def background_run_tests(
    repo: str | Path | None = None,
    *,
    paths: list[str] | None = None,
    changed_only: bool = False,
    cap_seconds: float = DEFAULT_CAP_SECONDS,
) -> BackgroundTestRun | dict[str, Any]:
    """Background twin of ``pytest_runner.run_tests``: same command
    construction, but the suite runs in the background.

    Returns a ``BackgroundTestRun`` — or a plain result dict when
    ``changed_only`` matches nothing (honest skip, same as ``run_tests``).
    """
    from ..tools import pytest_runner as _pytest_mod

    root = _pytest_mod._repo_root(repo)
    selected: list[str] | None = list(paths) if paths else None
    if changed_only:
        selected = _pytest_mod.select_changed_tests(str(root))
        if not selected:
            return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                    "seconds": 0.0, "selected": [],
                    "runner": "background",
                    "note": "no tests matched the changed files — nothing ran"}
    if _pytest_mod._has_pytest():
        targets = selected or (["tests"] if (root / "tests").is_dir()
                               else ["."])
        argv: list[str] | None = [
            sys.executable, "-m", "pytest", "-x", "-q", "--tb=short",
            "-rf", *targets]
    else:
        argv = _pytest_mod._unittest_cmd(root, selected)
    if argv is None:
        # honest skip, same as run_tests: no tests is not a failure.
        return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                "seconds": 0.0, "selected": [],
                "runner": "background",
                "note": "no tests directory — nothing ran"}
    return BackgroundTestRun(argv, cwd=root, cap_seconds=cap_seconds)


__all__ = ["BackgroundTestRun", "background_run_tests",
           "DEFAULT_CAP_SECONDS"]
