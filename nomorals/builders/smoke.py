"""Smoke tests for scaffolded projects.

:func:`smoke_test` accepts a :class:`ServeHandle`, an HTTP(S) URL string,
or a project directory:

* HTTP target -- ``GET /`` must return 200 within the deadline.
* Project directory of a CLI/console project -- the entrypoint is run
  with ``--help`` and must exit 0.

Returns a :class:`SmokeResult` with per-check details.  Failures carry
the actual stderr / HTTP body, never a vague message.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .run import RunConfig, ServeHandle, run_config

_log = get_logger(__name__)

__all__ = ["Check", "SmokeResult", "smoke_test"]


@dataclass
class Check:
    """One assertion inside a smoke test."""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class SmokeResult:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    elapsed: float = 0.0
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "target": self.target,
            "elapsed": round(self.elapsed, 3),
            "checks": [c.to_dict() for c in self.checks],
        }


def _smoke_http(url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    base = url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/", timeout=timeout) as resp:
            body = resp.read(4096)
            status = resp.status
    except Exception as exc:  # noqa: BLE001 -- surfaced in the check detail
        return [Check("GET / returns 200", False, f"request failed: {exc}")]
    snippet = body[:200].decode("utf-8", "replace")
    checks.append(Check(
        "GET / returns 200",
        status == 200,
        f"status={status} body_snippet={snippet!r}" if status != 200
        else f"status=200 body_snippet={snippet!r}",
    ))
    return checks


def _smoke_cli(config: RunConfig, timeout: float) -> list[Check]:
    try:
        proc = subprocess.run(
            [*config.command, "--help"],
            cwd=str(config.project_dir),
            stdin=subprocess.DEVNULL,  # interactive entrypoints must not hang on our stdin
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [Check("--help exits 0", False,
                      f"timed out after {timeout:.1f}s waiting for --help")]
    except OSError as exc:
        return [Check("--help exits 0", False, f"could not spawn: {exc}")]
    detail = (proc.stderr.strip() or proc.stdout.strip())[:300]
    return [Check(
        "--help exits 0",
        proc.returncode == 0,
        f"exit={proc.returncode} output={detail!r}" if proc.returncode != 0
        else f"exit=0 output={proc.stdout.strip()[:120]!r}",
    )]


def smoke_test(target: ServeHandle | str | Path, *,
               timeout: float = 10.0) -> SmokeResult:
    """Run a smoke test against ``target`` and return a structured result.

    ``target`` may be a :class:`ServeHandle` (HTTP smoke against its URL),
    an ``http(s)://`` URL string, or a project directory (CLI projects are
    smoked via ``--help``; HTTP projects are *not* auto-served -- use
    :func:`serve` first and pass the handle).
    """
    started = time.monotonic()
    if isinstance(target, ServeHandle):
        checks = _smoke_http(target.url, timeout)
        label = target.url
    elif isinstance(target, str) and target.startswith(("http://", "https://")):
        checks = _smoke_http(target, timeout)
        label = target
    else:
        config = run_config(target)
        label = str(config.project_dir)
        if config.kind == "http":
            raise ToolError(
                "smoke_test() will not auto-serve an HTTP project; "
                "call serve() first and pass the ServeHandle"
            )
        checks = _smoke_cli(config, timeout)
    ok = all(c.ok for c in checks)
    elapsed = time.monotonic() - started
    _log.info("smoke_test %s -> %s (%.2fs)", label, "OK" if ok else "FAIL", elapsed)
    return SmokeResult(ok=ok, checks=checks, elapsed=elapsed, target=label)
