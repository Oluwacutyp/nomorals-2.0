"""Smoke tests for scaffolded projects.

:func:`smoke_test` accepts a :class:`ServeHandle`, an HTTP(S) URL string,
or a project directory:

* HTTP target -- one or more :class:`HttpExpectation` probes are run
  (Kubernetes ``httpGet`` semantics: any 2xx–3xx is healthy, optional
  body-content assertion, per-probe timeout).
* Project directory of a CLI/console project -- the entrypoint is run
  with ``--help`` and must exit 0.

Returns a :class:`SmokeResult` with per-check details.  Failures carry
the actual stderr / HTTP body, never a vague message.
"""

from __future__ import annotations

import socket
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

__all__ = ["Check", "HttpExpectation", "SmokeResult", "smoke_test", "tcp_probe"]


@dataclass
class Check:
    """One assertion inside a smoke test."""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class HttpExpectation:
    """One HTTP probe: path, healthy statuses, optional body assertion.

    Mirrors Kubernetes ``httpGet`` probe semantics — any 2xx–3xx counts
    as healthy unless ``statuses`` pins the exact set.  ``body_contains``
    is the Docker HEALTHCHECK ``grep``: the response body must contain
    the substring.
    """

    path: str = "/"
    statuses: tuple[int, ...] = ()
    body_contains: str = ""
    timeout: float = 5.0
    #: Also assert this response header (name, expected value substring).
    header_contains: tuple[str, str] | None = None

    def label(self) -> str:
        bits = [f"GET {self.path}"]
        if self.statuses:
            bits.append(f"status in {sorted(self.statuses)}")
        else:
            bits.append("status 2xx/3xx")
        if self.body_contains:
            bits.append(f"body contains {self.body_contains!r}")
        return " ".join(bits)

    def healthy(self, status: int) -> bool:
        if self.statuses:
            return status in self.statuses
        return 200 <= status < 400


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


def _smoke_one(url: str, exp: HttpExpectation) -> Check:
    name = exp.label()
    try:
        req = urllib.request.Request(url.rstrip("/") + exp.path, method="GET")
        with urllib.request.urlopen(req, timeout=exp.timeout) as resp:
            body = resp.read(32768)
            status = resp.status
            headers = dict(resp.headers.items())
    except Exception as exc:  # noqa: BLE001 -- surfaced in the check detail
        return Check(name, False, f"request failed: {exc}")
    if not exp.healthy(status):
        snippet = body[:160].decode("utf-8", "replace")
        return Check(name, False,
                     f"status={status} body_snippet={snippet!r}")
    problems: list[str] = []
    if exp.body_contains:
        text = body.decode("utf-8", "replace")
        if exp.body_contains not in text:
            problems.append(f"body missing {exp.body_contains!r}")
    if exp.header_contains:
        hname, want = exp.header_contains
        got = next((v for k, v in headers.items()
                    if k.lower() == hname.lower()), "")
        if want not in got:
            problems.append(f"header {hname!r} missing {want!r} (got {got!r})")
    snippet = body[:120].decode("utf-8", "replace")
    if problems:
        return Check(name, False, "; ".join(problems) +
                     f" status={status} body_snippet={snippet!r}")
    return Check(name, True,
                 f"status={status} body_snippet={snippet!r}")


def _smoke_http(url: str, timeout: float,
                expectations: list[HttpExpectation] | None) -> list[Check]:
    expectations = expectations or [HttpExpectation(path="/", timeout=timeout)]
    return [_smoke_one(url, exp) for exp in expectations]


def tcp_probe(host: str, port: int, timeout: float = 5.0) -> Check:
    """Kubernetes ``tcpSocket`` probe: the port must accept a connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return Check(f"TCP {host}:{port} accepts", True, "connect ok")
    except OSError as exc:
        return Check(f"TCP {host}:{port} accepts", False, f"connect failed: {exc}")


def _smoke_cli(config: RunConfig, timeout: float,
               env: dict[str, str] | None = None) -> list[Check]:
    import os as _os

    child_env = dict(_os.environ)
    if env:
        child_env.update(env)
    try:
        proc = subprocess.run(
            [*config.command, "--help"],
            cwd=str(config.project_dir),
            env=child_env,
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
               timeout: float = 10.0,
               expectations: list[HttpExpectation] | None = None,
               env: dict[str, str] | None = None) -> SmokeResult:
    """Run a smoke test against ``target`` and return a structured result.

    ``target`` may be a :class:`ServeHandle` (HTTP smoke against its URL),
    an ``http(s)://`` URL string, or a project directory (CLI projects are
    smoked via ``--help``; HTTP projects are *not* auto-served -- use
    :func:`serve` first and pass the handle).

    ``expectations`` is a list of :class:`HttpExpectation` probes
    (default: ``GET /`` healthy on 2xx/3xx).  ``env`` is passed to the
    ``--help`` child for CLI projects.
    """
    started = time.monotonic()
    if isinstance(target, ServeHandle):
        checks = _smoke_http(target.url, timeout, expectations)
        label = target.url
    elif isinstance(target, str) and target.startswith(("http://", "https://")):
        checks = _smoke_http(target, timeout, expectations)
        label = target
    else:
        config = run_config(target)
        label = str(config.project_dir)
        if config.kind == "http":
            raise ToolError(
                "smoke_test() will not auto-serve an HTTP project; "
                "call serve() first and pass the ServeHandle"
            )
        checks = _smoke_cli(config, timeout, env=env)
    ok = all(c.ok for c in checks)
    elapsed = time.monotonic() - started
    _log.info("smoke_test %s -> %s (%.2fs)", label, "OK" if ok else "FAIL", elapsed)
    return SmokeResult(ok=ok, checks=checks, elapsed=elapsed, target=label)
