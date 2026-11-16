"""The Auditor gate (Antigravity Teamwork, Devon version).

Teamwork's Auditor validates work against the selected integrity mode
and checks test evidence against REAL command output — a mocked pass
is a failure, not a pass. Devon already has reviewers that read diffs
and claims (the delegation claim ledger, the swarm verifier leg, the
bench verifier for bench tasks), but none of them re-runs a project's
evidence command itself. This organ is that gate, wired into the
project verification path — it extends the existing spine, it is not
a second review system.

Integrity modes (per project, durable in KV — the ``projects`` table
is migration-frozen, so no column can be added):

- ``development`` (default): lenient. Re-run the evidence command;
  fail only fabricated outputs and facade implementations. Undeclared
  mocks are flagged as warnings.
- ``demo``: moderate. Adds: undeclared mocks fail; implementation
  files must not import test modules (no reading test source to
  reverse-engineer expected behaviour); the evidence command must be
  local (no delegating core work to an external tool).
- ``benchmark``: maximum strictness, a superset of demo. Every claim
  needs a re-runnable command (command-less evidence fails outright);
  audited source files are stdlib-only.

Rules the Auditor CANNOT check mechanically ("no copying core logic
from open source", "fully independent from-scratch implementation")
are recorded as ``manual`` findings — listed in the verdict, never
passed or failed. The Auditor never pretends to check what it cannot.

Layering: agents (L5). The production command runner is the tools
sandbox (``tools/shell.py::run_sandboxed``, L4) — the same runner
``CodingAgent._run`` wraps for the acceptance replay. Verdicts are
journaled to a JSON ledger (WorkflowStore shape: atomic tmp+replace,
corrupt → quarantine + degrade, bad rows skipped).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..storage.kv import KVStore
from .failures import record_failure

_log = get_logger(__name__)

INTEGRITY_MODES = ("development", "demo", "benchmark")
DEFAULT_MODE = "development"
INTEGRITY_KV_PREFIX = "auditor:integrity."
AUDIT_TIMEOUT_S = 60.0

#: Mock markers the Auditor scans test source for. A marker the claim
#: did not declare is an undeclared mock.
_MOCK_MARKERS = (
    "unittest.mock",
    "Mock(",
    "MagicMock(",
    "patch(",
    "monkeypatch",
    "mocker",
)

#: Words that mean the evidence command reaches outside the sandbox.
_REMOTE_TOKENS = ("curl", "wget", "ssh", "scp", "rsync")
_REMOTE_URLS = ("http://", "https://")

_MANUAL_RULES = {
    "demo": (
        "no copying core logic from open source",
    ),
    "benchmark": (
        "no copying core logic from open source",
        "fully independent from-scratch implementation",
    ),
}


class AuditorError(Exception):
    """The audit could not run, or the evidence failed the gate."""


@dataclass
class AuditEvidence:
    """One claim + the evidence for it.

    ``command`` is the command to re-run ("" = no re-runnable evidence).
    ``claimed_exit`` / ``claimed_output`` are what the producer claims
    the command produced; ``expected_output_hash`` is a frozen hash
    (e.g. from the project's acceptance ledger) the re-run output must
    still equal. ``test_files`` / ``source_files`` are paths under the
    workdir. ``declared_mocks`` names the mocks the claim discloses.
    """
    command: str = ""
    claimed_exit: int | None = None
    claimed_output: str = ""
    expected_output_hash: str = ""
    test_files: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    declared_mocks: tuple[str, ...] = ()


@dataclass
class AuditVerdict:
    passed: bool
    mode: str
    command: str = ""
    exit_code: int | None = None
    output_hash: str = ""
    rerun: bool = False
    reasons: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode,
            "command": self.command,
            "exit_code": self.exit_code,
            "output_hash": self.output_hash,
            "rerun": self.rerun,
            "reasons": list(self.reasons),
            "findings": [dict(f) for f in self.findings],
            "checked_at": self.checked_at,
        }


def _output_hash(text: str) -> str:
    """The acceptance ledger's hash shape (sha256, first 16 hex)."""
    return hashlib.sha256(
        str(text or "").encode("utf-8", "replace")).hexdigest()[:16]


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_auditor_path(home: Path | str | None = None) -> Path:
    """The durable audit ledger under a home (``<home>/data/auditor.json``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "auditor.json"


def default_run_command(command: str, *, timeout: float = AUDIT_TIMEOUT_S,
                        workdir: str | Path | None = None) -> dict[str, Any]:
    """Run a command in the tools sandbox. The production runner."""
    from ..tools.shell import run_sandboxed
    out = run_sandboxed(command, cwd=workdir or ".", timeout=timeout)
    return {
        "exit_code": out.get("exit_code"),
        "stdout": out.get("stdout") or "",
        "stderr": out.get("stderr") or "",
        "timed_out": bool(out.get("timed_out")),
    }


def _home_from_context(context: Any) -> Any:
    settings = getattr(context, "settings", None)
    if settings is not None:
        home = getattr(settings, "home", None)
        if home is not None:
            return home
    return getattr(context, "home", None)


def _validate_mode(mode: str) -> str:
    if mode not in INTEGRITY_MODES:
        raise AuditorError(
            f"unknown integrity mode {mode!r} — "
            f"expected one of {', '.join(INTEGRITY_MODES)}")
    return mode


def get_integrity_mode(db: Any, project_id: str) -> str:
    """The project's integrity mode, durable in KV. Default development."""
    try:
        raw = KVStore(db).get(INTEGRITY_KV_PREFIX + project_id)
    except Exception as exc:  # noqa: BLE001 — a dead store degrades
        record_failure("auditor.mode_read", exc)
        return DEFAULT_MODE
    mode = ""
    if isinstance(raw, dict):
        mode = str(raw.get("mode") or "")
    elif isinstance(raw, str):
        mode = raw.strip()
    return mode if mode in INTEGRITY_MODES else DEFAULT_MODE


def set_integrity_mode(db: Any, project_id: str, mode: str) -> str:
    """Set a project's integrity mode. Unknown modes raise."""
    _validate_mode(mode)
    KVStore(db).set_raw(INTEGRITY_KV_PREFIX + project_id,
                        json.dumps({"mode": mode}), "json")
    return mode


class AuditLedger:
    """Journaled audit verdicts, durable JSON, one file.

    ``path=None`` means in-memory only (tests, and runtimes with no
    home); the row shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.degraded_reason = ""
        self._rows: list[dict[str, Any]] = []
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — corrupt store degrades
            self.degraded_reason = f"corrupt audit ledger: {exc}"
            with contextlib.suppress(Exception):
                self.path.rename(
                    self.path.with_name(self.path.name + ".quarantined"))
            self._rows = []
            return
        if not isinstance(raw, list):
            self.degraded_reason = "audit ledger is not a list"
            self._rows = []
            return
        rows: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("id"), str) or not item["id"]:
                continue
            if not isinstance(item.get("project_id"), str):
                continue
            rows.append({
                "id": item["id"],
                "project_id": item["project_id"],
                "step_id": str(item.get("step_id") or ""),
                "mode": str(item.get("mode") or DEFAULT_MODE),
                "command": str(item.get("command") or ""),
                "verdict": bool(item.get("verdict", False)),
                "reasons": list(item.get("reasons") or []),
                "output_hash": str(item.get("output_hash") or ""),
                "ts": float(item.get("ts") or 0.0),
            })
        self._rows = rows

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path, json.dumps(self._rows, indent=2, default=str))

    def record(self, *, project_id: str, step_id: str = "", mode: str = "",
               command: str = "", verdict: bool = False,
               reasons: list[str] | None = None,
               output_hash: str = "") -> dict[str, Any]:
        row = {
            "id": new_short_id("aud"),
            "project_id": project_id,
            "step_id": step_id,
            "mode": mode or DEFAULT_MODE,
            "command": command,
            "verdict": bool(verdict),
            "reasons": list(reasons or []),
            "output_hash": output_hash,
            "ts": time.time(),
        }
        self._rows.append(row)
        try:
            self._save()
        except Exception as exc:  # noqa: BLE001 — journaling never fails a run
            record_failure("auditor.ledger_save", exc)
        return row

    def recent(self, project_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = [r for r in self._rows if r["project_id"] == project_id]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return [dict(r) for r in rows[:limit]]


def _discover_test_files(evidence: AuditEvidence,
                         workdir: Path | None) -> list[str]:
    """The test files the evidence covers: named, else found."""
    if evidence.test_files:
        return list(evidence.test_files)
    found: list[str] = []
    for token in re.split(r"\s+", evidence.command or ""):
        if token.endswith(".py"):
            found.append(token.strip("\"'"))
    if found:
        return found
    if workdir is not None:
        try:
            for path in sorted(workdir.rglob("test_*.py")):
                found.append(str(path.relative_to(workdir)))
            for path in sorted((workdir / "tests").rglob("*.py")):
                rel = str(path.relative_to(workdir))
                if rel not in found:
                    found.append(rel)
        except Exception as exc:  # noqa: BLE001 — discovery is best-effort
            record_failure("auditor.discover_tests", exc)
    return found


def _scan_mocks(workdir: Path | None,
                test_files: list[str]) -> dict[str, list[str]]:
    """Map test file -> mock markers found in its source."""
    hits: dict[str, list[str]] = {}
    for rel in test_files:
        path = (workdir / rel) if workdir is not None else Path(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — unreadable file is a finding
            hits[rel] = ["<unreadable>"]
            continue
        found = [m for m in _MOCK_MARKERS if m in text]
        if found:
            hits[rel] = found
    return hits


def _top_level_imports(path: Path) -> list[str] | None:
    """Top-level imported module names, or None when unparseable."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 — unparseable file is a finding
        return None
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name.split(".")[0] for a in node.names)
        elif (isinstance(node, ast.ImportFrom)
                and node.level == 0 and node.module):
            names.append(node.module.split(".")[0])
    return names


def _is_test_import(name: str) -> bool:
    return (name == "tests" or name.startswith("tests.")
            or name.startswith("test_"))


class Auditor:
    """Re-run test evidence against real command output."""

    def audit(self, evidence: AuditEvidence, *, mode: str,
              run_fn: Any = None,
              workdir: str | Path | None = None) -> AuditVerdict:
        _validate_mode(mode)
        work = Path(workdir) if workdir is not None else None
        reasons: list[str] = []
        findings: list[dict[str, Any]] = []

        def _finding(rule: str, severity: str, detail: str) -> None:
            findings.append({"rule": rule, "severity": severity,
                             "detail": detail})

        for rule in _MANUAL_RULES.get(mode, ()):
            _finding(f"manual/{rule}", "manual",
                     f"{rule} — the Auditor cannot check this "
                     "mechanically; listed, not passed or failed")

        exit_code: int | None = None
        output_hash = ""
        rerun = False
        stdout = ""
        if evidence.command:
            rerun = True
            # One runner contract: fn(command, *, timeout, workdir).
            # The module global is read at call time so tests can patch
            # default_run_command with the same contract.
            runner = run_fn or default_run_command
            try:
                out = runner(evidence.command, timeout=AUDIT_TIMEOUT_S,
                             workdir=work)
            except Exception as exc:  # noqa: BLE001 — a dead runner fails
                reasons.append(f"evidence command could not run: {exc}")
                return AuditVerdict(passed=False, mode=mode,
                                    command=evidence.command, rerun=True,
                                    reasons=reasons, findings=findings)
            if out.get("timed_out"):
                reasons.append(
                    f"evidence command timed out after {AUDIT_TIMEOUT_S:.0f}s")
                return AuditVerdict(
                    passed=False, mode=mode, command=evidence.command,
                    exit_code=out.get("exit_code"), rerun=True,
                    reasons=reasons, findings=findings)
            exit_code = out.get("exit_code")
            stdout = str(out.get("stdout") or "")
            output_hash = _output_hash(stdout)
            if (evidence.claimed_exit is not None
                    and exit_code != evidence.claimed_exit):
                reasons.append(
                    f"claimed exit {evidence.claimed_exit}, re-run exited "
                    f"{exit_code} — the evidence does not reproduce")
            if exit_code != 0:
                reasons.append(
                    f"evidence command exited {exit_code} — the claimed "
                    "pass did not happen")
            elif not stdout.strip():
                reasons.append(
                    "evidence command exited 0 with empty output — a "
                    "facade, not evidence")
            if (evidence.claimed_output
                    and _output_hash(evidence.claimed_output)
                    != output_hash):
                reasons.append(
                    "claimed output does not match the re-run output "
                    "— the evidence looks fabricated")
            if (evidence.expected_output_hash
                    and evidence.expected_output_hash != output_hash):
                reasons.append(
                    "re-run output hash does not match the frozen "
                    "acceptance hash — the evidence no longer reproduces")
        else:
            if mode == "benchmark":
                reasons.append(
                    "benchmark demands a re-runnable command for every "
                    "claim — this evidence has none")
            else:
                _finding("no-command", "warn",
                         "no command to re-run — nothing audited")

        # Mock scan: undeclared mocks fail in demo/benchmark, warn in dev.
        test_files = _discover_test_files(evidence, work)
        if evidence.command or test_files:
            mock_hits = _scan_mocks(work, test_files)
            if not test_files:
                _finding("mock-scan-skipped", "warn",
                         "no test files found — mock scan skipped")
            declared = [d.lower() for d in evidence.declared_mocks]
            for rel, markers in sorted(mock_hits.items()):
                if markers == ["<unreadable>"]:
                    _finding("mock-scan-unreadable", "warn",
                             f"could not read {rel} — mock scan skipped")
                    continue
                undeclared = []
                for marker in markers:
                    # An import marker (no call) is the conduit, not a
                    # mock itself: disclosing any mock discloses it.
                    if "(" not in marker and declared:
                        continue
                    if not any(d in marker.lower() for d in declared):
                        undeclared.append(marker)
                if not undeclared:
                    continue
                detail = (f"undeclared mock {', '.join(undeclared)} in "
                          f"{rel}")
                if mode in ("demo", "benchmark"):
                    reasons.append(detail + " — demo/benchmark demand "
                                            "mocks be declared")
                else:
                    _finding("mock-undeclared", "warn", detail)

        # Demo: implementation must not import test modules.
        if mode in ("demo", "benchmark"):
            for rel in evidence.source_files:
                path = (work / rel) if work is not None else Path(rel)
                names = _top_level_imports(path) if path.exists() else None
                if names is None:
                    _finding("test-import-unparseable", "warn",
                             f"could not parse {rel} — test-import scan "
                             "skipped")
                    continue
                for name in names:
                    if _is_test_import(name):
                        reasons.append(
                            f"{rel} imports test module {name!r} — the "
                            "implementation reads test source to "
                            "reverse-engineer expected behaviour")

        # Demo: the evidence command must be local.
        if mode in ("demo", "benchmark") and evidence.command:
            lowered = evidence.command.lower()
            if (any(t in lowered.split() for t in _REMOTE_TOKENS)
                    or any(u in lowered for u in _REMOTE_URLS)):
                reasons.append(
                    "evidence command reaches outside the sandbox — "
                    "demo/benchmark forbid delegating core work to an "
                    "external tool")

        # Benchmark: stdlib-only imports in the audited source files.
        if mode == "benchmark":
            stdlib = set(getattr(sys, "stdlib_module_names", ()))
            for rel in evidence.source_files:
                path = (work / rel) if work is not None else Path(rel)
                if not path.exists():
                    continue
                names = _top_level_imports(path)
                if names is None:
                    _finding("stdlib-unparseable", "warn",
                             f"could not parse {rel} — stdlib scan skipped")
                    continue
                for name in names:
                    if name in stdlib:
                        continue
                    local = ((work / f"{name}.py").exists()
                             if work is not None else False) or (
                                 (work / name / "__init__.py").exists()
                                 if work is not None else False)
                    if not local:
                        reasons.append(
                            f"non-stdlib import {name!r} in {rel} — "
                            "benchmark is stdlib-only")

        return AuditVerdict(
            passed=not reasons, mode=mode, command=evidence.command,
            exit_code=exit_code, output_hash=output_hash, rerun=rerun,
            reasons=reasons, findings=findings)


def audit_project_step(pm: Any, project: Any, *, step_id: str = "",
                       run_fn: Any = None,
                       workdir: str | Path | None = None,
                       ledger: AuditLedger | None = None,
                       ) -> AuditVerdict | None:
    """Audit one project step's evidence under the project's mode.

    Build steps carry a ``verify_cmd``: the evidence is that command
    plus the frozen acceptance hash for the step — the re-run output
    must still equal the hash the step froze when it passed. Steps
    with no verifiable evidence are audited only in ``benchmark``
    (where a command-less claim fails); otherwise there is nothing
    to audit and this returns None. Every verdict is journaled.
    """
    mode = get_integrity_mode(pm.db, project.id)
    verify_cmd = str(getattr(project, "verify_cmd", "") or "")
    if not verify_cmd and mode != "benchmark":
        return None
    expected_hash = ""
    if verify_cmd:
        step = step_id or ""
        try:
            rows = pm._acceptance_rows(project.id, ok_only=True)
            for row in reversed(rows):
                if not step or str(row.get("step_id") or "") == step:
                    expected_hash = str(row.get("output_hash") or "")
                    break
        except Exception as exc:  # noqa: BLE001 — no frozen row, audit anyway
            record_failure("auditor.frozen_hash", exc)
    work = workdir
    if work is None:
        try:
            from ..tools.filesystem import safe_path
            work = safe_path(pm.context, ".")
        except Exception:  # noqa: BLE001 — file scans degrade to skipped
            work = None
    source_files = []
    artifact = str(getattr(project, "artifact", "") or "")
    if artifact:
        source_files.append(artifact)
    evidence = AuditEvidence(
        command=verify_cmd, claimed_exit=0 if verify_cmd else None,
        expected_output_hash=expected_hash, source_files=tuple(source_files))
    verdict = Auditor().audit(evidence, mode=mode, run_fn=run_fn,
                              workdir=work)
    store = ledger
    if store is None:
        home = _home_from_context(getattr(pm, "context", None))
        store = AuditLedger(path=default_auditor_path(home))
    store.record(project_id=project.id, step_id=step_id, mode=mode,
                 command=verify_cmd, verdict=verdict.passed,
                 reasons=verdict.reasons, output_hash=verdict.output_hash)
    return verdict
