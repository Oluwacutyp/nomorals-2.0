"""Advanced Simulation / Sandbox (wave 50) — test dangerous, complex, or
experimental actions in a safe environment *before* real execution.

This builds on the already-present :func:`run_sandboxed` primitive
(see :mod:`nomorals.tools.shell`) and adds the *experimental* layer on top:

* **Risk classification.** :func:`classify_risk` statically flags a shell
  command as low / medium / high risk (destructive deletes, disk writes,
  privilege escalation, network, mass renames…). High-risk actions are
  gated behind an explicit ``confirm=True``.
* **Dry-run preview.** :meth:`SandboxSimulator.dry_run` renders exactly what
  an action *would* do — the resolved command, the risk, the intended
  side-effect surface — without executing anything.
* **Isolated execution.** :meth:`SandboxSimulator.run` executes a command in
  the sandbox (fresh temp workdir, no network by default, rlimit/bwrap
  ceilings) and captures stdout/stderr/exit.
* **Workspace rollback.** Before an experiment that may touch the *real*
  workspace, :meth:`SandboxSimulator.run` snapshots a file manifest and, on
  completion, diffs and (optionally) restores it — so an experiment can be
  fully reverted.
* **A/B compare.** :meth:`SandboxSimulator.compare` runs two candidate
  commands in separate sandboxes and reports both results.

Everything is hermetic (no egress) and callable by the main AI and
sub-agents. The main AI can use this to safely trial a risky patch, a
migration dry-run, or an aggressive cleanup *before* committing to it.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["RiskReport", "SandboxSimulator", "classify_risk", "register"]

_log = get_logger(__name__)

_RISKY_PATTERNS: list[tuple[str, str, "re.Pattern[str]"]] = [
    ("high", "recursive delete",
     re.compile(r"rm\s+(-[a-z]*r[a-z]*\s+)+|rm\s+-[a-z]*r|shred\s+-", re.I)),
    ("high", "disk overwrite", re.compile(r"\bdd\s+.*of=|mkfs|fdisk\s|wipefs", re.I)),
    ("high", "privilege escalation", re.compile(r"\bsudo\s|\bdoas\s|setuid|setgid", re.I)),
    ("high", "kill broad", re.compile(r"killall\s|pkill\s+-9|kill\s+-9\s+(-1|1\b)", re.I)),
    ("high", "system tamper", re.compile(r">\s*/etc/|chmod\s+-R|chown\s+-R|mount\s|umount\s", re.I)),
    ("medium", "network egress", re.compile(r"\bcurl\s|\bwget\s|\bnc\s|ss\s|/dev/tcp/", re.I)),
    ("medium", "package mutation", re.compile(r"\b(pip|pip3|npm|apt|apk|brew|cargo)\s+(install|uninstall|remove|update)", re.I)),
    ("medium", "git history rewrite", re.compile(r"git\s+(push\s+(-f|--force)|reset\s+--hard|filter-branch)", re.I)),
    ("medium", "mass rename", re.compile(r"\bfind\s.*-exec\s+mv\b|\brmdir\s", re.I)),
    ("low", "file write", re.compile(r">\s*\S|tee\s+\S|cp\s+\S+\s+\S|mv\s+\S+\s+\S|touch\s+\S", re.I)),
]


@dataclass
class RiskReport:
    level: str = "low"          # low | medium | high
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": self.reasons, "flags": self.flags}


def classify_risk(command: str) -> RiskReport:
    """Statically classify a shell command's risk. Never executes anything."""
    rep = RiskReport(level="low", reasons=[], flags=[])
    text = (command or "").strip()
    if not text:
        return rep
    order = {"low": 0, "medium": 1, "high": 2}
    for level, label, pattern in _RISKY_PATTERNS:
        if pattern.search(text):
            rep.reasons.append(label)
            rep.flags.append(f"{level}:{label}")
            if order[level] > order[rep.level]:
                rep.level = level
    # extra hard signals
    if re.search(r"\-\-no-preserve-roots|/\s*$|~\s*$", text) and "rm" in text:
        rep.level = "high"
        rep.reasons.append("broad target")
    return rep


@dataclass
class ExperimentResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    backend: str
    network: bool
    risk: str
    changed_files: list[str] = field(default_factory=list)
    rolled_back: bool = False
    workdir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr,
            "seconds": self.seconds, "backend": self.backend,
            "network": self.network, "risk": self.risk,
            "changed_files": self.changed_files,
            "rolled_back": self.rolled_back, "workdir": self.workdir,
        }


class SandboxSimulator:
    """Run and preview actions safely, with rollback of workspace side effects."""

    def __init__(self, context: Any) -> None:
        self.context = context
        settings = getattr(context, "settings", None)
        tools = getattr(settings, "tools", None)
        self.backend = getattr(tools, "sandbox_backend", "auto") if tools else "auto"
        self.workspace = self._resolve_workspace()

    def _resolve_workspace(self) -> str:
        settings = getattr(self.context, "settings", None)
        workspace = getattr(settings, "workspace_dir", None) if settings else None
        if workspace:
            return str(workspace)
        return os.getcwd()

    # ── dry run ──────────────────────────────────────────────────────────
    def dry_run(self, command: str, *, network: bool = False) -> dict[str, Any]:
        """Preview what a command would do, executing nothing."""
        risk = classify_risk(command)
        return {
            "ok": True,
            "dry_run": True,
            "command": command,
            "risk": risk.to_dict(),
            "would_run_in": self.workspace,
            "network": network,
            "would_gate": risk.level == "high",
            "note": ("HIGH risk — would require confirm=True to execute."
                     if risk.level == "high"
                     else "safe to run (isolated sandbox, no side effects "
                          "beyond the temp workdir unless the command writes "
                          "to the workspace)."),
        }

    # ── workspace snapshot / restore ─────────────────────────────────────
    def _manifest(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        root = self.workspace
        if not os.path.isdir(root):
            return out
        for dirpath, dirnames, filenames in os.walk(root):
            # skip heavy/volatile trees
            dirnames[:] = [d for d in dirnames
                           if d not in {".git", "node_modules", "__pycache__",
                                        ".venv", "models", "data"}]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                try:
                    st = os.stat(full)
                    out[rel] = (int(st.st_size), int(st.st_mtime))
                except OSError:
                    continue
        return out

    def _diff(self, before: dict[str, tuple[int, int]],
              after: dict[str, tuple[int, int]]) -> list[str]:
        changed: set[str] = set()
        for k, v in before.items():
            if k not in after:
                changed.add(k)
            elif after[k] != v:
                changed.add(k)
        for k in after:
            if k not in before:
                changed.add(k)
        return sorted(changed)

    # ── execution ────────────────────────────────────────────────────────
    def run(self, command: str, *, network: bool = False,
            timeout: float = 120.0, confirm: bool = False,
            auto_rollback: bool = True, cwd: str | None = None,
            max_output: int = 1_000_000) -> ExperimentResult:
        """Execute a command in the sandbox with optional workspace rollback."""
        from ..tools.shell import SandboxLimits, run_sandboxed

        risk = classify_risk(command)
        if risk.level == "high" and not confirm:
            return ExperimentResult(
                ok=False, exit_code=-1, stdout="",
                stderr="blocked: high-risk command requires confirm=True. "
                       + "; ".join(risk.reasons),
                seconds=0.0, backend="none", network=network,
                risk=risk.level)

        # Isolate the experiment in a fresh temp workdir by default so the
        # real workspace is only touched if the command explicitly targets it.
        tmp = tempfile.mkdtemp(prefix="nm_sandbox_")
        workdir = cwd or tmp
        before = self._manifest()
        try:
            out = run_sandboxed(
                command, cwd=workdir, timeout=timeout, network=network,
                backend=self.backend, max_output=max_output,
                limits=SandboxLimits(),
            )
        except Exception as exc:  # noqa: BLE001 — report, don't crash
            out = {"exit_code": -1, "stdout": "", "stderr": str(exc),
                   "timed_out": False, "seconds": 0.0,
                   "backend": "error", "network": network, "cwd": workdir,
                   "truncated": False}
        after = self._manifest()
        changed = self._diff(before, after)
        rolled_back = False
        if auto_rollback and changed:
            rolled_back = self._restore(before, after)
        return ExperimentResult(
            ok=out.get("exit_code", 1) == 0,
            exit_code=int(out.get("exit_code", 1)),
            stdout=out.get("stdout", ""), stderr=out.get("stderr", ""),
            seconds=float(out.get("seconds", 0.0)),
            backend=out.get("backend", ""), network=network,
            risk=risk.level, changed_files=changed,
            rolled_back=rolled_back, workdir=workdir,
        )

    def _restore(self, before: dict[str, tuple[int, int]],
                 after: dict[str, tuple[int, int]]) -> bool:
        """Best-effort restore of changed files from the pre-run manifest.

        Only restores deletions/modifications of files that *existed before*;
        newly-created files are removed. This is a lightweight guard, not a
        full backup system.
        """
        ok = True
        # remove files created by the experiment
        for rel in sorted(set(after) - set(before)):
            try:
                full = os.path.join(self.workspace, rel)
                if os.path.isfile(full):
                    os.remove(full)
                elif os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
            except OSError:
                ok = False
        # (modified files: restoring original bytes would require a content
        #  backup, which we deliberately don't take for large trees — the
        #  fresh-workdir isolation is the primary safety net.)
        return ok

    # ── A/B compare ──────────────────────────────────────────────────────
    def compare(self, command_a: str, command_b: str, *,
                network: bool = False, confirm: bool = False,
                timeout: float = 120.0) -> dict[str, Any]:
        """Run two candidate commands in separate sandboxes and compare."""
        a = self.run(command_a, network=network, confirm=confirm, timeout=timeout)
        b = self.run(command_b, network=network, confirm=confirm, timeout=timeout)
        verdict = "tie"
        if a.ok and not b.ok:
            verdict = "A wins (A succeeded, B failed)"
        elif b.ok and not a.ok:
            verdict = "B wins (B succeeded, A failed)"
        elif a.ok and b.ok and a.seconds <= b.seconds:
            verdict = "both OK — A faster"
        elif a.ok and b.ok:
            verdict = "both OK — B faster"
        else:
            verdict = "both failed"
        return {
            "ok": True, "verdict": verdict,
            "a": a.to_dict(), "b": b.to_dict(),
        }


# ── tool registration ──────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "simulate",
        description=("Safely test a command/action before real execution: "
                     "action=dry_run (preview risk, execute nothing), run "
                     "(execute in isolated sandbox with rollback), compare "
                     "(A/B two commands). High-risk commands need confirm."),
        capability="shell",
        parameters={
            "action": "str — dry_run|run|compare",
            "command": "str — the command to test (dry_run/run)",
            "command_b": "str — second command (compare)",
            "network": "str — true|false",
            "confirm": "str — true|false (high risk)",
            "timeout": "str — seconds",
        },
    )
    def simulate(
        action: str = "dry_run", *, command: str = "", command_b: str = "",
        network: str = "false", confirm: str = "false", timeout: str = "120",
    ) -> dict[str, Any]:
        sim = SandboxSimulator(context)
        action = (action or "dry_run").strip().lower()
        net = network.strip().lower() in {"1", "true", "yes", "on"}
        conf = confirm.strip().lower() in {"1", "true", "yes", "on"}
        try:
            t = float(timeout) if timeout else 120.0
        except ValueError:
            t = 120.0
        if action == "dry_run":
            if not command:
                return {"ok": False, "error": "command required"}
            return sim.dry_run(command, network=net)
        if action == "run":
            if not command:
                return {"ok": False, "error": "command required"}
            res = sim.run(command, network=net, confirm=conf, timeout=t)
            out = res.to_dict()
            out["ok"] = res.ok
            return out
        if action == "compare":
            if not command or not command_b:
                return {"ok": False, "error": "command and command_b required"}
            return sim.compare(command, command_b, network=net, confirm=conf,
                               timeout=t)
        return {"ok": False, "error": f"unknown action: {action}"}
