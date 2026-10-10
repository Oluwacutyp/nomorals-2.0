"""The framework self-improvement agent (a.k.a. the Evolver) — god tier.

An agent with deep access to the *entire* NoMorals framework: OSINT,
browser, code interpreter, memory, social connectors, scheduler, tools,
performance — any part of the repo.  It behaves like a senior engineer
constantly trying to make the system stronger:

1. **Research** — `research(topic)`: deep codebase evidence (module
   health, markers, untested areas, recent evolution history) fused with
   expert LLM analysis into a structured report with concrete,
   high-quality recommendations.
2. **Audit** — `audit()`: a deterministic scanner that detects real
   weaknesses: TODO/FIXME/HACK markers, dead stubs, modules with no
   tests, oversized files, long functions.  No model needed — always
   honest, always fast.
3. **Plan** — `plan(instruction, research_id, focus)`: the model writes
   an *edit list* with real file contents in context (not just names);
   every edit is validated against the working tree (exact, unique
   match; no escapes from the repo root).
4. **Apply** — `apply(proposal)`: writes the edits, runs the import
   smoke + the FULL unittest suite in a subprocess, and keeps the change
   only if everything passes.  Fail → `git reset --hard` restores the
   pre-change tree exactly.  Successful applies become real git commits
   on the configured evolution work branch (config `[evolution]`
   `work_branch`; default = the branch the repo is on), each tagged
   `evo/<id>` so it can be rolled back cleanly.
   **Publish** — `git.publish()`: fast-forwards the main branch to the
   evolution tip (FF-only, never auto-resolves conflicts) and optionally
   pushes — the owner's explicit "make it permanent" step.
5. **Revert** — `revert(proposal_id)`: `git revert` of a committed
   evolution (or a file-level checkout for on-disk changes) — clean,
   auditable undo.
6. **Autopilot** — `autopilot(steps)` (power mode): audit → pick the
   top actionable finding → plan → verify → apply → repeat.  Every
   step is independently gate-verified, so the bot never ships a crash
   even while improving itself.

Gating: in power mode plans apply directly; in normal mode a plan needs
explicit owner approval (`/evolve apply <id>`).  Everything — research
reports, proposals, verdicts, reverts — is kept in the state DB.

What the Evolver will never touch (these are the contract, not
limitations): secret files (.env, private keys, certificates), git
internals, and the safety-contract test suite itself.  Everything else
— including its own source — is fair game, verified by the gate.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.brain import brain_for
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = ["EvolutionAgent", "EvolutionGitController",
           "EvolutionProposal", "register"]

# Repository root: this file is <root>/nomorals/agents/evolution.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The Evolver works on ANY part of the framework.  The only things it
# may never touch — even with a perfect test run — are secrets, git
# plumbing, and the safety-contract tests (an agent that can edit the
# contract has no contract).  Everything else, including this very
# module, is fair game: edits are verified by the gate before they stay.
_FORBIDDEN_PATHS = (
    ".git",
    "tests/test_e2e_safety.py",  # the safety contract tests
)
_FORBIDDEN_SUBSTR = (".env", "id_rsa", ".pem", ".key")
_FORBIDDEN_EXACT = (".env", ".env.example")

_TEST_TIMEOUT = 1500  # seconds for the full suite (generous phone budget)
_BENCHMARK_TOLERANCE = 0.05  # regression allowed before a promotion is blocked


@dataclass
class EvolutionProposal:
    """A concrete, reviewable set of framework edits."""

    id: str
    instruction: str
    edits: list[dict[str, str]] = field(default_factory=list)
    status: str = "planned"  # planned | verifying | applied | rejected | reverted
    rationale: str = ""
    created_at: float = field(default_factory=time.time)
    verify_result: str = ""
    commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "edits": self.edits,
            "status": self.status,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "verify_result": self.verify_result,
            "commit": self.commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionProposal":
        return cls(
            id=str(data.get("id") or ""),
            instruction=str(data.get("instruction") or ""),
            edits=[e for e in (data.get("edits") or []) if isinstance(e, dict)],
            status=str(data.get("status") or "planned"),
            rationale=str(data.get("rationale") or ""),
            created_at=float(data.get("created_at") or time.time()),
            verify_result=str(data.get("verify_result") or ""),
            commit=str(data.get("commit") or ""),
        )


def _compact_hunks(edits: list[dict[str, Any]], *, max_edits: int = 6,
                   max_lines: int = 12) -> list[dict[str, Any]]:
    """Chat-sized unified diffs for the applied-result record.

    Captured at apply time because the pre-apply file content is gone
    afterwards.  Hard-capped: this record is stored as JSON in the state
    DB and rendered into chat, so it must stay small.
    """
    hunks: list[dict[str, Any]] = []
    for edit in edits[:max_edits]:
        if not isinstance(edit, dict):
            continue
        path = str(edit.get("path") or "?")
        old = str(edit.get("old") or "")
        new = str(edit.get("new") or "")
        if not old and new:
            body = new.splitlines()
            diff = [f"+ {ln}" for ln in body[:max_lines]]
            if len(body) > max_lines:
                diff.append(f"… +{len(body) - max_lines} more added lines")
        else:
            full = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="", n=2))
            body = full[2:]  # drop the ---/+++ file headers
            diff = body[:max_lines]
            if len(body) > max_lines:
                diff.append(f"… +{len(body) - max_lines} more diff lines")
        hunks.append({"path": path,
                      "diff": diff if diff else ["(no visible change)"]})
    return hunks


class EvolutionGitController:
    """Branch-aware git policy for the Evolver.

    Evolution changes are REAL COMMITS in the repo the agent runs in.
    This controller decides *where* they land and *how* they become
    permanent — and keeps both decisions auditable and reversible:

    - ``work_branch``: the branch evolution commits are made on.  Empty
      (default) = stay on whatever branch is checked out.  When set, the
      branch is created from the current HEAD the first time it is used,
      and evolution keeps building on its own tip afterwards.
    - ``main_branch``: the branch ``publish()`` fast-forwards.
    - ``push``: whether publish also ``git push``es.  Default off —
      pushing is a deliberate owner action, not an agent side-effect.

    Everything here is non-destructive by construction: FF-only merges
    (a diverged branch is refused, never auto-resolved), clean-tree
    checks before any branch switch, and every evolution commit carries
    an ``evo/<id>`` tag for one-command rollback.
    """

    def __init__(self, repo_root: Path, *, work_branch: str = "",
                 main_branch: str = "main", push: bool = False) -> None:
        self.repo_root = Path(repo_root)
        self.work_branch = (work_branch or "").strip()
        self.main_branch = (main_branch or "main").strip() or "main"
        self.push = bool(push)

    # ── introspection ────────────────────────────────────────────────────
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=self.repo_root,
                              capture_output=True, text=True, timeout=120)

    def branch(self) -> str:
        result = self._run("branch", "--show-current")
        return result.stdout.strip() if result.returncode == 0 else ""

    def remote(self) -> str:
        result = self._run("remote", "get-url", "origin")
        return result.stdout.strip() if result.returncode == 0 else ""

    def clean(self) -> bool:
        result = self._run("status", "--porcelain")
        return result.returncode == 0 and not result.stdout.strip()

    def status(self) -> dict[str, Any]:
        """Owner-facing git state: branch, remote, ahead/behind, policy."""
        out: dict[str, Any] = {
            "branch": self.branch(),
            "remote": self.remote(),
            "clean": self.clean(),
            "work_branch": self.work_branch or "(current branch)",
            "main_branch": self.main_branch,
            "push_on_publish": self.push,
        }
        ahead = self._run("rev-list", "--left-right", "--count",
                          "HEAD...@{u}")
        if ahead.returncode == 0:
            left, sep, right = ahead.stdout.partition("\t")
            try:
                out["ahead"] = int(left or 0)
                out["behind"] = int(right or 0) if sep else 0
            except ValueError as e:
                _log.debug("unparseable rev-list output: %s", e)
        return out

    # ── lifecycle ────────────────────────────────────────────────────────
    def ensure_work_branch(self) -> str:
        """Move to the configured work branch (if any) before editing.

        First use creates it from the current HEAD; later uses check it
        out as-is so evolution history accumulates on its own tip.
        Returns the branch evolution commits will land on.
        """
        current = self.branch()
        if not self.work_branch:
            return current
        if current == self.work_branch:
            return current
        if not self.clean():
            raise ToolError(
                "cannot switch to the evolution work branch "
                f"{self.work_branch!r}: the working tree is dirty — commit "
                "or revert current changes first")
        exists = self._run("rev-parse", "--verify", "--quiet",
                           f"refs/heads/{self.work_branch}")
        if exists.returncode == 0:
            result = self._run("checkout", self.work_branch)
        else:
            result = self._run("checkout", "-b", self.work_branch)
        if result.returncode != 0:
            raise ToolError(
                f"git checkout {self.work_branch!r} failed: "
                f"{(result.stderr or result.stdout)[-300:]}")
        return self.work_branch

    def commit_and_tag(self, message: str, tag: str) -> dict[str, str]:
        """``git add -A`` + commit + the ``evo/<id>`` rollback tag.

        Returns ``{commit, tag, branch}``; raises ToolError when the
        commit itself fails (the change stays on disk for inspection).
        """
        out: dict[str, str] = {"branch": self.branch()}
        add = self._run("add", "-A")
        commit = self._run("commit", "-m", message)
        if add.returncode != 0 and commit.returncode != 0:
            raise ToolError("git commit failed: "
                            f"{(commit.stderr or commit.stdout)[-300:]}")
        head = self._run("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            raise ToolError("git commit failed: no HEAD after commit")
        out["commit"] = head.stdout.strip()
        tagged = self._run("tag", "-f", tag)
        if tagged.returncode == 0:
            out["tag"] = tag
        return out

    def publish(self, target: str = "", *, push: bool | None = None) -> dict[str, Any]:
        """Fast-forward ``target`` to the current branch's tip.

        FF-only on purpose: if the branches have diverged the publish is
        REFUSED — the evolver never auto-resolves conflicts or writes
        merge commits.  After the merge it returns to the work branch and
        (when push is enabled) pushes both branches so the remote never
        diverges from the working copy.
        """
        target = (target or self.main_branch).strip()
        if push is None:
            push = self.push
        work = self.branch()
        if not work:
            raise ToolError("cannot determine the current branch")
        out: dict[str, Any] = {"target": target, "branch": work}
        if target == work:
            out["published"] = False
            out["reason"] = "already on the target branch — nothing to merge"
            if push:
                out.update(self._push(target))
            return out
        target_exists = self._run("rev-parse", "--verify", "--quiet",
                                  f"refs/heads/{target}")
        if target_exists.returncode != 0:
            raise ToolError(
                f"publish refused: branch {target!r} does not exist — "
                "create it first (the evolver does not guess branch names)")
        ancestor = self._run("merge-base", "--is-ancestor", target, "HEAD")
        if ancestor.returncode != 0:
            raise ToolError(
                f"publish refused: {target!r} has diverged from {work!r} — "
                "merge it by hand first; the evolver never auto-resolves "
                "conflicts")
        if not self.clean():
            raise ToolError(
                "working tree is dirty — commit or revert changes before "
                "publishing")
        to_target = self._run("checkout", target)
        if to_target.returncode != 0:
            raise ToolError(
                f"cannot check out {target!r}: "
                f"{(to_target.stderr or to_target.stdout)[-300:]}")
        merge = self._run("merge", "--ff-only", work)
        if merge.returncode != 0:
            self._run("checkout", work)
            raise ToolError(
                "fast-forward merge failed (branches moved?): "
                f"{(merge.stderr or merge.stdout)[-300:]}")
        head = self._run("rev-parse", "--short", "HEAD")
        self._run("checkout", work)
        out.update({
            "published": True,
            "method": "fast-forward",
            "commit": head.stdout.strip() if head.returncode == 0 else "",
        })
        if push:
            out.update(self._push(target))
            if target != work:
                out.update(self._push(work))
        return out

    def _push(self, branch: str) -> dict[str, Any]:
        if not self.remote():
            return {"pushed": False, "reason": "no 'origin' remote configured"}
        result = self._run("push", "origin", branch)
        if result.returncode != 0:
            return {"pushed": False,
                    "reason": (result.stderr or result.stdout)[-300:]}
        return {"pushed": True}


class EvolutionAgent:
    """Plans, verifies, and applies framework changes — tests first, always."""

    def __init__(self, context: Any, *, repo_root: Path | None = None) -> None:
        self.context = context
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        # branch policy for where evolution commits land (config:
        # [evolution] work_branch / main_branch / push_on_publish)
        settings = getattr(context, "settings", None)
        ev = getattr(settings, "evolution", None)
        self.git = EvolutionGitController(
            self.repo_root,
            work_branch=str(getattr(ev, "work_branch", "") or ""),
            main_branch=str(getattr(ev, "main_branch", "") or "main"),
            push=bool(getattr(ev, "push_on_publish", False)),
        )

    def git_status(self) -> dict[str, Any]:
        """Owner-facing view of the evolver's git policy + repo state."""
        return self.git.status()

    # ── 1. measure: the system's own telemetry ──────────────────────────────
    _METRICS_KEY = "evolution.metrics"

    def measure(self) -> dict[str, Any]:
        """Snapshot the system's health signals — fast, pure, hermetic.

        This is the *measure* half of the closed loop: test surface, code
        size, tool count, skill/failure/KG memory sizes, and the last
        benchmark score.  Every reading is appended to a bounded time
        series (kv ``evolution.metrics``) so later cycles can see deltas.
        """
        tests_dir = self.repo_root / "tests"
        test_files = test_lines = 0
        if tests_dir.is_dir():
            for p in sorted(tests_dir.glob("test_*.py")):
                test_files += 1
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    test_lines += sum(1 for l in text.splitlines()
                                      if "def test_" in l)
                except OSError as e:
                    _log.debug("unreadable test file %s: %s", p, e)
        prod_files = prod_lines = 0
        nomo = self.repo_root / "nomorals"
        if nomo.is_dir():
            for p in sorted(nomo.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                prod_files += 1
                try:
                    prod_lines += len(p.read_text(
                        encoding="utf-8", errors="replace").splitlines())
                except OSError as e:
                    _log.debug("unreadable source file %s: %s", p, e)
        tools = skills = kg_nodes = kg_edges = 0
        try:
            tools = len(getattr(self.context, "tools", None) or [])
        except TypeError:
            tools = 0
        db = getattr(self.context, "db", None)
        if db is not None:
            try:
                skills = db.query_one("SELECT COUNT(*) AS n FROM skills")["n"]
            except Exception:  # noqa: BLE001
                skills = 0
            try:
                kg_nodes = db.query_one(
                    "SELECT COUNT(*) AS n FROM kg_nodes")["n"]
                kg_edges = db.query_one(
                    "SELECT COUNT(*) AS n FROM kg_edges")["n"]
            except Exception:  # noqa: BLE001
                pass
        failures_24h = 0
        try:
            failures_24h = db.query_one(
                "SELECT COUNT(*) AS n FROM failures WHERE ts > ?",
                (time.time() - 86400.0,))["n"]
        except Exception:  # noqa: BLE001
            pass
        snapshot = {
            "ts": time.time(),
            "tests": test_lines,
            "test_files": test_files,
            "prod_lines": prod_lines,
            "prod_files": prod_files,
            "tools": tools,
            "skills": skills,
            "kg_nodes": kg_nodes,
            "kg_edges": kg_edges,
            "failures_24h": failures_24h,
        }
        # persist a bounded time series
        try:
            if db is not None:
                kv = KVStore(db)
                series: list[dict[str, Any]] = kv.get(self._METRICS_KEY, default=[])
                series.append(snapshot)
                series = series[-200:]
                kv.set(self._METRICS_KEY, series)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            pass
        # last benchmark score if one was stored
        try:
            val = KVStore(db).get_raw(self._BENCHMARK_KEY)
            if val:
                snapshot["benchmark"] = float(val)
        except Exception:  # noqa: BLE001
            pass
        return snapshot

    def metrics_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """The persisted measure time-series (oldest first)."""
        db = getattr(self.context, "db", None)
        if db is None:
            return []
        try:
            data = KVStore(db).get(self._METRICS_KEY, default=[])
            return data[-max(1, limit):]
        except Exception:  # noqa: BLE001
            return []

    def _outcome_context(self, instruction: str) -> str:
        """What the system knows about how this kind of change went before.

        Metric deltas from the last measure, recent evolution outcomes,
        failure lessons matching the instruction, and systemic skill traps.
        Empty parts are skipped; never raises.
        """
        parts: list[str] = []
        history = self.metrics_history(limit=3)
        if len(history) >= 2:
            a, b = history[-2], history[-1]
            parts.append(
                "System metrics (prev -> last): "
                f"tests {a.get('tests', 0)} -> {b.get('tests', 0)}, "
                f"prod lines {a.get('prod_lines', 0)} -> "
                f"{b.get('prod_lines', 0)}, "
                f"tools {a.get('tools', 0)} -> {b.get('tools', 0)}, "
                f"skills {a.get('skills', 0)} -> {b.get('skills', 0)}, "
                f"kg nodes {a.get('kg_nodes', 0)} -> "
                f"{b.get('kg_nodes', 0)}")
        try:
            db = self.context.db
            rows = db.query(
                "SELECT instruction, applied, reason FROM evolution_outcomes "
                "ORDER BY ts DESC LIMIT 5")
            if rows:
                recent = "; ".join(
                    ("applied: " if r["applied"] else "reverted: ") +
                    (r["instruction"][:70])
                    + (f" ({r['reason'][:60]})" if not r["applied"]
                       and r["reason"] else "")
                    for r in rows)
                parts.append("Recent evolution outcomes: " + recent)
        except Exception:  # noqa: BLE001
            pass
        try:
            from .failure import enrich_with_lessons
            prev = enrich_with_lessons(self.context, instruction, limit=3)
            if prev:
                parts.append(prev)
        except Exception:  # noqa: BLE001
            pass
        try:
            from .skills import SkillLibrary
            traps = SkillLibrary(self.context.db).trap_block(limit=3)
            if traps:
                parts.append(traps)
        except Exception:  # noqa: BLE001
            pass
        return "\n\n".join(parts)

    def record_outcome(self, proposal_id: str, *, applied: bool,
                       reason: str = "", commit: str = "", tag: str = "",
                       before: dict[str, Any] | None = None,
                       after: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist what one closed-loop cycle actually did, with the
        before/after metrics.  A reverted cycle is also recorded as a
        failure lesson so the next cycle knows this path was tried."""
        before = before or {}
        after = after or {}
        row_id = f"evo-out-{int(time.time() * 1000)}-{proposal_id[-6:]}"
        try:
            self.context.db.execute(
                "INSERT INTO evolution_outcomes (id, proposal_id, "
                "instruction, source, applied, reverted, reason, "
                "tests_before, tests_after, lines_before, lines_after, "
                "commit_id, tag, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row_id, proposal_id,
                 (getattr(self, "_last_instruction", "") or "")[:300],
                 "", int(applied), int(not applied), reason[:500],
                 int(before.get("tests", 0)), int(after.get("tests", 0)),
                 int(before.get("prod_lines", 0)),
                 int(after.get("prod_lines", 0)),
                 commit, tag, time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("could not record outcome: %s", exc)
        if not applied:
            try:
                from .failure import FailureAnalyzer
                FailureAnalyzer(self.context).record(
                    "evolution", f"reverted proposal {proposal_id}",
                    reason or "evolution change failed the test gate")
            except Exception:  # noqa: BLE001
                pass
        return {"id": row_id, "applied": applied, "before": before,
                "after": after}

    # ── 3. planning ─────────────────────────────────────────────────────────
    def plan(self, instruction: str, *, research_id: str = "",
             focus: str = "") -> EvolutionProposal:
        """Turn an instruction into concrete, reviewable repo edits.

        ``research_id`` seeds the prompt from a stored research report;
        ``focus`` is a comma-separated list of repo-relative files whose
        real contents are put in the model's context (max ~3 KB each).
        """
        instruction = (instruction or "").strip()
        if len(instruction) < 8:
            raise ToolError("evolution needs a real instruction (>= 8 chars)")
        self._last_instruction = instruction
        proposal = EvolutionProposal(
            id=f"evo-{int(time.time() * 1000)}",
            instruction=instruction,
        )
        context_pack = ""
        # the closed loop: what the system measured/learned before this
        # cycle shapes what it proposes now
        outcome_ctx = self._outcome_context(instruction)
        if outcome_ctx:
            context_pack += ("\n\nWhat the system knows from past cycles "
                             "(use it — avoid re-treading reverted paths, "
                             "respect the measured state):\n" + outcome_ctx)
        if research_id:
            report = KVStore(self.context.db).get(f"evolution.research.{research_id}")
            if report:
                try:
                    context_pack += (
                        "\nResearch report on this topic:\n"
                        + json.dumps({
                            k: report.get(k)
                            for k in ("findings", "recommendations")
                            if report.get(k)
                        }, indent=2)[:6000]
                    )
                except (ValueError, TypeError) as e:
                    _log.debug("skipping malformed research report: %s", e)
        if focus:
            parts = [p.strip() for p in focus.split(",") if p.strip()]
            for rel in parts[:6]:
                try:
                    target = (self.repo_root / rel).resolve()
                    target.relative_to(self.repo_root.resolve())
                    if not target.is_file():
                        continue
                    text = target.read_text(encoding="utf-8",
                                            errors="replace")
                    context_pack += (
                        f"\n\nCurrent contents of {rel} "
                        f"(first 3000 chars):\n{text[:3000]}")
                except (OSError, ValueError):
                    continue
        edits, rationale = self._generate_edits(instruction, context_pack)
        if not edits:
            raise ToolError(
                "no concrete edits could be generated — the model is unavailable "
                "or returned nothing usable. Nothing was changed.")
        proposal.edits = edits
        proposal.rationale = rationale
        self._save(proposal)
        _log.info("evolution plan %s: %d edits for %r",
                  proposal.id, len(edits), instruction[:80])
        return proposal

    def _generate_edits(self, instruction: str,
                        context_pack: str = "") -> tuple[list[dict[str, str]], str]:
        """LLM-first, JSON-only edit list.  Every candidate edit is checked
        against the tree before the proposal is accepted."""
        router = getattr(self.context, "router", None)
        if router is None:
            return [], ""
        from ..llm.base import Message, SamplingParams

        system = (
            "You are the Evolver: the self-improvement agent for the NoMorals "
            "framework (this repo). The owner wants a change. Produce EXACT "
            "text-replacement edits, nothing else.\n"
            "Rules:\n"
            "1. Reply with ONLY JSON: {\"rationale\": \"...\", "
            "\"edits\": [{\"path\": \"repo-relative/path.py\", "
            "\"old\": \"exact existing text\", \"new\": \"replacement text\"}]}\n"
            "2. `old` must be an EXACT substring of the current file "
            "(including whitespace) and must occur exactly once — include "
            "enough surrounding lines to be unique.\n"
            "3. Keep edits minimal and surgical. Match the existing code "
            "style. No deletions of unrelated code.\n"
            "4. Never edit .env, git internals, or files under tests/ that "
            "define the safety contract.\n"
            "5. If the request cannot be done with small edits, return "
            "{\"rationale\": \"...\", \"edits\": []}.\n"
            "The change will be applied and the FULL test suite run before "
            "it is kept — if it breaks anything it is reverted."
        )
        # Give the model the repo layout so it can target real files, plus
        # any research/file context the caller gathered.
        layout = self._repo_layout()
        user = (f"Repository layout:\n{layout}\n"
                f"{context_pack}\n\nOwner instruction: {instruction}")
        try:
            response = brain_for(self.context).chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=0.0, max_tokens=3000),
            task_kind="judge")
        except Exception as exc:  # noqa: BLE001 - planning must never crash chat
            _log.warning("evolver LLM call failed: %s", exc)
            return [], ""
        text = getattr(response, "text", "") if getattr(response, "ok", False) else ""
        if not text:
            return [], ""
        data = self._extract_json(text)
        if not data:
            return [], ""
        raw_edits = data.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            return [], str(data.get("rationale") or "")
        cleaned: list[dict[str, str]] = []
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip().lstrip("/")
            old = str(item.get("old") or "")
            new = str(item.get("new") or "")
            if not path or not old:
                continue
            if self._forbidden(path):
                continue
            try:
                target = (self.repo_root / path).resolve()
            except OSError:
                continue
            try:
                target.relative_to(self.repo_root.resolve())
            except ValueError:
                continue  # escapes the repo
            if not target.is_file():
                continue
            content = target.read_text(encoding="utf-8", errors="replace")
            if content.count(old) != 1:
                continue  # must match exactly once
            cleaned.append({"path": path, "old": old, "new": new})
        if not cleaned:
            return [], str(data.get("rationale") or "no valid edits produced")
        return cleaned, str(data.get("rationale") or "")

    def _repo_layout(self) -> str:
        """A compact tree the model can target: py files + key docs."""
        lines: list[str] = []
        for sub in ("nomorals", "tests"):
            base = self.repo_root / sub
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.py")):
                try:
                    rel = path.relative_to(self.repo_root)
                except ValueError:
                    continue
                lines.append(str(rel))
        return "\n".join(lines[:400])

    # ── 2. codebase audit (deterministic, always honest) ───────────────────
    def audit(self) -> dict[str, Any]:
        """Scan the whole framework for weaknesses and missing power.

        Pure heuristics, no model: TODO/FIXME/HACK markers, dead stubs
        (functions that only raise NotImplementedError or pass), modules
        with no matching test file, oversized files, and long functions.
        Each finding carries enough detail to seed a plan directly.
        """
        report: dict[str, Any] = {
            "scanned_at": time.time(),
            "files": 0,
            "lines": 0,
            "markers": [],
            "stubs": [],
            "untested_modules": [],
            "large_files": [],
            "long_functions": [],
            "findings": [],
        }
        py_files: list[Path] = []
        for sub in ("nomorals",):
            base = self.repo_root / sub
            if base.is_dir():
                py_files.extend(p for p in base.rglob("*.py")
                                if "__pycache__" not in p.parts)
        test_names = set()
        tests_base = self.repo_root / "tests"
        if tests_base.is_dir():
            for p in tests_base.rglob("test_*.py"):
                test_names.add(p.stem.replace("test_", ""))
                test_names.add(p.stem)

        marker_re = re.compile(
            r"^\s*#?\s*(TODO|FIXME|XXX|HACK|BUG)\b\s*:?\s*(.*)$")
        stub_re = re.compile(
            r"^\s*raise\s+NotImplementedError\b")

        for path in sorted(py_files):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            rel = str(path.relative_to(self.repo_root))
            report["files"] += 1
            report["lines"] += len(lines)
            if len(lines) > 900:
                report["large_files"].append({"path": rel, "lines": len(lines)})

            # markers + stubs + long functions in one pass
            fn_name = ""
            fn_start = 0
            for i, line in enumerate(lines, 1):
                m = marker_re.match(line)
                if m:
                    report["markers"].append({
                        "path": rel, "line": i, "tag": m.group(1),
                        "text": m.group(2).strip()[:140],
                    })
                mdef = re.match(r"^(\s*)def\s+([A-Za-z_]\w*)\s*\(", line)
                if mdef:
                    # close the previous function's span
                    if fn_name:
                        span = i - fn_start
                        if span > 140:
                            report["long_functions"].append(
                                {"path": rel, "function": fn_name,
                                 "lines": span})
                    fn_name = mdef.group(2)
                    fn_start = i
                if stub_re.search(line):
                    report["stubs"].append({
                        "path": rel, "line": i, "function": fn_name,
                    })
            if fn_name and len(lines) - fn_start + 1 > 140:
                report["long_functions"].append(
                    {"path": rel, "function": fn_name,
                     "lines": len(lines) - fn_start + 1})

            # untested module heuristic: nomorals/<...>/<mod>.py with no
            # tests/test_<mod>.py anywhere
            stem = path.stem
            if stem not in {"__init__", "__main__"} and stem not in test_names:
                report["untested_modules"].append(rel)

        # rank findings: things an engineer would actually fix
        findings: list[dict[str, str]] = []
        for m in report["markers"][:60]:
            if m["tag"] in {"TODO", "FIXME", "HACK"} and m["text"]:
                findings.append({
                    "kind": "marker",
                    "severity": "medium" if m["tag"] == "FIXME" else "low",
                    "where": f"{m['path']}:{m['line']}",
                    "what": f"{m['tag']}: {m['text']}",
                    "suggested": f"resolve the {m['tag']} in {m['path']} line "
                                 f"{m['line']}: {m['text'][:100]}",
                })
        for s in report["stubs"][:40]:
            findings.append({
                "kind": "stub",
                "severity": "high",
                "where": f"{s['path']}:{s['line']}",
                "what": f"dead stub: {s['function']}() raises "
                        f"NotImplementedError",
                "suggested": f"implement {s['function']}() in {s['path']} "
                             f"or remove the stub",
            })
        for u in report["untested_modules"][:40]:
            findings.append({
                "kind": "untested",
                "severity": "medium",
                "where": u,
                "what": "module has no matching test file",
                "suggested": f"add tests/test_{u.split('/')[-1].removesuffix('.py')}"
                             f".py covering {u}",
            })
        for lf in sorted(report["large_files"],
                         key=lambda x: -x["lines"])[:10]:
            findings.append({
                "kind": "large-file",
                "severity": "low",
                "where": lf["path"],
                "what": f"file is {lf['lines']} lines long",
                "suggested": f"consider splitting {lf['path']} (extract the "
                             f"cohesive blocks into sibling modules)",
            })
        order = {"high": 0, "medium": 1, "low": 2}
        findings.sort(key=lambda f: order.get(f["severity"], 3))
        report["findings"] = findings
        report["summary"] = (
            f"{report['files']} files / {report['lines']} lines scanned — "
            f"{len(report['markers'])} markers, {len(report['stubs'])} stubs, "
            f"{len(report['untested_modules'])} untested modules, "
            f"{len(report['large_files'])} large files, "
            f"{len(report['long_functions'])} long functions")
        return report

    # ── 1. deep research ────────────────────────────────────────────────────
    def research(self, topic: str) -> dict[str, Any]:
        """Deep research on anything that could improve the system.

        Fuses (a) live codebase evidence — the audit report plus targeted
        file excerpts around the topic — with (b) expert LLM analysis of
        the state of the art.  The result is a structured report with
        concrete, actionable recommendations, stored for audit and
        directly usable as plan context (`plan(..., research_id=...)`).
        """
        topic = (topic or "").strip()
        if len(topic) < 6:
            raise ToolError("research needs a real topic (>= 6 chars)")
        audit = self.audit()
        evidence = self._topic_evidence(topic, audit)
        history = [
            f"({p.status}) {p.instruction[:100]}"
            for p in self.list(10)
        ]
        router = getattr(self.context, "router", None)
        analysis = ""
        if router is not None:
            from ..llm.base import Message, SamplingParams

            system = (
                "You are a principal engineer researching improvements to "
                "the NoMorals autonomous-agent framework (the codebase "
                "evidence is below). Analyze the state of the art for the "
                "topic and produce concrete, high-quality recommendations "
                "that fit THIS codebase. Be specific: name the modules to "
                "touch, the approach, and the risk. No generic advice.\n"
                "Reply with ONLY JSON: {\"findings\": [\"...\"], "
                "\"recommendations\": [{\"title\": \"...\", "
                "\"targets\": [\"path.py\"], \"approach\": \"...\", "
                "\"risk\": \"low|medium|high\"}]}"
            )
            user = (
                f"Topic: {topic}\n\n"
                f"Codebase evidence:\n{evidence}\n\n"
                f"Recent evolution history:\n"
                + ("\n".join(history) if history else "(none yet)")
            )
            try:
                response = brain_for(self.context).chat(
                    [Message.system(system), Message.user(user)],
                    SamplingParams(temperature=0.2, max_tokens=4000),
                task_kind="judge")
                text = getattr(response, "text", "") \
                    if getattr(response, "ok", False) else ""
                if text:
                    data = self._extract_json(text)
                    if data:
                        analysis = json.dumps(data, indent=2)
            except Exception as exc:  # noqa: BLE001
                _log.warning("research LLM failed: %s", exc)

        recs = []
        if analysis:
            try:
                parsed = json.loads(analysis)
                recs = parsed.get("recommendations") or []
            except (ValueError, TypeError):
                recs = []
        report = {
            "topic": topic,
            "created_at": time.time(),
            "evidence_summary": audit["summary"],
            "findings": [],
            "recommendations": recs if isinstance(recs, list) else [],
            "raw_analysis": analysis,
            "evidence": evidence[:8000],
        }
        if not analysis:
            report["note"] = (
                "no LLM available — codebase evidence only; attach a model "
                "for the analysis half")
        # store for audit + later plan seeding
        try:
            KVStore(self.context.db).set_raw(
                f"evolution.research.{int(time.time() * 1000)}",
                json.dumps(report, default=str), "json")
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not persist research report: %s", exc)
        return report

    def _topic_evidence(self, topic: str, audit: dict[str, Any]) -> str:
        """Codebase excerpts most relevant to a topic."""
        words = [w for w in re.findall(r"[A-Za-z_]{3,}", topic.lower())
                 if w not in {"the", "and", "with", "from", "into", "for",
                              "about", "this", "that", "agent", "system"}]
        hits: dict[str, list[str]] = {}
        for path in (self.repo_root / "nomorals").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            low = text.lower()
            try:
                rel = str(path.relative_to(self.repo_root))
            except ValueError:
                rel = path.name
            # filename matches count double — "scanner.py" IS the scanner
            score = sum(low.count(w) + rel.lower().count(w) * 2 for w in words)
            if score > 2:
                rel = str(path.relative_to(self.repo_root))
                lines = text.splitlines()
                excerpt = "\n".join(
                    f"{i}: {ln}" for i, ln in enumerate(lines[:60], 1))
                hits[rel] = excerpt
        top = sorted(hits.items(), key=lambda kv: -len(kv[0]))[:6]
        rel_lines = [f"== {rel} =\n{body}" for rel, body in top]
        return "\n\n".join(rel_lines)[:12000]

    @staticmethod
    def _forbidden(path: str) -> bool:
        low = path.lower()
        if path in _FORBIDDEN_EXACT:
            return True
        for token in _FORBIDDEN_PATHS:
            if path == token or low.startswith(token + "/"):
                return True
        for token in _FORBIDDEN_SUBSTR:
            if token in low:
                return True
        return False

    # ── verification gate ───────────────────────────────────────────────────
    def verify(self) -> tuple[bool, str]:
        """Quick import smoke + the FULL test suite in a subprocess.

        The suite runs against the WORKING TREE (edits already written),
        so a failing test means the change is bad — full stop.
        """
        # Fast syntax check of the whole tree first: an edit that breaks
        # Python syntax must never reach the (much slower) suite.
        check_dirs = [d for d in ("nomorals", "tests")
                      if (self.repo_root / d).is_dir()]
        if check_dirs:
            compileall = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", *check_dirs],
                cwd=self.repo_root, capture_output=True, text=True,
                timeout=180,
            )
            if compileall.returncode != 0:
                detail = (compileall.stderr or compileall.stdout)[-800:]
                return False, f"syntax check failed:\n{detail}"
        # Full import smoke only when the complete framework is present
        # (a partial repo — e.g. a focused exercise tree — would fail for
        # reasons unrelated to the change).
        if (self.repo_root / "nomorals" / "agents"
            / "partner_runtime.py").is_file():
            smoke = subprocess.run(
                [sys.executable, "-c",
                 "import nomorals.agents.partner_runtime, nomorals.tools.registry;"
                 "from nomorals.tools.registry import ToolRegistry;"
                 "ToolRegistry().register_builtins();print('smoke ok')"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=120,
            )
            if smoke.returncode != 0:
                detail = (smoke.stderr or smoke.stdout)[-800:]
                return False, f"import smoke failed:\n{detail}"
        # Drop stale bytecode first: a .pyc compiled by a previous gate run
        # can mask a broken edit when the rewrite lands in the same second
        # with the same file size (stale pyc → old code runs → false pass).
        import shutil

        for pycache in list(self.repo_root.rglob("__pycache__")):
            shutil.rmtree(pycache, ignore_errors=True)
        suite_cmd = [sys.executable, "-B", "-m", "unittest",
                     "discover", "-s", "tests"]
        if (self.repo_root / "tests" / "__init__.py").is_file():
            suite_cmd += ["-t", "."]
        suite = subprocess.run(
            suite_cmd,
            cwd=self.repo_root, capture_output=True, text=True,
            timeout=_TEST_TIMEOUT,
        )
        # unittest reports its verdict on stderr
        combined = (suite.stderr or "") + (suite.stdout or "")
        tail = combined[-1500:]
        if suite.returncode == 0 and re.search(r"^OK", tail, re.M):
            ran = re.search(r"Ran (\d+) tests?", combined)
            count = int(ran.group(1)) if ran else 0
            if count == 0:
                # a gate that runs zero tests proves nothing
                return False, "test gate invalid: 0 tests were collected"
            return True, tail
        return False, f"test suite failed (exit {suite.returncode}):\n{tail}"

    # ── benchmark regression gate ───────────────────────────────────────────
    _BENCHMARK_KEY = "benchmark.baseline"

    def _benchmark_gate(self) -> dict[str, Any] | None:
        """Run the agent benchmark and compare to the stored baseline.

        Returns None when the gate doesn't apply (disabled, or the active
        provider can't be measured — mock/offline — and an unmeasurable
        benchmark must never block real work).  Otherwise
        ``{"ok": bool, "report": str}``.

        Baseline lifecycle: first measurable run records it; a score
        beyond tolerance below it blocks; a score above it raises it.
        """
        mode = str(getattr(self.context.settings.evolution, "benchmark",
                           "on") or "on").strip().lower()
        if mode == "off":
            return None
        from .benchmark import run_benchmark

        try:
            report = run_benchmark(self.context, limit=2)
        except Exception as exc:  # noqa: BLE001 — the gate must not crash apply
            _log.warning("benchmark gate failed to run: %s", exc)
            return None
        if not report.measurable or report.overall is None:
            return None
        baseline = self._load_benchmark_baseline()
        if baseline is None:
            self._save_benchmark_baseline(report.overall)
            return {"ok": True,
                    "report": f"benchmark baseline recorded: "
                              f"{report.overall:.2f} (4 dimensions)"}
        if report.overall < baseline - _BENCHMARK_TOLERANCE:
            dim_lines = "\n".join(
                f"  {d.name}: "
                + ("n/a" if d.score is None else f"{d.score:.2f}")
                for d in report.scores.values())
            return {"ok": False,
                    "report": (f"benchmark regression: {report.overall:.2f} "
                               f"< baseline {baseline:.2f} "
                               f"(tolerance {_BENCHMARK_TOLERANCE})\n"
                               f"dimensions:\n{dim_lines}")}
        if report.overall > baseline:
            self._save_benchmark_baseline(report.overall)
        return {"ok": True,
                "report": f"benchmark ok: {report.overall:.2f} "
                          f"(baseline {baseline:.2f})"}

    def _load_benchmark_baseline(self) -> float | None:
        db = getattr(self.context, "db", None)
        if db is None:
            return None
        try:
            val = KVStore(db).get(self._BENCHMARK_KEY)
            if val is not None:
                return float(val)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _save_benchmark_baseline(self, score: float) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            KVStore(db).set(self._BENCHMARK_KEY, round(float(score), 4))
        except Exception:  # noqa: BLE001
            _log.debug("benchmark baseline save failed", exc_info=True)

    # ── apply (verified-only) ───────────────────────────────────────────────
    def apply(self, proposal_id: str, *, verify: bool = True,
              commit: bool = False) -> dict[str, Any]:
        proposal = self._load(proposal_id)
        if proposal is None:
            raise ToolError(f"no evolution proposal {proposal_id!r}")
        if proposal.status == "applied":
            raise ToolError(f"proposal {proposal_id} was already applied")
        if not self._git_clean():
            raise ToolError(
                "working tree is dirty — commit or revert current changes "
                "before applying an evolution (the test gate needs a clean "
                "baseline to revert to)")

        # where do the commits land? (config [evolution] work_branch —
        # empty = stay on the current branch, the original behavior)
        self.git.ensure_work_branch()

        # the closed loop measures before it acts — the before-snapshot
        metrics_before = self.measure()
        self._last_instruction = proposal.instruction

        # always-on mid-task reasoning: one thinking pass over what we
        # are about to change — the system shows its work before it acts
        try:
            from .reasoning import ReasoningAgent
            check = ReasoningAgent(self.context).mid_task_check(
                f"apply evolution proposal {proposal_id} "
                f"({len(proposal.edits)} edits)",
                details=proposal.instruction[:300],
                hard_risks=[] if self._git_clean() else
                ["hard risk: working tree is not clean"])
            proposal.verify_result = (
                f"mid-task check: proceed={check['proceed']} "
                f"risks={check['risks'][:3]} advice={check['advice'][:120]}\n"
                + (proposal.verify_result or ""))
        except Exception:  # noqa: BLE001 - thinking is a bonus, not a gate
            pass

        # write the edits
        for edit in proposal.edits:
            target = (self.repo_root / edit["path"]).resolve()
            try:
                target.relative_to(self.repo_root.resolve())
            except ValueError:
                self._revert_tree()
                proposal.status = "rejected"
                proposal.verify_result = f"path escapes repo: {edit['path']}"
                self._save(proposal)
                raise ToolError(f"path escapes repo: {edit['path']}") from None
            if not target.exists():
                if edit["old"] != "":
                    # the plan referenced a file that was never created —
                    # restore the tree (edits already written are rolled
                    # back) and reject, like the stale-plan path above.
                    self._revert_tree()
                    proposal.status = "rejected"
                    proposal.verify_result = (
                        f"edit target missing: {edit['path']}")
                    self._save(proposal)
                    raise ToolError(
                        f"edit target missing: {edit['path']}")
                # new-file edit: create parent dirs, write the content,
                # continue to the next edit (no stale check — the file is
                # ours and the plan is the only author).
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit["new"], encoding="utf-8")
                continue
            content = target.read_text(encoding="utf-8")
            if content.count(edit["old"]) != 1:
                # stale plan: the file changed since planning
                self._revert_tree()
                proposal.status = "rejected"
                proposal.verify_result = (
                    f"edit no longer matches (file changed since plan): "
                    f"{edit['path']}")
                self._save(proposal)
                raise ToolError(
                    "plan is stale — the file changed since it was planned; "
                    "re-plan and retry")
            target.write_text(content.replace(edit["old"], edit["new"]),
                              encoding="utf-8")

        if not verify:
            from .power import power_mode_for

            if not power_mode_for(self.context).active:
                raise ToolError(
                    "skipping the test gate requires power mode — the bot's "
                    "safety contract is that unverified changes never land")
            proposal.status = "applied"
            proposal.verify_result = "SKIPPED (force, power mode)"
            out = self._applied_summary(proposal, commit, skip_verify=True)
            self._save(proposal)
            return out

        proposal.status = "verifying"
        self._save(proposal)
        ok, report = self.verify()
        if not ok:
            self._revert_tree()
            proposal.status = "reverted"
            proposal.verify_result = report
            self._save(proposal)
            self.record_outcome(proposal_id, applied=False,
                                reason="verification failed: " + report[:300],
                                before=metrics_before,
                                after=self.measure())
            return {
                "applied": False,
                "status": "reverted",
                "proposal": proposal_id,
                "reason": "verification failed — working tree restored exactly",
                "report": report,
            }
        # tests passed — now the benchmark regression check. A promotion
        # that makes the system measurably less intelligent is reverted
        # too: the gate verifies correctness AND intelligence. Skipped
        # when off, or when the active provider can't be measured
        # (mock/offline) — an unmeasurable benchmark never blocks.
        bench = self._benchmark_gate()
        if bench is not None and not bench["ok"]:
            self._revert_tree()
            proposal.status = "reverted"
            proposal.verify_result = report + "\n\n" + bench["report"]
            self._save(proposal)
            self.record_outcome(proposal_id, applied=False,
                                reason="benchmark regression: "
                                       + bench["report"][:300],
                                before=metrics_before,
                                after=self.measure())
            return {
                "applied": False,
                "status": "reverted",
                "proposal": proposal_id,
                "reason": ("benchmark regression — the change scores below "
                           "the recorded baseline; working tree restored"),
                "report": bench["report"],
            }
        if bench is not None:
            report = report + "\n\n" + bench["report"]

        # passed: keep it
        proposal.status = "applied"
        proposal.verify_result = report
        out = self._applied_summary(proposal, commit)
        self._save(proposal)
        # the closed loop measures after it acts — the after-snapshot,
        # and the delta is what the next cycle's plan() will see
        self.record_outcome(
            proposal_id, applied=True,
            reason=(report[:300] if bench is None else
                    "verified + benchmark ok"),
            commit=out.get("commit", ""), tag=out.get("tag", ""),
            before=metrics_before, after=self.measure())
        return out

    def _applied_summary(self, proposal: EvolutionProposal, commit: bool,
                         *, skip_verify: bool = False) -> dict[str, Any]:
        out = {
            "applied": True,
            "status": proposal.status,
            "proposal": proposal.id,
            "edits": [e["path"] for e in proposal.edits],
            # the pre-apply file content is gone after this point, so the
            # apply path records chat-sized hunks now — the upgrade digest
            # renders them later from the stored result.
            "hunks": _compact_hunks(proposal.edits),
            "verified": not skip_verify,
        }
        if commit:
            try:
                committed = self.git.commit_and_tag(
                    f"evolve: {proposal.instruction[:120]}",
                    f"evo/{proposal.id}",
                )
                proposal.commit = committed.get("commit", "")
                out["commit"] = proposal.commit
                out["branch"] = committed.get("branch", "")
                if committed.get("tag"):
                    out["tag"] = committed["tag"]
            except (ToolError, OSError, subprocess.SubprocessError) as exc:
                out["commit_error"] = str(exc)
        return out

    # ── 5. clean revert ─────────────────────────────────────────────────────
    def revert(self, proposal_id: str) -> dict[str, Any]:
        """Roll back an applied evolution cleanly.

        Committed evolutions get a `git revert` (a new inverse commit —
        history stays intact).  On-disk (uncommitted) evolutions get a
        file-level checkout of exactly the paths they touched.
        """
        proposal = self._load(proposal_id)
        if proposal is None:
            raise ToolError(f"no evolution proposal {proposal_id!r}")
        if proposal.status != "applied":
            raise ToolError(
                f"proposal {proposal_id} is {proposal.status!r}, not applied")
        out: dict[str, Any] = {"proposal": proposal_id, "method": ""}
        if proposal.commit:
            if not self._git_clean():
                raise ToolError(
                    "working tree is dirty — commit or revert current "
                    "changes before reverting an evolution")
            result = subprocess.run(
                ["git", "revert", "--no-edit", proposal.commit],
                cwd=self.repo_root, capture_output=True, text=True,
                timeout=120)
            if result.returncode == 0:
                tail = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.repo_root, capture_output=True, text=True)
                out["method"] = "git revert"
                out["revert_commit"] = tail.stdout.strip()
            else:
                # conflict or missing commit — fall back to tag-based reset
                tagged = subprocess.run(
                    ["git", "rev-parse", "--verify",
                     f"evo/{proposal.id}"],
                    cwd=self.repo_root, capture_output=True, text=True)
                if tagged.returncode == 0:
                    reset = subprocess.run(
                        ["git", "reset", "--hard",
                         f"evo/{proposal.id}~1"],
                        cwd=self.repo_root, capture_output=True,
                        timeout=120)
                    out["method"] = "git reset to evo/<id>~1"
                    out["ok"] = reset.returncode == 0
                else:
                    raise ToolError(
                        f"git revert failed and no evo tag exists: "
                        f"{(result.stderr or result.stdout)[-400:]}")
        else:
            # on-disk change: restore exactly the touched paths
            for edit in proposal.edits:
                target = (self.repo_root / edit["path"]).resolve()
                try:
                    target.relative_to(self.repo_root.resolve())
                except ValueError:
                    continue
                subprocess.run(["git", "checkout", "--", edit["path"]],
                               cwd=self.repo_root, capture_output=True,
                               timeout=60, check=False)
            out["method"] = "file checkout"
        # the working tree may now differ from what the tests expect —
        # verify the revert kept the system green
        ok, report = self.verify()
        out["verified"] = ok
        if not ok:
            out["report"] = report[-800:]
        proposal.status = "reverted"
        proposal.verify_result = f"manual revert via {out['method']}"
        self._save(proposal)
        out["ok"] = out.get("ok", True) and ok
        return out

    # ── upgrade-queue approval surface ──────────────────────────────────────
    def submit_to_queue(self, proposal_id: str) -> str:
        """File an evolution proposal into the owner's upgrade queue.

        This is THE approve/deny path for proposed framework changes:
        the queue notifies the owner, they approve or deny, and approval
        dispatches back to :meth:`apply`.  Never raises — a proposal that
        cannot be filed is still reviewable via ``evolve_plan``'s return
        and ``EvolutionAgent.list``.
        """
        proposal = self._load(proposal_id)
        if proposal is None:
            raise ToolError(f"no evolution proposal {proposal_id!r}")
        try:
            from .upgrade_queue import UpgradeQueue

            files = [str(e.get("path", "")) for e in (proposal.edits or [])
                     if e.get("path")]
            queue = UpgradeQueue(self.context)
            return queue.propose(
                title=f"evolution: {proposal.instruction[:90]}",
                rationale=(proposal.rationale or proposal.instruction)[:500],
                patch_plan={
                    "source": "evolution",
                    "evolution_proposal_id": proposal.id,
                    "instruction": proposal.instruction[:500],
                    "edit_paths": files,
                },
                files=files,
                tests=["full evolve gate: test suite + benchmark regression"],
                source="evolution",
            )
        except Exception as exc:  # noqa: BLE001 - filing is best-effort
            _log.warning("could not file evolution proposal %s to the "
                         "upgrade queue: %s", proposal_id, exc)
            return ""

    def reject(self, proposal_id: str) -> dict[str, Any]:
        """Mark a proposal rejected (the owner denied it in the upgrade
        queue).  It stays in the record — reverted/failed paths are how
        the next cycle learns — but it can never be applied afterwards."""
        proposal = self._load(proposal_id)
        if proposal is None:
            raise ToolError(f"no evolution proposal {proposal_id!r}")
        if proposal.status in {"applied", "reverted"}:
            raise ToolError(
                f"proposal {proposal_id} is {proposal.status!r} — "
                "revert it instead of rejecting")
        proposal.status = "rejected"
        self._save(proposal)
        return {"proposal": proposal_id, "status": "rejected"}

    # ── goal queue ──────────────────────────────────────────────────────────
    def queue(self, action: str = "list", instruction: str = "") -> list[str]:
        """Owner-queued improvement goals (consumed by autopilot)."""
        action = (action or "list").strip().lower()
        try:
            goals = KVStore(self.context.db).get("evolution.queue", default=[])
            if not isinstance(goals, list):
                goals = []
        except Exception:  # noqa: BLE001
            goals = []
        if action in {"add", "queue"}:
            instruction = (instruction or "").strip()
            if len(instruction) < 8:
                raise ToolError("queue needs a real instruction (>= 8 chars)")
            goals.append(instruction)
        elif action == "clear":
            goals = []
        elif action == "pop":
            goals = goals[1:]
        try:
            KVStore(self.context.db).set("evolution.queue", goals)
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not persist evolution queue: %s", exc)
        return goals

    def pop_goal(self) -> str:
        """Remove and return the next queued goal ('' when empty)."""
        goals = self.queue("list")
        if not goals:
            return ""
        self.queue("pop")
        return str(goals[0])

    # ── 6. autopilot: constantly make the system stronger ──────────────────
    def autopilot(self, max_steps: int = 3) -> dict[str, Any]:
        """Loop: pick the next improvement goal (owner queue first, then
        the top audit finding) → plan → gate → apply → repeat.

        Power mode only.  Every single step still runs the full test
        gate — autopilot can propose at machine speed, but it can never
        ship an unverified change.  Stops after two consecutive
        reverts (stop thrashing) or when no goals/findings remain.
        """
        from .power import power_mode_for

        if not power_mode_for(self.context).active:
            raise ToolError(
                "autopilot requires power mode — in normal mode, queue "
                "goals (/evolve queue add <goal>) and approve each plan "
                "manually")
        max_steps = max(1, min(int(max_steps or 3), 10))
        summary: dict[str, Any] = {
            "steps": 0, "applied": [], "reverted": [], "skipped": [],
            "stopped_reason": "",
        }
        consecutive_failures = 0
        audit_cache: dict[str, Any] | None = None
        for step in range(max_steps):
            summary["steps"] = step + 1
            # 1. what are we improving this step?
            goal = self.pop_goal()
            if goal:
                instruction = goal
                source = "queue"
            else:
                if audit_cache is None:
                    audit_cache = self.audit()
                findings = [f for f in audit_cache["findings"]
                            if f["where"] not in
                            {s.get("where") for s in summary["skipped"]}]
                findings = [f for f in findings
                            if f["kind"] in {"marker", "stub", "untested"}]
                if not findings:
                    summary["stopped_reason"] = "no more actionable findings"
                    break
                finding = findings[0]
                instruction = finding["suggested"]
                source = f"audit:{finding['where']}"
            # 2. plan + gate + apply
            try:
                proposal = self.plan(instruction)
            except ToolError as exc:
                summary["skipped"].append({"where": source, "reason": str(exc)})
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    summary["stopped_reason"] = "two consecutive failures"
                    break
                continue
            out = self.apply(proposal.id, verify=True, commit=True)
            if out.get("applied"):
                consecutive_failures = 0
                summary["applied"].append({
                    "step": step + 1, "source": source,
                    "proposal": proposal.id,
                    "instruction": instruction[:120],
                    "commit": out.get("commit", ""),
                    "tag": out.get("tag", ""),
                })
            else:
                consecutive_failures += 1
                summary["reverted"].append({
                    "step": step + 1, "source": source,
                    "proposal": proposal.id,
                    "instruction": instruction[:120],
                    "reason": (out.get("report") or "")[-300:],
                })
                # the failed edit changed the tree state the audit relies
                # on — rescan next step
                audit_cache = None
                if consecutive_failures >= 2:
                    summary["stopped_reason"] = "two consecutive reverts"
                    break
        if not summary["stopped_reason"]:
            summary["stopped_reason"] = "step budget exhausted"
        return summary

    # ── git safety helpers ──────────────────────────────────────────────────
    def _git_clean(self) -> bool:
        try:
            result = subprocess.run(["git", "status", "--porcelain"],
                                    cwd=self.repo_root, capture_output=True,
                                    text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and not result.stdout.strip()

    def _revert_tree(self) -> None:
        """Restore the pre-change tree exactly (we verified it was clean)."""
        for cmd in (
            ["git", "checkout", "--", "."],
            ["git", "clean", "-fd"],
        ):
            try:
                subprocess.run(cmd, cwd=self.repo_root, capture_output=True,
                               timeout=120, check=False)
            except (OSError, subprocess.SubprocessError):
                _log.warning("evolver revert step failed: %s", " ".join(cmd))

    # ── storage ─────────────────────────────────────────────────────────────
    def _save(self, proposal: EvolutionProposal) -> None:
        try:
            KVStore(self.context.db).set_raw(
                f"evolution.{proposal.id}",
                json.dumps(proposal.to_dict(), default=str), "json",
            )
        except Exception as exc:  # noqa: BLE001 - audit persistence is best-effort
            _log.warning("could not persist evolution proposal: %s", exc)

    def _load(self, proposal_id: str) -> EvolutionProposal | None:
        try:
            data = KVStore(self.context.db).get(f"evolution.{proposal_id}")
        except Exception:  # noqa: BLE001
            return None
        if not data:
            return None
        try:
            return EvolutionProposal.from_dict(data)
        except (ValueError, TypeError):
            return None

    def list(self, limit: int = 10) -> list[EvolutionProposal]:
        # proposal ids are 'evo-…' (plan) — scope to them so the other
        # evolution.* keys (metrics, queue, research) never crowd a
        # proposal out of the limit window
        try:
            # Keys are evo-<timestamp_ms>, so key order == time order.
            # Scan is ASC; reverse for most-recent-first.
            pairs = KVStore(self.context.db).scan("evolution.evo-", limit=limit)
            pairs = list(reversed(pairs))
        except Exception:  # noqa: BLE001
            return []
        out: list[EvolutionProposal] = []
        for _key, data in pairs:
            try:
                if not isinstance(data, dict):
                    continue
                out.append(EvolutionProposal.from_dict(data))
            except Exception:  # noqa: BLE001 - a bad row never kills the list
                continue
        return out

    # ── LLM JSON helper (same discipline as devon's) ────────────────────────
    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        from ..core.jsonutil import extract_json

        return extract_json(text)


# ── tool registration ───────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "evolve_plan",
        description=(
            "Framework self-improvement: turn an instruction into concrete, "
            "reviewable repo edits (validated against the tree). Nothing is "
            "changed yet — apply it with evolve_apply after review. "
            "research_id seeds it from a stored research report; focus puts "
            "specific files' contents in the model's context."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "instruction": "str — what to improve/change in the framework",
            "research_id": "str (optional) — from evolve_research storage key",
            "focus": "str (optional) — comma-separated repo files to read first",
        },
    )
    def evolve_plan(instruction: str, *, research_id: str = "",
                    focus: str = "") -> dict[str, Any]:
        agent = EvolutionAgent(context)
        proposal = agent.plan(
            instruction, research_id=research_id, focus=focus)
        out = proposal.to_dict()
        # the upgrade queue is the owner approve/deny surface for proposed
        # changes — file it there so approval dispatches to evolve_apply
        # and denial marks the proposal rejected.  (Power-mode autopilot
        # calls plan()+apply() directly and never goes through this tool.)
        out["upgrade_proposal_id"] = agent.submit_to_queue(proposal.id)
        return out

    @registry.register(
        "evolve_research",
        description=(
            "Deep research on anything that could improve the system "
            "(tools, agents, performance, new features, better methods): "
            "codebase evidence + expert analysis → structured "
            "recommendations, stored for audit."
        ),
        capability=Capability.FS_READ,
        parameters={"topic": "str — what to research"},
    )
    def evolve_research(topic: str) -> dict[str, Any]:
        report = EvolutionAgent(context).research(topic)
        return {
            "topic": report["topic"],
            "evidence_summary": report["evidence_summary"],
            "recommendations": report["recommendations"][:10],
            "findings": report["findings"][:10],
            "note": report.get("note", ""),
        }

    @registry.register(
        "evolve_audit",
        description=(
            "Scan the whole framework for weaknesses: TODO/FIXME/HACK "
            "markers, dead stubs, untested modules, oversized files, long "
            "functions — ranked, with a suggested fix per finding. "
            "Deterministic: no model, always honest."
        ),
        capability=Capability.FS_READ,
        parameters={"max_findings": "int (optional, 30)"},
    )
    def evolve_audit(*, max_findings: str = "") -> dict[str, Any]:
        try:
            cap = max(1, min(int(max_findings or 30), 200))
        except ValueError:
            cap = 30
        report = EvolutionAgent(context).audit()
        report["findings"] = report["findings"][:cap]
        return report

    @registry.register(
        "evolve_measure",
        description=(
            "Measure the system (the 'measure' half of the closed loop): "
            "fast telemetry snapshot — test count, prod line/file counts, "
            "tool/skill/KG sizes, 24h failure rate, last benchmark score — "
            "plus the bounded history of past snapshots and the recorded "
            "outcomes of recent evolution cycles."
        ),
        capability="memory.read",
        parameters={
            "action": "str — snapshot|history|outcomes",
            "limit": "int — rows to return",
        },
    )
    def evolve_measure(*, action: str = "snapshot", limit: str = "10") -> dict[str, Any]:
        agent = EvolutionAgent(context)
        action = (action or "snapshot").strip().lower()
        try:
            n = max(1, int(limit or 10))
        except ValueError:
            n = 10
        if action == "history":
            return {"history": agent.metrics_history(limit=n)}
        if action == "outcomes":
            try:
                rows = context.db.query(
                    "SELECT proposal_id, instruction, applied, reason, "
                    "tests_before, tests_after, lines_before, lines_after, "
                    "commit_id, tag, ts FROM evolution_outcomes "
                    "ORDER BY ts DESC LIMIT ?", (n,))
                return {"ok": True, "outcomes": [dict(r) for r in rows]}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        return {"ok": True, "snapshot": agent.measure()}

    @registry.register(
        "evolve_revert",
        description=(
            "Cleanly roll back an applied evolution: git revert (committed) "
            "or file checkout (on-disk), then re-runs the test gate to "
            "prove the system is green again."
        ),
        capability=Capability.FS_WRITE,
        parameters={"proposal_id": "str — an applied evo-… id"},
    )
    def evolve_revert(proposal_id: str) -> dict[str, Any]:
        return EvolutionAgent(context).revert(proposal_id)

    @registry.register(
        "evolve_auto",
        description=(
            "Autopilot (power mode): loop audit/queue → plan → FULL test "
            "gate → apply, repeatedly. Every step is gate-verified; two "
            "consecutive reverts stop it. The bot never ships unverified "
            "changes."
        ),
        capability=Capability.FS_WRITE,
        parameters={"steps": "int (optional, 3, max 10)"},
    )
    def evolve_auto(*, steps: str = "") -> dict[str, Any]:
        try:
            steps_i = int(steps or 3)
        except ValueError:
            steps_i = 3
        return EvolutionAgent(context).autopilot(steps_i)

    @registry.register(
        "evolve_git",
        description=(
            "Branch control for evolution commits. status: current branch, "
            "remote, ahead/behind, work/main branch policy. publish: "
            "fast-forward the main branch to the evolution tip (FF-only, "
            "diverged branches are refused) and optionally git push. "
            "Pushing is off unless configured/enabled — it is an owner "
            "action."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — status | publish",
            "branch": "str (optional, with publish) — target branch",
            "push": "bool (optional, with publish) — also git push",
        },
    )
    def evolve_git(*, action: str = "status", branch: str = "",
                   push: str = "") -> dict[str, Any]:
        agent = EvolutionAgent(context)
        action = (action or "status").strip().lower()
        if action == "publish":
            want_push: bool | None = None
            if str(push or "").strip():
                want_push = str(push).lower() in {"1", "true", "yes", "on"}
            return agent.git.publish(branch, push=want_push)
        return agent.git_status()

    @registry.register(
        "evolve_queue",
        description=(
            "Manage the autopilot goal queue: add improvement instructions "
            "the evolver will work through (owner queue is consumed before "
            "audit findings)."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — add | list | clear | pop",
            "instruction": "str (with add) — what to improve",
        },
    )
    def evolve_queue(*, action: str = "list", instruction: str = "") -> dict[str, Any]:
        goals = EvolutionAgent(context).queue(action, instruction)
        return {"goals": goals, "queued": len(goals)}

    @registry.register(
        "evolve_apply",
        description=(
            "Apply an evolution proposal: writes the edits, runs the FULL "
            "test suite, keeps the change only if everything passes "
            "(otherwise reverts automatically). Requires a clean git tree. "
            "verify=false skips the gate — power mode + explicit force only."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "proposal_id": "str — from evolve_plan",
            "commit": "bool (optional) — git commit the verified change",
            "verify": "bool (optional, true) — run the test gate",
        },
    )
    def evolve_apply(proposal_id: str, *, commit: str = "",
                     verify: str = "") -> dict[str, Any]:
        want_commit = str(commit or "").lower() in {"1", "true", "yes", "on"}
        do_verify = str(verify or "").lower() not in {"0", "false", "no", "off"}
        # the power-mode gate for verify=false lives in EvolutionAgent.apply
        return EvolutionAgent(context).apply(proposal_id, verify=do_verify,
                                             commit=want_commit)

    @registry.register(
        "evolve_list",
        description="List recent evolution proposals with their status.",
        capability=Capability.DB_READ,
        parameters={"limit": "int (optional, 10)"},
    )
    def evolve_list(*, limit: str = "") -> dict[str, Any]:
        try:
            limit_i = int(limit or 10)
        except ValueError:
            limit_i = 10
        proposals = EvolutionAgent(context).list(limit_i)
        return {
            "proposals": [
                {
                    "id": p.id,
                    "instruction": p.instruction[:120],
                    "status": p.status,
                    "edits": [e["path"] for e in p.edits],
                    "created_at": p.created_at,
                }
                for p in proposals
            ],
        }
