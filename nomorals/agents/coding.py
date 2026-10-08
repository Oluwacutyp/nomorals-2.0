"""The coding agent: draft -> run in the sandbox -> read the error -> fix.

This is the "not just a class" version: every step uses the real system.

* **Drafting** goes through the router — whatever model is actually active
  (groq / hf / local / mock), with the previous attempt's code and its exact
  stderr fed back in, so the model fixes the real failure, not a guess.
* **Writing** goes through the filesystem tool's ``safe_path`` — files land
  inside the workspace, traversal is rejected.
* **Running** goes through the sandbox (``run_sandboxed``) — bwrap/unshare/
  rlimit isolation, network off by default, process-group kill on timeout.
* **Every iteration** is journaled to ``coding_log`` so a whole session can
  be replayed or audited later.

The loop stops on a green acceptance run, on a clean model refusal, or when
``max_iterations`` is spent.  With the mock provider the loop fails fast and
loud ("model returned no code") rather than pretending to work.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import LLMResponse, Message, SamplingParams
from ..tools.filesystem import safe_path
from .patch import apply_unified_diff, parse_unified_diff
from .repo_map import build_repo_map

_log = get_logger(__name__)

#: Markers of context that the project layer INJECTS into a build task
#: before the coding agent sees it.  The owner's actual request is what
#: remains after the last injected block — reasoning gates (briefing,
#: draft review) must judge THAT, not the inflated execution prompt.
_SKILL_BLOCK_MARKER = "Relevant proven skills (apply their lessons):"


def core_request(task: str) -> str:
    """The owner's actual request with injected prior-art stripped.

    Projects prepend recalled-skill blocks (and goal knowledge) to the
    build task.  Those make the string long, but they are not new
    complexity: structuring or reviewing them would just spend model
    calls re-wrapping what was already injected.
    """
    t = (task or "").strip()
    idx = t.find(_SKILL_BLOCK_MARKER)
    if idx != -1:
        lines = t[idx:].splitlines()
        end = 1  # skip the marker line itself
        while end < len(lines) and (
                lines[end].startswith("- ") or not lines[end].strip()):
            end += 1
        t = "\n".join(lines[end:]).strip()
    return t

__all__ = ["CodingAgent", "CodingResult", "extract_code_block"]

_CODE_BLOCK_NL = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_CODE_BLOCK_INLINE = re.compile(r"```(?:python|py)?[ \t]+(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Pull the first fenced code block out of a model reply.

    Handles both the normal shape (code on the line after the fence) and the
    lazy one (code starting on the fence line).  Returns "" when there is no
    block — the caller treats that as a refusal, not as an empty file.
    """
    if not text:
        return ""
    for pattern in (_CODE_BLOCK_NL, _CODE_BLOCK_INLINE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip() + "\n"
    return ""


_JSON_BLOCK = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL)


def _parse_json_block(text: str) -> Any:
    """Pull the first fenced (```json) block out of a model reply and parse
    it. Returns None when there is no block or it is not valid JSON."""
    if not text:
        return None
    match = _JSON_BLOCK.search(text)
    raw = match.group(1) if match else text
    try:
        return json.loads(raw.strip())
    except (ValueError, TypeError):
        return None


_PATCH_BLOCK = re.compile(r"```(?:diff|patch)[ \t]*\n(.*?)```", re.DOTALL)


def _parse_patch_block(text: str) -> str | None:
    """Pull the first fenced ```diff (or ```patch) block out of a model
    reply.  Returns the raw unified-diff text, or None when there is no
    usable diff block."""
    if not text:
        return None
    match = _PATCH_BLOCK.search(text)
    if not match:
        return None
    patch_text = match.group(1).strip()
    if not patch_text or "--- " not in patch_text:
        return None
    return patch_text


def _parse_edits_block(text: str) -> list[dict[str, str]] | None:
    """Parse the surgical-edit protocol: ``{"edits": [{old_text, new_text}]}``.

    Returns the edit list (possibly empty), or None when the model gave
    nothing usable.
    """
    data = _parse_json_block(text)
    if not isinstance(data, dict):
        return None
    edits = data.get("edits")
    if not isinstance(edits, list):
        return None
    out: list[dict[str, str]] = []
    for edit in edits:
        if (isinstance(edit, dict)
                and isinstance(edit.get("old_text"), str)
                and isinstance(edit.get("new_text"), str)):
            out.append({"old_text": edit["old_text"],
                        "new_text": edit["new_text"]})
        else:
            return None
    return out


def _bg_progress(status: dict[str, Any]) -> None:
    """Live progress from a background test run (Phase D)."""
    _log.info("background tests: %d passed, %d failed, %d errors "
              "(%.1fs elapsed)", status["passed"], status["failed"],
              status["errors"], status["seconds"])


def _workdir_has_tests(workdir: Any) -> bool:
    """Does this workdir actually contain tests?  A simple single-file
    script (no tests/ dir, no test_*.py) should just run — not go through
    the pytest runner and print "nothing ran"."""
    from pathlib import Path
    root = Path(str(workdir))
    if (root / "tests").is_dir() or (root / "test").is_dir():
        return True
    # test files anywhere in the tree (bounded depth)
    try:
        for pat in ("test_*.py", "*_test.py"):
            for _ in root.glob(pat):
                return True
            for _ in root.glob(f"*/{pat}"):
                return True
    except Exception:  # noqa: BLE001 — unreadable dir, treat as no tests
        pass
    return False


def _substantive_lines(path: Any) -> int:
    """Non-blank, non-comment, non-docstring lines in a file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    in_docstring = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('"""') or s.startswith("'''"):
            # single-line docstring
            if len(s) > 6 and s.endswith(('"""', "'''")):
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if s.startswith("#"):
            continue
        count += 1
    return count


#: minimum substantive lines across changed files to count as an artifact
_MIN_ARTIFACT_LINES = 5


def _unified_diff(before: str, after: str, rel: str) -> str:
    """Unified diff of two file texts ("" when identical)."""
    if before == after:
        return ""
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))


# Phase C: focus line for the diff-review gate — the critic reads the
# full multi-file unified diff, not a single draft.
_DIFF_REVIEW_FOCUS = (
    "a unified diff of the complete multi-file change: does the diff "
    "implement the task, are there wrong operators, indices, or names, "
    "does any hunk break something the diff touches or contradict the "
    "task's acceptance criteria")

def _format_lint_result(lint_res: dict[str, Any]) -> str:
    """Lint violations as fix-loop feedback."""
    lines = ["ruff lint failed:"]
    for viol in lint_res.get("violations", [])[:10]:
        lines.append(f"{viol.get('file')}:{viol.get('line')}:{viol.get('col')}: "
                     f"{viol.get('code')} {viol.get('message')}")
    return "\n".join(lines)


@dataclass
class CodingResult:
    ok: bool
    iterations: int
    files: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""
    seconds: float = 0.0
    # Phase C: diff-review gate verdict — {"passed": bool, "rounds": int,
    # "objections": [str]}.  Empty when the gate never ran.
    review: dict[str, Any] = field(default_factory=dict)
    # Plan mode (item #4): when True the task paused BEFORE any file was
    # touched — the owner must approve plan_text (plan_id) before
    # execute_plan() runs it.
    needs_approval: bool = False
    plan_id: str = ""
    plan_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "iterations": self.iterations,
            "files": self.files,
            "output": self.output[-2000:],
            "error": self.error[-2000:],
            "seconds": round(self.seconds, 2),
            "review": self.review,
            "needs_approval": self.needs_approval,
            "plan_id": self.plan_id,
            "plan_text": self.plan_text,
        }


class CodingAgent:
    """One coding session against the real tools and the active model."""

    def __init__(self, context: Any, *, root: str | None = None) -> None:
        self.context = context
        self.db = context.db
        self.router = context.router
        # Optional explicit project root (used by `nm code --root`). When set,
        # files resolve under it with a containment check instead of the
        # workspace sandbox — so the agent can work on a real repo checkout.
        self._root = Path(root).expanduser().resolve() if root else None
        # Override in tests to isolate the error-recall index.
        self._recall_path: str | Path | None = None
        self._recall_index: Any | None = None
        # Git mission snapshot (snapshot()/rollback()); None when the
        # workdir is not a git repo or no mission is in flight.
        self._mission_snapshot: dict[str, Any] | None = None
        # Plan-mode approved scope: when execute_plan() sets this, patch
        # targets outside the set are rejected (no silent scope creep).
        self._plan_scope: set[str] | None = None
        self._mission_touched: list[str] = []
        # Checkpoint conversation state (item #17): what the agent was
        # working on, serialized into checkpoints and restored by rewind().
        self._last_task: str = ""
        self._last_plan_id: str = ""
        self._last_scope: list[str] = []
        self._last_iterations: int = 0

    def _error_recall(self) -> Any:
        """The Phase C embedding recall index (lazy, cached, best-effort)."""
        if self._recall_index is None:
            from .error_recall import ErrorRecallIndex

            self._recall_index = ErrorRecallIndex(store_path=self._recall_path)
        return self._recall_index

    def _resolve(self, rel: str) -> Path:
        """Resolve a project-relative path, honoring an explicit root."""
        if self._root is not None:
            candidate = (self._root / rel).resolve()
            try:
                candidate.relative_to(self._root)
            except ValueError as exc:
                raise ValueError(f"path {rel!r} escapes project root {self._root}") from exc
            return candidate
        return safe_path(self.context, rel)

    def _substantive_artifact(self, rels: list[str], workdir: Any) -> bool:
        """Did the run leave a real artifact?  Counts non-blank,
        non-comment lines across the changed files — an empty main.py
        (or a few comment lines) is NOT an artifact."""
        from pathlib import Path
        root = Path(str(workdir))
        total = 0
        for rel in rels or []:
            try:
                p = (root / rel) if not str(rel).startswith("/") else Path(rel)
                if p.is_file():
                    total += _substantive_lines(p)
            except Exception:  # noqa: BLE001 - one bad path never hides
                continue
        return total >= _MIN_ARTIFACT_LINES

    def _git(self, *args: str, cwd: Path | None = None,
             timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=cwd or self._resolve("."),
            capture_output=True, text=True, timeout=timeout)

    def _is_git_repo(self, workdir: Path) -> bool:
        try:
            proc = self._git("rev-parse", "--is-inside-work-tree",
                             cwd=workdir)
        except (OSError, subprocess.SubprocessError) as exc:
            _log.debug("snapshot: git probe failed: %s", exc)
            return False
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    # ── git mission rollback ──────────────────────────────────────────
    def snapshot(self) -> dict[str, Any] | None:
        """Snapshot the workdir before a mission starts.

        When the workdir is a git repo with a dirty tree, the dirty state
        is stashed under a mission id; otherwise the current HEAD is
        recorded.  Returns the snapshot dict, or None when the workdir is
        not a git repo (debug-logged, never a crash).
        """
        workdir = self._resolve(".")
        if not self._is_git_repo(workdir):
            _log.debug("snapshot: %s is not a git repo — skipping",
                       workdir)
            self._mission_snapshot = None
            return None
        mission_id = new_short_id("mission")
        head = self._git("rev-parse", "HEAD", cwd=workdir).stdout.strip()
        status = self._git("status", "--porcelain", cwd=workdir).stdout
        dirty = bool(status.strip())
        # Untracked files/dirs present BEFORE the mission: anything
        # untracked that appears later is a mission side effect and gets
        # removed by rollback().
        untracked_before = sorted(
            line[3:].strip().strip('"')
            for line in status.splitlines()
            if line.startswith("??"))
        stashed = False
        if dirty:
            proc = self._git("stash", "push", "-m",
                             f"coding-mission {mission_id}", cwd=workdir)
            if proc.returncode != 0:
                raise RuntimeError(
                    "mission snapshot failed: "
                    f"git stash push: {proc.stderr.strip()}")
            stashed = True
            _log.info("mission %s: stashed dirty workdir", mission_id)
        self._mission_snapshot = {
            "mission_id": mission_id, "head": head,
            "stashed": stashed, "repo": str(workdir),
            "untracked_before": untracked_before,
        }
        self._mission_touched = []
        return self._mission_snapshot

    def rollback(self) -> bool:
        """Restore the pre-mission state captured by snapshot().

        Reverts exactly the files the mission touched (tracked files via
        ``git checkout --``, mission-created new files by deletion), then
        pops the mission stash when one was taken.  Files the mission
        never touched are never modified.  Returns True on success, False
        when any step failed (the failure is logged loudly — callers
        surface it in the mission error).
        """
        snap = self._mission_snapshot
        if not snap:
            _log.debug("rollback: no snapshot — nothing to restore")
            return True
        workdir = Path(snap["repo"])
        ok = True
        for rel in self._mission_touched:
            target = workdir / rel
            try:
                target.resolve().relative_to(workdir.resolve())
            except ValueError:
                _log.error("rollback: %r escapes repo — skipping", rel)
                ok = False
                continue
            try:
                untracked = (self._git("ls-files", "--error-unmatch", rel,
                                      cwd=workdir).returncode != 0)
                if untracked:
                    # Mission-created file: checkout cannot restore it,
                    # so it must be removed for a true rollback.
                    if target.is_file():
                        target.unlink()
                        _log.info("rollback: removed mission-created %s",
                                  rel)
                else:
                    proc = self._git("checkout", "--", rel, cwd=workdir)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr.strip())
            except (OSError, RuntimeError,
                    subprocess.SubprocessError) as exc:
                _log.error("rollback: failed to revert %s: %s", rel, exc)
                ok = False
        if snap.get("stashed"):
            try:
                proc = self._git("stash", "pop", cwd=workdir)
            except (OSError, subprocess.SubprocessError) as exc:
                _log.error("rollback: git stash pop crashed: %s", exc)
                return False
            if proc.returncode != 0:
                _log.error("rollback: git stash pop failed: %s",
                           (proc.stderr.strip() or proc.stdout.strip()))
                return False
            _log.info("mission %s: stash popped", snap.get("mission_id"))
        # Remove untracked files/dirs the mission created as side effects
        # (.edit_backups, __pycache__, …).  Anything untracked BEFORE the
        # mission is in the snapshot set and is never touched.
        try:
            status = self._git("status", "--porcelain",
                               cwd=workdir).stdout
            current = {line[3:].strip().strip('"')
                       for line in status.splitlines()
                       if line.startswith("??")}
            before = set(snap.get("untracked_before", []))
            for rel in sorted(current - before):
                target = workdir / rel
                try:
                    target.resolve().relative_to(workdir.resolve())
                except ValueError:
                    _log.error("rollback: %r escapes repo — skipping", rel)
                    ok = False
                    continue
                try:
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    elif target.is_file() or target.is_symlink():
                        target.unlink()
                    else:
                        continue
                    _log.info("rollback: removed mission side effect %s",
                              rel)
                except OSError as exc:
                    _log.error("rollback: failed to remove %s: %s",
                               rel, exc)
                    ok = False
        except (OSError, subprocess.SubprocessError) as exc:
            _log.error("rollback: untracked cleanup failed: %s", exc)
            ok = False
        self._mission_snapshot = None
        self._mission_touched = []
        return ok

    # ── checkpoints + rewind (item #17) ──────────────────────────────
    def _checkpoint_store(self) -> Any:
        """The disk-backed checkpoint store for this agent's workdir."""
        from .checkpoints import CheckpointStore
        return CheckpointStore(
            settings=getattr(self.context, "settings", None))

    def _convo_state(self) -> dict[str, Any]:
        """Serializable snapshot of what the agent is working on."""
        return {
            "task": self._last_task,
            "plan_id": self._last_plan_id,
            "scope": list(self._last_scope),
            "iterations": self._last_iterations,
            "captured_at": time.time(),
        }

    def _restore_convo(self, convo: dict[str, Any]) -> None:
        """Restore conversation state from a checkpoint snapshot."""
        if not isinstance(convo, dict):
            return
        self._last_task = str(convo.get("task", "") or "")
        self._last_plan_id = str(convo.get("plan_id", "") or "")
        scope = convo.get("scope") or []
        self._last_scope = [str(s) for s in scope] \
            if isinstance(scope, list) else []
        try:
            self._last_iterations = int(convo.get("iterations", 0) or 0)
        except (TypeError, ValueError):
            self._last_iterations = 0

    def checkpoint(self, label: str = "",
                   *, convo: dict[str, Any] | None = None) -> Any:
        """Save a recovery checkpoint of the workdir + conversation state.

        Code is captured with ``git stash create`` (a stash *commit* —
        the user's own ``git stash`` list is never touched); untracked
        file contents are copied into the checkpoint dir.  Never raises:
        when git is unavailable a convo-only checkpoint is stored instead.
        """
        store = self._checkpoint_store()
        workdir = self._resolve(".")
        try:
            ckpt = store.capture(
                workdir, label=label,
                convo=convo if convo is not None else self._convo_state())
            return store.save(ckpt)
        except Exception as exc:  # noqa: BLE001 — checkpoint never kills a run
            _log.warning("checkpoint failed, storing convo-only: %s", exc)
            fallback = store.capture(
                Path("__no_repo__"), label=label,
                convo=self._convo_state())
            fallback.code_captured = False
            fallback.note = f"capture failed ({exc})"
            return store.save(fallback)

    def rewind(self, n: int = 1) -> str:
        """Restore the nth-latest checkpoint (n=1 → latest).

        A ``pre-rewind`` safety checkpoint is saved first, so the rewind
        itself is reversible.  Only working-tree files inside the
        checkpoint's scope are rewritten — HEAD is never moved.  Returns
        a human summary (never raises).
        """
        store = self._checkpoint_store()
        try:
            n = max(1, int(n))
        except (TypeError, ValueError):
            return f"bad checkpoint number {n!r} — /checkpoints to list."
        try:
            target = store.get_nth(n)
        except Exception as exc:  # noqa: BLE001
            return f"could not read checkpoints: {exc}"
        if target is None:
            return "no checkpoints yet — /checkpoint to save one first."
        safety = self.checkpoint(label="pre-rewind")
        summary = store.rewind_to(target)
        self._restore_convo(target.convo)
        return (summary +
                f"\n🛟 safety checkpoint {safety.id} (label 'pre-rewind') "
                "saved — /rewind 1 undoes this rewind.")

    def list_checkpoints(self) -> list[Any]:
        """All checkpoints, newest first. Never raises."""
        try:
            return self._checkpoint_store().list()
        except Exception as exc:  # noqa: BLE001
            _log.debug("list_checkpoints: %s", exc)
            return []

    def _rollback_after_failure(self, changed: list[str]) -> str:
        """Roll back a failed mission that made changes.

        Returns a warning suffix for the mission error ("" when the
        rollback succeeded or there was nothing to roll back).
        """
        if not changed:
            return ""
        self._mission_touched = list(dict.fromkeys(changed))
        try:
            rolled_back = self.rollback()
        except Exception as exc:  # noqa: BLE001 — never mask the mission error
            _log.error("mission rollback crashed: %s", exc)
            return (" | WARNING: mission rollback crashed — "
                    "working tree may be dirty")
        if not rolled_back:
            return (" | WARNING: mission rollback failed — "
                    "working tree may be dirty")
        _log.info("failed mission rolled back %d file(s)", len(changed))
        return ""

    def _explore_reads(
        self, plan: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        """Phase D explore phase: read every planned file in one parallel
        ``call_many`` block (bounded at 8 workers) instead of N sequential
        reads.

        Routes through the registry's ``fs_read`` so explore reads are
        audited like every other tool call.  Falls back to direct reads
        when no registry is available (minimal contexts) or when
        ``fs_read`` cannot resolve a path — e.g. the agent root sits
        outside the tool workspace, or the file exceeds ``fs_read``'s
        byte cap.
        """
        rels = [spec["path"] for spec in plan]
        tools = getattr(self.context, "tools", None)
        call_many = getattr(tools, "call_many", None)
        if call_many is not None:
            outcomes: Any = None
            try:
                outcomes = call_many(
                    [("fs_read", {"path": str(self._resolve(rel))})
                     for rel in rels],
                    max_workers=8,
                )
            except Exception as exc:  # noqa: BLE001 — fall back to direct
                _log.debug("parallel explore reads failed: %s", exc)
            if outcomes is not None:
                texts: dict[str, str | None] = {}
                for rel, outcome in zip(rels, outcomes):
                    content: str | None = None
                    if getattr(outcome, "ok", False):
                        value = outcome.value or {}
                        content = value.get("content")
                    if content is None:
                        p = self._resolve(rel)
                        content = (p.read_text(encoding="utf-8")
                                   if p.is_file() else None)
                    texts[rel] = content
                return texts
        # legacy sequential path (no registry on the context)
        texts = {}
        for spec in plan:
            p = self._resolve(spec["path"])
            texts[spec["path"]] = (
                p.read_text(encoding="utf-8") if p.is_file() else None
            )
        return texts

    def chat(self, prompt: str) -> Any:
        """Simple chat interface for tools like EditLoop and CodeExecutor."""
        return self.router.chat([{"role": "user", "content": prompt}])

    # ── public ──────────────────────────────────────────────────────────────
    def run(
        self,
        task: str,
        *,
        filename: str = "main.py",
        accept: str = "",
        max_iterations: int = 5,
        timeout: float = 60.0,
        seed_code: str = "",
        background_tests: bool = False,
        test_cap_seconds: float = 900.0,
        plan_mode: Any = "auto",
        _plan: list[dict[str, Any]] | None = None,
    ) -> CodingResult:
        """Run the surgical multi-file edit loop (audit Phase B).

        Plan which files change, edit them with exact-text replacement via
        the ``edit_file`` tool surface (never whole-file rewrites of
        existing files), verify with the pytest-aware runner, and gate on
        lint.  An explicitly passed ``accept`` command overrides the test
        runner (the ``--accept`` escape hatch for non-Python / exotic
        cases).  ``seed_code`` pre-loads the default file on round 1.
        ``background_tests`` (Phase D) runs the suite in the background
        while the lint gate runs concurrently, instead of blocking on it;
        ``test_cap_seconds`` caps the background run (default 15 min).

        ``plan_mode`` gates non-trivial work on owner approval:
        ``"auto"`` (default) pauses when the file plan is complex
        (>4 files, >2 new files, or >2 modules); ``True``/``"always"``
        always pauses; ``False``/``"never"`` never pauses.  A paused run
        returns ``CodingResult(needs_approval=True, plan_id=...,
        plan_text=...)`` with NO files touched — approve with
        :meth:`approve_and_execute` or :meth:`execute_plan` after
        :meth:`PlanStore.approve`.
        """
        from ..tools import lint as _lint_mod
        from ..tools import pytest_runner as _pytest_mod
        from ..tools.edit_loop import EditLoop

        started = time.perf_counter()
        use_runner = not accept
        accept = accept or f'python3 "{filename}"'
        # wave 68: a long/multi-clause build ask is structured into a
        # spec first — unchanged from Phase A.
        draft_task = task
        try:
            from .brief import BriefAgent, should_brief

            # gate on the CORE request (see core_request above)
            if should_brief(core_request(task)):
                brief = BriefAgent(self.context).refine(task, kind="code")
                if brief.by == "model":
                    draft_task = brief.as_goal()
        except Exception:  # noqa: BLE001 — structuring is best-effort
            pass

        workdir = self._resolve(".")
        # Item #17: track conversation state for checkpoints.
        self._last_task = task
        self._last_plan_id = ""
        self._last_scope = []
        self._last_iterations = 0
        # Simple scripts (no tests anywhere) just run — the pytest runner's
        # "nothing ran" output confuses more than it helps.
        if use_runner and not _workdir_has_tests(workdir):
            use_runner = False
        # Git mission rollback: snapshot the workdir before any change is
        # made.  A failed snapshot must never kill the mission — it just
        # means rollback() becomes a no-op (loudly logged).
        try:
            self.snapshot()
        except Exception as exc:  # noqa: BLE001
            _log.warning("mission snapshot failed — rollback disabled: %s",
                         exc)
            self._mission_snapshot = None
        editor = EditLoop(agent=None, project_root=str(workdir))

        # ── plan step: which files change and why ──
        plan = _plan if _plan is not None else self._plan_files(
            draft_task, filename, workdir)
        # ── plan-mode gate (item #4): non-trivial plans pause for owner
        # approval BEFORE any file is touched.  _plan is set only by
        # execute_plan() on an already-approved plan, so it skips the gate.
        if _plan is None and plan_mode not in (False, "never", "off", None):
            from .plan_mode import PlanStore, is_complex, render_plan
            mode = str(plan_mode).lower() if not isinstance(
                plan_mode, bool) else ("always" if plan_mode else "never")
            if mode == "always" or (mode == "auto" and is_complex(plan)):
                code_plan = PlanStore.new(draft_task, plan,
                                          approach="", risks=[])
                self._last_plan_id = code_plan.id
                self._last_scope = [str(s.get("path", ""))
                                    for s in plan if s.get("path")]
                return CodingResult(
                    ok=False,
                    iterations=0,
                    needs_approval=True,
                    plan_id=code_plan.id,
                    plan_text=render_plan(code_plan),
                    seconds=round(time.perf_counter() - started, 2),
                )
        self._last_scope = [str(s.get("path", "")) if isinstance(s, dict)
                            else str(s["path"])
                            for s in plan if s.get("path")]
        per_file = max(1, max_iterations // max(1, len(plan)))
        budgets = {spec["path"]: per_file for spec in plan}
        # Phase D: explore phase reads all planned files in one parallel
        # call_many block instead of N sequential reads.
        texts = self._explore_reads(plan)
        file_errors = {spec["path"]: "" for spec in plan}
        changed: list[str] = []
        last_error = ""
        rounds = max(1, max_iterations)
        # Phase C: diff-review gate state.  all_diffs keeps the latest diff
        # per file across attempts so the critic always sees the full
        # change; review_rounds bounds the critic->rework loop at 2.
        all_diffs: dict[str, str] = {}
        review_rounds = 0
        review_objections: list[str] = []
        gate_exhausted = False

        for attempt in range(1, rounds + 1):
            self._last_iterations = attempt  # item #17: checkpoint state
            if not any(v > 0 for v in budgets.values()):
                break  # every file spent its budget — stop, don't re-verify
            diffs: dict[str, str] = {}
            touched: list[str] = []
            for spec in plan:
                rel = spec["path"]
                if budgets[rel] <= 0:
                    continue
                path = self._resolve(rel)
                before = path.read_text(encoding="utf-8") if path.is_file() else ""
                is_new = texts[rel] is None or spec.get("new_file")
                if is_new:
                    # New files may still be drafted whole.
                    seeded = bool(seed_code) and rel == filename and attempt == 1
                    code = self._draft(
                        draft_task, rel,
                        (seed_code if seeded else texts[rel] or ""),
                        file_errors[rel] or last_error, attempt, seeded=seeded)
                    if not code:
                        rb_note = self._rollback_after_failure(changed)
                        return CodingResult(
                            ok=False,
                            iterations=attempt - 1,
                            error=("model returned no code block (check the "
                                   "active provider in /status)") + rb_note,
                            seconds=time.perf_counter() - started,
                        )
                    if attempt == 1:
                        code = self._reason_review_code(draft_task, code)
                    path.write_text(code, encoding="utf-8")
                    texts[rel] = code
                else:
                    change = self._draft_change(
                        draft_task, rel, texts[rel] or "",
                        file_errors[rel] or last_error, attempt, workdir)
                    if change is None:
                        file_errors[rel] = "model returned no usable edits"
                        budgets[rel] -= 1
                        continue
                    kind, payload = change
                    if kind == "patch":
                        # Unified-diff protocol: the model shipped a whole
                        # diff (possibly multi-file); apply it directly.
                        patch_ok, patch_err = self._apply_model_patch(
                            payload, workdir, texts, diffs, touched, changed)
                        budgets[rel] -= 1
                        if not patch_ok:
                            file_errors[rel] = patch_err
                            continue
                        file_errors[rel] = ""
                        # Phase B: the harsh reviewer runs over patch diffs
                        # too, not just surgical-edit diffs.
                        if diffs.get(rel):
                            flaws = self._review_flaws(
                                draft_task, diffs[rel])
                            if flaws:
                                file_errors[rel] = (
                                    "reviewer found flaws in the applied "
                                    "diff: " + "; ".join(flaws))
                        continue
                    edits = payload
                    if not edits:
                        continue  # model judges no change needed
                    apply_errors: list[str] = []
                    for edit in edits:
                        try:
                            editor.surgical_replace(
                                rel, edit["old_text"], edit["new_text"])
                        except (ValueError, FileNotFoundError) as exc:
                            apply_errors.append(str(exc))
                    texts[rel] = (path.read_text(encoding="utf-8")
                                  if path.is_file() else "")
                    if apply_errors:
                        file_errors[rel] = ("edit application failed: "
                                           + "; ".join(apply_errors))
                        budgets[rel] -= 1
                        continue
                    file_errors[rel] = ""
                budgets[rel] -= 1
                after = path.read_text(encoding="utf-8") if path.is_file() else ""
                diff_text = _unified_diff(before, after, rel)
                if not diff_text.strip():
                    continue
                diffs[rel] = diff_text
                touched.append(rel)
                if rel not in changed:
                    changed.append(rel)
                # Phase B: the harsh reviewer runs over EVERY applied diff,
                # not just attempt-1 new-file drafts. Flaws become next
                # round's fix prompt for that file.
                flaws = self._review_flaws(draft_task, diff_text)
                if flaws:
                    file_errors[rel] = (
                        "reviewer found flaws in the applied diff: "
                        + "; ".join(flaws))

            # ── verify ──
            bg_lint_res: Any = None
            if use_runner:
                if background_tests:
                    # Phase D: the suite runs in the background while the
                    # agent does the lint gate concurrently (prep work
                    # instead of blocking); poll surfaces live progress.
                    from .bg_tests import background_run_tests

                    bg = background_run_tests(
                        str(workdir), cap_seconds=test_cap_seconds)
                    if isinstance(bg, dict):
                        tres = bg  # honest skip: nothing matched
                    else:
                        bg_lint_res = _lint_mod.lint(
                            changed or [filename], repo=str(workdir))
                        tres = bg.wait(on_progress=_bg_progress)
                else:
                    tres = _pytest_mod.run_tests(
                        repo=str(workdir),
                        timeout=min(max(timeout * 5.0, 60.0), 600.0))
                green = (tres["ok"] and not tres["failed"]
                         and not tres["errors"])
                verify_out = _pytest_mod.format_test_result(tres)
                verify_err = "" if green else verify_out
            else:
                raw = self._run(accept, workdir, timeout)
                out_text = (raw.get("stdout") or "").strip()
                # Real acceptance: exit 0 AND (meaningful stdout OR a
                # substantive artifact).  An empty main.py that exits 0
                # with no output is FAILURE, not green — never report
                # "0 tests, empty file" as success.
                ran_something = bool(out_text)
                made_artifact = self._substantive_artifact(
                    changed or [filename], workdir)
                green = (raw["exit_code"] == 0 and not raw["timed_out"]
                         and (ran_something or made_artifact))
                verify_out = (raw.get("stdout") or "")[-4000:]
                verify_err = ((raw.get("stderr") or raw.get("stdout")
                               or "non-zero exit"))[-4000:]
                if not green:
                    if raw["exit_code"] == 0 and not raw["timed_out"]:
                        verify_err = (
                            "accept command exited 0 but produced no "
                            "output and no substantive artifact "
                            "(empty/trivial files) — not accepted as "
                            "success")
                        verify_out = (f"empty success rejected:\n{verify_err}")
                    else:
                        verify_out = (f"accept command failed "
                                      f"(exit {raw.get('exit_code')}):\n{verify_err}")
            vresult = {"exit_code": 0 if green else 1, "timed_out": False,
                       "stdout": verify_out, "stderr": verify_err}
            for rel in touched:
                self._journal(task, rel, attempt,
                              diffs.get(rel, "")[:60000], vresult)
            all_diffs.update({rel: d for rel, d in diffs.items() if d.strip()})
            _log.info("coding agent round %d: %s%s", attempt,
                      "green" if green else "failing",
                      "" if green else " — fixing")

            if green:
                # ── lint gate: runs AFTER tests go green; lint failures
                # become fix-iterations exactly like test failures.  In
                # background mode it already ran concurrently with the
                # suite — reuse it instead of running it twice. ──
                lint_res = (bg_lint_res if bg_lint_res is not None
                            else _lint_mod.lint(changed or [filename],
                                                repo=str(workdir)))
                if not lint_res["ruff_installed"]:
                    _log.info("ruff not installed — lint gate skipped "
                              "(honest skip, never a silent pass)")
                elif not lint_res["ok"]:
                    last_error = _format_lint_result(lint_res)
                    for rel in changed:
                        file_errors[rel] = last_error
                    for rel in touched:
                        self._journal(task, rel, attempt,
                                      diffs.get(rel, "")[:60000],
                                      {"exit_code": 1, "timed_out": False,
                                       "stdout": "", "stderr": last_error})
                    continue
                # ── Phase C: diff-review gate.  The critic reads the FULL
                # multi-file diff; its objections become one more targeted
                # fix iteration (max 2 review rounds — never infinite).
                gate_flaws: list[str] = []
                if changed:
                    full_diff = "\n".join(
                        all_diffs.get(rel, "") for rel in changed
                        if all_diffs.get(rel))
                    if full_diff.strip():
                        gate_flaws = self._review_flaws(
                            draft_task, full_diff, focus=_DIFF_REVIEW_FOCUS)
                if gate_flaws:
                    review_objections.extend(gate_flaws)
                    review_rounds += 1
                    if review_rounds < 2:
                        objection_block = (
                            "CODE REVIEW objections — address EACH with a "
                            "minimal edit:\n- " + "\n- ".join(gate_flaws))
                        for rel in changed:
                            file_errors[rel] = objection_block
                            self._journal(
                                task, rel, attempt,
                                all_diffs.get(rel, "")[:60000],
                                {"exit_code": 1, "timed_out": False,
                                 "stdout": "",
                                 "stderr": (f"code-review round "
                                            f"{review_rounds} objections:\n"
                                            f"{objection_block}")})
                        _log.info("code-review round %d found %d flaw(s); "
                                  "reworking", review_rounds, len(gate_flaws))
                        continue
                    # critic never approved and the 2-round budget is spent:
                    # ship the green diff WITH the objections attached —
                    # never a third rework round, never a hang.
                    gate_exhausted = True
                    break
                # closed loop (wave 66): distill per changed file.
                for rel in changed:
                    self._distill_session(task, rel, attempt)
                return CodingResult(
                    ok=True,
                    iterations=attempt,
                    files=changed or [filename],
                    output=verify_out[-2000:],
                    seconds=time.perf_counter() - started,
                    review={"passed": not gate_flaws,
                            "rounds": review_rounds,
                            "objections": review_objections},
                )
            last_error = verify_err
            for rel in touched:
                if not file_errors[rel]:
                    file_errors[rel] = verify_err[-2000:]

        if gate_exhausted:
            # Green diff, but the critic never approved within its 2-round
            # budget: ship it WITH the objections attached.
            for rel in changed:
                self._distill_session(task, rel, attempt)
            return CodingResult(
                ok=True,
                iterations=attempt,
                files=changed or [filename],
                output=verify_out[-2000:],
                seconds=time.perf_counter() - started,
                review={"passed": False,
                        "rounds": review_rounds,
                        "objections": review_objections},
            )
        stuck = [s["path"] for s in plan
                 if budgets[s["path"]] <= 0 and file_errors[s["path"]]]
        error = f"still failing after {rounds} attempts: {last_error[-500:]}"
        if stuck:
            error += f" | files that did not converge: {', '.join(stuck)}"
        # Failed mission after making changes: restore the pre-mission state.
        error += self._rollback_after_failure(changed)
        # Phase C: report-only critic on exhausted runs — the diff is still
        # reviewed, but no rework is scheduled.
        exhausted_objections: list[str] = []
        if changed:
            full_diff = "\n".join(all_diffs.get(rel, "") for rel in changed
                                  if all_diffs.get(rel))
            if full_diff.strip():
                exhausted_objections = self._review_flaws(
                    draft_task, full_diff, focus=_DIFF_REVIEW_FOCUS)
        return CodingResult(
            ok=False,
            iterations=rounds,
            error=error,
            seconds=time.perf_counter() - started,
            review={"passed": not exhausted_objections,
                    "rounds": review_rounds,
                    "objections": review_objections + exhausted_objections},
        )

    # ── plan mode: approve → execute (item #4) ────────────────────────

    def execute_plan(self, plan_id: str) -> CodingResult:
        """Execute an approved plan from :class:`plan_mode.PlanStore`.

        Returns a failed CodingResult (never raises) when the plan is
        unknown or not yet approved.  Execution stays within the approved
        scope: a model patch touching an unplanned file is rejected with
        a re-plan request instead of being silently applied.
        """
        from .plan_mode import PlanStore
        plan = PlanStore.get(plan_id)
        if plan is None:
            return CodingResult(
                ok=False, iterations=0,
                error=f"unknown plan {plan_id!r} — ask for a fresh plan")
        if not plan.approved:
            return CodingResult(
                ok=False, iterations=0,
                error=f"plan {plan_id} not approved — reply 'approve' first")
        specs = [{"path": str(f.get("path", "")),
                  "why": str(f.get("why", "")),
                  "new_file": bool(f.get("new_file"))}
                 for f in plan.files if f.get("path")]
        self._plan_scope = {s["path"] for s in specs}
        # Item #17: checkpoint the pre-execution state (after the approval
        # check) so a bad run rewinds cleanly.  Never kills the run.
        try:
            self.checkpoint(
                label=f"plan:{plan_id}",
                convo={"task": plan.task, "plan_id": plan_id,
                       "scope": sorted(self._plan_scope),
                       "iterations": 0, "captured_at": time.time()},
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("execute_plan: pre-run checkpoint failed: %s", exc)
        try:
            return self.run(plan.task, _plan=specs, plan_mode="never")
        finally:
            self._plan_scope = None

    def approve_and_execute(self, plan_id: str) -> CodingResult:
        """Approve ``plan_id`` and immediately execute it."""
        from .plan_mode import PlanStore
        plan = PlanStore.approve(plan_id)
        if plan is None:
            return CodingResult(
                ok=False, iterations=0,
                error=f"unknown plan {plan_id!r} — ask for a fresh plan")
        return self.execute_plan(plan_id)

    def _plan_files(self, task: str, default: str,
                    workdir: Path) -> list[dict[str, Any]]:
        """Ask the model which files the task touches (Phase B plan step).

        Returns ``[{path, why, new_file}]``. Falls back to the single
        default file when the model gives nothing usable — the loop stays
        single-file then, exactly like Phase A.
        """
        system = (
            "You are a build planner. Decide which files must be created or "
            "modified to complete the task. Respond with EXACTLY ONE fenced "
            "```json block shaped like "
            '{"files": [{"path": "relative/path.py", "why": "one line", '
            '"new_file": false}]} — paths are relative to the project root. '
            "No prose outside the block."
        )
        # Repo map as mission context: the planner sees the real layout
        # (top-level dirs, file purposes, symbol counts) instead of
        # guessing paths.  Best-effort — the mission never breaks over it.
        user = f"Task: {task}"
        try:
            repo_map = build_repo_map(workdir)
        except Exception as exc:  # noqa: BLE001
            _log.debug("repo map failed: %s", exc)
            repo_map = ""
        if repo_map:
            user += ("\n\nRepository map — real layout, prefer these paths "
                     "when choosing files:\n" + repo_map)
        response = self.router.chat(
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        specs = self._default_plan(default, workdir)
        if not getattr(response, "ok", False):
            return specs
        data = _parse_json_block(response.text)
        items = data.get("files") if isinstance(data, dict) else None
        if not items:
            return specs
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items[:12]:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path", "")).strip()
            if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                continue
            try:
                self._resolve(rel)
            except ValueError:
                continue  # escapes the project root
            if rel in seen:
                continue
            seen.add(rel)
            out.append({"path": rel, "why": str(item.get("why", ""))[:200],
                        "new_file": bool(item.get("new_file"))
                        and not (workdir / rel).is_file()})
        return out or specs

    def _default_plan(self, default: str,
                      workdir: Path) -> list[dict[str, Any]]:
        return [{"path": default, "why": "default target (plan step fallback)",
                 "new_file": not (workdir / default).is_file()}]

    def _draft_change(self, task: str, rel: str, current: str,
                      last_error: str, attempt: int,
                      workdir: Path) -> tuple[str, Any] | None:
        """One model call for an existing file; two answer protocols.

        Returns ``("edits", [{old_text, new_text}])`` for the surgical-edit
        protocol, ``("patch", patch_text)`` for a unified diff (the caller
        applies it via :meth:`_apply_model_patch`), or None when the model
        gave nothing usable.
        """
        system = (
            "You are a surgical code editor. Fix the file below with minimal "
            "changes. Answer with EXACTLY ONE fenced block, either:\n"
            "(a) a ```json block shaped like "
            '{"edits": [{"old_text": "<exact text copied verbatim from the file>", '
            '"new_text": "<replacement>"}]} — old_text must appear EXACTLY as '
            "written in the file (copy it verbatim, including whitespace) and "
            "should be unique — include surrounding context lines; or\n"
            "(b) a ```diff block holding a standard unified diff — use this "
            "for multi-file or large changes; paths are relative to the "
            "project root.\n"
            'For (a), if no change is needed, return {"edits": []}. '
            "No prose outside the block."
        )
        user = (f"Task: {task}\n\nFile: {rel}\n\nCurrent content:\n```\n"
                f"{current}\n```\n\nAttempt {attempt}.")
        if last_error:
            user += (f"\n\nThe last round FAILED with this exact output:\n```\n"
                     f"{last_error}\n```\nFix it with minimal edits.")
            # hard mid-loop recall (wave 67), same as _draft
            fixes = self._recall_error_fixes(last_error)
            if fixes:
                user += "\n" + fixes
        elif attempt == 1:
            traps = self._trap_warning()
            if traps:
                user += "\n\n" + traps
        response = self.router.chat(
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        if not getattr(response, "ok", False):
            _log.warning("coding agent: edit-model call failed: %s",
                         getattr(response, "error", "?"))
            return None
        edits = _parse_edits_block(response.text)
        if edits is not None:
            return ("edits", edits)
        patch_text = _parse_patch_block(response.text)
        if patch_text is not None:
            return ("patch", patch_text)
        return None

    def _apply_model_patch(self, patch_text: str, workdir: Path,
                           texts: dict[str, str | None],
                           diffs: dict[str, str], touched: list[str],
                           changed: list[str]) -> tuple[bool, str]:
        """Apply a model-supplied unified diff; refresh texts/diffs/touched.

        Returns ``(True, "")`` on full success, ``(False, reason)`` when the
        patch is unparseable, rejected, or any hunk failed (per-file
        all-or-nothing is enforced by apply_unified_diff).
        """
        try:
            file_patches = parse_unified_diff(patch_text)
        except ValueError as exc:
            return False, f"unparseable unified diff: {exc}"
        if not file_patches:
            return False, "unified diff contained no file patches"
        before: dict[str, str] = {}
        for fp in file_patches:
            try:
                p = self._resolve(fp.target_rel)
            except ValueError as exc:
                return False, f"patch target rejected: {exc}"
            # Plan-mode scope enforcement: an approved plan lists the
            # files the mission may touch. A patch reaching outside that
            # set is not silently applied — the caller re-plans instead.
            if (self._plan_scope is not None
                    and fp.target_rel not in self._plan_scope):
                return False, (
                    f"patch target {fp.target_rel!r} is outside the "
                    "approved plan scope — re-plan instead of silent "
                    "scope creep")
            before[fp.target_rel] = (p.read_text(encoding="utf-8")
                                     if p.is_file() else "")
        try:
            result = apply_unified_diff(patch_text, workdir)
        except ValueError as exc:
            return False, f"patch rejected: {exc}"
        failed = result["failed_hunks"]
        if failed:
            detail = "; ".join(
                f"{h['file']} hunk {h['hunk']}: {h['reason']}"
                for h in failed[:5])
            return False, f"patch application failed: {detail}"
        for target_rel in result["applied_files"]:
            p = self._resolve(target_rel)
            after = p.read_text(encoding="utf-8") if p.is_file() else ""
            texts[target_rel] = after
            diff_text = _unified_diff(before.get(target_rel, ""), after,
                                      target_rel)
            if diff_text.strip():
                diffs[target_rel] = diff_text
                if target_rel not in touched:
                    touched.append(target_rel)
                if target_rel not in changed:
                    changed.append(target_rel)
        return True, ""

    def _review_flaws(self, task: str, text: str,
                      focus: str | None = None) -> list[str]:
        """The harsh reviewer, factored out so Phase B can run it over
        every applied diff — not just attempt-1 new-file drafts."""
        from .reasoning import looks_complex, reasoning_enabled, review_text

        # gate on the CORE request (see core_request above)
        core = core_request(task)
        if not reasoning_enabled(self.context,
                                 complex_ok=looks_complex(core)
                                 or len(core) >= 80):
            return []
        return review_text(
            self.context, text,
            focus=focus or "the applied diff below: undefined names, wrong "
                           "indices, unhandled edge cases, or changes that "
                           "break the task's acceptance criteria") or []

    def sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Recent iterations, newest first (for inspection after the fact)."""
        try:
            return list(self.db.query(
                "SELECT id, task, filename, attempt, exit_code, timed_out, created_at "
                "FROM coding_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ))
        except Exception:  # noqa: BLE001 - journal is best-effort
            return []

    def _distill_session(self, task: str, filename: str,
                         iterations: int) -> None:
        """Turn a self-corrected build session into a reusable code skill
        (wave 66).

        Only sessions that FAILED at least once and then recovered carry
        a lesson.  The skill captures the exact error trail from the
        coding journal (error signature as the skill name, each failed
        attempt's stderr as the body, the final working code referenced
        from the journal), so the NEXT build that hits a similar error
        recalls it and applies the proven fix pattern instead of
        re-deriving it.  Best-effort: learning must never break a build.
        """
        if iterations < 2:
            return  # one-shot success: nothing to learn
        try:
            from .skills import SkillLibrary

            rows = list(self.db.query(
                "SELECT attempt, exit_code, stderr FROM coding_log "
                "WHERE task=? AND filename=? ORDER BY attempt DESC LIMIT 5",
                (task[:500], filename)))  # journal caps task at 500
            rows = list(reversed(rows))
            failed = [r for r in rows if int(r.get("exit_code", 0)) != 0
                      or (r.get("stderr") or "").strip()]
            if not failed:
                return
            # signature = the LAST non-empty line of the first failure —
            # for tracebacks that's the actual error ("ModuleNotFoundError:
            # ..."), not the "Traceback (most recent call last):" header
            lines = [ln.strip() for ln in
                     str(failed[0].get("stderr") or "").splitlines()
                     if ln.strip()]
            sig = " ".join((lines[-1] if lines else "").split())[:60]
            slug = re.sub(r"[^a-z0-9]+", "-", sig.lower()).strip("-")[:28]
            if not slug:
                slug = "runtime-error"
            trail = []
            for r in failed[:4]:
                err = " ".join(str(r.get("stderr") or "").split())[:200]
                trail.append(f"- attempt {r['attempt']} (exit "
                             f"{r['exit_code']}): {err or '(no stderr)'}")
            body = (
                f"Self-corrected build session ({iterations} iterations) "
                f"for task: {task[:200]}\n\nSandbox error trail "
                "(oldest first):\n" + "\n".join(trail) +
                f"\n\nResolving pattern: the final code for {filename} "
                "in the coding_log journal makes these errors go away. "
                "Before re-implementing from scratch on a similar task, "
                "recall this skill and apply the same fix pattern.")
            SkillLibrary(self.db).save(
                f"fix-{slug}", kind="code", body=body[:2000],
                description=f"Error -> fix trail for: {task[:120]}",
                tags=["coding", filename.replace(".py", "").replace(".", "-"),
                      slug.split("-")[0] if slug.split("-") else "error"],
                source="coding_session")
            _log.info("coding session distilled into skill fix-%s", slug)
            # Phase C: index the error signature + trail for embedding
            # recall — a renamed-variable variant of this error should
            # still find the fix.
            try:
                skill = SkillLibrary(self.db).get_by_name(f"fix-{slug}")
                if skill is not None:
                    self._error_recall().index(
                        skill.id, sig + " " + " ".join(trail))
            except Exception as exc:  # noqa: BLE001 — recall is best-effort
                _log.debug("error-recall indexing failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            _log.debug("session distillation failed: %s", exc)

    # ── internals ───────────────────────────────────────────────────────────
    def _draft(
        self, task: str, filename: str, current: str, last_error: str, attempt: int,
        seeded: bool = False,
    ) -> str:
        system = (
            f"You are a coding agent. Write complete, runnable Python for the file "
            f"'{filename}'. Respond with EXACTLY ONE fenced ```python code block "
            "containing the whole file and nothing else — no prose outside the block."
        )
        user = f"Task: {task}\n\nAttempt {attempt}."
        if current:
            if seeded:
                user += (
                    "\n\nExisting code from the previous step (EXTEND it — keep "
                    "what already works, do not start over):\n```\n"
                    f"{current}\n```"
                )
            else:
                user += f"\n\nPrevious code:\n```\n{current}\n```"
        if last_error:
            user += (
                f"\n\nIt was run and FAILED with this exact output:\n```\n{last_error}\n```\n"
                "Fix the code so the run succeeds."
            )
            # hard mid-loop recall (wave 67): a past session already fixed
            # THIS exact error family — inject the proven fix pattern.
            fixes = self._recall_error_fixes(last_error)
            if fixes:
                user += "\n" + fixes
        elif attempt == 1:
            # up-front systemic warning (wave 67): error families the
            # system keeps hitting across builds — avoid them before
            # they happen.
            traps = self._trap_warning()
            if traps:
                user += "\n\n" + traps
        response = self.router.chat(
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        if not getattr(response, "ok", False):
            _log.warning("coding agent: model call failed: %s", getattr(response, "error", "?"))
            return ""
        return extract_code_block(response.text)

    def _recall_error_fixes(self, last_error: str) -> str:
        """Phase C recall: embedding similarity search over distilled
        session skills, with the old deterministic string matcher as a
        fallback.  Returns a prompt block with the proven fix trail(s)
        and similarity scores, or '' when nothing matches.  Best-effort —
        memory must never break a build."""
        try:
            from .skills import SkillLibrary

            library = SkillLibrary(self.db)
            hits = self._error_recall().recall(last_error, top_k=3)
            skills: list[tuple[Any, float]] = []
            for hit in hits:
                skill = library.get(hit["skill_id"])
                if skill is not None and not skill.pruned:
                    skills.append((skill, hit["score"]))
            if not skills:
                # fallback: the wave-67 deterministic matcher
                for s in library.match_errors(last_error, limit=3):
                    skills.append((s, 0.0))
            if not skills:
                return ""
            lines = ["\nKNOWN FIX from a previous self-corrected session "
                     "for a SIMILAR error (apply the same pattern):\n"]
            for s, score in skills:
                body = s.body[:500].replace("\n", "\n    ")
                score_bit = f"similarity {score:.2f}, " if score else ""
                lines.append(f"  skill '{s.name}' ({score_bit}tried {s.uses}x, "
                             f"success {s.success_rate:.0%}): {body}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            _log.debug("error-fix recall failed: %s", exc)
            return ""

    def _trap_warning(self) -> str:
        """Wave 67 up-front systemic warning: 'known traps' accumulated
        across past builds.  Best-effort."""
        try:
            from .skills import SkillLibrary

            return SkillLibrary(self.db).trap_block(limit=4)
        except Exception as exc:  # noqa: BLE001
            _log.debug("trap warning failed: %s", exc)
            return ""

    def _reason_review_code(self, task: str, code: str) -> str:
        """Review the draft before it runs: a harsh reviewer finds real
        bugs (wrong indices, undefined names, missing edge cases); if it
        finds any, the code is revised once. Gated by NM_REASONING_MODE;
        power mode reviews every draft with a relaxed budget."""
        from .reasoning import revise_text

        flaws = self._review_flaws(
            task, code,
            focus="python code that must run without error: undefined "
                  "names, wrong indices, unhandled edge cases the task "
                  "implies")
        if not flaws:
            return code
        _log.info("coding draft review found %d flaw(s); revising", len(flaws))
        revised = revise_text(
            self.context, code, flaws,
            focus="the SAME format: exactly one fenced ```python block "
                  "containing the whole file")
        if revised == code:
            return code
        block = extract_code_block(revised)
        return block if block.strip() else code

    def _run(self, command: str, workdir: Path, timeout: float) -> dict[str, Any]:
        from ..tools.shell import SandboxLimits, run_sandboxed

        # The sandbox strips PATH down to system dirs; make sure the interpreter
        # that runs this process is still reachable (Termux keeps python under
        # its own prefix, not /usr/bin).
        python_dir = str(Path(sys.executable).parent)
        env = {"PATH": f"{python_dir}:/usr/local/bin:/usr/bin:/bin"}
        return run_sandboxed(
            command,
            cwd=workdir,
            timeout=timeout,
            env=env,
            limits=SandboxLimits(cpu_seconds=int(min(max(timeout, 5.0), 600.0))),
        )

    def _journal(
        self, task: str, filename: str, attempt: int, code: str, result: dict[str, Any]
    ) -> None:
        try:
            self.db.execute(
                "INSERT INTO coding_log "
                "(id, task, filename, attempt, exit_code, timed_out, stdout, stderr, code, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_short_id("code"),
                    task[:500],
                    filename,
                    attempt,
                    result["exit_code"] if isinstance(result.get("exit_code"), int) else -1,
                    int(bool(result.get("timed_out"))),
                    (result.get("stdout") or "")[:8000],
                    (result.get("stderr") or "")[:8000],
                    code[:60000],
                    time.time(),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - journaling must never break the loop
            _log.warning("coding_log insert failed: %s", exc)
