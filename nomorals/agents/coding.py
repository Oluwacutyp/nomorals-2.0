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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import LLMResponse, Message, SamplingParams
from ..tools.filesystem import safe_path

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "iterations": self.iterations,
            "files": self.files,
            "output": self.output[-2000:],
            "error": self.error[-2000:],
            "seconds": round(self.seconds, 2),
            "review": self.review,
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
        editor = EditLoop(agent=None, project_root=str(workdir))

        # ── plan step: which files change and why ──
        plan = self._plan_files(draft_task, filename, workdir)
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
                        return CodingResult(
                            ok=False,
                            iterations=attempt - 1,
                            error="model returned no code block (check the active provider in /status)",
                            seconds=time.perf_counter() - started,
                        )
                    if attempt == 1:
                        code = self._reason_review_code(draft_task, code)
                    path.write_text(code, encoding="utf-8")
                    texts[rel] = code
                else:
                    edits = self._draft_edits(
                        draft_task, rel, texts[rel] or "",
                        file_errors[rel] or last_error, attempt)
                    if edits is None:
                        file_errors[rel] = "model returned no usable edits"
                        budgets[rel] -= 1
                        continue
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
                green = raw["exit_code"] == 0 and not raw["timed_out"]
                verify_out = (raw.get("stdout") or "")[-4000:]
                verify_err = ((raw.get("stderr") or raw.get("stdout")
                               or "non-zero exit"))[-4000:]
                if not green:
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
        response = self.router.chat(
            [Message.system(system), Message.user(f"Task: {task}")],
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

    def _draft_edits(self, task: str, rel: str, current: str,
                     last_error: str, attempt: int) -> list[dict[str, str]] | None:
        """Ask the model for surgical edits to an existing file.

        Returns a list of ``{old_text, new_text}`` (possibly empty when the
        model judges no change is needed), or None when the model gave
        nothing usable.
        """
        system = (
            "You are a surgical code editor. Fix the file below with minimal "
            "exact-text replacements. Respond with EXACTLY ONE fenced ```json "
            "block shaped like "
            '{"edits": [{"old_text": "<exact text copied verbatim from the file>", '
            '"new_text": "<replacement>"}]}. old_text must appear EXACTLY as '
            "written in the file (copy it verbatim, including whitespace) and "
            "should be unique — include surrounding context lines. Change "
            "only what the task needs. If no change is needed, return "
            '{"edits": []}. No prose outside the block.'
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
        return _parse_edits_block(response.text)

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
