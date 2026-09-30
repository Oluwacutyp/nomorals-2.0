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


@dataclass
class CodingResult:
    ok: bool
    iterations: int
    files: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "iterations": self.iterations,
            "files": self.files,
            "output": self.output[-2000:],
            "error": self.error[-2000:],
            "seconds": round(self.seconds, 2),
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
    ) -> CodingResult:
        """Run the write->run->fix loop until the acceptance command is green.

        ``seed_code`` (wave 65) pre-loads the file with code from a PREVIOUS
        build step, so multi-step build projects extend the artifact instead
        of rewriting it from scratch each step.
        """
        started = time.perf_counter()
        accept = accept or f'python3 "{filename}"'
        # wave 68: a long/multi-clause build ask is structured into a
        # spec first (objective + steps + criteria) — the draft loop then
        # works from a spec, not from raw prose.  One bounded model call,
        # skipped for short asks, best-effort throughout.
        draft_task = task
        try:
            from .brief import BriefAgent, should_brief

            # gate on the CORE request: an execution prompt that already
            # carries injected prior-art is not a raw ask worth re-wrapping
            if should_brief(core_request(task)):
                brief = BriefAgent(self.context).refine(task, kind="code")
                if brief.by == "model":
                    draft_task = brief.as_goal()
        except Exception:  # noqa: BLE001 — structuring is best-effort
            pass
        current = seed_code or ""
        last_error = ""
        workdir = self._resolve(".")

        for attempt in range(1, max(max_iterations, 1) + 1):
            seeded = bool(seed_code) and attempt == 1 and not last_error
            code = self._draft(draft_task, filename, current, last_error,
                               attempt, seeded=seeded)
            if not code:
                return CodingResult(
                    ok=False,
                    iterations=attempt - 1,
                    error="model returned no code block (check the active provider in /status)",
                    seconds=time.perf_counter() - started,
                )

            # The reasoning pre-flight: a harsh reviewer reads the draft
            # BEFORE it spends a sandbox run; if it finds real flaws the
            # code is revised once. A revision that yields no code block
            # keeps the original draft.
            if attempt == 1:
                code = self._reason_review_code(draft_task, code)

            path = self._resolve(filename)
            path.write_text(code, encoding="utf-8")
            result = self._run(accept, workdir, timeout)
            self._journal(task, filename, attempt, code, result)
            _log.info(
                "coding agent attempt %d: exit=%s (%.2fs)%s",
                attempt,
                result["exit_code"],
                result["seconds"],
                "" if result["exit_code"] == 0 else " — fixing",
            )

            if result["exit_code"] == 0 and not result["timed_out"]:
                # closed loop (wave 66): a session that had to fix
                # itself is prior art — distill the error -> fix trail
                # into a reusable code skill.
                self._distill_session(task, filename, attempt)
                return CodingResult(
                    ok=True,
                    iterations=attempt,
                    files=[filename],
                    output=result["stdout"],
                    seconds=time.perf_counter() - started,
                )
            current = code
            last_error = (result["stderr"] or result["stdout"] or "non-zero exit")[-4000:]

        return CodingResult(
            ok=False,
            iterations=max_iterations,
            error=f"still failing after {max_iterations} attempts: {last_error[-500:]}",
            seconds=time.perf_counter() - started,
        )

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
        """Wave 67 hard recall: deterministic match of the current error
        against distilled session skills; returns a prompt block with the
        proven fix trail(s), or '' when nothing matches.  Best-effort —
        memory must never break a build."""
        try:
            from .skills import SkillLibrary

            hits = SkillLibrary(self.db).match_errors(last_error, limit=3)
            if not hits:
                return ""
            lines = ["\nKNOWN FIX from a previous self-corrected session "
                     "for THIS error (apply the same pattern):\n"]
            for s in hits:
                body = s.body[:500].replace("\n", "\n    ")
                lines.append(f"  skill '{s.name}' (tried {s.uses}x, success "
                             f"{s.success_rate:.0%}): {body}")
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
        from .reasoning import (looks_complex, reasoning_enabled,
                                revise_text, review_text)

        # gate on the CORE request — injected prior-art blocks are not
        # complexity (judging them would double every build's calls)
        core = core_request(task)
        if not reasoning_enabled(self.context,
                                 complex_ok=looks_complex(core)
                                 or len(core) >= 80):
            return code
        flaws = review_text(
            self.context, code,
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
