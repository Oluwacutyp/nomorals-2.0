"""Edit loop - read, diff, apply, test, iterate.

The core workflow for code editing:
1. Read current file
2. Generate diff (planned changes)
3. Apply diff to file
4. Run tests to validate
5. If tests fail, iterate (revert or fix)

Supports:
- Unified diff format
- Search/replace blocks
- Full file rewrites
- Automatic rollback on test failure
- Multi-step edit sequences

Usage:
    editor = EditLoop(agent, project_root="/path/to/repo")
    
    # Simple edit
    result = await editor.edit_file(
        "src/auth.py",
        instruction="Add rate limiting to login endpoint",
        test_command="pytest tests/test_auth.py -x",
    )
    
    # Multi-file edit
    result = await editor.edit_plan(
        instruction="Add caching to all API endpoints",
        files=["src/api/users.py", "src/api/posts.py"],
        test_command="pytest tests/ -x",
    )
    
    # Review diff before applying
    diff = await editor.plan_edit("src/auth.py", "Add JWT validation")
    print(diff)
    if input("Apply? ").lower() == "y":
        await editor.apply_diff("src/auth.py", diff)
"""

from __future__ import annotations

import asyncio
import difflib
import py_compile
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..agents.coding import CodingAgent
from ..core.logging_setup import get_logger

__all__ = [
    "EditLoop",
    "EditResult",
    "EditPlan",
    "TestResult",
    "EditConflictError",
    "EditSyntaxError",
    "DiffApplyError",
    "verify_format_preserved",
    "write_text_verified",
    "apply_unified_diff",
]

_log = get_logger(__name__)


@dataclass
class TestResult:
    """Result of running tests."""
    
    passed: bool
    output: str
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0
    return_code: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "duration": self.duration,
            "return_code": self.return_code,
        }


@dataclass
class EditPlan:
    """A planned edit with diff."""
    
    file_path: str
    original: str
    modified: str
    diff: str
    instruction: str = ""
    explanation: str = ""
    
    @property
    def is_noop(self) -> bool:
        return self.original == self.modified
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "diff": self.diff,
            "instruction": self.instruction,
            "explanation": self.explanation,
            "is_noop": self.is_noop,
        }


@dataclass
class EditResult:
    """Result of an edit operation."""
    
    success: bool
    file_path: str
    diff: str = ""
    test_result: Optional[TestResult] = None
    iterations: int = 0
    error: str = ""
    explanation: str = ""
    backup_path: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "iterations": self.iterations,
            "test_passed": self.test_result.passed if self.test_result else None,
            "error": self.error,
        }


# ── surgical edit machinery (module level, agent-testable) ─────────────────


class EditConflictError(ValueError):
    """An edit could not be applied cleanly: ambiguous, overlapping, or
    invalidated match. The target file is always left untouched."""


class EditSyntaxError(ValueError):
    """An edit produced syntactically invalid Python. The pre-edit content
    has already been restored when this is raised."""




def _locate_unique(text: str, needle: str, *, label: str) -> tuple[int, int]:
    """Return the (start, end) span of ``needle`` in ``text``.

    Raises :class:`EditConflictError` unless ``needle`` occurs exactly once.
    """
    if not needle:
        raise EditConflictError(f"{label}: old_text is empty")
    first = text.find(needle)
    if first < 0:
        raise EditConflictError(f"{label}: old_text not found")
    if text.find(needle, first + 1) >= 0:
        n = text.count(needle)
        raise EditConflictError(
            f"{label}: old_text occurs {n} times; it must be unique — "
            "include more surrounding context"
        )
    return (first, first + len(needle))


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def verify_format_preserved(
    original: str,
    modified: str,
    spans: list[tuple[int, int]],
) -> bool:
    """Check that every byte of ``original`` outside ``spans`` is unchanged.

    ``spans`` are (start, end) offsets into ``original`` covering the replaced
    regions. Every difference between ``original`` and ``modified`` must fall
    inside those spans; untouched regions must be byte-identical. Pure
    insertions are allowed only exactly at a span boundary (i.e. where a
    replacement grew the text).
    """
    merged = _merge_spans(spans)

    def covered(a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in merged)

    matcher = difflib.SequenceMatcher(None, original, modified, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if original[i1:i2] != modified[j1:j2]:
                return False
            continue
        if tag == "insert" and i1 == i2:
            if not any(s == i1 or e == i1 for s, e in merged):
                return False
            continue
        if not covered(i1, i2):
            return False
    return True


def _apply_edits_atomic(
    original: str,
    edits: list[tuple[str, str]],
) -> tuple[str, list[tuple[int, int]]]:
    """Validate and apply several (old_text, new_text) replacements.

    All ``old_text`` blocks are located in ``original`` first: each must occur
    exactly once, and no two spans may overlap. The edits are then applied one
    at a time to an evolving buffer, re-validating each ``old_text`` before
    applying it — so an edit whose match was consumed or duplicated by an
    earlier edit in the same batch is reported with both edit indexes.

    Returns (modified, spans). Raises :class:`EditConflictError` without
    touching anything on any failure (the caller writes only on success, so
    the operation is all-or-nothing).
    """
    if not edits:
        raise EditConflictError("edits must contain at least one (old_text, new_text) pair")

    # Phase 1 — locate every old_text in the ORIGINAL text.
    spans: list[tuple[int, int]] = []
    for i, (old_text, _new_text) in enumerate(edits):
        spans.append(_locate_unique(original, old_text, label=f"edit {i}"))

    # Phase 2 — reject overlapping spans before writing anything.
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    for a, b in zip(order, order[1:]):
        s1, e1 = spans[a]
        s2, e2 = spans[b]
        lo, hi = max(s1, s2), min(e1, e2)
        if lo < hi:
            shared = original[lo:hi]
            raise EditConflictError(
                f"edits {a} and {b} overlap: their old_text spans "
                f"[{s1}:{e1}] and [{s2}:{e2}] share [{lo}:{hi}] "
                f"({shared[:60]!r}) — disambiguate the old_text blocks"
            )

    # Phase 3 — apply sequentially against the evolving buffer.
    buffer = original
    for i, (old_text, new_text) in enumerate(edits):
        count = buffer.count(old_text)
        if count != 1:
            earlier = ", ".join(str(j) for j in range(i)) or "none"
            raise EditConflictError(
                f"edit {i} was invalidated by an earlier edit in the same batch "
                f"(applied edits: {earlier}): its old_text now occurs "
                f"{count} times instead of exactly once"
            )
        buffer = buffer.replace(old_text, new_text, 1)
    return buffer, spans


def _check_python_syntax(path: str | Path) -> None:
    """Raise :class:`EditSyntaxError` if ``path`` is a .py file that no longer
    compiles. Non-Python files are skipped — other grammars are not guessed."""
    path = Path(path)
    if path.suffix.lower() != ".py":
        return
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        exc = getattr(e, "exc_value", None)
        lineno = getattr(exc, "lineno", None) or "?"
        msg = getattr(exc, "msg", None) or str(e).strip().splitlines()[-1]
        raise EditSyntaxError(
            f"{path.name} has invalid Python syntax after edit "
            f"(line {lineno}: {msg})"
        ) from e


def write_text_verified(path: str | Path, original: str, modified: str) -> None:
    """Write ``modified`` to ``path``; if it is a .py file, compile-check the
    result and restore ``original`` on SyntaxError.

    Never leaves a syntactically broken .py behind: on failure the original
    content is written back and :class:`EditSyntaxError` is raised with the
    compile message and line number.
    """
    path = Path(path)
    path.write_text(modified, encoding="utf-8")
    try:
        _check_python_syntax(path)
    except EditSyntaxError:
        path.write_text(original, encoding="utf-8")
        raise


# ── unified diff engine (canonical home: nomorals.core.diff) ───────────────
# Re-exported here so existing ``edit_loop.DiffApplyError`` /
# ``edit_loop.apply_unified_diff`` imports keep working.
from ..core.diff import (
    _DEV_NULL,
    _apply_parsed_diff,
    _parse_unified_diff,
    DiffApplyError,
    apply_unified_diff,
)




class EditLoop:
    """Read → diff → apply → test → iterate loop."""
    
    def __init__(
        self,
        agent: CodingAgent,
        *,
        project_root: str = ".",
        max_iterations: int = 3,
        auto_backup: bool = True,
    ) -> None:
        self.agent = agent
        self.project_root = Path(project_root).resolve()
        self.max_iterations = max_iterations
        self.auto_backup = auto_backup
        _log.info(f"EditLoop initialized for {self.project_root}")
    
    async def edit_file(
        self,
        file_path: str,
        instruction: str,
        *,
        test_command: str = "",
        context_files: list[str] | None = None,
    ) -> EditResult:
        """Edit a file with instruction, optionally testing after.
        
        Args:
            file_path: Relative path to file
            instruction: What to change (natural language)
            test_command: Command to run tests (empty = skip tests)
            context_files: Additional files to provide as context
            
        Returns:
            EditResult with success status and details
        """
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            return EditResult(
                success=False,
                file_path=file_path,
                error=f"File not found: {file_path}",
            )
        
        # Backup original
        backup_path = ""
        if self.auto_backup:
            backup_path = self._backup_file(full_path)
        
        # Read current file
        original = full_path.read_text(encoding="utf-8", errors="ignore")
        
        # Build context
        context = ""
        if context_files:
            for ctx_file in context_files:
                ctx_path = self.project_root / ctx_file
                if ctx_path.exists():
                    context += f"\n\n--- {ctx_file} ---\n{ctx_path.read_text()}"
        
        # Generate edit
        for iteration in range(1, self.max_iterations + 1):
            _log.info(f"Edit iteration {iteration}/{self.max_iterations}")
            
            try:
                # Ask LLM for modified code
                current = full_path.read_text(encoding="utf-8", errors="ignore")
                
                prompt = self._build_edit_prompt(
                    file_path, current, instruction, context
                )
                
                response = self.agent.chat(prompt)
                
                # Extract code from response
                modified = self._extract_code(response.content)
                
                if not modified or modified == current:
                    return EditResult(
                        success=True,
                        file_path=file_path,
                        diff="",
                        iterations=iteration,
                        explanation="No changes needed",
                        backup_path=backup_path,
                    )
                
                # Generate diff
                diff = self._generate_diff(current, modified, file_path)
                
                # Apply changes
                write_text_verified(full_path, current, modified)
                
                # Run tests if specified
                if test_command:
                    test_result = await self.run_tests(test_command)
                    
                    if test_result.passed:
                        return EditResult(
                            success=True,
                            file_path=file_path,
                            diff=diff,
                            test_result=test_result,
                            iterations=iteration,
                            explanation=response.content[:500],
                            backup_path=backup_path,
                        )
                    else:
                        # Tests failed - try to fix
                        _log.warning(f"Tests failed on iteration {iteration}: {test_result.errors}")
                        
                        if iteration >= self.max_iterations:
                            # Rollback
                            if backup_path:
                                self._restore_backup(full_path, backup_path)
                            
                            return EditResult(
                                success=False,
                                file_path=file_path,
                                diff=diff,
                                test_result=test_result,
                                iterations=iteration,
                                error=f"Tests failed after {iteration} iterations",
                                backup_path=backup_path,
                            )
                        
                        # Add test errors to instruction for next iteration
                        instruction = (
                            f"{instruction}\n\n"
                            f"Tests failed with these errors:\n"
                            + "\n".join(test_result.errors[:5])
                        )
                        continue
                else:
                    # No tests, just return success
                    return EditResult(
                        success=True,
                        file_path=file_path,
                        diff=diff,
                        iterations=iteration,
                        explanation=response.content[:500],
                        backup_path=backup_path,
                    )
            
            except Exception as e:
                _log.error(f"Edit iteration {iteration} failed: {e}")
                if iteration >= self.max_iterations:
                    if backup_path:
                        self._restore_backup(full_path, backup_path)
                    return EditResult(
                        success=False,
                        file_path=file_path,
                        iterations=iteration,
                        error=str(e),
                        backup_path=backup_path,
                    )
        
        return EditResult(
            success=False,
            file_path=file_path,
            error="Max iterations reached",
            backup_path=backup_path,
        )
    
    async def plan_edit(
        self,
        file_path: str,
        instruction: str,
        *,
        context_files: list[str] | None = None,
    ) -> EditPlan:
        """Plan an edit without applying it.
        
        Args:
            file_path: Relative path to file
            instruction: What to change
            context_files: Additional files for context
            
        Returns:
            EditPlan with diff preview
        """
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        original = full_path.read_text(encoding="utf-8", errors="ignore")
        
        # Build context
        context = ""
        if context_files:
            for ctx_file in context_files:
                ctx_path = self.project_root / ctx_file
                if ctx_path.exists():
                    context += f"\n\n--- {ctx_file} ---\n{ctx_path.read_text()}"
        
        prompt = self._build_edit_prompt(file_path, original, instruction, context)
        response = self.agent.chat(prompt)
        
        modified = self._extract_code(response.content)
        if not modified:
            modified = original
        
        diff = self._generate_diff(original, modified, file_path)
        
        return EditPlan(
            file_path=file_path,
            original=original,
            modified=modified,
            diff=diff,
            instruction=instruction,
            explanation=response.content[:500],
        )
    
    async def apply_edit(self, plan: EditPlan) -> EditResult:
        """Apply a planned edit."""
        if plan.is_noop:
            return EditResult(
                success=True,
                file_path=plan.file_path,
                explanation="No changes to apply",
            )
        
        full_path = self.project_root / plan.file_path
        backup_path = ""
        
        if self.auto_backup:
            backup_path = self._backup_file(full_path)
        
        original = (
            full_path.read_text(encoding="utf-8", errors="ignore")
            if full_path.exists() else ""
        )
        write_text_verified(full_path, original, plan.modified)
        
        return EditResult(
            success=True,
            file_path=plan.file_path,
            diff=plan.diff,
            explanation=plan.explanation,
            backup_path=backup_path,
        )
    
    async def edit_plan(
        self,
        instruction: str,
        files: list[str],
        *,
        test_command: str = "",
    ) -> list[EditResult]:
        """Edit multiple files as part of a coordinated plan.
        
        Args:
            instruction: Overall instruction
            files: List of files to edit
            test_command: Tests to run after all edits
            
        Returns:
            List of EditResult for each file
        """
        results = []
        backups = {}
        
        try:
            for file_path in files:
                # Backup
                full_path = self.project_root / file_path
                if full_path.exists():
                    backups[file_path] = self._backup_file(full_path)
                
                result = await self.edit_file(
                    file_path,
                    instruction,
                    test_command="",  # Don't test individual files
                )
                results.append(result)
                
                if not result.success:
                    raise RuntimeError(f"Edit failed for {file_path}: {result.error}")
            
            # Run tests on full set
            if test_command:
                test_result = await self.run_tests(test_command)
                
                if not test_result.passed:
                    # Rollback all changes
                    for file_path, backup in backups.items():
                        full_path = self.project_root / file_path
                        self._restore_backup(full_path, backup)
                    
                    results[-1].test_result = test_result
                    results[-1].success = False
                    results[-1].error = "Tests failed after multi-file edit"
        
        except Exception as e:
            _log.error(f"Multi-file edit failed: {e}")
            # Rollback all
            for file_path, backup in backups.items():
                full_path = self.project_root / file_path
                self._restore_backup(full_path, backup)
            
            if results:
                results[-1].success = False
                results[-1].error = str(e)
        
        return results
    
    async def run_tests(self, command: str, *, timeout: int = 120) -> TestResult:
        """Run test command and parse results.
        
        Args:
            command: Test command to run
            timeout: Max seconds to wait
            
        Returns:
            TestResult with pass/fail and errors
        """
        start = time.time()
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            
            output = stdout.decode(errors="ignore") + stderr.decode(errors="ignore")
            duration = time.time() - start
            
            passed = proc.returncode == 0
            
            # Extract errors
            errors = []
            for line in output.splitlines():
                if any(marker in line.lower() for marker in ["error", "fail", "exception", "traceback"]):
                    errors.append(line.strip())
            
            return TestResult(
                passed=passed,
                output=output,
                errors=errors[:10],  # Limit to 10 errors
                duration=duration,
                return_code=proc.returncode,
            )
        
        except asyncio.TimeoutError:
            return TestResult(
                passed=False,
                output=f"Tests timed out after {timeout}s",
                errors=[f"Timeout after {timeout}s"],
                duration=timeout,
                return_code=-1,
            )
        except Exception as e:
            return TestResult(
                passed=False,
                output=str(e),
                errors=[str(e)],
                return_code=-1,
            )
    
    def _build_edit_prompt(
        self,
        file_path: str,
        current_code: str,
        instruction: str,
        context: str,
    ) -> str:
        """Build the prompt for the LLM to generate code."""
        context_block = f"\nADDITIONAL CONTEXT:{context}\n" if context else ""
        return f"""You are editing the file `{file_path}`.

CURRENT FILE CONTENTS:
```
{current_code}
```
{context_block}
INSTRUCTION: {instruction}

Return the COMPLETE modified file contents in a code block. Include ALL code, not just the changed parts.

```
<modified file contents here>
```
"""
    
    def _extract_code(self, response: str) -> str:
        """Extract code block from LLM response."""
        # Look for fenced code blocks
        patterns = [
            r"```(?:python|py|javascript|js|typescript|ts)?\n(.*?)```",
            r"```\n(.*?)```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # No code block found - maybe the whole response is code
        return response.strip()
    
    def surgical_replace(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        *,
        dry_run: bool = False,
    ) -> str:
        """Exact-text surgical replacement, no LLM involved.

        The ``old_text`` must occur exactly once in the file; zero or
        multiple matches raise :class:`ValueError` so a sloppy match can
        never silently edit the wrong place. Returns the unified diff.

        If the file is Python, the result is compile-checked; on SyntaxError
        the original content is restored and :class:`EditSyntaxError` is
        raised — a broken .py is never left behind.

        With ``dry_run=True`` nothing is written (no backup either); the
        diff that *would* be applied is returned.
        """
        full_path = self.project_root / file_path
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        original = full_path.read_text(encoding="utf-8", errors="ignore")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ValueError(f"old_text not found in {file_path}")
        if occurrences > 1:
            raise ValueError(
                f"old_text occurs {occurrences} times in {file_path}; "
                "it must be unique — include more surrounding context"
            )

        modified = original.replace(old_text, new_text, 1)
        diff = self._generate_diff(original, modified, file_path)
        if dry_run:
            return diff
        if self.auto_backup:
            self._backup_file(full_path)
        write_text_verified(full_path, original, modified)
        _log.info("surgical_replace %s (%d chars changed)", file_path,
                  abs(len(modified) - len(original)))
        return diff

    def surgical_replace_many(
        self,
        file_path: str,
        edits: list[tuple[str, str]],
        *,
        dry_run: bool = False,
    ) -> str:
        """Apply several exact-text replacements atomically, no LLM involved.

        Every ``old_text`` must occur exactly once in the file, and no two
        ``old_text`` spans may overlap; each edit is also re-validated against
        the evolving buffer so an edit invalidated by an earlier one in the
        same batch is reported. If ANY validation fails the file is left
        untouched (all-or-nothing). Untouched regions are verified
        byte-identical, and Python results are compile-checked with auto-revert
        on SyntaxError.

        Returns the unified diff of the combined change. With
        ``dry_run=True`` nothing is written; the would-be diff is returned.
        """
        full_path = self.project_root / file_path
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        edits = list(edits)
        original = full_path.read_text(encoding="utf-8", errors="ignore")
        modified, spans = _apply_edits_atomic(original, edits)
        if not verify_format_preserved(original, modified, spans):
            raise EditConflictError(
                f"internal error: format-preservation check failed for "
                f"{file_path}; file left untouched"
            )
        diff = self._generate_diff(original, modified, file_path)
        if dry_run:
            return diff
        if self.auto_backup:
            self._backup_file(full_path)
        write_text_verified(full_path, original, modified)
        _log.info("surgical_replace_many %s (%d edits applied)", file_path, len(edits))
        return diff

    def preview_replace(
        self,
        file_path: str,
        edits: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Preview what :meth:`surgical_replace_many` would do. Writes nothing.

        Returns ``{"diff", "would_change", "matches", "file_path"}`` where
        ``matches`` lists each edit's ``old_text`` with its occurrence count
        and 1-based line number (``None`` when not found).
        """
        full_path = self.project_root / file_path
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        edits = list(edits)
        original = full_path.read_text(encoding="utf-8", errors="ignore")
        modified, _spans = _apply_edits_atomic(original, edits)
        matches = []
        for old_text, _new_text in edits:
            count = original.count(old_text)
            idx = original.find(old_text)
            line = original.count("\n", 0, idx) + 1 if idx >= 0 else None
            matches.append({"old_text": old_text, "count": count, "line": line})
        return {
            "file_path": file_path,
            "diff": self._generate_diff(original, modified, file_path),
            "would_change": modified != original,
            "matches": matches,
        }

    def _generate_diff(self, original: str, modified: str, file_path: str) -> str:
        """Generate unified diff between original and modified."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        
        return "".join(diff)
    
    def _backup_file(self, file_path: Path) -> str:
        """Create backup of file."""
        backup_dir = self.project_root / ".edit_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_name = f"{file_path.name}.{int(time.time())}.bak"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def _restore_backup(self, file_path: Path, backup_path: str) -> None:
        """Restore file from backup."""
        if Path(backup_path).exists():
            shutil.copy2(backup_path, file_path)
            _log.info(f"Restored {file_path} from backup")


# ── registry hook ──────────────────────────────────────────────────────────
# Narrow, safe tools. The LLM-driven planning methods (plan_edit, edit_plan)
# stay inside the agent — what an orchestrator-spawned agent may call is the
# surgical surface below.


def _workspace_root(context: Any) -> Path:
    settings = getattr(context, "settings", None) if context is not None else None
    if settings is not None:
        try:
            return Path(settings.workspace_dir).resolve()
        except Exception:  # noqa: BLE001 — fall back to cwd
            pass
    return Path.cwd().resolve()


def register(registry: Any) -> None:
    """Attach the surgical edit tools to a registry."""
    from ..core.errors import ToolError
    from ..core.policy import Capability
    from .filesystem import safe_path

    context = registry.context

    @registry.register(
        "edit_file",
        description=("Surgical exact-text replacement: old_text must occur exactly "
                     "once in the file. No LLM involved."),
        capability=Capability.FS_WRITE,
    )
    def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        loop = EditLoop(agent=None, project_root=str(_workspace_root(context)))  # type: ignore[arg-type]
        rel = str(target.relative_to(loop.project_root))
        diff = loop.surgical_replace(rel, old_text, new_text)
        return {"path": str(target), "diff": diff}

    @registry.register(
        "apply_patch",
        description=("Apply a unified diff to the file at path. Uses the `patch` "
                     "binary when available, otherwise a built-in pure-Python "
                     "applier. Python results are compile-checked; a syntax "
                     "break reverts the change."),
        capability=Capability.FS_WRITE,
    )
    def apply_patch(path: str, unified_diff: str) -> dict[str, Any]:
        import subprocess as _sp

        target = safe_path(context, path, must_exist=True)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        root = _workspace_root(context)
        rel = str(target.relative_to(root))
        original = target.read_text(encoding="utf-8", errors="ignore")

        def _revert_syntax_break(reason: str) -> None:
            target.write_text(original, encoding="utf-8")
            raise ToolError(f"{reason}; change reverted: {target.name} kept its original content")

        patch_bin = shutil.which("patch")
        if patch_bin:
            with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as fh:
                fh.write(unified_diff)
                diff_file = fh.name
            try:
                proc = _sp.run(
                    [patch_bin, "-p1", "--no-backup-if-mismatch", "-i", diff_file,
                     str(target.relative_to(root))],
                    cwd=str(root), capture_output=True, text=True, timeout=60,
                )
            finally:
                Path(diff_file).unlink(missing_ok=True)
            if proc.returncode != 0:
                raise ToolError(f"patch failed: {(proc.stderr or proc.stdout).strip()[:500]}")
            try:
                _check_python_syntax(target)
            except EditSyntaxError as e:
                _revert_syntax_break(f"patch applied but broke Python syntax ({e})")
            return {"path": str(target), "applied": True,
                    "output": proc.stdout.strip()[:500]}

        # Pure-Python fallback for hosts without the `patch` binary (Termux).
        try:
            patches = _parse_unified_diff(unified_diff)
        except DiffApplyError as e:
            raise ToolError(f"could not parse diff: {e}")
        if not patches:
            raise ToolError("no file sections found in diff")
        texts: dict[str, str | None] = {}
        for fp in patches:
            key = fp.new_path if fp.new_path != "/dev/null" else fp.old_path
            p = root / key
            texts[key] = (
                p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else None
            )
        try:
            results = _apply_parsed_diff(patches, texts)
        except DiffApplyError as e:
            raise ToolError(f"pure-python patch failed: {e}")
        saved: dict[str, str | None] = {}
        touched: list[str] = []
        for key, new_text in results.items():
            p = root / key
            saved[key] = (
                p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else None
            )
            if new_text is None:
                if p.is_file():
                    p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(new_text, encoding="utf-8")
            touched.append(key)
        try:
            for key in touched:
                p = root / key
                if p.is_file():
                    _check_python_syntax(p)
        except EditSyntaxError as e:
            for key, old in saved.items():
                p = root / key
                if old is None:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(old, encoding="utf-8")
            raise ToolError(f"patch applied but broke Python syntax ({e}); all changes reverted")
        if rel not in touched:
            raise ToolError(f"diff did not touch {rel}")
        return {"path": str(target), "applied": True, "files": touched,
                "fallback": "pure-python"}

    @registry.register(
        "show_diff",
        description="Unified diff of a path against its committed git state.",
        capability=Capability.FS_READ,
    )
    def show_diff(path: str) -> dict[str, Any]:
        from .git import git_diff

        target = safe_path(context, path)
        root = _workspace_root(context)
        try:
            rel = str(target.relative_to(root))
        except ValueError:
            rel = str(target)
        return git_diff(None, [rel], str(root))
