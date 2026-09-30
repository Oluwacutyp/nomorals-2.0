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
                full_path.write_text(modified, encoding="utf-8")
                
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
        
        full_path.write_text(plan.modified, encoding="utf-8")
        
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
    
    def surgical_replace(self, file_path: str, old_text: str, new_text: str) -> str:
        """Exact-text surgical replacement, no LLM involved.

        The ``old_text`` must occur exactly once in the file; zero or
        multiple matches raise :class:`ValueError` so a sloppy match can
        never silently edit the wrong place. Returns the unified diff.
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
        if self.auto_backup:
            self._backup_file(full_path)
        full_path.write_text(modified, encoding="utf-8")
        _log.info("surgical_replace %s (%d chars changed)", file_path,
                  abs(len(modified) - len(original)))
        return self._generate_diff(original, modified, file_path)

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
        description="Apply a unified diff to the file at path (uses the `patch` binary).",
        capability=Capability.FS_WRITE,
    )
    def apply_patch(path: str, unified_diff: str) -> dict[str, Any]:
        import subprocess as _sp

        target = safe_path(context, path, must_exist=True)
        patch_bin = shutil.which("patch")
        if not patch_bin:
            raise ToolError("`patch` binary not found on PATH")
        root = _workspace_root(context)
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
        return {"path": str(target), "applied": True,
                "output": proc.stdout.strip()[:500]}

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
