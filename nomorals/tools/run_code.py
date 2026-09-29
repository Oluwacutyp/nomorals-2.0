"""Run code tool for executing code snippets."""

from __future__ import annotations

from typing import Any


def run_code(code: str = "", language: str = "python", **kwargs) -> dict[str, Any]:
    """Execute code and return results."""
    result = {
        "code": code,
        "language": language,
        "stdout": "",
        "stderr": "",
        "exit_code": 0
    }
    
    if language == "python":
        try:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            exec(code)
            result["stdout"] = sys.stdout.getvalue()
            sys.stdout = old_stdout
        except Exception as e:
            result["stderr"] = str(e)
            result["exit_code"] = 1
    
    return result


def register(registry: Any) -> None:
    """Register run_code tool with the registry."""
    registry.register(
        "run_code",
        run_code,
        description="Execute code and return results",
        capability="code",
        parameters={
            "code": {"type": "string", "description": "Code to execute"},
            "language": {"type": "string", "description": "Programming language"}
        }
    )
