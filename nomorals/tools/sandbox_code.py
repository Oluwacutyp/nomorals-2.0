"""Sandboxed code interpreter: write Python, run it, see the result.

The coding agent can *build* programs; this is the day-to-day tool for
*running* them — the AI (or the owner, via ``/py``) writes a snippet, it
executes under the same isolation the rest of the system uses (bwrap →
unshare → rlimit fallback, network off by default, process-group kill on
timeout), and the structured result comes back for the next decision.

Two things on top of plain shell execution:

* **REPL result capture** — the value of the *last expression* is returned
  (``x + 1`` yields ``11``), not only what was printed.
* **Named sessions with persistent state** — variables survive across calls
  (``session="alpha"``), so the interpreter can iterate like a real REPL:
  define in one call, use in the next. State is pickled inside the session
  working directory, which also doubles as the file workspace (everything
  the code writes shows up in ``files`` and survives for the next call).

Python is first-class; other languages go through the ``shell`` tool.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.ids import new_short_id
from ..core.policy import Capability
from .shell import SandboxLimits, detect_backend, run_sandboxed

__all__ = ["CodeInterpreter", "register", "SANDBOX_ROOT"]

#: Sessions live under the NM home; each is an isolated working directory.
SANDBOX_ROOT = Path(os.environ.get("NM_SANDBOX_ROOT", str(Path.home() / ".nomorals" / "sandbox")))

_SAFE_SESSION = re.compile(r"[^A-Za-z0-9._-]+")
MAX_TIMEOUT = 300.0
#: power-mode ceiling — long computations (big data wrangling, training
#: preps) are legitimate work; the cap exists to stop runaway loops, not
#: to ration real computation.
MAX_TIMEOUT_POWER = 1800.0
MAX_OUTPUT = 20_000


def _session_dir(session: str) -> Path:
    name = _SAFE_SESSION.sub("_", (session or "default").strip())[:64] or "default"
    path = SANDBOX_ROOT / "work" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _transform_last_expression(code: str) -> str | None:
    """Rewrite a trailing expression statement into ``__nm_result__ = <expr>``.

    Returns the rewritten source, or None when the code doesn't parse or has
    no trailing expression (in which case there is no REPL value to capture).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return None
    node = tree.body[-1]
    tree.body[-1] = ast.Assign(
        targets=[ast.Name(id="__nm_result__", ctx=ast.Store())],
        value=node.value,
        lineno=node.lineno,
        col_offset=node.col_offset,
    )
    try:
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001 - fall back to raw source, no capture
        return None


def _build_script(user_code: str) -> str:
    """Wrap user code in the state-in / result-out harness.

    Everything runs relative to the working directory (the sandbox binds
    only that directory as writable), so state and the result file never
    escape the session.
    """
    source = user_code or ""
    rewritten = _transform_last_expression(source)
    if rewritten is not None:
        source = rewritten  # ast.unparse round-trips any parseable code
    return (
        "import pickle as _nm_p\n"
        "_nm_ns = {}\n"
        "try:\n"
        "    _nm_ns = _nm_p.loads(open('__nm_state__.pkl', 'rb').read())\n"
        "except Exception:\n"
        "    _nm_ns = {}\n"
        "_nm_ns['__name__'] = '__nm_interpreter__'\n"
        f"exec(compile({source!r}, '<nm-interpreter>', 'exec'), _nm_ns)\n"
        "_nm_has = ('__nm_result__' in _nm_ns)\n"
        "try:\n"
        "    _nm_payload = {'result': repr(_nm_ns['__nm_result__'])[:100000] if _nm_has else None}\n"
        "except Exception:\n"
        "    _nm_payload = {'result': None}\n"
        "import json as _nm_j\n"
        "_nm_j.dump(_nm_payload, open('__nm_result__.json', 'w'))\n"
        "_nm_new = {k: v for k, v in _nm_ns.items() if not k.startswith('_')}\n"
        "try:\n"
        "    _nm_p.dump(_nm_new, open('__nm_state__.pkl', 'wb'))\n"
        "except Exception:\n"
        "    for _nm_k in [k for k, v in _nm_new.items()]:\n"
        "        try:\n"
        "            _nm_p.dumps(v)\n"
        "        except Exception:\n"
        "            del _nm_new[_nm_k]\n"
        "    _nm_p.dump(_nm_new, open('__nm_state__.pkl', 'wb'))\n"
    )


class CodeInterpreter:
    """Stateful, sandboxed Python execution with result capture."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else SANDBOX_ROOT
        self.max_timeout = MAX_TIMEOUT

    def _dir(self, session: str) -> Path:
        name = _SAFE_SESSION.sub("_", (session or "default").strip())[:64] or "default"
        path = self.root / "work" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── public ──────────────────────────────────────────────────────────────
    def run(
        self,
        code: str,
        *,
        language: str = "python",
        session: str = "default",
        timeout: float = 30.0,
        network: bool = False,
        backend: str = "auto",
        reset: bool = False,
    ) -> dict[str, Any]:
        """Execute ``code`` and return a structured observation."""
        language = (language or "python").lower().strip()
        if language not in {"", "python", "py"}:
            return {
                "ok": False, "error": (
                    f"language {language!r} not supported yet — python is first-class; "
                    "use the shell tool for other languages"
                ),
            }
        timeout = max(0.5, min(float(timeout), self.max_timeout))

        workdir = self._dir(session)
        if reset:
            for stale in workdir.glob("*"):
                try:
                    stale.unlink()
                except OSError:  # noqa: E103 - stale file cleanup is best-effort
                    pass
        if not (code or "").strip():
            if reset:
                return {
                    "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
                    "result": None, "files": [], "workdir": str(workdir),
                    "timed_out": False, "seconds": 0.0, "session": session,
                    "backend": "none", "note": "session reset",
                }
            return {"ok": False, "error": "empty code"}
        snippet = workdir / "__snippet__.py"
        snippet.write_text(_build_script(code), encoding="utf-8")
        result_file = workdir / "__nm_result__.json"
        if result_file.exists():
            result_file.unlink()

        started = time.perf_counter()
        try:
            raw = run_sandboxed(
                "python3 -B __snippet__.py",
                cwd=workdir,
                timeout=timeout,
                network=network,
                backend=backend,
                limits=SandboxLimits(
                    cpu_seconds=max(int(timeout) + 5, 10),
                    address_space_mb=4096,   # 4 GiB — a REPL, not a compiler
                    file_size_mb=256,
                    max_processes=32,
                ),
                max_output=MAX_OUTPUT * 4,
            )
        except Exception as exc:  # noqa: BLE001 - sandbox launch failure is a result
            return {"ok": False, "error": f"sandbox failed to start: {exc}", "seconds":
                    round(time.perf_counter() - started, 2)}

        stdout, stderr = raw.get("stdout", ""), raw.get("stderr", "")
        captured: Any = None
        if result_file.exists():
            try:
                captured = json.loads(result_file.read_text(encoding="utf-8")).get("result")
            except Exception:  # noqa: BLE001
                captured = None
        # a trailing expression that evaluated to None comes back as repr(None);
        # the string "None" itself would repr as "'None'", so this is unambiguous
        if captured == "None":
            captured = None

        files = sorted(
            p.name for p in workdir.iterdir()
            if p.is_file() and not p.name.startswith("__")
        )[:50]
        ok = raw.get("exit_code") == 0 and not raw.get("timed_out")
        return {
            "ok": ok,
            "language": "python",
            "session": (session or "default").strip() or "default",
            "exit_code": raw.get("exit_code"),
            "timed_out": bool(raw.get("timed_out")),
            "stdout": (stdout or "")[:MAX_OUTPUT],
            "stderr": (stderr or "")[:MAX_OUTPUT],
            "result": captured,
            "files": files,
            "workdir": str(workdir),
            "backend": raw.get("backend", detect_backend(backend)),
            "seconds": round(raw.get("seconds", time.perf_counter() - started), 2),
        }

    def session_info(self, session: str) -> dict[str, Any]:
        """What a session holds: file count + size (state lives in the dir)."""
        workdir = self._dir(session)
        files = [p for p in workdir.iterdir() if p.is_file() and not p.name.startswith("__")]
        state = workdir / "__nm_state__.pkl"
        return {
            "session": (session or "default").strip() or "default",
            "workdir": str(workdir),
            "files": [p.name for p in files][:50],
            "has_state": state.exists(),
            "state_bytes": state.stat().st_size if state.exists() else 0,
        }


def register(registry: Any) -> None:
    """Register the interpreter with the tool registry."""
    context = registry.context
    interpreter = CodeInterpreter()

    @registry.register(
        "code",
        description=(
            "run Python in the sandbox and get stdout/stderr/exit code plus the "
            "value of the last expression; named sessions keep variables across calls"
        ),
        capability=Capability.EXEC_CODE,
        parameters={
            "code": "str — python source to execute",
            "session": "str (optional, default 'default') — named REPL state",
            "timeout": "float seconds (optional, default 30, max 300)",
            "network": "bool (optional, default False) — allow outbound network",
            "reset": "bool (optional) — clear this session's state and files first",
        },
    )
    def code(
        code: str,
        session: str = "default",
        timeout: float = 30.0,
        network: bool = False,
        reset: bool = False,
    ) -> dict[str, Any]:
        from ..agents.power import power_mode_for

        # power mode lifts the 5-minute ceiling to 30 minutes
        interpreter.max_timeout = (MAX_TIMEOUT_POWER
                                   if power_mode_for(context).active
                                   else MAX_TIMEOUT)
        return interpreter.run(
            code, session=session, timeout=timeout, network=network, reset=reset)

    @registry.register(
        "code_session",
        description="inspect a code-interpreter session (files, persisted state)",
        capability=Capability.EXEC_CODE,
        parameters={"session": "str (optional, default 'default')"},
    )
    def code_session(session: str = "default") -> dict[str, Any]:
        return interpreter.session_info(session)
