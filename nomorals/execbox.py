"""Execution system — run code in many languages, safely, with real output.

:class:`CodeRunner` extends the sandbox the whole system already uses
(``bwrap → unshare → rlimit`` fallback, network off by default,
process-group kill on timeout) to a **language registry**:

* **interpreted** — python, node/js, bash/sh, ruby, perl, php, lua,
  tclsh, awk, julia, deno, bun (whichever binaries exist on the box)
* **compiled** — c (gcc), c++ (g++), rust (rustc), go (go) — compiled
  into the session workdir, then run

Every run gets its own workdir (``workspace/runs/<stamp>/``), so files the
code writes are captured and reported back in ``files``.  Results are
structured: stdout, stderr, exit code, wall time, sandbox backend, and a
``timed_out`` flag — no "did it work?" guesswork.

    from nomorals.execbox import CodeRunner
    box = CodeRunner(context)
    box.languages()                      # what's actually available here
    r = box.run("print(sum(range(10)))", lang="python")
    r = box.run("int main(){puts(\"ok\");return 0;}", lang="c")
    r = box.run("#!/usr/bin/env bash\necho hi", lang="bash")

Registered as the ``run_code`` tool.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm.brain import brain_for
from .core.errors import ToolError
from .core.logging_setup import get_logger
from .core.policy import Capability

_log = get_logger(__name__)

__all__ = ["CodeRunner", "Language", "register"]

MAX_TIMEOUT = 300.0
MAX_OUTPUT = 50_000

#: name → (binary candidates, needs compile?, compile command template,
#: run command template, default extension, shebangs)
_LANGS: dict[str, dict[str, Any]] = {
    "python": {"bins": ("python3", "python"), "ext": "py",
               "shebang": ("python",)},
    "javascript": {"bins": ("node",), "ext": "js", "shebang": ("node",)},
    "bash": {"bins": ("bash", "sh"), "ext": "sh", "shebang": ("bash", "sh")},
    "ruby": {"bins": ("ruby",), "ext": "rb", "shebang": ("ruby",)},
    "perl": {"bins": ("perl",), "ext": "pl", "shebang": ("perl",)},
    "php": {"bins": ("php",), "ext": "php", "shebang": ("php",)},
    "lua": {"bins": ("lua", "lua5.4", "lua5.3"), "ext": "lua",
            "shebang": ("lua",)},
    "tcl": {"bins": ("tclsh",), "ext": "tcl", "shebang": ("tclsh",)},
    "awk": {"bins": ("awk", "gawk", "mawk"), "ext": "awk", "shebang": ("awk",)},
    "julia": {"bins": ("julia",), "ext": "jl", "shebang": ("julia",)},
    "deno": {"bins": ("deno",), "ext": "ts", "shebang": ("deno",)},
    "bun": {"bins": ("bun",), "ext": "ts", "shebang": ("bun",)},
    "c": {"bins": ("gcc", "cc", "clang"), "ext": "c", "shebang": ("c",),
          "compile": "{bin} -O1 -o {work}/prog {src}",
          "run": "{work}/prog"},
    "cpp": {"bins": ("g++", "clang++"), "ext": "cpp",
            "shebang": ("g++", "c++"),
            "compile": "{bin} -O1 -o {work}/prog {src}",
            "run": "{work}/prog"},
    "rust": {"bins": ("rustc",), "ext": "rs", "shebang": ("rustc",),
             "compile": "{bin} -O -o {work}/prog {src}",
             "run": "{work}/prog"},
    "go": {"bins": ("go",), "ext": "go", "shebang": ("go",),
           "compile": "cd {work} && {bin} build -o prog {src}",
           "run": "cd {work} && ./prog"},
}

_EXT_TO_LANG = {"py": "python", "js": "javascript", "mjs": "javascript",
                "ts": "deno", "sh": "bash", "bash": "bash", "rb": "ruby",
                "pl": "perl", "pm": "perl", "php": "php", "lua": "lua",
                "tcl": "tcl", "awk": "awk", "jl": "julia", "c": "c",
                "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "rs": "rust",
                "go": "go"}


@dataclass
class Language:
    name: str
    binary: str = ""
    available: bool = False
    compiles: bool = False
    extensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "binary": self.binary,
                "available": self.available, "compiles": self.compiles,
                "extensions": list(self.extensions)}


class CodeRunner:
    """Multi-language sandboxed execution."""

    role = "runner"

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── capability discovery ──────────────────────────────────────────────
    def languages(self) -> dict[str, Language]:
        out: dict[str, Language] = {}
        for name, spec in _LANGS.items():
            binary = ""
            for cand in spec["bins"]:
                binary = shutil.which(cand) or ""
                if binary:
                    break
            out[name] = Language(
                name=name, binary=binary, available=bool(binary),
                compiles="compile" in spec,
                extensions=(spec.get("ext", ""),))
        return out

    def detect_language(self, code: str = "", filename: str = "") -> str:
        """lang by extension, shebang, or heuristic."""
        fname = (filename or "").strip()
        if fname:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in _EXT_TO_LANG:
                return _EXT_TO_LANG[ext]
        first = (code or "").strip().splitlines()[:1]
        if first and first[0].startswith("#!"):
            shebang = first[0][2:].strip().split()
            prog = os.path.basename(shebang[0]) if shebang else ""
            for name, spec in _LANGS.items():
                if prog in spec.get("shebang", ()):
                    return name
        lowered = (code or "").lower()
        if re.search(r"\b(def |import |print\()|open\(", lowered):
            return "python"
        if re.search(r"\b(function |const |let |var |require\(|console\.)",
                     lowered):
            return "javascript"
        if re.search(r"(^|\n)\s*(echo |export |source |readonly |cd /|"
                     r"if \[|for |while |#!/usr/bin/env (ba)?sh)|"
                     r";\s*(echo|export|local|readonly)\b|\b(echo|export)\b\s+"
                     r"[\"$A-Za-z_]", lowered):
            return "bash"
        if "int main(" in lowered:
            return "c"
        if re.search(r"\b(my|our|say)\s+\$\w|\bprint\s+\$\w", lowered):
            return "perl"
        if lowered.lstrip().startswith("<?php") or \
                re.search(r"\becho\s+[\"']", lowered):
            return "php"
        if re.search(r"\b(puts|attr_accessor|require_relative)\b", lowered):
            return "ruby"
        if re.search(r"\blocal\s+[a-z_]+\s*=", lowered):
            return "lua"
        if re.search(r"\b(puts|proc|set)\s+[\"{a-z]", lowered):
            return "tcl"
        if "package main" in lowered:
            return "go"
        if re.search(r"\bfn main\b|use std::", lowered):
            return "rust"
        if "println!" in lowered:
            return "julia"
        return ""

    # ── execution ─────────────────────────────────────────────────────────
    def run(self, code: str, *, lang: str = "", file: str = "",
            timeout: float = 30.0, stdin: str = "") -> dict[str, Any]:
        from .tools.shell import run_sandboxed

        if not (code or "").strip() and not file:
            raise ToolError("run_code needs code or file=")
        lang = (lang or "").strip().lower() or self.detect_language(code, file)
        if not lang:
            raise ToolError(
                "could not detect the language — pass lang= "
                f"({', '.join(sorted(self.languages()))})")
        spec = _LANGS.get(lang)
        if spec is None:
            raise ToolError(f"unsupported language {lang!r}")
        info = self.languages()[lang]
        if not info.available:
            raise ToolError(
                f"{lang} is not installed here — install "
                f"{' or '.join(spec['bins'])[:60]} first")
        timeout = max(1.0, min(float(timeout), MAX_TIMEOUT))

        if file:
            from .tools.filesystem import safe_path

            src_path = safe_path(self.context, file, must_exist=True)
            code = src_path.read_text(encoding="utf-8", errors="ignore")

        # session workdir
        from .tools.sandbox_code import _session_dir

        stamp = time.strftime("%H%M%S") + f"-{int(time.time() * 1000) % 10000}"
        work = _session_dir(f"run-{stamp}")
        ext = spec.get("ext") or "txt"
        src = work / f"main.{ext}"
        src.write_text(code, encoding="utf-8")

        # build the command
        if "compile" in spec:
            compile_cmd = spec["compile"].format(bin=info.binary, work=work,
                                                 src=src)
            run_cmd = spec["run"].format(bin=info.binary, work=work, src=src)
            command = f"{compile_cmd} && {run_cmd}"
        else:
            command = f"{info.binary} {src}"
        if lang == "awk":
            # awk reads the program, then data on stdin
            command = f"{info.binary} -f {src}"
        if stdin:
            command = f"{command} <<'NM_STDIN'\n{stdin}\nNM_STDIN"

        result = run_sandboxed(
            command, cwd=work, timeout=timeout, network=False, stdin="")
        stdout = (result.get("stdout") or "")[:MAX_OUTPUT]
        stderr = (result.get("stderr") or "")[:MAX_OUTPUT]
        exit_code = int(result.get("exit_code", -1))
        timed_out = bool(result.get("timed_out"))

        files = []
        for p in sorted(work.rglob("*")):
            if p.is_file() and p.name not in (f"main.{ext}",):
                try:
                    if p.stat().st_size <= 200_000:
                        files.append({"path": str(p.relative_to(work)),
                                      "bytes": p.stat().st_size})
                except (OSError, ValueError):
                    continue
        return {
            "language": lang,
            "binary": info.binary,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "ok": exit_code == 0 and not timed_out,
            "seconds": round(float(result.get("seconds", 0.0)), 3),
            "timed_out": timed_out,
            "backend": result.get("backend", ""),
            "workdir": str(work),
            "files": files[:50],
            "truncated": len(result.get("stdout") or "") > MAX_OUTPUT or
            len(result.get("stderr") or "") > MAX_OUTPUT,
        }

    # ── CI loop: run → fail → model rewrite → re-run until green ─────────
    def run_until_green(self, code: str, *, lang: str = "",
                        max_rounds: int = 4, timeout: float = 30.0,
                        expected: str = "") -> dict[str, Any]:
        """Run the code; while it fails, let the model fix it and re-run.

        Green means exit code 0 (plus ``expected`` text in stdout when
        given).  Each fix round sends the code + the failing output to the
        router and asks for a complete corrected rewrite.  Without an LLM
        backend the loop honestly reports the last failure — no pretending.
        """
        max_rounds = max(1, min(int(max_rounds), 10))
        current = code
        history: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        green = False

        for round_no in range(1, max_rounds + 1):
            final = self.run(current, lang=lang, timeout=timeout)
            ok = bool(final.get("ok"))
            if ok and expected:
                ok = expected in (final.get("stdout") or "")
            entry = {
                "round": round_no,
                "exit_code": final.get("exit_code"),
                "ok": ok,
                "timed_out": bool(final.get("timed_out")),
                "stdout_tail": (final.get("stdout") or "")[-400:],
                "stderr_tail": (final.get("stderr") or "")[-800:],
                "seconds": final.get("seconds"),
            }
            history.append(entry)
            if ok:
                green = True
                break
            if round_no >= max_rounds:
                break

            # ask the model for a fixed rewrite
            fixed, fix_note = self._fix_code(
                current, lang, final, expected)
            entry["fix_note"] = fix_note
            if fixed is None:
                break
            entry["fixed_by"] = "model"
            current = fixed

        return {
            "green": green,
            "rounds": len(history),
            "max_rounds": max_rounds,
            "final_code": current,
            "final_run": final,
            "history": history,
            "expected": expected,
            "note": "" if green else (
                "still red after "
                f"{len(history)} rounds — see the last stderr_tail for the "
                "failure; widen max_rounds or hand the final_code to a "
                "human/model with more context"),
        }

    # ── green = test suite: whole files & projects ─────────────────────────
    _TEST_FILE_RE = re.compile(r"^(test_.*|.*_test)\.py$|^tests?(/|$)")
    _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                  "build", "dist", ".next", ".tox", ".mypy_cache",
                  ".pytest_cache", "coverage", ".cache", ".local"}

    def _resolve_target(self, path: str) -> "Path":
        from .tools.filesystem import safe_path

        return safe_path(self.context, path, must_exist=True)

    def _test_command(self, root: "Path") -> tuple[list[str], str]:
        """Choose the green-criterion command for a path.

        * project dir  → its test suite (pytest when installed + present,
          else unittest discover; npm test for JS projects)
        * test file    → that file as a test run
        * other file   → the file itself, exit 0
        """
        import sys

        if root.is_dir():
            pkg = root / "package.json"
            if pkg.exists():
                try:
                    import json as _json

                    data = _json.loads(pkg.read_text(errors="ignore"))
                    if (data.get("scripts") or {}).get("test"):
                        npm = shutil.which("npm") or "npm"
                        return [npm, "test", "--silent"], "npm test"
                except (ValueError, OSError):  # noqa: E103 - falls through to unittest detection
                    pass
            py_tests = [
                p for p in root.rglob("*.py")
                if not any(part in self._SKIP_DIRS
                           for part in p.parts)
                and self._TEST_FILE_RE.match(p.name)
            ]
            if py_tests:
                # unittest is ALWAYS good enough: the green criterion must not
                # hinge on whether an optional package happens to be
                # installed here. pytest is chosen only when the suite
                # actually speaks pytest (imports/markers/fixtures).
                def _uses_pytest() -> bool:
                    for tp in py_tests[:20]:
                        try:
                            text = tp.read_text(errors="ignore")
                        except OSError:
                            continue
                        if "import pytest" in text or "pytest.mark" in text \
                                or "pytest.fixture" in text:
                            return True
                    return False

                try:
                    import importlib.util

                    have_pytest = importlib.util.find_spec("pytest") is not None
                except (ImportError, ValueError):
                    have_pytest = False
                if have_pytest and _uses_pytest():
                    return ([sys.executable, "-m", "pytest", "-q",
                             str(root)], "pytest -q <project>")
                tests_dir = str(root / "tests") if (root / "tests").is_dir() \
                    and any((root / "tests").rglob("test_*.py")) \
                    else str(root)
                # no top-level-dir arg: with cwd=root the project root is on
                # sys.path, which is what makes `import <module>` work from
                # the test files (passing a top-level dir made discover find
                # 0 tests and report a false green)
                return ([sys.executable, "-m", "unittest", "discover",
                         "-s", tests_dir, "-p", "test_*.py"],
                        "unittest discover <project>")
            main = root / "main.py"
            if not main.exists():
                pys = sorted(p for p in root.glob("*.py"))
                if pys:
                    main = pys[0]
            if main.exists():
                return [sys.executable, str(main)], f"python {main.name}"
            raise ToolError(
                f"no test suite or runnable entry found in {root} "
                "(test_*.py files, tests/ dir, package.json test script, "
                "or main.py)")
        # a single file
        if self._TEST_FILE_RE.match(root.name) and root.suffix == ".py":
            return [sys.executable, str(root)], f"python {root.name} (tests)"
        return [sys.executable, str(root)], f"python {root.name}"

    def _run_cmd(self, cmd: list[str], cwd: str,
                 timeout: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": (proc.stdout or "")[-20_000:],
            "stderr": (proc.stderr or "")[-20_000:],
            "timed_out": False,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    @staticmethod
    def _traceback_files(stderr: str) -> list[str]:
        """File names mentioned in a Python traceback."""
        out: list[str] = []
        for m in re.finditer(
                r'File "([^"]+\.py)", line \d+', stderr):
            name = m.group(1)
            if name not in out:
                out.append(name)
        return out

    def _project_context(self, root: "Path",
                         result: dict[str, Any]) -> tuple[str, list["Path"]]:
        """Layout + the files a failing round most likely touched."""
        files: list["Path"] = []
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_dir():
                    continue
                if any(part in self._SKIP_DIRS for part in p.parts):
                    continue
                files.append(p)
            files = files[:40]
        else:
            files = [root]
        lines = []
        if root.is_dir():
            lines.append(f"Project: {root}")
            for p in files:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"  {p.relative_to(root)}  ({size} B)")
        # include the files the traceback blames + the test files, full text
        blamed = [b for b in self._traceback_files(
            result.get("stderr", ""))]
        wanted: list["Path"] = []
        for b in blamed:
            for p in files:
                if str(p).endswith(b) or b.endswith(str(p)):
                    if p not in wanted:
                        wanted.append(p)
        for p in files:
            if self._TEST_FILE_RE.match(p.name) and p not in wanted:
                wanted.append(p)
        wanted = wanted[:6]
        for p in wanted:
            try:
                text = p.read_text(errors="ignore")[:15000]
            except OSError:
                continue
            rel = p.relative_to(root) if root.is_dir() else p.name
            lines.append(f"\n=== {rel} ===\n{text}")
        return "\n".join(lines), files

    def _fix_project(self, root: "Path", ctx: str,
                     run_result: dict[str, Any]) -> tuple[list[tuple["Path", str]], str]:
        """One rewrite round. Returns (list of (path, new_content), note)."""
        router = getattr(self.context, "router", None)
        if router is None:
            return [], ("no LLM backend available — automatic project "
                        "fixing needs a configured model (the suite ran "
                        "and failed; the failure is in final_run)")
        prompt = (
            "A project\'s test suite is FAILING. Fix the code until the "
            "suite is green.\n\n"
            f"Suite command output (exit {run_result.get('exit_code')}):\n"
            f"STDERR:\n{(run_result.get('stderr') or '')[-6000:]}\n\n"
            f"STDOUT (tail):\n{(run_result.get('stdout') or '')[-2000:]}\n\n"
            f"Project context (layout + relevant file contents):\n{ctx[:24000]}\n\n"
            "Reply with the COMPLETE corrected content of EACH file that "
            "needs to change, in this exact format — one section per "
            "file, a `# FILE: <relative/path>` line, then a fenced code "
            "block with the FULL file (never a diff, never a snippet, "
            "never placeholders):\n\n"
            "# FILE: path/to/file.py\n"
            "```python\n<full file>\n```\n\n"
            "Only files that must change. No explanation outside the "
            "sections."
        )
        try:
            from .llm.base import Message, SamplingParams

            resp = brain_for(self.context).chat(
                [Message.system(
                    "You are a surgical test-suite repair engine. You "
                    "make failing test suites green by rewriting the "
                    "minimal set of files. You always emit complete "
                    "files, never diffs."),
                 Message.user(prompt)],
                SamplingParams(temperature=0.1, max_tokens=8000), task_kind="chat")
            if not resp.ok or not (resp.text or "").strip():
                return [], f"model fix failed: {resp.error or 'empty response'}"
            out: list[tuple[Path, str]] = []
            for m in re.finditer(
                    r"#\s*FILE:\s*(\S+)\s*\n"
                    r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```",
                    resp.text, re.S):
                rel = m.group(1).strip()
                content = m.group(2).rstrip("\n") + "\n"
                if not self._CODE_TOKENS.search(content):
                    continue  # chatty garbage, not code
                target = (root / rel) if root.is_dir() else root
                out.append((target, content))
            if not out:
                return [], ("model returned no usable FILE sections "
                            f"(got: {(resp.text or '')[:80].strip()!r})")
            return out, f"rewrote {len(out)} file(s) by model"
        except Exception as exc:  # noqa: BLE001
            return [], f"model fix error: {exc}"

    def run_project_until_green(self, path: str, *,
                                max_rounds: int = 4,
                                timeout: float = 180.0,
                                command: str = "") -> dict[str, Any]:
        """Run a FILE or whole PROJECT until its test suite is green.

        Green = exit 0 of the suite: unittest/pytest for Python,
        ``npm test`` for JS projects, the file itself for a plain script.
        While red, the model gets the failing output + the relevant files
        and rewrites what must change; the suite re-runs each round.
        Without an LLM backend the loop honestly reports the last failure.
        """
        max_rounds = max(1, min(int(max_rounds), 10))
        root = self._resolve_target(path)
        if command:
            cmd = command.split()
            label = command
        else:
            cmd, label = self._test_command(root)
        cwd = str(root if root.is_dir() else root.parent)
        history: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        green = False
        files_changed: list[str] = []
        for round_no in range(1, max_rounds + 1):
            try:
                final = self._run_cmd(cmd, cwd, timeout)
            except subprocess.TimeoutExpired:
                final = {"exit_code": -1, "ok": False, "stdout": "",
                         "stderr": f"timed out after {timeout}s",
                         "timed_out": True, "seconds": timeout}
            entry = {
                "round": round_no, "exit_code": final.get("exit_code"),
                "ok": final.get("ok"),
                "timed_out": bool(final.get("timed_out")),
                "stdout_tail": (final.get("stdout") or "")[-400:],
                "stderr_tail": (final.get("stderr") or "")[-800:],
                "seconds": final.get("seconds"),
            }
            history.append(entry)
            if final.get("ok"):
                green = True
                break
            if round_no >= max_rounds:
                break
            ctx, _files = self._project_context(root, final)
            fixes, note = self._fix_project(root, ctx, final)
            entry["fix_note"] = note
            if not fixes:
                break
            for target, content in fixes:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                files_changed.append(str(target))
            entry["fixed_files"] = [str(t) for t, _ in fixes]

        return {
            "green": green,
            "target": str(root),
            "criterion": label,
            "rounds": len(history),
            "max_rounds": max_rounds,
            "files_changed": files_changed,
            "final_run": final,
            "history": history,
            "note": "" if green else (
                "still red after "
                f"{len(history)} rounds — last failure is in "
                "final_run.stderr_tail; widen max_rounds or hand "
                "files_changed + the suite output to a human"),
        }

    def _fix_code(self, code: str, lang: str,
                  run_result: dict[str, Any],
                  expected: str = "") -> tuple[str | None, str]:
        """One rewrite round. Returns (new_code or None, note)."""
        router = getattr(self.context, "router", None)
        if router is None:
            return None, ("no LLM backend available — automatic fixing "
                          "needs a configured model (the code ran and "
                          "failed; the failure is in final_run)")
        lang = lang or (run_result.get("language") or "")
        spec = (f"Expected in stdout: {expected!r}\n" if expected else "")
        prompt = (
            f"The following {lang or 'code'} was executed and FAILED.\n\n"
            f"Code:\n```\n{code}\n```\n\n"
            f"Exit code: {run_result.get('exit_code')}\n"
            f"Stderr:\n{(run_result.get('stderr') or '')[-2000:]}\n"
            f"Stdout:\n{(run_result.get('stdout') or '')[-2000:]}\n\n"
            f"{spec}"
            "Fix it. Reply with ONLY the complete corrected program in a "
            "single fenced code block — no explanation, no truncation, no "
            "placeholders."
        )
        try:
            from .llm.base import Message, SamplingParams

            resp = brain_for(self.context).chat(
                [Message.system(
                    "You are a surgical code repair engine. You rewrite "
                    "failing programs so they run cleanly and produce the "
                    "requested output. You never lecture; you only emit "
                    "the fixed code."),
                 Message.user(prompt)],
                SamplingParams(temperature=0.1, max_tokens=8000), task_kind="chat")
            if not resp.ok or not (resp.text or "").strip():
                return None, f"model fix failed: {resp.error or 'empty response'}"
            fixed = self._extract_code_block(resp.text, lang)
            if not fixed or not fixed.strip():
                return None, ("model returned no usable code "
                              f"(got: {(resp.text or '')[:80].strip()!r})")
            return fixed, "rewritten by model"
        except Exception as exc:  # noqa: BLE001
            return None, f"model fix error: {exc}"

    #: token that should appear somewhere in real source code
    _CODE_TOKENS = re.compile(
        r"(?m)^\s*(def |class |import |from |return |print\(|if |elif |"
        r"else:|for |while |function |const |let |var |echo |exit |main|"
        r"#include|package |public |fun |fn |SELECT |INSERT |BEGIN |END |"
        r"#!|//|\{|\}|=)")

    @classmethod
    def _extract_code_block(cls, text: str, lang: str = "") -> str:
        """Pull the (longest) fenced block out of a model reply.  Without
        fences, the raw text is used only if it plausibly IS code — this
        keeps chatty non-compliant replies ("working on it…") out of the
        sandbox."""
        m = re.findall(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, re.S)
        blocks = [b for b in m if b.strip()]
        if blocks:
            return max(blocks, key=len).strip()
        raw = text.strip()
        if len(raw) < 20 or not cls._CODE_TOKENS.search(raw):
            return ""
        return raw


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "run_code",
        description=(
            "Execute code in the sandbox and get structured results: python, "
            "javascript, bash, ruby, perl, php, lua, tcl, awk, julia, deno, "
            "bun, c, cpp, rust, go (whatever is installed). Returns stdout, "
            "stderr, exit code, wall time, timeout flag, and files the code "
            "wrote. action=run (code|file, lang, timeout, stdin) | "
            "until_green (code|project, lang, max_rounds, expected) — CI "
            "loop: run, and on failure the model rewrites and it runs "
            "again until exit 0. With project= (a file or whole project "
            "path) green = the test suite (unittest/pytest/npm test). "
            "| languages."
        ),
        capability=Capability.EXEC_SHELL,
    )
    def run_code(action: str = "run", code: str = "", file: str = "",
                 lang: str = "", timeout: float = 30.0,
                 stdin: str = "", max_rounds: int = 4,
                 expected: str = "", project: str = "") -> dict[str, Any]:
        box = CodeRunner(context)
        if action == "languages":
            return {"languages": [L.to_dict() for L in box.languages().values()]}
        if action == "until_green":
            if project:
                return box.run_project_until_green(
                    project, max_rounds=max_rounds, timeout=timeout)
            return box.run_until_green(code, lang=lang,
                                       max_rounds=max_rounds,
                                       timeout=timeout, expected=expected)
        if action != "run":
            raise ToolError(f"unknown run_code action {action!r}")
        return box.run(code, lang=lang, file=file, timeout=timeout,
                       stdin=stdin)
