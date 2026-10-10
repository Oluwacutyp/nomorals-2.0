"""Error-handling scanner: find every place the codebase mishandles errors.

``scan(paths)`` walks Python sources with the AST and reports findings:

* E101 bare ``except:`` — catches everything including KeyboardInterrupt.
* E102 ``except BaseException`` — same problem, spelled out.
* E103 swallowed exception — the ``except`` body is just ``pass``.
* E104 ignored exception — a broad ``except`` that never touches the
  exception value (no logging, no classify, no re-raise, no record).
* E105 ``except`` tuple that redundantly includes ``Exception``
  alongside specific types.
* E106 ``except KeyboardInterrupt`` — almost always wrong; let it
  propagate or catch it deliberately at the top level.

The scanner is descriptive, not a gate: it reports, it never rewrites.
Run ``python -m nomorals.tools.error_scan [paths...]`` for a report,
or ``scan()`` for machine-readable findings.
"""

from __future__ import annotations

import ast
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Finding", "ScanReport", "scan", "main", "register"]

SEVERITY = {"E101": "error", "E102": "error", "E103": "error",
            "E104": "warning", "E105": "info", "E106": "warning"}

MESSAGES = {
    "E101": "bare except: catches KeyboardInterrupt/SystemExit too",
    "E102": "except BaseException: catches KeyboardInterrupt/SystemExit too",
    "E103": "swallowed exception: except body is just 'pass'",
    "E104": "broad except ignores the exception value entirely",
    "E105": "redundant except tuple: specific types already covered by Exception",
    "E106": "except KeyboardInterrupt: let it propagate unless this is a deliberate top-level shutdown hook",
}


@dataclass
class Finding:
    rule: str
    file: str
    line: int
    col: int
    message: str
    severity: str = field(init=False)
    context: str = ""

    def __post_init__(self) -> None:
        self.severity = SEVERITY[self.rule]

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "severity": self.severity, "file": self.file,
                "line": self.line, "col": self.col, "message": self.message,
                "context": self.context}


@dataclass
class ScanReport:
    findings: list[Finding]
    files_scanned: int
    files_failed: int
    elapsed_s: float

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "files_scanned": self.files_scanned,
                "files_failed": self.files_failed,
                "elapsed_s": round(self.elapsed_s, 2),
                "findings": [f.to_dict() for f in self.findings]}


def _handler_names(node: ast.ExceptHandler) -> list[str]:
    """Resolve the exception names in an except clause to dotted strings."""
    if node.type is None:
        return []
    names: list[str] = []

    def dotted(n: ast.AST) -> str:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            return dotted(n.value) + "." + n.attr
        return "?"

    target = node.type
    if isinstance(target, ast.Tuple):
        return [dotted(e) for e in target.elts]
    return [dotted(target)]


def _is_broad(names: list[str]) -> bool:
    return any(n in ("Exception", "BaseException") or n.endswith(".Exception")
               for n in names)


def _body_uses_exception(body: list[ast.stmt], exc_name: str | None) -> bool:
    """True if the handler body references, records, or re-raises the error."""
    if not body:
        return False
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(stmt, ast.Raise):
            return True  # re-raise (bare or chained) is honest
        if isinstance(stmt, ast.Name) and exc_name and stmt.id == exc_name:
            return True
        if isinstance(stmt, ast.Call):
            func = stmt.func
            fname = ""
            if isinstance(func, ast.Attribute):
                fname = func.attr
            elif isinstance(func, ast.Name):
                fname = func.id
            # logging it, classifying it, or recording it counts as handling
            if fname in {"classify", "error", "warning", "exception", "critical",
                         "mark_failed", "record", "report"}:
                return True
    return False


# ruff codes the codebase already uses to mark deliberate catches, mapped to
# our rules.  A bare "# noqa" suppresses everything on that line.
_NOQA_MAP = {"BLE001": ("E101", "E102", "E103", "E104"),  # blind except (whole handler accepted)
             "S110": ("E103",),                   # try-except-pass
             "B110": ("E103",)}                    # try-except-pass (bugbear)


def _noqa_suppresses(line: str, rule: str) -> bool:
    # find "noqa" anywhere inside a trailing comment, e.g.
    # "# noqa: BLE001", "# noqa", "# pragma: no cover, noqa: E103"
    m = re.search(r"#.*?\bnoqa\b(.*)$", line)
    if not m:
        return False
    tail = m.group(1).strip()
    if not tail.startswith(":"):
        return True  # bare "noqa" suppresses all
    codes = [c.strip().split()[0] for c in tail[1:].split(",") if c.strip()]
    for code in codes:
        if code == rule or rule in _NOQA_MAP.get(code, ()):
            return True
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, source_lines: list[str]) -> None:
        self.path = path
        self.lines = source_lines
        self.findings: list[Finding] = []

    def _add(self, rule: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", 1)
        ctx = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        # the author already reviewed this catch: don't re-flag it
        if _noqa_suppresses(self.lines[line - 1] if 0 < line <= len(self.lines) else "", rule):
            return
        self.findings.append(Finding(rule=rule, file=self.path, line=line,
                                     col=getattr(node, "col_offset", 0),
                                     message=MESSAGES[rule], context=ctx))

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            names = _handler_names(handler)
            if handler.type is None:
                self._add("E101", handler)
                continue
            if "BaseException" in names:
                self._add("E102", handler)
                continue
            if "KeyboardInterrupt" in names:
                self._add("E106", handler)
            broad = _is_broad(names)
            if broad and len(names) > 1:
                self._add("E105", handler)
            body = handler.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                self._add("E103", handler)
            elif broad and not _body_uses_exception(body, handler.name):
                self._add("E104", handler)
        self.generic_visit(node)


def _iter_py_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    # skip vendored / generated code: not ours to fix
    return [f for f in files
            if "vendor" not in f.parts and "node_modules" not in f.parts
            and "__pycache__" not in f.parts]


def scan(paths: list[str] | None = None,
         repo: str | None = None) -> ScanReport:
    """Scan Python sources for error-handling problems. Never rewrites code."""
    started = time.perf_counter()
    roots = paths or ([repo] if repo else ["."])
    py_files = _iter_py_files(roots)
    findings: list[Finding] = []
    failed = 0
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            failed += 1
            _log.warning("error_scan: cannot parse %s: %s", path, exc)
            continue
        visitor = _Visitor(str(path), text.splitlines())
        visitor.visit(tree)
        findings.extend(visitor.findings)
    findings.sort(key=lambda f: (f.file, f.line))
    return ScanReport(findings=findings, files_scanned=len(py_files),
                      files_failed=failed,
                      elapsed_s=time.perf_counter() - started)


def format_report(report: ScanReport) -> str:
    from . import _style as _style

    lines = [_style.banner(
        f"error_scan — {report.files_scanned} files, "
        f"{len(report.findings)} findings ({len(report.errors)} errors)",
        sub=f"completed in {report.elapsed_s:.1f}s")]
    if report.files_failed:
        lines.append(_style.status_line(
            _style.WARN,
            f"{report.files_failed} files could not be parsed"))
    current = ""
    for f in report.findings:
        if f.file != current:
            current = f.file
            lines.append("")
            lines.append(_style.colorize(current, "heading"))
        status = _style.FAIL if f.severity == "error" else _style.WARN
        lines.append(_style.status_line(
            status, f"{f.line}:{f.col} [{f.rule}] {f.message}"))
        if f.context:
            lines.append(_style.colorize(f"      > {f.context}", "dim"))
    if not report.findings:
        lines.append(_style.status_line(_style.OK, "clean: no error-handling "
                                                   "problems found"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    paths = [a for a in args if not a.startswith("-")]
    as_json = "--json" in args
    report = scan(paths or None)
    if as_json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


def register(registry: Any) -> None:
    """Attach the error scanner to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "error_scan",
        description=("Static error-handling scan: finds bare except, "
                     "swallowed exceptions, ignored broad catches, "
                     "BaseException/KeyboardInterrupt traps. Returns "
                     "{ok, files_scanned, findings}. Read-only, never "
                     "rewrites code."),
        capability=Capability.FS_READ,
    )
    def _error_scan(paths: list[str] | None = None,
                    repo: str | None = None) -> dict[str, Any]:
        return scan(paths, repo).to_dict()


if __name__ == "__main__":
    sys.exit(main())
