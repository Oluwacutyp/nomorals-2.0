"""Live survey of Devon's own system: organs, wiring, dead paths, weak spots.

This is the survey half of the arena researcher (doc 15 R-2-14 — the
live version of the one-off module-gap audit). It walks the checked-out
tree under ``nomorals/`` and returns findings, each carrying a
``file:line`` anchor that verifies against the surveyed root
(:func:`verify_anchor`), the module's importers, and the callers or
evidence that prove the finding is real — not vibes.

Finding kinds:

* ``dead_module`` — a production module with zero production importers
  under ``nomorals/`` AND zero dotted-name references anywhere under
  ``nomorals/``. Entry points (L7 / ``cmdline`` / ``builders`` / top-level
  layer modules from :data:`nomorals.top_level_layer.TOP_LEVEL_LAYER`)
  are never candidates: they are reached from outside the tree.
* ``stub`` — a function whose whole body is one ``raise
  NotImplementedError`` or only ``pass`` / ``...`` / a docstring.
  ``@abstractmethod`` contracts are excluded: they are the interface,
  not a stub.
* ``unwired_tool`` — an entry of the agent-tool registry
  (:data:`nomorals.tools.agents.AGENT_TOOL_MODULES`) whose module cannot
  be imported, or imports but exposes no ``register`` hook, so the tool
  never reaches the runtime.
* ``silent_swallow`` — a module with three or more bare
  ``except: pass`` / ``except Exception: pass`` sites (SOUL.md §1: a
  swallowed error is a silent failure, not a fix).

Known limitation, stated so the owner can judge it: a module loaded
only by a dynamic path (``importlib.util.spec_from_file_location``,
a variable ``import_module`` name) is invisible to a static scan and
can be flagged dead. The finding's evidence states exactly what was
measured, and proposals are never auto-applied — the owner denies a
false positive, and the denial is recorded.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import time
from pathlib import Path

#: Modules with three or more silent-swallow sites get a finding.
SILENT_SWALLOW_THRESHOLD = 3

#: Packages whose modules are reached from outside the tree (L7 entry
#: points, CLI, builders) and are never dead-module candidates.
ENTRY_POINT_PACKAGES = {"api", "builders", "cmdline"}

#: A dotted ``nomorals....`` reference anywhere in a file's text keeps
#: a module alive (covers string-form dynamic references).
_DOTTED_REF_RE = re.compile(r"nomorals(?:\.[A-Za-z0-9_]+)+")

#: Survey-cache format version. Bumped when the cached per-file shape
#: changes; a cache written by another version is ignored, never read.
CACHE_VERSION = 1

#: Default survey-cache filename under ``<home>/data/`` (beside the
#: arena store's ``arena_researcher.json``).
SURVEY_CACHE_FILENAME = "arena_survey_cache.json"


def default_survey_cache_path(home: str | Path | None) -> Path | None:
    """Default survey-cache path for *home*, or None with no home."""
    if not home:
        return None
    return Path(home) / "data" / SURVEY_CACHE_FILENAME


def _load_survey_cache(path: str | Path,
                       root: Path) -> dict[str, dict]:
    """Cached per-file parse results, or {} when there is no usable one.

    A cache is usable only when its version matches and it was written
    for this same repo root. A missing, corrupt, or stale cache is not
    an error — the survey just re-parses everything this run.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    if str(raw.get("root") or "") != str(root):
        return {}
    files = raw.get("files")
    if not isinstance(files, dict):
        return {}
    kept: dict[str, dict] = {}
    for rel, entry in files.items():
        if not isinstance(entry, dict):
            continue
        if not {"size", "mtime_ns", "dotted", "imports", "refs",
                "stubs", "swallows", "entry"} <= set(entry):
            continue
        kept[str(rel)] = entry
    return kept


def _save_survey_cache(path: str | Path, root: Path,
                       entries: dict[str, dict]) -> None:
    """Persist the per-file cache atomically (tmp+replace).

    Best-effort like the arena store: a cache that cannot be written is
    a missed optimisation, never a failed survey.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"version": CACHE_VERSION, "root": str(root),
             "files": entries}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _is_stub_body(body: list[ast.stmt]) -> bool:
    """True when a function body does nothing at all."""
    meaningful = [s for s in body
                  if not (isinstance(s, ast.Expr)
                          and isinstance(s.value, ast.Constant)
                          and isinstance(s.value.value, str))]
    if not meaningful:
        return True
    if len(meaningful) != 1:
        return False
    only = meaningful[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) \
            and only.value.value is Ellipsis:
        return True
    if isinstance(only, ast.Raise) and isinstance(only.exc, ast.Call) \
            and isinstance(only.exc.func, ast.Name) \
            and only.exc.func.id == "NotImplementedError":
        return True
    return (isinstance(only, ast.Raise)
            and isinstance(only.exc, ast.Name)
            and only.exc.id == "NotImplementedError")


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) \
                and dec.attr == "abstractmethod":
            return True
    return False


def _is_silent_handler(handler: ast.ExceptHandler) -> bool:
    """True when an except handler swallows the error silently."""
    if not handler.body:
        return False
    for stmt in handler.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) \
                and isinstance(stmt.value, ast.Constant) \
                and stmt.value.value is Ellipsis:
            continue
        return False
    return True


def _dotted_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative(dotted: str, level: int,
                      module: str | None) -> str | None:
    """Resolve a relative import to its absolute dotted name.

    *dotted* is the importing module's dotted name; its package is
    everything but the last segment. Level 1 is the current package,
    level 2 its parent, and so on.
    """
    package = dotted.split(".")[:-1]
    if level > len(package) + 1:
        return None
    base = package[:len(package) - level + 1]
    if module:
        base.append(module)
    return ".".join(base)


def _collect_imports(tree: ast.Module, dotted: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                base = _resolve_relative(dotted, node.level, node.module) \
                    or ""
            if base:
                imported.add(base)
            # `from X import Y` also imports X.Y (the name, not just
            # the package) — dropping it made every such module look
            # dead.
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base:
                    imported.add(base + "." + alias.name)
    return imported


def _is_entry_module(tree: ast.Module) -> bool:
    """True when a module is an entry point (top-level main / __main__)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == "main":
            return True
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) \
                    and isinstance(test.left, ast.Name) \
                    and test.left.id == "__name__" \
                    and any(isinstance(c, ast.Constant)
                            and c.value == "__main__"
                            for c in test.comparators):
                return True
    return False


def _entry_points(root: Path) -> set[str]:
    """Dotted names that are reached from outside the tree."""
    names: set[str] = set()
    try:
        from nomorals.top_level_layer import TOP_LEVEL_LAYER
    except Exception:
        TOP_LEVEL_LAYER = {}
    for name in TOP_LEVEL_LAYER:
        names.add("nomorals." + name)
    for pkg in ENTRY_POINT_PACKAGES:
        names.add("nomorals." + pkg)
    return names


def _tool_module_line(root: Path, name: str) -> int:
    registry = root / "nomorals" / "tools" / "agents.py"
    try:
        for lineno, line in enumerate(
                registry.read_text(encoding="utf-8").splitlines(), 1):
            if (f'"{name}"') in line or (f"'{name}'") in line:
                return lineno
    except OSError:
        pass
    return 1


class SurveyFindings(list):
    """A survey result that is also its own budget receipt.

    A plain ``list`` subclass, so every caller that treats the
    survey as a list keeps working.  ``partial`` is True when the
    read+parse pass stopped on its deadline before covering the
    whole tree (``surveyed_files`` of ``total_files``); a partial
    survey's findings are real, just incomplete — the tick files
    them and says so in its run record instead of pretending the
    tree was fully covered.
    """

    def __init__(self, items: list[dict] | None = None, *,
                 partial: bool = False, surveyed_files: int = 0,
                 total_files: int = 0, seconds: float = 0.0) -> None:
        super().__init__(items or [])
        self.partial = partial
        self.surveyed_files = surveyed_files
        self.total_files = total_files
        self.seconds = seconds


def survey_repo(root: str | Path, *,
                tool_modules: list[str] | None = None,
                deadline_s: float | None = None,
                cache_path: str | Path | None = None) -> SurveyFindings:
    """Survey *root* and return ranked-ready findings.

    *root* is the repo root (the directory holding ``nomorals/``).
    Findings are dicts with ``id`` (stable), ``kind``, ``path``,
    ``line``, ``organ``, ``evidence`` and ``count``.

    ``deadline_s`` bounds the read+parse pass: the survey stops
    starting new files once the budget is spent and returns what it
    found, flagged ``partial`` on the returned list (a plain ``list``
    subclass, so every existing caller keeps working).  It is a
    TRIPWIRE, not the reason the survey completes: a survey with no
    budget is how a scheduled tick wedged past the scheduler's 300s
    wall (2026-11-21: the live survey reads ~2,900 files and the wall
    fired mid-walk in ``Path.open``) — callers that run inside a
    capped job MUST pass one.

    ``cache_path`` is what makes the survey actually COMPLETE instead
    of merely stopping on time: per-file parse results (imports,
    stubs, swallow count, dotted refs, entry flag) are cached keyed
    by ``(path, size, mtime_ns)`` — the same fingerprint every build
    tool uses — so a warm survey only ``stat``s unchanged files and
    re-parses exactly the files that changed. The cross-file
    aggregation still runs over the whole tree every run, from the
    cached per-file data in memory. A missing, corrupt, or
    root-mismatched cache is ignored, never fatal; without a cache
    path the survey behaves exactly as before (full re-parse).
    """
    started = time.monotonic()
    deadline = (started + float(deadline_s)
                if deadline_s is not None and float(deadline_s) > 0
                else None)
    root = Path(root)
    pkg_root = root / "nomorals"
    files = sorted(p for p in pkg_root.rglob("*.py")
                   if "__pycache__" not in p.parts)
    entries = _entry_points(root)
    importers: dict[str, set[str]] = {}
    dotted_refs: set[str] = set()
    per_file: dict[Path, dict] = {}
    surveyed = 0
    cache: dict[str, dict] = (
        _load_survey_cache(cache_path, root)
        if cache_path is not None else {})
    cache_dirty = False
    for path in files:
        if deadline is not None and time.monotonic() >= deadline:
            break
        rel = str(path.relative_to(root))
        dotted = _dotted_name(root, path)
        entry: dict | None = None
        if cache_path is not None:
            try:
                st = path.stat()
            except OSError:
                continue
            cached = cache.get(rel)
            if (cached is not None
                    and cached.get("size") == st.st_size
                    and cached.get("mtime_ns") == st.st_mtime_ns
                    and cached.get("dotted") == dotted):
                entry = cached
        if entry is None:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            surveyed += 1
            refs = _DOTTED_REF_RE.findall(text)
            dotted_refs.update(refs)
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError:
                continue
            imported = _collect_imports(tree, dotted)
            stubs: list[dict] = []
            swallows = 0
            is_entry = _is_entry_module(tree)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not _is_abstract(node) \
                        and _is_stub_body(node.body):
                    stubs.append({"name": node.name, "line": node.lineno})
                elif isinstance(node, ast.ExceptHandler) \
                        and _is_silent_handler(node):
                    swallows += 1
            entry = {"dotted": dotted,
                     "imports": sorted(imported),
                     "refs": sorted(refs),
                     "stubs": stubs, "swallows": swallows,
                     "entry": is_entry}
            if cache_path is not None:
                try:
                    st = path.stat()
                except OSError:
                    st = None
                if st is not None:
                    cache[rel] = {"size": st.st_size,
                                  "mtime_ns": st.st_mtime_ns, **entry}
                    cache_dirty = True
        else:
            surveyed += 1
            dotted_refs.update(entry.get("refs") or [])
        for name in entry.get("imports") or []:
            importers.setdefault(name, set()).add(dotted)
        per_file[path] = {"dotted": dotted,
                          "stubs": entry.get("stubs") or [],
                          "swallows": entry.get("swallows") or 0,
                          "entry": bool(entry.get("entry"))}
    if cache_path is not None:
        known = {str(p.relative_to(root)) for p in files}
        for rel in [r for r in cache if r not in known]:
            del cache[rel]
            cache_dirty = True
        if cache_dirty:
            _save_survey_cache(cache_path, root, cache)
    findings: SurveyFindings = SurveyFindings(
        [], partial=(surveyed < len(files)),
        surveyed_files=surveyed, total_files=len(files),
        seconds=round(time.monotonic() - started, 1))
    for path in files:
        info = per_file.get(path)
        if info is None:
            continue
        rel = str(path.relative_to(root))
        organ = path.relative_to(pkg_root).parts[0]
        dotted = info["dotted"]
        if path.name != "__init__.py" \
                and not info.get("entry") \
                and dotted not in entries \
                and not any(dotted == e or dotted.startswith(e + ".")
                            for e in entries) \
                and dotted not in importers \
                and dotted not in dotted_refs:
            findings.append({
                "id": "ar-dead-{}".format(dotted.replace(".", "-")),
                "kind": "dead_module",
                "path": rel,
                "line": 1,
                "organ": organ,
                "evidence": (
                    "zero production importers in nomorals/ and zero "
                    "dotted-name references in nomorals/"),
                "count": 1,
            })
        for stub in info["stubs"]:
            findings.append({
                "id": f"ar-stub-{dotted.replace('.', '-')}-{stub['line']}",
                "kind": "stub",
                "path": rel,
                "line": stub["line"],
                "organ": organ,
                "evidence": ("function {} has no implementation "
                             "(raises NotImplementedError or empty body)".format(stub["name"])),
                "count": 1,
            })
        if info["swallows"] >= SILENT_SWALLOW_THRESHOLD:
            findings.append({
                "id": "ar-swallow-{}".format(dotted.replace(".", "-")),
                "kind": "silent_swallow",
                "path": rel,
                "line": 1,
                "organ": organ,
                "evidence": (f"{info['swallows']} silent except-pass sites "
                             "in one module; errors vanish without a trace"),
                "count": info["swallows"],
            })
    if tool_modules is None:
        try:
            from nomorals.tools.agents import AGENT_TOOL_MODULES
            tool_modules = list(AGENT_TOOL_MODULES)
        except Exception:
            tool_modules = []
    for name in tool_modules:
        module = None
        for candidate in (f"nomorals.agents.{name}",
                          f"nomorals.{name}"):
            try:
                module = importlib.import_module(candidate)
                break
            except Exception:
                continue
        line = _tool_module_line(root, name)
        if module is None:
            findings.append({
                "id": f"ar-unwired-{name}",
                "kind": "unwired_tool",
                "path": "nomorals/tools/agents.py",
                "line": line,
                "organ": "tools",
                "evidence": (f"tool module {name} cannot be imported, so the "
                             "registry entry never reaches the runtime"),
                "count": 1,
            })
        elif not hasattr(module, "register"):
            findings.append({
                "id": f"ar-unwired-{name}",
                "kind": "unwired_tool",
                "path": "nomorals/tools/agents.py",
                "line": line,
                "organ": "tools",
                "evidence": (f"tool module {name} imports but exposes no "
                             "register hook, so its tool never reaches "
                             "the runtime"),
                "count": 1,
            })
    findings.seconds = round(time.monotonic() - started, 1)
    return findings


def verify_anchor(root: str | Path, path: str, line: int) -> bool:
    """True when *path*:*line* exists in the surveyed root."""
    target = Path(root) / path
    if not target.is_file():
        return False
    try:
        total = sum(1 for _ in target.open(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return False
    return 1 <= line <= total
