"""Specialist subagent roster (Wave A, stream 3).

Eight purpose-built coding-pipeline roles with a uniform
``run(input) -> result`` interface: every input and result is a dataclass
(no free-form dicts), and every subagent is constructed per-mission with no
shared mutable state, so a roster can run in parallel via
:func:`run_parallel`.

**Module layout.** One module, not seven. The eight roles share their
skeleton (model-call helper, JSON extraction, path validation, the fused
runner and its merge functions); each role's own logic is 40-120 lines, so
splitting them would produce seven near-identical import headers wrapping
one small class each. This mirrors the existing precedent of
``nomorals/agents/coding.py`` (agent + result + helpers in one module) and
``nomorals/agents/role_specs.py`` (spec + registry + swarm agent together).

**Reuse decisions (survey before building).**

* *Implementer* does not reimplement surgical editing: it imports and calls
  :meth:`nomorals.tools.edit_loop.EditLoop.surgical_replace_many` — the
  canonical exact-text batch machinery (atomic, exact-once per edit,
  syntax-checked). Per-edit failure attribution comes from the same
  module's :meth:`preview_replace` matches, never from a second matcher.
* *Tester* and *Refactorer* run tests through
  :func:`nomorals.tools.pytest_runner.run_tests` (pytest when present,
  unittest discover otherwise) — the same runner the ``run_tests`` tool
  exposes.
* Each role carries a ``role_spec`` drawn from
  :data:`nomorals.agents.role_specs.BUILTIN_ROLES` (architect/critic/tester/
  coder/researcher) so the roster speaks the same contract language as the
  swarm layer. :class:`SwarmAgent` itself is *not* subclassed: its contract
  is ``run(payload dict) -> SimpleNamespace(output, ok, error)`` bound to a
  tool registry and a full ``AgentContext`` — forcing the roster's
  ``run(dataclass) -> dataclass`` parallel-safe contract through it would be
  adapter mush, not reuse. The roster reuses the swarm's *model-call
  pattern* (``router.chat`` with ``Message``/``SamplingParams``) instead.
* The roster is **not** wired into ``tools/agents.py``'s
  ``AGENT_TOOL_MODULES``: that list is for modules exposing a
  ``register(registry)`` hook of *tool* functions. These subagents are L5
  orchestration workers, not registry tools; the natural registration is
  the :data:`ROSTER` dict below.

**Honesty rules.** A model failure never becomes a silent fake: the planner
falls back to an explicit single-file plan (``fallback=True``, logged);
the reviewer degrades to rule-based-only (``model_review="skipped"``); the
tester/refactorer/implementer/api-designer return ``ok=False`` results with
the reason instead of pretending. The tester never claims a pass without
running the tests; the refactorer reverts its own edits when post-refactor
tests newly fail.
"""

from __future__ import annotations

import abc
import ast
import builtins
import concurrent.futures
import dataclasses
import difflib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import Message, SamplingParams
from ..tools.edit_loop import EditConflictError, EditLoop
from ..tools.pytest_runner import run_tests
from .role_specs import BUILTIN_ROLES, RoleSpec

__all__ = [
    "ROSTER",
    "MERGERS",
    "Subagent",
    "Planner", "PlanInput", "PlanItem", "PlanResult", "RepoFile",
    "Implementer", "ImplementInput", "ImplementResult", "EditOutcome",
    "Reviewer", "ReviewInput", "ReviewResult", "Flaw", "FlawCategory",
    "Tester", "TestInput", "TesterResult",
    "Refactorer", "RefactorInput", "RefactorResult",
    "DepHunter", "DepInput", "DepResult",
    "ApiDesigner", "ApiDesignInput", "ApiDesignResult",
    "AgentRun", "FusedResult",
    "run_parallel",
]

_log = get_logger(__name__)

#: Hard cap on planner output; the mission bounds the plan at 12 files.
MAX_PLAN_FILES = 12

#: Severities, ordered.
SEVERITIES = ("low", "medium", "high", "critical")
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


# ── shared helpers ─────────────────────────────────────────────────────────


def _safe_rel(root: Path, rel: str) -> str:
    """Normalize a repo-relative path and reject escapes.

    Raises ValueError on absolute paths, ``..`` escapes, or empty paths.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("empty path")
    rel = rel.strip()
    if os.path.isabs(rel):
        raise ValueError(f"absolute path not allowed: {rel!r}")
    norm = os.path.normpath(rel).replace(os.path.sep, "/")
    if norm == ".." or norm.startswith("../"):
        raise ValueError(f"path escapes the project root: {rel!r}")
    return norm


def _dotted_path(rel: str) -> str:
    """``nomorals/tools/x.py`` -> ``nomorals.tools.x`` (for imports)."""
    return Path(rel).with_suffix("").as_posix().replace("/", ".")


class Subagent(abc.ABC):
    """Base for the roster: per-mission instance, dataclass in/out.

    ``router`` is anything with ``.chat(messages, params)`` returning an
    object with ``.ok`` / ``.text`` / ``.error`` (the LLMRouter, or a
    provider such as the mock used in tests). ``None`` means "no model":
    roles degrade explicitly rather than failing.

    Abstract: instantiating the bare base is a ``TypeError`` at
    construction time, not a ``NotImplementedError`` at call time — a
    subagent that cannot ``run`` must fail fast.
    """

    roster_type: str = "base"
    #: Shared contract spec from BUILTIN_ROLES; never mutated.
    role_spec: RoleSpec | None = None

    def __init__(self, router: Any = None, project_root: str = ".") -> None:
        self.router = router
        self.project_root = Path(project_root).expanduser().resolve()

    # ── model ────────────────────────────────────────────────────────────
    def _model_json(self, system: str, user: str, *,
                    max_tokens: int = 2048,
                    temperature: float = 0.2) -> tuple[Any | None, str]:
        """One model call; returns ``(data, error)`` — error "" on success.

        The Brain's ``chat_json`` owns the parse + repair loop now; the
        old hand-rolled ``_extract_json`` + one-shot parse is gone.
        """
        if self.router is None:
            return None, "no model available"
        try:
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import Brain

            brain = self.router if isinstance(self.router, Brain) else Brain(router=self.router)
            data, response = brain.chat_json(
                [Message.system(system),
                 Message.user(user)],
                task_kind="plan",
                params=SamplingParams(temperature=temperature,
                                      max_tokens=max_tokens, json_mode=True),
            )
        except Exception as exc:  # noqa: BLE001 - model failures are results
            return None, f"model call failed: {exc}"
        if not isinstance(data, dict):
            err = getattr(response, "error", "") or "no JSON in model output"
            return None, f"model error: {err[:200]}"
        return data, ""

    # ── paths ────────────────────────────────────────────────────────────
    def _inside(self, rel: str) -> Path:
        """Repo-relative path validated to stay inside the project root."""
        return self.project_root / _safe_rel(self.project_root, rel)

    @abc.abstractmethod
    def run(self, inp: Any) -> Any:
        """Run one mission input and return its result dataclass.

        Every concrete role implements this; failures are returned as
        explicit ``ok=False`` results, never raised — except for
        programming errors (bad input types, broken project root), which
        fail fast here in the caller via :func:`run_parallel`.
        """


# ── 1. Planner ───────────────────────────────────────────────────────────


@dataclass
class RepoFile:
    path: str
    purpose: str = ""


@dataclass
class PlanItem:
    path: str
    why: str
    new_file: bool
    acceptance: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class PlanInput:
    task: str
    repo_map: list[Any] = field(default_factory=list)  # RepoFile | dict | tuple
    max_files: int = MAX_PLAN_FILES

    def files(self) -> list[RepoFile]:
        out: list[RepoFile] = []
        for entry in self.repo_map:
            if isinstance(entry, RepoFile):
                out.append(entry)
            elif isinstance(entry, dict):
                out.append(RepoFile(path=str(entry.get("path", "")),
                                    purpose=str(entry.get("purpose", ""))))
            elif isinstance(entry, (tuple, list)) and entry:
                out.append(RepoFile(
                    path=str(entry[0]),
                    purpose=str(entry[1]) if len(entry) > 1 else ""))
        return out


@dataclass
class PlanResult:
    items: list[PlanItem]
    fallback: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Planner(Subagent):
    """Task + repo map -> bounded file-level plan.

    Falls back to an explicit single-file plan (``fallback=True``, logged)
    when there is no model or the model gives nothing usable — never a
    silent empty plan.
    """

    roster_type = "planner"
    role_spec = BUILTIN_ROLES["architect"]

    _SYSTEM = ("You are a software architect. Produce minimal, file-level "
               "implementation plans. Prefer editing existing files over "
               "creating new ones.")

    def _fallback(self, inp: PlanInput, reason: str) -> PlanResult:
        _log.warning("planner fallback (%s)", reason)
        files = inp.files()
        if files:
            first = files[0]
            try:
                path = _safe_rel(self.project_root, first.path)
            except ValueError:
                path = "notes/plan.md"
                item = PlanItem(path=path, why=reason, new_file=True,
                                acceptance="human review")
            else:
                item = PlanItem(
                    path=path, new_file=False,
                    why=(f"fallback: {reason}; begin at the most relevant "
                         "mapped file"),
                    acceptance="human review")
        else:
            item = PlanItem(path="notes/plan.md", why=reason, new_file=True,
                            acceptance="human review")
        return PlanResult(items=[item], fallback=True, note=reason)

    def run(self, inp: PlanInput) -> PlanResult:
        max_files = max(1, min(int(inp.max_files or MAX_PLAN_FILES),
                               MAX_PLAN_FILES))
        repo_lines = "\n".join(
            f"- {f.path} — {f.purpose}" for f in inp.files()) or "(no repo map)"
        data, error = self._model_json(
            self._SYSTEM,
            f"Produce a file-level implementation plan.\n\nTask: {inp.task}\n\n"
            f"Repository map (path — purpose):\n{repo_lines}\n\n"
            f"Return JSON: {{\"items\": [{{\"path\": \"repo/relative/path.py\", "
            f"\"why\": \"...\", \"new_file\": true, \"acceptance\": \"how to "
            f"verify\"}}]}}.\nRules: at most {max_files} items; every path "
            f"must stay inside the project root (no \"..\", no absolute "
            f"paths); keep the plan minimal.")
        if error:
            return self._fallback(inp, error)
        raw_items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(raw_items, list) or not raw_items:
            return self._fallback(inp, "model returned no plan items")
        items: list[PlanItem] = []
        seen: set[str] = set()
        dropped = 0
        for raw in raw_items:
            if not isinstance(raw, dict):
                dropped += 1
                continue
            try:
                path = _safe_rel(self.project_root, str(raw.get("path", "")))
            except ValueError:
                dropped += 1
                continue
            if path in seen:
                dropped += 1
                continue
            seen.add(path)
            items.append(PlanItem(
                path=path,
                why=str(raw.get("why", ""))[:500],
                new_file=bool(raw.get("new_file", False)),
                acceptance=str(raw.get("acceptance", ""))[:500]))
        if not items:
            return self._fallback(inp, "no usable plan items after validation")
        truncated = len(items) > max_files
        items = items[:max_files]
        note = ""
        if dropped:
            note += f"dropped {dropped} invalid/duplicate item(s). "
        if truncated:
            note += f"truncated to the {max_files}-file bound."
        return PlanResult(items=items, note=note.strip())

# ── 2. Implementer ───────────────────────────────────────────────────────


@dataclass
class ImplementInput:
    path: str
    instruction: str
    file_content: str = ""   # current content; "" for new files
    new_file: bool = False
    context: str = ""        # extra context (neighbor files, error text)


@dataclass
class EditOutcome:
    old_text: str
    new_text: str
    applied: bool
    diff: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ImplementResult:
    path: str
    edits: list[EditOutcome]
    ok: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Implementer(Subagent):
    """Plan item + instruction -> surgical edits via edit_loop.

    All edits go through :meth:`EditLoop.surgical_replace_many` (atomic:
    all-or-nothing, exact-once per edit, syntax-checked). On success every
    edit is marked applied with the combined diff; on failure the file is
    untouched and each edit gets its failure reason from the same module's
    ``preview_replace`` match data — no second matcher, no reimplementation.
    """

    roster_type = "implementer"
    role_spec = BUILTIN_ROLES["coder"]

    _SYSTEM = ("You are a coding specialist. Make small, surgical edits. "
               "Match old_text EXACTLY (character for character, including "
               "indentation) or the edit cannot apply.")

    def run(self, inp: ImplementInput) -> ImplementResult:
        try:
            rel = _safe_rel(self.project_root, inp.path)
        except ValueError as exc:
            return ImplementResult(path=inp.path, edits=[], ok=False,
                                   note=str(exc))
        target = self.project_root / rel
        if inp.new_file:
            return self._create_file(rel, target, inp)
        return self._edit_file(rel, target, inp)

    # ── new files ──────────────────────────────────────────────────────
    def _create_file(self, rel: str, target: Path,
                     inp: ImplementInput) -> ImplementResult:
        data, error = self._model_json(
            self._SYSTEM,
            f"Write the complete contents of the NEW file `{rel}`.\n\n"
            f"Purpose / instruction: {inp.instruction}\n"
            + (f"Extra context:\n{inp.context}\n" if inp.context else "")
            + "Return JSON: {\"content\": \"<complete file text>\"}.")
        if error:
            return ImplementResult(path=rel, edits=[], ok=False, note=error)
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, str) or not content.strip():
            return ImplementResult(path=rel, edits=[], ok=False,
                                   note="model returned no file content")
        if not content.endswith("\n"):
            content += "\n"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ImplementResult(path=rel, edits=[], ok=False,
                                   note=f"write failed: {exc}")
        diff = "".join(difflib.unified_diff(
            [], content.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        outcome = EditOutcome(old_text="", new_text=content, applied=True,
                              diff=diff)
        return ImplementResult(path=rel, edits=[outcome], ok=True,
                               note="created new file")

    # ── existing files ─────────────────────────────────────────────────
    def _edit_file(self, rel: str, target: Path,
                   inp: ImplementInput) -> ImplementResult:
        if not target.is_file():
            return ImplementResult(
                path=rel, edits=[], ok=False,
                note=f"file not found: {rel} (pass new_file=True to create it)")
        current = (inp.file_content if inp.file_content
                   else target.read_text(encoding="utf-8", errors="ignore"))
        data, error = self._model_json(
            self._SYSTEM,
            f"Propose surgical edits for the file `{rel}`.\n\n"
            f"Instruction: {inp.instruction}\n"
            + (f"Extra context:\n{inp.context}\n" if inp.context else "")
            + f"Current file contents:\n```\n{current[:12000]}\n```\n\n"
            f"Return JSON: {{\"edits\": [{{\"old_text\": \"<exact existing "
            f"text>\", \"new_text\": \"<replacement>\"}}]}}. Every old_text "
            f"must occur exactly once in the file above.")
        if error:
            return ImplementResult(path=rel, edits=[], ok=False, note=error)
        raw_edits = data.get("edits") if isinstance(data, dict) else None
        if not isinstance(raw_edits, list) or not raw_edits:
            return ImplementResult(path=rel, edits=[], ok=False,
                                   note="model returned no usable edits")
        pairs: list[tuple[str, str]] = []
        for raw in raw_edits:
            if (isinstance(raw, dict)
                    and isinstance(raw.get("old_text"), str)
                    and isinstance(raw.get("new_text"), str)):
                pairs.append((raw["old_text"], raw["new_text"]))
        if not pairs:
            return ImplementResult(path=rel, edits=[], ok=False,
                                   note="model returned no usable edits")
        loop = EditLoop(agent=None, project_root=str(self.project_root),
                        auto_backup=True)
        try:
            diff = loop.surgical_replace_many(rel, pairs)
        except FileNotFoundError as exc:
            return ImplementResult(path=rel, edits=[], ok=False, note=str(exc))
        except (EditConflictError, ValueError) as exc:
            return self._attribute_failures(rel, pairs, str(exc))
        outcomes = [EditOutcome(old_text=o, new_text=n, applied=True,
                                diff=diff)
                    for o, n in pairs]
        return ImplementResult(path=rel, edits=outcomes, ok=True,
                               note=f"applied {len(pairs)} edit(s)")

    def _attribute_failures(self, rel: str,
                            pairs: list[tuple[str, str]],
                            reason: str) -> ImplementResult:
        """Atomic batch failed: the file is untouched. Attribute the failure
        to the specific edit named in the canonical error (the sibling
        machinery labels failures ``edit N: ...``); every edit in a
        rejected batch is reported as not applied — honestly, since
        all-or-nothing means none landed."""
        culprit: int | None = None
        match = re.search(r"edit (\d+)", reason)
        if match:
            try:
                culprit = int(match.group(1))
            except ValueError:
                culprit = None
        outcomes: list[EditOutcome] = []
        for i, (old, new) in enumerate(pairs):
            if culprit is not None and i == culprit:
                error = reason[:300]
            else:
                error = ("not applied: batch rejected atomically, file "
                         "untouched")
            outcomes.append(EditOutcome(old_text=old, new_text=new,
                                        applied=False, error=error))
        return ImplementResult(
            path=rel, edits=outcomes, ok=False,
            note=f"atomic batch failed, file untouched: {reason[:300]}")


# ── 3. Reviewer ──────────────────────────────────────────────────────────


class FlawCategory(str, Enum):
    """Fixed flaw categories. Anything outside this set is dropped."""
    UNDEFINED_NAME = "undefined-name"
    WRONG_INDEX = "wrong-index"
    WRONG_OPERATOR = "wrong-operator"
    UNHANDLED_EDGE = "unhandled-edge"
    CONTRACT_VIOLATION = "contract-violation"
    SECURITY_RISK = "security-risk"
    DEAD_CODE = "dead-code"


@dataclass
class Flaw:
    category: str   # one of FlawCategory's values
    file: str
    line: int       # new-file line number (0 when unknown)
    severity: str   # low | medium | high | critical
    message: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ReviewInput:
    diff: str  # unified diff


@dataclass
class ReviewResult:
    flaws: list[Flaw]
    model_review: str = "ok"  # "ok" | "skipped"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _ParsedFile:
    path: str
    is_new: bool
    added: set[int] = field(default_factory=set)  # new-file line numbers
    full_new: str | None = None  # full new content (new files only)


_DIFF_HUNK = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _parse_diff(diff: str) -> list[_ParsedFile]:
    """Parse a unified diff into per-file added-line sets.

    New files (``--- /dev/null``) are reconstructed in full, so analysis on
    them is exact.
    """
    files: list[_ParsedFile] = []
    cur: _ParsedFile | None = None
    old_side = ""
    new_lineno = 0
    for raw in (diff or "").splitlines():
        if raw.startswith("--- "):
            old_side = raw[4:].strip()
        elif raw.startswith("+++ "):
            new_side = raw[4:].strip()
            path = new_side[2:] if new_side.startswith("b/") else new_side
            cur = _ParsedFile(path=path, is_new=(old_side == "/dev/null"),
                              full_new="" if old_side == "/dev/null" else None)
            files.append(cur)
        elif raw.startswith("@@ ") and cur is not None:
            match = _DIFF_HUNK.match(raw)
            new_lineno = int(match.group(1)) if match else 0
        elif cur is not None:
            if raw.startswith("\\"):
                continue  # "\ No newline at end of file"
            if raw.startswith("+"):
                cur.added.add(new_lineno)
                if cur.full_new is not None:
                    cur.full_new += raw[1:] + "\n"
                new_lineno += 1
            elif raw.startswith("-"):
                continue
            else:  # context line
                text = raw[1:] if raw.startswith(" ") else raw
                if cur.full_new is not None:
                    cur.full_new += text + "\n"
                new_lineno += 1
    return files


class _Scope:
    __slots__ = ("parent", "defined", "used", "global_names", "star")

    def __init__(self, parent: "_Scope | None" = None) -> None:
        self.parent = parent
        self.defined: set[str] = set()
        self.used: list[tuple[str, int]] = []  # (name, lineno)
        self.global_names: set[str] = set()
        self.star = False  # `from x import *` seen: names may come from it


class _NameVisitor(ast.NodeVisitor):
    """Pyflakes-style undefined-name / unused-import detection.

    Scope-aware: functions, classes, lambdas and comprehensions get their
    own scopes; names resolve outward through enclosing scopes. ``global``
    uses resolve at module level. A ``from x import *`` in a scope
    suppresses undefined-name reports for that scope (the names genuinely
    cannot be known).
    """

    def __init__(self) -> None:
        self.module = _Scope()
        self.scope = self.module
        self.all_scopes = [self.module]
        self.imports: list[tuple[str, int]] = []  # (bound name, lineno)
        self.dunder_all: set[str] = set()

    # ── scope plumbing ─────────────────────────────────────────────────
    def _push(self) -> None:
        child = _Scope(parent=self.scope)
        self.all_scopes.append(child)
        self.scope = child

    def _pop(self) -> None:
        assert self.scope.parent is not None
        self.scope = self.scope.parent

    def _define(self, name: str) -> None:
        self.scope.defined.add(name)

    def _use(self, name: str, lineno: int) -> None:
        if name in self.scope.global_names:
            self.module.used.append((name, lineno))
        else:
            self.scope.used.append((name, lineno))

    def _bind_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._define(target.id)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt)
        else:
            self.visit(target)  # attribute/subscript: uses its value

    # ── names ──────────────────────────────────────────────────────────
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._define(node.id)
        else:  # Load and Del both require the name to exist
            self._use(node.id, node.lineno)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.global_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = (alias.asname or alias.name).split(".")[0]
            self._define(bound)
            self.imports.append((bound, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(a.name == "*" for a in node.names):
            self.scope.star = True
            return
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            self._define(bound)
            self.imports.append((bound, node.lineno))

    # ── definitions: defaults/decorators run in the enclosing scope ────
    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef
                        ) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        args = node.args
        arg_nodes = (list(args.posonlyargs) + list(args.args)
                     + list(args.kwonlyargs))
        for arg in arg_nodes:
            if arg.annotation is not None:
                self.visit(arg.annotation)
        for opt in (args.vararg, args.kwarg):
            if opt is not None and opt.annotation is not None:
                self.visit(opt.annotation)
        for default in list(args.defaults) + [d for d in args.kw_defaults
                                              if d is not None]:
            self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self._define(node.name)
        self._push()
        for arg in arg_nodes:
            self._define(arg.arg)
        if args.vararg is not None:
            self._define(args.vararg.arg)
        if args.kwarg is not None:
            self._define(args.kwarg.arg)
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        for default in list(args.defaults) + [d for d in args.kw_defaults
                                              if d is not None]:
            self.visit(default)
        self._push()
        for arg in (list(args.posonlyargs) + list(args.args)
                    + list(args.kwonlyargs)):
            self._define(arg.arg)
        if args.vararg is not None:
            self._define(args.vararg.arg)
        if args.kwarg is not None:
            self._define(args.kwarg.arg)
        self.visit(node.body)
        self._pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw.value)
        self._define(node.name)
        self._push()
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    def _visit_comp(self, node: ast.AST) -> None:
        gens = node.generators  # type: ignore[attr-defined]
        if gens:
            self.visit(gens[0].iter)  # outermost iter: enclosing scope
        self._push()
        for i, gen in enumerate(gens):
            if i:
                self.visit(gen.iter)
            self._bind_target(gen.target)
            for cond in gen.ifs:
                self.visit(cond)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)  # type: ignore[attr-defined]
        self._pop()

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_GeneratorExp = _visit_comp
    visit_DictComp = _visit_comp

    # ── statements ─────────────────────────────────────────────────────
    def visit_Assign(self, node: ast.Assign) -> None:
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    self.dunder_all.add(elt.value)
        self.visit(node.value)
        for target in node.targets:
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.annotation is not None:
            self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._use(node.target.id, node.target.lineno)
            self._define(node.target.id)
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._define(node.target.id)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._define(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            bound: list[str] = []
            self._pattern_names(case.pattern, bound)
            for name in bound:
                self._define(name)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)

    def _pattern_names(self, pattern: ast.AST, out: list[str]) -> None:
        for child in ast.walk(pattern):
            if isinstance(child, (ast.MatchAs, ast.MatchStar)):
                if child.name:
                    out.append(child.name)
            elif isinstance(child, ast.MatchMapping):
                if child.rest:
                    out.append(child.rest)

    # ── results ────────────────────────────────────────────────────────
    def undefined_names(self) -> list[tuple[str, int]]:
        """(name, lineno) used but not defined in any enclosing scope."""
        builtin_names = set(dir(builtins)) | {
            "__name__", "__file__", "__doc__", "__package__", "__spec__",
            "__loader__", "__cached__", "__annotations__", "__debug__",
        }
        flaws: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for scope in self.all_scopes:
            for name, lineno in scope.used:
                if name in builtin_names or (name, lineno) in seen:
                    continue
                found = False
                cursor: _Scope | None = scope
                while cursor is not None:
                    if name in cursor.defined or cursor.star:
                        found = True
                        break
                    cursor = cursor.parent
                if not found:
                    seen.add((name, lineno))
                    flaws.append((name, lineno))
        return flaws

    def unused_imports(self) -> list[tuple[str, int]]:
        """(name, lineno) imported but never used.

        Only actual *uses* (loads) count — the import binding itself being
        defined must not count, or nothing would ever be reported.
        """
        used: set[str] = set()
        for scope in self.all_scopes:
            used.update(name for name, _ in scope.used)
        out = []
        for name, lineno in self.imports:
            if name not in used and name not in self.dunder_all:
                out.append((name, lineno))
        return out


class Reviewer(Subagent):
    """Unified diff -> flaws.

    Rule-based pass (real detection, not vibes): syntax errors via
    ``compile`` and undefined names / unused imports via a scope-aware AST
    walk, attributed only to added lines. A syntax error maps to the fixed
    ``contract-violation`` category (critical) since the fixed category set
    has no syntax bucket. Model pass covers the remaining categories
    (wrong-index, wrong-operator, unhandled-edge, contract-violation,
    security-risk, dead-code); when the model is unavailable or fails, the
    result degrades to rule-based-only with ``model_review="skipped"``.
    """

    roster_type = "reviewer"
    role_spec = BUILTIN_ROLES["critic"]

    _SYSTEM = ("You are an adversarial code reviewer. Find real defects "
               "only — no style nits, no speculation. Every flaw must point "
               "at a concrete line.")

    _MODEL_CATEGORIES = ("wrong-index", "wrong-operator", "unhandled-edge",
                         "contract-violation", "security-risk", "dead-code")

    def run(self, inp: ReviewInput) -> ReviewResult:
        files = _parse_diff(inp.diff)
        if not files:
            return ReviewResult(flaws=[], model_review="skipped",
                                note="empty or unparseable diff")
        flaws: list[Flaw] = []
        for parsed in files:
            flaws.extend(self._rule_review(parsed))
        model_flaws, model_status, model_note = self._model_review(inp.diff)
        flaws.extend(model_flaws)
        flaws = self._dedupe(flaws)
        flaws.sort(key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0),
                                  f.file, f.line, f.message))
        note = model_note
        return ReviewResult(flaws=flaws, model_review=model_status, note=note)

    # ── rule-based pass ────────────────────────────────────────────────
    def _rule_review(self, parsed: _ParsedFile) -> list[Flaw]:
        if not parsed.path.endswith(".py"):
            return []
        content, full = self._file_content(parsed)
        if content is None:
            return []
        flaws: list[Flaw] = []
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            line = exc.lineno or 0
            if not parsed.added or line in parsed.added:
                flaws.append(Flaw(
                    category=FlawCategory.CONTRACT_VIOLATION.value,
                    file=parsed.path, line=line, severity="critical",
                    message=f"syntax error: {exc.msg}"))
            return flaws
        visitor = _NameVisitor()
        visitor.visit(tree)
        if full:
            added = parsed.added
            sev = "high"
        else:
            # Partial-file analysis: only the hunk text is visible, so an
            # undefined name might be defined elsewhere in the file. Report
            # at low severity; skip unused-import detection entirely (it
            # cannot be sound without the whole file).
            added = set(range(1, len(content.splitlines()) + 1))
            sev = "low"
        for name, lineno in visitor.undefined_names():
            if lineno in added:
                flaws.append(Flaw(
                    category=FlawCategory.UNDEFINED_NAME.value,
                    file=parsed.path, line=lineno, severity=sev,
                    message=f"undefined name {name!r}"))
        if full:
            for name, lineno in visitor.unused_imports():
                if lineno in added:
                    flaws.append(Flaw(
                        category=FlawCategory.DEAD_CODE.value,
                        file=parsed.path, line=lineno, severity="low",
                        message=f"imported but unused: {name}"))
        return flaws

    def _file_content(self, parsed: _ParsedFile
                      ) -> tuple[str | None, bool]:
        """(content, full). full=False means partial hunk-only analysis."""
        if parsed.is_new and parsed.full_new is not None:
            return parsed.full_new, True
        on_disk = self.project_root / parsed.path
        if on_disk.is_file():
            try:
                return on_disk.read_text(encoding="utf-8", errors="ignore"), True
            except OSError:
                return None, False
        return None, False

    # ── model pass ─────────────────────────────────────────────────────
    def _model_review(self, diff: str
                      ) -> tuple[list[Flaw], str, str]:
        data, error = self._model_json(
            self._SYSTEM,
            f"Review the following unified diff for real defects. Categories "
            f"(use exactly one): {', '.join(self._MODEL_CATEGORIES)}. "
            f"Severity: low, medium, high, critical.\n\n"
            f"Return JSON: {{\"flaws\": [{{\"category\": \"...\", "
            f"\"file\": \"...\", \"line\": <new-file line number>, "
            f"\"severity\": \"...\", \"message\": \"...\"}}]}}.\n\n"
            f"Diff:\n{diff[:12000]}")
        if error:
            return [], "skipped", f"model review skipped: {error}"
        raw_flaws = data.get("flaws") if isinstance(data, dict) else None
        if not isinstance(raw_flaws, list):
            return [], "skipped", "model review skipped: no flaws list"
        flaws: list[Flaw] = []
        dropped = 0
        coerced = 0
        for raw in raw_flaws:
            if not isinstance(raw, dict):
                dropped += 1
                continue
            category = str(raw.get("category", ""))
            if category not in FlawCategory._value2member_map_:
                dropped += 1
                continue
            severity = str(raw.get("severity", "low"))
            if severity not in _SEVERITY_RANK:
                severity = "low"
                coerced += 1
            try:
                line = int(raw.get("line", 0))
            except (TypeError, ValueError):
                line = 0
            flaws.append(Flaw(category=category,
                              file=str(raw.get("file", ""))[:200],
                              line=line, severity=severity,
                              message=str(raw.get("message", ""))[:500]))
        note = ""
        if dropped:
            note += f"dropped {dropped} model flaw(s) with invalid category. "
        if coerced:
            note += f"coerced {coerced} invalid severit(ies) to low."
        return flaws, "ok", note.strip()

    @staticmethod
    def _dedupe(flaws: list[Flaw]) -> list[Flaw]:
        seen: set[tuple[str, str, int, str]] = set()
        out: list[Flaw] = []
        for flaw in flaws:
            key = (flaw.category, flaw.file, flaw.line, flaw.message)
            if key not in seen:
                seen.add(key)
                out.append(flaw)
        return out

# ── 4. Tester ──────────────────────────────────────────────────────────


def _count_test_cases(code: str) -> int:
    """Number of test cases defined in a test file (AST-counted).

    Counts ``test*`` methods on ``TestCase`` subclasses plus bare
    ``test*`` functions (which pytest collects). Zero means the file
    verifies nothing, however cleanly it "passes".
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
                (b.attr if isinstance(b, ast.Attribute) else
                 getattr(b, "id", "")) == "TestCase" for b in node.bases):
            count += sum(1 for item in node.body
                         if isinstance(item, ast.FunctionDef)
                         and item.name.startswith("test"))
    count += sum(1 for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name.startswith("test"))
    return count


@dataclass
class TestInput:
    changed_files: list[str] = field(default_factory=list)  # repo-relative
    focus: str = ""  # extra instruction, e.g. "cover the new retry path"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TesterResult:
    ok: bool
    tests_written: list[str] = field(default_factory=list)
    discarded: list[dict[str, str]] = field(default_factory=list)
    passed: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)
    errors: int = 0
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Tester(Subagent):
    """Changed files -> generated, syntax-checked, actually-run tests.

    The model writes focused unittest suites; each candidate is
    compile-checked, then collection-checked (discarded with a reported
    reason when it does not even collect), then run via
    ``tools.pytest_runner.run_tests``. ``ok`` is True only when at least
    one generated suite collected, ran, and produced zero failures/errors.
    A pass is never claimed without running.
    """

    roster_type = "tester"
    role_spec = BUILTIN_ROLES["tester"]

    #: Max changed files handled per run (bounded work).
    MAX_FILES = 6
    #: Where generated suites live: a scratch dir outside the real tests/
    #: tree (kept out of the committed suite; the result records what ran).
    #: No leading dot — the unittest fallback needs an importable module.
    GEN_DIR = "subagent_tests"

    _SYSTEM = ("You are a testing specialist. Write small, deterministic "
               "unittest suites. No network, no randomness, no sleeps, no "
               "files outside tmp.")

    def run(self, inp: TestInput) -> TesterResult:
        if self.router is None:
            return TesterResult(ok=False,
                                note="no model available — no tests generated")
        gen_dir = self.project_root / self.GEN_DIR
        gen_dir.mkdir(parents=True, exist_ok=True)
        result = TesterResult(ok=False)
        ran_any = False
        for rel in inp.changed_files[:self.MAX_FILES]:
            try:
                norm = _safe_rel(self.project_root, rel)
            except ValueError as exc:
                result.note += f"skipped {rel!r}: {exc}. "
                continue
            src = self.project_root / norm
            if not src.is_file() or src.suffix != ".py":
                result.note += f"skipped {norm!r}: not a Python file. "
                continue
            code, gen_error = self._generate(norm, src, inp.focus)
            stem = f"test_subgen_{Path(norm).stem}_{new_short_id()[:8]}"
            if gen_error:
                result.discarded.append({"name": stem + ".py",
                                         "reason": gen_error})
                continue
            assert code is not None
            try:
                compile(code, stem + ".py", "exec")
            except SyntaxError as exc:
                result.discarded.append(
                    {"name": stem + ".py",
                     "reason": f"syntax error: {exc.msg} (line {exc.lineno})"})
                continue
            test_path = gen_dir / (stem + ".py")
            test_path.write_text(code, encoding="utf-8")
            rel_test = str(test_path.relative_to(self.project_root))
            collect_error = self._check_collects(rel_test, code)
            if collect_error:
                result.discarded.append({"name": rel_test,
                                         "reason": collect_error})
                continue
            ran = run_tests(paths=[rel_test], repo=str(self.project_root),
                            timeout=120.0)
            if ran.get("ran") == 0 and not ran.get("failed"):
                # unittest fallback ran nothing (e.g. pytest-style bare
                # functions with no pytest installed): not a verification.
                result.discarded.append(
                    {"name": rel_test,
                     "reason": "no tests ran (nothing collected)"})
                continue
            ran_any = True
            result.seconds += float(ran.get("seconds", 0.0))
            result.tests_written.append(rel_test)
            result.passed += int(ran.get("passed", 0))
            result.errors += int(ran.get("errors", 0))
            for failure in ran.get("failed", []) or []:
                failure = dict(failure)
                failure["suite"] = rel_test
                result.failed.append(failure)
            if not ran.get("ok"):
                result.note += (f"suite {rel_test} did not pass cleanly. ")
        if not result.tests_written and not result.discarded:
            result.note += "nothing to test: no usable Python files given. "
        result.ok = (ran_any and not result.failed and result.errors == 0
                     and bool(result.tests_written))
        if not result.ok and ran_any and not result.failed:
            result.note += "some suites were discarded before running. "
        result.note = result.note.strip()
        return result

    def _generate(self, rel: str, src: Path,
                  focus: str) -> tuple[str | None, str]:
        """Ask the model for a test file. Returns (code, error)."""
        try:
            tree = ast.parse(src.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            tree = None
        names: list[str] = []
        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)) and not node.name.startswith("_"):
                    names.append(node.name)
        dotted = _dotted_path(rel)
        data, error = self._model_json(
            self._SYSTEM,
            f"Write a focused unittest suite for the Python module "
            f"`{dotted}` (file `{rel}`).\n"
            f"Public callables to cover: "
            f"{', '.join(names) if names else 'all public functions/classes'}.\n"
            + (f"Extra focus: {focus}\n" if focus else "")
            + f"Rules: use unittest.TestCase classes, runnable by pytest; "
            f"import the module as `import {dotted} as _target`; small "
            f"deterministic cases only; keep the file under 150 lines.\n"
            f"Return JSON: {{\"code\": \"<complete test file text>\"}}.")
        if error:
            return None, error
        code = data.get("code") if isinstance(data, dict) else None
        if not isinstance(code, str) or not code.strip():
            return None, "model returned no test code"
        return code, ""

    def _check_collects(self, rel_test: str, code: str) -> str:
        """Verify the generated suite collects; '' when it does.

        Three gates: (1) the file must define at least one test case
        (AST-counted, so a file with no tests is discarded rather than
        "passing" vacuously); (2) with pytest, ``--collect-only``; without
        pytest, an import of the test module in a subprocess (catches
        collection-time import errors the same way).
        """
        if _count_test_cases(code) == 0:
            return "no test cases found in generated file"
        dotted = _dotted_path(rel_test)
        if importlib.util.find_spec("pytest") is not None:
            cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                   rel_test]
        else:
            cmd = [sys.executable, "-c", f"import {dotted}"]
        try:
            proc = subprocess.run(cmd, cwd=str(self.project_root),
                                  capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "collection timed out after 60s"
        except OSError as exc:
            return f"could not run collection check: {exc}"
        if proc.returncode == 0:
            return ""
        if proc.returncode == 5:
            return "no tests collected"
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-800:]
        return f"collection failed: {tail}"


# ── 5. Refactorer ──────────────────────────────────────────────────────


@dataclass
class RefactorInput:
    path: str   # repo-relative file to refactor
    goal: str   # e.g. "extract the retry loop into a helper"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RefactorResult:
    path: str
    applied: bool
    diff: str = ""
    tests_before: dict[str, Any] = field(default_factory=dict)
    tests_after: dict[str, Any] = field(default_factory=dict)
    reverted: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _tests_for_module(root: Path, rel: str) -> list[str]:
    """Test files covering a module: the stem-matching heuristic from
    ``tools.pytest_runner.select_changed_tests``, applied to one file
    instead of the git working tree."""
    p = Path(rel)
    if "test" in p.name.lower() and p.suffix == ".py":
        return [rel] if (root / rel).is_file() else []
    stem = p.stem
    if not stem or stem.startswith("_"):
        return []
    tdir = root / "tests"
    if not tdir.is_dir():
        return []
    return sorted(str(tf.relative_to(root)) for tf in tdir.rglob("test_*.py")
                  if stem in tf.stem)


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(run.get("ok")),
        "passed": int(run.get("passed", 0)),
        "failed": [f.get("test_id", "?") for f in run.get("failed", []) or []],
        "errors": int(run.get("errors", 0)),
        "seconds": float(run.get("seconds", 0.0)),
        "note": str(run.get("note", "") or ""),
    }


class Refactorer(Subagent):
    """File + goal -> behavior-preserving surgical edits.

    Runs the file's existing tests BEFORE and AFTER via
    ``tools.pytest_runner.run_tests``. When the post-refactor run has
    *newly* failing tests (or new errors) compared to before, the edits are
    reverted from the in-memory snapshot and the result reports
    ``reverted=True`` honestly instead of claiming success.
    """

    roster_type = "refactorer"
    role_spec = BUILTIN_ROLES["coder"]

    _SYSTEM = ("You are a refactoring specialist. Preserve behavior "
               "EXACTLY: no logic changes, no API changes, no new "
               "dependencies. Match old_text EXACTLY or the edit cannot "
               "apply.")

    def run(self, inp: RefactorInput) -> RefactorResult:
        try:
            rel = _safe_rel(self.project_root, inp.path)
        except ValueError as exc:
            return RefactorResult(path=inp.path, applied=False, note=str(exc))
        target = self.project_root / rel
        if not target.is_file():
            return RefactorResult(path=rel, applied=False,
                                  note=f"file not found: {rel}")
        if self.router is None:
            return RefactorResult(path=rel, applied=False,
                                  note="no model available — no refactor "
                                       "proposed")
        test_files = _tests_for_module(self.project_root, rel)
        before = self._run_tests(test_files)
        original = target.read_text(encoding="utf-8", errors="ignore")
        data, error = self._model_json(
            self._SYSTEM,
            f"Propose surgical refactor edits for `{rel}`.\n\n"
            f"Goal: {inp.goal}\n\n"
            f"Current file:\n```\n{original[:12000]}\n```\n\n"
            f"Return JSON: {{\"edits\": [{{\"old_text\": \"<exact existing "
            f"text>\", \"new_text\": \"<replacement>\"}}]}}. Behavior must "
            f"be identical before and after.")
        result = RefactorResult(path=rel, applied=False,
                                tests_before=_summarize_run(before))
        if error:
            result.note = error
            result.tests_after = result.tests_before
            return result
        raw_edits = data.get("edits") if isinstance(data, dict) else None
        pairs = [(e["old_text"], e["new_text"]) for e in raw_edits
                 if isinstance(e, dict)
                 and isinstance(e.get("old_text"), str)
                 and isinstance(e.get("new_text"), str)] \
            if isinstance(raw_edits, list) else []
        if not pairs:
            result.note = "model returned no usable edits"
            result.tests_after = result.tests_before
            return result
        loop = EditLoop(agent=None, project_root=str(self.project_root),
                        auto_backup=True)
        try:
            diff = loop.surgical_replace_many(rel, pairs)
        except (FileNotFoundError, EditConflictError, ValueError) as exc:
            result.note = f"edits rejected, file untouched: {exc}"
            result.tests_after = result.tests_before
            return result
        result.diff = diff
        after = self._run_tests(test_files)
        result.tests_after = _summarize_run(after)
        new_failures = (set(result.tests_after["failed"])
                        - set(result.tests_before["failed"]))
        new_errors = (result.tests_after["errors"]
                      > result.tests_before["errors"])
        if new_failures or new_errors:
            target.write_text(original, encoding="utf-8")
            result.reverted = True
            result.applied = False
            result.note = ("post-refactor tests newly failed "
                           f"({sorted(new_failures)}"
                           f"{'; new errors' if new_errors else ''}) — "
                           "edits reverted")
            return result
        result.applied = True
        if not test_files:
            result.note = ("no existing tests cover this file — refactor "
                           "applied without test verification")
        else:
            result.note = (f"tests before={result.tests_before['passed']} "
                           f"passed/{len(result.tests_before['failed'])} "
                           f"failed, after={result.tests_after['passed']} "
                           f"passed/{len(result.tests_after['failed'])} "
                           f"failed; no new failures")
        return result

    def _run_tests(self, test_files: list[str]) -> dict[str, Any]:
        if not test_files:
            return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                    "seconds": 0.0,
                    "note": "no existing tests found for this module"}
        return run_tests(paths=test_files, repo=str(self.project_root),
                         timeout=180.0)

# ── 6. Dependency hunter ───────────────────────────────────────────────


@dataclass
class DepInput:
    name: str  # dependency name as a user would write it, e.g. "requests"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DepResult:
    name: str
    verdict: str  # stdlib | repo-existing | pip-available | unknown
    detail: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _find_repo_modules(keyword: str) -> list[str]:
    """Nomorals modules whose name covers ``keyword`` (filesystem scan —
    no importing of every module)."""
    pkg_dir = Path(__file__).resolve().parent.parent  # nomorals/
    hits: list[str] = []
    spec = importlib.util.find_spec(f"nomorals.{keyword}")
    if spec is not None and spec.origin:
        hits.append(f"nomorals.{keyword}")
    for sub in ("tools", "agents", "memory", "llm", "integrations", "core",
                "storage"):
        subdir = pkg_dir / sub
        if not subdir.is_dir():
            continue
        for path in subdir.glob("*.py"):
            if (path.stem != "__init__" and keyword in path.stem
                    and f"nomorals.{sub}.{path.stem}" not in hits):
                hits.append(f"nomorals.{sub}.{path.stem}")
    return sorted(hits)


def _pypi_latest(name: str, timeout: float = 5.0) -> str | None:
    """Latest version on PyPI, or None (not found / network down).

    A short timeout and total exception tolerance: a network failure
    reports "unknown", never blocks.
    """
    url = ("https://pypi.org/pypi/"
           + urllib.parse.quote(name, safe="") + "/json")
    request = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-dep-hunter/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        version = payload.get("info", {}).get("version")
        return str(version) if version else None
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return None


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return None


class DepHunter(Subagent):
    """Before adding a dependency: stdlib -> repo -> PyPI -> unknown.

    * stdlib: ``sys.stdlib_module_names`` membership (no import needed).
    * repo: does ``nomorals/`` already ship a module for this?
    * pip: PyPI JSON API with a 5s timeout; network failure reports
      "unknown", never blocks and never guesses.
    """

    roster_type = "dep_hunter"
    role_spec = BUILTIN_ROLES["researcher"]

    def run(self, inp: DepInput) -> DepResult:
        name = (inp.name or "").strip()
        if not name:
            return DepResult(name="", verdict="unknown", detail="empty name",
                             recommendation="no dependency name was given")
        keyword = name.lower().replace("-", "_")
        # 1. stdlib
        if keyword in sys.stdlib_module_names:
            return DepResult(
                name=name, verdict="stdlib",
                detail=f"{keyword!r} is in the Python standard library",
                recommendation="use the standard library — no new "
                               "dependency needed")
        # 2. repo
        hits = _find_repo_modules(keyword)
        if hits:
            shown = ", ".join(hits[:5]) + ("…" if len(hits) > 5 else "")
            return DepResult(
                name=name, verdict="repo-existing", detail=shown,
                recommendation=(f"nomorals already ships {hits[0]} — reuse "
                                "it instead of adding a dependency"))
        # 3. PyPI (short timeout; failure -> unknown)
        version = _pypi_latest(name)
        if version:
            installed = _installed_version(name)
            detail = (f"PyPI latest {version}; "
                      + (f"installed {installed}" if installed
                         else "not currently installed"))
            return DepResult(
                name=name, verdict="pip-available", detail=detail,
                recommendation="available on PyPI — add it only if the "
                               "standard library and existing repo modules "
                               "cannot cover the need")
        # 4. unknown
        return DepResult(
            name=name, verdict="unknown",
            detail=("PyPI lookup failed or the package was not found "
                    "(the network may be unavailable)"),
            recommendation="could not verify — do not add blindly; confirm "
                           "the exact package name and network access first")


# ── 7. API designer ────────────────────────────────────────────────────


@dataclass
class ApiDesignInput:
    feature: str
    sample_modules: list[str] | None = None  # repo-relative; defaults below

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ApiDesignResult:
    stub_code: str
    imports_ok: bool
    import_error: str = ""
    consistency_notes: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: Default convention samples: modules that exemplify the repo's idioms
#: (module docstring, __all__, dataclass results with to_dict()).
DEFAULT_SAMPLE_MODULES = (
    "nomorals/agents/coding.py",
    "nomorals/tools/edit_loop.py",
    "nomorals/agents/role_specs.py",
)


def _convention_summary(root: Path, modules: list[str]) -> str:
    """Describe the repo's API conventions, sampled from real modules."""
    lines: list[str] = []
    for rel in modules:
        path = root / rel
        if not path.is_file():
            lines.append(f"- {rel}: not found, skipped")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="ignore"))
        except (OSError, SyntaxError):
            lines.append(f"- {rel}: unparseable, skipped")
            continue
        doc = "yes" if ast.get_docstring(tree) else "no"
        has_all = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__"
                    for t in n.targets)
            for n in ast.walk(tree))
        dataclasses_used = any(
            isinstance(n, ast.ClassDef)
            and any((d.id if isinstance(d, ast.Name) else
                     getattr(d, "attr", ""))
                    == "dataclass" for d in n.decorator_list)
            for n in ast.walk(tree))
        to_dict = any(isinstance(n, ast.FunctionDef) and n.name == "to_dict"
                      for n in ast.walk(tree))
        funcs = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and not n.name.startswith("_")]
        classes = [n.name for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef)]
        lines.append(
            f"- {rel}: module docstring={doc}, __all__="
            f"{'yes' if has_all else 'no'}, dataclass results="
            f"{'yes' if dataclasses_used else 'no'}, to_dict()="
            f"{'yes' if to_dict else 'no'}, public functions={funcs[:6]}, "
            f"classes={classes[:6]}")
    return "\n".join(lines)


def _check_stub_conventions(stub_code: str) -> list[str]:
    """Real consistency checks on the proposed stub vs repo idioms."""
    notes: list[str] = []
    try:
        tree = ast.parse(stub_code)
    except SyntaxError as exc:
        return [f"differs: stub does not parse ({exc.msg})"]
    if ast.get_docstring(tree):
        notes.append("ok: module docstring present")
    else:
        notes.append("differs: no module docstring "
                     "(repo modules open with one)")
    all_names: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))):
            all_names = [e.value for e in node.value.elts
                         if isinstance(e, ast.Constant)
                         and isinstance(e.value, str)]
    if all_names:
        notes.append(f"ok: __all__ present ({len(all_names)} names)")
    else:
        notes.append("differs: no __all__ (repo modules declare one)")
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    bad_funcs = [f.name for f in funcs
                 if not re.fullmatch(r"[a-z][a-z0-9_]*", f.name)
                 and not f.name.startswith("__")]
    bad_classes = [c.name for c in classes
                   if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", c.name)]
    if bad_funcs:
        notes.append(f"differs: non-snake_case functions: {bad_funcs[:5]}")
    else:
        notes.append("ok: function naming is snake_case")
    if bad_classes:
        notes.append(f"differs: non-PascalCase classes: {bad_classes[:5]}")
    else:
        notes.append("ok: class naming is PascalCase")
    dataclass_names = {
        c.name for c in classes
        if any((d.id if isinstance(d, ast.Name) else getattr(d, "attr", ""))
               == "dataclass" for d in c.decorator_list)}
    if dataclass_names:
        notes.append(f"ok: dataclass result types: "
                     f"{sorted(dataclass_names)[:6]}")
    else:
        notes.append("differs: no dataclass result types "
                     "(repo convention: dataclass results, not dicts)")
    dict_returns = [f.name for f in funcs
                    if isinstance(f.returns, ast.Name) and f.returns.id == "dict"]
    if dict_returns:
        notes.append(f"differs: free-form dict returns in {dict_returns[:5]} "
                     "(repo convention prefers dataclass results)")
    return notes


class ApiDesigner(Subagent):
    """Feature description + sampled repo conventions -> importable stub.

    The model proposes the public surface; the stub is written to a temp
    file and actually imported (``imports_ok`` proves it). Consistency
    notes are real AST checks against observed repo idioms, not vibes.
    """

    roster_type = "api_designer"
    role_spec = BUILTIN_ROLES["architect"]

    _SYSTEM = ("You are an API designer. Propose minimal, consistent public "
               "surfaces: small function/class counts, dataclass results, "
               "no free-form dicts.")

    def run(self, inp: ApiDesignInput) -> ApiDesignResult:
        if self.router is None:
            return ApiDesignResult(stub_code="", imports_ok=False,
                                   note="no model available — no stub "
                                        "proposed")
        modules = (inp.sample_modules if inp.sample_modules is not None
                   else list(DEFAULT_SAMPLE_MODULES))
        summary = _convention_summary(self.project_root, modules)
        data, error = self._model_json(
            self._SYSTEM,
            f"Design the public Python API for this feature:\n\n{inp.feature}\n\n"
            f"Repository conventions observed in existing modules:\n{summary}\n\n"
            f"Return JSON: {{\"stub\": \"<complete Python stub>\"}}.\n"
            f"The stub must: be valid Python; import cleanly (define or "
            f"import every name it references); open with a module "
            f"docstring; define __all__; use dataclasses for result types "
            f"(never free-form dict returns); use snake_case functions and "
            f"PascalCase classes; keep method bodies as `...`.")
        if error:
            return ApiDesignResult(stub_code="", imports_ok=False, note=error)
        stub = data.get("stub") if isinstance(data, dict) else None
        if not isinstance(stub, str) or not stub.strip():
            return ApiDesignResult(stub_code="", imports_ok=False,
                                   note="model returned no stub code")
        tmpdir = tempfile.mkdtemp(prefix="subagent_api_")
        try:
            stub_path = os.path.join(tmpdir, "api_stub.py")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(stub if stub.endswith("\n") else stub + "\n")
            imports_ok, import_error = self._try_import(stub_path)
        finally:
            # The stub is validated by import, not by keeping the file:
            # never leave scratch dirs behind in the temp tree.
            shutil.rmtree(tmpdir, ignore_errors=True)
        notes = _check_stub_conventions(stub)
        return ApiDesignResult(stub_code=stub, imports_ok=imports_ok,
                               import_error=import_error,
                               consistency_notes=notes,
                               note="stub imported cleanly" if imports_ok
                               else "stub failed to import")

    @staticmethod
    def _try_import(path: str) -> tuple[bool, str]:
        name = "subagent_api_stub"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                return False, "could not build import spec"
            module = importlib.util.module_from_spec(spec)
            # Register before exec: dataclasses resolve string
            # annotations (``from __future__ import annotations``) via
            # sys.modules, and an unregistered module makes every
            # dataclass stub fail with a confusing AttributeError.
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
            exported = getattr(module, "__all__", None)
            if isinstance(exported, (list, tuple)):
                missing = [n for n in exported if not hasattr(module, n)]
                if missing:
                    return False, (f"__all__ names missing after import: "
                                   f"{missing[:5]}")
            return True, ""
        except Exception as exc:  # noqa: BLE001 - import errors are results
            return False, f"{type(exc).__name__}: {exc}"


# ── 8. Fused parallel runner ───────────────────────────────────────────


@dataclass
class AgentRun:
    """One subagent's outcome: success with its result dataclass, or an
    explicit failure record. Failures never propagate as exceptions and are
    never silently dropped."""
    name: str
    ok: bool
    seconds: float
    result: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return data


@dataclass
class FusedResult:
    runs: list[AgentRun]
    merged: Any
    merge: str
    ok: bool        # True only when every run succeeded
    succeeded: int
    failed: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge_reviewer(runs: list[AgentRun]) -> dict[str, Any]:
    seen: dict[tuple[str, str, int, str], Flaw] = {}
    for run in runs:
        if not run.ok or run.result is None:
            continue
        for flaw in run.result.flaws:
            key = (flaw.category, flaw.file, flaw.line, flaw.message)
            seen.setdefault(key, flaw)
    flaws = sorted(seen.values(),
                   key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0),
                                  f.file, f.line, f.message))
    return {"flaws": [f.to_dict() for f in flaws], "count": len(flaws)}


def _merge_planner(runs: list[AgentRun]) -> dict[str, Any]:
    items: dict[str, PlanItem] = {}
    for run in runs:
        if not run.ok or run.result is None:
            continue
        for item in run.result.items:
            items.setdefault(item.path, item)
    ordered = [items[k] for k in sorted(items)]
    return {"items": [i.to_dict() for i in ordered], "count": len(ordered)}


def _merge_implementer(runs: list[AgentRun]) -> dict[str, Any]:
    applied: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for run in runs:
        if not run.ok or run.result is None:
            failed.append({"run": run.name, "error": run.error})
            continue
        result = run.result
        for edit in result.edits:
            entry = {"run": run.name, "path": result.path,
                     "old_text": edit.old_text[:80], "error": edit.error}
            (applied if edit.applied else failed).append(entry)
    return {"applied": applied, "failed": failed}


def _merge_tester(runs: list[AgentRun]) -> dict[str, Any]:
    passed = 0
    failed: dict[str, dict[str, Any]] = {}
    written: set[str] = set()
    for run in runs:
        if not run.ok or run.result is None:
            continue
        result = run.result
        passed += result.passed
        written.update(result.tests_written)
        for failure in result.failed:
            failed[str(failure.get("test_id", "?"))] = failure
    ordered = [failed[k] for k in sorted(failed)]
    return {"passed": passed, "failed": ordered,
            "tests_written": sorted(written)}


def _merge_refactorer(runs: list[AgentRun]) -> dict[str, Any]:
    out = []
    for run in runs:
        result = run.result
        out.append({
            "run": run.name, "ok": run.ok,
            "path": getattr(result, "path", ""),
            "applied": bool(getattr(result, "applied", False)),
            "reverted": bool(getattr(result, "reverted", False)),
            "error": run.error,
        })
    out.sort(key=lambda d: (d["path"], d["run"]))
    return {"refactors": out}


def _merge_dep_hunter(runs: list[AgentRun]) -> dict[str, Any]:
    out = []
    for run in runs:
        result = run.result
        if run.ok and result is not None:
            out.append({"name": result.name, "verdict": result.verdict,
                        "detail": result.detail,
                        "recommendation": result.recommendation})
        else:
            out.append({"name": run.name, "verdict": "unknown",
                        "detail": run.error or "agent failed",
                        "recommendation": "agent failed — treat as unknown"})
    out.sort(key=lambda d: d["name"])
    return {"dependencies": out}


def _merge_api_designer(runs: list[AgentRun]) -> dict[str, Any]:
    stubs = []
    notes: set[str] = set()
    for run in runs:
        result = run.result
        if run.ok and result is not None and result.imports_ok:
            stubs.append({"run": run.name, "stub_code": result.stub_code})
            notes.update(result.consistency_notes)
    return {"stubs": stubs, "consistency_notes": sorted(notes)}


#: Deterministic merge function per roster type.
MERGERS: dict[str, Callable[[list[AgentRun]], Any]] = {
    "reviewer": _merge_reviewer,
    "planner": _merge_planner,
    "implementer": _merge_implementer,
    "tester": _merge_tester,
    "refactorer": _merge_refactorer,
    "dep_hunter": _merge_dep_hunter,
    "api_designer": _merge_api_designer,
}


def _invoke(agent: Subagent, name: str, inp: Any) -> AgentRun:
    started = time.perf_counter()
    try:
        result = agent.run(inp)
    except Exception as exc:  # noqa: BLE001 - failures are records, not raises
        return AgentRun(name=name, ok=False,
                        seconds=time.perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}")
    return AgentRun(name=name, ok=True,
                    seconds=time.perf_counter() - started, result=result)


def run_parallel(subagents: list[Subagent], inputs: list[Any], *,
                 timeout: float = 60.0,
                 merge: str | None = None) -> FusedResult:
    """Run N subagent ``run()`` calls concurrently and merge the results.

    Each subagent gets its own thread and its own per-agent ``timeout``;
    a timeout or an exception becomes an explicit :class:`AgentRun` with
    ``ok=False`` — it never kills the batch and is never silently dropped.
    ``merged`` is produced by the deterministic merge function for
    ``merge`` (default: the first subagent's roster type); runs of other
    roster types are excluded from the merge but kept in ``runs``.
    """
    if len(subagents) != len(inputs):
        raise ValueError(
            f"subagents ({len(subagents)}) and inputs ({len(inputs)}) "
            "must have the same length")
    names = [getattr(agent, "name", None)
             or f"{getattr(agent, 'roster_type', 'agent')}-{i}"
             for i, agent in enumerate(subagents)]
    runs: list[AgentRun | None] = [None] * len(subagents)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(subagents)),
            thread_name_prefix="subagent") as executor:
        futures = {
            executor.submit(_invoke, agent, name, inp): i
            for i, (agent, name, inp)
            in enumerate(zip(subagents, names, inputs))
        }
        for future, i in futures.items():
            try:
                runs[i] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                runs[i] = AgentRun(name=names[i], ok=False, seconds=timeout,
                                   error=f"timed out after {timeout}s")
            except Exception as exc:  # noqa: BLE001 - defensive; _invoke catches
                runs[i] = AgentRun(name=names[i], ok=False, seconds=0.0,
                                   error=f"{type(exc).__name__}: {exc}")
    finished = [r for r in runs if r is not None]
    merge_key = (merge or (getattr(subagents[0], "roster_type", "")
                           if subagents else ""))
    merger = MERGERS.get(merge_key)
    if merger is not None:
        relevant = [r for r, a in zip(finished, subagents)
                    if getattr(a, "roster_type", "") == merge_key]
        merged: Any = merger(relevant)
    else:
        merged = [r.to_dict() for r in finished]
    ok = all(r.ok for r in finished) and bool(finished)
    return FusedResult(runs=finished, merged=merged, merge=merge_key, ok=ok,
                       succeeded=sum(1 for r in finished if r.ok),
                       failed=sum(1 for r in finished if not r.ok))


#: The roster: role name -> subagent class. Construct per mission; instances
#: hold no shared mutable state and are safe to run via run_parallel.
ROSTER: dict[str, type[Subagent]] = {
    "planner": Planner,
    "implementer": Implementer,
    "reviewer": Reviewer,
    "tester": Tester,
    "refactorer": Refactorer,
    "dep_hunter": DepHunter,
    "api_designer": ApiDesigner,
}


def register(registry: Any) -> None:
    """Expose the specialist subagent roster as agent tools.

    The roster (planner, implementer, reviewer, tester, refactorer,
    dep_hunter, api_designer) provides L5 orchestration workers with a
    uniform run(input) -> result interface. This hook exposes roster
    listing and parallel execution through the tool registry.
    """

    @registry.register(
        "subagent_roster",
        description=(
            "List the specialist subagent roster (planner, implementer, "
            "reviewer, tester, refactorer, dep_hunter, api_designer) with "
            "their roles. These are L5 orchestration workers for coding pipelines."
        ),
        capability="agent.subagents",
        parameters={},
    )
    def _subagent_roster() -> dict[str, Any]:
        return {
            "ok": True,
            "roster": sorted(ROSTER.keys()),
            "description": (
                "Specialist coding-pipeline subagents. Each has a uniform "
                "run(input) -> result interface and is safe for parallel execution."
            ),
        }

    @registry.register(
        "subagent_run",
        description=(
            "Run a specialist subagent from the roster. role is one of: "
            "planner, implementer, reviewer, tester, refactorer, dep_hunter, "
            "api_designer. input_json is the JSON-encoded input dataclass."
        ),
        capability="agent.subagents",
        parameters={
            "role": "str — roster role name",
            "input_json": "str — JSON-encoded input for the subagent",
        },
    )
    def _subagent_run(role: str, input_json: str = "{}") -> dict[str, Any]:
        import dataclasses
        import json

        cls = ROSTER.get((role or "").strip().lower())
        if cls is None:
            return {
                "ok": False,
                "error": f"unknown role {role!r}; roster: {sorted(ROSTER.keys())}",
            }
        try:
            data = json.loads(input_json or "{}")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"bad input_json: {exc}"}
        try:
            agent = cls()
            # Build the input dataclass from the provided dict
            input_cls = None
            for attr in ("__dataclass_params__",):
                _ = attr
            # Find the Input dataclass: <Role>Input naming convention
            import re as _re

            input_name = f"{cls.__name__}Input"
            input_cls = globals().get(input_name)
            if input_cls is None or not dataclasses.is_dataclass(input_cls):
                # Fall back: try any dataclass ending in "Input" defined here
                for _n, _v in list(globals().items()):
                    if (
                        _n.endswith("Input")
                        and dataclasses.is_dataclass(_v)
                        and _n.lower().startswith(cls.__name__.lower()[:4])
                    ):
                        input_cls = _v
                        break
            if input_cls is None:
                return {"ok": False, "error": f"no Input dataclass found for {role}"}
            field_names = {f.name for f in dataclasses.fields(input_cls)}
            kwargs = {k: v for k, v in data.items() if k in field_names}
            inp = input_cls(**kwargs)
            result = agent.run(inp)
            if dataclasses.is_dataclass(result):
                return {"ok": True, "result": dataclasses.asdict(result)}
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
