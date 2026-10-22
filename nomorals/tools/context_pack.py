"""Context packs for the coding loop's edit calls (pass #64, W6).

The drafting loop used to paste the WHOLE target file into every edit
round: prompt cost scaled with file size instead of task size, and the
model edited blind to the symbols the file depends on and to the tests
that would judge the change.  The ranked-context machinery existed —
:func:`nomorals.tools.repo_context.rank_context` ranks whole FILES for
the *planning* context (gap-audit G8: CLOSED for CodeExecutor, PARTIAL
for the main loop) — but the drafting loop never called any of it.

A context pack is the bounded, step-steered assembly one edit call
actually needs:

  1. the target file's RELEVANT REGION — the module header (docstring,
     imports, constants) plus the top-level definitions the step is
     about, spliced in file order with explicit omission markers that
     NAME the definitions left out (never silent amputation);
  2. the file's IMPORT SURFACE — full bodies of the neighboring
     definitions the target imports and references (bounded), plus
     signature lines for everything else it imports from the repo;
  3. its COVERING TESTS — the selection the verify round would run
     (pass #62: :func:`pytest_runner.select_changed_tests_detailed`,
     import graph ∪ stem, with provenance), rendered as test names +
     assertion text.

Ranking is steered by the step (#60): the task plus the step's own
instructions/acceptance — and, on repair rounds, the identifiers in
the last error — decide which definitions, imports and tests rank
first.  The pack has a stated char budget; sections hold shares of it
(target 55% / imports 27% / tests 18%) and everything cut is recorded
in :meth:`ContextPack.metadata` (``trimmed``), which the loop surfaces
on ``CodingResult.context_packs``.

Fallbacks are honest, never silent: an unparseable or small target
file, or a pack build failure, returns ``mode="whole-file"`` — the
pre-#64 prompt, byte for byte — with the reason in ``notes``.  A
broken import graph degrades the imports/tests sections to the
repo_context helpers (mirroring #62's stem fallback) and says so.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from . import repo_context
from .import_graph import module_name_for

_log = get_logger(__name__)

#: Default pack budget, in characters (repo_context estimates tokens
#: at 4 chars/token, so this is ~6,000 tokens of edit context).
DEFAULT_PACK_BUDGET_CHARS = 24_000

#: A target file at or under this size is sent whole — a region cut
#: would save nothing and risk hiding the exact lines an edit needs.
SMALL_FILE_CHARS = 6_000

#: Section shares of the budget.
_TARGET_SHARE = 0.55
_IMPORTS_SHARE = 0.27
_TESTS_SHARE = 0.18

#: A definition rides in the region only on a real relevance signal:
#: a focus-name match, a distinctive focus name in its body, a
#: referenced-by name plus a signature hit, …  The top-ranked
#: definition always rides regardless (it is the best guess); the
#: rest must clear this bar — a stray generic word ("total", "file")
#: in a body is not a signal, and bulk that clears nothing is omitted
#: BY NAME instead of diluting the region.
_MIN_SEGMENT_SCORE = 20

#: Bounds inside the imports/tests sections.
_NEIGHBOR_DEF_MAX_CHARS = 1_400
_SIGS_PER_MODULE = 14
_SIG_LINE_MAX_CHARS = 200
_TEST_FILES_MAX = 3
_ASSERTS_PER_FILE = 10
_ASSERT_MAX_CHARS = 200

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class TrimmedItem:
    """One thing the budget cut — always recorded, never silent."""

    kind: str
    label: str
    chars: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label,
                "chars": self.chars, "reason": self.reason}


@dataclass
class ContextPack:
    """One edit call's assembled context + its accounting."""

    rel: str
    mode: str  # "pack" | "whole-file"
    text: str
    budget_chars: int
    whole_chars: int
    sections: list[dict[str, Any]] = field(default_factory=list)
    trimmed: list[TrimmedItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    region_ranges: list[tuple[int, int]] = field(default_factory=list)
    omitted_defs: list[str] = field(default_factory=list)
    covering_tests: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "mode": self.mode,
            "budget_chars": self.budget_chars,
            "used_chars": len(self.text),
            "whole_file_chars": self.whole_chars,
            "sections": self.sections,
            "trimmed": [t.as_dict() for t in self.trimmed],
            "notes": list(self.notes),
            "region_ranges": [list(r) for r in self.region_ranges],
            "omitted_defs": list(self.omitted_defs),
            "covering_tests": list(self.covering_tests),
        }


def whole_file_pack(rel: str, current: str, budget_chars: int,
                    note: str) -> ContextPack:
    """The pre-#64 behavior as a pack: whole file, reason recorded."""
    return ContextPack(
        rel=rel, mode="whole-file", text=current,
        budget_chars=budget_chars, whole_chars=len(current),
        sections=[{"kind": "target", "label": f"{rel} (whole file)",
                   "chars": len(current)}],
        notes=[note])


# ── focus ─────────────────────────────────────────────────────────────


def _focus(task: str, step: dict[str, Any] | None,
           last_error: str) -> tuple[list[str], list[str]]:
    """(focus names, focus keywords) steering every ranking decision.

    Names are identifiers as written (python names are case-sensitive);
    keywords come from repo_context's extractor (stopword-filtered).
    Repair rounds steer by the last error's identifiers too — a failed
    apply or a failing test names the symbols the next round must see.
    """
    parts = [task or ""]
    if step:
        parts.append(str(step.get("instructions") or ""))
        parts.append(str(step.get("acceptance") or ""))
    if last_error:
        parts.append(str(last_error)[-1500:])
    text = "\n".join(parts)
    names: list[str] = []
    seen: set[str] = set()
    for m in _IDENT_RE.finditer(text):
        w = m.group(0)
        if len(w) >= 3 and w not in seen:
            seen.add(w)
            names.append(w)
    return names, repo_context.extract_keywords(text, limit=10)


def _score(name: str, sig_text: str, body_text: str,
           focus_names: set[str], focus_tokens: set[str],
           used_names: set[str]) -> int:
    """Deterministic relevance score for one definition/segment."""
    score = 0
    if name:
        if name in focus_names:
            score += 100
        if name in used_names:
            score += 15
        for tok in focus_tokens:
            if tok == name or (len(tok) >= 4 and tok in name):
                score += 40
    # Distinct focus names the body mentions: a large definition (a
    # class holding the focused method) mentions several and must
    # outrank small definitions that mention one generic word.
    # Distinctive names (underscored or long) weigh more than short
    # generic ones ("step", "file"); the bonus caps at +100.
    body_bonus = 0
    for fn in focus_names:
        if fn != name and re.search(rf"\b{re.escape(fn)}\b", body_text):
            body_bonus += 25 if ("_" in fn or len(fn) >= 7) else 6
    score += min(body_bonus, 100)
    for tok in focus_tokens:
        if tok in sig_text.lower():
            score += 10
    return score


# ── target region ─────────────────────────────────────────────────────


class _Segment:
    """A line range of the target file with one primary definition."""

    __slots__ = ("name", "kind", "start", "end", "score", "node")

    def __init__(self, name: str, kind: str, start: int, end: int,
                 node: Any = None) -> None:
        self.name = name
        self.kind = kind  # "header" | "def" | "class"
        self.start = start  # 1-based, inclusive
        self.end = end
        self.score = 0
        self.node = node


def _segments(lines: list[str], tree: ast.Module) -> list[_Segment]:
    """Partition the file: header + one segment per top-level def/class.

    Each def segment extends to the next def's first line, so stray
    top-level statements ride with the definition above them.
    """
    blocks: list[_Segment] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            start = node.lineno
            if node.decorator_list:
                start = min(start, min(d.lineno
                                       for d in node.decorator_list))
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            blocks.append(_Segment(node.name, kind, start,
                                   node.end_lineno or start, node))
    segs: list[_Segment] = []
    first = blocks[0].start if blocks else len(lines) + 1
    if first > 1:
        segs.append(_Segment("", "header", 1, first - 1))
    for i, b in enumerate(blocks):
        end = (blocks[i + 1].start - 1) if i + 1 < len(blocks) \
            else len(lines)
        segs.append(_Segment(b.name, b.kind, b.start, max(b.end, end),
                             b.node))
    if not blocks and lines:
        segs.append(_Segment("", "header", 1, len(lines)))
    return segs


def _class_subregion(seg: _Segment, lines: list[str],
                     focus_names: set[str], focus_tokens: set[str],
                     used_names: set[str],
                     room: int) -> tuple[str, list[str]] | None:
    """Class header + best-scoring methods when the whole class
    doesn't fit ``room``.  Returns (text, omitted_method_names) or
    None when sub-selection can't help."""
    node = seg.node
    if not isinstance(node, ast.ClassDef):
        return None
    methods = [n for n in node.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(methods) < 2:
        return None
    first_method = methods[0]
    head_end = first_method.lineno
    if first_method.decorator_list:
        head_end = min(head_end,
                       min(d.lineno for d in first_method.decorator_list))
    head = lines[seg.start - 1:head_end - 1]
    head_text = "\n".join(head)
    if len(head_text) >= room:
        return None
    scored: list[tuple[int, int, ast.AST]] = []
    for m in methods:
        m_start = m.lineno
        if m.decorator_list:
            m_start = min(m_start, min(d.lineno
                                       for d in m.decorator_list))
        m_end = m.end_lineno or m_start
        body = "\n".join(lines[m_start - 1:m_end])
        scored.append((_score(m.name, body.split("\n", 1)[0], body,
                              focus_names, focus_tokens, used_names),
                       m_start, m))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[ast.AST] = []
    truncated: dict[int, str] = {}  # id(method) -> shown text
    used = len(head_text)
    for _s, _st, m in scored:
        m_start = m.lineno
        if m.decorator_list:
            m_start = min(m_start, min(d.lineno
                                   for d in m.decorator_list))
        m_end = m.end_lineno or m_start
        cost = sum(len(lines[i]) + 1
                   for i in range(m_start - 1, m_end))
        if used + cost > room:
            if chosen:
                continue
            # The best method alone exceeds the room: show its head
            # with a truncation marker rather than nothing.
            keep: list[str] = []
            acc = used
            for ln in lines[m_start - 1:m_end]:
                if keep and acc + len(ln) + 1 > room:
                    break
                keep.append(ln)
                acc += len(ln) + 1
            if not keep:
                return None
            truncated[id(m)] = (
                "\n".join(keep)
                + f"\n    # [context pack: method '{m.name}' truncated"
                f" — lines {m_start + len(keep)}-{m_end} not shown]")
            chosen.append(m)
            used = acc
            continue
        chosen.append(m)
        used += cost
    if not chosen:
        return None
    chosen_ids = {id(m) for m in chosen}
    omitted = [m.name for m in methods if id(m) not in chosen_ids]
    parts = [head_text]
    last = head_end - 1  # 0-based index consumed so far
    for m in sorted(chosen, key=lambda n: n.lineno):
        m_start = m.lineno
        if m.decorator_list:
            m_start = min(m_start, min(d.lineno
                                   for d in m.decorator_list))
        m_end = m.end_lineno or m_start
        if m_start - 1 > last:
            parts.append(
                f"    # [context pack: methods omitted here: "
                f"{', '.join(omitted)}]")
            omitted = []
        parts.append(truncated.get(id(m))
                     or "\n".join(lines[m_start - 1:m_end]))
        last = m_end
    if omitted:
        parts.append(
            f"    # [context pack: methods omitted: {', '.join(omitted)}]")
    return "\n".join(parts), [m.name for m in methods
                              if id(m) not in chosen_ids]


def _target_region(lines: list[str], tree: ast.Module,
                   focus_names: set[str], focus_tokens: set[str],
                   budget: int,
                   trimmed: list[TrimmedItem]) -> tuple[
                       str, list[tuple[int, int]], list[str]]:
    """Assemble the target region under ``budget`` chars.

    Returns (region_text, included_line_ranges, omitted_def_names).
    """
    total = "\n".join(lines)
    if len(total) <= budget:
        return total, [(1, len(lines))], []
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            used_names.add(node.attr)
    segs = _segments(lines, tree)

    def seg_text(s: _Segment) -> str:
        return "\n".join(lines[s.start - 1:s.end])

    for s in segs:
        if s.kind != "header":
            text = seg_text(s)
            sig = text.split("\n", 1)[0]
            s.score = _score(s.name, sig, text, focus_names,
                             focus_tokens, used_names)
    header = next((s for s in segs if s.kind == "header"), None)
    defs = [s for s in segs if s.kind != "header"]
    # most relevant first; ties keep file order (stable, deterministic)
    ranked = sorted(defs, key=lambda s: -s.score)

    chosen: list[_Segment] = []
    rendered: dict[int, str] = {}  # id(seg) -> text actually shown
    used = 0
    if header is not None:
        htext = seg_text(header)
        if len(htext) > budget:
            keep = budget
            htext = (htext[:keep]
                     + "\n# [context pack: module header truncated]")
            trimmed.append(TrimmedItem(
                "target", "module header", len(seg_text(header)),
                "header alone exceeded the target share; truncated"))
        rendered[id(header)] = htext
        chosen.append(header)
        used += len(htext)
    for i, s in enumerate(ranked):
        text = seg_text(s)
        if i > 0 and s.score < _MIN_SEGMENT_SCORE:
            trimmed.append(TrimmedItem(
                "definition", s.name or "(statements)", len(text),
                "below the relevance threshold for this step "
                f"(score {s.score} < {_MIN_SEGMENT_SCORE})"))
            continue
        room = budget - used
        if room <= 0:
            trimmed.append(TrimmedItem(
                "definition", s.name or "(statements)", len(text),
                "target share spent"))
            continue
        if len(text) <= room:
            rendered[id(s)] = text
            chosen.append(s)
            used += len(text)
            continue
        if i == 0:
            # The single most relevant definition is ALWAYS shown —
            # sub-select a class's methods, else truncate with a
            # marker naming what was cut.
            sub = _class_subregion(s, lines, focus_names, focus_tokens,
                                   used_names, room)
            if sub is not None:
                sub_text, _om = sub
                rendered[id(s)] = sub_text
                chosen.append(s)
                used += len(sub_text)
                trimmed.append(TrimmedItem(
                    "definition", s.name, len(text) - len(sub_text),
                    "class shown as header + relevant methods"))
                continue
            keep_lines: list[str] = []
            cost = 0
            for ln in lines[s.start - 1:s.end]:
                if keep_lines and cost + len(ln) + 1 > room:
                    break
                keep_lines.append(ln)
                cost += len(ln) + 1
            shown = "\n".join(keep_lines)
            marker = (f"# [context pack: '{s.name}' truncated — "
                      f"lines {s.start + len(keep_lines)}-{s.end} "
                      "not shown]")
            rendered[id(s)] = shown + "\n" + marker
            chosen.append(s)
            used += len(shown) + len(marker)
            trimmed.append(TrimmedItem(
                "definition", s.name, len(text) - len(shown),
                "definition truncated to the target share"))
            continue
        trimmed.append(TrimmedItem(
            "definition", s.name or "(statements)", len(text),
            "target share spent on higher-ranked definitions"))
    chosen_ids = {id(s) for s in chosen}
    omitted = [s.name for s in defs if id(s) not in chosen_ids and s.name]

    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 1  # next unconsumed 1-based line
    for s in sorted(chosen, key=lambda x: x.start):
        if s.start > cursor:
            gap_names = [d.name for d in defs
                         if d.start >= cursor and d.end < s.start
                         and d.name]
            parts.append(
                f"# [context pack: lines {cursor}-{s.start - 1} omitted"
                + (f" — defines: {', '.join(gap_names)}]"
                   if gap_names else "]"))
        parts.append(rendered[id(s)])
        ranges.append((s.start, s.end))
        cursor = s.end + 1
    if cursor <= len(lines):
        gap_names = [d.name for d in defs if d.start >= cursor and d.name]
        parts.append(
            f"# [context pack: lines {cursor}-{len(lines)} omitted"
            + (f" — defines: {', '.join(gap_names)}]" if gap_names
               else "]"))
    return "\n".join(parts), ranges, omitted


# ── import surface ────────────────────────────────────────────────────


def _module_file_map(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in repo_context._iter_py_files(root):
        name = module_name_for(root, path)
        if name and name not in out:
            out[name] = path
    return out


def _imported_modules(root: Path, rel: str, tree: ast.Module,
                      mfmap: dict[str, Path]) -> tuple[set[str], dict[str, str]]:
    """(module names the target imports, local alias -> module).

    From the target's own import statements, resolved against the repo
    module map (longest known prefix wins, like the import graph).
    """
    own = module_name_for(root, root / rel) or ""
    own_pkg = own.rpartition(".")[0]
    modules: set[str] = set()
    aliases: dict[str, str] = {}

    def resolve(dotted: str) -> str | None:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in mfmap:
                return cand
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                hit = resolve(a.name)
                if hit:
                    modules.add(hit)
                    local = a.asname or a.name.split(".")[0]
                    # `import a.b` binds `a`; `import a.b as x` binds x
                    aliases[local] = hit if a.asname else resolve(
                        a.name.split(".")[0]) or hit
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                dotted = repo_context._resolve_relative(
                    own_pkg, node.level, node.module or "")
            else:
                dotted = node.module or ""
            if not dotted:
                continue
            hit = resolve(dotted)
            if hit:
                modules.add(hit)
            for a in node.names:
                if a.name == "*":
                    continue
                sub = resolve(f"{dotted}.{a.name}")
                if sub:
                    modules.add(sub)
                    aliases[a.asname or a.name] = sub
                elif hit:
                    aliases[a.asname or a.name] = hit
    return modules, aliases


def _referenced_names(tree: ast.Module,
                      aliases: dict[str, str]) -> list[tuple[str, str]]:
    """(module, name) pairs the target actually references: names it
    imported with ``from`` and attributes it reads off module aliases."""
    pairs: set[tuple[str, str]] = set()
    # from-imports: local name -> the source name it binds (its module
    # comes from the alias map, keyed by the same local name)
    from_pairs: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    from_pairs[a.asname or a.name] = a.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) \
                and isinstance(node.value, ast.Name):
            mod = aliases.get(node.value.id)
            if mod:
                pairs.add((mod, node.attr))
    # from-imported names that are referenced anywhere as bare names
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for local, src_name in from_pairs.items():
        if local in used:
            mod = aliases.get(local)
            if mod:
                pairs.add((mod, src_name))
    return sorted(pairs)


def _def_source(path: Path, name: str) -> str | None:
    """Full source of the top-level def/class ``name`` in ``path``."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == name:
            seg = ast.get_source_segment(text, node)
            if seg:
                return seg
    return None


def _signature_lines(path: Path) -> list[str]:
    """Signature lines (+ first doc line) of a module's public
    top-level definitions — its import surface in miniature."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8",
                                        errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            try:
                sig = f"def {node.name}({ast.unparse(node.args)})"
                if node.returns is not None:
                    sig += f" -> {ast.unparse(node.returns)}"
                sig += ":"
            except Exception:  # noqa: BLE001 — unparse is best-effort
                sig = f"def {node.name}(...):"
            out.append(sig[:_SIG_LINE_MAX_CHARS])
            doc = (ast.get_docstring(node) or "").split("\n", 1)[0]
            if doc:
                out.append(f"    # {doc[:120]}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:  # noqa: BLE001
                bases = ""
            out.append(f"class {node.name}({bases}):"
                       if bases else f"class {node.name}:")
            methods = [n for n in node.body
                       if isinstance(n, (ast.FunctionDef,
                                         ast.AsyncFunctionDef))
                       and not n.name.startswith("_")]
            for m in methods[:8]:
                try:
                    msig = (f"    def {m.name}"
                            f"({ast.unparse(m.args)}):")
                except Exception:  # noqa: BLE001
                    msig = f"    def {m.name}(...):"
                out.append(msig[:_SIG_LINE_MAX_CHARS])
        if len(out) >= _SIGS_PER_MODULE * 2:
            break
    return out[:_SIGS_PER_MODULE * 2]


def _imports_section(root: Path, rel: str, tree: ast.Module,
                     mfmap: dict[str, Path],
                     focus_names: set[str],
                     share: int,
                     trimmed: list[TrimmedItem],
                     notes: list[str]) -> list[tuple[str, str, str]]:
    """(kind, label, text) items: referenced neighbor definitions
    first, then per-module signature blocks — cut to ``share``."""
    modules, aliases = _imported_modules(root, rel, tree, mfmap)
    # #62's graph resolves the same question with repo-wide knowledge
    # (submodule edges from `from pkg import sub`); union it in when
    # it builds, degrade to the AST read when it doesn't.
    try:
        from .import_graph import build_import_graph

        graph = build_import_graph(root)
        for mod in graph.file_edges.get(rel, set()):
            if mod in mfmap:
                modules.add(mod)
    except Exception as exc:  # noqa: BLE001 — degrade, never crash
        notes.append(
            "import graph unavailable for the context pack "
            f"({type(exc).__name__}: {exc}); import surface read from "
            "the target's own import statements")
    items: list[tuple[int, str, str, str]] = []  # score, kind, label, text
    referenced = _referenced_names(tree, aliases)
    for mod, name in referenced:
        path = mfmap.get(mod)
        if path is None:
            continue
        body = _def_source(path, name)
        if body is None:
            continue
        label = f"{path.relative_to(root)}: {name} (referenced)"
        if len(body) > _NEIGHBOR_DEF_MAX_CHARS:
            sigs = _signature_lines(path)
            start = next((i for i, s in enumerate(sigs)
                          if s.startswith(f"def {name}(")
                          or s.startswith(f"class {name}")
                          or f" {name}(" in s), None)
            if start is None:
                body = body[:_NEIGHBOR_DEF_MAX_CHARS]
            elif sigs[start].startswith("class "):
                # a class's surface is its header + method signatures
                body = "\n".join(sigs[start:start + 10])
            else:
                body = sigs[start]
            label += " — signature only (definition too large)"
        score = 100 if name in focus_names else 50
        items.append((score, "neighbor-def", label, body))
    for mod in sorted(modules):
        path = mfmap.get(mod)
        if path is None:
            continue
        sigs = _signature_lines(path)
        if not sigs:
            continue
        label = f"{path.relative_to(root)} — import surface"
        text = "\n".join(sigs)
        score = 30 if any(fn in text for fn in focus_names) else 10
        items.append((score, "import-signatures", label, text))
    items.sort(key=lambda t: (-t[0], t[2]))
    out: list[tuple[str, str, str]] = []
    used = 0
    for _score_v, kind, label, text in items:
        cost = len(text) + len(label) + 8
        if out and used + cost > share:
            trimmed.append(TrimmedItem(
                kind, label, len(text),
                "imports share spent on higher-ranked items"))
            continue
        out.append((kind, label, text))
        used += cost
    return out


# ── covering tests ────────────────────────────────────────────────────


_REASON_RANK = {"graph-direct": 0, "graph-transitive": 1}


def _test_items(root: Path, rel: str, share: int,
                trimmed: list[TrimmedItem],
                notes: list[str]) -> tuple[list[tuple[str, str, str]],
                                            list[str]]:
    """Covering tests for ``rel``: the verify round's own selection
    (#62, graph ∪ stem with provenance), as names + assertions."""
    selected: list[tuple[str, str]] = []
    try:
        from .pytest_runner import select_changed_tests_detailed

        picked, reasons, meta = select_changed_tests_detailed(
            str(root), [rel])
        selected = [(r, reasons.get(r, "")) for r in picked]
        if meta.get("graph_status") == "degraded":
            notes.append(
                "covering-test selection ran stem-only: import graph "
                f"degraded ({meta.get('graph_error', 'unknown error')})")
    except Exception as exc:  # noqa: BLE001 — degrade to stem names
        notes.append(
            "covering-test selection unavailable for the context pack "
            f"({type(exc).__name__}: {exc}); falling back to "
            "same-name test files")
        selected = [(r, "stem") for r in
                    repo_context.test_files_for(root, rel)]
    selected.sort(key=lambda t: (
        min((_REASON_RANK.get(part, 2)
             for part in t[1].split("+")), default=2), t[0]))
    items: list[tuple[str, str, str]] = []
    covered: list[str] = []
    used = 0
    for test_rel, reason in selected:
        if len(covered) >= _TEST_FILES_MAX:
            trimmed.append(TrimmedItem(
                "tests", test_rel, 0,
                f"covering-test cap ({_TEST_FILES_MAX} files)"))
            continue
        path = root / test_rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError, ValueError):
            items.append(("tests", f"{test_rel} (selected: {reason})",
                          "# (test file could not be parsed — "
                          "assertions not extracted)"))
            covered.append(test_rel)
            continue
        lines_out: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test"):
                asserts: list[str] = []
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert):
                        seg = ast.get_source_segment(text, sub) or ""
                        seg = " ".join(seg.split())
                        if seg:
                            asserts.append(seg[:_ASSERT_MAX_CHARS])
                lines_out.append(f"def {node.name}(...):")
                for a in asserts[:_ASSERTS_PER_FILE]:
                    lines_out.append(f"    {a}")
                if not asserts:
                    lines_out.append("    # (no assert statements)")
        body = "\n".join(lines_out) or "# (no test functions found)"
        label = f"{test_rel} (selected: {reason or 'stem'})"
        cost = len(body) + len(label) + 8
        if items and used + cost > share:
            trimmed.append(TrimmedItem(
                "tests", test_rel, len(body),
                "tests share spent on higher-ranked test files"))
            continue
        items.append(("tests", label, body))
        covered.append(test_rel)
        used += cost
    return items, covered


# ── assembly ──────────────────────────────────────────────────────────


def build_context_pack(
    root: str | Path,
    rel: str,
    current: str,
    *,
    task: str = "",
    step: dict[str, Any] | None = None,
    last_error: str = "",
    budget_chars: int = DEFAULT_PACK_BUDGET_CHARS,
) -> ContextPack:
    """Build the context pack for one edit call.  NEVER raises: any
    build failure degrades to the whole-file pack with the reason in
    ``notes`` (the loop's pre-#64 behavior, honestly labeled)."""
    budget = max(1_000, int(budget_chars))
    try:
        return _build(Path(root), rel, current, task=task, step=step,
                      last_error=last_error, budget=budget)
    except Exception as exc:  # noqa: BLE001 — the pack must never break
        _log.warning("context pack build failed for %s: %s", rel, exc)
        return whole_file_pack(
            rel, current, budget,
            f"context pack could not be built ({type(exc).__name__}: "
            f"{exc}); whole file sent, as before pass #64")


def _build(root: Path, rel: str, current: str, *, task: str,
           step: dict[str, Any] | None, last_error: str,
           budget: int) -> ContextPack:
    if len(current) <= SMALL_FILE_CHARS:
        return whole_file_pack(
            rel, current, budget,
            f"target file is {len(current)} chars (<= {SMALL_FILE_CHARS})"
            " — small enough to send whole; no region cut needed")
    try:
        tree = ast.parse(current)
    except (SyntaxError, ValueError) as exc:
        pack = whole_file_pack(
            rel, current, budget,
            f"target file does not parse ({type(exc).__name__}) — no "
            "region can be selected; whole file sent, as before "
            "pass #64")
        return pack
    focus_names_list, focus_tokens_list = _focus(task, step, last_error)
    focus_names = set(focus_names_list)
    focus_tokens = set(focus_tokens_list)

    trimmed: list[TrimmedItem] = []
    notes: list[str] = []
    lines = current.split("\n")
    region_budget = int(budget * _TARGET_SHARE)
    region_text, ranges, omitted = _target_region(
        lines, tree, focus_names, focus_tokens, region_budget, trimmed)
    whole_region = len(ranges) == 1 and ranges[0] == (1, len(lines)) \
        and not omitted

    mfmap = _module_file_map(root)
    import_items = _imports_section(
        root, rel, tree, mfmap, focus_names,
        int(budget * _IMPORTS_SHARE), trimmed, notes)
    test_items, covered = _test_items(
        root, rel, int(budget * _TESTS_SHARE), trimmed, notes)

    header_lines = [
        f"# ── context pack for {rel} (pass #64) ──",
        f"# whole file: {len(current)} chars / {len(lines)} lines; "
        f"pack budget: {budget} chars",
    ]
    if whole_region:
        header_lines.append("# target: shown whole (fits its share)")
    else:
        shown = ", ".join(f"{a}-{b}" for a, b in ranges)
        header_lines.append(
            f"# target region: lines {shown}; omitted definitions are "
            "named in the markers")

    sections: list[dict[str, Any]] = []
    parts: list[str] = ["\n".join(header_lines)]
    sections.append({"kind": "target",
                     "label": f"{rel} (region)" if not whole_region
                     else f"{rel} (whole)",
                     "chars": len(region_text)})
    parts.append(region_text)

    def render(items: list[tuple[str, str, str]], title: str) -> None:
        if not items:
            return
        parts.append(f"\n# ── {title} ──")
        for kind, label, text in items:
            parts.append(f"# {label}\n{text}")
            sections.append({"kind": kind, "label": label,
                             "chars": len(text)})

    render(import_items, "import surface: definitions this file uses")
    render(test_items, "covering tests (verify will run these)")

    text = "\n".join(parts)
    # Global backstop: if shares + headers still overflow the stated
    # budget, drop whole items (tests first, then imports) — recorded.
    droppable = [("tests", test_items), ("imports", import_items)]
    while len(text) > budget and any(items for _k, items in droppable):
        for kind, items in droppable:
            if items and len(text) > budget:
                k2, label, body = items.pop()
                trimmed.append(TrimmedItem(
                    k2, label, len(body),
                    "pack budget backstop: dropped lowest-priority "
                    "section item"))
                sections[:] = [s for s in sections
                               if not (s["kind"] == k2
                                       and s["label"] == label)]
                parts = [p for p in parts
                         if not p.startswith(f"# {label}\n")]
                text = "\n".join(parts)
    if len(text) > budget:
        text = (text[:budget]
                + "\n# [context pack: hard-truncated at the pack "
                "budget]")
        trimmed.append(TrimmedItem(
            "target", rel, len(current) - budget,
            "pack exceeded the budget with the target region alone; "
            "hard-truncated"))
    return ContextPack(
        rel=rel, mode="pack", text=text, budget_chars=budget,
        whole_chars=len(current), sections=sections, trimmed=trimmed,
        notes=notes, region_ranges=ranges, omitted_defs=omitted,
        covering_tests=covered)
