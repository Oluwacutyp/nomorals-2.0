"""The Android reference graph (Pass 2) — row 8 of the census.

Devon already decompiles APKs through ApkForge's provisioned jadx
(``forge.jadx`` → a tree of readable Java). What she could not do
is *query* that tree: who calls this class, what does that method
touch, which classes implement this interface. This module builds
a class/method/reference graph over a jadx sources tree and
answers those questions as structured data, which the engine
folds into Evidence — never as a text dump.

The builder is a heuristic scanner over decompiled Java, and it
says so wherever its answers land: references resolve by name, so
overloaded methods merge into one node, reflection and dynamic
dispatch are invisible to it, method bodies are delimited by
declaration spans rather than an AST, and obfuscated names stay
obfuscated. Within those stated limits the answers are exact
counts over the tree that was scanned — the scan is the evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "GraphError", "build_graph",
    "sources_root_of", "callers", "callees", "class_profile",
    "search", "LIMITATIONS",
]

PROVIDER_ID = "devon.android-graph"
PROVIDER_NAME = "Devon Android reference graph over jadx output"
PROVIDER_VERSION = "1"

MAX_FILES = 20000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SAMPLE = 100

LIMITATIONS = (
    "references resolve by name over decompiled sources: "
    "overloaded methods share one node, and calls through "
    "reflection, dynamic proxies or native methods are invisible",
    "method bodies are declaration spans in the decompiled text, "
    "not an AST — a reference is credited to the method whose "
    "span contains it",
    "comments and string literals are stripped before scanning, "
    "so names mentioned only in prose do not create edges",
    "obfuscated names stay obfuscated — the graph mirrors the "
    "decompile it was built from (jadx, mapping-restored when "
    "ApkForge applied a mapping)",
)


class GraphError(RuntimeError):
    """The graph could not be built or the query has no referent."""


_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_TYPE_RE = re.compile(
    r"\b(class|interface|enum|@interface)\s+([A-Za-z_]\w*)"
    r"(?:\s+extends\s+([\w.]+))?(?:\s+implements\s+([\w.,\s]+?))?"
    r"\s*\{")
_METHOD_RE = re.compile(
    r"(?:^|\n)\s*(?:@[\w.]+(?:\([^)]*\))?\s*)*"
    r"(?:public|private|protected|static|final|abstract|"
    r"synchronized|native|strictfp|\s)+?"
    r"[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[\w.,\s]+)?\{")
_FIELD_TYPE_RE = re.compile(
    r"(?:private|protected|public)\s+(?:static\s+)?(?:final\s+)?"
    r"([A-Z][\w.]*)\s+(?:\w+\s*[,;=])")
_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(")
_BARE_CALL_RE = re.compile(r"(?<![\w.])(\w+)\s*\(")
_NEW_RE = re.compile(r"\bnew\s+([A-Z][\w.]*)\s*\(")
_TYPE_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")

_JAVA_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "new",
    "super", "this", "synchronized", "assert", "throw", "case",
    "do", "else", "try", "instanceof",
})


def _strip_noise(text: str) -> str:
    """Blank out comments and string/char literals (keeping
    newlines, so line numbers survive)."""
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                out.append("  ")
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                state = "string"
                out.append(" ")
                i += 1
                continue
            if ch == "'":
                state = "char"
                out.append(" ")
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line_comment":
            out.append("\n" if ch == "\n" else " ")
            if ch == "\n":
                state = "code"
            i += 1
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        # string / char literal
        quote = '"' if state == "string" else "'"
        if ch == "\\" and i + 1 < n:
            out.append("  ")
            i += 2
            continue
        out.append("\n" if ch == "\n" else " ")
        if ch == quote or ch == "\n":
            state = "code"
        i += 1
    return "".join(out)


def _parse_file(path: Path, rel: str) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_FILE_BYTES:
        return None
    text = _strip_noise(raw.decode("utf-8", errors="replace"))
    pkg_match = _PACKAGE_RE.search(text)
    package = pkg_match.group(1) if pkg_match else ""
    type_match = _TYPE_RE.search(text)
    if not type_match:
        return None
    kind, simple = type_match.group(1), type_match.group(2)
    fqcn = f"{package}.{simple}" if package else simple
    record: dict[str, Any] = {
        "fqcn": fqcn, "simple": simple, "package": package,
        "kind": kind, "file": rel,
        "extends": type_match.group(3) or "",
        "implements": [x.strip() for x in
                       (type_match.group(4) or "").split(",")
                       if x.strip()],
        "methods": {}, "field_types": [], "_spans": [],
        "_text": text,
    }
    methods = list(_METHOD_RE.finditer(text))
    for idx, m in enumerate(methods):
        name = m.group(1)
        if name in _JAVA_KEYWORDS:
            continue
        start = m.start()
        end = methods[idx + 1].start() if idx + 1 < len(methods) \
            else len(text)
        line = text.count("\n", 0, start) + 1
        sig = " ".join(text[start:m.end()].split())[:200]
        record["methods"].setdefault(name, {"signature": sig,
                                            "line": line})
        # (span start, body start, span end, name): the body starts
        # after the declaration's opening brace, so a method's own
        # signature is never scanned as a call to itself
        record["_spans"].append((start, m.end(), end, name))
    record["field_types"] = sorted(set(_FIELD_TYPE_RE.findall(text)))
    return record


def sources_root_of(path: str | Path) -> Path:
    """Resolve a jadx output dir (or a sources dir, or a single
    tree) to the directory holding the .java files."""
    base = Path(path).expanduser()
    if not base.is_dir():
        raise GraphError(f"no directory at {path!r}")
    for candidate in (base / "sources", base):
        if candidate.is_dir() and any(candidate.rglob("*.java")):
            return candidate
    raise GraphError(
        f"no .java sources under {base} — this does not look like "
        "a jadx output tree")


def build_graph(sources: str | Path) -> dict[str, Any]:
    """Scan a jadx sources tree → the reference graph.

    Returns {"classes": {fqcn: record}, "edges": [...], "stats":
    {...}, "collisions": [...]}. Edge records are deduplicated
    with counts: {from_class, from_method, to_class, to_method,
    kind, count}.
    """
    root = sources_root_of(sources)
    files = sorted(root.rglob("*.java"))[:MAX_FILES]
    classes: dict[str, dict[str, Any]] = {}
    by_simple: dict[str, list[str]] = {}
    for path in files:
        rel = str(path.relative_to(root))
        record = _parse_file(path, rel)
        if record is None:
            continue
        classes[record["fqcn"]] = record
        by_simple.setdefault(record["simple"], []).append(
            record["fqcn"])
    collisions = sorted(s for s, lst in by_simple.items()
                        if len(lst) > 1)

    def resolve(simple: str, from_pkg: str) -> str | None:
        hits = by_simple.get(simple) or []
        if len(hits) == 1:
            return hits[0]
        for hit in hits:
            if hit.rsplit(".", 1)[0] == from_pkg:
                return hit
        return None

    edges: dict[tuple, dict[str, Any]] = {}

    def add_edge(frm: str, method: str, to: str, to_method: str,
                 kind: str) -> None:
        key = (frm, method, to, to_method, kind)
        row = edges.get(key)
        if row is None:
            edges[key] = {"from_class": frm, "from_method": method,
                          "to_class": to, "to_method": to_method,
                          "kind": kind, "count": 1}
        else:
            row["count"] += 1

    for fqcn, record in classes.items():
        text = record["_text"]
        pkg = record["package"]
        field_var_types: dict[str, str] = {}
        for m in re.finditer(
                r"\b([A-Z][\w.]*)\s+([a-z_]\w*)\s*[;=]", text):
            resolved = resolve(m.group(1), pkg)
            if resolved:
                field_var_types[m.group(2)] = resolved
        spans = record["_spans"] or [(0, 0, len(text), "")]
        for _start, body_start, end, method_name in spans:
            body = text[body_start:end]
            for call in _CALL_RE.finditer(body):
                owner, callee = call.group(1), call.group(2)
                target = None
                if owner[0].isupper():
                    target = resolve(owner, pkg)
                elif owner in field_var_types:
                    target = field_var_types[owner]
                elif owner in ("this", "super"):
                    target = fqcn
                if target is None or callee in _JAVA_KEYWORDS:
                    continue
                add_edge(fqcn, method_name, target, callee, "call")
            for new_m in _NEW_RE.finditer(body):
                target = resolve(new_m.group(1), pkg)
                if target and target != fqcn:
                    add_edge(fqcn, method_name, target,
                             new_m.group(1), "instantiates")
            for bare in _BARE_CALL_RE.finditer(body):
                name = bare.group(1)
                if name in record["methods"] and \
                        name not in _JAVA_KEYWORDS:
                    add_edge(fqcn, method_name, fqcn, name, "call")
    out_classes = {}
    for fqcn, record in classes.items():
        out_classes[fqcn] = {
            k: v for k, v in record.items()
            if not k.startswith("_")}
    edge_list = sorted(edges.values(),
                       key=lambda e: (e["from_class"],
                                      e["from_method"],
                                      e["to_class"], e["to_method"]))
    return {
        "classes": out_classes,
        "edges": edge_list,
        "collisions": collisions,
        "stats": {
            "source_root": str(root),
            "files_scanned": len(files),
            "classes": len(out_classes),
            "methods": sum(len(c["methods"])
                           for c in out_classes.values()),
            "edges": len(edge_list),
            "references": sum(e["count"] for e in edge_list),
        },
    }


def _split_target(target: str) -> tuple[str, str]:
    """'com.x.Foo.bar' / 'Foo.bar' / 'Foo' → (class part, method)."""
    target = (target or "").strip()
    if "." in target:
        head, _, last = target.rpartition(".")
        if last and last[0].islower():
            return head, last
    return target, ""


def _resolve_class(graph: dict[str, Any], ref: str) -> str:
    classes = graph["classes"]
    if ref in classes:
        return ref
    hits = [fq for fq, rec in classes.items()
            if rec["simple"] == ref]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise GraphError(f"no class matches {ref!r} in this graph")
    raise GraphError(
        f"{ref!r} is ambiguous in this graph: " + ", ".join(hits))


def callers(graph: dict[str, Any], target: str) -> dict[str, Any]:
    """Who calls X — X a class ('Foo', 'com.x.Foo') or a method
    ('Foo.bar'). Every edge into the target, grouped by caller."""
    class_ref, method = _split_target(target)
    fqcn = _resolve_class(graph, class_ref)
    rows = [e for e in graph["edges"]
            if e["to_class"] == fqcn
            and (not method or e["to_method"] == method)]
    return {
        "target": target, "resolved_class": fqcn,
        "method": method,
        "callers": rows[:MAX_SAMPLE],
        "caller_count": len(rows),
        "references": sum(e["count"] for e in rows),
        "truncated": len(rows) > MAX_SAMPLE,
    }


def callees(graph: dict[str, Any], target: str) -> dict[str, Any]:
    """What X touches — X a class or one of its methods."""
    class_ref, method = _split_target(target)
    fqcn = _resolve_class(graph, class_ref)
    rows = [e for e in graph["edges"]
            if e["from_class"] == fqcn
            and (not method or e["from_method"] == method)]
    return {
        "target": target, "resolved_class": fqcn,
        "method": method,
        "callees": rows[:MAX_SAMPLE],
        "callee_count": len(rows),
        "references": sum(e["count"] for e in rows),
        "truncated": len(rows) > MAX_SAMPLE,
    }


def class_profile(graph: dict[str, Any],
                  target: str) -> dict[str, Any]:
    record = graph["classes"][_resolve_class(graph, target)]
    inbound = [e for e in graph["edges"]
               if e["to_class"] == record["fqcn"]]
    outbound = [e for e in graph["edges"]
                if e["from_class"] == record["fqcn"]]
    return {
        "class": record,
        "inbound_count": len(inbound),
        "outbound_count": len(outbound),
        "callers_sample": inbound[:20],
        "callees_sample": outbound[:20],
    }


def search(graph: dict[str, Any], text: str) -> dict[str, Any]:
    needle = (text or "").lower()
    if not needle:
        raise GraphError("search needs text")
    class_hits = [fq for fq in graph["classes"]
                  if needle in fq.lower()]
    method_hits = [
        {"class": fq, "method": name}
        for fq, rec in graph["classes"].items()
        for name in rec["methods"] if needle in name.lower()]
    return {"text": text,
            "classes": class_hits[:MAX_SAMPLE],
            "class_count": len(class_hits),
            "methods": method_hits[:MAX_SAMPLE],
            "method_count": len(method_hits)}


def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, graph: dict[str, Any],
                  graph_ref: str = "") -> dict[str, Any]:
    """Record the graph census itself as Evidence (the query
    answers are recorded separately by the engine)."""
    stats = graph["stats"]
    rec = store.record(
        predicate_type="android.class_graph", operation="graph.build",
        provider_id=PROVIDER_ID, provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        parameters={"source": stats["source_root"]},
        result={"classes": stats["classes"],
                "methods": stats["methods"],
                "edges": stats["edges"],
                "references": stats["references"],
                "files_scanned": stats["files_scanned"],
                "name_collisions": graph["collisions"][:32]},
        subject_sha256=subject_sha256, subject_name=subject_name,
        limitations=list(LIMITATIONS),
        locations=[{"type": "artifact-path",
                    "value": stats["source_root"]}])
    return {"evidence_ids": [rec.evidence_id],
            "graph_evidence_id": rec.evidence_id,
            "unknowns": []}
