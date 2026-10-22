"""ApkForge safe patch engine — ReVanced-fingerprint-inspired,
Devon-native, fail-closed (design §3.4).

A patch is a JSON document:

    {"name": "license-check-neutralize",
     "target_apk_sha256": "…",                (optional; checked when set)
     "match": {"class": "com/example/LicenseContentProvider",
               "method": "onCreate", "prototype": "()Z",
               "fingerprint": {"strings": ["license"],
                               "opcodes": ["invoke-virtual",
                                           "move-result"]}},
     "op": "replace_body" | "return_const" | "insert_before" |
           "insert_after" | "replace_seq",
     "smali": "…lines…",                      (body / inserted / replacement)
     "value": "1",                            (return_const)
     "expect_matches": 1,
     "note": "why this patch exists"}

The engine locates candidate methods in the decoded smali by
class + method + prototype, cross-checks the fingerprint (strings
present in the body; opcode sequence appearing in order), and
requires the surviving match count to equal ``expect_matches``
EXACTLY.  Any deviation — zero matches, two matches, fingerprint
miss — refuses the patch and changes nothing; the refusal is
journaled like every other op.  Nothing anywhere else in the class
freehand-writes smali files: brain-authored and owner-authored
patches pass through this same gate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .edit import apply_edit, decoded_dir
from .project import Project

__all__ = ["PatchError", "apply_patch", "parse_patch", "find_matches"]


class PatchError(RuntimeError):
    """A designed patch refusal/failure — names the exact cause."""


_OPS = ("replace_body", "return_const", "insert_before",
        "insert_after", "replace_seq")

_METHOD_RE = re.compile(
    r"^\.method\s+(?P<flags>.*?)\s*(?P<name>[^\s(]+)"
    r"(?P<proto>\([^\s]*)$")


def parse_patch(doc: Any) -> dict[str, Any]:
    if isinstance(doc, str):
        import json

        try:
            doc = json.loads(doc)
        except json.JSONDecodeError as exc:
            raise PatchError(f"patch is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise PatchError("patch must be a JSON object")
    match = doc.get("match") or {}
    for key in ("class", "method"):
        if not match.get(key):
            raise PatchError(f"patch.match.{key} is required")
    op = doc.get("op") or ""
    if op not in _OPS:
        raise PatchError(f"patch.op must be one of {_OPS} — got {op!r}")
    if op != "return_const" and not str(doc.get("smali") or "").strip():
        raise PatchError(f"patch {op} needs a 'smali' body")
    return doc


def _smali_roots(proj: Project) -> list[Path]:
    ddir = decoded_dir(proj)
    if not ddir.exists():
        return []
    roots = [ddir / "smali"]
    roots += sorted(p for p in ddir.glob("smali_classes*")
                    if p.is_dir())
    return [r for r in roots if r.exists()]


def _class_file(proj: Project, class_name: str) -> Path | None:
    rel = class_name.strip().strip("L;").replace(".", "/") + ".smali"
    for root in _smali_roots(proj):
        cand = root / rel
        if cand.exists():
            return cand
    return None


class _Method:
    __slots__ = ("path", "start", "end", "lines", "name", "proto",
                 "header")

    def __init__(self, path: Path, start: int, lines: list[str]):
        self.path = path
        self.start = start
        self.lines = lines
        self.end = start + len(lines) - 1
        m = _METHOD_RE.match(lines[0].strip())
        self.header = lines[0]
        self.name = m.group("name") if m else ""
        self.proto = m.group("proto") if m else ""


def _methods_in(path: Path) -> list[_Method]:
    lines = path.read_text(encoding="utf-8",
                           errors="replace").splitlines()
    out: list[_Method] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(".method"):
            j = i
            while j < len(lines) and \
                    lines[j].strip() != ".end method":
                j += 1
            if j < len(lines):
                out.append(_Method(path, i, lines[i:j + 1]))
                i = j
        i += 1
    return out


def _instruction_lines(method: _Method) -> list[tuple[int, str, str]]:
    """(relative index, opcode, raw line) for real instruction
    lines — directives, labels and comments excluded."""
    out = []
    for rel, line in enumerate(method.lines):
        s = line.strip()
        if not s or s.startswith((".", ":", "#")):
            continue
        out.append((rel, s.split()[0].rstrip(","), line))
    return out


def _fingerprint_ok(method: _Method, fingerprint: dict) -> bool:
    body = "\n".join(method.lines)
    for s in fingerprint.get("strings") or []:
        if str(s) not in body:
            return False
    want = [str(o) for o in (fingerprint.get("opcodes") or [])]
    if want:
        have = [op for _rel, op, _raw in _instruction_lines(method)]
        pos = 0
        for op in have:
            if pos < len(want) and op == want[pos]:
                pos += 1
        if pos < len(want):
            return False
    return True


def _seq_span(method: _Method,
              fingerprint: dict) -> tuple[int, int] | None:
    """Relative line span [first, last] of the fingerprint opcode
    sequence inside the method (for insert/replace_seq anchoring)."""
    want = [str(o) for o in (fingerprint.get("opcodes") or [])]
    if not want:
        return None
    pos = 0
    first: int | None = None
    last: int | None = None
    for rel, op, _raw in _instruction_lines(method):
        if pos < len(want) and op == want[pos]:
            if first is None:
                first = rel
            last = rel
            pos += 1
    if pos == len(want) and first is not None:
        return (first, last)  # type: ignore[return-value]
    return None


def find_matches(proj: Project, match: dict) -> list[_Method]:
    path = _class_file(proj, str(match["class"]))
    if path is None:
        return []
    fingerprint = match.get("fingerprint") or {}
    out = []
    for method in _methods_in(path):
        if method.name != str(match["method"]):
            continue
        proto = str(match.get("prototype") or "")
        if proto and method.proto != proto:
            continue
        if fingerprint and not _fingerprint_ok(method, fingerprint):
            continue
        out.append(method)
    return out


def _return_const_body(proto: str, value: str) -> list[str]:
    ret = proto.split(")")[-1] if ")" in proto else ""
    v = str(value).strip()
    if ret in ("Z", "B", "C", "S", "I"):
        n = int(v, 0)
        hx = f"{n:#x}" if n >= 0 else f"-{abs(n):#x}"
        if -8 <= n <= 7:
            ins = f"const/4 v0, {hx}"
        elif -32768 <= n <= 32767:
            ins = f"const/16 v0, {hx}"
        else:
            ins = f"const v0, {hx}"
        return ["    .locals 1", f"    {ins}", "    return v0"]
    if ret == "J":
        return ["    .locals 2", f"    const-wide/16 v0, {int(v, 0)}",
                "    return-wide v0"]
    if ret == "Ljava/lang/String;":
        esc = v.replace("\\", "\\\\").replace('"', '\\"')
        return ["    .locals 1", f'    const-string v0, "{esc}"',
                "    return-object v0"]
    if ret == "V":
        return ["    .locals 0", "    return-void"]
    raise PatchError(
        f"return_const doesn't know return type {ret!r} — use "
        "replace_body with an explicit smali body instead")


def apply_patch(proj: Project, doc: Any) -> dict[str, Any]:
    """Validate, locate, and apply one patch document — or refuse
    it whole, journaled either way."""
    patch = parse_patch(doc)
    name = str(patch.get("name") or "unnamed-patch")
    match = patch["match"]
    want_sha = str(patch.get("target_apk_sha256") or "")
    if want_sha and want_sha != proj.sha256:
        proj.journal("patch_refused",
                     f"{name}: target_apk_sha256 does not match "
                     f"this project's original ({proj.sha256[:16]}…)")
        raise PatchError(
            f"patch {name!r} targets sha256 {want_sha[:16]}… but "
            f"this project's original is {proj.sha256[:16]}… — "
            "refused, nothing changed")
    if not _smali_roots(proj):
        raise PatchError("project is not decoded — run decode first")
    matches = find_matches(proj, match)
    expect = int(patch.get("expect_matches", 1))
    if len(matches) != expect:
        proj.journal(
            "patch_refused",
            f"{name}: expected {expect} match(es) for "
            f"{match['class']}->{match['method']}"
            f"{match.get('prototype', '')}, found {len(matches)} "
            "— fail-closed, nothing changed",
            patch=name)
        raise PatchError(
            f"patch {name!r} refused: expected exactly {expect} "
            f"matching method(s), found {len(matches)} — "
            "fail-closed, nothing changed")
    method = matches[0]
    rel = str(method.path.relative_to(decoded_dir(proj)))
    op = patch["op"]
    fingerprint = match.get("fingerprint") or {}
    span = None
    if op in ("insert_before", "insert_after", "replace_seq"):
        span = _seq_span(method, fingerprint)
        if span is None:
            raise PatchError(
                f"patch {name!r}: op {op} needs a fingerprint "
                "opcode sequence that matches inside the method — "
                "none found, nothing changed")
    smali_lines = [ln for ln in
                   str(patch.get("smali") or "").splitlines()]

    def mutate() -> None:
        lines = method.path.read_text(
            encoding="utf-8", errors="replace").splitlines()
        head = lines[:method.start]
        tail = lines[method.end + 1:]
        if op == "replace_body":
            body = [method.header, *smali_lines, ".end method"]
        elif op == "return_const":
            body = [method.header,
                    *_return_const_body(method.proto,
                                        str(patch.get("value", "1"))),
                    ".end method"]
        elif op == "insert_before":
            at = method.start + span[0]  # type: ignore[index]
            new = (lines[:at] + smali_lines
                   + lines[at:])
            method.path.write_text("\n".join(new) + "\n",
                                   encoding="utf-8")
            return
        elif op == "insert_after":
            at = method.start + span[1] + 1  # type: ignore[index]
            new = (lines[:at] + smali_lines + lines[at:])
            method.path.write_text("\n".join(new) + "\n",
                                   encoding="utf-8")
            return
        elif op == "replace_seq":
            first = method.start + span[0]  # type: ignore[index]
            last = method.start + span[1]  # type: ignore[index]
            new = lines[:first] + smali_lines + lines[last + 1:]
            method.path.write_text("\n".join(new) + "\n",
                                   encoding="utf-8")
            return
        else:  # pragma: no cover — parse_patch gates ops
            raise PatchError(f"unsupported op {op}")
        method.path.write_text("\n".join(head + body + tail) + "\n",
                               encoding="utf-8")

    result = apply_edit(
        proj, "patch",
        f"{name}: {op} {match['class']}->{match['method']}"
        f"{match.get('prototype', '')}",
        [rel], mutate, patch=name,
        note=str(patch.get("note") or ""))
    result["patch"] = name
    return result
