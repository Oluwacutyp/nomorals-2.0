"""Parity runner — the machine-checked parity TOML.

Port of the *design* of the pdfcraft parity runner (lane 99): a TOML
file listing real fixture .docx files, the operations applied to
each, and the expected outcomes — and a runner that EXECUTES them
against the real engine and reports pass/fail per case with measured
values. Parity is measured, never asserted: a case whose expectation
is wrong fails loudly, a missing fixture fails (never a silent
skip), and an operation that raises unexpectedly fails with the
exception named.

Schema (version 1):

    schema_version = 1

    [[cases]]
    id = "simple-inspect"
    fixture = "simple.docx"
    ops = ["inspect"]
    [cases.expect]
    paragraphs = 4
    title = "Simple Fixture"

Ops: inspect, replace, restyle-run, add-heading, add-table,
add-list, set-metadata, open-refused. Expect keys: paragraphs,
tables, headers, footers, images, title, author, text_contains
(list — every entry must appear in the resulting text),
parts_preserved (bool — every part except the ones the op may
touch stays byte-identical), untouched_paragraphs_changed (int),
error_kind.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import DocumentError
from .edit import WordEditor
from .model import WordDocument

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "word-parity.toml"

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent \
    / "tests" / "fixtures" / "wordcraft"

#: Parts an edit is allowed to touch; everything else must survive
#: byte-identical for ``parts_preserved`` to hold.
_EDITABLE_PARTS = frozenset({
    "word/document.xml", "word/styles.xml", "word/numbering.xml",
    "docProps/core.xml", "[Content_Types].xml",
    "word/_rels/document.xml.rels",
})


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise DocumentError(f"cannot read parity TOML {path}: {exc}",
                            kind="io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DocumentError(f"bad parity TOML {path}: {exc}",
                            kind="parse") from exc
    if data.get("schema_version") != 1:
        raise DocumentError(
            f"parity TOML {path} is not schema version 1",
            kind="unsupported_format")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise DocumentError(f"parity TOML {path} has no [[case]] rows",
                            kind="parse")
    return data


def _apply_op(op: str, editor: WordEditor,
              args: dict[str, Any]) -> None:
    if op == "inspect":
        return
    if op == "replace":
        editor.replace_text(str(args["old"]), str(args["new"]))
    elif op == "restyle-run":
        editor.set_run_format(
            int(args["para"]), int(args["run"]),
            bold=args.get("bold"), italic=args.get("italic"),
            color=args.get("color"),
            size_half_points=args.get("size_half_points"))
    elif op == "add-heading":
        editor.add_heading(str(args["text"]), int(args.get("level", 1)))
    elif op == "add-table":
        editor.add_table(list(args["headers"]),
                         [list(r) for r in args["rows"]])
    elif op == "add-list":
        editor.add_list_item(str(args["text"]),
                             ordered=bool(args.get("ordered", False)))
    elif op == "set-metadata":
        editor.set_metadata(**{k: str(v)
                               for k, v in args.items()})
    elif op == "open-refused":
        return
    else:
        raise DocumentError(f"unknown parity op {op!r}",
                            kind="invalid_argument")


def _run_case(case: dict[str, Any], fixtures_dir: Path
              ) -> dict[str, Any]:
    """Execute one [[case]]; returns {id, status, measured, expected,
    detail}. A failure is a reported row, never a raised exception."""
    case_id = str(case.get("id", "?"))
    expected = case.get("expect", {})
    measured: dict[str, Any] = {}
    try:
        fixture = case.get("fixture", "")
        fixture_path = fixtures_dir / str(fixture)
        if not fixture_path.is_file():
            return {"id": case_id, "status": "fail",
                    "measured": {"fixture_missing": str(fixture)},
                    "expected": expected,
                    "detail": f"fixture {fixture!r} is not on disk — "
                              f"a missing fixture is a failure, never "
                              f"a silent skip"}
        original = fixture_path.read_bytes()
        ops = case.get("ops", ["inspect"])
        args = case.get("op_args", {})
        if "open-refused" in ops:
            try:
                WordDocument.from_bytes(original)
            except DocumentError as exc:
                measured["error_kind"] = exc.kind
                if expected.get("error_kind") == exc.kind:
                    return {"id": case_id, "status": "pass",
                            "measured": measured, "expected": expected,
                            "detail": "refused honestly with kind "
                                      f"{exc.kind!r}"}
                return {"id": case_id, "status": "fail",
                        "measured": measured, "expected": expected,
                        "detail": f"refused with kind {exc.kind!r}, "
                                  f"expected {expected.get('error_kind')!r}"}
            return {"id": case_id, "status": "fail",
                    "measured": measured, "expected": expected,
                    "detail": "expected a refusal, the file opened"}
        editor = WordEditor.from_bytes(original)
        for op in ops:
            _apply_op(op, editor, args)
        saved = editor.save()
        doc = WordDocument.from_bytes(saved)
        measured["paragraphs"] = len(doc.paragraphs)
        measured["tables"] = len(doc.tables)
        measured["headers"] = len(doc.headers)
        measured["footers"] = len(doc.footers)
        measured["images"] = len(doc.images)
        measured["title"] = doc.core_properties.get("title", "")
        measured["author"] = doc.core_properties.get("author", "")
        text = doc.plain_text()
        for want in expected.get("text_contains", []):
            measured.setdefault("text_contains", {})[want] = want in text
        if expected.get("parts_preserved") is not None:
            # the EDITOR's document carries the touched-paragraph
            # tracking; the re-parsed doc does not
            report = editor.document.fidelity_report(original)
            changed = set(report["parts_changed"])
            measured["parts_preserved"] = (
                changed <= _EDITABLE_PARTS
                and report["untouched_paragraphs_changed"] == 0)
            measured["parts_changed"] = sorted(changed)
        if "untouched_paragraphs_changed" in expected:
            report = editor.document.fidelity_report(original)
            measured["untouched_paragraphs_changed"] = report[
                "untouched_paragraphs_changed"]
        if "error_kind" in expected:
            measured["error_kind"] = None
    except DocumentError as exc:
        measured["error_kind"] = exc.kind
        if expected.get("error_kind") == exc.kind:
            return {"id": case_id, "status": "pass",
                    "measured": measured, "expected": expected,
                    "detail": f"refused honestly with kind {exc.kind!r}"}
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised DocumentError kind {exc.kind!r}: "
                          f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - a case must report, not raise
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised {type(exc).__name__}: {exc}"}
    failures: list[str] = []
    for key, want in expected.items():
        if key == "text_contains":
            for text_want, found in measured.get(
                    "text_contains", {}).items():
                if not found:
                    failures.append(
                        f"text {text_want!r} not in result")
        elif key in ("parts_preserved", "untouched_paragraphs_changed",
                     "paragraphs", "tables", "headers", "footers",
                     "images", "title", "author", "error_kind"):
            got = measured.get(key)
            if got != want:
                failures.append(f"{key}: measured {got!r}, "
                                f"expected {want!r}")
    if failures:
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": "; ".join(failures)}
    return {"id": case_id, "status": "pass",
            "measured": measured, "expected": expected,
            "detail": "all expectations met"}


def run_parity(path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None
               ) -> dict[str, Any]:
    """Run every case in the parity TOML; returns {passed, failed,
    cases}. A missing fixture file fails its case — never a skip."""
    data = load_parity(path)
    base = Path(fixtures_dir) if fixtures_dir is not None \
        else _FIXTURES_DIR
    cases = [_run_case(case, base) for case in data["cases"]]
    passed = sum(1 for c in cases if c["status"] == "pass")
    return {"passed": passed, "failed": len(cases) - passed,
            "cases": cases}
