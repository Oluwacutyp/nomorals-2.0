"""Parity runner — the machine-checked parity TOML.

Port of the *design* of storytold/pdfcraft ``parity/*.toml`` + ``cargo
xtask parity``: a TOML file listing real fixture PDFs, the operations
applied to each, and the expected outcomes — and a runner that EXECUTES
them against the real engine and reports pass/fail per case with
measured values.  Parity is measured, never asserted: a case whose
expectation is wrong fails loudly, a missing fixture fails (never a
silent skip), and an operation that raises unexpectedly fails with
the exception named.

Schema (version 1):

    [schema]
    version = 1

    [[case]]
    id = "simple-inspect"
    fixture = "simple-3page.pdf"
    ops = ["inspect"]
    [case.expect]
    page_count = 3
    text_contains = ["Alpha beta gamma 1"]
    repaired = false

Ops: inspect, incremental_metadata, incremental_rotate,
incremental_annotate, incremental_stamp, open_repaired, open_refused.
Expect keys: page_count, text_contains (list, checked against the
resulting file), title, prefix_preserved (bool — the edited file
starts with the original bytes), repaired (bool), error_kind.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import DocumentError
from .edit import PdfEditor
from .model import PdfDocument

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "pdfcraft-parity.toml"


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
    if not isinstance(data.get("schema"), dict) \
            or data["schema"].get("version") != 1:
        raise DocumentError(
            f"parity TOML {path} is not schema version 1",
            kind="unsupported_format")
    if not isinstance(data.get("case"), list):
        raise DocumentError(f"parity TOML {path} has no [[case]] rows",
                            kind="parse")
    return data


def _run_case(case: dict[str, Any], fixtures_dir: Path
              ) -> dict[str, Any]:
    """Execute one [[case]]; returns {id, status, measured, expected,
    detail}.  A failure is a reported row, never a raised exception."""
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
        result = original
        for op in ops:
            result = _apply_op(op, result, case)
        doc = PdfDocument.from_bytes(result)
        measured["page_count"] = doc.page_count
        measured["repaired"] = doc.repaired
        if ops and ops != ["open_repaired"]:
            measured["prefix_preserved"] = result[:len(original)] == original \
                if result is not original else True
        for want in expected.get("text_contains", []):
            text = doc.text()
            measured.setdefault("text_contains", {})[want] = want in text
        for want in expected.get("annotations_contains", []):
            found = any(a.get("contents") == want
                        for a in doc.annotations())
            measured.setdefault("annotations_contains", {})[want] = found
        if "title" in expected:
            measured["title"] = doc.info.get("Title")
        if "error_kind" in expected:
            measured["error_kind"] = None
    except DocumentError as exc:
        measured["error_kind"] = exc.kind
        if expected.get("error_kind") == exc.kind:
            return {"id": case_id, "status": "pass",
                    "measured": measured, "expected": expected,
                    "detail": f"refused honestly with kind "
                              f"{exc.kind!r}"}
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised DocumentError kind {exc.kind!r}: "
                          f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - the runner reports, never raises
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"unexpected {type(exc).__name__}: {exc}"}
    failures: list[str] = []
    if "page_count" in expected \
            and measured.get("page_count") != expected["page_count"]:
        failures.append(
            f"page_count {measured.get('page_count')} != "
            f"{expected['page_count']}")
    if "repaired" in expected \
            and measured.get("repaired") != expected["repaired"]:
        failures.append(
            f"repaired {measured.get('repaired')} != "
            f"{expected['repaired']}")
    if "prefix_preserved" in expected \
            and measured.get("prefix_preserved") \
            != expected["prefix_preserved"]:
        failures.append("prefix not preserved by the incremental save")
    for want, found in measured.get("text_contains", {}).items():
        if not found:
            failures.append(f"text {want!r} missing from the result")
    for want, found in measured.get("annotations_contains", {}).items():
        if not found:
            failures.append(
                f"annotation {want!r} missing from the result")
    if "title" in expected \
            and measured.get("title") != expected["title"]:
        failures.append(
            f"title {measured.get('title')!r} != "
            f"{expected['title']!r}")
    if "error_kind" in expected:
        failures.append(
            f"expected refusal kind {expected['error_kind']!r}, "
            f"the file opened instead")
    if failures:
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": "; ".join(failures)}
    return {"id": case_id, "status": "pass",
            "measured": measured, "expected": expected,
            "detail": "all expectations measured, not asserted"}


def _apply_op(op: str, data: bytes, case: dict[str, Any]) -> bytes:
    """One parity op over ``data``; returns the resulting bytes."""
    if op == "inspect":
        return data
    if op == "open_repaired":
        return data
    if op == "open_refused":
        # The refusal happens in _run_case's PdfDocument.from_bytes;
        # reaching here means the file opened — which the expectation
        # check will fail on.
        return data
    editor = PdfEditor.from_bytes(data)
    if op == "incremental_metadata":
        editor.set_info(title=str(case.get("title", "Parity Title")))
        return editor.save_incremental()
    if op == "incremental_rotate":
        editor.rotate_page(int(case.get("page", 1)),
                           int(case.get("degrees", 90)))
        return editor.save_incremental()
    if op == "incremental_annotate":
        editor.add_text_annotation(
            int(case.get("page", 1)),
            tuple(case.get("rect", [72, 700, 200, 720])),  # type: ignore[arg-type]
            str(case.get("annotation", "Parity note.")))
        return editor.save_incremental()
    if op == "incremental_stamp":
        editor.stamp_text(int(case.get("page", 1)),
                          str(case.get("stamp", "PARITY STAMP")))
        return editor.save_incremental()
    raise DocumentError(f"unknown parity op {op!r}",
                        kind="invalid_argument")


def run_parity(path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None) -> dict[str, Any]:
    """Run every [[case]] in a parity TOML; returns {passed, failed,
    cases}.  ``fixtures_dir`` defaults to the test fixtures directory
    (``tests/fixtures/pdfcraft`` relative to the repo root)."""
    data = load_parity(path)
    if fixtures_dir is None:
        fixtures_dir = (Path(__file__).resolve().parent.parent.parent
                        .parent / "tests" / "fixtures" / "pdfcraft")
    fixtures = Path(fixtures_dir)
    cases = [_run_case(case, fixtures) for case in data["case"]]
    passed = sum(1 for c in cases if c["status"] == "pass")
    return {"passed": passed, "failed": len(cases) - passed,
            "cases": cases}
