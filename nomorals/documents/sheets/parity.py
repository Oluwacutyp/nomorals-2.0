"""Parity runner — the machine-checked parity TOML.

Port of the *design* of the wordcraft parity runner (lane 100): a
TOML file listing real fixture .xlsx files, the operations applied
to each, and the expected outcomes — and a runner that EXECUTES
them against the real engine and reports pass/fail per case with
measured values. Parity is measured, never asserted: a case whose
expectation is wrong fails loudly, a missing fixture fails (never
a silent skip), and an operation that raises unexpectedly fails
with the exception named.

Schema (version 1):

    schema_version = 1

    [[cases]]
    id = "simple-inspect"
    fixture = "simple.xlsx"
    ops = ["inspect"]
    [cases.expect]
    sheets = ["Data", "Summary"]
    title = "Simple Fixture"
    [cases.expect.values]
    "Data!D6" = 51

Ops: inspect, recalc, set-value, set-formula, restyle,
insert-rows, insert-cols, add-sheet, set-metadata, open-refused.
Expect keys: sheets, title, author, values ({"Sheet!Ref": value}),
formulas ({"Sheet!Ref": "SUM(...)"}), cached ({"Sheet!Ref":
value-or-error-display}), parts_preserved (bool — every part
except the ones the op may touch stays byte-identical),
untouched_cells_changed (int), error_kind.

Unlike the wordcraft runner, ``failed`` here is the LIST of
failed case ids (not a count) — the gridcraft tool surface
reports the names, not just the number.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import DocumentError
from .calc import recalc
from .edit import SheetEditor
from .model import CellError, Workbook

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "sheets-parity.toml"

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent \
    / "tests" / "fixtures" / "gridcraft"

#: Parts an edit is allowed to touch; everything else must survive
#: byte-identical for ``parts_preserved`` to hold. Any worksheet
#: part counts (the sheet an edit touched), plus the shared
#: strings/styles/workbook/core plumbing an edit may rewrite.
_EDITABLE_PREFIXES = ("xl/worksheets/",)
_EDITABLE_PARTS = frozenset({
    "xl/sharedStrings.xml", "xl/styles.xml", "xl/workbook.xml",
    "docProps/core.xml", "[Content_Types].xml",
    "xl/_rels/workbook.xml.rels",
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


def _apply_op(op: str, editor: SheetEditor,
              args: dict[str, Any]) -> None:
    if op in ("inspect", "recalc", "open-refused"):
        return
    if op == "set-value":
        editor.set_value(str(args["sheet"]), str(args["ref"]),
                         args["value"])
    elif op == "set-formula":
        editor.set_formula(str(args["sheet"]), str(args["ref"]),
                           str(args["formula"]))
    elif op == "restyle":
        editor.restyle(str(args["sheet"]), str(args["ref"]),
                       **{k: v for k, v in args.items()
                          if k not in ("sheet", "ref")})
    elif op == "insert-rows":
        editor.insert_rows(str(args["sheet"]), int(args["at"]),
                           int(args.get("count", 1)))
    elif op == "insert-cols":
        editor.insert_cols(str(args["sheet"]), int(args["at"]),
                           int(args.get("count", 1)))
    elif op == "add-sheet":
        editor.add_sheet(str(args["name"]))
    elif op == "set-metadata":
        editor.set_metadata(**{k: str(v)
                               for k, v in args.items()})
    else:
        raise DocumentError(f"unknown parity op {op!r}",
                            kind="invalid_argument")


def _cell_key(ref: str) -> tuple[str, str]:
    sheet, _, cell = ref.rpartition("!")
    return sheet, cell


def _measure(wb: Workbook, expected: dict[str, Any],
             measured: dict[str, Any]) -> None:
    for key in expected.get("values", {}):
        sheet, cell = _cell_key(key)
        measured.setdefault("values", {})[key] = \
            wb.sheet(sheet).value(cell)
    for key in expected.get("formulas", {}):
        sheet, cell = _cell_key(key)
        measured.setdefault("formulas", {})[key] = \
            wb.sheet(sheet).cell(cell).formula
    for key in expected.get("cached", {}):
        sheet, cell = _cell_key(key)
        cached = wb.sheet(sheet).cell(cell).cached
        if isinstance(cached, CellError):
            cached = cached.display
        measured.setdefault("cached", {})[key] = cached


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
                Workbook.from_bytes(original)
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
        wb = Workbook.from_bytes(original)
        editor = SheetEditor(wb)
        for op in ops:
            _apply_op(op, editor, args)
        recalc(wb)
        saved = editor.to_bytes()
        wb = Workbook.from_bytes(saved)
        measured["sheets"] = wb.sheet_names
        measured["title"] = wb.core_properties.get("title", "")
        measured["author"] = wb.core_properties.get("creator", "")
        _measure(wb, expected, measured)
        if expected.get("parts_preserved") is not None:
            report = editor.workbook.fidelity_report(original)
            changed = set(report["changed_parts"])
            allowed = changed <= _EDITABLE_PARTS or all(
                part in _EDITABLE_PARTS
                or part.startswith(_EDITABLE_PREFIXES)
                for part in changed)
            measured["parts_preserved"] = (
                allowed
                and report["untouched_cells_changed"] == 0)
            measured["parts_changed"] = sorted(changed)
        if "untouched_cells_changed" in expected:
            report = editor.workbook.fidelity_report(original)
            measured["untouched_cells_changed"] = report[
                "untouched_cells_changed"]
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
        if key in ("values", "formulas", "cached"):
            for ref, want_value in want.items():
                got = measured.get(key, {}).get(ref)
                if got != want_value:
                    failures.append(f"{key}[{ref}]: measured "
                                    f"{got!r}, expected {want_value!r}")
        elif key in ("parts_preserved", "untouched_cells_changed",
                     "sheets", "title", "author", "error_kind"):
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
    cases} where ``failed`` is the list of failed case ids. A
    missing fixture file fails its case — never a skip."""
    data = load_parity(path)
    base = Path(fixtures_dir) if fixtures_dir is not None \
        else _FIXTURES_DIR
    cases = [_run_case(case, base) for case in data["cases"]]
    passed = sum(1 for c in cases if c["status"] == "pass")
    return {"passed": passed,
            "failed": [c["id"] for c in cases
                       if c["status"] != "pass"],
            "cases": cases}
