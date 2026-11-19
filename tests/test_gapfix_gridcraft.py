"""GridCraft port — failing-first tests (lane 102, census 3A.9).

A real spreadsheet organ over real .xlsx bytes: OOXML package,
workbook model, formula parser/evaluator, calc dependency engine,
fidelity-preserving editor, command-sequence fuzzer, machine-checked
parity TOML and SHA-256 attribution DB. Fixtures are hand-built
real .xlsx files (see ~/workspace/gridcraft-verify/gen_fixtures.py);
a missing fixture is a failure, never a silent skip.
"""

from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.documents.sheets import (
    CellError,
    SheetEditor,
    Workbook,
)
from nomorals.documents.sheets import run_parity as run_gridcraft_parity
from nomorals.documents.sheets import (
    verify_attribution as verify_gridcraft_attribution,
)
from nomorals.documents.sheets.calc import find_cycles, recalc
from nomorals.documents.sheets.formula import (
    evaluate,
    parse_formula,
)
from nomorals.documents.sheets.fuzz import run_fuzzer
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gridcraft"

VALID = ("simple.xlsx", "formulas.xlsx", "cycle.xlsx", "styled.xlsx",
         "values.xlsx", "openpyxl.xlsx")


def _load(name: str) -> Workbook:
    return Workbook.from_bytes((FIXTURES / name).read_bytes())


def _small() -> Workbook:
    """A blank workbook with a 3x3 grid of numbers, for formula tests."""
    wb = Workbook.blank()
    ed = SheetEditor(wb)
    for row, values in enumerate(((10, 2, 3), (20, 5, 6), (30, 7, 9)),
                                 start=1):
        for col, value in enumerate(values, start=1):
            ed.set_value("Sheet1", _ref(row, col), value)
    return wb


def _ref(row: int, col: int) -> str:
    letters = ""
    n = col
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


class FixtureTests(unittest.TestCase):
    def test_all_fixtures_present(self) -> None:
        for name in VALID + ("broken.xlsx", "no-workbook.xlsx",
                             "corrupt-xml.xlsx"):
            self.assertTrue((FIXTURES / name).exists(), name)

    def test_valid_fixtures_are_real_xlsx_packages(self) -> None:
        for name in VALID:
            with zipfile.ZipFile(FIXTURES / name) as zf:
                self.assertIn("xl/workbook.xml", zf.namelist(), name)


class InspectTests(unittest.TestCase):
    def test_simple_workbook(self) -> None:
        wb = _load("simple.xlsx")
        self.assertEqual(wb.sheet_names, ["Data", "Summary"])
        self.assertEqual(wb.core_properties.get("title"),
                         "Simple Fixture")
        data = wb.sheet("Data")
        self.assertEqual(data.value("A1"), "Quarterly Sales")
        self.assertEqual(data.cell("D3").formula, "B3*C3")
        self.assertEqual(data.cell("D3").cached, 18)
        self.assertEqual(data.cell("D6").formula, "SUM(D3:D5)")
        self.assertEqual(data.cell("D6").cached, 51)
        self.assertEqual(data.merged_cells, ["A1:D1"])
        self.assertEqual(wb.defined_names.get("GrandTotal"),
                         "Data!$D$6")
        summary = wb.sheet("Summary")
        self.assertEqual(summary.cell("B1").formula, "Data!D6")
        self.assertEqual(summary.cell("B1").cached, 51)

    def test_formulas_fixture(self) -> None:
        wb = _load("formulas.xlsx")
        calc = wb.sheet("Calc")
        self.assertEqual(calc.cell("A2").formula, "A1*2")
        self.assertEqual(calc.cell("A5").formula,
                         "VLOOKUP(2,Lookup!A2:B4,2,FALSE)")
        self.assertEqual(calc.cell("A5").cached, "Bola")
        self.assertEqual(calc.cell("B5").cached, "Chidi")
        self.assertEqual(calc.cell("A6").cached, 15)

    def test_styled_cells(self) -> None:
        wb = _load("styled.xlsx")
        ws = wb.sheet("Styled")
        header = ws.cell("A1")
        self.assertTrue(header.style.bold)
        self.assertEqual(header.style.font_color, "FFFFFFFF")
        self.assertEqual(header.style.fill, "FF1F4E79")
        self.assertEqual(ws.cell("A2").style.number_format,
                         '"₦"#,##0.00')
        self.assertEqual(ws.cell("A3").style.number_format, "yyyy-mm-dd")

    def test_value_types(self) -> None:
        wb = _load("values.xlsx")
        ws = wb.sheet("Values")
        self.assertEqual(ws.value("A1"), 42)
        self.assertEqual(ws.value("B1"), -3.25)
        self.assertEqual(ws.value("C1"), 1500.0)
        self.assertIs(ws.value("D1"), True)
        self.assertIs(ws.value("D2"), False)
        self.assertEqual(ws.value("A2"), "hello")
        self.assertEqual(ws.value("B2"), "wörld ✓")
        self.assertEqual(ws.value("C2"), "inline text")
        self.assertIsNone(ws.value("B3"))
        self.assertEqual(ws.cell("A3").style.number_format, "yyyy-mm-dd")

    def test_openpyxl_producer(self) -> None:
        wb = _load("openpyxl.xlsx")
        self.assertEqual(wb.sheet_names, ["Produced", "Second"])
        ws = wb.sheet("Produced")
        self.assertEqual(ws.value("A1"), "Produced by openpyxl")
        self.assertTrue(ws.cell("A1").style.bold)
        self.assertEqual(ws.cell("A4").formula, "SUM(A2:A3)")
        self.assertEqual(ws.cell("A4").cached, 15)
        second = wb.sheet("Second")
        self.assertEqual(second.cell("A1").formula, "Produced!A4*10")
        self.assertEqual(second.cell("A1").cached, 150)

    def test_to_dict_round_trip(self) -> None:
        wb = _load("simple.xlsx")
        data = wb.to_dict()
        self.assertEqual(data["sheets"][0]["name"], "Data")
        self.assertEqual(data["defined_names"]["GrandTotal"],
                         "Data!$D$6")


class FormulaTests(unittest.TestCase):
    def test_arithmetic_precedence(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=1+2*3"), 7)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=(1+2)*3"), 9)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=2^3^2"), 512)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=10%"), 0.1)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=-5+2"), -3)

    def test_references_and_ranges(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=A1"), 10)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=$A$1"), 10)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=A1+B1"), 12)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=SUM(A1:A3)"), 60)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=SUM(A1:C1)"), 15)

    def test_math_functions(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=AVERAGE(A1:A3)"),
                         20)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=COUNT(A1:A3)"), 3)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=MIN(A1:C3)"), 2)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=MAX(A1:C3)"), 30)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=ABS(-4)"), 4)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=ROUND(3.567,2)"),
                         3.57)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=MOD(10,3)"), 1)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=SQRT(16)"), 4)
        self.assertEqual(evaluate(wb, "Sheet1", "A1", "=POWER(2,10)"),
                         1024)
        # 10*2 + 20*5 + 30*7 = 330 (the grid's B column is 2/5/7).
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=SUMPRODUCT(A1:A3,B1:B3)"), 330)

    def test_logic_functions(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=IF(A1>15,"big","small")'), "small")
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=AND(A1>5,B1>1)"),
                      True)
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=OR(A1>50,B1>1)"),
                      True)
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=NOT(A1>50)"), True)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=IFERROR(1/0,99)"), 99)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=IFS(A1>50,1,A1>5,2,TRUE,3)"), 2)

    def test_text_functions(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=CONCAT("a","b","c")'), "abc")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '="x"&"y"'), "xy")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=LEFT("hello",2)'), "he")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=RIGHT("hello",3)'), "llo")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=MID("hello",2,3)'), "ell")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=LEN("hello")'), 5)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=UPPER("abc")'), "ABC")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=LOWER("ABC")'), "abc")
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  '=TRIM("  a  b  ")'), "a b")

    def test_lookup_functions(self) -> None:
        wb = _load("formulas.xlsx")
        self.assertEqual(evaluate(wb, "Calc", "A1",
                                  "=VLOOKUP(2,Lookup!A2:B4,2,FALSE)"),
                         "Bola")
        self.assertEqual(evaluate(wb, "Calc", "A1",
                                  "=INDEX(Lookup!B2:B4,3)"), "Chidi")
        self.assertEqual(evaluate(wb, "Calc", "A1",
                                  "=MATCH(3,Lookup!A2:A4,0)"), 3)

    def test_let_and_named_range(self) -> None:
        wb = _small()
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=LET(x,5,x*3)"), 15)
        simple = _load("simple.xlsx")
        self.assertEqual(evaluate(simple, "Summary", "A1",
                                  "=GrandTotal*2"), 102)

    def test_date_functions(self) -> None:
        wb = _small()
        # 46344 is the Excel serial for 2026-11-18, verified against
        # openpyxl's to_excel (45974 is 2025-11-13, not this date).
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=DATE(2026,11,18)"), 46344)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=YEAR(DATE(2026,11,18))"), 2026)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=MONTH(DATE(2026,11,18))"), 11)
        self.assertEqual(evaluate(wb, "Sheet1", "A1",
                                  "=DAY(DATE(2026,11,18))"), 18)

    def test_error_values(self) -> None:
        wb = _small()
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=1/0"),
                      CellError.DIV0)
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=FOO(1)"),
                      CellError.NAME)
        self.assertIs(evaluate(wb, "Sheet1", "A1", '="a"+1'),
                      CellError.VALUE)
        self.assertIs(evaluate(wb, "Sheet1", "A1", "=Missing!A1"),
                      CellError.REF)
        self.assertIs(evaluate(wb, "Sheet1", "A1",
                               "=VLOOKUP(99,A1:A3,1,FALSE)"),
                      CellError.NA)
        self.assertEqual(CellError.DIV0.display, "#DIV/0!")
        self.assertEqual(CellError.CIRC.display, "#CIRC!")

    def test_parse_errors_raise(self) -> None:
        with self.assertRaises(ValueError):
            parse_formula("=1+")
        with self.assertRaises(ValueError):
            parse_formula("=(1+2")
        with self.assertRaises(ValueError):
            parse_formula("1+2")


class CalcTests(unittest.TestCase):
    def test_recalc_chain(self) -> None:
        wb = _load("formulas.xlsx")
        result = recalc(wb)
        self.assertEqual(result.cycles, [])
        calc = wb.sheet("Calc")
        self.assertEqual(calc.cell("A2").cached, 20)
        self.assertEqual(calc.cell("B2").cached, 30)
        self.assertEqual(calc.cell("A3").cached, "big")
        self.assertIs(calc.cell("A4").cached, CellError.DIV0)
        self.assertIs(calc.cell("B4").cached, CellError.REF)
        self.assertIs(calc.cell("C4").cached, CellError.NAME)

    def test_dirty_propagation(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.set_value("Data", "B3", 100)
        recalc(wb)
        self.assertEqual(wb.sheet("Data").cell("D3").cached, 150)
        self.assertEqual(wb.sheet("Data").cell("D6").cached, 183)
        self.assertEqual(wb.sheet("Summary").cell("B1").cached, 183)

    def test_cycle_is_named(self) -> None:
        wb = _load("cycle.xlsx")
        result = recalc(wb)
        self.assertEqual(len(result.cycles), 1)
        cycle = result.cycles[0]
        self.assertIn("Loop!A1", cycle)
        self.assertIn("Loop!B1", cycle)
        self.assertIs(wb.sheet("Loop").cell("A1").cached,
                      CellError.CIRC)
        self.assertIs(wb.sheet("Loop").cell("B1").cached,
                      CellError.CIRC)
        # Downstream of a cycle is an error too, not a stale value.
        self.assertIs(wb.sheet("Loop").cell("A2").cached,
                      CellError.CIRC)

    def test_find_cycles(self) -> None:
        wb = _load("cycle.xlsx")
        cycles = find_cycles(wb)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), {"Loop!A1", "Loop!B1"})
        self.assertEqual(find_cycles(_load("simple.xlsx")), [])


class EditTests(unittest.TestCase):
    def test_set_value_and_formula(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.set_value("Data", "A7", "Total rows")
        ed.set_formula("Data", "B7", "=COUNT(B3:B5)")
        recalc(wb)
        self.assertEqual(wb.sheet("Data").value("A7"), "Total rows")
        self.assertEqual(wb.sheet("Data").cell("B7").cached, 3)

    def test_set_formula_validates(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        with self.assertRaises(ValueError):
            ed.set_formula("Data", "A7", "=1+")

    def test_insert_row_shifts_refs(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.insert_rows("Data", 3, 1)
        data = wb.sheet("Data")
        # The old row-3 item moved to row 4; the SUM follows it.
        self.assertEqual(data.value("A4"), "Apples")
        self.assertEqual(data.cell("D4").formula, "B4*C4")
        self.assertEqual(data.cell("D7").formula, "SUM(D4:D6)")
        recalc(wb)
        self.assertEqual(data.cell("D7").cached, 51)
        # The named range followed the grand total down one row.
        self.assertEqual(wb.defined_names["GrandTotal"], "Data!$D$7")

    def test_insert_column_shifts_refs(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.insert_cols("Data", 2, 1)
        data = wb.sheet("Data")
        self.assertEqual(data.cell("E3").formula, "C3*D3")
        # A column insert moves D6 to E6 (same row); E7 would be a
        # row insert, not a column insert.
        self.assertEqual(data.cell("E6").formula, "SUM(E3:E5)")
        recalc(wb)
        self.assertEqual(data.cell("E6").cached, 51)

    def test_add_and_rename_sheet(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.add_sheet("Notes")
        self.assertEqual(wb.sheet_names, ["Data", "Summary", "Notes"])
        ed.rename_sheet("Notes", "Scratch")
        self.assertEqual(wb.sheet_names, ["Data", "Summary", "Scratch"])
        ed.set_value("Scratch", "A1", "hello")
        self.assertEqual(wb.sheet("Scratch").value("A1"), "hello")
        with self.assertRaises(ValueError):
            ed.add_sheet("Data")

    def test_remove_sheet(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.remove_sheet("Summary")
        self.assertEqual(wb.sheet_names, ["Data"])
        with self.assertRaises(ValueError):
            ed.remove_sheet("Data")

    def test_restyle_cell(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.restyle("Data", "A3", bold=True, color="FFFF0000",
                   number_format="#,##0.00")
        cell = wb.sheet("Data").cell("A3")
        self.assertTrue(cell.style.bold)
        self.assertEqual(cell.style.font_color, "FFFF0000")
        self.assertEqual(cell.style.number_format, "#,##0.00")

    def test_set_metadata(self) -> None:
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.set_metadata(title="Renamed Fixture", author="Devon")
        self.assertEqual(wb.core_properties["title"], "Renamed Fixture")
        self.assertEqual(wb.core_properties["creator"], "Devon")

    def test_create_blank(self) -> None:
        wb = Workbook.blank()
        self.assertEqual(wb.sheet_names, ["Sheet1"])
        ed = SheetEditor(wb)
        ed.set_value("Sheet1", "A1", 1)
        ed.set_formula("Sheet1", "A2", "=A1*2")
        recalc(wb)
        self.assertEqual(wb.sheet("Sheet1").cell("A2").cached, 2)
        out = wb.to_bytes()
        again = Workbook.from_bytes(out)
        self.assertEqual(again.sheet("Sheet1").cell("A2").cached, 2)


class FidelityTests(unittest.TestCase):
    def test_no_edit_round_trip_preserves_every_part(self) -> None:
        for name in VALID:
            wb = _load(name)
            report = wb.fidelity_report(wb.to_bytes())
            self.assertEqual(report["changed_parts"], [], name)
            self.assertEqual(report["identical_parts"],
                             report["total_parts"], name)

    def test_edit_preserves_untouched_parts_and_cells(self) -> None:
        # fidelity_report compares the workbook against REFERENCE
        # bytes — the original file, captured before the edit — not
        # against the edited output (edited-vs-edited is trivially
        # empty, which is what the first version of this test passed).
        original = (FIXTURES / "simple.xlsx").read_bytes()
        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.set_value("Data", "B3", 100)
        ed.to_bytes()
        report = wb.fidelity_report(original)
        self.assertNotIn("xl/styles.xml", report["changed_parts"])
        self.assertNotIn("docProps/core.xml", report["changed_parts"])
        self.assertIn("xl/worksheets/sheet1.xml",
                      report["changed_parts"])
        self.assertEqual(report["untouched_cells_changed"], 0)

    def test_edited_file_opens_in_openpyxl(self) -> None:
        import openpyxl

        wb = _load("simple.xlsx")
        ed = SheetEditor(wb)
        ed.set_value("Data", "B3", 100)
        ed.set_formula("Data", "A7", "=SUM(D3:D5)")
        raw = ed.to_bytes()
        book = openpyxl.load_workbook(io.BytesIO(raw), data_only=False)
        self.assertEqual(book["Data"]["B3"].value, 100)
        self.assertEqual(book["Data"]["A7"].value, "=SUM(D3:D5)")
        self.assertEqual(book["Data"]["D6"].value, "=SUM(D3:D5)")


class ErrorTests(unittest.TestCase):
    def test_broken_zip_refused(self) -> None:
        with self.assertRaises(ValueError):
            _load("broken.xlsx")

    def test_missing_workbook_refused(self) -> None:
        with self.assertRaises(ValueError):
            _load("no-workbook.xlsx")

    def test_corrupt_xml_refused(self) -> None:
        with self.assertRaises(ValueError):
            _load("corrupt-xml.xlsx")

    def test_empty_bytes_refused(self) -> None:
        with self.assertRaises(ValueError):
            Workbook.from_bytes(b"")

    def test_unknown_sheet_refused(self) -> None:
        wb = _load("simple.xlsx")
        with self.assertRaises(ValueError):
            wb.sheet("Nope")
        with self.assertRaises(ValueError):
            SheetEditor(wb).set_value("Nope", "A1", 1)

    def test_bad_cell_ref_refused(self) -> None:
        wb = _load("simple.xlsx")
        with self.assertRaises(ValueError):
            SheetEditor(wb).set_value("Data", "not-a-ref", 1)


class ParityTests(unittest.TestCase):
    def test_parity_toml_loads(self) -> None:
        from nomorals.documents.sheets.parity import load_parity

        data = load_parity()
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(len(data["cases"]), 10)

    def test_parity_runs_green(self) -> None:
        result = run_gridcraft_parity()
        failed = [c["id"] for c in result["cases"]
                  if c["status"] != "pass"]
        self.assertEqual(failed, [])
        self.assertEqual(len(result["cases"]), 10)

    def test_parity_missing_fixture_is_a_failure(self) -> None:
        from nomorals.documents.sheets.parity import run_parity

        result = run_parity(fixtures_dir=FIXTURES / "does-not-exist")
        self.assertTrue(any(c["status"] == "fail"
                            for c in result["cases"]))


class AttributionTests(unittest.TestCase):
    def test_attribution_toml_loads(self) -> None:
        from nomorals.documents.sheets.attribution import load_attribution

        data = load_attribution()
        self.assertTrue(len(data["assets"]) >= 12)

    def test_attribution_verifies_against_clone(self) -> None:
        result = verify_gridcraft_attribution()
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["skipped"], [])


class FuzzTests(unittest.TestCase):
    def test_fuzzer_deterministic(self) -> None:
        first = run_fuzzer(seed=42, commands=200)
        second = run_fuzzer(seed=42, commands=200)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.failures, [])

    def test_fuzzer_invariants_hold(self) -> None:
        for seed in (1, 7, 99):
            result = run_fuzzer(seed=seed, commands=200)
            self.assertEqual(result.failures, [], f"seed {seed}")
            self.assertEqual(result.commands_run, 200)

    def test_fuzzer_names_injected_cycle(self) -> None:
        result = run_fuzzer(seed=5, commands=50, inject_cycle=True)
        self.assertTrue(result.cycles_found)
        self.assertEqual(result.failures, [])


class ToolTests(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(context=SimpleNamespace(
            home=Path(__file__).resolve().parent.parent))
        registry.register_builtins()
        return registry

    def _call(self, registry: ToolRegistry, **kwargs) -> dict:
        outcome = registry.call("gridcraft", **kwargs)
        if hasattr(outcome, "value"):
            return outcome.value
        return outcome

    def test_tool_is_registered(self) -> None:
        registry = self._registry()
        spec = registry.get("gridcraft")
        self.assertIsNotNone(spec)

    def test_tool_inspect(self) -> None:
        registry = self._registry()
        result = self._call(registry, action="inspect",
                            sources=[str(FIXTURES / "simple.xlsx")])
        self.assertIn("Data", result["reply"])
        self.assertEqual(result["sheets"][0]["name"], "Data")

    def test_tool_set_and_recalc_round_trip(self) -> None:
        import tempfile

        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "edited.xlsx")
            result = self._call(
                registry, action="set",
                sources=[str(FIXTURES / "simple.xlsx")],
                output=out, sheet="Data", ref="B3", text="100")
            self.assertEqual(result["value"], 100)
            again = self._call(registry, action="recalc", sources=[out])
            self.assertEqual(again["cells"]["Data!D6"], 183)

    def test_tool_parity_and_attribution(self) -> None:
        registry = self._registry()
        parity = self._call(registry, action="parity")
        self.assertEqual(parity["failed"], [])
        attribution = self._call(registry, action="attribution")
        self.assertEqual(attribution["failed"], [])

    def test_tool_unknown_action(self) -> None:
        # The registry never raises for a bad action — it returns an
        # error outcome, the same contract wordcraft's test pins.
        registry = self._registry()
        outcome = registry.call("gridcraft", action="explode")
        self.assertFalse(outcome.ok)
        self.assertIn("unknown", str(outcome.error).lower())


if __name__ == "__main__":
    unittest.main()
