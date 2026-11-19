"""Deterministic command-sequence fuzzer for the GridCraft port.

Port of the *design* of the census line ("command-sequence fuzzer")
and upstream ``crates/engine`` (the command catalogue in
``cmd.rs``): a seeded ``random.Random`` drives random edit/command
sequences against the real engine and asserts the invariants after
every checkpoint — no crash, the recalculation converges, the
dependency graph stays acyclic except for cycles the run itself
names, and the workbook still round-trips through ``to_bytes``.

Deterministic: the same seed always drives the same commands and
the same digest (no wall clock, no RAND()/NOW() in generated
formulas, sorted fingerprints). A failure is a recorded invariant
violation, never a raised exception — ``run_fuzzer`` always
returns a result.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from ..errors import DocumentError
from .calc import find_cycles, recalc
from .edit import SheetEditor
from .model import CellError, Workbook, a1_ref

__all__ = ["FuzzResult", "run_fuzzer"]

_MAX_ROW = 8
_MAX_COL = 6  # A..F

_FORMULA_TEMPLATES = (
    "={r1}+{r2}",
    "={r1}*{r2}",
    "=SUM({rng})",
    "=AVERAGE({rng})",
    "=IF({r1}>5,1,0)",
    "=IF({r1}>5,\"big\",\"small\")",
    "={r1}&{r2}",
    "=MAX({rng})",
    "=MIN({rng})",
    "=COUNT({rng})",
    "=ROUND({r1}/3,2)",
    "={r1}-{r2}",
)

_VALUE_POOL = (0, 1, 2, 3, 5, 7, 10, 12.5, -4, "alpha", "beta", True,
               False)

_NUMBER_FORMATS = ("General", "#,##0.00", "0%", "yyyy-mm-dd")


@dataclass
class FuzzResult:
    """The outcome of one fuzz run."""

    seed: int
    commands_run: int
    cycles_found: list[list[str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "commands_run": self.commands_run,
            "cycles_found": [list(c) for c in self.cycles_found],
            "failures": list(self.failures),
            "digest": self.digest,
        }


def _fingerprint(wb: Workbook) -> str:
    """A sorted, deterministic snapshot of the workbook's cells."""
    parts: list[str] = []
    for ws in wb.sheets:
        for (row, col) in sorted(ws.cells):
            cell = ws.cells[(row, col)]
            cached = cell.cached
            if isinstance(cached, CellError):
                cached = cached.display
            parts.append(f"{ws.name}!{a1_ref(row, col)}="
                         f"{cell.value!r}|{cell.formula!r}|{cached!r}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _random_ref(rng: random.Random) -> str:
    return a1_ref(rng.randint(1, _MAX_ROW), rng.randint(1, _MAX_COL))


def _random_formula(rng: random.Random, target: str) -> str:
    template = rng.choice(_FORMULA_TEMPLATES)
    ref1, ref2 = _random_ref(rng), _random_ref(rng)
    rng_ref = (f"{_random_ref(rng)}:{_random_ref(rng)}")
    text = template.format(r1=ref1, r2=ref2, rng=rng_ref)
    # A formula must not reference its own cell — that would be a
    # deliberate cycle, not a random command.
    if target in text:
        return "=1+2"
    return text


def _snapshot(wb: Workbook) -> dict[str, Any]:
    """Cached values per formula cell, in display form."""
    out: dict[str, Any] = {}
    for ws in wb.sheets:
        for cell in ws.cells.values():
            if cell.formula is None:
                continue
            cached = cell.cached
            if isinstance(cached, CellError):
                cached = cached.display
            out[f"{ws.name}!{cell.ref}"] = cached
    return out


def _check_invariants(wb: Workbook, step: int,
                      failures: list[str]) -> None:
    """The invariants, checked at every checkpoint."""
    try:
        first = recalc(wb)
        snap_first = _snapshot(wb)
        second = recalc(wb)
        snap_second = _snapshot(wb)
    except Exception as exc:  # noqa: BLE001 - the failure is the report
        failures.append(f"step {step}: recalc raised "
                        f"{type(exc).__name__}: {exc}")
        return
    if first.cycles != second.cycles:
        failures.append(f"step {step}: cycle list changed between "
                        f"recalcs: {first.cycles} vs {second.cycles}")
    if snap_first != snap_second:
        diffs = [key for key in snap_first
                 if snap_first.get(key) != snap_second.get(key)]
        failures.append(f"step {step}: recalculation did not "
                        f"converge: {diffs[:5]}")
    cycles = find_cycles(wb)
    if sorted(map(sorted, cycles)) != sorted(
            map(sorted, first.cycles)):
        failures.append(f"step {step}: find_cycles disagrees with "
                        f"recalc cycles: {cycles} vs {first.cycles}")
    try:
        raw = wb.to_bytes()
        again_wb = Workbook.from_bytes(raw)
    except Exception as exc:  # noqa: BLE001 - the failure is the report
        failures.append(f"step {step}: round-trip raised "
                        f"{type(exc).__name__}: {exc}")
        return
    if again_wb.sheet_names != wb.sheet_names:
        failures.append(f"step {step}: sheet names changed across "
                        f"round-trip: {again_wb.sheet_names} vs "
                        f"{wb.sheet_names}")
    for ws in wb.sheets:
        other = again_wb.sheets_by_name.get(ws.name)
        if other is None:
            failures.append(f"step {step}: sheet {ws.name!r} lost "
                            f"across round-trip")
            continue
        for key, cell in ws.cells.items():
            mine = (cell.value, cell.formula)
            theirs_cell = other.cells.get(key)
            theirs = (theirs_cell.value, theirs_cell.formula) \
                if theirs_cell is not None else None
            if mine != theirs:
                failures.append(
                    f"step {step}: {ws.name}!{cell.ref} changed "
                    f"across round-trip: {mine!r} vs {theirs!r}")
                break


def run_fuzzer(seed: int = 0, commands: int = 100,
               inject_cycle: bool = False) -> FuzzResult:
    """Drive ``commands`` random edit commands against a fresh
    workbook, checking the invariants at every checkpoint. With
    ``inject_cycle``, two cells are wired into a cycle at the end
    and the run must name it."""
    rng = random.Random(seed)
    wb = Workbook.blank()
    ed = SheetEditor(wb)
    sheets = ["Sheet1"]
    log: list[str] = []
    failures: list[str] = []
    sheet_counter = 1

    def pick_sheet() -> str:
        return rng.choice(sheets)

    for step in range(1, commands + 1):
        op = rng.choice((
            "set-value", "set-value", "set-formula", "set-formula",
            "insert-row", "insert-col", "restyle", "add-sheet",
            "remove-sheet", "rename-sheet", "set-metadata",
        ))
        detail = op
        try:
            if op == "set-value":
                sheet = pick_sheet()
                ref = _random_ref(rng)
                value = rng.choice(_VALUE_POOL)
                ed.set_value(sheet, ref, value)
                detail = f"{op} {sheet}!{ref}={value!r}"
            elif op == "set-formula":
                sheet = pick_sheet()
                ref = _random_ref(rng)
                formula = _random_formula(rng, ref)
                ed.set_formula(sheet, ref, formula)
                detail = f"{op} {sheet}!{ref}={formula}"
            elif op == "insert-row":
                sheet = pick_sheet()
                at = rng.randint(1, _MAX_ROW)
                ed.insert_rows(sheet, at, 1)
                detail = f"{op} {sheet}@{at}"
            elif op == "insert-col":
                sheet = pick_sheet()
                at = rng.randint(1, _MAX_COL)
                ed.insert_cols(sheet, at, 1)
                detail = f"{op} {sheet}@{at}"
            elif op == "restyle":
                sheet = pick_sheet()
                ref = _random_ref(rng)
                ed.restyle(sheet, ref, bold=rng.choice((True, False)),
                           number_format=rng.choice(_NUMBER_FORMATS))
                detail = f"{op} {sheet}!{ref}"
            elif op == "add-sheet":
                sheet_counter += 1
                name = f"Sheet{sheet_counter}"
                ed.add_sheet(name)
                sheets.append(name)
                detail = f"{op} {name}"
            elif op == "remove-sheet":
                if len(sheets) <= 1:
                    # One sheet left: the engine would refuse, so the
                    # command is a deliberate no-op, still counted.
                    detail = f"{op} SKIPPED (one sheet left)"
                else:
                    name = rng.choice(sheets)
                    ed.remove_sheet(name)
                    sheets.remove(name)
                    detail = f"{op} {name}"
            elif op == "rename-sheet":
                old = rng.choice(sheets)
                sheet_counter += 1
                new = f"Renamed{sheet_counter}"
                ed.rename_sheet(old, new)
                sheets[sheets.index(old)] = new
                detail = f"{op} {old}->{new}"
            elif op == "set-metadata":
                ed.set_metadata(title=f"fuzz-{seed}-{step}")
                detail = f"{op} title=fuzz-{seed}-{step}"
        except DocumentError as exc:
            # A random command may be refused (a bad ref, a missing
            # sheet) — that is the engine refusing honestly, not a
            # fuzzer failure. It still counts as a run command.
            detail = f"{op} REFUSED {exc.kind}"
        log.append(f"{step}:{detail}:{_fingerprint(wb)}")
        if step % 25 == 0 or step == commands:
            _check_invariants(wb, step, failures)

    cycles_found: list[list[str]] = []
    if inject_cycle:
        sheet = sheets[0]
        ed.set_formula(sheet, "A1", "=B1")
        ed.set_formula(sheet, "B1", "=A1")
        result = recalc(wb)
        cycles_found = result.cycles
        if not cycles_found:
            failures.append("injected cycle was not found by recalc")
        _check_invariants(wb, commands + 1, failures)
        log.append(f"inject:{sheet}!A1<->B1:{_fingerprint(wb)}")

    digest = hashlib.sha256("\n".join(log).encode()).hexdigest()
    return FuzzResult(seed=seed, commands_run=commands,
                      cycles_found=cycles_found, failures=failures,
                      digest=digest)
