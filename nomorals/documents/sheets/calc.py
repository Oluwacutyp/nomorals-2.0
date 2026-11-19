"""Calc dependency engine for the GridCraft port — stdlib only.

Port of the *design* of upstream ``crates/calc`` (``recalc.rs``:
``rebuild`` / ``recalc_all`` / ``cells_changed``): a dependency
graph over formula cells, topological recalculation order, and
cycle detection that names the cycle. A cycle's cells evaluate to
the upstream's explicit ``#CIRC!`` (Excel shows 0) and everything
downstream of a cycle is an error too, never a stale value.

``recalc`` recomputes every formula cell in dependency order.
``recalc_changed`` recomputes only the dirty cells and their
dependents — the incremental path, same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import DocumentError
from .formula import dependencies, evaluate, parse_formula
from .model import CellError, a1_ref, parse_a1

__all__ = ["RecalcResult", "find_cycles", "recalc", "recalc_changed"]

_MAX_CELLS = 1_000_000


@dataclass
class RecalcResult:
    """The outcome of one recalculation pass."""

    changed: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": list(self.changed),
            "cycles": [list(c) for c in self.cycles],
            "errors": dict(self.errors),
        }


def _formula_cells(workbook: Any) -> dict[tuple[str, int, int], Any]:
    """Every formula cell, keyed ``(sheet, row, col)``."""
    cells: dict[tuple[str, int, int], Any] = {}
    for ws in workbook.sheets:
        for key, cell in ws.cells.items():
            if cell.formula is not None:
                cells[(ws.name, key[0], key[1])] = cell
    if len(cells) > _MAX_CELLS:
        raise DocumentError(
            f"workbook has {len(cells)} formula cells, over the "
            f"{_MAX_CELLS} cap", kind="invalid_argument")
    return cells


def _edges(workbook: Any) -> dict[tuple[str, int, int],
                                  set[tuple[str, int, int]]]:
    """Formula-cell → formula-cells it reads (ranges expanded).

    Ranges are clamped to the sheet's used bounds, so a whole-column
    ref (``A:A``) expands to the rows that actually exist instead of
    a million rows.
    """
    formula_cells = _formula_cells(workbook)
    bounds: dict[str, tuple[int, int]] = {}
    for ws in workbook.sheets:
        bounds[ws.name] = (ws.max_row, ws.max_column)
    edges: dict[tuple[str, int, int], set[tuple[str, int, int]]] = {}
    for key, cell in formula_cells.items():
        sheet = key[0]
        if cell.ast is None:
            try:
                # Stored formulas carry no '='; the parser requires it.
                cell.ast = parse_formula("=" + (cell.formula or ""))
            except Exception:
                cell.ast = None
        deps: set[tuple[str, int, int]] = set()
        if cell.ast is not None:
            for dep_sheet, r1, c1, r2, c2 in dependencies(cell.ast):
                name = dep_sheet or sheet
                max_row, max_col = bounds.get(name, (0, 0))
                lo_r, hi_r = max(1, r1), min(r2, max_row)
                lo_c, hi_c = max(1, c1), min(c2, max_col)
                for row in range(lo_r, hi_r + 1):
                    for col in range(lo_c, hi_c + 1):
                        if (name, row, col) in formula_cells:
                            deps.add((name, row, col))
        edges[key] = deps
    return edges


def find_cycles(workbook: Any) -> list[list[str]]:
    """Cycles in the formula graph, each as ``Sheet!Ref`` strings.

    Iterative DFS with an explicit stack (a deep chain must not
    blow the interpreter's recursion limit) — each back edge closes
    one cycle, reported in the order the cycle runs.
    """
    edges = _edges(workbook)
    state: dict[tuple[str, int, int], int] = {}
    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for start in sorted(edges):
        if state.get(start, 0) != 0:
            continue
        # Stack frames: (node, iterator over its deps).
        stack: list[tuple[tuple[str, int, int],
                          Any]] = [(start, iter(sorted(edges[start])))]
        path: list[tuple[str, int, int]] = [start]
        state[start] = 1
        while stack:
            node, it = stack[-1]
            advanced = False
            for dep in it:
                if state.get(dep, 0) == 0:
                    state[dep] = 1
                    path.append(dep)
                    stack.append((dep, iter(sorted(edges[dep]))))
                    advanced = True
                    break
                if state.get(dep) == 1:
                    at = path.index(dep)
                    cycle = [f"{s}!{a1_ref(r, c)}"
                             for s, r, c in path[at:]]
                    frozen = tuple(sorted(cycle))
                    if frozen not in seen:
                        seen.add(frozen)
                        cycles.append(cycle)
            if not advanced:
                stack.pop()
                state[node] = 2
                if path and path[-1] == node:
                    path.pop()
    return cycles


def _topological_order(
        edges: dict[tuple[str, int, int], set[tuple[str, int, int]]],
        skip: set[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Kahn order over the graph minus the skipped (cyclic) nodes."""
    nodes = [key for key in edges if key not in skip]
    in_degree: dict[tuple[str, int, int], int] = {key: 0 for key in nodes}
    dependents: dict[tuple[str, int, int],
                     set[tuple[str, int, int]]] = {
                         key: set() for key in nodes}
    for key in nodes:
        for dep in edges[key]:
            if dep in in_degree:
                in_degree[key] += 1
                dependents[dep].add(key)
    ready = sorted(key for key in nodes if in_degree[key] == 0)
    order: list[tuple[str, int, int]] = []
    while ready:
        key = ready.pop(0)
        order.append(key)
        for nxt in sorted(dependents[key]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    return order


def recalc(workbook: Any) -> RecalcResult:
    """Recompute every formula cell in dependency order.

    Cycle cells get ``#CIRC!``; cells downstream of a cycle get
    ``#CIRC!`` too, by error propagation — never a stale value.
    """
    formula_cells = _formula_cells(workbook)
    edges = _edges(workbook)
    cycles = find_cycles(workbook)
    cyclic: set[tuple[str, int, int]] = set()
    for cycle in cycles:
        for ref in cycle:
            sheet, _, cell_ref = ref.rpartition("!")
            row, col = parse_a1(cell_ref)
            cyclic.add((sheet, row, col))
    old_cached = {key: cell.cached
                  for key, cell in formula_cells.items()}
    for key in cyclic:
        formula_cells[key].cached = CellError.CIRC
    # Clear every other formula cell's stale cache first: in
    # topological order each evaluation then only reads fresh values.
    for key, cell in formula_cells.items():
        if key not in cyclic:
            cell.cached = None
    order = _topological_order(edges, cyclic)
    # Any node the topological sort could not place is part of a
    # cycle the DFS missed (defensive: can only happen on a graph
    # the cycle finder disagrees with) — mark it too.
    placed = set(order)
    for key in formula_cells:
        if key not in placed and key not in cyclic:
            formula_cells[key].cached = CellError.CIRC
            cyclic.add(key)
    changed: list[str] = []
    errors: dict[str, str] = {}
    for key in order:
        sheet, row, col = key
        cell = formula_cells[key]
        cell.cached = None  # force a fresh evaluation
        value = evaluate(workbook, sheet, a1_ref(row, col),
                         cell.formula or "")
        cell.cached = value
        label = f"{sheet}!{a1_ref(row, col)}"
        changed.append(label)
        if isinstance(value, CellError):
            errors[label] = value.display
    for key in cyclic:
        sheet, row, col = key
        label = f"{sheet}!{a1_ref(row, col)}"
        changed.append(label)
        errors[label] = CellError.CIRC.display
    _mark_changed_sheets(workbook, formula_cells, old_cached)
    return RecalcResult(changed=changed, cycles=cycles, errors=errors)


def _mark_changed_sheets(workbook: Any,
                         formula_cells: dict[tuple[str, int, int], Any],
                         old_cached: dict[tuple[str, int, int], Any]
                         ) -> None:
    """Mark every sheet dirty whose cached values moved.

    A recalculation that changes a cached value changes the file —
    without this, ``to_bytes`` would write back the stale cached
    values (the parity runner caught it: cross-sheet dependents
    re-opened with their old numbers).
    """
    for key, cell in formula_cells.items():
        if key not in old_cached:
            continue
        if cell.cached != old_cached[key]:
            sheet = key[0]
            ws = workbook.sheets_by_name.get(sheet)
            if ws is not None:
                workbook.mark_sheet_dirty(ws.part)


def recalc_changed(workbook: Any,
                   changed: list[tuple[str, int, int]]) -> RecalcResult:
    """Recompute only the given cells and their dependents.

    ``changed`` names cells whose values moved (usually after an
    edit); the reverse graph walks out to every formula that reads
    them, directly or transitively.
    """
    formula_cells = _formula_cells(workbook)
    edges = _edges(workbook)
    dependents: dict[tuple[str, int, int],
                     set[tuple[str, int, int]]] = {}
    for key, deps in edges.items():
        for dep in deps:
            dependents.setdefault(dep, set()).add(key)
    dirty: set[tuple[str, int, int]] = set()
    queue = list(changed)
    while queue:
        key = queue.pop(0)
        for nxt in dependents.get(key, ()):
            if nxt not in dirty:
                dirty.add(nxt)
                queue.append(nxt)
    for key in changed:
        if key in formula_cells:
            dirty.add(key)
    cycles = find_cycles(workbook)
    cyclic: set[tuple[str, int, int]] = set()
    for cycle in cycles:
        for ref in cycle:
            sheet, _, cell_ref = ref.rpartition("!")
            row, col = parse_a1(cell_ref)
            cyclic.add((sheet, row, col))
    old_cached = {key: formula_cells[key].cached for key in dirty
                  if key in formula_cells}
    order = _topological_order(
        {k: v for k, v in edges.items() if k in dirty}, cyclic)
    changed_out: list[str] = []
    errors: dict[str, str] = {}
    for key in order:
        sheet, row, col = key
        cell = formula_cells[key]
        cell.cached = None
        value = evaluate(workbook, sheet, a1_ref(row, col),
                         cell.formula or "")
        cell.cached = value
        label = f"{sheet}!{a1_ref(row, col)}"
        changed_out.append(label)
        if isinstance(value, CellError):
            errors[label] = value.display
    for key in dirty & cyclic:
        sheet, row, col = key
        formula_cells[key].cached = CellError.CIRC
        label = f"{sheet}!{a1_ref(row, col)}"
        changed_out.append(label)
        errors[label] = CellError.CIRC.display
    _mark_changed_sheets(workbook, formula_cells, old_cached)
    return RecalcResult(changed=changed_out, cycles=cycles,
                        errors=errors)
