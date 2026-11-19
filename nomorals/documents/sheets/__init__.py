"""GridCraft port — the spreadsheet model, calc engine, editor,
fuzzer, parity runner, and attribution DB for Devon's documents
organ.

Spec-decomposed like the upstream crates (``crates/model`` +
``crates/xlsx`` → :mod:`model` + :mod:`ooxml`, ``crates/formula`` +
``crates/functions`` → :mod:`formula`, ``crates/calc`` → :mod:`calc`,
``crates/engine`` command sequences → :mod:`fuzz`, ``parity/*.toml``
→ :mod:`parity`, ``ATTRIBUTION.toml`` → :mod:`attribution`); dual
MIT/Apache-2.0 upstream, attribution in
``nomorals/documents/sheets-attribution.toml``.

Entry points: :class:`Workbook` (open + inventory),
:class:`SheetEditor` (fidelity-preserving edits), :func:`recalc`
and :func:`find_cycles` (the calc engine), :func:`run_fuzzer`
(the deterministic command-sequence fuzzer), :func:`run_parity`
(machine-checked parity TOML), :func:`verify_attribution` (SHA-256
provenance check). OOXML primitives (:class:`XlsxPackage`,
:func:`parse_xml_bounded`, :func:`serialize_xml`) are exported for
callers that work at the package level.
"""

from __future__ import annotations

from .attribution import (
    DEFAULT_ATTRIBUTION_PATH,
    DEFAULT_CLONE_ROOT,
    load_attribution,
    verify_attribution,
)
from .calc import RecalcResult, find_cycles, recalc, recalc_changed
from .edit import SheetEditor
from .formula import (
    evaluate,
    parse_formula,
)
from .fuzz import FuzzResult, run_fuzzer
from .model import (
    Cell,
    CellError,
    CellStyle,
    Workbook,
    Worksheet,
)
from .ooxml import (
    XlsxPackage,
    parse_xml_bounded,
    read_package,
    serialize_xml,
    write_package,
)
from .parity import DEFAULT_PARITY_PATH, load_parity, run_parity

__all__ = [
    "DEFAULT_ATTRIBUTION_PATH",
    "DEFAULT_CLONE_ROOT",
    "DEFAULT_PARITY_PATH",
    "Cell",
    "CellError",
    "CellStyle",
    "FuzzResult",
    "RecalcResult",
    "SheetEditor",
    "Workbook",
    "Worksheet",
    "XlsxPackage",
    "evaluate",
    "find_cycles",
    "load_attribution",
    "load_parity",
    "parse_formula",
    "parse_xml_bounded",
    "read_package",
    "recalc",
    "recalc_changed",
    "run_fuzzer",
    "run_parity",
    "serialize_xml",
    "verify_attribution",
    "write_package",
]
