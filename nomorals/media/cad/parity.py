"""Machine-checked parity for the CAD organ.

A parity TOML lists real corpus drawings + operations + expected
outcomes; :func:`run_parity` runs every case and fails on any
mismatch. A missing fixture FAILS its case — never a silent skip
(the same habit as the pdfcraft/wordcraft/vectorcraft/layout organs).
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

from .codec import export_dxf, import_drawing, import_dxf, parse_document
from .edit import CadEditor
from .errors import CadError
from .model import CadDocument
from .render import render_document
from .solver import solve

__all__ = ["PARITY_TOML", "run_parity"]

PARITY_TOML = Path(__file__).resolve().parent.parent / "cad-parity.toml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_fixture(name: str, toml_path: Path) -> Path | None:
    path = Path(name)
    candidates = [path] if path.is_absolute() else [
        toml_path.parent / name,
        _repo_root() / "tests" / "fixtures" / "cadcraft" / name,
        _repo_root() / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _facts_for(fixture: Path, ops: list[str]) -> dict:
    """Run the ops on a fresh load of the fixture and measure the
    facts they produce."""
    raw = fixture.read_bytes()
    facts: dict = {}
    for op in ops:
        if op == "inspect":
            doc = _load(fixture, raw)
            facts.update({"entities": doc.entity_count(),
                          "constraints": doc.constraint_count(),
                          "layers": len(doc.layers)})
        elif op == "roundtrip":
            doc = parse_document(raw)
            from .codec import to_document_bytes
            facts["byte_identical"] = to_document_bytes(doc) == raw
        elif op == "render":
            doc = _load(fixture, raw)
            png = render_document(doc, width=400, height=300)
            facts["png_ok"] = png[:8] == b"\x89PNG\r\n\x1a\n" \
                and len(png) > 0
        elif op == "import":
            doc = import_drawing(fixture.name, raw)
            facts["entities"] = doc.entity_count()
        elif op == "dxf_roundtrip":
            doc = import_dxf(raw)
            again = import_dxf(export_dxf(doc))
            facts["entities"] = again.entity_count()
            facts["identical"] = [
                str(e) for e in again.entities.values()] == [
                str(e) for e in doc.entities.values()]
        elif op == "solve":
            doc = _load(fixture, raw)
            report = solve(doc)
            facts.update({"converged": report["converged"],
                          "status": report["status"],
                          "dof": report["dof"]})
        elif op == "tangent_solve":
            doc = _load(fixture, raw)
            report = solve(doc)
            facts["converged"] = report["converged"]
            line = next(e for e in doc.entities.values()
                        if type(e).__name__ == "Line")
            circle = next(e for e in doc.entities.values()
                          if type(e).__name__ == "Circle")
            direction = (line.b[0] - line.a[0], line.b[1] - line.a[1])
            length = math.hypot(*direction)
            direction = (direction[0] / length, direction[1] / length)
            gap = abs(direction[0] * (circle.center[1] - line.a[1])
                      - direction[1] * (circle.center[0] - line.a[0]))
            # The tangent invariant is gap == radius, not gap == the
            # starting radius: the solver shares the move across all
            # free parameters including the radius, so measure the
            # gap error against the solved radius.
            facts["gap_error"] = gap - circle.radius
        elif op == "trim":
            doc = _load(fixture, raw)
            editor = CadEditor(doc)
            first = next(iter(doc.entities))
            facts["trimmed"] = editor.trim(first, (8.0, 0.0))
            kept = doc.entities[first]
            facts["kept_end_x"] = kept.b[0]
        elif op == "fillet":
            doc = _load(fixture, raw)
            editor = CadEditor(doc)
            ids = list(doc.entities)
            arc_id = editor.fillet(ids[0], ids[1], 2.0,
                                   (9.0, 0.0), (10.0, 1.0))
            facts["filleted"] = arc_id is not None
            if arc_id is not None:
                arc = doc.entities[arc_id]
                facts["arc_radius"] = arc.radius
                facts["arc_center_x"] = arc.center[0]
                facts["arc_center_y"] = arc.center[1]
        else:
            raise CadError(f"unknown parity op {op!r}",
                           kind="invalid_argument")
    return facts


def _load(fixture: Path, raw: bytes) -> CadDocument:
    if fixture.suffix.lower() == ".dxf":
        return import_dxf(raw)
    return parse_document(raw)


def _matches(measured, expected) -> bool:
    if isinstance(expected, float) or isinstance(measured, float):
        try:
            return abs(float(measured) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    return measured == expected


def run_parity(path: str | Path | None = None) -> dict:
    """Run every parity case in the TOML. Returns
    ``{"passed": n, "failed": [ids], "cases": [...]}`` — a missing
    fixture fails its case."""
    toml_path = Path(path) if path is not None else PARITY_TOML
    try:
        with open(toml_path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise CadError(f"cannot read parity TOML {toml_path}: {exc}",
                       kind="io") from exc
    cases = data.get("cases", [])
    results: list[dict] = []
    failed: list[str] = []
    for case in cases:
        case_id = str(case.get("id", "?"))
        fixture_name = str(case.get("fixture", ""))
        ops = list(case.get("ops", []))
        expect = dict(case.get("expect", {}))
        fixture = _resolve_fixture(fixture_name, toml_path)
        if fixture is None:
            results.append({"id": case_id, "status": "fail",
                            "detail": f"missing fixture {fixture_name}"})
            failed.append(case_id)
            continue
        try:
            facts = _facts_for(fixture, ops)
        except Exception as exc:
            results.append({"id": case_id, "status": "fail",
                            "detail": f"{exc.__class__.__name__}: {exc}"})
            failed.append(case_id)
            continue
        mismatches = {key: (facts.get(key), value)
                      for key, value in expect.items()
                      if not _matches(facts.get(key), value)}
        if mismatches:
            results.append({"id": case_id, "status": "fail",
                            "detail": f"mismatches: {mismatches}"})
            failed.append(case_id)
        else:
            results.append({"id": case_id, "status": "pass",
                            "detail": f"facts: {facts}"})
    return {"passed": len(results) - len(failed), "failed": failed,
            "cases": results}
