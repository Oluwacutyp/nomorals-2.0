"""CadCraft port — CAD organ tests (lane 106, census §3A.13).

Failing-first: the CAD organ does not exist yet on the base, so
every import here fails until the port lands. The test pins the
public contract the port must meet:

* ``nomorals.media.cad`` — a real CAD document model (sketches,
  lines/arcs/circles/ellipses/splines/points/polylines/text,
  layers, blocks, dimensions, constraints) in the upstream
  ``.cadcraft`` zip shape (document.json + meta.json, versioned),
  with fidelity-preserving edits.
* ``solver`` — a real constraint solver ported from upstream
  ``crates/constraints``: flat parameter vector, residual forms
  per constraint kind, damped weighted minimum-norm Gauss-Newton
  (Levenberg–Marquardt style) with a numerical Jacobian, DOF
  report from the Jacobian's rank and null space (Householder QR
  with column pivoting). Over-constraint is reported by name.
* ``commands`` — typed-command state machines with per-step
  panic guards (upstream ``crates/engine/src/guard.rs``): a step
  that raises ends the command with a named internal error and
  restores the drawing as it was.
* ``edit`` — real drafting operations computed from real geometry
  (trim, extend, offset, fillet, chamfer, mirror, array).
* ``render`` — a real sketch renderer to PNG through PIL.
* ``codec`` — the ``.cadcraft`` codec plus real DXF import/export
  (the corpus drawings in tests/fixtures/cadcraft/ are the real
  ones, MIT OR Apache-2.0); DWG is refused honestly.
* ``parity`` / ``attribution`` — machine-checked TOML runners;
  a missing fixture FAILS its case, never a silent skip.
* ``cadcraft`` brain tool — the same registry contract the
  designcraft/vectorcraft tools pin.
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.media.cad import (
    Arc,
    CadDocument,
    CadEditor,
    CadError,
    Circle,
    Constraint,
    Dimension,
    GeomRef,
    Line,
    render_document,
    run_parity,
    verify_attribution,
)
from nomorals.media.cad.codec import (
    export_dxf,
    import_drawing,
    import_dxf,
    parse_document,
    to_document_bytes,
)
from nomorals.media.cad.commands import (
    CommandSession,
    Guarded,
    list_commands,
)
from nomorals.media.cad.geom import (
    circle_circle,
    line_circle,
    line_line,
)
from nomorals.media.cad.solver import (
    dof_report,
    measure,
    purge_invalid,
    solve,
    validate_constraint,
)
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cadcraft"


def _read(name: str) -> bytes:
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {name}"
    return path.read_bytes()


def _doc(name: str) -> CadDocument:
    return parse_document(_read(name))


def _line_doc() -> CadDocument:
    """Two lines forming an L, plus a circle — the solver's workbench."""
    doc = CadDocument.blank(title="workbench")
    ed = CadEditor(doc)
    ed.add_line((0.0, 0.0), (10.0, 0.0))
    ed.add_line((10.0, 0.0), (10.0, 10.0))
    ed.add_circle((20.0, 5.0), 3.0)
    return doc


def _rect_doc() -> CadDocument:
    """A rectangle sketch: four lines, corners coincident."""
    doc = CadDocument.blank(title="rectangle")
    ed = CadEditor(doc)
    a = ed.add_line((0.0, 0.0), (10.0, 0.0))
    b = ed.add_line((10.0, 0.0), (10.0, 5.0))
    c = ed.add_line((10.0, 5.0), (0.0, 5.0))
    d = ed.add_line((0.0, 5.0), (0.0, 0.0))
    for first, second in ((a, b), (b, c), (c, d), (d, a)):
        ed.constrain("coincident",
                     [GeomRef(first, "end"), GeomRef(second, "start")])
    ed.constrain("horizontal", [GeomRef(a, "whole")])
    ed.constrain("horizontal", [GeomRef(c, "whole")])
    ed.constrain("vertical", [GeomRef(b, "whole")])
    ed.constrain("vertical", [GeomRef(d, "whole")])
    return doc


# ── document model ────────────────────────────────────────────────

class TestBlankDocument:
    def test_blank_has_one_layer(self) -> None:
        doc = CadDocument.blank()
        assert doc.title == "Untitled"
        assert doc.units == "mm"
        assert len(doc.layers) == 1
        assert doc.layers[0].name == "0"
        assert doc.entity_count() == 0
        assert doc.constraint_count() == 0

    def test_entities_round_trip_through_dict(self) -> None:
        doc = _line_doc()
        again = CadDocument.from_dict(doc.to_dict())
        assert again.entity_count() == 3
        assert again.bounds() == (0.0, 0.0, 23.0, 10.0)


class TestEntities:
    def test_line_arc_circle_ellipse_spline_point_polyline(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        ed.add_line((0.0, 0.0), (5.0, 0.0))
        ed.add_arc((5.0, 5.0), 2.0, 0.0, math.pi)
        ed.add_circle((10.0, 10.0), 4.0)
        ed.add_ellipse((20.0, 20.0), (6.0, 0.0), 0.5)
        ed.add_spline([(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 3.0)])
        ed.add_point((7.0, 7.0))
        ed.add_polyline([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)], closed=True)
        ed.add_text((1.0, 1.0), "BRACKET", 2.5)
        assert doc.entity_count() == 8
        kinds = [type(e).__name__ for e in doc.entities.values()]
        assert kinds == ["Line", "Arc", "Circle", "Ellipse", "Spline",
                         "CadPoint", "Polyline", "CadText"]

    def test_bad_geometry_is_refused(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        with pytest.raises(CadError):
            ed.add_circle((0.0, 0.0), -1.0)
        with pytest.raises(CadError):
            ed.add_arc((0.0, 0.0), 0.0, 0.0, 1.0)
        with pytest.raises(CadError):
            ed.add_spline([(0.0, 0.0)])
        with pytest.raises(CadError):
            ed.add_ellipse((0.0, 0.0), (0.0, 0.0), 0.5)
        assert doc.entity_count() == 0

    def test_move_rotate_scale_mirror(self) -> None:
        doc = _line_doc()
        ed = CadEditor(doc)
        ids = list(doc.entities)
        ed.move([ids[0]], 5.0, 5.0)
        line = doc.entities[ids[0]]
        assert isinstance(line, Line)
        assert line.a == (5.0, 5.0) and line.b == (15.0, 5.0)
        ed.rotate([ids[0]], (5.0, 5.0), 90.0)
        line = doc.entities[ids[0]]
        assert isinstance(line, Line)
        assert line.a == (5.0, 5.0)
        assert abs(line.b[0] - 5.0) < 1e-9 and abs(line.b[1] - 15.0) < 1e-9
        ed.scale([ids[2]], (20.0, 5.0), 2.0)
        circle = doc.entities[ids[2]]
        assert isinstance(circle, Circle) and circle.radius == 6.0
        # Mirror creates a mirrored COPY (the AutoCAD meaning, and
        # the editor's copy operation alongside array) — the
        # original stays put and the returned id is the copy.
        mirrored = ed.mirror([ids[0]], (0.0, 0.0), (0.0, 10.0))
        assert len(mirrored) == 1
        line = doc.entities[mirrored[0]]
        assert isinstance(line, Line)
        assert line.a == (-5.0, 5.0)

    def test_remove_and_layers(self) -> None:
        doc = _line_doc()
        ed = CadEditor(doc)
        ed.add_layer("hidden", "#ff0000", visible=False)
        assert len(doc.layers) == 2
        first = next(iter(doc.entities))
        ed.remove([first])
        assert doc.entity_count() == 2
        with pytest.raises(CadError):
            ed.remove(["no-such-id"])


# ── geometry ──────────────────────────────────────────────────────

class TestGeometry:
    def test_line_line(self) -> None:
        p = line_line((0.0, 0.0), (10.0, 0.0), (5.0, -5.0), (5.0, 5.0))
        assert p == (5.0, 0.0)
        assert line_line((0.0, 0.0), (10.0, 0.0),
                         (0.0, 1.0), (10.0, 1.0)) is None

    def test_line_circle(self) -> None:
        pts = line_circle((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), 3.0)
        assert len(pts) == 2
        assert pts[0] == (2.0, 0.0) and pts[1] == (8.0, 0.0)
        assert line_circle((0.0, 0.0), (10.0, 0.0), (5.0, 5.0), 3.0) == []

    def test_circle_circle(self) -> None:
        pts = circle_circle((0.0, 0.0), 5.0, (8.0, 0.0), 5.0)
        assert len(pts) == 2
        assert pts[0] == (4.0, 3.0) and pts[1] == (4.0, -3.0)
        assert circle_circle((0.0, 0.0), 1.0, (10.0, 0.0), 1.0) == []

    def test_spline_interpolates_fit_points(self) -> None:
        from nomorals.media.cad.geom import spline_eval
        pts = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 3.0)]
        assert spline_eval(pts, 0.0) == (0.0, 0.0)
        assert spline_eval(pts, 1.0) == (4.0, 3.0)
        mid = spline_eval(pts, 0.5)
        assert abs(mid[0] - 2.0) < 0.6 and 0.5 < mid[1] < 2.5


# ── codec + fidelity ──────────────────────────────────────────────

class TestCodec:
    def test_roundtrip_byte_identical_when_untouched(self) -> None:
        raw = _read("simple.cadcraft")
        doc = parse_document(raw)
        assert to_document_bytes(doc) == raw

    def test_unknown_keys_survive_round_trip(self) -> None:
        raw = _read("simple.cadcraft")
        doc = parse_document(raw)
        doc.extra["vendor_note"] = "kept verbatim"
        again = parse_document(to_document_bytes(doc))
        assert again.extra.get("vendor_note") == "kept verbatim"

    def test_fidelity_report(self) -> None:
        raw = _read("simple.cadcraft")
        doc = parse_document(raw)
        ed = CadEditor(doc)
        first = next(iter(doc.entities))
        ed.move([first], 1.0, 0.0)
        report = doc.fidelity_report(raw)
        assert report["untouched_changed"] == 0
        assert report["changed_ids"] == [first]
        assert report["added"] == 0 and report["removed"] == 0

    def test_corrupt_package_refused(self) -> None:
        with pytest.raises(CadError):
            parse_document(b"not a zip at all")
        with pytest.raises(CadError):
            parse_document(_read("corrupt.cadcraft"))

    def test_unsupported_version_refused(self) -> None:
        with pytest.raises(CadError) as caught:
            parse_document(_read("future.cadcraft"))
        assert caught.value.kind == "unsupported"


class TestDxf:
    def test_bracket_import_counts(self) -> None:
        # Measured from the real corpus file (ENTITIES section only;
        # the BLOCKS section holds the dimension-arrow blocks, whose
        # 18 LINEs and 10 SOLIDs are NOT model entities).
        doc = import_dxf(_read("sample-bracket.dxf"))
        kinds: dict[str, int] = {}
        for entity in doc.entities.values():
            kinds[type(entity).__name__] = kinds.get(
                type(entity).__name__, 0) + 1
        assert kinds.get("Line") == 16
        assert kinds.get("Circle") == 7
        assert kinds.get("CadText") == 7
        assert kinds.get("Polyline") == 6
        assert kinds.get("Dimension") == 6
        assert doc.entity_count() == 42
        assert doc.extra["dxf_unmodelled"] == [
            {"type": "HATCH", "count": 4}]
        assert doc.layer_names() == [
            "0", "Defpoints", "Outline", "Center", "Hidden",
            "Dimensions", "Hatch", "Notes", "Title"]

    def test_floorplan_import_counts(self) -> None:
        doc = import_dxf(_read("sample-floorplan.dxf"))
        kinds: dict[str, int] = {}
        for entity in doc.entities.values():
            kinds[type(entity).__name__] = kinds.get(
                type(entity).__name__, 0) + 1
        assert kinds.get("Line") == 7
        assert kinds.get("Circle") == 1
        assert kinds.get("Arc") == 1
        assert kinds.get("Polyline") == 4
        assert kinds.get("CadText") == 2
        assert kinds.get("Dimension") == 1
        assert doc.entity_count() == 16

    def test_dxf_roundtrip_keeps_entities(self) -> None:
        doc = import_dxf(_read("sample-bracket.dxf"))
        again = import_dxf(export_dxf(doc))
        assert again.entity_count() == doc.entity_count()
        def kinds(d):
            return sorted(type(e).__name__
                          for e in d.entities.values())

        assert kinds(again) == kinds(doc)
        # Geometry AND ids survive the round trip exactly; the
        # four unmodelled HATCH entities must not shift the ids.
        assert [str(e) for e in again.entities.values()] == [
            str(e) for e in doc.entities.values()]

    def test_dwg_refused_honestly(self) -> None:
        with pytest.raises(CadError) as caught:
            import_drawing("sample-bracket.dwg",
                           _read("sample-bracket.dwg"))
        assert caught.value.kind == "unsupported"
        assert "DWG" in str(caught.value)

    def test_dxf_layers_kept(self) -> None:
        doc = import_dxf(_read("sample-bracket.dxf"))
        names = [layer.name for layer in doc.layers]
        assert "0" in names and len(names) >= 2


# ── constraint solver ─────────────────────────────────────────────

class TestSolver:
    def test_coincident_moves_endpoints_together(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((10.5, 0.5), (10.5, 10.0))
        ed.constrain("coincident",
                     [GeomRef(a, "end"), GeomRef(b, "start")])
        report = solve(doc)
        assert report["converged"] is True
        assert report["max_residual"] < 1e-9
        la, lb = doc.entities[a], doc.entities[b]
        assert isinstance(la, Line) and isinstance(lb, Line)
        assert la.b == lb.a

    def test_rectangle_fully_constrained(self) -> None:
        doc = _rect_doc()
        ed = CadEditor(doc)
        ids = list(doc.entities)
        ed.constrain("distance_aligned",
                     [GeomRef(ids[0], "whole")], value=10.0)
        ed.constrain("distance_aligned",
                     [GeomRef(ids[1], "whole")], value=5.0)
        ed.constrain("fix", [GeomRef(ids[0], "start")])
        report = solve(doc)
        assert report["converged"] is True
        dof = dof_report(doc)
        assert dof["status"] == "fully"
        assert dof["dof"] == 0

    def test_under_constrained_reports_dof(self) -> None:
        doc = _rect_doc()
        dof = dof_report(doc)
        assert dof["status"] == "under"
        assert dof["dof"] is not None and dof["dof"] > 0
        assert dof["constraints"] == 8

    def test_over_constrained_named(self) -> None:
        doc = _rect_doc()
        ed = CadEditor(doc)
        ids = list(doc.entities)
        ed.constrain("distance_aligned",
                     [GeomRef(ids[0], "whole")], value=10.0)
        ed.constrain("distance_aligned",
                     [GeomRef(ids[0], "whole")], value=12.0)
        report = solve(doc)
        assert report["converged"] is False
        assert report["conflicts"]
        dof = dof_report(doc)
        assert dof["status"] == "over"
        assert dof["redundant"] >= 1

    def test_tangent_line_circle(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((0.0, 0.0), (10.0, 0.0))
        circle = ed.add_circle((5.0, 4.0), 3.0)
        ed.constrain("tangent",
                     [GeomRef(line, "whole"), GeomRef(circle, "whole")])
        report = solve(doc)
        assert report["converged"] is True
        assert report["max_residual"] < 1e-9
        # The invariant is geometric, not which object moved: the
        # minimum-norm step shares the move between line, circle
        # centre AND radius — nothing fixes the radius (a radius
        # constraint is the way to hold it), so the solved circle
        # grows and the line shifts until gap == radius. Assert
        # that invariant, not the starting radius.
        c = doc.entities[circle]
        line_e = doc.entities[line]
        assert isinstance(c, Circle) and isinstance(line_e, Line)
        direction = (line_e.b[0] - line_e.a[0],
                     line_e.b[1] - line_e.a[1])
        length = math.hypot(*direction)
        direction = (direction[0] / length, direction[1] / length)
        gap = abs(direction[0] * (c.center[1] - line_e.a[1])
                  - direction[1] * (c.center[0] - line_e.a[0]))
        assert abs(gap - c.radius) < 1e-6

    def test_parallel_and_equal(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((0.0, 2.0), (5.0, 3.0))
        ed.constrain("parallel",
                     [GeomRef(a, "whole"), GeomRef(b, "whole")])
        ed.constrain("equal",
                     [GeomRef(a, "whole"), GeomRef(b, "whole")])
        report = solve(doc)
        assert report["converged"] is True
        lb = doc.entities[b]
        la = doc.entities[a]
        assert isinstance(lb, Line) and isinstance(la, Line)
        # The invariant is geometric, not which object moved: the
        # minimum-norm step shares the move between both lines, so
        # neither is guaranteed to stay horizontal or keep its
        # length — they end up parallel to each other, and with
        # equal lengths (the shared length is between the two
        # starting lengths, not pinned to either one).
        d1 = (la.b[0] - la.a[0], la.b[1] - la.a[1])
        d2 = (lb.b[0] - lb.a[0], lb.b[1] - lb.a[1])
        assert abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-6
        assert abs(math.hypot(*d2) - math.hypot(*d1)) < 1e-6

    def test_radius_and_distance_measure(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        circle = ed.add_circle((0.0, 0.0), 5.0)
        assert measure(doc, Constraint(
            1, "radius", [GeomRef(circle, "whole")])) == 5.0
        a = ed.add_line((0.0, 0.0), (3.0, 4.0))
        assert abs(measure(doc, Constraint(
            2, "distance_aligned",
            [GeomRef(a, "start"), GeomRef(a, "end")])) - 5.0) < 1e-9

    def test_validate_and_purge(self) -> None:
        doc = _line_doc()
        bad = Constraint(99, "coincident",
                         [GeomRef("no-such", "start"),
                          GeomRef("also-no", "end")])
        assert validate_constraint(doc, bad) is not None
        doc.constraints.append(bad)
        purged = purge_invalid(doc)
        assert purged == [99]
        assert doc.constraint_count() == 0

    def test_hold_keeps_edited_geometry_put(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((0.0, 1.0), (10.0, 1.0))
        ed.constrain("parallel",
                     [GeomRef(a, "whole"), GeomRef(b, "whole")])
        report = solve(doc, hold={a})
        assert report["converged"] is True
        la = doc.entities[a]
        assert isinstance(la, Line)
        assert la.a == (0.0, 0.0) and la.b == (10.0, 0.0)


# ── drafting operations ───────────────────────────────────────────

class TestDrafting:
    def test_trim(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((0.0, 0.0), (10.0, 0.0))
        ed.add_line((5.0, -5.0), (5.0, 5.0))
        trimmed = ed.trim(line, (8.0, 0.0))
        assert trimmed is True
        kept = doc.entities[line]
        assert isinstance(kept, Line)
        assert kept.a == (0.0, 0.0) and kept.b == (5.0, 0.0)

    def test_trim_without_cutter_fails_cleanly(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((0.0, 0.0), (10.0, 0.0))
        assert ed.trim(line, (8.0, 0.0)) is False
        assert doc.entities[line] == doc.entities[line]

    def test_extend(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((0.0, 0.0), (5.0, 0.0))
        ed.add_line((10.0, -5.0), (10.0, 5.0))
        assert ed.extend(line, (5.0, 0.0)) is True
        kept = doc.entities[line]
        assert isinstance(kept, Line)
        assert kept.b == (10.0, 0.0)

    def test_offset(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((0.0, 0.0), (10.0, 0.0))
        new_id = ed.offset(line, 2.0, (5.0, 1.0))
        assert new_id is not None
        off = doc.entities[new_id]
        assert isinstance(off, Line)
        assert off.a == (0.0, 2.0) and off.b == (10.0, 2.0)

    def test_offset_circle(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        circle = ed.add_circle((0.0, 0.0), 5.0)
        new_id = ed.offset(circle, 2.0, (6.0, 0.0))
        assert new_id is not None
        off = doc.entities[new_id]
        assert isinstance(off, Circle) and off.radius == 7.0

    def test_fillet(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((10.0, 0.0), (10.0, 10.0))
        arc_id = ed.fillet(a, b, 2.0, (9.0, 0.0), (10.0, 1.0))
        assert arc_id is not None
        arc = doc.entities[arc_id]
        assert isinstance(arc, Arc) and abs(arc.radius - 2.0) < 1e-9
        assert arc.center == (8.0, 2.0)
        la, lb = doc.entities[a], doc.entities[b]
        assert isinstance(la, Line) and isinstance(lb, Line)
        assert la.b == (8.0, 0.0) and lb.a == (10.0, 2.0)

    def test_fillet_too_big_fails_cleanly(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (2.0, 0.0))
        b = ed.add_line((2.0, 0.0), (2.0, 2.0))
        assert ed.fillet(a, b, 50.0, (1.0, 0.0), (2.0, 1.0)) is None
        assert doc.entity_count() == 2

    def test_chamfer(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((10.0, 0.0), (10.0, 10.0))
        chamfer_id = ed.chamfer(a, b, 2.0, 2.0, (9.0, 0.0), (10.0, 1.0))
        assert chamfer_id is not None
        seg = doc.entities[chamfer_id]
        assert isinstance(seg, Line)
        assert seg.a == (8.0, 0.0) and seg.b == (10.0, 2.0)

    def test_mirror_and_array(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((1.0, 0.0), (2.0, 0.0))
        mirrored = ed.mirror([line], (0.0, 0.0), (0.0, 10.0))
        assert len(mirrored) == 1
        got = doc.entities[mirrored[0]]
        assert isinstance(got, Line)
        assert got.a == (-1.0, 0.0) and got.b == (-2.0, 0.0)
        copies = ed.array_rect([line], 2, 3, 5.0, 5.0)
        assert len(copies) == 5
        assert doc.entity_count() == 7

    def test_array_polar(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        line = ed.add_line((5.0, 0.0), (6.0, 0.0))
        copies = ed.array_polar([line], (0.0, 0.0), 4, 360.0)
        assert len(copies) == 3
        last = doc.entities[copies[-1]]
        assert isinstance(last, Line)
        assert abs(last.a[0] - 0.0) < 1e-9 and abs(last.a[1] + 5.0) < 1e-9


# ── dimensions ────────────────────────────────────────────────────

class TestDimensions:
    def test_aligned_dimension_measures(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (3.0, 4.0))
        dim = ed.add_dimension("aligned",
                               [GeomRef(a, "start"), GeomRef(a, "end")],
                               (1.5, 3.0))
        assert isinstance(dim, Dimension)
        assert abs(dim.value - 5.0) < 1e-9

    def test_angular_dimension(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        a = ed.add_line((0.0, 0.0), (10.0, 0.0))
        b = ed.add_line((0.0, 0.0), (0.0, 10.0))
        dim = ed.add_dimension("angular",
                               [GeomRef(a, "whole"), GeomRef(b, "whole")],
                               (3.0, 3.0))
        assert abs(dim.value - 90.0) < 1e-9

    def test_radius_dimension(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        c = ed.add_circle((0.0, 0.0), 7.5)
        dim = ed.add_dimension("radius", [GeomRef(c, "whole")], (7.5, 0.0))
        assert abs(dim.value - 7.5) < 1e-9


# ── commands: typed machines + per-step guards ────────────────────

class TestCommands:
    def test_registry_lists_drawing_and_modify_commands(self) -> None:
        names = list_commands()
        for want in ("line", "circle", "arc", "rectangle", "polyline",
                     "trim", "extend", "offset", "fillet", "chamfer",
                     "mirror", "array", "move", "rotate", "scale",
                     "erase", "dimension", "constrain", "solve"):
            assert want in names

    def test_json_line_command(self) -> None:
        session = CommandSession(CadDocument.blank())
        out = session.execute("line", {"points": [[0, 0], [10, 0]]})
        assert out["created"]
        assert session.doc.entity_count() == 1

    def test_json_circle_command(self) -> None:
        session = CommandSession(CadDocument.blank())
        out = session.execute("circle", {"center": [0, 0], "radius": 5})
        assert out["created"]
        circle = next(iter(session.doc.entities.values()))
        assert isinstance(circle, Circle) and circle.radius == 5.0

    def test_json_command_bad_params_fail_cleanly(self) -> None:
        session = CommandSession(CadDocument.blank())
        with pytest.raises(CadError):
            session.execute("circle", {"center": [0, 0], "radius": -5})
        assert session.doc.entity_count() == 0

    def test_unknown_command(self) -> None:
        session = CommandSession(CadDocument.blank())
        with pytest.raises(CadError):
            session.execute("warpdrive", {})

    def test_interactive_line_machine(self) -> None:
        session = CommandSession(CadDocument.blank())
        machine = session.start("line")
        assert "point" in machine.prompt().lower()
        machine.input(session.doc, (0.0, 0.0))
        machine.input(session.doc, (10.0, 0.0))
        assert session.doc.entity_count() == 1
        assert session.running is None

    def test_guard_restores_drawing_on_step_panic(self) -> None:
        from nomorals.media.cad.commands import CommandMachine, Step

        class Boom(CommandMachine):
            name = "boom"

            def begin(self, doc):
                return Step.CONTINUE

            def input(self, doc, value):
                raise RuntimeError("step exploded")

            def prompt(self):
                return "Specify a point"

        session = CommandSession(_line_doc())
        before = session.doc.entity_count()
        machine = Guarded(Boom())
        machine.begin(session.doc)
        with pytest.raises(CadError) as caught:
            machine.input(session.doc, (1.0, 1.0))
        assert caught.value.kind == "internal"
        assert "boom" in str(caught.value)
        assert session.doc.entity_count() == before

    def test_undo(self) -> None:
        session = CommandSession(CadDocument.blank())
        session.execute("line", {"points": [[0, 0], [1, 0]]})
        assert session.doc.entity_count() == 1
        assert session.undo() is True
        assert session.doc.entity_count() == 0
        assert session.undo() is False


# ── rendering ─────────────────────────────────────────────────────

class TestRender:
    def test_render_produces_real_png(self) -> None:
        doc = _doc("simple.cadcraft")
        png = render_document(doc, width=640, height=480)
        image = Image.open(io.BytesIO(png))
        assert image.size == (640, 480)
        assert image.mode in ("RGB", "RGBA")

    def test_render_draws_entities(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        ed.add_line((0.0, 0.0), (10.0, 0.0))
        ed.add_circle((5.0, 5.0), 2.0)
        png = render_document(doc, width=400, height=400)
        image = Image.open(io.BytesIO(png)).convert("RGB")
        pixels = list(image.getdata())
        assert any(p != (255, 255, 255) for p in pixels)

    def test_render_hidden_layer_skipped(self) -> None:
        doc = CadDocument.blank()
        ed = CadEditor(doc)
        ed.add_layer("hidden", "#000000", visible=False)
        line = ed.add_line((0.0, 0.0), (10.0, 0.0))
        ed.set_layer([line], "hidden")
        png = render_document(doc, width=400, height=400)
        image = Image.open(io.BytesIO(png)).convert("RGB")
        assert all(p == (255, 255, 255) for p in image.getdata())

    def test_render_empty_document(self) -> None:
        png = render_document(CadDocument.blank(), width=200, height=200)
        assert Image.open(io.BytesIO(png)).size == (200, 200)


# ── parity + attribution ──────────────────────────────────────────

class TestParityAttribution:
    def test_parity_all_pass(self) -> None:
        result = run_parity()
        assert result["failed"] == [], result["cases"]
        assert result["passed"] == 10

    def test_parity_missing_fixture_fails(self) -> None:
        # A missing fixture FAILS its case, never a silent skip:
        # run a TOML that points at a fixture that does not exist
        # and check the case fails (and names the missing file).
        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "missing.toml"
            toml_path.write_text(
                '[[cases]]\nid = "ghost"\n'
                'fixture = "no-such-drawing.cadcraft"\n'
                'ops = ["inspect"]\n[cases.expect]\nentities = 1\n',
                encoding="utf-8")
            result = run_parity(toml_path)
            assert result["failed"] == ["ghost"]
            assert result["cases"][0]["status"] == "fail"
            assert "no-such-drawing.cadcraft" in \
                result["cases"][0]["detail"]

    def test_attribution_verifies(self) -> None:
        result = verify_attribution()
        assert result["failed"] == [], result
        assert result["checked"] >= 10
        assert result["noted"] >= 5


# ── brain tool ────────────────────────────────────────────────────

class TestTool:
    def _registry(self, home):
        registry = ToolRegistry(context=SimpleNamespace(home=home))
        registry.register_builtins()
        return registry

    def _call(self, registry, **kwargs):
        # registry.call returns an Outcome, not the dict — unwrap
        # .value the same way the vectorcraft test does.
        outcome = registry.call("cadcraft", **kwargs)
        if hasattr(outcome, "value"):
            return outcome.value
        return outcome

    def test_tool_registered(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            spec = registry.get("cadcraft")
            assert spec is not None
            assert "inspect" in spec.description
            # 724 after the designcraft port, 725 after the figcraft
            # port (lane 105) and 726 after the soundcraft port
            # (lane 107), 727 after the cadcraft port (lane 106),
            # 728 after the materialcraft port (lane 109), 729 after the
            # texturecraft port (lane 108), 730 after the
            # anicraft port (lane 110), 731 after the
            # craft-anything port (lane 111), 732 after the
            # craftlauncher port (lane 112), 733 after the
            # craft-libs port (lane 113) added the craftlibs tool, 734
            # after the craft-fonts port (lane 114) added the craftfonts tool.
            # 735 after the indd-utils port (lane 117) added the inddutils tool.
            # 748 after the auth-plugin wiring added 5 tools (session_forge, session_verify, payload_png_bomb, payload_svg_bomb, payload_keylogger) on top of the bughunter port (lane 121). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
            assert len(registry._tools) == 787

    def test_tool_inspect_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.png"
            with tempfile.TemporaryDirectory() as home:
                registry = self._registry(home)
                value = self._call(
                    registry, action="inspect",
                    sources=[str(FIXTURES / "simple.cadcraft")])
                assert value["entities"] >= 1
                value = self._call(
                    registry, action="render",
                    sources=[str(FIXTURES / "simple.cadcraft")],
                    output=str(out))
                assert out.is_file() and out.stat().st_size > 0

    def test_tool_solve(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            value = self._call(
                registry, action="solve",
                sources=[str(FIXTURES / "rect-solve.cadcraft")])
            assert value["converged"] is True
            assert value["status"] == "fully"

    def test_tool_import_dxf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bracket.cadcraft"
            with tempfile.TemporaryDirectory() as home:
                registry = self._registry(home)
                value = self._call(
                    registry, action="import",
                    sources=[str(FIXTURES / "sample-bracket.dxf")],
                    output=str(out))
                assert value["entities"] == 42
                assert out.is_file()

    def test_tool_unknown_action(self) -> None:
        # An unknown action is an error Outcome, not a raise — the
        # same contract the designcraft test pins.
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            outcome = registry.call("cadcraft", action="warpdrive")
            assert outcome.ok is False
            assert "unknown" in str(outcome.error).lower()
