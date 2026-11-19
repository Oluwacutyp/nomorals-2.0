"""Typed command state machines for CadCraft.

Upstream (crates/engine/src/cmd.rs, machines.rs, guard.rs): every
drafting action is a command with a typed spec — an id (the AutoCAD
command name, lowercase), a list of typed steps, and a Guarded wrapper
that restores the drawing when a step fails. A bad step fails the step
with a named reason; it never corrupts the sketch.

This module is the Python port of that shape:

* :class:`CommandMachine` — one command, one step at a time. ``input``
  feeds the current step a typed value and either advances, finishes
  (returning the created entity ids), or raises
  :class:`~nomorals.media.cad.errors.CadError` naming the step that
  refused the value.
* :class:`Guarded` — wraps a machine; on any step failure it restores
  the document to the snapshot taken at ``begin`` and reports the
  failure as a ``CadError`` with kind ``"internal"`` and the step in
  the message — the drawing is never left half-edited.
* :class:`CommandSession` — the interactive session: ``execute`` runs
  a whole command from a parameter dict (the scripted path),
  ``start``/``feed``/``undo`` drive one machine interactively.
* :func:`list_commands` — the registry of every command id the organ
  knows, with its prompt text.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .edit import CadEditor
from .errors import CadError
from .model import CadDocument, GeomRef, Sub
from .solver import dof_report, solve


class Step:
    """Step outcomes, mirroring upstream's machine step results."""
    CONTINUE = "continue"
    DONE = "done"


@dataclass
class CommandSpec:
    """The typed description of one command (upstream CommandSpec)."""
    id: str
    description: str
    steps: tuple[str, ...] = ()
    # The machine class that implements this command.
    machine: type | None = None


class CommandMachine:
    """Base class for a typed drafting command (upstream machines.rs).

    A machine owns one :class:`CadEditor` over the session document
    and a step counter. ``begin`` resets it; ``prompt`` names what the
    current step wants; ``input`` consumes one typed value.
    """

    name: str = "command"
    steps: tuple[str, ...] = ()

    def __init__(self, editor: CadEditor | None = None) -> None:
        self.editor = editor
        self.step = 0
        self.created: list[str] = []

    @property
    def running(self) -> bool:
        return self.step < len(self.steps)

    def begin(self, doc: CadDocument | None = None) -> None:
        self.step = 0
        self.created = []

    def prompt(self) -> str:
        if not self.running:
            return f"{self.name}: done."
        return self.steps[self.step]

    def input(self, doc: CadDocument, value: object) -> str:
        """Feed one typed value to the current step.

        Returns ``Step.CONTINUE`` or ``Step.DONE``. Raises CadError
        naming the step when the value is refused.
        """
        raise NotImplementedError

    def _fail(self, what: str, detail: str) -> CadError:
        return CadError(f"{self.name} step {self.step + 1} "
                        f"({self.steps[self.step]}): {what}: {detail}",
                        kind="invalid_argument")


def _point(value: object) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    raise CadError(f"expected a point (x, y), got {value!r}",
                   kind="invalid_argument")


def _number(value: object, what: str = "a number") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CadError(f"expected {what}, got {value!r}",
                       kind="invalid_argument")
    return float(value)


def _entity_id(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    raise CadError(f"expected an entity id, got {value!r}",
                   kind="invalid_argument")


class LineMachine(CommandMachine):
    name = "line"
    steps = ("first point", "second point")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._first: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._first = _point(value)
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            second = _point(value)
            if second == self._first:
                raise self._fail("zero-length line",
                                 "second point equals the first")
            self.created = [self.editor.add_line(self._first, second)]
            self.step = 2
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class CircleMachine(CommandMachine):
    name = "circle"
    steps = ("centre point", "radius")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._centre: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._centre = _point(value)
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            radius = _number(value, "a radius")
            if radius <= 0:
                raise self._fail("bad radius",
                                 f"radius must be positive, got {radius}")
            self.created = [self.editor.add_circle(self._centre, radius)]
            self.step = 2
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ArcMachine(CommandMachine):
    name = "arc"
    steps = ("centre point", "radius", "start angle (degrees)",
             "end angle (degrees)")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._centre: tuple[float, float] | None = None
        self._radius = 0.0
        self._start = 0.0

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._centre = _point(value)
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._radius = _number(value, "a radius")
            if self._radius <= 0:
                raise self._fail("bad radius",
                                 f"radius must be positive, got {self._radius}")
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            self._start = _number(value, "an angle in degrees")
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            end = _number(value, "an angle in degrees")
            if math.isclose(end, self._start):
                raise self._fail("zero sweep",
                                 "start and end angles are equal")
            self.created = [self.editor.add_arc(
                self._centre, self._radius, self._start, end)]
            self.step = 4
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class RectangleMachine(CommandMachine):
    name = "rectangle"
    steps = ("first corner", "opposite corner")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._first: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._first = _point(value)
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            second = _point(value)
            if second == self._first:
                raise self._fail("zero-size rectangle",
                                 "corners are identical")
            self.created = [self.editor.add_polyline(
                [self._first,
                 (second[0], self._first[1]),
                 second,
                 (self._first[0], second[1])], closed=True)]
            self.step = 2
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class PolylineMachine(CommandMachine):
    name = "polyline"
    steps = ("next vertex (or 'done')",)

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._vertices: list[tuple[float, float]] = []

    @property
    def running(self) -> bool:
        # Polyline stays open until the user says done.
        return self.step == 0

    def prompt(self) -> str:
        return (f"polyline: vertex {len(self._vertices) + 1} "
                "(or 'done' to finish)")

    def input(self, doc: CadDocument, value: object) -> str:
        if value == "done" or value == "close":
            if len(self._vertices) < 2:
                raise self._fail("too few vertices",
                                 "a polyline needs at least two points")
            self.created = [self.editor.add_polyline(
                self._vertices, closed=(value == "close"))]
            self.step = 1
            return Step.DONE
        point = _point(value)
        if self._vertices and point == self._vertices[-1]:
            raise self._fail("duplicate vertex",
                             "point equals the previous vertex")
        self._vertices.append(point)
        return Step.CONTINUE


class _TwoEntityMachine(CommandMachine):
    """Shared shape for commands that pick two entities then values."""

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._first_id = ""
        self._second_id = ""

    def _pick(self, value: object, step: int) -> str:
        eid = _entity_id(value)
        if eid not in self.editor.doc.entities:
            raise self._fail("unknown entity", f"no entity {eid!r}")
        if step == 0:
            self._first_id = eid
        else:
            if eid == self._first_id:
                raise self._fail("same entity twice",
                                 "pick two different entities")
            self._second_id = eid
        self.step += 1
        return Step.CONTINUE


class TrimMachine(_TwoEntityMachine):
    name = "trim"
    steps = ("entity to trim", "pick point on the side to remove")

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            return self._pick(value, 0)
        if self.step == 1:
            pick = _point(value)
            if not self.editor.trim(self._first_id, pick):
                raise self._fail("nothing to trim",
                                 "no boundary cuts this entity here")
            self.created = [self._first_id]
            self.step = 2
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ExtendMachine(_TwoEntityMachine):
    name = "extend"
    steps = ("entity to extend", "pick point towards the boundary")

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            return self._pick(value, 0)
        if self.step == 1:
            pick = _point(value)
            if not self.editor.extend(self._first_id, pick):
                raise self._fail("nothing to extend to",
                                 "no boundary lies in that direction")
            self.created = [self._first_id]
            self.step = 2
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class OffsetMachine(CommandMachine):
    name = "offset"
    steps = ("entity to offset", "distance", "side point")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._distance = 0.0

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._distance = _number(value, "a distance")
            if self._distance == 0:
                raise self._fail("zero distance",
                                 "offset distance must not be zero")
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            side = _point(value)
            new_id = self.editor.offset(self._eid, self._distance, side)
            if new_id is None:
                raise self._fail("offset failed",
                                 "entity type cannot be offset")
            self.created = [new_id]
            self.step = 3
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class FilletMachine(_TwoEntityMachine):
    name = "fillet"
    steps = ("first entity", "second entity", "radius",
             "pick point near the first entity's kept end",
             "pick point near the second entity's kept end")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._radius = 0.0
        self._pick1: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            return self._pick(value, 0)
        if self.step == 1:
            return self._pick(value, 1)
        if self.step == 2:
            self._radius = _number(value, "a radius")
            if self._radius <= 0:
                raise self._fail("bad radius",
                                 f"radius must be positive, got {self._radius}")
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            self._pick1 = _point(value)
            self.step = 4
            return Step.CONTINUE
        if self.step == 4:
            pick2 = _point(value)
            arc_id = self.editor.fillet(self._first_id, self._second_id,
                                        self._radius, self._pick1, pick2)
            if arc_id is None:
                raise self._fail("fillet impossible",
                                 "radius too large for this corner, or "
                                 "the entities do not meet")
            self.created = [arc_id]
            self.step = 5
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ChamferMachine(_TwoEntityMachine):
    name = "chamfer"
    steps = ("first entity", "second entity", "first distance",
             "second distance",
             "pick point near the first entity's kept end",
             "pick point near the second entity's kept end")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._d1 = 0.0
        self._d2 = 0.0
        self._pick1: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            return self._pick(value, 0)
        if self.step == 1:
            return self._pick(value, 1)
        if self.step == 2:
            self._d1 = _number(value, "a distance")
            if self._d1 <= 0:
                raise self._fail("bad distance",
                                 f"distance must be positive, got {self._d1}")
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            self._d2 = _number(value, "a distance")
            if self._d2 <= 0:
                raise self._fail("bad distance",
                                 f"distance must be positive, got {self._d2}")
            self.step = 4
            return Step.CONTINUE
        if self.step == 4:
            self._pick1 = _point(value)
            self.step = 5
            return Step.CONTINUE
        if self.step == 5:
            pick2 = _point(value)
            line_id = self.editor.chamfer(
                self._first_id, self._second_id, self._d1, self._d2,
                self._pick1, pick2)
            if line_id is None:
                raise self._fail("chamfer impossible",
                                 "distances too large for this corner, "
                                 "or the entities do not meet")
            self.created = [line_id]
            self.step = 6
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class MirrorMachine(CommandMachine):
    name = "mirror"
    steps = ("entity id to mirror", "mirror line first point",
             "mirror line second point")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._p1: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._p1 = _point(value)
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            p2 = _point(value)
            if p2 == self._p1:
                raise self._fail("zero-length mirror line",
                                 "mirror points are identical")
            self.created = self.editor.mirror([self._eid], self._p1, p2)
            self.step = 3
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ArrayMachine(CommandMachine):
    name = "array"
    steps = ("entity id", "rows", "columns", "row spacing",
             "column spacing")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._rows = 1
        self._cols = 1
        self._dx = 0.0

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._rows = int(_number(value, "a row count"))
            if self._rows < 1:
                raise self._fail("bad row count",
                                 f"rows must be at least 1, got {self._rows}")
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            self._cols = int(_number(value, "a column count"))
            if self._cols < 1:
                raise self._fail("bad column count",
                                 f"columns must be at least 1, got {self._cols}")
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            self._dx = _number(value, "a spacing")
            self.step = 4
            return Step.CONTINUE
        if self.step == 4:
            dy = _number(value, "a spacing")
            self.created = self.editor.array_rect(
                [self._eid], self._rows, self._cols, self._dx, dy)
            self.step = 5
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class MoveMachine(CommandMachine):
    name = "move"
    steps = ("entity id", "delta x", "delta y")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._dx = 0.0

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._dx = _number(value, "a delta")
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            dy = _number(value, "a delta")
            self.editor.move([self._eid], self._dx, dy)
            self.created = [self._eid]
            self.step = 3
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class RotateMachine(CommandMachine):
    name = "rotate"
    steps = ("entity id", "centre point", "angle (degrees)")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._centre: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._centre = _point(value)
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            angle = _number(value, "an angle in degrees")
            self.editor.rotate([self._eid], self._centre, angle)
            self.created = [self._eid]
            self.step = 3
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ScaleMachine(CommandMachine):
    name = "scale"
    steps = ("entity id", "centre point", "factor")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._eid = ""
        self._centre: tuple[float, float] | None = None

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            self._eid = _entity_id(value)
            if self._eid not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._eid!r}")
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._centre = _point(value)
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            factor = _number(value, "a scale factor")
            if factor <= 0:
                raise self._fail("bad factor",
                                 f"factor must be positive, got {factor}")
            self.editor.scale([self._eid], self._centre, factor)
            self.created = [self._eid]
            self.step = 3
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class EraseMachine(CommandMachine):
    name = "erase"
    steps = ("entity id",)

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            eid = _entity_id(value)
            if eid not in self.editor.doc.entities:
                raise self._fail("unknown entity", f"no entity {eid!r}")
            self.editor.remove([eid])
            self.created = [eid]
            self.step = 1
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class DimensionMachine(CommandMachine):
    name = "dimension"
    steps = ("kind (aligned, angular, radius, diameter)",
             "first entity id", "second entity id or ''",
             "dimension line point")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._kind = "aligned"
        self._first = ""
        self._second = ""

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            if not isinstance(value, str) or value not in (
                    "aligned", "angular", "radius", "diameter"):
                raise self._fail(
                    "bad dimension kind",
                    "expected aligned, angular, radius or diameter")
            self._kind = value
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._first = _entity_id(value)
            if self._first not in self.editor.doc.entities:
                raise self._fail("unknown entity",
                                 f"no entity {self._first!r}")
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            if value not in ("", None):
                self._second = _entity_id(value)
                if self._second not in self.editor.doc.entities:
                    raise self._fail("unknown entity",
                                     f"no entity {self._second!r}")
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            position = _point(value)
            refs = [GeomRef(self._first, Sub.WHOLE)]
            if self._second:
                refs.append(GeomRef(self._second, Sub.WHOLE))
            dim = self.editor.add_dimension(self._kind, refs, position)
            self.created = [dim.id]
            self.step = 4
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class ConstrainMachine(CommandMachine):
    name = "constrain"
    steps = ("kind", "first reference (entity id or id:sub)",
             "second reference or ''", "value or ''")

    def __init__(self, editor: CadEditor) -> None:
        super().__init__(editor)
        self._kind = ""
        self._first = ""
        self._second = ""

    @staticmethod
    def _ref(value: object) -> GeomRef:
        if not isinstance(value, str) or not value:
            raise CadError(f"expected a reference, got {value!r}",
                           kind="invalid_argument")
        if ":" in value:
            handle, _, sub = value.partition(":")
            return GeomRef(handle, Sub.parse(sub))
        return GeomRef(value, Sub.WHOLE)

    def input(self, doc: CadDocument, value: object) -> str:
        if self.step == 0:
            if not isinstance(value, str) or not value:
                raise self._fail("bad constraint kind",
                                 "expected a constraint kind name")
            self._kind = value
            self.step = 1
            return Step.CONTINUE
        if self.step == 1:
            self._first = value  # type: ignore[assignment]
            self._ref(value)
            self.step = 2
            return Step.CONTINUE
        if self.step == 2:
            if value not in ("", None):
                self._second = value  # type: ignore[assignment]
                self._ref(value)
            self.step = 3
            return Step.CONTINUE
        if self.step == 3:
            amount = None
            if value not in ("", None):
                amount = _number(value, "a value")
            refs = [self._ref(self._first)]
            if self._second:
                refs.append(self._ref(self._second))
            self.editor.constrain(self._kind, refs, amount)
            self.created = []
            self.step = 4
            return Step.DONE
        raise self._fail("already finished", "no more input expected")


class SolveMachine(CommandMachine):
    name = "solve"
    steps = ()

    def input(self, doc: CadDocument, value: object) -> str:
        solve(self.editor.doc)
        self.created = []
        self.step = 1
        return Step.DONE


#: Every command the organ knows, in census order.
COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("line", "Draw a line between two points",
                LineMachine.steps, LineMachine),
    CommandSpec("circle", "Draw a circle from a centre and a radius",
                CircleMachine.steps, CircleMachine),
    CommandSpec("arc", "Draw an arc from a centre, radius and angles",
                ArcMachine.steps, ArcMachine),
    CommandSpec("rectangle", "Draw a closed rectangle from two corners",
                RectangleMachine.steps, RectangleMachine),
    CommandSpec("polyline", "Draw a polyline, one vertex at a time",
                PolylineMachine.steps, PolylineMachine),
    CommandSpec("trim", "Cut an entity at the nearest boundary",
                TrimMachine.steps, TrimMachine),
    CommandSpec("extend", "Extend an entity to the nearest boundary",
                ExtendMachine.steps, ExtendMachine),
    CommandSpec("offset", "Copy an entity at a distance",
                OffsetMachine.steps, OffsetMachine),
    CommandSpec("fillet", "Round a corner with an arc",
                FilletMachine.steps, FilletMachine),
    CommandSpec("chamfer", "Cut a corner with a straight segment",
                ChamferMachine.steps, ChamferMachine),
    CommandSpec("mirror", "Mirror an entity across a line",
                MirrorMachine.steps, MirrorMachine),
    CommandSpec("array", "Repeat an entity in rows and columns",
                ArrayMachine.steps, ArrayMachine),
    CommandSpec("move", "Move an entity by a delta",
                MoveMachine.steps, MoveMachine),
    CommandSpec("rotate", "Rotate an entity about a centre",
                RotateMachine.steps, RotateMachine),
    CommandSpec("scale", "Scale an entity about a centre",
                ScaleMachine.steps, ScaleMachine),
    CommandSpec("erase", "Delete an entity",
                EraseMachine.steps, EraseMachine),
    CommandSpec("dimension", "Add a dimension",
                DimensionMachine.steps, DimensionMachine),
    CommandSpec("constrain", "Add a constraint",
                ConstrainMachine.steps, ConstrainMachine),
    CommandSpec("solve", "Solve the sketch's constraints",
                SolveMachine.steps, SolveMachine),
)

_COMMAND_MAP = {spec.id: spec for spec in COMMAND_SPECS}


def list_commands() -> list[str]:
    """The id of every command the organ knows, in census order."""
    return [spec.id for spec in COMMAND_SPECS]


def command_specs() -> list[dict[str, object]]:
    """Every command id with its description and prompt steps."""
    return [{"id": spec.id, "description": spec.description,
             "steps": list(spec.steps)} for spec in COMMAND_SPECS]


class Guarded:
    """A machine wrapped in a per-step panic guard (upstream guard.rs).

    The document is snapshotted at ``begin``. If any step raises,
    the snapshot is restored before the error surfaces, so the sketch
    is exactly as it was before the command started — never half
    edited. The raised error names the step and the reason.
    """

    def __init__(self, machine: CommandMachine) -> None:
        self.machine = machine
        self._snapshot: CadDocument | None = None
        self._doc: CadDocument | None = None

    @property
    def name(self) -> str:
        return self.machine.name

    @property
    def running(self) -> bool:
        return self.machine.running

    def begin(self, doc: CadDocument | None = None) -> None:
        if doc is None and self.machine.editor is not None:
            doc = self.machine.editor.doc
        if doc is None:
            raise CadError("guarded command needs a document",
                           kind="invalid_argument")
        self._doc = doc
        self._snapshot = copy.deepcopy(doc)
        self.machine.begin(doc)

    def prompt(self) -> str:
        return self.machine.prompt()

    def input(self, doc: CadDocument, value: object) -> str:
        try:
            return self.machine.input(doc, value)
        except CadError as exc:
            self._restore()
            raise CadError(
                f"guarded {self.machine.name}: step failed, drawing "
                f"restored — {exc}", kind="internal") from exc
        except Exception as exc:  # pragma: no cover - defence in depth
            self._restore()
            raise CadError(
                f"guarded {self.machine.name}: step panicked "
                f"({exc.__class__.__name__}: {exc}), drawing restored",
                kind="internal") from exc

    def _restore(self) -> None:
        if self._snapshot is None or self._doc is None:
            return
        doc = self._doc
        restored = self._snapshot
        doc.title = restored.title
        doc.units = restored.units
        doc.layers = copy.deepcopy(restored.layers)
        doc.entities = copy.deepcopy(restored.entities)
        doc.constraints = copy.deepcopy(restored.constraints)
        doc.blocks = copy.deepcopy(restored.blocks)
        doc.extra = copy.deepcopy(restored.extra)
        doc._next_id = restored._next_id
        doc._dirty_ids = set(restored._dirty_ids)
        self._snapshot = None


class CommandSession:
    """An interactive drafting session over one document.

    ``execute`` runs a whole command from a parameter dict (the
    scripted path — each step still goes through the guarded machine,
    one value at a time). ``start``/``feed`` drive one machine
    interactively; ``undo`` pops the last finished command's snapshot.
    """

    def __init__(self, doc: CadDocument) -> None:
        self.doc = doc
        self.editor = CadEditor(doc)
        self.machine: Guarded | None = None
        self._undo_stack: list[CadDocument] = []

    @property
    def running(self) -> Guarded | None:
        """The active machine, or None when no command is running."""
        if self.machine is not None and self.machine.running:
            return self.machine
        return None

    def prompt(self) -> str:
        if self.machine is None:
            return "no command running — start one with start()."
        return self.machine.prompt()

    def start(self, command: str) -> Guarded:
        spec = _COMMAND_MAP.get(command)
        if spec is None or spec.machine is None:
            raise CadError(f"unknown command {command!r}",
                           kind="invalid_argument")
        machine = Guarded(spec.machine(self.editor))
        machine.begin(self.doc)
        self.machine = machine
        return machine

    def feed(self, value: object) -> str:
        if self.machine is None:
            raise CadError("no command running", kind="invalid_argument")
        result = self.machine.input(self.doc, value)
        if result == Step.DONE:
            self.machine = None
        return result

    def execute(self, command: str,
                params: dict[str, object] | None = None) -> dict:
        """Run a whole command from a parameter dict.

        The dict keys are the machine's step names in plain form
        (``points``, ``centre``, ``radius``, …); each command documents
        the keys it reads. Returns ``{"created": [...]}``.
        """
        params = params or {}
        spec = _COMMAND_MAP.get(command)
        if spec is None:
            raise CadError(f"unknown command {command!r}",
                           kind="invalid_argument")
        before = copy.deepcopy(self.doc)
        machine = self.start(command)
        try:
            values = _params_to_steps(command, params)
            for value in values:
                result = self.feed(value)
                if result == Step.DONE:
                    break
            if self.running:
                raise CadError(
                    f"{command}: missing inputs — still waiting for "
                    f"{self.prompt()}", kind="invalid_argument")
        except Exception:
            self.machine = None
            raise
        created = list(machine.machine.created)
        self._undo_stack.append(before)
        return {"created": created}

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        before = self._undo_stack.pop()
        self.doc.title = before.title
        self.doc.units = before.units
        self.doc.layers = copy.deepcopy(before.layers)
        self.doc.entities = copy.deepcopy(before.entities)
        self.doc.constraints = copy.deepcopy(before.constraints)
        self.doc.blocks = copy.deepcopy(before.blocks)
        self.doc.extra = copy.deepcopy(before.extra)
        self.doc._next_id = before._next_id
        self.machine = None
        return True

    def solve(self, hold: dict | None = None) -> dict:
        return solve(self.doc, hold=hold or {})

    def dof(self) -> dict:
        return dof_report(self.doc)


def _params_to_steps(command: str,
                     params: dict[str, object]) -> list[object]:
    """Translate a parameter dict into the machine's step values."""
    p = params
    if command == "line":
        points = p.get("points") or [p.get("a"), p.get("b")]
        return list(points)
    if command == "circle":
        return [p.get("centre", p.get("center")), p.get("radius")]
    if command == "arc":
        return [p.get("centre", p.get("center")), p.get("radius"),
                p.get("start", 0.0), p.get("end", 90.0)]
    if command == "rectangle":
        return [p.get("a"), p.get("b")]
    if command == "polyline":
        vertices = list(p.get("vertices") or [])
        vertices.append("close" if p.get("closed") else "done")
        return vertices
    if command == "trim":
        return [p.get("entity"), p.get("pick")]
    if command == "extend":
        return [p.get("entity"), p.get("pick")]
    if command == "offset":
        return [p.get("entity"), p.get("distance"), p.get("side")]
    if command == "fillet":
        return [p.get("first"), p.get("second"), p.get("radius"),
                p.get("pick1"), p.get("pick2")]
    if command == "chamfer":
        return [p.get("first"), p.get("second"), p.get("d1"),
                p.get("d2"), p.get("pick1"), p.get("pick2")]
    if command == "mirror":
        return [p.get("entity"), p.get("p1"), p.get("p2")]
    if command == "array":
        return [p.get("entity"), p.get("rows", 1), p.get("cols", 1),
                p.get("dx", 0.0), p.get("dy", 0.0)]
    if command == "move":
        delta = p.get("delta", (0.0, 0.0))
        return [p.get("entity"), delta[0], delta[1]]
    if command == "rotate":
        return [p.get("entity"), p.get("centre", p.get("center")),
                p.get("angle", 0.0)]
    if command == "scale":
        return [p.get("entity"), p.get("centre", p.get("center")),
                p.get("factor", 1.0)]
    if command == "erase":
        return [p.get("entity")]
    if command == "dimension":
        return [p.get("kind", "aligned"), p.get("first"),
                p.get("second", ""), p.get("position")]
    if command == "constrain":
        return [p.get("kind"), p.get("first"), p.get("second", ""),
                p.get("value", "")]
    if command == "solve":
        return [None]
    raise CadError(f"unknown command {command!r}",
                   kind="invalid_argument")
