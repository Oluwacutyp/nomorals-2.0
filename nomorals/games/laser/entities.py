"""Laser Asteroids entities — the drawable, collidable objects.

Ported from ``entities/*.py`` in echelon/laser-asteroids (MIT, Copyright
(c) 2012 Brandon Thomas), translated to Python 3 (``xrange`` → ``range``,
no ``pygame``/``thread`` imports — those served the original's joystick
threads, not the shapes). Each entity is a point generator: ``produce()``
yields ``(x, y, r, g, b)`` laser points for one drawing of the shape.

Serialization: every entity round-trips through :meth:`Entity.to_dict` /
:func:`entity_from_dict` (plain JSON-able dicts) so the game state can
live in a room's ``state`` blob and be rehydrated per move.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

from .constants import (
    ASTEROID_HEALTH_MAX,
    BULLET_DO_BLANKING,
    BULLET_EDGE_SAMPLE_PTS,
    BULLET_LENGTH,
    CMAX,
    HEALTHBAR_EDGE_SAMPLE_PTS,
    HEALTHBAR_VERTEX_SAMPLE_PTS,
    LASER_POWER_DENOM,
    PARTICLE_LIFE_MAX,
    PARTICLE_LIFE_MIN,
    SHIP_EDGE_SAMPLE_PTS,
    SHIP_HEALTH_MAX,
    SHIP_VERTEX_SAMPLE_PTS,
    SQUARE_EDGE_SAMPLE_PTS,
    SQUARE_VERTEX_SAMPLE_PTS,
)

__all__ = [
    "Asteroid",
    "Bullet",
    "Entity",
    "HealthBar",
    "Particle",
    "Ship",
    "entity_from_dict",
]

# A laser point: (x, y, r, g, b) in galvo units / 16-bit colour.
Point = tuple[float, float, int, int, int]


def _rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Apply the laser power divisor, exactly as the original did.

    Note the original's asymmetry: red and blue are divided, green is
    passed through unless the divisor exceeds 4. Preserved, not fixed.
    """
    rr = 0 if not r else int(r / LASER_POWER_DENOM)
    gg = 0 if not g or LASER_POWER_DENOM > 4 else g
    bb = 0 if not b else int(b / LASER_POWER_DENOM)
    return rr, gg, bb


def _make_line(pt1: dict[str, float], pt2: dict[str, float],
               steps: int, r: int, g: int, b: int) -> list[Point]:
    """Interpolate a straight run of points between two vertices."""
    xdiff = pt1["x"] - pt2["x"]
    ydiff = pt1["y"] - pt2["y"]
    line: list[Point] = []
    for i in range(steps):
        j = float(i) / steps
        line.append((pt1["x"] - xdiff * j, pt1["y"] - ydiff * j, r, g, b))
    return line


def _rotate_translate(pts: list[dict[str, float]], theta: float,
                      x: float, y: float) -> list[dict[str, float]]:
    """Rotate vertices by ``theta`` around the origin, then move to (x, y)."""
    out = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for p in pts:
        px, py = p["x"], p["y"]
        out.append({"x": px * cos_t - py * sin_t + x,
                    "y": py * cos_t + px * sin_t + y})
    return out


class Entity:
    """Base class: position, velocity, colour, collision, point cache.

    Handles collision detection, x/y/velocity/colour, blanking marks, and
    the first/last point cache the PointStream algorithm needs.
    """

    # Static object counter.
    ID_COUNTER = 0

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = 0, g: int = 0, b: int = 0) -> None:
        self.x = x
        self.y = y
        self.xVel = 0.0
        self.yVel = 0.0
        self.rotation = 0.0

        self.r = r
        self.g = g
        self.b = b
        self.doBlanking = True
        self.skipDraw = False

        self.collides = True
        self.collisionRadius = 1000
        self.destroy = False

        # Cached first and last points (part of the PointStream algo).
        self.firstPt: Point = (0, 0, 0, 0, 0)
        self.lastPt: Point = (0, 0, 0, 0, 0)

        self.entityId = Entity.ID_COUNTER
        Entity.ID_COUNTER += 1

    def produce(self) -> Iterator[Point]:
        """Generate this entity's laser points for one drawing."""
        self.lastPt = (0, 0, 0, 0, 0)
        yield self.lastPt

    def cacheFirstPt(self) -> None:
        """Cache the first point produced, so the galvos can track to the
        next object without starting to draw."""
        for pt in self.produce():
            self.firstPt = pt
            break

    def checkCollide(self, other: Entity) -> bool:
        """Circle test: distance < sum of collision radii."""
        rad = other.collisionRadius + self.collisionRadius
        hyp = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        return hyp < rad

    # ── serialization ────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": type(self).__name__}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    def describe(self) -> str:
        return (f"{type(self).__name__}#{self.entityId} "
                f"({self.x:.0f},{self.y:.0f})")


def entity_from_dict(data: dict[str, Any]) -> Entity:
    """Rehydrate an entity serialized by :meth:`Entity.to_dict`."""
    kinds = {"Entity": Entity, "Ship": Ship, "Asteroid": Asteroid,
             "Bullet": Bullet, "Particle": Particle, "HealthBar": HealthBar}
    cls = kinds[data["kind"]]
    obj = cls.__new__(cls)
    state = dict(data)
    state.pop("kind", None)
    for key in ("firstPt", "lastPt"):
        if isinstance(state.get(key), list):
            state[key] = tuple(state[key])
    obj.__dict__.update(state)
    return obj


class Ship(Entity):
    """The player's ship — a five-vertex arrow that rotates by ``theta``."""

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = 0, g: int = 0, b: int = 0,
                 rgb: tuple[int, int, int] | None = None,
                 radius: float = 1200) -> None:
        if rgb:
            r, g, b = rgb
        super().__init__(x, y, r, g, b)
        self.radius = radius
        self.drawn = False
        self.theta = 0.0

    def produce(self) -> Iterator[Point]:
        ed = self.radius / 2
        pts = _rotate_translate(
            [{"x": ed, "y": ed}, {"x": -ed, "y": ed},
             {"x": -ed - ed * 2, "y": 0}, {"x": -ed, "y": -ed},
             {"x": ed, "y": -ed}],
            self.theta, self.x, self.y)
        r, g, b = _rgb(self.r, self.g, self.b)

        def line(a: int, c: int) -> list[Point]:
            return _make_line(pts[a], pts[c], SHIP_EDGE_SAMPLE_PTS, r, g, b)

        dwell = SHIP_VERTEX_SAMPLE_PTS
        half = int(round(dwell / 2.0))
        p: Point | None = None
        p = line(0, 1)[0]
        for _ in range(half):
            yield p  # type: ignore[misc]
        for p in line(0, 1):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(1, 2):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(2, 3):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(3, 4):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(4, 0):
            self.lastPt = p
            yield p
        for _ in range(half):
            self.lastPt = p  # type: ignore[assignment]
            yield p  # type: ignore[misc]
        self.drawn = True


class Asteroid(Entity):
    """A rotating square rock with health. Destroyed at 0."""

    HEALTH_MAX = ASTEROID_HEALTH_MAX

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = 0, g: int = 0, b: int = 0,
                 radius: float = 8200) -> None:
        super().__init__(x, y, r, g, b)
        self.drawn = False
        self.theta = 0.0
        self.thetaRate = 0.0
        self.radius = radius
        self.collisionRadius = radius
        self.health = Asteroid.HEALTH_MAX

    def subtract(self, health: int) -> None:
        self.health = max(self.health - health, 0)

    def produce(self) -> Iterator[Point]:
        ed = self.radius
        pts = _rotate_translate(
            [{"x": ed, "y": ed}, {"x": -ed, "y": ed},
             {"x": -ed, "y": -ed}, {"x": ed, "y": -ed}],
            self.theta, self.x, self.y)
        r, g, b = _rgb(self.r, self.g, self.b)

        def line(a: int, c: int) -> list[Point]:
            return _make_line(pts[a], pts[c], SQUARE_EDGE_SAMPLE_PTS, r, g, b)

        dwell = SQUARE_VERTEX_SAMPLE_PTS
        half = int(round(dwell / 2.0))
        p: Point | None = None
        p = line(0, 1)[0]
        for _ in range(half):
            yield p  # type: ignore[misc]
        for p in line(0, 1):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(1, 2):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(2, 3):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(3, 0):
            self.lastPt = p
            yield p
        for _ in range(half):
            self.lastPt = p  # type: ignore[assignment]
            yield p  # type: ignore[misc]
        self.drawn = True


class Bullet(Entity):
    """A shot: a short line along its firing angle."""

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = CMAX, g: int = CMAX, b: int = CMAX,
                 rgb: tuple[int, int, int] = (CMAX, CMAX, CMAX),
                 shotAngle: float = 0,
                 length: float = BULLET_LENGTH) -> None:
        super().__init__(x, y, rgb[0], rgb[1], rgb[2])
        self.drawn = False
        self.length = length
        self.shotAngle = shotAngle
        self.theta = shotAngle
        self.doBlanking = BULLET_DO_BLANKING

    def produce(self) -> Iterator[Point]:
        r, g, b = _rgb(self.r, self.g, self.b)
        pts = [{"x": 0, "y": 0},
               {"x": self.length * math.cos(self.shotAngle),
                "y": self.length * math.sin(self.shotAngle)}]
        for pt in pts:
            pt["x"] += self.x
            pt["y"] += self.y
        for p in _make_line(pts[0], pts[1], BULLET_EDGE_SAMPLE_PTS, r, g, b):
            self.lastPt = p
            yield p
        self.drawn = True


class Particle(Entity):
    """Explosion debris: a burst of points at one spot, short-lived."""

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = 0, g: int = 0, b: int = 0,
                 rng: random.Random | None = None) -> None:
        super().__init__(x, y, r, g, b)
        self.radius = 1
        self.drawn = False
        self.life = (rng or random).randint(PARTICLE_LIFE_MIN,
                                            PARTICLE_LIFE_MAX)
        self.xVel = 0
        self.yVel = 0

    def produce(self) -> Iterator[Point]:
        for _ in range(20):
            yield (self.x, self.y, self.r, self.g, self.b)


class HealthBar(Entity):
    """The player's health bar — green for health left, red for lost."""

    HEALTH_MAX = SHIP_HEALTH_MAX

    def __init__(self, x: float = 0, y: float = 0,
                 r: int = 0, g: int = 0, b: int = 0,
                 radius: float = 1200) -> None:
        super().__init__(x, y, r, g, b)
        self.radius = radius
        self.drawn = False
        self.skipDraw = False
        self.health = HealthBar.HEALTH_MAX

    def subtract(self, health: int) -> None:
        self.health = max(self.health - health, 0)

    def produce(self) -> Iterator[Point]:
        full = self.radius / 2
        h_perc = self.health / float(HealthBar.HEALTH_MAX)
        ih_perc = 1.0 - h_perc
        red = full * (1.0 - h_perc)
        y = 4000
        x_shift = 200
        pts = [
            {"x": full + x_shift, "y": y},
            {"x": -full * 2 + full * 3 * ih_perc + x_shift, "y": y},
            {"x": full - full * 3 + x_shift, "y": y},
            {"x": -full * 2 + x_shift, "y": y},
        ]
        for pt in pts:
            pt["x"] += self.x
            pt["y"] += self.y
        r, g, b = _rgb(self.r, self.g, self.b)

        def line(a: int, c: int, rr: int = r, gg: int = g,
                 bb: int = b) -> list[Point]:
            return _make_line(pts[a], pts[c], HEALTHBAR_EDGE_SAMPLE_PTS,
                              rr, gg, bb)

        dwell = HEALTHBAR_VERTEX_SAMPLE_PTS
        half = int(round(dwell / 2.0))
        p: Point | None = None
        # GREEN (health remaining)
        p = line(0, 1)[0]
        for _ in range(half):
            yield p  # type: ignore[misc]
        for p in line(0, 1):
            yield p
        for _ in range(dwell):
            yield p
        for p in line(1, 0):
            self.lastPt = p
            yield p
        for _ in range(half):
            self.lastPt = p  # type: ignore[assignment]
            yield p  # type: ignore[misc]
        # RED (health lost)
        if red:
            p = line(2, 3, rr=CMAX, gg=0, bb=0)[0]
            for _ in range(half):
                yield p  # type: ignore[misc]
            for p in line(2, 3, rr=CMAX, gg=0, bb=0):
                yield p
            for _ in range(dwell):
                yield p
            for p in line(3, 2, rr=CMAX, gg=0, bb=0):
                self.lastPt = p
                yield p
            for _ in range(half):
                self.lastPt = p  # type: ignore[assignment]
                yield p  # type: ignore[misc]
        self.drawn = True
