"""Laser Asteroids — the 2012 Georgia Game Jam laser-projector game,
ported into Devon's games organ.

The game flies a ship through a field of rocks on a laser projector's
galvo field: move, aim, shoot, don't get hit. The original drove an
Ether Dream DAC through a GPL driver (carved out — see
:mod:`nomorals.games.laser.sinks`); this port plays headless, rendering
real laser point-stream frames, a text board, and PNG frames, and
refuses a live projector by name when none is attached.

Original: echelon/laser-asteroids (MIT, Copyright (c) 2012 Brandon
Thomas). The port keeps the original's field, tuning, shapes, spawn
patterns, and collision rules; it replaces the three threads with one
deterministic step, makes the score real, and splits big rocks.
"""
from __future__ import annotations

from .colors import BULLET_COLORS, COLOR_PINK
from .constants import (
    ASTEROID_MAX_RADIUS,
    ASTEROID_MIN_RADIUS,
    BOARD_COLS,
    BOARD_ROWS,
    CMAX,
    MAX_NUM_BULLETS,
    MAX_NUM_ENEMIES,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    SHIP_HEALTH_MAX,
)
from .entities import (
    Asteroid,
    Bullet,
    Entity,
    HealthBar,
    Particle,
    Ship,
    entity_from_dict,
)
from .game import COMMANDS, LaserGame, score_for_radius
from .sinks import (
    FrameSink,
    LaserError,
    LaserSink,
    NullSink,
    projector_sink,
    projector_unavailable,
)
from .stream import PointStream

__all__ = [
    "ASTEROID_MAX_RADIUS",
    "ASTEROID_MIN_RADIUS",
    "BOARD_COLS",
    "BOARD_ROWS",
    "BULLET_COLORS",
    "CMAX",
    "COLOR_PINK",
    "COMMANDS",
    "MAX_NUM_BULLETS",
    "MAX_NUM_ENEMIES",
    "MAX_X",
    "MAX_Y",
    "MIN_X",
    "MIN_Y",
    "SHIP_HEALTH_MAX",
    "Asteroid",
    "Bullet",
    "Entity",
    "FrameSink",
    "HealthBar",
    "LaserError",
    "LaserGame",
    "LaserSink",
    "NullSink",
    "Particle",
    "PointStream",
    "Ship",
    "entity_from_dict",
    "projector_sink",
    "projector_unavailable",
    "score_for_radius",
]
