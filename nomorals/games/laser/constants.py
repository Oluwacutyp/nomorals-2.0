"""Laser Asteroids constants — the galvo field and the game tuning.

Ported from ``globalvals.py`` in echelon/laser-asteroids (MIT, Copyright
(c) 2012 Brandon Thomas). The trailing GUI globals of that file (balls,
PLAYERS, BOUNCE_*, PADDLE_*) belonged to other laser experiments, not this
game, and were left behind. The values below are the game's own.
"""
from __future__ import annotations

import math

# ── the laser field ──────────────────────────────────────────────────────
# Positional boundaries of the galvos, in laser units. The original
# normally used 28000 each way and ran this game at a reduced field for
# presentation; the reduced field is the shipped one.
MAX_X = 13000
MIN_X = -20000
MAX_Y = 17000
MIN_Y = 0

# Max color value the hardware allows (16-bit colour space).
CMAX = 65535

# Laser power divisor: point colours are divided by this before they hit
# the stream. 1.0 = full power.
LASER_POWER_DENOM = 1.0

# ── game parameters ──────────────────────────────────────────────────────
MAX_NUM_ENEMIES = 3
MAX_NUM_BULLETS = 30

SHIP_IS_INVINCIBLE = False
SHIP_HEALTH_MAX = 100
SHIP_HEALTH_ASTEROID_HIT = 10
ASTEROID_HEALTH_MAX = 30
ASTEROID_HEALTH_BULLET_HIT = 10

# ── blanking and tracking (point-stream algorithm tweaks) ────────────────
BLANK_SAMPLE_PTS = 30
SHOW_TRACKING_PATH = False
TRACKING_SAMPLE_PTS = 12

# ── sizes ────────────────────────────────────────────────────────────────
SHIP_SIZE = 800
ASTEROID_MIN_RADIUS = 1500
ASTEROID_MAX_RADIUS = 3500

# Asteroids at or above this radius split into two children when destroyed
# (classic Asteroids behaviour the original lacked — the original simply
# destroyed them). Children are half the radius, half the health.
ASTEROID_SPLIT_MIN_RADIUS = 1800

# Score for destroying an asteroid, by size band. The original never
# incremented its score at all; the bands pay small rocks more.
SCORE_LARGE = 25    # radius >= 3000
SCORE_MEDIUM = 50   # radius >= 2000
SCORE_SMALL = 100   # radius < 2000

# ── particle effects ─────────────────────────────────────────────────────
PARTICLE_SPAWN_MIN = 3
PARTICLE_SPAWN_MAX = 5
PARTICLE_LIFE_MIN = 15
PARTICLE_LIFE_MAX = 30

# ── speeds and velocities ────────────────────────────────────────────────
ASTEROID_VEL_MAG_MAX = 100
ASTEROID_VEL_MAG_MIN = 50
ASTEROID_SPIN_VEL_MAG_MAX = math.pi / 70
ASTEROID_SPIN_VEL_MAG_MIN = math.pi / 400
BULLET_SPEED = 3000
PARTICLE_MAX_X_VEL = 500
PARTICLE_MIN_X_VEL = -500
PARTICLE_MAX_Y_VEL = 500
PARTICLE_MIN_Y_VEL = -500

# ── entity drawing specifics ─────────────────────────────────────────────
BULLET_DO_BLANKING = True
BULLET_EDGE_SAMPLE_PTS = 7
BULLET_LENGTH = 400
SHIP_EDGE_SAMPLE_PTS = 12
SHIP_VERTEX_SAMPLE_PTS = 2
SQUARE_EDGE_SAMPLE_PTS = 12
SQUARE_VERTEX_SAMPLE_PTS = 5

HEALTHBAR_EDGE_SAMPLE_PTS = 7
HEALTHBAR_VERTEX_SAMPLE_PTS = 10

# ── input ────────────────────────────────────────────────────────────────
# Direct translation speed per tick (the original's right-stick speed).
SIMPLE_TRANSLATION_SPD = 800
# Fire cooldown in ticks: the original waited 150 ms between shots at a
# 20 ms tick, i.e. ~8 ticks. Tick-based, so replays are deterministic.
FIRE_COOLDOWN_TICKS = 8
# Rotation step per `left` / `right` command (radians).
ROTATE_STEP = math.pi / 12

# ── rendering ────────────────────────────────────────────────────────────
# Text-board size for the chat/CLI render.
BOARD_COLS = 40
BOARD_ROWS = 20
# PNG frame size (pixels).
PNG_WIDTH = 640
PNG_HEIGHT = 480
