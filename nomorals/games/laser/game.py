"""Laser Asteroids — the headless game.

Ported from ``main.py`` (the ``game_thread`` + ``joystick_thread`` halves)
in echelon/laser-asteroids (MIT, Copyright (c) 2012 Brandon Thomas).
The original ran three threads (DAC stream, joystick input, game loop);
this port is one deterministic ``step()`` — the threads were the
original's own design flaw (its "Weird out of range error" hack, lists
mutated from two threads at once). Every move is a pure state transition
driven by the game's own ``random.Random``, so a seeded game replays
exactly.

Input replaces the PyGame joystick: ``up``/``down``/``left``/``right``
are the original's right-stick direct translation, ``aim <deg>`` is the
left-stick rotation (the nose points at ``deg`` — see :meth:`heading`),
``fire`` is the trigger, ``tick [n]`` advances the world.

Where the port beats the reference: the score is real (the original
never incremented it), and big asteroids split into two children
(classic Asteroids behaviour the original lacked).
"""
from __future__ import annotations

import math
import random
from typing import Any

from .colors import BULLET_COLORS, COLOR_PINK
from .constants import (
    ASTEROID_HEALTH_BULLET_HIT,
    ASTEROID_HEALTH_MAX,
    ASTEROID_MAX_RADIUS,
    ASTEROID_MIN_RADIUS,
    ASTEROID_SPIN_VEL_MAG_MAX,
    ASTEROID_SPIN_VEL_MAG_MIN,
    ASTEROID_SPLIT_MIN_RADIUS,
    ASTEROID_VEL_MAG_MAX,
    ASTEROID_VEL_MAG_MIN,
    BOARD_COLS,
    BOARD_ROWS,
    BULLET_SPEED,
    CMAX,
    FIRE_COOLDOWN_TICKS,
    MAX_NUM_BULLETS,
    MAX_NUM_ENEMIES,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    PARTICLE_MAX_X_VEL,
    PARTICLE_MAX_Y_VEL,
    PARTICLE_MIN_X_VEL,
    PARTICLE_MIN_Y_VEL,
    PARTICLE_SPAWN_MAX,
    PARTICLE_SPAWN_MIN,
    PNG_HEIGHT,
    PNG_WIDTH,
    ROTATE_STEP,
    SCORE_LARGE,
    SCORE_MEDIUM,
    SCORE_SMALL,
    SHIP_HEALTH_ASTEROID_HIT,
    SHIP_IS_INVINCIBLE,
    SHIP_SIZE,
    SIMPLE_TRANSLATION_SPD,
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
from .sinks import LaserError, LaserSink
from .stream import PointStream

__all__ = ["LaserGame", "LaserError"]

# Commands the game understands (the adapter and the CLI share this).
COMMANDS = ("up", "down", "left", "right", "aim", "fire", "tick",
            "rotate", "render", "png")


def score_for_radius(radius: float) -> int:
    """Points for destroying an asteroid of this size — small pays more."""
    if radius >= 3000:
        return SCORE_LARGE
    if radius >= 2000:
        return SCORE_MEDIUM
    return SCORE_SMALL


def _rng_to_json(rng: random.Random) -> list[Any]:
    """Serialize a Random's state to JSON-able lists."""
    version, internal, gauss = rng.getstate()

    def conv(value: Any) -> Any:
        if isinstance(value, tuple):
            return [conv(v) for v in value]
        return value

    return [version, conv(internal), gauss]


def _rng_from_json(data: list[Any]) -> random.Random:
    """Restore a Random from :func:`_rng_to_json` output."""
    rng = random.Random()

    def conv(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(conv(v) for v in value)
        return value

    rng.setstate((data[0], conv(data[1]), data[2]))
    return rng


class LaserGame:
    """One laser-asteroids session. Pure state, no I/O."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.ship = Ship(0, 0, rgb=COLOR_PINK, radius=SHIP_SIZE)
        self.healthbar = HealthBar(0, 0, r=0, g=CMAX, b=0,
                                   radius=SHIP_SIZE)
        self.ship.x = (MIN_X + MAX_X) / 2
        self.ship.y = (MAX_Y - MIN_Y) / 2
        self.healthbar.x = self.ship.x
        self.healthbar.y = self.ship.y - 2500
        self.bullets: list[Bullet] = []
        self.asteroids: list[Asteroid] = []
        self.particles: list[Particle] = []
        self.score = 0
        self.kills = 0
        self.shots_fired = 0
        self.tick = 0
        self.game_over = False
        self._last_fire_tick = -FIRE_COOLDOWN_TICKS
        self._color_index = 0

    # ── input ────────────────────────────────────────────────────────────
    @property
    def heading(self) -> float:
        """Where the ship's nose points, radians. The arrow's tip sits at
        ``theta + pi`` (the original's geometry), so heading is that."""
        return (self.ship.theta + math.pi) % (2 * math.pi)

    def aim(self, degrees: float) -> None:
        """Point the nose at ``degrees`` (0 = east, 90 = north)."""
        self.ship.theta = math.radians(degrees) - math.pi

    def rotate(self, direction: int) -> None:
        """Rotate the ship: +1 left (counter-clockwise), -1 right."""
        self.ship.theta += direction * ROTATE_STEP

    def translate(self, dx: int, dy: int) -> None:
        """Move the ship directly, clamped to the galvo field (the
        original's right-stick translation)."""
        x = self.ship.x + dx * SIMPLE_TRANSLATION_SPD
        y = self.ship.y + dy * SIMPLE_TRANSLATION_SPD
        self.ship.x = max(MIN_X, min(MAX_X, x))
        self.ship.y = max(MIN_Y, min(MAX_Y, y))

    def fire(self) -> bool:
        """Shoot if the cooldown has elapsed and the bullet cap allows.
        Returns True when a bullet left the ship."""
        if self.game_over:
            return False
        if self.tick - self._last_fire_tick < FIRE_COOLDOWN_TICKS:
            return False
        if len(self.bullets) >= MAX_NUM_BULLETS:
            return False
        color = BULLET_COLORS[self._color_index % len(BULLET_COLORS)]
        self._color_index += 1
        bullet = Bullet(self.ship.x, self.ship.y, rgb=color,
                        shotAngle=self.heading)
        self.bullets.append(bullet)
        self.shots_fired += 1
        self._last_fire_tick = self.tick
        return True

    # ── world ────────────────────────────────────────────────────────────
    @property
    def objects(self) -> list[Entity]:
        """Everything the point stream draws, in draw order."""
        return ([self.ship, self.healthbar] + self.bullets
                + self.asteroids + self.particles)

    def spawn_enemy(self) -> Asteroid:
        """Spawn one asteroid from a corner or an edge, as the original's
        eight spawn patterns did (coords kept exactly; the original's
        corner labels were backwards, the coords are what matter)."""
        spawn_type = self.rng.randint(0, 7)

        def vel() -> int:
            return self.rng.randint(ASTEROID_VEL_MAG_MIN,
                                    ASTEROID_VEL_MAG_MAX)
        if spawn_type == 0:
            x, y, x_vel, y_vel = MIN_X, MAX_Y, vel(), -vel()
        elif spawn_type == 1:
            x, y, x_vel, y_vel = MIN_X, MIN_Y, vel(), vel()
        elif spawn_type == 2:
            x, y, x_vel, y_vel = MAX_X, MIN_Y, -vel(), vel()
        elif spawn_type == 3:
            x, y, x_vel, y_vel = MAX_X, MAX_Y, -vel(), -vel()
        elif spawn_type == 4:
            x, y = self.rng.randint(MIN_X, MAX_X), MAX_Y
            x_vel = vel() * (1 if self.rng.randint(0, 1) else -1)
            y_vel = -vel()
        elif spawn_type == 5:
            x, y = MIN_X, self.rng.randint(MIN_Y, MAX_Y)
            x_vel = vel()
            y_vel = vel() * (1 if self.rng.randint(0, 1) else -1)
        elif spawn_type == 6:
            x, y = self.rng.randint(MIN_X, MAX_X), MIN_Y
            x_vel = vel() * (1 if self.rng.randint(0, 1) else -1)
            y_vel = vel()
        else:
            x, y = MAX_X, self.rng.randint(MIN_Y, MAX_Y)
            x_vel, y_vel = -vel(), vel() * (1 if self.rng.randint(0, 1)
                                            else -1)
        radius = self.rng.randint(ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS)
        enemy = Asteroid(x, y, r=CMAX, g=CMAX, b=0, radius=radius)
        enemy.xVel = x_vel
        enemy.yVel = y_vel
        rate = self.rng.uniform(ASTEROID_SPIN_VEL_MAG_MIN,
                                ASTEROID_SPIN_VEL_MAG_MAX)
        enemy.thetaRate = rate * (1 if self.rng.randint(0, 1) else -1)
        self.asteroids.append(enemy)
        return enemy

    def _spawn_particles(self, x: float, y: float) -> None:
        count = self.rng.randint(PARTICLE_SPAWN_MIN, PARTICLE_SPAWN_MAX)
        for _ in range(count):
            p = Particle(x, y, CMAX, CMAX, CMAX, rng=self.rng)
            p.xVel = self.rng.randint(PARTICLE_MIN_X_VEL,
                                      PARTICLE_MAX_X_VEL)
            p.yVel = self.rng.randint(PARTICLE_MIN_Y_VEL,
                                      PARTICLE_MAX_Y_VEL)
            self.particles.append(p)

    def _split(self, enemy: Asteroid) -> list[Asteroid]:
        """A big destroyed asteroid becomes two children at half radius
        and half health, flying apart."""
        children = []
        base = math.atan2(enemy.yVel, enemy.xVel)
        for sign in (+1, -1):
            angle = base + sign * math.pi / 3
            speed = max(50.0, math.hypot(enemy.xVel, enemy.yVel))
            child = Asteroid(enemy.x, enemy.y, r=CMAX, g=CMAX, b=0,
                             radius=enemy.radius / 2)
            # Half of max health: the parent is at 0 when it splits,
            # so half of its *current* health would be nothing.
            child.health = max(1, ASTEROID_HEALTH_MAX // 2)
            child.xVel = speed * math.cos(angle)
            child.yVel = speed * math.sin(angle)
            child.thetaRate = enemy.thetaRate * -sign
            children.append(child)
        return children

    def _in_field(self, obj: Entity) -> bool:
        return MIN_X <= obj.x <= MAX_X and MIN_Y <= obj.y <= MAX_Y

    def apply(self, command: str) -> str | None:
        """Apply one input command. Returns a note, or None when the
        command did nothing worth saying."""
        parts = (command or "").strip().lower().split()
        if not parts:
            return None
        verb = parts[0]
        if verb == "up":
            self.translate(0, 1)
        elif verb == "down":
            self.translate(0, -1)
        elif verb == "left":
            self.translate(-1, 0)
        elif verb == "right":
            self.translate(1, 0)
        elif verb == "aim":
            if len(parts) < 2:
                return "aim needs degrees — aim 90"
            try:
                self.aim(float(parts[1]))
            except ValueError:
                return f"aim needs a number, not {parts[1]!r}"
        elif verb == "fire":
            if not self.fire():
                return "can't fire yet — cooling down"
        elif verb == "rotate":
            if len(parts) < 2 or parts[1] not in ("left", "right"):
                return "rotate left | rotate right"
            self.rotate(+1 if parts[1] == "left" else -1)
        else:
            return (f"unknown move {verb!r} — up, down, left, right, "
                    "aim <deg>, fire, tick [n], render, png")
        return None

    def advance(self) -> list[str]:
        """One 20 ms tick of the world. Returns event lines."""
        events: list[str] = []
        if self.game_over:
            # The field keeps sparkling after the end, as the original did.
            if self.rng.randint(0, 20) == 0:
                self._spawn_particles(
                    self.rng.randint(MIN_X, MAX_X),
                    self.rng.randint(MIN_Y, MAX_Y))
            self._integrate()
            self.tick += 1
            return events

        # Spawn.
        if len(self.asteroids) < MAX_NUM_ENEMIES \
                and self.rng.randint(0, 5) == 0:
            self.spawn_enemy()

        # Bullet ↔ asteroid collisions.
        for bullet in list(self.bullets):
            for enemy in list(self.asteroids):
                if enemy.destroy or bullet.destroy:
                    continue
                if enemy.checkCollide(bullet):
                    bullet.destroy = True
                    enemy.subtract(ASTEROID_HEALTH_BULLET_HIT)
                    if enemy.health <= 0:
                        enemy.destroy = True
                        self.kills += 1
                        points = score_for_radius(enemy.radius)
                        self.score += points
                        self._spawn_particles(enemy.x, enemy.y)
                        events.append(
                            f"asteroid destroyed +{points}")
                        if enemy.radius >= ASTEROID_SPLIT_MIN_RADIUS:
                            kids = self._split(enemy)
                            self.asteroids.extend(kids)
                            events.append(
                                "it split into two smaller rocks")

        # Ship ↔ asteroid collisions.
        for enemy in list(self.asteroids):
            if enemy.destroy:
                continue
            if enemy.checkCollide(self.ship):
                enemy.destroy = True
                self._spawn_particles(enemy.x, enemy.y)
                if not SHIP_IS_INVINCIBLE:
                    self.healthbar.subtract(SHIP_HEALTH_ASTEROID_HIT)
                    events.append(
                        f"rock hit the ship — health "
                        f"{self.healthbar.health}")
                    if self.healthbar.health <= 0:
                        self.ship.destroy = True
                        self.healthbar.destroy = True
                        self._spawn_particles(self.ship.x, self.ship.y)
                        self.game_over = True
                        events.append(
                            f"game over — final score {self.score}")

        self._integrate()
        # The health bar rides above the ship.
        self.healthbar.x = self.ship.x
        self.healthbar.y = self.ship.y - 2500
        self.tick += 1
        return events

    def _integrate(self) -> None:
        """Move everything one tick; cull the out-of-bounds and the dead."""
        for bullet in self.bullets:
            bullet.x += BULLET_SPEED * math.cos(bullet.theta)
            bullet.y += BULLET_SPEED * math.sin(bullet.theta)
            if not self._in_field(bullet):
                bullet.destroy = True
        for enemy in self.asteroids:
            enemy.x += enemy.xVel
            enemy.y += enemy.yVel
            if not self._in_field(enemy):
                enemy.destroy = True
                continue
            enemy.theta += enemy.thetaRate
        for particle in self.particles:
            particle.x += particle.xVel
            particle.y += particle.yVel
            if not self._in_field(particle):
                particle.destroy = True
                continue
            particle.life -= 1
            if particle.life <= 0:
                particle.destroy = True
        self.bullets = [b for b in self.bullets if not b.destroy]
        self.asteroids = [e for e in self.asteroids if not e.destroy]
        self.particles = [p for p in self.particles if not p.destroy]

    def step(self, commands: list[str] | None = None) -> list[str]:
        """Apply input commands, then advance one tick."""
        notes: list[str] = []
        if self.game_over:
            return ["game over — start a new game to play again"]
        for command in commands or []:
            note = self.apply(command)
            if note:
                notes.append(note)
        return notes + self.advance()

    def tick_steps(self, count: int) -> list[str]:
        """Advance the world ``count`` ticks with no input."""
        events: list[str] = []
        for _ in range(max(0, count)):
            events.extend(self.advance())
            if self.game_over:
                break
        return events

    # ── rendering ────────────────────────────────────────────────────────
    def render_frame(self, sink: LaserSink | None = None) -> list[Any]:
        """One frame of laser points, written to ``sink`` when given.
        Destroyed objects (a dead ship, a dead health bar) leave the
        frame — the stream culls them, but the stream works on a copy,
        so the game filters them first."""
        frame = PointStream(
            [o for o in self.objects if not o.destroy]).frame()
        if sink is not None:
            sink.write(frame)
        return frame

    def render_text(self) -> str:
        """The field as a text board, for chat and the CLI."""
        grid = [["·" for _ in range(BOARD_COLS)]
                for _ in range(BOARD_ROWS)]

        def plot(x: float, y: float, glyph: str) -> None:
            col = int((x - MIN_X) / (MAX_X - MIN_X) * BOARD_COLS)
            row = int((y - MIN_Y) / (MAX_Y - MIN_Y) * BOARD_ROWS)
            col = max(0, min(BOARD_COLS - 1, col))
            row = max(0, min(BOARD_ROWS - 1, row))
            grid[BOARD_ROWS - 1 - row][col] = glyph

        for particle in self.particles:
            plot(particle.x, particle.y, "*")
        for bullet in self.bullets:
            plot(bullet.x, bullet.y, "-")
        for enemy in self.asteroids:
            plot(enemy.x, enemy.y,
                 "O" if enemy.radius >= 2500 else "o")
        if not self.ship.destroy:
            plot(self.ship.x, self.ship.y, "@")
        lines = [
            f"score: {self.score}   health: {self.healthbar.health}   "
            f"rocks: {len(self.asteroids)}   tick: {self.tick}",
        ]
        lines.extend(" ".join(row) for row in grid)
        if self.game_over:
            lines.append(f"game over — final score {self.score}")
        else:
            lines.append("move: up / down / left / right / aim <deg> / "
                         "fire / tick [n]")
        return "\n".join(lines)

    def render_png(self, path: str) -> str:
        """Draw one frame to a PNG file. Returns the path written."""
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise LaserError(
                "render_unavailable",
                "PIL is not installed, so no PNG frame can be drawn") \
                from exc
        frame = self.render_frame()
        image = Image.new("RGB", (PNG_WIDTH, PNG_HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for x, y, r, g, b in frame:
            px = int((x - MIN_X) / (MAX_X - MIN_X) * (PNG_WIDTH - 1))
            py = int((y - MIN_Y) / (MAX_Y - MIN_Y) * (PNG_HEIGHT - 1))
            py = PNG_HEIGHT - 1 - py
            draw.point((px, py),
                       fill=(r * 255 // CMAX, g * 255 // CMAX,
                             b * 255 // CMAX))
        image.save(path, format="PNG")
        return path

    def describe(self) -> str:
        """One-line status for /status."""
        return (f"score {self.score}, health {self.healthbar.health}, "
                f"{len(self.asteroids)} rocks, {len(self.bullets)} shots "
                f"in flight, tick {self.tick}")

    # ── serialization ────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "kills": self.kills,
            "shots_fired": self.shots_fired,
            "tick": self.tick,
            "game_over": self.game_over,
            "last_fire_tick": self._last_fire_tick,
            "color_index": self._color_index,
            "ship": self.ship.to_dict(),
            "healthbar": self.healthbar.to_dict(),
            "bullets": [b.to_dict() for b in self.bullets],
            "asteroids": [e.to_dict() for e in self.asteroids],
            "particles": [p.to_dict() for p in self.particles],
            "rng": _rng_to_json(self.rng),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaserGame:
        game = cls.__new__(cls)
        game.rng = _rng_from_json(data["rng"])
        game.ship = entity_from_dict(data["ship"])
        game.healthbar = entity_from_dict(data["healthbar"])
        game.bullets = [entity_from_dict(b) for b in data["bullets"]]
        game.asteroids = [entity_from_dict(e) for e in data["asteroids"]]
        game.particles = [entity_from_dict(p) for p in data["particles"]]
        game.score = int(data["score"])
        game.kills = int(data["kills"])
        game.shots_fired = int(data["shots_fired"])
        game.tick = int(data["tick"])
        game.game_over = bool(data["game_over"])
        game._last_fire_tick = int(data["last_fire_tick"])
        game._color_index = int(data["color_index"])
        # Keep entity ids unique across the rehydrated game.
        ids = [e.entityId for e in game.objects]
        if ids:
            Entity.ID_COUNTER = max(Entity.ID_COUNTER, max(ids) + 1)
        return game
