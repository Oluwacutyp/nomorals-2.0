"""§3C.5 laser-asteroids port — the 2012 Georgia Game Jam game in Devon.

Covers the organ (entities, point stream, sinks, game loop), the engine
adapter, the registry wiring, and the CLI surface. No laser projector
exists here or on EC2, so the projector path is tested as a named
refusal, and the game is proven headless: real point-stream frames, a
real text board, a real PNG.
"""
from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

from nomorals.agents.coremind import (
    COMMAND_STARTABLE,
    GAME_ALIASES,
    _game_intent,
)
from nomorals.games.engine import GameEngine
from nomorals.games.games.arcade import ARCADE_GAMES
from nomorals.games.laser import (
    Asteroid,
    Bullet,
    Entity,
    FrameSink,
    HealthBar,
    LaserError,
    LaserGame,
    Particle,
    PointStream,
    Ship,
    projector_sink,
    score_for_radius,
)
from nomorals.games.players import Player
from nomorals.social.chat.control import (
    GAME_COMMANDS,
    PUBLIC_GAME_COMMANDS,
    parse_control,
)
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(Ctx(db), send=lambda c, t: sent.append((c, t)))
    return engine, db, sent


ADA = Player.from_sender("telegram", "456", "Ada")


def _fresh_game(seed: int = 1) -> LaserGame:
    Entity.ID_COUNTER = 0
    return LaserGame(random.Random(seed))


# ── entities ─────────────────────────────────────────────────────────────

class EntityTests(unittest.TestCase):
    def test_collide_is_circle_distance(self):
        a = Asteroid(0, 0, radius=1000)
        b = Asteroid(1500, 0, radius=1000)
        self.assertTrue(a.checkCollide(b))   # 1500 < 1000 + 1000
        c = Asteroid(2500, 0, radius=1000)
        self.assertFalse(a.checkCollide(c))  # 2500 > 2000

    def test_ship_produces_70_points_and_rotates(self):
        ship = Ship(0, 0, rgb=(100, 100, 100), radius=800)
        pts = list(ship.produce())
        self.assertEqual(len(pts), 70)
        for x, y, _r, _g, _b in pts:
            self.assertLessEqual(abs(x), 1200)
            self.assertLessEqual(abs(y), 1200)
        ship.theta = math.pi / 2
        rotated = list(ship.produce())
        self.assertEqual(len(rotated), 70)
        self.assertNotEqual([p[:2] for p in pts],
                            [p[:2] for p in rotated])

    def test_asteroid_health_clamps_at_zero(self):
        rock = Asteroid(0, 0, radius=2000)
        self.assertEqual(rock.health, 30)
        rock.subtract(10)
        self.assertEqual(rock.health, 20)
        rock.subtract(99)
        self.assertEqual(rock.health, 0)

    def test_asteroid_produces_67_points(self):
        self.assertEqual(len(list(Asteroid(0, 0, radius=2000).produce())),
                         67)

    def test_bullet_is_a_400_line_along_its_angle(self):
        bullet = Bullet(100, 200, shotAngle=0)
        pts = list(bullet.produce())
        self.assertEqual(len(pts), 7)
        xs = [p[0] for p in pts]
        self.assertAlmostEqual(min(xs), 100)
        self.assertAlmostEqual(max(xs), 100 + 400, delta=60)
        self.assertTrue(all(p[1] == 200 for p in pts))

    def test_particle_burst_is_20_points_with_bounded_life(self):
        p = Particle(0, 0, 65535, 65535, 65535, rng=random.Random(7))
        self.assertEqual(len(list(p.produce())), 20)
        self.assertGreaterEqual(p.life, 15)
        self.assertLessEqual(p.life, 30)

    def test_healthbar_shrinks_with_damage(self):
        full = HealthBar(0, 0, radius=1200)
        self.assertEqual(len(list(full.produce())), 34)
        empty = HealthBar(0, 0, radius=1200)
        empty.subtract(100)
        self.assertEqual(len(list(empty.produce())), 68)

    def test_entity_round_trips_through_dict(self):
        ship = Ship(10, 20, rgb=(1, 2, 3), radius=800)
        ship.theta = 0.5
        from nomorals.games.laser import entity_from_dict
        back = entity_from_dict(ship.to_dict())
        self.assertIsInstance(back, Ship)
        self.assertEqual((back.x, back.y), (10, 20))
        self.assertEqual(back.theta, 0.5)
        self.assertEqual(back.entityId, ship.entityId)


# ── point stream + sinks ─────────────────────────────────────────────────

class StreamTests(unittest.TestCase):
    def test_frame_counts_blanking_tracking_and_shapes(self):
        game = _fresh_game()
        frame = PointStream(game.objects).frame()
        # ship: 30 blank-in + 1 first + 70 shape + 30 blank-out + 12 track
        # health bar: 30 + 1 + 34 + 30 + 12 → 250 total, 144 dark.
        self.assertEqual(len(frame), 250)
        dark = [p for p in frame if p[2] == 0 and p[3] == 0 and p[4] == 0]
        self.assertEqual(len(dark), 144)

    def test_stream_culls_destroyed_objects(self):
        ship = Ship(0, 0, rgb=(1, 1, 1))
        rock = Asteroid(5000, 5000, radius=1000)
        rock.destroy = True
        stream = PointStream([ship, rock])
        stream.frame()
        self.assertEqual(stream.objects, [ship])

    def test_read_pulls_n_points(self):
        stream = PointStream([Ship(0, 0, rgb=(1, 1, 1))])
        self.assertEqual(len(stream.read(50)), 50)

    def test_projector_sink_refuses_by_name(self):
        with self.assertRaises(LaserError) as ctx:
            projector_sink()
        self.assertEqual(ctx.exception.code, "projector_unavailable")

    def test_frame_sink_keeps_frames(self):
        game = _fresh_game()
        sink = FrameSink()
        frame = game.render_frame(sink)
        self.assertEqual(sink.frame_count, 1)
        self.assertEqual(sink.latest, frame)
        self.assertGreater(sink.points_written, 0)


# ── the game loop ────────────────────────────────────────────────────────

class GameLoopTests(unittest.TestCase):
    def test_spawn_uses_all_eight_patterns_in_field(self):
        game = _fresh_game(seed=42)
        seen = set()
        for _ in range(300):
            rock = game.spawn_enemy()
            self.assertGreaterEqual(rock.x, -20000)
            self.assertLessEqual(rock.x, 13000)
            self.assertGreaterEqual(rock.y, 0)
            self.assertLessEqual(rock.y, 17000)
            seen.add((rock.x, rock.y))
        self.assertGreater(len(seen), 50)

    def test_fire_cooldown_is_eight_ticks(self):
        game = _fresh_game()
        self.assertTrue(game.fire())
        self.assertFalse(game.fire())
        game.tick_steps(7)
        self.assertFalse(game.fire())
        game.tick_steps(1)
        self.assertTrue(game.fire())

    def test_bullet_cap_is_thirty(self):
        game = _fresh_game()
        game._last_fire_tick = -100
        for _ in range(40):
            game._last_fire_tick = game.tick - 8
            game.fire()
        self.assertEqual(len(game.bullets), 30)

    def _aimed_shot(self, game: LaserGame, radius: int,
                    health: int = 30) -> Asteroid:
        """Park a rock dead ahead of the ship's nose, motionless, far
        enough out that the bullet needs a full tick of travel before
        the collision check sees it (collisions run before movement)."""
        game.aim(0)  # nose east
        rock = Asteroid(game.ship.x + 5000, game.ship.y,
                        r=65535, g=65535, b=0, radius=radius)
        rock.health = health
        rock.xVel = 0
        rock.yVel = 0
        game.asteroids.append(rock)
        return rock

    def test_bullet_hit_damages_then_destroys_and_scores(self):
        game = _fresh_game()
        rock = self._aimed_shot(game, radius=1500)
        game.fire()
        game.tick_steps(2)  # tick 1 flies the bullet on; tick 2 hits
        self.assertEqual(rock.health, 20)
        self.assertEqual(game.score, 0)
        for _ in range(2):
            game._last_fire_tick = game.tick - 8
            game.fire()
            game.tick_steps(2)
        self.assertEqual(rock.health, 0)
        self.assertNotIn(rock, game.asteroids)
        self.assertEqual(game.score, 100)  # small rock pays 100
        self.assertEqual(game.kills, 1)
        self.assertGreater(len(game.particles), 0)

    def test_big_rock_splits_into_two_children(self):
        game = _fresh_game()
        self._aimed_shot(game, radius=3000)
        for _ in range(3):
            game._last_fire_tick = game.tick - 8
            game.fire()
            game.tick_steps(2)
        self.assertEqual(game.score, 25)  # large rock pays 25
        # Children carry half health; ambient spawns carry full health.
        kids = [e for e in game.asteroids if e.health == 15]
        self.assertEqual(len(kids), 2)
        self.assertTrue(all(k.radius == 1500 for k in kids))

    def test_ship_hit_costs_health_then_ends_the_game(self):
        game = _fresh_game()
        for hit in range(10):
            rock = Asteroid(game.ship.x, game.ship.y, radius=1500)
            rock.xVel = 0
            rock.yVel = 0
            game.asteroids.append(rock)
            events = game.step([])
            self.assertEqual(game.healthbar.health, 100 - 10 * (hit + 1))
            if hit < 9:
                self.assertFalse(game.game_over)
            else:
                self.assertTrue(game.game_over)
                self.assertIn("game over", " ".join(events))
        self.assertEqual(game.step([]),
                         ["game over — start a new game to play again"])

    def test_out_of_bounds_objects_are_culled(self):
        game = _fresh_game()
        bullet = Bullet(12000, 8500, shotAngle=0)
        game.bullets.append(bullet)
        game.tick_steps(1)  # 12000 + 3000 > MAX_X 13000
        self.assertEqual(game.bullets, [])

    def test_translate_clamps_to_the_field(self):
        game = _fresh_game()
        game.ship.x = 13000
        game.translate(1, 0)
        self.assertEqual(game.ship.x, 13000)
        game.ship.y = 0
        game.translate(0, -1)
        self.assertEqual(game.ship.y, 0)

    def test_state_round_trips_and_replays_deterministically(self):
        moves = [["fire"], ["up"], ["tick"], ["fire"], ["tick"]]

        def _play() -> dict:
            game = _fresh_game(seed=9)
            for command in moves:
                if command == ["tick"]:
                    game.tick_steps(5)
                else:
                    game.step(command)
            return game.to_dict()

        first = _play()
        second = _play()
        self.assertEqual(first, second)
        restored = LaserGame.from_dict(first)
        self.assertEqual(restored.to_dict(), first)

    def test_score_bands(self):
        self.assertEqual(score_for_radius(3500), 25)
        self.assertEqual(score_for_radius(2500), 50)
        self.assertEqual(score_for_radius(1500), 100)

    def test_render_text_shows_score_health_and_board(self):
        game = _fresh_game()
        text = game.render_text()
        self.assertIn("score: 0", text)
        self.assertIn("health: 100", text)
        self.assertIn("@", text)  # the ship

    def test_render_png_writes_a_real_png(self):
        import os
        import tempfile
        game = _fresh_game()
        game.tick_steps(30)
        # TemporaryDirectory honours TMPDIR, which the prove script
        # points under the verify dir — never /tmp.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "laser-test-frame.png")
            out = game.render_png(path)
            with open(out, "rb") as fh:
                magic = fh.read(8)
            self.assertEqual(magic, b"\x89PNG\r\n\x1a\n")
            self.assertGreater(os.path.getsize(out), 100)


# ── the engine adapter ───────────────────────────────────────────────────

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _start(self, chat: str = "t:g"):
        room, _msgs = self.engine.start(chat, "laser", ADA, kind="dm")
        return room

    def test_laser_is_registered_and_listed(self):
        self.assertIn("laser", self.engine.games)
        self.assertEqual(len(self.engine.games), 45)
        self.assertIn("/game laser", self.engine.list_games())
        self.assertIn("laser", [g.name for g in ARCADE_GAMES])

    def test_start_renders_the_board(self):
        room = self._start()
        self.assertEqual(room.game, "laser")
        self.assertEqual(room.state["score"], 0)
        self.assertEqual(room.state["healthbar"]["health"], 100)

    def test_moves_drive_the_game(self):
        room = self._start()
        out = self.engine.move("t:g", "fire", ADA)
        self.assertTrue(any("score:" in line for line in out))
        out = self.engine.move("t:g", "tick 30", ADA)
        self.assertGreaterEqual(room.state["tick"], 30)
        # The board comes back after every move.
        self.assertTrue(any("score:" in line for line in out))

    def test_unknown_move_gets_the_help_line(self):
        self._start()
        out = self.engine.move("t:g", "dance", ADA)
        self.assertTrue(any("move:" in line for line in out))

    def test_casual_chat_is_not_a_move(self):
        game = self.engine.games["laser"]
        self.assertFalse(game.is_move_text("fire that up"))
        self.assertFalse(game.is_move_text("hello there"))
        self.assertFalse(game.is_move_text("aim abc"))
        self.assertFalse(game.is_move_text("tick abc"))
        self.assertFalse(game.is_move_text("rotate up"))
        self.assertTrue(game.is_move_text("fire"))
        self.assertTrue(game.is_move_text("aim 90"))
        self.assertTrue(game.is_move_text("tick 10"))
        self.assertTrue(game.is_move_text("rotate left"))

    def test_score_and_status(self):
        room = self._start()
        game = self.engine.games["laser"]
        self.assertEqual(game.score(room, ADA), 0)
        self.assertIn("score 0", game.describe_state(room))
        self.assertFalse(game.is_over(room))

    def test_game_over_finishes_the_room(self):
        room = self._start()
        room.state["healthbar"]["health"] = 10
        game = self.engine.games["laser"]
        # Park a rock on the ship through the organ and step once.
        from nomorals.games.laser import LaserGame as _LG
        live = _LG.from_dict(room.state)
        rock = Asteroid(live.ship.x, live.ship.y, radius=1500)
        rock.xVel = 0
        rock.yVel = 0
        live.asteroids.append(rock)
        live.step([])
        room.state = {**room.state, **live.to_dict()}
        self.assertTrue(game.is_over(room))

    def test_direct_command_parses(self):
        cmd = parse_control("/laser")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "laser")


# ── the CLI surface ────────────────────────────────────────────────────

class CliTests(unittest.TestCase):
    def _args(self, **kw: object):
        args = SimpleNamespace(
            game_action="", game_name="", game_move=[], png="",
            json=False)
        for key, value in kw.items():
            setattr(args, key, value)
        return args

    def test_parser_builds_the_game_command(self):
        from nomorals.cmdline.parser import _parser
        parser = _parser()
        args = parser.parse_args(["game", "start", "laser"])
        self.assertEqual(args.game_action, "start")
        self.assertEqual(args.game_name, "laser")
        args = parser.parse_args(["game", "move", "fire"])
        self.assertEqual(args.game_move, ["fire"])
        args = parser.parse_args(["game", "render", "--png", "f.png"])
        self.assertEqual(args.png, "f.png")

    def test_dispatch_registers_game(self):
        from nomorals.cmdline.dispatch import _COMMANDS
        self.assertIn("game", _COMMANDS)

    def test_cli_list_start_move_state_quit(self):
        from nomorals.cmdline.commands.game import _cmd_game
        db = Database(":memory:")
        db.migrate()
        ctx = Ctx(db)
        self.assertEqual(
            _cmd_game(self._args(game_action="list"), ctx), 0)
        self.assertEqual(
            _cmd_game(self._args(game_action="start",
                                 game_name="laser"), ctx), 0)
        self.assertEqual(
            _cmd_game(self._args(game_action="move",
                                 game_move=["fire"]), ctx), 0)
        self.assertEqual(
            _cmd_game(self._args(game_action="state"), ctx), 0)
        self.assertEqual(
            _cmd_game(self._args(game_action="render"), ctx), 0)
        self.assertEqual(
            _cmd_game(self._args(game_action="quit"), ctx), 0)
        # After quit there is no live game.
        self.assertEqual(
            _cmd_game(self._args(game_action="state"), ctx), 1)

    def test_cli_move_without_a_game_fails(self):
        from nomorals.cmdline.commands.game import _cmd_game
        db = Database(":memory:")
        db.migrate()
        self.assertEqual(
            _cmd_game(self._args(game_action="move",
                                 game_move=["fire"]), Ctx(db)), 1)

    def test_cli_start_unknown_game_fails(self):
        from nomorals.cmdline.commands.game import _cmd_game
        db = Database(":memory:")
        db.migrate()
        self.assertEqual(
            _cmd_game(self._args(game_action="start",
                                 game_name="nope"), Ctx(db)), 1)


# ── wiring: NL alias + direct command ────────────────────────────────────────

class WiringTests(unittest.TestCase):
    """Laser must be startable the way every other engine game is: by
    natural language in the owner's DM and by a direct /<name> command
    outside it (test_games_f2 gates both structurally)."""

    def test_nl_alias_launches_laser(self):
        for text in ("let's play laser", "play laser asteroids with me"):
            with self.subTest(text=text):
                intent = _game_intent(text, None)
                self.assertIsNotNone(intent)
                self.assertEqual(
                    (intent.kind, intent.action, intent.target),
                    ("game", "start", "laser"))

    def test_laser_in_alias_command_and_startable_lists(self):
        self.assertIn("laser", set(GAME_ALIASES.values()))
        self.assertIn("laser", GAME_COMMANDS)
        self.assertIn("laser", COMMAND_STARTABLE)

    def test_direct_laser_command_parses(self):
        parsed = parse_control("/laser")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "laser")
        parsed = parse_control("/game laser")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "game")

    def test_laser_not_on_public_menu(self):
        # Deliberate: PUBLIC_GAME_COMMANDS is a 32-command reconstruction
        # pinned by test_public_bot_gating, awaiting verification against
        # the BotFather list already set — laser stays off it until that
        # list is verified. /game laser still reaches it for everyone.
        self.assertNotIn("laser", PUBLIC_GAME_COMMANDS)
