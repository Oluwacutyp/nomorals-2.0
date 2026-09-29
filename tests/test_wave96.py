"""Wave 96 — deeper games and a phone that can diagnose itself.

Games:
* the RPG campaign now runs ALL 16 scenes (the finale was unreachable
  at 10);
* the world town gets four seasons, a weighted event table (rain,
  caravans, festivals, droughts, plague, raids, merchants), four new
  buildings (well, wall, temple, granary) and population milestones;
* the battle arena gets focus (+50% next hit), fury (two 80% swings,
  cooldown) and 10% crits — and the house brain finally reads the
  arena's real stat keys (it used to defend against a ghost stat);
* the escape room is six locks deep.

Hugging Face:
* ``nm hf-doctor`` — one command on the phone, a numbered report:
  token, endpoint (the retired domain is auto-remapped), model,
  reachability, a live probe, and the first fallback model that
  actually answers;
* 410 Gone (retired serverless domain) is now a named error and a
  self-heal trigger, not a silent failure;
* the fallback chain carries catalog-verified models (live-verified
  against router.huggingface.co/v1/models on 2026-09-21).

Everything hermetic: in-memory db, no model, scripted moves, and the
HF tests mock the wire (no network in CI).
"""
from __future__ import annotations

import random
import types
import unittest

from nomorals.games import Player
from nomorals.games.ai import GameMind
from nomorals.games.engine import GameEngine
from nomorals.games.games.ambitious import (
    BUILDINGS,
    FARM_YIELD,
    SEASONS,
    BattleArenaGame,
    EscapeRoomGame,
    season_of,
)
from nomorals.games.games.medium import RPG_SCENES, RpgAdventureGame

try:  # package import under unittest discover / pytest
    from tests.test_wave85_games import ADA, drive, make_engine
except ImportError:  # direct run: fall back to the tests directory
    import sys
    from pathlib import Path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_wave85_games import (  # noqa: PLC0415
        ADA, drive, make_engine)


# ── 1. the world town: seasons, events, buildings ────────────────────────────

class WorldTownTests(unittest.TestCase):
    def test_season_wheel(self):
        self.assertEqual(season_of(1), "spring")
        self.assertEqual(season_of(10), "spring")
        self.assertEqual(season_of(11), "summer")
        self.assertEqual(season_of(30), "autumn")
        self.assertEqual(season_of(31), "winter")
        self.assertEqual(season_of(41), "spring")
        self.assertEqual(len(SEASONS), 4)
        # farm pays more in autumn than winter — the whole point
        self.assertGreater(FARM_YIELD["autumn"], FARM_YIELD["winter"])

    def test_new_buildings_are_buildable(self):
        for name in ("well", "wall", "temple", "granary"):
            spec = BUILDINGS[name]
            self.assertTrue(spec["effect"])
            self.assertGreaterEqual(spec["cost_g"], 0)

    def test_town_never_goes_negative_and_seasons_turn(self):
        """40 scripted actions, every seed: food/gold/tools stay >= 0,
        population stays >= 1, and a full year passes."""
        for seed in range(6):
            engine, _ = make_engine()
            room, _ = engine.start(f"world-{seed}", "world", ADA, kind="dm")
            room.seed = 1000 + seed  # deterministic stream
            room._rng_instance = None
            s = room.state
            seen_seasons = {season_of(s["day"])}
            moves = ["farm", "trade", "mine", "build house", "craft",
                     "rest", "build granary", "build well"]
            for mv in range(40):
                engine.move(f"world-{seed}", moves[mv % len(moves)], ADA)
                s = room.state
                self.assertGreaterEqual(s["food"], 0, f"food < 0 day {s['day']}")
                self.assertGreaterEqual(s["gold"], 0)
                self.assertGreaterEqual(s["tools"], 0)
                self.assertGreaterEqual(s["pop"], 1)
                seen_seasons.add(season_of(s["day"]))
            self.assertEqual(len(seen_seasons), 4,
                             "a year must pass through all four seasons")
            self.assertGreater(s["day"], 40)
            engine.shutdown()

    def test_winter_farm_pays_the_winter_rate(self):
        engine, _ = make_engine()
        room, _ = engine.start("world-winter", "world", ADA, kind="dm")
        room.state["day"] = 31  # winter
        out = engine.move("world-winter", "farm", ADA)
        joined = "\n".join(out)
        self.assertIn("winter fields", joined)
        self.assertIn(f"give +{FARM_YIELD['winter']} food", joined)
        engine.shutdown()


# ── 2. the battle arena: focus, fury, crits, a brain that reads stats ────────

class ArenaCombatTests(unittest.TestCase):
    def _mind(self, seed: int) -> GameMind:
        return GameMind(seed=seed)

    def _state(self) -> dict:
        game = BattleArenaGame()
        s = game.new_state(random.Random(1))
        return s

    def test_house_reads_real_stat_keys(self):
        """The bug: the brain read 'attack'/'defense' while the arena
        stores 'atk'/'def', so the house never defended."""
        mind = self._mind(1)
        me = {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
              "potions": 1, "fury_cd": 0}
        tank = {"hp": 50, "atk": 30, "def": 5}  # far stronger than us
        move = mind.combat_move(me, tank)
        self.assertEqual(move["action"], "defend")
        # old-style keys still work (external callers)
        move = mind.combat_move({"hp": 50, "max_hp": 50, "attack": 10,
                                 "defense": 5, "potions": 0},
                                {"hp": 50, "attack": 30, "defense": 5})
        self.assertEqual(move["action"], "defend")

    def test_focus_finishes_the_fight(self):
        mind = self._mind(1)
        me = {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
              "potions": 0, "fury_cd": 0}
        foe = {"hp": 12, "atk": 5, "def": 0}  # one focused hit kills
        move = mind.combat_move(me, foe)
        self.assertEqual(move["action"], "focus")
        me["focused"] = True
        move = mind.combat_move(me, foe)
        self.assertEqual(move["action"], "attack")

    def test_fury_respects_the_cooldown(self):
        mind = self._mind(1)
        me = {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
              "potions": 0, "fury_cd": 2}
        foe = {"hp": 50, "atk": 5, "def": 5}
        self.assertNotEqual(mind.combat_move(me, foe)["action"], "fury")
        me["fury_cd"] = 0
        self.assertEqual(mind.combat_move(me, foe)["action"], "fury")

    def test_focused_hit_never_underperforms(self):
        """Same seed, same rolls: a focused hit must do at least as
        much damage as the same swing unfocused (crit or no crit)."""
        game = BattleArenaGame()
        for seed in range(25):
            s1, s2 = self._state(), self._state()
            s2["you"]["focused"] = True
            r1 = types.SimpleNamespace(state=s1)
            r2 = types.SimpleNamespace(state=s2)
            h1 = game._hit(r1, "you", "house", self._mind(seed))
            h2 = game._hit(r2, "you", "house", self._mind(seed))
            d1 = s1["house"]["max_hp"] - s1["house"]["hp"]
            d2 = s2["house"]["max_hp"] - s2["house"]["hp"]
            self.assertGreaterEqual(d2, d1, f"seed {seed}: {h1} vs {h2}")
            if s2["you"].get("focused"):
                self.fail("focus must be consumed by the hit")

    def test_crits_happen(self):
        game = BattleArenaGame()
        seen_crit = False
        for seed in range(200):
            s = self._state()
            room = types.SimpleNamespace(state=s)
            msg = game._hit(room, "you", "house", self._mind(seed))
            if "CRIT" in msg:
                seen_crit = True
                break
        self.assertTrue(seen_crit, "no crit in 200 seeded swings")

    def test_fury_lands_two_swings_and_cools_down(self):
        engine, _ = make_engine()
        room, _ = engine.start("arena-chat", "arena", ADA, kind="dm")
        out = engine.move("arena-chat", "fury", ADA)
        text = "\n".join(out)
        self.assertGreaterEqual(text.count("lands"), 2,
                                "fury must land two swings")
        self.assertEqual(room.state["you"]["fury_cd"], 2)
        # immediate re-fury is refused
        out = engine.move("arena-chat", "fury", ADA)
        self.assertTrue(any("warming up" in m for m in out))
        engine.shutdown()

    def test_arena_finishes(self):
        engine, _ = make_engine()
        moves = ["attack", "focus", "attack", "fury", "defend", "potion"]
        i = {"n": 0}

        def resp(room, state):
            m = moves[i["n"] % len(moves)]
            i["n"] += 1
            return m
        live, _ = drive(engine, "arena2-chat", "arena", ADA,
                        responder=resp, max_moves=60)
        self.assertTrue(live is None or live.status == "finished")
        engine.shutdown()


# ── 3. the rpg: all sixteen scenes, the finale included ──────────────────────

class RpgCampaignTests(unittest.TestCase):
    def test_campaign_spans_all_sixteen_scenes(self):
        game = RpgAdventureGame()
        s = game.new_state(random.Random(1))
        self.assertEqual(s["scenes"], len(RPG_SCENES))
        self.assertEqual(s["scenes"], 16)

    def test_finale_is_reachable(self):
        engine, _ = make_engine()
        room, _ = engine.start("rpg-chat", "rpg", ADA, kind="dm")
        finale_text = RPG_SCENES[-1]["text"][:30]
        saw_finale = False

        def pick(room, state):
            """Play it safe: heal when low, else take the choice with
            the smallest HP delta for the current scene."""
            sh = state["sheets"][ADA.key]
            if sh["hp"] < 12 and sh["potions"] > 0:
                return "potion"
            choices = RPG_SCENES[state["scene"] % len(RPG_SCENES)]["choices"]
            # the SAFEST way in is the LARGEST (least negative) HP delta
            idx = max(range(len(choices)),
                      key=lambda i: choices[i][4])  # dhp
            return "123"[idx]

        for _ in range(24):
            if room.state.get("done"):
                break
            if room.state["sheets"][ADA.key]["out"]:
                break
            out = engine.move("rpg-chat", pick(room, room.state), ADA)
            if finale_text in "\n".join(out):
                saw_finale = True
        s = room.state
        self.assertFalse(s["sheets"][ADA.key]["out"],
                         "safe play must survive all 16 scenes")
        self.assertTrue(saw_finale, "the last scene never appeared")
        self.assertTrue(s["done"], "campaign must finish after 16 scenes")
        self.assertEqual(s["scene"], 16)
        engine.shutdown()


# ── 4. the escape room: six locks ────────────────────────────────────────────

class EscapeRoomTests(unittest.TestCase):
    def test_six_locks(self):
        game = EscapeRoomGame()
        s = game.new_state(random.Random(1))
        self.assertEqual(len(s["puzzles"]), 6)
        answers = {p["answer"] for p in s["puzzles"]}
        self.assertEqual(answers, {"26", "echo", "24", "south",
                                   "162", "kayak"})

    @staticmethod
    def _solo_escape(engine, chat):
        """Start an escape room with the house teammate silenced, so
        the table's fate depends only on the scripted player."""
        engine.games["escape"].ai_turn = lambda room, mind: []
        room, _ = engine.start(chat, "escape", ADA, kind="dm")
        return room

    def test_full_escape(self):
        engine, _ = make_engine()
        room = self._solo_escape(engine, "escape-chat")
        from nomorals.games.games.ambitious import PUZZLES as LOCKS
        for p in LOCKS:
            engine.move("escape-chat", f"answer {p['answer']}", ADA)
        self.assertTrue(room.state.get("done"), "six correct answers "
                                                "must open the door")
        engine.shutdown()

    def test_still_three_strikes(self):
        engine, _ = make_engine()
        room = self._solo_escape(engine, "escape2-chat")
        for _ in range(3):
            engine.move("escape2-chat", "answer definitely-wrong-xyz", ADA)
        self.assertTrue(room.state.get("done"),
                        "three strikes seals the room")
        engine.shutdown()


# ── 5. the hf provider: 410 handling, the doctor, the fallback chain ─────────

def _json_loads(body: str):
    import json as _json
    try:
        return _json.loads(body)
    except Exception:  # noqa: BLE001
        return {}


def _raw(ok: bool, status: int = 200, body: str = "{}",
         url: str = "https://x") -> types.SimpleNamespace:
    from nomorals.core.errors import NoMoralsError
    name = {400: "bad request", 401: "unauthorized", 403: "unauthorized",
            404: "not found", 410: "gone", 429: "too many requests"}.get(
                status, "error")

    def raise_for_status():
        if not ok:
            raise NoMoralsError(f"{status} {name} from {url}",
                                retryable=status in (429, 500, 502, 503))

    return types.SimpleNamespace(
        ok=ok, status_code=status, text=body,
        json=lambda: _json_loads(body), raise_for_status=raise_for_status,
    )


def _hf_provider(**over):
    from nomorals.llm.providers.hf_serverless import HFServerlessProvider
    kwargs = dict(token="hf_tok_abc123",
                  model="some/Model-8B",
                  base_url="https://router.huggingface.co/hf-inference")
    kwargs.update(over)
    return HFServerlessProvider(**kwargs)


class HfDoctorTests(unittest.TestCase):
    def test_doctor_reports_missing_token(self):
        from nomorals.llm.providers.hf_serverless import hf_doctor
        import nomorals.llm.providers.hf_serverless as hfmod
        p = _hf_provider(token="")
        # catalog + probes all fail offline; the report must not raise
        orig_catalog = hfmod.HFServerlessProvider.fetch_catalog
        hfmod.HFServerlessProvider.fetch_catalog = staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
        p.http.post_json = lambda url, payload: _raw(False, 0, "")
        try:
            report = hf_doctor(p)
        finally:
            hfmod.HFServerlessProvider.fetch_catalog = orig_catalog
        joined = "\n".join(report)
        self.assertIn("1. token: MISSING", joined)
        self.assertIn("NM_HF_TOKEN", joined)
        self.assertIn("VERDICT", joined)
        self.assertIn("NOT working", joined)

    def test_doctor_remaps_retired_domain(self):
        from nomorals.llm.providers.hf_serverless import hf_doctor
        import nomorals.llm.providers.hf_serverless as hfmod
        p = _hf_provider(base_url="https://api-inference.huggingface.co")
        orig_catalog = hfmod.HFServerlessProvider.fetch_catalog
        hfmod.HFServerlessProvider.fetch_catalog = staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
        p.http.post_json = lambda url, payload: _raw(False, 0, "")
        try:
            report = hf_doctor(p)
        finally:
            hfmod.HFServerlessProvider.fetch_catalog = orig_catalog
        self.assertNotIn("api-inference", p.base_url,
                         "retired domain must be remapped in place")
        self.assertIn("hf-inference", p.base_url)
        self.assertIn("RETIRED", "\n".join(report))
        self.assertIn("auto-remapped", "\n".join(report))

    def test_doctor_finds_working_fallback(self):
        from nomorals.llm.providers.hf_serverless import hf_doctor
        import nomorals.llm.providers.hf_serverless as hfmod
        p = _hf_provider(base_url="https://router.huggingface.co/v1")
        catalog = [
            {"id": "some/Model-8B", "providers": [{"status": "live"}]},
            {"id": "prism-ml/Ternary-Bonsai-27B-gguf",
             "providers": [{"status": "live"}]},
        ]
        orig_catalog = hfmod.HFServerlessProvider.fetch_catalog
        hfmod.HFServerlessProvider.fetch_catalog = staticmethod(
            lambda *a, **k: catalog)
        # configured model 400s, the fallback answers
        def post(url, payload):
            if payload["model"] == "prism-ml/Ternary-Bonsai-27B-gguf":
                return _raw(True, 200,
                            '{"choices": [{"message": {"content": "OK"}}]}')
            return _raw(False, 400, '{"error": "model not hosted"}')
        p.http.post_json = post
        try:
            report = hf_doctor(p)
        finally:
            hfmod.HFServerlessProvider.fetch_catalog = orig_catalog
        joined = "\n".join(report)
        self.assertIn("fallback found: prism-ml/Ternary-Bonsai-27B-gguf",
                      joined)
        self.assertIn("NM_HF_MODEL=prism-ml/Ternary-Bonsai-27B-gguf", joined)

    def test_410_on_retired_domain_is_named(self):
        from nomorals.core.errors import ProviderError
        p = _hf_provider(base_url="https://api-inference.huggingface.co")
        p.http.post_json = lambda url, payload: _raw(
            False, 410, "Gone: the Inference API is retired")
        try:
            p._post_checked(p.chat_url(), {"x": 1})
            self.fail("410 must raise")
        except ProviderError as exc:
            self.assertIn("RETIRED", str(exc))
            self.assertIn("NM_HF_BASE_URL", str(exc))
            self.assertFalse(exc.retryable)

    def test_chat_heals_a_410_with_remap(self):
        from nomorals.llm.base import Message, SamplingParams
        p = _hf_provider(base_url="https://api-inference.huggingface.co")
        import nomorals.llm.providers.hf_serverless as hfmod
        # catalog: the configured model is gone, a fallback is live
        catalog = [{"id": "zai-org/GLM-5.3-Flash",
                    "providers": [{"status": "live"}]}]
        orig_catalog = hfmod.HFServerlessProvider.fetch_catalog
        hfmod.HFServerlessProvider.fetch_catalog = staticmethod(
            lambda *a, **k: catalog)
        calls = []

        def post(url, payload):
            calls.append(url)
            if "GLM-5.3-Flash" in payload.get("model", ""):
                return _raw(True, 200,
                            '{"choices": [{"message": {"content": '
                            '"ok"}}], "usage": {}}')
            return _raw(False, 410, "Gone")
        p.http.post_json = post
        try:
            resp = p.chat([Message.user("hi")], SamplingParams(max_tokens=4))
        finally:
            hfmod.HFServerlessProvider.fetch_catalog = orig_catalog
        self.assertEqual(resp.text, "ok")
        self.assertIn("hf-inference", p.base_url,
                      "410 on the retired domain must remap the base url")
        self.assertFalse(resp.error)


class HfFallbackChainTests(unittest.TestCase):
    def test_chain_has_live_verified_models(self):
        from nomorals.llm.providers.hf_serverless import (
            ROUTER_FALLBACK_MODELS)
        # verified against the live /v1 catalog 2026-09-21
        for mid in ("prism-ml/Ternary-Bonsai-27B-gguf",
                    "zai-org/GLM-5.3-Flash",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "deepseek-ai/DeepSeek-V4.1-Flash"):
            self.assertIn(mid, ROUTER_FALLBACK_MODELS)
        # uncensored family still leads the chain
        self.assertTrue(ROUTER_FALLBACK_MODELS[0].lower()
                        .find("stheno") >= 0 or
                        "uncensored" in ROUTER_FALLBACK_MODELS[0].lower())


if __name__ == "__main__":
    unittest.main()
