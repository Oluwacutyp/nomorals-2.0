"""Math Class is playable: /math wired end to end.

Regression for the orphan bug: MathCommandHandler existed but /math was
not in CONTROL_COMMANDS and MathClass was never wired into the /game
engine, so /math fell through to the brain in every chat. These tests
cover registration, the handler subcommands, persistence across
handler instances (save/load round-trip), the runtime handler, and the
gating (owner DM works, outsider group works, outsider DM gets
conversation only).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.games.math_commands import MathCommandHandler
from nomorals.games.math_persist import (
    dump_state, load_math_state, load_state, save_math_state)
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS, CONTROL_COMMANDS, GAME_COMMANDS, LIST_GROUPS,
    LIST_ONELINERS, _HELP_GROUPS, caller_may_run, parse_control)


def _msg(text: str, *, chat_key: str = "telegram:999",
         owner: bool = False, kind: str = ChatKind.DM) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=kind),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
    )


class _RecordingGateway:
    adapters: dict = {}
    sent: list[tuple[str, str]]

    def __init__(self) -> None:
        self.sent = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


def _correct_answer(handler: MathCommandHandler, player: str) -> str:
    """Server-side answer for the player's active problem (test only)."""
    pid = handler._active[player]
    return handler.math.store["active"][pid].answer_text


class RegistrationTest(unittest.TestCase):
    def test_math_registered_everywhere(self) -> None:
        self.assertIn("math", CONTROL_COMMANDS)
        self.assertIn("math", COMMAND_DETAILS)
        self.assertIn("math", LIST_ONELINERS)
        self.assertIn("math", GAME_COMMANDS)
        for _, names in _HELP_GROUPS:
            if "game" in names:
                self.assertIn("math", names)
        for _, names in LIST_GROUPS:
            if "game" in names:
                self.assertIn("math", names)

    def test_parse_control_math(self) -> None:
        cmd = parse_control("/math answer 42")
        assert cmd is not None
        self.assertEqual(cmd.kind, "math")

    def test_caller_may_run_math_is_game(self) -> None:
        # Games are social: non-owners may run them (the DM gate lives
        # in the runtime, not in caller_may_run).
        self.assertTrue(caller_may_run("math", True))
        self.assertTrue(caller_may_run("math", False))


class HandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = MathCommandHandler()
        self.player = "telegram:999"

    def test_bare_math_starts_problem(self) -> None:
        out = self.handler.handle("/math", self.player)
        self.assertTrue(out["ok"])
        self.assertEqual(out["kind"], "problem")
        self.assertIn("question", out["problem"])
        self.assertNotIn("answer", out["problem"])

    def test_tier_start(self) -> None:
        out = self.handler.handle("/math easy", self.player)
        self.assertTrue(out["ok"])
        self.assertEqual(out["problem"]["tier"], "easy")

    def test_answer_correct_pays_when_verified(self) -> None:
        self.handler.economy.verify_member(self.player)
        self.handler.handle("/math", self.player)
        out = self.handler.handle(
            f"/math answer {_correct_answer(self.handler, self.player)}",
            self.player)
        self.assertTrue(out["ok"])
        self.assertTrue(out["correct"])
        self.assertTrue(out["reward"]["paid"])
        self.assertGreater(out["reward"]["coins"], 0)

    def test_answer_correct_blocked_when_unverified(self) -> None:
        self.handler.handle("/math", self.player)
        out = self.handler.handle(
            f"/math answer {_correct_answer(self.handler, self.player)}",
            self.player)
        self.assertTrue(out["correct"])
        self.assertFalse(out["reward"]["paid"])

    def test_answer_wrong(self) -> None:
        self.handler.handle("/math", self.player)
        out = self.handler.handle("/math answer nope-not-a-number",
                                  self.player)
        self.assertTrue(out["ok"])
        self.assertFalse(out["correct"])
        self.assertIn("solution", out)

    def test_answer_without_active(self) -> None:
        out = self.handler.handle("/math answer 42", self.player)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "no_active_problem")

    def test_leaderboard_streak_balance(self) -> None:
        self.handler.economy.verify_member(self.player)
        self.handler.handle("/math", self.player)
        self.handler.handle(
            f"/math answer {_correct_answer(self.handler, self.player)}",
            self.player)
        leaders = self.handler.handle("/math leaderboard", self.player)
        self.assertTrue(leaders["ok"])
        self.assertTrue(any(r["player"] == self.player
                            for r in leaders["leaders"]))
        streak = self.handler.handle("/math streak", self.player)
        self.assertTrue(streak["ok"])
        self.assertEqual(streak["streak"], 1)
        balance = self.handler.handle("/math balance", self.player)
        self.assertTrue(balance["ok"])
        self.assertGreater(balance["balance"]["coins"], 0)

    def test_daily(self) -> None:
        out = self.handler.handle("/math daily", self.player)
        self.assertTrue(out["ok"])
        self.assertTrue(out.get("daily"))
        self.assertIn("question", out["problem"])

    def test_set_advances_through_problems(self) -> None:
        out = self.handler.handle("/math set", self.player)
        self.assertTrue(out["ok"])
        self.assertEqual(out["kind"], "timed_set")
        self.assertEqual(len(out["problems"]), 10)
        self.assertEqual(out["set_progress"], "1/10")
        first = self.handler.handle(
            f"/math answer {_correct_answer(self.handler, self.player)}",
            self.player)
        self.assertTrue(first["ok"])
        self.assertEqual(first["set_progress"], "2/10")
        self.assertIn("next_problem", first)
        for _ in range(9):
            res = self.handler.handle(
                f"/math answer {_correct_answer(self.handler, self.player)}",
                self.player)
            self.assertTrue(res["ok"])
        self.assertTrue(res.get("set_complete"))

    def test_duel_and_duel_answer(self) -> None:
        out = self.handler.handle("/math duel @ada", self.player)
        self.assertTrue(out["ok"])
        self.assertEqual(out["kind"], "duel")
        self.assertEqual(len(out["problems"]), 5)
        duel_id = out["duel_id"]
        # Round 0: answer from the duel's stored problem (test only).
        duel = self.handler.math.store["duels"][duel_id]
        ans = duel.problems[0].answer_text
        res = self.handler.handle(
            f"/math duel_answer {duel_id} 0 {ans}", self.player)
        self.assertTrue(res["ok"])
        self.assertTrue(res["correct"])

    def test_teach_and_upvote(self) -> None:
        self.handler.economy.verify_member(self.player)
        self.handler.economy.verify_member("telegram:ada")
        self.handler.handle("/math", self.player)
        pid = self.handler._active[self.player]
        taught = self.handler.handle(
            f"/math teach {pid} Because the fractions cancel.", self.player)
        self.assertTrue(taught["ok"])
        expl_id = taught["explanation_id"]
        voted = self.handler.handle(f"/math upvote {expl_id}", "telegram:ada")
        self.assertTrue(voted["ok"])
        self.assertTrue(voted["paid"])


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-math-")
        self.path = os.path.join(self.tmp.name, "math_state.json")
        self._old = os.environ.get("NOMORALS_MATH_STATE_PATH")
        os.environ["NOMORALS_MATH_STATE_PATH"] = self.path

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("NOMORALS_MATH_STATE_PATH", None)
        else:
            os.environ["NOMORALS_MATH_STATE_PATH"] = self._old
        self.tmp.cleanup()

    def test_save_load_round_trip_preserves_state(self) -> None:
        handler = MathCommandHandler()
        handler.economy.verify_member("telegram:999")
        handler.handle("/math", "telegram:999")
        self.assertTrue(save_math_state(handler))
        loaded = load_math_state()
        assert loaded is not None
        # Pre-answer state survived the round-trip (empty tier clears,
        # no ledger entries yet).
        self.assertEqual(loaded._tier_clears, handler._tier_clears)
        # Active problem survived; answering it after reload grades.
        pid = loaded._active["telegram:999"]
        ans = loaded.math.store["active"][pid].answer_text
        out = loaded.handle(f"/math answer {ans}", "telegram:999")
        self.assertTrue(out["ok"])
        self.assertTrue(out["correct"])
        # Answering on the reloaded handler persists again (first clear
        # recorded, ledger credited).
        self.assertIn("telegram:999", loaded._tier_clears)
        self.assertGreater(
            loaded.economy.balance("telegram:999")["coins"], 0)

    def test_fraction_answer_value_round_trips(self) -> None:
        from fractions import Fraction

        from nomorals.games.math_class import Problem

        handler = MathCommandHandler()
        problem = Problem(
            problem_id="test_frac", seed=1, tier="easy", topic="fractions",
            format="quickfire", question="1/2 + 1/4 = ?",
            answer_value=Fraction(3, 4), answer_text="3/4",
            solution="Common denominator 4.", solution_steps=[],
            created_at=0.0)
        handler.math.store["active"]["test_frac"] = problem
        handler._active["telegram:999"] = "test_frac"
        data = dump_state(handler)
        reloaded = load_state(data)
        got = reloaded.math.store["active"]["test_frac"].answer_value
        self.assertIsInstance(got, Fraction)
        self.assertEqual(got, Fraction(3, 4))

    def test_corrupt_file_returns_none(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertIsNone(load_math_state())

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(load_math_state())


class RuntimeMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-math-runtime-")
        self.math_path = os.path.join(self.tmp.name, "math_state.json")
        self.modes_path = os.path.join(self.tmp.name, "chat_modes.json")
        self._old_math = os.environ.get("NOMORALS_MATH_STATE_PATH")
        self._old_modes = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_MATH_STATE_PATH"] = self.math_path
        os.environ["NOMORALS_CHAT_MODES_PATH"] = self.modes_path
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.runtime = PartnerRuntime(self.context, gateway=_RecordingGateway())

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        if self._old_math is None:
            os.environ.pop("NOMORALS_MATH_STATE_PATH", None)
        else:
            os.environ["NOMORALS_MATH_STATE_PATH"] = self._old_math
        if self._old_modes is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old_modes
        self.tmp.cleanup()

    def test_handle_control_owner_dm_math_starts_problem(self) -> None:
        out = self.runtime.handle_control(
            "/math", "telegram:999",
            message=_msg("/math", owner=True))
        self.assertIn("🧮", out)
        self.assertIn("Answer with /math answer", out)

    def test_handle_control_answer_grades_and_persists(self) -> None:
        self.runtime.handle_control(
            "/math", "telegram:999", message=_msg("/math", owner=True))
        handler = self.runtime._math_handler()
        # The runtime player key is derived from the chat key; fetch it
        # from the handler's own active map.
        player_key = next(iter(handler._active))
        ans = handler.math.store["active"][handler._active[player_key]].answer_text
        out = self.runtime.handle_control(
            f"/math answer {ans}", "telegram:999",
            message=_msg(f"/math answer {ans}", owner=True))
        self.assertIn("✅ Correct!", out)

    def test_handle_control_outsider_dm_gets_redirect(self) -> None:
        out = self.runtime.handle_control(
            "/math", "telegram:999", message=_msg("/math"))
        self.assertIn("group game", out)
        self.assertNotIn("Answer with /math answer", out)

    def test_handle_control_outsider_group_gets_game(self) -> None:
        out = self.runtime.handle_control(
            "/math", "telegram:12345",
            message=_msg("/math", kind=ChatKind.GROUP))
        self.assertIn("🧮", out)
        self.assertIn("Answer with /math answer", out)

    def test_game_math_via_control_game(self) -> None:
        out = self.runtime._control_game(
            "math", "telegram:999",
            player=self.runtime._game_player_for_key("telegram:999"),
            kind="group", is_owner=False)
        self.assertIn("🧮", out)

    def test_process_owner_dm_math(self) -> None:
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None):
            self.runtime._process(_msg("/math", owner=True))
        self.assertTrue(any("🧮" in text for _, text in gateway.sent))

    def test_process_outsider_group_math(self) -> None:
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None):
            self.runtime._process(_msg("/math", kind=ChatKind.GROUP))
        self.assertTrue(any("🧮" in text for _, text in gateway.sent))

    def test_process_outsider_dm_math_falls_through_to_conversation(self) -> None:
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None), \
                mock.patch.object(
                    self.runtime.brain, "handle_message",
                    return_value=SimpleNamespace(parts=["hello back"])):
            self.runtime._process(_msg("/math"))
        # No math problem sent; the brain's conversation reply was.
        self.assertFalse(any("🧮" in text for _, text in gateway.sent))
        self.assertTrue(any("hello back" in text for _, text in gateway.sent))

    def test_math_state_persists_across_runtime_calls(self) -> None:
        self.runtime.handle_control(
            "/math", "telegram:999", message=_msg("/math", owner=True))
        handler = self.runtime._math_handler()
        player_key = next(iter(handler._active))
        ans = handler.math.store["active"][handler._active[player_key]].answer_text
        self.runtime.handle_control(
            f"/math answer {ans}", "telegram:999",
            message=_msg(f"/math answer {ans}", owner=True))
        # A fresh runtime reading the same file sees the solved state.
        runtime2 = PartnerRuntime(self.context, gateway=_RecordingGateway())
        try:
            handler2 = runtime2._math_handler()
            out = handler2.handle("/math streak", player_key)
            self.assertEqual(out["streak"], 1)
        finally:
            try:
                runtime2.stop()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
