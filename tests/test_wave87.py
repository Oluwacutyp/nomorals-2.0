"""Wave 87 — the Agent OS transformation: the Core Mind + the games organ.

Covers:
* intent detection (strong triggers AND the false-trigger battery)
* the structural owner-DM-only gate (non-owner chats cannot launch anything
  from natural language — commands are the only trigger there)
* clarification flow (ask → answer / cancel, persisted across restarts)
* dispatch routing to the organs (games, research, mission, status)
* profile-aware sizing of the research fan-out
* the model disambiguation layer (used only in the uncertain band, with a
  deterministic fallback)
* the real chat flow through the Social Operator (owner DM routes, non-owner
  falls through to conversation, /hangman in a group starts a real room)
* the command center: 18 direct game commands + /mind, all documented
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.coremind import (
    CoreMind,
    understand,
)
from nomorals.core.config import load_settings
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    GAME_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    _HELP_GROUPS,
    help_text,
    parse_control,
)


# ── hermetic fakes ──────────────────────────────────────────────────────────

class FakeMemory:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.updates: list[tuple[str, dict]] = []

    def remember(self, content, *, kind="episode", importance=0.5, source="",
                 agent="", metadata=None, ttl_seconds=0.0, index=True,
                 tags=None, origin="") -> str:
        rid = f"m{len(self.records) + 1}"
        self.records.append({"id": rid, "content": content, "kind": kind,
                             "metadata": metadata or {}, "source": source})
        return rid

    def update(self, record_id, **changes) -> int:
        self.updates.append((record_id, changes))
        return 1


class FakeRouter:
    """The model layer: answers the intent-classification prompt."""

    def __init__(self, payload: str, *, ok: bool = True) -> None:
        self.payload = payload
        self.ok = ok
        self.calls = 0

    def complete(self, prompt, **kw):
        self.calls += 1
        if not self.ok:
            return SimpleNamespace(ok=False, text="", error="down")
        return SimpleNamespace(ok=True, text=self.payload)


class FakeRoom:
    def __init__(self, game: str) -> None:
        self.game = game
        self.chat_key = ""


class FakeGameEngine:
    def __init__(self, live_game: str | None = None) -> None:
        self.live_game = live_game
        self.started: list[tuple[str, str]] = []

    def live(self, chat_key: str):
        if self.live_game:
            room = FakeRoom(self.live_game)
            room.chat_key = chat_key
            return room
        return None

    def rooms(self):
        return []


class FakeRuntime:
    """The Social Operator, faked: records what the mind dispatches."""

    def __init__(self, live_game: str | None = None) -> None:
        self.engine = FakeGameEngine(live_game)
        self.game_calls: list[tuple[str, str, Any]] = []
        self.is_operator = True
        self.owner_chats = {"local:console"}

    def _is_operator(self, message):
        key = message.chat.key
        return self.is_operator and (key.endswith(":console")
                                     or key in self.owner_chats)

    def _game_engine(self):
        return self.engine

    def _game_player(self, message):
        return SimpleNamespace(key=f"{message.chat.platform}:tester")

    def _control_game(self, tail, chat_key, *, player=None, kind="dm"):
        self.game_calls.append((tail, chat_key, player))
        return f"GAME-STARTED:{tail}"


def _owner_dm_msg(text: str, chat_key: str = "local:console"):
    """A message whose .chat.kind is the real ChatKind.DM."""
    from nomorals.social.chat import ChatKind

    chat, sid = chat_key.split(":", 1)
    return SimpleNamespace(
        text=text,
        sender="you",
        chat=SimpleNamespace(key=chat_key, platform=chat, kind=ChatKind.DM),
        incoming=True,
    )


def _make_mind(*, settings=None, router=None) -> tuple[CoreMind, FakeMemory, SimpleNamespace]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-w87-")
    if settings is None:
        settings = load_settings(overrides={"home": tmp.name})
    memory = FakeMemory()
    context = SimpleNamespace(settings=settings, extras={}, memory=memory,
                              router=router)
    mind = CoreMind(context)
    mind._tmp = tmp  # keep alive for the test's lifetime
    return mind, memory, context


class Tune:
    def __init__(self, kind="vps", vcpu_target=4) -> None:
        self.profile = SimpleNamespace(kind=kind)
        self.vcpu_target = vcpu_target


# ── 1. intent detection: strong triggers ────────────────────────────────────

class GameTriggerTests(unittest.TestCase):
    def test_start_triggers(self) -> None:
        for text, name in [("let's play hangman", "hangman"),
                           ("start mafia", "mafia"),
                           ("play wordchain", "wordchain"),
                           ("begin the rpg", "rpg"),
                           ("let's do trivia", "trivia")]:
            cands = understand(text)
            self.assertTrue(cands, text)
            self.assertEqual(cands[0].kind, "game", text)
            self.assertEqual(cands[0].action, "start", text)
            self.assertEqual(cands[0].target, name, text)

    def test_aliases_resolve(self) -> None:
        self.assertEqual(understand("play word chain")[0].target, "wordchain")
        self.assertEqual(understand("start king of the hill")[0].target, "king")
        self.assertEqual(understand("let's play 20 questions")[0].target, "20q")
        self.assertEqual(understand("start the dungeon")[0].target, "rpg")

    def test_continue_semantics(self) -> None:
        cands = understand("continue the RPG")
        self.assertEqual(cands[0].action, "start")  # no live room → start it
        self.assertEqual(cands[0].target, "rpg")
        cands = understand("resume mafia", live_game="mafia")
        self.assertEqual(cands[0].action, "resume")
        cands = understand("keep going", live_game="mafia")
        self.assertEqual(cands[0].action, "resume")
        self.assertEqual(cands[0].target, "mafia")

    def test_board_and_economy_intents(self) -> None:
        cands = understand("show leaderboard")
        self.assertEqual((cands[0].kind, cands[0].action), ("game", "board"))
        cands = understand("show me the mafia board")
        self.assertEqual((cands[0].action, cands[0].target), ("board", "mafia"))
        cands = understand("what's my balance")
        self.assertEqual(cands[0].action, "economy")

    def test_false_triggers_never_launch(self) -> None:
        """The wave 87 spec battery: none of these may produce a game intent."""
        for text in ("game", "games", "gaming", "this game is boring",
                     "football game", "I played a game last night",
                     "that's a great game show", "video game night",
                     "we should play some games", "game over",
                     "I love gaming", "the game of life"):
            cands = understand(text)
            games = [c for c in cands if c.kind == "game"]
            self.assertEqual(games, [], f"{text!r} must not trigger games")

    def test_play_without_name_asks(self) -> None:
        cands = understand("let's play")
        self.assertEqual(cands[0].action, "ask")
        cands = understand("let's play a game")
        self.assertEqual(cands[0].action, "ask")


class OtherIntentTests(unittest.TestCase):
    def test_research(self) -> None:
        cands = understand("research the history of jazz")
        self.assertEqual(cands[0].kind, "research")
        self.assertEqual(cands[0].target, "the history of jazz")
        cands = understand("investigate why the api is slow")
        self.assertEqual(cands[0].kind, "research")

    def test_build(self) -> None:
        cands = understand("build me a website that does invoices")
        self.assertEqual(cands[0].kind, "build")
        self.assertEqual(cands[0].target, "a website that does invoices")
        cands = understand("write a script to rename all my files")
        self.assertEqual(cands[0].kind, "build")

    def test_download_and_browse_with_url(self) -> None:
        cands = understand("download https://example.com/file.mp4")
        self.assertEqual(cands[0].kind, "download")
        cands = understand("get me the video from https://youtu.be/abc")
        self.assertEqual(cands[0].kind, "download")
        cands = understand("open https://news.ycombinator.com for me")
        self.assertEqual(cands[0].kind, "browse")

    def test_missions(self) -> None:
        cands = understand("mission: fix the deploy")
        self.assertEqual((cands[0].kind, cands[0].action), ("mission", "add"))
        self.assertEqual(cands[0].target, "fix the deploy")
        cands = understand("continue the mission")
        self.assertEqual(cands[0].action, "run")
        cands = understand("mission status")
        self.assertEqual(cands[0].action, "list")

    def test_status(self) -> None:
        for text in ("status", "what's active", "what are you doing"):
            cands = understand(text)
            self.assertEqual(cands[0].kind, "status", text)

    def test_plain_chat_has_no_intents(self) -> None:
        for text in ("hey, how are you?", "lol", "what's the weather like",
                     "tell me about your day"):
            self.assertEqual(understand(text), [], text)


# ── 2. CoreMind: gating, clarification, routing ─────────────────────────────

class CoreMindGatingTests(unittest.TestCase):
    def test_owner_dm_dispatches_game(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        reply = mind.handle("let's play hangman",
                            message=_owner_dm_msg("let's play hangman"),
                            chat_key="local:console")
        self.assertIn("GAME-STARTED:hangman", reply or "")
        self.assertEqual(runtime.game_calls[0][0], "hangman")
        self.assertEqual(runtime.game_calls[0][1], "local:console")

    def test_non_owner_dm_can_never_launch(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        runtime.is_operator = False
        mind.runtime = runtime
        for text in ("let's play hangman", "research the history of jazz",
                     "build me a bot", "mission: do the thing"):
            reply = mind.handle(text,
                                message=_owner_dm_msg(text, "local:stranger"),
                                chat_key="local:stranger")
            self.assertIsNone(reply, text)
        self.assertEqual(runtime.game_calls, [])

    def test_owner_in_group_still_command_only(self) -> None:
        """Owner DMs get NL routing; groups do not (even for the owner)."""
        from nomorals.social.chat import ChatKind

        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        msg = SimpleNamespace(text="let's play hangman", sender="you",
                              chat=SimpleNamespace(key="local:g1",
                                                   platform="local",
                                                   kind=ChatKind.GROUP),
                              incoming=True)
        # operator=True (it's the owner) but the chat is a GROUP
        self.assertIsNone(mind.handle("let's play hangman", message=msg,
                                      chat_key="local:g1"))
        self.assertEqual(runtime.game_calls, [])

    def test_console_key_dispatches(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        reply = mind.handle("let's play hangman",
                            message=_owner_dm_msg("let's play hangman"),
                            chat_key="local:console")
        self.assertIn("GAME-STARTED:hangman", reply or "")


class ClarificationTests(unittest.TestCase):
    def test_ask_then_answer_resolves(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        q = mind.handle("let's play",
                        message=_owner_dm_msg("let's play"),
                        chat_key="local:console")
        self.assertIn("which one", q or "")
        self.assertTrue(mind.pending())
        reply = mind.handle("hangman",
                            message=_owner_dm_msg("hangman"),
                            chat_key="local:console")
        self.assertIn("GAME-STARTED:hangman", reply or "")
        self.assertEqual(mind.pending(), {})

    def test_answer_with_other_name_resolves_to_that_name(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        mind.handle("let's play", message=_owner_dm_msg("let's play"),
                    chat_key="local:console")
        reply = mind.handle("trivia", message=_owner_dm_msg("trivia"),
                            chat_key="local:console")
        self.assertIn("GAME-STARTED:trivia", reply or "")

    def test_cancel_clears_pending(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        mind.handle("let's play", message=_owner_dm_msg("let's play"),
                    chat_key="local:console")
        reply = mind.handle("never mind", message=_owner_dm_msg("never mind"),
                            chat_key="local:console")
        self.assertIn("scrapped", reply or "")
        self.assertEqual(mind.pending(), {})
        self.assertEqual(runtime.game_calls, [])

    def test_long_unrelated_message_clears_and_falls_through(self) -> None:
        mind, _mem, _ctx = _make_mind()
        runtime = FakeRuntime()
        mind.runtime = runtime
        mind.handle("let's play", message=_owner_dm_msg("let's play"),
                    chat_key="local:console")
        reply = mind.handle("actually can you research the state of solid state batteries",
                            message=_owner_dm_msg(
                                "actually can you research the state of solid state batteries"),
                            chat_key="local:console")
        self.assertEqual(mind.pending(), {})
        self.assertIn("researching", reply or "")

    def test_pending_survives_restart(self) -> None:
        mind, _mem, ctx = _make_mind()
        mind._set_pending("local:console",
                          SimpleNamespace(kind="game", action="ask",
                                          target="", route="games"),
                          "which one — hangman, mafia?")
        mind2, _mem2, _ = _make_mind(settings=ctx.settings)
        pending = mind2._get_pending("local:console")
        self.assertIsNotNone(pending)
        self.assertIn("which one", pending["question"])

    def test_objective_written_to_memory(self) -> None:
        mind, memory, _ctx = _make_mind()
        job_id = mind._new_job(SimpleNamespace(kind="research",
                                               target="the history of jazz",
                                               route="research_swarm"))
        self.assertTrue(memory.records)
        self.assertEqual(memory.records[0]["kind"], "objective")
        self.assertIn("the history of jazz", memory.records[0]["content"])
        mind._job_done(job_id, True, "done note")
        self.assertEqual(memory.updates[0][1]["metadata"]["status"], "done")


class RoutingAndContinuityTests(unittest.TestCase):
    def test_job_registry_and_status(self) -> None:
        mind, memory, _ctx = _make_mind()
        j1 = mind._new_job(SimpleNamespace(kind="research", target="jazz",
                                           route="research_swarm"))
        j2 = mind._new_job(SimpleNamespace(kind="build", target="a bot",
                                           route="coding"))
        mind._job_done(j1, True, "report saved")
        text = mind.status()
        self.assertIn("core mind", text)
        self.assertIn("jazz", text)
        self.assertIn("1 active", text)
        self.assertIn("1 done", text)
        self.assertIn("last objective", text)

    def test_tune_scales_research_workers(self) -> None:
        mind, _mem, ctx = _make_mind()
        ctx.extras["tune"] = Tune("termux", 2)
        self.assertEqual(mind._research_workers(), 2)
        ctx.extras["tune"] = Tune("vps", 4)
        self.assertEqual(mind._research_workers(), 3)
        ctx.extras["tune"] = Tune("workstation", 16)
        self.assertEqual(mind._research_workers(), 4)
        ctx.extras = {}
        self.assertEqual(mind._research_workers(), 3)

    def test_model_check_used_only_in_uncertain_band(self) -> None:
        router = FakeRouter('{"kind": "research", "target": "x", '
                            '"confidence": 0.9, "why": "model says so"}')
        mind, _mem, ctx = _make_mind(router=router)
        # strong deterministic intent → no model call
        intent = mind.decide("research the history of jazz")
        self.assertEqual(intent.kind, "research")
        self.assertEqual(router.calls, 0)
        # plain chat → no model call
        intent = mind.decide("hey, how are you?")
        self.assertEqual(intent.kind, "chat")
        self.assertEqual(router.calls, 0)
        # uncertain build (0.6) → model consulted and trusted
        intent = mind.decide("build something interesting")
        self.assertEqual(intent.kind, "research")
        self.assertEqual(intent.why, "model: model says so")
        self.assertEqual(router.calls, 1)

    def test_model_down_falls_back_to_deterministic(self) -> None:
        router = FakeRouter("garbage", ok=False)
        mind, _mem, ctx = _make_mind(router=router)
        intent = mind.decide("build something interesting")
        self.assertEqual(intent.kind, "build")

    def test_multi_part_goal(self) -> None:
        mind, _mem, _ctx = _make_mind()
        intent = mind.decide("research the history of jazz and build me a bot")
        self.assertEqual(intent.kind, "multi")
        self.assertEqual(intent.route, "orchestrator")


# ── 3. the real chat flow through the Social Operator ───────────────────────

class _RuntimeFixture:
    def __init__(self) -> None:
        from tests.test_w85c import _ScriptedRouter, RecordingAdapter
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.social.chat import ChatGateway

        self._scripted = _ScriptedRouter
        self._recording = RecordingAdapter
        self._build_context = build_context
        self._partner = PartnerRuntime
        self._gateway = ChatGateway
        self._cleanups = []

    def make(self) -> tuple:
        tmp = tempfile.TemporaryDirectory(prefix="nm-w87-rt-")
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = self._build_context(settings, with_executor=False,
                                      with_tools=False)
        context.router = self._scripted()
        adapter = self._recording("local")
        gateway = self._gateway({"local": adapter}, db=context.db)
        runtime = self._partner(context, gateway=gateway)
        runtime.brain.presence_rng = random.Random(34)
        self._cleanups.append((context, tmp))
        return runtime, adapter, settings

    def close(self) -> None:
        for context, tmp in self._cleanups:
            try:
                context.close()
            except Exception:  # noqa: BLE001
                pass
            tmp.cleanup()


def _msg(chat_key: str, text: str, sender: str = "you", kind="dm",
         mentioned: bool = False):
    from nomorals.social.chat import ChatKind, ChatMessage, ChatRef

    chat, cid = chat_key.split(":", 1)
    ref = ChatRef(platform=chat, chat_id=cid,
                  kind={"dm": ChatKind.DM, "group": ChatKind.GROUP}[kind])
    return ChatMessage(chat=ref, incoming=True, text=text, sender=sender,
                       mentioned=mentioned)


class ChatFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _RuntimeFixture()
        self.runtime, self.adapter, self.settings = self.fx.make()

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            pass
        self.fx.close()

    def test_owner_dm_research_routes_with_ack(self) -> None:
        self.runtime._process(_msg("local:console",
                                   "research the history of jazz"))
        acks = [t for t in self.adapter.sent if "researching" in t]
        self.assertTrue(acks, f"no research ack in {self.adapter.sent}")
        self.assertIn("route: research_swarm", acks[0])

    def test_owner_dm_game_nl_start_real_engine(self) -> None:
        self.runtime._process(_msg("local:console", "let's play numberguess"))
        text = "".join(self.adapter.sent)
        self.assertIn("🎮", text)  # the table-opening banner
        room = self.runtime._game_engine().live("local:console")
        self.assertIsNotNone(room)
        self.assertEqual(room.game, "numberguess")

    def test_owner_dm_status_routes(self) -> None:
        self.runtime._process(_msg("local:console", "what's active"))
        text = "".join(self.adapter.sent)
        self.assertIn("core mind", text)

    def test_non_owner_nl_never_launches(self) -> None:
        self.runtime._process(_msg("local:stranger", "let's play hangman",
                                   sender="newbie"))
        self.assertIsNone(self.runtime._game_engine().live("local:stranger"))
        # it fell through to the normal conversation flow (the scripted model)
        self.assertTrue(self.adapter.sent)
        self.assertNotIn("GAME-STARTED", "".join(self.adapter.sent))

    def test_non_owner_group_command_starts_game(self) -> None:
        self.runtime._process(_msg("local:g1", "/hangman", sender="someone",
                                   kind="group"))
        room = self.runtime._game_engine().live("local:g1")
        self.assertIsNotNone(room)
        self.assertEqual(room.game, "hangman")

    def test_active_group_game_routers_moves_not_brain(self) -> None:
        self.runtime._process(_msg("local:g1", "/wordchain", sender="someone",
                                   kind="group"))
        before = len(self.adapter.sent)
        self.runtime._process(_msg("local:g1", "apple", sender="someone2",
                                   kind="group"))
        new = "".join(self.adapter.sent[before:])
        self.assertNotIn("mhm, that tracks", new)  # not the brain's reply
        self.assertTrue(new.strip(), "the live game swallowed the move silently")

    def test_mind_command_status_in_chat(self) -> None:
        reply = self.runtime.handle_control("/mind status", "local:console")
        self.assertIn("core mind", reply)

    def test_mind_command_goal_in_chat(self) -> None:
        reply = self.runtime.handle_control(
            "/mind research the history of jazz", "local:console")
        self.assertIn("decision: research", reply)

    def test_mind_command_clear(self) -> None:
        reply = self.runtime.handle_control("/mind clear", "local:console")
        self.assertIn("cleared", reply)

    def test_per_game_command_in_chat_list(self) -> None:
        reply = self.runtime.handle_control("/list games", "local:console")
        for name in ("hangman", "mafia", "wordchain", "numberguess"):
            self.assertIn(f"/{name}", reply)


# ── 4. command center completeness ──────────────────────────────────────────

class CommandCenterTests(unittest.TestCase):
    def test_all_commands_documented(self) -> None:
        help_groups: set[str] = set()
        for _, kinds in _HELP_GROUPS:
            help_groups.update(kinds)
        list_groups: set[str] = set()
        for _, kinds in LIST_GROUPS:
            list_groups.update(kinds)
        for kind in CONTROL_COMMANDS:
            detail = COMMAND_DETAILS.get(kind)
            self.assertIsNotNone(detail, f"/{kind} has no detail page")
            for field in ("what", "usage", "example"):
                self.assertTrue(detail.get(field), f"/{kind} missing {field}")
            if kind not in ("commands", "menu"):
                self.assertIn(kind, LIST_ONELINERS, f"/{kind} has no oneliner")
            self.assertIn(kind, help_groups, f"/{kind} missing from /help catalog")
            self.assertIn(kind, list_groups, f"/{kind} missing from /list catalog")

    def test_game_commands_parse(self) -> None:
        self.assertEqual(len(GAME_COMMANDS), 18)
        for name in GAME_COMMANDS:
            cmd = parse_control(f"/{name}")
            self.assertIsNotNone(cmd, name)
            self.assertEqual(cmd.kind, name, name)
        # "arena" stays the self-improvement arena — that game is /game arena
        self.assertEqual(parse_control("/arena").kind, "arena")
        self.assertNotIn("arena", GAME_COMMANDS)

    def test_help_text_mentions_every_command(self) -> None:
        text = help_text()
        for kind in CONTROL_COMMANDS:
            self.assertIn(f"/{kind}", text, f"help text misses /{kind}")

    def test_game_alias_not_a_command(self) -> None:
        self.assertIsNone(parse_control("/20q"))
        self.assertIsNone(parse_control("/word chain"))


if __name__ == "__main__":
    unittest.main()
