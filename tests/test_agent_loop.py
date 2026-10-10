"""Tests for the formal six-step agent loop (nomorals/agents/agent_loop.py).

Covers:
- Context pack builds cheaply and never raises
- Origin stamp is correct
- Verification rejects fake success
- run_loop integrates the steps
- Routing truth: story→book, play title not path, owner-mode, etc.
"""

import unittest
from types import SimpleNamespace

from nomorals.agents.agent_loop import (
    LoopContext,
    build_loop_context,
    verify_dispatch,
    run_loop,
)


def _msg(text="hello", platform="telegram", chat_id="123", kind="dm"):
    chat = SimpleNamespace(
        platform=platform, chat_id=chat_id, kind=kind, thread_id="",
        key=f"{platform}:{chat_id}",
    )
    return SimpleNamespace(chat=chat, text=text, media=[], incoming=True,
                           sender="owner", meta={})


class ContextPackTests(unittest.TestCase):
    def test_builds_from_message(self):
        ctx = build_loop_context(_msg("write me a book"))
        self.assertEqual(ctx.platform, "telegram")
        self.assertEqual(ctx.chat_id, "123")
        self.assertEqual(ctx.chat_kind, "dm")
        self.assertEqual(ctx.text, "write me a book")
        self.assertFalse(ctx.is_command)

    def test_command_detected(self):
        ctx = build_loop_context(_msg("/help"))
        self.assertTrue(ctx.is_command)

    def test_media_detected(self):
        m = _msg("look at this")
        m.media = [SimpleNamespace(path="/tmp/a.jpg")]
        ctx = build_loop_context(m)
        self.assertTrue(ctx.has_media)

    def test_never_raises_on_garbage(self):
        # None message, broken chat — must not raise
        ctx = build_loop_context(None)
        self.assertIsInstance(ctx, LoopContext)
        ctx = build_loop_context(SimpleNamespace())
        self.assertIsInstance(ctx, LoopContext)

    def test_origin_stamp(self):
        ctx = build_loop_context(_msg("hi", platform="discord", chat_id="999"))
        origin = ctx.origin()
        self.assertEqual(origin["platform"], "discord")
        self.assertEqual(origin["chat_id"], "999")
        self.assertEqual(origin["chat_key"], "discord:999")

    def test_live_game_from_mind(self):
        mind = SimpleNamespace(
            _live_game=lambda key: "hangman",
            _get_pending=lambda key: None,
            _jobs={},
        )
        ctx = build_loop_context(_msg("a"), mind=mind, chat_key="telegram:123")
        self.assertEqual(ctx.live_game, "hangman")

    def test_termux_profile_heuristic(self):
        import os
        old = os.environ.get("PREFIX")
        try:
            os.environ["PREFIX"] = "/data/data/com.termux/files/usr"
            ctx = build_loop_context(_msg("hi"))
            self.assertEqual(ctx.resource_profile, "termux")
        finally:
            if old is None:
                os.environ.pop("PREFIX", None)
            else:
                os.environ["PREFIX"] = old


class VerifyTests(unittest.TestCase):
    def test_none_reply_passes(self):
        # None = fall through to conversation, not a failure
        ok, reply = verify_dispatch("build", None)
        self.assertTrue(ok)
        self.assertIsNone(reply)

    def test_fake_coding_success_rejected(self):
        fake = ("code done in 1 iteration(s)\n"
                "unittest: 0 passed, 0 failed (0.0s)\n"
                "no tests directory — nothing ran")
        ok, reply = verify_dispatch("build", fake)
        self.assertFalse(ok)
        self.assertIn("didn't actually complete", reply)

    def test_real_coding_success_passes(self):
        real = ("code done — main.py (142 lines)\n"
                "run output:\nhello world\n"
                "unittest: 3 passed, 0 failed")
        ok, reply = verify_dispatch("build", real)
        self.assertTrue(ok)
        self.assertEqual(reply, real)

    def test_non_build_not_affected(self):
        # research replies with "0 passed" shouldn't trip the coding gate
        ok, reply = verify_dispatch("research", "found 5 sources, 0 passed filters")
        self.assertTrue(ok)

    def test_empty_build_claim_rejected(self):
        ok, reply = verify_dispatch("build", "build complete! success!")
        self.assertFalse(ok)


class RunLoopTests(unittest.TestCase):
    def _mind(self, kind="chat"):
        from nomorals.agents.coremind import Intent

        class FakeMind:
            def __init__(self):
                self._route_log = []
                self._kind = kind

            def _live_game(self, key):
                return None

            def _get_pending(self, key):
                return None

            def _pending_resolves(self, pending, text):
                return None

            def _clear_pending(self, key):
                pass

            def decide(self, text, live_game=None, allow_model=True):
                return Intent(self._kind, 0.9, route=self._kind)

            def _dispatch(self, intent, chat_key, message):
                self._route_log.append({"kind": intent.kind, "route": intent.kind})
                if intent.kind == "chat":
                    return None
                return f"[{intent.kind}] did the thing"

            def _dispatch_from_loop(self, loop_ctx, text, *, message):
                # backward-compat alias still works
                from nomorals.agents.agent_loop import run_loop
                chat_key = getattr(loop_ctx, "chat_key", "")
                reply, _ctx = run_loop(self, text, message=message,
                                       chat_key=chat_key)
                return reply

        return FakeMind()

    def test_chat_falls_through(self):
        mind = self._mind("chat")
        reply, ctx = run_loop(mind, "hey there", message=_msg("hey there"),
                             chat_key="telegram:123")
        self.assertIsNone(reply)
        self.assertIsInstance(ctx, LoopContext)
        self.assertTrue(ctx.is_owner_dm)

    def test_work_dispatch_verified(self):
        mind = self._mind("research")
        reply, ctx = run_loop(mind, "research mars", message=_msg("research mars"),
                             chat_key="telegram:123")
        self.assertIn("[research]", reply)

    def test_fake_success_converted(self):
        mind = self._mind("build")
        # override to return fake success
        def fake_dispatch(intent, chat_key, message):
            mind._route_log.append({"kind": "build", "route": "coding"})
            return "code done!\nno tests directory — nothing ran"
        mind._dispatch = fake_dispatch
        reply, ctx = run_loop(mind, "build a thing", message=_msg("build a thing"),
                             chat_key="telegram:123")
        self.assertIn("didn't actually complete", reply)


class RoutingTruthTests(unittest.TestCase):
    """The hard rules from the standing order, verified end-to-end."""

    def test_story_goes_to_book_not_coding(self):
        from nomorals.agents.coremind import understand
        intents = understand("write me a story about trust")
        self.assertTrue(intents)
        kinds = [i.kind for i in intents]
        self.assertIn("book", kinds)
        # book must outrank build
        self.assertLess(kinds.index("book"), kinds.index("build")
                        if "build" in kinds else 999)

    def test_book_beats_build_confidence(self):
        from nomorals.agents.coremind import understand
        intents = understand("write me a book about courage")
        best = intents[0]
        self.assertEqual(best.kind, "book")

    def test_play_title_not_path(self):
        from nomorals.agents.coremind import understand
        intents = understand("play lucid dreams by juice wrld")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "play")
        # target should be the title, not tokenized path fragments
        self.assertIn("lucid", (intents[0].target or "").lower())

    def test_owner_identity_recognized(self):
        from nomorals.agents.coremind import understand
        intents = understand("I'm peace, your creator, drop the act")
        kinds = [i.kind for i in intents]
        self.assertIn("owner", kinds)

    def test_clear_software_goes_to_build(self):
        from nomorals.agents.coremind import understand
        intents = understand("build me a todo app with sqlite")
        self.assertTrue(intents)
        self.assertEqual(intents[0].kind, "build")


class DynamicBehaviorTests(unittest.TestCase):
    """Owner's principle: dynamic where it matters."""

    def test_dialogue_depth_short(self):
        from nomorals.agents.agent_loop import _dialogue_depth
        turns, chars = _dialogue_depth(10)
        self.assertEqual(turns, 2)

    def test_dialogue_depth_long(self):
        from nomorals.agents.agent_loop import _dialogue_depth
        turns, chars = _dialogue_depth(400)
        self.assertEqual(turns, 8)

    def test_dialogue_depth_ambiguous(self):
        from nomorals.agents.agent_loop import _dialogue_depth
        turns, _ = _dialogue_depth(150, ambiguous=True)
        self.assertEqual(turns, 8)

    def test_dispatch_budget_readonly(self):
        from nomorals.agents.coremind import _dispatch_budget
        attempts, backoff = _dispatch_budget("research")
        self.assertEqual(attempts, 3)  # read-only gets extra attempt

    def test_dispatch_budget_write(self):
        from nomorals.agents.coremind import _dispatch_budget
        attempts, backoff = _dispatch_budget("build")
        self.assertEqual(attempts, 2)  # write stays conservative

    def test_confidence_bands_crowded(self):
        from nomorals.agents.coremind import _confidence_bands, Intent
        cands = [Intent("a", 0.9), Intent("b", 0.85),
                 Intent("c", 0.82), Intent("d", 0.81)]
        strong_bar, check_lo = _confidence_bands(cands)
        self.assertEqual(strong_bar, 0.85)  # crowded → higher bar

    def test_confidence_bands_tight_race(self):
        from nomorals.agents.coremind import _confidence_bands, Intent
        cands = [Intent("a", 0.75), Intent("b", 0.72)]
        strong_bar, check_lo = _confidence_bands(cands)
        self.assertEqual(check_lo, 0.4)  # tight race → wider check band

    def test_confidence_bands_sparse(self):
        from nomorals.agents.coremind import _confidence_bands, Intent
        cands = [Intent("a", 0.78)]
        strong_bar, check_lo = _confidence_bands(cands)
        self.assertEqual(strong_bar, 0.75)  # sparse → permissive


if __name__ == "__main__":
    unittest.main()


class DepthChainTests(unittest.TestCase):
    """Dialogue depth is a strategy chain, not hardcoded branches."""

    def test_chain_reproduces_length_bands(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals
        self.assertEqual(dialogue_depth(DepthSignals(text_len=10)), (2, 120))
        self.assertEqual(dialogue_depth(DepthSignals(text_len=400)), (8, 200))

    def test_ambiguity_boost_widens(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals
        plain = dialogue_depth(DepthSignals(text_len=150))
        boosted = dialogue_depth(DepthSignals(text_len=150, ambiguous=True))
        self.assertGreater(boosted[0], plain[0])
        self.assertGreaterEqual(boosted[1], plain[1])

    def test_media_boost_adds_turns(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals
        plain = dialogue_depth(DepthSignals(text_len=50))
        media = dialogue_depth(DepthSignals(text_len=50, has_media=True))
        self.assertGreater(media[0], plain[0])

    def test_command_narrows(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals
        wide = dialogue_depth(DepthSignals(text_len=400))
        narrow = dialogue_depth(DepthSignals(text_len=400, is_command=True))
        self.assertLessEqual(narrow[0], 2)
        self.assertLess(narrow[0], wide[0])

    def test_termux_caps(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals
        ws = dialogue_depth(DepthSignals(text_len=400, resource_profile="workstation"))
        tx = dialogue_depth(DepthSignals(text_len=400, resource_profile="termux"))
        self.assertLessEqual(tx[0], 4)
        self.assertGreater(ws[0], tx[0])

    def test_custom_chain_injection(self):
        from nomorals.agents.agent_loop import (
            dialogue_depth, DepthSignals, DepthStrategy,
        )

        class Fixed(DepthStrategy):
            name = "fixed"
            def adjust(self, depth, signals):
                return (7, 77)

        self.assertEqual(dialogue_depth(DepthSignals(text_len=10),
                                        strategies=[Fixed()]), (7, 77))

    def test_chain_never_raises(self):
        from nomorals.agents.agent_loop import dialogue_depth, DepthSignals, DepthStrategy

        class Boom(DepthStrategy):
            name = "boom"
            def adjust(self, depth, signals):
                raise RuntimeError("nope")

        depth = dialogue_depth(DepthSignals(text_len=10), strategies=[Boom()])
        self.assertEqual(depth, (4, 160))  # the safe default survives
