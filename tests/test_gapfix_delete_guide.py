"""Deletion runbooks (AccountKiller build, 52-websites #28).

The registry holds a runbook per service — steps, difficulty,
grace window, export-first warning, delete URL — and the walk
takes the owner through it one step at a time, stopping at every
step that needs the owner's own hand. A stale guide is labelled
stale, never served as current. An unknown service is answered
honestly, never invented.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.accounts.deletion import (
    DIFFICULTIES,
    STALE_AFTER_DAYS,
    DeletionGuide,
    DeletionStep,
    DeletionStore,
    is_stale,
    parse_delete_request,
    render_guide,
    slugify_service,
    walk_advance,
    walk_current,
    walk_reset,
)


class _DecoRegistry:
    """A registry double that captures register() calls."""

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.registered: dict[str, Any] = {}

    def register(self, name: str, fn: Any = None, **_kw: Any) -> Any:
        def deco(f: Any) -> Any:
            self.registered[name] = f
            return f

        if fn is not None:
            return deco(fn)
        return deco


def _store_path(home: str) -> Path:
    return Path(home) / "data" / "deletion_guides.json"


class OrganTests(unittest.TestCase):
    def test_seeds_load_twenty_one_guides(self) -> None:
        store = DeletionStore("")
        guides = store.list_all()
        self.assertEqual(21, len(guides))
        seen = {guide.difficulty for guide in guides}
        self.assertEqual(set(DIFFICULTIES), seen)
        for guide in guides:
            self.assertGreaterEqual(len(guide.steps), 3)
            self.assertEqual("2026-11-18", guide.checked_at)

    def test_fixture_guide_renders_completely(self) -> None:
        store = DeletionStore("")
        guide = store.get("facebook")
        assert guide is not None
        text = render_guide(guide)
        self.assertIn("medium", text)
        self.assertIn("30 days", text)
        self.assertIn("Export your data first", text)
        self.assertIn(
            "Delete link: https://www.facebook.com/help/224562897555674",
            text,
        )
        for step in guide.steps:
            self.assertIn(step.text, text)
        self.assertIn("[YOUR HAND]", text)
        self.assertIn("Note: The old facebook.com/help/delete_account",
                      text)

    def test_stale_guide_labelled_not_served_as_current(self) -> None:
        # Build the stale guide directly: checked 2025, long past the
        # 180-day staleness window.
        stale = DeletionGuide(
            service="myspace",
            steps=(DeletionStep(text="Open the deletion page."),
                   DeletionStep(text="Confirm with your password.",
                                owner_hand=True)),
            difficulty="medium",
            grace_days=0,
            checked_at="2025-01-01",
        )
        self.assertTrue(is_stale(stale))
        text = render_guide(stale)
        self.assertIn("STALE", text)
        self.assertIn("last checked 2025-01-01", text)
        self.assertIn("(unverified)", text)
        fresh = DeletionGuide(
            service="myspace",
            steps=(DeletionStep(text="Open the deletion page."),),
            difficulty="medium",
            checked_at=date.today().isoformat(),
        )
        self.assertFalse(is_stale(fresh))
        self.assertNotIn("STALE", render_guide(fresh))

    def test_impossible_rating_survives_rendering(self) -> None:
        store = DeletionStore("")
        guide = store.get("archive.org")
        assert guide is not None
        self.assertEqual("impossible", guide.difficulty)
        text = render_guide(guide)
        self.assertIn("impossible", text)
        self.assertIn("no self-service deletion", text)

    def test_alias_resolution(self) -> None:
        store = DeletionStore("")
        self.assertEqual("x", store.get("twitter").service)
        self.assertEqual("facebook", store.get("fb").service)
        self.assertEqual("google", store.get("gmail").service)
        self.assertEqual("facebook", store.get("Facebook").service)
        self.assertIsNone(store.get("myspace"))

    def test_slugify(self) -> None:
        self.assertEqual("internet-archive",
                         slugify_service("Internet Archive"))
        self.assertEqual("x", slugify_service("X"))
        self.assertEqual("archive-org", slugify_service("archive.org"))

    def test_walk_advances_and_stops_at_owner_hand(self) -> None:
        store = DeletionStore("")
        guide = store.get("facebook")
        assert guide is not None
        store.reset_walk("facebook")
        first = walk_current(store, guide)
        self.assertIn("step 1 of 5", first)
        self.assertNotIn("This one is yours", first)
        second = walk_advance(store, guide)
        self.assertIn("step 2 of 5", second)
        walk_advance(store, guide)
        fourth = walk_advance(store, guide)
        self.assertIn("step 4 of 5", fourth)
        self.assertIn("This one is yours", fourth)
        fifth = walk_advance(store, guide)
        self.assertIn("step 5 of 5", fifth)
        done = walk_advance(store, guide)
        self.assertIn("That's the whole facebook runbook", done)
        self.assertIsNone(store.active_walk())

    def test_walk_reset_restarts(self) -> None:
        store = DeletionStore("")
        guide = store.get("spotify")
        assert guide is not None
        walk_advance(store, guide)
        walk_advance(store, guide)
        self.assertEqual(2, store.walk_position("spotify"))
        restarted = walk_reset(store, guide)
        self.assertEqual(0, store.walk_position("spotify"))
        self.assertIn("step 1 of 5", restarted)

    def test_add_persists_and_search_finds(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = DeletionStore(home)
            guide = DeletionGuide(
                service="signal",
                steps=(DeletionStep(text="Open Settings → Account."),
                       DeletionStep(
                           text="Tap Delete account and confirm.",
                           owner_hand=True)),
                difficulty="easy",
                grace_days=0,
            )
            store.add(guide)
            self.assertTrue(_store_path(home).exists())
            reloaded = DeletionStore(home)
            self.assertIsNotNone(reloaded.get("signal"))
            self.assertEqual(1, len(reloaded.search("signal")))
            self.assertEqual("signal",
                             reloaded.search("signal")[0].service)
            self.assertEqual(22, len(reloaded.list_all()))

    def test_add_rejects_bad_difficulty(self) -> None:
        with self.assertRaises(ValueError):
            DeletionGuide(
                service="nope",
                steps=(DeletionStep(text="Do the thing."),),
                difficulty="trivial",
            )

    def test_corrupt_store_quarantined_and_reseeded(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            store = DeletionStore(home)
            self.assertEqual(21, len(store.list_all()))
            store.reset_walk("facebook")  # persists the store to disk
            path = _store_path(home)
            self.assertTrue(path.exists())
            path.write_text("{not json", encoding="utf-8")
            reseeded = DeletionStore(home)
            self.assertEqual(21, len(reseeded.list_all()))
            quarantined = list(Path(home).glob("data/*.corrupt-*.json"))
            self.assertEqual(1, len(quarantined))

    def test_parse_shapes(self) -> None:
        store = DeletionStore("")
        known = store.known_names()
        self.assertEqual(
            {"kind": "guide", "service": "Facebook",
             "has_account_word": True},
            parse_delete_request("how do I delete my Facebook account?",
                                 known))
        self.assertEqual(
            {"kind": "walk", "service": "spotify",
             "has_account_word": False},
            parse_delete_request("close my spotify", known))
        self.assertEqual(
            {"kind": "walk", "service": "netflix",
             "has_account_word": True},
            parse_delete_request(
                "walk me through deleting my netflix account", known))
        self.assertEqual(
            {"kind": "guide", "service": "X",
             "has_account_word": True},
            parse_delete_request("delete my X account", known))
        self.assertIsNone(parse_delete_request("close my laptop", known))
        self.assertIsNone(parse_delete_request("delete my account", known))
        self.assertIsNone(parse_delete_request("/delete my facebook",
                                               known))
        self.assertIsNone(parse_delete_request("how's your day going?",
                                               known))
        parsed = parse_delete_request(
            "how do i delete my myspace account?", known)
        assert parsed is not None
        self.assertEqual("guide", parsed["kind"])
        self.assertTrue(parsed["has_account_word"])


class ToolTests(unittest.TestCase):
    def _tool(self, home: str):
        from nomorals.tools.delete_guide import register

        ctx = SimpleNamespace(settings=SimpleNamespace(home=home))
        deco = _DecoRegistry(context=ctx)
        register(deco)
        return deco.registered["delete_guide"]

    def test_tool_registered_by_builtins(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("delete_guide", reg._tools)

    def test_guide_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            out = fn(action="guide", service="facebook")
            self.assertTrue(out["ok"])
            self.assertEqual("medium", out["difficulty"])
            self.assertEqual(30, out["grace_days"])
            self.assertFalse(out["stale"])
            self.assertIn("Export your data first", out["text"])
            self.assertIn("Delete link:", out["text"])

    def test_unknown_service_honest_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            out = fn(action="guide", service="myspace")
            self.assertFalse(out["ok"])
            self.assertIn("myspace", out["error"])
            self.assertIn("won't invent one", out["error"])
            self.assertNotIn("text", out)

    def test_walk_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            first = fn(action="walk", service="spotify")
            self.assertTrue(first["ok"])
            self.assertEqual(0, first["step"])
            self.assertEqual(5, first["of"])
            self.assertIn("step 1 of 5", first["text"])
            second = fn(action="walk", service="spotify", done="done")
            self.assertEqual(1, second["step"])
            self.assertIn("step 2 of 5", second["text"])
            last = None
            for _ in range(5):
                last = fn(action="walk", service="spotify", done="done")
            assert last is not None
            self.assertTrue(last["finished"])
            self.assertIn("That's the whole spotify runbook", last["text"])
            restarted = fn(action="walk", service="spotify", done="reset")
            self.assertEqual(0, restarted["step"])
            self.assertFalse(restarted["finished"])

    def test_list_search_status_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            listed = fn(action="list")
            self.assertEqual(21, listed["count"])
            found = fn(action="search", query="spotify")
            self.assertEqual(1, found["count"])
            self.assertEqual("spotify", found["guides"][0]["service"])
            status = fn(action="status")
            self.assertEqual(21, status["count"])
            self.assertEqual(0, status["stale_count"])
            self.assertEqual(STALE_AFTER_DAYS, status["stale_after_days"])
            self.assertEqual(str(_store_path(home)), status["store"])

    def test_add_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            out = fn(
                action="add",
                service="signal",
                steps=("Open Settings → Account.\n"
                       "HAND: Tap Delete account and confirm."),
                difficulty="easy",
                grace_days="0",
            )
            self.assertTrue(out["ok"])
            self.assertEqual(2, out["steps"])
            self.assertIn("[YOUR HAND]", out["text"])
            guide = fn(action="guide", service="signal")
            self.assertTrue(guide["ok"])
            with self.assertRaises(ValueError):
                fn(action="add", service="bad", steps="Do it.",
                   difficulty="trivial")
            with self.assertRaises(ValueError):
                fn(action="explode")

    def test_stale_guide_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            fn(action="add", service="myspace",
               steps="Open the deletion page.",
               difficulty="medium")
            path = _store_path(home)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["guides"]["myspace"]["checked_at"] = "2025-01-01"
            path.write_text(json.dumps(raw), encoding="utf-8")
            out = fn(action="guide", service="myspace")
            self.assertTrue(out["ok"])
            self.assertTrue(out["stale"])
            self.assertIn("STALE", out["text"])


class HookTests(unittest.TestCase):
    def _hook(self, home: str, text: str):
        from nomorals.agents.partner.runtime import PartnerRuntime
        from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

        runtime_double = SimpleNamespace(
            context=SimpleNamespace(
                settings=SimpleNamespace(home=home)))
        chat = ChatRef(platform="local", chat_id="console",
                       kind=ChatKind.DM, peer="you")
        message = ChatMessage(chat=chat, incoming=True, text=text,
                              sender="you")
        return PartnerRuntime._delete_guide_hook(runtime_double, message)

    def test_hook_guide(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            reply = self._hook(
                home, "how do I delete my Facebook account?")
            assert reply is not None
            self.assertIn("medium", reply)
            self.assertIn("30 days", reply)
            self.assertIn("Export your data first", reply)

    def test_hook_walk_then_done(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            first = self._hook(home, "close my spotify")
            assert first is not None
            self.assertIn("step 1 of 5", first)
            second = self._hook(home, "done")
            assert second is not None
            self.assertIn("step 2 of 5", second)

    def test_hook_unknown_service_honest(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            reply = self._hook(
                home, "how do I delete my myspace account?")
            assert reply is not None
            self.assertIn("myspace", reply)
            self.assertIn("won't invent one", reply)

    def test_hook_ignores_non_requests(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            self.assertIsNone(self._hook(home, "close my laptop"))
            self.assertIsNone(self._hook(home, "how's your day going?"))
            self.assertIsNone(self._hook(home, "done"))


if __name__ == "__main__":
    unittest.main()
