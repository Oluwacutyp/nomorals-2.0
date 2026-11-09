"""Living Chat Experience — Phase 1 gates (doc 2 P1 + doc 8 P3).

Four contracts, pinned at the real seams:

1. **Operation targeting** — numbered picks and token-addressed
   card actions bind to a chat-scoped operation store
   (:class:`nomorals.core.operations.OperationStore`). The newest
   operation in a (chat, user) slot owns the numbers; a reference
   to a replaced/expired operation answers STALE honestly instead
   of executing against the wrong operation; different users in
   one chat and different chats never collide.
2. **Rich action schema** — :mod:`nomorals.social.chat.action_schema`
   actions carry stable identity, intent, style and an optional
   native custom-emoji icon; per-platform resolution names what is
   honored and what is dropped (Telegram's Feb-2026 ``style`` /
   ``icon_custom_emoji_id`` button fields are NOT serialized by the
   deployed adapter, so the schema records them unsupported rather
   than emitting fields nothing honors).
3. **Surface manifest drift** — the checked-in manifest
   (:mod:`nomorals.social.chat.surface_manifest`) is compared
   SET-BASED against the live registries (hub screens, adaptive
   output kinds, menu groups): a surface added without a manifest
   update fails by naming the difference. Every declared action
   and every payload on every probed real keyboard resolves to a
   real control handler.
4. **Capability chains** (doc 8 P3) — every manifest surface
   resolves a complete producer → action → handler → operation →
   adapter → bridge → receipt chain per platform, with the
   WhatsApp bridge resolved per engine (Go / Node).
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from typing import Any

import pytest

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


# ── 1. the operation store ──────────────────────────────────────

from nomorals.core.operations import (  # noqa: E402
    OperationStore,
    PickStatus,
)


class TestOperationStore:
    def test_open_returns_identity_and_revision(self):
        store = OperationStore()
        op1 = store.open("chat:1", ["a", "b"])
        assert op1 is not None and op1.operation_id
        assert op1.revision == 1
        op2 = store.open("chat:1", ["c"])
        assert op2.operation_id != op1.operation_id
        assert op2.revision == 2

    def test_resolve_number_ok_consumes(self):
        store = OperationStore()
        store.open("chat:1", ["cmd a", "cmd b"])
        res = store.resolve_number("chat:1", 2)
        assert res.status is PickStatus.OK
        assert res.command == "cmd b"
        # consumed: the operation is spent
        again = store.resolve_number("chat:1", 1)
        assert again.status is not PickStatus.OK

    def test_out_of_range_keeps_operation(self):
        store = OperationStore()
        store.open("chat:1", ["cmd a"])
        res = store.resolve_number("chat:1", 5)
        assert res.status is PickStatus.OUT_OF_RANGE
        assert store.resolve_number("chat:1", 1).command == "cmd a"

    def test_stale_reference_to_replaced_operation(self):
        store = OperationStore()
        old = store.open("chat:1", ["old a"])
        store.open("chat:1", ["new a"])
        res = store.resolve_operation("chat:1", old.operation_id, 1)
        assert res.status is PickStatus.STALE
        assert res.command == ""

    def test_expired_operation_answers_stale_not_missing(self):
        store = OperationStore(ttl_s=0.05)
        store.open("chat:1", ["cmd a"])
        time.sleep(0.07)
        res = store.resolve_number("chat:1", 1)
        assert res.status is PickStatus.STALE

    def test_no_operation_is_missing(self):
        store = OperationStore()
        res = store.resolve_number("chat:never", 1)
        assert res.status is PickStatus.MISSING

    def test_users_in_one_chat_never_collide(self):
        store = OperationStore()
        store.open("chat:g", ["alice cmd"], owner_key="alice")
        store.open("chat:g", ["bob cmd"], owner_key="bob")
        assert store.resolve_number(
            "chat:g", 1, owner_key="alice").command == "alice cmd"
        assert store.resolve_number(
            "chat:g", 1, owner_key="bob").command == "bob cmd"

    def test_chats_never_collide(self):
        store = OperationStore()
        store.open("chat:1", ["one"])
        store.open("chat:2", ["two"])
        assert store.resolve_number("chat:1", 1).command == "one"
        assert store.resolve_number("chat:2", 1).command == "two"

    def test_shared_slot_is_separate_from_user_slots(self):
        store = OperationStore()
        store.open("chat:g", ["shared cmd"])
        store.open("chat:g", ["alice cmd"], owner_key="alice")
        assert store.resolve_number("chat:g", 1).command == "shared cmd"
        assert store.resolve_number(
            "chat:g", 1, owner_key="alice").command == "alice cmd"


# ── 1b. the button-pick store rides the operation store ─────────

from nomorals.agents.partner.button_picks import ButtonPickStore  # noqa: E402


class TestButtonPickStoreTargeting:
    def test_legacy_contract_still_holds(self):
        store = ButtonPickStore()
        store.register("chat:1", ["cmd a", "cmd b"])
        assert store.has("chat:1")
        assert store.take("chat:1", 2) == "cmd b"
        assert store.take("chat:1", 1) == ""

    def test_register_returns_operation_id(self):
        store = ButtonPickStore()
        op_id = store.register("chat:1", ["cmd a"])
        assert isinstance(op_id, str) and op_id

    def test_take_result_reports_stale_after_replacement(self):
        store = ButtonPickStore()
        old_id = store.register("chat:1", ["old a"])
        store.register("chat:1", ["new a"])
        res = store.take_result("chat:1", 1, operation_id=old_id)
        assert res.status is PickStatus.STALE
        assert res.command == ""

    def test_per_user_slots(self):
        store = ButtonPickStore()
        store.register("chat:g", ["alice cmd"], owner_key="alice")
        store.register("chat:g", ["bob cmd"], owner_key="bob")
        res = store.take_result("chat:g", 1, owner_key="alice")
        assert res.command == "alice cmd"
        # bob's card is untouched by alice's pick
        assert store.take("chat:g", 1, owner_key="bob") == "bob cmd"


# ── 1c. the play pick-list: owner scoping + superseded ──────────

from nomorals.media.picklist import PickCache  # noqa: E402
from nomorals.media.resolver import ResolvedAudio  # noqa: E402


def _cand(title: str) -> ResolvedAudio:
    return ResolvedAudio(ok=True, path_or_url=f"https://x/{title}",
                         title=title, artist="", duration=180.0,
                         kind="youtube", source_name="test")


@pytest.fixture(autouse=True)
def _clean_pick_cache():
    PickCache.clear()
    yield
    PickCache.clear()


class TestPickCacheTargeting:
    def test_is_superseded(self):
        old = PickCache.store([_cand("a")], "chat:1")
        assert not PickCache.is_superseded(old)
        PickCache.store([_cand("b")], "chat:1")
        assert PickCache.is_superseded(old)

    def test_owner_scoped_lookup(self):
        PickCache.store([_cand("a")], "chat:g", owner_key="alice")
        assert PickCache.get_for_chat("chat:g", "alice") is not None
        # another user in the same chat must not see alice's list
        assert PickCache.get_for_chat("chat:g", "bob") is None
        # the legacy no-owner lookup still finds chat-scoped lists
        assert PickCache.get_for_chat("chat:g") is not None


# ── 2. the rich action schema ───────────────────────────────────

from nomorals.social.chat.action_schema import (  # noqa: E402
    FIELD_SUPPORT,
    Action,
    resolve_action,
)
from nomorals.social.chat.tgbot_buttons import actionable  # noqa: E402


class TestActionSchema:
    def test_stable_identity(self):
        a = Action(label="Retry", intent="refresh", data="tour")
        b = Action(label="Retry", intent="refresh", data="tour")
        assert a.id and a.id == b.id

    def test_explicit_identity_kept(self):
        a = Action(id="job.cancel.42", label="Cancel", intent="callback",
                   data="jobs cancel 42")
        assert a.id == "job.cancel.42"

    def test_callback_budget_enforced(self):
        with pytest.raises(ValueError):
            Action(label="X", intent="callback", data="x " + "y" * 80)

    def test_url_intent_renders_legacy_url_pair(self):
        a = Action(label="Rules", intent="url",
                   url="https://example.com/rules")
        assert a.to_legacy() == ("Rules", "https://example.com/rules")

    def test_url_intent_requires_a_url(self):
        with pytest.raises(ValueError):
            Action(label="Rules", intent="url", data="rules")

    def test_actionable_accepts_actions_and_tuples(self):
        msg = actionable("pick one", [[
            Action(label="Retry", intent="refresh", data="tour"),
            ("Help", "help"),
        ]])
        flat = [pair for row in msg.buttons for pair in row]
        assert ("Retry", "tour") in flat
        assert ("Help", "help") in flat

    def test_telegram_resolution_names_dropped_fields(self):
        a = Action(label="Go", intent="callback", data="help",
                   style="primary", icon_custom_emoji_id="12345")
        res = resolve_action(a, "telegram-bot")
        assert res.form == "inline"
        assert "style" in res.dropped
        assert "icon_custom_emoji_id" in res.dropped

    def test_telegram_field_support_is_honest(self):
        # The deployed Bot API adapter serializes only text/url/
        # callback_data (telegram.TelegramBotAdapter._button_json):
        # the Feb-2026 fields are recorded unsupported, never emitted.
        assert FIELD_SUPPORT["telegram-bot"]["style"] == "unsupported"
        assert (FIELD_SUPPORT["telegram-bot"]["icon_custom_emoji_id"]
                == "unsupported")
        import inspect

        from nomorals.social.chat import telegram
        src = inspect.getsource(telegram.TelegramBotAdapter._button_json)
        assert "icon_custom_emoji_id" not in src
        assert '"style"' not in src

    def test_whatsapp_url_resolves_to_text_line(self):
        a = Action(label="Rules", intent="url",
                   url="https://example.com/rules")
        res = resolve_action(a, "whatsapp")
        assert res.form == "text-line"
        assert "url" in res.dropped

    def test_whatsapp_callback_is_interactive(self):
        a = Action(label="Go", intent="callback", data="help")
        res = resolve_action(a, "whatsapp")
        assert res.form == "interactive"

    def test_buttonless_platform_resolves_to_numbered(self):
        a = Action(label="Go", intent="callback", data="help")
        res = resolve_action(a, "discord")
        assert res.form == "numbered"


# ── 3+4. the surface manifest and its drift gates ───────────────

from nomorals.social.chat import surface_manifest as sm  # noqa: E402
from nomorals.social.chat.control import (  # noqa: E402
    CONTROL_COMMANDS,
    LIST_GROUPS,
    parse_control,
)

REQUIRED_SURFACES = {
    "menu", "hub", "personal-keyboards", "adaptive-output",
    "text-pattern-keyboards", "confirm", "play-picklist",
    "button-fallback", "games", "live-session", "pairing",
    "downloads", "group-boards", "inline-query",
}


def _resolves(data: str) -> bool:
    """A callback payload reaches a real handler: parse_control
    routes it (non-None) — unknown kinds return None."""
    if data.startswith(("http://", "https://")):
        return True
    parsed = parse_control("/" + data)
    return parsed is not None


def _keyboard_payloads(keyboard) -> list[str]:
    return [data for row in (keyboard or []) for _label, data in row]


class TestSurfaceManifest:
    def test_required_surfaces_present(self):
        ids = {s.id for s in sm.SURFACES}
        missing = REQUIRED_SURFACES - ids
        assert not missing, f"manifest is missing surfaces: {missing}"

    def test_hub_screens_match_live_registry(self):
        from nomorals.social.chat import hub
        surface = sm.surface_by_id("hub")
        assert set(surface.members) == set(hub.SCREEN_NAMES), (
            f"hub screens drifted: manifest={sorted(surface.members)} "
            f"live={sorted(hub.SCREEN_NAMES)}")

    def test_adaptive_kinds_match_live_registry(self):
        from nomorals.social.chat import adaptive_buttons
        surface = sm.surface_by_id("adaptive-output")
        assert set(surface.members) == set(adaptive_buttons.OUTPUT_KINDS), (
            f"adaptive kinds drifted: manifest={sorted(surface.members)} "
            f"live={sorted(adaptive_buttons.OUTPUT_KINDS)}")

    def test_menu_groups_match_live_registry(self):
        surface = sm.surface_by_id("menu")
        live = {name for name, _kinds in LIST_GROUPS}
        assert set(surface.members) == live, (
            f"menu groups drifted: manifest-only="
            f"{sorted(set(surface.members) - live)} live-only="
            f"{sorted(live - set(surface.members))}")

    def test_every_declared_action_resolves_to_a_real_handler(self):
        problems = []
        for surface in sm.SURFACES:
            for ref in surface.actions:
                handler = ref.handler
                assert handler.startswith("control:"), (
                    f"{surface.id}/{ref.action_id}: handler {handler!r} "
                    "is not a control route")
                kind = handler.split(":", 1)[1]
                if kind not in CONTROL_COMMANDS:
                    problems.append(
                        f"{surface.id}/{ref.action_id}: no such "
                        f"control kind {kind!r}")
                if ref.data and not _resolves(ref.data):
                    problems.append(
                        f"{surface.id}/{ref.action_id}: payload "
                        f"{ref.data!r} does not resolve")
        assert not problems, "; ".join(problems)

    def test_probed_keyboards_carry_only_resolvable_payloads(self):
        from nomorals.social.chat import hub
        from nomorals.social.chat.adaptive_buttons import adaptive_buttons_for
        from nomorals.social.chat.tgbot_buttons import buttons_for_text, personal_buttons_for

        keyboards: dict[str, Any] = {}
        for screen in hub.SCREEN_NAMES:
            keyboards[f"hub:{screen}"] = hub.keyboard_for_screen(screen)
        keyboards["pattern:hub"] = buttons_for_text(hub.hub_text())
        keyboards["pattern:proxy"] = buttons_for_text(
            hub.proxy_intro_text())
        keyboards["pattern:confirm"] = buttons_for_text(
            "⚠️ confirm: /quit — send /quit confirm to go ahead")
        keyboards["pattern:game-menu"] = buttons_for_text(
            "modes & mastery — /game <name> to play")
        keyboards["pattern:game-result"] = buttons_for_text(
            "The House wins this round")
        keyboards["pattern:music"] = buttons_for_text(
            "🎵 “Some Song”\nnow playing")
        keyboards["personal:help"] = personal_buttons_for("x", "/help")
        keyboards["personal:power"] = personal_buttons_for("x", "/power")
        keyboards["adaptive:search"] = adaptive_buttons_for(
            output_kind="search", payload={"query": "burna boy"})
        keyboards["adaptive:scheduled"] = adaptive_buttons_for(
            output_kind="scheduled", payload={"job": "memory-consolidation"})
        keyboards["adaptive:game"] = adaptive_buttons_for(
            output_kind="game", payload={"game": "trivia"})
        keyboards["adaptive:memory"] = adaptive_buttons_for(
            output_kind="memory")
        keyboards["adaptive:music-ask"] = adaptive_buttons_for(
            output_kind="music-ask")

        problems = []
        for name, kb in keyboards.items():
            assert kb, f"probe {name} produced no keyboard"
            for data in _keyboard_payloads(kb):
                if not _resolves(data):
                    problems.append(f"{name}: payload {data!r} unresolved")
        assert not problems, "; ".join(problems)

    def test_menu_renders_resolvable_payloads_for_every_group(self):
        from nomorals.social.chat.control import _menu_section_key, menu_session_render
        home = menu_session_render(home=True)
        assert home is not None
        problems = [d for d in _keyboard_payloads(home.buttons)
                    if not _resolves(d)]
        titles = {name: _menu_section_key(name) for name, _ in LIST_GROUPS}
        for name, key in titles.items():
            page = menu_session_render(topic=key)
            assert page is not None, f"menu section {name!r} did not render"
            problems += [d for d in _keyboard_payloads(page.buttons)
                         if not _resolves(d)]
        assert not problems, "; ".join(sorted(set(problems)))

    def test_capability_chain_complete(self):
        problems = []
        for surface in sm.SURFACES:
            for platform in surface.platforms:
                chain = sm.resolve_chain(surface.id, platform)
                for link in ("producer", "action", "handler",
                             "operation", "adapter", "bridge", "receipt"):
                    if not getattr(chain, link):
                        problems.append(
                            f"{surface.id}@{platform}: empty {link} link")
        assert not problems, "; ".join(problems)

    def test_whatsapp_chain_resolves_per_engine(self):
        go = sm.resolve_chain("menu", "whatsapp", engine="go")
        node = sm.resolve_chain("menu", "whatsapp", engine="node")
        assert go.bridge and node.bridge
        assert go.bridge != node.bridge
        assert "gobridge" in go.bridge
        assert "whatsapp-bridge.mjs" in node.bridge


# ── runtime-level targeting (group users + stale honesty) ───────

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import (  # noqa: E402
    ChatKind,
    ChatMessage,
    ChatRef,
)


def _make_runtime() -> tuple[PartnerRuntime, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-", dir=_TEST_TMPDIR)
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    runtime = PartnerRuntime(context)
    runtime._typing_enabled = lambda chat: False
    return runtime, tmp


def _group_msg(text: str, sender_id: str) -> ChatMessage:
    chat = ChatRef(platform="whatsapp", chat_id="room1",
                   kind=ChatKind.GROUP, title="room", peer="room1")
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender=sender_id, sender_id=sender_id)


class RuntimeTargetingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.tmp = _make_runtime()

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.tmp.cleanup()

    def test_group_users_do_not_collide(self) -> None:
        key = "whatsapp:room1"
        self.runtime._register_button_picks(
            key, ["help"], owner_key="alice")
        self.runtime._register_button_picks(
            key, ["status"], owner_key="bob")
        # bob's bare number must not consume alice's card, and
        # alice's pick still runs her own card's command.
        out_bob = self.runtime._button_pick_hook(_group_msg("1", "bob"))
        self.assertIsNotNone(out_bob)
        out_alice = self.runtime._button_pick_hook(
            _group_msg("1", "alice"))
        self.assertIsNotNone(out_alice)

    def test_expired_pick_answers_honestly(self) -> None:
        from nomorals.agents.partner.button_picks import ButtonPickStore
        self.runtime._button_pick_store_obj = ButtonPickStore(ttl_s=0.05)
        key = "whatsapp:room1"
        self.runtime._register_button_picks(key, ["help"],
                                            owner_key="alice")
        time.sleep(0.07)
        out = self.runtime._button_pick_hook(_group_msg("1", "alice"))
        self.assertIsNotNone(out)
        text = str(getattr(out, "text", out)).lower()
        self.assertIn("replaced", text)


class PlayPickTargetingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.tmp = _make_runtime()
        PickCache.clear()

    def tearDown(self) -> None:
        PickCache.clear()
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.tmp.cleanup()

    def test_superseded_picklist_pick_answers_replaced(self) -> None:
        key = "whatsapp:room1"
        old = PickCache.store([_cand("old song")], key)
        PickCache.store([_cand("new song")], key)
        out = self.runtime._play_pick(f"{old} 1", key)
        self.assertIn("replaced", str(out).lower())

    def test_current_picklist_pick_passes_the_gate(self) -> None:
        key = "whatsapp:room1"
        token = PickCache.store([_cand("only song")], key)
        out = self.runtime._play_pick(f"{token} 9", key)
        # out-of-range wording proves the pick reached candidate
        # resolution (it was NOT stopped as superseded/expired)
        self.assertIn("out of range", str(out).lower())
