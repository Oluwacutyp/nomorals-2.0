"""Living Chat Experience — Phase 2 gates (doc 2 P2 + doc 4 journeys).

Contracts pinned here:

1. **Living cards** — a card carries identity + revision and is
   DURABLE (the runtime-created ``living_cards`` table, Batch B
   pattern — migrations.py untouched). Content updates EDIT the
   recorded message in place on surfaces whose adapter supports
   text edits; on surfaces without an edit seam the update is an
   honest superseding send and the record moves. The mechanism is
   recorded per platform and agrees with platform_limits evidence.
2. **Revision fencing** — a card's buttons carry (card, revision,
   index). A tap at the current revision resolves the action's
   command; a tap against a superseded revision answers STALE
   with the CURRENT card state and never yields the command; an
   unknown card answers MISSING; another owner's card is not
   addressable.
3. **Restart-resumable** — the card record alone (fresh stores,
   same database) is enough for the first post-restart tap to
   work and for a post-restart revision to edit the ORIGINAL
   message.
4. **Progress tenant** — the download progress card renders from
   the measured progress stream's snapshots (core.progress_events).
5. **Startup journey** (doc 4) — the readiness card's verdict is
   computed from real facts: recovery-required beats degraded
   beats ready, and every row carries its evidence.
6. **Restart journey** (doc 4) — the dying process's side of the
   card can never write "confirmed"; only the new-process
   finalizer, reading the restart-truth receipt, closes the card.
7. **Operation identity** — the operation store accepts a
   caller-named id so a durable card and its operation share one
   identity (closes the P1 seam additively).
8. **Blocked-matches picklist owner scope** — the offer stores
   its pick-list under the requesting user (the dispatch's
   play_owner_var), like every other pick surface after P1.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="nm-test-", dir=_TEST_TMPDIR)


# ── 1+2. the card model, the durable store, fencing ─────────────

from nomorals.core.operations import OperationStore  # noqa: E402
from nomorals.social.chat.action_schema import Action  # noqa: E402
from nomorals.social.chat.living_cards import (  # noqa: E402
    CARD_UPDATE_FORMS,
    Card,
    CardHost,
    CardStore,
    TapStatus,
    render_card,
    resolve_card_tap,
    update_form,
)
from nomorals.storage.db import Database  # noqa: E402


def _card(**kw: Any) -> Card:
    base = dict(
        card_id="c0ffee42", family="generic",
        chat_key="telegram-bot:42", owner_key="owner1",
        platform="telegram-bot", chat_id="42",
        title="Deploy status", body="working on it",
        actions=(Action(label="Platforms", data="platforms"),
                 Action(label="Status", data="status")))
    base.update(kw)
    return Card(**base)


class TestCardStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmpdir()
        self.db = Database(os.path.join(self.tmp.name, "cards.db"))
        self.store = CardStore(self.db)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_trip(self) -> None:
        card = _card(rows=(("stage", "swap"),), state={"k": 1})
        self.assertTrue(self.store.save(card))
        back = self.store.get("c0ffee42")
        self.assertIsNotNone(back)
        assert back is not None
        self.assertEqual(back.title, "Deploy status")
        self.assertEqual(back.rows, (("stage", "swap"),))
        self.assertEqual(back.state, {"k": 1})
        self.assertEqual(back.actions[0].data, "platforms")
        self.assertEqual(back.revision, 1)

    def test_revision_cas_refuses_stale_save(self) -> None:
        card = _card()
        self.assertTrue(self.store.save(card))
        newer = _card(revision=2, body="second")
        self.assertTrue(self.store.save(newer, expect_revision=1))
        stale = _card(revision=2, body="stale overwrite")
        self.assertFalse(self.store.save(stale, expect_revision=1))
        back = self.store.get("c0ffee42")
        assert back is not None
        self.assertEqual(back.body, "second")
        self.assertEqual(back.revision, 2)

    def test_latest_and_live_for_chat(self) -> None:
        self.store.save(_card(card_id="aaaa0001"))
        self.store.save(_card(card_id="aaaa0002", terminal=True))
        live = self.store.live_for_chat("telegram-bot:42")
        self.assertEqual([c.card_id for c in live], ["aaaa0001"])
        latest = self.store.latest("generic", "telegram-bot:42")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.card_id, "aaaa0002")


class TestOperationNamedId(unittest.TestCase):
    def test_open_with_caller_named_id(self) -> None:
        store = OperationStore()
        op = store.open("chat:1", ["platforms"],
                        operation_id="c0ffee42", kind="living-card")
        self.assertIsNotNone(op)
        assert op is not None
        self.assertEqual(op.operation_id, "c0ffee42")
        res = store.resolve_operation("chat:1", "c0ffee42", 1)
        self.assertEqual(res.command, "platforms")


class TestRenderForms(unittest.TestCase):
    def test_telegram_keyboard_carries_fenced_data(self) -> None:
        out = render_card(_card(), "telegram-bot")
        self.assertEqual(out.form, "inline")
        datas = [d for row in out.keyboard for _l, d in row]
        self.assertEqual(
            datas, ["cardtap c0ffee42 1 1", "cardtap c0ffee42 1 2"])

    def test_whatsapp_form_is_interactive(self) -> None:
        out = render_card(_card(platform="whatsapp"), "whatsapp")
        self.assertEqual(out.form, "interactive")
        datas = [d for row in out.keyboard for _l, d in row]
        self.assertEqual(datas[0], "cardtap c0ffee42 1 1")

    def test_buttonless_platform_gets_numbered_fallback(self) -> None:
        out = render_card(_card(platform="discord"), "discord")
        self.assertEqual(out.form, "numbered")
        self.assertEqual(out.keyboard, [])
        self.assertIn("1 — Platforms", out.text)
        self.assertIn("2 — Status", out.text)

    def test_declared_style_and_icon_are_dropped_not_emitted(self) -> None:
        from nomorals.social.chat.action_schema import resolve_action
        action = Action(label="Go", data="status", style="primary",
                        icon_custom_emoji_id="123")
        resolved = resolve_action(action, "telegram-bot")
        self.assertIn("style", resolved.dropped)
        self.assertIn("icon_custom_emoji_id", resolved.dropped)
        out = render_card(_card(actions=(action,)), "telegram-bot")
        for row in out.keyboard:
            for pair in row:
                self.assertEqual(len(pair), 2)

    def test_update_form_matches_platform_limit_evidence(self) -> None:
        from nomorals.social.chat.platform_limits import for_platform
        for platform, form in CARD_UPDATE_FORMS.items():
            limits = for_platform(platform)
            expected = "edit" if limits.supports_text_edit \
                else "supersede"
            self.assertEqual(form, expected, platform)
        self.assertEqual(update_form("telegram-bot"), "edit")
        self.assertEqual(update_form("whatsapp"), "edit")
        self.assertEqual(update_form("discord"), "supersede")
        self.assertEqual(update_form("sms"), "supersede")


class TestTapFencing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmpdir()
        self.db = Database(os.path.join(self.tmp.name, "cards.db"))
        self.store = CardStore(self.db)
        self.store.save(_card())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_current_revision_tap_resolves_command(self) -> None:
        out = resolve_card_tap(self.store, "c0ffee42", 1, 2,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.OK)
        self.assertEqual(out.command, "status")

    def test_stale_revision_tap_answers_with_current_card(self) -> None:
        self.store.save(_card(revision=2, body="moved on"),
                        expect_revision=1)
        out = resolve_card_tap(self.store, "c0ffee42", 1, 1,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.STALE)
        self.assertEqual(out.command, "")
        self.assertIsNotNone(out.card)
        assert out.card is not None
        self.assertEqual(out.card.body, "moved on")
        self.assertEqual(out.card.revision, 2)

    def test_unknown_card_is_missing(self) -> None:
        out = resolve_card_tap(self.store, "deadbeef", 1, 1,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.MISSING)

    def test_out_of_range_does_not_burn_the_card(self) -> None:
        out = resolve_card_tap(self.store, "c0ffee42", 1, 9,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.OUT_OF_RANGE)
        again = resolve_card_tap(self.store, "c0ffee42", 1, 1,
                                 chat_key="telegram-bot:42",
                                 owner_key="owner1")
        self.assertEqual(again.status, TapStatus.OK)

    def test_another_owners_card_is_not_addressable(self) -> None:
        out = resolve_card_tap(self.store, "c0ffee42", 1, 1,
                               chat_key="telegram-bot:42",
                               owner_key="intruder")
        self.assertEqual(out.status, TapStatus.MISSING)


# ── 3. the host: edit in place, supersede, restart-resume ───────


class _Transport:
    """A fake chat surface: records sends and edits."""

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.fail_edits = False
        self._n = 0

    def send(self, platform: str, chat: Any, text: str,
             buttons: Any = None) -> tuple[bool, str]:
        self._n += 1
        mid = f"m{self._n}"
        self.sends.append({"platform": platform, "text": text,
                           "buttons": buttons, "message_id": mid})
        return True, mid

    def edit(self, chat: Any, message_id: str, text: str,
             buttons: Any = None) -> bool:
        self.edits.append({"message_id": message_id, "text": text,
                           "buttons": buttons})
        return not self.fail_edits


class _Chat:
    platform = "telegram-bot"
    chat_id = "42"
    key = "telegram-bot:42"


class _DiscordChat(_Chat):
    platform = "discord"
    key = "discord:42"


class TestCardHost(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmpdir()
        self.db_path = os.path.join(self.tmp.name, "cards.db")
        self.store = CardStore(Database(self.db_path))
        self.transport = _Transport()
        self.host = CardHost(self.store, send=self.transport.send,
                             edit=self.transport.edit)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_publish_then_revise_edits_in_place(self) -> None:
        card = self.host.publish(_card(), _Chat())
        self.assertTrue(card.delivered)
        self.assertEqual(card.message_id, "m1")
        self.assertEqual(len(self.transport.sends), 1)
        revised = self.host.revise(card, _Chat(), body="halfway")
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.message_id, "m1")
        self.assertEqual(len(self.transport.sends), 1)
        self.assertEqual(len(self.transport.edits), 1)
        self.assertEqual(self.transport.edits[0]["message_id"], "m1")
        self.assertIn("halfway", self.transport.edits[0]["text"])

    def test_failed_edit_falls_back_to_a_fresh_send(self) -> None:
        card = self.host.publish(_card(), _Chat())
        self.transport.fail_edits = True
        revised = self.host.revise(card, _Chat(), body="again")
        self.assertEqual(revised.message_id, "m2")
        self.assertEqual(len(self.transport.sends), 2)
        back = self.store.get("c0ffee42")
        assert back is not None
        self.assertEqual(back.message_id, "m2")

    def test_supersede_platform_never_edits(self) -> None:
        chat = _DiscordChat()
        card = self.host.publish(
            _card(platform="discord", chat_key="discord:42"), chat)
        revised = self.host.revise(card, chat, body="v2")
        self.assertEqual(self.transport.edits, [])
        self.assertEqual(len(self.transport.sends), 2)
        self.assertEqual(revised.message_id, "m2")

    def test_restart_resume_from_the_durable_record_alone(self) -> None:
        card = self.host.publish(_card(), _Chat())
        self.host.revise(card, _Chat(), body="before the restart")
        # Simulate the restart: brand-new store + host + transport
        # over the SAME database file; nothing in memory survives.
        store2 = CardStore(Database(self.db_path))
        transport2 = _Transport()
        host2 = CardHost(store2, send=transport2.send,
                         edit=transport2.edit)
        # The first post-restart interaction: a tap at the current
        # revision works from the durable record alone.
        out = resolve_card_tap(store2, "c0ffee42", 2, 1,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.OK)
        self.assertEqual(out.command, "platforms")
        # A post-restart revision edits the ORIGINAL message.
        back = store2.get("c0ffee42")
        assert back is not None
        revised = host2.revise(back, _Chat(), body="after restart")
        self.assertEqual(revised.message_id, "m1")
        self.assertEqual(transport2.edits[0]["message_id"], "m1")
        self.assertEqual(transport2.sends, [])
        # And a tap cut before that revision is now stale.
        out = resolve_card_tap(store2, "c0ffee42", 2, 1,
                               chat_key="telegram-bot:42",
                               owner_key="owner1")
        self.assertEqual(out.status, TapStatus.STALE)


# ── 4. the progress tenant ──────────────────────────────────────

from nomorals.core.progress_events import (  # noqa: E402
    DOWNLOADING,
    SEARCHING,
    ProgressTracker,
)
from nomorals.social.chat.living_cards import (  # noqa: E402
    apply_progress_snapshot,
    progress_card,
)


class TestProgressTenant(unittest.TestCase):
    def test_card_revises_through_the_measured_stages(self) -> None:
        tracker = ProgressTracker("dl1234", title="some song")
        card = progress_card(tracker, chat_key="whatsapp:7",
                             platform="whatsapp", chat_id="7")
        self.assertEqual(card.card_id, "dl1234")
        self.assertEqual(card.family, "progress")
        tracker.set_stage(SEARCHING)
        apply_progress_snapshot(card, tracker.snapshot())
        out = render_card(card, "whatsapp")
        self.assertIn("some song", out.text)
        tracker.set_stage(DOWNLOADING)
        tracker.report_bytes(512, 1024)
        apply_progress_snapshot(card, tracker.snapshot())
        out = render_card(card, "whatsapp")
        self.assertIn("50%", out.text)
        tracker.finish("delivered")
        apply_progress_snapshot(card, tracker.snapshot())
        self.assertTrue(card.terminal)
        out = render_card(card, "whatsapp")
        self.assertIn("delivered", out.text)


# ── 5. the startup journey ──────────────────────────────────────

from nomorals.social.chat.living_cards import (  # noqa: E402
    build_readiness_card,
    readiness_verdict,
)


def _facts(**kw: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "storage": True,
        "recovery": None,
        "platforms": [("telegram-bot", "up", ""),
                      ("whatsapp", "up", "")],
        "brain": ("ready", ""),
        "scheduler": ("running", ""),
        "agency": {"gateway": "ready", "search": "ready: live",
                   "gigs": "ready", "research": "ready",
                   "wisdom": "ready", "impossible": "ready",
                   "deals": "delegated: no synchronous hunter",
                   "planner": "delegated: injected graphs",
                   "expansion": "delegated: own schedules"},
        "missions": None,
    }
    facts.update(kw)
    return facts


class TestStartupJourney(unittest.TestCase):
    def test_ready_when_everything_is_up(self) -> None:
        self.assertEqual(readiness_verdict(_facts()), "ready")
        card = build_readiness_card(_facts())
        self.assertEqual(card.family, "readiness")
        self.assertEqual(card.status, "ready")

    def test_degraded_when_a_platform_is_down(self) -> None:
        facts = _facts(platforms=[("telegram-bot", "up", ""),
                                  ("whatsapp", "down", "bridge down")])
        self.assertEqual(readiness_verdict(facts), "degraded")
        card = build_readiness_card(facts)
        text = render_card(card, "telegram-bot").text
        self.assertIn("whatsapp", text)
        self.assertIn("bridge down", text)

    def test_degraded_when_the_brain_is_degraded(self) -> None:
        facts = _facts(brain=("degraded", "last reply held"))
        self.assertEqual(readiness_verdict(facts), "degraded")

    def test_degraded_when_agency_reports_unavailable(self) -> None:
        agency = dict(_facts()["agency"])
        agency["research"] = "unavailable: construction failed"
        self.assertEqual(readiness_verdict(_facts(agency=agency)),
                         "degraded")

    def test_recovery_required_beats_degraded(self) -> None:
        facts = _facts(
            recovery={"recovered": True, "action": "rollback_failed"},
            platforms=[("whatsapp", "down", "bridge down")])
        self.assertEqual(readiness_verdict(facts),
                         "recovery-required")
        card = build_readiness_card(facts)
        self.assertEqual(card.status, "recovery-required")

    def test_unconfirmed_restart_is_recovery_required(self) -> None:
        facts = _facts(recovery={"action": "restart_unconfirmed"})
        self.assertEqual(readiness_verdict(facts),
                         "recovery-required")

    def test_clean_recovery_is_information_not_a_verdict(self) -> None:
        facts = _facts(recovery={"recovered": True,
                                 "action": "rolled_back"})
        self.assertEqual(readiness_verdict(facts), "ready")


# ── 6. the restart journey ──────────────────────────────────────

from nomorals.social.chat.living_cards import (  # noqa: E402
    advance_restart_card,
    build_restart_card,
    finalize_restart_card,
    restart_phase_lines,
)


class _Report:
    ok = True
    steps = [{"name": "preflight", "ok": True},
             {"name": "pull", "ok": True},
             {"name": "health", "ok": True}]
    to_sha = "abcdef1234567890"


class TestRestartJourney(unittest.TestCase):
    def test_phases_progress_and_the_dying_side_never_confirms(self) -> None:
        card = build_restart_card(chat_key="telegram-bot:42",
                                  platform="telegram-bot",
                                  chat_id="42", kind="update")
        lines = restart_phase_lines(card)
        self.assertIn("preflight", lines[0])
        card = advance_restart_card(card, _Report())
        self.assertFalse(card.terminal)
        self.assertNotEqual(card.status, "confirmed")
        lines = restart_phase_lines(card)
        joined = "\n".join(lines)
        self.assertIn("restart", joined)
        # Even a forced advance on the dying side stops at handoff.
        card = advance_restart_card(card, _Report(), phase="restart")
        self.assertFalse(card.terminal)
        self.assertNotEqual(card.status, "confirmed")

    def test_finalizer_confirms_from_the_receipt(self) -> None:
        card = build_restart_card(chat_key="telegram-bot:42",
                                  platform="telegram-bot",
                                  chat_id="42", kind="update")
        card = advance_restart_card(card, _Report(), phase="restart")
        card.state["to_sha"] = "abcdef1234567890"
        done = finalize_restart_card(card, {
            "action": "restart_confirmed",
            "running_sha": "abcdef1234567890",
            "to_sha": "abcdef1234567890"})
        self.assertTrue(done.terminal)
        self.assertEqual(done.status, "confirmed")
        self.assertIn("abcdef123456", done.body)

    def test_finalizer_reports_unconfirmed_honestly(self) -> None:
        card = build_restart_card(chat_key="telegram-bot:42",
                                  platform="telegram-bot",
                                  chat_id="42", kind="update")
        done = finalize_restart_card(card, {
            "action": "restart_unconfirmed",
            "running_sha": "999999999999",
            "to_sha": "abcdef1234567890"})
        self.assertTrue(done.terminal)
        self.assertEqual(done.status, "attention")
        self.assertIn("NOT confirmed", done.body)


# ── 7. runtime wiring: the cardtap interception ─────────────────

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import (  # noqa: E402
    ChatKind,
    ChatMessage,
    ChatRef,
)


def _make_runtime() -> tuple[PartnerRuntime, tempfile.TemporaryDirectory]:
    tmp = _tmpdir()
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


def _dm_msg(text: str, sender_id: str = "owner1") -> ChatMessage:
    chat = ChatRef(platform="telegram-bot", chat_id="42",
                   kind=ChatKind.DM, title="dm", peer="42")
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender=sender_id, sender_id=sender_id)


class TestRuntimeCardTap(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.tmp = _make_runtime()
        self.store = self.runtime._living_card_store()
        self.store.save(_card())

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.tmp.cleanup()

    def test_tap_at_current_revision_dispatches_the_command(self) -> None:
        out = self.runtime.handle_control(
            "/cardtap c0ffee42 1 1", "telegram-bot:42",
            message=_dm_msg("/cardtap c0ffee42 1 1"))
        self.assertTrue(str(out).strip())
        self.assertNotIn("moved on", str(out))

    def test_stale_tap_answers_with_the_current_card(self) -> None:
        self.store.save(_card(revision=2, body="the new state"),
                        expect_revision=1)
        out = self.runtime.handle_control(
            "/cardtap c0ffee42 1 1", "telegram-bot:42",
            message=_dm_msg("/cardtap c0ffee42 1 1"))
        self.assertIn("moved on", str(out))
        self.assertIn("the new state", str(out))

    def test_unknown_card_tap_answers_honestly(self) -> None:
        out = self.runtime.handle_control(
            "/cardtap deadbeef 1 1", "telegram-bot:42",
            message=_dm_msg("/cardtap deadbeef 1 1"))
        self.assertIn("gone", str(out).lower())


# ── 8. the blocked-matches picklist owner scope ─────────────────


class TestPicklistOfferOwnerScope(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.media.picklist import PickCache
        PickCache.clear()

    def tearDown(self) -> None:
        from nomorals.media.picklist import PickCache
        PickCache.clear()

    def test_offer_binds_to_the_requesting_user(self) -> None:
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import (
            RuntimeMediaMixin,
            play_owner_var,
        )
        from nomorals.media.picklist import PickCache

        mixin = RuntimeMediaMixin()
        chat = SimpleNamespace(key="whatsapp:room1",
                               platform="whatsapp", chat_id="room1")
        items = [{"path": "https://example.invalid/song.mp3",
                  "title": "Some Song", "artist": "An Artist",
                  "duration": 200.0, "kind": "url"}]
        token = play_owner_var.set("u1")
        try:
            offer = mixin._play_picklist_offer(
                items, [], [], "some song", chat)
        finally:
            play_owner_var.reset(token)
        self.assertTrue(offer)
        found = PickCache.get_for_chat("whatsapp:room1",
                                       owner_key="u1")
        self.assertIsNotNone(found)
        assert found is not None
        # The entry itself is BOUND to the requester: another
        # user's bare number in the same chat must not resolve it
        # (on the pre-P2 code the entry carried no owner, so any
        # user's number in the group consumed the offer).
        self.assertEqual(found[1].owner_key, "u1")
        self.assertIsNone(
            PickCache.get_for_chat("whatsapp:room1", owner_key="u2"))


if __name__ == "__main__":
    unittest.main()
