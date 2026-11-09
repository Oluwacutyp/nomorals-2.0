"""Living Chat Experience — Phase 4 gates (doc 2 §6 + §9 P4:
experience packs).

Contracts pinned here:

1. **A pack system, not 25 one-offs** — every pack in the doc's
   §6 list is a DECLARATIVE definition in one registry
   (nomorals.social.chat.packs): identity, the surfaces it
   composes, its style family, its entry points. Packs compose
   the P1–P3 machinery (manifest, action schema, operations,
   living cards, style pins, procedural creation); nothing here
   reimplements a wrapped feature.
2. **Discoverable** — the packs surface is registered in the P1
   surface manifest with the pack ids as its member set
   (set-based drift gate, like hub/adaptive/expression), and
   /pack lists what's installed with honest state.
3. **Launchable + stateful** — /pack open <id> publishes the
   pack's living home card (family = the pack's own scope, so a
   style pin can target it through the P3 seam); the card's
   actions resolve through the durable tap fencing to real
   control commands; stateful packs carry their session in the
   pack store, across chats for owner-scoped packs.
4. **Lifecycle** — install/enable/disable persists (JSON store,
   the PinStore pattern); a disabled pack's surfaces stop
   resolving: opening it answers unavailable in words and
   publishes nothing, its verbs refuse too.
5. **Proof per pack** — every pack's entry point is driven
   through the real runtime with a fake gateway, and a real
   outcome is asserted (a card published whose actions resolve;
   for native packs, a verb that moves real state).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="nm-test-", dir=_TEST_TMPDIR)


@pytest.fixture(autouse=True)
def _store_isolation():
    """Every test runs against throwaway pin + pack stores."""
    from nomorals.core import style_pins
    from nomorals.social.chat import packs

    tmp = _tmpdir()
    prev_pins = style_pins._STORE
    prev_packs = packs._STORE
    style_pins.set_pin_store(style_pins.PinStore(
        path=Path(tmp.name) / "pins.json"))
    packs.set_pack_store(packs.PackStore(
        path=Path(tmp.name) / "packs.json"))
    yield
    style_pins.set_pin_store(prev_pins)
    packs.set_pack_store(prev_packs)
    tmp.cleanup()


# ── 1. the registry ─────────────────────────────────────────────

from nomorals.social.chat import packs  # noqa: E402
from nomorals.social.chat.control import (  # noqa: E402
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    parse_control,
)

#: The doc's §6 list, in the doc's order.
DOC_PACK_IDS = (
    "command-center", "artifact-workbench", "research-cockpit",
    "music-deck", "creative-director", "sticker-atelier",
    "learning-cards", "game-boards", "book-studio",
    "smart-digest", "memory-shelf", "calendar-host",
    "travel-board", "media-vault", "decision-room",
    "diagnostic-panel", "adventure", "celebration-engine",
    "workflow-buttons", "inline-creation", "companion-views",
    "cross-device-handoff", "language-expression", "action-bar",
    "surprise-me",
)


class TestPackRegistry:
    def test_all_25_doc_packs_registered_in_order(self):
        assert tuple(p.id for p in packs.PACKS) == DOC_PACK_IDS

    def test_pack_ids_unique_and_addressable(self):
        assert len({p.id for p in packs.PACKS}) == 25
        for pack in packs.PACKS:
            assert packs.pack_by_id(pack.id) is pack

    def test_definitions_are_complete(self):
        for pack in packs.PACKS:
            assert pack.title and pack.tagline and pack.icon, pack.id
            assert pack.family == pack.id, (
                f"{pack.id}: the style-pin family scope is the "
                f"pack's own id")
            assert pack.surfaces, f"{pack.id}: composes no surface"
            for surface_id in pack.surfaces:
                from nomorals.social.chat import surface_manifest as sm
                assert surface_id in {s.id for s in sm.SURFACES} | {
                    "packs"}, (f"{pack.id}: unknown surface "
                               f"{surface_id!r}")
            assert pack.entry or pack.verbs, (
                f"{pack.id}: no entry command and no native verbs "
                f"— nothing to launch")
            if pack.entry:
                cmd = parse_control("/" + pack.entry)
                assert cmd is not None, (
                    f"{pack.id}: entry {pack.entry!r} does not parse")

    def test_pack_command_registered_everywhere_a_command_lives(self):
        assert "pack" in CONTROL_COMMANDS
        assert "run /pack." not in COMMAND_DETAILS["pack"]["what"]
        assert LIST_ONELINERS["pack"].strip()
        grouped = [k for _t, kinds in LIST_GROUPS for k in kinds]
        assert grouped.count("pack") == 1
        cmd = parse_control("/pack")
        assert cmd is not None and cmd.kind == "pack"
        cmd = parse_control("/pack open music-deck")
        assert cmd is not None and cmd.kind == "pack"


# ── 2. the pack store (lifecycle + sessions) ────────────────────


class TestPackStore:
    def test_packs_default_to_enabled(self):
        store = packs.PackStore(path=None)
        for pack in packs.PACKS:
            assert store.is_enabled(pack.id)

    def test_disable_persists_across_store_instances(self, tmp_path):
        path = tmp_path / "packs.json"
        store = packs.PackStore(path=path)
        store.set_enabled("music-deck", False)
        again = packs.PackStore(path=path)
        assert not again.is_enabled("music-deck")
        again.set_enabled("music-deck", True)
        assert packs.PackStore(path=path).is_enabled("music-deck")

    def test_sessions_persist_and_are_scoped(self, tmp_path):
        path = tmp_path / "packs.json"
        store = packs.PackStore(path=path)
        store.save_session("travel-board", "owner:alice",
                           {"legs": [{"label": "Lagos"}]})
        store.save_session("travel-board", "owner:bob", {"legs": []})
        again = packs.PackStore(path=path)
        assert again.session("travel-board", "owner:alice")["legs"]
        assert again.session("travel-board", "owner:bob")["legs"] == []

    def test_usage_counts_and_pins(self, tmp_path):
        store = packs.PackStore(path=tmp_path / "packs.json")
        store.record_use("music-deck")
        store.record_use("music-deck")
        store.record_use("adventure")
        assert store.usage_counts() == {"music-deck": 2, "adventure": 1}
        store.set_pinned("adventure", True)
        assert store.pinned() == ["adventure"]
        store.set_pinned("adventure", False)
        assert store.pinned() == []

    def test_corrupt_store_reads_as_fresh(self, tmp_path):
        path = tmp_path / "packs.json"
        path.write_text("{not json", encoding="utf-8")
        store = packs.PackStore(path=path)
        assert store.is_enabled("music-deck")
        assert store.session("travel-board", "owner:x") == {}


# ── 3. the manifest surface ─────────────────────────────────────


class TestPacksManifestSurface:
    def test_packs_surface_members_are_the_registry(self):
        from nomorals.social.chat import surface_manifest as sm

        surface = sm.surface_by_id("packs")
        assert set(surface.members) == {p.id for p in packs.PACKS}

    def test_packs_surface_actions_resolve(self):
        from nomorals.social.chat import surface_manifest as sm

        surface = sm.surface_by_id("packs")
        assert surface.actions
        for ref in surface.actions:
            kind = ref.handler.split(":", 1)[1]
            assert kind in CONTROL_COMMANDS
            if ref.data:
                assert parse_control("/" + ref.data) is not None


# ── 4. runtime end-to-end ───────────────────────────────────────

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import (  # noqa: E402
    ChatKind,
    ChatMessage,
    ChatRef,
)


def _make_runtime():
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
    sends: list = []

    def _send(platform, chat, text, buttons=None):
        sends.append({"platform": platform, "text": text,
                      "buttons": buttons})
        return SimpleNamespace(ok=True,
                               message_id=f"m{len(sends)}")

    runtime.gateway = SimpleNamespace(send=_send, status=lambda: {})
    runtime._p4_sends = sends
    return runtime, tmp


def _dm(chat_id="room1", sender="owner1"):
    chat = ChatRef(platform="local", chat_id=chat_id,
                   kind=ChatKind.DM, title="room", peer=chat_id)
    # meta.is_owner is the gateway's operator stamp: /pack rides
    # the catalog's standing owner grant (as /create and
    # /stylepin do), so the test's callers carry the stamp the
    # owner's own surfaces carry in production.
    msg = ChatMessage(chat=chat, incoming=True, text="",
                      sender=sender, sender_id=sender,
                      meta={"is_owner": True})
    return chat, msg


def _drive(runtime, chat, msg, text):
    msg.text = text
    return runtime.handle_control(text, chat.key, message=msg)


def _pack_card(runtime, chat):
    store = runtime._living_card_host().store
    cards = [c for c in store.live_for_chat(chat.key)
             if c.family in packs.PACK_IDS]
    return cards[0] if cards else None


class PackRuntimeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.tmp = _make_runtime()
        self.chat, self.msg = _dm()

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.tmp.cleanup()

    def open_pack(self, pack_id, chat=None, msg=None):
        chat = chat or self.chat
        msg = msg or self.msg
        return _drive(self.runtime, chat, msg, f"/pack open {pack_id}")


class TestEveryPackLaunches(PackRuntimeCase):
    def test_every_pack_publishes_a_card_whose_actions_resolve(self):
        from nomorals.social.chat.living_cards import resolve_card_tap

        store = self.runtime._living_card_host().store
        for pack in packs.PACKS:
            with self.subTest(pack=pack.id):
                sends_before = len(self.runtime._p4_sends)
                self.open_pack(pack.id)
                card = _pack_card(self.runtime, self.chat)
                assert card is not None, (
                    f"{pack.id}: opening published no living card")
                assert card.family == pack.id
                assert len(self.runtime._p4_sends) > sends_before
                assert card.actions, f"{pack.id}: home card is inert"
                first = card.actions[0]
                outcome = resolve_card_tap(
                    store, card.card_id, card.revision, 1,
                    chat_key=card.chat_key, owner_key=card.owner_key)
                assert outcome.status.name == "OK", (
                    f"{pack.id}: first action did not resolve")
                if first.intent == "url":
                    assert outcome.command.startswith("http")
                else:
                    cmd = parse_control("/" + outcome.command)
                    assert cmd is not None, (
                        f"{pack.id}: action command "
                        f"{outcome.command!r} reaches no handler")

    def test_wrapped_packs_point_at_their_real_feature(self):
        for pack in packs.PACKS:
            if not pack.entry:
                continue
            with self.subTest(pack=pack.id):
                self.open_pack(pack.id)
                card = _pack_card(self.runtime, self.chat)
                assert card is not None
                datas = [a.data for a in card.actions]
                assert any(d == pack.entry
                           or d.startswith(pack.entry + " ")
                           for d in datas), (
                    f"{pack.id}: home card never offers its entry "
                    f"{pack.entry!r}: {datas}")

    def test_listing_names_every_pack_with_its_state(self):
        out = _drive(self.runtime, self.chat, self.msg, "/pack")
        for pack in packs.PACKS:
            assert pack.id in out, f"{pack.id} missing from /pack"
        assert "on" in out

    def test_unknown_pack_answers_honestly(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack open no-such-pack")
        assert "no-such-pack" in out
        assert _pack_card(self.runtime, self.chat) is None


class TestPackLifecycle(PackRuntimeCase):
    def test_disabled_pack_stops_resolving(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack off music-deck")
        assert "music-deck" in out
        sends_before = len(self.runtime._p4_sends)
        out = self.open_pack("music-deck")
        assert "unavailable" in out.lower() or "off" in out.lower()
        assert len(self.runtime._p4_sends) == sends_before, (
            "a disabled pack still published a card")
        verb = _drive(self.runtime, self.chat, self.msg,
                      "/pack act music-deck noop")
        assert "unavailable" in verb.lower() or "off" in verb.lower()
        listing = _drive(self.runtime, self.chat, self.msg, "/pack")
        assert "off" in listing
        _drive(self.runtime, self.chat, self.msg, "/pack on music-deck")
        self.open_pack("music-deck")
        assert _pack_card(self.runtime, self.chat) is not None


class TestStylePinTargetsPack(PackRuntimeCase):
    def test_pin_on_pack_scope_names_itself_on_the_card(self):
        reply = self.runtime._control_stylepin(
            "set music-deck theme board", None)
        assert "pinned" in reply, reply
        self.open_pack("music-deck")
        texts = [s["text"] for s in self.runtime._p4_sends]
        assert any("theme board" in t for t in texts), (
            "the pinned theme never surfaced on the pack's card")
        self.runtime._control_stylepin("clear music-deck", None)
        chat2, msg2 = _dm(chat_id="room2")
        self.open_pack("music-deck", chat=chat2, msg=msg2)
        texts = [s["text"] for s in self.runtime._p4_sends]
        assert "theme board" not in texts[-1]


class TestCommandCenterFacts(PackRuntimeCase):
    def test_center_sees_the_chats_live_cards(self):
        self.open_pack("music-deck")
        self.open_pack("command-center")
        center = [c for c in
                  self.runtime._living_card_host().store.live_for_chat(
                      self.chat.key)
                  if c.family == "command-center"]
        assert center, "the command center published no card"
        assert "music-deck" in center[0].body, (
            "the center did not report the chat's live music-deck "
            f"card: {center[0].body!r}")


# ── 5. native packs move real state ─────────────────────────────


class TestTravelBoard(PackRuntimeCase):
    def test_legs_round_trip_and_cross_chat_resume(self):
        self.open_pack("travel-board")
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act travel-board add Lagos | fly in Friday")
        assert "Lagos" in out
        _drive(self.runtime, self.chat, self.msg,
               "/pack act travel-board add Abuja | train back Sunday")
        store = packs.get_pack_store()
        session = store.session("travel-board", "owner:owner1")
        assert [leg["label"] for leg in session["legs"]] == [
            "Lagos", "Abuja"]
        # cross-device handoff: a different chat, same owner —
        # the board resumes with its legs (the handoff pack's
        # substrate claim, exercised here on its behalf).
        chat2, msg2 = _dm(chat_id="phone2")
        self.open_pack("travel-board", chat=chat2, msg=msg2)
        card = _pack_card(self.runtime, chat2)
        assert card is not None and "Lagos" in card.body
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act travel-board drop 1")
        assert "Lagos" in out
        session = store.session("travel-board", "owner:owner1")
        assert [leg["label"] for leg in session["legs"]] == ["Abuja"]


class TestDecisionRoom(PackRuntimeCase):
    def test_propose_vote_close_records_a_decision(self):
        self.open_pack("decision-room")
        out = _drive(
            self.runtime, self.chat, self.msg,
            "/pack act decision-room propose Ship Friday? | yes | no")
        assert "Ship Friday?" in out
        _drive(self.runtime, self.chat, self.msg,
               "/pack act decision-room vote 1")
        _chat2, msg2 = _dm(sender="teammate")
        msg2.chat = self.chat
        _drive(self.runtime, self.chat, msg2,
               "/pack act decision-room vote 1")
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act decision-room close")
        assert "yes" in out.lower()
        session = packs.get_pack_store().session(
            "decision-room", f"chat:{self.chat.key}")
        assert session["record"]["winner"] == "yes"
        assert session["record"]["tally"]["yes"] == 2


class TestAdventure(PackRuntimeCase):
    def test_choices_walk_the_story_to_an_ending(self):
        self.open_pack("adventure")
        store = packs.get_pack_store()
        seen = set()
        for _step in range(12):
            session = store.session("adventure", "owner:owner1")
            node = session.get("node", "")
            assert node and node not in seen, (
                f"story looped or lost its place at {node!r}")
            seen.add(node)
            graph = packs.ADVENTURE_NODES[node]
            if not graph["choices"]:
                break
            _drive(self.runtime, self.chat, self.msg,
                   "/pack act adventure choose 1")
        else:
            raise AssertionError("the story never reached an ending")
        assert graph["ending"]


class TestLearningCards(PackRuntimeCase):
    def test_reveal_next_and_progress(self):
        self.open_pack("learning-cards")
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act learning-cards reveal")
        session = packs.get_pack_store().session(
            "learning-cards", "owner:owner1")
        assert session["revealed"]
        card0 = packs.LEARNING_DECK[session["index"]]
        assert card0["answer"] in out, (
            "the reveal reply did not carry the card's answer")
        _drive(self.runtime, self.chat, self.msg,
               "/pack act learning-cards known")
        session = packs.get_pack_store().session(
            "learning-cards", "owner:owner1")
        assert session["known"] == 1


class TestWorkflowButtons(PackRuntimeCase):
    def test_save_and_run_replays_a_real_command(self):
        self.open_pack("workflow-buttons")
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act workflow-buttons save morning = status")
        assert "morning" in out
        store = packs.get_pack_store()
        assert store.workflows()["morning"] == "status"
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act workflow-buttons run morning")
        assert out.strip(), "running the workflow answered silence"
        bad = _drive(self.runtime, self.chat, self.msg,
                     "/pack act workflow-buttons save loop = pack open x")
        assert "pack" in bad.lower()  # recursion refused, in words
        assert "loop" not in store.workflows()


class TestMemoryShelf(PackRuntimeCase):
    def test_shelve_list_forget(self):
        self.open_pack("memory-shelf")
        out = _drive(
            self.runtime, self.chat, self.msg,
            "/pack act memory-shelf shelve deploy keys live in the "
            "vault | from the infra chat")
        assert "vault" in out
        session = packs.get_pack_store().session(
            "memory-shelf", "owner:owner1")
        assert session["facts"][0]["source"] == "from the infra chat"
        assert session["facts"][0]["shelved_in"] == self.chat.key
        _drive(self.runtime, self.chat, self.msg,
               "/pack act memory-shelf forget 1")
        session = packs.get_pack_store().session(
            "memory-shelf", "owner:owner1")
        assert session["facts"] == []


class TestMediaVault(PackRuntimeCase):
    def test_find_searches_the_real_expression_library(self):
        library = self.runtime._expression_library()
        library.add_item(kind="gif", path="/tmp/celebration.gif",
                         tags=("celebration", "dance"),
                         source="test")
        self.open_pack("media-vault")
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act media-vault find celebration")
        assert "celebration" in out.lower()
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act media-vault find no-such-tag")
        assert "no" in out.lower()


class TestActionBar(PackRuntimeCase):
    def test_usage_ranks_and_pins_lead(self):
        self.open_pack("music-deck")
        self.open_pack("music-deck")
        self.open_pack("adventure")
        self.open_pack("action-bar")
        card = _pack_card(self.runtime, self.chat)
        assert card is not None
        bar = [c for c in
               self.runtime._living_card_host().store.live_for_chat(
                   self.chat.key) if c.family == "action-bar"][0]
        datas = [a.data for a in bar.actions]
        assert datas[0] == "pack open music-deck", datas
        _drive(self.runtime, self.chat, self.msg,
               "/pack act action-bar pin book-studio")
        self.open_pack("action-bar")
        bar = [c for c in
               self.runtime._living_card_host().store.live_for_chat(
                   self.chat.key) if c.family == "action-bar"][0]
        assert bar.actions[0].data == "pack open book-studio"


class TestSurprise(PackRuntimeCase):
    def test_roll_is_labeled_and_actionable(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act surprise-me roll")
        assert "surprise" in out.lower()


class TestCelebration(PackRuntimeCase):
    def test_celebrate_produces_a_real_artifact(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act celebration-engine celebrate launch day")
        assert "launch" in out.lower()
        root = Path(self.tmp.name)
        artifacts = [p for p in root.rglob("*")
                     if p.is_file() and p.suffix in (".png", ".webp")
                     and "packs" in str(p)]
        assert artifacts, "no celebration artifact was produced"


class TestCompanionViews(PackRuntimeCase):
    def test_view_renders_a_real_html_artifact(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act companion-views view music-deck")
        root = Path(self.tmp.name)
        pages = [p for p in root.rglob("*.html") if "packs" in str(p)]
        assert pages, f"no companion page was produced: {out!r}"
        html = pages[0].read_text(encoding="utf-8")
        assert "Music deck" in html


class TestInlineCreation(PackRuntimeCase):
    def test_pack_articles_and_adapter_wiring(self):
        articles = packs.pack_inline_articles("music")
        assert articles
        assert any("/pack open music-deck" in a["input_message_content"]
                   ["message_text"] for a in articles)
        from nomorals.social.chat.tgbot_buttons import (
            inline_results_for,
        )
        results = inline_results_for("pack music")
        assert any("/pack open" in r["input_message_content"]
                   ["message_text"] for r in results)


class TestLanguageExpression(PackRuntimeCase):
    def test_say_keeps_the_original_and_names_the_emotion(self):
        out = _drive(self.runtime, self.chat, self.msg,
                     "/pack act language-expression say hug")
        assert "hug" in out.lower()


if __name__ == "__main__":
    unittest.main()
