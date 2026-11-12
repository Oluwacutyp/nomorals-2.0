"""Pins for the owner's 2026-11-12 live defect: a /play pick-list reply "1"
met with silence.

The owner sent ``/play haza by corizo`` at 12:55 AM; Devon answered with
the numbered pick-list (``pick:d5a5bbeb``); he replied ``1`` at 1:00 AM —
five minutes later, right at the old 300-second TTL boundary — and
nothing came back: no track, no error. The hook returned None for an
expired pick and the bare number fell through to the Core Mind, which
had nothing to say to a bare "1".

Two fixes pinned here:

1. Silence ends: a bare number in a chat that HAD a pick-list answers
   honestly ("that pick-list expired ...") instead of returning None.
   A bare number in a chat that NEVER had one still flows on untouched.
2. The TTL is 30 minutes, not 5 — a human reads a list slower than a
   cache expires.
"""

import time

import pytest

from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
from nomorals.media import picklist as pl
from nomorals.media.picklist import PICK_TTL, PickCache
from nomorals.media.resolver import ResolvedAudio
from nomorals.social.chat.base import ChatMessage, ChatRef


@pytest.fixture(autouse=True)
def _clean_cache():
    PickCache.clear()
    yield
    PickCache.clear()


def _cand(title):
    return ResolvedAudio(ok=True, path_or_url=f"https://x/{title}",
                         title=title, artist="", duration=180.0,
                         kind="youtube", source_name="test")


def _msg(text, chat_key="whatsapp:2348065574087", sender="owner",
         sender_id=None, meta=None):
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id),
        incoming=True, text=text, sender=sender,
        sender_id=sender_id if sender_id is not None else sender,
        meta=dict(meta or {}))


def _hook(text, **kw):
    mixin = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    return mixin._play_pick_hook(_msg(text, **kw))


def _reply_text(reply):
    return getattr(reply, "text", reply) or ""


class TestExpiredPickAnswersHonestly:
    def test_owner_sequence_expired_reply_not_silence(self):
        # The owner's exact sequence: list offered, reply "1" after
        # the TTL — the hook must ANSWER, not return None.
        token = PickCache.store(
            [_cand("Haza")], "whatsapp:2348065574087",
            ttl=0.05, owner_key="owner")
        assert token
        time.sleep(0.08)
        reply = _hook("1", sender="owner")
        assert reply is not None
        assert "expired" in _reply_text(reply)
        assert "30 min" in _reply_text(reply)

    def test_consumed_pick_number_reply_answers_honestly(self):
        # Single-use list, already picked: a second "1" gets the
        # honest answer, not silence.
        token = PickCache.store(
            [_cand("Haza")], "whatsapp:2348065574087",
            owner_key="owner")
        assert PickCache.consume(token) is not None
        reply = _hook("1", sender="owner")
        assert reply is not None
        assert "expired" in _reply_text(reply)

    def test_expired_pick_does_not_answer_another_user(self):
        # Owner scoping survives expiry: another user's bare number
        # in the same chat is not the owner's expired pick.
        token = PickCache.store(
            [_cand("Haza")], "whatsapp:2348065574087",
            ttl=0.05, owner_key="owner")
        assert token
        time.sleep(0.08)
        assert _hook("1", sender="someone-else") is None

    def test_bare_number_without_pick_history_flows_on(self):
        # A stray "1" in a chat that never had a pick-list is not a
        # pick attempt — the hook returns None and the turn flows on.
        assert _hook("1") is None
        assert _hook("7", chat_key="wa:never-had-one") is None

    def test_live_pick_still_routes_to_control_play(self):
        # The fix must not touch the live path: a live pick still
        # resolves through /play pick.
        token = PickCache.store(
            [_cand("Haza")], "wa:1", owner_key="owner")
        mixin = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        seen = {}

        def fake_control_play(rest, chat_key, **kw):
            seen["rest"] = rest
            return "picked:stub"

        mixin._control_play = fake_control_play
        reply = mixin._play_pick_hook(
            _msg("1", chat_key="wa:1", sender="owner"))
        assert reply == "picked:stub"
        assert seen["rest"] == f"pick {token} 1"


class TestPickTtl:
    def test_pick_ttl_is_thirty_minutes(self):
        # A human reads a list slower than 5 minutes.
        assert PICK_TTL == 1800.0
        assert getattr(pl, "PICK_TOMBSTONE_TTL", None) == 1800.0

    def test_fresh_store_clears_expired_memory(self):
        # A new list supersedes the expired one: the chat has a LIVE
        # pick again, not an expired memory of one.
        PickCache.store([_cand("A")], "wa:1", ttl=0.05)
        time.sleep(0.08)
        PickCache.prune()
        assert PickCache.had_expired_pick("wa:1") is True
        PickCache.store([_cand("B")], "wa:1")
        assert PickCache.had_expired_pick("wa:1") is False

    def test_tombstone_pruned_after_window(self):
        # The "had a pick" memory is bounded: past the window a bare
        # number is a stray number again, and the hook stays silent.
        PickCache.store([_cand("A")], "wa:1", ttl=0.05)
        time.sleep(0.08)
        PickCache.prune()
        assert PickCache.had_expired_pick("wa:1") is True
        for slot in list(PickCache._expired):
            PickCache._expired[slot] = (
                time.time() - getattr(pl, "PICK_TOMBSTONE_TTL", 1800.0) - 1)
        assert PickCache.had_expired_pick("wa:1") is False
        assert _hook("1", chat_key="wa:1") is None


class TestLidPhoneFormsResolveSamePick:
    """The proven live hole (register #29, second form).

    WhatsApp addresses one human two ways — the phone JID and the
    privacy LID — and the bridge reports them "in either order,
    depending on the account's addressing migration". The pick slot
    keyed off the RAW sender id, so a /play offered on the phone
    form and a "1" arriving on the LID form missed a LIVE pick and
    met silence — the tombstone answer would be a lie there ("that
    pick-list expired" for a list that is alive). The owner key
    must fold both forms to the canonical phone form, exactly like
    the game identity resolution already does.
    """

    PHONE = "2348065574087@s.whatsapp.net"
    LID = "123456789012345@lid"
    CHAT = "whatsapp:2348065574087"

    def _live_pick_hook(self, reply_msg):
        mixin = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        seen = {}

        def fake_control_play(rest, chat_key, **kw):
            seen["rest"] = rest
            return "picked:stub"

        mixin._control_play = fake_control_play
        reply = mixin._play_pick_hook(reply_msg)
        return reply, seen

    def test_phone_offer_lid_reply_resolves_live_pick(self):
        from nomorals.agents.partner.runtime import _pick_owner_key

        play_msg = _msg("/play haza by corizo", chat_key=self.CHAT,
                        sender_id=self.PHONE,
                        meta={"sender_alt_id": self.LID})
        assert _pick_owner_key(play_msg) == self.PHONE
        token = PickCache.store(
            [_cand("Haza")], self.CHAT,
            owner_key=_pick_owner_key(play_msg))
        reply_msg = _msg("1", chat_key=self.CHAT, sender_id=self.LID,
                         meta={"sender_alt_id": self.PHONE})
        reply, seen = self._live_pick_hook(reply_msg)
        assert reply == "picked:stub"
        assert seen["rest"] == f"pick {token} 1"

    def test_lid_offer_phone_reply_resolves_live_pick(self):
        from nomorals.agents.partner.runtime import _pick_owner_key

        play_msg = _msg("/play haza by corizo", chat_key=self.CHAT,
                        sender_id=self.LID,
                        meta={"sender_alt_id": self.PHONE})
        assert _pick_owner_key(play_msg) == self.PHONE
        token = PickCache.store(
            [_cand("Haza")], self.CHAT,
            owner_key=_pick_owner_key(play_msg))
        reply_msg = _msg("1", chat_key=self.CHAT, sender_id=self.PHONE,
                         meta={"sender_alt_id": self.LID})
        reply, seen = self._live_pick_hook(reply_msg)
        assert reply == "picked:stub"
        assert seen["rest"] == f"pick {token} 1"

    def test_other_humans_lid_does_not_resolve_owners_pick(self):
        # The fold merges one human's two forms — never two humans.
        from nomorals.agents.partner.runtime import _pick_owner_key

        play_msg = _msg("/play haza by corizo", chat_key=self.CHAT,
                        sender_id=self.PHONE,
                        meta={"sender_alt_id": self.LID})
        PickCache.store(
            [_cand("Haza")], self.CHAT,
            owner_key=_pick_owner_key(play_msg))
        other = _msg("1", chat_key=self.CHAT,
                     sender_id="999999999999999@lid",
                     meta={"sender_alt_id": "2348000000000@s.whatsapp.net"})
        reply, _seen = self._live_pick_hook(other)
        assert reply is None
