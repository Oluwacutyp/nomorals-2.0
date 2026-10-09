"""Tests for the social rebuild: identity, relationships, triage, gating."""

import json
import os
import tempfile

from nomorals.partner.social_gate import (
    ACTOR_OWNER,
    ACTOR_OUTSIDER,
    check_tool_call,
    grant_for,
)
from nomorals.social.chat.base import ChatKind
from nomorals.social.identity import CONF_HIGH, IdentityStore
from nomorals.social.relationships import RelationshipTracker
from nomorals.social.triage import (
    TIER_CRITICAL,
    TIER_IMPORTANT,
    TIER_NOISE,
    TIER_ROUTINE,
    triage_message,
)


class _Msg:
    def __init__(self, text: str = ""):
        self.text = text


# ── identity ─────────────────────────────────────────────────────

def _tmp_store():
    path = os.path.join(tempfile.mkdtemp(), "ids.json")
    return IdentityStore(path=path), path


def test_identity_register_and_find():
    store, _ = _tmp_store()
    p = store.register("whatsapp", "2348012345678@s.whatsapp.net",
                       display_name="Mama")
    assert store.find("whatsapp", "2348012345678@s.whatsapp.net").person_id == p.person_id


def test_identity_phone_autolink():
    store, _ = _tmp_store()
    p1 = store.register("whatsapp", "2348012345678@s.whatsapp.net",
                        display_name="Mama")
    # Same phone on Telegram auto-links (HIGH confidence).
    p2 = store.register("telegram", "99112233", display_name="Mama Vee",
                        phone="+2348012345678")
    assert p1.person_id == p2.person_id
    assert len(p2.identities) == 2


def test_identity_no_merge_on_name_alone():
    store, _ = _tmp_store()
    p1 = store.register("whatsapp", "111@s.whatsapp.net", display_name="John")
    p2 = store.register("telegram", "222", display_name="John")
    assert p1.person_id != p2.person_id
    # But it suggests the link for owner confirmation.
    suggestions = store.suggest_links("telegram", "222")
    assert any(s["person_id"] == p1.person_id for s in suggestions)


def test_identity_explicit_link_and_unlink():
    store, _ = _tmp_store()
    p1 = store.register("discord", "d1", display_name="Vee")
    ok = store.link(p1.person_id, "telegram", "t9", signal="owner confirmed",
                    confidence=CONF_HIGH, display_name="Vee")
    assert ok
    assert store.find("telegram", "t9").person_id == p1.person_id
    assert store.unlink(p1.person_id, "telegram", "t9")
    assert store.find("telegram", "t9").person_id != p1.person_id


def test_identity_low_confidence_link_refused():
    store, _ = _tmp_store()
    p1 = store.register("discord", "d1", display_name="Vee")
    assert not store.link(p1.person_id, "telegram", "t9",
                          signal="name similarity", confidence="low")


def test_identity_persists():
    store, path = _tmp_store()
    p = store.register("whatsapp", "555@s.whatsapp.net", display_name="Zed")
    store2 = IdentityStore(path=path)
    assert store2.find("whatsapp", "555@s.whatsapp.net").person_id == p.person_id


# ── relationships ────────────────────────────────────────────────

def _tmp_tracker():
    path = os.path.join(tempfile.mkdtemp(), "rels.json")
    return RelationshipTracker(path=path)


def test_relationship_closeness_derived():
    t = _tmp_tracker()
    for _ in range(20):
        t.note_inbound("p1", display_name="Ada")
    for _ in range(5):
        t.note_outbound("p1", owner_initiated=True)
    rel = t.get("p1")
    assert rel.closeness > 0.4  # owner initiation weighs heavily


def test_relationship_cadence():
    t = _tmp_tracker()
    t.note_inbound("p1", display_name="New")
    assert t.get("p1").cadence() == "new"


def test_relationship_owner_initiation_matters_most():
    t = _tmp_tracker()
    for _ in range(30):  # lots of inbound, no owner initiation
        t.note_inbound("p1", display_name="A")
    t2 = _tmp_tracker()
    for _ in range(3):  # few messages, but owner-initiated
        t2.note_outbound("p2", owner_initiated=True, display_name="B")
    assert t2.get("p2").closeness > t.get("p1").closeness


def test_relationship_needs_reconnect():
    t = _tmp_tracker()
    t.note_inbound("p1", display_name="Old Friend")
    for _ in range(10):
        t.note_outbound("p1", owner_initiated=True)
    rel = t.get("p1")
    rel.last_inbound = rel.last_outbound = 0.0  # long ago
    quiet = t.needs_reconnect(days=30)
    assert any(r.person_id == "p1" for r in quiet)


# ── triage ───────────────────────────────────────────────────────

def test_triage_owner_is_critical():
    s = triage_message(_Msg("dad's in the hospital, call me now"),
                       is_owner=False, closeness=0.9, kind="dm")
    assert s.tier in (TIER_CRITICAL, TIER_IMPORTANT)
    assert s.score > 0.45


def test_triage_group_chatter_is_noise():
    s = triage_message(_Msg("lol"), is_owner=False, closeness=0.1, kind="group")
    assert s.tier == TIER_NOISE
    assert not s.buzz


def test_triage_mention_boosts():
    base = triage_message(_Msg("what do you think?"), is_owner=False,
                          closeness=0.3, kind="group")
    mentioned = triage_message(_Msg("what do you think?"), is_owner=False,
                               closeness=0.3, kind="group", mentioned=True)
    assert mentioned.score > base.score


def test_triage_question_from_close_contact():
    s = triage_message(_Msg("can you send me the file?"), is_owner=False,
                       closeness=0.8, kind="dm")
    assert s.tier in (TIER_CRITICAL, TIER_IMPORTANT)


def test_triage_ack_never_important():
    s = triage_message(_Msg("ok"), is_owner=False, closeness=0.9, kind="dm")
    assert s.score <= 0.45  # capped, even from close contacts


# ── social gate (code-enforced) ──────────────────────────────────

def test_gate_owner_gets_everything():
    g = grant_for(is_owner=True, chat_kind=ChatKind.GROUP)
    assert g.actor == ACTOR_OWNER
    assert g.may_send and g.may_admin and g.may_access_memory


def test_gate_outsider_dm_gets_nothing():
    g = grant_for(is_owner=False, chat_kind=ChatKind.DM)
    assert g.actor == ACTOR_OUTSIDER
    assert not g.may_send and not g.may_access_memory
    allowed, _ = check_tool_call("social_send", grant=g)
    assert not allowed


def test_gate_outsider_group_gets_public_games():
    g = grant_for(is_owner=False, chat_kind=ChatKind.GROUP)
    allowed, _ = check_tool_call("game_join", grant=g)
    assert allowed
    denied, reason = check_tool_call("social_send", grant=g)
    assert not denied
    assert "not available" in reason


def test_gate_outsider_cannot_reach_memory():
    g = grant_for(is_owner=False, chat_kind=ChatKind.GROUP)
    allowed, _ = check_tool_call("memory_recall", grant=g)
    assert not allowed
    allowed, _ = check_tool_call("telegram_ban", grant=g)
    assert not allowed
