"""Community mirroring completion (R2-26 named follow-ups).

The command-driven /community mirror shipped with two follow-ups
deliberately not done: no live inbound hook (GroupMirror.
handle_incoming had zero production callers) and no chat verbs for
the roster link/unlink (UnifiedRoster.link/unlink had zero
production callers, so resolve() could never return a member and
mirror attribution always fell back to display name). These tests
pin both:

- /community mirror inbound: a plain inbound group message in an
  armed community mirrors to the community's groups on the other
  platforms, on the group cycle, after moderation, with the
  command path's gates (armed-only, system-output, channels held)
  and loop safety via the shared engine's seen_ids + origin
  exclusion. Commands, empty text, unarmed communities,
  untagged groups and moderated-away messages never mirror.
- /community member: add / link / unlink / list over the
  per-group roster, so a member's platform identities can be
  linked from chat and mirror attribution resolves to the linked
  identity instead of the display name.

Rig copied from tests/test_gapfix_community_mirror.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.community.bridges import merge_rosters
from nomorals.community.mirroring import MirrorEngine
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

WA1 = "whatsapp:120363025000001@g.us"
WA2 = "whatsapp:120363025000002@g.us"
TG1 = "telegram-bot:-100111"
TG2 = "telegram-bot:-100222"
DC1 = "discord:999888"
NOW = 1780000000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_community(surface, cid="c1", name="Devs",
                    keys=(WA1, WA2, TG1), designate=True):
    from nomorals.social.chat import group_community as _gc

    for key in keys:
        _gc.set_community_tag(surface, key, cid, name)
        if designate:
            surface.set_designated(key, True, by="owner")
    return surface


class _FakeAdapter:
    """Records send() calls; returns a SendResult-shaped verdict."""

    def __init__(self, *, fail_keys=()):
        self.sent: list[tuple[str, str]] = []
        self.fail_keys = set(fail_keys)

    def send(self, chat, text):
        key = f"{chat.platform}:{chat.chat_id}"
        self.sent.append((key, text))
        if key in self.fail_keys:
            return SimpleNamespace(ok=False, message_id="",
                                   error="bridge down")
        return SimpleNamespace(ok=True, message_id="m1", error="")


def _adapters():
    return {"whatsapp": _FakeAdapter(), "telegram-bot": _FakeAdapter()}


def _rig(tmp_path: Path, *, adapters=None, operator=True):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: operator  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: True  # type: ignore[method-assign]
    adapters = adapters if adapters is not None else {}
    mixin._adapter_for = lambda platform: adapters.get(platform)  # type: ignore[method-assign]
    mixin.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    mixin._community_bridge_dir = str(tmp_path / "bridges")  # type: ignore[attr-defined]
    mixin._community_memory_root = str(tmp_path / "memory")  # type: ignore[attr-defined]
    mixin._community_mirror_engine_inst = MirrorEngine()  # type: ignore[attr-defined]
    return mixin, surface


def _dm_msg(sender_id="owner1", *, platform="whatsapp", sender="someone"):
    chat = SimpleNamespace(key=f"{platform}:{sender_id}",
                           platform=platform, kind=ChatKind.DM,
                           chat_id=sender_id)
    return SimpleNamespace(chat=chat, text="", meta={},
                           sender=sender, sender_id=sender_id)


def _group_msg(key=WA1, *, sender_id="u-wa-1", sender="Ada",
               username="ada", text="hello all", message_id="m-in-1",
               platform=None):
    platform = platform or str(key).split(":", 1)[0]
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind=ChatKind.GROUP,
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender=sender, sender_username=username,
        text=text, meta={}, reply_to="", message_id=message_id,
        incoming=True)


def _text(out) -> str:
    return getattr(out, "text", out)


def _arm(mixin, tmp_path):
    mixin._community_service()["arms"].arm("community-mirror")


def _dm_session(mixin, tmp_path, *, groups=None):
    mixin._console_session_map()["whatsapp:owner1"] = {
        "groups": groups or {"g1": WA1}, "at": time.time()}


# ── organ: merge_rosters ─────────────────────────────────────────────

def test_merge_rosters_unions_members_and_platforms():
    from nomorals.community.bridges import UnifiedRoster

    r1 = UnifiedRoster()
    m1 = r1.add_member("Adaeze")
    r1.link(m1.id, "whatsapp", "u-wa-1")
    r2 = UnifiedRoster()
    m2 = r2.add_member("Bola")
    r2.link(m2.id, "telegram", "u-tg-1")
    merged = merge_rosters([r1, r2])
    assert merged.resolve("whatsapp", "u-wa-1") is not None
    assert merged.resolve("whatsapp", "u-wa-1").display_name == "Adaeze"
    assert merged.resolve("telegram", "u-tg-1").display_name == "Bola"
    assert merged.resolve("whatsapp", "nobody") is None


def test_merge_rosters_first_wins_on_conflict():
    from nomorals.community.bridges import UnifiedRoster

    r1 = UnifiedRoster()
    m1 = r1.add_member("First")
    r1.link(m1.id, "whatsapp", "u-shared")
    r2 = UnifiedRoster()
    m2 = r2.add_member("Second")
    r2.link(m2.id, "whatsapp", "u-shared")
    merged = merge_rosters([r1, r2])
    assert merged.resolve("whatsapp", "u-shared").display_name == "First"


# ── inbound hook ─────────────────────────────────────────────────────

def test_inbound_mirrors_plain_text_to_sibling_platform(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    assert out["mirrored"] is True
    assert out["sent"] == 1
    assert adapters["telegram-bot"].sent == [
        (TG1, "[Ada @ whatsapp] hello all")]
    assert adapters["whatsapp"].sent == []  # origin never echoed


def test_inbound_disarmed_mirrors_nothing(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []
    assert adapters["whatsapp"].sent == []


def test_inbound_no_community_mirrors_nothing(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []


def test_inbound_command_text_never_mirrors(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(text="/play something"),
        {"actions": []})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []


def test_inbound_empty_text_never_mirrors(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(text="   "), {"actions": []})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []


def test_inbound_dedup_same_message_id_twice(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    first = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    second = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    assert first["mirrored"] is True
    assert second["mirrored"] is False
    assert len(adapters["telegram-bot"].sent) == 1


def test_inbound_deleted_message_never_mirrors(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(),
        {"actions": [{"kind": "delete", "rule": "scamlink"}]})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []


def test_inbound_banned_message_never_mirrors(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(),
        {"actions": [{"kind": "ban", "rule": "raid"}]})
    assert out["mirrored"] is False
    assert adapters["telegram-bot"].sent == []


def test_inbound_system_output_disallowed_target_held(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    # WA1 designated (origin, passes); TG1 is a bot-platform room
    # so it passes the §33 gate by key; DC1 merely sits there —
    # not designated, not a bot platform — so the gate holds it.
    _seed_community(surface, keys=(WA1, TG1, DC1), designate=False)
    surface.set_designated(WA1, True, by="owner")
    _arm(mixin, tmp_path)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    assert out["mirrored"] is True
    assert out["sent"] == 1
    assert adapters["telegram-bot"].sent == [
        (TG1, "[Ada @ whatsapp] hello all")]
    assert any(DC1 in h for h in out.get("held", []))


def test_inbound_linked_member_attributed_by_identity(tmp_path):
    from nomorals.community.bridges import BridgeStore

    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    store = BridgeStore(str(tmp_path / "bridges"))
    roster = store.load(WA1)
    member = roster.add_member("Adaeze")
    roster.link(member.id, "whatsapp", "u-wa-1")
    store.save(WA1, roster)
    out = mixin._community_mirror_inbound(
        surface, WA1, _group_msg(sender="ada_nick"), {"actions": []})
    assert out["mirrored"] is True
    assert adapters["telegram-bot"].sent == [
        (TG1, "[Adaeze @ whatsapp] hello all")]


def test_inbound_remembers_message_in_group_memory(tmp_path):
    from nomorals.community.mirroring import recall_group

    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    mixin._community_mirror_inbound(
        surface, WA1, _group_msg(), {"actions": []})
    recalled = recall_group(
        "c1", key="msg:mirror-inbound:c1:whatsapp:m-in-1",
        root=str(tmp_path / "memory"))
    assert recalled and "hello all" in recalled[0]["value"]
    assert "Ada" in recalled[0]["value"]


def test_inbound_wired_into_group_surface_turn(tmp_path):
    """The hook rides the real group cycle: a plain message
    through _group_surface_turn mirrors to the sibling group."""
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _arm(mixin, tmp_path)
    mixin._probe_bot_claim = lambda *a, **k: None  # type: ignore[method-assign]
    mixin._group_admin_ok = lambda message: False  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: None  # type: ignore[method-assign]
    mixin._send = lambda *a, **k: None  # type: ignore[method-assign]
    mixin._claim_system_send = lambda *a, **k: True  # type: ignore[method-assign]
    mixin._execute_group_action = lambda *a, **k: False  # type: ignore[method-assign]
    mixin._send_action_notice = lambda *a, **k: None  # type: ignore[method-assign]
    mixin._sensor_flag_proposal = lambda *a, **k: None  # type: ignore[method-assign]
    for name in ("_post_math_round_closes", "_post_conflict_feed",
                 "_post_proposal_feed", "_execute_pending_pin",
                 "_drive_all_channels", "_cleanup_auto_drive",
                 "_wa_join_request_drive", "_p5_topic_duty"):
        setattr(mixin, name, lambda *a, **k: None)
    mixin._resolve_pending_ask = lambda *a, **k: None  # type: ignore[method-assign]
    consumed = mixin._group_surface_turn(_group_msg())
    assert consumed is False  # ordinary message flows on
    assert adapters["telegram-bot"].sent == [
        (TG1, "[Ada @ whatsapp] hello all")]
    assert adapters["whatsapp"].sent == []


# ── /community member verbs ──────────────────────────────────────────

def test_member_add_link_list_unlink_round_trip(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community(
        "member add g1 Adaeze", _dm_msg()))
    assert "Adaeze" in out and "member" in out
    member_id = out.split("member ", 1)[1].split()[0]
    out = _text(mixin._control_community(
        f"member link g1 {member_id} whatsapp u-wa-1", _dm_msg()))
    assert "linked" in out.lower()
    out = _text(mixin._control_community("member list g1", _dm_msg()))
    assert "Adaeze" in out and "whatsapp:u-wa-1" in out
    out = _text(mixin._control_community(
        f"member unlink g1 {member_id} whatsapp", _dm_msg()))
    assert "unlinked" in out.lower()
    out = _text(mixin._control_community("member list g1", _dm_msg()))
    assert "Adaeze" in out and "whatsapp:u-wa-1" not in out


def test_member_link_by_unique_prefix(tmp_path):
    mixin, surface = _rig(tmp_path)
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community(
        "member add g1 Adaeze", _dm_msg()))
    member_id = out.split("member ", 1)[1].split()[0]
    out = _text(mixin._control_community(
        f"member link g1 {member_id[:12]} whatsapp u-wa-1", _dm_msg()))
    assert "linked" in out.lower()


def test_member_link_unknown_member_refuses(tmp_path):
    mixin, surface = _rig(tmp_path)
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community(
        "member link g1 member-nope whatsapp u-wa-1", _dm_msg()))
    assert "no member" in out.lower()
    assert "member list" in out.lower()


def test_member_link_bad_platform_refuses(tmp_path):
    mixin, surface = _rig(tmp_path)
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community(
        "member add g1 Adaeze", _dm_msg()))
    member_id = out.split("member ", 1)[1].split()[0]
    out = _text(mixin._control_community(
        f"member link g1 {member_id} myspace u-1", _dm_msg()))
    assert "myspace" in out.lower()
    assert "telegram" in out.lower() and "whatsapp" in out.lower()


def test_member_list_empty_names_the_group(tmp_path):
    mixin, surface = _rig(tmp_path)
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community("member list g1", _dm_msg()))
    assert "no members" in out.lower() and WA1 in out


def test_member_verbs_need_admin(tmp_path):
    mixin, surface = _rig(tmp_path, operator=False)
    mixin._console_is_admin = lambda k, sender: False  # type: ignore[method-assign]
    _seed_community(surface)
    _dm_session(mixin, tmp_path)
    out = _text(mixin._control_community(
        "member add g1 Adaeze", _dm_msg()))
    assert "don't admin" in out.lower()


def test_member_usage_lists_member_verbs(tmp_path):
    mixin, surface = _rig(tmp_path)
    out = _text(mixin._control_community("help", _dm_msg()))
    assert "member add" in out and "member link" in out
    assert "member unlink" in out and "member list" in out
