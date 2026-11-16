"""Community mirroring cluster (audit gap R2-26).

The cluster (nomorals/community/mirroring.py, bridges.py,
group_memory.py, persona.py) is dormant — zero production callers.
These tests pin the wiring:

- MirrorEngine targets accept several groups per platform + pacing
  (organ extension, backward-compatible with the pinned str form).
- /community mirror — cross-platform mirroring via the real engine,
  dry-run-first, arm-gated live, held targets named, dedup honoured.
- /cgroup memory — group memory + persona via remember_group /
  recall_group / build_persona, inside the isolation boundary.

Rig copied from tests/test_gapfix_community_xpost.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

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
    mixin._community_mirror_engine_inst = MirrorEngine()  # type: ignore[attr-defined]
    return mixin, surface


def _dm_msg(sender_id="owner1", *, platform="whatsapp", sender="someone"):
    chat = SimpleNamespace(key=f"{platform}:{sender_id}",
                           platform=platform, kind=ChatKind.DM,
                           chat_id=sender_id)
    return SimpleNamespace(chat=chat, text="", meta={},
                           sender=sender, sender_id=sender_id)


def _group_msg(key=WA1, *, sender_id="owner1"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform="whatsapp",
                             kind=ChatKind.GROUP,
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Owner",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _text(out) -> str:
    return getattr(out, "text", out)


# ── organ: MirrorEngine list targets + pacing ──────────────────────────

def test_engine_list_targets_mirror_each_group():
    sent: list[tuple[str, str]] = []
    engine = MirrorEngine(
        senders={"whatsapp": lambda t, x: sent.append((t, x)) or True},
        targets={"telegram": ["tg-1", "tg-2"],
                 "whatsapp": ["wa-1", "wa-2"],
                 "discord": ["dc-1"]})
    msg = {"group_id": "g1", "author_name": "ada",
           "origin_platform": "telegram", "text": "hi",
           "message_id": "m1"}
    deliveries = engine.mirror(msg)
    by_platform = {}
    for d in deliveries:
        by_platform.setdefault(d.to_platform, []).append(d)
    # origin skipped; both whatsapp groups sent; discord has no sender
    assert "telegram" not in by_platform
    assert [d.to_target for d in by_platform["whatsapp"]] == ["wa-1", "wa-2"]
    assert all(d.sent for d in by_platform["whatsapp"])
    assert len(by_platform["discord"]) == 1
    assert by_platform["discord"][0].sent is False
    assert sent and "[ada @ telegram] hi" in sent[0][1]
    # dedup: same message_id never mirrored twice, across all groups
    assert engine.mirror(msg) == []


def test_engine_list_targets_paces_between_deliveries():
    sent: list[tuple[str, str]] = []
    sleeps: list[float] = []
    engine = MirrorEngine(
        senders={"whatsapp": lambda t, x: sent.append((t, x)) or True},
        targets={"whatsapp": ["wa-1", "wa-2"]})
    msg = {"group_id": "g1", "author_name": "ada",
           "origin_platform": "telegram", "text": "hi",
           "message_id": "m1"}
    deliveries = engine.mirror(msg, sleep=sleeps.append, pace_seconds=3.0)
    assert len(deliveries) == 2
    assert sleeps == [3.0]  # paced between groups, not before the first


def test_engine_str_targets_still_work():
    # The pinned str form must survive the list extension.
    sent: list[tuple[str, str]] = []
    engine = MirrorEngine(
        senders={"whatsapp": lambda t, x: sent.append((t, x)) or True},
        targets={"telegram": "tg-1", "whatsapp": "wa-1"})
    msg = {"group_id": "g1", "author_name": "ada",
           "origin_platform": "telegram", "text": "hi",
           "message_id": "m1"}
    deliveries = engine.mirror(msg)
    assert len(deliveries) == 1
    assert deliveries[0].to_target == "wa-1"
    assert deliveries[0].sent is True


# ── handler: mirror preview ────────────────────────────────────────────

def _adapters():
    return {"whatsapp": _FakeAdapter(), "telegram-bot": _FakeAdapter()}


def test_mirror_preview_shows_plan_and_sends_nothing(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "mirror Devs | hello all", _dm_msg()))
    assert "mirror" in out.lower() and "dry" in out.lower()
    assert TG1 in out  # the other platform is the plan
    assert "[someone @ whatsapp] hello all" in out
    assert "arm first" in out
    assert adapters["whatsapp"].sent == []
    assert adapters["telegram-bot"].sent == []


def test_mirror_preview_does_not_burn_dedup(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _text(mixin._control_community(
        "mirror Devs | hello all", _dm_msg()))
    _text(mixin._control_community("arm mirror", _dm_msg()))
    out = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "sent 1 of 1" in out
    assert len(adapters["telegram-bot"].sent) == 1


def test_mirror_preview_empty_text_refuses(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = _text(mixin._control_community("mirror Devs |   ", _dm_msg()))
    assert "empty" in out.lower()
    assert adapters["telegram-bot"].sent == []


def test_mirror_unknown_community_refuses(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "mirror Nope | hello", _dm_msg()))
    assert "Nope" in out and "no community" in out
    assert adapters["telegram-bot"].sent == []


def test_mirror_held_group_named_not_sent(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    # WA1 designated (passes the §33 gate); WA2 merely sits there.
    _seed_community(surface, keys=(WA1, WA2, TG1), designate=False)
    surface.set_designated(WA1, True, by="owner")
    surface.set_designated(TG1, True, by="owner")
    out = _text(mixin._control_community(
        "mirror Devs | hello all", _dm_msg(platform="telegram-bot")))
    assert "held" in out.lower() and WA2 in out and TG1 in out
    assert WA1 in out  # the one planned target


# ── handler: arm gate + live ───────────────────────────────────────────

def test_mirror_live_disarmed_refuses_and_sends_nothing(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "/community arm mirror" in out
    assert adapters["telegram-bot"].sent == []


def test_mirror_live_armed_sends_with_attribution(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    assert "ARMED" in _text(mixin._control_community(
        "arm mirror", _dm_msg()))
    out = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "sent 1 of 1" in out
    assert adapters["telegram-bot"].sent == [
        (TG1, "[someone @ whatsapp] hello all")]
    assert adapters["whatsapp"].sent == []  # origin never echoed


def test_mirror_live_from_telegram_sends_to_whatsapp(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _text(mixin._control_community("arm mirror", _dm_msg()))
    out = _text(mixin._control_community(
        "mirror Devs | hello all | live",
        _dm_msg(platform="telegram-bot")))
    assert "sent 2 of 2" in out
    assert adapters["whatsapp"].sent == [
        (WA1, "[someone @ telegram] hello all"),
        (WA2, "[someone @ telegram] hello all")]
    assert adapters["telegram-bot"].sent == []


def test_mirror_live_second_identical_run_already_mirrored(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    _text(mixin._control_community("arm mirror", _dm_msg()))
    first = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "sent 1 of 1" in first
    second = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "already mirrored" in second.lower()
    assert len(adapters["telegram-bot"].sent) == 1


def test_mirror_no_adapter_target_reports_error_not_silent(tmp_path):
    adapters = {"whatsapp": _FakeAdapter(), "telegram-bot": _FakeAdapter()}
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface, keys=(WA1, TG1, DC1))
    _text(mixin._control_community("arm mirror", _dm_msg()))
    out = _text(mixin._control_community(
        "mirror Devs | hello all | live", _dm_msg()))
    assert "sent 1 of 2" in out
    assert DC1 in out  # the discord delivery failed honestly
    assert adapters["telegram-bot"].sent == [
        (TG1, "[someone @ whatsapp] hello all")]


def test_mirror_arm_mirror_disarm_mirror_status(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    # Bare arm arms the POST tool only — mirror stays disarmed.
    assert "ARMED" in _text(mixin._control_community("arm", _dm_msg()))
    out = _text(mixin._control_community(
        "mirror Devs | hello | live", _dm_msg()))
    assert "/community arm mirror" in out
    assert adapters["telegram-bot"].sent == []
    assert "ARMED" in _text(mixin._control_community(
        "arm mirror", _dm_msg()))
    status = _text(mixin._control_community("status", _dm_msg()))
    assert "mirror" in status.lower() and "armed" in status.lower()
    assert "live runs: 0" in status  # the post line stays byte-identical
    assert "disarmed" in _text(mixin._control_community(
        "disarm mirror", _dm_msg())).lower()
    out = _text(mixin._control_community(
        "mirror Devs | hello | live", _dm_msg()))
    assert "/community arm mirror" in out


def test_mirror_in_group_is_dm_only(tmp_path):
    adapters = _adapters()
    mixin, surface = _rig(tmp_path, adapters=adapters)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "mirror Devs | hello", _group_msg()))
    assert "DM" in out
    assert adapters["telegram-bot"].sent == []


# ── /cgroup memory ─────────────────────────────────────────────────────

def _cgroup_setup(tmp_path, monkeypatch):
    import nomorals.community.group_memory as gmem
    import nomorals.community.groups as gmod
    from nomorals.community.groups import control_cgroup

    monkeypatch.setattr(gmod, "_DATA_DIR", tmp_path / "themed")
    monkeypatch.setattr(gmem, "default_community_root",
                        lambda: tmp_path / "community")

    created = control_cgroup("new Testers | testing", context=None,
                             chat=None, sender="alice", sender_id="alice")
    m = re.search(r"`(grp_[^`]+)`", created)
    assert m, created
    gid = m.group(1)
    joined = control_cgroup(f"join {gid}", context=None, chat=None,
                            sender="alice", sender_id="alice")
    assert "Welcome" in joined, joined
    return control_cgroup, gid


def test_cgroup_memory_remember_and_recall(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    out = control_cgroup(f"memory {gid} remember motto = ship it",
                         context=None, chat=None,
                         sender="alice", sender_id="alice")
    assert "motto" in out
    out = control_cgroup(f"memory {gid} recall motto", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "ship it" in out


def test_cgroup_memory_nonmember_cannot_remember(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    out = control_cgroup(f"memory {gid} remember motto = ship it",
                         context=None, chat=None,
                         sender="mallory", sender_id="mallory")
    # "only members" — the usage text (unknown-command fallthrough)
    # contains "member" but never this refusal, so this fails pre-wire.
    assert "only members" in out.lower()
    out = control_cgroup(f"memory {gid} recall motto", context=None,
                         chat=None, sender="mallory", sender_id="mallory")
    assert "only members" in out.lower()


def test_cgroup_memory_search_and_list(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    control_cgroup(f"memory {gid} remember motto = ship it",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    control_cgroup(f"memory {gid} remember snack = suya",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    out = control_cgroup(f"memory {gid} search suya", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "snack" in out and "suya" in out
    assert "motto" not in out
    out = control_cgroup(f"memory {gid} list", context=None, chat=None,
                         sender="alice", sender_id="alice")
    assert "motto" in out and "snack" in out


def test_cgroup_memory_forget_needs_owner_or_admin(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    control_cgroup(f"memory {gid} remember motto = ship it",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    # bob joins as a plain member: can read, cannot forget.
    control_cgroup(f"join {gid}", context=None, chat=None,
                   sender="bob", sender_id="bob")
    out = control_cgroup(f"memory {gid} forget motto", context=None,
                         chat=None, sender="bob", sender_id="bob")
    assert "owner" in out.lower() or "admin" in out.lower()
    out = control_cgroup(f"memory {gid} recall motto", context=None,
                         chat=None, sender="bob", sender_id="bob")
    assert "ship it" in out  # still there
    out = control_cgroup(f"memory {gid} forget motto", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "forgot" in out.lower() or "removed" in out.lower()
    out = control_cgroup(f"memory {gid} recall motto", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "never stored" in out.lower() or "no such" in out.lower()


def test_cgroup_memory_persona_renders_from_facts(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    control_cgroup(f"memory {gid} remember persona:name = The Crew",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    control_cgroup(f"memory {gid} remember persona:traits = blunt; funny",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    control_cgroup(f"memory {gid} remember motto = ship it",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    out = control_cgroup(f"memory {gid} persona", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "The Crew" in out
    assert "blunt" in out
    assert "motto" in out and "ship it" in out


def test_cgroup_memory_persona_empty_group_says_so(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    out = control_cgroup(f"memory {gid} persona", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "no stored history" in out.lower()


def test_cgroup_memory_cross_group_isolation(tmp_path, monkeypatch):
    control_cgroup, gid = _cgroup_setup(tmp_path, monkeypatch)
    control_cgroup(f"memory {gid} remember motto = ship it",
                   context=None, chat=None,
                   sender="alice", sender_id="alice")
    created = control_cgroup("new Others | other", context=None,
                             chat=None, sender="alice", sender_id="alice")
    gid2 = re.search(r"`(grp_[^`]+)`", created).group(1)
    control_cgroup(f"join {gid2}", context=None, chat=None,
                   sender="alice", sender_id="alice")
    out = control_cgroup(f"memory {gid2} recall motto", context=None,
                         chat=None, sender="alice", sender_id="alice")
    assert "ship it" not in out
    assert "never stored" in out.lower() or "no such" in out.lower()
