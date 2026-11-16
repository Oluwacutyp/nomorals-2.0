"""WhatsApp Community cross-post (audit gap R2-21).

The organ (nomorals/social/chat/whatsapp_communities.py) plans and
previews a cross-post — plan_crosspost / preview_crosspost /
CommunityRateLimiter — with zero production callers. These tests pin
the wiring: the /community post verb (preview-first, arm-gated,
rate-limited, per-group send) and the new run_crosspost runner.

Rig copied from tests/test_gapfix_whatsapp_digest.py: RuntimeGroupMixin
with monkey-patched seams, bare GroupSurface over a tmp GroupStore.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.whatsapp_communities import (
    CommunityPost,
    CommunityRateLimiter,
    plan_crosspost,
)

WA1 = "whatsapp:120363025000001@g.us"
WA2 = "whatsapp:120363025000002@g.us"
WA3 = "whatsapp:120363025000003@g.us"
TG1 = "telegram:-100111"
NOW = 1780000000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed_community(surface, cid="c1", name="Devs", keys=(WA1, WA2),
                    designate=True):
    from nomorals.social.chat import group_community as _gc

    for key in keys:
        _gc.set_community_tag(surface, key, cid, name)
        if designate and key.startswith("whatsapp:"):
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


class _Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def _rig(tmp_path: Path, *, adapter=None, operator=True):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: operator  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: True  # type: ignore[method-assign]
    mixin._adapter_for = lambda platform: adapter  # type: ignore[method-assign]
    mixin.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    clock = _Clock()
    mixin._community_limiter = CommunityRateLimiter(clock=clock)  # type: ignore[attr-defined]
    mixin._community_clock = clock  # type: ignore[attr-defined]
    return mixin, surface


def _dm_msg(sender_id="owner1", *, platform="whatsapp"):
    chat = SimpleNamespace(key=f"{platform}:{sender_id}",
                           platform=platform, kind=ChatKind.DM,
                           chat_id=sender_id)
    return SimpleNamespace(chat=chat, text="", meta={},
                           sender="someone", sender_id=sender_id)


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


# ── organ: run_crosspost ─────────────────────────────────────────────────

def test_run_crosspost_sends_each_delivery_paced():
    from nomorals.social.chat.whatsapp_communities import run_crosspost

    plan = plan_crosspost(CommunityPost(
        community="Devs", text="hello all", groups=[WA1, WA2]))
    assert plan.ok
    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []

    def send(group, text):
        calls.append((group, text))
        return {"ok": True, "message_id": "m1"}

    record = run_crosspost(plan, send=send, sleep=sleeps.append,
                           pace_seconds=3.0)
    assert record["sent"] == 2 and record["failed"] == 0
    assert calls == [(WA1, "📣 [Devs] hello all"),
                     (WA2, "📣 [Devs] hello all")]
    assert sleeps == [3.0]  # paced between groups, not before the first


def test_run_crosspost_one_failure_does_not_kill_rest():
    from nomorals.social.chat.whatsapp_communities import run_crosspost

    plan = plan_crosspost(CommunityPost(
        community="Devs", text="hello all", groups=[WA1, WA2]))
    assert plan.ok
    calls: list[str] = []

    def send(group, text):
        calls.append(group)
        if group == WA1:
            raise RuntimeError("bridge down")
        return {"ok": True, "message_id": "m1"}

    record = run_crosspost(plan, send=send, sleep=lambda s: None,
                           pace_seconds=3.0)
    assert record["sent"] == 1 and record["failed"] == 1
    assert calls == [WA1, WA2]
    failed = [r for r in record["results"] if not r["ok"]]
    assert len(failed) == 1 and "bridge down" in failed[0]["error"]


def test_plan_rate_limit_refuses_second_post_in_a_minute():
    clock = _Clock()
    limiter = CommunityRateLimiter(clock=clock)
    first = plan_crosspost(
        CommunityPost(community="Devs", text="one", groups=[WA1]),
        limiter=limiter)
    assert first.ok
    second = plan_crosspost(
        CommunityPost(community="Devs", text="two", groups=[WA1]),
        limiter=limiter)
    assert not second.ok and "rate limit" in second.reason
    clock.now += 61.0
    third = plan_crosspost(
        CommunityPost(community="Devs", text="three", groups=[WA1]),
        limiter=limiter)
    assert third.ok


# ── handler: preview ─────────────────────────────────────────────────────

def test_post_preview_shows_exact_text_and_sends_nothing(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "post Devs | hello all", _dm_msg()))
    assert "📣 Community: Devs" in out
    assert "groups (2)" in out
    assert "hello all" in out
    assert "arm first" in out
    assert adapter.sent == []
    # Preview must not burn the 1/min rate token.
    assert mixin._community_limiter.remaining() == 1


def test_post_preview_empty_text_refuses(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community("post Devs |   ", _dm_msg()))
    assert "empty" in out.lower()
    assert adapter.sent == []


def test_post_unknown_community_refuses(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "post Nope | hello", _dm_msg()))
    assert "Nope" in out and "no community" in out
    assert adapter.sent == []


def test_post_telegram_only_community_refuses(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface, keys=(TG1,))
    out = _text(mixin._control_community(
        "post Devs | hello", _dm_msg()))
    assert "WhatsApp" in out
    assert adapter.sent == []


def test_post_held_group_named_not_sent(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    # WA1 designated (passes the §33 gate); WA2 merely sits there.
    _seed_community(surface, keys=(WA1, WA2), designate=False)
    surface.set_designated(WA1, True, by="owner")
    out = _text(mixin._control_community(
        "post Devs | hello all", _dm_msg()))
    assert "groups (1)" in out
    assert "held" in out.lower() and WA2 in out
    assert adapter.sent == []


# ── handler: arm gate + live ─────────────────────────────────────────────

def test_post_live_disarmed_refuses_and_sends_nothing(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "post Devs | hello all | live", _dm_msg()))
    assert "/community arm" in out
    assert adapter.sent == []


def test_post_live_armed_sends_to_each_group(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    assert "ARMED" in _text(mixin._control_community("arm", _dm_msg()))
    out = _text(mixin._control_community(
        "post Devs | hello all | live", _dm_msg()))
    assert "sent 2 of 2" in out
    assert adapter.sent == [(WA1, "📣 [Devs] hello all"),
                            (WA2, "📣 [Devs] hello all")]


def test_post_live_rate_limited_refuses_second_run(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    _text(mixin._control_community("arm", _dm_msg()))
    first = _text(mixin._control_community(
        "post Devs | one | live", _dm_msg()))
    assert "sent 2 of 2" in first
    second = _text(mixin._control_community(
        "post Devs | two | live", _dm_msg()))
    assert "rate limit" in second
    assert len(adapter.sent) == 2  # nothing new sent


def test_post_arm_disarm_status_verbs(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    assert "ARMED" in _text(mixin._control_community("arm", _dm_msg()))
    status = _text(mixin._control_community("status", _dm_msg()))
    assert "armed" in status.lower()
    assert "live runs: 0" in status
    assert "1 post/min" in status
    assert "disarmed" in _text(
        mixin._control_community("disarm", _dm_msg())).lower()
    status = _text(mixin._control_community("status", _dm_msg()))
    assert "disarmed" in status.lower()
    out = _text(mixin._control_community(
        "post Devs | hello | live", _dm_msg()))
    assert "/community arm" in out
    assert adapter.sent == []


def test_post_in_group_is_dm_only(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community(
        "post Devs | hello", _group_msg()))
    assert "DM" in out
    assert adapter.sent == []


def test_show_still_resolves_by_name(tmp_path):
    adapter = _FakeAdapter()
    mixin, surface = _rig(tmp_path, adapter=adapter)
    _seed_community(surface)
    out = _text(mixin._control_community("show Devs", _dm_msg()))
    assert "Devs" in out and WA1 in out and WA2 in out
