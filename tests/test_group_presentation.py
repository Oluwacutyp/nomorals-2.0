"""Presentation build, slice A (owner order 2026-11-03):
formatting + inline buttons are platform-wide, buttons are
authored from the SAME live state that renders the display, and
nothing is hardcoded pattern-matching.

Pins:
- RuntimeGroupMixin._send EXISTS and reaches the gateway (the
  live bug: welcomes called a method defined nowhere — every
  join's welcome died on AttributeError).  This file never stubs
  _send; the join flow runs the real one.
- A join's welcome carries buttons derived from group state
  (sparks row present exactly when the economy runs).
- _send_control_reply unpacks an ActionableMessage into
  text+buttons+HTML parse mode; a plain string sends bare.
- The drop button's label carries THIS drop's per-claim amount.
- The shop offers buy buttons only for items the member can
  actually afford at render time.
- Settings buttons show current state and carry the flip.
- Group help escapes a hostile group title under HTML mode.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"
T0 = 1_800_000_000.0


class _Gateway:
    def __init__(self):
        self.sent: list = []

    def send(self, platform, chat, text, **kw):
        self.sent.append({"platform": platform, "text": text, **kw})
        return {"ok": True}


class _Adapter:
    def admin_member(self, chat, user_id):
        return {"status": "administrator"}


def _rig(tmp_path: Path):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        operator_ids = frozenset()

        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

    rt = _RT()
    rt.gateway = _Gateway()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, sender_id="admin1", sender="Admin", text="",
         meta=None, title="Builders"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title=title),
        sender_id=sender_id, sender=sender, sender_username="boss",
        sender_role="", text=text, meta=meta or {}, reply_to="",
        message_id="900")


def _fund(surface: GroupSurface, member: str, amount: int) -> None:
    econ = surface.economy
    econ.verify_member(KEY, member)
    econ.award(KEY, member, amount, "seed", now=T0)


def _flat_buttons(msg) -> list:
    return [btn for row in (msg.buttons or []) for btn in row]


# ── the _send regression (welcomes were dead live) ────────────

def test_send_reaches_the_gateway_with_buttons(tmp_path):
    rt, _surface = _rig(tmp_path)
    chat = SimpleNamespace(platform="telegram-bot", key=KEY)
    rt._send(chat, "hello", buttons=[[("A", "a")]],
             parse_mode="HTML")
    assert len(rt.gateway.sent) == 1
    sent = rt.gateway.sent[0]
    assert sent["text"] == "hello"
    assert sent["buttons"] == [[("A", "a")]]
    assert sent["parse_mode"] == "HTML"


def test_join_welcome_sends_through_the_real_send(tmp_path):
    rt, _surface = _rig(tmp_path)
    join = _msg(meta={"service": "join", "new_members": [
        {"id": "u9", "username": "newbie", "name": "New B"}]})
    assert rt._group_surface_turn(join) is True
    welcomes = [s for s in rt.gateway.sent if "welcome" in
                str(s["text"]).lower()]
    assert welcomes, "the welcome must actually be sent"
    flat = [btn for row in (welcomes[0].get("buttons") or [])
            for btn in row]
    datas = {data for _label, data in flat}
    # economy runs here by default -> the sparks row is present
    assert {"rules", "balance", "quests"} <= datas


# ── the control-reply channel ─────────────────────────────────

def test_send_control_reply_unpacks_actionable():
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.social.chat.tgbot_buttons import actionable

    rt = object.__new__(PartnerRuntime)
    gw = _Gateway()
    rt.gateway = gw
    rt._typing_for = lambda chat, text: None  # noqa: ARG005
    msg = _msg()
    rt._send_control_reply(
        msg, actionable("<b>hi</b>", [[("Go", "go")]]))
    sent = gw.sent[-1]
    # The unpack contract stands; the always-present stamp (owner
    # law 2026-11-05) rides under the body.
    assert sent["text"] == (
        "<b>hi</b>\n\n<i>𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁</i>")
    assert sent["buttons"] == [[("Go", "go")]]
    assert sent["parse_mode"] == "HTML"
    rt._send_control_reply(msg, "plain reply")
    sent = gw.sent[-1]
    # A plain unadorned reply takes the stamp's lightest form.
    assert sent["text"] == (
        "plain reply\n<i>𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁</i>")
    assert sent["buttons"] is None
    # HTML because the stamp's italic rides — the transports render
    # it down per platform (Telegram entity, WhatsApp _…_).
    assert sent["parse_mode"] == "HTML"


# ── buttons authored from live state ──────────────────────────

def test_drop_button_carries_this_drops_amount(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "admin1", 150)  # award() respects the daily cap
    out = rt._control_drop("100 25", _msg())
    flat = _flat_buttons(out)
    labels = {label for label, _data in flat}
    assert "🎁 Claim 25 sparks" in labels


def test_shop_buttons_only_for_what_the_member_can_afford(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "u1", 35)  # affords nickname_change (20), shout (30)
    out = rt._control_gshop("", _msg(sender_id="u1", sender="U One"))
    datas = {data for _label, data in _flat_buttons(out)}
    assert "gshop buy nickname_change" in datas
    assert "gshop buy shout_out" in datas
    assert "gshop buy pinned_message" not in datas  # costs 60
    assert "not yet" in out.text  # the display says so too


def test_settings_buttons_reflect_current_state(tmp_path):
    rt, _surface = _rig(tmp_path)
    out = rt._control_gsettings("", _msg())
    datas = {data for _label, data in _flat_buttons(out)}
    assert "gsettings welcome off" in datas  # welcome is ON now
    rt._control_gsettings("welcome off", _msg())
    out = rt._control_gsettings("", _msg())
    datas = {data for _label, data in _flat_buttons(out)}
    assert "gsettings welcome on" in datas  # now it offers the flip


def test_group_help_escapes_title_and_matches_economy(tmp_path):
    rt, surface = _rig(tmp_path)
    out = rt._control_group_help(_msg(title="Evil <b>Crew</b>"))
    assert "Evil &lt;b&gt;Crew&lt;/b&gt;" in out.text
    datas = {data for _label, data in _flat_buttons(out)}
    assert "balance" in datas  # economy on by default
    surface.update_settings(KEY, economy_enabled=False)
    out = rt._control_group_help(_msg())
    datas = {data for _label, data in _flat_buttons(out)}
    assert "balance" not in datas
    assert "sparks (the group purse)" not in out.text
