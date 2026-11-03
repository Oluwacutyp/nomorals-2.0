"""Presentation slice C — from the owner's live-test round 2
(2026-11-03): "/work@Nomorals_bot got a chatty reply instead of
running" and "formatting can be better, still not complete".

Pins:
- parse_control strips the @BotName addressee Telegram's group
  menu inserts — the addressed form IS the command; an unknown
  addressed command still falls through to conversation.
- The remaining group economy replies (mine/work/quests/claim/
  pay) are formatted ActionableMessages with purse buttons.
- Quest claim buttons exist exactly for quests the member can
  claim right now (authored from the same progress rendered).
- Rules render escaped under HTML; the member card escapes a
  hostile display name; the settings usage block is readable
  <code> lines, not a run-on.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.control import parse_control
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"
T0 = 1_800_000_000.0


def _flat(msg) -> dict:
    return {data: label for row in (msg.buttons or [])
            for label, data in row}


# ── the @-form parse fix ──────────────────────────────────────

def test_addressed_command_parses_as_the_command():
    cmd = parse_control("/work@Nomorals_bot")
    assert cmd is not None and cmd.kind == "work"
    cmd = parse_control("/gsettings@Nomorals_bot welcome off")
    assert cmd is not None and cmd.kind == "gsettings"
    assert cmd.tail == "welcome off"
    cmd = parse_control("/mute@Nomorals_bot 10m")
    assert cmd is not None and cmd.kind == "mute"


def test_addressed_unknown_command_still_conversation():
    assert parse_control("/nosuchthing@Nomorals_bot") is None
    assert parse_control("/work") is not None  # plain form intact


# ── group rig (same shape as the presentation tests) ──────────

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
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, sender_id="u1", sender="User", text=""):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender, sender_username="user1",
        sender_role="", text=text, meta={}, reply_to="",
        message_id="900")


def test_mine_and_work_replies_carry_purse_buttons(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.handle_message(KEY, "u1", username="user1", name="User",
                           text="hello group")
    out = rt._control_mine(_msg())
    assert "balance" in _flat(out) and "work" in _flat(out)
    out = rt._control_work(_msg())
    assert "mine" in _flat(out) and "balance" in _flat(out)
    assert "<b>+" in out.text


def test_quest_claim_button_only_for_ready_quests(tmp_path):
    rt, surface = _rig(tmp_path)
    out = rt._control_quests("", _msg())
    assert not any(d.startswith("quests claim")
                   for d in _flat(out))
    # presence first (that's what verifies a real member),
    # then the qualifying activity
    surface.handle_message(KEY, "u1", username="user1", name="User",
                           text="hi all")
    surface.economy.record_activity(KEY, "u1", "invite_joined")
    out = rt._control_quests("", _msg())
    datas = _flat(out)
    assert datas.get("quests claim invite_1") == \
        "🎁 Claim: Bring a Friend"
    claimed = rt._control_quests("claim invite_1", _msg())
    assert "<b>+25 sparks</b>" in claimed.text


def test_rules_and_member_card_escape_html(tmp_path):
    rt, surface = _rig(tmp_path)
    surface.update_settings(
        KEY, rules=["No <b>spam</b> & no flooding"])
    out = rt._control_rules(_msg())
    assert "No &lt;b&gt;spam&lt;/b&gt; &amp; no flooding" in out
    assert "<b>Group rules</b>" in out
    surface.handle_message(KEY, "u7", username="evil",
                           name="Ev<i>l", text="hi")
    card = rt._control_member("@evil", _msg())
    assert "Ev&lt;i&gt;l" in card


def test_gsettings_usage_is_readable_code_lines(tmp_path):
    rt, _surface = _rig(tmp_path)
    out = rt._control_gsettings("", _msg(sender_id="admin1"))
    assert "<code>/gsettings flood N</code>" in out.text
    assert "<code>/gsettings filter add WORD</code>" in out.text
    assert "flood n ·" not in out.text  # the old run-on is gone
