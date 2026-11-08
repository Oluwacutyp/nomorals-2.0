"""Menu grid + decision buttons + live pager (owner reports,
2026-11-07).

Pins:

* the menu keyboards are GRIDS — category buttons pack multiple
  per row in the live menu session (home render and section
  pages) and in the personal category keyboard, never one
  stretched full-width line each; the button walk (callbacks, in
  order) is unchanged by the packing;
* the weekly gift proposal announces through the proposal feed:
  a Card whose Approve/Deny buttons carry the proposal id in
  callback data — the id NEVER appears in member text, and no
  text instructs a typed command the room would refuse;
* an admin decision resolves the proposal IN PLACE (the posted
  announcement edits to its decided state, buttons cleared);
  a non-admin is refused and the proposal stays open; deny
  closes it without moving money; /deny exists as its own group
  handler (it used to fall through to the owner-proposals one);
* /approve and /deny are decidable where proposals post: the
  handle_control grant passes them inside a group (the handler
  enforces the admin rule) while a DM keeps the owner-only
  grant; approve/deny are NOT public commands;
* a bare /help on a buttons+edit surface opens the live menu
  home; an ADDRESSED bare menu (/menu@Nomorals_bot) parses as
  the same bare trigger the public-bot gate now honors;
* choice questions carry buttons AND complete from a typed
  answer: the mute-length ask (the owner's live instance —
  "for how long should I mute @Whynot_Xander?" answered with a
  plain "2h" that completed nothing) renders duration buttons
  (tap executes), records a pending ask that the plain-text
  reply "2h" resolves identically (admin-gated, one-shot), and
  the ⭐ button sets the standing default while muting; the
  /flag category ask rides the same mechanism.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore (rhythm suite), a RuntimeGroupMixin stub (economy
suite), the bare PartnerRuntime (public-bot gating suite).
"""
from __future__ import annotations

import calendar
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatRef, SendResult
from nomorals.social.chat.control import (
    GROUP_DECISION_COMMANDS,
    PUBLIC_COMMANDS,
    menu_session_render,
    parse_control,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.tgbot_buttons import personal_buttons_for

KEY = "telegram-bot:-1001234567"


def _ts(y, mo, d, h=0):
    return float(calendar.timegm((y, mo, d, h, 0, 0, 0, 0, 0)))


DAY = _ts(2023, 11, 14, 12)

LONG_A = ("The London open works because the Asian range gives "
          "you a clean box to trade against. When price sweeps "
          "one side of that box and closes back inside, the "
          "other side is usually the target for the morning. "
          "I wait for the sweep, mark the extreme, and only "
          "then look for an entry on the way back through.")
LONG_B = ("For anyone journaling trades: write the reason BEFORE "
          "the entry, not after the exit. The version of you "
          "that exists in the trade is emotional and certain; "
          "the version that wrote the plan was neither. When "
          "the two disagree mid-trade, believe the paper, not "
          "the pulse in your ears at the time.")
LONG_C = ("Risk first: size from the stop, never from the "
          "feeling. The stop is the market telling you where "
          "you are wrong; the size is just arithmetic after "
          "that. Every blow-up I have witnessed started as "
          "a position sized by confidence instead.")


def _flat(buttons):
    return [data for row in (buttons or []) for _label, data in row]


# ── item 1: the grid ────────────────────────────────────────────


class MenuGridTests(unittest.TestCase):
    def test_home_render_packs_categories(self):
        for platform, public_only in (
                ("telegram", False), ("telegram-personal", False),
                ("telegram-bot", True)):
            render = menu_session_render(
                platform=platform, public_only=public_only,
                home=True)
            rows = render.buttons
            self.assertTrue(rows, platform)
            # No row wider than the packing, and the grid really
            # packs: more than half the rows hold two buttons.
            self.assertTrue(
                all(len(row) <= 2 for row in rows), platform)
            packed = sum(1 for row in rows if len(row) == 2)
            self.assertGreater(packed, len(rows) // 2, platform)
            # All commands closes the render on its own row.
            self.assertEqual(len(rows[-1]), 1, platform)

    def test_home_walk_order_survives_packing(self):
        render = menu_session_render(
            platform="telegram", public_only=False, home=True)
        flat = _flat(render.buttons)
        # Section callbacks in menu order, Power right after
        # Games, and the All-commands row re-points at the first
        # section (the long-standing opener invariant).
        self.assertEqual(flat[-1], flat[0])
        games = flat.index("list games")
        self.assertEqual(flat[games + 1], "power status")
        # The only duplicate in the walk is that closing
        # All-commands re-point at the first section.
        self.assertEqual(len(flat), len(set(flat)) + 1)

    def test_section_page_nav_is_packed_too(self):
        render = menu_session_render(
            "games", platform="telegram", public_only=False)
        rows = render.buttons
        nav = rows[2:]  # after the turn row + Menu home row
        self.assertTrue(nav)
        self.assertTrue(all(len(row) <= 2 for row in nav))
        self.assertTrue(any(len(row) == 2 for row in nav))
        home = menu_session_render(
            platform="telegram", public_only=False, home=True)
        # The section page's nav is the home walk minus the
        # home-only Power row and the closing All-commands row.
        expected = [data for data in _flat(home.buttons)[:-1]
                    if data != "power status"]
        self.assertEqual(_flat(nav), expected)

    def test_personal_category_keyboard_is_a_grid(self):
        kb = personal_buttons_for("the full catalog …", "/help")
        self.assertIsNotNone(kb)
        assert kb is not None
        self.assertTrue(all(len(row) <= 2 for row in kb))
        self.assertGreaterEqual(
            sum(1 for row in kb if len(row) == 2), 4)
        # All commands still closes the grid.
        self.assertEqual(_flat(kb)[-1], "list")


# ── surface rig: the gift proposal feed ─────────────────────────


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _open_gift_proposal(surface: GroupSurface,
                        week: int = 0) -> str:
    """Drive the real rhythm flow until the weekly gift proposal
    opens; ``week`` shifts the messages into a later week (the
    gift is a weekly event, so a second proposal in one test
    needs a later week than the first)."""
    base = DAY + week * 7 * 86400
    block = dict(surface.store.settings(KEY)["rhythm"])
    block.update({"review": {"mode": "auto"}})
    surface.update_settings(KEY, rhythm=block)
    surface.economy.treasury_bonus(KEY, 5000, "seed", now=base)
    notices: list[str] = []
    for i, text in enumerate((LONG_A, LONG_B, LONG_C)):
        out = surface.handle_message(
            KEY, "7", username="u7", name="User 7", text=text,
            message_id=f"m7-w{week}-{i}", ts=base + i * 3600)
        notices.extend(out.get("notices") or [])
    proposals = surface.open_proposals(KEY)
    gift = [pid for pid, p in proposals.items()
            if p.get("by") == "rhythm"]
    assert gift, proposals
    # The announcement NO LONGER rides the text notices: no gift
    # text, no proposal id, no typed instruction anywhere.
    joined = "\n".join(notices)
    assert "Gift proposal" not in joined
    assert gift[0] not in joined
    assert "/approve" not in joined
    return gift[0]


def test_gift_proposal_queues_a_button_announcement(tmp_path):
    surface = _surface(tmp_path)
    pid = _open_gift_proposal(surface)
    feed = surface._extras(KEY).get("proposal_feed") or []
    assert len(feed) == 1, feed
    item = feed[0]
    assert item["proposal"] == pid
    assert item["amount"] > 0 and item["recipient"] == "7"
    taken = surface.proposal_feed_take(KEY)
    assert taken == feed
    assert surface.proposal_feed_take(KEY) == []


def test_deny_spend_closes_without_moving_money(tmp_path):
    surface = _surface(tmp_path)
    pid = _open_gift_proposal(surface)
    treasury = surface.economy.treasury_balance(KEY)
    wallet = surface.economy.balance(KEY, "7")
    out = surface.deny_spend(KEY, pid, by="99", now=DAY)
    assert out["ok"], out
    assert surface.open_proposals(KEY) == {}
    assert surface.economy.treasury_balance(KEY) == treasury
    assert surface.economy.balance(KEY, "7") == wallet
    again = surface.deny_spend(KEY, pid, by="99", now=DAY)
    assert not again["ok"] and again["reason"] == "no_proposal"


# ── mixin rig: drain, decide in place, refusals ─────────────────


class _Adapter:
    """Platform word on members + a recording message editor."""

    def __init__(self, admins):
        self.admins = set(admins)
        self.edits: list = []
        self.restricts: list = []

    def admin_member(self, chat, user_id):
        if str(user_id) in self.admins:
            return {"status": "administrator"}
        return {"status": "member"}

    def admin_restrict(self, chat, user_id, minutes=None,
                       until=None):
        self.restricts.append((str(user_id), minutes))
        return {"ok": True}

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     parse_mode=""):
        self.edits.append((message_id, text, buttons))
        return SendResult(ok=True, platform=chat.platform,
                          message_id=message_id)


class _Gateway:
    def __init__(self):
        self.sends: list = []
        self._n = 0

    def send(self, platform, chat, text, **kwargs):
        self._n += 1
        self.sends.append(SimpleNamespace(
            platform=platform, chat=chat, text=str(text),
            kwargs=kwargs))
        return SendResult(ok=True, platform=platform,
                          message_id=str(900 + self._n))


def _mixin_rig(tmp_path: Path, admins=("admin2",)):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    adapter = _Adapter(admins)
    gateway = _Gateway()

    class _RT(RuntimeGroupMixin):
        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return adapter

        def _is_operator(self, message):  # type: ignore[no-untyped-def]
            return False

    rt = _RT()
    surface = _surface(tmp_path)
    rt._group_surface_obj = surface
    rt.gateway = gateway
    return rt, surface, adapter, gateway


def _gmsg(*, sender_id="admin2", sender="Admin"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind=ChatKind.GROUP, title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username="boss", text="", meta={},
        message_id="1")


def _gmsg_text(text, *, sender_id="admin2", sender="Admin"):
    msg = _gmsg(sender_id=sender_id, sender=sender)
    msg.text = text
    return msg


def test_proposal_feed_posts_buttons_and_decides_in_place(tmp_path):
    rt, surface, adapter, gateway = _mixin_rig(tmp_path)
    pid = _open_gift_proposal(surface)
    msg = _gmsg()
    rt._post_proposal_feed(surface, KEY, msg)
    assert len(gateway.sends) == 1, gateway.sends
    post = gateway.sends[0]
    # The announcement text: contents named, NO proposal id, NO
    # typed instruction.
    assert pid not in post.text
    assert "/approve" not in post.text and "/deny" not in post.text
    assert "User 7" in post.text
    buttons = post.kwargs.get("buttons")
    assert _flat(buttons) == [f"approve {pid}", f"deny {pid}"], \
        buttons
    # A non-admin decision attempt is refused; nothing moves and
    # the announcement is untouched.
    refused = rt._control_approve(pid, _gmsg(sender_id="member1",
                                             sender="Member"))
    assert refused == "approvals are for admins.", refused
    assert pid in surface.open_proposals(KEY)
    assert adapter.edits == []
    # An admin approves: the treasury pays the recipient and the
    # posted announcement EDITS to its decided state (the empty
    # reply means the edit is the answer).
    before = surface.economy.balance(KEY, "7")
    out = rt._control_approve(pid, msg)
    assert out == "", out
    assert surface.open_proposals(KEY) == {}
    assert surface.economy.balance(KEY, "7") > before
    assert len(adapter.edits) == 1, adapter.edits
    mid, text, buttons = adapter.edits[0]
    assert mid == "901" and buttons == []
    assert "approved" in text and pid not in text


def test_deny_button_path_closes_in_place(tmp_path):
    rt, surface, adapter, gateway = _mixin_rig(tmp_path)
    pid = _open_gift_proposal(surface)
    rt._post_proposal_feed(surface, KEY, _gmsg())
    assert len(gateway.sends) == 1
    treasury = surface.economy.treasury_balance(KEY)
    out = rt._control_deny(pid, _gmsg())
    assert out == "", out
    assert surface.open_proposals(KEY) == {}
    assert surface.economy.treasury_balance(KEY) == treasury
    assert len(adapter.edits) == 1
    _mid, text, buttons = adapter.edits[0]
    assert "denied" in text and buttons == [] and pid not in text
    # A second decision on the closed proposal is an honest miss.
    out = rt._control_deny(pid, _gmsg())
    assert "no open proposal" in out, out


def test_bare_approve_lists_proposals_as_buttons(tmp_path):
    rt, surface, _adapter, _gateway = _mixin_rig(tmp_path)
    pid = _open_gift_proposal(surface)
    out = rt._control_approve("", _gmsg())
    assert pid not in out.text
    assert _flat(out.buttons) == [f"approve {pid}", f"deny {pid}"]


# ── the grant: decidable in the room, owner-only in DM ──────────


def _bare_runtime(tmp_path: Path):
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.storage.db import Database

    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(db=db)
    rt._owner_chats = set()
    surface = _surface(tmp_path)
    rt._group_surface_obj = surface
    adapter = _Adapter({"admin2"})
    rt.gateway = SimpleNamespace(
        adapters={"telegram-bot": adapter})
    return rt, surface


def _chat_msg(chat, text, *, sender_id, sender="Admin"):
    return SimpleNamespace(chat=chat, text=text, meta={},
                           sender=sender, sender_id=sender_id)


def test_group_approve_and_deny_pass_the_grant(tmp_path):
    rt, surface = _bare_runtime(tmp_path)
    pid = _open_gift_proposal(surface)
    chat = ChatRef(platform="telegram-bot", chat_id="-1001234567",
                   kind=ChatKind.GROUP)
    assert chat.key == KEY
    # A non-admin in the group reaches the HANDLER's refusal —
    # not the owner grant denial, not the public-hat refusal.
    out = rt.handle_control(
        f"/approve {pid}", chat.key,
        message=_chat_msg(chat, f"/approve {pid}",
                          sender_id="member1", sender="Member"))
    assert out == "approvals are for admins.", out
    # An admin's typed /approve RUNS in the room (the typed path
    # the old announcement instructed now works where it posts).
    out = rt.handle_control(
        f"/approve {pid}", chat.key,
        message=_chat_msg(chat, f"/approve {pid}",
                          sender_id="admin2"))
    text = out.text if hasattr(out, "text") else str(out)
    assert "approved and executed" in text, text
    assert pid not in text
    assert surface.open_proposals(KEY) == {}
    # /deny likewise (a later week: the gift is weekly).
    pid2 = _open_gift_proposal(surface, week=1)
    out = rt.handle_control(
        f"/deny {pid2}", chat.key,
        message=_chat_msg(chat, f"/deny {pid2}",
                          sender_id="admin2"))
    text = out.text if hasattr(out, "text") else str(out)
    assert "denied" in text, text
    assert surface.open_proposals(KEY) == {}


def test_decisions_stay_owner_only_in_dm(tmp_path):
    rt, surface = _bare_runtime(tmp_path)
    pid = _open_gift_proposal(surface)
    dm = ChatRef(platform="telegram-personal", chat_id="42",
                 kind=ChatKind.DM)
    out = rt.handle_control(
        f"/approve {pid}", dm.key,
        message=_chat_msg(dm, f"/approve {pid}",
                          sender_id="stranger", sender="Stranger"))
    assert out == "that one's just for the owner.", out
    assert pid in surface.open_proposals(KEY)


def test_decision_kinds_are_not_public_commands():
    assert frozenset({"approve", "deny"}) == GROUP_DECISION_COMMANDS
    for kind in GROUP_DECISION_COMMANDS:
        assert kind not in PUBLIC_COMMANDS


# ── item 3/4 shapes: addressed menu parses bare; /help session ──


def test_addressed_bare_menu_parses_as_bare():
    cmd = parse_control("/menu@Nomorals_bot")
    assert cmd is not None
    assert cmd.kind == "menu" and cmd.tail == ""
    assert cmd.addressee == "nomorals_bot"
    tailed = parse_control("/menu@Nomorals_bot somegroup")
    assert tailed is not None and tailed.tail != ""


class _DmGateway:
    def __init__(self, adapter, platform):
        self.adapters = {platform: adapter}
        self.sends: list = []

    def send(self, platform, chat, text, **kwargs):
        self.sends.append(SimpleNamespace(
            platform=platform, chat=chat, text=text, kwargs=kwargs))
        return SendResult(ok=True, platform=platform,
                          message_id="100")


def test_bare_help_opens_the_live_menu_home():
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.storage.db import Database

    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(db=db)
    chat = ChatRef(platform="telegram-personal", chat_id="42",
                   kind=ChatKind.DM)
    rt._owner_chats = {chat.key}
    rt.gateway = _DmGateway(_Adapter(set()), "telegram-personal")
    rt._typing_for = lambda chat_, text: None
    msg = SimpleNamespace(chat=chat, text="/help", meta={},
                          sender="owner", sender_id="42")
    out = rt.handle_control("/help", chat.key, message=msg)
    assert not isinstance(out, str)
    assert out.session == "menu"


# ── the mute-length ask (owner live instance, 2026-11-07) ──────
# In his 35-member group Devon asked in plain text: "for how long
# should I mute @Whynot_Xander? /mute @Whynot_Xander 10m|2h|1d —
# or set a standing default: /gsettings mute default 2h." The
# owner typed "2h" as a plain reply; the ask completed nothing
# and he drove three typed steps by hand. The ask now carries
# buttons, and the typed answer resolves it identically.


def _seed_member(surface, uid="777", username="xander",
                 name="Xander"):
    surface.handle_message(KEY, uid, username=username, name=name,
                           text="hello group",
                           message_id=f"seed-{uid}", ts=DAY)


def test_mute_ask_carries_duration_buttons(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    out = rt._control_mute("@xander", _gmsg())
    assert not isinstance(out, str)
    assert "for how long should I mute" in out.text
    # No typed-command instruction remains in the question.
    assert "/mute" not in out.text
    assert "/gsettings" not in out.text
    assert _flat(out.buttons) == [
        "mute @xander 10m", "mute @xander 2h", "mute @xander 1d",
        "mute @xander 2h +default"], out.buttons
    assert adapter.restricts == []
    ask = surface._extras(KEY).get("pending_ask") or {}
    assert ask.get("kind") == "mute_duration"
    assert ask.get("user_id") == "777"


def test_mute_ask_button_tap_executes(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    rt._control_mute("@xander", _gmsg())
    # The tap re-enters as the command the button names.
    out = rt._control_mute("@xander 2h", _gmsg())
    assert "muted for 2h" in out, out
    assert adapter.restricts == [("777", 120)], adapter.restricts
    assert not (surface._extras(KEY).get("pending_ask") or {})


def test_mute_ask_text_reply_executes_identically(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    rt._control_mute("@xander", _gmsg())
    out = rt._resolve_pending_ask(surface, KEY, _gmsg_text("2h"))
    assert out is not None and "muted for 2h" in out, out
    assert adapter.restricts == [("777", 120)], adapter.restricts
    # One-shot: a second "2h" is ordinary chat now.
    assert rt._resolve_pending_ask(
        surface, KEY, _gmsg_text("2h")) is None


def test_mute_ask_answers_are_gated_and_exact(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    rt._control_mute("@xander", _gmsg())
    # A non-admin's "2h" resolves nothing.
    assert rt._resolve_pending_ask(
        surface, KEY, _gmsg_text(
            "2h", sender_id="member1", sender="Member")) is None
    # A non-answer from the asker resolves nothing either.
    assert rt._resolve_pending_ask(
        surface, KEY, _gmsg_text("soon")) is None
    assert adapter.restricts == []
    # The ask is still live for the real answer.
    out = rt._resolve_pending_ask(surface, KEY, _gmsg_text("10m"))
    assert out is not None and "muted for 10m" in out, out
    assert adapter.restricts == [("777", 10)]


def test_mute_ask_star_button_sets_the_standing_default(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    out = rt._control_mute("@xander 2h +default", _gmsg())
    assert "muted for 2h" in out, out
    assert surface.store.settings(KEY)["mute_default_s"] == 7200
    assert adapter.restricts == [("777", 120)]


def test_flag_ask_carries_category_buttons_and_text_reply(tmp_path):
    rt, surface, adapter, _gw = _mixin_rig(tmp_path)
    _seed_member(surface)
    out = rt._control_flag("@xander", _gmsg())
    assert not isinstance(out, str)
    assert "which kind of flag" in out.text
    flat = _flat(out.buttons)
    assert len(flat) == 6, flat
    assert "flag @xander scam" in flat
    assert "flag @xander other" in flat
    # The text reply stages the flag identically to a tap.
    out = rt._resolve_pending_ask(surface, KEY, _gmsg_text("scam"))
    assert out is not None
    proposals = surface.flag_proposals(KEY)
    assert proposals and proposals[0].get("category") == "scam", \
        proposals


if __name__ == "__main__":
    unittest.main()
