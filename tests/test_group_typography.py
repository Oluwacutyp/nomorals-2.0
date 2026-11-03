"""Typography system (DESIGN §27, owner rulings 2026-11-03
08:52 + 09:04): one shared layer renders every group surface —
theme packs (never a hardcoded look), per-class presets (games
lined, panels clean), tabular amounts, flavor as a quote, and
the owner's footer stamp on presented surfaces only.

Pins:
- primitives render per platform: Telegram HTML (bold title,
  mono amounts, blockquote flavor, italic footer + stamp),
  WhatsApp native marks, Discord markdown — well-formed on
  each.
- theme resolution is settings-driven: group theme beats the
  house default; a per-class override beats the theme; class
  mapping sends games to the board preset and economy panels
  to clean.
- the stamp: present on a balance panel, absent on a
  conversational reply, exactly once even on a re-render,
  custom text honored, group-disabled → no stamp.
- the economy balance + deposit replies render through the
  organ (structure, not golden strings): title, tabular stat
  rows, the disclaimer folded into a quote, the stamp last.
- settings round-trip: /gsettings theme / class / stamp
  grammar persists into the store and changes later renders.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.typography import (
    DEFAULT_STAMP_TEXT,
    Card,
    fmt_amount,
    render_card,
    resolve_style,
    sparks,
)

KEY = "telegram-bot:-1001"
T0 = 1_800_000_000.0


def _panel() -> Card:
    card = Card("economy", title="Ada", icon="💰")
    card.stats([("Wallet", sparks(29), True),
                ("Banked", sparks(1250), True)])
    card.flavor("Sparks are game points only. They have no "
                "monetary value.")
    card.note("banked sparks are safe from robbery")
    return card


# ── primitives per platform ───────────────────────────────────

def test_telegram_render_has_the_house_structure():
    out = render_card(_panel(), platform="telegram-bot")
    assert out.parse_mode == "HTML"
    assert "💰 <b>Ada</b>" in out.text                # title
    assert "<b>Wallet</b>: <code>29 sparks</code>" in out.text
    assert "<code>1,250 sparks</code>" in out.text    # tabular
    assert "<blockquote>" in out.text                 # flavor
    assert "Sparks are game points only." in out.text
    assert "<i>banked sparks are safe from robbery</i>" in out.text
    assert out.text.rstrip().endswith(
        f"<i>{DEFAULT_STAMP_TEXT}</i>")               # stamp last


def test_whatsapp_render_uses_native_marks():
    out = render_card(_panel(), platform="whatsapp")
    assert out.parse_mode == ""
    assert "💰 *Ada*" in out.text
    assert "*Wallet*: ```29 sparks```" in out.text
    assert "▍ Sparks are game points only." in out.text
    assert out.text.rstrip().endswith(f"_{DEFAULT_STAMP_TEXT}_")
    assert "<b>" not in out.text and "<code>" not in out.text


def test_discord_render_is_markdown():
    out = render_card(_panel(), platform="discord")
    assert "💰 **Ada**" in out.text
    assert "**Wallet**: `29 sparks`" in out.text
    assert "> Sparks are game points only." in out.text
    assert out.text.rstrip().endswith(f"_{DEFAULT_STAMP_TEXT}_")


def test_amounts_group_and_tabulate():
    assert fmt_amount(12500) == "12,500"
    assert fmt_amount(29) == "29"
    assert sparks(1000000) == "1,000,000 sparks"


def test_long_flavor_collapses_expandable_on_telegram():
    card = Card("economy", title="Panel")
    card.flavor("x " * 200)
    out = render_card(card, platform="telegram-bot")
    assert "<blockquote expandable>" in out.text


# ── theme resolution ──────────────────────────────────────────

def test_class_mapping_games_board_economy_clean():
    style = resolve_style(None, "game")
    assert style.preset == "board" and style.theme == "house"
    style = resolve_style(None, "economy")
    assert style.preset == "clean"


def test_group_theme_and_class_override_win():
    settings = {"presentation": {"theme": "clean"}}
    assert resolve_style(settings, "game").preset == "clean"
    settings = {"presentation": {"theme": "house",
                                 "classes": {"economy": "board"}}}
    style = resolve_style(settings, "economy")
    assert style.preset == "board" and style.theme == "house"
    settings = {"presentation": {"theme": "board"}}
    assert resolve_style(settings, "economy").preset == "board"


def test_board_preset_lines_and_grids_the_render():
    card = Card("game", title="Slots", icon="🎰")
    card.board("🍒 🍋 🔔")
    card.stats([("Balance", sparks(40), True),
                ("Bet", sparks(10), True)])
    out = render_card(card, platform="telegram-bot")
    assert "━" in out.text                       # the lined look
    assert "<pre>" in out.text                   # board + grid
    assert "🍒 🍋 🔔" in out.text
    assert "Balance" in out.text and "40 sparks" in out.text
    clean = render_card(card, platform="telegram-bot",
                        settings={"presentation":
                                  {"theme": "clean"}})
    assert "━" not in clean.text                 # theme flips it


def test_unknown_theme_and_class_fall_back_safely():
    style = resolve_style({"presentation": {"theme": "nope"}},
                          "game")
    assert style.theme == "house"
    assert resolve_style(None, "nonsense").surface_class == "chat"


# ── the stamp ─────────────────────────────────────────────────

def test_stamp_absent_on_conversational_and_moderation():
    chat = Card("chat", title="Reply")
    chat.line("just talking")
    out = render_card(chat, platform="telegram-bot")
    assert DEFAULT_STAMP_TEXT not in out.text
    mod = Card("moderation", title="Strike ledger", icon="⚖️")
    mod.stats([("Ada", "2 strikes")])
    out = render_card(mod, platform="telegram-bot")
    assert DEFAULT_STAMP_TEXT not in out.text


def test_stamp_exactly_once_and_idempotent():
    card = _panel()
    once = render_card(card, platform="telegram-bot").text
    assert once.count(DEFAULT_STAMP_TEXT) == 1
    # a re-render of already-stamped text (message edits) must
    # not stack a second signature
    from nomorals.social.chat.typography import _append_stamp

    again = _append_stamp(once, DEFAULT_STAMP_TEXT, "html")
    assert again == once


def test_stamp_custom_text_and_group_disable():
    settings = {"presentation": {"stamp_text": "Powered by X"}}
    out = render_card(_panel(), platform="telegram-bot",
                      settings=settings)
    assert out.text.rstrip().endswith("<i>Powered by X</i>")
    settings = {"presentation": {"stamp": False}}
    out = render_card(_panel(), platform="telegram-bot",
                      settings=settings)
    assert DEFAULT_STAMP_TEXT not in out.text
    settings = {"presentation": {"stamp_classes": ["game"]}}
    out = render_card(_panel(), platform="telegram-bot",
                      settings=settings)
    assert DEFAULT_STAMP_TEXT not in out.text  # economy unlisted


# ── the live surfaces, through the runtime ────────────────────

class _Adapter:
    def admin_member(self, chat, user_id):
        return {"status": "administrator"}


def _rig(tmp_path: Path):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    class _RT(RuntimeGroupMixin):
        operator_ids = frozenset()

        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

    rt = _RT()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    return rt, surface


def _msg(*, sender_id="u1", sender="User", platform="telegram-bot"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform=platform,
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender, sender_username="user1",
        sender_role="", text="", meta={}, reply_to="",
        message_id="900")


def _fund(surface: GroupSurface, member: str, amount: int) -> None:
    econ = surface.economy
    econ.verify_member(KEY, member)
    econ.award(KEY, member, amount, "seed", now=T0)


def test_balance_panel_structure_not_wall_of_text(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "u1", 29)
    out = rt._control_balance(_msg())
    text = out.text
    assert "<b>User</b>" in text                       # title
    assert "<b>Wallet</b>: <code>29 sparks</code>" in text
    assert "<blockquote" in text                       # disclaimer
    assert "Sparks are game points only." in text
    assert text.rstrip().endswith(
        f"<i>{DEFAULT_STAMP_TEXT}</i>")                # stamp
    datas = {d for row in out.buttons for _l, d in row}
    assert "deposit all" in datas  # buttons still match display


def test_deposit_reply_is_a_card(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "u1", 50)
    out = rt._control_deposit("all", _msg())
    assert "<b>Deposit</b>" in out.text
    assert "**" not in out.text  # dialect fully rendered
    assert "<b>Wallet</b>: <code>0 sparks</code>" in out.text
    assert "<b>Banked</b>: <code>50 sparks</code>" in out.text
    assert out.text.rstrip().endswith(
        f"<i>{DEFAULT_STAMP_TEXT}</i>")


def test_balance_on_whatsapp_uses_whatsapp_marks(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "u1", 29)
    out = rt._control_balance(_msg(platform="whatsapp"))
    assert "<b>" not in out.text and "<code>" not in out.text
    assert "*Wallet*: ```29 sparks```" in out.text
    assert out.text.rstrip().endswith(f"_{DEFAULT_STAMP_TEXT}_")


def test_settings_round_trip_theme_stamp_class(tmp_path):
    rt, surface = _rig(tmp_path)
    admin = _msg(sender_id="admin1", sender="Admin")
    rt._control_gsettings("stamp off", admin)
    _fund(surface, "u1", 29)
    out = rt._control_balance(_msg())
    assert DEFAULT_STAMP_TEXT not in out.text
    rt._control_gsettings("stamp on", admin)
    rt._control_gsettings("stamp text Powered by the house", admin)
    out = rt._control_balance(_msg())
    assert out.text.rstrip().endswith(
        "<i>Powered by the house</i>")
    # theme + class overrides persist and change later renders
    rt._control_gsettings("theme board", admin)
    settings = surface.store.settings(KEY)
    assert settings["presentation"]["theme"] == "board"
    assert resolve_style(settings, "economy").preset == "board"
    out = rt._control_balance(_msg())
    assert "━" in out.text
    rt._control_gsettings("class economy clean", admin)
    settings = surface.store.settings(KEY)
    assert resolve_style(settings, "economy").preset == "clean"
    # the panel itself reports the presentation state
    panel = rt._control_gsettings("", admin)
    assert "Presentation" in panel.text
    assert "<code>/gsettings theme house|clean|board</code>" \
        in panel.text


def test_gsettings_stamp_toggle_button_flips(tmp_path):
    rt, _surface = _rig(tmp_path)
    admin = _msg(sender_id="admin1", sender="Admin")
    out = rt._control_gsettings("", admin)
    datas = {d for row in out.buttons for _l, d in row}
    assert "gsettings stamp off" in datas  # stamp is ON now
    rt._control_gsettings("stamp off", admin)
    out = rt._control_gsettings("", admin)
    datas = {d for row in out.buttons for _l, d in row}
    assert "gsettings stamp on" in datas


def test_slots_result_is_a_board(tmp_path):
    rt, surface = _rig(tmp_path)
    _fund(surface, "u1", 100)
    out = rt._control_slots("10", _msg())
    assert "━" in out.text                      # board preset
    assert "<pre>" in out.text                  # the reels block
    assert out.text.rstrip().endswith(
        f"<i>{DEFAULT_STAMP_TEXT}</i>")         # game results stamp


def test_games_can_go_clean_per_group(tmp_path):
    rt, surface = _rig(tmp_path)
    admin = _msg(sender_id="admin1", sender="Admin")
    rt._control_gsettings("class game clean", admin)
    _fund(surface, "u1", 100)
    out = rt._control_slots("10", _msg())
    assert "━" not in out.text
