"""WhatsApp menu — the XMD frame over the real catalog
(owner order 2026-11-05: "one Whatsapp bot menu like that",
replying to a pasted FEI XMD BOT menu).

The presentation is the pasted format — a header box, one boxed
section per category, a powered-by footer.  The content is
Devon's own registry, and these pins hold the two together:

- the header's fields are honest (real prefix grammar, the
  repo's own version, the owner, owner-gated mode) and its
  command total EQUALS the number of command lines rendered;
- drift, both directions: every command line in the menu
  resolves in the dispatcher registry, and every
  WhatsApp-visible registry command is rendered — byte-exact,
  never stylized (small caps are for titles only);
- the platform-impossible kinds stay off the surface;
- the full menu rides in ONE message under WhatsApp's real
  65,536-char protocol cap (platform_limits);
- the footer stamp comes from the typography theme system
  (surface class "menu"), not a literal in the renderer;
- WhatsApp group help renders the same cut-down content —
  §33 claim block included — in the same frame grammar, with
  its buttons intact; Telegram group help is byte-unchanged
  in form (no frame, HTML render).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    WHATSAPP_UNSUPPORTED_KINDS,
    whatsapp_menu,
)
from nomorals.social.chat.platform_limits import for_platform
from nomorals.social.chat.style import small_caps
from nomorals.version import __version__

_CMD_LINE = re.compile(r"^┃ ✫✧│ (/[a-z0-9_]+)", re.M)
_HEADER_TOTAL = re.compile(r"ᴛᴏᴛᴀʟ ᴄᴏᴍᴍᴀɴᴅs: (\d+)")


def _menu_commands(menu: str) -> list[str]:
    return _CMD_LINE.findall(menu)


# ── the header box: honest fields ────────────────────────────

def test_header_box_shape_and_fields():
    menu = whatsapp_menu()
    lines = menu.split("\n")
    assert lines[0] == "╭━━〔 🤖 ᴅᴇᴠᴏɴ 〕━━┈⊷"
    header = lines[1:8]
    assert all(ln.startswith("┃ ✫ ") for ln in header)
    assert lines[8] == "╰━━━━━━━━━━━━┈⊷"
    joined = "\n".join(header)
    # owner, status, the parser's REAL prefix grammar, the
    # repo's own version, the posture in plain words
    assert "ᴏᴡɴᴇʀ: peacethefirst" in joined
    assert "sᴛᴀᴛᴜs: Online" in joined
    assert "ᴘʀᴇꜰɪx: /" in joined
    assert f"ᴠᴇʀsɪᴏɴ: {__version__}" in joined
    assert "ᴍᴏᴅᴇ: owner-gated" in joined
    # a greeting, daypart only — the surface knows no name
    assert "ʜᴇʟʟᴏ:" in joined and "here's what I can do" in joined


def test_header_total_equals_rendered_commands():
    menu = whatsapp_menu()
    m = _HEADER_TOTAL.search(menu)
    assert m, "header carries no command total"
    rendered = _menu_commands(menu)
    assert int(m.group(1)) == len(rendered)
    assert len(rendered) == \
        len(CONTROL_COMMANDS) - len(WHATSAPP_UNSUPPORTED_KINDS)


# ── drift pins, both directions ──────────────────────────────

def test_every_rendered_command_resolves_in_registry():
    for token in _menu_commands(whatsapp_menu()):
        kind = token[1:]
        assert kind in CONTROL_COMMANDS, token
        assert kind not in WHATSAPP_UNSUPPORTED_KINDS, token


def test_every_whatsapp_visible_command_is_rendered():
    rendered = {t[1:] for t in _menu_commands(whatsapp_menu())}
    visible = {k for k in CONTROL_COMMANDS
               if k not in WHATSAPP_UNSUPPORTED_KINDS}
    assert rendered == visible


def test_command_strings_are_byte_exact_ascii():
    # small caps are presentation for TITLES only: a command a
    # member cannot type verbatim is a broken menu.
    for token in _menu_commands(whatsapp_menu()):
        assert token.isascii(), token
        assert token == token.lower(), token


def test_platform_impossible_kinds_absent():
    menu = whatsapp_menu()
    for kind in WHATSAPP_UNSUPPORTED_KINDS:
        assert f"/{kind}" not in _menu_commands(menu)


# ── one message, real cap, theme stamp ───────────────────────

def test_full_menu_fits_one_whatsapp_message():
    cap = for_platform("whatsapp").max_text
    assert cap == 65536
    assert len(whatsapp_menu()) < cap


def test_footer_stamp_resolves_through_typography():
    from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT, resolve_style
    style = resolve_style(None, "menu")
    assert style.stamp  # the menu class is a presented surface
    assert style.stamp_text == DEFAULT_STAMP_TEXT
    assert whatsapp_menu().rstrip().endswith(
        f"_{DEFAULT_STAMP_TEXT}_")


def test_sections_are_registry_categories_in_boxes():
    menu = whatsapp_menu()
    # one box per category, frames balanced
    assert menu.count("╭━━〔") == menu.count("╰━━━━━━━━━━━━┈⊷")
    assert menu.count("╭━━〔") >= 13  # header + 12+ categories
    # the registry's own groupings, small-capped as titles
    for title in ("ᴠᴏɪᴄᴇ & ᴠɪsɪᴏɴ", "ɢᴀᴍᴇs", "ᴛᴏᴏʟs",
                  "ɢʀᴏᴜᴘs & ᴄᴏᴍᴍᴜɴɪᴛɪᴇs"):
        assert f"〔 {title} 〕" in menu or title in menu


def test_page_view_same_frame_and_junk_falls_back():
    page = whatsapp_menu("games")
    assert page.startswith("╭━━〔 🎮 ɢᴀᴍᴇs 〕━━┈⊷")
    assert "/game" in _menu_commands(page)
    assert page.rstrip().endswith("_Powered by peacethefirst_")
    assert whatsapp_menu("no-such-section") == whatsapp_menu()


def test_small_caps_titles_only():
    assert small_caps("Devon") == "ᴅᴇᴠᴏɴ"
    assert small_caps("games — 41") == "ɢᴀᴍᴇs — 41"
    assert small_caps("") == ""
    # non-letters and emoji pass through untouched
    assert small_caps("🎮 games!") == "🎮 ɢᴀᴍᴇs!"


# ── group help in the same frame (WhatsApp only) ─────────────

def _group_help_rig(tmp_path, key, platform):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    class _Gateway:
        def send(self, platform, chat, text, **kw):
            return {"ok": True}

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

    class _RT(RuntimeGroupMixin):
        operator_ids = frozenset()

        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

    rt = _RT()
    rt.gateway = _Gateway()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    msg = SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Crew"),
        sender_id="m1", sender="Member", sender_username="",
        sender_role="", text="/help", meta={}, reply_to="",
        message_id="1")
    return rt, surface, msg


def test_whatsapp_group_help_framed_with_claim_block(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    out = rt._control_group_help(msg)
    text = out.text
    # same cut-down content, §33 claim block included
    assert "what I do here" in text
    assert "not switched on yet" in text
    assert "/gsettings system output on" in text
    # …in the frame grammar
    assert "╭━━〔 ᴛʜɪs ʀᴏᴏᴍ, ᴏɴ ᴡʜᴀᴛsᴀᴘᴘ: 〕━━┈⊷" in text
    assert "╭━━〔 ꜰᴏʀ ᴇᴠᴇʀʏᴏɴᴇ: 〕━━┈⊷" in text
    assert "┃ /rules — the group rules" in text
    # the platform-impossible family stays off this surface too
    assert "/mute" not in text and "/unmute" not in text
    # buttons authored from the same state still ride along
    assert out.buttons


def test_whatsapp_group_help_claimed_framed(tmp_path):
    rt, surface, msg = _group_help_rig(
        tmp_path, "whatsapp:120363@g.us", "whatsapp")
    surface.set_designated("whatsapp:120363@g.us", True, by="owner")
    out = rt._control_group_help(msg)
    assert "switched on — the owner designated this group" \
        in out.text
    assert "╭━━〔" in out.text


def test_telegram_group_help_not_framed(tmp_path):
    rt, _surface, msg = _group_help_rig(
        tmp_path, "telegram:-1001", "telegram")
    out = rt._control_group_help(msg)
    assert "╭━━〔" not in out.text
    assert "<b>" in out.text  # the HTML render, unchanged
    assert "/mute" in out.text  # Telegram keeps its full surface
