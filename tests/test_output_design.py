"""Output design: platform adaptation, WhatsApp game formatting, styled menus.

Covers:
* nomorals/social/chat/platforms.py — detect_platform, format_for_platform,
  to_whatsapp/to_telegram/to_plain/to_markdown, chunk_text
* nomorals/games/whatsapp_format.py — WA renderers for stats/battle/
  leaderboard/achievements/status/menu/coins
* nomorals/social/chat/style.py — menu_section/menu_item/render_menu
* nomorals/social/chat/control.py — help_menu (styled, platform-aware)
* WhatsAppAdapter.send — HTML → WhatsApp markdown conversion hook

All offline. Every test asserts never-raises on garbage input.
"""

from __future__ import annotations

import unittest

from nomorals.games import whatsapp_format as wf
from nomorals.social.chat import platforms as plat
from nomorals.social.chat import style


class DetectPlatformTests(unittest.TestCase):
    def test_chat_key_prefix(self):
        self.assertEqual(plat.detect_platform("whatsapp:12345"), "whatsapp")
        self.assertEqual(plat.detect_platform("telegram:99"), "telegram")
        self.assertEqual(plat.detect_platform("discord:abc"), "discord")

    def test_aliases(self):
        self.assertEqual(plat.detect_platform("wa:1"), "whatsapp")
        self.assertEqual(plat.detect_platform("tg:1"), "telegram")

    def test_unknown_defaults_telegram(self):
        self.assertEqual(plat.detect_platform("mystery:1"), "telegram")

    def test_garbage_never_raises(self):
        for bad in (None, "", 123, [], {}, object()):
            self.assertEqual(plat.detect_platform(bad), "telegram")


class FormatForPlatformTests(unittest.TestCase):
    def test_telegram_passthrough(self):
        html = "<b>hi</b> <code>/x</code>"
        self.assertEqual(plat.format_for_platform(html, "telegram"), html)

    def test_telegram_upgrades_markdown(self):
        out = plat.format_for_platform("**hi** `code`", "telegram")
        self.assertIn("<b>hi</b>", out)
        self.assertIn("<code>code</code>", out)

    def test_whatsapp_converts_html(self):
        out = plat.format_for_platform("<b>hi</b> <code>/x</code> <i>it</i>", "whatsapp")
        self.assertIn("*hi*", out)
        self.assertIn("```/x```", out)
        self.assertIn("_it_", out)
        self.assertNotIn("<b>", out)

    def test_whatsapp_double_star(self):
        out = plat.format_for_platform("**bold**", "whatsapp")
        self.assertEqual(out, "*bold*")

    def test_whatsapp_links(self):
        out = plat.format_for_platform("[docs](https://x.y)", "whatsapp")
        self.assertIn("https://x.y", out)
        self.assertNotIn("[", out)

    def test_sms_strips_all(self):
        out = plat.format_for_platform("<b>hi</b> **yo** `c`", "sms")
        self.assertNotIn("<", out)
        self.assertNotIn("*", out)
        self.assertIn("hi", out)

    def test_web_markdown(self):
        out = plat.format_for_platform("<b>hi</b>", "web")
        self.assertIn("**hi**", out)

    def test_unknown_platform_is_telegram(self):
        self.assertEqual(plat.format_for_platform("**x**", "carrier-pigeon"),
                         plat.format_for_platform("**x**", "telegram"))

    def test_whatsapp_code_no_double_wrap(self):
        # triple-backtick blocks must survive untouched (no ```x``` → `````x`````)
        self.assertEqual(plat.format_for_platform("```/duel``` ok", "whatsapp"),
                         "```/duel``` ok")
        self.assertEqual(plat.format_for_platform("mix `a` and ```b```", "whatsapp"),
                         "mix ```a``` and ```b```")

    def test_garbage_never_raises(self):
        for bad in (None, 123, [], {}):
            plat.format_for_platform(bad, "whatsapp")
            plat.format_for_platform("ok", bad)


class ChunkTests(unittest.TestCase):
    def test_short_single_chunk(self):
        self.assertEqual(plat.chunk_text("hello", 100), ["hello"])

    def test_long_splits_on_paragraphs(self):
        text = "\n\n".join(f"para {i} " + "x" * 50 for i in range(10))
        chunks = plat.chunk_text(text, 200)
        self.assertTrue(len(chunks) > 1)
        for c in chunks:
            self.assertLessEqual(len(c), 200)

    def test_huge_paragraph_hard_splits(self):
        text = "word " * 500
        chunks = plat.chunk_text(text, 100)
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_empty(self):
        self.assertEqual(plat.chunk_text("", 100), [""])

    def test_garbage_never_raises(self):
        for bad in (None, 123, [], {}):
            self.assertIsInstance(plat.chunk_text(bad, 100), list)


class WhatsAppGameFormatTests(unittest.TestCase):
    def test_is_whatsapp(self):
        self.assertTrue(wf.is_whatsapp("whatsapp:123"))
        self.assertTrue(wf.is_whatsapp("whatsapp"))
        self.assertFalse(wf.is_whatsapp("telegram:123"))
        self.assertFalse(wf.is_whatsapp(None))

    def test_format_game_text_whatsapp(self):
        out = wf.format_game_text("<b>hi</b>", "whatsapp:1")
        self.assertIn("*hi*", out)

    def test_format_game_text_telegram_untouched(self):
        html = "<b>hi</b>"
        self.assertEqual(wf.format_game_text(html, "telegram:1"), html)

    def test_render_stats(self):
        stats = {"strength": 5, "stamina": 8, "mana": 3, "intelligence": 4,
                 "unspent": 2}
        out = wf.render_stats_wa("Ada", stats, {"strength": 1})
        self.assertIn("*📊 Ada's attributes*", out)
        self.assertIn("⚔️", out)
        self.assertIn("(+1 gear)", out)
        self.assertIn("*2*", out)
        self.assertNotIn("<b>", out)

    def test_render_battle(self):
        out = wf.render_battle_wa("Duel", ["Ada strikes!", "Bola parries."], "Ada", "Bola")
        self.assertIn("*⚔️ Duel*", out)
        self.assertIn("🏆 *Ada wins!*", out)
        self.assertIn("_1._", out)

    def test_render_leaderboard(self):
        out = wf.render_leaderboard_wa("Top players", [("Ada", 120), ("Bola", 95), ("Zed", 80), ("Kim", 10)])
        self.assertIn("🥇 *Ada*", out)
        self.assertIn("🥈 *Bola*", out)
        self.assertIn("🥉 *Zed*", out)
        self.assertIn("4. *Kim*", out)

    def test_render_leaderboard_empty(self):
        out = wf.render_leaderboard_wa("Top", [])
        self.assertIn("no rankings yet", out)

    def test_render_achievements(self):
        out = wf.render_achievements_wa("Ada", [{"title": "First Blood", "description": "win a duel"}, "Veteran"])
        self.assertIn("🏅", out)
        self.assertIn("*First Blood*", out)
        self.assertIn("*Veteran*", out)

    def test_render_game_status(self):
        out = wf.render_game_status_wa("chess", ["board: mid", "turn 12"], "Ada to move")
        self.assertIn("*🎮 chess*", out)
        self.assertIn("👉", out)

    def test_render_game_menu(self):
        out = wf.render_game_menu_wa([("chess", "classic"), ("duel", "fight")])
        self.assertIn("`/game chess`", out)
        self.assertIn("`/game duel`", out)

    def test_render_coins(self):
        out = wf.render_coins_wa("Ada", 250, ["iron sword"])
        self.assertIn("*250*", out)
        self.assertIn("iron sword", out)

    def test_wa_bar(self):
        bar = wf.wa_bar(5, 10, 10)
        self.assertEqual(bar, "█████░░░░░")
        self.assertEqual(wf.wa_bar(10, 10, 4), "████")

    def test_wa_chunks(self):
        chunks = wf.wa_chunks("<b>" + "x" * 5000 + "</b>", limit=1000)
        self.assertTrue(len(chunks) > 1)
        for c in chunks:
            self.assertLessEqual(len(c), 1000)
            self.assertNotIn("<b>", c)

    def test_garbage_never_raises(self):
        wf.render_stats_wa(None, None)
        wf.render_battle_wa(None, None)
        wf.render_leaderboard_wa(None, None)
        wf.render_achievements_wa(None, None)
        wf.render_game_status_wa(None, None)
        wf.render_game_menu_wa(None)
        wf.render_coins_wa(None, None)
        wf.wa_bar("x", "y")
        wf.wa_chunks(None)


class MenuStyleTests(unittest.TestCase):
    def test_section_telegram(self):
        self.assertEqual(style.menu_section("🎮", "games", "telegram"), "🎮 <b>GAMES</b>")

    def test_section_whatsapp(self):
        self.assertEqual(style.menu_section("🎮", "games", "whatsapp"), "🎮 *GAMES*")

    def test_item_telegram(self):
        self.assertEqual(style.menu_item("/game list", "all games", "telegram"),
                         "• <code>/game list</code> — all games")

    def test_item_whatsapp(self):
        self.assertEqual(style.menu_item("/game list", "all games", "whatsapp"),
                         "• `/game list` — all games")

    def test_render_menu(self):
        out = style.render_menu([("🎮", "games", [("/game list", "all games")])], "whatsapp")
        self.assertIn("🎮 *GAMES*", out)
        self.assertIn("`/game list`", out)

    def test_render_menu_telegram(self):
        out = style.render_menu([("🎮", "games", [("/game list", "all games")])], "telegram")
        self.assertIn("<b>GAMES</b>", out)
        self.assertIn("<code>/game list</code>", out)

    def test_garbage_never_raises(self):
        style.menu_section(None, None, None)
        style.menu_item(None, None, None)
        style.render_menu(None, None)
        style.render_menu("junk", "junk")


class HelpMenuTests(unittest.TestCase):
    def test_help_menu_renders(self):
        from nomorals.social.chat.control import help_menu
        out = help_menu("whatsapp")
        self.assertIn("*DEVON — COMMAND MENU*", out)
        self.assertIn("🔍 *SEARCH*", out)

    def test_help_menu_telegram(self):
        from nomorals.social.chat.control import help_menu
        out = help_menu("telegram")
        self.assertIn("<b>SEARCH</b>", out)

    def test_help_menu_coverage_matches_help_text(self):
        from nomorals.social.chat.control import help_menu, help_text, CONTROL_COMMANDS
        menu = help_menu("whatsapp")
        missing = [k for k in CONTROL_COMMANDS if f"/{k}" not in menu and f"/{k}" in help_text()]
        self.assertEqual(missing, [], f"menu dropped commands: {missing[:5]}")

    def test_help_text_unchanged(self):
        from nomorals.social.chat.control import help_text
        text = help_text()
        self.assertIn("control commands", text)
        self.assertIn("/status /platforms /help", text)

    def test_garbage_never_raises(self):
        from nomorals.social.chat.control import help_menu
        help_menu(None)
        help_menu("junk")
        help_menu(123)


class WhatsAppAdapterHookTests(unittest.TestCase):
    def test_send_converts_html(self):
        import inspect
        from nomorals.social.chat import whatsapp as wamod
        src = inspect.getsource(wamod.WhatsAppAdapter.send)
        self.assertIn("format_for_platform", src)


if __name__ == "__main__":
    unittest.main()
