"""FontLibrary F3 — the fancy-style engine as data.

Pins:
* the generated tables are proven against the Unicode blocks
  themselves (boundaries, the math-italic h hole, exception
  glyphs, per-family digit coverage);
* the lookalike tables stay inside their claimed scripts and
  the catalogue records their fallbacks honestly;
* overlays compose (base map + strike + wrapper stacks);
* the 89-name generator union resolves — minus the seven
  banner brand names that stay unclaimed, pinned as absent;
* the legacy 18 styles and style.small_caps are BYTE-IDENTICAL
  to the pre-engine outputs, and the duplicate small-caps table
  is gone;
* a style miss is named, with the nearest real names offered.
"""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

from nomorals.social.chat import emoji_art, style
from nomorals.social.chat import text_styles as ts

SAMPLE = "Devon ARENA HhEeOoSsXxQq 0123456789"


# ── generated tables vs the Unicode blocks ───────────────────

def test_math_alphabet_block_boundaries():
    for key, ub, lb in (
            ("serif_bold", 0x1D400, 0x1D41A),
            ("serif_bold_italic", 0x1D468, 0x1D482),
            ("script_bold", 0x1D4D0, 0x1D4EA),
            ("fraktur_bold", 0x1D56C, 0x1D586),
            ("sans", 0x1D5A0, 0x1D5BA),
            ("sans_bold", 0x1D5D4, 0x1D5EE),
            ("sans_italic", 0x1D608, 0x1D622),
            ("sans_bold_italic", 0x1D63C, 0x1D656),
            ("monospace", 0x1D670, 0x1D68A)):
        table = ts.BASE_MAPS[key]
        assert table["A"] == chr(ub)
        assert table["Z"] == chr(ub + 25)
        assert table["a"] == chr(lb)
        assert table["z"] == chr(lb + 25)


def test_math_italic_h_hole_and_exception_glyphs():
    # U+1D455 is a reserved hole in the block — Unicode never
    # assigned it; the glyph lives at U+210E PLANCK CONSTANT.
    try:
        unicodedata.name(chr(0x1D455))
        raise AssertionError("U+1D455 unexpectedly assigned")
    except ValueError:
        pass
    assert ts.BASE_MAPS["serif_italic"]["h"] == "ℎ"
    # the Letterlike-Symbols residents of script/fraktur/
    # double-struck ride as recorded exceptions
    assert ts.BASE_MAPS["script"]["B"] == "ℬ"
    assert ts.BASE_MAPS["script"]["e"] == "ℯ"
    assert ts.BASE_MAPS["script"]["o"] == "ℴ"
    assert ts.BASE_MAPS["fraktur"]["C"] == "ℭ"
    assert ts.BASE_MAPS["fraktur"]["I"] == "ℑ"
    assert ts.BASE_MAPS["double_struck"]["C"] == "ℂ"
    assert ts.BASE_MAPS["double_struck"]["R"] == "ℝ"
    assert ts.BASE_MAPS["double_struck"]["Z"] == "ℤ"


def test_math_digits_only_where_unicode_has_them():
    for key, base in (("serif_bold", 0x1D7CE),
                      ("sans", 0x1D7E2), ("sans_bold", 0x1D7EC),
                      ("monospace", 0x1D7F6),
                      ("double_struck", 0x1D7D8)):
        assert ts.BASE_MAPS[key]["0"] == chr(base)
        assert ts.BASE_MAPS[key]["9"] == chr(base + 9)
    for key in ("serif_italic", "serif_bold_italic", "script",
                "script_bold", "fraktur", "fraktur_bold",
                "sans_italic", "sans_bold_italic"):
        assert "0" not in ts.BASE_MAPS[key]


def test_generated_values_are_real_block_glyphs():
    # alphabets with no exceptions: every value's Unicode name
    # proves it lives in the Mathematical Alphanumeric block
    for key in ("serif_bold", "serif_bold_italic", "script_bold",
                "fraktur_bold", "sans", "sans_bold", "sans_italic",
                "sans_bold_italic", "monospace"):
        for glyph in ts.BASE_MAPS[key].values():
            assert unicodedata.name(glyph).startswith(
                "MATHEMATICAL"), (key, glyph)


def test_enclosed_sets_boundaries():
    circled = ts.BASE_MAPS["circled"]
    assert circled["A"] == chr(0x24B6)
    assert circled["a"] == chr(0x24D0)
    assert circled["0"] == "⓪"
    assert circled["9"] == chr(0x2460 + 8)
    neg = ts.BASE_MAPS["negative_circled"]
    assert neg["A"] == chr(0x1F150)
    assert neg["a"] == chr(0x1F150)  # lowercase folds up
    assert neg["0"] == "⓿"
    assert ts.BASE_MAPS["squared"]["Z"] == chr(0x1F130 + 25)
    assert ts.BASE_MAPS["negative_squared"]["Z"] == chr(
        0x1F170 + 25)
    paren = ts.BASE_MAPS["parenthesized"]
    assert paren["a"] == chr(0x249C)
    assert paren["A"] == chr(0x1F110)
    assert ts.BASE_MAPS["regional"]["a"] == chr(0x1F1E6)
    fw = ts.BASE_MAPS["fullwidth"]
    assert fw["A"] == chr(0x21 + 0xFEE0 + 0x20)
    assert fw[" "] == "　"


# ── lookalike tables: in-script, honestly partial ────────────

def _names(table):
    return [unicodedata.name(g, "") for g in table.values()]


def test_lookalike_tables_stay_in_their_scripts():
    assert all("CYRILLIC" in n
               for n in _names(ts.BASE_MAPS["cyrillic"]))
    assert all("GREEK" in n for n in _names(ts.BASE_MAPS["greek"]))
    assert all("ARABIC" in n
               for n in _names(ts.BASE_MAPS["arabic"]))
    assert all("ARMENIAN" in n
               for n in _names(ts.BASE_MAPS["wizard"]))
    assert all(0x13A0 <= ord(g) <= 0x13F5
               for g in ts.BASE_MAPS["cherokee"].values())
    assert all(0x10330 <= ord(g) <= 0x1034A
               for g in ts.BASE_MAPS["gothic"].values())
    assert all("LISU" in n for n in _names(ts.BASE_MAPS["lisu"]))
    assert all(any(s in n for s in ("THAI", "ARMENIAN"))
               for n in _names(ts.BASE_MAPS["lefthanded"]))


def test_partial_tables_record_their_fallbacks():
    sup = ts.resolve_style("superscript")
    assert sup is not None and sup.lookalike
    assert "q" in sup.fallbacks and "C" in sup.fallbacks
    sub = ts.resolve_style("subscript")
    assert sub is not None and sub.lookalike
    assert "b" in sub.fallbacks and "A" in sub.fallbacks
    cyr = ts.resolve_style("cyrillic")
    assert cyr is not None and cyr.lookalike and cyr.fallbacks
    # and the pass-through behaviour matches the record
    assert ts.apply_style("q", "superscript") == "q"
    assert ts.apply_style("a", "superscript") == "ᵃ"


# ── overlays compose ─────────────────────────────────────────

def test_overlay_stacks_on_map_and_wrapper():
    spec = ts.StyleSpec(name="pin", slug="pin", base="serif_bold",
                        overlays=("strike",), wrapper="sparkle")
    out = ts.apply_spec("ab", spec)
    assert out == "✨ " + "𝐚" + "̶" + "𝐛" + "̶" + " ✨"


def test_named_overlays_exact_marks():
    assert ts.apply_style("ab", "strikethrough") == "a̶b̶"
    assert ts.apply_style("ab", "underlined") == "a̲b̲"
    assert ts.apply_style("ab", "slashed") == "a̷b̷"
    # overlays skip spaces (a struck space is invisible noise)
    assert ts.apply_style("a b", "strikethrough") == "a̶ b̶"


def test_glitch_overlay_is_deterministic():
    one = ts.apply_style("devon", "glitch")
    assert one == ts.apply_style("devon", "glitch")
    assert len(one) > len("devon")


def test_wrapper_forms():
    assert ts.apply_style("hi", "sparkle") == "✨ hi ✨"
    # ♥ between every character, ends included (Picsart Hearts)
    assert ts.apply_style("hi", "hearts") == "♥h♥i♥"
    assert ts.apply_style("hi", "barcode") == "|h|i|"
    assert ts.apply_style("ab", "arrow flow") == "a→b"


# ── the catalogue: the union resolves, the unclaimed don't ──

_UNION_SHIPPED = (
    # Namecheap's 16
    "blackletter", "gothic bold", "circled", "negative circled",
    "monospace", "double struck", "script", "bold script",
    "squared", "negative squared", "serif bold",
    "serif bold italic", "sans serif", "sans serif bold",
    "sans serif italic", "sans serif bold italic",
    # Picsart main + category pages (observed names)
    "light text bubbles", "fraktur", "cryptic italic",
    "math sans", "script bold italic", "fairytale", "hearts",
    "emojis text", "dark text bubbles", "fraktur bold",
    "monospace text", "subscript", "superscript", "full width",
    "parenthesis", "light text squares", "cyrillic", "mirrored",
    "japanese", "arabic", "inverted", "mixed style 1",
    "handwriting", "modern greek", "wizard", "lefthanded",
    "mono upper", "manuscript", "funky", "dark text squares",
    "up and down", "math style", "math sans bold italic",
    "math italic", "math bold", "math sans bold",
    "sweets emojis", "veggies emojis", "math sans italic",
    "diagonal strikes", "ancient style 1", "acute accents",
    "underlined by dashes", "metal umlauts", "mixed style 2",
    "mixed style 3", "delta", "strikethrough", "x cross",
    "stroked", "bandaid", "fading effect", "arrows",
    "air quotes", "food emojis", "volume", "princess style",
    "cupido", "music notes", "love emojis", "sparkles",
    "finish", "drinks emojis", "barcode", "bubbly bubbles",
    "standard text art", "squared letters", "shadows",
    "rectangles", "italic letters", "slant", "block letters",
    "lines and dots",
)

_UNCLAIMED = ("Modern", "Retro Vibes", "Ogre", "Small Letters",
              "Stick Letters", "Sub-Zero", "3D Text Art")


def test_catalogue_covers_the_union():
    # 89 probes: the 82 shipped union styles, seven of them
    # probed under both generators' names for the same style
    assert len(_UNION_SHIPPED) == 89
    for name in _UNION_SHIPPED:
        assert ts.resolve_style(name) is not None, name
        assert ts.apply_style("Aa", name), name


def test_unclaimed_banner_names_stay_unclaimed():
    for name in _UNCLAIMED:
        assert ts.resolve_style(name) is None, name
        assert ts.apply_style("Aa", name) is None, name


def test_catalogue_is_data_sized():
    assert len(ts.CATALOGUE) >= 95
    assert len(ts.style_names()) == len(ts.CATALOGUE)


def test_banner_variants():
    block = ts.apply_style("HI", "block letters")
    lines = block.splitlines()
    assert len(lines) == 5 and "█" in block
    dots = ts.compose_banner("A", "dots")
    assert "●" in dots and "█" not in dots
    slant = ts.compose_banner("A", "slant")
    assert slant.splitlines()[0].startswith(" ")
    assert ts.compose_banner("", "block") == ""


# ── legacy byte-identity (the pre-engine behaviour, pinned) ──

def _oracle_variant(bu, bl, bd, exc=None):
    """The ORIGINAL emoji_art table builder, reimplemented from
    its spec — the independent oracle the engine's generated
    tables must match exactly."""
    table = dict(exc or {})
    for i in range(26):
        table.setdefault(chr(ord("A") + i), chr(bu + i))
        table.setdefault(chr(ord("a") + i), chr(bl + i))
    if bd:
        for i in range(10):
            table.setdefault(str(i), chr(bd + i))
    return table


def test_legacy_tables_match_the_oracle():
    tables = emoji_art._FANCY_TABLES
    assert tables["bold"] == _oracle_variant(0x1D400, 0x1D41A, 0x1D7CE)
    assert tables["italic"] == _oracle_variant(0x1D434, 0x1D44E, 0, {"h": "ℎ"})
    assert tables["bolditalic"] == _oracle_variant(0x1D468, 0x1D482, 0)
    assert tables["script"] == _oracle_variant(
        0x1D49C, 0x1D4B6, 0,
        {"B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ",
        "L": "ℒ", "M": "ℳ", "R": "ℛ", "e": "ℯ", "g": "ℊ",
        "o": "ℴ"})
    assert tables["fraktur"] == _oracle_variant(
        0x1D504, 0x1D51E, 0,
        {"C": "ℭ", "H": "ℌ", "I": "ℑ", "R": "ℜ", "Z": "ℨ"})
    assert tables["doublestruck"] == _oracle_variant(
        0x1D538, 0x1D552, 0x1D7D8,
        {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ",
        "R": "ℝ", "Z": "ℤ"})
    assert tables["monospace"] == _oracle_variant(0x1D670, 0x1D68A, 0x1D7F6)
    assert tables["sans"] == _oracle_variant(0x1D5A0, 0x1D5BA, 0x1D7E2)
    assert tables["sansbold"] == _oracle_variant(0x1D5D4, 0x1D5EE, 0x1D7EC)
    circled = _oracle_variant(0x24B6, 0x24D0, 0, {"0": "⓪"})
    for i in range(1, 10):
        circled[str(i)] = chr(0x2460 + i - 1)
    assert tables["circled"] == circled
    assert tables["fullwidth"] == {
        chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)}
    assert tables["smallcaps"] == dict(zip(
        "abcdefghijklmnopqrstuvwxyz",
        "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ", strict=True))


def test_legacy_outputs_byte_identical():
    # expected strings captured from the pre-engine code
    assert emoji_art.fancy_text(SAMPLE, "bold") == (
        "𝐃𝐞𝐯𝐨𝐧 𝐀𝐑𝐄𝐍𝐀 𝐇𝐡𝐄𝐞𝐎𝐨𝐒𝐬𝐗𝐱𝐐𝐪 "
        "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗")
    assert emoji_art.fancy_text(SAMPLE, "script") == (
        "𝒟ℯ𝓋ℴ𝓃 𝒜ℛℰ𝒩𝒜 ℋ𝒽ℰℯ𝒪ℴ𝒮𝓈𝒳𝓍𝒬𝓆 "
        "0123456789")
    assert emoji_art.fancy_text(SAMPLE, "circled") == (
        "Ⓓⓔⓥⓞⓝ ⒶⓇⒺⓃⒶ ⒽⓗⒺⓔⓄⓞⓈⓢⓍⓧⓆⓠ "
        "⓪①②③④⑤⑥⑦⑧⑨")
    assert emoji_art.fancy_text(SAMPLE, "smallcaps") == (
        "Dᴇᴠᴏɴ ARENA HʜEᴇOᴏSsXxQǫ 0123456789")
    assert emoji_art.fancy_text(SAMPLE, "sparkle") == (
        "✨ Devon ARENA HhEeOoSsXxQq 0123456789 ✨")
    assert emoji_art.fancy_text(SAMPLE, "frame") == (
        "【 Devon ARENA HhEeOoSsXxQq 0123456789 】")
    assert emoji_art.fancy_text(SAMPLE, "waves") == (
        "˗ˏˋ Devon ARENA HhEeOoSsXxQq 0123456789 ˎˊ˗")
    assert len(emoji_art.FANCY_STYLES) == 18
    assert emoji_art.fancy_text("hi", "nope") is None


def test_duplicate_small_caps_table_is_dead():
    assert "_SMALL_CAPS" not in vars(style)
    # byte-identical to the pre-engine outputs (captured)
    assert style.small_caps(SAMPLE) == (
        "ᴅᴇᴠᴏɴ ᴀʀᴇɴᴀ ʜʜᴇᴇᴏᴏssxxǫǫ 0123456789")
    assert style.small_caps("Devon") == "ᴅᴇᴠᴏɴ"
    # and it reads from the engine's one table
    for ch in "abcdefghijklmnopqrstuvwxyz":
        assert style.small_caps(ch) == ts.small_caps_glyph(ch)
        assert style.small_caps(ch.upper()) == ts.small_caps_glyph(ch)


# ── named miss ───────────────────────────────────────────────

def test_named_miss_offers_nearest_styles():
    assert ts.apply_style("hi", "nosuchstyle") is None
    assert ts.resolve_style("nosuchstyle") is None
    near = ts.suggest_styles("scrip")
    assert "Script" in near
    near = ts.suggest_styles("fraktur bold")
    assert "Fraktur Bold" in near


# ── the /emojiart surface ────────────────────────────────────

class _Chat:
    def __init__(self, platform, key):
        from nomorals.social.chat.base import ChatKind

        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = ChatKind.DM


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


def _runtime(tmp_path, adapters, operators=("7",)):
    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
        def __init__(self):
            self.gateway = _Gateway(adapters)
            self.operators = set(operators)
            self.context = SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(tmp_path)))

        def _send(self, chat, text, **kw):
            return SimpleNamespace(ok=True, message_id="1")

        def _is_operator(self, message):
            return str(message.sender_id or "") in self.operators

    return _Runtime()


def test_emojiart_surface_resolves_the_catalogue(tmp_path):
    rt = _runtime(tmp_path, {"whatsapp": object()})
    chat = _Chat("whatsapp", "whatsapp:5")
    # legacy name — byte-identical through the surface
    out = str(rt._control_emojiart("fancy bold devon", _Msg(chat)))
    assert out == "𝐝𝐞𝐯𝐨𝐧"
    # a catalogue-only style resolves through the engine
    out = str(rt._control_emojiart(
        "fancy fraktur_bold devon", _Msg(chat)))
    assert out == ts.apply_style("devon", "fraktur_bold")
    assert out != "devon"
    out = str(rt._control_emojiart(
        "fancy strikethrough devon", _Msg(chat)))
    assert out == ts.apply_style("devon", "strikethrough")
    # the browse verb lists the catalogue
    out = str(rt._control_emojiart("styles", _Msg(chat)))
    assert "Fraktur Bold" in out and "Barcode" in out


def test_emojiart_surface_miss_is_named_with_suggestions(tmp_path):
    rt = _runtime(tmp_path, {"whatsapp": object()})
    chat = _Chat("whatsapp", "whatsapp:5")
    out = str(rt._control_emojiart(
        "fancy scrip devon", _Msg(chat)))
    assert "no style called 'scrip'" in out
    assert "Script" in out  # the nearest real name is offered
    assert "devon" not in out.split("—")[0]
