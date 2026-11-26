"""CMUdict tiers — the arpabet.rs pattern (§3C.9).

Takes the DESIGN of echelon/arpabet.rs (BSD-4-Clause, Brandon
Thomas 2020) — a CMUdict dictionary layered as tiers, with the
lookup order and the failure modes named — and writes Devon's own
code for it in ``nomorals/media/arpabet.py``, wired into the
existing English G2P rung (``nomorals/media/g2p_en.py``) so
``pronounce()`` and the TTS/singing consumers get it for free.

What is pinned here:
- the full Arpabet inventory the donor declares: 31 consonants,
  76 vowels (19 bases x 4 stresses), 10 punctuation tokens,
  PHONEME_MAP of 107 entries, u8 codes consonants 1-31,
  vowels 101-176, punctuation 201-209, [end] 254;
- the parser: ``WORD PH PH PH`` lines, ``;;;`` comments, words
  lowercased, phonemes uppercased, first entry wins within a
  file, alternates ``WORD(1)`` parsed but only the first kept;
  a malformed line fails the load naming its line number and
  the offending text — never a silent skip; an empty file is
  refused loudly;
- the tier order: overrides > user > core, and every lookup
  names the tier that answered (``G2PWord.tier``) and the source
  (``lexicon_override`` / ``lexicon_user`` / ``lexicon``);
- the strict loader: a user file with a malformed line raises
  instead of loading the good lines; the full inventory now
  loads (AX / DX / EL / Q used to be silently dropped);
- wiring: the ``tonal_g2p`` tool's ``arpabet`` action names the
  tier and its new ``tiers`` action reports the tiers,
  ``nm g2p tiers|load|override`` works from the terminal, and
  ``/g2p tiers|load|override`` answers in chat; the registry
  stays at 786 tools — no new tool was needed (785 after C.8 added
  the robot tool; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media import g2p_en as g
from nomorals.media.arpabet import (
    ALL_CONSONANTS,
    ALL_PUNCTUATION,
    ALL_VOWELS,
    PHONEME_MAP,
    Arpabet,
    ArpabetError,
    SentenceToken,
    Tier,
    TieredLexicon,
    load_from_file,
    load_from_str,
    sentence_to_u8,
    token_to_u8,
    tokenize_sentence,
)
from nomorals.media.g2p_en import g2p_en, lexicon_stats

# A small CMUdict-format fixture: one new word, one core word
# redefined, one alternate, one comment line.
USER_DICT = """\
;;; a test dictionary
ZQXWIBBLE  Z K W IH1 B AH0 L
HELLO  HH EH1 L OW1
HELLO(1)  HH AH0 L OW1 ER0
"""


@pytest.fixture(autouse=True)
def _clean_lexicon():
    """Each test starts from a pristine lexicon and restores
    whatever it found — the tiers are process-global, and a
    test's loaded file must never leak into the next suite."""
    user = dict(g._user_lexicon)
    user_files = list(g._user_loaded_from)
    overrides = dict(g._overrides_lexicon)
    override_files = list(g._overrides_loaded_from)
    cmudict = os.environ.get("NM_CMUDICT_PATH")
    overrides_env = os.environ.get("NM_G2P_OVERRIDES_PATH")
    g._user_lexicon.clear()
    g._user_loaded_from.clear()
    g._overrides_lexicon.clear()
    g._overrides_loaded_from.clear()
    os.environ.pop("NM_CMUDICT_PATH", None)
    os.environ.pop("NM_G2P_OVERRIDES_PATH", None)
    yield
    g._user_lexicon.clear()
    g._user_lexicon.update(user)
    g._user_loaded_from[:] = user_files
    g._overrides_lexicon.clear()
    g._overrides_lexicon.update(overrides)
    g._overrides_loaded_from[:] = override_files
    if cmudict is None:
        os.environ.pop("NM_CMUDICT_PATH", None)
    else:
        os.environ["NM_CMUDICT_PATH"] = cmudict
    if overrides_env is None:
        os.environ.pop("NM_G2P_OVERRIDES_PATH", None)
    else:
        os.environ["NM_G2P_OVERRIDES_PATH"] = overrides_env


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestArpabetOrgan:
    def test_full_inventory_counts(self):
        assert len(ALL_CONSONANTS) == 31
        assert len(ALL_VOWELS) == 76
        assert len(ALL_PUNCTUATION) == 10
        assert len(PHONEME_MAP) == 107

    def test_inventory_covers_the_old_blind_spots(self):
        # The old g2p_en sets missed these — a user file using
        # them had those phonemes silently dropped on load.
        assert "AX" in ALL_VOWELS and "AXR" in ALL_VOWELS
        assert "IX" in ALL_VOWELS and "UX" in ALL_VOWELS
        assert "DX" in ALL_CONSONANTS and "EL" in ALL_CONSONANTS
        assert "EM" in ALL_CONSONANTS and "EN" in ALL_CONSONANTS
        assert "NX" in ALL_CONSONANTS and "Q" in ALL_CONSONANTS
        assert "WH" in ALL_CONSONANTS

    def test_vowel_stress_grades(self):
        for base in ("AA", "AE", "AH", "AO", "AW", "AX", "AXR",
                     "AY", "EH", "ER", "EY", "IH", "IX", "IY",
                     "OW", "OY", "UH", "UW", "UX"):
            assert base in ALL_VOWELS
            assert base + "0" in ALL_VOWELS
            assert base + "1" in ALL_VOWELS
            assert base + "2" in ALL_VOWELS

    def test_u8_codes_match_the_donor(self):
        assert token_to_u8(SentenceToken("phoneme", "B")) == 1
        assert token_to_u8(SentenceToken("phoneme", "AA1")) == 103
        assert token_to_u8(SentenceToken("phoneme", "ZH")) == 31
        assert token_to_u8(SentenceToken("punctuation", "[start]")) == 201
        assert token_to_u8(SentenceToken("punctuation", "[end]")) == 254
        assert token_to_u8(SentenceToken("punctuation", "[comma]")) == 203

    def test_load_from_str_first_entry_wins(self):
        arp = load_from_str(USER_DICT)
        assert arp.get_polyphone("hello") == ["HH", "EH1", "L", "OW1"]
        assert arp.get_polyphone("zqxwibble") == [
            "Z", "K", "W", "IH1", "B", "AH0", "L"]
        assert len(arp) == 2
        assert sorted(arp.keys()) == ["hello", "zqxwibble"]
        assert list(arp) == ["hello", "zqxwibble"]  # __iter__ == keys()

    def test_load_from_str_comments_and_case(self):
        arp = load_from_str(";;; only a comment\n\nGAME  g ey1 m\n")
        assert arp.get_polyphone("GAME") == ["G", "EY1", "M"]
        assert arp.get_polyphone("game") == ["G", "EY1", "M"]

    def test_load_from_str_invalid_format_names_line(self):
        with pytest.raises(ArpabetError) as exc:
            load_from_str("GOOD  G UH1 D\nWAT\n")
        assert exc.value.kind == "invalid_format"
        assert exc.value.line_number == 2
        assert exc.value.text == "WAT"

    def test_load_from_str_empty_file(self):
        with pytest.raises(ArpabetError) as exc:
            load_from_str(";;; nothing but a comment\n")
        assert exc.value.kind == "empty_file"

    def test_load_from_file_missing_is_io(self, tmp_path):
        with pytest.raises(ArpabetError) as exc:
            load_from_file(str(tmp_path / "nope.dict"))
        assert exc.value.kind == "io"

    def test_insert_validates_phonemes(self):
        arp = Arpabet.new()
        with pytest.raises(ArpabetError) as exc:
            arp.insert("nope", ["HH", "BOGUS1"])
        assert exc.value.kind == "invalid_phoneme"
        assert "nope" not in arp

    def test_insert_returns_old_and_remove_drops(self):
        arp = Arpabet.new()
        assert arp.insert("cat", ["K", "AE1", "T"]) is None
        assert arp.insert("cat", ["K", "AE1", "T", "S"]) == [
            "K", "AE1", "T"]
        assert arp.remove("cat") == ["K", "AE1", "T", "S"]
        assert arp.remove("cat") is None

    def test_combine_second_wins(self):
        a = load_from_str("CAT  K AE1 T\n")
        b = load_from_str("CAT  K AE1 T S\nDOG  D AO1 G\n")
        merged = Arpabet.new().combine(a).combine(b)
        assert merged.get_polyphone("cat") == ["K", "AE1", "T", "S"]
        assert merged.get_polyphone("dog") == ["D", "AO1", "G"]

    def test_merge_from_second_wins(self):
        a = load_from_str("CAT  K AE1 T\n")
        b = load_from_str("CAT  K AE1 T S\n")
        a.merge_from(b)
        assert a.get_polyphone("cat") == ["K", "AE1", "T", "S"]


class TestTieredLexicon:
    def _lexicon(self):
        core = Tier("core", "lexicon")
        core.arpabet.insert("hello", ["HH", "AH0", "L", "OW1"])
        core.arpabet.insert("cat", ["K", "AE1", "T"])
        user = Tier("user", "lexicon_user")
        user.arpabet.insert("hello", ["HH", "EH1", "L", "OW1"])
        user.arpabet.insert("dog", ["D", "AO1", "G"])
        override = Tier("override", "lexicon_override")
        override.arpabet.insert("cat", ["K", "AE1", "T", "S"])
        return TieredLexicon([override, user, core])

    def test_precedence_override_beats_user_beats_core(self):
        lex = self._lexicon()
        assert lex.lookup("cat")[0] == ["K", "AE1", "T", "S"]
        assert lex.lookup("hello")[0] == ["HH", "EH1", "L", "OW1"]
        assert lex.lookup("dog")[0] == ["D", "AO1", "G"]

    def test_lookup_names_tier_and_source(self):
        lex = self._lexicon()
        assert lex.lookup("cat") == (["K", "AE1", "T", "S"],
                                     "override", "lexicon_override")
        assert lex.lookup("hello") == (["HH", "EH1", "L", "OW1"],
                                       "user", "lexicon_user")
        assert lex.lookup("dog") == (["D", "AO1", "G"],
                                     "user", "lexicon_user")

    def test_lookup_unknown_returns_none(self):
        assert self._lexicon().lookup("blorp") is None

    def test_load_into_tier_counts_new_coverage_only(self, tmp_path):
        lex = self._lexicon()
        path = _write(tmp_path, "user.dict",
                      "HELLO  HH AH0 L OW1\nNEWWORD  N UW1 W ER0 D\n")
        # hello is already covered by the core tier — only the
        # genuinely new word counts, even though both land in
        # the user tier.
        assert lex.load_into_tier(path, "user") == 1
        assert lex.lookup("newword")[1] == "user"

    def test_load_into_tier_strict_malformed_raises(self, tmp_path):
        lex = self._lexicon()
        path = _write(tmp_path, "bad.dict",
                      "GOOD  G UH1 D\nWAT\n")
        with pytest.raises(ArpabetError) as exc:
            lex.load_into_tier(path, "user")
        assert exc.value.kind == "invalid_format"
        assert lex.lookup("good") is None
        assert path not in lex.tier("user").files

    def test_load_into_tier_empty_file_raises(self, tmp_path):
        lex = self._lexicon()
        path = _write(tmp_path, "empty.dict", "")
        with pytest.raises(ArpabetError) as exc:
            lex.load_into_tier(path, "user")
        assert exc.value.kind == "empty_file"

    def test_load_into_tier_unknown_tier(self, tmp_path):
        lex = self._lexicon()
        path = _write(tmp_path, "x.dict", "WORD  W ER1 D\n")
        with pytest.raises(ArpabetError) as exc:
            lex.load_into_tier(path, "nope")
        assert exc.value.kind == "unknown_tier"

    def test_tier_stats(self):
        lex = self._lexicon()
        assert lex.stats() == {
            "tiers": {"override": 1, "user": 2, "core": 2},
            "total": 3,
            "files": {"override": [], "user": [], "core": []},
        }


class TestSentence:
    def test_tokenize_sentence_punctuation_tokens(self):
        s = tokenize_sentence(
            "Hello, world!",
            lambda w: ["HH", "AH0"] if w == "hello" else None)
        assert [t.to_str() for t in s.tokens] == [
            "[start]", "HH", "AH0", "[comma]", "[space]",
            "[exclamation]", "[end]"]
        assert s.unknown == ["world"]

    def test_sentence_to_u8_codes_and_unknown(self):
        codes, unknown = sentence_to_u8(
            "Hi!",
            lambda w: ["HH", "AY1"] if w == "hi" else None)
        assert codes == [201, 11, 131, 206, 254]
        assert unknown == []

    def test_sentence_to_u8_unknown_word_named(self):
        codes, unknown = sentence_to_u8("Blorp!", lambda w: None)
        assert codes == [201, 206, 254]
        # the unknown list names the word as it was written
        assert unknown == ["Blorp"]


class TestG2pWiring:
    def test_core_word_names_core_tier(self):
        result = g2p_en("hello")
        assert result.source == "lexicon"
        assert result.tier == "core"
        assert result.phonemes == ["HH", "AH0", "L", "OW1"]

    def test_user_word_names_user_tier(self, tmp_path):
        path = _write(tmp_path, "user.dict", USER_DICT)
        assert g.load_cmudict(path) == 1
        result = g2p_en("zqxwibble")
        assert result.source == "lexicon_user"
        assert result.tier == "user"
        assert result.phonemes == ["Z", "K", "W", "IH1", "B", "AH0", "L"]
        # the user layer's redefinition of hello wins over core
        assert g2p_en("hello").phonemes == ["HH", "EH1", "L", "OW1"]

    def test_override_wins_over_user_and_core(self, tmp_path):
        user = _write(tmp_path, "user.dict", USER_DICT)
        over = _write(tmp_path, "over.dict",
                      "HELLO  HH AH0 L OW1\nZQXWIBBLE  Z K W IH1 B AH0 L Z\n")
        assert g.load_cmudict(user) == 1
        assert g.load_overrides(over) == 0  # both already covered
        result = g2p_en("hello")
        assert result.source == "lexicon_override"
        assert result.tier == "override"
        assert result.phonemes == ["HH", "AH0", "L", "OW1"]
        assert g2p_en("zqxwibble").phonemes[-1] == "Z"

    def test_overrides_env_var(self, tmp_path, monkeypatch):
        path = _write(tmp_path, "over.dict",
                      "HELLO  HH EH1 L OW1\n")
        monkeypatch.setenv("NM_G2P_OVERRIDES_PATH", path)
        assert g2p_en("hello").source == "lexicon_override"

    def test_malformed_user_file_raises_not_silent_skip(self, tmp_path):
        path = _write(tmp_path, "bad.dict", "GOOD  G UH1 D\nWAT\n")
        with pytest.raises(ValueError):
            g.load_cmudict(path)
        assert g2p_en("good").source != "lexicon_user"
        assert path not in g.lexicon_stats()["user_files"]

    def test_full_inventory_phoneme_loads(self, tmp_path):
        # AX / DX / EL / Q used to be silently dropped on load.
        path = _write(tmp_path, "full.dict",
                      "BOTTLE  B AA1 DX AH0 L\n"
                      "TABLE  T EY1 B AX L\n"
                      "TRAVEL  T R AE1 V AX L\n")
        assert g.load_cmudict(path) == 3
        assert g2p_en("bottle").phonemes == [
            "B", "AA1", "DX", "AH0", "L"]
        assert g2p_en("table").phonemes == [
            "T", "EY1", "B", "AX", "L"]

    def test_lexicon_stats_reports_tiers(self, tmp_path):
        path = _write(tmp_path, "user.dict", USER_DICT)
        g.load_cmudict(path)
        stats = lexicon_stats()
        assert stats["tiers"] == {
            "override": 0, "user": 2, "core": 145}
        assert stats["total"] == 146  # hello was already in core
        assert stats["override_files"] == []
        assert stats["user_files"] == [path]

    def test_blorp_still_learned(self):
        result = g2p_en("blorp")
        assert result.source == "learned"
        assert result.tier == ""
        assert result.phonemes == ["B", "L", "AO1", "R", "P"]


class TestWiring:
    def test_tool_count_still_788(self):
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # No new tool in this lane: the tiers ride the existing
        # tonal_g2p tool, the nm g2p CLI and the /g2p chat verb.
        # 785 after C.8 added the robot tool; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class.
        assert len(registry._tools) == 788
        assert "tonal_g2p" in registry._tools

    def test_tool_tonal_g2p_arpabet_action_names_tier(self):
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        outcome = registry.call("tonal_g2p", action="arpabet",
                                word="hello")
        assert getattr(outcome, "ok", False), outcome
        assert outcome.value["phonemes"] == ["HH", "AH0", "L", "OW1"]
        assert outcome.value["source"] == "lexicon"
        assert outcome.value["tier"] == "core"

    def test_tool_tonal_g2p_tiers_action(self):
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        outcome = registry.call("tonal_g2p", action="tiers")
        assert getattr(outcome, "ok", False), outcome
        assert outcome.value["tiers"] == {
            "override": 0, "user": 0, "core": 145}
        assert outcome.value["total"] == 145

    def test_cli_g2p_tiers_load_override(self, tmp_path, capsys):
        from nomorals.cmdline.dispatch import main

        user = _write(tmp_path, "user.dict", USER_DICT)
        over = _write(tmp_path, "over.dict",
                      "HELLO  HH AH0 L OW1\n")
        assert main(["g2p", "load", user]) == 0
        assert "added: 1" in capsys.readouterr().out
        assert main(["g2p", "override", over]) == 0
        assert "added: 0" in capsys.readouterr().out
        assert main(["g2p", "tiers"]) == 0
        out = capsys.readouterr().out
        assert "'override': 1" in out
        assert "'user': 2" in out
        assert "'core': 145" in out
        assert "total: 146" in out
        assert main(["g2p", "pronounce", "hello"]) == 0
        out = capsys.readouterr().out
        assert "HH AH0 L OW1" in out
        assert "lexicon_override" in out

    def test_cli_g2p_load_bad_file_fails(self, tmp_path, capsys):
        from nomorals.cmdline.dispatch import main

        path = _write(tmp_path, "bad.dict", "GOOD  G UH1 D\nWAT\n")
        assert main(["g2p", "load", path]) == 1
        assert "failed" in capsys.readouterr().err

    def test_chat_g2p_tiers_and_load(self, tmp_path):
        from nomorals.agents.partner.runtime_voice import (
            RuntimeVoiceMixin,
        )

        mixin = RuntimeVoiceMixin()
        reply = mixin._control_g2p("tiers", "test")
        assert "override: 0 words" in reply
        assert "user: 0 words" in reply
        assert "core: 145 words" in reply
        user = _write(tmp_path, "user.dict", USER_DICT)
        reply = mixin._control_g2p(f"load {user}", "test")
        assert "user tier (+1 new words)" in reply
        reply = mixin._control_g2p("zqxwibble", "test")
        assert "Z K W IH1 B AH0 L" in reply
        assert "lexicon_user" in reply
        over = _write(tmp_path, "over.dict",
                      "HELLO  HH AH0 L OW1\n")
        reply = mixin._control_g2p(f"override {over}", "test")
        assert "override tier (+0 new words)" in reply
        assert "HH AH0 L OW1" in mixin._control_g2p("hello", "test")
        assert "g2p load failed" in mixin._control_g2p(
            "load /nonexistent.dict", "test")


if __name__ == "__main__":
    import unittest

    unittest.main()
