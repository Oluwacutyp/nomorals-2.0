"""§3C.11 sentence.rs — TTS tokenization + segmentation.

The donor vectors (echelon/sentence.rs `src/lib.rs` tests) are
reproduced exactly as (kind, text) sequences; the fixes (multi-
punctuation runs, ellipsis, em-dash, quotes) and the roadmap kinds
(currency, percentage, ordinal, date, time, emoji) are pinned on
top; segmentation pins the abbreviation rules and the ambiguous
flag; wiring pins the three voice consumers, the CLI and the chat
verb. No new brain tool: the registry stays at 787.
"""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from nomorals.media.sentence import (
    Punctuation,
    segment,
    segment_iter,
    segment_text,
    tokenize,
    tokenize_iter,
)
from nomorals.tools.registry import ToolRegistry


def _kinds(text: str) -> list[tuple[str, str]]:
    return [(t.kind, t.text) for t in tokenize(text)]


def _texts(text: str) -> list[str]:
    return segment_text(text)


class DonorVectorTests(unittest.TestCase):
    """Every test case from the donor's src/lib.rs, exact."""

    def test_simple_sentence(self):
        self.assertEqual(_kinds("this is an example"), [
            ("Word", "this"), ("Word", "is"), ("Word", "an"),
            ("Word", "example"),
        ])

    def test_simple_sentence_with_punctuation(self):
        self.assertEqual(
            _kinds("This, right here, is a sentence."), [
                ("Word", "This"), ("Punctuation", ","),
                ("Word", "right"), ("Word", "here"),
                ("Punctuation", ","), ("Word", "is"),
                ("Word", "a"), ("Word", "sentence"),
                ("Punctuation", "."),
            ])

    def test_hyphenated_words(self):
        self.assertEqual(_kinds("Please double-check the drive-thru"), [
            ("Word", "Please"),
            ("HyphenatedWord", "double-check"), ("Word", "the"),
            ("HyphenatedWord", "drive-thru"),
        ])
        self.assertEqual(
            _kinds("Please double-check the drive-thru - pretty-please."),
            [
                ("Word", "Please"),
                ("HyphenatedWord", "double-check"), ("Word", "the"),
                ("HyphenatedWord", "drive-thru"),
                ("Punctuation", "-"),
                ("HyphenatedWord", "pretty-please"),
                ("Punctuation", "."),
            ])

    def test_apostrophe_words(self):
        self.assertEqual(_kinds("Gotta catch 'em all."), [
            ("Word", "Gotta"), ("Word", "catch"),
            ("ApostrophenatedWord", "'em"), ("Word", "all"),
            ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("It ain't you, darlin'"), [
            ("Word", "It"), ("ApostrophenatedWord", "ain't"),
            ("Word", "you"), ("Punctuation", ","),
            ("ApostrophenatedWord", "darlin'"),
        ])
        self.assertEqual(_kinds("That isn't freakin'."), [
            ("Word", "That"), ("ApostrophenatedWord", "isn't"),
            ("ApostrophenatedWord", "freakin'"), ("Punctuation", "."),
        ])
        self.assertEqual(
            _kinds("How're y'all doin' at the O'Grady's'?"), [
                ("ApostrophenatedWord", "How're"),
                ("ApostrophenatedWord", "y'all"),
                ("ApostrophenatedWord", "doin'"),
                ("Word", "at"), ("Word", "the"),
                ("ApostrophenatedWord", "O'Grady's'"),
                ("Punctuation", "?"),
            ])
        self.assertEqual(_kinds("'nuff said"), [
            ("ApostrophenatedWord", "'nuff"), ("Word", "said"),
        ])
        self.assertEqual(_kinds("It's 5 o'clock"), [
            ("ApostrophenatedWord", "It's"), ("Integer", "5"),
            ("ApostrophenatedWord", "o'clock"),
        ])

    def test_integers(self):
        self.assertEqual(_kinds("9 out of 10 agree"), [
            ("Integer", "9"), ("Word", "out"), ("Word", "of"),
            ("Integer", "10"), ("Word", "agree"),
        ])
        self.assertEqual(_kinds("1, 2, 3, 100."), [
            ("Integer", "1"), ("Punctuation", ","),
            ("Integer", "2"), ("Punctuation", ","),
            ("Integer", "3"), ("Punctuation", ","),
            ("Integer", "100"), ("Punctuation", "."),
        ])

    def test_real_numbers(self):
        self.assertEqual(_kinds("The total comes to 25.15."), [
            ("Word", "The"), ("Word", "total"), ("Word", "comes"),
            ("Word", "to"), ("RealNumber", "25.15"),
            ("Punctuation", "."),
        ])

    def test_comma_numbers(self):
        self.assertEqual(
            _kinds("1,000,000 people have 1,234.56 points"), [
                ("CommaFormattedInteger", "1,000,000"),
                ("Word", "people"), ("Word", "have"),
                ("CommaFormattedRealNumber", "1,234.56"),
                ("Word", "points"),
            ])

    def test_colon(self):
        self.assertEqual(_kinds("one: two"), [
            ("Word", "one"), ("Punctuation", ":"), ("Word", "two"),
        ])
        self.assertEqual(_kinds("one : two"), [
            ("Word", "one"), ("Punctuation", ":"), ("Word", "two"),
        ])
        self.assertEqual(_kinds("this:"), [
            ("Word", "this"), ("Punctuation", ":"),
        ])

    def test_question(self):
        self.assertEqual(_kinds("what? no"), [
            ("Word", "what"), ("Punctuation", "?"), ("Word", "no"),
        ])
        self.assertEqual(_kinds("what ? no"), [
            ("Word", "what"), ("Punctuation", "?"), ("Word", "no"),
        ])
        self.assertEqual(_kinds("what?"), [
            ("Word", "what"), ("Punctuation", "?"),
        ])
        self.assertEqual(_kinds("?"), [("Punctuation", "?")])

    def test_exclamation(self):
        self.assertEqual(_kinds("yes! that"), [
            ("Word", "yes"), ("Punctuation", "!"), ("Word", "that"),
        ])
        self.assertEqual(_kinds("yes ! that"), [
            ("Word", "yes"), ("Punctuation", "!"), ("Word", "that"),
        ])
        self.assertEqual(_kinds("yes!"), [
            ("Word", "yes"), ("Punctuation", "!"),
        ])

    def test_semicolon(self):
        self.assertEqual(_kinds("one; two"), [
            ("Word", "one"), ("Punctuation", ";"), ("Word", "two"),
        ])
        self.assertEqual(_kinds("one ; two"), [
            ("Word", "one"), ("Punctuation", ";"), ("Word", "two"),
        ])
        self.assertEqual(_kinds("one;"), [
            ("Word", "one"), ("Punctuation", ";"),
        ])

    def test_dash(self):
        self.assertEqual(_kinds("but- no"), [
            ("Word", "but"), ("Punctuation", "-"), ("Word", "no"),
        ])
        self.assertEqual(_kinds("but - no"), [
            ("Word", "but"), ("Punctuation", "-"), ("Word", "no"),
        ])
        self.assertEqual(_kinds("but-"), [
            ("Word", "but"), ("Punctuation", "-"),
        ])

    def test_urls(self):
        self.assertEqual(_kinds("Go to https://google.com"), [
            ("Word", "Go"), ("Word", "to"),
            ("Url", "https://google.com"),
        ])
        self.assertEqual(_kinds("Go to https://www.google.com."), [
            ("Word", "Go"), ("Word", "to"),
            ("Url", "https://www.google.com"), ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("My website is http://127.0.0.1"), [
            ("Word", "My"), ("Word", "website"), ("Word", "is"),
            ("Url", "http://127.0.0.1"),
        ])
        self.assertEqual(_kinds(
            "My website is http://127.0.0.1/my/page.html?foo=bar&bin=baz#hah"
        ), [
            ("Word", "My"), ("Word", "website"), ("Word", "is"),
            ("Url",
             "http://127.0.0.1/my/page.html?foo=bar&bin=baz#hah"),
        ])

    def test_hashtags(self):
        self.assertEqual(_kinds("#hashtag"), [("Hashtag", "#hashtag")])
        self.assertEqual(_kinds("This is #rust #awesomeness!"), [
            ("Word", "This"), ("Word", "is"),
            ("Hashtag", "#rust"), ("Hashtag", "#awesomeness"),
            ("Punctuation", "!"),
        ])

    def test_usernames(self):
        self.assertEqual(_kinds("@echelon"),
                         [("UsernameMention", "@echelon")])
        self.assertEqual(_kinds("The author is @echelon."), [
            ("Word", "The"), ("Word", "author"), ("Word", "is"),
            ("UsernameMention", "@echelon"), ("Punctuation", "."),
        ])

    def test_empty_strings(self):
        for text in ("", "  ", "\t\t", "\n\n\n", "\n \t \n"):
            self.assertEqual(tokenize(text), [], repr(text))

    def test_donor_no_infinite_loop_inputs_terminate(self):
        # The donor only asserts these terminate; Devon tokenizes
        # them properly instead of leaving one Unknown behind.
        self.assertEqual(_kinds("."), [("Punctuation", ".")])
        self.assertEqual(_kinds("..."), [("Punctuation", "...")])
        self.assertEqual(_kinds(". . ."), [
            ("Punctuation", "."), ("Punctuation", "."),
            ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("yes!!!!!"), [
            ("Word", "yes"),
            ("Punctuation", "!"), ("Punctuation", "!"),
            ("Punctuation", "!"), ("Punctuation", "!"),
            ("Punctuation", "!"),
        ])
        self.assertEqual(_kinds("yes!!!!1??"), [
            ("Word", "yes"),
            ("Punctuation", "!"), ("Punctuation", "!"),
            ("Punctuation", "!"), ("Punctuation", "!"),
            ("Integer", "1"),
            ("Punctuation", "?"), ("Punctuation", "?"),
        ])
        self.assertEqual(_kinds("dashes--emdash"), [
            ("Word", "dashes"), ("Punctuation", "--"),
            ("Word", "emdash"),
        ])
        self.assertEqual(_kinds("This does not work!?"), [
            ("Word", "This"), ("Word", "does"), ("Word", "not"),
            ("Word", "work"), ("Punctuation", "!"),
            ("Punctuation", "?"),
        ])
        self.assertEqual(_kinds("will-o'-the-wisp"), [
            ("HyphenatedWord", "will-o'-the-wisp"),
        ])
        self.assertEqual(_kinds("I'm sorry you can't do it."), [
            ("ApostrophenatedWord", "I'm"), ("Word", "sorry"),
            ("Word", "you"), ("ApostrophenatedWord", "can't"),
            ("Word", "do"), ("Word", "it"), ("Punctuation", "."),
        ])
        self.assertEqual(_kinds('That is "good" enough'), [
            ("Word", "That"), ("Word", "is"),
            ("Punctuation", '"'), ("Word", "good"),
            ("Punctuation", '"'), ("Word", "enough"),
        ])


class BeyondDonorKindTests(unittest.TestCase):
    """The donor roadmap kinds, built for real."""

    def test_currency(self):
        self.assertEqual(_kinds("It costs $12.50."), [
            ("Word", "It"), ("Word", "costs"),
            ("Currency", "$12.50"), ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("£3 and €1,000.00 and ₦1,500"), [
            ("Currency", "£3"), ("Word", "and"),
            ("Currency", "€1,000.00"), ("Word", "and"),
            ("Currency", "₦1,500"),
        ])

    def test_percentage(self):
        self.assertEqual(_kinds("45% off"), [
            ("Percentage", "45%"), ("Word", "off"),
        ])

    def test_ordinal(self):
        self.assertEqual(_kinds("1st 2nd 3rd 21st place"), [
            ("Ordinal", "1st"), ("Ordinal", "2nd"),
            ("Ordinal", "3rd"), ("Ordinal", "21st"),
            ("Word", "place"),
        ])

    def test_date(self):
        self.assertEqual(_kinds("on 2026-11-20."), [
            ("Word", "on"), ("Date", "2026-11-20"),
            ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("on 20/11/2026."), [
            ("Word", "on"), ("Date", "20/11/2026"),
            ("Punctuation", "."),
        ])

    def test_time(self):
        self.assertEqual(_kinds("at 10:30."), [
            ("Word", "at"), ("Time", "10:30"), ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("at 10:30am."), [
            ("Word", "at"), ("Time", "10:30am"), ("Punctuation", "."),
        ])
        self.assertEqual(_kinds("at 10:30 a.m."), [
            ("Word", "at"), ("Time", "10:30 a.m"),
            ("Punctuation", "."),
        ])

    def test_emoji(self):
        self.assertEqual(_kinds("I love it 😂!"), [
            ("Word", "I"), ("Word", "love"), ("Word", "it"),
            ("Emoji", "😂"), ("Punctuation", "!"),
        ])
        self.assertEqual(_kinds("❤️"), [("Emoji", "❤️")])
        self.assertEqual(_kinds("👍🏽 ok"), [
            ("Emoji", "👍🏽"), ("Word", "ok"),
        ])

    def test_version_number_is_unknown_like_the_donor(self):
        # "version2.0" matches no donor kind either — Unknown, not a
        # faked RealNumber with a word glued on.
        self.assertEqual(_kinds("version2.0"), [
            ("Unknown", "version2.0"),
        ])

    def test_offsets_slice_back(self):
        text = "Hello, world! It costs $12.50."
        for tok in tokenize(text):
            self.assertEqual(text[tok.start:tok.end], tok.text)
        self.assertEqual(tokenize(text)[0].start, 0)
        self.assertEqual(tokenize(text)[-1].end, len(text))

    def test_iterators_match_lists(self):
        text = "Dr. Smith paid $12.50 for 1,000 items! Really?"
        self.assertEqual(list(tokenize_iter(text)), tokenize(text))
        self.assertEqual([s.text for s in segment_iter(text)],
                         segment_text(text))

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            tokenize(42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            segment(None)  # type: ignore[arg-type]


class SegmentationTests(unittest.TestCase):
    def test_abbreviations_do_not_end_sentences(self):
        self.assertEqual(
            _texts("Dr. Smith went home. He left."),
            ["Dr. Smith went home.", "He left."])
        self.assertEqual(
            _texts("Mr. Jones and Mrs. Jones met Prof. Plum."),
            ["Mr. Jones and Mrs. Jones met Prof. Plum."])

    def test_dotted_abbreviations(self):
        self.assertEqual(
            _texts("Use e.g. this and i.e. that. Done."),
            ["Use e.g. this and i.e. that.", "Done."])
        self.assertEqual(
            _texts("The U.S.A. is big. Yes."),
            ["The U.S.A. is big.", "Yes."])

    def test_decimals_and_numbers_do_not_split(self):
        self.assertEqual(_texts("It costs $3.14. Really?"),
                         ["It costs $3.14.", "Really?"])
        self.assertEqual(_texts("Pi is 3.14. Ok."),
                         ["Pi is 3.14.", "Ok."])
        self.assertEqual(_texts("No. 5 arrived. Fig. 3 shows it."),
                         ["No. 5 arrived.", "Fig. 3 shows it."])

    def test_endable_abbreviations(self):
        self.assertEqual(
            _texts("Bring pens, etc. and paper. Done."),
            ["Bring pens, etc. and paper.", "Done."])
        self.assertEqual(
            _texts("Bring pens, etc. He left."),
            ["Bring pens, etc.", "He left."])

    def test_am_pm_end_sentences(self):
        # am/pm are time words, not sentence-suppressing
        # abbreviations: "10 am." ends a sentence.
        self.assertEqual(_texts("He arrived at 10 am. He left."),
                         ["He arrived at 10 am.", "He left."])
        self.assertEqual(_texts("He arrived at 10:30 a.m. He left."),
                         ["He arrived at 10:30 a.m.", "He left."])

    def test_initial_is_ambiguous(self):
        sentences = segment("J. Smith arrived. He stayed.")
        self.assertEqual([s.text for s in sentences],
                         ["J. Smith arrived.", "He stayed."])
        self.assertTrue(sentences[0].ambiguous)
        self.assertFalse(sentences[1].ambiguous)
        self.assertEqual(sentences[0].terminator, Punctuation.Period)

    def test_quotes_attach(self):
        self.assertEqual(_texts('He said "go." She left.'),
                         ['He said "go."', "She left."])

    def test_ellipsis_rules(self):
        # Capitalized next word: a boundary.
        self.assertEqual(_texts("Wait... What?"),
                         ["Wait...", "What?"])
        # Lowercase next word: a trailing-off pause, one sentence.
        self.assertEqual(_texts("He trailed off... and then left."),
                         ["He trailed off... and then left."])

    def test_blank_line_forces_boundary(self):
        self.assertEqual(
            _texts("Hello world\n\nGoodbye world"),
            ["Hello world", "Goodbye world"])

    def test_no_terminal_punctuation(self):
        self.assertEqual(_texts("Hello world"), ["Hello world"])
        self.assertEqual(_texts(""), [])
        self.assertEqual(_texts("   "), [])

    def test_terminators_and_offsets(self):
        sentences = segment("Hello! Really? Yes.")
        self.assertEqual([s.terminator for s in sentences], [
            Punctuation.Exclamation, Punctuation.Question,
            Punctuation.Period,
        ])
        text = "Hello! Really? Yes."
        for s in sentences:
            self.assertEqual(text[s.start:s.end], s.text)
        self.assertEqual(sentences[0].start, 0)
        self.assertEqual(sentences[-1].end, len(text))

    def test_street_abbreviation_is_a_known_limitation(self):
        # "St." is a title (St. Louis) even though it can also be a
        # street: "Main St." merges with the next sentence. The
        # merge error is the lesser evil for TTS, and it is
        # documented in the organ, not hidden.
        self.assertEqual(_texts("St. Louis is nice."),
                         ["St. Louis is nice."])
        self.assertEqual(_texts("Main St. He moved."),
                         ["Main St. He moved."])


class WiringTests(unittest.TestCase):
    def test_director_segment_uses_the_engine(self):
        from nomorals.voice.director import PerformanceTuner

        self.assertEqual(
            PerformanceTuner().segment("Dr. Smith went home. He left."),
            ["Dr. Smith went home.", "He left."])

    def test_tts_split_sentences(self):
        from nomorals.voice.tts import _split_sentences

        self.assertEqual(
            _split_sentences("Dr. Smith went home. He left."),
            ["Dr. Smith went home.", "He left."])
        # Hard line breaks still split (the old behaviour).
        self.assertEqual(
            _split_sentences("First line.\nSecond line."),
            ["First line.", "Second line."])
        self.assertEqual(_split_sentences(""), [])
        self.assertEqual(_split_sentences("  "), [])

    def test_tool_count_still_807(self):
        # No new brain tool in this lane: the sentence surface is
        # `nm sentence` + `/sentence` over the shared engine.
        # 785 after C.8 added the robot tool; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 807)

    def test_cli_split(self):
        from nomorals.cmdline.dispatch import main

        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["sentence", "split",
                       "Dr. Smith went home. He left."])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("1. [0:20] Dr. Smith went home.", out)
        self.assertIn("2. [21:29] He left.", out)

    def test_cli_tokens(self):
        from nomorals.cmdline.dispatch import main

        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["sentence", "tokens", "Hello, world!"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Word: 'Hello' [0:5]", out)
        self.assertIn("Punctuation: ',' [5:6]", out)

    def test_cli_split_json(self):
        import json

        from nomorals.cmdline.dispatch import main

        buf = StringIO()
        with redirect_stdout(buf):
            rc = main(["sentence", "--json", "split",
                       "J. Smith arrived. He stayed."])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 2)
        self.assertTrue(data[0]["ambiguous"])
        self.assertEqual(data[0]["terminator"], ".")

    def test_chat_sentence(self):
        from nomorals.agents.partner.runtime_voice import (
            RuntimeVoiceMixin,
        )

        mixin = RuntimeVoiceMixin()
        reply = mixin._control_sentence(
            "Dr. Smith went home. He left.", "test")
        self.assertIn("1. Dr. Smith went home.", reply)
        self.assertIn("2. He left.", reply)
        reply = mixin._control_sentence("tokens Hello, world!", "test")
        self.assertIn("Word: 'Hello'", reply)
        self.assertIn("Punctuation: ','", reply)
        self.assertIn("/sentence <text>",
                      mixin._control_sentence("", "test"))

    def test_sentence_not_on_public_menu(self):
        # Owner-only verb: PUBLIC_COMMANDS stays 279.
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        self.assertEqual(len(PUBLIC_COMMANDS), 279)
        self.assertNotIn("sentence", PUBLIC_COMMANDS)

    def test_sentence_has_one_list_group_and_help(self):
        from nomorals.social.chat.control import (
            _HELP_GROUPS,
            COMMAND_DETAILS,
            LIST_GROUPS,
            LIST_ONELINERS,
        )

        groups = [g for g, kinds in LIST_GROUPS if "sentence" in kinds]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], "voice & vision")
        help_groups = [g for g, kinds in _HELP_GROUPS
                       if "sentence" in kinds]
        self.assertEqual(len(help_groups), 1)
        self.assertEqual(help_groups[0], "voice & vision")
        self.assertIn("sentence", COMMAND_DETAILS)
        self.assertIn("sentence", LIST_ONELINERS)


if __name__ == "__main__":
    unittest.main()
