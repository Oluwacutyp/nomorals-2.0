"""Pass #218 pins: social/chat/mock_screens/render.py — 1 exclusion, 0 converts.

- 116: ``_participant_color`` malformed-hex probe. The colour is parsed
  as ``#rrggbb``; when ``int(..., 16)`` raises ValueError the code falls
  through to the deterministic palette colour (sha256 of the participant
  id). The fallback is always a valid palette colour derived from the
  participant itself — the render never fails and nothing is fabricated.
  A participant with a malformed colour gets exactly the colour they
  would get with no colour at all.

mock_screens/render.py calls no failures-ledger record function anywhere,
so NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use plain values, never a swallow.
"""

from nomorals.agents import failures as ag_fail
from nomorals.social.chat.mock_screens.authoring import demo_screen
from nomorals.social.chat.mock_screens.designs import GROUP_NAME_COLORS
from nomorals.social.chat.mock_screens.render import _participant_color
from nomorals.social.chat.mock_screens.schema import Participant


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _screen():
    return demo_screen("group", "android")


# ── L116: the malformed-hex probe falls through to the palette ────────


def test_116_valid_hex_returns_the_parsed_rgb():
    before = _snap()
    assert _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#ff0000"),
        _screen()) == (255, 0, 0)
    assert _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#00F0FF"),
        _screen()) == (0, 240, 255)
    assert _snap() == before


def test_116_malformed_hex_falls_through_to_the_palette():
    # "#zzzzzz" starts with "#" and is 7 chars, so the parse is
    # attempted and ValueError is swallowed; the palette colour
    # (by deterministic hash of id) is used instead.
    before = _snap()
    screen = _screen()
    malformed = _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#zzzzzz"), screen)
    no_colour = _participant_color(
        Participant(id="adaeze", name="Adaeze", color=""), screen)
    assert malformed == no_colour
    assert malformed in GROUP_NAME_COLORS
    assert _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#gggggg"),
        screen) == no_colour
    assert _snap() == before


def test_116_palette_is_deterministic_per_participant():
    before = _snap()
    screen = _screen()
    first = _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#zzzzzz"), screen)
    second = _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#zzzzzz"), screen)
    assert first == second
    other = _participant_color(
        Participant(id="chidi", name="Chidi", color="#zzzzzz"), screen)
    assert other in GROUP_NAME_COLORS
    assert _snap() == before


def test_116_blank_and_short_colours_use_the_palette():
    before = _snap()
    screen = _screen()
    blank = _participant_color(
        Participant(id="adaeze", name="Adaeze", color="   "), screen)
    plain = _participant_color(
        Participant(id="adaeze", name="Adaeze"), screen)
    assert blank == plain
    # "#fff" is too short to enter the parse at all — same palette.
    short = _participant_color(
        Participant(id="adaeze", name="Adaeze", color="#fff"), screen)
    assert short == plain
    assert _snap() == before
