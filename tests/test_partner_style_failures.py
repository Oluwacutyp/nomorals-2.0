"""Tail-33 pins for nomorals/partner/style.py — 0 converts / 9 exclusions.

Each test pins the fallback value of one error site plus a control half
proving the except path was taken (trigger == fallback AND control !=
fallback). The file itself contains no exception handlers, so it adds
zero census sites.

style.py calls no record function anywhere (grep-proven), so NOT-recording
is pinned with the tail-8 snapshot method: recent_failures() before/after
each trigger must be unchanged — a style failure leaves no trace anywhere.

Site 1021 is pinned as OBSERVED behavior only and is FLAGGED for parent
adjudication (see SURVEY.md): an overrides/mood failure discards the whole
computed profile for fixed defaults, and the /style renderer shows those
defaults with no failure marker. The pin does not endorse that behavior.
"""

import random
import sys
from types import SimpleNamespace

from nomorals.agents import failures as ag_fail
from nomorals.partner import style as S
from nomorals.social.chat.platform_limits import for_platform


def _snap():
    return list(ag_fail.recent_failures())


def _prof(**kw):
    d = dict(audience="owner", platform="telegram", max_chars=640,
             target_chars=300, emoji_budget=1, allow_sections=False,
             allow_lists=True, allow_markup=False)
    d.update(kw)
    return S.StyleProfile(**d)


def test_307_repair_echo_failure_returns_empty():
    before = _snap()
    assert S.repair_echo("i love you so much", 12345) == ""
    assert S.repair_echo(
        "i love you so much",
        "i love you so much and also pizza is great") == \
        "and also pizza is great"
    assert _snap() == before


def test_330_lexicon_hits_failure_returns_zero():
    before = _snap()
    assert S.lexicon_hits(12345, ["hello"]) == 0
    assert S.lexicon_hits("hello world hello", ["hello", "world"]) == 2
    assert _snap() == before


def test_732_platform_hard_cap_falls_back_to_mirrored_table(monkeypatch):
    before = _snap()
    monkeypatch.setitem(sys.modules,
                        "nomorals.social.chat.platform_limits", None)
    assert S.platform_hard_cap("telegram") == 4096
    assert S.platform_hard_cap("sms") == 160
    monkeypatch.undo()
    for key in ["telegram", "telegram-bot", "telegram-personal", "whatsapp",
                "discord", "sms", "web", "local"]:
        assert S._PLATFORM_HARD_CAPS[key] == for_platform(key).max_text
    assert S.platform_hard_cap("telegram") == 4096
    assert _snap() == before


def test_816_measure_style_failure_returns_empty_measurement():
    before = _snap()
    assert S.measure_style(123).n == 0
    m = S.measure_style(["hello world this is fine", "short one here ok",
                         "another message here"])
    assert m.n == 3 and m.median_chars == 20.0
    assert _snap() == before


def test_1021_style_for_chat_failure_returns_fixed_defaults():
    # OBSERVED ONLY — flagged for parent adjudication, not endorsed.
    before = _snap()
    d = S.style_for_chat(overrides={"target_chars": "nope"})
    assert (d.audience, d.platform, d.target_chars, d.max_chars,
            d.emoji_budget, d.measured.n) == ("owner", "telegram", 300,
                                              640, 1, 0)
    c = S.style_for_chat(overrides={"target_chars": 200})
    assert (c.target_chars, c.max_chars, c.emoji_budget) == (200, 480, 2)
    d2 = S.style_for_chat(mood_values={"frustration": "high"})
    assert (d2.target_chars, d2.max_chars, d2.emoji_budget) == (300, 640, 1)
    c2 = S.style_for_chat(mood_values={"frustration": 10})
    assert (c2.target_chars, c2.max_chars, c2.emoji_budget) == (300, 480, 2)
    assert _snap() == before


def test_1038_style_prompt_lines_failure_returns_empty():
    before = _snap()
    assert S.style_prompt_lines(_prof(emoji_budget="many")) == []
    assert len(S.style_prompt_lines(_prof())) == 5
    assert _snap() == before


def test_1096_purposeful_emoji_failure_returns_text_unchanged():
    before = _snap()
    t = "great \U0001f602 news \U0001f389 here \U0001f3ae"
    assert S.purposeful_emoji(t, "many") == t
    assert S.purposeful_emoji(t, 1) == "great \U0001f602 news here"
    assert _snap() == before


def test_1196_format_reply_failure_returns_text_unchanged():
    before = _snap()
    t = "great \U0001f602 news \U0001f389 here \U0001f3ae"
    assert S.format_reply(t, _prof(max_chars="many")) == t
    assert S.format_reply(t, _prof()) == "great \U0001f602 news here"
    assert _snap() == before


def test_1335_apply_texting_voice_failure_returns_draft_unchanged():
    before = _snap()
    sp = SimpleNamespace(lowercase_bias=1.0, text_speak={})
    draft = "Hello World THIS Is Great"
    assert S.apply_texting_voice(draft, sp, {"energy": "high"}, "tired",
                                 random.Random(1)) == draft
    assert S.apply_texting_voice(
        draft, sp, {"energy": 60, "frustration": 10, "distance": 30},
        "tired", random.Random(1)) == draft.lower()
    assert _snap() == before
