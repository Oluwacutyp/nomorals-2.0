"""Pass tail-96 pins: docs/codebeast/build_500k.py — 3 sites, all exclusions.

A standalone dataset-builder script (not a runtime module): its
normalizers process untrusted dataset rows by the million, where a
malformed fragment must be skipped, never fatal to the build and
never repaired into fabricated content.

94 — ``_as_list``: a string field that is not valid JSON -> None,
and None means "field absent" to every caller (the row is dropped
by the normalizers' own emptiness checks). EXCLUSION: parser
semantics — unparseable is indistinguishable from absent, honestly.

110 — ``norm_ultra``: one message stored as a broken JSON string is
skipped (``continue``); the row's remaining messages still normalize.
EXCLUSION: row-level data hygiene. The skip is surgical (proven
below: output with the broken message equals output without it),
and when the skip leaves <2 turns the row drops to None — dropped,
never padded into a fake conversation.

154 — ``norm_conv``: same pattern, same ruling, ShareGPT shape.

The module is loaded by file path (importlib SourceFileLoader) —
it lives outside the package and no sys.path manipulation is used.
"""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[1]
         / "docs" / "codebeast" / "build_500k.py")
_loader = SourceFileLoader("build_500k", str(_PATH))
_spec = spec_from_loader("build_500k", _loader)
b5 = module_from_spec(_spec)
_loader.exec_module(b5)


# ── 94: _as_list ─────────────────────────────────────────────────

def test_94_as_list_unparseable_string_is_absent():
    assert b5._as_list("{not json") is None
    assert b5._as_list("") is None
    # controls: valid JSON list parses; non-list JSON and non-list
    # values are also "absent" by the same contract
    assert b5._as_list('["x", 2]') == ["x", 2]
    assert b5._as_list("5") is None
    assert b5._as_list(["a"]) == ["a"]
    assert b5._as_list(None) is None


# ── 110: norm_ultra skips one broken message ─────────────────────

_GOOD_USER = '{"role": "user", "content": "hi there"}'
_GOOD_ASST = {"role": "assistant", "content": "hello!"}


def test_110_norm_ultra_skips_broken_message_string():
    row = {"messages": [_GOOD_USER, "{broken json", _GOOD_ASST]}
    assert b5.norm_ultra(row) == [["user", "hi there"],
                                  ["assistant", "hello!"]]
    # surgical: identical to the row that never had the broken entry
    clean = {"messages": [_GOOD_USER, _GOOD_ASST]}
    assert b5.norm_ultra(row) == b5.norm_ultra(clean)


def test_110_norm_ultra_row_drops_when_skip_leaves_too_few_turns():
    row = {"messages": [_GOOD_USER, "{broken json"]}
    assert b5.norm_ultra(row) is None


# ── 154: norm_conv skips one broken conversation item ────────────

def test_154_norm_conv_skips_broken_item_string():
    row = {"conversations": [
        '{"from": "human", "value": "a question"}',
        "{broken json",
        {"from": "gpt", "value": "an answer"},
    ]}
    assert b5.norm_conv(row) == [["user", "a question"],
                                 ["assistant", "an answer"]]
    clean = {"conversations": [
        '{"from": "human", "value": "a question"}',
        {"from": "gpt", "value": "an answer"},
    ]}
    assert b5.norm_conv(row) == b5.norm_conv(clean)


def test_154_norm_conv_row_drops_when_skip_leaves_too_few_turns():
    row = {"conversations": ["{broken json",
                             {"from": "gpt", "value": "an answer"}]}
    assert b5.norm_conv(row) is None
