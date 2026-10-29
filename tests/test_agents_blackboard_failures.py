"""Tail-37 pins for nomorals/agents/blackboard.py — 8 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback, or the
mock call log proves the failing operation was attempted). The file itself
contains no exception handlers, so it adds zero census sites.

blackboard.py calls no record function anywhere (grep-proven: only _log),
so NOT-recording is pinned with the tail-8 snapshot method:
agency recent_failures() before/after each trigger must be unchanged — a
swallowed failure leaves no trace anywhere. (The module's own _log calls
are not failure records.)

Verdict notes (full grounds in SURVEY.md):
- 93: finalizer guard; fires only on a half-constructed Subscription, whose
  token was never registered, so nothing is left linked. No claim made.
- 157: "(unprintable)" is self-describing; the entry and audit metadata
  are intact. The value itself is stored, only the preview falls back.
- 182: the write already succeeded before watcher dispatch, so post()'s
  return is honest; the gap is observability (trigger failures ARE logged,
  watcher failures are not), not a false claim. No production watch()
  callers today — latent.
- 327: no production callers (latent); the only trigger is a
  type-violating limit (caller bug) or a lock failure. [] is the
  documented empty value either way.
- 374: documented "one bad entry skips" contract; the restored count is
  honest. No production callers (latent).
- 376: the restored count is always accurate — no unguarded code inside
  the outer try can raise after a partial restore. No production
  callers (latent).
- 402: "blackboard (render failed)" explicitly marks failure, distinct
  from the "(empty)" render. No production callers (latent).
- 467: the pass IS the fallthrough to the documented coercion branches
  (dicts/lists recursed, everything else stringified). Only called by
  save(), which has no production callers (latent).
"""

from nomorals.agents import failures as ag_fail
from nomorals.agents.blackboard import Blackboard, Subscription


def _snap():
    return list(ag_fail.recent_failures())


class _BadStr:
    def __str__(self):
        raise RuntimeError("boom")


def test_093_del_half_constructed_does_not_raise():
    before = _snap()
    half = object.__new__(Subscription)
    half.__del__()  # must not raise: cancel() hits missing attrs, caught
    assert _snap() == before
    board = Blackboard()
    tok = object()
    sub = Subscription(board, tok)
    board._links[tok] = ("k*", lambda e: None, sub)
    sub.cancel()
    assert tok not in board._links


def test_157_audit_preview_unprintable():
    board = Blackboard()
    before = _snap()
    bad = _BadStr()
    entry = board.post("k", bad)
    assert entry.value is bad
    assert board._audit[-1]["value_preview"] == "(unprintable)"
    assert board._audit[-1]["key"] == "k" and board._audit[-1]["version"] == 1
    assert _snap() == before
    board.post("k2", "hello")
    assert board._audit[-1]["value_preview"] == "hello"


def test_182_watcher_failure_does_not_break_write():
    board = Blackboard()
    attempted = []

    def raiser(e):
        attempted.append(e.key)
        raise RuntimeError("watcher boom")

    board.watch("k*", raiser)
    before = _snap()
    entry = board.post("k", 1)
    assert entry.value == 1
    assert attempted == ["k"]  # callback WAS attempted: except path taken
    assert board.get("k") == 1
    assert _snap() == before
    seen = []
    board.watch("j*", lambda e: seen.append(e.key))
    board.post("j", 2)
    assert seen == ["j"]


def test_327_history_empty_on_bad_limit():
    board = Blackboard()
    board.post("k", "v1")
    board.post("k", "v2")
    rows = board.history("k")
    assert len(rows) == 2 and all(r["key"] == "k" for r in rows)
    assert board.history("nope") == []
    before = _snap()
    assert board.history("k", limit="bogus") == []
    assert _snap() == before


def test_374_bad_entry_skips_count_honest():
    board = Blackboard()
    before = _snap()
    n = board.import_snapshot({"entries": {
        "good": {"key": "good", "value": 1, "version": 1},
        "bad": {"key": "bad", "value": 2, "version": "xx"},
    }})
    assert n == 1
    assert board.get("good") == 1 and board.get("bad") is None
    assert _snap() == before
    board2 = Blackboard()
    n2 = board2.import_snapshot({"entries": {
        "a": {"key": "a", "value": 1, "version": 1},
        "c": {"key": "c", "value": 3, "version": 2},
    }})
    assert n2 == 2


def test_376_non_dict_import_restores_zero():
    board = Blackboard()
    before = _snap()
    assert board.import_snapshot(["not", "a", "dict"]) == 0
    assert board.import_snapshot({"entries": ["a", "list"]}) == 0
    assert board.import_snapshot(None) == 0
    assert _snap() == before


def test_402_render_failed_labelled():
    board = Blackboard()
    board.post("k", _BadStr())
    before = _snap()
    assert board.render_board() == "blackboard (render failed)"
    assert _snap() == before
    empty = Blackboard()
    assert empty.render_board().endswith("(empty)")
    empty.post("k", "v")
    assert "k" in empty.render_board()


def test_467_json_safe_coerces():
    obj = object()
    assert Blackboard._json_safe(obj) == str(obj)
    assert Blackboard._json_safe({"a": obj}) == {"a": str(obj)}
    assert Blackboard._json_safe([obj]) == [str(obj)]
    assert Blackboard._json_safe({"a": 1}) == {"a": 1}
    assert Blackboard._json_safe(5) == 5
