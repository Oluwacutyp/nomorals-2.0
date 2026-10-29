"""Tail-44 pins for nomorals/browser/tab_tools.py — 0 converts /
6 exclusions, 2 sites flagged (46, 54).

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the except path was
taken (trigger == fallback AND control != fallback), asserting via
the double's own call log that the failing operation itself was
attempted. The file contains no exception handlers, so it adds
zero census sites.

tab_tools.py (and all four solver consumers) call no
failures-ledger record function anywhere (grep-proven: no
record_failure import or call), so NOT-recording is pinned with
the tail-8 snapshot method: agents recent_failures() before/after
each trigger must be unchanged.

Verdict notes (full grounds in the tail-44 SURVEY.md):
- 46 FLAGGED: tab.html() failure -> main doc html "", identical
  to a genuinely empty page; the solvers' settle checks render
  "cleared in-session" ok-True claims from a fresh docs read, so
  a settle-time read failure can dress up as a cleared widget.
  Fix = contract change (unreadable != absent); pin is OBSERVED.
- 54 FLAGGED: frames_html() failure -> main doc only, the
  documented bonus-view degrade; same settle chain as 46, and
  the operative one for frame widgets. Pin is OBSERVED.
- 75: click_at_px failure -> pass; the normalized hand decides
  the result (fallback chain, no claim at the swallow).
- 90: normalized-path failure -> False, the documented value;
  landed-then-raised can only misreport toward failure.
- 112: type_text TypeError -> the signature probe for the
  single-arg focused variant; the fallback types for real.
- 114: type_text failure -> the focused fallback is attempted
  for real and alone decides the result.
- 126: fallback-block failure -> False ("typing itself failed").
- 139: press_key failure -> False; consumers discard the bool
  by design and let the settle re-read decide the solve.
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/nm-tail44-tree")

from nomorals.agents import failures as ag_fail
from nomorals.browser import tab_tools as tt


def _snap():
    return list(ag_fail.recent_failures())


class FakeTab:
    """Duck-typed tab double; every primitive logs to .calls.
    A primitive set to None reads as absent (production checks
    callable()). type_mode: ok2 | focused | raise."""

    def __init__(self, url="https://ex.test/page"):
        self.url = url
        self.calls = []
        self.html_value = "<main>page</main>"
        self.html_exc = None
        self.frames_value = []
        self.frames_exc = None
        self.click_at_px_exc = None
        self.evaluate_value = {"w": 200.0, "h": 100.0}
        self.evaluate_exc = None
        self.click_norm_exc = None
        self.type_mode = "ok2"
        self.box_value = {"x": 10.0, "y": 20.0,
                          "width": 100.0, "height": 40.0}
        self.box_exc = None
        self.press_exc = None

    def html(self):
        self.calls.append(("html",))
        if self.html_exc is not None:
            raise self.html_exc
        return self.html_value

    def frames_html(self):
        self.calls.append(("frames_html",))
        if self.frames_exc is not None:
            raise self.frames_exc
        return self.frames_value

    def click_at_px(self, x, y):
        self.calls.append(("click_at_px", x, y))
        if self.click_at_px_exc is not None:
            raise self.click_at_px_exc

    def evaluate(self, js):
        self.calls.append(("evaluate", js))
        if self.evaluate_exc is not None:
            raise self.evaluate_exc
        return self.evaluate_value

    def click_at(self, nx, ny):
        self.calls.append(("click_at", nx, ny))
        if self.click_norm_exc is not None:
            raise self.click_norm_exc

    def type_text(self, *args):
        self.calls.append(("type_text", args))
        if self.type_mode == "ok2":
            return None
        if len(args) == 2:
            if self.type_mode == "focused":
                raise TypeError("takes 1 arg")
            raise RuntimeError("targeted typing boom")
        return None

    def element_box(self, selector):
        self.calls.append(("element_box", selector))
        if self.box_exc is not None:
            raise self.box_exc
        return self.box_value

    def press_key(self, key):
        self.calls.append(("press_key", key))
        if self.press_exc is not None:
            raise self.press_exc


def _names(tab):
    return [c[0] for c in tab.calls]


def test_46_main_html_failure_yields_empty_main_doc():
    before = _snap()
    t = FakeTab()
    t.html_exc = RuntimeError("read boom")
    t.frames_value = [{"url": "https://ex.test/fr", "html": "<fr>"}]
    docs = tt.docs_for_tab(t)
    assert ("html",) in t.calls  # the read was attempted
    assert docs[0] == {"url": t.url, "html": ""}
    assert len(docs) == 2 and docs[1]["html"] == "<fr>"
    control = tt.docs_for_tab(FakeTab())
    assert control[0]["html"] == "<main>page</main>"
    assert _snap() == before


def test_54_frames_failure_yields_main_doc_only():
    before = _snap()
    t = FakeTab()
    t.frames_exc = RuntimeError("frames boom")
    docs = tt.docs_for_tab(t)
    assert ("frames_html",) in t.calls
    assert docs == [{"url": t.url, "html": "<main>page</main>"}]
    t2 = FakeTab()
    t2.frames_value = [{"url": t2.url, "html": "<main>page</main>"},
                       {"url": "https://ex.test/fr", "html": "<fr>"}]
    docs2 = tt.docs_for_tab(t2)
    assert len(docs2) == 2  # main-frame dupe dropped, frame kept
    assert docs2[1]["url"] == "https://ex.test/fr"
    assert _snap() == before


def test_75_click_at_px_failure_falls_through_to_normalized():
    before = _snap()
    t = FakeTab()
    t.click_at_px_exc = RuntimeError("px boom")
    assert tt.click_px(t, 50.0, 25.0) is True
    assert _names(t)[0] == "click_at_px"  # px hand attempted first
    assert ("click_at", 0.25, 0.25) in t.calls  # real normalized click
    t2 = FakeTab()
    assert tt.click_px(t2, 50.0, 25.0) is True
    assert "click_at" not in _names(t2)  # control never needs it
    assert _snap() == before


def test_90_normalized_path_failure_returns_false():
    before = _snap()
    t = FakeTab()
    t.click_at_px = None
    t.evaluate_exc = RuntimeError("eval boom")
    assert tt.click_px(t, 50.0, 25.0) is False
    assert "evaluate" in _names(t)
    assert "click_at" not in _names(t)  # never reached
    t = FakeTab()
    t.click_at_px = None
    t.click_norm_exc = RuntimeError("norm boom")
    assert tt.click_px(t, 50.0, 25.0) is False
    assert "click_at" in _names(t)  # the click was attempted
    t2 = FakeTab()
    t2.click_at_px = None
    assert tt.click_px(t2, 50.0, 25.0) is True
    assert ("click_at", 0.25, 0.25) in t2.calls
    assert _snap() == before


def test_112_type_text_typeerror_uses_focused_fallback():
    before = _snap()
    t = FakeTab()
    t.type_mode = "focused"
    assert tt.type_answer(t, "#in", "abc") is True
    assert ("type_text", ("#in", "abc")) in t.calls  # 2-arg attempted
    assert ("element_box", "#in") in t.calls
    assert ("click_at_px", 60.0, 40.0) in t.calls  # box centre
    assert ("type_text", ("abc",)) in t.calls  # focused typing landed
    t2 = FakeTab()
    assert tt.type_answer(t2, "#in", "abc") is True
    assert "element_box" not in _names(t2)  # control: direct hand
    assert _snap() == before


def test_114_type_text_failure_uses_focused_fallback():
    before = _snap()
    t = FakeTab()
    t.type_mode = "raise"
    assert tt.type_answer(t, "#in", "abc") is True
    assert ("type_text", ("#in", "abc")) in t.calls
    assert ("type_text", ("abc",)) in t.calls
    t2 = FakeTab()
    assert tt.type_answer(t2, "#in", "abc") is True
    assert "element_box" not in _names(t2)
    assert _snap() == before


def test_126_fallback_box_failure_returns_false():
    before = _snap()
    t = FakeTab()
    t.type_mode = "raise"
    t.box_exc = RuntimeError("box boom")
    assert tt.type_answer(t, "#in", "abc") is False
    assert ("element_box", "#in") in t.calls  # fallback was entered
    assert ("type_text", ("abc",)) not in t.calls  # nothing typed
    t2 = FakeTab()
    t2.type_mode = "raise"
    assert tt.type_answer(t2, "#in", "abc") is True  # healthy box
    assert _snap() == before


def test_139_press_key_failure_returns_false():
    before = _snap()
    t = FakeTab()
    t.press_exc = RuntimeError("key boom")
    assert tt.press_key(t, "Enter") is False
    assert ("press_key", "Enter") in t.calls
    assert tt.press_key(FakeTab(), "Enter") is True
    t3 = FakeTab()
    t3.press_key = None
    assert tt.press_key(t3, "Enter") is False  # no primitive at all
    assert _snap() == before
