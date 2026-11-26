"""Gate-discovery contract tests (gapfix121).

The contract lives in nomorals/browser/gate_discovery.py and
is consumed by social/chat/webnav.py.  These tests prove the
eight handoff findings are actually fixed in code — not just
named — against scripted probes (no live browser needed):

F1 click mis-targeting · F2 apostrophe disagreement ·
F3 hidden dialog treated as standing · F4 fixed 600 ms
verification · F5 missing frame discovery · F6 ignored
ARIA controls · F7 remembered-consent suppression ·
F8 swallowed failures.
"""
import pytest

from nomorals.browser.gate_discovery import (
    GateDiscovery,
    JobProjection,
    discover_gates,
    fold_text,
    norm_text,
    run_gate_pass,
)
from nomorals.social.chat import webnav
from nomorals.social.chat.webnav import (
    GateBlocked,
    GateMemory,
)

COOKIE_PAGE = (
    "<html><body><div class=\"cookie-banner\">We use cookies "
    "<button id=\"cookie-accept\">Accept all cookies</button>"
    "</div></body></html>")
CONSENT_PAGE = (
    "<html><body><div class=\"onetrust\">We value your privacy "
    "<button>Accept all</button></div></body></html>")
AGE_PAGE = (
    "<html><body><div class=\"modal\">You must be 18 "
    "<button>I\u2019m 18 or older</button></div></body></html>")
CONTENT_PAGE = "<html><body><h1>Real Page</h1></body></html>"
DOB_PAGE = (
    "<html><body><div class=\"modal\">Verify your age "
    "<form><input name=\"dob\" type=\"date\"></form></div>"
    "</body></html>")
FRAME_CONSENT = (
    "<html><body><div class=\"onetrust\">We value your privacy "
    "<button>Accept all</button></div></body></html>")


# ── scripted probes ─────────────────────────────────────────


class FakeProbe:
    """A PageProbe driven by a script.  click_gate moves to
    the page the gate was hiding (unless told to refuse);
    html() only clears after ``clear_after_waits`` waits,
    which proves the settle loop waits past a fixed sleep."""

    def __init__(self, html, *, url="https://x.example/",
                 after_gate=None, click_works=True,
                 clear_after_waits=0, frames=()):
        self._html = html
        self._after_gate = after_gate
        self._click_works = click_works
        self._clear_after_waits = clear_after_waits
        self._frames = list(frames)
        self._url = url
        self.gate_clicks = []
        self.waits = []
        self._waits_done = 0

    def url(self):
        return self._url

    def title(self):
        return "Real Page"

    def html(self):
        if self._clear_after_waits and \
                self._waits_done < self._clear_after_waits:
            return self._html
        if self._after_gate is not None and self.gate_clicks:
            return self._after_gate
        return self._html

    def reload(self):
        pass

    def wait(self, seconds):
        self.waits.append(seconds)
        self._waits_done += 1

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        return self._click_works

    def click_checkbox(self):
        return True

    def frames(self):
        # A clicked frame gate clears: once the pass has
        # clicked, the frame no longer shows a gate.
        if self.gate_clicks:
            return []
        return list(self._frames)


class OldProbe:
    """A probe WITHOUT the frames() seam — the contract must
    still work on it (tolerant discovery)."""

    def __init__(self, html, *, url="https://x.example/",
                 after_gate=None):
        self._html = html
        self._after_gate = after_gate
        self._url = url
        self.gate_clicks = []

    def url(self):
        return self._url

    def title(self):
        return "Real Page"

    def html(self):
        if self._after_gate is not None and self.gate_clicks:
            return self._after_gate
        return self._html

    def reload(self):
        pass

    def wait(self, seconds):
        pass

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        return True

    def click_checkbox(self):
        return True


class FailingClickProbe(FakeProbe):
    """The FIRST click overall fails; later clicks work.
    Proves a failed attempt is receipted even when the pass
    ends open — never swallowed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failed_first = False

    def click_gate(self, gate):
        self.gate_clicks.append(gate.kind)
        if not self._failed_first:
            self._failed_first = True
            return False
        if self._after_gate is not None:
            self._html = self._after_gate
        return True


class FakeLocator:
    def __init__(self, present: bool, page: "FakePage"):
        self._present = present
        self._page = page

    @property
    def first(self) -> "FakeLocator":
        # Playwright's Locator.first: the locator scoped to
        # the first matching element. The fake's count /
        # is_visible already encode presence, so the first
        # match is this locator itself.
        return self

    def count(self):
        return 1 if self._present else 0

    def is_visible(self):
        return self._present

    def click(self, timeout=3000):
        self._page.clicked.append("click")


class FakeFrame:
    def __init__(self, page: "FakePage", name: str):
        self._page = page
        self.name = name
        self.url = f"https://frame.example/{name}"

    def locator(self, selector: str):
        return self._page.locator(selector, frame=self)


class FakePage:
    """A Playwright-shaped page for the clicker: text/role
    lookups match on the RAW form or the folded form, the
    way a real page's text does — a folded-only lookup
    misses a curly-quote control."""

    def __init__(self, *, texts=(), selectors=(),
                 roles=(), frames=()):
        self._texts = set(texts)
        self._selectors = set(selectors)
        self._roles = set(roles)
        self._frames = [FakeFrame(self, name)
                        for name in frames]
        self.clicked: list[str] = []
        self.searched: list[str] = []
        self.main_frame = object()

    @property
    def frames(self):
        return [self.main_frame, *self._frames]

    def _text_matches(self, wanted: str) -> bool:
        return any(wanted == have or norm_text(wanted) ==
                   norm_text(have) for have in self._texts)

    def locator(self, selector: str, frame=None):
        self.searched.append(selector)
        if selector.startswith("text="):
            return FakeLocator(
                self._text_matches(selector[5:]), self)
        return FakeLocator(selector in self._selectors, self)

    def get_by_role(self, role: str, name: str = "",
                    exact: bool = False):
        self.searched.append(f"role:{role}:{name}")
        hit = any(r == role and self._text_matches(name)
                  for r, _n in self._roles)
        return FakeLocator(hit, self)


# ── F2: folding is shared ───────────────────────────────────


def test_fold_text_folds_curly_quotes():
    assert fold_text("I\u2019m 18 or older") == "I'm 18 or older"
    assert norm_text("I\u2019m 18 or Older") == "i'm 18 or older"


def test_age_gate_curly_apostrophe_detected_with_raw_label():
    gates = webnav.detect_gates(
        AGE_PAGE, "https://videos.example/")
    assert [g.kind for g in gates] == ["age"]
    assert gates[0].label == "i'm 18 or older"
    assert gates[0].raw_label == "I\u2019m 18 or older"


def test_clicker_tries_raw_then_folded_form():
    page = FakePage(texts={"I\u2019m 18 or older"},
                    roles={("button", "I\u2019m 18 or older")},
                    selectors={"#age-ok"})
    probe = webnav._PlaywrightProbe(page)
    gates = webnav.detect_gates(
        AGE_PAGE, "https://videos.example/")
    assert probe.click_gate(gates[0]) is True
    assert page.clicked == ["click"]
    assert probe.last_click is not None
    assert "I\u2019m 18 or older" in probe.last_click["summary"]


def test_clicker_selector_first_scoped_to_frame():
    page = FakePage(texts={"Accept all"},
                    selectors={"#cookie-accept"})
    probe = webnav._PlaywrightProbe(page)
    gates = webnav.detect_gates(
        COOKIE_PAGE, "https://x.example/")
    assert probe.click_gate(gates[0]) is True
    assert probe.last_click["strategy"] == \
        "selector #cookie-accept"


# ── F3: hidden dialogs are not standing gates ───────────────


def test_hidden_dialog_is_not_a_gate():
    page = (
        "<html><body><div class=\"modal\" style=\"display:none\">"
        "You must be 18 <button>I\u2019m 18 or older</button>"
        "</div></body></html>")
    assert webnav.detect_gates(page, "https://v.example/") == []


def test_hidden_attribute_and_visibility_hidden():
    for style in ("hidden", "style=\"visibility:hidden\"",
                  "style=\"visibility:collapse\""):
        page = (
            f"<html><body><div class=\"modal\" {style}>"
            "We use cookies <button>Accept all cookies</button>"
            "</div></body></html>")
        assert webnav.detect_gates(
            page, "https://x.example/") == []


def test_hidden_dialog_does_not_shadow_visible_banner():
    page = (
        "<html><body>"
        "<div class=\"modal\" style=\"display:none\">You must "
        "be 18 <button>I\u2019m 18 or older</button></div>"
        "<div class=\"cookie-banner\">We use cookies "
        "<button id=\"cookie-accept\">Accept all cookies"
        "</button></div></body></html>")
    gates = webnav.detect_gates(page, "https://x.example/")
    assert [g.kind for g in gates] == ["cookie"]


# ── F6: ARIA controls are collected ─────────────────────────


def test_aria_role_button_with_label_is_a_gate():
    page = (
        "<html><body><div>We value your privacy "
        "<div role=\"button\" aria-label=\"Accept all\">"
        "\u2713</div></div></body></html>")
    gates = webnav.detect_gates(page, "https://x.example/")
    assert [g.kind for g in gates] == ["consent"]
    assert gates[0].label == "accept all"


def test_icon_only_button_names_itself_by_aria_label():
    page = (
        "<html><body><div>We use cookies "
        "<button aria-label=\"Accept all cookies\">\u2713"
        "</button></div></body></html>")
    gates = webnav.detect_gates(page, "https://x.example/")
    assert [g.kind for g in gates] == ["cookie"]


def test_worded_button_keeps_visible_text_over_aria_label():
    page = (
        "<html><body><div>We use cookies "
        "<button aria-label=\"Ignore me\">Accept all</button>"
        "</div></body></html>")
    gates = webnav.detect_gates(page, "https://x.example/")
    assert gates[0].label == "accept all"


# ── F5: frame discovery ─────────────────────────────────────


def test_frame_only_gate_is_invisible_without_frames():
    assert webnav.detect_gates(
        CONTENT_PAGE, "https://x.example/") == []


def test_frame_only_gate_is_discovered_with_frames():
    gates = webnav.detect_gates(
        CONTENT_PAGE, "https://x.example/",
        frames=[("onetrust", FRAME_CONSENT)])
    assert [g.kind for g in gates] == ["consent"]
    assert gates[0].frame == "onetrust"
    assert gates[0].source == "frame:onetrust"


def test_discovery_scans_main_and_frame_gates():
    gates = discover_gates(
        COOKIE_PAGE, "https://x.example/",
        frames=[("onetrust", FRAME_CONSENT)])
    # Consent outranks cookie, even from a frame.
    assert [(g.kind, g.source) for g in gates] == [
        ("consent", "frame:onetrust"), ("cookie", "main")]


def test_pass_discovers_frame_gate_through_probe():
    probe = FakeProbe(CONTENT_PAGE, after_gate=CONTENT_PAGE,
                      frames=[("onetrust", FRAME_CONSENT)])
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["consent"]
    assert probe.gate_clicks == ["consent"]


# ── F4: settle loop, not a fixed wait ───────────────────────


def test_pass_waits_for_slow_clearing_gate():
    # The gate only clears after 3 waits.  A fixed 0.6 s
    # wait + one re-read would report "stayed up".
    probe = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE,
                      clear_after_waits=3)
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["cookie"]
    assert sum(probe.waits) > 0.6


def test_receipt_records_settle_checks():
    probe = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE,
                      clear_after_waits=2)
    projection = run_gate_pass(probe)
    assert projection.final_state == "open"
    receipt = projection.receipts[0]
    assert receipt.outcome == "cleared"
    assert receipt.verification == "rediscovered_absent"
    assert receipt.attempts >= 2


# ── F7: memory never skips a detected gate ──────────────────


def test_memory_never_skips_a_standing_gate():
    memory = GateMemory()
    first = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE)
    webnav.run_interstitial_pass(first, memory=memory)
    assert memory.knows("x.example", "cookie")
    # The gate stands again on a fresh page: the pass must
    # still click it and verify it gone.
    second = FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE)
    actions = webnav.run_interstitial_pass(second, memory=memory)
    assert [a.kind for a in actions] == ["cookie"]
    assert second.gate_clicks == ["cookie"]


# ── F8: failed attempts are receipted, never swallowed ──────


def test_failed_attempt_is_receipted_on_success():
    # A page stacking two gates: the consent click fails,
    # the cookie click clears and the page opens.  The pass
    # ends open — and the failed attempt is still receipted.
    stacked = (
        "<html><body><div class=\"onetrust\">We value your "
        "privacy <button>Accept all</button></div>"
        "<div class=\"cookie-banner\">We use cookies "
        "<button id=\"cookie-accept\">Accept all cookies"
        "</button></div></body></html>")
    probe = FailingClickProbe(stacked, after_gate=CONTENT_PAGE)
    projection = run_gate_pass(probe)
    assert projection.final_state == "open"
    outcomes = [r.outcome for r in projection.receipts]
    assert outcomes == ["would_not_click", "cleared"]
    assert "would not click" in projection.diagnostics()


def test_gate_that_stays_up_carries_its_receipts():
    probe = FakeProbe(COOKIE_PAGE, after_gate=COOKIE_PAGE)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "still standing" in str(err.value)
    assert err.value.projection is not None
    assert err.value.projection.final_state == "gate_standing"
    assert err.value.projection.receipts
    assert err.value.projection.receipts[0].outcome == "stayed"


def test_gate_that_will_not_click_blocks_honestly():
    probe = FakeProbe(COOKIE_PAGE, click_works=False)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "would not click" in str(err.value)


def test_dob_gate_is_named_and_never_invented():
    probe = FakeProbe(DOB_PAGE)
    with pytest.raises(GateBlocked) as err:
        webnav.run_interstitial_pass(probe)
    assert "date of birth" in str(err.value)
    assert probe.gate_clicks == []
    assert err.value.projection is not None
    assert err.value.projection.final_state == "refused"


# ── projection shape ────────────────────────────────────────


def test_projection_merge_keeps_every_receipt():
    first = run_gate_pass(
        FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE))
    second = run_gate_pass(
        FakeProbe(CONSENT_PAGE, after_gate=CONTENT_PAGE))
    merged = JobProjection.merge(first, second)
    assert [r.outcome for r in merged.receipts] == [
        "cleared", "cleared"]
    assert merged.final_state == "open"
    assert merged.cleared == ["cookie", "consent"]


def test_projection_to_dict_round_trip():
    projection = run_gate_pass(
        FakeProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE))
    data = projection.to_dict()
    assert data["final_state"] == "open"
    assert data["receipts"][0]["outcome"] == "cleared"
    assert data["receipts"][0]["observation"]["kind"] == "cookie"


def test_no_gates_is_an_open_projection():
    projection = run_gate_pass(FakeProbe(CONTENT_PAGE))
    assert projection.final_state == "open"
    assert projection.receipts == []
    assert projection.diagnostics() == "no gates standing"


def test_old_style_probe_without_frames_seam_still_works():
    probe = OldProbe(COOKIE_PAGE, after_gate=CONTENT_PAGE)
    actions = webnav.run_interstitial_pass(probe)
    assert [a.kind for a in actions] == ["cookie"]


def test_discovery_is_pure():
    once = discover_gates(COOKIE_PAGE, "https://x.example/")
    twice = discover_gates(COOKIE_PAGE, "https://x.example/")
    assert once == twice
    assert GateDiscovery.discover(
        COOKIE_PAGE, "https://x.example/") == once
