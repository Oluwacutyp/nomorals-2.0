"""Sweep tests: browser module upgrades (mined-then-built).

Covers the new behavior in errors (expanded WAF detection, retry
guidance, styled summaries), pacing (per-action cadence, reading pauses,
stats), forms (shadow-piercing resolver, candidates, form groups),
liveview (caption styles, icons, progress), service (RenderedTab snapshot,
telemetry, dialogs, new interactions, pdf), daemon (new ops), and styles.
Rendered-tab tests use fake page drivers — no real browser is launched.
"""

import socket
import time

import pytest

from nomorals.browser import daemon as daemon_mod
from nomorals.browser import errors, forms, liveview, pacing, styles
from nomorals.browser.daemon import (
    DaemonError,
    _handle_op,
    _handle_pacing_op,
    _recv_frame,
    _send_frame,
)
from nomorals.browser.service import (
    BrowserError,
    BrowserService,
    RenderedTab,
)


# ── errors: expanded WAF detection ───────────────────────────────────────────

@pytest.mark.parametrize("kwargs,expected", [
    ({"set_cookies": ["_abck=abc123; Path=/; Domain=.x.com"]}, "akamai"),
    ({"set_cookies": ["bm_sz=xyz; Path=/"]}, "akamai"),
    ({"headers": {"x-iinfo": "1-2-3"}}, "imperva"),
    ({"headers": {"x-kpsdk-ct": "abc"}}, "kasada"),
    ({"headers": {"x-datadome": "protected"}}, "datadome"),
    ({"headers": {"x-amzn-waf-action": "challenge"}}, "aws-waf"),
    ({"set_cookies": ["aws-waf-token=abc; Path=/"]}, "aws-waf"),
    ({"html": '<div class="cf-turnstile" data-sitekey="x"></div>'}, "turnstile"),
    ({"html": "<script src='https://arkoselabs.com/api.js'>"}, "arkose"),
    ({"html": "initGeetest4({captchaId: 'x'})"}, "geetest"),
    ({"html": "var istlWasHere = true;"}, "f5-shape"),
    ({"html": "___utmvc = 'abc';"}, "imperva"),
    ({"set_cookies": ["incap_ses_123=abc; Path=/"]}, "imperva"),
    ({"set_cookies": ["_px3=abc; Path=/"]}, "perimeterx"),
    ({"title": "Just a moment..."}, "cloudflare-challenge"),
    # no false positives on ordinary pages
    ({"html": "<p>please complete the captcha below</p>"}, ""),
    ({"html": "<html><body>hello</body></html>"}, ""),
])
def test_detect_challenge_new_wafs(kwargs, expected):
    assert errors.detect_challenge(**kwargs) == expected


def test_429_honors_retry_after():
    err = errors.classify_http_status(
        429, url="https://x.test",
        headers={"Retry-After": "90"})
    assert isinstance(err, errors.BrowserBotDetectedError)
    assert err.detection == "rate-limit"
    assert err.retryable is True
    assert err.retry_after_seconds == 90


def test_retryable_flags():
    assert errors.classify_exception(TimeoutError("t")).retryable is True
    assert errors.BrowserSiteError("x", status=503).retryable is True
    assert errors.BrowserSiteError("x", status=404).retryable is False
    assert errors.BrowserBotDetectedError("x").retryable is False
    assert errors.BrowserBotDetectedError(
        "x", detection="rate-limit").retryable is True


def test_with_message_preserves_retry_guidance():
    err = errors.classify_http_status(
        429, headers={"Retry-After": "30"})
    copy = err.with_message("new message")
    assert copy.retryable is True
    assert copy.retry_after_seconds == 30
    assert str(copy) == "new message"


def test_summarize_card():
    err = errors.classify_http_status(
        429, url="https://x.test", headers={"Retry-After": "60"})
    rich = errors.summarize(err, style="rich")
    assert "rate-limit" in rich
    assert "next steps:" in rich
    assert "retryable: yes" in rich
    plain = errors.summarize(err, style="plain")
    assert "rate-limit" in plain


def test_describe_carries_icon_and_retry():
    err = errors.BrowserNetworkError("dns blew up", reason="dns")
    payload = errors.describe(err)
    assert payload["icon"]
    assert payload["retryable"] is True
    assert payload["detail"]["reason"] == "dns"


# ── pacing ───────────────────────────────────────────────────────────────────

def test_careful_preset_per_action_cadence():
    p = pacing.Pacing.careful()
    assert p.enabled
    assert p._cadence_for("submit") == (1400, 800)
    assert p._cadence_for("hover") == (700, 500)


def test_per_action_pause_overrides_global(monkeypatch):
    slept = []
    monkeypatch.setattr(pacing.time, "sleep", lambda s: slept.append(s))
    p = pacing.Pacing(enabled=True, delay_ms=10, jitter_ms=0,
                      per_action={"submit": (50, 0)})
    p.pause("submit")
    p.pause("hover")
    assert slept == pytest.approx([0.05, 0.01])
    stats = p.stats()
    assert stats["pauses"] == 2
    assert stats["slept_s"] == pytest.approx(0.06)


def test_read_pause_proportional(monkeypatch):
    slept = []
    monkeypatch.setattr(pacing.time, "sleep", lambda s: slept.append(s))
    p = pacing.Pacing(enabled=True, delay_ms=10, jitter_ms=0)
    p.read(100)  # 20 words at 220wpm ≈ 5.45s
    assert slept and slept[0] == pytest.approx(5.45, rel=0.05)


def test_pacing_fail_loud_and_noop():
    with pytest.raises(ValueError):
        pacing.Pacing(enabled=False, delay_ms=100)
    with pytest.raises(ValueError):
        pacing.Pacing(enabled=False, per_action={"submit": (50, 0)})
    assert pacing.Pacing.disabled().pause("click") == 0.0


def test_pacing_from_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("NOMORALS_BROWSER_PACING", "fast,slow")
    with pytest.raises(ValueError):
        pacing.pacing_from_env()
    monkeypatch.setenv("NOMORALS_BROWSER_PACING", "400,300")
    p = pacing.pacing_from_env()
    assert (p.delay_ms, p.jitter_ms) == (400, 300)


# ── forms ────────────────────────────────────────────────────────────────────

class _FakePage:
    """Duck-typed page: canned evaluate payloads per call."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def evaluate(self, js, *args):
        self.calls.append(args)
        if not self.payloads:
            return []
        return self.payloads.pop(0)


def _candidate(**kw):
    base = {"tag": "input", "type": "text", "by": "label", "score": 90,
            "name": "email", "id": "email", "label": "Email",
            "placeholder": "", "aria": "", "shadow": False,
            "in_frame": False}
    base.update(kw)
    return base


def test_resolve_pins_winner():
    page = _FakePage([[_candidate(score=90, shadow=True, ord=3)], "ok"])
    selector, info = forms.resolve(page, "email")
    assert selector == forms.MARKER_SELECTOR
    assert info["by"] == "label"
    assert info["shadow"] is True
    # candidates script first (bare field name), then the ordinal pin
    assert page.calls[0] == ("email",)
    assert page.calls[1] == (3,)


def test_resolve_candidates_does_not_pin():
    page = _FakePage([[_candidate(), _candidate(score=30, by="name~")]])
    cands = forms.resolve_candidates(page, "email")
    assert len(cands) == 2
    assert cands[0]["score"] == 90
    assert len(page.calls) == 1  # no pin call
    assert page.calls[0] == ("email",)


def test_field_not_found_carries_candidates_and_hint():
    page = _FakePage([[], [_candidate(label="Email Address", score=45,
                                      by="label~", ord=0)]])
    with pytest.raises(forms.FieldNotFound) as excinfo:
        forms.resolve(page, "emial")
    assert "did you mean 'Email Address'?" in str(excinfo.value)
    assert len(excinfo.value.candidates) == 1


def test_resolve_legacy_single_winner_protocol():
    """Duck-typed drivers speaking the old single-dict protocol still
    resolve (ord=-1: no pin call needed, they pinned inline)."""
    page = _FakePage([{"tag": "input", "type": "email", "by": "name",
                       "score": 70, "name": "email", "id": ""}])
    selector, info = forms.resolve(page, "email")
    assert selector == forms.MARKER_SELECTOR
    assert info["by"] == "name"
    assert len(page.calls) == 1  # no pin script for legacy infos


def test_resolve_legacy_fallback_without_js():
    selector, info = forms.resolve(object(), "email")
    assert info is None
    assert 'input[name="email"]' in selector


def test_describe_forms_groups():
    page = _FakePage([[{"index": 0, "id": "login", "name": "",
                        "action": "/go", "method": "post",
                        "fields": [{"name": "u"}, {"name": "p"}],
                        "submit": {"tag": "button", "type": "submit"}}]])
    groups = forms.describe_forms(page)
    assert groups[0]["id"] == "login"
    assert len(groups[0]["fields"]) == 2
    assert groups[0]["submit"]["tag"] == "button"


def test_forms_js_blocks_balanced():
    for name in ("_CANDIDATES_JS", "_CLEAR_JS", "_DESCRIBE_JS",
                 "_SET_VALUE_JS", "_READ_VALUE_JS", "_DESCRIBE_FORMS_JS",
                 "_DEEP_WALK_JS"):
        js = getattr(forms, name)
        assert js.count("{") == js.count("}"), name
        assert js.count("(") == js.count(")"), name
        # the deep walker is actually referenced where it matters
        if name not in {"_DEEP_WALK_JS"}:
            assert "__nmDeepEach" in js or name == "_DEEP_WALK_JS", name


def test_normalize_date_still_works():
    assert forms.normalize_date("4 Oct 2026") == "2026-10-04"
    with pytest.raises(ValueError):
        forms.normalize_date("not a date")


# ── liveview ─────────────────────────────────────────────────────────────────

class _FakeChat:
    pass


class _FakeAdapter:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("adapter down")

    def send(self, chat, text):
        self._maybe_fail()
        self.sent.append(("send", text))
        return _FakeResult(True, "m1")

    def send_media(self, chat, media, caption=""):
        self._maybe_fail()
        self.sent.append(("send_media", caption))
        return _FakeResult(True, "m2")

    def edit_media(self, chat, message_id, media, caption=""):
        self._maybe_fail()
        self.sent.append(("edit_media", caption))
        return _FakeResult(True, message_id)


class _FakeResult:
    def __init__(self, ok, message_id):
        self.ok = ok
        self.message_id = message_id


def _make_view(adapter, clock_time=None, style="rich"):
    import tempfile
    shot_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    shot_file.close()
    now = [clock_time if clock_time is not None else 1000.0]

    def clock():
        return now[0]

    view = liveview.LiveView(lambda: shot_file.name, profile="workstation",
                             min_interval_s=8.0, clock=clock,
                             caption_style=style)
    view.start(_FakeChat(), adapter, task_label="test task")
    return view, now


def test_liveview_rich_captions():
    adapter = _FakeAdapter()
    view, now = _make_view(adapter)
    opening = adapter.sent[0][1]
    assert "test task" in opening
    now[0] += 9.0  # pass the throttle window
    view.update("clicked the submit button")
    caption = adapter.sent[-1][1]
    assert "\U0001F446" in caption  # click icon
    assert "clicked the submit button" in caption
    assert "1/25" in caption  # progress
    assert "9s" in caption  # elapsed
    now[0] += 9.0
    view.update("submitted the form")
    assert "\U0001F4E8" in adapter.sent[-1][1]  # submit icon


def test_liveview_throttles_and_coalesces():
    adapter = _FakeAdapter()
    view, now = _make_view(adapter)
    view.update("first action")  # throttled: no new message
    assert len(adapter.sent) == 1
    now[0] += 9.0
    view.update("second action")
    assert len(adapter.sent) == 2
    assert "second action" in adapter.sent[-1][1]


def test_liveview_finish_and_fail_frames():
    adapter = _FakeAdapter()
    view, now = _make_view(adapter)
    now[0] += 65.0
    view.finish("all done")
    caption = adapter.sent[-1][1]
    assert caption.startswith("\u2705")
    assert "1m05s" in caption
    assert "all done" in caption

    adapter2 = _FakeAdapter()
    view2, _ = _make_view(adapter2)
    view2.fail("boom")
    assert adapter2.sent[-1][1].startswith("\u274C")
    assert "boom" in adapter2.sent[-1][1]


def test_liveview_compact_style():
    adapter = _FakeAdapter()
    view, now = _make_view(adapter, style="compact")
    now[0] += 9.0
    view.update("hovered the menu")
    caption = adapter.sent[-1][1]
    assert "test task" in caption
    assert "/" not in caption  # no progress counter in compact mode


def test_liveview_never_raises():
    adapter = _FakeAdapter(fail=True)
    view, _ = _make_view(adapter)
    view.update("whatever")
    view.finish("done")
    view.fail("x")
    view.detach()  # no exceptions anywhere


def test_action_icon_and_elapsed():
    assert liveview.action_icon("navigated to https://x") == "\U0001F310"
    assert liveview.action_icon("typed into search") == "\u2328\uFE0F"
    assert liveview.action_icon("something weird") == "\U0001F50D"
    assert liveview.format_elapsed(5) == "5s"
    assert liveview.format_elapsed(125) == "2m05s"


# ── service: RenderedTab with fake drivers ───────────────────────────────────

def _make_tab(tmp_path):
    return RenderedTab(
        tab_id="t1",
        session_name="s1",
        storage_state_path=str(tmp_path / "state.json"),
        playwright=object(),
    )


class _FakeDialog:
    def __init__(self, kind="alert", message="hi"):
        self.type = kind
        self.message = message
        self.accepted = False
        self.dismissed = False

    def accept(self):
        self.accepted = True

    def dismiss(self):
        self.dismissed = True


def test_dialog_policy_default_accepts_and_records(tmp_path):
    tab = _make_tab(tmp_path)
    dlg = _FakeDialog()
    tab._on_dialog(dlg)
    assert dlg.accepted is True
    assert tab.dialogs()["count"] == 1
    assert tab.dialogs()["dialogs"][0]["message"] == "hi"


def test_dialog_policy_record_leaves_open(tmp_path):
    tab = _make_tab(tmp_path)
    assert tab.set_dialog_policy("record")["policy"] == "record"
    dlg = _FakeDialog()
    tab._on_dialog(dlg)
    assert dlg.accepted is False  # left for manual handling
    assert tab.dialogs(clear=True)["dialogs"][0]["policy"] == "record"
    assert tab.dialogs()["count"] == 0
    with pytest.raises(BrowserError):
        tab.set_dialog_policy("explode")


def test_telemetry_buffers_start_empty(tmp_path):
    tab = _make_tab(tmp_path)
    assert tab.console_messages()["messages"] == []
    assert tab.network_requests()["requests"] == []
    assert tab.dialogs()["dialogs"] == []


def test_console_and_network_filters(tmp_path):
    tab = _make_tab(tmp_path)

    class Msg:
        type = "error"
        location = {"url": "https://x", "lineNumber": 3,
                    "columnNumber": 1}

        def text(self):
            return "boom"

    class Req:
        url = "https://x/api"
        method = "GET"

    class Resp:
        url = "https://x/api"
        status = 500
        request = Req()

    tab._on_console_msg(Msg())
    tab._on_request(Req())
    tab._on_response(Resp())
    assert tab.console_messages(level="error")["count"] == 1
    assert tab.console_messages(level="warning")["count"] == 0
    assert tab.network_requests()["count"] == 2
    failed = tab.network_requests(failed_only=True)
    assert failed["count"] == 1
    assert failed["requests"][0]["status"] == 500


def test_resolve_ref_and_target_selector(tmp_path):
    tab = _make_tab(tmp_path)
    tab._snap_refs = {"e4": "aria-ref=e4"}
    assert tab.resolve_ref("ref:e4") == "aria-ref=e4"
    assert tab.resolve_ref("e4") == "aria-ref=e4"
    with pytest.raises(BrowserError, match="take snapshot"):
        RenderedTab("t9", "s", str(tmp_path / "s.json"),
                    object()).resolve_ref("e1")
    assert tab._target_selector("ref:e4", "click") == "aria-ref=e4"
    assert tab._target_selector("#main", "click") == "#main"
    assert tab._target_selector("Sign in", "click") == "text=Sign in"
    with pytest.raises(BrowserError):
        tab._target_selector("", "click")


class _AriaPage:
    """Fake page exposing locator().aria_snapshot()."""

    def __init__(self, tree):
        self._tree = tree

    def locator(self, sel):
        tree = self._tree

        class _Loc:
            def aria_snapshot(self):
                return tree

        return _Loc()


class _JsPage:
    """Fake page with evaluate() but no aria_snapshot."""

    def __init__(self, payload):
        self._payload = payload

    def evaluate(self, js, *args):
        return self._payload


def test_snapshot_aria_source(tmp_path):
    tab = _make_tab(tmp_path)
    tab.url = "https://example.com"
    tab._page = _AriaPage('- button "Go" [ref=e4]\n- textbox "q" [ref=e7]')
    result = tab.snapshot()
    assert result["source"] == "aria"
    assert result["refs"] == {"e4": "aria-ref=e4", "e7": "aria-ref=e7"}
    assert tab.resolve_ref("ref:e4") == "aria-ref=e4"


def test_snapshot_fallback_source(tmp_path):
    tab = _make_tab(tmp_path)
    tab.url = "https://example.com"
    tab._page = _JsPage(
        '- textbox "q" [ref=f1]\n__REF__f1__SEL__input#q\n- button [ref=f2]')
    result = tab.snapshot()
    assert result["source"] == "fallback"
    assert result["refs"] == {"f1": "input#q"}
    assert "[ref=f1]" in result["snapshot"]
    assert "__REF__" not in result["snapshot"]


def test_press_key_validation(tmp_path):
    tab = _make_tab(tmp_path)
    tab.url = "https://example.com"
    tab._page = object()
    with pytest.raises(BrowserError, match="needs a key"):
        tab.press("")


def test_scroll_bad_direction(tmp_path):
    tab = _make_tab(tmp_path)
    tab.url = "https://example.com"
    tab._page = object()
    with pytest.raises(BrowserError, match="unknown scroll direction"):
        tab.scroll("sideways")


def test_type_text_needs_text(tmp_path):
    tab = _make_tab(tmp_path)
    tab.url = "https://example.com"
    tab._page = object()
    with pytest.raises(BrowserError, match="needs text"):
        tab.type_text("#q", "")


def test_pdf_without_loaded_page_fails_fast(tmp_path):
    tab = _make_tab(tmp_path)
    with pytest.raises(BrowserError, match="no loaded page"):
        tab.pdf()


def test_service_stats(tmp_path):
    svc = BrowserService(data_dir=str(tmp_path / "bsvc"))
    stats = svc.stats()
    assert stats["sessions"] == []
    assert stats["plain_tabs"] == 0
    assert stats["rendered_tabs"] == 0
    assert stats["downloads"] == 0
    assert stats["pacing"]["enabled"] is False
    assert stats["proxy_pool_attached"] is False


def test_service_set_pacing_per_action(tmp_path):
    svc = BrowserService(data_dir=str(tmp_path / "bsvc2"))
    desc = svc.set_pacing(enabled=True, delay_ms=100, jitter_ms=50,
                           per_action={"submit": [500, 100]})
    assert desc["per_action"] == {"submit": [500, 100]}
    assert svc.pacing._cadence_for("submit") == (500, 100)
    with pytest.raises(BrowserError):
        svc.set_pacing(per_action={"submit": "fast"})


# ── daemon: framing + new ops ────────────────────────────────────────────────

def test_daemon_frame_roundtrip():
    a, b = socket.socketpair()
    try:
        _send_frame(a, {"op": "ping", "params": {"x": [1, 2]}})
        assert _recv_frame(b, 5.0) == {"op": "ping", "params": {"x": [1, 2]}}
    finally:
        a.close()
        b.close()


def test_daemon_unknown_op(tmp_path):
    svc = BrowserService(data_dir=str(tmp_path / "dsvc"))
    with pytest.raises(DaemonError, match="unknown daemon op"):
        _handle_op(svc, "frobnicate", {})


def test_daemon_stats_op(tmp_path):
    svc = BrowserService(data_dir=str(tmp_path / "dsvc2"))
    result = _handle_op(svc, "stats", {})["result"]
    assert result["pacing"]["enabled"] is False
    assert result["rendered_tabs"] == 0


def test_daemon_pacing_careful_preset(tmp_path):
    svc = BrowserService(data_dir=str(tmp_path / "dsvc3"))
    result = _handle_pacing_op(svc, {"action": "set", "preset": "careful"})
    assert result["enabled"] is True
    assert result["per_action"]["submit"] == [1400, 800]
    with pytest.raises(DaemonError):
        _handle_pacing_op(svc, {"action": "set", "preset": "reckless"})


def test_daemon_mutating_ops_cover_new_actions():
    for op in ("r_type", "r_dblclick", "r_press", "r_drag",
               "r_reload", "r_forward", "r_back"):
        assert op in daemon_mod._MUTATING_OPS


# ── styles ───────────────────────────────────────────────────────────────────

def test_styles_themes():
    err = errors.BrowserNetworkError("dns blew up", reason="dns")
    rich = styles.error_card(err, style="rich")
    assert "\U0001F50C" in rich
    plain = styles.error_card(err, style="plain")
    assert "\U0001F50C" not in plain
    assert "dns blew up" in plain

    snap = styles.snapshot_block(
        {"snapshot": '- button "Go" [ref=e4]', "refs": {"e4": "x"},
         "source": "aria", "truncated": False, "url": "https://x"},
        style="rich")
    assert "ref:<id>" in snap
    assert "[ref=e4]" in snap

    block = styles.stats_block(
        {"sessions": ["cli"], "plain_tabs": 1, "rendered_tabs": 2,
         "downloads": 3, "pacing": {"enabled": False},
         "proxy_pool_attached": True})
    assert "2 rendered" in block
    assert "off" in styles.pacing_line({"enabled": False})
