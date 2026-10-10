"""Tests for the rendered Chromium engine on the spine browser tool."""

import pytest

from nomorals.tools.browser import (
    RenderedBrowserSession,
    _playwright_available,
    _RENDERED_ACTIONS,
    get_rendered_session,
)


def test_playwright_check_returns_tuple():
    ok, reason = _playwright_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    if not ok:
        assert "playwright" in reason.lower()


def test_rendered_actions_cover_agent_vocabulary():
    for act in ("open", "text", "observe", "screenshot", "click", "fill",
                "submit", "wait", "scroll", "hover", "press", "extract",
                "task", "state", "close"):
        assert act in _RENDERED_ACTIONS, f"missing rendered action: {act}"


def test_rendered_session_do_unknown_action():
    sess = RenderedBrowserSession(name="test-nope")
    with pytest.raises(Exception, match="unknown browser action"):
        sess.do("frobnicate")


def test_rendered_session_requires_playwright():
    ok, _ = _playwright_available()
    if ok:
        pytest.skip("playwright installed — honest-error path not testable")
    sess = RenderedBrowserSession(name="test-missing")
    with pytest.raises(Exception, match="[Pp]laywright"):
        sess.open("https://example.com")


def test_rendered_session_state_without_tab():
    sess = RenderedBrowserSession(name="test-state")
    st = sess.state()
    assert st["engine"] == "rendered"
    assert st["session"] == "test-state"
    assert "playwright" in st


def test_get_rendered_session_named():
    s1 = get_rendered_session("unit-a")
    s2 = get_rendered_session("unit-a")
    assert s1 is s2
    s1.close()


def test_osint_browser_specs_wired():
    from nomorals.search.osint_browser import BROWSER_OSINT_SPECS
    from nomorals.search.sources import SOURCE_SPECS
    names = [s[0] for s in SOURCE_SPECS]
    for name, _rtype, _desc, _cls in BROWSER_OSINT_SPECS:
        assert name in names, f"{name} not in federated sources"


def test_research_fallback_helper_exists():
    from nomorals.research.pipeline import _browser_fetch_fallback
    assert callable(_browser_fetch_fallback)
    # never raises, returns "" when unwired
    assert _browser_fetch_fallback(None, "https://example.com", 100) == ""
