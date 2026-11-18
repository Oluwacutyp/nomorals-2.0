"""Failing-first tests for the chronic browser ToolError (lane 24,
defect 5).

Root cause (proven by a hasattr probe on the real classes):
``RenderedBrowserSession._tab_obj`` calls ``svc.open_tab()`` — but
``open_tab`` exists only on ``SessionHandle``, NOT on
``BrowserService``.  Every rendered browser-tool call therefore
raises AttributeError, ``_wrap`` converts it to a ToolError, and
selfheal ESCALATE logs it and moves on — hundreds of times, with
no recovery and no fallback.

The fixes under test:

* the tab opens through ``open_rendered_tab`` (the API that exists
  on BrowserService), falling back to the session handle's
  ``open_tab`` only when the service has no rendered tab;
* when the rendered engine is DEAD (Chromium cannot launch — the
  defect-1 launch failure), an ``engine="auto"`` call falls back to
  the http engine for READ-ONLY actions, marks the rendered session
  dead so the next call goes straight to http, and names the
  fallback in the result;
* an explicitly requested rendered engine NEVER falls back;
* a rendered-only action (screenshot) never falls back;
* an action failure on a LIVE page ("no such element") is not an
  engine failure and never falls back.

``browser()`` is a closure built inside ``register(registry)``, so
the tests build it through a fake registry and pull the tool fn
out of the recorded spec.  Playwright is not installed on this
box, so the auto-pick is exercised by patching the module's
``_playwright_available`` to report it present.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nomorals.tools.browser as browser_mod  # noqa: E402
from nomorals.core.errors import ToolError  # noqa: E402
from nomorals.tools.browser import RenderedBrowserSession  # noqa: E402

LAUNCH_TIMEOUT_MSG = (
    "rendered tab could not launch chromium: BrowserType.launch: "
    "Timeout 180000ms exceeded. rendered tabs require the "
    "'playwright' package and a chromium build: pip install "
    "playwright && playwright install chromium")


class _FakeRegistry:
    """Records the tool specs ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


def _browser_fn():
    registry = _FakeRegistry()
    browser_mod.register(registry)
    return registry.tools["browser"]


class _DeadRendered:
    """A rendered session whose engine is dead: every call raises
    the launch-timeout error the real service raises."""

    def __init__(self):
        self.do_calls = 0

    def do(self, action, **kwargs):
        self.do_calls += 1
        raise ToolError(f"browser {action} failed: {LAUNCH_TIMEOUT_MSG}")


class _HttpSession:
    """A plain http session: read actions answer."""

    def __init__(self):
        self.do_calls = []

    def do(self, action, **kwargs):
        self.do_calls.append(action)
        return {"engine": "http", "action": action,
                "url": kwargs.get("url", ""), "ok": True}


class _FakeService:
    """A BrowserService-shaped object: ``open_rendered_tab`` exists,
    ``open_tab`` does not — exactly like the real class."""

    def __init__(self):
        self.opened = []

    def open_rendered_tab(self, session_name, url=None):
        self.opened.append((session_name, url))
        return {"ok": True, "tab": "rendered"}


class _LegacyService:
    """A service with no rendered tab at all: the old session-handle
    route must still work."""

    def open_session(self, session_name):
        return _FakeHandle(session_name)


class _FakeHandle:
    def __init__(self, name):
        self.name = name
        self.opened = []

    def open_tab(self, url=None):
        self.opened.append(url)
        return {"ok": True, "tab": "handle"}


class TabOpenTests(unittest.TestCase):
    def test_tab_opens_via_open_rendered_tab(self):
        svc = _FakeService()
        sess = RenderedBrowserSession(name="t")
        sess._svc = svc  # noqa: SLF001 - the injection seam
        self.assertEqual(sess._tab_obj(), {"ok": True, "tab": "rendered"})
        self.assertEqual(svc.opened, [("t", None)])

    def test_legacy_service_falls_back_to_session_handle(self):
        svc = _LegacyService()
        sess = RenderedBrowserSession(name="t")
        sess._svc = svc  # noqa: SLF001 - the injection seam
        self.assertEqual(sess._tab_obj(), {"ok": True, "tab": "handle"})


class EngineFallbackTests(unittest.TestCase):
    def setUp(self):
        self.browser = _browser_fn()

    def _run(self, action, rendered, http, engine="auto"):
        with patch.object(browser_mod, "_playwright_available",
                         return_value=(True, "")), \
                patch.object(browser_mod, "get_rendered_session",
                             return_value=rendered), \
                patch.object(browser_mod, "get_session",
                             return_value=http):
            return self.browser(action=action, url="https://example.com",
                                engine=engine)

    def test_auto_falls_back_to_http_for_read_actions(self):
        rendered, http = _DeadRendered(), _HttpSession()
        result = self._run("open", rendered, http)
        self.assertEqual(result.get("engine"), "http")
        self.assertIn("fallback", result)
        self.assertEqual(rendered.do_calls, 1)
        self.assertEqual(http.do_calls, ["open"])

    def test_dead_engine_is_remembered_within_the_process(self):
        rendered, http = _DeadRendered(), _HttpSession()
        self._run("open", rendered, http)
        self._run("text", rendered, http)
        self.assertEqual(rendered.do_calls, 1)
        self.assertEqual(http.do_calls, ["open", "text"])

    def test_explicit_rendered_never_falls_back(self):
        rendered, http = _DeadRendered(), _HttpSession()
        with self.assertRaises(ToolError):
            self._run("open", rendered, http, engine="rendered")
        self.assertEqual(http.do_calls, [])

    def test_rendered_only_action_never_falls_back(self):
        rendered, http = _DeadRendered(), _HttpSession()
        with self.assertRaises(ToolError):
            self._run("screenshot", rendered, http)
        self.assertEqual(http.do_calls, [])

    def test_action_failure_on_a_live_page_never_falls_back(self):
        class _LiveRendered:
            def do(self, action, **kwargs):
                raise ToolError("browser click failed: no such element: "
                                "#submit")
        rendered, http = _LiveRendered(), _HttpSession()
        with self.assertRaises(ToolError):
            self._run("click", rendered, http)
        self.assertEqual(http.do_calls, [])


if __name__ == "__main__":
    unittest.main()
