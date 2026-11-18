"""Failing-first tests for the rendered-tab launch path (lane 24).

Defect 1, traced from the repo + the owner's EC2 log
(``rendered tab could not launch chromium: BrowserType.launch:
Timeout 180000ms exceeded``):

* ``RenderedTab._ensure_page`` called ``new_page()`` TWICE — the
  first page (with the console/network/dialog telemetry listeners
  attached) was orphaned by the second call, so the returned page
  had NO listeners and the rendered capture stage's network log
  stayed empty forever.  The second call was the original line
  (d84a7f20); the first was added by the a230642f browser sweep
  without removing it — a regression, not a design.
* the launch carried no timeout of its own (Playwright's 180s
  default burned the chain's whole budget) and no constrained-box
  launch args for a 2 vCPU / 2 GB box.

These tests drive ``_ensure_page`` with a fake Playwright stack
(Playwright is not installed on this box) and pin: one page
created, listeners on THAT page, an explicit launch timeout, and
low-resource launch args.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.browser.service import RenderedTab  # noqa: E402


class FakePage:
    def __init__(self):
        self.listeners = {}

    def on(self, event, handler):
        self.listeners.setdefault(event, []).append(handler)


class FakeContext:
    def __init__(self):
        self.pages = []
        self.listeners = {}
        self.new_page_calls = 0

    def new_page(self):
        self.new_page_calls += 1
        page = FakePage()
        self.pages.append(page)
        return page

    def on(self, event, handler):
        self.listeners.setdefault(event, []).append(handler)


class FakeBrowser:
    def __init__(self):
        self.context = FakeContext()
        self.context_kwargs = None

    def new_context(self, **kwargs):
        self.context_kwargs = dict(kwargs)
        return self.context


class FakeChromium:
    def __init__(self):
        self.browser = FakeBrowser()
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = dict(kwargs)
        return self.browser


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()


def _make_tab(tmpdir):
    state_path = Path(tmpdir) / "playwright-storage.json"
    tab = RenderedTab(
        tab_id="t1",
        session_name="downloader",
        storage_state_path=state_path,
        playwright=FakePlaywright(),
        stealth={"enabled": False},
    )
    # The estate is an enhancement, not the launch: keep the fake
    # stack hermetic — no real cookie estate is consulted.
    tab._estate = lambda: None  # noqa: SLF001 - test seam
    return tab


class EnsurePageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_one_page_created_with_telemetry(self):
        """The listeners land on the page that is returned."""
        tab = _make_tab(self.tmp)
        page = tab._ensure_page()
        ctx = tab._playwright.chromium.browser.context
        self.assertEqual(ctx.new_page_calls, 1,
                         "a second new_page() orphans the listened page")
        self.assertIs(page, ctx.pages[0])
        for event in ("console", "request", "response", "dialog"):
            self.assertIn(event, page.listeners,
                          f"telemetry listener {event!r} missing from "
                          "the returned page")

    def test_launch_has_explicit_timeout_and_low_resource_args(self):
        """Fail fast on a stuck launch; keep Chromium small on 2 GB."""
        tab = _make_tab(self.tmp)
        tab._ensure_page()
        kwargs = tab._playwright.chromium.launch_kwargs
        self.assertIn("timeout", kwargs,
                      "launch must bound itself instead of taking "
                      "Playwright's 180s default")
        self.assertLessEqual(int(kwargs["timeout"]), 60_000,
                             "the launch never completed in 180s — "
                             "waiting longer is not the fix")
        args = list(kwargs.get("args") or [])
        for flag in ("--no-sandbox", "--disable-dev-shm-usage",
                     "--disable-gpu"):
            self.assertIn(flag, args, f"missing low-resource arg {flag}")

    def test_second_ensure_page_call_reuses_the_page(self):
        tab = _make_tab(self.tmp)
        first = tab._ensure_page()
        second = tab._ensure_page()
        self.assertIs(first, second)
        ctx = tab._playwright.chromium.browser.context
        self.assertEqual(ctx.new_page_calls, 1)


if __name__ == "__main__":
    unittest.main()
