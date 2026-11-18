"""Failing-first tests for cause naming (lane 24, defect 6).

The same window's analysis said "[timeout] Media Error: ...
The service is slow or overloaded. Try again later or increase
the timeout value." — for a Chromium launch that timed out ON THE
BOX and a page that exposed no link.  The evidence pointed at the
box and the player, never the service.  Two naming failures:

* the KB's generic "timeout" pattern matches ANY message with
  "timeout" in it, so a local browser-launch timeout reads as a
  slow remote service;
* ``_generate_explanation_for_unknown`` uses only the exception
  TYPE and discards the message, so the ToolError's real cause
  (``'BrowserService' object has no attribute 'open_tab'``) never
  surfaces — the log shows only "An unexpected error occurred
  during browser: ToolError."

These tests pin: a launch timeout names the local browser, a
wiring AttributeError names the code bug (and is NOT retryable),
and an unknown error's explanation carries its message.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.error_intelligence import ErrorIntelligence  # noqa: E402
from nomorals.core.errors import ToolError  # noqa: E402


def _analyze(message, operation="browser"):
    intel = ErrorIntelligence()
    return intel.analyze(ToolError(message), context={"operation": operation})


class LaunchTimeoutTests(unittest.TestCase):
    def test_launch_timeout_names_the_local_browser(self):
        analysis = _analyze(
            "browser open failed: rendered tab could not launch "
            "chromium: BrowserType.launch: Timeout 180000ms exceeded. "
            "rendered tabs require the 'playwright' package and a "
            "chromium build: pip install playwright && playwright "
            "install chromium")
        self.assertEqual(analysis.error_code, "browser_launch_timeout")
        self.assertIn("chromium", analysis.root_cause.lower())
        self.assertNotIn("slow or overloaded",
                         analysis.suggested_fix.lower())
        self.assertFalse(analysis.retryable)

    def test_remote_timeout_still_reads_as_remote(self):
        """The generic timeout pattern survives for a genuine
        remote timeout — only the local-launch shape is carved
        out."""
        analysis = _analyze("request to https://api.example.com timed "
                            "out after 30s")
        self.assertEqual(analysis.error_code, "timeout")
        self.assertTrue(analysis.retryable)


class WiringBugTests(unittest.TestCase):
    def test_missing_attribute_names_the_code_bug(self):
        analysis = _analyze(
            "browser open failed: 'BrowserService' object has no "
            "attribute 'open_tab'")
        self.assertEqual(analysis.error_code, "wiring_attribute_error")
        self.assertIn("wiring bug", analysis.explanation.lower())
        self.assertFalse(analysis.retryable)
        self.assertIn("retry", analysis.suggested_fix.lower())
        self.assertIn("not", analysis.suggested_fix.lower())

    def test_unknown_explanation_carries_the_message(self):
        analysis = _analyze("browser frobnicate failed: frobnicate "
                            "exploded (code 7)")
        self.assertIn("frobnicate exploded", analysis.explanation)


if __name__ == "__main__":
    unittest.main()
