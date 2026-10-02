"""Retry-policy tests for the browser tool (wave D production behaviors).

The policy under test:

* transient failures (timeout/reset/5xx) are retried with backoff, but
  ONLY for idempotent requests (GET navigation);
* a form POST is attempted exactly once — retrying could duplicate the
  submission;
* multi-step ``task()`` retries only idempotent steps (open/text/
  markdown/links/extract/back/wait); fill/submit/click never retry;
* 4xx responses are never retried; every terminal failure raises a real
  ToolError naming the method and URL.

All network I/O is mocked at the opener level — no live requests.
"""

from __future__ import annotations

import io
import unittest
import urllib.error
from unittest import mock

from nomorals.core.errors import ToolError
from nomorals.tools.browser import BrowserSession, parse_html


class _FakeHeaders(dict):
    pass


class _FakeResponse:
    """Minimal stand-in for the object urllib's opener returns."""

    def __init__(self, body: bytes = b"<html><head><title>t</title></head>"
                                    b"<body>hi</body></html>",
                 url: str = "http://example.test/",
                 status: int = 200) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers = _FakeHeaders({"Content-Type": "text/html; charset=utf-8"})

    def read(self, n: int = -1) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _session(retries: int = 2) -> BrowserSession:
    return BrowserSession(retries=retries, respect_robots=False)


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "boom", {}, io.BytesIO(b"err"))


_FORM_HTML = (
    '<html><head><title>f</title></head><body>'
    '<form method="post" action="/go">'
    '<input name="q" value="a">'
    "</form></body></html>"
)


class FetchRetryTests(unittest.TestCase):
    def test_get_retries_transient_then_succeeds(self) -> None:
        sess = _session(retries=2)
        opener = mock.Mock()
        opener.open.side_effect = [
            urllib.error.URLError("connection reset"),
            _FakeResponse(),
        ]
        sess._opener = opener
        result = sess._fetch("http://example.test/")
        self.assertEqual(result["status"], 200)
        self.assertEqual(opener.open.call_count, 2)

    def test_get_gives_up_after_bounded_attempts(self) -> None:
        sess = _session(retries=2)
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.URLError("down")
        sess._opener = opener
        with self.assertRaises(ToolError) as ctx:
            sess._fetch("http://example.test/")
        self.assertEqual(opener.open.call_count, 3)  # initial + 2 retries
        self.assertIn("GET", str(ctx.exception))
        self.assertIn("http://example.test/", str(ctx.exception))

    def test_get_retries_5xx(self) -> None:
        sess = _session(retries=2)
        opener = mock.Mock()
        opener.open.side_effect = [
            _http_error("http://example.test/", 503),
            _FakeResponse(),
        ]
        sess._opener = opener
        result = sess._fetch("http://example.test/")
        self.assertEqual(result["status"], 200)
        self.assertEqual(opener.open.call_count, 2)

    def test_4xx_never_retried(self) -> None:
        sess = _session(retries=3)
        opener = mock.Mock()
        opener.open.side_effect = _http_error("http://example.test/", 404)
        sess._opener = opener
        result = sess._fetch("http://example.test/")
        self.assertEqual(result["status"], 404)
        self.assertEqual(opener.open.call_count, 1)

    def test_post_form_not_retried_on_transient_failure(self) -> None:
        sess = _session(retries=3)
        sess.url = "http://example.test/form"
        sess.dom = parse_html(_FORM_HTML)
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.URLError("timeout mid-post")
        sess._opener = opener
        with self.assertRaises(ToolError):
            sess.submit(target="0")
        # exactly one attempt: the server may already have the submission
        self.assertEqual(opener.open.call_count, 1)

    def test_post_form_not_retried_on_5xx(self) -> None:
        sess = _session(retries=3)
        sess.url = "http://example.test/form"
        sess.dom = parse_html(_FORM_HTML)
        opener = mock.Mock()
        opener.open.side_effect = _http_error("http://example.test/go", 500)
        sess._opener = opener
        result = sess.submit(target="0")
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 500)
        self.assertEqual(opener.open.call_count, 1)

    def test_get_form_submit_still_retries(self) -> None:
        html = _FORM_HTML.replace('method="post"', 'method="get"')
        sess = _session(retries=2)
        sess.url = "http://example.test/form"
        sess.dom = parse_html(html)
        opener = mock.Mock()
        opener.open.side_effect = [
            urllib.error.URLError("blip"),
            _FakeResponse(url="http://example.test/go?q=a"),
        ]
        sess._opener = opener
        result = sess.submit(target="0")
        self.assertTrue(result["ok"])
        self.assertEqual(opener.open.call_count, 2)


class TaskRetryTests(unittest.TestCase):
    def _task_session(self) -> BrowserSession:
        sess = _session()
        sess.url = "http://example.test/form"
        sess.dom = parse_html(_FORM_HTML)
        return sess

    def test_idempotent_step_retried_once(self) -> None:
        sess = self._task_session()
        calls = {"n": 0}

        def fake_step(act: str, step: dict) -> dict:
            calls["n"] += 1
            if act == "open" and calls["n"] == 1:
                raise ToolError("transient blip")
            return {"ok": True}

        with mock.patch.object(sess, "_run_task_step", side_effect=fake_step):
            report = sess.task([{"act": "open", "url": "http://example.test/"}],
                               stop_on_error=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["steps"][0]["attempts"], 2)
        self.assertTrue(report["steps"][0]["ok"])

    def test_submit_step_never_retried(self) -> None:
        sess = self._task_session()
        calls = {"n": 0}

        def fake_step(act: str, step: dict) -> dict:
            calls["n"] += 1
            raise ToolError("server hiccup")

        with mock.patch.object(sess, "_run_task_step", side_effect=fake_step):
            report = sess.task([{"act": "submit", "target": "0"}],
                               stop_on_error=False)
        step = report["steps"][0]
        self.assertFalse(step["ok"])
        self.assertEqual(step["attempts"], 1)
        self.assertEqual(calls["n"], 1)
        self.assertIn("server hiccup", step["error"])

    def test_click_and_fill_never_retried(self) -> None:
        sess = self._task_session()

        def fake_step(act: str, step: dict) -> dict:
            raise ToolError("nope")

        with mock.patch.object(sess, "_run_task_step", side_effect=fake_step):
            report = sess.task(
                [{"act": "click", "target": "0"},
                 {"act": "fill", "name": "q", "value": "x"}],
                stop_on_error=False)
        for step in report["steps"]:
            self.assertEqual(step["attempts"], 1,
                             f"{step['act']} must not retry")

    def test_failed_task_report_says_where(self) -> None:
        sess = self._task_session()

        def fake_step(act: str, step: dict) -> dict:
            if act == "open":
                return {"ok": True}
            raise ToolError("form exploded")

        with mock.patch.object(sess, "_run_task_step", side_effect=fake_step):
            report = sess.task(
                [{"act": "open", "url": "http://example.test/"},
                 {"act": "submit", "target": "0"}],
                stop_on_error=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["steps_done"], 2)
        last = report["steps"][1]
        self.assertEqual(last["act"], "submit")
        self.assertFalse(last["ok"])
        self.assertIn("form exploded", last["error"])


if __name__ == "__main__":
    unittest.main()
