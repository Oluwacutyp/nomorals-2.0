"""Owner-directed opens are not crawling (owner
screenshot, 2026-11-08): Devon refused the owner's own
x.com link with "robots.txt disallows …" inside a job
whose failure rendered as a raw Python dict, because
the browse job was a urllib fetch wearing a job's
clothes — there was no web handoff at all.

Pinned here:

- the fetch gate is scoped by ORIGIN: an owner-directed
  BrowserSession (origin="owner") opens a
  robots-disallowed URL; an autonomous session is still
  refused, before any network is touched (today's
  politeness, unchanged);
- the browse JOB opens through the real browser organ
  (webnav.navigate: fingerprint ladder, persistent
  profile, takeover factory attached) and returns the
  page's content — robots is never consulted;
- when no browser engine exists on the host, the job's
  fetch fallback is owner-directed too (robots does not
  gate the owner's own link on any rung);
- a failed browse job renders a plain sentence — never
  a "{'ok': False, 'steps': …}" dict repr.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nomorals.agents.coremind import CoreMind, Intent
from nomorals.core.errors import ToolError
from nomorals.social.chat import webnav
from nomorals.tools import browser as B

OWNER_URL = "https://x.com/i/status/2119069260060201273"

PAGE_HTML = (b"<html><head><title>Post</title></head>"
             b"<body><h1>A post</h1>"
             b"<p>the owner link content</p></body></html>")


# ── transport fakes (the test_tools_browser_failures shape) ──


class _Headers(dict):
    def get(self, key, default=None):
        if key == "Content-Type":
            return "text/html; charset=utf-8"
        return default


class _Resp:
    def __init__(self, body):
        self._body = body
        self._pos = 0
        self.headers = _Headers()
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if self._pos >= len(self._body):
            return b""
        if n is None or n < 0:
            chunk = self._body[self._pos:]
        else:
            chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def geturl(self):
        return OWNER_URL


class _Opener:
    def __init__(self, resp):
        self._resp = resp
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        return self._resp


def _deny_robots(monkeypatch):
    monkeypatch.setattr(
        B, "_robots_allowed", lambda url, ua: False)


# ── the gate, scoped by origin ────────────────────────────


def test_owner_session_opens_robots_disallowed_url(monkeypatch):
    _deny_robots(monkeypatch)
    sess = B.BrowserSession(origin="owner", retries=0)
    sess._opener = _Opener(_Resp(PAGE_HTML))
    out = sess.open(OWNER_URL)
    assert out["ok"] is True
    assert "the owner link content" in sess.text()["text"]


def test_autonomous_session_still_refused(monkeypatch):
    _deny_robots(monkeypatch)
    sess = B.BrowserSession(retries=0)
    opener = _Opener(_Resp(PAGE_HTML))
    sess._opener = opener
    with pytest.raises(ToolError, match="robots.txt disallows"):
        sess.open(OWNER_URL)
    assert opener.calls == 0  # refused before the network


def test_owner_fetch_bytes_ignores_robots(monkeypatch, tmp_path):
    _deny_robots(monkeypatch)
    sess = B.BrowserSession(origin="owner", retries=0)
    sess._opener = _Opener(_Resp(PAGE_HTML))
    dest = tmp_path / "page.bin"
    out = sess.fetch_bytes(OWNER_URL, dest)
    assert out["bytes"] == len(PAGE_HTML)
    assert dest.read_bytes() == PAGE_HTML


def test_autonomous_fetch_bytes_still_refused(monkeypatch,
                                              tmp_path):
    _deny_robots(monkeypatch)
    sess = B.BrowserSession(retries=0)
    sess._opener = _Opener(_Resp(PAGE_HTML))
    with pytest.raises(ToolError, match="robots.txt disallows"):
        sess.fetch_bytes(OWNER_URL, tmp_path / "page.bin")


def test_origin_defaults_to_autonomous_and_normalizes():
    assert B.BrowserSession(retries=0).origin == "autonomous"
    assert B.BrowserSession(
        origin="owner", retries=0).origin == "owner"
    assert B.BrowserSession(
        origin="anything-else", retries=0).origin == "autonomous"


# ── the browse job: real browser handoff ──────────────────


class _FakeSettings:
    def __init__(self, mind=None):
        self._mind = mind

    def resolve(self, key):
        raise RuntimeError("no settings in tests")

    def __getattr__(self, name):
        if name == "mind" and self._mind is not None:
            return self._mind
        raise AttributeError(name)


class _FakeContext:
    def __init__(self, mind=None):
        self.settings = _FakeSettings(mind)
        self.extras = {}
        self.memory = None
        self.router = None
        self.db = None


def _mind():
    return CoreMind(_FakeContext(mind=SimpleNamespace(
        max_inflight=4, inflight_acquire_timeout_s=5.0)))


class _FakeNavSession:
    def __init__(self, html):
        self.probe = SimpleNamespace(html=lambda: html)
        self.exited = False

    def __exit__(self, *a):
        self.exited = True


def _capture_job(mind, monkeypatch):
    """Run _dispatch_browse's job closure synchronously."""
    captured = {}

    def fake_send_async(chat_key, job, job_id, note,
                        kind="job"):
        captured["job"] = job
        return note

    monkeypatch.setattr(mind, "_send_async", fake_send_async)
    intent = Intent("browse", 0.9, target=OWNER_URL,
                    route="browser")
    mind._dispatch_browse(intent, "job-1", "chat-1", None)
    return captured["job"]


def test_browse_job_opens_owner_link_through_browser(
        monkeypatch):
    _deny_robots(monkeypatch)  # robots refuses this URL —
    # the browser handoff must never even consult it
    calls = {}

    def fake_navigate(url, **kw):
        calls["url"] = url
        calls.update(kw)
        return _FakeNavSession(
            "<html><body><h1>Post</h1>"
            "<p>the owner link content</p></body></html>"), \
            SimpleNamespace(final_url=url, title="Post")

    monkeypatch.setattr(webnav, "navigate", fake_navigate)
    job = _capture_job(_mind(), monkeypatch)
    out = job()
    assert "the owner link content" in out
    assert calls["url"] == OWNER_URL
    assert calls["ladder"], "the fingerprint ladder rides"
    assert calls["takeover_budget_s"] > 120.0
    assert "{'ok':" not in out


def test_browse_job_wall_failure_is_a_sentence(monkeypatch):
    def walled(url, **kw):
        raise webnav.WallBlocked(
            "bot wall (hard) at " + url + " did not clear "
            "within the takeover hold — nothing was "
            "captured", wall="hard",
            attempts=["managed wait-out"])

    monkeypatch.setattr(webnav, "navigate", walled)
    job = _capture_job(_mind(), monkeypatch)
    out = job()
    assert out.startswith("❌")
    assert "bot wall" in out
    assert OWNER_URL in out
    assert "{'ok':" not in out
    assert "'steps'" not in out
    assert "session" not in out.split("—")[0]


def test_browse_job_fetch_fallback_is_owner_directed(
        monkeypatch):
    _deny_robots(monkeypatch)

    def unavailable(url, **kw):
        raise webnav.NavUnavailable(
            "no browser engine on this host")

    monkeypatch.setattr(webnav, "navigate", unavailable)
    monkeypatch.setattr(
        B.BrowserSession, "_build_opener",
        lambda self: _Opener(_Resp(PAGE_HTML)))
    mind = _mind()
    out = mind._browse_job(OWNER_URL, job_id="job-9")
    assert "the owner link content" in out
    assert "{'ok':" not in out


def test_browse_job_empty_page_is_a_sentence(monkeypatch):
    def empty(url, **kw):
        return _FakeNavSession(
            "<html><body></body></html>"), \
            SimpleNamespace(final_url=url, title="")

    monkeypatch.setattr(webnav, "navigate", empty)
    mind = _mind()
    out = mind._browse_job(OWNER_URL)
    assert out.startswith("❌")
    assert "no readable text" in out
    assert "{'ok':" not in out
