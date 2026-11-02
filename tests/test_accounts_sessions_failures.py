"""Tail-113 pins: nomorals/accounts/sessions.py census sites (461, 757).

Both sites guard the best-effort read of an HTTPError body; in both
the verdict survives intact and only the diagnostic detail degrades:

461 — ``SessionManager.probe``: the error body cannot be read ->
      body "". The classification stays fail-closed: an HTTPError
      code always lands on a code-driven failure rung (403 ->
      logged_out; any >= 400 -> unknown/logged_out), and with
      ok_markers supplied the empty body fails the marker check
      (-> unknown). The body only ever refines WHICH failure.
757 — ``refresh_oauth_token``: the rejection body cannot be read ->
      body stays "" and TokenRefreshError is still raised with the
      HTTP code and the recoverable classification intact; only the
      body excerpt in the message is lost.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.accounts.sessions import SessionManager, TokenRefreshError


class _ExplodingFP:
    def read(self, *args, **kwargs):
        raise RuntimeError("body gone")

    def close(self):
        pass


def _http_error(code: int, fp) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", None, fp)


class _RaisingOpener:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def open(self, req, timeout=None):
        raise self.exc


def _manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr.get_session = lambda service, username: SimpleNamespace(
        cookies={"sid": "x"})
    return mgr


def test_probe_unreadable_body_403_still_logged_out() -> None:
    mgr = _manager()
    opener = _RaisingOpener(_http_error(403, _ExplodingFP()))
    with mock.patch.object(urllib.request, "build_opener", return_value=opener):
        result = mgr.probe("svc", "user", "http://x/page")
    assert result["ok"] is False
    assert result["status"] == "logged_out"
    assert result["http_status"] == 403


def test_probe_unreadable_body_with_ok_markers_is_unknown() -> None:
    mgr = _manager()
    opener = _RaisingOpener(_http_error(403, _ExplodingFP()))
    with mock.patch.object(urllib.request, "build_opener", return_value=opener):
        result = mgr.probe("svc", "user", "http://x/page",
                           ok_markers=("dashboard",))
    assert result["ok"] is False
    assert result["status"] == "unknown"


def test_probe_readable_body_marker_control() -> None:
    mgr = _manager()
    opener = _RaisingOpener(
        _http_error(403, io.BytesIO(b"account locked page")))
    with mock.patch.object(urllib.request, "build_opener", return_value=opener):
        result = mgr.probe(
            "svc", "user", "http://x/page",
            marker_status={"locked": ("account locked",)})
    assert result["status"] == "locked"


def test_refresh_unreadable_body_error_intact() -> None:
    mgr = SessionManager.__new__(SessionManager)
    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=_http_error(500, _ExplodingFP())):
        with pytest.raises(TokenRefreshError) as caught:
            mgr.refresh_oauth_token(
                "svc", "u", "rt", "cid", "sec", "http://x/token")
    assert "(HTTP 500)" in str(caught.value)
    assert caught.value.recoverable is True
    assert str(caught.value).rstrip().endswith(":")


def test_refresh_readable_body_control() -> None:
    mgr = SessionManager.__new__(SessionManager)
    body = io.BytesIO(b'{"error":"invalid_grant"}')
    with mock.patch.object(urllib.request, "urlopen",
                           side_effect=_http_error(400, body)):
        with pytest.raises(TokenRefreshError) as caught:
            mgr.refresh_oauth_token(
                "svc", "u", "rt", "cid", "sec", "http://x/token")
    assert "invalid_grant" in str(caught.value)
    assert caught.value.recoverable is False
