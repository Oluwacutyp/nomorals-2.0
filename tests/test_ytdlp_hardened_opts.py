"""yt-dlp hardened opts — the exploration-sweep hang fix (2026-11-14).

The exploration sweep (morning) timed out at 1800s on EC2: yt-dlp's
``ssl_load_certs`` hung ~125s per SSL context enumerating the Windows
cert store (certifi not importable there), and a socket read hung
~180s with no timeout. The fix is one shared helper every
``yt_dlp.YoutubeDL`` in Devon starts from: a bounded ``socket_timeout``
plus a certifi self-provision so yt-dlp uses the bundle file.
"""

import sys
from unittest import mock

from nomorals.media.cookies import (
    YT_DLP_SOCKET_TIMEOUT_S,
    ensure_ytdlp_certs,
    ytdlp_hardened_opts,
)

# ── the opts themselves ──────────────────────────────────────────────────

def test_hardened_opts_carries_socket_timeout_and_skip_download():
    opts = ytdlp_hardened_opts()
    assert opts["socket_timeout"] == YT_DLP_SOCKET_TIMEOUT_S == 30
    assert opts["skip_download"] is True
    assert opts["quiet"] is True
    assert opts["no_warnings"] is True


def test_hardened_opts_download_shape_drops_skip_download():
    opts = ytdlp_hardened_opts(skip_download=False)
    assert opts["socket_timeout"] == 30
    assert "skip_download" not in opts


def test_hardened_opts_returns_a_fresh_dict_each_call():
    a = ytdlp_hardened_opts()
    b = ytdlp_hardened_opts()
    assert a is not b
    a["socket_timeout"] = 999
    assert b["socket_timeout"] == 30


# ── the certifi ensure ───────────────────────────────────────────────────

def test_ensure_ytdlp_certs_noop_when_certifi_present():
    # certifi IS importable on this box — the ensure must be a silent
    # no-op, never a pip run, never a raise.
    ensure_ytdlp_certs()
    ensure_ytdlp_certs()


def test_ensure_ytdlp_certs_never_raises_when_certifi_missing(monkeypatch):
    import nomorals.media.cookies as cookies

    monkeypatch.setitem(sys.modules, "certifi", None)
    monkeypatch.setattr(cookies, "_CERTIFI_PROVISION_ATTEMPTED", False)
    # Under pytest the provision is skipped by design (sys.modules
    # holds pytest) — the contract under test is: never raises.
    ensure_ytdlp_certs()


def test_ensure_ytdlp_certs_attempts_pip_once_when_missing_outside_pytest(monkeypatch):
    import nomorals.media.cookies as cookies

    monkeypatch.setitem(sys.modules, "certifi", None)
    monkeypatch.setattr(cookies, "_CERTIFI_PROVISION_ATTEMPTED", False)
    # Simulate a non-test process: pytest not in sys.modules.
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        result = mock.Mock()
        result.returncode = 0
        return result

    monkeypatch.setattr("subprocess.run", fake_run)
    ensure_ytdlp_certs()
    assert len(ran) == 1
    assert ran[0][-1] == "certifi"
    # A second call does not re-attempt.
    ensure_ytdlp_certs()
    assert len(ran) == 1


# ── the sweep path carries the bound ─────────────────────────────────────

class _FakeYDL:
    """Captures the opts a YoutubeDL is built with."""

    seen_opts: list = []

    def __init__(self, opts):
        type(self).seen_opts.append(dict(opts))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, query, download=False):
        return {"entries": [
            {"id": "dQw4w9WgXcQ", "title": "t", "duration": 10,
             "uploader": "u"}]}


def test_youtube_search_many_passes_socket_timeout(monkeypatch):
    from nomorals.media.playback import PlaybackEngine

    fake_module = mock.Mock()
    fake_module.YoutubeDL = _FakeYDL
    _FakeYDL.seen_opts.clear()
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    out = PlaybackEngine._youtube_search_many("some query", limit=3)
    assert out and out[0]["video_id"] == "dQw4w9WgXcQ"
    assert _FakeYDL.seen_opts, "YoutubeDL was never constructed"
    opts = _FakeYDL.seen_opts[0]
    assert opts["socket_timeout"] == 30
    assert opts["skip_download"] is True
