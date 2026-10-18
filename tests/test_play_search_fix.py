"""Regression tests for the /play search failure (owner paste: /play
"goodbye" failed on ALL sources + a misleading, duplicated hint).

Bugs fixed:
1. Misleading hint: _finalize_failure fired the "yt-dlp not installed"
   hint on ANY "yt-dlp" substring — which matched the
   github.com/yt-dlp URL inside a YouTube BOT-CHECK error, claiming
   yt-dlp was "not installed" when it had just run. Now the install
   hint fires only when yt-dlp is VERIFIED missing (ytdlp_available)
   AND an error says so; bot-check errors get the cookies help.
2. Hint duplicated: _control_play appended resolution.hint AFTER
   explain_failure (which already includes it). Now once.
3. SoundCloud MONETIZE treated as preview-only: policy != ALLOW
   discarded real full tracks. Only the hard preview policies
   (PREVIEW/SNIP/BLOCK/DENY) count; MONETIZE is accepted (the real
   stream URL is the only preview-only proof, checked by the
   adapter's stream_url).
4. SoundCloud search fetched exactly 1 track and failed the whole
   strategy when it was preview-only. Now fetches up to 5 and skips
   preview-only ones.
5. YouTube bot-check error now names the cookies fix explicitly.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import resolver
from nomorals.media.resolver import (
    ResolutionError,
    ResolvedAudio,
    SourceResolver,
    YT_DLP_HINT,
    ytdlp_available,
)
from nomorals.media.cookies import BOT_COOKIE_HELP, YTDLP_INSTALL_HELP


def _track(title, policy=""):
    return {"permalink_url": f"https://soundcloud.com/a/{title}",
            "title": title, "artist": "Artist",
            "duration_ms": 180000, "policy": policy}


class FakeSoundCloud:
    def __init__(self, tracks):
        self.tracks = tracks

    def search_tracks(self, query, limit=1):
        return self.tracks[:limit]


def _ctx():
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=""))


# ── 1. hint classification: verified facts, not substring matches ──

BOTCHECK_ERROR = (
    "youtube-search: youtube search failed: ERROR: [youtube] VBS4-hMEHfg: "
    "Sign in to confirm you're not a bot. Use --cookies-from-browser or "
    "--cookies for the authentication. See "
    "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp "
    "for how to manually pass cookies."
)


def test_botcheck_error_with_ytdlp_present_gets_cookies_hint_not_install(
        monkeypatch):
    """The exact owner-paste error: yt-dlp RAN (bot check), so the hint
    must be cookies — never 'not installed'."""
    monkeypatch.setattr(resolver, "ytdlp_available", lambda: True)
    result = ResolvedAudio(errors=[BOTCHECK_ERROR])
    out = SourceResolver._finalize_failure(result)
    assert "cookies" in out.hint.lower()
    assert "not installed" not in out.hint
    assert out.hint == BOT_COOKIE_HELP


def test_missing_ytdlp_verified_gets_install_hint(monkeypatch):
    monkeypatch.setattr(resolver, "ytdlp_available", lambda: False)
    result = ResolvedAudio(errors=[
        "youtube-search: youtube search needs yt-dlp (not installed here)"])
    out = SourceResolver._finalize_failure(result)
    assert out.hint == YT_DLP_HINT
    # Official install form (yt-dlp wiki), not an invented one-liner.
    assert 'pip install -U "yt-dlp[default]"' in out.hint
    assert "https://github.com/yt-dlp/yt-dlp/wiki/Installation" in out.hint


def test_ytdlp_available_detects_module_and_cli(monkeypatch):
    import importlib.util
    import shutil

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object())
    assert ytdlp_available() is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/yt-dlp")
    assert ytdlp_available() is True
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert ytdlp_available() is False


def test_install_help_is_official_not_invented():
    assert YTDLP_INSTALL_HELP == YT_DLP_HINT
    assert "winget install --source winget --id yt-dlp.yt-dlp" in YTDLP_INSTALL_HELP
    assert "pkg install python python-pip" in YTDLP_INSTALL_HELP
    assert "ffmpeg" in YTDLP_INSTALL_HELP


def test_cookie_help_is_official_faq_grounded():
    assert "--cookies-from-browser chrome" in BOT_COOKIE_HELP
    assert "--cookies cookies.txt" in BOT_COOKIE_HELP
    assert "Netscape" in BOT_COOKIE_HELP
    assert "Get cookies.txt LOCALLY" in BOT_COOKIE_HELP
    assert "malware" in BOT_COOKIE_HELP


# ── 2. hint printed once (end-to-end through _control_play) ──

def test_control_play_failure_hint_printed_once():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = mock.Mock()
    rt._ref_from_key = lambda key: None  # CLI path (no chat)

    fake_resolution = ResolvedAudio(
        ok=False,
        attempts=["workspace-scan — failed", "youtube-search — failed"],
        errors=["workspace-scan: no local audio matches 'goodbye'",
                BOTCHECK_ERROR],
        hint=BOT_COOKIE_HELP)
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine"):
        SR.return_value.resolve.return_value = fake_resolution
        text = rt._control_play("goodbye", "")
    assert text.startswith("couldn't find 'goodbye'")
    # The hint body appears exactly once — not once as "hint: ..." plus
    # once bare (the old duplication).
    assert text.count("YouTube is blocking automated requests") == 1
    assert "hint:" in text


# ── 3. MONETIZE is a full track, not preview-only ──

def test_monetize_is_not_preview_only():
    assert SourceResolver._preview_reason({"policy": "MONETIZE"}) == ""
    assert SourceResolver._preview_reason({"policy": "ALLOW"}) == ""
    assert SourceResolver._preview_reason({"policy": ""}) == ""
    assert SourceResolver._preview_reason({}) == ""


def test_hard_preview_policies_still_rejected():
    for policy in ("PREVIEW", "SNIP", "BLOCK", "DENY"):
        assert SourceResolver._preview_reason({"policy": policy})
    assert SourceResolver._preview_reason({"preview": True})
    assert SourceResolver._preview_reason(
        {"preview_url": "https://example.com/preview.mp3"})


def test_monetize_track_resolves():
    r = SourceResolver(_ctx())
    out = r._track_result(_track("welcome and goodbye", "MONETIZE"),
                          "SoundCloud search")
    assert out.ok
    assert out.title == "welcome and goodbye"


# ── 4. SoundCloud search skips preview-only tracks ──

def test_soundcloud_search_skips_preview_first_track():
    sc = FakeSoundCloud([
        _track("preview one", "PREVIEW"),
        _track("welcome and goodbye", "MONETIZE"),
    ])
    r = SourceResolver(_ctx(), soundcloud=sc)
    out = r._s_soundcloud_search("goodbye")
    assert out.ok
    assert out.title == "welcome and goodbye"


def test_soundcloud_search_all_preview_fails_honestly():
    sc = FakeSoundCloud([_track("p1", "PREVIEW"), _track("p2", "SNIP")])
    r = SourceResolver(_ctx(), soundcloud=sc)
    with pytest.raises(ResolutionError, match="all 2 .* preview-only"):
        r._s_soundcloud_search("goodbye")


# ── 5. YouTube bot-check names the cookies fix ──

def test_youtube_search_botcheck_names_cookies(monkeypatch):
    from nomorals.media.playback import PlaybackEngine

    monkeypatch.setattr(
        PlaybackEngine, "_youtube_search_id",
        classmethod(lambda cls, q: (_ for _ in ()).throw(
            RuntimeError(BOTCHECK_ERROR))))
    r = SourceResolver(_ctx())
    with pytest.raises(ResolutionError) as excinfo:
        r._s_youtube_search("goodbye")
    assert "cookies" in str(excinfo.value).lower()
    assert "not installed" not in str(excinfo.value)
