"""Tests for nomorals/media/cookies.py — YouTube cookie discovery for yt-dlp."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nomorals.media.cookies import (
    BOT_COOKIE_HELP,
    COOKIES_ENV,
    cookies_status,
    find_cookies_file,
    is_bot_detection_error,
    ytdlp_cookie_args,
    ytdlp_cookie_opts,
)


def _write(tmp_path, name="youtube-cookies.txt", content="# Netscape\n.youtube.com\tTRUE\n"):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ── discovery ─────────────────────────────────────────────────────────────

def test_missing_everywhere_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_cookies_file() is None


def test_home_dot_nomorals_found(tmp_path, monkeypatch):
    d = tmp_path / ".nomorals"
    d.mkdir()
    (d / "youtube-cookies.txt").write_text("# Netscape\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    assert find_cookies_file() == str(d / "youtube-cookies.txt")


def test_env_var_override(tmp_path, monkeypatch):
    custom = _write(tmp_path, "custom.txt")
    monkeypatch.setenv(COOKIES_ENV, custom)
    monkeypatch.setenv("HOME", str(tmp_path))  # no file here
    assert find_cookies_file() == custom


def test_termux_prefix_location(tmp_path, monkeypatch):
    d = tmp_path / "etc" / "nomorals"
    d.mkdir(parents=True)
    (d / "youtube-cookies.txt").write_text("# Netscape\n")
    monkeypatch.setenv("PREFIX", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    assert find_cookies_file() == str(d / "youtube-cookies.txt")


def test_cwd_fallback(tmp_path, monkeypatch):
    _write(tmp_path)  # ./youtube-cookies.txt
    monkeypatch.setenv("HOME", str(tmp_path / "nope"))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_cookies_file() == str(tmp_path / "youtube-cookies.txt")


def test_empty_file_ignored(tmp_path, monkeypatch):
    d = tmp_path / ".nomorals"
    d.mkdir()
    (d / "youtube-cookies.txt").write_text("")  # empty → skip
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_cookies_file() is None


def test_discovery_order_home_first(tmp_path, monkeypatch):
    # home file wins over env var when both exist (order: home → prefix → env → cwd)
    d = tmp_path / ".nomorals"
    d.mkdir()
    home_file = d / "youtube-cookies.txt"
    home_file.write_text("# home\n")
    env_file = _write(tmp_path, "env.txt")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(COOKIES_ENV, env_file)
    assert find_cookies_file() == str(home_file)


def test_never_raises_on_broken_home(tmp_path, monkeypatch):
    # HOME points at a regular file, not a dir — discovery must not crash
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    monkeypatch.setenv("HOME", str(f))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_cookies_file() is None  # no exception


# ── yt-dlp wiring helpers ─────────────────────────────────────────────────

def test_cookie_opts_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert ytdlp_cookie_opts() == {}
    assert ytdlp_cookie_args() == []


def test_cookie_opts_include_path_when_found(tmp_path, monkeypatch):
    d = tmp_path / ".nomorals"
    d.mkdir()
    p = d / "youtube-cookies.txt"
    p.write_text("# Netscape\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(COOKIES_ENV, raising=False)
    assert ytdlp_cookie_opts() == {"cookiefile": str(p)}
    assert ytdlp_cookie_args() == ["--cookies", str(p)]


# ── bot detection ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "Sign in to confirm you're not a bot",
    "ERROR: [youtube] abc123: Sign in to confirm you are not a bot. Use --cookies-from-browser or --cookies for the authentication.",
    "youtube download failed: not a bot check failed",
])
def test_bot_detection_matches(msg):
    assert is_bot_detection_error(msg) is True


@pytest.mark.parametrize("msg", [
    "network timeout",
    "DRM protected",
    "pip install yt-dlp",
    "",
    None,
    12345,
])
def test_bot_detection_no_false_positives(msg):
    assert is_bot_detection_error(msg) is False


def test_bot_help_is_two_lines():
    assert len(BOT_COOKIE_HELP.strip().split(". ")) <= 3
    assert "youtube-cookies.txt" in BOT_COOKIE_HELP
    assert "~/.nomorals" in BOT_COOKIE_HELP


# ── status ────────────────────────────────────────────────────────────────

def test_status_reports_checked_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    st = cookies_status()
    assert st["found"] is None
    assert isinstance(st["checked"], list) and len(st["checked"]) >= 2
    assert st["env_var"] == COOKIES_ENV


def test_status_never_raises(tmp_path, monkeypatch):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    monkeypatch.setenv("HOME", str(f))
    st = cookies_status()
    assert st["found"] is None
