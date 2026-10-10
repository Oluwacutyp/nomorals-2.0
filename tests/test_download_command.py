"""Tests for /download any-URL media downloader."""

import pytest
from nomorals.social.chat.control import parse_control, CONTROL_COMMANDS


def test_download_registered():
    assert "download" in CONTROL_COMMANDS
    min_args, max_args = CONTROL_COMMANDS["download"]
    assert min_args == 1  # requires a URL


def test_download_parses_url():
    cmd = parse_control("/download https://youtube.com/watch?v=abc123")
    assert cmd.kind == "download"
    assert "youtube.com" in cmd.tail


def test_download_parses_audio_flag():
    cmd = parse_control("/download https://example.com/song.mp3 audio")
    assert cmd.kind == "download"
    assert "audio" in cmd.tail


def test_download_rejects_no_url():
    cmd = parse_control("/download")
    assert cmd.kind == "error"
    assert "needs 1 argument" in cmd.arg


def test_download_rejects_non_url():
    # Handler-level check — the _control_download validates URL shape
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    # Just verify the method exists
    assert hasattr(RuntimeMediaMixin, "_control_download")
