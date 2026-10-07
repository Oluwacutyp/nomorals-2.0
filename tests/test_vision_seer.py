"""Tests for the vision ("eyes") system."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nomorals.vision.seer import Seer, VisionUnavailable, get_seer, see
from nomorals.vision.screenshot import (
    ScreenshotUnavailable,
    screenshot_from_file,
)


# ── helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def png_file(tmp_path):
    p = tmp_path / "shot.png"
    # Minimal valid PNG header + junk (we mock the vision call anyway)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return p


@pytest.fixture
def mock_router():
    router = MagicMock()
    resp = MagicMock()
    resp.text = "A red button in the top-right corner."
    router.describe_image.return_value = resp
    return router


# ── Seer ──────────────────────────────────────────────────────────────

class TestSeer:
    def test_see_via_router(self, png_file, mock_router):
        seer = Seer(router=mock_router)
        result = seer.see(png_file, "Where is the button?")
        assert "red button" in result
        mock_router.describe_image.assert_called_once()
        # Prompt must enforce the sensor role
        prompt = mock_router.describe_image.call_args[0][1]
        assert "not an agent" in prompt.lower() or "sensor" in prompt.lower()

    def test_see_missing_file(self, tmp_path):
        seer = Seer(router=MagicMock())
        with pytest.raises(Exception, match="not found"):
            seer.see(tmp_path / "nope.png", "what?")

    def test_see_no_vision_provider(self, png_file):
        router = MagicMock()
        router.describe_image.side_effect = Exception(
            "no registered provider supports vision"
        )
        seer = Seer(router=router)
        # Local path will also fail (no lifecycle) → VisionUnavailable
        with pytest.raises(VisionUnavailable, match="vision unavailable"):
            with patch.dict("sys.modules", {}):
                # Force the local path to fail fast
                with patch.object(
                    seer, "_see_via_local",
                    side_effect=VisionUnavailable("no local"),
                ):
                    seer.see(png_file, "what?")

    def test_see_empty_response(self, png_file):
        router = MagicMock()
        resp = MagicMock()
        resp.text = "   "
        router.describe_image.return_value = resp
        seer = Seer(router=router)
        # Empty router response falls through to local; with no local
        # available it must fail fast, not return an empty string.
        with pytest.raises(VisionUnavailable, match="vision unavailable"):
            seer.see(png_file)

    def test_sensor_prompt_has_no_decision_language(self):
        from nomorals.vision.seer import SEER_SYSTEM_PROMPT
        p = SEER_SYSTEM_PROMPT.lower()
        assert "describe" in p
        assert "do not make decisions" in p or "not make decisions" in p

    def test_get_seer_singleton(self):
        assert get_seer() is get_seer()

    def test_see_convenience(self, png_file, mock_router):
        import nomorals.vision.seer as seer_mod
        with patch.object(seer_mod, "_default_seer", Seer(router=mock_router)):
            result = see(png_file, "what?")
            assert "red button" in result


class TestLazyLoading:
    def test_unload_timer_scheduled(self, png_file, mock_router):
        # Router path doesn't touch local loading
        seer = Seer(router=mock_router, idle_timeout=0.2)
        seer.see(png_file)
        assert not seer._local_loaded  # router path never loads local

    def test_manual_unload(self, mock_router):
        seer = Seer(router=mock_router)
        seer._local_loaded = True
        with patch.object(seer, "_do_unload_local") as mock_unload:
            seer.unload_local()
            mock_unload.assert_called_once()
        assert not seer._local_loaded

    def test_idle_unload_fires(self, mock_router):
        seer = Seer(router=mock_router, idle_timeout=0.1)
        seer._local_loaded = True
        seer._local_last_used = time.time() - 1.0  # idle
        with patch.object(seer, "_do_unload_local") as mock_unload:
            seer._schedule_unload()
            time.sleep(0.3)
            mock_unload.assert_called_once()

    def test_idle_unload_reschedules_if_used(self, mock_router):
        seer = Seer(router=mock_router, idle_timeout=0.1)
        seer._local_loaded = True
        seer._local_last_used = time.time()  # just used
        with patch.object(seer, "_do_unload_local") as mock_unload:
            with patch.object(seer, "_schedule_unload") as mock_resched:
                seer._idle_unload()
                mock_unload.assert_not_called()
                mock_resched.assert_called_once()


# ── screenshots ───────────────────────────────────────────────────────

class TestScreenshots:
    def test_from_file_ok(self, png_file):
        assert screenshot_from_file(png_file) == png_file

    def test_from_file_missing(self, tmp_path):
        with pytest.raises(ScreenshotUnavailable, match="not found"):
            screenshot_from_file(tmp_path / "nope.png")

    def test_from_file_not_image(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("hello")
        with pytest.raises(ScreenshotUnavailable, match="not an image"):
            screenshot_from_file(p)

    def test_from_file_empty(self, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        with pytest.raises(ScreenshotUnavailable, match="empty"):
            screenshot_from_file(p)


# ── game state from screenshot ────────────────────────────────────────

class TestVisualGameState:
    def test_state_from_screenshot(self, png_file):
        import nomorals.vision.seer as seer_mod
        mock_router = MagicMock()
        resp = MagicMock()
        resp.text = (
            "Health bar at 75%. Balance: ₦1,250. Level 5. "
            "A green ATTACK button bottom-center."
        )
        mock_router.describe_image.return_value = resp
        with patch.object(seer_mod, "_default_seer", Seer(router=mock_router)):
            from nomorals.games.player.state import state_from_screenshot
            state = state_from_screenshot(
                str(png_file), url="https://game.example/play"
            )
            assert state.url == "https://game.example/play"
            assert state.visual  # visual description stored
            assert state.numbers.get("naira") == 1250.0
            assert state.numbers.get("level") == 5.0
            assert "ATTACK" in state.text

    def test_visual_in_summary(self, png_file):
        from nomorals.games.player.state import GameState
        state = GameState(
            url="u", title="t", text="x", visual="A dragon."
        )
        assert "dragon" in state.summary().lower()


# ── tool registration ─────────────────────────────────────────────────

class TestRegistration:
    def test_see_registered(self):
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry()
        from nomorals.tools.seer import register
        register(registry)
        assert "see" in registry._tools
        spec = registry._tools["see"]
        assert "image" in spec.description.lower()
