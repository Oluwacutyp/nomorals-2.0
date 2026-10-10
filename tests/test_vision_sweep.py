"""Vision sweep tests: mined-then-built upgrades to nomorals/vision/.

- seer.py: context bug fix (self._context), image normalization, transient
  retries, bytes input, context-manager unload.
- screenshot.py: host screen capture strategy chain.
- native.py: ocr_text, resize_for_model, screen_marks (Set-of-Mark),
  mark_prompt_snippet, analyze() OCR section.
"""

from __future__ import annotations

import io
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from nomorals.vision import native as N
from nomorals.vision import screenshot as SC
from nomorals.vision.screenshot import ScreenshotUnavailable
from nomorals.vision.seer import (
    UNTRUSTED_VISION_PREFIX,
    Seer,
    VisionUnavailable,
    _is_transient,
    _prepare_bytes,
    see,
    see_bytes,
)
from nomorals.core.errors import ToolError


# ── helpers ───────────────────────────────────────────────────────────

def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scene_png(size=(320, 240)) -> bytes:
    img = Image.new("RGB", size, (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 60, 180, 120], fill=(30, 144, 255))
    d.text((110, 80), "Click me", fill=(255, 255, 255))
    return _png(img)


@pytest.fixture
def png_file(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(_scene_png())
    return p


@pytest.fixture
def mock_router():
    router = MagicMock()
    resp = MagicMock()
    resp.text = "A blue button at top-center."
    router.describe_image.return_value = resp
    return router


# ── seer: context bug fix ─────────────────────────────────────────────

class TestSeerContext:
    def test_injected_router_used_directly(self, png_file, mock_router):
        seer = Seer(router=mock_router)
        result = seer.see(png_file, "what?")
        assert "blue button" in result
        mock_router.describe_image.assert_called_once()
        args = mock_router.describe_image.call_args[0]
        assert isinstance(args[0], bytes) and args[0].startswith(b"\x89PNG")
        assert "sensor" in args[1].lower()

    def test_context_passed_to_brain_for(self, png_file):
        import nomorals.vision.seer as seer_mod
        ctx = SimpleNamespace(router=MagicMock())
        brain = MagicMock()
        resp = MagicMock()
        resp.text = "seen via brain"
        brain.describe_image.return_value = resp
        with patch.object(seer_mod, "brain_for", return_value=brain) as bf:
            seer = Seer(context=ctx)  # no injected router
            result = seer.see(png_file)
        bf.assert_called_once_with(ctx)
        assert "seen via brain" in result

    def test_no_context_falls_back_to_shared_brain(self, png_file):
        import nomorals.vision.seer as seer_mod
        brain = MagicMock()
        resp = MagicMock()
        resp.text = "shared brain saw it"
        brain.describe_image.return_value = resp
        with patch.object(seer_mod, "brain_for", return_value=brain) as bf:
            result = Seer().see(png_file)
        bf.assert_called_once_with(None)
        assert "shared brain saw it" in result

    def test_result_marked_untrusted(self, png_file, mock_router):
        result = Seer(router=mock_router).see(png_file)
        assert result.startswith(UNTRUSTED_VISION_PREFIX)

    def test_see_bytes_convenience(self, mock_router):
        import nomorals.vision.seer as seer_mod
        with patch.object(seer_mod, "_default_seer",
                          Seer(router=mock_router)):
            result = see_bytes(_scene_png(), "q?")
        assert "blue button" in result

    def test_see_accepts_bytes(self, mock_router):
        result = Seer(router=mock_router).see(_scene_png(), "q?")
        assert "blue button" in result

    def test_empty_bytes_rejected(self, mock_router):
        with pytest.raises(ToolError):
            Seer(router=mock_router).see_bytes(b"")

    def test_context_manager_unloads(self, mock_router):
        seer = Seer(router=mock_router)
        with patch.object(seer, "unload_local") as unl:
            with seer:
                pass
            unl.assert_called_once()


# ── seer: image normalization ─────────────────────────────────────────

class TestPrepareBytes:
    def test_huge_image_downscaled(self):
        big = _png(Image.new("RGB", (3000, 2000), (10, 20, 30)))
        out = _prepare_bytes(big, 1568)
        img = Image.open(io.BytesIO(out))
        assert max(img.size) <= 1568
        assert out.startswith(b"\x89PNG")

    def test_small_image_untouched(self):
        small = _scene_png()
        assert _prepare_bytes(small, 1568) == small

    def test_bmp_converted_to_png(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (1, 2, 3)).save(buf, format="BMP")
        out = _prepare_bytes(buf.getvalue(), 1568)
        assert out.startswith(b"\x89PNG")

    def test_garbage_bytes_pass_through(self):
        assert _prepare_bytes(b"not an image", 1568) == b"not an image"

    def test_see_sends_downscaled_payload(self, mock_router):
        big = _png(Image.new("RGB", (3000, 2000), (10, 20, 30)))
        Seer(router=mock_router, max_dimension=800).see_bytes(big)
        payload = mock_router.describe_image.call_args[0][0]
        assert max(Image.open(io.BytesIO(payload)).size) <= 800


# ── seer: transient retries ───────────────────────────────────────────

class TestRetries:
    def _brain(self, effects):
        brain = MagicMock()
        resp = MagicMock()
        resp.text = "recovered"
        brain.describe_image.side_effect = effects + [resp]
        return brain

    def test_transient_retried_then_succeeds(self):
        import nomorals.vision.seer as seer_mod
        brain = self._brain([TimeoutError("timed out"),
                             ConnectionError("connection reset")])
        with patch.object(seer_mod, "brain_for", return_value=brain), \
             patch.object(seer_mod.time, "sleep") as slp:
            result = Seer().see_bytes(_scene_png())
        assert "recovered" in result
        assert brain.describe_image.call_count == 3
        assert slp.call_count == 2  # exponential backoff slept

    def test_permanent_error_not_retried(self):
        import nomorals.vision.seer as seer_mod
        brain = MagicMock()
        brain.describe_image.side_effect = ValueError("bad request")
        with patch.object(seer_mod, "brain_for", return_value=brain), \
             patch.object(seer_mod.time, "sleep") as slp:
            seer = Seer()
            with patch.object(seer, "_see_via_local",
                              side_effect=VisionUnavailable("no local")):
                with pytest.raises(VisionUnavailable):
                    seer.see_bytes(_scene_png())
        assert brain.describe_image.call_count == 1
        slp.assert_not_called()

    def test_no_vision_provider_fails_fast(self):
        import nomorals.vision.seer as seer_mod
        brain = MagicMock()
        brain.describe_image.side_effect = Exception(
            "no registered provider supports vision")
        with patch.object(seer_mod, "brain_for", return_value=brain):
            with pytest.raises(VisionUnavailable, match="no registered"):
                Seer().see_bytes(_scene_png())

    def test_is_transient_markers(self):
        assert _is_transient(TimeoutError("timed out"))
        assert _is_transient(Exception("429 too many requests"))
        assert _is_transient(ConnectionError("connection reset by peer"))
        assert _is_transient(Exception("503 service unavailable"))
        assert not _is_transient(ValueError("bad request"))
        assert not _is_transient(Exception(
            "no registered provider supports vision"))


# ── screenshot: host capture chain ────────────────────────────────────

class TestCaptureScreen:
    def _no_env(self):
        return patch.dict(os.environ,
                          {"PATH": "/nonexistent"}, clear=True)

    def test_no_backends_raises_with_hints(self):
        with self._no_env(), \
             patch.object(SC, "_which", return_value=None), \
             patch.dict(sys.modules, {"mss": None}), \
             patch.object(SC.sys, "platform", "linux"):
            with pytest.raises(ScreenshotUnavailable) as ei:
                SC.capture_screen()
        assert "pip install mss" in str(ei.value)

    def test_termux_backend(self, tmp_path):
        dest = tmp_path / "s.png"
        dest.write_bytes(_scene_png())
        with patch.object(SC, "_which",
                          side_effect=lambda *n: "/bin/termux-screenshot"
                          if "termux-screenshot" in n else None), \
             patch.object(SC, "_run_capture", return_value=dest):
            assert SC.capture_screen(dest=dest) == dest

    def test_mss_backend(self, tmp_path):
        dest = tmp_path / "s.png"
        fake_mss = MagicMock()
        ctx = MagicMock()
        fake_mss.mss.return_value.__enter__.return_value = ctx

        def _shot(output=""):
            Path(output).write_bytes(_scene_png())
        ctx.shot.side_effect = _shot
        with patch.object(SC, "_which", return_value=None), \
             patch.dict(sys.modules, {"mss": fake_mss}):
            got = SC.capture_screen(dest=dest)
        assert got == dest and dest.is_file()

    def test_grim_on_wayland(self, tmp_path):
        dest = tmp_path / "s.png"
        dest.write_bytes(_scene_png())
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
             patch.object(SC, "_which",
                          side_effect=lambda *n: "/usr/bin/grim"
                          if n == ("grim",) else None), \
             patch.object(SC, "_run_capture", return_value=dest), \
             patch.object(SC.sys, "platform", "linux"):
            assert SC.capture_screen(dest=dest) == dest

    def test_scrot_not_used_on_wayland(self):
        # scrot captures black on Wayland — must never be attempted there.
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
             patch.object(SC, "_which",
                          side_effect=lambda *n: "/usr/bin/scrot"
                          if n == ("scrot",) else None), \
             patch.dict(sys.modules, {"mss": None}), \
             patch.object(SC.sys, "platform", "linux"):
            with pytest.raises(ScreenshotUnavailable):
                SC.capture_screen()


# ── native: ocr_text ──────────────────────────────────────────────────

@pytest.fixture
def fake_tesseract(tmp_path, monkeypatch):
    """A fake tesseract binary emitting a fixed TSV (words: Hello world)."""
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "1\t1\t0\t0\t0\t0\t0\t0\t320\t240\t-1\t\n"
        "2\t1\t1\t0\t0\t0\t100\t60\t120\t60\t-1\t\n"
        "4\t1\t1\t1\t1\t0\t100\t60\t120\t60\t85\tHello world\n"
        "5\t1\t1\t1\t1\t1\t100\t60\t50\t60\t90\tHello\n"
        "5\t1\t1\t1\t1\t2\t155\t60\t65\t60\t80\tworld\n"
    )
    script = tmp_path / "tesseract"
    script.write_text("#!/bin/sh\nprintf '%s' '" + tsv.replace("'", "'\\''")
                      + "'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("NM_OCR_BINARY", str(script))
    return script


class TestOcrText:
    def test_transcribes_with_confidence(self, fake_tesseract):
        result = N.ocr_text(_scene_png())
        assert result["text"] == "Hello world"
        assert result["words"] == 2
        assert result["lines"] == 1
        assert result["mean_conf"] == pytest.approx(85.0)
        assert result["method"] == "tesseract-ocr-tsv"
        assert "untrusted" in result["untrusted_note"].lower()

    def test_psm_recorded(self, fake_tesseract):
        result = N.ocr_text(_scene_png(), psm=6)
        assert result["psm"] == 6

    def test_missing_binary_raises_clearly(self, monkeypatch):
        monkeypatch.delenv("NM_OCR_BINARY", raising=False)
        with patch.object(N, "_tesseract_binary", return_value=None):
            with pytest.raises(N.NativeUnavailable, match="tesseract"):
                N.ocr_text(_scene_png())


# ── native: resize_for_model ──────────────────────────────────────────

class TestResizeForModel:
    def test_downscales_to_budget(self):
        big = _png(Image.new("RGB", (3000, 2000), (5, 5, 5)))
        out = N.resize_for_model(big, 1568)
        assert max(Image.open(io.BytesIO(out)).size) <= 1568

    def test_fitting_image_unchanged(self):
        small = _scene_png()
        assert N.resize_for_model(small, 1568) == small


# ── native: screen_marks (Set-of-Mark) ────────────────────────────────

class TestScreenMarks:
    def _elements(self):
        return [
            {"bbox_1000": {"x": 300, "y": 250, "w": 260, "h": 120},
             "text": "Click me"},
            {"bbox_1000": {"x": 600, "y": 700, "w": 120, "h": 80},
             "text": ""},
        ]

    def test_marks_drawn_and_numbered(self):
        result = N.screen_marks(_scene_png(), elements=self._elements())
        assert result["count"] == 2
        assert result["png"].startswith(b"\x89PNG")
        ids = [e["id"] for e in result["elements"]]
        assert ids == [1, 2]
        first = result["elements"][0]
        assert first["center_1000"] == {"x": 300 + 130, "y": 250 + 60}
        assert result["method"] == "set-of-marks-native"

    def test_marks_at_model_resolution(self):
        big = _png(Image.new("RGB", (3000, 2000), (20, 20, 20)))
        result = N.screen_marks(big, elements=self._elements(),
                                max_dimension=1000)
        assert result["width"] == 1000
        # 0-1000 bboxes stay valid on the downscaled marks image
        img = Image.open(io.BytesIO(result["png"]))
        assert img.size == (1000, 667)

    def test_marked_image_differs_from_source(self):
        src = _scene_png()
        result = N.screen_marks(src, elements=self._elements())
        assert result["png"] != src  # marks were actually drawn

    def test_no_backends_raises_clearly(self, monkeypatch):
        monkeypatch.delenv("NM_OCR_BINARY", raising=False)
        with patch.object(N, "_tesseract_binary", return_value=None):
            N._BACKENDS["cv2"] = None
            try:
                with pytest.raises(N.NativeUnavailable,
                                   match="elements="):
                    N.screen_marks(_scene_png())
            finally:
                N._BACKENDS.pop("cv2", None)

    def test_merge_dedupes_contours_on_ocr(self):
        ocr = [{"bbox_1000": {"x": 100, "y": 100, "w": 200, "h": 50},
                "text": "hi", "source": "ocr"}]
        contour = [{"bbox_1000": {"x": 110, "y": 105, "w": 180, "h": 40},
                    "text": "", "source": "contour"}]
        merged = N._merge_marks(ocr, contour)
        assert len(merged) == 1 and merged[0]["source"] == "ocr"

    def test_merge_keeps_reading_order(self):
        els = [
            {"bbox_1000": {"x": 500, "y": 100, "w": 50, "h": 50},
             "text": "b", "source": "ocr"},
            {"bbox_1000": {"x": 100, "y": 100, "w": 50, "h": 50},
             "text": "a", "source": "ocr"},
        ]
        merged = N._merge_marks(els, [])
        assert [e["text"] for e in merged] == ["a", "b"]


class TestMarkPromptSnippet:
    def test_snippet_lists_elements(self):
        els = [{"id": 1, "text": "OK"}, {"id": 2, "text": ""}]
        snippet = N.mark_prompt_snippet(els)
        assert "[1] OK" in snippet
        assert "[2]" in snippet
        assert "NUMBER" in snippet
        assert "coordinates" in snippet.lower()


# ── native: analyze includes OCR ──────────────────────────────────────

class TestAnalyzeOcr:
    def test_analyze_has_ocr_section(self, fake_tesseract):
        report = N.analyze(_scene_png())
        assert report["ocr"]["available"] is True
        assert report["ocr"]["text"] == "Hello world"

    def test_analyze_reports_ocr_unavailable(self, monkeypatch):
        monkeypatch.delenv("NM_OCR_BINARY", raising=False)
        with patch.object(N, "_tesseract_binary", return_value=None):
            report = N.analyze(_scene_png())
        assert report["ocr"]["available"] is False
        assert "why" in report["ocr"]


# ── package exports ───────────────────────────────────────────────────

class TestPackageExports:
    def test_new_exports(self):
        import nomorals.vision as V
        assert V.capture_screen is SC.capture_screen
        assert V.see_bytes is see_bytes
        for name in ("see", "see_bytes", "capture_screen",
                     "capture_screenshot", "screenshot_from_file",
                     "native", "register"):
            assert name in V.__all__, name
