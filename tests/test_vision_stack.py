"""Tests for the vision utility stack (build-map #23).

All offline: heavy libraries (rembg, realesrgan, insightface, gfpgan)
are mocked or asserted absent — no network, no GPU, no model downloads.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture()
def img():
    return Image.new("RGB", (64, 64), (200, 30, 30))


# ── lazy loading: no heavy imports at module load ─────────────────────────

HEAVY = ("rembg", "realesrgan", "basicsr", "insightface", "gfpgan")


def test_no_heavy_imports_at_module_load():
    import nomorals.media_edit.segment as segment
    import nomorals.media_edit.upscale as upscale
    import nomorals.media_edit.faceswap as faceswap
    assert segment and upscale and faceswap
    for mod in HEAVY:
        assert mod not in sys.modules, f"{mod} imported at module load"


# ── segment.py ────────────────────────────────────────────────────────────

def test_segment_unknown_model(img):
    from nomorals.media_edit.segment import remove_background, SegmentationError
    with pytest.raises(SegmentationError, match="unknown segmentation model"):
        remove_background(img, model="nope")


def test_segment_missing_lib_clear_error(img):
    from nomorals.media_edit import segment
    from nomorals.media_edit.segment import (
        remove_background, SegmentationError)
    segment._SESSIONS.clear()
    with patch.dict(sys.modules, {"rembg": None}):
        # block the real import: make the import machinery raise
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "rembg" or name.startswith("rembg."):
                raise ImportError("No module named 'rembg'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SegmentationError,
                               match="pip install rembg"):
                remove_background(img, model="u2netp")


def test_segment_default_model_termux():
    from nomorals.media_edit import segment
    with patch.object(segment, "get_profile_kind", return_value="termux"):
        assert segment.default_model() == "u2netp"
    with patch.object(segment, "get_profile_kind", return_value="laptop"):
        assert segment.default_model() == "birefnet-general"
    with patch.object(segment, "get_profile_kind",
                      return_value="workstation"):
        assert segment.default_model() == "birefnet-general"


def test_segment_list_models():
    from nomorals.media_edit.segment import list_models
    models = list_models()
    assert any(m["name"] == "u2netp" for m in models)
    assert any(m["name"] == "birefnet-general" for m in models)


def test_segment_remove_background_mocked(img):
    """With a fake rembg, remove_background returns RGBA."""
    from nomorals.media_edit import segment
    from nomorals.media_edit.segment import remove_background
    segment._SESSIONS.clear()
    # _require_rembg returns (new_session, remove); _session caches both
    with patch.object(segment, "_require_rembg",
                      return_value=(lambda m: object(),
                                    lambda pil, session=None:
                                    pil.convert("RGBA"))):
        out = remove_background(img, model="u2netp")
    assert out.mode == "RGBA"
    assert out.size == img.size
    segment._SESSIONS.clear()


# ── upscale.py ────────────────────────────────────────────────────────────

def test_upscale_auto_tile():
    from nomorals.media_edit.upscale import _auto_tile
    assert _auto_tile(100, 100, 0) == 0      # small → whole image
    assert _auto_tile(2048, 2048, 0) == 512  # large → tiled
    assert _auto_tile(2048, 2048, 256) == 256  # explicit wins


def test_upscale_unknown_model(img):
    from nomorals.media_edit.upscale import upscale, UpscaleError
    with pytest.raises(UpscaleError, match="unknown SR model"):
        upscale(img, model="nope")


def test_upscale_scale_mismatch(img):
    from nomorals.media_edit.upscale import upscale, UpscaleError
    with pytest.raises(UpscaleError, match="x4, not x2"):
        upscale(img, scale=2.0, model="RealESRGAN_x4plus")


def test_upscale_termux_blocked(img):
    from nomorals.media_edit import upscale as up
    from nomorals.media_edit.upscale import upscale, UpscaleError
    with patch.object(up, "get_profile_kind", return_value="termux"):
        with pytest.raises(UpscaleError, match="laptop/workstation"):
            upscale(img, model="RealESRGAN_x4plus")


def test_upscale_missing_lib_clear_error(img):
    from nomorals.media_edit import upscale as up
    from nomorals.media_edit.upscale import upscale, UpscaleError
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in ("realesrgan", "basicsr") or name.startswith(
                ("realesrgan.", "basicsr.")):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch.object(up, "get_profile_kind", return_value="laptop"):
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(UpscaleError,
                               match="pip install realesrgan"):
                upscale(img, model="RealESRGAN_x4plus")


def test_supir_workstation_gate(img):
    from nomorals.media_edit import upscale as up
    from nomorals.media_edit.upscale import upscale, UpscaleError
    with patch.object(up, "get_profile_kind", return_value="laptop"):
        with pytest.raises(UpscaleError, match="workstation profile"):
            upscale(img, model="supir")


def test_supir_missing_repo_clear_error(img):
    from nomorals.media_edit import upscale as up
    from nomorals.media_edit.upscale import upscale, UpscaleError
    with patch.object(up, "get_profile_kind", return_value="workstation"):
        with patch.dict("os.environ", {"SUPIR_REPO": ""}, clear=False):
            import os
            os.environ.pop("SUPIR_REPO", None)
            with pytest.raises(UpscaleError, match="git clone.*SUPIR"):
                upscale(img, model="supir")


# ── faceswap.py ───────────────────────────────────────────────────────────

def test_faceswap_termux_blocked(img):
    from nomorals.media_edit import faceswap as fs
    from nomorals.media_edit.faceswap import swap_face, FaceSwapError
    with patch.object(fs, "get_profile_kind", return_value="termux"):
        with pytest.raises(FaceSwapError, match="laptop/workstation"):
            swap_face(img, img)


def test_faceswap_no_face_clear_error(img):
    from nomorals.media_edit import faceswap as fs
    from nomorals.media_edit.faceswap import swap_face, FaceSwapError
    with patch.object(fs, "get_profile_kind", return_value="laptop"):
        with patch.object(fs, "_analyzer") as analyzer:
            analyzer.return_value.get.return_value = []
            with pytest.raises(FaceSwapError, match="no face detected"):
                swap_face(img, img)


def test_faceswap_target_index_out_of_range(img):
    from nomorals.media_edit import faceswap as fs
    from nomorals.media_edit.faceswap import swap_face, FaceSwapError

    class FakeFace:
        bbox = (0, 0, 10, 10)

    with patch.object(fs, "get_profile_kind", return_value="laptop"):
        with patch.object(fs, "_faces", return_value=[FakeFace()]):
            with patch.object(fs, "_swapper"):
                with pytest.raises(FaceSwapError, match="out of range"):
                    swap_face(img, img, target_index=5)


def test_faceswap_missing_insightface(img):
    from nomorals.media_edit import faceswap as fs
    from nomorals.media_edit.faceswap import swap_face, FaceSwapError
    fs._ANALYZER = None
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "insightface" or name.startswith("insightface."):
            raise ImportError("No module named 'insightface'")
        return real_import(name, *args, **kwargs)

    with patch.object(fs, "get_profile_kind", return_value="laptop"):
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(FaceSwapError,
                               match="pip install insightface"):
                swap_face(img, img, restore=False)
    fs._ANALYZER = None


# ── model cache ───────────────────────────────────────────────────────────

def test_model_cache_path_resolution(tmp_path, monkeypatch):
    from nomorals.media_edit import _model_cache as mc
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path))
    assert mc.cache_dir() == tmp_path
    assert not mc.cached("nope.pth")
    # existing non-empty file → returned immediately, no download
    f = tmp_path / "m.pth"
    f.write_bytes(b"weights")
    assert mc.model_path("m.pth") == f


def test_model_cache_missing_no_url():
    from nomorals.media_edit._model_cache import (
        model_path, ModelCacheError, cache_dir)
    import os
    os.environ.pop("NM_MODELS_DIR", None)
    with pytest.raises(ModelCacheError, match="no download URL"):
        # unique name so it can't exist in the real cache
        model_path("definitely-not-a-real-model-xyz123.pth")


# ── op registration ───────────────────────────────────────────────────────

def test_ops_registered():
    import nomorals.media_edit.generate  # noqa: F401 — runs _register()
    from nomorals.media_edit.images import OP_ALLOWLIST, _OP_FUNCS
    for name in ("bg_remove_v2", "upscale_sr", "faceswap"):
        assert name in OP_ALLOWLIST, f"{name} not in allowlist"
        assert name in _OP_FUNCS, f"{name} has no function"


def test_legacy_bg_remove_still_registered():
    import nomorals.media_edit.generate  # noqa: F401
    from nomorals.media_edit.images import OP_ALLOWLIST
    assert "bg_remove" in OP_ALLOWLIST  # backward compat


# ── NL patterns ───────────────────────────────────────────────────────────

def test_nl_bgremove_matches():
    from nomorals.agents.coremind import _image_intent
    for text in ("remove the background", "remove background",
                 "remove the background from this image"):
        intent = _image_intent(text)
        assert intent is not None and intent.kind == "vision_bgremove", text


def test_nl_upscale_matches():
    from nomorals.agents.coremind import _image_intent
    for text in ("upscale this image", "upscale this", "upscale"):
        intent = _image_intent(text)
        assert intent is not None and intent.kind == "vision_upscale", text


def test_nl_faceswap_matches():
    from nomorals.agents.coremind import _image_intent
    for text in ("swap faces", "swap face", "swap faces with this image"):
        intent = _image_intent(text)
        assert intent is not None and intent.kind == "vision_faceswap", text


def test_nl_vision_no_misfire():
    from nomorals.agents.coremind import _image_intent
    for text in ("what's the background of this story",
                 "upscale the business",
                 "let's swap faces at the party",
                 "remove the background music"):
        intent = _image_intent(text)
        assert intent is None or intent.kind not in (
            "vision_bgremove", "vision_upscale", "vision_faceswap"), text
