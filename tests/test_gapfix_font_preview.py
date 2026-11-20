"""Font preview service (census §3B.9) — storytold/font-preview-service
in Devon.

PATTERN ONLY: the upstream has no licence, so nothing of it is
vendored — the service shape (request/response preview, health,
cache by font+text+size+style, honest refusals, bounded render
queue) is written fresh on Devon's typography organ. Renders are
real: the vendored OFL Inter/Anton faces from the repo, through
the real shaping stack (Pillow/raqm) and the organ's bounded,
journaled render cache.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.typography import (
    FontPreviewService,
    PreviewBusy,
    PreviewError,
    PreviewRequest,
    PreviewTooLarge,
    TypographyService,
    UnsupportedPreviewFormat,
)
from nomorals.media.typography.fontlib import FontNotStocked

_VENDORED = Path(__file__).resolve().parent.parent / "nomorals" \
    / "media" / "typography" / "fonts"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _service(tmp_path, **kwargs):
    return TypographyService(roots=[str(_VENDORED)],
                             store_dir=tmp_path / "store", **kwargs)


def _preview(tmp_path, **kwargs):
    return FontPreviewService(_service(tmp_path), **kwargs)


def _registry():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


def _call(registry, name, **kwargs):
    outcome = registry.call(name, **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


# ── real renders ──────────────────────────────────────────────────

def test_real_render_end_to_end(tmp_path):
    svc = _preview(tmp_path)
    result = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    raw = result.path.read_bytes()
    assert raw[:8] == PNG_MAGIC
    assert result.width > 100 and result.height == 84  # 48 * 1.25 + padding
    assert result.png_bytes == len(raw) == result.path.stat().st_size
    assert len(result.sha256) == 64
    assert result.cache_hit is False
    assert result.family == "Inter"
    assert result.subfamily == "Regular"
    assert result.italic is False
    assert result.size == 48
    assert result.format == "png"
    assert result.missing_glyphs == ()
    assert result.glyphs_needed == 7  # H e l o D v n
    assert result.direction == "ltr"
    assert result.engine in ("raqm", "basic")
    assert result.style_note == ""


def test_cache_hit_returns_same_bytes(tmp_path):
    svc = _preview(tmp_path)
    first = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    second = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    assert second.cache_hit is True
    assert second.path == first.path
    assert second.sha256 == first.sha256


def test_size_changes_render(tmp_path):
    svc = _preview(tmp_path)
    small = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=24))
    big = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    assert small.sha256 != big.sha256
    assert small.height == 54  # 24 * 1.25 + padding
    assert big.height == 84
    assert small.width < big.width


def test_font_changes_render(tmp_path):
    svc = _preview(tmp_path)
    inter = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    anton = svc.preview(PreviewRequest(
        family="Anton", text="Hello Devon", size=48))
    assert anton.family == "Anton"
    assert anton.sha256 != inter.sha256


def test_regular_style_is_not_italic(tmp_path):
    # The organ's semantic selection prefers italic faces for the
    # "text" intent — a plain select() lands on Inter Italic.
    # The preview's requested style is authoritative, so the
    # default regular preview must render the regular face.
    svc = _preview(tmp_path)
    result = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48))
    assert result.italic is False
    assert result.subfamily == "Regular"


def test_italic_style_picks_italic_face(tmp_path):
    svc = _preview(tmp_path)
    result = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48, style="italic"))
    assert result.italic is True
    assert result.subfamily == "Italic"
    assert result.style_note == ""


def test_bold_style_renders_and_reports_honestly(tmp_path):
    svc = _preview(tmp_path)
    result = svc.preview(PreviewRequest(
        family="Inter", text="Hello Devon", size=48, style="bold"))
    assert result.family == "Inter"
    # Inter's vendored faces both parse at weight 400, so a bold
    # request still renders the 400 face — and says so in the
    # weight it reports, never a fabricated 700.
    assert result.weight == 400
    assert result.style_note == ""


def test_missing_glyphs_reported_not_boxed_silently(tmp_path):
    svc = _preview(tmp_path)
    result = svc.preview(PreviewRequest(
        family="Inter", text="Hello \u4f60", size=48))
    assert "\u4f60" in result.missing_glyphs
    assert result.glyphs_needed == 5  # H e l o + CJK char


# ── honest refusals ───────────────────────────────────────────────

def test_missing_font_refused_by_name(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(FontNotStocked) as exc:
        svc.preview(PreviewRequest(
            family="No Such Family XYZ", text="Hello", size=48))
    assert "No Such Family XYZ" in str(exc.value)


def test_unsupported_format_refused_by_name(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(UnsupportedPreviewFormat) as exc:
        svc.preview(PreviewRequest(
            family="Inter", text="Hello", size=48, format="webp"))
    assert "webp" in str(exc.value)


def test_oversized_size_refused(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(PreviewTooLarge) as exc:
        svc.preview(PreviewRequest(
            family="Inter", text="Hello", size=9999))
    assert "9999" in str(exc.value)


def test_oversized_text_refused(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(PreviewTooLarge) as exc:
        svc.preview(PreviewRequest(
            family="Inter", text="x" * 2001, size=48))
    assert "2001" in str(exc.value)


def test_empty_text_refused(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(PreviewError):
        svc.preview(PreviewRequest(family="Inter", text="   ", size=48))


def test_unknown_style_refused(tmp_path):
    svc = _preview(tmp_path)
    with pytest.raises(PreviewError) as exc:
        svc.preview(PreviewRequest(
            family="Inter", text="Hello", size=48, style="swash"))
    assert "swash" in str(exc.value)


def test_busy_queue_refused_with_retry_after(tmp_path):
    svc = _preview(tmp_path, max_concurrent=1, queue_timeout_s=0.05)
    assert svc._slots.acquire(blocking=False)
    try:
        with pytest.raises(PreviewBusy) as exc:
            svc.preview(PreviewRequest(
                family="Inter", text="Hello", size=48))
        assert exc.value.retry_after == 0.05
    finally:
        svc._slots.release()


# ── health ────────────────────────────────────────────────────────

def test_health_shape(tmp_path):
    svc = _preview(tmp_path)
    health = svc.health()
    assert health["status"] == "ok"
    assert health["fonts_indexed"] >= 50
    assert health["families"] >= 20
    assert "Inter" in svc.service.catalogue().families()
    assert health["cache"]["files"] >= 0
    assert health["engine"] in ("raqm", "basic")
    assert health["renderer"] == "font-preview-1"
    assert health["concurrency"]["max_concurrent"] == 4
    assert health["concurrency"]["queue_timeout_s"] == 2.0


# ── tool wiring: no new tool, preview rides craftfonts ───────────

def test_tool_count_stays_778(tmp_path):
    # No new brain tool: the preview rides the existing craftfonts
    # tool (its render action already reached the typography
    # organ), so the registry count did not move then; 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber.
    assert len(_registry()._tools) == 781


def test_tool_preview_action_renders_png(tmp_path):
    registry = _registry()
    out = tmp_path / "preview.png"
    result = _call(
        registry, "craftfonts", action="preview", value="Hello Devon",
        output=str(out), params={
            "family": "Inter", "size": 48, "home": str(tmp_path)})
    assert result["ok"] is True, result
    assert result["family"] == "Inter"
    assert result["width"] > 100 and result["height"] == 84
    assert result["cache_hit"] is False
    assert out.is_file()
    assert out.read_bytes()[:8] == PNG_MAGIC
    assert out.read_bytes() == Path(result["path"]).read_bytes()
    # A second identical call is a cache hit with the same bytes.
    again = _call(
        registry, "craftfonts", action="preview", value="Hello Devon",
        output=str(tmp_path / "preview2.png"), params={
            "family": "Inter", "size": 48, "home": str(tmp_path)})
    assert again["ok"] is True, again
    assert again["cache_hit"] is True
    assert again["png_sha256"] == result["png_sha256"]


def test_tool_preview_missing_font_is_ok_false(tmp_path):
    registry = _registry()
    result = _call(
        registry, "craftfonts", action="preview", value="Hello",
        params={"family": "No Such Family XYZ",
                "home": str(tmp_path)})
    assert result["ok"] is False
    assert "No Such Family XYZ" in result["error"]
    assert result["error_kind"] == "FontNotStocked"


def test_tool_preview_bad_format_is_ok_false(tmp_path):
    registry = _registry()
    result = _call(
        registry, "craftfonts", action="preview", value="Hello",
        params={"family": "Inter", "format": "svg",
                "home": str(tmp_path)})
    assert result["ok"] is False
    assert "svg" in result["error"]
    assert result["error_kind"] == "UnsupportedPreviewFormat"


def test_tool_health_action(tmp_path):
    registry = _registry()
    result = _call(registry, "craftfonts", action="health",
                   params={"home": str(tmp_path)})
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["fonts_indexed"] >= 50
