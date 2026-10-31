"""Tail-77 failure pins — nomorals/media_edit/studio.py.

The six guarded sites are a font-resolver ladder rung and five
Pillow ``close()`` teardowns.  ``find_font``'s contract is in its
docstring — resolve by path, then fuzzy name discovery, then the
bundled DejaVu files, then ``load_default()``; a font file that
exists but will not load falls through to the next rung.  The
``close()`` sites sit in ``finally`` blocks or after the artifact
is already saved: the composite / collage / render / compare
results are fully computed before the close is attempted, and no
caller branches on a close outcome.  These pins prove a failing
close (or an unloadable font file) never takes the result down
with it.

Probed live first (nm-tail77-verify/probes.py) with real Pillow.
"""
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from nomorals.media_edit import studio


def _boom():
    raise RuntimeError("close failed")


def _src_png(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (12, 12), (123, 40, 200)).save(src)
    return src


def test_112_unloadable_font_path_falls_through_the_ladder(tmp_path):
    corrupt = tmp_path / "corrupt.ttf"
    corrupt.write_bytes(b"this is not a font at all")
    # control: the raw load the guard protects really does fail
    with pytest.raises(OSError):
        ImageFont.truetype(str(corrupt), 20)
    # the guard lets resolution fall through to a working font
    font = studio.find_font(str(corrupt), 20)
    assert font is not None
    # and the terminal rung exists for a name nothing matches
    assert studio.find_font("no-such-font-name-zzz", 20) is not None


def test_973_layer_close_failure_keeps_the_composite(monkeypatch):
    base = Image.new("RGB", (40, 40), (10, 10, 10))
    content = Image.new("RGBA", (10, 10), (200, 0, 0, 255))
    content.close = _boom
    monkeypatch.setattr(
        studio, "render_shape_layer",
        lambda size, spec: (content, (0, 0)))
    out = studio.composite_layers(base, [{"type": "shape"}])
    # the layer was composited BEFORE the finally-block close ran
    assert out.getpixel((5, 5)) == (200, 0, 0)
    assert out.size == (40, 40)


def test_1052_collage_extra_close_failure_keeps_the_collage(monkeypatch):
    extra = Image.new("RGB", (10, 10), (0, 200, 0))
    extra.close = _boom
    monkeypatch.setattr(studio, "load_image", lambda p: extra)
    out = studio.op_collage(Image.new("RGB", (10, 10), (0, 0, 200)),
                            images=["whatever.png"], template="strip")
    # strip = base + gap(8) + extra, all before the finally close
    assert out.size == (28, 10)


def test_1993_source_close_failure_keeps_the_render(tmp_path,
                                                    monkeypatch):
    src = _src_png(tmp_path)
    real_load = studio.load_image

    def load_and_patch(path):
        img = real_load(path)
        img.close = _boom  # only the SOURCE image's close fails
        return img

    monkeypatch.setattr(studio, "load_image", load_and_patch)
    proj = studio.EditStudio(source=src)
    res = proj._render_image(out_dir=tmp_path / "out")
    assert res["kind"] == "image"
    assert Path(res["output"]).exists()
    assert res["bytes"] > 0


def test_2001_output_close_failure_keeps_the_render(tmp_path,
                                                    monkeypatch):
    src = _src_png(tmp_path)
    real_load = studio.load_image
    patched = []

    def load_and_wrap(path):
        img = real_load(path)
        real_copy = img.copy

        def copy_with_bad_close():
            out = real_copy()
            out.close = _boom  # only the OUTPUT image's close fails
            patched.append(out)
            return out

        img.copy = copy_with_bad_close
        return img

    monkeypatch.setattr(studio, "load_image", load_and_wrap)
    proj = studio.EditStudio(source=src)
    res = proj._render_image(out_dir=tmp_path / "out")
    assert res["kind"] == "image"
    assert Path(res["output"]).exists()
    assert res["bytes"] > 0
    # proof the failing close really was on the render path
    assert patched and patched[-1].close is _boom


def test_2125_compare_close_failure_keeps_the_export(tmp_path,
                                                     monkeypatch):
    src = _src_png(tmp_path)
    real_cr = studio.compare_render

    def cr_with_bad_close(before, after, **kw):
        comp = real_cr(before, after, **kw)
        comp.close = _boom
        return comp

    monkeypatch.setattr(studio, "compare_render", cr_with_bad_close)
    proj = studio.EditStudio(source=src)
    res = proj.compare(out_dir=tmp_path / "cmp")
    assert res["mode"] == "side-by-side"
    assert Path(res["output"]).exists()
