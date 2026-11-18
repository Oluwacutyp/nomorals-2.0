"""52-websites #14 (Remove.bg): the composed background-removal ladder.

The organ composes the rembg model rung with the classical mask
ladder; every cutout says which rung served it. These tests prove
the composition, not the model: rembg/cv2/SAM are absent on this
box, so the live ladder here is flood → box, and the model +
GrabCut + SAM rungs are exercised through the injectable seams —
the same seams the organ offers production callers.

Corpus floors are the manifest's own: the cutout mask for a
fixture must hold the fixture's ``auto`` floor where the ladder
can run it (disc/block/gradient/speckled), and the ring/twin
fixtures prove the border-colour guard — the centre of a ring is
background, and a flood from it must fail the rung rather than
serve the hole as the subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nomorals.media.imggen import mask_corpus
from nomorals.media.imggen.bgremove import (
    bgremove_status,
    cutout_from_mask,
    remove_background,
    remove_background_file,
)
from nomorals.media.imggen.document import ImageDocument
from nomorals.media.imggen.masks import ImgGenError, MaskResult
from nomorals.tools import bgremove as bgremove_tool

MANIFEST = json.loads(
    (Path(__file__).parent / "golden" / "mask_corpus.json")
    .read_text(encoding="utf-8"))
FLOORS = {f["name"]: f["floors"] for f in MANIFEST["fixtures"]}


class _Registry:
    def __init__(self, context):
        self.context = context
        self.handlers = {}

    def register(self, name, **meta):
        def deco(fn):
            self.handlers[name] = fn
            return fn
        return deco


class _Ctx:
    def __init__(self, home):
        self.settings = _Settings(home)


class _Settings:
    def __init__(self, home):
        self.home = str(home)


def _fixture(name):
    return mask_corpus.build_fixture(name)


def _iou(mask, gt):
    return mask_corpus.iou_score(mask, gt)


def _failing_model_fn(image, *, model=""):
    raise ImgGenError(
        "rembg rung unavailable: background removal needs rembg: "
        "pip install rembg (u2netp model is only ~4.7 MB)")


def _alpha_model_fn(alpha):
    """A fake model rung returning a fixed alpha channel."""
    def run(image, *, model=""):
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(
            (np.asarray(alpha) * 255).astype(np.uint8), "L"))
        return rgba
    return run


def _mask_fn(results):
    """A fake mask ladder: tier -> MaskResult or raised error."""
    def run(image, *, point=None, box=None, tier="auto",
            feather_radius=8, **_kw):
        entry = results[tier]
        if isinstance(entry, Exception):
            raise entry
        return entry
    return run


def _mask_from_gt(gt, tier, detail="fake mask"):
    arr = (np.asarray(gt, dtype=bool) * 255).astype(np.uint8)
    return MaskResult(mask=Image.fromarray(arr, "L"), tier=tier,
                      detail=detail)


# ---------------------------------------------------------------------------
# The live ladder on this box (rembg/cv2/SAM absent: flood → box)
# ---------------------------------------------------------------------------

def test_disc_auto_lands_on_flood_and_holds_its_floor():
    image, gt, _prompt = _fixture("disc_on_field")
    res = remove_background(image, model_fn=_failing_model_fn)
    assert res.rung == "flood"
    assert _iou(res.mask, gt) >= FLOORS["disc_on_field"]["auto"]
    assert res.image.mode == "RGBA"
    assert res.image.split()[3].getpixel((96, 96)) > 200
    assert res.image.split()[3].getpixel((0, 0)) < 50
    assert 0.2 < res.coverage < 0.35
    assert res.detail.startswith("numpy region-grow")


def test_block_gradient_bg_holds_its_floor():
    image, gt, _prompt = _fixture("block_gradient_bg")
    res = remove_background(image, model_fn=_failing_model_fn)
    assert res.rung == "flood"
    assert _iou(res.mask, gt) >= FLOORS["block_gradient_bg"]["auto"]


def test_speckled_disc_holds_its_flood_floor():
    image, gt, _prompt = _fixture("speckled_disc")
    res = remove_background(image, model_fn=_failing_model_fn)
    assert res.rung == "flood"
    assert _iou(res.mask, gt) >= FLOORS["speckled_disc"]["flood"]


def test_ring_centre_does_not_become_the_subject():
    image, gt, _prompt = _fixture("thin_ring")
    res = remove_background(image, model_fn=_failing_model_fn)
    # The centre pixel is the hole — background. Flood from it
    # must fail the border-colour guard, never serve the hole.
    assert res.rung == "box+feather"
    flood = [a for a in res.attempts if a["rung"] == "flood"][0]
    assert flood["ok"] is False
    assert "matches the border" in flood["reason"]
    assert "last resort" in " ".join(res.notes)


def test_twin_discs_centre_does_not_become_the_subject():
    image, gt, _prompt = _fixture("twin_discs")
    res = remove_background(image, model_fn=_failing_model_fn)
    assert res.rung == "box+feather"
    flood = [a for a in res.attempts if a["rung"] == "flood"][0]
    assert flood["ok"] is False
    assert "matches the border" in flood["reason"]


def test_flat_image_reports_full_coverage():
    image = Image.new("RGB", (64, 64), (200, 30, 30))
    res = remove_background(image, model_fn=_failing_model_fn)
    assert res.rung == "flood"
    assert res.coverage > 0.98
    assert "fill the frame" in " ".join(res.notes)


# ---------------------------------------------------------------------------
# The ladder order, through the injectable seams
# ---------------------------------------------------------------------------

def test_rembg_rung_first_when_provisioned():
    image, gt, _prompt = _fixture("disc_on_field")
    called = []

    def guard_mask_fn(*_a, **_kw):
        called.append(True)
        raise AssertionError("mask ladder must not run after rembg")

    res = remove_background(
        image, model_fn=_alpha_model_fn(gt), mask_fn=guard_mask_fn)
    assert res.rung == "rembg"
    assert _iou(res.mask, gt) == pytest.approx(1.0)
    assert called == []


def test_fallthrough_order_sam_grabcut_flood():
    image, gt, _prompt = _fixture("disc_on_field")
    fake = _mask_fn({
        "sam": ImgGenError("no SAM checkpoint on disk"),
        "grabcut": _mask_from_gt(gt, "grabcut", "fake OpenCV grabcut"),
        "flood": _mask_from_gt(gt, "flood"),
        "box": _mask_from_gt(gt, "box+feather"),
    })
    res = remove_background(image, model_fn=_failing_model_fn,
                            mask_fn=fake)
    assert res.rung == "grabcut"
    assert res.detail == "fake OpenCV grabcut"
    assert [a["rung"] for a in res.attempts] == [
        "rembg", "sam", "grabcut"]
    assert res.attempts[0]["ok"] is False
    assert res.attempts[1]["ok"] is False
    assert res.attempts[2]["ok"] is True


def test_grabcut_rect_serves_both_discs():
    # Background removal keeps every foreground object: on the
    # twin-discs scene the rect rung's mask is the union, not the
    # point-prompted left disc alone.
    image, gt, _prompt = _fixture("twin_discs")
    union = np.zeros_like(gt)
    yy, xx = np.mgrid[0:192, 0:192]
    union |= (xx - 58) ** 2 + (yy - 96) ** 2 <= 34 ** 2
    union |= (xx - 134) ** 2 + (yy - 96) ** 2 <= 34 ** 2
    fake = _mask_fn({
        "sam": ImgGenError("no SAM checkpoint on disk"),
        "grabcut": _mask_from_gt(union, "grabcut"),
        "flood": _mask_from_gt(gt, "flood"),
        "box": _mask_from_gt(union, "box+feather"),
    })
    res = remove_background(image, model_fn=_failing_model_fn,
                            mask_fn=fake)
    assert res.rung == "grabcut"
    assert _iou(res.mask, union) == pytest.approx(1.0)


def test_all_rungs_down_raises_naming_every_rung():
    image, _gt, _prompt = _fixture("disc_on_field")
    fake = _mask_fn({
        "sam": ImgGenError("no SAM checkpoint on disk"),
        "grabcut": ImgGenError("OpenCV not installed"),
        "flood": ImgGenError("numpy is not installed"),
        "box": ImgGenError("Pillow is not installed"),
    })
    with pytest.raises(ImgGenError) as caught:
        remove_background(image, model_fn=_failing_model_fn,
                          mask_fn=fake)
    msg = str(caught.value)
    for rung in ("rembg", "sam", "grabcut", "flood", "box"):
        assert rung in msg, msg


def test_forced_rembg_names_the_rung_when_missing():
    image, _gt, _prompt = _fixture("disc_on_field")
    with pytest.raises(ImgGenError) as caught:
        remove_background(image, tier="rembg",
                          model_fn=_failing_model_fn)
    assert "rembg" in str(caught.value)
    assert "pip install rembg" in str(caught.value)


def test_forced_box_is_a_feathered_rect():
    image, gt, _prompt = _fixture("disc_on_field")
    res = remove_background(image, tier="box",
                            model_fn=_failing_model_fn)
    assert res.rung == "box+feather"
    # The bgremove box is the whole-frame inset rect (margin
    # 0.06), NOT the manifest's tight fixture box — its IoU
    # ceiling is disc-area / rect-area ≈ 0.35, not the
    # manifest's 0.55 tight-box floor. Assert the rect shape
    # and the inset geometry instead.
    assert res.mask.getpixel((96, 96)) > 127
    assert res.mask.getpixel((2, 2)) < 127
    assert _iou(res.mask, gt) >= 0.30
    assert "last resort" in " ".join(res.notes)


def test_unknown_tier_is_refused():
    image, _gt, _prompt = _fixture("disc_on_field")
    with pytest.raises(ImgGenError, match="unknown tier"):
        remove_background(image, tier="magic",
                          model_fn=_failing_model_fn)


def test_non_image_is_refused():
    with pytest.raises(ImgGenError, match="takes a PIL image"):
        remove_background("not-an-image")


def test_cutout_from_mask_applies_alpha():
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((3, 3), 255)
    out = cutout_from_mask(image, mask)
    assert out.mode == "RGBA"
    assert out.getpixel((3, 3))[3] == 255
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((3, 3))[:3] == (10, 20, 30)


# ---------------------------------------------------------------------------
# Files, status, the tool surface
# ---------------------------------------------------------------------------

def test_remove_background_file_writes_png(tmp_path):
    image, _gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.jpg"
    image.save(src)
    record = remove_background_file(src, tmp_path / "cut.jpg")
    assert record["output"].endswith(".png")
    out = Image.open(record["output"])
    assert out.mode == "RGBA"
    assert record["rung"] == "flood"
    assert 0.2 < record["coverage"] < 0.35
    assert record["bytes"] > 0
    assert Path(record["output"]).exists()


def test_remove_background_file_missing_source(tmp_path):
    with pytest.raises(ImgGenError, match="no such image"):
        remove_background_file(tmp_path / "nope.png",
                               tmp_path / "out.png")


def test_bgremove_status_shape():
    status = bgremove_status()
    assert status["ladder"] == ["rembg", "sam", "grabcut", "flood",
                                "box"]
    assert set(status["masks"]) == {"sam", "grabcut", "flood",
                                    "box", "file"}
    assert status["rembg"]["usable"] is False
    assert "pip install rembg" in status["rembg"]["reason"]


def _tool(tmp_path):
    reg = _Registry(_Ctx(tmp_path))
    bgremove_tool.register(reg)
    return reg.handlers["remove_background"]


def test_tool_remove_status_models(tmp_path):
    image, _gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    tool = _tool(tmp_path)
    status = tool(action="status")
    assert status["ok"] is True
    assert status["ladder"][0] == "rembg"
    models = tool(action="models")
    assert models["ok"] is True
    assert any(m["name"] == "u2netp"
               for m in models["rembg_models"])
    assert "grabcut" in models["classical_rungs"]
    done = tool(action="remove", path=str(src))
    assert done["ok"] is True
    assert done["rung"] == "flood"
    assert 0.2 < done["coverage"] < 0.35
    assert Path(done["output"]).exists()
    assert Image.open(done["output"]).mode == "RGBA"
    # The cutout lands as a Studio document beside the PNG, so
    # Studio work can continue from it.
    doc_path = Path(done["document"])
    assert doc_path.exists()
    assert doc_path.suffix == ".json"
    doc = ImageDocument.load(doc_path)
    assert [o.op for o in doc.ops] == ["remove_background"]
    assert doc.render().image.mode == "RGBA"


def test_tool_default_out_lands_under_home(tmp_path):
    image, _gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    tool = _tool(tmp_path)
    done = tool(action="remove", path=str(src))
    out = Path(done["output"])
    assert out.parent == tmp_path / "generations"
    assert out.name.startswith("bgcut-")
    assert out.suffix == ".png"
    assert Path(done["document"]).parent == tmp_path / "generations"
    assert Path(done["document"]).name == out.name.replace(
        ".png", ".json")


def test_tool_refuses_plainly(tmp_path):
    tool = _tool(tmp_path)
    with pytest.raises(ValueError, match="needs a photo path"):
        tool(action="remove", path="")
    with pytest.raises(ValueError, match="no such photo"):
        tool(action="remove", path=str(tmp_path / "nope.png"))
    with pytest.raises(ValueError, match="unknown remove_background"):
        tool(action="explode")


# ---------------------------------------------------------------------------
# The image-document op: replay, undo, save/load
# ---------------------------------------------------------------------------

def test_document_op_replay_undo(tmp_path):
    image, gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    doc = ImageDocument(str(src), title="cutout")
    doc.remove_background()
    rendered = doc.render()
    assert rendered.image.mode == "RGBA"
    assert rendered.provenance[0]["op"] == "remove_background"
    assert rendered.provenance[0]["rung"].startswith("bgremove (")
    assert rendered.provenance[0]["rung"].endswith("flood)")
    assert _iou(rendered.image.split()[3], gt) >= \
        FLOORS["disc_on_field"]["auto"]
    cut = rendered.image.tobytes()
    doc.undo()
    plain = doc.render()
    assert plain.image.tobytes() == image.tobytes()
    assert plain.provenance == []
    doc.redo()
    again = doc.render()
    assert again.image.tobytes() == cut
    saved = tmp_path / "doc.json"
    doc.save(saved)
    loaded = ImageDocument.load(saved)
    assert loaded.content_hash() == doc.content_hash()
    assert loaded.render().image.tobytes() == cut


def test_document_op_unknown_param_raises(tmp_path):
    image, _gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    doc = ImageDocument(str(src))
    doc.push("remove_background", bogus=1)
    with pytest.raises(ImgGenError, match="unknown params"):
        doc.render()


def test_document_op_forced_tier_recorded(tmp_path):
    image, _gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    doc = ImageDocument(str(src))
    doc.remove_background(tier="box")
    rendered = doc.render()
    assert rendered.provenance[0]["rung"] == "bgremove (box+feather)"


# ---------------------------------------------------------------------------
# The Studio flow: the op is reachable from the Studio, and Studio
# work continues from the cutout
# ---------------------------------------------------------------------------

def test_studio_remove_background_renders_through_the_document(tmp_path):
    from nomorals.media.imggen.studio import Studio

    image, gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    out = tmp_path / "cut.png"
    paths = Studio().remove_background(
        str(src), save_to=str(out))
    assert paths == [str(out)]
    assert Image.open(out).mode == "RGBA"
    sidecar = Path(str(out) + ".txt")
    assert "bgremove (flood)" in sidecar.read_text(encoding="utf-8")
    assert _iou(Image.open(out).split()[3], gt) >= \
        FLOORS["disc_on_field"]["auto"]


def test_studio_remove_background_document_continues(tmp_path):
    from nomorals.media.imggen.studio import Studio

    image, gt, _prompt = _fixture("disc_on_field")
    src = tmp_path / "photo.png"
    image.save(src)
    studio = Studio()
    doc = studio.remove_background_document(str(src))
    assert [o.op for o in doc.ops] == ["remove_background"]
    # Studio work continues from the cutout: chain another op,
    # undo it, save the project, re-open it, render later.
    doc.blur(radius=1.0)
    assert [o.op for o in doc.ops] == ["remove_background", "blur"]
    doc.undo()
    assert [o.op for o in doc.ops] == ["remove_background"]
    saved = tmp_path / "cutout.json"
    doc.save(saved)
    reopened = ImageDocument.load(saved)
    rendered = reopened.render()
    assert rendered.image.mode == "RGBA"
    assert rendered.provenance[0]["op"] == "remove_background"
    assert _iou(rendered.image.split()[3], gt) >= \
        FLOORS["disc_on_field"]["auto"]


def test_cutout_coverage_reads_back_off_a_png(tmp_path):
    image, gt, _prompt = _fixture("disc_on_field")
    res = remove_background(image, model_fn=_failing_model_fn)
    out = tmp_path / "cut.png"
    res.image.save(out)
    from nomorals.media.imggen.bgremove import cutout_coverage

    with Image.open(out) as saved:
        assert cutout_coverage(saved) == pytest.approx(
            res.coverage, abs=0.01)
