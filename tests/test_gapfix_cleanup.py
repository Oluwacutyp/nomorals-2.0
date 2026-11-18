"""Tests for the composed remove-object loop (52-websites #15).

Fixtures are deterministic PIL/numpy scenes built from their ground
truth, in the mask-corpus idiom: every scene is drawn *from* its
ground-truth mask, so the truth is exact by construction.

Proof contract (from the report), executed through the image systems
the op lives in — never a standalone executor:
- fixture images with planted objects → removal region filled,
  outside-region pixels byte-identical, via the
  :class:`~nomorals.media.imggen.document.ImageDocument` remove-object
  op and the :class:`~nomorals.media.imggen.studio.Studio`;
- description → mask resolution cases (colour, OCR text, faces,
  spatial filter, explicit mask honoured);
- the op is an image-document op: undo pops it, replay is
  deterministic, and the Studio can continue from it with a second
  op;
- large-mask fills are labelled synthesised, never a silent retouch;
- a description no rung can locate refuses honestly.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen.cleanup import (
    SYNTHESISED_FRACTION,
    locate_description,
    mask_fraction,
    parse_description,
    remove_object_document,
)
from nomorals.media.imggen.document import ImageDocument
from nomorals.media.imggen.masks import MaskRef
from nomorals.media.imggen.studio import Studio

SIZE = (192, 192)
BG = (120, 170, 220)  # sky blue


def _scene_disc():
    """Blue field, solid red disc at centre (r=40)."""
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    d.ellipse([56, 56, 136, 136], fill=(255, 0, 0))
    gt = Image.new("L", SIZE, 0)
    ImageDraw.Draw(gt).ellipse([56, 56, 136, 136], fill=255)
    return img, gt


def _scene_two_discs():
    """Blue field, red disc left + red disc right."""
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    d.ellipse([24, 72, 84, 132], fill=(255, 0, 0))
    d.ellipse([108, 72, 168, 132], fill=(255, 0, 0))
    left = Image.new("L", SIZE, 0)
    ImageDraw.Draw(left).ellipse([24, 72, 84, 132], fill=255)
    right = Image.new("L", SIZE, 0)
    ImageDraw.Draw(right).ellipse([108, 72, 168, 132], fill=255)
    return img, left, right


def _scene_text():
    """White field, black SALE text — letter pixels are the truth."""
    img = Image.new("RGB", SIZE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((48, 80), "SALE", fill=(0, 0, 0))
    gt = Image.new("L", SIZE, 0)
    ImageDraw.Draw(gt).text((48, 80), "SALE", fill=255)
    return img, gt


def _scene_face():
    """Blue field, solid skin-tone disc as a stand-in face."""
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    d.ellipse([72, 56, 120, 104], fill=(220, 180, 140))
    gt = Image.new("L", SIZE, 0)
    ImageDraw.Draw(gt).ellipse([72, 56, 120, 104], fill=255)
    return img, gt


def _iou(a, b) -> float:
    aa = np.asarray(a.convert("L")) > 127
    bb = np.asarray(b.convert("L")) > 127
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return inter / union if union else 1.0


def _outside_mask_identical(before, after, mask) -> bool:
    """Pixels where the mask alpha is exactly 0 are untouched.

    This is the guarantee the inpaint op makes structurally when it
    carries ``preserve_outside`` — the feather band (alpha > 0) is
    part of the removal region, so ground-truth-based checks would
    fail there.
    """
    a = np.asarray(before.convert("RGB"))
    b = np.asarray(after.convert("RGB"))
    keep = np.asarray(mask.convert("L")) == 0
    return bool((a[keep] == b[keep]).all())


def _outside_identical(before, after, gt) -> bool:
    """Pixels outside a hard ground-truth mask are untouched.

    Only valid when the removal mask equals the ground truth
    (explicit masks) — description masks are feathered, so use
    :func:`_outside_mask_identical` for those.
    """
    a = np.asarray(before.convert("RGB"))
    b = np.asarray(after.convert("RGB"))
    keep = ~(np.asarray(gt.convert("L")) > 127)
    return bool((a[keep] == b[keep]).all())


def _dilated(gt, px: int):
    """Ground truth dilated by ``px`` — covers the feather band."""
    from PIL import ImageFilter

    return gt.convert("L").filter(ImageFilter.MaxFilter(px * 2 + 1))


def _eroded(gt, px: int):
    """Ground truth eroded by ``px`` — the fill's deep interior,
    away from the antialiased edge band."""
    from PIL import ImageFilter

    return gt.convert("L").filter(ImageFilter.MinFilter(px * 2 + 1))


def _run_doc(tmp_path, img, description=None, mask=None,
             locator=None, method="auto"):
    """Run the remove-object document end to end.

    Returns ``(doc, rendered, resolved_mask)`` — the resolved mask
    is the one the op itself resolved from the doc's own recipe.
    """
    src = tmp_path / "scene.png"
    img.save(src)
    doc = remove_object_document(
        str(src), description=description, mask=mask,
        method=method, locator=locator)
    rendered = doc.render()
    ref = MaskRef.from_dict(doc.ops[0].params["mask"])
    resolved = ref.resolve(img.convert("RGB"))
    return doc, rendered, resolved


def _ocr_stub_for(box1000, text):
    def _ocr(data: bytes):
        return {"lines": [{"bbox_1000": box1000, "text": text,
                           "conf": 90.0}],
                "words": 1, "blocks": 1, "method": "stub"}
    return _ocr


def _faces_stub_for(box1000):
    def _faces(data: bytes):
        return {"count": 1, "boxes": [box1000],
                "method": "stub", "note": ""}
    return _faces


# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------

def test_parse_kinds():
    assert parse_description("remove the watermark")["kind"] == "text"
    assert parse_description('remove the word "SALE"')["kind"] == "text"
    assert parse_description("remove the person on the left")["kind"] \
        == "person"
    # Person words take precedence over the colour: "the man in red"
    # is a person, with red carried as the qualifier.
    intent = parse_description("remove the man in red")
    assert intent["kind"] == "person"
    assert intent["colour"] == ("red", (255, 0, 0))
    assert parse_description("remove the red disc")["kind"] == "colour"
    assert parse_description("remove the dog")["kind"] == "object"


def test_parse_spatial_intersects():
    intent = parse_description("remove the person on the top left")
    assert intent["spatial"] == (0.0, 0.0, 0.34, 0.34)
    assert parse_description("remove the disc")["spatial"] is None


# ---------------------------------------------------------------------------
# Colour rung: the red disc, through the document
# ---------------------------------------------------------------------------

def test_remove_red_disc_by_description(tmp_path):
    img, gt = _scene_disc()
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the red disc")
    assert resolved.tier in ("flood", "grabcut", "sam")
    assert _iou(resolved.mask, gt) >= 0.9
    assert _outside_mask_identical(img, rendered.image, resolved.mask)
    # The disc is gone: the filled region is field-toned, not red.
    # Classical quality, stated honestly — a harmonic fill is a
    # smooth interpolation of the surroundings, not a perfect field
    # (the antialiased rim tints the edge band), so the bar is the
    # eroded interior within 50 of the field on every channel, plus
    # the whole disc decisively away from red.
    filled = np.asarray(rendered.image.convert("RGB")).astype(float)
    inside = np.asarray(gt.convert("L")) > 127
    mean_inside = filled[inside].mean(axis=0)
    assert mean_inside[0] < 180  # was 255
    assert mean_inside[2] > 120  # was 0
    core = np.asarray(_eroded(gt, 8).convert("L")) > 127
    mean_core = filled[core].mean(axis=0)
    assert abs(mean_core - np.array(BG)).max() < 50
    assert mask_fraction(resolved.mask) < SYNTHESISED_FRACTION
    assert rendered.provenance[0]["note"].startswith("mask rung")


def test_spatial_filter_picks_one_disc(tmp_path):
    img, left, right = _scene_two_discs()
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the red disc on the left")
    assert _iou(resolved.mask, left) >= 0.9
    assert _iou(resolved.mask, right) < 0.1
    assert _outside_mask_identical(img, rendered.image, resolved.mask)
    # The right disc survives untouched.
    after = np.asarray(rendered.image.convert("RGB"))
    before = np.asarray(img.convert("RGB"))
    keep_right = np.asarray(right.convert("L")) > 127
    assert bool((after[keep_right] == before[keep_right]).all())


# ---------------------------------------------------------------------------
# Text rung (injected OCR — tesseract is absent on this box)
# ---------------------------------------------------------------------------

def test_remove_text_with_injected_ocr(tmp_path):
    img, gt = _scene_text()
    # The SALE text sits at ~(48..95, 80..92); give the stub that box.
    ocr = _ocr_stub_for({"x": 250, "y": 416, "w": 250, "h": 62}, "SALE")
    found = locate_description(img, 'remove the word "SALE"', ocr=ocr)
    assert found["how"] == "ocr-text"
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description='remove the word "SALE"',
        locator=lambda im, d: locate_description(im, d, ocr=ocr))
    letters = np.asarray(gt.convert("L")) > 127
    mask = np.asarray(resolved.mask.convert("L")) > 127
    recall = float((mask & letters).sum() / letters.sum())
    assert recall >= 0.9
    assert _outside_mask_identical(img, rendered.image, resolved.mask)


def test_remove_watermark_without_target_words(tmp_path):
    img, gt = _scene_text()
    ocr = _ocr_stub_for({"x": 250, "y": 416, "w": 250, "h": 62}, "SALE")
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the watermark",
        locator=lambda im, d: locate_description(im, d, ocr=ocr))
    assert resolved.tier in ("flood", "grabcut", "sam", "box+feather")
    assert _outside_mask_identical(img, rendered.image, resolved.mask)


# ---------------------------------------------------------------------------
# Person rung (injected faces — cv2 is absent on this box)
# ---------------------------------------------------------------------------

def test_remove_person_with_injected_faces(tmp_path):
    img, gt = _scene_face()
    # Face disc centre (96, 80) r=24 → 0-1000 box (375, 291, 250, 250).
    faces = _faces_stub_for({"x": 375, "y": 291, "w": 250, "h": 250})
    found = locate_description(img, "remove the person", faces=faces)
    assert found["how"] == "face-point"
    assert found["box"] is None
    assert found["point"] == (96, 80)
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the person",
        locator=lambda im, d: locate_description(im, d, faces=faces))
    assert _iou(resolved.mask, gt) >= 0.8
    assert _outside_mask_identical(img, rendered.image, resolved.mask)


def test_two_faces_fall_through_to_contrast(tmp_path):
    """Two faces is not a refusal: the faces rung asks which one,
    and the ladder falls through to the next rung — the single
    salient object is still found by background-contrast."""
    img, gt = _scene_face()

    def _two_faces(data: bytes):
        return {"count": 2,
                "boxes": [{"x": 100, "y": 291, "w": 250, "h": 250},
                          {"x": 650, "y": 291, "w": 250, "h": 250}],
                "method": "stub", "note": ""}

    found = locate_description(img, "remove the person",
                               faces=_two_faces)
    assert found["how"] == "background-contrast"
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the person",
        locator=lambda im, d: locate_description(im, d,
                                                 faces=_two_faces))
    assert _iou(resolved.mask, gt) >= 0.8
    assert _outside_mask_identical(img, rendered.image, resolved.mask)


# ---------------------------------------------------------------------------
# Explicit mask honoured; large mask labelled; unknown refuses
# ---------------------------------------------------------------------------

def test_explicit_mask_honoured_over_description(tmp_path):
    img, gt = _scene_disc()
    mask_path = tmp_path / "mask.png"
    gt.save(mask_path)
    doc, rendered, resolved = _run_doc(
        tmp_path, img, mask=str(mask_path),
        description="remove the red disc")
    assert resolved.tier == "file"
    assert "explicit mask given — description ignored" in doc.notes
    assert _outside_identical(img, rendered.image, gt)
    assert _iou(resolved.mask, gt) == 1.0


def test_large_mask_labelled_synthesised(tmp_path):
    img, _gt = _scene_disc()
    big = Image.new("L", SIZE, 0)
    ImageDraw.Draw(big).rectangle([0, 0, 130, 130], fill=255)
    mask_path = tmp_path / "mask.png"
    big.save(mask_path)
    src = tmp_path / "scene.png"
    img.save(src)
    out = Studio().remove_object(
        str(src), mask=str(mask_path),
        save_to=str(tmp_path / "cleaned.png"))
    assert out["mask_fraction"] >= SYNTHESISED_FRACTION
    assert out["synthesised"] is True
    sidecar = Path(out["paths"][0] + ".txt")
    assert "synthesised content" in sidecar.read_text()


def test_unknown_description_refuses_honestly(tmp_path):
    img, _gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    with pytest.raises(ImgGenError) as excinfo:
        remove_object_document(
            str(src), description="remove the purple elephant")
    msg = str(excinfo.value)
    assert "purple elephant" in msg
    assert "tried:" in msg
    assert "colour-region" in msg


def test_needs_mask_or_description(tmp_path):
    img, _gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    with pytest.raises(ImgGenError, match="needs a mask or a description"):
        remove_object_document(str(src))


def test_unknown_method_refuses(tmp_path):
    img, _gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    with pytest.raises(ImgGenError, match="unknown inpaint method"):
        remove_object_document(str(src),
                               description="remove the red disc",
                               method="watermelon")


def test_neural_without_pipeline_refuses(tmp_path):
    img, _gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    # No checkpoint on this box: the Studio says so, honestly.
    with pytest.raises(ImgGenError, match="no native checkpoint"):
        Studio().remove_object(str(src),
                               description="remove the red disc",
                               method="neural")


# ---------------------------------------------------------------------------
# Methods: outside-mask pixels byte-identical on every CPU rung
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["numpy", "auto"])
def test_outside_identical_per_method(tmp_path, method):
    img, gt = _scene_disc()
    _doc, rendered, resolved = _run_doc(
        tmp_path, img, description="remove the red disc", method=method)
    assert _outside_mask_identical(img, rendered.image, resolved.mask)
    assert rendered.rungs[0]


# ---------------------------------------------------------------------------
# The guarantee lives in the op, not in a wrapper
# ---------------------------------------------------------------------------

def test_remove_object_op_preserves_outside(tmp_path):
    img, gt = _scene_disc()
    _doc, rendered, _resolved = _run_doc(
        tmp_path, img, description="remove the red disc")
    assert "byte-identical" in rendered.provenance[0]["note"]


def test_plain_inpaint_op_does_not_preserve(tmp_path):
    """A plain inpaint op carries no preserve flag — its note must
    not claim the guarantee. Guards the _run_inpaint restructure
    against leaking the flag onto every inpaint op."""
    img, gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    mask_path = tmp_path / "mask.png"
    gt.save(mask_path)
    doc = ImageDocument(str(src))
    doc.inpaint(mask=MaskRef.file(str(mask_path)), method="numpy")
    rendered = doc.render()
    assert "byte-identical" not in rendered.provenance[0]["note"]


# ---------------------------------------------------------------------------
# Document op: undo, redo, deterministic replay, Studio continues
# ---------------------------------------------------------------------------

def test_document_replay_and_undo(tmp_path):
    img, gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    doc = remove_object_document(str(src),
                                 description="remove the red disc")
    assert len(doc.ops) == 1
    assert doc.ops[0].op == "inpaint"
    first = doc.render()
    assert _outside_identical(img, first.image, _dilated(gt, 12))
    assert first.provenance[0]["note"].startswith("mask rung")
    second = doc.render()
    assert np.asarray(first.image).tobytes() \
        == np.asarray(second.image).tobytes()
    assert doc.undo() is not None
    assert len(doc.ops) == 0
    undone = doc.render()
    assert np.asarray(undone.image).tobytes() \
        == np.asarray(img.convert("RGB")).tobytes()
    assert doc.redo() is not None
    redone = doc.render()
    assert np.asarray(redone.image).tobytes() \
        == np.asarray(first.image).tobytes()


def test_studio_continues_from_remove_object(tmp_path):
    """The Studio can continue from the remove-object document: a
    second op lands on top of the cleaned image, and the render
    carries both ops' provenance."""
    img, gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    doc = remove_object_document(str(src),
                                 description="remove the red disc")
    remove_only = doc.render()
    doc.blur(radius=2.0)
    assert len(doc.ops) == 2
    continued = doc.render()
    assert len(continued.provenance) == 2
    assert continued.provenance[0]["op"] == "inpaint"
    assert continued.provenance[1]["op"] == "blur"
    assert "byte-identical" in continued.provenance[0]["note"]
    assert _outside_identical(img, continued.image, _dilated(gt, 20))
    assert np.asarray(continued.image).tobytes() \
        != np.asarray(remove_only.image).tobytes()


def test_document_with_explicit_mask(tmp_path):
    img, gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    mask_path = tmp_path / "mask.png"
    gt.save(mask_path)
    doc = remove_object_document(str(src), mask=str(mask_path))
    rendered = doc.render()
    assert _outside_identical(img, rendered.image, gt)


def test_document_needs_mask_or_description(tmp_path):
    img, _gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    with pytest.raises(ImgGenError, match="needs a mask or a description"):
        remove_object_document(str(src))


def test_document_rejects_pil_mask(tmp_path):
    img, gt = _scene_disc()
    src = tmp_path / "scene.png"
    img.save(src)
    with pytest.raises(ImgGenError, match="cannot embed mask pixels"):
        remove_object_document(str(src), mask=gt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_mask_fraction_math():
    assert mask_fraction(Image.new("L", (10, 10), 0)) == 0.0
    assert mask_fraction(Image.new("L", (10, 10), 255)) == 1.0
    half = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(half).rectangle([0, 0, 9, 4], fill=255)
    assert mask_fraction(half) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Brain tool
# ---------------------------------------------------------------------------

def test_tool_registered_and_rejects_empty():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "remove_object" in reg._tools
    tool = reg._tools["remove_object"]
    with pytest.raises(Exception, match="needs a description"):
        tool.fn("some.png")


def test_tool_end_to_end(tmp_path):
    from nomorals.tools.registry import ToolRegistry

    ws = tmp_path / "workspace"
    ws.mkdir()
    img, gt = _scene_disc()
    src = ws / "scene.png"
    img.save(src)
    settings = SimpleNamespace(workspace_dir=str(ws))
    context = SimpleNamespace(settings=settings, workspace=str(ws))
    reg = ToolRegistry(context=context)
    reg.register_builtins()
    tool = reg._tools["remove_object"]
    out = tool.fn("scene.png", description="remove the red disc")
    assert out["paths"]
    assert out["mask_rung"] in ("flood", "grabcut", "sam")
    assert out["synthesised"] is False
    assert Path(out["paths"][0]).exists()
    got = Image.open(out["paths"][0]).convert("RGB")
    assert _outside_identical(img, got, _dilated(gt, 12))
    sidecar = Path(out["paths"][0] + ".txt")
    assert sidecar.exists()
    assert "remove_object: mask rung" in sidecar.read_text()
