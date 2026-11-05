"""Echelon Wave 2 · Image port acceptance (plan §2.1a/b/d).

Pins the ImageDocument contract:

* an image edit is data — source ref + ordered op list over ONE
  vocabulary (media_edit PIL ops + imggen neural/develop ops);
* render replays from the source: byte-identical on every replay,
  source never modified, output never overwrites the source;
* façade agreement: undo/redo like EditStudio, compile/rebuild
  against the canonical edit_engine Document, and bridge to
  EditStudio sessions over the shared PIL vocabulary — with loud
  refusals (never silent drops) where a bridge cannot express an op;
* per-rung honesty: every render stage names the rung that produced
  it; neural ops without a pipeline raise rather than pass the
  image through unstyled; CPU tiers label themselves non-neural;
* the develop stage (§2.1b) is deterministic tone math, and masks
  (§2.1d) are first-class MaskRefs whose provenance is recorded.
"""

import hashlib
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from nomorals.media import imggen
from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen.develop import develop_image
from nomorals.media.imggen.document import (CLASSIC_OPS, IMAGE_OPS,
                                            ImageDocument)
from nomorals.media.imggen.masks import MaskRef, mask_status
from nomorals.media.imggen.studio import Studio

W, H = 64, 48


def _fixture(path):
    """Two-tone + saturated-block fixture: dark left, bright right,
    a vivid red square — enough structure for develop/mask maths."""
    img = Image.new("RGB", (W, H), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([W // 2, 0, W - 1, H - 1], fill=(220, 220, 220))
    d.rectangle([4, 4, 20, 20], fill=(230, 30, 30))
    img.save(path)
    return path


@pytest.fixture
def src(tmp_path):
    return _fixture(tmp_path / "src.png")


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── the document model ───────────────────────────────────────────────

def test_ops_are_data_with_stable_ids(src):
    doc = ImageDocument(str(src), title="t").resize(10, 10).develop(
        exposure=0.5)
    d = doc.to_dict()
    assert d["kind"] == "devon-image-document"
    assert [o["op"] for o in d["ops"]] == ["resize", "develop"]
    assert all(o["op"] in IMAGE_OPS for o in d["ops"])
    assert d["ops"][1]["params"] == {"exposure": 0.5}
    assert d["ops"][1]["label"]


def test_unknown_op_rejected_by_name(src):
    with pytest.raises(ImgGenError, match="unknown image op"):
        ImageDocument(str(src)).push("levitate")


def test_non_jsonable_params_rejected(src):
    with pytest.raises(ImgGenError, match="JSON-able"):
        ImageDocument(str(src)).push("resize", width=object(), height=5)


def test_serialization_round_trip_and_hash(src, tmp_path):
    doc = (ImageDocument(str(src), title="rt")
           .develop(shadows=0.3)
           .inpaint("fix", mask=MaskRef.of_box((1, 1, 9, 9)),
                    method="numpy"))
    clone = ImageDocument.from_dict(doc.to_dict())
    assert clone.to_dict() == doc.to_dict()
    assert clone.content_hash() == doc.content_hash()
    path = doc.save(tmp_path / "proj.json")
    loaded = ImageDocument.load(path)
    assert loaded.to_dict() == doc.to_dict()
    with pytest.raises(ImgGenError, match="not an image document"):
        ImageDocument.from_dict({"kind": "something-else"})


def test_undo_redo_like_the_facade(src):
    doc = ImageDocument(str(src)).resize(8, 8).blur(1).rotate(90)
    assert doc.undo().op == "rotate"
    assert [o.op for o in doc.ops] == ["resize", "blur"]
    assert doc.redo().op == "rotate"
    doc.undo()
    doc.push("flip", direction="vertical")  # push clears the redo stack
    assert doc.redo() is None
    assert ImageDocument(str(src)).undo() is None


def test_branch_is_independent(src):
    doc = ImageDocument(str(src)).resize(8, 8)
    branch = doc.branch().blur(2)
    assert [o.op for o in doc.ops] == ["resize"]
    assert [o.op for o in branch.ops] == ["resize", "blur"]


# ── render: replay, determinism, source safety ───────────────────────

def _rich_doc(src):
    return (ImageDocument(str(src), title="rich")
            .develop(exposure=0.4, shadows=0.3, saturation=0.2)
            .resize(32, 24)
            .inpaint("fill", mask=MaskRef.of_box((2, 2, 10, 10),
                                                  feather=2),
                     method="numpy")
            .blur(0.5))


def test_render_is_byte_identical_across_replays(src, tmp_path):
    doc = _rich_doc(src)
    a = _png_bytes(doc.render().image)
    b = _png_bytes(doc.render().image)
    assert a == b
    reloaded = ImageDocument.load(doc.save(tmp_path / "p.json"))
    assert _png_bytes(reloaded.render().image) == a


def test_render_never_touches_source(src, tmp_path):
    before = _sha(src)
    doc = _rich_doc(src)
    doc.render()
    rec = doc.render_to(tmp_path / "out.png")
    assert rec["kind"] == "image"
    assert rec["ops"] == ["develop", "resize", "inpaint", "blur"]
    assert len(rec["rungs"]) == 4
    assert rec["bytes"] > 0
    assert _sha(src) == before
    with pytest.raises(ImgGenError, match="must not overwrite"):
        doc.render_to(src)


def test_every_op_records_its_rung(src):
    r = _rich_doc(src).render()
    by_op = {p["op"]: p for p in r.provenance}
    assert by_op["develop"]["rung"] == "develop (numpy, deterministic)"
    assert by_op["resize"]["rung"] == "pil (media_edit.images)"
    assert "numpy" in by_op["inpaint"]["rung"]
    assert "non-neural" in by_op["inpaint"]["rung"]
    assert "box+feather" in by_op["inpaint"]["note"]


# ── op coverage: the classic PIL vocabulary ──────────────────────────

_CLASSIC_PARAMS = {
    "resize": {"width": 20, "height": 12, "mode": "exact"},
    "crop": {"box": (4, 4, 40, 30)},
    "rotate": {"angle": 90},
    "flip": {"direction": "horizontal"},
    "blur": {"radius": 1.5},
    "border": {"width": 3, "color": "red"},
    "enhance": {"brightness": 1.2},
    "thumbnail": {"size": 16},
    "annotate_text": {"text": "HI"},
    "annotate_shape": {"shape": "rectangle"},
    "duotone": {},
    "round_corners": {"radius": 8},
    "gradient_overlay": {},
    "meme": {"top": "TOP"},
    "stack": {},   # images injected per-test
    "grid": {},    # images injected per-test
    "watermark": {},  # logo injected per-test
}


def test_every_classic_op_renders(src, tmp_path):
    assert set(_CLASSIC_PARAMS) == set(CLASSIC_OPS)
    other = _fixture(tmp_path / "other.png")
    logo = tmp_path / "logo.png"
    Image.new("RGBA", (8, 8), (0, 255, 0, 200)).save(logo)
    for name in CLASSIC_OPS:
        params = dict(_CLASSIC_PARAMS[name])
        if name in ("stack", "grid"):
            params["images"] = [str(other)]
        if name == "watermark":
            params["logo"] = str(logo)
        doc = ImageDocument(str(src)).push(name, **params)
        out = doc.render()
        assert out.image is not None, name
        assert out.rungs == ["pil (media_edit.images)"], name


def test_classic_geometry_ops(src):
    img = ImageDocument(str(src)).push(
        "resize", width=20, height=12, mode="exact").render().image
    assert img.size == (20, 12)
    img = ImageDocument(str(src)).push("rotate", angle=90).render().image
    assert img.size == (H, W)
    img = ImageDocument(str(src)).push(
        "crop", box=(4, 4, 40, 30)).render().image
    assert img.size == (36, 26)


# ── the develop stage (plan §2.1b) ───────────────────────────────────

def _gray(v: int) -> Image.Image:
    return Image.new("RGB", (8, 8), (v, v, v))


def test_develop_identity_is_a_copy():
    img = _gray(100)
    out = develop_image(img)
    assert out is not img
    assert list(out.getdata()) == list(img.getdata())


def test_develop_exposure_is_ev_gain():
    out = develop_image(_gray(64), exposure=1.0)
    assert out.getpixel((0, 0))[0] == pytest.approx(128, abs=2)
    out = develop_image(_gray(64), exposure=-1.0)
    assert out.getpixel((0, 0))[0] == pytest.approx(32, abs=2)


def test_develop_shadows_and_highlights_are_weighted():
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (30, 30, 30))
    img.putpixel((1, 0), (225, 225, 225))
    out = develop_image(img, shadows=0.8)
    dark_gain = out.getpixel((0, 0))[0] - 30
    bright_gain = out.getpixel((1, 0))[0] - 225
    assert dark_gain > bright_gain + 10
    out = develop_image(img, highlights=-0.8)
    dark_loss = 30 - out.getpixel((0, 0))[0]
    bright_loss = 225 - out.getpixel((1, 0))[0]
    assert bright_loss > dark_loss + 10


def test_develop_saturation_and_temperature():
    img = Image.new("RGB", (1, 1), (200, 100, 50))
    out = develop_image(img, saturation=-1.0)
    r, g, b = out.getpixel((0, 0))
    assert r == g == b
    warm = develop_image(img, temperature=1.0).getpixel((0, 0))
    assert warm[0] > 200 and warm[2] < 50


def test_develop_vibrance_spares_the_vivid():
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (128, 110, 110))   # muted
    img.putpixel((1, 0), (255, 0, 0))       # vivid

    def spread(px):
        return max(px) - min(px)

    out = develop_image(img, vibrance=0.8)
    muted_gain = spread(out.getpixel((0, 0))) - spread((128, 110, 110))
    vivid_gain = spread(out.getpixel((1, 0))) - spread((255, 0, 0))
    assert muted_gain > vivid_gain


def test_develop_rejects_unknown_controls_and_ranges(src):
    with pytest.raises(ImgGenError, match="unknown develop control"):
        develop_image(_gray(10), glow=0.5)
    with pytest.raises(ImgGenError, match=r"must be in"):
        develop_image(_gray(10), exposure=9.0)
    with pytest.raises(ImgGenError, match="unknown controls"):
        ImageDocument(str(src)).develop(glow=0.5).render()


def test_develop_is_deterministic():
    img = _fixture_image()
    a = _png_bytes(develop_image(img, exposure=0.3, shadows=0.5,
                                 temperature=0.4, vibrance=0.6))
    b = _png_bytes(develop_image(img, exposure=0.3, shadows=0.5,
                                 temperature=0.4, vibrance=0.6))
    assert a == b


def _fixture_image() -> Image.Image:
    img = Image.new("RGB", (W, H), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([W // 2, 0, W - 1, H - 1], fill=(220, 220, 220))
    d.rectangle([4, 4, 20, 20], fill=(230, 30, 30))
    return img


def test_regional_develop_only_moves_the_mask(src):
    ref = MaskRef.of_box((0, 0, 16, 48), feather=0)
    doc = ImageDocument(str(src)).develop(mask=ref, exposure=1.0)
    r = doc.render()
    out = np.asarray(r.image.convert("RGB")).astype(int)
    base = np.asarray(_fixture_image().convert("RGB")).astype(int)
    # inside the mask the dark tone roughly doubles (EV +1)
    assert out[24, 4].mean() > base[24, 4].mean() + 20
    # outside the mask: untouched
    assert out[24, 60].tolist() == base[24, 60].tolist()
    assert "mask rung" in r.provenance[0]["note"]
    assert "box+feather" in r.provenance[0]["note"]


# ── masks as first-class refs (plan §2.1d) ───────────────────────────

def test_maskref_box_round_trip_and_resolve():
    ref = MaskRef.of_box((2, 3, 20, 30), feather=4)
    assert MaskRef.from_dict(ref.to_dict()) == ref
    img = _fixture_image()
    res = ref.resolve(img)
    assert res.tier == "box+feather"
    assert res.mask.size == img.size
    stamped = ref.resolved(res)
    assert stamped.rung == "box+feather"
    assert ref.rung == ""  # the original is unchanged (frozen)


def test_maskref_point_flood_covers_the_region():
    img = Image.new("RGB", (32, 32), (10, 10, 200))
    ImageDraw.Draw(img).rectangle([16, 0, 31, 31], fill=(200, 10, 10))
    ref = MaskRef.of_point((8, 16), feather=0)
    res = ref.resolve(img)
    assert res.tier == "flood"
    arr = np.asarray(res.mask)
    assert arr[16, 8] > 127 and arr[16, 24] < 127


def test_maskref_file_loads_and_resizes(tmp_path):
    mask_path = tmp_path / "m.png"
    Image.new("L", (16, 16), 255).save(mask_path)
    res = MaskRef.file(str(mask_path)).resolve(_fixture_image())
    assert res.tier == "file"
    assert res.mask.size == (W, H)
    assert np.asarray(res.mask).min() == 255


def test_maskref_auto_with_locator_box():
    seen = {}

    def predictor(image, text):
        seen["text"] = text
        return (4, 4, 24, 24)

    ref = MaskRef.auto(text="the red square", tier="box")
    res = ref.resolve(_fixture_image(), box_predictor=predictor)
    assert seen["text"] == "the red square"
    assert res.tier == "box+feather"


def test_maskref_rejects_unknown_kind():
    with pytest.raises(ImgGenError, match="unknown MaskRef kind"):
        MaskRef.from_dict({"kind": "psychic"})


def test_mask_status_states_every_rung():
    st = mask_status()
    assert set(st) == {"sam", "grabcut", "flood", "box", "file"}
    for rung in st.values():
        assert isinstance(rung["usable"], bool)
        assert isinstance(rung["reason"], str) and rung["reason"]
    assert st["flood"]["usable"] is True       # numpy is a hard dep here
    if not st["sam"]["usable"]:
        assert "checkpoint" in st["sam"]["reason"] \
            or "package" in st["sam"]["reason"]


# ── imggen ops: tiers + honesty ──────────────────────────────────────

def test_inpaint_cpu_tier_labels_itself(src):
    doc = ImageDocument(str(src)).inpaint(
        "fill", mask=MaskRef.of_box((4, 4, 20, 20), feather=2),
        method="numpy")
    r = doc.render()
    assert "numpy-voronoi-diffuse" in r.rungs[0]
    assert "non-neural" in r.rungs[0]


def test_inpaint_auto_without_pipeline_uses_cpu_rung(src):
    doc = ImageDocument(str(src)).inpaint(
        "fill", mask=MaskRef.of_box((4, 4, 20, 20), feather=2),
        method="auto")
    r = doc.render(pipeline=None)
    assert "neural" not in r.rungs[0] or "non-neural" in r.rungs[0]


def test_outpaint_cpu_extends_canvas(src):
    doc = ImageDocument(str(src)).outpaint("more sky", tier="cpu",
                                           left=10)
    r = doc.render()
    assert r.image.size == (W + 10, H)
    assert "cpu-outpaint" in r.rungs[0]


def test_upscale_classical(src):
    r = ImageDocument(str(src)).upscale(2.0).render()
    assert r.image.size == (2 * W, 2 * H)
    assert "classical" in r.rungs[0] and "non-neural" in r.rungs[0]


def test_img2img_without_pipeline_refuses_passthrough(src):
    doc = ImageDocument(str(src)).img2img("make it anime", strength=0.7)
    with pytest.raises(ImgGenError, match="unstyled"):
        doc.render(pipeline=None)


def test_upscale_diffusion_without_pipeline_raises(src):
    doc = ImageDocument(str(src)).upscale(2.0, diffusion=True)
    with pytest.raises(ImgGenError, match="diffusion=True needs"):
        doc.render(pipeline=None)


def test_inpaint_neural_without_pipeline_raises(src):
    doc = ImageDocument(str(src)).inpaint(
        "fill", mask=MaskRef.of_box((4, 4, 20, 20)), method="neural")
    with pytest.raises(ImgGenError, match="method='neural' needs"):
        doc.render(pipeline=None)


def test_img2img_with_fake_pipeline(monkeypatch, src):
    from nomorals.media.imggen import edit as edit_mod

    calls = []

    def fake_i2i(pipeline, image, prompt, strength=0.6, **kw):
        calls.append((pipeline, prompt, strength))
        return [Image.new("RGB", image.size, (9, 99, 199))]

    monkeypatch.setattr(edit_mod, "img2img", fake_i2i)
    pipe = object()
    doc = ImageDocument(str(src)).img2img("anime", strength=0.7, seed=7)
    r = doc.render(pipeline=pipe)
    assert calls == [(pipe, "anime", 0.7)]
    assert r.image.getpixel((0, 0)) == (9, 99, 199)
    assert "neural img2img" in r.rungs[0]


def test_inpaint_neural_with_fake_pipeline(monkeypatch, src):
    from nomorals.media.imggen import edit as edit_mod

    def fake_inpaint(pipeline, image, mask, prompt, feather=8, **kw):
        assert mask.size == image.size
        return [Image.new("RGB", image.size, (1, 2, 3))]

    monkeypatch.setattr(edit_mod, "inpaint", fake_inpaint)
    doc = ImageDocument(str(src)).inpaint(
        "fill", mask=MaskRef.of_box((4, 4, 20, 20)), method="neural")
    r = doc.render(pipeline=object())
    assert r.rungs[0] == "neural-blended-diffusion"
    assert "mask rung" in r.provenance[0]["note"]


# ── canonical Document + EditStudio bridges ──────────────────────────

def test_to_document_compiles_ops_verbatim(src):
    doc = (ImageDocument(str(src), title="poster")
           .develop(exposure=0.2).resize(10, 10))
    ed = doc.to_document()
    clip = ed.track("image").clips[0]
    assert clip.path == str(src)
    assert [(e.name, e.params) for e in clip.effects] == [
        ("develop", {"exposure": 0.2}),
        ("resize", {"width": 10, "height": 10}),
    ]
    assert ed.doc_hash == ed.compute_hash()


def test_document_round_trip_renders_identically(src):
    doc = _rich_doc(src)
    back = ImageDocument.from_document(doc.to_document())
    assert [o.op for o in back.ops] == [o.op for o in doc.ops]
    assert _png_bytes(back.render().image) == \
        _png_bytes(doc.render().image)


def test_from_document_rejects_foreign_effects(src):
    from nomorals.media.edit_engine.document import image_document
    from nomorals.media.edit_engine.timeline import Effect

    ed = image_document(str(src), title="x")
    ed.tracks[0].clips[0].effects = (Effect(name="filter",
                                            params={"preset": "noir"}),)
    with pytest.raises(ImgGenError, match="not in the image op"):
        ImageDocument.from_document(ed.seal())


def test_edit_studio_bridge_round_trip(src):
    doc = ImageDocument(str(src)).resize(24, 18).blur(1).flip("vertical")
    st = doc.to_edit_studio()
    assert [o["op"] for o in st.ops] == ["resize", "blur", "flip"]
    back = ImageDocument.from_edit_studio(st)
    assert [o.op for o in back.ops] == ["resize", "blur", "flip"]
    assert _png_bytes(back.render().image) == \
        _png_bytes(doc.render().image)


def test_edit_studio_bridge_refuses_imggen_ops(src):
    from nomorals.media_edit import MediaEditError

    doc = ImageDocument(str(src)).develop(exposure=0.3)
    with pytest.raises(MediaEditError, match="no EditStudio"):
        doc.to_edit_studio()


def test_from_edit_studio_refuses_foreign_ops(src):
    from nomorals.media_edit.studio import EditStudio

    st = EditStudio(str(src), kind="image").filter("vivid")
    with pytest.raises(ImgGenError, match="not in the image op"):
        ImageDocument.from_edit_studio(st)


# ── Studio integration ───────────────────────────────────────────────

def test_studio_document_builders_are_unrendered(src, tmp_path):
    st = Studio()
    mask_path = tmp_path / "m.png"
    Image.new("L", (W, H), 255).save(mask_path)
    d1 = st.img2img_document(str(src), "anime", strength=0.5)
    assert [o.op for o in d1.ops] == ["img2img"]
    assert d1.ops[0].params["strength"] == 0.5
    d2 = st.inpaint_document(str(src), str(mask_path), "fill")
    assert d2.ops[0].params["mask"]["kind"] == "file"
    d3 = st.inpaint_document(str(src), MaskRef.of_box((1, 1, 5, 5)),
                             "fill")
    assert d3.ops[0].params["mask"]["kind"] == "box"
    with pytest.raises(ImgGenError, match="cannot embed"):
        st.inpaint_document(str(src), Image.new("L", (W, H)), "fill")
    d4 = st.outpaint_document(str(src), "more", left=12)
    assert d4.ops[0].params["left"] == 12
    d5 = st.upscale_document(str(src), 3.0)
    assert d5.ops[0].params["scale"] == 3.0


def test_studio_upscale_classic_api_unchanged(src, tmp_path):
    st = Studio()
    out = tmp_path / "up.png"
    paths = st.upscale(str(src), scale=2.0, save_to=str(out))
    assert paths == [str(out)]
    with Image.open(out) as im:
        assert im.size == (2 * W, 2 * H)
    sidecar = open(str(out) + ".txt").read()
    assert "rung: upscale: classical" in sidecar


def test_studio_render_document_cpu_inpaint_honest_sidecar(src, tmp_path):
    st = Studio()
    doc = st.inpaint_document(str(src),
                              MaskRef.of_box((4, 4, 20, 20), feather=2),
                              "fill", method="auto")
    out = tmp_path / "filled.png"
    paths = st.render_document(doc, save_to=str(out))
    assert paths == [str(out)]
    sidecar = open(str(out) + ".txt").read()
    # no torch/checkpoint in this env → the pipeline could not be
    # built, the CPU rung ran, and the sidecar says both things.
    assert "rung: inpaint:" in sidecar
    assert "non-neural" in sidecar


def test_studio_img2img_classic_api_via_fake_pipeline(monkeypatch, src,
                                                      tmp_path):
    from nomorals.media.imggen import edit as edit_mod

    def fake_i2i(pipeline, image, prompt, strength=0.6, **kw):
        return [Image.new("RGB", image.size, (5, 6, 7))]

    monkeypatch.setattr(edit_mod, "img2img", fake_i2i)
    st = Studio()
    monkeypatch.setattr(st, "pipeline", lambda: object())
    out = tmp_path / "restyled.png"
    paths = st.img2img(str(src), "anime", strength=0.7,
                       save_to=str(out))
    assert paths == [str(out)]
    with Image.open(out) as im:
        assert im.convert("RGB").getpixel((0, 0)) == (5, 6, 7)
    assert "rung: img2img: neural img2img" in open(
        str(out) + ".txt").read()


def test_studio_generate_document_carries_provenance(monkeypatch,
                                                     tmp_path):
    gen_path = tmp_path / "gen.png"
    Image.new("RGB", (16, 16), (3, 3, 3)).save(gen_path)
    st = Studio()

    class FakePipe:
        pass

    def fake_generate(prompt, **kw):
        st._pipeline = FakePipe()
        st.last_warnings = ["tokenizer fallback notice"]
        return [str(gen_path)]

    monkeypatch.setattr(st, "generate", fake_generate)
    doc = st.generate_document("a cat", seed=42)
    assert doc.source == str(gen_path)
    assert doc.source_prompt == "a cat"
    assert "FakePipe" in doc.source_rung
    assert "tokenizer fallback notice" in doc.notes
    assert "seed=42" in doc.notes
    assert doc.ops == []  # generation is provenance, not an op


def test_imggen_package_lazy_exports():
    assert imggen.ImageDocument is ImageDocument
    assert imggen.MaskRef is MaskRef
