"""Tests for the Squoosh codec ladder (52-websites #17).

Proof contract (from the report), executed through the image systems
the op lives in — never a standalone executor:
- corpus images -> reported sizes equal actual file sizes on disk
  (the recommended trial written via out_dir re-stats to the same
  bytes; the Studio render of the document reproduces the trial);
- a trial claiming a smaller file at the same dimensions is verified
  on disk, not asserted (decode the written file, compare dims);
- absent rungs are named (a hidden encoder makes the AVIF trial
  "unavailable" with the reason, never a silent skip);
- the recommendation picks the best trial under a stated constraint
  (budget, preferred codec, lossless-first, smallest-with-floor);
- PSNR is reported only where actually computed (lossless trials
  carry None, lossy trials a float).

Fixtures are deterministic formula-built images — no randomness, no
network, no real photos needed: a photo-like gradient with
high-frequency detail (JPEG/WebP lose PSNR on it, AVIF wins) and a
flat image (lossless WebP is tiny and wins outright).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen.document import ImageDocument
from nomorals.media.imggen.optimize import (
    CODECS,
    OptimizeTrial,
    _pick,
    _target_spec,
    document_from_report,
    optimize_document,
    optimize_image,
    optimize_status,
)
from nomorals.media.imggen.studio import Studio


def _photo(path: Path, w: int = 400, h: int = 300) -> Path:
    """Photo-like: smooth gradients plus high-frequency detail in
    the blue channel, which lossy codecs smear (so PSNR separates
    the trials) and lossless codecs keep exactly."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 255) // w, (y * 255) // h,
                        128 + ((x * y) % 60))
    img.save(path)
    return path


def _flat(path: Path, w: int = 200, h: int = 150) -> Path:
    Image.new("RGB", (w, h), (40, 120, 200)).save(path)
    return path


def _big(path: Path) -> Path:
    """2500x1800 — over WhatsApp's 1920 long edge, so the ladder
    resizes before trialling and the document must replay that."""
    src = Image.new("RGB", (400, 288))
    px = src.load()
    for y in range(288):
        for x in range(400):
            px[x, y] = ((x * 255) // 400, (y * 255) // 288,
                        128 + ((x + y) % 40))
    src.resize((2500, 1800), Image.BICUBIC).save(path)
    return path


def _decode(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def _trial(codec: str, quality, size_bytes: int, psnr):
    return OptimizeTrial(codec=codec, quality=quality,
                         size_bytes=size_bytes, width=100, height=100,
                         psnr_db=psnr, status="ok")


# ── reported sizes are real ──────────────────────────────────────────


def test_reported_sizes_equal_disk_sizes(tmp_path):
    src = _photo(tmp_path / "photo.png")
    out_dir = tmp_path / "out"
    report = optimize_image(src, target="whatsapp", out_dir=out_dir)
    assert report.recommended is not None
    best = report.trials[report.recommended]
    assert best.path, "the recommended trial is written to disk"
    written = Path(best.path)
    assert written.exists()
    assert written.stat().st_size == best.size_bytes
    # every ok trial's measured size is its real byte length
    for trial in report.trials:
        if trial.status == "ok":
            assert trial.size_bytes == len(trial._data)


def test_smaller_file_same_dimensions_verified_on_disk(tmp_path):
    src = _flat(tmp_path / "flat.png")
    source_bytes = src.stat().st_size
    out_dir = tmp_path / "out"
    report = optimize_image(src, target="whatsapp", out_dir=out_dir)
    best = report.trials[report.recommended]
    assert best.size_bytes < source_bytes
    got = _decode(Path(best.path))
    assert got.size == (best.width, best.height)
    assert (best.width, best.height) == (200, 150)


# ── the recommendation under a stated constraint ────────────────────


def test_recommendation_fits_budget(tmp_path):
    src = _photo(tmp_path / "photo.png")
    report = optimize_image(src, target="whatsapp")
    best = report.trials[report.recommended]
    assert best.size_bytes <= 1_000_000


def test_recommendation_nothing_fits_names_closest(tmp_path):
    src = _photo(tmp_path / "photo.png")
    report = optimize_image(src, target="whatsapp", max_bytes=10)
    assert report.recommended is None
    assert report.closest is not None
    near = report.trials[report.closest]
    assert near.status == "ok"
    # the closest is the smallest ok trial
    ok = [t for t in report.trials if t.status == "ok"]
    assert near.size_bytes == min(t.size_bytes for t in ok)


def test_lossless_first_when_it_fits(tmp_path):
    src = _flat(tmp_path / "flat.png")
    report = optimize_image(src, target="whatsapp")
    best = report.trials[report.recommended]
    assert best.psnr_db is None, "identical pixels beat every lossy trial"
    assert best.size_bytes == min(
        t.size_bytes for t in report.trials
        if t.status == "ok" and t.psnr_db is None)


def test_pick_preferred_codec_within_margin():
    # webp is the best lossy trial, jpeg 0.5 dB behind it; the
    # lossless PNG is over budget. WhatsApp prefers jpeg, so jpeg
    # wins the margin rule.
    trials = [
        _trial("jpeg", 90, 50_000, 39.0),
        _trial("webp", 90, 48_000, 39.5),
        _trial("png", None, 2_000_000, None),
    ]
    spec = _target_spec("whatsapp")
    pick, _ = _pick(trials, spec, 1_000_000, "whatsapp", 30.0)
    assert trials[pick].codec == "jpeg"
    # 1.5 dB behind: outside the 1.0 dB margin, webp wins outright.
    trials[0].psnr_db = 38.0
    pick, _ = _pick(trials, spec, 1_000_000, "whatsapp", 30.0)
    assert trials[pick].codec == "webp"


def test_pick_smallest_honours_psnr_floor():
    trials = [
        _trial("jpeg", 70, 9_000, 28.0),   # below the 30 dB floor
        _trial("webp", 70, 11_000, 31.0),  # smallest that passes
        _trial("avif", 60, 40_000, 41.0),
        _trial("webp", None, 60_000, None),  # lossless passes, bigger
    ]
    spec = _target_spec("smallest")
    pick, _ = _pick(trials, spec, 0, "smallest", 30.0)
    assert trials[pick].codec == "webp"
    assert trials[pick].quality == 70
    # nothing passes the floor: no recommendation, closest named.
    for t in trials:
        if t.psnr_db is not None:
            t.psnr_db = 20.0
    trials[3].status = "failed"  # the lossless trial must fail too —
    # identical pixels pass any floor, so it would win otherwise.
    pick, closest = _pick(trials, spec, 0, "smallest", 30.0)
    assert pick is None
    assert closest is not None
    assert trials[closest].size_bytes == 9_000


# ── PSNR only where computed ─────────────────────────────────────────


def test_psnr_only_where_computed(tmp_path):
    src = _photo(tmp_path / "photo.png")
    report = optimize_image(src, target="whatsapp")
    table = report.format_table()
    for trial in report.trials:
        if trial.status != "ok":
            continue
        if trial.quality is None:
            assert trial.psnr_db is None
        else:
            assert isinstance(trial.psnr_db, float)
            assert trial.psnr_db > 0
    assert "identical" in table
    assert "dB" in table


# ── absent rungs are named ───────────────────────────────────────────


def test_absent_rung_named(monkeypatch, tmp_path):
    import nomorals.media.imggen.optimize as opt

    monkeypatch.setattr(opt.shutil, "which", lambda name: None)
    src = _photo(tmp_path / "photo.png")
    report = optimize_image(src, target="whatsapp", codecs=["avif"])
    trial = report.trials[0]
    assert trial.codec == "avif"
    assert trial.status == "unavailable"
    assert trial.reason, "an absent rung says why, never silently"
    assert "avif" in report.rungs_unavailable[0]
    status = optimize_status()
    assert "ffmpeg-avif" in status["unavailable"]
    assert "heifenc-avif" in status["unavailable"]


def test_status_names_every_codec():
    status = optimize_status()
    for codec in CODECS:
        named = (f"pillow-{codec}" in status["rungs"]
                 or f"ffmpeg-{codec}" in status["rungs"]
                 or f"heifenc-{codec}" in status["rungs"])
        assert named, f"{codec} is named available or unavailable"
    assert status["ssim"].startswith("unavailable")


# ── the document flow ────────────────────────────────────────────────


def test_document_flow_reproduces_trial(tmp_path):
    src = _flat(tmp_path / "flat.png")
    studio = Studio()
    paths = studio.optimize(src, target="whatsapp",
                            save_to=str(tmp_path / "result"))
    assert len(paths) == 1
    out = Path(paths[0])
    assert out.suffix == ".webp"
    got = _decode(out)
    assert got.size == (200, 150)
    assert got.format == "WEBP"
    # the trial table rides in the sidecar
    sidecar = Path(str(out) + ".txt")
    assert sidecar.exists()
    assert "webp lossless" in sidecar.read_text()


def test_document_flow_replays_resize(tmp_path):
    src = _big(tmp_path / "big.png")
    doc = optimize_document(src, target="whatsapp")
    ops = [o.op for o in doc.ops]
    assert ops == ["resize", "optimize"]
    studio = Studio()
    paths = studio.render_document(doc, save_to=str(tmp_path / "big-out"))
    got = _decode(Path(paths[0]))
    assert got.size == (1920, 1382), "long edge 1920, aspect kept"


def test_render_to_honours_op_and_refuses_overwrite(tmp_path):
    src = _photo(tmp_path / "photo.png")
    doc = ImageDocument(str(src)).optimize(codec="webp", quality=None)
    rec = doc.render_to(tmp_path / "result.png")
    out = Path(rec["output"])
    assert out.suffix == ".webp", "the extension follows the encoding"
    assert _decode(out).format == "WEBP"
    # the overwrite guard fires on the rewritten path: a png op
    # keeps the source's own extension, so rendering onto the
    # source itself is still refused.
    png_doc = ImageDocument(str(src)).optimize(codec="png")
    with pytest.raises(ImgGenError, match="must not overwrite"):
        png_doc.render_to(src)
    # an explicit fmt still wins over the op
    rec = doc.render_to(tmp_path / "override.png", fmt="JPEG",
                        quality=70)
    assert Path(rec["output"]).suffix == ".jpg"
    assert _decode(Path(rec["output"])).format == "JPEG"


def test_document_from_report_refuses_when_nothing_fits(tmp_path):
    src = _photo(tmp_path / "photo.png")
    report = optimize_image(src, target="whatsapp", max_bytes=10)
    assert report.recommended is None
    with pytest.raises(ImgGenError, match="no trial fits whatsapp"):
        document_from_report(report)


def test_invalid_codec_quality_target():
    with pytest.raises(ImgGenError, match="unknown optimize codec"):
        ImageDocument("x.png").optimize(codec="bogus")
    with pytest.raises(ImgGenError, match="out of range"):
        ImageDocument("x.png").optimize(quality=0)
    with pytest.raises(ImgGenError, match="out of range"):
        ImageDocument("x.png").optimize(quality=101)
    with pytest.raises(ImgGenError, match="unknown optimize target"):
        optimize_image("x.png", target="nope")
    with pytest.raises(ImgGenError, match="max_bytes"):
        optimize_image("x.png", max_bytes=-5)


# ── brain tool ───────────────────────────────────────────────────────


def test_tool_end_to_end(tmp_path):
    from nomorals.tools import optimize as opt_mod
    from nomorals.tools.registry import ToolRegistry

    src = _photo(tmp_path / "photo.png")
    context = SimpleNamespace(
        settings=SimpleNamespace(home=str(tmp_path)))
    reg = ToolRegistry(context=context)
    opt_mod.register(reg)
    tool = reg._tools["optimize_image"]
    result = tool.fn(path=str(src), target="whatsapp",
                     out=str(tmp_path / "tool-result"))
    assert result["ok"] is True
    out = Path(result["output"])
    assert out.exists()
    assert result["bytes"] == out.stat().st_size
    assert result["bytes"] == result["recommended"]["bytes"]
    assert result["bytes"] < result["source_bytes"]
    assert "trials" in result and len(result["trials"]) >= 3
    assert "★" in result["table"]
    doc_path = Path(result["document"])
    assert doc_path.exists()
    doc = json.loads(doc_path.read_text())
    assert any(o["op"] == "optimize" for o in doc["ops"])
    status = tool.fn(action="status")
    assert status["ok"] is True
    assert "rungs" in status
    with pytest.raises(ValueError, match="unknown optimize_image"):
        tool.fn(action="bogus")
    with pytest.raises(ValueError, match="needs a photo path"):
        tool.fn(path="")


def test_tool_registered_by_builtins():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "optimize_image" in reg._tools


# ── chat intent ──────────────────────────────────────────────────────


@pytest.mark.parametrize("text,target", [
    ("compress this for WhatsApp", "whatsapp"),
    ("compress this photo for telegram", "telegram"),
    ("optimise this image for the web", "web"),
    ("optimize this for archive", "archive"),
    ("make this smaller", "whatsapp"),
    ("make this image as small as possible", "smallest"),
    ("shrink this photo", "whatsapp"),
])
def test_optimize_intent(text, target):
    from nomorals.agents.coremind import _image_intent

    intent = _image_intent(text)
    assert intent is not None
    assert intent.kind == "vision_optimize"
    assert intent.action == "optimize"
    assert intent.meta["target"] == target


@pytest.mark.parametrize("text", [
    "remove the background",
    "upscale this image",
    "compress the meeting notes",
    "compress this video for WhatsApp",
])
def test_optimize_intent_misses(text):
    from nomorals.agents.coremind import _image_intent

    intent = _image_intent(text)
    if text.startswith("remove"):
        assert intent.kind == "vision_bgremove"
    elif text.startswith("upscale"):
        assert intent.kind == "vision_upscale"
    else:
        assert intent is None
