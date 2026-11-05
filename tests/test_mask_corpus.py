"""§1.1d acceptance: the mask ladder, measured.

The corpus (``nomorals.media.imggen.mask_corpus``) builds
synthetic stills whose ground truth is exact by construction;
``tests/golden/mask_corpus.json`` records, per fixture, the IoU
floor each rung must hold and the IoU actually measured when the
manifest was written. These tests rebuild every fixture, run the
real ladder rung by rung, and fail when a rung that is usable on
this machine falls below its floor — a rung regression is a test
failure, not a production surprise.

Rungs that are not usable here (GrabCut without cv2, SAM without
a checkpoint) are not silently skipped: the manifest records why
they carry no floor, and the scored set must match what
``mask_status()`` says this machine can actually run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomorals.media.imggen import mask_corpus
from nomorals.media.imggen.masks import CV2_AVAILABLE, mask_status

MANIFEST = json.loads(
    (Path(__file__).parent / "golden" / "mask_corpus.json")
    .read_text(encoding="utf-8"))
BY_NAME = {f["name"]: f for f in MANIFEST["fixtures"]}
NAMES = mask_corpus.fixture_names()


def test_manifest_covers_exactly_the_corpus():
    assert MANIFEST["kind"] == "mask_corpus"
    assert sorted(BY_NAME) == sorted(NAMES)
    for name, entry in BY_NAME.items():
        assert entry["floors"], name
        for rung, floor in entry["floors"].items():
            assert 0.0 < floor < 1.0, (name, rung, floor)
            assert rung in entry["measured"], (name, rung)
            # A floor above the measured value would fail the
            # suite on the machine that wrote the manifest.
            assert floor <= entry["measured"][rung]["iou"], (
                name, rung)


def test_sam_gap_is_recorded_not_invented():
    assert MANIFEST["sam"]["floor"] is None
    assert "not scored" in MANIFEST["sam"]["note"]


@pytest.mark.parametrize("name", NAMES)
def test_fixture_is_deterministic_with_sane_truth(name):
    image, gt, prompt = mask_corpus.build_fixture(name)
    image2, gt2, _ = mask_corpus.build_fixture(name)
    assert image.tobytes() == image2.tobytes()
    assert (gt == gt2).all()
    frac = float(gt.mean())
    assert 0.02 < frac < 0.60, (name, frac)
    entry = BY_NAME[name]
    assert abs(frac - entry["gt_fraction"]) < 0.01
    assert prompt == entry["prompt"]


@pytest.mark.parametrize("name", NAMES)
def test_ladder_holds_its_floors(name):
    entry = BY_NAME[name]
    scores = mask_corpus.score_fixture(name)
    status = mask_status()
    # Ladder-selection property: auto must land essentially on
    # the best rung this machine can run — the empty-mask defect
    # the corpus was built to catch fails exactly here.
    scored = [s["iou"] for r, s in scores.items()
              if r != "auto" and "iou" in s]
    if scored and "iou" in scores["auto"]:
        assert scores["auto"]["iou"] >= max(scored) - 0.02, (
            name, scores["auto"], scores)
    for rung, floor in entry["floors"].items():
        got = scores[rung]
        if "error" in got:
            # An error is only honest when this machine truly
            # cannot run the rung (or the rung does not apply
            # to the fixture's prompt) — anything else means
            # the rung broke and hid behind the error path.
            assert rung == "grabcut" and not CV2_AVAILABLE, (
                name, rung, got["error"])
            assert not status["grabcut"]["usable"]
            continue
        assert got["iou"] >= floor, (
            f"{name}/{rung}: IoU {got['iou']:.4f} below the "
            f"recorded floor {floor} (tier {got['tier']})")
        recorded = entry["measured"][rung]
        if got["tier"] == recorded["tier"]:
            # Same producing tier as the manifest run: the
            # number must land near the recorded measurement,
            # not merely above the floor.
            assert abs(got["iou"] - recorded["iou"]) <= 0.05, (
                name, rung, got["iou"], recorded["iou"])


def test_auto_never_accepts_an_empty_mask():
    """The corpus's founding defect, pinned: GrabCut's point
    seeding returned empty masks that auto_mask accepted as
    success. On a cv2 machine the auto rung for a point prompt
    must now produce a real mask (and a high-IoU one)."""
    if not CV2_AVAILABLE:
        pytest.skip("needs cv2 to exercise the grabcut rung")
    scores = mask_corpus.score_fixture("disc_on_field")
    assert "iou" in scores["grabcut"], (
        "grabcut produced no mask on a point prompt: "
        f"{scores['grabcut']}")
    assert scores["grabcut"]["iou"] >= 0.9
    assert scores["auto"]["tier"] == "grabcut"
    assert scores["auto"]["iou"] >= 0.9
