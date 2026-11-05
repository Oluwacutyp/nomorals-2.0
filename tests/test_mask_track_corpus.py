"""Mask tracking acceptance: the tracker, measured.

The corpus (``nomorals.media.edit_engine.mask_track_corpus``)
builds synthetic sequences whose ground-truth motion is exact by
construction; ``tests/golden/mask_track_corpus.json`` records, per
fixture, the floors the tracker must hold and what it actually
measured when the manifest was written. These tests rebuild every
fixture, re-run the real tracker, and fail below a floor — a
tracker regression is a test failure, not a production surprise.

The adversarial fixtures carry expectations, not floors: when the
content vanishes or the scene cuts, the tracker must flag the loss
at the event frame and hold its last fix; on a featureless field
it must refuse outright and name texture as the reason. A silent
'tracked' on any of those would be a fabricated success.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nomorals.media.edit_engine import mask_track, mask_track_corpus

MANIFEST = json.loads(
    (Path(__file__).parent / "golden" / "mask_track_corpus.json")
    .read_text(encoding="utf-8"))
BY_NAME = {f["name"]: f for f in MANIFEST["fixtures"]}
NAMES = mask_track_corpus.fixture_names()


def test_manifest_covers_exactly_the_corpus():
    assert MANIFEST["kind"] == "mask_track_corpus"
    assert sorted(BY_NAME) == sorted(NAMES)
    assert MANIFEST["tracker"]["min_score"] == \
        mask_track.DEFAULT_MIN_SCORE
    assert MANIFEST["tracker"]["search_radius_px"] == \
        mask_track.DEFAULT_SEARCH_RADIUS
    for name, entry in BY_NAME.items():
        floors = entry["floors"]
        assert 0.0 < floors["min_iou"] < 1.0, name
        assert floors["max_offset_err_px"] >= 0.0, name
        # a floor stricter than the recorded measurement would
        # fail the suite on the machine that wrote the manifest
        assert floors["max_offset_err_px"] >= \
            entry["measured"]["max_offset_err_px"], name
        assert floors["min_iou"] <= entry["measured"]["min_iou"], name


def test_fixtures_are_deterministic():
    for name in NAMES:
        frames_a, mask_a, truth_a, _ = mask_track_corpus.build_sequence(
            name)
        frames_b, mask_b, truth_b, _ = mask_track_corpus.build_sequence(
            name)
        assert truth_a == truth_b, name
        assert (mask_a == mask_b).all(), name
        assert len(frames_a) == len(frames_b), name
        for a, b in zip(frames_a, frames_b, strict=True):
            assert (a == b).all(), name


@pytest.mark.parametrize("name", NAMES)
def test_tracker_holds_its_floors(name):
    entry = BY_NAME[name]
    got = mask_track_corpus.score_fixture(name)
    floors, expect = entry["floors"], entry["expect"]
    assert got["status"] == expect["status"], (name, got)
    assert got["max_offset_err_px"] <= floors["max_offset_err_px"], (
        name, got)
    assert got["min_iou"] >= floors["min_iou"], (name, got)
    if "first_lost_frame" in expect:
        assert got["first_lost_frame"] == expect["first_lost_frame"], (
            name, got)
    if expect.get("holds_after_event"):
        assert got["holds_after_event"] is True, (name, got)
    if "reason_must_name" in expect:
        assert expect["reason_must_name"] in got["reason"], (name, got)


def test_tracking_is_deterministic():
    frames, mask, _truth, _meta = mask_track_corpus.build_sequence(
        "translate_diagonal")
    a = mask_track.track_sequence(frames, mask)
    b = mask_track.track_sequence(frames, mask)
    assert a.offsets == b.offsets
    assert a.scores == b.scores
    assert a.status == b.status == "tracked"


def test_shift_mask_clips_never_wraps():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:7] = True
    out = mask_track.shift_mask(m, 5, 0)
    # cols 3..6 land on 8..11: only 8..9 survive — the rest is
    # clipped away, never wrapped to the left edge
    assert int((out > 127).sum()) == 3 * 2
    assert not out[:, :8].any()
    assert (out[2:5, 8:10] > 127).all()
    out0 = mask_track.shift_mask(m, 0, 0)
    assert ((out0 > 127) == m).all()
