"""Explainer family tests (Echelon Wave 1 close-out · plan §1.5d).

The plan's acceptance: an explainer fixture (a labelled diagram
build) renders deterministically, and text is composited by the
deterministic text path. The suite pins the whole shape: scene
validation as build errors, per-kind build progression in the pure
frame functions, compile records through MotionProject, end-to-end
renders diffed against recorded goldens (frame-sampled stats, the
1.2e harness pattern), double-render determinism, caption timing
provenance, and the scope refusals that stay refused.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from nomorals.media.motion_studio._core import (
    MotionStudioError,
    probe_duration,
)
from nomorals.media.motion_studio.explainers import (
    EXPLAINER_KINDS,
    frame_image,
    render_explainer,
    validate_design,
)
from nomorals.media.motion_studio.project import (
    MotionProject,
    MotionSegment,
    chart_project,
    diagram_project,
    equation_project,
    explainer_project,
)
from nomorals.media.motion_studio.studio import make_explainer

from tests.edit_engine_golden import diff_stats, frame_stats

NEEDS_FFMPEG = pytest.mark.skipif(
    os.system("ffmpeg -hide_banner -version > /dev/null 2>&1") != 0,
    reason="ffmpeg not installed")

SIZE = (320, 180)
FPS = 30
DURATION = 3.0

DIAGRAM_SCENE = {
    "engine": "diagram", "title": "Request flow",
    "nodes": [
        {"id": "in", "label": "Input", "x": 0.20, "y": 0.52, "step": 0},
        {"id": "mid", "label": "Model", "x": 0.52, "y": 0.52, "step": 1},
        {"id": "out", "label": "Answer", "x": 0.84, "y": 0.52, "step": 2},
    ],
    "edges": [
        {"from": "in", "to": "mid", "label": "feeds"},
        {"from": "mid", "to": "out"},
    ],
}

EQUATION_SCENE = {
    "engine": "equation", "title": "Energy",
    "lines": [
        {"text": "E = m c^2"},
        {"text": "m = 2, c = 3", "note": "plug in"},
        {"text": "E = 2 x 9 = 18", "box": True},
    ],
}

CHART_SCENE = {
    "engine": "chart", "chart": "bar", "title": "Runs per day",
    "ylabel": "count",
    "series": [
        {"label": "Mon", "value": 3},
        {"label": "Tue", "value": 7},
        {"label": "Wed", "value": 5},
    ],
}

SCENES = {"diagram": DIAGRAM_SCENE, "equation": EQUATION_SCENE,
          "chart": CHART_SCENE}

GOLDENS_PATH = Path(__file__).parent / "golden" / "explainer_goldens.json"


def build_project(scene: dict) -> MotionProject:
    return explainer_project([scene], per_scene=DURATION,
                             size=SIZE, fps=FPS)


def _close_to(img: np.ndarray, rgb: tuple[int, int, int],
              tol: int = 40) -> int:
    d = np.abs(img.astype(int) - np.array(rgb)).sum(axis=2)
    return int((d < tol * 3).sum())


# ---------------------------------------------------------------------------
# validation — scenes are data; malformed data is a build error
# ---------------------------------------------------------------------------

def test_validate_normalizes_defaults():
    spec = validate_design(copy.deepcopy(DIAGRAM_SCENE))
    assert spec["kind"] == "diagram"
    assert spec["steps"] == 3
    # an edge builds once both endpoints exist, unless authored
    assert [e["step"] for e in spec["edges"]] == [1, 2]
    spec = validate_design(copy.deepcopy(EQUATION_SCENE))
    assert spec["steps"] == 3  # each line its own step, in order
    spec = validate_design(copy.deepcopy(CHART_SCENE))
    assert spec["ymax"] == pytest.approx(7 * 1.15)
    # validation does not mutate the authored design
    scene = copy.deepcopy(DIAGRAM_SCENE)
    validate_design(scene)
    assert scene == DIAGRAM_SCENE


@pytest.mark.parametrize("design,frag", [
    ({"engine": "explainer"}, "1.5d"),
    ({"engine": "explainer", "kind": "particles"}, "1.5d"),
    ({"engine": "hologram"}, "1.5d"),
    ({"engine": "diagram", "nodes": []}, "non-empty 'nodes'"),
    ({"engine": "diagram", "nodes": [
        {"id": "a", "x": 0.5, "y": 0.5}, {"id": "a", "x": 0.2, "y": 0.2}]},
     "duplicate node id"),
    ({"engine": "diagram", "nodes": [{"id": "a", "x": 1.4, "y": 0.5}]},
     "normalized 0..1"),
    ({"engine": "diagram",
      "nodes": [{"id": "a", "x": 0.5, "y": 0.5}],
      "edges": [{"from": "a", "to": "ghost"}]}, "not a node id"),
    ({"engine": "diagram", "nodes": [
        {"id": "a", "x": 0.5, "y": 0.5, "color": "red"}]}, "bad color"),
    ({"engine": "equation", "lines": []}, "non-empty 'lines'"),
    ({"engine": "equation", "lines": [{"text": "  "}]}, "empty 'text'"),
    ({"engine": "chart", "chart": "pie",
      "series": [{"label": "a", "value": 1}]}, "must be one of"),
    ({"engine": "chart", "series": [{"label": "a", "value": -2}]},
     "must be ≥ 0"),
    ({"engine": "chart", "series": []}, "non-empty 'series'"),
])
def test_validation_refusals_name_the_field(design, frag):
    with pytest.raises(MotionStudioError, match=frag):
        validate_design(design)


# ---------------------------------------------------------------------------
# frame functions — the build actually builds, deterministically
# ---------------------------------------------------------------------------

def test_diagram_builds_node_by_node():
    early = np.asarray(frame_image(DIAGRAM_SCENE, SIZE, 0.05, DURATION))
    late = np.asarray(frame_image(DIAGRAM_SCENE, SIZE, 2.95, DURATION))
    # node 'out' carries palette green (#A3E635): absent early, drawn late
    assert _close_to(early, (0xA3, 0xE6, 0x35)) < 30
    assert _close_to(late, (0xA3, 0xE6, 0x35)) > 300
    assert late.mean() > early.mean() + 2.0


def test_equation_accrues_lines():
    counts = []
    for t in (0.3, 1.5, 2.9):
        img = np.asarray(frame_image(EQUATION_SCENE, SIZE, t, DURATION))
        counts.append(int((img > 100).sum()))
    assert counts[0] < counts[1] < counts[2]


def test_chart_bars_grow_and_line_draws_on():
    early = np.asarray(frame_image(CHART_SCENE, SIZE, 0.05, DURATION))
    late = np.asarray(frame_image(CHART_SCENE, SIZE, 2.95, DURATION))
    # first bar is palette sky (#38BDF8)
    assert _close_to(early, (0x38, 0xBD, 0xF8)) < 120
    assert _close_to(late, (0x38, 0xBD, 0xF8)) > 800
    line_scene = dict(CHART_SCENE, chart="line")
    early = np.asarray(frame_image(line_scene, SIZE, 0.05, DURATION))
    late = np.asarray(frame_image(line_scene, SIZE, 2.95, DURATION))
    assert _close_to(late, (0xF4, 0x72, 0xB6)) > _close_to(
        early, (0xF4, 0x72, 0xB6)) + 30  # second point's pink arrives


def test_frame_function_is_pure():
    a = np.asarray(frame_image(DIAGRAM_SCENE, SIZE, 1.234, DURATION))
    b = np.asarray(frame_image(DIAGRAM_SCENE, SIZE, 1.234, DURATION))
    assert np.array_equal(a, b)


def test_caption_words_reveal_on_their_times():
    words = [{"text": "alpha", "start": 0.0, "end": 0.8},
             {"text": "omega", "start": 2.2, "end": 3.0}]
    early = np.asarray(frame_image(
        DIAGRAM_SCENE, SIZE, 0.4, DURATION, words=words,
        timing_source="measured"))
    late = np.asarray(frame_image(
        DIAGRAM_SCENE, SIZE, 2.6, DURATION, words=words,
        timing_source="measured"))
    # the caption band (bottom strip) gains the second word's ink
    band = slice(int(SIZE[1] * 0.82), SIZE[1])
    assert (late[band] > 100).sum() > (early[band] > 100).sum()


# ---------------------------------------------------------------------------
# compile — MotionProject records state the tier, the kind, the provenance
# ---------------------------------------------------------------------------

def test_compile_records_per_kind(tmp_path):
    for i, (kind, scene) in enumerate(SCENES.items()):
        proj = build_project(scene)
        doc, rec = proj.to_document(workdir=tmp_path / f"c{i}")
        (seg,) = rec["segments"]
        assert seg["tier"] == "procedural"
        assert f"({kind})" in seg["rung"]
        assert seg["explainer"]["kind"] == kind
        assert seg["explainer"]["steps"] >= 3
        clip = doc.tracks[0].clips[0]
        assert Path(clip.path).exists()


def test_compile_timing_provenance_is_labelled(tmp_path):
    words = [{"text": "measured", "start": 0.0, "end": 0.9},
             {"text": "words", "start": 1.0, "end": 1.9}]
    proj = diagram_project(
        DIAGRAM_SCENE["nodes"], DIAGRAM_SCENE["edges"], title="t",
        duration=2.0, words=words, timing_source="measured",
        size=SIZE, fps=FPS)
    _, rec = proj.to_document(workdir=tmp_path / "m")
    assert rec["segments"][0]["timing_sources"] == {"measured": 2}
    proj = diagram_project(
        DIAGRAM_SCENE["nodes"], DIAGRAM_SCENE["edges"], title="t",
        duration=2.0, text="plain narration here", size=SIZE, fps=FPS)
    _, rec = proj.to_document(workdir=tmp_path / "e")
    assert rec["segments"][0]["timing_sources"] == {"estimated": 3}


def test_explainer_alias_and_roundtrip(tmp_path):
    scene = {"engine": "explainer", "kind": "chart",
             "series": [{"label": "a", "value": 2}]}
    proj = explainer_project([scene], per_scene=1.0, size=SIZE, fps=FPS)
    _, rec = proj.to_document(workdir=tmp_path / "a")
    assert rec["segments"][0]["explainer"]["kind"] == "chart"
    proj.seal()
    clone = MotionProject.from_dict(proj.to_dict())
    assert clone.to_dict() == proj.to_dict()
    assert clone.compute_hash() == proj.doc_hash


def test_compile_refusals_stay_refused(tmp_path):
    for design, frag in [
            ({"engine": "explainer"}, "1.5d"),
            ({"engine": "particles"}, "1.5d"),
            ({"engine": ""}, "1.5d")]:
        proj = MotionProject(segments=[MotionSegment(
            uid="x", kind="procedural", duration_s=1.0, design=design)])
        with pytest.raises(MotionStudioError, match=frag):
            proj.to_document(workdir=tmp_path / "x")


def test_kind_constructors_build_their_scenes(tmp_path):
    for proj in (
            diagram_project(DIAGRAM_SCENE["nodes"],
                            DIAGRAM_SCENE["edges"], duration=1.0,
                            size=SIZE, fps=FPS),
            equation_project(EQUATION_SCENE["lines"], duration=1.0,
                             size=SIZE, fps=FPS),
            chart_project(CHART_SCENE["series"], duration=1.0,
                          size=SIZE, fps=FPS)):
        assert proj.segments[0].kind == "procedural"
        doc, rec = proj.to_document(workdir=tmp_path / proj.title)
        assert Path(doc.tracks[0].clips[0].path).exists()


# ---------------------------------------------------------------------------
# renders — end-to-end through render_document, goldens, determinism
# ---------------------------------------------------------------------------

@NEEDS_FFMPEG
@pytest.mark.parametrize("kind", list(EXPLAINER_KINDS))
def test_render_end_to_end_matches_golden(tmp_path, kind):
    proj = build_project(SCENES[kind])
    out = tmp_path / f"{kind}.mp4"
    res = proj.render(out=out, workdir=tmp_path / "r")
    assert Path(res["output"]).exists()
    assert probe_duration(res["output"]) == pytest.approx(
        DURATION, abs=0.2)
    assert res["rungs"][0]["stage"] == "motion:scene1"
    assert f"({kind})" in res["rungs"][0]["rung"]
    goldens = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    for label in ("early", "mid", "final"):
        golden = goldens[kind][label]
        stats = frame_stats(res["output"], golden["t"],
                            workdir=tmp_path)
        assert diff_stats(stats, golden["stats"]) == [], (
            f"{kind}/{label}: "
            + "; ".join(diff_stats(stats, golden["stats"])))


@NEEDS_FFMPEG
def test_double_render_is_deterministic(tmp_path):
    proj = build_project(DIAGRAM_SCENE)
    stats = []
    for i in range(2):
        out = tmp_path / f"det{i}.mp4"
        proj.render(out=out, workdir=tmp_path / f"dr{i}")
        stats.append(frame_stats(out, 2.9, workdir=tmp_path))
    assert stats[0] == stats[1]


@NEEDS_FFMPEG
def test_render_explainer_direct_and_make_explainer(tmp_path):
    out = tmp_path / "direct.mp4"
    rec = render_explainer(CHART_SCENE, out, duration=2.0, size=SIZE,
                           fps=FPS, text="bars grow in order")
    assert rec["kind"] == "chart" and rec["frames"] == 60
    assert rec["timing_sources"] == {"estimated": 4}
    assert probe_duration(out) == pytest.approx(2.0, abs=0.2)
    made = make_explainer([dict(DIAGRAM_SCENE, duration=1.5)],
                          out=tmp_path / "made.mp4", size=SIZE, fps=FPS)
    assert Path(made).exists()
    assert probe_duration(made) == pytest.approx(1.5, abs=0.2)


def test_parity_row_records_the_family_as_shipped():
    parity = (Path(__file__).parent.parent
              / "nomorals" / "media" / "PARITY.md").read_text(
                  encoding="utf-8")
    assert "Explainer family (plan §1.5d)" in parity
    assert "shipped (§1.5d)" in parity
    assert "deferred to plan §1.5d" not in parity
