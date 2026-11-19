"""Manim-class explainer presets — named, selectable, narrated (lane 85).

The recheck (2026-11-18) proved the gap: ``motion_studio/explainers.py``
carries diagram/equation/chart validators + ``list_explainers()`` — a
preset set by another name at best, never shipped as named presets with
a chat surface, and no code-trace kind at all. These tests pin the real
build: four named presets over the existing render path (PIL scene
renderer + code backend + TTS narration), an unknown name refused with
the list, and frames that build step by step — never a blank video
passed off as an explainer.
"""
from __future__ import annotations

import pytest
from PIL import Image

from nomorals.media.motion_studio._core import probe_duration
from nomorals.media.motion_studio.explainers import (
    frame_image,
    validate_design,
)
from nomorals.media.videogen import VideogenError, generate
from nomorals.media.videogen.narration import scene_narration
from nomorals.media.videogen.presets import (
    PRESETS,
    build_preset,
    list_presets,
    render_preset,
)

SIZE = (640, 360)

PRESET_NAMES = ("equation-walkthrough", "diagram-build",
                "code-trace", "chart-build")

CODE_DESIGN = {
    "engine": "code",
    "kind": "code",
    "title": "Trace: x doubles",
    "lines": [
        {"text": "x = 1", "values": "x = 1", "step": 0},
        {"text": "x = x + 1", "values": "x = 2", "step": 1},
        {"text": "print(x)", "values": "prints 2", "step": 2},
    ],
}


# -- the registry ---------------------------------------------------------------

def test_list_presets_names_all_four():
    names = {p["name"] for p in list_presets()}
    assert names == set(PRESET_NAMES)
    assert set(PRESETS) == set(PRESET_NAMES)
    for entry in list_presets():
        assert entry["description"]
        assert entry["params"]


def test_build_preset_unknown_names_the_known():
    with pytest.raises(VideogenError) as excinfo:
        build_preset("no-such-preset", lines="a | b")
    for name in PRESET_NAMES:
        assert name in str(excinfo.value)


def test_render_preset_unknown_refuses():
    with pytest.raises(VideogenError) as excinfo:
        render_preset("no-such-preset", lines="a | b")
    for name in PRESET_NAMES:
        assert name in str(excinfo.value)


# -- builders: content in, valid explainer design out ---------------------------

def test_equation_walkthrough_builds_valid_equation():
    design = build_preset("equation-walkthrough",
                          lines="E = mc^2 | F = ma",
                          title="Two laws")
    spec = validate_design(design)
    assert spec["kind"] == "equation"
    assert spec["title"] == "Two laws"
    assert spec["steps"] == 2
    assert [ln["text"] for ln in spec["lines"]] == ["E = mc^2", "F = ma"]
    # every step is boxed when it is current — the Manim highlight idiom
    assert all(ln["box"] for ln in spec["lines"])


def test_equation_walkthrough_needs_lines():
    with pytest.raises(VideogenError):
        build_preset("equation-walkthrough", lines="   ")


def test_diagram_build_builds_valid_diagram():
    design = build_preset("diagram-build", nodes="A | B | C",
                          title="Chain")
    spec = validate_design(design)
    assert spec["kind"] == "diagram"
    assert len(spec["nodes"]) == 3
    assert len(spec["edges"]) == 2
    assert spec["steps"] == 3
    assert [n["label"] for n in spec["nodes"]] == ["A", "B", "C"]


def test_diagram_build_needs_nodes():
    with pytest.raises(VideogenError):
        build_preset("diagram-build", nodes="")


def test_code_trace_builds_valid_code_with_values():
    design = build_preset(
        "code-trace",
        code="x = 1 | x = x + 1 | print(x)",
        values="x = 1 | x = 2 | prints 2",
        title="Trace: x doubles")
    spec = validate_design(design)
    assert spec["kind"] == "code"
    assert spec["steps"] == 3
    assert [ln["text"] for ln in spec["lines"]] == [
        "x = 1", "x = x + 1", "print(x)"]
    assert [ln["values"] for ln in spec["lines"]] == [
        "x = 1", "x = 2", "prints 2"]


def test_code_trace_values_shorter_than_lines_is_fine():
    design = build_preset("code-trace", code="a = 1 | b = 2",
                          values="a = 1")
    spec = validate_design(design)
    assert spec["lines"][0]["values"] == "a = 1"
    assert spec["lines"][1]["values"] == ""


def test_code_trace_values_longer_than_lines_refused():
    with pytest.raises(VideogenError) as excinfo:
        build_preset("code-trace", code="a = 1",
                     values="a = 1 | b = 2")
    assert "values" in str(excinfo.value)


def test_code_trace_empty_code_refused():
    with pytest.raises(VideogenError):
        build_preset("code-trace", code="   ")


def test_chart_build_builds_valid_chart():
    design = build_preset("chart-build", series="a:3, b:5",
                          title="Two bars")
    spec = validate_design(design)
    assert spec["kind"] == "chart"
    assert [p["label"] for p in spec["series"]] == ["a", "b"]
    assert [p["value"] for p in spec["series"]] == [3.0, 5.0]


def test_chart_build_bad_series_refused():
    with pytest.raises(VideogenError) as excinfo:
        build_preset("chart-build", series="a:three")
    assert "a:three" in str(excinfo.value)


# -- the code kind itself -------------------------------------------------------

def test_code_kind_validation_refuses_empty_lines():
    from nomorals.media.motion_studio._core import MotionStudioError

    with pytest.raises(MotionStudioError):
        validate_design({"engine": "code", "kind": "code", "lines": []})
    with pytest.raises(MotionStudioError):
        validate_design({"engine": "code", "kind": "code",
                         "lines": [{"text": "  "}]})


def test_code_frames_build_step_by_step():
    early = frame_image(CODE_DESIGN, SIZE, 0.05, 3.0)
    late = frame_image(CODE_DESIGN, SIZE, 2.9, 3.0)
    assert early.tobytes() != late.tobytes()
    # the late frame has more lit content than the near-empty first one:
    # count pixels that are not the dark background gradient's start.
    import numpy as np

    def lit(img: Image.Image) -> int:
        arr = np.asarray(img.convert("RGB")).astype(int)
        return int((arr.sum(axis=2) > 120).sum())

    assert lit(late) > lit(early)


# -- renders: real MP4s, preset named in the result ------------------------------

@pytest.mark.parametrize("name,kwargs", [
    ("equation-walkthrough",
     {"lines": "E = mc^2 | F = ma", "title": "Two laws"}),
    ("diagram-build", {"nodes": "A | B | C", "title": "Chain"}),
    ("code-trace",
     {"code": "x = 1 | x = x + 1 | print(x)",
      "values": "x = 1 | x = 2 | prints 2", "title": "Trace"}),
    ("chart-build", {"series": "a:3, b:5", "title": "Two bars"}),
])
def test_render_preset_renders_a_real_mp4(tmp_path, name, kwargs):
    out = tmp_path / f"{name}.mp4"
    res = render_preset(name, duration_s=2.0, fps=12.0, size=SIZE,
                        out=str(out), narrate=False, **kwargs)
    assert res.engine == "code"
    assert res.meta["preset"] == name
    assert name in res.note
    assert out.is_file() and out.stat().st_size > 1000
    assert abs(probe_duration(str(out)) - 2.0) < 0.4


def test_generate_threads_preset_into_meta(tmp_path):
    out = tmp_path / "preset.mp4"
    res = generate("", backend="code", scene=dict(CODE_DESIGN),
                   duration_s=2.0, fps=12.0, size=SIZE, out=str(out),
                   narrate=False, preset="code-trace")
    assert res.meta["preset"] == "code-trace"
    assert "code-trace" in res.note
    assert out.is_file()


# -- narration walks the same steps ---------------------------------------------

def test_code_trace_narration_names_the_values():
    segments = scene_narration(CODE_DESIGN, duration=3.0)
    assert [s["step"] for s in segments] == [0, 1, 2]
    assert "x equals 2" in segments[1]["text"]
    assert "prints 2" in segments[2]["text"]


def test_equation_walkthrough_narration_walks_the_lines():
    design = build_preset("equation-walkthrough",
                          lines="E = mc^2 | F = ma", title="Two laws")
    segments = scene_narration(design, duration=2.0)
    assert [s["step"] for s in segments] == [0, 1]
    assert "Two laws" in segments[0]["text"]
    assert "F = ma" in segments[1]["text"]


# -- chat surface: /video explain -----------------------------------------------

def test_chat_video_explain_usage_lists_presets():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    reply = RuntimeMediaMixin._control_video_explain(object(), "")
    assert "usage" in reply
    for name in PRESET_NAMES:
        assert name in reply


def test_chat_video_explain_unknown_preset_names_the_known():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    reply = RuntimeMediaMixin._control_video_explain(
        object(), "no-such-preset something")
    assert "unknown" in reply
    for name in PRESET_NAMES:
        assert name in reply


def test_chat_video_explain_renders_and_names_the_file(tmp_path,
                                                        monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    monkeypatch.setenv("HOME", str(tmp_path))
    reply = RuntimeMediaMixin._control_video_explain(
        object(),
        "code-trace x = 1 | x = x + 1 --values \"x = 1 | x = 2\" "
        "--seconds 1 --silent")
    assert "code" in reply
    assert "code-trace" in reply
    path = reply.splitlines()[0][2:].strip()
    assert path.endswith(".mp4")
    assert (tmp_path / ".nomorals" / "videos"
            / path.split("/")[-1]).is_file()
