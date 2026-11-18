"""Tests for the code-video backend — backend="code" in videogen.

No model, no GPU: a scene spec renders deterministic seek(t) frames
to a real MP4. The Anthropic code-video shape (owner link 2026-10-31):
a video is code — every frame is a pure function of time, t = frame / fps.

The diffusion path is asserted unchanged beside it (have-it-all):
code for crisp, exact scenes; diffusion for organic footage; the router
picks per job from the scene-kind signal, never a user choice between rungs.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from nomorals.media.motion_studio._core import probe_duration
from nomorals.media.motion_studio.explainers import (
    frame_image,
    validate_design,
)
from nomorals.media.videogen import (
    VideogenError,
    capability_report,
    generate,
)
from nomorals.media.videogen.codegen import render_code_scene
from nomorals.media.videogen.formats import (
    FORMATS,
    build_format,
    list_formats,
    render_format,
)
from nomorals.media.videogen.registry import BACKENDS

DIAGRAM = {
    "engine": "diagram",
    "kind": "diagram",
    "title": "Lagos to Abuja",
    "nodes": [
        {"id": "lagos", "label": "Lagos", "x": 0.15, "y": 0.5,
         "shape": "circle", "step": 0},
        {"id": "abuja", "label": "Abuja", "x": 0.85, "y": 0.5,
         "shape": "circle", "step": 1},
    ],
    "edges": [{"from": "lagos", "to": "abuja", "label": "750 km",
               "step": 1}],
}

SIZE = (640, 360)


# -- registry + capability -----------------------------------------------------

def test_code_is_a_registered_backend():
    assert "code" in BACKENDS
    assert set(BACKENDS) >= {"ltx", "wan", "code"}


def test_capability_report_names_the_code_rung():
    report = capability_report()
    assert "code" in report
    assert "scene" in report


# -- code render: a real MP4, deterministic ------------------------------------

def test_code_renders_diagram_to_a_real_mp4(tmp_path):
    out = tmp_path / "route.mp4"
    # narrate=False pins the silent contract this test owns — exact
    # duration, exact frames. The narrated contract (duration grows to
    # fit the voice) is covered by test_gapfix_vidnarrate.py.
    res = generate("Lagos to Abuja", backend="code", scene=dict(DIAGRAM),
                   duration_s=2.0, fps=12.0, size=SIZE, out=str(out),
                   narrate=False)
    assert res.engine == "code"
    assert res.routed == "code"
    assert out.is_file() and out.stat().st_size > 1000
    assert abs(probe_duration(str(out)) - 2.0) < 0.4
    assert res.meta["frames"] == 24
    assert res.meta["render_hash"]
    assert res.meta["contact_sheet"]
    sheet = Image.open(res.meta["contact_sheet"])
    assert sheet.size[0] > 0 and sheet.size[1] > 0
    assert "code" in res.message()


def test_code_frames_are_deterministic(tmp_path):
    # the frame function is pure: same design, same t, same pixels
    a = frame_image(DIAGRAM, SIZE, 0.5, 2.0)
    b = frame_image(DIAGRAM, SIZE, 0.5, 2.0)
    assert a.tobytes() == b.tobytes()
    # and a whole render re-renders identically — the determinism receipt
    # (silent here: this test pins frames/duration, narration's stretch
    # is covered by test_gapfix_vidnarrate.py)
    rec1 = render_code_scene(dict(DIAGRAM), tmp_path / "one.mp4",
                             duration=2.0, fps=12.0, size=SIZE,
                             narrate=False)
    rec2 = render_code_scene(dict(DIAGRAM), tmp_path / "two.mp4",
                             duration=2.0, fps=12.0, size=SIZE,
                             narrate=False)
    assert rec1["render_hash"] == rec2["render_hash"]
    assert rec1["frames"] == rec2["frames"] == 24


def test_code_render_record_names_the_scene(tmp_path):
    rec = render_code_scene(dict(DIAGRAM), tmp_path / "rec.mp4",
                            duration=1.0, fps=12.0, size=SIZE)
    assert rec["engine"] == "code"
    assert rec["source"] == "design"
    assert rec["kind"] == "diagram"
    assert rec["title"] == "Lagos to Abuja"
    assert rec["deterministic"] is True


# -- routing: scene-kind signal, never a user choice ----------------------------

def test_auto_routes_a_scene_to_code(tmp_path):
    out = tmp_path / "auto.mp4"
    res = generate("Lagos to Abuja", scene=dict(DIAGRAM),
                   duration_s=1.0, fps=12.0, size=SIZE, out=str(out))
    assert res.engine == "code"
    assert res.routed == "code"


def test_organic_prompt_never_takes_the_code_rung(tmp_path):
    out = tmp_path / "organic.mp4"
    res = generate("a drone shot over Lagos at sunset",
                   duration_s=1.0, fps=12.0, size=SIZE, out=str(out))
    assert res.engine != "code"
    assert res.routed in ("neural", "motion")


def test_code_without_a_scene_raises():
    with pytest.raises(VideogenError, match="scene"):
        generate("a route map", backend="code", duration_s=1.0)


def test_code_with_a_diffusion_model_raises():
    with pytest.raises(VideogenError, match="code"):
        generate("a route map", backend="code", scene=dict(DIAGRAM),
                 model="ltx-distilled", duration_s=1.0)


def test_code_with_a_lora_raises(tmp_path, monkeypatch):
    # a real installed LoRA, so the conflict — not the unknown-name
    # error — is what fires
    lora_dir = tmp_path / "loras" / "ltx" / "testlora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DEVON_MODELS_DIR", str(tmp_path))
    with pytest.raises(VideogenError, match="code"):
        generate("a route map", backend="code", scene=dict(DIAGRAM),
                 lora=["testlora"], duration_s=1.0)


def test_diffusion_backend_with_a_scene_raises():
    with pytest.raises(VideogenError, match="code"):
        generate("a route map", backend="ltx", scene=dict(DIAGRAM),
                 duration_s=1.0)


def test_motion_backend_with_a_scene_raises():
    with pytest.raises(VideogenError, match="code"):
        generate("a route map", backend="motion", scene=dict(DIAGRAM),
                 duration_s=1.0)


# -- formats: presets as reusable commands --------------------------------------

def test_list_formats_names_the_first_three():
    names = [f["name"] for f in list_formats()]
    assert names == ["map-route", "headline", "teardown"]
    assert set(FORMATS) == {"map-route", "headline", "teardown"}


def test_format_map_route_builds_a_valid_diagram():
    design = build_format("map-route", route="Lagos to Abuja")
    spec = validate_design(design)
    assert spec["kind"] == "diagram"
    labels = [n["label"] for n in spec["nodes"]]
    assert labels == ["Lagos", "Abuja"]
    assert len(spec["edges"]) == 1
    assert spec["edges"][0]["label"]


def test_format_map_route_takes_a_stops_list():
    design = build_format("map-route",
                          stops=["Lagos", "Ibadan", "Abuja"])
    spec = validate_design(design)
    assert [n["label"] for n in spec["nodes"]] == [
        "Lagos", "Ibadan", "Abuja"]
    assert len(spec["edges"]) == 2


def test_format_headline_builds_a_valid_equation():
    design = build_format(
        "headline", headline="Devon renders video from code",
        lines=["no video model", "deterministic frames"])
    spec = validate_design(design)
    assert spec["kind"] == "equation"
    assert spec["lines"][0]["text"] == "Devon renders video from code"
    assert spec["lines"][0]["box"] is True
    assert len(spec["lines"]) == 3


def test_format_teardown_builds_a_numbered_diagram():
    design = build_format(
        "teardown", thing="keyboard",
        parts=["keycap", "switch", "plate", "case"])
    spec = validate_design(design)
    assert spec["kind"] == "diagram"
    assert [n["label"] for n in spec["nodes"]] == [
        "1. keycap", "2. switch", "3. plate", "4. case"]
    assert spec["title"].lower().startswith("keyboard")


def test_format_unknown_names_the_known_formats():
    with pytest.raises(VideogenError, match="map-route"):
        build_format("no-such-format", headline="x")


def test_format_missing_params_raise_named():
    with pytest.raises(VideogenError, match="headline"):
        build_format("headline", headline="  ")
    with pytest.raises(VideogenError, match="parts"):
        build_format("teardown", thing="keyboard", parts=[])
    with pytest.raises(VideogenError, match="route|stops"):
        build_format("map-route")


def test_format_renders_a_real_mp4(tmp_path):
    out = tmp_path / "map-route.mp4"
    # silent here: this test pins the exact 2.0s duration; the
    # narrated render (duration grows to fit the voice) is covered by
    # test_gapfix_vidnarrate.py.
    res = render_format("map-route", route="Lagos to Abuja",
                        duration_s=2.0, fps=12.0, size=SIZE, out=str(out),
                        narrate=False)
    assert res.engine == "code"
    assert out.is_file() and out.stat().st_size > 1000
    assert abs(probe_duration(str(out)) - 2.0) < 0.4


# -- surfaces: chat commands, CLI, tool -----------------------------------------

def test_chat_map_route_renders_and_names_the_file(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    monkeypatch.setenv("HOME", str(tmp_path))
    reply = RuntimeMediaMixin._control_code_format(
        object(), "map-route", "Lagos to Abuja --seconds 1")
    assert "code" in reply
    path = reply.splitlines()[0][2:].strip()
    assert path.endswith(".mp4")
    assert (tmp_path / ".nomorals" / "videos" / path.split("/")[-1]).is_file()


def test_chat_format_usage_on_empty_tail():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    assert "usage" in RuntimeMediaMixin._control_code_format(
        object(), "map-route", "")
    assert "usage" in RuntimeMediaMixin._control_code_format(
        object(), "teardown", "keyboard")


def test_cli_code_renders_map_route(tmp_path, capsys):
    import argparse

    from nomorals.cmdline.commands.video import _cmd_code

    out = tmp_path / "cli-route.mp4"
    args = argparse.Namespace(format="map-route", spec="Lagos to Abuja",
                              stops="", lines="", parts="", seconds=1.0,
                              out=str(out))
    assert _cmd_code(args, None) == 0
    assert out.is_file()
    assert "code" in capsys.readouterr().out


def test_cli_code_teardown_needs_parts(capsys):
    import argparse

    from nomorals.cmdline.commands.video import _cmd_code

    args = argparse.Namespace(format="teardown", spec="keyboard",
                              stops="", lines="", parts="", seconds=1.0,
                              out="")
    assert _cmd_code(args, None) == 1
    assert "usage" in capsys.readouterr().err


def test_tool_video_generate_takes_backend_and_scene():
    import inspect

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_builtins()
    sig = inspect.signature(registry._tools["video_generate"].fn)
    assert "backend" in sig.parameters
    assert "scene" in sig.parameters


# -- HTML scenes: the browser headless-capture substrate -------------------------

class _FakeTab:
    """A tab that paints a solid colour per seek — the browser boundary,
    faked; the frame loop, the t sequence and the encode are all real."""

    def __init__(self):
        self.seeks: list[float] = []
        self.url = ""

    def navigate(self, url):
        self.url = url
        return {"url": url}

    def evaluate(self, js, arg=None):
        self.seeks.append(float(arg))
        return True

    def screenshot_bytes(self):
        t = self.seeks[-1] if self.seeks else 0.0
        shade = int(40 + t * 30) % 256
        img = Image.new("RGB", (320, 180), (shade, 80, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        pass


class _FakeSession:
    def __init__(self):
        self.tab = _FakeTab()

    def open_tab(self, url=""):
        return self.tab

    def close(self):
        pass


def test_html_scene_seeks_and_captures_per_frame(tmp_path):
    session = _FakeSession()
    scene = {"engine": "html",
             "html": "<html><body><h1>hi</h1></body></html>"}
    rec = render_code_scene(scene, tmp_path / "html.mp4",
                            duration=1.0, fps=12.0, size=SIZE,
                            browser_session_factory=lambda: session)
    assert rec["source"] == "html"
    assert rec["frames"] == 12
    assert (tmp_path / "html.mp4").is_file()
    seeks = session.tab.seeks
    assert len(seeks) == 12
    assert seeks == sorted(seeks)
    assert abs(seeks[0] - 0.0) < 1e-9
    assert abs(seeks[-1] - (11 / 12.0)) < 1e-9


def test_html_scene_without_a_browser_fails_honestly(tmp_path):
    def _no_browser():
        raise RuntimeError(
            "CdpDriver needs Playwright: the package and its chromium "
            "build are missing")

    scene = {"engine": "html", "html": "<html><body>hi</body></html>"}
    with pytest.raises(VideogenError, match="browser"):
        render_code_scene(scene, tmp_path / "nope.mp4",
                          duration=1.0, fps=12.0, size=SIZE,
                          browser_session_factory=_no_browser)
