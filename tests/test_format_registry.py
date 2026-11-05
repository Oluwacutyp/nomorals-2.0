"""§1.5b — the format registry: formats as parameterized data.

Acceptance (plan §1.5b): one registry where a format = a named
parameter set over the Scene engine; the three incumbent systems
(contentops styles, pure_render packs, director STYLE_PACKS) mapped
onto it, not rewritten; registry validation fails loudly; a format
provably steers a render; formats extendable as data (scaffold →
file → load). Plus the §1.5e seam: the camera3d format resolves as
data and refuses at apply.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from nomorals.media.edit_engine import formats as F
from nomorals.media.edit_engine.formats import (
    FormatError, FormatSpec, apply_format, formats_data, get_format,
    list_formats, load_format_file, pack_for_format, register_format,
    resolve_format, scaffold_format, validate_format,
)
from nomorals.media.edit_engine.pure_render import apply_pack
from nomorals.media.edit_engine.spec import EditSpec
from nomorals.media.edit_engine.text import TextLayer
from nomorals.media_edit.videos import run_ffmpeg


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "fmt"
    d.mkdir()
    return d


def _clip(workdir, name="c.mp4", seconds=1.0):
    p = workdir / name
    run_ffmpeg(["-f", "lavfi", "-i",
                f"color=c=red:s=320x240:r=30:d={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)])
    return str(p)


def _probe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


# ---------------------------------------------------------------------------
# Bootstrap + totality: every incumbent name resolves or is named
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_no_bootstrap_issues_on_shipped_set(self):
        assert F.bootstrap_issues() == []

    def test_registry_is_populated(self):
        names = list_formats()
        for expected in ("phonk", "documentary", "vlog", "minimal",
                         "neutral", "naturalist", "noir", "neon",
                         "golden", "portrait", "landscape", "square",
                         "portrait45", "tiktok_short", "youtube_shorts",
                         "youtube_video", "instagram_square",
                         "instagram_portrait", "camera3d"):
            assert expected in names


class TestStyleMappingTotality:
    """Style entries are probed from the styles' own build_spec
    output — re-probing here must reproduce them exactly."""

    def test_every_style_resolves_and_matches_its_probe(self):
        from nomorals.media.contentops import styles as S

        for name in S.list_styles():
            entry = get_format(name)
            probe = S.build_spec(
                name, captions="format registry probe words",
                music="format-probe.mp3")
            assert (entry.width, entry.height, entry.fps) == (
                probe.width, probe.height, probe.fps)
            layer = probe.text_layers[0]
            assert entry.caption_style == layer.style
            assert entry.caption_position == layer.position
            assert entry.caption_size == layer.size
            assert entry.target_lufs == probe.audio.target_lufs
            sig = inspect.signature(
                S.get_style(name).build_spec).parameters
            assert entry.caption_wpm == sig["caption_wpm"].default
            assert entry.grade_pack == {"style": name}
            assert "contentops.styles" in entry.source

    def test_style_format_accessor_serves_the_registry_entry(self):
        from nomorals.media.contentops.styles import style_format
        data = style_format("phonk")
        assert data["name"] == "phonk"
        assert data["grade_pack"] == {"style": "phonk"}


class TestDesignMappingTotality:
    def test_design_entries_equal_style_packs(self):
        from nomorals.media.director.design import STYLE_PACKS

        design = {n for n in list_formats()
                  if get_format(n).design_pack is not None}
        assert design == set(STYLE_PACKS)
        for name in design:
            entry = get_format(name)
            assert entry.design_pack == name
            assert entry.groups() == ["design"]
            assert entry.label == STYLE_PACKS[name]["label"]


class TestAspectMappingTotality:
    def test_aspect_entries_match_format_sizes(self):
        from nomorals.media.motion_studio.grading import FORMAT_SIZES

        for aspect, key in FORMAT_SIZES.items():
            entry = get_format(key)
            num, den = aspect.split(":")
            assert Fraction(entry.width, entry.height) == Fraction(
                int(num), int(den))
            assert "FORMAT_SIZES" in entry.source


# ---------------------------------------------------------------------------
# Validation — fail loudly, naming the parameter
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_minimal_format(self):
        assert validate_format(FormatSpec(
            name="ok_fmt", label="x", width=320, height=240)) == []

    @pytest.mark.parametrize("spec,fragment", [
        (FormatSpec(name="Bad Name", label="x", width=1, height=1),
         "slug"),
        (FormatSpec(name="nolabel", label="", width=1, height=1),
         "label"),
        (FormatSpec(name="empty_fmt", label="x"), "no parameter group"),
        (FormatSpec(name="half", label="x", width=100), "together"),
        (FormatSpec(name="neg", label="x", width=-1, height=10),
         "positive int"),
        (FormatSpec(name="badcap", label="x", caption_style="nope"),
         "caption_style"),
        (FormatSpec(name="badpos", label="x", caption_position="nope"),
         "caption_position"),
        (FormatSpec(name="badsafe", label="x", safe={"bottom": 0.9}),
         "[0, 0.5)"),
        (FormatSpec(name="badplat", label="x", audio_platform="nope"),
         "audio_platform"),
        (FormatSpec(name="badloud", label="x", target_lufs=-99.0),
         "[-70, 0]"),
        (FormatSpec(name="baddesign", label="x", design_pack="nope"),
         "design_pack"),
        (FormatSpec(name="badgrade", label="x",
                    grade_pack={"style": "nope"}), "grade_pack style"),
        (FormatSpec(name="cutgrade", label="x",
                    grade_pack={"style": "phonk",
                                "spec_overrides": {"width": 10}}),
         "never the cut"),
        (FormatSpec(name="selfx", label="x", extends="selfx",
                    width=1, height=1), "extend itself"),
    ])
    def test_invalid_formats_name_the_problem(self, spec, fragment):
        issues = validate_format(spec)
        assert issues and any(fragment in i for i in issues), issues
        with pytest.raises(FormatError):
            register_format(spec)

    def test_duplicate_registration_with_different_spec_raises(self):
        spec = FormatSpec(name="dup_fmt", label="one",
                          width=100, height=100)
        register_format(spec)
        register_format(spec)  # same object: idempotent
        with pytest.raises(FormatError):
            register_format(FormatSpec(name="dup_fmt", label="two",
                                       width=200, height=200))

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(FormatError, match="unknown format keys"):
            FormatSpec.from_dict({"name": "x", "label": "y",
                                  "widt": 100})


# ---------------------------------------------------------------------------
# Resolution — total, provenance per parameter, require fails loudly
# ---------------------------------------------------------------------------

class TestResolution:
    def test_tiktok_short_provenance(self):
        r = resolve_format("tiktok_short")
        assert (r.width, r.height, r.fps) == (1080, 1920, 30)
        assert r.provenance["width"] == "base:portrait"
        assert r.provenance["safe"] == "format"
        assert r.provenance["audio_platform"] == "format"
        assert r.provenance["fps"] == "default"
        assert r.loudness["platform"] == "tiktok"
        assert r.loudness["target_lufs"] == -14.0
        assert set(r.groups) >= {"canvas", "safe", "captions", "audio"}

    def test_design_only_format_resolves_with_defaults_named(self):
        r = resolve_format("noir")
        assert r.design_pack == "noir"
        assert r.groups == ["design"]
        assert r.provenance["width"] == "default"
        assert r.grade_pack is None

    def test_require_fails_loudly_for_unspoken_groups(self):
        with pytest.raises(FormatError, match="canvas"):
            resolve_format("noir", require=("canvas",))
        # groups spoken through a base count:
        r = resolve_format("tiktok_short",
                           require=("canvas", "safe", "audio"))
        assert r.name == "tiktok_short"

    def test_unknown_format_and_unknown_base_raise(self):
        with pytest.raises(FormatError, match="unknown format"):
            resolve_format("no_such_format")
        register_format(FormatSpec(name="orphan_fmt", label="x",
                                   extends="ghost_base", width=10,
                                   height=10))
        with pytest.raises(FormatError, match="unknown format"):
            resolve_format("orphan_fmt")

    def test_extends_cycle_raises(self):
        register_format(FormatSpec(name="cycle_a", label="x",
                                   extends="cycle_b", width=10,
                                   height=10))
        register_format(FormatSpec(name="cycle_b", label="x",
                                   extends="cycle_a", width=10,
                                   height=10))
        with pytest.raises(FormatError, match="cycle"):
            resolve_format("cycle_a")

    def test_resolution_is_deterministic(self):
        a = resolve_format("youtube_shorts").to_dict()
        b = resolve_format("youtube_shorts").to_dict()
        assert a == b
        assert a["canvas"] == {"width": 1080, "height": 1920, "fps": 30}
        assert a["captions"]["style"] == "hormozi"
        assert a["loudness"]["platform"] == "youtube"


# ---------------------------------------------------------------------------
# Application — a format steers a spec, then a real render
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_format():
    name = "unittest_tiny"
    if name not in list_formats():
        register_format(FormatSpec(
            name=name, label="tiny test format", width=320, height=240,
            fps=30, caption_style="hormozi", caption_position="bottom",
            caption_size=40, safe={"bottom": 0.25},
            target_lufs=-16.0, source="test"))
    return name


class TestApplyFormat:
    def test_canvas_and_caption_defaults_steer(self, tiny_format):
        spec = EditSpec(
            clips=[{"path": "a.mp4", "duration": 1.0}],
            text_layers=[TextLayer(text="hello", start=0, end=1)],
            width=640, height=480, fps=24)
        out, rec = apply_format(spec, tiny_format)
        assert (out.width, out.height, out.fps) == (320, 240, 30)
        layer = out.text_layers[0]
        assert layer.style == "hormozi"          # default → format's
        assert layer.size == 40                  # default → format's
        assert rec["captions_applied"] == 1
        assert rec["loudness"]["target_lufs"] == -16.0
        # input untouched
        assert (spec.width, spec.text_layers[0].style) == (640, "karaoke")

    def test_safe_zone_raises_caption_margin(self):
        # tiktok_short declares bottom safe 0.22 over a 1920 canvas:
        # captions must clear 422px (the default margin is 120).
        spec = EditSpec(
            clips=[{"path": "a.mp4", "duration": 1.0}],
            text_layers=[TextLayer(text="hello", start=0, end=1)])
        out, rec = apply_format(spec, "tiktok_short")
        assert (out.width, out.height, out.fps) == (1080, 1920, 30)
        assert out.text_layers[0].margin_v == 422
        assert rec["captions_applied"] == 1

    def test_layer_choices_are_kept(self, tiny_format):
        chosen = TextLayer(text="mine", start=0, end=1, style="mrbeast",
                           size=100, margin_v=200)
        spec = EditSpec(clips=[{"path": "a.mp4", "duration": 1.0}],
                        text_layers=[chosen])
        out, rec = apply_format(spec, tiny_format)
        layer = out.text_layers[0]
        assert (layer.style, layer.size, layer.margin_v) == (
            "mrbeast", 100, 200)
        assert rec["captions_applied"] == 0

    def test_dict_spec_form(self, tiny_format):
        data = {"clips": [{"path": "a.mp4", "duration": 1.0}],
                "width": 10, "height": 10, "fps": 1, "text_layers": []}
        out, rec = apply_format(data, tiny_format)
        assert (out["width"], out["height"], out["fps"]) == (320, 240, 30)
        assert rec["format"] == tiny_format

    def test_format_steers_a_real_render(self, workdir, tiny_format):
        clip = _clip(workdir)
        spec = EditSpec(
            clips=[{"path": clip, "duration": 1.0}],
            text_layers=[TextLayer(text="steering proof", start=0.1,
                                   end=0.9)],
            width=640, height=480, fps=30, out=str(workdir / "o.mp4"))
        out_spec, _rec = apply_format(spec, tiny_format)
        from nomorals.media.edit_engine.spec import render
        out = render(out_spec, workdir=workdir / "rw")
        assert _probe_dims(out) == (320, 240)
        assert out_spec.text_layers[0].style == "hormozi"


# ---------------------------------------------------------------------------
# Packs — the pure_render adapter
# ---------------------------------------------------------------------------

class TestPackAdapter:
    BASE = {"clips": [{"path": "a.mp4", "duration": 1.0}],
            "width": 1080, "height": 1920}

    def test_format_pack_equals_style_pack(self):
        assert apply_pack(self.BASE, {"format": "phonk"}) == \
            apply_pack(self.BASE, {"style": "phonk"})

    def test_explicit_overrides_win_over_format(self):
        packed = apply_pack(self.BASE, {"format": "phonk",
                                        "spec_overrides": {"crf": 18}})
        assert packed["crf"] == 18

    def test_pack_for_format_and_unknown(self):
        assert pack_for_format("phonk") == {"style": "phonk"}
        assert pack_for_format("noir") == {}
        with pytest.raises(FormatError):
            pack_for_format("no_such_format")
        with pytest.raises(FormatError):
            apply_pack(self.BASE, {"format": "no_such_format"})

    def test_format_pack_render_cache_is_deterministic(self, workdir):
        from nomorals.media.edit_engine.pure_render import render_pure

        calls = []

        def stub_renderer(spec):
            calls.append(1)
            Path(spec.out).write_bytes(b"fake-mp4")
            return spec.out

        src = workdir / "src.mp4"
        src.write_bytes(b"source")
        kw = dict(cache_dir=workdir / "cache", renderer=stub_renderer)
        one = render_pure(str(src), dict(self.BASE),
                          {"format": "phonk"}, **kw)
        two = render_pure(str(src), dict(self.BASE),
                          {"format": "phonk"}, **kw)
        assert one["cache_key"] == two["cache_key"]
        assert two["cached"] is True and len(calls) == 1


# ---------------------------------------------------------------------------
# Styles consumption — build_spec(format=...)
# ---------------------------------------------------------------------------

class TestStylesConsumption:
    def test_format_steers_style_canvas(self):
        from nomorals.media.contentops.styles import build_spec
        spec = build_spec("minimal", format="youtube_video",
                          captions="steering test words",
                          music="m.mp3")
        assert (spec.width, spec.height) == (1920, 1080)

    def test_explicit_values_win_over_format(self):
        from nomorals.media.contentops.styles import build_spec
        spec = build_spec("minimal", format="youtube_video", width=720,
                          captions="steering test words", music="m.mp3")
        assert spec.width == 720 and spec.height == 1080

    def test_format_audio_and_caption_reach_phonk(self):
        from nomorals.media.contentops.styles import build_spec
        spec = build_spec("phonk", format="minimal",
                          captions="one two three", music="m.mp3")
        # minimal's probed format declares target -16 + caption style
        # minimal; phonk's signature accepts both keys.
        assert spec.audio.target_lufs == -16.0
        assert spec.text_layers[0].style == "minimal"

    def test_no_format_means_incumbent_behavior(self):
        from nomorals.media.contentops.styles import build_spec
        spec = build_spec("minimal", captions="plain words here",
                          music="m.mp3")
        assert (spec.width, spec.height) == (1080, 1920)
        assert spec.audio.target_lufs == -16.0


# ---------------------------------------------------------------------------
# Scene boundary — scenes_to_edit_spec(format=...)
# ---------------------------------------------------------------------------

class TestSceneBoundary:
    def test_scenes_inherit_format(self, workdir):
        from PIL import Image
        from nomorals.media.edit_engine.scene import (
            Scene, scenes_to_edit_spec)

        img = workdir / "still.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
        scenes = [
            Scene(kind="still", source=str(img), duration_s=1.0),
            Scene(kind="text", source="caption words", duration_s=1.0),
        ]
        spec = scenes_to_edit_spec(scenes, format="youtube_shorts")
        assert (spec.width, spec.height) == (1080, 1920)
        assert spec.format_application["format"] == "youtube_shorts"
        assert spec.text_layers[0].style == "hormozi"


# ---------------------------------------------------------------------------
# Extension as data — scaffold → fill → load; NM_FORMAT_DIRS
# ---------------------------------------------------------------------------

class TestScaffoldAndLoad:
    def test_scaffold_round_trip(self, workdir):
        path = scaffold_format("my_house_fmt", dest=workdir / "h.json")
        assert Path(path).exists()
        with pytest.raises(FileExistsError):
            scaffold_format("my_house_fmt", dest=workdir / "h.json")
        # unfilled skeleton fails loudly (no parameter group yet)
        with pytest.raises(FormatError, match="parameter group"):
            load_format_file(path)
        data = json.loads(Path(path).read_text())
        data["label"] = "house format"
        data["caption_style"] = "minimal"
        Path(path).write_text(json.dumps(data))
        spec = load_format_file(path)
        assert spec.name == "my_house_fmt"
        r = resolve_format("my_house_fmt")
        assert r.caption_style == "minimal"
        assert r.provenance["width"] == "base:portrait"

    def test_env_dirs_load_at_bootstrap(self, workdir):
        fmt_dir = workdir / "fmts"
        fmt_dir.mkdir()
        (fmt_dir / "envfmt.format.json").write_text(json.dumps({
            "name": "env_loaded_fmt", "label": "from env dir",
            "width": 200, "height": 200}))
        env = dict(os.environ)
        env["NM_FORMAT_DIRS"] = str(fmt_dir)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        proc = subprocess.run(
            [sys.executable, "-c",
             "from nomorals.media.edit_engine.formats import list_formats;"
             "print('env_loaded_fmt' in list_formats())"],
            capture_output=True, text=True, env=env,
            cwd=str(Path(__file__).resolve().parents[1]))
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True"


# ---------------------------------------------------------------------------
# The §1.5e seam — camera3d resolves as data; at apply it refuses
# by name with no rung registered and records the serving rung
# with one
# ---------------------------------------------------------------------------

class TestCamera3dSeam:
    def test_camera3d_resolves_as_data(self):
        r = resolve_format("camera3d")
        assert r.renderer == "camera3d"
        assert (r.width, r.height) == (1920, 1080)
        data = formats_data()
        row = next(f for f in data["formats"] if f["name"] == "camera3d")
        assert row["renderer"] == "camera3d"

    def test_camera3d_refuses_at_apply(self):
        # The refusal is the NO-RUNG state (§1.5e): with the
        # renderer registry emptied the seam refuses by name,
        # verbatim. With a rung registered the apply resolves
        # instead — pinned by the next test.
        from nomorals.media.edit_engine.scene3d import (
            get_renderer,
            register_renderer,
            registered_renderers,
            unregister_renderer,
        )
        saved = {name: get_renderer(name)
                 for name in registered_renderers()}
        for name in saved:
            unregister_renderer(name)
        try:
            spec = EditSpec(clips=[{"path": "a.mp4", "duration": 1.0}])
            with pytest.raises(FormatError, match="unavailable"):
                apply_format(spec, "camera3d")
        finally:
            for renderer in saved.values():
                register_renderer(renderer)

    def test_camera3d_apply_records_the_serving_rung(self):
        # With a §1.5e rung registered, apply resolves through the
        # registry and the record names the rung serving the
        # format3d surface. (Registered explicitly here: the
        # registry is shared session state — test_scene3d's
        # fixture deliberately leaves it empty.)
        from nomorals.media.edit_engine import register_rust_renderer
        from nomorals.media.edit_engine.scene3d import registered_renderers
        register_rust_renderer()
        assert registered_renderers()
        spec = EditSpec(clips=[{"path": "a.mp4", "duration": 1.0}])
        _spec, record = apply_format(spec, "camera3d")
        assert record["renderer3d"]["renderer"] in registered_renderers()
        assert record["renderer3d"]["surface"] == "format3d"


# ---------------------------------------------------------------------------
# Surfaces — brain tool, CLI, PARITY record
# ---------------------------------------------------------------------------

class _StubRegistry:
    context = None

    def __init__(self):
        self.fns = {}

    def register(self, name, description=None, capability=None):
        def deco(fn):
            self.fns[name] = fn
            return fn
        return deco


class TestToolSurface:
    def test_studio_formats_tool(self):
        from nomorals.tools.studio import register
        reg = _StubRegistry()
        register(reg)
        tool = reg.fns["studio_formats"]
        listing = tool()
        assert listing["ok"] and listing["count"] == len(list_formats())
        one = tool("tiktok_short")
        assert one["ok"] and one["format"]["name"] == "tiktok_short"
        bad = tool("no_such_format")
        assert bad["ok"] is False and "unknown format" in bad["reason"]


class TestCliSurface:
    def test_formats_list_and_show(self, capsys):
        from nomorals.cmdline.dispatch import main
        assert main(["video", "formats"]) == 0
        out = capsys.readouterr().out
        assert "tiktok_short" in out and "camera3d" in out
        assert main(["video", "formats", "show", "tiktok_short",
                     "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["canvas"] == {"width": 1080, "height": 1920,
                                   "fps": 30}
        assert shown["loudness"]["platform"] == "tiktok"
        assert main(["video", "formats", "show", "nope"]) == 1

    def test_formats_scaffold_cli(self, workdir, capsys):
        from nomorals.cmdline.dispatch import main
        dest = workdir / "cli_fmt.format.json"
        assert main(["video", "formats", "scaffold", "cli_fmt",
                     "--out", str(dest)]) == 0
        assert dest.exists()
        assert json.loads(dest.read_text())["name"] == "cli_fmt"

    def test_project_format_flag_steers_canvas(self, workdir, capsys,
                                               monkeypatch):
        from nomorals.cmdline.dispatch import main
        from nomorals.media.motion_studio.project import (
            MotionProject, MotionSegment)

        captured = {}

        def fake_render(self, out=None, **kw):
            captured.update(self.output)
            return {"output": "x.mp4",
                    "motion_project": {"project_hash": "h" * 16,
                                       "segments": [1]}}

        monkeypatch.setattr(MotionProject, "render", fake_render)
        proj = MotionProject(
            title="fmt", segments=[MotionSegment(
                uid="one", kind="text", text="hi", duration_s=1.0)])
        path = proj.save(str(workdir / "p.json"))
        assert main(["video", "motion", "project", path]) == 0
        plain = dict(captured)
        assert main(["video", "motion", "project", path,
                     "--format", "landscape"]) == 0
        assert (captured["width"], captured["height"]) == (1920, 1080)
        assert (plain["width"], plain["height"]) != (1920, 1080) \
            or plain["fps"] != captured["fps"]


class TestParityRecord:
    def test_parity_md_carries_the_formats_rows(self):
        text = Path("nomorals/media/PARITY.md").read_text()
        assert "## Formats — format registry (plan §1.5b)" in text
        assert "camera3d" in text
