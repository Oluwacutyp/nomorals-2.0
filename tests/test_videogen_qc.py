"""Variants-matrix QC for the motion studio — contact sheet + winner.

Lane 7 (scout 17 alternate A8): the code-video rung already writes a
5-frame contact sheet beside every MP4 and edit_engine/qc.py already
measures a finished render — but nothing renders the SAME job across
rungs/params, sheets every variant, and names a winner. This module is
that discipline: a variants matrix for any render job, a contact sheet
per variant, one matrix sheet, and a QC report naming which variant
passed/failed and why — so a render is SEEN before it ships.
"""

import json

import pytest

from nomorals.media.videogen.capabilities import VideogenError

SCENE = {
    "engine": "diagram",
    "title": "QC test diagram",
    "nodes": [
        {"id": "a", "label": "Start", "x": 0.2, "y": 0.5},
        {"id": "b", "label": "End", "x": 0.8, "y": 0.5},
    ],
    "edges": [{"from": "a", "to": "b", "label": "go"}],
}

SMALL = (320, 180)


def test_plan_scene_job_has_code_motion_and_diffusion():
    from nomorals.media.videogen.qc import plan_variants

    variants = plan_variants(scene=SCENE, prompt="qc test")
    names = [v["name"] for v in variants]
    assert names == ["code", "code-480p", "motion", "diffusion"]
    assert variants[0]["backend"] == "code"
    assert variants[1]["size"] == (854, 480)
    assert variants[1]["fidelity"] == "scene"
    assert variants[2]["fidelity"] == "prompt"
    assert variants[2]["backend"] == "motion"
    assert variants[3]["backend"] == "diffusion"


def test_plan_prompt_only_job_has_no_code_variant():
    from nomorals.media.videogen.qc import plan_variants

    variants = plan_variants(prompt="a neon alley in rain")
    names = [v["name"] for v in variants]
    assert names == ["motion", "diffusion"]


def test_plan_empty_job_raises():
    from nomorals.media.videogen.qc import plan_variants

    with pytest.raises(VideogenError):
        plan_variants()


def _passing_variant(name, fidelity, black, exposure, aesthetic):
    return {"name": name, "fidelity": fidelity, "verdict": "pass",
            "measure": {"measurements": {
                "black_frame_fraction": black}},
            "gate": {"axes": {
                "exposure_clipping": {"score": exposure},
                "aesthetic": {"score": aesthetic}}}}


def test_rank_noise_ties_go_to_plan_order():
    from nomorals.media.videogen.qc import _rank_key

    # 0.002 apart on aesthetic — sampling noise, not a quality signal.
    a = _passing_variant("code", "scene", 0.0, 0.999, 0.854)
    b = _passing_variant("code-480p", "scene", 0.0, 0.999, 0.856)
    assert _rank_key(a, True) == _rank_key(b, True)


def test_rank_real_gap_beats_plan_order():
    from nomorals.media.videogen.qc import _rank_key

    # 0.13 apart on aesthetic — a real gap, the better variant wins
    # even though it sits later in the plan.
    a = _passing_variant("code", "scene", 0.0, 0.999, 0.854)
    b = _passing_variant("motion", "prompt", 0.0, 1.0, 0.984)
    assert _rank_key(b, False) < _rank_key(a, False)


def test_rank_scene_fidelity_beats_prettier_stand_in():
    from nomorals.media.videogen.qc import _rank_key

    # the motion stand-in measures prettier, but for a scene job the
    # scene-faithful variant ranks first.
    code = _passing_variant("code", "scene", 0.0, 0.999, 0.854)
    motion = _passing_variant("motion", "prompt", 0.0, 1.0, 0.984)
    assert _rank_key(code, True) < _rank_key(motion, True)


def test_matrix_over_real_scene_renders_and_names_winner(tmp_path):
    from nomorals.media.videogen.qc import render_matrix

    report = render_matrix(
        scene=SCENE, prompt="qc test", duration_s=1.0,
        out_dir=tmp_path / "matrix", size=SMALL, fps=8.0)
    assert report["kind"] == "variants_matrix"
    assert report["winner"] == "code"
    assert "renders the scene itself" in report["winner_reason"]
    assert "deterministic" in report["winner_reason"]
    by_name = {v["name"]: v for v in report["variants"]}
    # diffusion has no GPU on this box — recorded unavailable, honestly.
    assert by_name["diffusion"]["status"] == "unavailable"
    assert by_name["diffusion"]["reason"]
    # every rendered variant carries a real MP4 + contact sheet + verdict.
    for name in ("code", "code-480p", "motion"):
        v = by_name[name]
        assert v["status"] == "rendered", v
        assert v["verdict"] in ("pass", "fail", "unmeasured"), v
        assert v["path"].endswith(".mp4")
        assert v["contact_sheet"].endswith(".png")
        assert v["measure"]["probe"]["has_video"] is True
        assert v["gate"]["axes"]["exposure_clipping"]["status"] in (
            "pass", "fail", "unknown")
    assert by_name["code"]["verdict"] == "pass"
    # the matrix sheet + the JSON report exist on disk.
    from pathlib import Path
    assert Path(report["matrix_sheet"]).is_file()
    assert Path(report["report_path"]).is_file()
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved["winner"] == "code"
    assert saved["variants"][0]["name"] == "code"


def test_matrix_prompt_only_names_motion_winner(tmp_path):
    """Prompt-only job: fidelity ties, the visual axes decide — the
    motion stand-in is the only rendered variant here."""
    from nomorals.media.videogen.qc import render_matrix

    report = render_matrix(prompt="a neon alley in rain", duration_s=1.0,
                           out_dir=tmp_path / "m", size=SMALL, fps=8.0)
    assert report["winner"] == "motion"
    by_name = {v["name"]: v for v in report["variants"]}
    assert by_name["motion"]["verdict"] == "pass"
    assert by_name["diffusion"]["status"] == "unavailable"


def test_matrix_code_variant_is_deterministic(tmp_path):
    from nomorals.media.videogen.qc import render_matrix

    first = render_matrix(scene=SCENE, prompt="qc test", duration_s=1.0,
                          out_dir=tmp_path / "m1", size=SMALL, fps=8.0)
    second = render_matrix(scene=SCENE, prompt="qc test", duration_s=1.0,
                           out_dir=tmp_path / "m2", size=SMALL, fps=8.0)
    h1 = [v for v in first["variants"] if v["name"] == "code"][0]
    h2 = [v for v in second["variants"] if v["name"] == "code"][0]
    assert h1["measure"]["probe"]["duration_s"] == pytest.approx(
        h2["measure"]["probe"]["duration_s"], abs=0.05)
    assert h1["render_hash"] == h2["render_hash"]


def test_qc_existing_on_a_real_render(tmp_path):
    from nomorals.media.videogen.codegen import render_code_scene
    from nomorals.media.videogen.qc import qc_existing

    rec = render_code_scene(SCENE, tmp_path / "clip.mp4", duration=1.0,
                            size=SMALL, fps=8.0)
    out = qc_existing(rec["path"], out_dir=tmp_path / "qc")
    assert out["ship"] is False  # the edit-engine gate needs a document
    assert out["verdict"] == "pass"  # the visual axes all pass
    assert out["contact_sheet"].endswith(".png")
    assert out["measure"]["probe"]["has_video"] is True
    assert out["gate"]["axes"]["black_frames"]["status"] == "pass"


def test_qc_existing_missing_file_raises(tmp_path):
    from nomorals.media.videogen.qc import qc_existing

    with pytest.raises(VideogenError):
        qc_existing(str(tmp_path / "nope.mp4"))


def test_matrix_empty_job_raises(tmp_path):
    from nomorals.media.videogen.qc import render_matrix

    with pytest.raises(VideogenError):
        render_matrix(out_dir=tmp_path / "m")


def test_matrix_message_names_winner_and_paths(tmp_path):
    from nomorals.media.videogen.qc import matrix_message, render_matrix

    report = render_matrix(scene=SCENE, prompt="qc test", duration_s=1.0,
                           out_dir=tmp_path / "m", size=SMALL, fps=8.0)
    msg = matrix_message(report)
    assert "winner: code" in msg
    assert "matrix sheet" in msg
    assert "diffusion" in msg and "UNAVAILABLE" in msg


# -- surfaces: brain tool, chat verb, CLI -----------------------------------

def test_video_qc_tool_registered():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_builtins()
    assert "video_qc" in registry._tools
    tool = registry._tools["video_qc"]
    with pytest.raises(ValueError):
        tool.fn()


def test_chat_qc_matrix_names_winner(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    monkeypatch.setenv("HOME", str(tmp_path))
    reply = RuntimeMediaMixin._video_studio_qc(
        object(), '"a neon alley in rain" --seconds 1')
    assert "winner: motion" in reply
    assert "matrix sheet" in reply
    assert "diffusion" in reply and "UNAVAILABLE" in reply


def test_chat_qc_usage_on_empty_tail():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    assert "usage" in RuntimeMediaMixin._video_studio_qc(object(), "")


def test_cli_qc_existing_render(tmp_path, capsys):
    import argparse

    from nomorals.cmdline.commands.video import _cmd_qc
    from nomorals.media.videogen.codegen import render_code_scene

    rec = render_code_scene(SCENE, tmp_path / "clip.mp4", duration=1.0,
                            size=SMALL, fps=8.0)
    args = argparse.Namespace(target=rec["path"], seconds=5.0,
                              json=False, out="")
    assert _cmd_qc(args, None) == 0
    out = capsys.readouterr().out
    assert "verdict: PASS" in out
    assert "contact sheet" in out
