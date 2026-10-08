"""Tests for Wan 2.2 local text-to-video (build-map #26).

All offline: ComfyUI is mocked (FakeComfy stands in for the HTTP layer),
the router's probes are monkeypatched, no network, no GPU.
"""

import json
from pathlib import Path

import pytest

from nomorals.media_edit.video_models import (
    BUDGETS,
    INTENTS,
    VIDEO_MODELS,
    VideoModelError,
    VideoModelRouter,
    VideoRoute,
    generate_video,
)

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "nomorals" / \
    "media_edit" / "workflows"


@pytest.fixture()
def router():
    return VideoModelRouter()


@pytest.fixture()
def workstation_comfy(monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.video_models.get_profile_kind",
        lambda: "workstation")
    monkeypatch.setattr(
        "nomorals.media_edit.video_models._comfy_reachable", lambda: True)
    monkeypatch.setattr(
        "nomorals.media_edit.video_models._cuda_vram_gb", lambda: 24.0)


@pytest.fixture()
def nothing_available(monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.video_models.get_profile_kind",
        lambda: "laptop")
    monkeypatch.setattr(
        "nomorals.media_edit.video_models._comfy_reachable", lambda: False)
    monkeypatch.delenv("GOOGLE_FLOW_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("KLING_API_KEY", raising=False)


# -- workflow template ------------------------------------------------------

def test_wan_workflow_is_valid_json():
    wf = json.loads((WORKFLOWS_DIR / "wan22_t2v.json").read_text())
    assert isinstance(wf, dict) and len(wf) >= 7
    kinds = {n.get("class_type") for n in wf.values()}
    for expected in ("WanVideoModelLoader", "WanVideoVAELoader",
                     "WanVideoTextEncode", "WanVideoSampler",
                     "WanVideoDecode", "VHS_VideoCombine"):
        assert expected in kinds, f"missing node {expected}"


def test_wan_workflow_parameterization():
    from nomorals.media_edit.comfy import _wan22_t2v_workflow
    wf = _wan22_t2v_workflow(
        prompt="a cat in lagos", negative_prompt="blurry",
        width=1280, height=720, num_frames=80, steps=20, seed=42,
        cfg=6.0, model_file="wan.safetensors", vae_file="vae.safetensors")

    def single(class_type):
        found = [n for n in wf.values()
                 if n.get("class_type") == class_type]
        assert len(found) == 1, class_type
        return found[0]

    enc = single("WanVideoTextEncode")["inputs"]
    assert enc["positive_prompt"] == "a cat in lagos"
    assert enc["negative_prompt"] == "blurry"
    sampler = single("WanVideoSampler")["inputs"]
    assert sampler["seed"] == 42 and sampler["steps"] == 20
    assert sampler["num_frames"] == 80 and sampler["cfg"] == 6.0
    latent = single("WanVideoEmptyLatent")["inputs"]
    assert (latent["width"], latent["height"]) == (1280, 720)
    assert single("WanVideoModelLoader")["inputs"]["model"] == \
        "wan.safetensors"


def test_wan_workflow_bad_frames():
    from nomorals.media_edit.comfy import _wan22_t2v_workflow
    from nomorals.media_edit.generate import GenerativeEditError
    with pytest.raises(GenerativeEditError):
        _wan22_t2v_workflow(prompt="x", negative_prompt=None, width=None,
                            height=None, num_frames=0, steps=None,
                            seed=None, cfg=None, model_file=None,
                            vae_file=None)


# -- registry ----------------------------------------------------------------

def test_registry_fields():
    required = {"license", "vram_gb", "speed", "quality", "backend",
                "workflow", "paid", "key_env", "setup", "capabilities"}
    for name, m in VIDEO_MODELS.items():
        assert required <= set(m), name
        assert m["license"], name


def test_wan_is_apache():
    assert VIDEO_MODELS["wan22-14b"]["license"] == "Apache 2.0"
    assert VIDEO_MODELS["wan22-5b"]["license"] == "Apache 2.0"


def test_list_models():
    names = {m["name"] for m in VideoModelRouter().list_models()}
    assert {"wan22-14b", "wan22-5b", "veo-3.1", "minimax-h3", "kling"} <= names


# -- routing -----------------------------------------------------------------

def test_route_local_14b_on_big_vram(router, workstation_comfy):
    route = router.route("video", budget="standard")
    assert isinstance(route, VideoRoute)
    assert route.model == "wan22-14b"
    assert route.backend == "comfy" and not route.paid


def test_route_local_5b_on_small_vram(router, workstation_comfy,
                                      monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.video_models._cuda_vram_gb", lambda: 16.0)
    route = router.route("video", budget="standard")
    assert route.model == "wan22-5b"


def test_route_local_5b_when_vram_unprobed(router, workstation_comfy,
                                          monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.video_models._cuda_vram_gb", lambda: None)
    route = router.route("video", budget="standard")
    assert route.model == "wan22-5b"
    assert "unprobed" in route.reason


def test_route_hero_prefers_veo_when_key_set(router, workstation_comfy,
                                            monkeypatch):
    monkeypatch.setenv("GOOGLE_FLOW_API_KEY", "test-key")
    route = router.route("video", budget="hero")
    assert route.model == "veo-3.1" and route.paid


def test_route_hero_falls_back_to_wan_without_key(router, workstation_comfy,
                                                 monkeypatch):
    monkeypatch.delenv("GOOGLE_FLOW_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    route = router.route("video", budget="hero")
    assert route.model.startswith("wan22") and not route.paid


def test_route_motion_prefers_kling(router, workstation_comfy, monkeypatch):
    monkeypatch.setenv("KLING_API_KEY", "test-key")
    route = router.route("video", budget="motion")
    assert route.model == "kling" and route.paid


def test_route_termux_never_local(router, nothing_available, monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.video_models.get_profile_kind",
        lambda: "termux")
    with pytest.raises(VideoModelError) as exc:
        router.route("video", budget="standard")
    assert "workstation" in str(exc.value)


def test_route_nothing_available(router, nothing_available):
    with pytest.raises(VideoModelError) as exc:
        router.route("video", budget="standard")
    msg = str(exc.value)
    assert "MINIMAX_API_KEY" in msg  # honest setup steps, not fake


def test_route_bad_intent_and_budget(router):
    with pytest.raises(VideoModelError):
        router.route("dance")
    with pytest.raises(VideoModelError):
        router.route("video", budget="imax")


# -- NL patterns --------------------------------------------------------------

def test_video_nl_matches():
    from nomorals.agents.coremind import _video_intent
    it = _video_intent("generate a 5-second video of a cat in lagos")
    assert it is not None and it.kind == "video_gen"
    assert it.meta["duration_s"] == 5
    assert "cat in lagos" in it.meta["prompt"]
    it = _video_intent("make a video of ocean waves")
    assert it is not None and it.meta["duration_s"] == 5
    it = _video_intent("create a 10s video of a market")
    assert it is not None and it.meta["duration_s"] == 10


def test_video_nl_misfires():
    from nomorals.agents.coremind import _video_intent
    assert _video_intent("make a video call") is None
    assert _video_intent("generate a video idea") is None
    assert _video_intent("hello") is None
    assert _video_intent("generate a 0-second video of x") is None
    assert _video_intent("generate a 99-second video of x") is None


def test_video_intent_in_understand():
    from nomorals.agents.coremind import understand
    intents = understand("generate a 5-second video of a cat")
    kinds = [i.kind for i in intents]
    assert "video_gen" in kinds


# -- execution (mocked ComfyUI) -------------------------------------------------

class FakeComfy:
    """Stands in for ComfyUIBackend.generate_video — records the call,
    writes a fake mp4, returns its path."""

    def __init__(self):
        self.calls = []

    def generate_video(self, prompt, **kw):
        self.calls.append((prompt, kw))
        out = Path(kw.get("out_dir") or "/tmp") / "fake.mp4"
        out.write_bytes(b"FAKEVIDEO")
        return out


def test_generate_video_local_path(monkeypatch, tmp_path):
    import nomorals.media_edit.video_models as vm
    from nomorals.media_edit.video_models import VideoRoute

    class _FixedRouter:
        def route(self, *a, **k):
            return VideoRoute(backend="comfy", model="wan22-5b",
                              workflow="wan22_t2v", paid=False, fps=16,
                              max_duration_s=10, reason="test")

    fake = FakeComfy()
    monkeypatch.setattr(vm, "VideoModelRouter", _FixedRouter)
    monkeypatch.setattr("nomorals.media_edit.comfy.ComfyUIBackend",
                        lambda **kw: fake)
    path = generate_video("a cat", duration_s=2, out_dir=tmp_path)
    assert path.exists() and path.suffix == ".mp4"
    assert fake.calls and fake.calls[0][0] == "a cat"
    assert fake.calls[0][1]["duration_s"] == 2


def test_generate_video_duration_cap(monkeypatch):
    import nomorals.media_edit.video_models as vm

    class _FixedRouter:
        def route(self, *a, **k):
            from nomorals.media_edit.video_models import VideoRoute
            return VideoRoute(backend="comfy", model="wan22-5b",
                              workflow="wan22_t2v", paid=False, fps=16,
                              max_duration_s=10, reason="test")

    monkeypatch.setattr(vm, "VideoModelRouter", _FixedRouter)
    with pytest.raises(VideoModelError, match="up to 10s"):
        generate_video("a cat", duration_s=30)


def test_generate_video_empty_prompt():
    with pytest.raises(VideoModelError, match="needs a prompt"):
        generate_video("   ")


def test_generate_video_unconfigured_api_stub(monkeypatch):
    import nomorals.media_edit.video_models as vm

    class _PaidRouter:
        def route(self, *a, **k):
            from nomorals.media_edit.video_models import VideoRoute
            return VideoRoute(backend="minimax", model="minimax-h3",
                              workflow=None, paid=True, fps=24,
                              max_duration_s=10, reason="test")

    monkeypatch.setattr(vm, "VideoModelRouter", _PaidRouter)
    with pytest.raises(VideoModelError, match="no connector yet"):
        generate_video("a cat", duration_s=2)
