"""Pass V0+V1 — videogen drift repair + registry/LoRA plumbing.

Covers: genedit's repaired videogen calls (V0), the hero-clip surface
(chat + CLI), the video registry (stock entries, local diffusers dir
classification, LoRA discovery/resolution), backend LoRA application
via FakePipe diffusers stand-ins, pipeline model/LoRA forwarding +
capability declines naming the selected model, per-scene chain
overrides, and the chat/CLI flag surfaces. Everything offline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types

import pytest

import nomorals.media.videogen.capabilities as C
from nomorals.media.videogen import registry
from nomorals.media.videogen.capabilities import VideogenError

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg needed")


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------

def _mk_model_dir(root, name, class_name):
    d = os.path.join(root, name)
    os.makedirs(d)
    if class_name is not None:
        with open(os.path.join(d, "model_index.json"), "w") as f:
            json.dump({"_class_name": class_name,
                       "_diffusers_version": "0.30.0"}, f)
    return d


def _mk_lora(root, backend, name, *, as_file=False):
    base = os.path.join(root, "loras", backend)
    os.makedirs(base, exist_ok=True)
    if as_file:
        p = os.path.join(base, f"{name}.safetensors")
        with open(p, "wb") as f:
            f.write(b"fake")
        return p
    d = os.path.join(base, name)
    os.makedirs(d)
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA"}, f)
    with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
        f.write(b"fake")
    return d


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("DEVON_MODELS_DIR", str(root))
    return str(root)


class TestRegistry:
    def test_stock_entries_present(self, models_root):
        entries = {e["name"]: e for e in registry.list_models()}
        assert set(entries) >= {"ltx-distilled", "ltx-base", "wan-ti2v-5b"}
        d = entries["ltx-distilled"]
        assert d["backend"] == "ltx" and d["default"] is True
        assert d["source"] == "Lightricks/LTX-Video-0.9.7-distilled"
        assert d["vram_gb"] == 8.0 and d["license_key"] == "ltx"
        w = entries["wan-ti2v-5b"]
        assert w["backend"] == "wan" and w["vram_gb"] == 16.0
        assert w["source"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert entries["ltx-base"]["distilled"] is False

    def test_local_dirs_classified(self, models_root):
        _mk_model_dir(models_root, "my-ltx", "LTXPipeline")
        _mk_model_dir(models_root, "my-wan", "WanImageToVideoPipeline")
        _mk_model_dir(models_root, "my-sd", "StableDiffusionPipeline")
        _mk_model_dir(models_root, "broken", None)
        entries = {e["name"]: e for e in registry.list_models()}
        assert entries["my-ltx"]["backend"] == "ltx"
        assert entries["my-ltx"]["loadable"] is True
        assert entries["my-wan"]["backend"] == "wan"
        sd = entries["my-sd"]
        assert sd["loadable"] is False
        assert "StableDiffusionPipeline" in sd["loadable_reason"]
        # a dir without model_index.json is not a diffusers model: it is
        # not listed, and resolving its path says so honestly
        broken = registry.resolve(os.path.join(models_root, "broken"))
        assert broken["loadable"] is False
        assert "model_index.json" in broken["loadable_reason"]

    def test_lora_discovery_per_backend(self, models_root):
        _mk_lora(models_root, "ltx", "anime")
        _mk_lora(models_root, "wan", "cinematic", as_file=True)
        loras = registry.list_loras()
        by_name = {l["name"]: l for l in loras}
        assert by_name["anime"]["backend"] == "ltx"
        assert by_name["anime"]["format"] == "dir"
        assert by_name["cinematic"]["backend"] == "wan"
        assert by_name["cinematic"]["format"] == "file"
        assert [l["name"] for l in registry.list_loras("ltx")] == ["anime"]
        # model entries advertise their backend's loras
        entries = {e["name"]: e for e in registry.list_models()}
        assert "anime" in entries["ltx-distilled"]["loras"]
        assert "cinematic" in entries["wan-ti2v-5b"]["loras"]

    def test_resolve_by_name_and_path(self, models_root):
        d = _mk_model_dir(models_root, "hero-ltx", "LTXPipeline")
        by_name = registry.resolve("HERO-LTX")  # case-insensitive
        assert by_name["source"] == d and by_name["backend"] == "ltx"
        by_path = registry.resolve(d)
        assert by_path["name"] == "hero-ltx" and by_path["loadable"]
        assert registry.resolve("no-such-model") is None
        bad = registry.resolve(_mk_model_dir(models_root, "x", "FluxPipeline"))
        assert bad["loadable"] is False and "FluxPipeline" in (
            bad["loadable_reason"])

    def test_normalize_loras_shapes(self, models_root):
        anime = _mk_lora(models_root, "ltx", "anime")
        _mk_lora(models_root, "wan", "cinematic", as_file=True)
        assert registry.normalize_loras(None) == []
        one = registry.normalize_loras("anime")
        assert one == [{"name": "anime", "source": anime, "weight": 1.0,
                        "backend": "ltx"}]
        weighted = registry.normalize_loras("anime:0.8")
        assert weighted[0]["weight"] == 0.8
        mixed = registry.normalize_loras(
            ["anime", ("cinematic", 0.5), {"name": "anime", "weight": 2.0}])
        assert [m["name"] for m in mixed] == ["anime", "cinematic", "anime"]
        assert mixed[1]["weight"] == 0.5 and mixed[2]["weight"] == 2.0

    def test_normalize_loras_unknown_raises_naming_it(self, models_root):
        with pytest.raises(VideogenError, match="nope-lora"):
            registry.normalize_loras(["nope-lora"])


# ---------------------------------------------------------------------------
# Backend loaders — FakePipe diffusers stand-ins
# ---------------------------------------------------------------------------

class _FakeVAE:
    def enable_tiling(self):
        pass

    def enable_slicing(self):
        pass


class FakePipe:
    """Records from_pretrained + LoRA calls; callable like a pipeline."""

    instances: list = []
    fail_lora_for: str = ""

    def __init__(self, model_id):
        self.model_id = model_id
        self.vae = _FakeVAE()
        self.lora_calls: list = []
        self.adapter_calls: list = []
        FakePipe.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id, **kw):
        pipe = cls(model_id)
        pipe.from_pretrained_kw = kw
        return pipe

    def enable_model_cpu_offload(self):
        pass

    def to(self, device):
        return self

    def load_lora_weights(self, source, **kw):
        self.lora_calls.append((source, kw))
        if FakePipe.fail_lora_for and FakePipe.fail_lora_for in str(source):
            raise RuntimeError("bogus adapter weights")

    def set_adapters(self, names, adapter_weights=None):
        self.adapter_calls.append((list(names), list(adapter_weights or [])))

    def __call__(self, **kw):
        frame = types.SimpleNamespace()
        return types.SimpleNamespace(frames=[[frame]])


def _fake_diffusers(monkeypatch):
    FakePipe.instances = []
    FakePipe.fail_lora_for = ""
    mod = types.ModuleType("diffusers")
    for cls_name in ("LTXPipeline", "LTXImageToVideoPipeline",
                     "LTXConditionPipeline", "WanPipeline",
                     "WanImageToVideoPipeline"):
        setattr(mod, cls_name, type(cls_name, (FakePipe,), {}))
    utils = types.ModuleType("diffusers.utils")

    def _export(frames, path, fps=8):
        with open(path, "wb") as f:
            f.write(b"fake-mp4")

    utils.export_to_video = _export
    utils.load_image = lambda *a, **k: types.SimpleNamespace(size=(64, 64))
    mod.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    # the backends build a CUDA torch.Generator mid-generate; the real
    # CPU-only torch raises there, so stand in a minimal torch too
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"

    class _Gen:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            return self

    fake_torch.Generator = _Gen
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return mod


def _neural_env(monkeypatch, vram=24.0):
    import importlib
    monkeypatch.setattr(C, "torch_cuda_available", lambda: True)
    monkeypatch.setattr(C, "cuda_vram_gb", lambda: vram)
    monkeypatch.setattr(C, "diffusers_available", lambda: True)
    monkeypatch.setattr(C, "get_profile_kind", lambda: "workstation")
    for mod_name in ("nomorals.media.videogen.ltx_backend",
                     "nomorals.media.videogen.wan_backend"):
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "diffusers_available", lambda: True)
        monkeypatch.setattr(mod, "torch_cuda_available", lambda: True)


class TestBackendLoaders:
    def test_ltx_selected_model_reaches_from_pretrained(
            self, monkeypatch, tmp_path, models_root):
        _fake_diffusers(monkeypatch)
        _neural_env(monkeypatch)
        from nomorals.media.videogen.ltx_backend import LTXBackend
        be = LTXBackend(model_id="org/custom-ltx", distilled=False)
        out = be.generate("a cat", duration_s=1.0, out=str(tmp_path / "o.mp4"))
        assert os.path.isfile(out)
        pipe = FakePipe.instances[-1]
        assert pipe.model_id == "org/custom-ltx"

    def test_wan_selected_model_reaches_from_pretrained(
            self, monkeypatch, tmp_path, models_root):
        _fake_diffusers(monkeypatch)
        _neural_env(monkeypatch)
        from nomorals.media.videogen.wan_backend import WanBackend
        src = _mk_model_dir(models_root, "local-wan", "WanPipeline")
        be = WanBackend(model_id=src)
        out = be.generate("a dog", duration_s=1.0, out=str(tmp_path / "o.mp4"))
        assert os.path.isfile(out)
        assert FakePipe.instances[-1].model_id == src

    def test_two_loras_applied_in_order_with_weights(
            self, monkeypatch, tmp_path, models_root):
        _fake_diffusers(monkeypatch)
        _neural_env(monkeypatch)
        from nomorals.media.videogen.ltx_backend import LTXBackend
        d1 = _mk_lora(models_root, "ltx", "anime")
        f2 = _mk_lora(models_root, "ltx", "neon", as_file=True)
        be = LTXBackend(loras=[(d1, 0.7), (f2, 0.3)])
        be.generate("a cat", duration_s=1.0, out=str(tmp_path / "o.mp4"))
        pipe = FakePipe.instances[-1]
        assert pipe.lora_calls[0] == (d1, {"adapter_name": "devon_lora_0"})
        assert pipe.lora_calls[1] == (
            os.path.dirname(f2),
            {"weight_name": "neon.safetensors",
             "adapter_name": "devon_lora_1"})
        assert pipe.adapter_calls == [
            (["devon_lora_0", "devon_lora_1"], [0.7, 0.3])]

    def test_lora_failure_raises_naming_the_lora(
            self, monkeypatch, tmp_path, models_root):
        _fake_diffusers(monkeypatch)
        _neural_env(monkeypatch)
        from nomorals.media.videogen.ltx_backend import LTXBackend
        d1 = _mk_lora(models_root, "ltx", "anime")
        FakePipe.fail_lora_for = "anime"
        be = LTXBackend(loras=[(d1, 1.0)])
        with pytest.raises(VideogenError, match="anime"):
            be.generate("a cat", duration_s=1.0,
                        out=str(tmp_path / "o.mp4"))


# ---------------------------------------------------------------------------
# Pipeline selection
# ---------------------------------------------------------------------------

def _fake_cap(monkeypatch, *, available=True, backend="wan", vram=24.0):
    import nomorals.media.videogen.pipeline as P
    cap = C.NeuralCapability(available, backend if available else "none",
                             vram, "workstation", True,
                             "fake capability" if available
                             else "fake decline reason")
    monkeypatch.setattr(P, "neural_capability",
                        lambda prefer="auto", min_vram_gb=None: cap)
    return P


class TestPipelineSelection:
    def test_model_and_lora_forwarded(self, monkeypatch, models_root):
        P = _fake_cap(monkeypatch)
        _mk_lora(models_root, "wan", "cinematic")
        seen = {}

        def fake_neural(prompt, backend, **kw):
            seen.update(backend=backend, **kw)
            return "/tmp/fake.mp4"

        monkeypatch.setattr(P, "_neural_generate", fake_neural)
        res = P.generate("a wave", model="wan-ti2v-5b",
                         lora=["cinematic:0.9"])
        assert res.engine == "wan" and res.routed == "neural"
        assert seen["backend"] == "wan"
        assert seen["model_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert seen["loras"][0]["name"] == "cinematic"
        assert seen["loras"][0]["weight"] == 0.9
        assert res.meta["model"] == "wan-ti2v-5b"
        assert res.meta["loras"] == ["cinematic"]

    def test_over_floor_model_declines_naming_model_and_floor(
            self, monkeypatch, models_root):
        # real capability probe over an 8GB fake GPU; a local Wan-dir
        # model carries the 16GB floor
        _neural_env(monkeypatch, vram=8.0)
        src = _mk_model_dir(models_root, "big-wan", "WanPipeline")
        import nomorals.media.videogen.pipeline as P
        with pytest.raises(VideogenError) as ei:
            P.generate("a wave", model="big-wan", allow_fallback=False)
        assert "big-wan" in str(ei.value) and "16" in str(ei.value)
        assert src  # the same model resolves by path too
        assert registry.resolve(src)["name"] == "big-wan"

    def test_over_floor_fallback_note_names_model(
            self, monkeypatch, models_root):
        _neural_env(monkeypatch, vram=8.0)
        import nomorals.media.videogen.pipeline as P
        res = P.generate("a calm sea", model="wan-ti2v-5b")
        assert res.routed == "motion"
        assert "wan-ti2v-5b" in res.note and "16" in res.note
        assert res.meta["requested_model"] == "wan-ti2v-5b"

    def test_unknown_model_and_conflicts_raise(self, monkeypatch,
                                               models_root):
        _fake_cap(monkeypatch)
        import nomorals.media.videogen.pipeline as P
        with pytest.raises(VideogenError, match="unknown video model"):
            P.generate("x", model="nope-model")
        with pytest.raises(VideogenError, match="conflicts"):
            P.generate("x", model="ltx-base", backend="wan")
        with pytest.raises(VideogenError, match="explicit"):
            P.generate("x", model="org/some-model")
        with pytest.raises(VideogenError, match="motion"):
            P.generate("x", backend="motion", model="ltx-base")

    def test_lora_backend_mismatch_raises(self, monkeypatch, models_root):
        _fake_cap(monkeypatch, backend="ltx")
        _mk_lora(models_root, "wan", "cinematic")
        import nomorals.media.videogen.pipeline as P
        with pytest.raises(VideogenError, match="cinematic"):
            P.generate("x", model="ltx-distilled", lora=["cinematic"])

    def test_capability_report_lists_registry(self, monkeypatch,
                                              models_root):
        _mk_lora(models_root, "ltx", "anime")
        import nomorals.media.videogen.pipeline as P
        report = P.capability_report()
        assert "ltx-distilled" in report and "wan-ti2v-5b" in report
        assert "anime" in report and "16GB" in report

    def test_hero_clip_with_selection(self, monkeypatch, models_root):
        P = _fake_cap(monkeypatch, backend="ltx")
        seen = {}

        def fake_neural(prompt, backend, **kw):
            seen.update(backend=backend, **kw)
            return "/tmp/hero.mp4"

        monkeypatch.setattr(P, "_neural_generate", fake_neural)
        res = P.request_hero_clip("a lighthouse", model="ltx-base")
        assert res["ok"] is True and res["path"] == "/tmp/hero.mp4"
        assert res["model"] == "ltx-base"
        assert seen["model_id"] == "Lightricks/LTX-Video"
        assert seen["distilled"] is False

    def test_hero_clip_bad_selection_is_honest_decline(
            self, monkeypatch, models_root):
        _fake_cap(monkeypatch)
        import nomorals.media.videogen.pipeline as P
        res = P.request_hero_clip("x", model="nope-model")
        assert res["ok"] is False and "nope-model" in res["reason"]


# ---------------------------------------------------------------------------
# Chaining overrides
# ---------------------------------------------------------------------------

def _tiny_clip(path: str, color: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:s=64x64:d=1:r=8", "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True)
    return path


class _FakeChainBackend:
    made: list = []

    def __init__(self, name, source, loras, distilled, clip):
        self.name = name
        self.model_id = source or (
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers" if name == "wan"
            else "Lightricks/LTX-Video-0.9.7-distilled")
        self.loras = loras
        self.distilled = distilled
        self._clip = clip
        self.calls: list = []
        _FakeChainBackend.made.append(self)

    def require(self):
        return None

    def generate(self, prompt, **kw):
        self.calls.append(kw)
        return self._clip

    def unload(self):
        pass


@needs_ffmpeg
class TestChainingOverrides:
    def test_scene_override_and_chain_default(self, monkeypatch, tmp_path,
                                             models_root):
        _neural_env(monkeypatch, vram=24.0)
        anime = _mk_lora(models_root, "ltx", "anime")
        import nomorals.media.videogen.chaining as CH
        clip = _tiny_clip(str(tmp_path / "c.mp4"), "navy")
        _FakeChainBackend.made = []
        monkeypatch.setattr(
            CH, "_make_backend",
            lambda name, source, loras, distilled: _FakeChainBackend(
                name, source, loras, distilled, clip))
        out = str(tmp_path / "film.mp4")
        report = CH.chain_scenes(
            [{"prompt": "open sea", "duration_s": 1},
             {"prompt": "neon alley", "duration_s": 1,
              "model": "ltx-distilled", "lora": ["anime"]}],
            out=out, transition="cut")
        assert os.path.isfile(out)
        assert report.scenes[0].engine == "wan"  # chain default (auto→wan)
        assert report.scenes[0].model == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        s2 = report.scenes[1]
        assert s2.engine == "ltx"
        assert s2.model == "Lightricks/LTX-Video-0.9.7-distilled"
        assert s2.loras == [{"name": "anime", "source": anime,
                             "weight": 1.0}]
        by_name = {b.name: b for b in _FakeChainBackend.made}
        assert set(by_name) == {"wan", "ltx"}
        assert by_name["ltx"].loras[0]["source"] == anime
        assert "model:" in report.summary()

    def test_unavailable_scene_override_raises_naming_scene(
            self, monkeypatch, tmp_path, models_root):
        _neural_env(monkeypatch, vram=10.0)  # LTX ok (8), Wan not (16)
        import nomorals.media.videogen.chaining as CH
        clip = _tiny_clip(str(tmp_path / "c.mp4"), "navy")
        monkeypatch.setattr(
            CH, "_make_backend",
            lambda name, source, loras, distilled: _FakeChainBackend(
                name, source, loras, distilled, clip))
        with pytest.raises(VideogenError, match="scene 1"):
            CH.chain_scenes(
                [{"prompt": "ok ltx", "duration_s": 1},
                 {"prompt": "too big", "duration_s": 1,
                  "model": "wan-ti2v-5b"}],
                out=str(tmp_path / "film.mp4"), backend="ltx",
                transition="cut")


# ---------------------------------------------------------------------------
# V0: genedit contract + sceneintel path
# ---------------------------------------------------------------------------

class TestGeneditContract:
    def test_text_to_shot_uses_real_api(self, monkeypatch, tmp_path):
        import nomorals.media.genedit as G
        import nomorals.media.videogen.pipeline as P
        seen = {}

        def fake_generate(prompt, **kw):
            seen.update(kw)
            return P.VideoResult(path="/tmp/shot.mp4", engine="ltx",
                                 routed="neural", note="n", prompt=prompt)

        monkeypatch.setattr(P, "generate", fake_generate)
        shot = G.text_to_shot("a robot waves", duration_s=3.0,
                              workdir=str(tmp_path))
        assert "out" in seen and "out_path" not in seen
        assert seen["out"].endswith(".mp4")
        assert shot.backend == "ltx" and shot.path == "/tmp/shot.mp4"
        assert shot.mode == "t2v"

    def test_image_to_shot_uses_out_kwarg(self, monkeypatch, tmp_path):
        import nomorals.media.genedit as G
        seen = {}

        class FakeLTX:
            def check(self):
                return {"available": True, "reason": "fake"}

            def generate(self, prompt, **kw):
                seen.update(kw)
                return kw["out"]

        import nomorals.media.videogen.ltx_backend as LB
        monkeypatch.setattr(LB, "LTXBackend", FakeLTX)
        img = tmp_path / "in.png"
        img.write_bytes(b"png")
        shot = G.image_to_shot(str(img), "waves move", duration_s=2.0)
        assert "out" in seen and "out_path" not in seen
        assert shot.backend == "ltx" and shot.mode == "i2v"

    @needs_ffmpeg
    def test_extend_shot_uses_out_kwarg(self, monkeypatch, tmp_path):
        import nomorals.media.genedit as G
        src = _tiny_clip(str(tmp_path / "seed.mp4"), "red")
        seen = {}

        class FakeLTX:
            def check(self):
                return {"available": True, "reason": "fake"}

            def generate(self, prompt, **kw):
                seen.update(kw)
                shutil.copyfile(src, kw["out"])
                return kw["out"]

        import nomorals.media.videogen.ltx_backend as LB
        monkeypatch.setattr(LB, "LTXBackend", FakeLTX)
        shot = G.extend_shot(src, "keep going", seconds=2.0,
                             workdir=str(tmp_path))
        assert "out" in seen and "out_path" not in seen
        assert seen["mode"] == "extend"
        assert shot.mode == "extend" and shot.backend == "ltx"


# ---------------------------------------------------------------------------
# Surfaces: chat + CLI
# ---------------------------------------------------------------------------

class TestSurfaces:
    def test_chat_video_gen_model_lora_parse(self, monkeypatch):
        import nomorals.media.videogen as VG
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
        seen = {}

        def fake_generate(prompt, **kw):
            seen.update(prompt=prompt, **kw)
            return types.SimpleNamespace(
                message=lambda: "rendered ok")

        monkeypatch.setattr(VG, "generate", fake_generate)
        out = RuntimeMediaMixin()._video_studio_gen(
            '"a neon alley" --model wan-ti2v-5b --lora anime:0.8 '
            '--lora neon --seed 7')
        assert out == "rendered ok"
        assert seen["model"] == "wan-ti2v-5b"
        assert seen["lora"] == ["anime:0.8", "neon"]
        assert seen["seed"] == 7

    def test_chat_shorts_hero_ok_and_decline(self, monkeypatch):
        import nomorals.media.videogen as VG
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
        seen = {}

        def fake_hero(prompt, **kw):
            seen.update(prompt=prompt, **kw)
            return {"ok": True, "path": "/tmp/hero.mp4", "backend": "ltx",
                    "model": "ltx-distilled", "loras": []}

        monkeypatch.setattr(VG, "request_hero_clip", fake_hero)
        mixin = RuntimeMediaMixin()
        out = mixin._control_shorts('hero "a neon alley" --seconds 3')
        assert "/tmp/hero.mp4" in out and "ltx" in out
        assert seen["prompt"] == "a neon alley"
        assert seen["duration_s"] == 3.0

        monkeypatch.setattr(VG, "request_hero_clip", lambda p, **kw: {
            "ok": False, "reason": "no CUDA GPU detected",
            "suggestion": "use motion_studio.kenburns"})
        out = mixin._control_shorts('hero "a wave"')
        assert "unavailable" in out and "no CUDA" in out

    def test_cli_video_generate_flags_parse(self):
        from nomorals.cmdline import parser as parser_mod
        ns = parser_mod._parser().parse_args(
            ["video", "generate", "a cat", "--model", "wan-ti2v-5b",
             "--lora", "anime:0.8", "--lora", "neon"])
        assert ns.video_action == "generate"
        assert ns.model == "wan-ti2v-5b"
        assert ns.lora == ["anime:0.8", "neon"]

    def test_cli_shorts_hero_parse_and_handler(self, monkeypatch):
        from nomorals.cmdline import parser as parser_mod
        from nomorals.cmdline.commands import shorts as shorts_cmd
        assert "hero" in shorts_cmd.HANDLERS
        ns = parser_mod._parser().parse_args(
            ["shorts", "hero", "a breaking wave", "--model", "ltx-base",
             "--lora", "anime:0.5"])
        assert ns.shorts_action == "hero" and ns.model == "ltx-base"
        assert ns.lora == ["anime:0.5"]

        import nomorals.media.videogen as VG
        monkeypatch.setattr(VG, "request_hero_clip", lambda p, **kw: {
            "ok": True, "path": "/tmp/hero.mp4", "backend": "ltx",
            "model": "ltx-base", "loras": ["anime"]})
        assert shorts_cmd._cmd_hero(ns, None) == 0
        monkeypatch.setattr(VG, "request_hero_clip", lambda p, **kw: {
            "ok": False, "reason": "no CUDA", "suggestion": "plan B"})
        assert shorts_cmd._cmd_hero(ns, None) == 1

    def test_cli_generate_forwards_model_lora(self, monkeypatch):
        import argparse
        import nomorals.media.videogen as VG
        from nomorals.cmdline.commands import video as video_cmd
        seen = {}

        def fake_generate(prompt, **kw):
            seen.update(prompt=prompt, **kw)
            return types.SimpleNamespace(message=lambda: "ok")

        monkeypatch.setattr(VG, "generate", fake_generate)
        ns = argparse.Namespace(prompt="a cat", backend="auto", mode="t2v",
                                image="", seconds=5.0, seed=0, out="",
                                model="ltx-base", lora=["anime:0.8"])
        assert video_cmd._cmd_generate(ns, None) == 0
        assert seen["model"] == "ltx-base"
        assert seen["lora"] == ["anime:0.8"]
