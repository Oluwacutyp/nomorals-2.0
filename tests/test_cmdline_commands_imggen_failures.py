"""Pins for Pass tail-111: nomorals/cmdline/commands/imggen.py — 3 exclusions.

137 E104 _cmd_imggen_train: the vision-captioner import (agents.seer)
    fails -> see_fn=None, the dataset manifest falls back to filename
    captions, and the printed summary DISCLOSES it (0 vision-captioned).
155 E104 _cmd_imggen_train: get_profile_kind import/call fails ->
    profile="laptop", the documented mid-tier sizing default (base_ch
    64). Profile-detection degradation, same family as the core/profile
    pins (tail-104). Flagged-accepted remainder (no noqa on the guard).
204 E103 _cmd_imggen_lora_train: the same probe fails -> pass, so the
    termux refusal is skipped and LoRA proceeds. Held as an exclusion:
    interactive owner-run command, detection-default family; the
    termux control below proves the refusal still fires when the probe
    works. Flagged-accepted remainder (no noqa on the guard).

The command's heavy imports are faked in sys.modules so the pins run
offline with no torch; every fake is restored by monkeypatch.
"""

from __future__ import annotations

import sys
import types

import pytest

import nomorals.media.imggen as imggen_pkg
from nomorals.cmdline.commands import imggen as imggen_cmd


def _raising_module(name: str, msg: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    def _ga(attr, _m=msg):
        raise RuntimeError(_m)

    mod.__getattr__ = _ga
    return mod


class _Rig:
    """Fake imggen engine + torch, installed via monkeypatch."""

    def __init__(self, monkeypatch):
        self.manifest_see: list = []
        self.unet_kwargs: list = []

        rig = self

        def fake_manifest(data, see_fn=None):
            rig.manifest_see.append(see_fn)
            return {"images": 3,
                    "vision_captions": 3 if see_fn else 0,
                    "filename_captions": 0 if see_fn else 3,
                    "manifest": "m.json"}

        class FakeDS:
            def __init__(self, *a, **k):
                pass

        class FakeUNet:
            def __init__(self, **k):
                rig.unet_kwargs.append(k)

        class FakeEnc:
            def __init__(self, **k):
                pass

        class FakeCfg:
            def __init__(self, **k):
                pass

        def fake_train(*a, **k):
            return {"steps": 7, "final_loss": 0.25,
                    "final_checkpoint": "ck.pt"}

        class FakePipe:
            unet = object()
            text_encoder = object()

        data_mod = types.ModuleType("nomorals.media.imggen.data")
        data_mod.ImageFolderDataset = FakeDS
        data_mod.build_dataset_manifest = fake_manifest
        pipe_mod = types.ModuleType("nomorals.media.imggen.pipeline")
        pipe_mod.TextEncoder = FakeEnc
        pipe_mod.load_native_checkpoint = lambda base: FakePipe()
        train_mod = types.ModuleType("nomorals.media.imggen.train")
        train_mod.TrainConfig = FakeCfg
        train_mod.train = fake_train
        train_mod.train_text_conditional = fake_train
        unet_mod = types.ModuleType("nomorals.media.imggen.unet")
        unet_mod.UNet = FakeUNet
        unet_mod.count_parameters = lambda m: 42
        lora_mod = types.ModuleType("nomorals.media.imggen.lora")
        lora_mod.inject_lora = lambda unet, r, alpha: ["mod1"]
        torch_mod = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_data = types.ModuleType("torch.utils.data")
        torch_data.DataLoader = lambda *a, **k: object()
        torch_mod.utils = torch_utils

        for name, mod in [
            ("nomorals.media.imggen.data", data_mod),
            ("nomorals.media.imggen.pipeline", pipe_mod),
            ("nomorals.media.imggen.train", train_mod),
            ("nomorals.media.imggen.unet", unet_mod),
            ("nomorals.media.imggen.lora", lora_mod),
            ("torch", torch_mod),
            ("torch.utils", torch_utils),
            ("torch.utils.data", torch_data),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(imggen_pkg, "TORCH_AVAILABLE", True)

        self.kind = ["laptop"]
        profiles_ok = types.ModuleType("nomorals.core.profiles")
        profiles_ok.get_profile_kind = lambda pinned="": self.kind[0]
        self.profiles_ok = profiles_ok
        self.see_sentinel = lambda *a, **k: "cap"  # noqa: E731
        seer_ok = types.ModuleType("nomorals.agents.seer")
        seer_ok.see = self.see_sentinel
        self.seer_ok = seer_ok

    def install_profiles(self, monkeypatch, mode: str):
        if mode == "boom":
            monkeypatch.setitem(
                sys.modules, "nomorals.core.profiles",
                _raising_module("nomorals.core.profiles", "profiles dead"))
        else:
            monkeypatch.setitem(sys.modules, "nomorals.core.profiles",
                                self.profiles_ok)

    def install_seer(self, monkeypatch, mode: str):
        if mode == "boom":
            monkeypatch.setitem(
                sys.modules, "nomorals.agents.seer",
                _raising_module("nomorals.agents.seer", "seer dead"))
        else:
            monkeypatch.setitem(sys.modules, "nomorals.agents.seer",
                                self.seer_ok)


TRAIN_ARGS = types.SimpleNamespace(
    data="ds", size=64, batch=2, run="r1", timesteps=10, lr=0.001,
    steps=5, checkpoint_every=2, seed=0)
LORA_ARGS = types.SimpleNamespace(
    base="ck.pt", rank=4, alpha=8, data="ds", size=64, batch=2, run="r1",
    lr=0.001, steps=5, checkpoint_every=2, seed=0)


@pytest.fixture()
def rig(monkeypatch):
    return _Rig(monkeypatch)


# ---------------------------------------------------------------- 137

def test_train_seer_crash_falls_back_to_filename_captions(rig, monkeypatch,
                                                          capsys):
    rig.install_seer(monkeypatch, "boom")
    rig.install_profiles(monkeypatch, "ok")
    assert imggen_cmd._cmd_imggen_train(TRAIN_ARGS, None) == 0
    assert rig.manifest_see == [None]
    out = capsys.readouterr().out
    assert "0 vision-captioned, 3 filename-captioned" in out


def test_train_seer_healthy_is_passed_through(rig, monkeypatch, capsys):
    rig.install_seer(monkeypatch, "ok")
    rig.install_profiles(monkeypatch, "ok")
    assert imggen_cmd._cmd_imggen_train(TRAIN_ARGS, None) == 0
    assert rig.manifest_see == [rig.see_sentinel]
    assert "3 vision-captioned" in capsys.readouterr().out


# ---------------------------------------------------------------- 155

def test_train_profile_crash_defaults_to_laptop_sizing(rig, monkeypatch,
                                                       capsys):
    rig.install_seer(monkeypatch, "ok")
    rig.install_profiles(monkeypatch, "boom")
    assert imggen_cmd._cmd_imggen_train(TRAIN_ARGS, None) == 0
    assert rig.unet_kwargs[0]["base_channels"] == 64
    assert "termux profile" not in capsys.readouterr().out


def test_train_profile_termux_control_gets_tiny_sizing(rig, monkeypatch,
                                                       capsys):
    rig.install_seer(monkeypatch, "ok")
    rig.install_profiles(monkeypatch, "ok")
    rig.kind[0] = "termux"
    assert imggen_cmd._cmd_imggen_train(TRAIN_ARGS, None) == 0
    assert rig.unet_kwargs[0]["base_channels"] == 32
    assert "termux profile: tiny" in capsys.readouterr().out


# ---------------------------------------------------------------- 204

def test_lora_profile_crash_skips_refusal_and_proceeds(rig, monkeypatch,
                                                       capsys):
    rig.install_profiles(monkeypatch, "boom")
    assert imggen_cmd._cmd_imggen_lora_train(LORA_ARGS, None) == 0
    assert "LoRA injected" in capsys.readouterr().out


def test_lora_profile_termux_control_refuses(rig, monkeypatch, capsys):
    rig.install_profiles(monkeypatch, "ok")
    rig.kind[0] = "termux"
    assert imggen_cmd._cmd_imggen_lora_train(LORA_ARGS, None) == 1
    assert "needs at least a laptop" in capsys.readouterr().err


def test_lora_profile_laptop_control_proceeds(rig, monkeypatch, capsys):
    rig.install_profiles(monkeypatch, "ok")
    assert imggen_cmd._cmd_imggen_lora_train(LORA_ARGS, None) == 0
    assert "LoRA injected" in capsys.readouterr().out
