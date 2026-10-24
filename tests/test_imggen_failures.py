"""Pins for Pass #112: the imggen failure ledger.

Covers the ledger contract (cap, unprintable exceptions, rendering) and
every converted site in the native image generation engine, with the
fallback defaults held exactly.
"""

from __future__ import annotations

import sys
import types

import pytest

from nomorals.media.imggen import failures as F
from nomorals.media.imggen import data as imgdata
from nomorals.media.imggen import studio as imgstudio


@pytest.fixture(autouse=True)
def _clean_ledger():
    F.clear_failures()
    yield
    F.clear_failures()


# ---------------------------------------------------------------- ledger


def test_ledger_basics_and_bound():
    for i in range(250):
        F.record_failure(f"site.{i % 3}", ValueError(f"boom {i}"))
    assert F.failure_count() == 250
    recent = F.recent_failures()
    assert len(recent) == 20
    assert recent[0].site == "site.0"  # 249 % 3 == 0, newest first
    assert len(F.recent_failures(1000)) == 200  # ring cap, count uncapped


def test_ledger_unprintable_exc_and_render():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    F.record_failure("pin.bad", BadStr())
    entry = F.recent_failures()[0]
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message
    rendered = F.format_failures()
    assert "pin.bad" in rendered and "BadStr" in rendered
    F.clear_failures()
    assert F.format_failures() == "no imggen failures recorded."
    assert F.failure_count() == 0


def test_direct_import_surface():
    from nomorals.media.imggen.failures import (
        Failure,
        clear_failures,
        failure_count,
        format_failures,
        recent_failures,
        record_failure,
    )

    assert callable(record_failure)
    assert callable(recent_failures)
    assert callable(failure_count)
    assert callable(clear_failures)
    assert callable(format_failures)
    assert Failure.__dataclass_params__.frozen


# ------------------------------------------------------- converted sites


def test_vision_caption_failure_recorded_and_none_held():
    def _boom(path, prompt):
        raise RuntimeError("vision stack down")

    assert imgdata.caption_with_devon_vision("x.png", see_fn=_boom) is None
    entries = F.recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "data.caption_with_devon_vision"
    assert entries[0].exc_type == "RuntimeError"


def test_vision_caption_healthy_records_nothing():
    def _see(path, prompt):
        return {"description": "a red cat on a roof"}

    assert imgdata.caption_with_devon_vision("x.png", see_fn=_see) == "a red cat on a roof"
    assert F.recent_failures() == []


def test_manifest_verify_failure_recorded_and_skipped_held(tmp_path):
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8), color="red").save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image at all")
    summary = imgdata.build_dataset_manifest(str(tmp_path), use_vision=False)
    assert summary["images"] == 1
    assert summary["skipped"] == 1
    entries = F.recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "data.build_dataset_manifest.verify"


def test_profile_kind_failure_recorded_and_laptop_held(monkeypatch):
    from nomorals.core import profiles

    def _boom():
        raise RuntimeError("profiles broken")

    monkeypatch.setattr(profiles, "get_profile_kind", _boom)
    assert imgstudio._profile_kind() == "laptop"
    entries = F.recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "studio._profile_kind"
    assert entries[0].exc_type == "RuntimeError"


def test_profile_kind_healthy_records_nothing():
    assert imgstudio._profile_kind() in ("termux", "laptop", "workstation")
    assert F.recent_failures() == []


def test_optimizer_state_failure_recorded_and_weights_held(monkeypatch, tmp_path):
    import nomorals.media.imggen as pkg

    payload = {
        "format": "devon-imggen-1",
        "model_state": {"w": 1},
        "optimizer_state": {"m": 2},
        "torch_rng": b"rng",
    }
    torch_mod = types.ModuleType("torch")
    torch_mod.load = lambda path, map_location=None, weights_only=False: payload
    torch_mod.set_rng_state = lambda state: None
    torch_mod.no_grad = lambda: (lambda fn: fn)  # EMA decorates methods at class time
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    nn_mod = types.ModuleType("torch.nn")

    class _FakeModule:
        pass

    nn_mod.Module = _FakeModule
    optim_mod = types.ModuleType("torch.optim")

    class _FakeOptimizer:
        pass

    optim_mod.Optimizer = _FakeOptimizer
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _FakeLoader:
        pass

    data_mod.DataLoader = _FakeLoader
    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.utils = utils_mod
    for name, mod in (("torch", torch_mod), ("torch.nn", nn_mod),
                      ("torch.optim", optim_mod), ("torch.utils", utils_mod),
                      ("torch.utils.data", data_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pkg, "TORCH_AVAILABLE", True)

    import nomorals.media.imggen.train as train_mod

    class _Model:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    class _Optimizer:
        def load_state_dict(self, state):
            raise RuntimeError("optimizer shape changed for real")

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    model = _Model()
    out = train_mod.load_checkpoint(str(ckpt), model, optimizer=_Optimizer())
    assert out is payload  # weights still valid, resume proceeds
    assert model.loaded == {"w": 1}
    entries = F.recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "train.load_checkpoint.optimizer_state"


# ------------------------------------------------------------- scan pin


def test_subsystem_scan_remainder_is_exactly_the_deliberate_exclusion():
    """imggen's only remaining finding is diffusion._as_numpy's ImportError
    probe — a hot-path availability probe (called per sampling step), where
    torch-absent is the designed numpy fallback. Asserted by rule + file +
    context, never by line number."""
    from nomorals.tools import error_scan as es

    report = es.scan(["nomorals/media/imggen"])
    by_file = {}
    for f in report.findings:
        by_file.setdefault(f.file.split("imggen/")[-1], []).append(f)
    assert "failures.py" not in by_file
    assert set(by_file) == {"diffusion.py"}
    remaining = by_file["diffusion.py"]
    assert len(remaining) == 1
    assert remaining[0].rule == "E103"
    assert remaining[0].context.strip().startswith("except ImportError")
