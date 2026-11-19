"""VITS fine-tuning port (storytold/vits-finetuning, MIT).

Two halves:

* Numpy/stdlib (runs anywhere): text front end (``text``), configs
  (``configs``), audio (``audio``), math (``mathutils``), dataset
  pipeline (``data``), preparation (``preprocess`` / ``prepare``).
* Torch (lazy, honest error without torch): the model (``models``),
  layers (``modules`` / ``attentions`` / ``transforms`` / ``commons``),
  STFT + mel (``stft`` / ``mel_processing``), losses (``losses``),
  checkpoint helpers (``train_utils``), trainer (``train``),
  inference frontend (``infer``), TinyBERT (``bertfe``) and TorchMoji
  (``moji``) context frontends, and the TTS backend (``backend``).

Nothing here imports torch at package import time.
"""

from . import audio, configs, data, mathutils, prepare, preprocess
from .configs import HParams, list_configs, load_config, validate
from .prepare import prepare_run

__all__ = [
    "audio", "configs", "data", "mathutils", "prepare", "preprocess",
    "HParams", "list_configs", "load_config", "validate", "prepare_run",
]


def __getattr__(name):
    """Lazy torch half: imported on first use, with the torch gate."""
    _torch_names = {
        "VITSTrainer": ("train", "VITSTrainer"),
        "train_from_config": ("train", "train_from_config"),
        "VITSFrontend": ("infer", "VITSFrontend"),
        "export_torchscript": ("infer", "export_torchscript"),
        "VITSBackend": ("backend", "VITSBackend"),
        "vits_model_dir": ("backend", "vits_model_dir"),
        "list_vits_models": ("backend", "list_vits_models"),
        "SynthesizerTrn": ("models", "SynthesizerTrn"),
    }
    if name in _torch_names:
        from ._torch import require_torch
        require_torch(f"nomorals.voice.vits.{name}")
        import importlib
        module_name, attr = _torch_names[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
