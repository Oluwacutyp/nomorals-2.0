"""VITS TTS backend: serve a fine-tuned VITS voice through ``UniversalTTS``.

A fine-tuned VITS voice is an exported TorchScript model
(``<name>.pt``) plus its config (``<name>.json``) in the VITS model
dir (``~/.nomorals/voice/vits/``, overridable with ``NM_VITS_DIR``).
The backend is available only when torch is importable AND at least
one such model exists — otherwise ``available_backends()`` skips it
and construction raises a ``RuntimeError`` naming the dir and how to
produce a model (``voice/prepare_vits.py`` + ``vits.train`` + export).
"""

import os

from .infer import VITSFrontend


def vits_model_dir() -> str:
    d = os.environ.get("NM_VITS_DIR",
                       os.path.expanduser("~/.nomorals/voice/vits"))
    os.makedirs(d, exist_ok=True)
    return d


def list_vits_models() -> list[str]:
    """Names of export pairs (``<name>.pt`` + ``<name>.json``) in the model dir."""
    d = vits_model_dir()
    names = []
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return names
    for entry in entries:
        if not entry.endswith(".pt"):
            continue
        name = entry[:-3]
        if os.path.isfile(os.path.join(d, name + ".json")):
            names.append(name)
    return names


def _vits_available() -> bool:
    try:
        from ._torch import torch_available
        if not torch_available():
            return False
    except ImportError:
        return False
    return bool(list_vits_models())


class VITSBackend:
    """Fine-tuned VITS voice (iSTFTNet + Avocodo generator)."""

    name = "vits"
    supports_native_tags = False
    supports_cloning = True  # a fine-tune IS the owner's (or a speaker's) voice
    supports_streaming = False  # whole-utterance only
    sample_rate = 22050

    def __init__(self, model: str = "") -> None:
        self.model_name = model or (list_vits_models()[0] if list_vits_models() else "")
        if not self.model_name:
            raise RuntimeError(
                f"no VITS voice found — export a fine-tuned model into {vits_model_dir()}/ "
                "(<name>.pt + <name>.json); see nomorals/voice/vits/train.py"
            )
        d = vits_model_dir()
        self.model_path = os.path.join(d, self.model_name + ".pt")
        self.config_path = os.path.join(d, self.model_name + ".json")
        self.frontend = VITSFrontend().load(self.model_path, self.config_path)
        self.sample_rate = self.frontend.sampling_rate

    def synthesize(self, text: str, voice=None, *, instruct: str = ""):
        """Synthesize text; ``instruct`` (when given) is the emotion text."""
        return self.frontend.infer(text, in_em=instruct or "")
