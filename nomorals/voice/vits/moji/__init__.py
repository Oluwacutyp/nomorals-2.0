"""TorchMoji emotion frontend (inference only).

The weights (``pytorch_model.bin``) are not shipped in the upstream
clone and are not vendored here; ``TorchMoji()`` raises a
``FileNotFoundError`` naming the exact path until they are placed at
``nomorals/voice/vits/data/pytorch_model.bin``. The tokenizer works
without torch.
"""

from .global_variables import PRETRAINED_PATH, VOCAB_PATH

__all__ = ["TorchMoji", "PRETRAINED_PATH", "VOCAB_PATH"]


def __getattr__(name):
    if name == "TorchMoji":
        from .._torch import require_torch
        require_torch("the TorchMoji emotion frontend")
        from .moji import TorchMoji
        return TorchMoji
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
