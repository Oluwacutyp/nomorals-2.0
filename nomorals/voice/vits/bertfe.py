"""TinyBERT context frontend.

Faithful port of upstream ``bertfe.py``: ``huawei-noah/TinyBERT_General_4L_312D``
loaded through transformers, single-sentence inference returning
``(encoded_layers [1, tokens, 312], pooled [1, 312])``. The transformers
import is lazy — constructing the frontend without transformers (or
without network access to the model hub) raises a named error instead
of failing at import time. Weights are cached in Devon's voice model
store (see ``model_files``) when a ``cache_dir`` is given.
"""

import os

DEFAULT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"


class BERTFrontEnd:
    def __init__(self, is_cuda=False, model_name=DEFAULT_MODEL, cache_dir=None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "the TinyBERT frontend needs torch; it is not installed"
            ) from exc
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "the TinyBERT frontend needs the transformers package; "
                "it is not installed"
            ) from exc
        self.is_cuda = is_cuda
        self.model_name = model_name
        load_kwargs = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        try:
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        except (OSError, ValueError) as exc:
            raise OSError(
                f"TinyBERT model {model_name!r} could not be loaded (network or cache): {exc}. "
                "Run once with network access, or place the model in the "
                "voice model store first."
            ) from exc
        if is_cuda:
            self.model = self.model.cuda()
        self._torch = torch

    def infer(self, in_txt):
        """Single-sentence inference: (hidden states, pooled)."""
        inputs = self.tokenizer(in_txt, return_tensors="pt")
        if self.is_cuda:
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
        with self._torch.no_grad():
            encoded_layers, pooled = self.model(**inputs, return_dict=False)
        return encoded_layers, pooled


def default_cache_dir():
    """Voice model store directory for TinyBERT weights."""
    try:
        from nomorals.voice.model_files import voice_model_dir
        return voice_model_dir()
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".nomorals", "voice", "models")
