"""The precomputed store — where conditions and latents live on disk.

Upstream ``Trainer.prepare_precomputations`` (finetrainers/trainer.py)
writes, per dataset item, two files into::

    {data_root}/{model_name}_{cleaned_model_id}_precomputed/
        conditions/conditions-{process}-{index}.pt
        latents/latents-{process}-{index}.pt

and ``PrecomputedDataset`` (finetrainers/dataset.py) reads them back
by sorted filename order, asserting the two sides have equal counts.
This module ports that layout, the naming, the
``should_perform_precomputation`` decision (finetrainers/utils/data_utils.py),
and the pair loading — with one profile adaptation, recorded in the
survey: the tensor codec is a protocol.

* :class:`TorchTensorIO` is upstream's format exactly: ``torch.save`` /
  ``torch.load(..., weights_only=True)``, ``.pt`` files. It is the
  default whenever torch imports.
* :class:`NumpyTensorIO` is the Devon-native fallback for machines
  without torch (this box, Termux, EC2's CPU-only paths): the same dict
  keys, the same filenames, but ``.npz`` files holding numpy arrays.
  A store never mixes codecs — the codec is chosen once per store and
  recorded in the manifest, so a reader always knows what it is opening.

The layout, pairing, counting, and validation are codec-agnostic, so a
store precomputed on a GPU box (torch) and a store precomputed here
(numpy) are interchangeable at every layer except the tensor file
reader — which is exactly the layer the codec names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ...core.logging_setup import get_logger
from .constants import (
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
    PRECOMPUTED_MANIFEST_NAME,
)
from .errors import ConditioningError, ConditioningUnavailable

__all__ = [
    "NumpyTensorIO",
    "PrecomputedStore",
    "TensorIO",
    "TorchTensorIO",
    "precomputation_dir",
    "precomputation_dir_name",
    "require_torch",
    "select_tensor_io",
    "should_perform_precomputation",
    "string_to_filename",
]

_log = get_logger(__name__)


def string_to_filename(text: str) -> str:
    """Verbatim port of ``finetrainers/utils/file_utils.py``.

    Model ids become directory-safe by replacing ``space / : . , ; ! ?``
    with ``-`` — so ``Lightricks/LTX-Video`` becomes
    ``Lightricks-LTX-Video`` in the precomputation directory name.
    """
    return (
        text.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )


def precomputation_dir_name(model_name: str, model_id: str) -> str:
    """``{model_name}_{cleaned_model_id}_precomputed`` — upstream's rule."""
    return f"{model_name}_{string_to_filename(model_id)}_{PRECOMPUTED_DIR_NAME}"


def precomputation_dir(
    data_root: str | Path, model_name: str, model_id: str
) -> Path:
    """The precomputation directory for a (model, checkpoint) pair."""
    return Path(data_root) / precomputation_dir_name(model_name, model_id)


def should_perform_precomputation(
    precomputation_dir_path: str | Path,
) -> bool:
    """Whether the precompute pass must run for this directory.

    Verbatim port of ``finetrainers/utils/data_utils.py`` (plus the
    ``.npz`` half of the codec adaptation): if both subdirectories exist
    and hold equal, non-zero file counts, the store is complete and the
    pass is skipped; if the counts disagree, the stale files are deleted
    and the pass re-runs; otherwise the pass runs. Returns ``True`` when
    the caller must (re)compute.
    """
    directory = Path(precomputation_dir_path)
    conditions_dir = directory / PRECOMPUTED_CONDITIONS_DIR_NAME
    latents_dir = directory / PRECOMPUTED_LATENTS_DIR_NAME
    if conditions_dir.exists() and latents_dir.exists():
        num_conditions = len(
            list(conditions_dir.glob("*.pt"))
            + list(conditions_dir.glob("*.npz"))
        )
        num_latents = len(
            list(latents_dir.glob("*.pt")) + list(latents_dir.glob("*.npz"))
        )
        if num_conditions != num_latents:
            _log.warning(
                f"Number of precomputed conditions ({num_conditions}) does "
                f"not match number of precomputed latents ({num_latents})."
                f"Cleaning up precomputed directories and re-running "
                f"precomputation."
            )
            for file in list(conditions_dir.glob("*.pt")) + list(
                conditions_dir.glob("*.npz")
            ):
                file.unlink()
            for file in list(latents_dir.glob("*.pt")) + list(
                latents_dir.glob("*.npz")
            ):
                file.unlink()
            return True
        if num_conditions > 0:
            _log.info(
                f"Found {num_conditions} precomputed conditions and latents."
            )
            return False
    _log.info("Precomputed data not found. Running precomputation.")
    return True


def require_torch():
    """Import torch, or fail honestly naming what provides it."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ConditioningUnavailable(
            "torch is not installed on this machine, so the .pt tensor "
            "codec is unavailable. Install torch (>=2.5.1 per upstream "
            "requirements.txt) or use the numpy codec "
            "(NumpyTensorIO), which stores the same conditions as .npz."
        ) from exc
    import torch

    return torch


class TensorIO(Protocol):
    """One tensor codec: save/load a ``{str: array-like}`` dict."""

    suffix: str

    def save(self, path: Path, payload: dict[str, Any]) -> None: ...

    def load(self, path: Path) -> dict[str, Any]: ...


class TorchTensorIO:
    """Upstream's codec: ``torch.save`` / ``torch.load``, ``.pt`` files."""

    suffix = ".pt"

    def save(self, path: Path, payload: dict[str, Any]) -> None:
        torch = require_torch()
        torch.save(payload, path.as_posix())

    def load(self, path: Path) -> dict[str, Any]:
        torch = require_torch()
        return torch.load(path, map_location="cpu", weights_only=True)


class NumpyTensorIO:
    """Devon-native codec: ``numpy.savez`` / ``numpy.load``, ``.npz`` files.

    Used when torch is absent. Array values round-trip as arrays;
    scalar values (``num_frames``, ``height``, ``width``) round-trip as
    Python scalars — 0-d arrays are converted back with ``.item()`` on
    load, so a payload read back compares equal to the payload written.
    """

    suffix = ".npz"

    def save(self, path: Path, payload: dict[str, Any]) -> None:
        arrays = {key: np.asarray(value) for key, value in payload.items()}
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    def load(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            out: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                out[key] = value.item() if value.ndim == 0 else value
            return out


def select_tensor_io() -> TensorIO:
    """The best codec available on this machine: torch, else numpy."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return NumpyTensorIO()
    return TorchTensorIO()


class PrecomputedStore:
    """A precomputed (conditions, latents) store on disk.

    Mirrors upstream ``PrecomputedDataset`` (finetrainers/dataset.py):
    with ``model_name`` + ``cleaned_model_id`` the store lives under
    ``{data_root}/{model_name}_{cleaned_model_id}_precomputed/``; without
    them it lives under ``{data_root}/precomputed/``. Pairs are matched
    by sorted filename order and the two sides must have equal counts —
    upstream's assert, with upstream's message.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        model_name: str | None = None,
        cleaned_model_id: str | None = None,
        tensor_io: TensorIO | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        if model_name and cleaned_model_id:
            directory = self.data_root / (
                f"{model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"
            )
        else:
            directory = self.data_root / PRECOMPUTED_DIR_NAME
        self.directory = directory
        self.latents_path = directory / PRECOMPUTED_LATENTS_DIR_NAME
        self.conditions_path = directory / PRECOMPUTED_CONDITIONS_DIR_NAME
        self.tensor_io = tensor_io or select_tensor_io()
        self.latent_names = sorted(
            name for name in os.listdir(self.latents_path)
            if name.endswith(self.tensor_io.suffix)
        ) if self.latents_path.exists() else []
        self.condition_names = sorted(
            name for name in os.listdir(self.conditions_path)
            if name.endswith(self.tensor_io.suffix)
        ) if self.conditions_path.exists() else []
        if len(self.latent_names) != len(self.condition_names):
            raise ConditioningError(
                "Number of captions and videos do not match"
            )

    def __len__(self) -> int:
        return len(self.latent_names)

    def pairs(self) -> list[tuple[Path, Path]]:
        """(latent path, condition path) per index, in sorted order."""
        return [
            (
                self.latents_path / self.latent_names[index],
                self.conditions_path / self.condition_names[index],
            )
            for index in range(len(self))
        ]

    def load_pair(self, index: int) -> dict[str, Any]:
        """Upstream ``PrecomputedDataset.__getitem__``: one pair's dicts."""
        latent_path, condition_path = self.pairs()[index]
        return {
            "latent_conditions": self.tensor_io.load(latent_path),
            "text_conditions": self.tensor_io.load(condition_path),
        }

    @property
    def manifest_path(self) -> Path:
        return self.directory / PRECOMPUTED_MANIFEST_NAME

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        """Record provenance next to the store (a Devon addition)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "codec": self.tensor_io.suffix.lstrip("."),
            "latents": len(self.latent_names),
            "conditions": len(self.condition_names),
            **manifest,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return self.manifest_path

    def read_manifest(self) -> dict[str, Any]:
        with open(self.manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ConditioningError(
                f"precomputed manifest {self.manifest_path} must hold an "
                f"object, got {type(data).__name__}."
            )
        return data
