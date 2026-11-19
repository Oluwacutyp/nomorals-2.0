"""Precompute orchestration — a port of ``Trainer.prepare_precomputations``.

Upstream (finetrainers/trainer.py) runs the precompute pass in two
sweeps over the dataset: first every prompt through
``prepare_conditions`` (text encoder), then every video through
``prepare_latents`` (VAE), writing
``conditions-0-{i}.pt`` / ``latents-0-{i}.pt`` into the store and
replacing the training dataloader with a :class:`PrecomputedStore`.
This module ports that orchestration as :class:`Precomputer`, single
process (upstream's ``accelerator.process_index`` is always ``0``
here), with the encoder calls injected so the same code runs against
real GPU encoders and against deterministic stand-ins.

Faithful rules, all ported:

* precompute only runs with ``batch_size == 1`` (upstream's
  ``ValueError``, same message);
* caption dropout with the ``"empty"`` technique warns that it is not
  supported with precomputation yet (upstream's warning, same text);
* an already-complete store is reused, never recomputed
  (``should_perform_precomputation``); a mismatched store is cleaned
  and re-run;
* conditions are written first, latents second, in dataset order.

Devon additions: a ``manifest.json`` is written next to the store
recording the model, the codec, and the counts; and the report returned
by :meth:`Precomputer.run` says whether the store was reused or built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core.logging_setup import get_logger
from .configs import ConditioningConfig
from .encoders import get_condition_spec, prepare_conditions
from .errors import ConditioningError
from .formats import CaptionedVideos
from .precomputed import (
    PrecomputedStore,
    TensorIO,
    precomputation_dir,
    select_tensor_io,
    should_perform_precomputation,
    string_to_filename,
)

__all__ = ["PrecomputeReport", "Precomputer"]

_log = get_logger(__name__)

#: ``encode_text(prompts, max_sequence_length) -> dict`` — see
#: :mod:`nomorals.training.conditioning.encoders`.
EncodeText = Callable[..., dict[str, Any]]

#: ``encode_video(video_path) -> dict`` — must return at least a
#: ``"latents"`` array (``[B, C, F, H, W]``) and, for LTX, the stored
#: ``latents_mean`` / ``latents_std`` plus ``num_frames`` / ``height`` /
#: ``width`` (upstream's ``prepare_latents(precompute=True)`` returns
#: exactly those).
EncodeVideo = Callable[..., dict[str, Any]]


@dataclass
class PrecomputeReport:
    """What :meth:`Precomputer.run` did, in plain data."""

    store_dir: Path
    model_name: str
    model_id: str
    codec: str
    reused: bool
    conditions: int = 0
    latents: int = 0
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_dir": str(self.store_dir),
            "model_name": self.model_name,
            "model_id": self.model_id,
            "codec": self.codec,
            "reused": self.reused,
            "conditions": self.conditions,
            "latents": self.latents,
            "manifest_path": None
            if self.manifest_path is None
            else str(self.manifest_path),
        }


@dataclass
class Precomputer:
    """Run the precompute pass for one dataset + one model family."""

    dataset: CaptionedVideos
    config: ConditioningConfig
    encode_text: EncodeText
    encode_video: EncodeVideo
    tensor_io: TensorIO | None = None
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.spec = get_condition_spec(self.config.model_name)
        self._tensor_io = self.tensor_io or select_tensor_io()
        # Root: an explicit Precomputer output_dir wins, then the config's
        # --output-dir (upstream default "output"), then the data root.
        root = Path(
            self.output_dir
            or self.config.output_dir
            or self.config.data_root
            or "."
        )
        self.store_dir = precomputation_dir(
            root, self.config.model_name, self.config.model_id
        )

    def run(self) -> PrecomputeReport:
        """Precompute (or reuse) the store. Returns the report."""
        if self.config.batch_size != 1:
            raise ConditioningError(
                "Precomputation is only supported with batch size 1. This "
                "will be supported in future."
            )
        if (
            self.config.caption_dropout_p > 0
            and self.config.caption_dropout_technique == "empty"
        ):
            _log.warning(
                "Caption dropout is not supported with precomputation yet. "
                "This will be supported in the future."
            )
        if not should_perform_precomputation(self.store_dir):
            _log.info("Precomputed conditions and latents found. Loading precomputed data.")
            store = PrecomputedStore(
                self.store_dir.parent,
                model_name=self.config.model_name,
                cleaned_model_id=_cleaned_id(self.config.model_id),
                tensor_io=self._tensor_io,
            )
            return PrecomputeReport(
                store_dir=self.store_dir,
                model_name=self.config.model_name,
                model_id=self.config.model_id,
                codec=self._tensor_io.suffix.lstrip("."),
                reused=True,
                conditions=len(store),
                latents=len(store),
                manifest_path=store.manifest_path
                if store.manifest_path.exists()
                else None,
            )

        _log.info("Precomputed conditions and latents not found. Running precomputation.")
        conditions_dir = self.store_dir / "conditions"
        latents_dir = self.store_dir / "latents"
        conditions_dir.mkdir(parents=True, exist_ok=True)
        latents_dir.mkdir(parents=True, exist_ok=True)

        # Sweep 1 — conditions (text encoder), in dataset order.
        for index, _prompt in enumerate(self.dataset.prompts):
            text_conditions = prepare_conditions(
                self.spec,
                self.encode_text,
                self.dataset.prompt_for(
                    index, id_token=self.config.id_token
                ),
                max_sequence_length=self.config.effective_max_sequence_length,
            )
            target = conditions_dir / (
                f"conditions-0-{index}{self._tensor_io.suffix}"
            )
            self._tensor_io.save(target, text_conditions)
            _log.debug(
                "Precomputing conditions for batch %d/%d",
                index + 1, len(self.dataset),
            )

        # Sweep 2 — latents (VAE), in dataset order.
        for index, video_path in enumerate(self.dataset.video_paths):
            latent_conditions = self.encode_video(video_path)
            if "latents" not in latent_conditions:
                raise ConditioningError(
                    f"encode_video for {video_path} returned no 'latents' "
                    f"(keys: {', '.join(sorted(latent_conditions))})."
                )
            target = latents_dir / (
                f"latents-0-{index}{self._tensor_io.suffix}"
            )
            self._tensor_io.save(target, latent_conditions)
            _log.debug(
                "Precomputing latents for batch %d/%d",
                index + 1, len(self.dataset),
            )

        _log.info("Precomputation complete")
        store = PrecomputedStore(
            self.store_dir.parent,
            model_name=self.config.model_name,
            cleaned_model_id=_cleaned_id(self.config.model_id),
            tensor_io=self._tensor_io,
        )
        manifest_path = store.write_manifest(
            {
                "model_name": self.config.model_name,
                "model_id": self.config.model_id,
                "config": self.config.to_dict(),
            }
        )
        return PrecomputeReport(
            store_dir=self.store_dir,
            model_name=self.config.model_name,
            model_id=self.config.model_id,
            codec=self._tensor_io.suffix.lstrip("."),
            reused=False,
            conditions=len(store),
            latents=len(store),
            manifest_path=manifest_path,
        )


def _cleaned_id(model_id: str) -> str:
    return string_to_filename(model_id)


def demo_encoders(spec_name: str) -> tuple[EncodeText, EncodeVideo]:
    """Deterministic stand-in encoders for verification (no weights).

    The text encoder returns fixed pseudo-embeddings derived from the
    prompt text (seeded by the prompt's bytes, so the same prompt always
    yields the same array); the video encoder returns fixed pseudo-latents
    derived from the video file's bytes. These exist so the precompute
    path runs end to end on a machine with no GPU and no model weights —
    they are NOT real model outputs and are never presented as such.
    """

    def encode_text(
        prompts: list[str], max_sequence_length: int
    ) -> dict[str, Any]:
        import hashlib

        seed = int.from_bytes(
            hashlib.sha256("\n".join(prompts).encode("utf-8")).digest()[:4],
            "little",
        )
        rng = np.random.default_rng(seed)
        out: dict[str, Any] = {
            "prompt_embeds": rng.standard_normal(
                (len(prompts), max_sequence_length, 8), dtype=np.float32
            ).astype(np.float32),
        }
        if spec_name == "ltx_video":
            out["prompt_attention_mask"] = np.ones(
                (len(prompts), max_sequence_length), dtype=np.int64
            )
        if spec_name == "hunyuan_video":
            out["prompt_attention_mask"] = np.ones(
                (len(prompts), max_sequence_length), dtype=np.int64
            )
            out["pooled_prompt_embeds"] = rng.standard_normal(
                (len(prompts), 8), dtype=np.float32
            ).astype(np.float32)
        return out

    def encode_video(video_path: str | Path) -> dict[str, Any]:
        import hashlib

        data = Path(video_path).read_bytes()
        seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        latents = rng.standard_normal((1, 4, 8, 8, 8), dtype=np.float32).astype(
            np.float32
        )
        out: dict[str, Any] = {"latents": latents}
        if spec_name == "ltx_video":
            out["latents_mean"] = np.zeros(4, dtype=np.float32)
            out["latents_std"] = np.ones(4, dtype=np.float32)
            out["num_frames"] = 8
            out["height"] = 8
            out["width"] = 8
        return out

    return encode_text, encode_video


def run_precompute_from_config(
    config: ConditioningConfig,
    *,
    encode_text: EncodeText | None = None,
    encode_video: EncodeVideo | None = None,
    demo: bool = False,
) -> PrecomputeReport:
    """Load the dataset and run the precompute pass for ``config``.

    With ``demo=True`` the deterministic stand-in encoders run (local
    verification, no weights). Otherwise the real condition models load
    via :func:`encoders.load_condition_models` — which fails honestly
    when diffusers/transformers or the weights are absent.
    """
    from .encoders import load_condition_models
    from .formats import load_dataset

    dataset = load_dataset(
        config.data_root or ".",
        config.caption_column,
        config.video_column,
        dataset_file=config.dataset_file,
        remove_llm_prefixes=config.remove_common_llm_caption_prefixes,
    )
    if encode_text is None or encode_video is None:
        if demo:
            encode_text, encode_video = demo_encoders(config.model_name)
        else:
            from .encoders import (
                RealEncoders,
                get_condition_spec,
                load_condition_models,
                load_latent_models,
            )

            spec = get_condition_spec(config.model_name)
            models = load_condition_models(
                config.model_name,
                config.model_id,
                revision=config.revision,
                cache_dir=config.cache_dir,
            )
            vae = load_latent_models(
                config.model_name,
                config.model_id,
                revision=config.revision,
                cache_dir=config.cache_dir,
            )["vae"]
            real = RealEncoders(
                spec, models, vae=vae,
                max_sequence_length=config.effective_max_sequence_length,
            )
            encode_text = real.encode_text
            encode_video = real.encode_video
    precomputer = Precomputer(
        dataset, config, encode_text, encode_video,
        output_dir=config.output_dir,
    )
    return precomputer.run()


# Re-exported for callers that post-process a precomputed pair.
__all__ += ["demo_encoders", "run_precompute_from_config"]
