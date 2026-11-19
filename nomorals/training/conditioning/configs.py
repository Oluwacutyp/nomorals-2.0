"""Conditioning configuration — the args that carry conditioning, ported from upstream ``args.py``.

Upstream ``finetrainers/args.py`` (1,151 lines) covers the whole trainer;
this module ports only the arguments that shape conditioning data:
dataset selection, resolution buckets, caption dropout, the precompute
switch, LLM-prefix stripping, and the identifier token. Flag names,
choices, and defaults are upstream's; the dataclass validates with
upstream's error wording.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_IMAGE_RESOLUTION_BUCKETS,
    DEFAULT_VIDEO_RESOLUTION_BUCKETS,
)
from .encoders import get_condition_spec
from .errors import ConditioningError

__all__ = [
    "ConditioningConfig",
    "parse_args",
    "parse_resolution_bucket",
    "parse_video_resolution_bucket",
]

#: Upstream ``--caption_dropout_technique`` choices (args.py).
CAPTION_DROPOUT_TECHNIQUES = ("empty", "zero")

#: Upstream ``--video_reshape_mode`` choices (args.py; "we don't support
#: this" upstream — the value is carried, the reshape itself is a no-op
#: unless the caller wires it).
VIDEO_RESHAPE_MODES = ("center", "random", "none")

#: Upstream ``--training_type`` choices (models.py).
TRAINING_TYPES = ("lora", "full-finetune")


def parse_resolution_bucket(text: str) -> tuple[int, ...]:
    """Parse one ``--video_resolution_buckets`` entry (``49x512x768``)."""
    try:
        parts = tuple(int(part) for part in text.split("x"))
    except ValueError as exc:
        raise ConditioningError(
            f"invalid resolution bucket {text!r} (expected e.g. "
            f"'49x512x768')"
        ) from exc
    if not parts or any(part <= 0 for part in parts):
        raise ConditioningError(
            f"invalid resolution bucket {text!r}: all dimensions must be "
            f"positive"
        )
    return parts


def parse_video_resolution_bucket(text: str) -> tuple[int, int, int]:
    """Parse one 3D bucket; upstream asserts exactly three dimensions."""
    bucket = parse_resolution_bucket(text)
    assert len(bucket) == 3, (
        f"Expected 3D resolution bucket, got {len(bucket)}D resolution bucket"
    )
    return bucket  # type: ignore[return-value]


@dataclass
class ConditioningConfig:
    """Everything one conditioning run needs, with upstream defaults.

    ``model_name`` selects the family (``cogvideox`` / ``hunyuan_video``
    / ``ltx_video``); ``pretrained_model_name_or_path`` is the HF repo
    (defaults to the family's upstream checkpoint). ``max_sequence_length``
    defaults to the family's value when left ``None``.
    """

    data_root: str | Path | None = None
    dataset_file: str | Path | None = None
    caption_column: str = "prompts.txt"
    video_column: str = "videos.txt"
    model_name: str = "ltx_video"
    training_type: str = "lora"
    pretrained_model_name_or_path: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    precompute_conditions: bool = False
    caption_dropout_p: float = 0.0
    caption_dropout_technique: str = "empty"
    remove_common_llm_caption_prefixes: bool = False
    id_token: str | None = None
    video_resolution_buckets: list[tuple[int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_VIDEO_RESOLUTION_BUCKETS)
    )
    image_resolution_buckets: list[tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_IMAGE_RESOLUTION_BUCKETS)
    )
    video_reshape_mode: str | None = None
    max_sequence_length: int | None = None
    output_dir: str | Path | None = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.validate()

    @property
    def model_id(self) -> str:
        """The checkpoint this run builds conditions against."""
        if self.pretrained_model_name_or_path:
            return str(self.pretrained_model_name_or_path)
        return get_condition_spec(self.model_name).model_id

    @property
    def effective_max_sequence_length(self) -> int:
        if self.max_sequence_length is not None:
            return self.max_sequence_length
        return get_condition_spec(self.model_name).max_sequence_length

    def validate(self) -> None:
        """Upstream's argument checks, with upstream's messages."""
        if self.data_root is None and self.dataset_file is None:
            raise ConditioningError(
                "At least one of `data_root` or `dataset_file` should be "
                "provided."
            )
        get_condition_spec(self.model_name)
        if self.training_type not in TRAINING_TYPES:
            raise ConditioningError(
                f"Training type {self.training_type} not supported for "
                f"model {self.model_name}. Supported training types are: "
                f"{list(TRAINING_TYPES)}"
            )
        if self.caption_dropout_technique not in CAPTION_DROPOUT_TECHNIQUES:
            raise ConditioningError(
                f"caption_dropout_technique must be one of "
                f"{list(CAPTION_DROPOUT_TECHNIQUES)}, got "
                f"{self.caption_dropout_technique!r}"
            )
        if (
            self.video_reshape_mode is not None
            and self.video_reshape_mode not in VIDEO_RESHAPE_MODES
        ):
            raise ConditioningError(
                f"video_reshape_mode must be one of "
                f"{list(VIDEO_RESHAPE_MODES)}, got "
                f"{self.video_reshape_mode!r}"
            )
        if not 0.0 <= self.caption_dropout_p <= 1.0:
            raise ConditioningError(
                f"caption_dropout_p must be between 0 and 1, got "
                f"{self.caption_dropout_p}"
            )
        if self.batch_size < 1:
            raise ConditioningError(
                f"batch_size must be at least 1, got {self.batch_size}"
            )

    def to_dict(self) -> dict[str, object]:
        """Plain-data view, for the precompute manifest and logs."""
        return {
            "data_root": None if self.data_root is None else str(self.data_root),
            "dataset_file": None
            if self.dataset_file is None
            else str(self.dataset_file),
            "caption_column": self.caption_column,
            "video_column": self.video_column,
            "model_name": self.model_name,
            "training_type": self.training_type,
            "model_id": self.model_id,
            "precompute_conditions": self.precompute_conditions,
            "caption_dropout_p": self.caption_dropout_p,
            "caption_dropout_technique": self.caption_dropout_technique,
            "remove_common_llm_caption_prefixes": (
                self.remove_common_llm_caption_prefixes
            ),
            "id_token": self.id_token,
            "max_sequence_length": self.effective_max_sequence_length,
            "batch_size": self.batch_size,
        }


def parse_args(argv: Sequence[str] | None = None) -> ConditioningConfig:
    """Parse the conditioning flags (upstream names, upstream defaults)."""
    parser = argparse.ArgumentParser(
        description="Precompute video-model conditions (FineTrainers port)."
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--caption-column", type=str, default="prompts.txt")
    parser.add_argument("--video-column", type=str, default="videos.txt")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ltx_video",
        choices=["cogvideox", "hunyuan_video", "ltx_video"],
    )
    parser.add_argument(
        "--training-type",
        type=str,
        default="lora",
        choices=list(TRAINING_TYPES),
    )
    parser.add_argument("--pretrained-model-name-or-path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--precompute-conditions", action="store_true",
        help="Whether or not to precompute the conditionings for the model.",
    )
    parser.add_argument("--caption-dropout-p", type=float, default=0.0)
    parser.add_argument(
        "--caption-dropout-technique",
        type=str,
        default="empty",
        choices=list(CAPTION_DROPOUT_TECHNIQUES),
    )
    parser.add_argument(
        "--remove-common-llm-caption-prefixes", action="store_true"
    )
    parser.add_argument("--id-token", type=str, default=None)
    parser.add_argument(
        "--video-resolution-buckets",
        type=parse_video_resolution_bucket,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--video-reshape-mode",
        type=str,
        default=None,
        choices=list(VIDEO_RESHAPE_MODES),
    )
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    return ConditioningConfig(
        data_root=args.data_root,
        dataset_file=args.dataset_file,
        caption_column=args.caption_column,
        video_column=args.video_column,
        model_name=args.model_name,
        training_type=args.training_type,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        precompute_conditions=args.precompute_conditions,
        caption_dropout_p=args.caption_dropout_p,
        caption_dropout_technique=args.caption_dropout_technique,
        remove_common_llm_caption_prefixes=(
            args.remove_common_llm_caption_prefixes
        ),
        id_token=args.id_token,
        video_resolution_buckets=(
            args.video_resolution_buckets
            or list(DEFAULT_VIDEO_RESOLUTION_BUCKETS)
        ),
        video_reshape_mode=args.video_reshape_mode,
        max_sequence_length=args.max_sequence_length,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
