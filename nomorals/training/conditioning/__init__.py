"""L3 — training.conditioning: conditioning data for video fine-tuning.

A port of upstream FineTrainers' conditioning-data machinery
(storytold/FineTrainers-Conditioning, upstream aarnphm/FineTrainers
snapshot commit 9f0fdb5, Apache-2.0) into Devon's training organ:
captioned-video dataset formats, resolution buckets, the precomputed
(conditions, latents) store, the per-model condition specs, and the
precompute orchestration that turns a captioned dataset into training
input for CogVideoX / HunyuanVideo / LTX-Video LoRA runs.

The consumer is Devon's videogen organ (``nomorals/media/videogen``):
LTX-Video is the video model Devon actually runs, and its LoRAs live
under the models root — a precomputed store built here is the training
input a future LTX LoRA run consumes. This package stays dependency
light on purpose (stdlib + numpy; torch/diffusers/transformers only
behind lazy imports) so the data path runs on any box while the GPU
half fails honestly where the weights are absent.
"""

from __future__ import annotations

from .buckets import BucketSampler, VideoItem
from .configs import ConditioningConfig, parse_args
from .constants import (
    COMMON_LLM_START_PHRASES,
    DEFAULT_VIDEO_RESOLUTION_BUCKETS,
)
from .encoders import (
    CONDITION_SPECS,
    ConditionSpec,
    RealEncoders,
    get_condition_spec,
    load_condition_models,
    load_latent_models,
    post_latent_preparation,
    prepare_conditions,
)
from .errors import ConditioningError, ConditioningUnavailable
from .formats import CaptionedVideos, load_dataset
from .frames import decode_video_frames, preprocess_video, probe_video
from .precompute import Precomputer, PrecomputeReport, demo_encoders
from .precomputed import (
    NumpyTensorIO,
    PrecomputedStore,
    TorchTensorIO,
    select_tensor_io,
    should_perform_precomputation,
)

__all__ = [
    "BucketSampler",
    "CaptionedVideos",
    "ConditionSpec",
    "ConditioningConfig",
    "ConditioningError",
    "ConditioningUnavailable",
    "COMMON_LLM_START_PHRASES",
    "CONDITION_SPECS",
    "DEFAULT_VIDEO_RESOLUTION_BUCKETS",
    "NumpyTensorIO",
    "PrecomputeReport",
    "PrecomputedStore",
    "Precomputer",
    "RealEncoders",
    "TorchTensorIO",
    "VideoItem",
    "decode_video_frames",
    "demo_encoders",
    "get_condition_spec",
    "load_condition_models",
    "load_dataset",
    "load_latent_models",
    "parse_args",
    "post_latent_preparation",
    "preprocess_video",
    "prepare_conditions",
    "probe_video",
    "select_tensor_io",
    "should_perform_precomputation",
]
