"""Condition encoders — the per-model condition specs and the post-latent math.

Upstream, each model family (``finetrainers/cogvideox``,
``finetrainers/hunyuan_video``, ``finetrainers/ltx_video``) ships a
``*_T2V_LORA_CONFIG`` dict naming the same hooks: ``load_condition_models``
(tokenizer + text encoder from the HF repo), ``prepare_conditions``
(prompt → text-condition dict), ``prepare_latents`` (video → VAE
latents), ``post_latent_preparation`` (latents → training-ready tokens).
This module ports the *spec* half of those configs — model ids,
``max_sequence_length``, and the exact condition-dict keys each family
produces — plus the pure-math half of ``post_latent_preparation``
(CogVideoX scaling + frame padding, LTX normalisation + patch packing)
in numpy, so it runs on this box.

The GPU half — loading a T5/Llama/CLIP encoder or a VAE and running it
— lives behind :func:`load_condition_models`, which imports
diffusers/transformers lazily and fails honestly
(:class:`ConditioningUnavailable`) naming the missing package and the HF
repo when they are not present. The actual encode step is injected by
the caller as plain callables (see :mod:`nomorals.training.conditioning.precompute`),
so the orchestration is fully testable without weights.

Condition keys, verified against the upstream sources:

* CogVideoX (T5, max 226): ``{"prompt_embeds"}``.
* HunyuanVideo (Llama + CLIP, max 256): ``{"prompt_embeds",
  "prompt_attention_mask", "pooled_prompt_embeds", "guidance"}`` where
  guidance is the scalar guidance ``* 1000.0``.
* LTX-Video (T5, max 128): ``{"prompt_embeds",
  "prompt_attention_mask"}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core.logging_setup import get_logger
from .constants import DEVON_MODEL_IDS
from .errors import ConditioningError, ConditioningUnavailable
from .precomputed import require_torch

__all__ = [
    "CONDITION_SPECS",
    "ConditionSpec",
    "RealEncoders",
    "get_condition_spec",
    "load_condition_models",
    "load_latent_models",
    "normalize_latents",
    "pack_latents",
    "pad_frames",
    "post_latent_preparation",
    "prepare_conditions",
    "scale_latents",
    "to_channel_first",
    "to_frame_first",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class ConditionSpec:
    """One model family's conditioning contract (upstream's config dict, reduced).

    ``text_encoder`` names the encoder family (``"t5"`` or
    ``"llama+clip"``); ``condition_keys`` are the exact keys
    ``prepare_conditions`` returns; ``model_id`` is upstream's default
    checkpoint; ``devon_runs`` lists the checkpoints Devon's videogen
    organ actually loads for this backend.
    """

    model_name: str
    model_id: str
    text_encoder: str
    max_sequence_length: int
    condition_keys: tuple[str, ...]
    vae: str
    guidance: bool = False
    devon_runs: tuple[str, ...] = ()


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "cogvideox": ConditionSpec(
        model_name="cogvideox",
        model_id="THUDM/CogVideoX-5b",
        text_encoder="t5",
        max_sequence_length=226,
        condition_keys=("prompt_embeds",),
        vae="cogvideox",
        devon_runs=DEVON_MODEL_IDS["cogvideox"]["devon_runs"],
    ),
    "hunyuan_video": ConditionSpec(
        model_name="hunyuan_video",
        model_id="hunyuanvideo-community/HunyuanVideo",
        text_encoder="llama+clip",
        max_sequence_length=256,
        condition_keys=(
            "prompt_embeds",
            "prompt_attention_mask",
            "pooled_prompt_embeds",
            "guidance",
        ),
        vae="hunyuan",
        guidance=True,
        devon_runs=DEVON_MODEL_IDS["hunyuan_video"]["devon_runs"],
    ),
    "ltx_video": ConditionSpec(
        model_name="ltx_video",
        model_id="Lightricks/LTX-Video",
        text_encoder="t5",
        max_sequence_length=128,
        condition_keys=("prompt_embeds", "prompt_attention_mask"),
        vae="ltx",
        devon_runs=DEVON_MODEL_IDS["ltx_video"]["devon_runs"],
    ),
}


def get_condition_spec(model_name: str) -> ConditionSpec:
    """The spec for a model family, or upstream's unsupported-model error."""
    try:
        return CONDITION_SPECS[model_name]
    except KeyError as exc:
        raise ConditioningError(
            f"Model {model_name} not supported. Supported models are: "
            f"{list(CONDITION_SPECS.keys())}"
        ) from exc


def load_latent_models(
    model_name: str,
    model_id: str | None = None,
    *,
    vae_dtype: Any = None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Load the VAE for a model family (upstream's ``load_latent_models``).

    CogVideoX: ``AutoencoderKLCogVideoX``; HunyuanVideo:
    ``AutoencoderKLHunyuanVideo``; LTX-Video: ``AutoencoderKLLTXVideo`` —
    each from ``{model_id}``'s ``vae`` subfolder. diffusers is imported
    lazily; missing means an honest :class:`ConditioningUnavailable`.
    """
    spec = get_condition_spec(model_name)
    model_id = model_id or spec.model_id
    try:
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise ConditioningUnavailable(
            "diffusers is not installed on this machine, so the "
            f"{model_name} VAE cannot load. Install diffusers "
            "(>=0.32.1 per upstream requirements.txt)."
        ) from exc
    from diffusers import (
        AutoencoderKLCogVideoX,
        AutoencoderKLHunyuanVideo,
        AutoencoderKLLTXVideo,
    )

    vae_cls = {
        "cogvideox": AutoencoderKLCogVideoX,
        "hunyuan_video": AutoencoderKLHunyuanVideo,
        "ltx_video": AutoencoderKLLTXVideo,
    }[model_name]
    kwargs: dict[str, Any] = {}
    if vae_dtype is not None:
        kwargs["torch_dtype"] = vae_dtype
    _log.info("loading %s VAE from %s", model_name, model_id)
    vae = vae_cls.from_pretrained(
        model_id, subfolder="vae", revision=revision, cache_dir=cache_dir,
        **kwargs,
    )
    return {"vae": vae}


class RealEncoders:
    """GPU-backed encoders built from loaded diffusers components.

    The real half of :func:`prepare_conditions` / ``prepare_latents``:
    text encoding runs the loaded tokenizer + text encoder(s) exactly
    as upstream's per-family ``prepare_conditions`` does (T5 at the
    family's ``max_sequence_length``; HunyuanVideo's Llama + CLIP with
    its prompt template and guidance), and latent encoding runs the
    loaded VAE's ``_encode`` on frames decoded by
    :mod:`nomorals.training.conditioning.frames` — the precompute path
    (unscaled latents, upstream's ``precompute=True`` branch).

    Components are injected (loaded by :func:`load_condition_models` /
    :func:`load_latent_models`, or handed in by a caller such as the
    videogen organ) — this package never imports a higher layer to fetch
    them. Without torch the constructor fails honestly; without a VAE,
    ``encode_video`` fails honestly naming the VAE.
    """

    def __init__(
        self,
        spec: ConditionSpec,
        models: dict[str, Any],
        *,
        vae: Any = None,
        max_sequence_length: int | None = None,
    ) -> None:
        self.spec = spec
        self.models = models
        self.vae = vae
        self.max_sequence_length = (
            max_sequence_length or spec.max_sequence_length
        )
        require_torch()

    def encode_text(
        self, prompts: list[str], max_sequence_length: int | None = None
    ) -> dict[str, Any]:
        """Encode prompts with the loaded text encoder(s)."""
        require_torch()
        length = max_sequence_length or self.max_sequence_length
        if self.spec.text_encoder == "t5":
            tokenizer = self.models["tokenizer"]
            text_encoder = self.models["text_encoder"]
            inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            embeds = text_encoder(inputs.input_ids.to(text_encoder.device))[0]
            out: dict[str, Any] = {
                "prompt_embeds": embeds.to(
                    dtype=text_encoder.dtype, device=text_encoder.device
                )
            }
            if self.spec.model_name == "ltx_video":
                out["prompt_attention_mask"] = inputs.attention_mask.bool().to(
                    text_encoder.device
                )
            return out
        # HunyuanVideo: Llama embeds + mask, CLIP pooled embeds, guidance.
        # Port of _get_llama_prompt_embeds + _get_clip_prompt_embeds:
        # the template wraps every caption, the token budget grows by
        # crop_start (95), the encoder runs with output_hidden_states
        # and the 3rd-from-last hidden layer is taken, then the first
        # crop_start positions are cropped off embeds and mask.
        tokenizer = self.models["tokenizer"]
        text_encoder = self.models["text_encoder"]
        tokenizer_2 = self.models["tokenizer_2"]
        text_encoder_2 = self.models["text_encoder_2"]
        template = _HUNYUAN_PROMPT_TEMPLATE
        crop_start = _HUNYUAN_CROP_START
        templated = [template.format(prompt) for prompt in prompts]
        inputs = tokenizer(
            templated,
            padding="max_length",
            max_length=length + crop_start,
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        hidden = text_encoder(
            input_ids=inputs.input_ids.to(text_encoder.device),
            attention_mask=inputs.attention_mask.to(text_encoder.device),
            output_hidden_states=True,
        ).hidden_states[-3]
        hidden = hidden.to(dtype=text_encoder.dtype)
        mask = inputs.attention_mask.to(text_encoder.device)
        if crop_start > 0:
            hidden = hidden[:, crop_start:]
            mask = mask[:, crop_start:]
        inputs_2 = tokenizer_2(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        pooled = text_encoder_2(
            inputs_2.input_ids.to(text_encoder_2.device)
        )[0]
        return {
            "prompt_embeds": hidden,
            "prompt_attention_mask": mask.bool().view(len(prompts), -1),
            "pooled_prompt_embeds": pooled.to(
                dtype=text_encoder_2.dtype, device=text_encoder_2.device
            ),
        }

    def encode_video(self, video_path: str | Path) -> dict[str, Any]:
        """Encode one video with the loaded VAE (precompute branch)."""
        from .frames import preprocess_video

        torch = require_torch()
        if self.vae is None:
            raise ConditioningUnavailable(
                f"no VAE is loaded for {self.spec.model_name}, so latents "
                f"cannot be encoded. Load one with load_latent_models() "
                f"({self.spec.model_id}, subfolder 'vae')."
            )
        video = preprocess_video(video_path)
        tensor = torch.from_numpy(video).unsqueeze(0)
        # Both families feed the VAE [B, C, F, H, W]: upstream's
        # prepare_latents permutes (0, 2, 1, 3, 4) on the [B, F, C, H, W]
        # dataset tensor.
        tensor = tensor.permute(0, 2, 1, 3, 4).contiguous()
        tensor = tensor.to(device=self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            if getattr(self.vae, "use_slicing", False) and tensor.shape[0] > 1:
                encoded = torch.cat(
                    [self.vae._encode(part) for part in tensor.split(1)]
                )
            else:
                encoded = self.vae._encode(tensor)
        out: dict[str, Any] = {"latents": encoded}
        if self.spec.model_name == "ltx_video":
            _, _, num_frames, height, width = encoded.shape
            out.update(
                {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "latents_mean": self.vae.latents_mean,
                    "latents_std": self.vae.latents_std,
                }
            )
        return out


#: HunyuanVideo's prompt template (upstream ``prepare_conditions``
#: default ``prompt_template``) — the system instruction wrapped around
#: every caption before the Llama encode — and its ``crop_start``: the
#: first 95 token positions of the templated prompt are the template
#: itself, cropped off the embeds after encoding.
_HUNYUAN_CROP_START = 95

_HUNYUAN_PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by "
    "detailing the following aspects: "
    "1. The main content and theme of the video. "
    "2. The color, shape, size, texture, quantity, text, and spatial "
    "relationships of the objects. "
    "3. Actions, events, behaviors temporal relationships, physical "
    "movement changes of the objects. "
    "4. background environment, light, style and atmosphere. "
    "5. camera angles, movements, and transitions used in the video:"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


def load_condition_models(
    model_name: str,
    model_id: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Load the tokenizer + text encoder(s) for a model family.

    Port of upstream's ``load_condition_models`` per family (CogVideoX:
    T5 tokenizer + T5EncoderModel; HunyuanVideo: Llama tokenizer/model +
    CLIP tokenizer/model; LTX-Video: T5 tokenizer + T5EncoderModel),
    all from ``{model_id}``'s ``tokenizer`` / ``text_encoder``
    subfolders. diffusers and transformers are imported lazily; if
    either is missing the call fails honestly instead of pretending the
    models loaded. The returned dict is keyed exactly as upstream keys
    it (``tokenizer`` / ``text_encoder``, plus ``tokenizer_2`` /
    ``text_encoder_2`` for HunyuanVideo).
    """
    spec = get_condition_spec(model_name)
    model_id = model_id or spec.model_id
    try:
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise ConditioningUnavailable(
            "diffusers is not installed on this machine, so the "
            f"{model_name} condition models cannot load. Install "
            "diffusers (>=0.32.1 per upstream requirements.txt)."
        ) from exc
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ConditioningUnavailable(
            "transformers is not installed on this machine, so the "
            f"{model_name} text encoder cannot load. Install "
            "transformers (>=4.45.2 per upstream requirements.txt)."
        ) from exc
    from transformers import (
        AutoTokenizer,
        CLIPTextModel,
        CLIPTokenizer,
        LlamaModel,
        T5EncoderModel,
        T5Tokenizer,
    )

    _log.info("loading %s condition models from %s", model_name, model_id)
    if spec.text_encoder == "t5":
        tokenizer = T5Tokenizer.from_pretrained(
            model_id, subfolder="tokenizer", revision=revision,
            cache_dir=cache_dir,
        )
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", revision=revision,
            cache_dir=cache_dir,
        )
        return {"tokenizer": tokenizer, "text_encoder": text_encoder}
    # llama+clip (HunyuanVideo): two encoders, upstream's exact keys.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir
    )
    text_encoder = LlamaModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision,
        cache_dir=cache_dir,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", revision=revision,
        cache_dir=cache_dir,
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder_2", revision=revision,
        cache_dir=cache_dir,
    )
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
    }


# ---------------------------------------------------------------------------
# prepare_conditions: dispatch over injected encoder callables.
# ---------------------------------------------------------------------------

#: ``encode_text(prompt, max_sequence_length) -> dict`` — the caller's
#: text encoder (a real T5/Llama+CLIP encode on GPU, or a deterministic
#: stand-in for verification). The returned dict must carry the spec's
#: ``condition_keys`` except ``guidance``, which this module adds.
EncodeText = Callable[..., dict[str, Any]]


def prepare_conditions(
    spec: ConditionSpec,
    encode_text: EncodeText,
    prompt: str | list[str],
    *,
    guidance: float = 1.0,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    """Build the text-condition dict for one prompt, per the spec.

    Port of the three families' ``prepare_conditions`` at the dispatch
    level: a lone string becomes a one-element list (all three families
    do this); the text encoder runs at the family's
    ``max_sequence_length``; HunyuanVideo adds ``guidance`` as the
    scalar ``* 1000.0`` (upstream: ``torch.tensor([guidance]) * 1000.0``).
    The result's keys must be exactly the spec's ``condition_keys`` —
    a short dict fails loudly instead of training on a missing tensor.
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    conditions = encode_text(
        prompts,
        max_sequence_length=max_sequence_length or spec.max_sequence_length,
    )
    if spec.guidance:
        conditions = dict(conditions)
        conditions["guidance"] = np.array([guidance], dtype=np.float32) * 1000.0
    missing = [key for key in spec.condition_keys if key not in conditions]
    if missing:
        raise ConditioningError(
            f"prepare_conditions for {spec.model_name} is missing "
            f"condition keys: {', '.join(missing)} (expected "
            f"{', '.join(spec.condition_keys)})."
        )
    return {key: conditions[key] for key in spec.condition_keys}


# ---------------------------------------------------------------------------
# prepare_latents: shape helpers + post-latent math (numpy ports).
# ---------------------------------------------------------------------------


def to_channel_first(video: np.ndarray) -> np.ndarray:
    """``[B, F, C, H, W] -> [B, C, F, H, W]`` (CogVideoX's permute).

    A 4D input is a single image ``[B, C, H, W]`` — upstream unsqueezes
    the frame axis (``unsqueeze(2)``), which already leaves it
    channel-first, so no permute applies.
    """
    array = np.asarray(video)
    if array.ndim == 4:
        return np.ascontiguousarray(array[:, :, np.newaxis, :, :])
    if array.ndim != 5:
        raise ConditioningError(
            f"Expected 5D tensor, got {array.ndim}D tensor"
        )
    return np.ascontiguousarray(array.transpose(0, 2, 1, 3, 4))


def to_frame_first(video: np.ndarray) -> np.ndarray:
    """``[B, C, F, H, W] -> [B, F, C, H, W]`` (LTX's permute)."""
    array = np.asarray(video)
    if array.ndim == 4:
        array = array[:, :, np.newaxis, :, :]
    if array.ndim != 5:
        raise ConditioningError(
            f"Expected 5D tensor, got {array.ndim}D tensor"
        )
    return np.ascontiguousarray(array.transpose(0, 2, 1, 3, 4))


def scale_latents(
    latents: np.ndarray, scaling_factor: float, invert_scale_latents: bool
) -> np.ndarray:
    """CogVideoX's VAE scaling: multiply unless the config inverts it."""
    if not invert_scale_latents:
        return np.asarray(latents, dtype=np.float64) * scaling_factor
    return np.asarray(latents)


def pad_frames(latents: np.ndarray, patch_size_t: int | None) -> np.ndarray:
    """CogVideoX's ``_pad_frames``: repeat the last frame to a multiple.

    Faithful port, including upstream's quirk: when the frame count is
    already divisible by ``patch_size_t``, ``additional_frames`` is a
    whole ``patch_size_t`` (not 0) — upstream pads anyway. ``None`` or
    ``1`` means no padding.
    """
    array = np.asarray(latents)
    if patch_size_t is None or patch_size_t == 1:
        return array
    num_frames = array.shape[2]
    additional = patch_size_t - num_frames % patch_size_t
    if additional > 0:
        last = array[:, :, -1:, :, :]
        padding = np.repeat(last, additional, axis=2)
        array = np.concatenate([array, padding], axis=2)
    return array


def normalize_latents(
    latents: np.ndarray,
    latents_mean: np.ndarray,
    latents_std: np.ndarray,
    scaling_factor: float = 1.0,
) -> np.ndarray:
    """LTX's ``_normalize_latents``: ``(x - mean) * scale / std`` per channel."""
    array = np.asarray(latents, dtype=np.float64)
    mean = np.asarray(latents_mean, dtype=np.float64).reshape(1, -1, 1, 1, 1)
    std = np.asarray(latents_std, dtype=np.float64).reshape(1, -1, 1, 1, 1)
    return (array - mean) * scaling_factor / std


def pack_latents(
    latents: np.ndarray, patch_size: int = 1, patch_size_t: int = 1
) -> np.ndarray:
    """LTX's ``_pack_latents``: patch ``[B, C, F, H, W]`` into tokens.

    Verbatim reshape/permute/flatten in numpy: the result is
    ``[B, F//p_t * H//p * W//p, C * p_t * p * p]``.
    """
    array = np.asarray(latents)
    batch, channels, num_frames, height, width = array.shape
    post_frames = num_frames // patch_size_t
    post_height = height // patch_size
    post_width = width // patch_size
    patched = array.reshape(
        batch,
        channels,
        post_frames,
        patch_size_t,
        post_height,
        patch_size,
        post_width,
        patch_size,
    )
    patched = patched.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return patched.reshape(
        batch,
        post_frames * post_height * post_width,
        channels * patch_size_t * patch_size * patch_size,
    )


def post_latent_preparation(
    spec: ConditionSpec,
    latents: np.ndarray,
    *,
    scaling_factor: float | None = None,
    invert_scale_latents: bool = False,
    patch_size_t: int | None = None,
    latents_mean: np.ndarray | None = None,
    latents_std: np.ndarray | None = None,
    patch_size: int = 1,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """The training-ready latents for one precomputed item, per family.

    CogVideoX (``finetrainers/cogvideox/lora.py``): scale, pad frames to
    a ``patch_size_t`` multiple, permute to ``[B, F, C, H, W]``.
    LTX-Video (``finetrainers/ltx_video/lora.py``): normalise with the
    stored mean/std, pack into tokens, and keep the pre-packing shape
    (``num_frames`` / ``height`` / ``width``) alongside — upstream's
    ``prepare_latents(precompute=True)`` stores those three in the file
    for exactly this step.
    """
    array = np.asarray(latents)
    if spec.model_name == "cogvideox":
        if scaling_factor is not None:
            array = scale_latents(array, scaling_factor, invert_scale_latents)
        array = pad_frames(array, patch_size_t)
        array = to_frame_first(array)
        return {"latents": array}
    if spec.model_name == "ltx_video":
        if latents_mean is None or latents_std is None:
            raise ConditioningError(
                "ltx_video post_latent_preparation needs latents_mean and "
                "latents_std (stored in the precomputed latent file)."
            )
        array = normalize_latents(array, latents_mean, latents_std)
        packed = pack_latents(array, patch_size, patch_size_t or 1)
        return {
            "latents": packed,
            "num_frames": num_frames if num_frames is not None else array.shape[2],
            "height": height if height is not None else array.shape[3],
            "width": width if width is not None else array.shape[4],
        }
    raise ConditioningError(
        f"post_latent_preparation is not implemented for "
        f"{spec.model_name} (supported: cogvideox, ltx_video)."
    )
