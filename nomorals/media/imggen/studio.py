"""Devon Image Studio — the high-level API over the imggen organ.

This is what chat (``/image``) and the CLI (``nm imggen``) talk to.
Everything here is honest about profile limits: capabilities are
gated at call time with clear messages, never silently missing.

- :class:`Studio`: generate/batch/img2img/inpaint/outpaint/upscale,
  backed by the best available pipeline (native checkpoint first,
  then the legacy generative backends).
- Checkpoint management: list, describe, merge (EMA + LoRA bake).
- LoRA management: list trained adapters, train new ones.
- :func:`training_dashboard`: read a run's loss.jsonl → summary with
  sparkline, for chat/CLI display.
- Character consistency: :func:`character_sheet` generates a
  multi-pose reference grid locked to one seed + prompt anchor; the
  same anchor reproduces the character later.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass

from . import ImgGenError, TORCH_AVAILABLE, checkpoint_dir
from .failures import record_failure

__all__ = [
    "Studio",
    "StudioConfig",
    "training_dashboard",
    "character_sheet",
    "merge_checkpoints",
]


def _profile_kind() -> str:
    try:
        from ...core.profiles import get_profile_kind

        return get_profile_kind()
    except Exception as e:
        record_failure("studio._profile_kind", e)
        return "laptop"


@dataclass
class StudioConfig:
    """Studio-wide defaults, overridable per call."""

    checkpoint: str = ""      # run name, registry model name, or path;
                              # "" = registry default (native, then SD1.5)
    lora: str = ""            # LoRA adapter name to apply
    lora_scale: float = 1.0
    device: str = ""


class Studio:
    """The image studio. Lazily builds the pipeline on first use."""

    def __init__(self, config: StudioConfig | None = None) -> None:
        self.config = config or StudioConfig()
        self._pipeline = None
        self._pipeline_key: tuple = ()
        #: Disclosures from the most recent generate() (e.g. the
        #: hashing-tokenizer fallback warning for SD1.5 models whose
        #: checkpoint has no BPE vocab nearby). Also written into the
        #: output sidecar and the imggen log — never silent.
        self.last_warnings: list[str] = []

    # -- pipeline ----------------------------------------------------
    def _resolve_entry(self) -> dict:
        """Resolve StudioConfig.checkpoint through the registry.

        Accepts a native run name, a registry model name, or a path.
        "" (default) picks the registry's default loadable model —
        Devon's native class first, then SD1.5. A registered but
        not-yet-loadable model (SDXL/FLUX in this pass) raises with
        its registry reason instead of vanishing.
        """
        from .registry import list_models, resolve

        want = self.config.checkpoint
        if want:
            entry = resolve(want)
            if entry is None:
                names = ", ".join(e["name"] for e in list_models()) \
                    or "none installed"
                raise ImgGenError(
                    f"checkpoint {want!r} not found in the imggen "
                    f"registry; available: {names}")
            if not entry["loadable"]:
                raise ImgGenError(
                    f"{entry['name']} ({entry['model_class']}) is "
                    f"installed but not loadable: "
                    f"{entry['loadable_reason']}")
            return entry
        entries = list_models()
        loadable = [e for e in entries if e["loadable"]]
        if loadable:
            return loadable[0]
        if entries:
            first = entries[0]
            raise ImgGenError(
                f"image models are installed but none is loadable "
                f"yet — e.g. {first['name']} ({first['model_class']}): "
                f"{first['loadable_reason']}")
        raise ImgGenError(
            "no native checkpoint yet — train one with "
            "`nm imggen train --data <folder>` (needs torch; "
            "a tiny 64px model trains on CPU in minutes), or drop an "
            "SD1.5 .safetensors into the imggen checkpoint dir")

    def pipeline(self):
        """The resolved pipeline, built once and cached.

        Native entries load through the original native loader;
        SD1.5 entries load through Devon's own SD implementation
        behind the Studio-protocol adapter.
        """
        from .pipeline import load_native_checkpoint

        key = (self.config.checkpoint, self.config.device)
        if self._pipeline is None or self._pipeline_key != key:
            if not TORCH_AVAILABLE:
                raise ImgGenError(
                    "native image generation needs torch: "
                    "pip install torch")
            entry = self._resolve_entry()
            if entry["model_class"] == "native":
                self._pipeline = load_native_checkpoint(
                    entry["path"], device=self.config.device)
            elif entry["model_class"] == "sd15":
                from .sdcompat import SD15Pipeline, SD15StudioAdapter

                self._pipeline = SD15StudioAdapter(
                    SD15Pipeline.from_safetensors(
                        entry["path"], device=self.config.device))
            else:  # registry said loadable; classes are native|sd15
                raise ImgGenError(
                    f"registry marks {entry['name']} "
                    f"({entry['model_class']}) loadable but Studio has "
                    f"no loader for that class")
            self._pipeline_key = key
        return self._pipeline

    def _gate(self, capability: str, min_profile: str = "laptop") -> None:
        """Profile gate with an honest message. Never silent."""
        order = ["termux", "laptop", "workstation"]
        have = _profile_kind()
        if have not in order:
            have = "laptop"
        if order.index(have) < order.index(min_profile):
            raise ImgGenError(
                f"{capability} needs at least a {min_profile} profile "
                f"(current: {have}); the capability exists, this "
                f"machine just can't run it well")

    # -- generation --------------------------------------------------
    def generate(self, prompt: str, **kwargs):
        """Text-to-image. kwargs → PipelineConfig fields + save path.

        ``aspect_ratio`` ("16:9", "9:16", …) is accepted and translated
        to width/height via :func:`resolve_aspect_ratio` — it is not a
        PipelineConfig field, so without the translation it was
        silently dropped by the field filter below and a 16:9 request
        rendered square. Explicit width/height always win.
        """
        from .pipeline import PipelineConfig, resolve_aspect_ratio

        save_to = kwargs.pop("save_to", None)
        aspect = kwargs.pop("aspect_ratio", None)
        if aspect:
            aw, ah = resolve_aspect_ratio(str(aspect))
            kwargs.setdefault("width", aw)
            kwargs.setdefault("height", ah)
        pipe = self.pipeline()
        cfg = PipelineConfig(**{k: v for k, v in kwargs.items()
                                 if k in PipelineConfig.__dataclass_fields__})
        images = pipe.generate(prompt, cfg)
        self.last_warnings = list(
            getattr(pipe, "last_warnings", None) or [])
        if self.last_warnings:
            import logging

            log = logging.getLogger("nomorals.media.imggen")
            for w in self.last_warnings:
                log.warning("studio generate: %s", w)
        return self._finalize(images, prompt, save_to)

    def batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Batch generation → list of saved paths."""
        paths: list[str] = []
        for p in prompts:
            paths.extend(self.generate(p, **kwargs))
        return paths

    def img2img(self, image_path: str, prompt: str,
                strength: float = 0.6, **kwargs):
        from PIL import Image

        from .edit import img2img as _i2i

        save_to = kwargs.pop("save_to", None)
        img = Image.open(image_path)
        out = _i2i(self.pipeline(), img, prompt, strength=strength,
                   **kwargs)
        return self._finalize(out, prompt, save_to)

    def inpaint(self, image_path: str, mask_path: str, prompt: str,
                **kwargs):
        from PIL import Image

        from .edit import inpaint as _inp

        save_to = kwargs.pop("save_to", None)
        out = _inp(self.pipeline(), Image.open(image_path),
                   Image.open(mask_path), prompt, **kwargs)
        return self._finalize(out, prompt, save_to)

    def outpaint(self, image_path: str, prompt: str, **kwargs):
        from PIL import Image

        from .edit import outpaint as _out

        save_to = kwargs.pop("save_to", None)
        ext = {k: kwargs.pop(k) for k in
               ("left", "right", "top", "bottom") if k in kwargs}
        out = _out(self.pipeline(), Image.open(image_path), prompt,
                   **ext, **kwargs)
        return self._finalize(out, prompt, save_to)

    def upscale(self, image_path: str, scale: float = 2.0,
                diffusion: bool = False, prompt: str = "",
                creativity: float | None = None,
                save_to: str | None = None) -> list[str]:
        """Upscale. Classical is instant; diffusion synthesizes detail.

        ``creativity`` (0..1, diffusion only): structure-preserving →
        creative reinterpretation slider (see upscale.upscale_diffusion).
        """
        from PIL import Image

        from .upscale import upscale_classical, upscale_diffusion

        img = Image.open(image_path)
        if diffusion:
            self._gate("diffusion upscaling", "laptop")
            out = upscale_diffusion(self.pipeline(), img, prompt,
                                    scale=scale,
                                    creativity=creativity)
        else:
            if creativity is not None:
                raise ImgGenError(
                    "creativity needs diffusion=True (classical "
                    "upscale is always structure-preserving)")
            out = upscale_classical(img, scale=scale)
        return self._finalize([out], f"upscale x{scale}", save_to)

    @staticmethod
    def _default_name(i: int) -> str:
        # Nanosecond stamp: the old 1-second name made back-to-back
        # generations (batch loops, character sheets) overwrite each
        # other's file AND sidecar prompt.
        return f"imggen_{time.time_ns()}_{i}.png"

    def _target_name(self, save_to: str | None, i: int, count: int,
                     out_dir: str) -> str:
        """Resolve where image ``i`` of ``count`` lands.

        ``save_to`` conventions across callers:
        - a directory (existing, or trailing "/") → a unique generated
          name inside it (the tools pass their out-dir this way; the
          old code treated it as a file path and died in PIL);
        - a file path with a multi-image batch → indexed siblings
          (``name_1.png``, ``name_2.png``…) so no image in the batch
          overwrites another;
        - a file path, single image → exactly that path;
        - None → a unique generated name under the default out dir.
        """
        if save_to and (os.path.isdir(save_to)
                        or save_to.endswith(("/", os.sep))):
            os.makedirs(save_to, exist_ok=True)
            return os.path.join(save_to, self._default_name(i))
        if save_to:
            if count > 1:
                stem, ext = os.path.splitext(save_to)
                return f"{stem}_{i + 1}{ext or '.png'}"
            return save_to
        return os.path.join(out_dir, self._default_name(i))

    def _finalize(self, images, prompt: str,
                  save_to: str | None) -> list[str]:
        out_dir = os.path.expanduser(
            "~/.nomorals/imggen/output")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            name = self._target_name(save_to, i, len(images), out_dir)
            if not os.path.isabs(name):
                name = os.path.join(out_dir, name)
            img.save(name)
            # Sidecar prompt for reproducibility — plus any standing
            # disclosures from the pipeline (e.g. tokenizer fallback),
            # so a saved image never hides how it was made.
            with open(name + ".txt", "w") as f:
                f.write(prompt)
                for w in self.last_warnings:
                    f.write(f"\nwarning: {w}")
            paths.append(name)
        return paths

    # -- checkpoints -------------------------------------------------
    def list_checkpoints(self) -> list[dict]:
        from .pipeline import list_native_checkpoints

        return list_native_checkpoints()

    def describe_checkpoint(self, path_or_run: str) -> dict:
        """Architecture, step, loss tail — no model load needed."""
        import torch

        path = self._resolve_run(path_or_run)
        payload = torch.load(path, map_location="cpu",
                             weights_only=False)
        if payload.get("format") != "devon-imggen-1":
            raise ImgGenError(f"not a Devon checkpoint: {path}")
        hist = payload.get("loss_history", [])
        return {
            "path": path,
            "format": payload["format"],
            "step": payload.get("step"),
            "model_config": payload.get("model_config", {}),
            "train_config": payload.get("train_config", {}),
            "loss_tail": hist[-5:] if hist else [],
            "loss_min": min(hist) if hist else None,
        }

    def _resolve_run(self, path_or_run: str) -> str:
        from .pipeline import list_native_checkpoints

        for c in list_native_checkpoints():
            if path_or_run in (c["path"], c["run"]):
                return c["path"]
        if os.path.exists(path_or_run):
            return path_or_run
        raise ImgGenError(f"checkpoint not found: {path_or_run}")


def merge_checkpoints(base_path: str, other_path: str,
                      alpha: float = 0.5, out_path: str = "") -> str:
    """Weighted merge of two checkpoints: W = (1-a)*W_base + a*W_other.

    Same architecture required. Useful for blending a base model with
    a fine-tune, or two style adapters, without retraining.
    """
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise ImgGenError("alpha must be in [0, 1]")
    if not TORCH_AVAILABLE:
        raise ImgGenError("merging needs torch")
    a = torch.load(base_path, map_location="cpu", weights_only=False)
    b = torch.load(other_path, map_location="cpu", weights_only=False)
    for name, p in (("base", a), ("other", b)):
        if not isinstance(p, dict) or p.get("format") != "devon-imggen-1":
            raise ImgGenError(f"{name} is not a Devon checkpoint")
    sa, sb = a["model_state"], b["model_state"]
    if set(sa) != set(sb):
        raise ImgGenError("checkpoints have different architectures; "
                          "cannot merge")
    merged = {k: (1 - alpha) * sa[k].float() + alpha * sb[k].float()
              for k in sa}
    out = out_path or os.path.join(
        os.path.dirname(base_path),
        f"merged_a{alpha:.2f}.pt")
    payload = dict(a)
    payload["model_state"] = merged
    payload["merged_from"] = [base_path, other_path]
    payload["merge_alpha"] = alpha
    tmp = out + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


def training_dashboard(run_name: str) -> str:
    """Render a run's loss.jsonl as a text dashboard (chat/CLI)."""
    run_dir = os.path.join(checkpoint_dir(), run_name)
    log_path = os.path.join(run_dir, "loss.jsonl")
    if not os.path.exists(log_path):
        # Maybe it's a checkpoint path.
        if os.path.isdir(run_name):
            run_dir = run_name
            log_path = os.path.join(run_dir, "loss.jsonl")
        if not os.path.exists(log_path):
            raise ImgGenError(f"no training log for run {run_name!r}")
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                steps.append(rec["step"])
                losses.append(rec["loss"])
            except (json.JSONDecodeError, KeyError):
                continue
    if not losses:
        return f"run {run_name}: log exists but has no entries yet"
    # Sparkline of loss (downsampled to 40 points).
    blocks = "▁▂▃▄▅▆▇█"
    n = min(40, len(losses))
    idx = [round(i * (len(losses) - 1) / (n - 1)) for i in range(n)] \
        if n > 1 else [0]
    vals = [losses[i] for i in idx]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    spark = "".join(blocks[min(7, int((v - lo) / span * 7))]
                       for v in vals)
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    lines = [
        f"📊 training dashboard: {run_name}",
        f"steps: {steps[-1]} · points: {len(losses)}",
        f"loss: start {losses[0]:.4f} → now {losses[-1]:.4f} "
        f"(min {min(losses):.4f})",
        f"trend: {spark}",
        f"checkpoints: {len(ckpts)}"
        + (f" (latest: {os.path.basename(ckpts[-1])})" if ckpts else ""),
    ]
    return "\n".join(lines)


def character_sheet(studio: Studio, anchor: str, poses: list[str],
                    seed: int = 42, **kwargs) -> list[str]:
    """Character consistency: one seed + prompt anchor, many poses.

    The anchor (e.g. "a yoruba warrior woman, red aso-oke, scar on
    left cheek") stays fixed; each pose appends to it. Same seed →
    same face structure across the sheet; the sheet doubles as the
    reference for future generations with the same anchor+seed.
    """
    paths: list[str] = []
    for i, pose in enumerate(poses):
        prompt = f"{anchor}, {pose}"
        paths.extend(studio.generate(
            prompt, seed=seed + i, **kwargs))
    return paths
