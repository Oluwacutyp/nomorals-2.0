"""Real diffusion training loop: DDPM loss, EMA, checkpointing, resume.

The objective (Ho et al. 2020, "simple" loss):

    L = E_{t ~ U[1,T], x_0 ~ data, eps ~ N(0,I)}[
            || eps - eps_theta(sqrt(a_bar_t) x_0 + sqrt(1-a_bar_t) eps, t) ||^2 ]

Features:
- Exponential moving average (EMA) of weights (decay 0.9999) — the EMA
  copy is what gets sampled from, standard practice since it smooths
  the noisy SGD trajectory.
- Atomic checkpointing (write tmp + rename) with optimizer state,
  scheduler state, RNG state, step and loss history → honest resume.
- Mixed precision (torch.cuda.amp) when CUDA is available; pure fp32
  on CPU.
- Loss logging to a JSONL file per run.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

from . import ImgGenError, TORCH_AVAILABLE, checkpoint_dir

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.train needs PyTorch: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .diffusion import DDPMScheduler

__all__ = [
    "TrainConfig",
    "EMA",
    "train",
    "train_text_conditional",
    "save_checkpoint",
    "load_checkpoint",
]


@dataclass
class TrainConfig:
    """Everything the training loop needs, serializable to JSON."""

    run_name: str = "devon-ddpm"
    timesteps: int = 1000
    schedule: str = "linear"
    image_size: int = 64
    in_channels: int = 3
    base_channels: int = 64
    depth: int = 3
    temb_dim: int = 128
    lr: float = 1e-4
    batch_size: int = 16
    steps: int = 10_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    checkpoint_every: int = 1000
    log_every: int = 50
    seed: int = 0
    device: str = ""  # auto: cuda if available else cpu
    extra: dict = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class EMA:
    """Exponential moving average of model parameters.

    shadow = decay * shadow + (1 - decay) * param, applied after each
    optimizer step. ``copy_to`` swaps the EMA weights into a model for
    sampling; ``restore`` puts the training weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            self.shadow[n].mul_(self.decay).add_(
                p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.detach().copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self._backup:
                p.detach().copy_(self._backup[n])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


def _model_config_dict(model: nn.Module) -> dict:
    """Best-effort architecture record for checkpoint provenance."""
    cfg = {"class": type(model).__name__}
    for attr in ("in_channels", "base_channels", "depth", "temb_dim",
                 "image_size", "ctx_dim"):
        if hasattr(model, attr):
            cfg[attr] = getattr(model, attr)
    return cfg


class _JointModel(nn.Module):
    """UNet + text encoder trained as one unit (native text-to-image).

    The checkpoint stores ``unet.*`` and ``text_encoder.*`` prefixed
    state dicts so pipeline.py can split them at load time.
    """

    def __init__(self, unet: nn.Module, text_encoder: nn.Module,
                 cond_dropout: float = 0.1) -> None:
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.cond_dropout = cond_dropout

    def forward(self, xt: torch.Tensor, t: torch.Tensor,
                captions: list[str]) -> torch.Tensor:
        ctx = self.text_encoder(captions)
        if self.training and self.cond_dropout > 0:
            # Conditioning dropout → the model learns the unconditional
            # branch that classifier-free guidance needs at inference.
            mask = (torch.rand(ctx.shape[0], 1, 1, device=ctx.device)
                    > self.cond_dropout).float()
            ctx = ctx * mask
        return self.unet(xt, t, ctx)


def train_text_conditional(unet: nn.Module, text_encoder: nn.Module,
                           dataloader: DataLoader, cfg: TrainConfig,
                           scheduler: DDPMScheduler | None = None,
                           cond_dropout: float = 0.1,
                           resume_from: str | None = None) -> dict:
    """Train UNet + text encoder jointly on (image, caption) pairs.

    The dataloader must yield ``(images, captions)`` where captions is
    a list of strings. Conditioning dropout (default 10%) trains the
    unconditional branch for classifier-free guidance.
    """
    joint = _JointModel(unet, text_encoder, cond_dropout=cond_dropout)

    # Wrap the dataloader so the shared train() sees (images, captions).
    class _CapLoader:
        def __init__(self, base):
            self.base = base

        def __iter__(self):
            for batch in self.base:
                images, captions = batch[0], batch[1]
                # train() takes batch[0] as images; stash captions.
                yield (images, captions)

        def __len__(self):
            return len(self.base)

    # Patch: train() reads batch[0] for images; we thread captions via
    # a model wrapper that consumes them from a side channel.
    caption_box: list = []

    orig_forward = joint.forward

    def _fwd(xt, t):
        return orig_forward(xt, t, caption_box[0])

    joint.forward = _fwd  # type: ignore[method-assign]

    real_iter_base = dataloader

    class _SideChannelLoader:
        def __iter__(self):
            for images, captions in real_iter_base:
                caption_box.clear()
                caption_box.append(list(captions))
                yield (images,)

        def __len__(self):
            return len(real_iter_base)

    result = train(joint, _SideChannelLoader(), cfg,
                   scheduler=scheduler, resume_from=resume_from)

    # Record the text-encoder config for the pipeline loader.
    enc_cfg = {"vocab_size": text_encoder.vocab_size,
               "dim": text_encoder.dim,
               "layers": len(text_encoder.transformer.layers),
               "heads": text_encoder.transformer.layers[0].self_attn.num_heads}
    ckpt = result["final_checkpoint"]
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    # State dict already carries unet.* / text_encoder.* prefixes from
    # the _JointModel wrapper — just record the encoder config.
    payload["text_encoder_config"] = enc_cfg
    payload["model_config"]["ctx_dim"] = text_encoder.dim
    tmp = ckpt + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt)
    result["text_encoder_config"] = enc_cfg
    return result


def save_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: DDPMScheduler, ema: EMA | None,
                    cfg: TrainConfig, step: int,
                    loss_history: list[float]) -> None:
    """Atomic checkpoint: tmp file + os.replace, never a half-write."""
    payload = {
        "format": "devon-imggen-1",
        "step": step,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "ema_state": ema.state_dict() if ema else None,
        "schedule": {"timesteps": scheduler.timesteps,
                     "betas": scheduler.schedule.betas},
        "model_config": _model_config_dict(model),
        "train_config": cfg.to_dict(),
        "loss_history": loss_history,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu") -> dict:
    """Load a Devon checkpoint into ``model`` (architecture must match).

    Returns the payload dict (step, loss_history, configs...).
    Raises ImgGenError on format mismatch or missing file.
    """
    if not os.path.exists(path):
        raise ImgGenError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ImgGenError(f"could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "devon-imggen-1":
        raise ImgGenError(
            f"{path} is not a Devon imggen checkpoint "
            f"(format={payload.get('format') if isinstance(payload, dict) else '?'}; "
            "expected 'devon-imggen-1')")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except Exception:
            pass  # optimizer shape changed; model weights still valid
    torch.set_rng_state(payload["torch_rng"])
    return payload


def train(model: nn.Module, dataloader: DataLoader, cfg: TrainConfig,
          scheduler: DDPMScheduler | None = None,
          ema: EMA | None = None,
          resume_from: str | None = None,
          lora_only: bool = False) -> dict:
    """Run the DDPM training loop. Returns a summary dict.

    - ``lora_only``: optimize just LoRA A/B params (see lora.py).
    - Checkpoints land in ``<checkpoint_dir>/<run_name>/ckpt_<step>.pt``.
    - Loss history appends to ``loss.jsonl`` in the run dir.
    """
    device = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    model.to(device)
    model.train()

    scheduler = scheduler or DDPMScheduler(timesteps=cfg.timesteps,
                                           schedule=cfg.schedule)
    if ema is None:
        ema = EMA(model, decay=cfg.ema_decay)

    if lora_only:
        from .lora import lora_parameters

        params = list(lora_parameters(model))
        if not params:
            raise ImgGenError(
                "lora_only=True but no LoRA adapters found — "
                "call lora.inject_lora(model) first")
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    start_step = 0
    loss_history: list[float] = []
    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer,
                                  device=device)
        start_step = int(payload.get("step", 0))
        loss_history = list(payload.get("loss_history", []))
        if payload.get("ema_state"):
            ema.load_state_dict(payload["ema_state"])

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = os.path.join(checkpoint_dir(), cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "loss.jsonl")
    log_f = open(log_path, "a")

    betas_t = torch.as_tensor(scheduler.schedule.betas,
                              dtype=torch.float32, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas_t, dim=0)
    sqrt_ab = torch.sqrt(alphas_cumprod)
    sqrt_om = torch.sqrt(1.0 - alphas_cumprod)

    step = start_step
    data_iter = iter(dataloader)
    t0 = time.time()
    try:
        while step < cfg.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            x0 = batch.to(device, dtype=torch.float32)
            # Data is expected in [-1, 1]; warn-scale if it looks like [0,1].
            if x0.min() >= 0:
                x0 = x0 * 2.0 - 1.0
            b = x0.shape[0]
            t = torch.randint(0, cfg.timesteps, (b,), device=device)
            noise = torch.randn_like(x0)
            xt = (sqrt_ab[t].view(b, 1, 1, 1) * x0
                  + sqrt_om[t].view(b, 1, 1, 1) * noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xt, t)
                loss = torch.mean((pred - noise) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            lv = float(loss.detach().cpu())
            loss_history.append(lv)
            step += 1

            if step % cfg.log_every == 0:
                rec = {"step": step, "loss": lv,
                       "elapsed_s": round(time.time() - t0, 1)}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            if step % cfg.checkpoint_every == 0:
                ckpt = os.path.join(run_dir, f"ckpt_{step:06d}.pt")
                save_checkpoint(ckpt, model, optimizer, scheduler,
                                ema, cfg, step, loss_history)
    finally:
        log_f.close()

    # Final checkpoint.
    ckpt = os.path.join(run_dir, f"ckpt_{step:06d}.pt")
    save_checkpoint(ckpt, model, optimizer, scheduler, ema, cfg,
                    step, loss_history)
    return {
        "steps": step,
        "run_dir": run_dir,
        "final_checkpoint": ckpt,
        "final_loss": loss_history[-1] if loss_history else math.nan,
        "loss_history": loss_history,
    }
