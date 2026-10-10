"""The Unsloth backend: QLoRA finetunes of 7B–8B models on a single GPU.

This is the production path for real transformer finetunes — the one that runs
in a Colab notebook (see ``docs/nomorals-qlora.ipynb``) or on any CUDA box.
It loads the base model in 4-bit, trains LoRA adapters, and saves three
artifacts: the PEFT adapter, the tokenizer, and a GGUF build so the artifact
is loadable by the llama.cpp provider on a phone.

Import-safe by design: nothing here imports torch/unsloth at module level, so
the package loads (and reports "not available") on a phone with zero GPU
dependencies.

The metrics contract is the same as every other backend: a gate-shaped dict
with a numeric ``score`` in (0, 1] computed from held-out negative
log-likelihood, so the promotion gate judges an Unsloth run on exactly the
same terms as a native one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from ...core.logging_setup import get_logger
from ..dataset import Example
from .base import BackendResult

__all__ = ["UnslothBackend", "dataset_from_examples", "guess_template", "dry_run"]

_log = get_logger(__name__)

#: LoRA targets that exist on Qwen/LLaMA/Mistral-style attention + MLP.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

#: Cap on how much held-out text we score, and per sample — evaluation must
#: stay a fraction of training cost, not a second training run.
EVAL_MAX_SAMPLES = 16
EVAL_MAX_TOKENS = 512


def guess_template(base_model: str) -> str:
    """Chat template name for LLaMA-Factory / tokenizers, from the model id.

    Substring matching needs care: "dolphin" contains "phi" — so the Phi
    check demands a word boundary or a following version digit.
    """
    m = (base_model or "").lower()
    if "qwen" in m:
        return "qwen"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    if "gemma" in m:
        return "gemma"
    if re.search(r"phi-?\d", m) or re.search(r"\bphi\b", m):
        return "phi"
    return "llama"


def dataset_from_examples(
    examples: Sequence[Example], *, add_generation_prompt: bool = False
) -> list[dict[str, str]]:
    """Examples → pre-formatted ChatML rows.

    Pre-formatted text with ``dataset_text_field="text"`` is the most
    version-stable contract across trl/unsloth releases, and it reuses the
    exact ChatML formatting the rest of the pipeline already trusts.
    """
    return [
        {"text": example.to_chatml(add_generation_prompt=add_generation_prompt)}
        for example in examples
    ]


class UnslothBackend:
    """QLoRA via Unsloth. Usable only where CUDA + unsloth are installed."""

    name = "unsloth"

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "torch is not installed (pip install torch — or run on Colab)"
        try:
            import unsloth  # noqa: F401
        except ImportError:
            return False, "unsloth is not installed (pip install unsloth — or run on Colab)"
        if not torch.cuda.is_available():
            return False, "no CUDA GPU visible — unsloth trains on GPU only"
        return True, f"CUDA OK ({torch.cuda.get_device_name(0)})"

    # ── the run ──────────────────────────────────────────────────────────────

    def train(
        self,
        train: Sequence[Example],
        evaluation: Sequence[Example] = (),
        *,
        output_dir: str | Path,
        base_model: str = "",
        settings: Any = None,
        on_step: Callable[[int, float], None] | None = None,
    ) -> BackendResult:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        s = settings or _Defaults()

        if not train:
            raise ValueError("unsloth backend: training set is empty")
        base = base_model or str(getattr(s, "base_model", "") or "")
        if not base:
            raise ValueError(
                "unsloth backend: no base_model given — pass one "
                "(e.g. 'cognitivecomputations/dolphin-2.9-llama3-8b')"
            )

        max_seq_length = _clamp(int(getattr(s, "max_seq_len", 2048)), 256, 8192)
        dataset = dataset_from_examples(train)

        # ── load + LoRA (all heavy imports live here, on purpose) ──────────
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
        except ImportError as exc:
            raise RuntimeError(
                f"unsloth backend needs unsloth + trl: {exc} — "
                "on Colab use !pip install unsloth, elsewhere pip install unsloth"
            ) from exc

        _log.info("unsloth: loading %s (4-bit QLoRA, seq %d)", base, max_seq_length)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=max_seq_length,
            dtype=None,  # let Unsloth pick bfloat16/fp16 for the hardware
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(getattr(s, "lora_r", 16) or 16),
            target_modules=DEFAULT_TARGET_MODULES,
            lora_alpha=int(getattr(s, "lora_alpha", 32) or 32),
            lora_dropout=float(getattr(s, "lora_dropout", 0.05) or 0.0),
            bias="none",
        )

        # ── train ───────────────────────────────────────────────────────────
        losses: list[float] = []

        class _LossProbe:
            def __init__(self, trainer: Any) -> None:
                self.trainer = trainer

            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
                if isinstance(logs, dict) and isinstance(logs.get("loss"), (int, float)):
                    losses.append(float(logs["loss"]))
                if on_step and getattr(state, "global_step", 0):
                    loss = logs.get("loss") if isinstance(logs, dict) else None
                    if isinstance(loss, (int, float)):
                        on_step(int(state.global_step), float(loss))

        use_bf16 = _wants_bf16()
        train_cfg_cls = _sft_config_class()
        train_cfg_kwargs: dict[str, Any] = dict(
            output_dir=str(out / "runs"),
            per_device_train_batch_size=int(getattr(s, "batch_size", 4) or 4),
            gradient_accumulation_steps=int(getattr(s, "gradient_accumulation", 4) or 4),
            warmup_ratio=0.1,
            learning_rate=float(getattr(s, "learning_rate", 2e-5) or 2e-5),
            num_train_epochs=int(getattr(s, "epochs", 3) or 3),
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            gradient_checkpointing="unsloth",
            report_to=[],
            max_seq_length=max_seq_length,
            remove_unused_columns=False,
        )
        # NEFTune: free instruction-following gain (Jain et al.).  Probed
        # like the liger flag — an old trl/transformers raises TypeError on
        # an unknown kwarg, so never assume it.
        train_cfg_kwargs.update(_neftune_kwargs(train_cfg_cls))
        # assistant_only_loss: loss on assistant turns only — the model
        # learns to write answers, not to predict user turns.  Probed the
        # same way; silently skipped on old trl.
        train_cfg_kwargs.update(_assistant_only_kwargs(train_cfg_cls))
        train_cfg = train_cfg_cls(**train_cfg_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            # packing: short rows waste most of each window on padding.
            packing=True,
            args=train_cfg,
        )
        trainer.add_callback(_LossProbe(trainer))
        _log.info("unsloth: training %d rows for %d epoch(s)", len(dataset), train_cfg.num_train_epochs)
        trainer.train()

        # ── save: adapter + tokenizer + GGUF (the phone-friendly artifact) ──
        adapter_dir = out / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(out))
        gguf_info = _save_gguf(model, tokenizer, out)

        # ── score the held-out set (same contract as the native backend) ────
        eval_loss = _eval_nll(model, tokenizer, evaluation, max_tokens=EVAL_MAX_TOKENS)
        if eval_loss is None:
            # No held-out set (or evaluation failed): report the last training
            # loss, and say so — a train-loss score must not masquerade as
            # validation in the gate history.
            final_loss = losses[-1] if losses else 0.0
            metrics = {
                "train_loss": round(final_loss, 6),
                "score": round(1.0 / (1.0 + final_loss), 6) if final_loss else 0.0,
            }
            info_extra = {"score_basis": "train_loss (no held-out eval)"}
        else:
            final_loss = losses[-1] if losses else eval_loss
            metrics = {
                "train_loss": round(final_loss, 6),
                "eval_loss": round(eval_loss, 6),
                "perplexity": round(_exp(eval_loss), 4),
                "score": round(1.0 / (1.0 + eval_loss), 6),
            }
            info_extra = {"score_basis": "eval_nll"}

        info = {
            "backend": self.name,
            "base_model": base,
            "template": guess_template(base),
            "adapter": str(adapter_dir),
            "max_seq_length": max_seq_length,
            "steps": int(getattr(trainer.state, "global_step", 0) or 0),
            "loss_history": [round(v, 5) for v in losses[-50:]],
            **gguf_info,
            **info_extra,
        }
        _log.info(
            "unsloth: done — %d steps, final loss %.4f, eval %s",
            info["steps"], final_loss,
            f"{eval_loss:.4f}" if eval_loss is not None else "n/a",
        )
        return BackendResult(output_path=str(out), metrics=metrics, info=info)


# ── helpers ──────────────────────────────────────────────────────────────────


class _Defaults:
    """Field access for backends called without settings."""

    max_seq_len = 2048
    epochs = 3
    batch_size = 4
    gradient_accumulation = 4
    learning_rate = 2e-5
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    base_model = ""


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _wants_bf16() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()
    except Exception:  # noqa: BLE001
        return False


def _sft_config_class() -> Any:
    """SFTConfig (new trl) or TrainingArguments (older trl)."""
    try:
        from trl import SFTConfig

        return SFTConfig
    except ImportError:
        from transformers import TrainingArguments

        return TrainingArguments


def _accepts_kwarg(config_cls: Any, name: str) -> bool:
    """Does this config class accept ``name``?  Old trl/transformers raise
    TypeError on unknown kwargs — probe, never assume."""
    try:
        import inspect

        return name in inspect.signature(config_cls.__init__).parameters
    except Exception:  # noqa: BLE001
        return False


def _neftune_kwargs(config_cls: Any) -> dict[str, Any]:
    if _accepts_kwarg(config_cls, "neftune_noise_alpha"):
        return {"neftune_noise_alpha": 5}
    return {}


def _assistant_only_kwargs(config_cls: Any) -> dict[str, Any]:
    if _accepts_kwarg(config_cls, "assistant_only_loss"):
        return {"assistant_only_loss": True}
    return {}


def dry_run(
    base_model: str = "",
    settings: Any = None,
    *,
    train_rows: int = 0,
) -> dict[str, Any]:
    """Validate a run's configuration WITHOUT training.

    Catches the mistakes that burn a Colab session: empty dataset, missing
    base model id, unavailable backend, absurd hyperparameters.  Returns
    ``{"ok", "checks": [...], "warnings": [...]}`` — every check names
    what it verified.
    """
    from . import get_backend

    checks: list[dict[str, str]] = []
    warnings: list[str] = []

    def _check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": "yes" if ok else "no",
                       "detail": detail})

    backend = get_backend("unsloth")
    available, reason = backend.available()
    _check("backend available", available, reason)
    base = base_model or str(getattr(settings, "base_model", "") or "")
    _check("base model set", bool(base), base or "no base_model given")
    _check("training rows", train_rows > 0,
           f"{train_rows} rows" if train_rows else "empty training set")
    s = settings or _Defaults()
    lr = float(getattr(s, "learning_rate", 2e-5) or 2e-5)
    _check("learning rate sane", 1e-6 <= lr <= 1e-2, f"lr={lr}")
    seq = int(getattr(s, "max_seq_len", 2048) or 2048)
    _check("sequence length sane", 256 <= seq <= 8192, f"seq={seq}")
    r = int(getattr(s, "lora_r", 16) or 16)
    alpha = int(getattr(s, "lora_alpha", 32) or 32)
    _check("lora alpha ≈ 2× rank", alpha == 2 * r,
           f"r={r} alpha={alpha}" + ("" if alpha == 2 * r
                                     else " (recipe says alpha ≈ 2× rank)"))
    if not available:
        warnings.append(f"backend not available here: {reason}")
    ok = all(c["ok"] == "yes" for c in checks)
    return {"ok": ok, "checks": checks, "warnings": warnings}


def _exp(x: float) -> float:
    import math

    return math.exp(min(20.0, max(-20.0, x)))


def _save_gguf(model: Any, tokenizer: Any, out: Path) -> dict[str, Any]:
    """Best-effort GGUF export so llama.cpp (and the phone) can load it.

    Tries the modern call shapes in order; a failure here must never sink an
    otherwise-good run — the PEFT adapter is the primary artifact.
    """
    try:
        gguf_dir = out / "gguf"
        attempts: list[tuple[str, Callable[[], None]]] = []
        attempts.append((
            "model.save_pretrained_gguf(q8_0)",
            lambda: model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q8_0"),
        ))
        try:
            from unsloth import FastLanguageModel

            attempts.append((
                "FastLanguageModel.save_for_gguf(q8_0)",
                lambda: FastLanguageModel.save_for_gguf(
                    model, str(gguf_dir), tokenizer, quantization_method="q8_0"
                ),
            ))
        except ImportError:  # noqa: E103 - unsloth optional, other export attempts follow
            pass
        attempts.append(("model.save_pretrained_gguf()",
                         lambda: model.save_pretrained_gguf(str(gguf_dir), tokenizer)))
        for label, call in attempts:
            try:
                call()
            except (AttributeError, TypeError, ImportError):
                continue  # this build doesn't have that entry point
            except Exception as exc:  # noqa: BLE001
                return {"gguf": "failed", "gguf_error": f"{label}: {str(exc)[:200]}"}
            files = sorted(p.name for p in gguf_dir.glob("*.gguf"))
            return {"gguf": "ok", "gguf_via": label, "gguf_files": files[:4]}
        return {"gguf": "failed", "gguf_error": "no save_pretrained_gguf entry point available"}
    except Exception as exc:  # noqa: BLE001
        return {"gguf": "failed", "gguf_error": str(exc)[:200]}


def _eval_nll(
    model: Any,
    tokenizer: Any,
    evaluation: Sequence[Example],
    *,
    max_samples: int = EVAL_MAX_SAMPLES,
    max_tokens: int = EVAL_MAX_TOKENS,
) -> float | None:
    """Mean negative log-likelihood over held-out rows. None when we cannot
    score (no eval set, or the model refuses) — the caller reports that."""
    if not evaluation:
        return None
    try:
        import torch

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for example in list(evaluation)[:max_samples]:
                text = example.to_chatml()
                ids = tokenizer(
                    text, add_special_tokens=False,
                    return_tensors="pt", truncation=True, max_length=max_tokens,
                ).input_ids
                if ids.shape[1] < 8:
                    continue
                outputs = model(input_ids=ids)
                logits = outputs.logits[0, :-1, :]
                targets = ids[0, 1:]
                loss_f = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                    reduction="sum",
                )
                total += float(loss_f)
                count += int(targets.numel())
        if count == 0:
            return None
        return total / count
    except Exception as exc:  # noqa: BLE001 — scoring is optional garnish
        _log.warning("unsloth eval skipped: %s", exc)
        return None
