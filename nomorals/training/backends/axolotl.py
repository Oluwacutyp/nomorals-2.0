"""The Axolotl backend: reproducible, YAML-driven finetunes on any GPU box.

Where the backends fit:

* ``native`` — zero-dependency, runs on a phone;
* ``unsloth`` — fastest single-GPU QLoRA, but the OSS build is
  single-GPU only (multi-GPU is a paid tier);
* ``axolotl`` — THIS backend: YAML-configured, reproducible, and
  multi-GPU/multi-node over FSDP2/DeepSpeed **for free**.  This is the
  "graduate" path when a run outgrows one GPU, and the audit path when
  a run must be re-runnable from a version-controlled file;
* ``llama_factory`` — breadth + the LlamaBoard web UI;
* ``mlx`` — Apple Silicon.

It follows the same adapter shape as ``llama_factory``: write a real
dataset file (OpenAI-messages ``{"messages": [...]}`` rows, axolotl's
recommended ``chat_template`` format), write a real training YAML,
execute ``axolotl train`` with output streamed into our logs, and read
``trainer_state.json`` back out for the promotion gate.

Liger-Kernel is enabled in the YAML whenever it is importable on the
training machine (see :mod:`nomorals.training.liger`): +20% throughput,
-60% VRAM, same math — pure win for a free-tier box.

Import-safe: the CLI is located at call time; without it the backend
reports "not available" and the pipeline falls back to another backend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from ...core.logging_setup import get_logger
from ..dataset import Example
from ..liger import axolotl_liger_block, liger_available
from .base import BackendResult
from .llama_factory import _loss_from_line, _step_from_line

__all__ = ["AxolotlBackend", "build_axolotl_yaml", "write_axolotl_dataset"]

_log = get_logger(__name__)

#: QLoRA targets that exist on Qwen/LLaMA/Mistral/Phi/Gemma-style blocks.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _to_openai_messages(example: Example) -> list[dict[str, str]]:
    """Example turns → OpenAI ``messages`` rows (axolotl's recommended
    ``chat_template`` dataset format)."""
    out: list[dict[str, str]] = []
    for turn in example.turns:
        role = "system" if turn.role == "system" else (
            "assistant" if turn.role == "assistant" else "user"
        )
        out.append({"role": role, "content": turn.content})
    return out


def write_axolotl_dataset(
    examples: Sequence[Example], dataset_dir: Path, *, name: str
) -> Path:
    """Write ``<name>.jsonl`` with one ``{"messages": [...]}`` row per
    example — the exact shape axolotl's ``chat_template`` loader wants."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{name}.jsonl"
    dropped = 0
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            messages = _to_openai_messages(example)
            if len(messages) < 2:
                dropped += 1
                continue
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    if dropped:
        _log.warning("axolotl: dropped %d sub-2-message examples", dropped)
    return path


def build_axolotl_yaml(
    *,
    base_model: str,
    output_dir: Path,
    data_path: Path,
    settings: Any = None,
) -> str:
    """A complete, runnable axolotl training config (SFT + QLoRA).

    Keys follow the axolotl docs (base example configs + the
    Custom-Integrations page for the Liger block).  Sized for free-tier
    reality: 4-bit QLoRA, gradient checkpointing, bf16 auto, sample
    packing on.
    """
    s = settings or _Defaults()
    lines: list[str] = [
        "### model",
        f"base_model: {base_model}",
        "model_type: AutoModelForCausalLM",
        "tokenizer_type: AutoTokenizer",
        "trust_remote_code: true",
        "load_in_4bit: true",
        "strict: false",
        "",
        "### chat template (the model's own)",
        "chat_template: tokenizer_default",
        "",
        "### dataset",
        "datasets:",
        f"  - path: {data_path}",
        "    type: chat_template",
        "val_set_size: 0.0",
        "",
        "### output",
        f"output_dir: {output_dir / 'axolotl-out'}",
        "logging_steps: 10",
        "save_steps: 500",
        "saves_per_epoch: 1",
        "evals_per_epoch: 0",
        "",
        "### train",
        f"sequence_len: {int(getattr(s, 'max_seq_len', 2048) or 2048)}",
        "sample_packing: true",
        "adapter: lora",
        f"lora_r: {int(getattr(s, 'lora_r', 16) or 16)}",
        f"lora_alpha: {int(getattr(s, 'lora_alpha', 32) or 32)}",
        f"lora_dropout: {float(getattr(s, 'lora_dropout', 0.05) or 0.0)}",
        "lora_target_linear: true",
        f"micro_batch_size: {int(getattr(s, 'batch_size', 2) or 2)}",
        f"gradient_accumulation_steps: {int(getattr(s, 'gradient_accumulation', 4) or 4)}",
        f"num_epochs: {int(getattr(s, 'epochs', 3) or 3)}",
        f"learning_rate: {float(getattr(s, 'learning_rate', 2e-4) or 2e-4)}",
        "optimizer: adamw_torch",
        "lr_scheduler: cosine",
        "warmup_ratio: 0.1",
        "weight_decay: 0.0",
        "bf16: auto",
        "tf32: true",
        "gradient_checkpointing: true",
        "gradient_checkpointing_kwargs:",
        "  use_reentrant: false",
    ]
    ok, reason = liger_available()
    if ok:
        lines += ["", "### liger-kernel (free fused kernels: -60% VRAM)"]
        lines += axolotl_liger_block()
    else:
        lines += [
            "",
            "# liger-kernel not installed here — the run works without it,",
            f"# just hungrier ({reason.split('(')[0].strip()});",
            "# pip install liger-kernel to enable the fused kernels.",
        ]
    return "\n".join(lines) + "\n"


class _Defaults:
    max_seq_len = 2048
    epochs = 3
    batch_size = 2
    gradient_accumulation = 4
    learning_rate = 2e-4
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05


class AxolotlBackend:
    """Delegates to ``axolotl train`` when it is installed.

    Why axolotl alongside unsloth: unsloth OSS is single-GPU only
    (multi-GPU is a paid tier) and optimizes for speed on one card;
    axolotl is the free, YAML-reproducible, multi-GPU path
    (FSDP2/DeepSpeed) and ships first-class Liger-Kernel + Cut
    Cross-Entropy plugins.  Both emit standard PEFT adapters — the
    promotion gate never knows which one trained.
    """

    name = "axolotl"

    def available(self) -> tuple[bool, str]:
        cli = shutil.which("axolotl")
        if cli:
            return True, f"axolotl CLI found ({cli})"
        import importlib.util

        if importlib.util.find_spec("axolotl") is not None:
            return True, "axolotl package installed (CLI used at call time)"
        return False, "axolotl is not installed (pip install axolotl — needs a CUDA GPU)"

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
        if not train:
            raise ValueError("axolotl backend: training set is empty")
        base = base_model or str(getattr(settings, "base_model", "") or "")
        if not base:
            raise ValueError("axolotl backend: no base_model given")

        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"axolotl backend is not available: {reason}")
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        cli = shutil.which("axolotl") or "axolotl"
        data_path = write_axolotl_dataset(train, out / "data", name="nm-corpus")
        yaml_text = build_axolotl_yaml(
            base_model=base, output_dir=out, data_path=data_path, settings=settings
        )
        yaml_path = out / "nm_axolotl.yml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        _log.info("axolotl: running %s train %s", cli, yaml_path)

        env = dict(os.environ)
        proc = subprocess.Popen(
            [cli, "train", str(yaml_path)],
            cwd=str(out),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            _log.debug("axolotl: %s", line)
            if on_step and "loss" in line:
                step = _step_from_line(line)
                loss = _loss_from_line(line)
                if step is not None and loss is not None:
                    on_step(step, loss)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                "axolotl exited "
                f"{rc} — last output: " + " | ".join(tail[-8:])[:500]
            )

        # trainer_state.json carries the loss history → gate metrics
        axo_out = out / "axolotl-out"
        candidates = sorted(axo_out.glob("checkpoint-*/trainer_state.json"))
        state_path = candidates[-1] if candidates else axo_out / "trainer_state.json"
        metrics: dict[str, Any] = {}
        liger_ok, _ = liger_available()
        info: dict[str, Any] = {
            "backend": self.name, "base_model": base,
            "liger_enabled": liger_ok,
        }
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                log_history = state.get("log_history") or []
                losses = [float(h["loss"]) for h in log_history if "loss" in h]
                steps = int(state.get("global_step") or 0)
                if losses:
                    final_loss = losses[-1]
                    metrics = {
                        "train_loss": round(final_loss, 6),
                        "score": round(1.0 / (1.0 + final_loss), 6),
                    }
                    info["loss_history"] = [round(v, 5) for v in losses[-50:]]
                    info["score_basis"] = "train_loss (no separate eval)"
                info["steps"] = steps
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                info["metrics_error"] = f"could not parse trainer_state.json: {exc}"
        if not metrics:
            raise RuntimeError(
                "axolotl reported success but no trainer_state.json with losses — "
                "the run produced no usable artifact"
            )

        adapters = sorted(axo_out.rglob("adapter_model.safetensors"))
        info["adapter"] = str(adapters[0].parent) if adapters else str(axo_out)
        _log.info("axolotl: done — final loss %.4f (%s)",
                  metrics["train_loss"],
                  "liger ON" if liger_ok else "liger off")
        return BackendResult(output_path=str(out), metrics=metrics, info=info)
