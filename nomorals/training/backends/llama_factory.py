"""The LLaMA-Factory backend: run a finetune through the LLaMA-Factory CLI.

For teams that already standardize on LLaMA-Factory (its dataset registry,
recipe zoo, and web UI), this adapter is the door in: it writes a real
dataset file, a real dataset_info.json entry, and a real training YAML into
the run directory, then executes ``llamafactory-cli train`` with its output
streamed into our logs, and reads the trainer state back out for the
promotion gate.

Import-safe: the CLI is located at call time; without it the backend reports
"not available" and the pipeline falls back to another backend.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from ...core.logging_setup import get_logger
from ..dataset import Example
from .base import BackendResult
from .unsloth import guess_template

__all__ = ["LlamaFactoryBackend", "build_yaml", "write_dataset_files"]

_log = get_logger(__name__)


def _to_messages(example: Example) -> list[dict[str, str]]:
    """Example turns → LLaMA-Factory ``messages`` format."""
    out: list[dict[str, str]] = []
    for turn in example.turns:
        role = "system" if turn.role == "system" else ("assistant" if turn.role == "assistant" else "user")
        out.append({"role": role, "content": turn.content})
    return out


def write_dataset_files(examples: Sequence[Example], dataset_dir: Path, *, name: str) -> Path:
    """Write ``<name>.jsonl`` (+ dataset_info.json) LLaMA-Factory can consume."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps({"messages": _to_messages(example)}, ensure_ascii=False) + "\n")

    info_path = dataset_dir / "dataset_info.json"
    info: dict[str, Any] = {}
    if info_path.exists():
        try:
            loaded = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                info = loaded
        except json.JSONDecodeError:
            _log.warning("dataset_info.json unreadable — rewriting from scratch")
    info[name] = {
        "file_name": f"{name}.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_yaml(
    *,
    name: str,
    base_model: str,
    output_dir: Path,
    dataset_dir: Path,
    settings: Any = None,
) -> str:
    """A complete, runnable LLaMA-Factory training config (stage: sft + LoRA)."""
    s = settings or _Defaults()
    rows = {
        "### model",
        f"model_name_or_path: {base_model}",
        "trust_remote_code: true",
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        f"lora_rank: {int(getattr(s, 'lora_r', 16) or 16)}",
        f"lora_alpha: {int(getattr(s, 'lora_alpha', 32) or 32)}",
        f"lora_dropout: {float(getattr(s, 'lora_dropout', 0.05) or 0.0)}",
        "lora_target: all-linear",
        "",
        "### dataset",
        f"dataset: {name}",
        "template: " + guess_template(base_model),
        "cutoff_len: " + str(int(getattr(s, "max_seq_len", 2048) or 2048)),
        "overwrite_cache: true",
        "preprocessing_num_workers: 2",
        "",
        "### output",
        f"output_dir: {output_dir / 'trainer'}",
        f"log_dir: {output_dir / 'logs'}",
        "logging_steps: 10",
        "save_steps: 500",
        "plot_loss: true",
        "overwrite_output_dir: true",
        "",
        "### train",
        f"per_device_train_batch_size: {int(getattr(s, 'batch_size', 4) or 4)}",
        f"gradient_accumulation_steps: {int(getattr(s, 'gradient_accumulation', 4) or 4)}",
        f"learning_rate: {float(getattr(s, 'learning_rate', 2e-5) or 2e-5)}",
        "num_train_epochs: " + str(float(int(getattr(s, "epochs", 3) or 3))),
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.1",
        "bf16: true",
        "gradient_checkpointing: true",
        "",
        "### eval",
        "do_eval: false",
        "val_size: 0.0",
    }
    return "\n".join(rows) + "\n"


class _Defaults:
    max_seq_len = 2048
    epochs = 3
    batch_size = 4
    gradient_accumulation = 4
    learning_rate = 2e-5
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05


class LlamaFactoryBackend:
    """Delegates to ``llamafactory-cli`` when it is installed."""

    name = "llama_factory"

    def available(self) -> tuple[bool, str]:
        cli = shutil.which("llamafactory-cli")
        if cli:
            return True, f"llamafactory-cli found ({cli})"
        if _spec_exists("llamafactory"):
            return True, "llamafactory package installed (entry point used at call time)"
        return False, "LLaMA-Factory is not installed (pip install llamafactory[torch] — or use the unsloth backend on Colab)"

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
            raise ValueError("llama_factory backend: training set is empty")
        base = base_model or str(getattr(settings, "base_model", "") or "")
        if not base:
            raise ValueError("llama_factory backend: no base_model given")

        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"llama_factory backend is not available: {reason}")
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        cli = shutil.which("llamafactory-cli") or "llamafactory-cli"
        dataset_name = "nm-corpus"
        write_dataset_files(train, out / "data", name=dataset_name)
        yaml_text = build_yaml(
            name=dataset_name, base_model=base,
            output_dir=out, dataset_dir=out / "data", settings=settings,
        )
        yaml_path = out / "nm_train.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        (out / "data").joinpath("dataset_info.json")  # already written above
        _log.info("llama_factory: running %s train %s", cli, yaml_path)

        env = dict(os.environ)
        env["LLAMAFACTORY_CACHE"] = str(out / "cache")
        proc = subprocess.Popen(
            [cli, "train", str(yaml_path)],
            cwd=str(out / "data"),  # dataset_info.json is read relative to cwd/data
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
            _log.debug("llamafactory: %s", line)
            if on_step and "loss" in line:
                step = _step_from_line(line)
                loss = _loss_from_line(line)
                if step is not None and loss is not None:
                    on_step(step, loss)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                "llamafactory-cli exited "
                f"{rc} — last output: " + " | ".join(tail[-8:])[:500]
            )

        # trainer_state.json carries the loss history → gate metrics
        state_path = out / "trainer" / "checkpoint-0" / "trainer_state.json"
        if not state_path.exists():
            candidates = sorted((out / "trainer").glob("checkpoint-*/trainer_state.json"))
            state_path = candidates[-1] if candidates else state_path
        metrics: dict[str, Any] = {}
        info: dict[str, Any] = {"backend": self.name, "base_model": base}
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
                "llamafactory-cli reported success but no trainer_state.json with losses — "
                "the run produced no usable artifact"
            )

        adapter_dir = out / "trainer" / "sft"
        if not adapter_dir.exists():
            candidates = [p for p in (out / "trainer").rglob("adapter_model.safetensors")]
            adapter_dir = candidates[0].parent if candidates else out / "trainer"
        info["adapter"] = str(adapter_dir)
        return BackendResult(output_path=str(out), metrics=metrics, info=info)


def _spec_exists(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


_STEP = re.compile(r"{'loss':\s*[\d.]+.*?'epoch':.*?}|step[_ ]?(\d+)|'step':\s*(\d+)")
_LOSS = re.compile(r"['\"]?loss['\"]?\s*[:=]\s*([\d.]+)")


def _loss_from_line(line: str) -> float | None:
    m = _LOSS.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _step_from_line(line: str) -> int | None:
    m = re.search(r"'step':\s*(\d+)", line) or re.search(r"step[_ ]?(\d+)", line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
