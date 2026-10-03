"""The MLX backend: LoRA finetunes on Apple Silicon, free and local.

Apple Silicon has no CUDA, so torch-based backends (unsloth, axolotl)
cannot train there at all — but an M-series Mac's unified memory is a
legitimately good training box (a 32 GB M1 Max comfortably LoRAs a 7B
in 4-bit).  Apple's MLX framework (Apache-2.0) plus ``mlx-lm`` (MIT)
is the free, first-party path: ``python -m mlx_lm lora`` with LoRA/
DoRA/QLoRA support, SFT/DPO/GRPO algorithms, and adapters that export
to GGUF for the phone.

This backend is import-safe AND platform-safe: on Linux/Windows or a
machine without ``mlx_lm`` it reports "not available" with the exact
reason and the pipeline falls back to another backend.  The MLX stack
only installs on ARM macs, so the import probe is the whole check.

Data contract: ``mlx_lm lora --data <dir>`` wants ``train.jsonl`` and
``valid.jsonl`` with ShareGPT-ish ``{"messages": [...]}`` rows; it
trains for ``--iters`` (not epochs), writes the adapter to
``--adapter-path``, and prints ``Iter N: Train loss X`` lines that we
parse for the promotion gate.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from ...core.logging_setup import get_logger
from ..dataset import Example
from .base import BackendResult

__all__ = ["MLXBackend", "build_mlx_args", "parse_mlx_losses", "write_mlx_data"]

_log = get_logger(__name__)

_LOSS_LINE = re.compile(r"Iter\s+(\d+)\s*:.*?Train loss\s+([\d.]+)")
_VAL_LINE = re.compile(r"Iter\s+(\d+)\s*:.*?Val loss\s+([\d.]+)")


def _to_openai_messages(example: Example) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in example.turns:
        role = "system" if turn.role == "system" else (
            "assistant" if turn.role == "assistant" else "user"
        )
        out.append({"role": role, "content": turn.content})
    return out


def write_mlx_data(
    train: Sequence[Example],
    evaluation: Sequence[Example],
    data_dir: Path,
) -> dict[str, int]:
    """Write ``train.jsonl`` (+ ``valid.jsonl`` when a held-out set is
    given) in the ``{"messages": [...]}`` shape ``mlx_lm lora`` reads."""
    data_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, examples in (("train", train), ("valid", evaluation)):
        if not examples:
            continue
        path = data_dir / f"{name}.jsonl"
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for example in examples:
                messages = _to_openai_messages(example)
                if len(messages) < 2:
                    continue
                fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                n += 1
        if n == 0:
            path.unlink(missing_ok=True)
        else:
            counts[name] = n
    if not counts.get("train"):
        raise ValueError("mlx backend: no usable training rows after formatting")
    return counts


def build_mlx_args(
    *,
    base_model: str,
    data_dir: Path,
    adapter_dir: Path,
    settings: Any = None,
    has_validation: bool = False,
    train_rows: int = 1,
) -> list[str]:
    """The ``python -m mlx_lm lora`` argv.

    mlx-lm trains by *iters*, not epochs: iters = epochs × steps-per-epoch
    (floored at 10 so a tiny set still moves the weights).  ``--steps-per-eval``
    is only passed when a validation file exists — without it mlx_lm would
    eval against nothing.
    """
    s = settings or _Defaults()
    batch = max(1, int(getattr(s, "batch_size", 4) or 4))
    epochs = max(1, int(getattr(s, "epochs", 3) or 3))
    iters = max(10, epochs * max(1, train_rows // batch))
    args = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(batch),
        "--learning-rate", str(float(getattr(s, "learning_rate", 1e-5) or 1e-5)),
        "--lora-layers", str(int(getattr(s, "lora_layers", 16) or 16)),
        "--rank", str(int(getattr(s, "lora_r", 8) or 8)),
        "--adapter-path", str(adapter_dir),
        "--save-every", "25",
    ]
    if has_validation:
        args += ["--steps-per-eval", "25", "--val-batches", "10"]
    return args


def parse_mlx_losses(lines: Sequence[str]) -> dict[str, Any]:
    """Pull (iter, train loss) and (iter, val loss) out of mlx_lm's
    stdout.  Returns the last losses seen plus the full train history."""
    train_hist: list[tuple[int, float]] = []
    val_hist: list[tuple[int, float]] = []
    for line in lines:
        m = _LOSS_LINE.search(line)
        if m:
            train_hist.append((int(m.group(1)), float(m.group(2))))
        m = _VAL_LINE.search(line)
        if m:
            val_hist.append((int(m.group(1)), float(m.group(2))))
    out: dict[str, Any] = {"train_history": train_hist, "val_history": val_hist}
    if train_hist:
        out["final_train_loss"] = train_hist[-1][1]
        out["iters"] = train_hist[-1][0]
    if val_hist:
        out["final_val_loss"] = val_hist[-1][1]
    return out


class _Defaults:
    batch_size = 4
    epochs = 3
    learning_rate = 1e-5  # mlx_lm's own default for LoRA
    lora_r = 8            # mlx_lm's default rank
    lora_layers = 16      # mlx_lm's default: last N layers get adapters


class MLXBackend:
    """LoRA via ``mlx_lm`` — the only training backend that runs on a Mac.

    On a 32 GB M-series machine this is genuinely competitive with a
    free-tier T4 for 7B-class LoRA (unified memory, no VRAM ceiling to
    negotiate), and it costs nothing.  Everywhere else it cleanly
    reports itself unavailable.
    """

    name = "mlx"

    def available(self) -> tuple[bool, str]:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False, (
                f"MLX runs on Apple Silicon only (this is "
                f"{sys.platform}/{platform.machine()})"
            )
        if importlib.util.find_spec("mlx_lm") is None:
            return False, "mlx_lm is not installed (pip install mlx-lm — macOS ARM only)"
        return True, "mlx_lm available — training on Apple Silicon"

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
            raise ValueError("mlx backend: training set is empty")
        base = base_model or str(getattr(settings, "base_model", "") or "")
        if not base:
            raise ValueError(
                "mlx backend: no base_model given — pass an HF id or a local "
                "path (mlx-community repos carry pre-quantized 4-bit weights)"
            )

        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"mlx backend is not available: {reason}")
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        counts = write_mlx_data(train, evaluation, out / "data")
        adapter_dir = out / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        argv = build_mlx_args(
            base_model=base,
            data_dir=out / "data",
            adapter_dir=adapter_dir,
            settings=settings,
            has_validation=bool(counts.get("valid")),
            train_rows=counts["train"],
        )
        _log.info("mlx: running %s", " ".join(argv[2:]))

        env = dict(os.environ)
        proc = subprocess.Popen(
            argv,
            cwd=str(out),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            _log.debug("mlx_lm: %s", line)
            if on_step:
                m = _LOSS_LINE.search(line)
                if m:
                    on_step(int(m.group(1)), float(m.group(2)))
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                "mlx_lm lora exited "
                f"{rc} — last output: " + " | ".join(lines[-8:])[:500]
            )

        parsed = parse_mlx_losses(lines)
        if "final_train_loss" not in parsed and "final_val_loss" not in parsed:
            raise RuntimeError(
                "mlx_lm reported success but printed no loss lines — "
                "the run produced no usable metrics"
            )
        # Prefer the validation loss for the gate when mlx_lm measured one;
        # it is the honest generalization signal.
        basis_loss = parsed.get("final_val_loss", parsed.get("final_train_loss"))
        assert basis_loss is not None
        metrics = {
            "train_loss": round(float(parsed.get("final_train_loss", basis_loss)), 6),
            "score": round(1.0 / (1.0 + float(basis_loss)), 6),
        }
        if "final_val_loss" in parsed:
            metrics["eval_loss"] = round(float(parsed["final_val_loss"]), 6)
        info = {
            "backend": self.name,
            "base_model": base,
            "adapter": str(adapter_dir),
            "score_basis": (
                "val_loss (mlx_lm measured)" if "final_val_loss" in parsed
                else "train_loss (no validation set)"
            ),
            "iters": parsed.get("iters", 0),
            "train_rows": counts["train"],
            "loss_history": [round(v, 5) for _, v in parsed["train_history"][-50:]],
        }
        adapter_files = sorted(p.name for p in adapter_dir.glob("adapters*"))
        if adapter_files:
            info["adapter_files"] = adapter_files[:4]
        _log.info("mlx: done — %s", info["score_basis"])
        return BackendResult(output_path=str(out), metrics=metrics, info=info)
