"""Universal wave: the MLX backend (Apple Silicon).

MLX is the ONLY training backend that runs on a Mac (no CUDA there),
and the unified-memory architecture makes M-series machines
legitimately good LoRA boxes.  These tests pin the data writer, the
``mlx_lm lora`` argv builder (iters math, validation flags), the loss
parser, registration, and the platform gate — on Linux, where the
backend must cleanly report itself unavailable.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.training.backends import (
    KNOWN_BACKENDS,
    available_backends,
    get_backend,
)
from nomorals.training.backends.mlx import (
    MLXBackend,
    build_mlx_args,
    parse_mlx_losses,
    write_mlx_data,
)
from nomorals.training.dataset import Example, Turn


def _example(i: int = 0) -> Example:
    return Example(
        turns=[
            Turn("user", f"What is the capital of region {i}?"),
            Turn("assistant", f"The capital of region {i} is Town{i}."),
        ]
    )


class MLXDataTests(unittest.TestCase):
    def test_writes_train_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            counts = write_mlx_data([_example(0), _example(1)], [_example(2)],
                                    Path(d))
            self.assertEqual(counts, {"train": 2, "valid": 1})
            train_rows = [json.loads(l) for l in
                          (Path(d) / "train.jsonl").read_text(
                              encoding="utf-8").splitlines()]
        self.assertEqual(train_rows[0]["messages"][0]["role"], "user")
        self.assertEqual(train_rows[0]["messages"][1]["role"], "assistant")

    def test_valid_omitted_when_no_eval_set(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            counts = write_mlx_data([_example()], [], Path(d))
            self.assertEqual(counts, {"train": 1})
            self.assertFalse((Path(d) / "valid.jsonl").exists())

    def test_empty_corpus_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_mlx_data([], [], Path(d))


class MLXArgsTests(unittest.TestCase):
    def _args(self, **kw) -> list[str]:
        return build_mlx_args(
            base_model=kw.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
            data_dir=Path("/tmp/d"),
            adapter_dir=Path("/tmp/a"),
            settings=kw.get("settings", SimpleNamespace(
                batch_size=4, epochs=3, learning_rate=1e-5, lora_r=8,
                lora_layers=16)),
            has_validation=kw.get("has_validation", False),
            train_rows=kw.get("train_rows", 100),
        )

    def test_invokes_mlx_lm_lora(self) -> None:
        argv = self._args()
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:4], ["-m", "mlx_lm", "lora"])
        self.assertIn("--train", argv)

    def test_iters_math(self) -> None:
        # epochs(3) * ceil(100/4) = 75
        argv = self._args()
        self.assertIn("75", argv[argv.index("--iters") + 1])

    def test_iters_floor(self) -> None:
        argv = self._args(train_rows=2)
        self.assertEqual(argv[argv.index("--iters") + 1], "10")

    def test_validation_flags_only_with_valid_set(self) -> None:
        self.assertNotIn("--steps-per-eval", self._args(has_validation=False))
        argv = self._args(has_validation=True)
        self.assertIn("--steps-per-eval", argv)
        self.assertIn("--val-batches", argv)

    def test_adapter_path_and_model(self) -> None:
        argv = self._args()
        self.assertEqual(argv[argv.index("--model") + 1],
                         "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.assertEqual(argv[argv.index("--adapter-path") + 1], "/tmp/a")


class MLXLossParseTests(unittest.TestCase):
    _LINES = [
        "Iter 1: Train loss 5.432, Learning Rate 1e-05, It/sec 2.1",
        "Iter 25: Train loss 3.210, Learning Rate 1e-05, It/sec 2.3",
        "Iter 25: Val loss 3.456, Val took 4.2s",
        "some unrelated line",
        "Iter 50: Train loss 2.100, Learning Rate 1e-05, It/sec 2.4",
    ]

    def test_parses_train_and_val(self) -> None:
        out = parse_mlx_losses(self._LINES)
        self.assertEqual(out["final_train_loss"], 2.1)
        self.assertEqual(out["final_val_loss"], 3.456)
        self.assertEqual(out["iters"], 50)
        self.assertEqual(len(out["train_history"]), 3)

    def test_empty_lines(self) -> None:
        out = parse_mlx_losses(["nothing here"])
        self.assertNotIn("final_train_loss", out)


class MLXRegistrationTests(unittest.TestCase):
    def test_registered_and_named(self) -> None:
        self.assertIn("mlx", KNOWN_BACKENDS)
        backend = get_backend("mlx")
        self.assertIsInstance(backend, MLXBackend)
        self.assertEqual(backend.name, "mlx")

    def test_unavailable_on_linux(self) -> None:
        ok, reason = get_backend("mlx").available()
        # this sandbox is linux — the gate must say so plainly
        if sys.platform != "darwin":
            self.assertFalse(ok)
            self.assertIn("Apple Silicon", reason)

    def test_available_true_on_mocked_mac(self) -> None:
        backend = get_backend("mlx")
        with patch.object(sys, "platform", "darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("importlib.util.find_spec",
                   return_value=SimpleNamespace(name="mlx_lm")):
            ok, reason = backend.available()
        self.assertTrue(ok)
        self.assertIn("Apple Silicon", reason)

    def test_available_backends_lists_mlx_without_crashing(self) -> None:
        rows = {r["name"]: r for r in available_backends()}
        self.assertIn("mlx", rows)
        self.assertIsInstance(rows["mlx"]["available"], bool)
        self.assertTrue(rows["mlx"]["reason"])

    def test_train_rejects_empty_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                get_backend("mlx").train([], output_dir=d, base_model="x")

    def test_train_requires_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                get_backend("mlx").train([_example()], output_dir=d)

    def test_train_refuses_when_unavailable(self) -> None:
        backend = get_backend("mlx")
        with patch.object(MLXBackend, "available",
                          return_value=(False, "no mac here")):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(RuntimeError) as ctx:
                    backend.train([_example()], output_dir=d, base_model="x")
        self.assertIn("not available", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
