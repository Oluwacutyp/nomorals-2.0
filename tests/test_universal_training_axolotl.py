"""Universal wave: the Axolotl backend.

Axolotl is the free, YAML-reproducible, multi-GPU scale-out path
(unsloth OSS is single-GPU only; its multi-GPU tiers are paid).  These
tests exercise the pure surfaces — dataset writer, YAML builder
(including the Liger plugin block), registration, and error paths —
on a machine with no axolotl installed, exactly like the existing
llama_factory tests.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.training import liger as liger_mod
from nomorals.training.backends import (
    KNOWN_BACKENDS,
    available_backends,
    get_backend,
)
from nomorals.training.backends.axolotl import (
    AxolotlBackend,
    build_axolotl_yaml,
    write_axolotl_dataset,
)
from nomorals.training.dataset import Example, Turn


def _example(i: int = 0) -> Example:
    return Example(
        turns=[
            Turn("user", f"Explain topic {i} in detail."),
            Turn("assistant", f"Here is the explanation of topic {i}."),
        ]
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        max_seq_len=1024, epochs=2, batch_size=2, gradient_accumulation=4,
        learning_rate=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    )


class AxolotlDatasetTests(unittest.TestCase):
    def test_writes_openai_messages_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = write_axolotl_dataset([_example(0), _example(1)],
                                         Path(d), name="nm-corpus")
            self.assertTrue(path.exists())
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["messages"][0]["role"], "user")
        self.assertEqual(rows[0]["messages"][1]["role"], "assistant")
        self.assertIn("Explain topic 0", rows[0]["messages"][0]["content"])

    def test_drops_sub_two_message_examples(self) -> None:
        bad = Example(turns=[Turn("user", "only a user turn")])
        with tempfile.TemporaryDirectory() as d:
            path = write_axolotl_dataset([bad, _example(0)], Path(d), name="x")
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)


class AxolotlYamlTests(unittest.TestCase):
    def _yaml(self, **kw) -> str:
        with tempfile.TemporaryDirectory() as d:
            return build_axolotl_yaml(
                base_model=kw.get("base_model", "Qwen/Qwen2.5-7B-Instruct"),
                output_dir=Path(d),
                data_path=Path(d) / "data" / "nm-corpus.jsonl",
                settings=kw.get("settings", _settings()),
            )

    def test_core_keys(self) -> None:
        y = self._yaml()
        for key in ("base_model: Qwen/Qwen2.5-7B-Instruct",
                    "chat_template: tokenizer_default",
                    "type: chat_template",
                    "adapter: lora",
                    "lora_r: 16",
                    "sequence_len: 1024",
                    "num_epochs: 2",
                    "load_in_4bit: true",
                    "sample_packing: true"):
            self.assertIn(key, y)

    def test_settings_plumbing(self) -> None:
        y = self._yaml()
        self.assertIn("micro_batch_size: 2", y)
        self.assertIn("gradient_accumulation_steps: 4", y)
        self.assertIn("learning_rate: 0.0002", y)

    def test_liger_block_when_available(self) -> None:
        with patch.object(liger_mod, "_spec_exists", return_value=True):
            y = self._yaml()
        self.assertIn("axolotl.integrations.liger.LigerPlugin", y)
        self.assertIn("liger_fused_linear_cross_entropy: true", y)

    def test_liger_note_when_missing(self) -> None:
        with patch.object(liger_mod, "_spec_exists", return_value=False):
            y = self._yaml()
        self.assertNotIn("LigerPlugin", y)
        self.assertIn("liger-kernel", y)

    def test_default_settings_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            y = build_axolotl_yaml(
                base_model="x", output_dir=Path(d),
                data_path=Path(d) / "d.jsonl", settings=None)
        self.assertIn("adapter: lora", y)


class AxolotlRegistrationTests(unittest.TestCase):
    def test_registered_and_named(self) -> None:
        self.assertIn("axolotl", KNOWN_BACKENDS)
        backend = get_backend("axolotl")
        self.assertIsInstance(backend, AxolotlBackend)
        self.assertEqual(backend.name, "axolotl")

    def test_unknown_name_still_fails_loudly(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_backend("axololt")  # typo
        self.assertIn("axolotl", str(ctx.exception))

    def test_available_backends_lists_axolotl_without_crashing(self) -> None:
        rows = {r["name"]: r for r in available_backends()}
        self.assertIn("axolotl", rows)
        # not installed in this sandbox → unavailable, with a clear reason
        self.assertFalse(rows["axolotl"]["available"])
        self.assertIn("pip install axolotl", rows["axolotl"]["reason"])

    def test_train_rejects_empty_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                get_backend("axolotl").train(
                    [], output_dir=d, base_model="Qwen/Qwen2.5-7B-Instruct")

    def test_train_requires_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                get_backend("axolotl").train([_example()], output_dir=d)

    def test_train_refuses_when_cli_missing(self) -> None:
        backend = get_backend("axolotl")
        with patch.object(AxolotlBackend, "available",
                          return_value=(False, "nope")):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(RuntimeError) as ctx:
                    backend.train([_example()], output_dir=d,
                                  base_model="Qwen/Qwen2.5-7B-Instruct")
        self.assertIn("not available", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
