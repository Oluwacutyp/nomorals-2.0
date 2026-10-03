"""Offline tests for the training backends: registry, pure helpers, and the
pipeline dispatch contract.

These run on a machine with no torch, no CUDA, and no LLaMA-Factory — exactly
the phone case. The heavy backends are exercised through their pure surfaces
(dataset/YAML/template writers, availability reporting, error paths) and the
registry/dispatch layer, so a missing dependency degrades to a clear error
instead of an import crash.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.training.backends import (
    KNOWN_BACKENDS,
    available_backends,
    get_backend,
)
from nomorals.training.backends.base import BackendResult
from nomorals.training.backends.llama_factory import (
    LlamaFactoryBackend,
    build_yaml,
    write_dataset_files,
)
from nomorals.training.backends.llama_factory import _loss_from_line, _step_from_line
from nomorals.training.backends.native import NativeBackend, default_train_config
from nomorals.training.backends.unsloth import (
    UnslothBackend,
    dataset_from_examples,
    guess_template,
)
from nomorals.training.dataset import Example, Turn
from nomorals.training.evaluate import (
    GoldenCase,
    REFUSAL_MARKERS,
    build_gate_scores,
    default_golden_cases,
    evaluate_native,
    grade_generation,
    grade_golden_set,
)
from nomorals.training.trainer import TrainedModel
from nomorals.training.tokenize import BPETokenizer

_TORCH_PRESENT = importlib.util.find_spec("torch") is not None


def _example(i: int = 0) -> Example:
    return Example(
        turns=[
            Turn("user", f"Tell me about topic {i}. The weather in Enugu was warm today."),
            Turn("assistant", f"Here is what I know about topic {i}. It is a warm climate."),
        ]
    )


def _corpus(n: int = 10) -> list[Example]:
    return [_example(i) for i in range(n)]


# ── registry ─────────────────────────────────────────────────────────────────


class RegistryTests(unittest.TestCase):
    def test_known_backends_are_the_five_implemented(self) -> None:
        self.assertEqual(
            set(KNOWN_BACKENDS),
            {"native", "unsloth", "llama_factory", "axolotl", "mlx"},
        )

    def test_default_backend_is_native(self) -> None:
        self.assertIs(get_backend("").name, "native")
        self.assertIs(get_backend("  Native ").name, "native")

    def test_unknown_backend_raises_with_the_valid_list(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_backend("bogus")
        for name in KNOWN_BACKENDS:
            self.assertIn(name, str(ctx.exception))

    def test_available_backends_reports_every_known_one(self) -> None:
        rows = {row["name"]: row for row in available_backends()}
        self.assertEqual(set(rows), set(KNOWN_BACKENDS))
        for row in rows.values():
            self.assertIsInstance(row["available"], bool)
            self.assertIsInstance(row["reason"], str)
            if not row["available"]:
                self.assertTrue(row["reason"].strip())
        # native has no dependencies — it must never report unavailable
        self.assertTrue(rows["native"]["available"])

    def test_every_backend_implements_the_protocol(self) -> None:
        for name, cls in KNOWN_BACKENDS.items():
            backend = cls()
            self.assertEqual(backend.name, name)
            ok, reason = backend.available()
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(reason, str)
            self.assertTrue(callable(backend.train))

    def test_import_safety_without_torch(self) -> None:
        """The whole point of the lazy imports: on a phone (no torch) every
        backend module imports and reports its availability honestly."""
        from nomorals.training import backends  # noqa: F401

        rows = {row["name"]: row for row in available_backends()}
        if not _TORCH_PRESENT:
            self.assertFalse(rows["unsloth"]["available"])
            self.assertIn("torch", rows["unsloth"]["reason"])
            self.assertFalse(rows["llama_factory"]["available"])


class BackendResultTests(unittest.TestCase):
    def test_ok_is_driven_by_output_path(self) -> None:
        self.assertFalse(BackendResult().ok)
        self.assertTrue(BackendResult(output_path="/tmp/x").ok)


# ── native backend ───────────────────────────────────────────────────────────


class NativeBackendTests(unittest.TestCase):
    def test_default_train_config_honours_settings_and_clamps(self) -> None:
        settings = SimpleNamespace(
            max_seq_len=100, learning_rate=1e-6, epochs=2, batch_size=8
        )
        config = default_train_config(settings)
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 8)
        # learning rate floored at 1e-4, context window capped at 16, floor 2
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(config.context_window, 16)
        small = default_train_config(SimpleNamespace(
            max_seq_len=1, learning_rate=1e-3, epochs=0, batch_size=0))
        self.assertEqual(small.context_window, 2)
        self.assertEqual(small.epochs, 1)
        self.assertEqual(small.batch_size, 1)

    def test_train_end_to_end_produces_artifacts_and_gate_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-native-") as tmp:
            result = NativeBackend().train(_corpus(), _corpus(2), output_dir=tmp)
            self.assertTrue(result.ok)
            out = Path(result.output_path)
            self.assertTrue((out / "model.json").is_file())
            self.assertTrue((out / "tokenizer.json").is_file())
            metrics = result.metrics
            self.assertIn("train_loss", metrics)
            self.assertIn("score", metrics)
            self.assertGreater(metrics["score"], 0.0)
            self.assertLessEqual(metrics["score"], 1.0)
            self.assertGreater(metrics.get("perplexity", 0), 1.0)
            self.assertEqual(result.info["architecture"], "native-mlp-lm")

    def test_empty_training_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NativeBackend().train([], output_dir=tempfile.mkdtemp())

    def test_saved_model_loads_and_evaluate_native_scores_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-native-eval-") as tmp:
            result = NativeBackend().train(_corpus(), _corpus(3), output_dir=tmp)
            model = TrainedModel.load(tmp)
            tokenizer = BPETokenizer.load(Path(tmp) / "tokenizer.json")
            scored = evaluate_native(model, tokenizer, _corpus(4))
        self.assertIn("eval_loss", scored)
        self.assertGreater(scored["eval_loss"], 0.0)
        self.assertGreater(scored["perplexity"], 1.0)
        self.assertGreater(scored["tokens"], 0)


# ── unsloth backend (pure surfaces) ──────────────────────────────────────────


class UnslothBackendTests(unittest.TestCase):
    def test_guess_template_by_family(self) -> None:
        self.assertEqual(guess_template("Qwen/Qwen2.5-7B-Instruct"), "qwen")
        self.assertEqual(guess_template("mistralai/Mixtral-8x7B"), "mistral")
        self.assertEqual(guess_template("google/gemma-2-9b"), "gemma")
        self.assertEqual(guess_template("microsoft/Phi-3.5-mini"), "phi")
        self.assertEqual(guess_template("cognitivecomputations/dolphin-2.9-llama3-8b"), "llama")
        self.assertEqual(guess_template(""), "llama")

    def test_dataset_from_examples_is_preformatted_chatml(self) -> None:
        examples = _corpus(3)
        rows = dataset_from_examples(examples)
        self.assertEqual(len(rows), 3)
        for row, example in zip(rows, examples):
            self.assertEqual(row, {"text": example.to_chatml()})
        prompted = dataset_from_examples(examples, add_generation_prompt=True)
        # preformatted rows end on the (empty) assistant turn when prompted
        self.assertTrue(prompted[0]["text"].rstrip().endswith(
            "<|im_start|>assistant".replace("<|im_start|>", "<|im_start|>")))
        self.assertNotIn("<|im_end|>\n<|im_start|>assistant\n<|im_end|>", prompted[0]["text"])

    def test_available_reports_missing_dependencies_honestly(self) -> None:
        ok, reason = UnslothBackend().available()
        if not _TORCH_PRESENT:
            self.assertFalse(ok)
            self.assertIn("torch", reason)

    def test_train_requires_a_base_model(self) -> None:
        backend = UnslothBackend()
        with self.assertRaises(ValueError) as ctx:
            backend.train(_corpus(2), output_dir=tempfile.mkdtemp())
        self.assertIn("base_model", str(ctx.exception))

    def test_train_without_unsloth_raises_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-unsloth-") as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                UnslothBackend().train(
                    _corpus(2), output_dir=tmp,
                    base_model="cognitivecomputations/dolphin-2.9-llama3-8b",
                )
        self.assertIn("unsloth", str(ctx.exception))

    def test_empty_training_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UnslothBackend().train([], output_dir=tempfile.mkdtemp(),
                                   base_model="x/y")


# ── llama-factory backend (pure surfaces) ────────────────────────────────────


class LlamaFactoryBackendTests(unittest.TestCase):
    def test_to_messages_maps_roles(self) -> None:
        from nomorals.training.backends.llama_factory import _to_messages

        example = Example(turns=[
            Turn("system", "You are terse."),
            Turn("user", "hi"),
            Turn("assistant", "hello"),
            Turn("tool", "odd role"),
        ])
        self.assertEqual(_to_messages(example), [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "odd role"},
        ])

    def test_write_dataset_files_emits_sharegpt_jsonl_and_info(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-lf-data-") as tmp:
            data = Path(tmp)
            path = write_dataset_files(_corpus(3), data, name="nm-corpus")
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            first = json.loads(lines[0])
            self.assertEqual(first["messages"][0]["role"], "user")
            self.assertEqual(first["messages"][1]["role"], "assistant")

            info = json.loads((data / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["nm-corpus"]["formatting"], "sharegpt")
            self.assertEqual(info["nm-corpus"]["file_name"], "nm-corpus.jsonl")

            # a second dataset merges into the same dataset_info.json
            write_dataset_files(_corpus(1), data, name="second")
            info2 = json.loads((data / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertIn("nm-corpus", info2)
            self.assertIn("second", info2)

    def test_build_yaml_is_a_complete_sft_lora_config(self) -> None:
        settings = SimpleNamespace(
            max_seq_len=1024, epochs=2, batch_size=6,
            gradient_accumulation=8, learning_rate=1e-5,
            lora_r=32, lora_alpha=64, lora_dropout=0.1,
        )
        text = build_yaml(
            name="nm-corpus",
            base_model="cognitivecomputations/dolphin-2.9-llama3-8b",
            output_dir=Path("/out"), dataset_dir=Path("/out/data"),
            settings=settings,
        )
        for needle in (
            "model_name_or_path: cognitivecomputations/dolphin-2.9-llama3-8b",
            "stage: sft",
            "do_train: true",
            "finetuning_type: lora",
            "lora_rank: 32",
            "lora_alpha: 64",
            "lora_dropout: 0.1",
            "lora_target: all-linear",
            "dataset: nm-corpus",
            "template: llama",
            "cutoff_len: 1024",
            "per_device_train_batch_size: 6",
            "gradient_accumulation_steps: 8",
            "learning_rate: 1e-05",
            "num_train_epochs: 2.0",
        ):
            self.assertIn(needle, text)
        self.assertIn("output_dir: /out/trainer", text)

    def test_build_yaml_template_follows_the_model_family(self) -> None:
        text = build_yaml(
            name="d", base_model="Qwen/Qwen2.5-7B-Instruct",
            output_dir=Path("/o"), dataset_dir=Path("/o/data"),
        )
        self.assertIn("template: qwen", text)

    def test_available_reports_honestly_when_missing(self) -> None:
        ok, reason = LlamaFactoryBackend().available()
        if not importlib.util.find_spec("llamafactory") and not _which_llama_factory():
            self.assertFalse(ok)
            self.assertTrue(reason.strip())

    def test_train_without_base_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LlamaFactoryBackend().train(_corpus(1), output_dir=tempfile.mkdtemp())

    def test_loss_and_step_line_parsing(self) -> None:
        self.assertAlmostEqual(
            _loss_from_line("{'loss': 1.234, 'grad_norm': 0.5, 'epoch': 0.33}"), 1.234)
        self.assertAlmostEqual(
            _loss_from_line("{'eval_loss': 0.9, 'step': 42}"), 0.9)
        self.assertIsNone(_loss_from_line("{'grad_norm': 0.5}"))
        self.assertEqual(_step_from_line("{'loss': 1.0, 'step': 42}"), 42)
        self.assertEqual(_step_from_line("loss 0.9 step 7"), 7)
        self.assertIsNone(_step_from_line("no numbers here"))

    def test_train_without_the_cli_raises_a_clear_error(self) -> None:
        if _which_llama_factory() or importlib.util.find_spec("llamafactory"):
            self.skipTest("llamafactory is installed here")
        with tempfile.TemporaryDirectory(prefix="nm-lf-") as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                LlamaFactoryBackend().train(
                    _corpus(1), output_dir=tmp, base_model="x/y")
        self.assertIn("not available", str(ctx.exception))


def _which_llama_factory() -> str | None:
    import shutil

    return shutil.which("llamafactory-cli")


# ── evaluation (backend-blind) ───────────────────────────────────────────────


class GoldenSetTests(unittest.TestCase):
    def test_grade_generation_accepts_and_rejects(self) -> None:
        case = GoldenCase(prompt="q", must_contain=("108",),
                          must_not_contain=REFUSAL_MARKERS, min_chars=10)
        ok, reasons = grade_generation(case, "The answer is 108 and nothing else.")
        self.assertTrue(ok, reasons)
        ok, reasons = grade_generation(case, "I'm sorry, I can't answer that.")
        self.assertFalse(ok)
        self.assertTrue(any("forbidden" in r for r in reasons))

    def test_refusal_markers_are_case_insensitive(self) -> None:
        case = GoldenCase(prompt="q", must_not_contain=("as an AI",))
        ok, _ = grade_generation(case, "As an AI I would say…")
        self.assertFalse(ok)

    def test_default_golden_cases_have_real_prompts(self) -> None:
        cases = default_golden_cases()
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertTrue(case.prompt.strip())

    def test_grade_golden_set_passes_a_cooperative_generator(self) -> None:
        def generate(prompt: str) -> str:
            if "27 times 4" in prompt:
                return "108"
            if "fox" in prompt:
                return "the quick brown fox"
            return "hey, how's it going — long enough to count as a real answer"

        report = grade_golden_set(default_golden_cases(), generate)
        self.assertEqual(report["golden_passed"], report["golden_total"])
        self.assertEqual(report["golden_score"], 1.0)
        self.assertEqual(report["golden_failures"], [])

    def test_grade_golden_set_flags_refusals(self) -> None:
        report = grade_golden_set(
            default_golden_cases(), lambda prompt: "I'm sorry, I can't help with that.")
        self.assertLess(report["golden_passed"], report["golden_total"])
        self.assertTrue(report["golden_failures"])
        self.assertTrue(report["golden_failures"][0]["reasons"])

    def test_generator_exceptions_become_failure_rows(self) -> None:
        def boom(prompt: str) -> str:
            raise RuntimeError("provider down")

        report = grade_golden_set(default_golden_cases(), boom)
        self.assertEqual(report["golden_passed"], 0)
        self.assertEqual(len(report["golden_failures"]), 4)
        self.assertIn("generator raised", report["golden_failures"][0]["reasons"][0])


class GateScoresTests(unittest.TestCase):
    def test_loss_only(self) -> None:
        scores = build_gate_scores(train_loss=1.0, backend="native")
        self.assertAlmostEqual(scores["score"], 0.5)
        self.assertEqual(scores["score_basis"], "loss (train)")
        scores = build_gate_scores(eval_loss=1.0, train_loss=0.5, backend="unsloth")
        self.assertAlmostEqual(scores["score"], 0.5)
        self.assertEqual(scores["score_basis"], "loss (eval)")

    def test_golden_only(self) -> None:
        scores = build_gate_scores(golden={"golden_score": 0.75, "golden_failures": []})
        self.assertAlmostEqual(scores["score"], 0.75)
        self.assertEqual(scores["score_basis"], "golden only")

    def test_blend_is_70_30(self) -> None:
        scores = build_gate_scores(
            eval_loss=1.0,  # loss score 0.5
            golden={"golden_score": 1.0, "golden_failures": []},
        )
        self.assertAlmostEqual(scores["score"], 0.7 * 0.5 + 0.3 * 1.0)
        self.assertEqual(scores["score_basis"], "0.7*loss + 0.3*golden")

    def test_score_is_clamped_to_the_unit_interval(self) -> None:
        scores = build_gate_scores(train_loss=0.0)
        self.assertLessEqual(scores["score"], 1.0)
        scores = build_gate_scores(train_loss=1e6)
        self.assertGreaterEqual(scores["score"], 0.0)

    def test_no_signal_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_gate_scores()

    def test_golden_failures_are_recorded(self) -> None:
        failures = [{"description": "x", "prompt": "q", "reasons": ["r"]}]
        scores = build_gate_scores(
            eval_loss=1.0, golden={"golden_score": 0.5, "golden_failures": failures})
        self.assertEqual(scores["golden_failures"], failures)


# ── pipeline dispatch (native e2e + honest external failures) ───────────────


class PipelineBackendDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.core.config import Settings

        self.home = tempfile.mkdtemp(prefix="nm-be-dispatch-")
        self.context = build_context(
            Settings(home=self.home),
            with_executor=False, with_router=False,
            with_memory=False, with_tools=False,
        )
        self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def _seed(self) -> None:
        db = self.context.db
        for index in range(8):
            db.insert(
                "memories",
                {
                    "id": f"memory-{index}", "kind": "episode",
                    "content": f"A durable training example with a distinct useful topic {index}.",
                    "created_at": float(index + 1), "updated_at": float(index + 1),
                },
            )
        self.context.settings.training.epochs = 1
        self.context.settings.training.eval_split = 0.2

    def _job(self):
        from nomorals.self_improvement import SelfImprovementJob
        from nomorals.training.policy import RetrainingPolicy

        return SelfImprovementJob(
            self.context, policy=RetrainingPolicy(dataset_growth_threshold=1, interval_seconds=0)
        )

    def test_explicit_backend_is_recorded_on_the_run_row(self) -> None:
        self._seed()
        result = self._job().run(force=True, backend="native")
        self.assertTrue(result.ok)
        run = self._job().runs.get(result.run_id)
        self.assertEqual(run.backend, "native")
        self.assertIn("score", run.metrics)

    def test_external_backend_without_base_model_fails_loudly(self) -> None:
        self._seed()
        result = self._job().run(force=True, backend="unsloth")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "failed")
        self.assertIn("NM_TRAINING_BASE_MODEL", result.error)
        from nomorals.training.registry import RunStatus

        run = self._job().runs.get(result.run_id)
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertIn("base model id", run.error)

    def test_external_backend_unavailable_fails_loudly(self) -> None:
        if _TORCH_PRESENT:
            self.skipTest("torch is present — the honest failure path is machine-specific")
        self._seed()
        result = self._job().run(
            force=True, backend="unsloth",
            base_model="cognitivecomputations/dolphin-2.9-llama3-8b",
        )
        self.assertFalse(result.ok)
        self.assertIn("not available on this machine", result.error)
        self.assertIn("torch", result.error)


# ── CLI surface ──────────────────────────────────────────────────────────────


class CLITrainingFlagsTests(unittest.TestCase):
    def test_parser_accepts_the_new_flags(self) -> None:
        from nomorals.cli import _parser

        args = _parser().parse_args(["train", "--backends"])
        self.assertTrue(args.backends)
        args = _parser().parse_args(
            ["train", "--run", "--backend", "unsloth", "--base-model", "x/y"])
        self.assertEqual(args.backend, "unsloth")
        self.assertEqual(args.base_model, "x/y")
        # defaults stay native
        args = _parser().parse_args(["train"])
        self.assertEqual(args.backend, "")

    def test_cmd_train_backends_lists_honest_availability(self) -> None:
        import io
        from contextlib import redirect_stdout

        from nomorals.agents.context import build_context
        from nomorals.cli import _parser, _cmd_train
        from nomorals.core.config import Settings

        home = tempfile.mkdtemp(prefix="nm-cli-be-")
        with build_context(
            Settings(home=home),
            with_executor=False, with_router=False,
            with_memory=False, with_tools=False,
        ) as context:
            args = _parser().parse_args(["train", "--backends"])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = _cmd_train(args, context)
        self.assertEqual(code, 0)
        out = buffer.getvalue()
        self.assertIn("configured backend:", out)
        self.assertIn("ok   native", out)
        for name in ("unsloth", "llama_factory"):
            self.assertIn(name, out)


if __name__ == "__main__":
    unittest.main()
