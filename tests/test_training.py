"""L3 — training: codecs, tokenizer, preprocessing, native trainer, promotion gate.

The gate assertions matter most. A self-improvement loop without a working gate
monotonically degrades the system, because every run that "trains successfully"
gets shipped whether or not it improved anything.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.core.errors import NotFound, ValidationError
from nomorals.llm.registry import ModelRegistry
from nomorals.training.dataset import DatasetRegistry, Example, Turn, decode_example, to_chatml
from nomorals.training.preprocess import clean_text, dedupe, hamming, prepare, quality_filter, simhash, split
from nomorals.training.registry import RunStatus, TrainingRegistry
from nomorals.training.tokenize import BPETokenizer, _pretokenize
from nomorals.training.trainer import NativeTrainer, TrainConfig, TrainedModel
from nomorals.agents.context import build_context
from nomorals.core.config import Settings


def _examples(count: int = 40) -> list[Example]:
    # Long enough to clear the quality filter's 8-char prompt floor; the filter
    # correctly rejects "count 5" / "value 5" as teaching the model nothing.
    return [
        Example(
            turns=[
                Turn("user", f"please count the items in batch number {n}"),
                Turn("assistant", f"batch number {n} contains exactly {n} items."),
            ]
        )
        for n in range(count)
    ]


class CodecTests(unittest.TestCase):
    def test_chatml_marks_roles(self):
        text = to_chatml([Turn("user", "hi"), Turn("assistant", "hello")])
        self.assertIn("user", text)
        self.assertIn("assistant", text)
        self.assertEqual(text.count("<|im_start|>"), 2)

    def test_generation_prompt_appends_an_open_assistant_turn(self):
        text = to_chatml([Turn("user", "hi")], add_generation_prompt=True)
        self.assertTrue(text.endswith("assistant\n"))

    def test_alpaca_is_sniffed_by_its_keys(self):
        example = decode_example({"instruction": "Add", "input": "2+2", "output": "4"})
        self.assertIn("Add", example.prompt)
        self.assertIn("2+2", example.prompt)
        self.assertEqual(example.completion, "4")

    def test_sharegpt_maps_gpt_to_assistant(self):
        example = decode_example(
            {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
        )
        self.assertEqual([t.role for t in example.turns], ["user", "assistant"])

    def test_openai_messages_shape(self):
        example = decode_example({"messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(example.turns[0].role, "user")

    def test_pretrain_text_becomes_a_single_turn(self):
        example = decode_example({"text": "raw corpus"}, kind="pretrain")
        self.assertEqual(example.completion, "")

    def test_malformed_rows_raise_parse_error(self):
        from nomorals.core.errors import ParseError

        with self.assertRaises(ParseError):
            decode_example({"conversations": []})
        with self.assertRaises(ParseError):
            decode_example({"instruction": ""})
        with self.assertRaises(ParseError):
            decode_example({})

    def test_example_dict_round_trip(self):
        original = Example(turns=[Turn("user", "q"), Turn("assistant", "a")], source="s")
        restored = Example.from_dict(original.to_dict())
        self.assertEqual([t.content for t in restored.turns], ["q", "a"])
        self.assertEqual(restored.source, "s")


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.corpus = ["the quick brown fox jumps "] * 120 + ["over the lazy dog "] * 90

    def test_pretokenize_separates_punctuation(self):
        self.assertEqual(_pretokenize("Hi, there!")[:4], ["Hi", ",", " ", "there"])

    def test_training_produces_a_usable_vocabulary(self):
        tokenizer = BPETokenizer.train(self.corpus, vocab_size=400)
        self.assertGreater(tokenizer.vocab_size, 256)
        self.assertGreater(len(tokenizer.merges), 0)

    def test_encode_decode_round_trips_exactly(self):
        tokenizer = BPETokenizer.train(self.corpus, vocab_size=400)
        text = "the quick brown fox"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_utf8_survives_the_byte_level_round_trip(self):
        tokenizer = BPETokenizer.train(self.corpus + ["café naïve 日本語 "], vocab_size=400)
        text = "café naïve"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_training_is_deterministic(self):
        a = BPETokenizer.train(self.corpus, vocab_size=350)
        b = BPETokenizer.train(self.corpus, vocab_size=350)
        self.assertEqual(a.merges, b.merges)
        self.assertEqual(a.encode("the quick"), b.encode("the quick"))

    def test_merges_actually_reduce_sequence_length(self):
        tokenizer = BPETokenizer.train(self.corpus, vocab_size=400)
        self.assertLess(len(tokenizer.encode("the the the the")), len("the the the the"))

    def test_empty_corpus_is_rejected(self):
        with self.assertRaises(ValidationError):
            BPETokenizer.train([], vocab_size=400)

    def test_absurdly_small_vocabulary_is_rejected(self):
        with self.assertRaises(ValidationError):
            BPETokenizer.train(self.corpus, vocab_size=10)

    def test_save_and_load_preserve_behaviour(self):
        tokenizer = BPETokenizer.train(self.corpus, vocab_size=400)
        path = Path(tempfile.mkdtemp()) / "tok.json"
        self.addCleanup(shutil.rmtree, path.parent, ignore_errors=True)
        tokenizer.save(path)
        restored = BPETokenizer.load(path)
        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)
        self.assertEqual(restored.encode("the quick"), tokenizer.encode("the quick"))

    def test_loading_garbage_is_a_parse_error(self):
        from nomorals.core.errors import ParseError

        path = Path(tempfile.mkdtemp()) / "bad.json"
        self.addCleanup(shutil.rmtree, path.parent, ignore_errors=True)
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(ParseError):
            BPETokenizer.load(path)


class PreprocessTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace_and_strips_control(self):
        self.assertEqual(clean_text("a   b\n\n\n\nc\x00d"), "a b\n\ncd")

    def test_simhash_is_close_for_similar_text_and_far_for_unrelated(self):
        similar = hamming(simhash("the quick brown fox"), simhash("the quick brown fox jumps"))
        distant = hamming(simhash("the quick brown fox"), simhash("quantum chromodynamics"))
        self.assertLess(similar, distant)

    def test_dedupe_drops_near_duplicates_but_keeps_distinct_rows(self):
        examples = _examples(20)
        examples.append(_examples(6)[5])  # an exact duplicate of one already present
        self.assertEqual(len(dedupe(examples)), 20)

    def test_quality_filter_drops_degenerate_and_tiny_rows(self):
        examples = _examples(10)
        examples.append(Example(turns=[Turn("user", "x"), Turn("assistant", "y")]))
        examples.append(Example(turns=[Turn("user", "repeat"), Turn("assistant", "spam " * 200)]))
        self.assertEqual(len(quality_filter(examples)), 10)

    def test_split_is_deterministic_and_disjoint(self):
        examples = _examples(40)
        a_train, a_eval = split(examples, eval_fraction=0.2, seed=7)
        b_train, b_eval = split(examples, eval_fraction=0.2, seed=7)
        self.assertEqual([id(e) for e in a_train], [id(e) for e in b_train])
        self.assertFalse({id(e) for e in a_train} & {id(e) for e in a_eval})

    def test_prepare_runs_the_whole_pipeline(self):
        examples = _examples(40)
        examples.append(Example(turns=[Turn("user", "x"), Turn("assistant", "y")]))
        train, evaluation, stats = prepare(examples, eval_fraction=0.2)
        self.assertEqual(stats.input, 41)
        self.assertEqual(stats.filtered, 40)
        self.assertEqual(len(train) + len(evaluation), 40)

    def test_prepare_on_empty_input_does_not_crash(self):
        train, evaluation, stats = prepare([])
        self.assertEqual((train, evaluation, stats.input), ([], [], 0))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.examples = _examples(50)
        self.tokenizer = BPETokenizer.train(
            [e.to_chatml() for e in self.examples], vocab_size=400
        )
        self.train, self.evaluation, _ = prepare(self.examples, eval_fraction=0.2)

    def test_config_validation(self):
        with self.assertRaises(ValidationError):
            TrainConfig(learning_rate=0).validate()
        with self.assertRaises(ValidationError):
            TrainConfig(epochs=0).validate()
        with self.assertRaises(ValidationError):
            TrainConfig(hidden_size=0).validate()

    def test_loss_decreases_over_epochs(self):
        model, metrics = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=16, context_window=8, epochs=4, batch_size=6, learning_rate=0.08),
        ).fit(self.train, self.evaluation)
        history = metrics.train_loss_history
        self.assertEqual(len(history), 4)
        self.assertLess(history[-1], history[0], f"loss did not fall: {history}")

    def test_eval_loss_is_reported_when_a_held_out_set_exists(self):
        _, metrics = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=16, context_window=8, epochs=2, batch_size=6),
        ).fit(self.train, self.evaluation)
        self.assertTrue(metrics.eval_loss_history)
        self.assertNotEqual(metrics.best_eval_loss, float("inf"))

    def test_perplexity_is_the_exponent_of_the_loss(self):
        _, metrics = NativeTrainer(
            self.tokenizer, TrainConfig(hidden_size=8, context_window=8, epochs=1, batch_size=6)
        ).fit(self.train, self.evaluation)
        self.assertAlmostEqual(
            metrics.perplexity, min(1e8, 2.718281828 ** metrics.best_eval_loss), delta=1.0
        )

    def test_scores_are_shaped_for_the_promotion_gate(self):
        _, metrics = NativeTrainer(
            self.tokenizer, TrainConfig(hidden_size=8, context_window=8, epochs=1, batch_size=6)
        ).fit(self.train, self.evaluation)
        scores = metrics.as_scores()
        for key in ("train_loss", "eval_loss", "perplexity", "score"):
            self.assertIn(key, scores)
        self.assertGreater(scores["score"], 0.0)
        self.assertLessEqual(scores["score"], 1.0)

    def test_empty_training_set_is_rejected(self):
        with self.assertRaises(ValidationError):
            NativeTrainer(self.tokenizer, TrainConfig()).fit([])

    def test_cancel_stops_training_early(self):
        trainer = NativeTrainer(
            self.tokenizer, TrainConfig(hidden_size=8, context_window=8, epochs=50, batch_size=4)
        )
        trainer.cancel()
        _, metrics = trainer.fit(self.train)
        self.assertLess(metrics.steps, 5)

    def test_model_saves_and_reloads(self):
        model, _ = NativeTrainer(
            self.tokenizer, TrainConfig(hidden_size=8, context_window=8, epochs=1, batch_size=6)
        ).fit(self.train)
        directory = Path(tempfile.mkdtemp()) / "model"
        self.addCleanup(shutil.rmtree, directory.parent, ignore_errors=True)
        model.save(directory)
        restored = TrainedModel.load(directory)
        self.assertEqual(restored.vocab_size, model.vocab_size)
        self.assertEqual(restored.config.hidden_size, model.config.hidden_size)


class PromotionGateTests(unittest.TestCase):
    """The gate is the difference between self-improvement and self-degradation."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-gate-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.runs = TrainingRegistry(self.context.db)
        self.models = ModelRegistry(self.context.db)
        self.models.register("incumbent", kind="foundation", activate=True)
        self.models.record_eval("incumbent", {"score": 0.50})

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def _run(self, name: str, score: float) -> object:
        run = self.runs.start(name, base_model="incumbent", output_model=f"{name}-model")
        run.metrics = {"score": score, "eval_loss": 1.0 / max(score, 1e-6)}
        return self.runs.save(run)

    def test_a_better_model_passes_and_promotes(self):
        run = self._run("better", 0.80)
        self.assertTrue(self.runs.evaluate(run))
        self.assertTrue(self.runs.promote(run))
        self.assertEqual(self.models.active().name, "better-model")
        self.assertEqual(self.runs.get(run.id).status, RunStatus.PROMOTED)

    def test_a_worse_model_is_rejected_and_the_incumbent_stays_live(self):
        run = self._run("worse", 0.10)
        self.assertFalse(self.runs.evaluate(run))
        self.assertFalse(self.runs.promote(run))
        self.assertEqual(self.models.active().name, "incumbent")
        self.assertEqual(self.runs.get(run.id).status, RunStatus.REJECTED)

    def test_promotion_without_evaluation_is_refused(self):
        run = self.runs.start("unevaluated", output_model="unevaluated-model")
        with self.assertRaises(ValidationError):
            self.runs.promote(run)

    def test_force_promotes_but_records_the_override(self):
        run = self._run("forced", 0.10)
        self.runs.evaluate(run)
        self.assertTrue(self.runs.promote(run, force=True))
        self.assertEqual(self.models.active().name, "forced-model")
        saved = self.runs.get(run.id)
        self.assertTrue(saved.metadata.get("forced_promotion"))
        self.assertEqual(saved.status, RunStatus.PROMOTED)

    def test_evaluation_requires_numeric_metrics(self):
        run = self.runs.start("empty", output_model="empty-model")
        run.metrics = {"note": "no numbers here"}
        self.runs.save(run)
        with self.assertRaises(ValidationError):
            self.runs.evaluate(run)

    def test_started_and_finished_at_are_stamped(self):
        run = self._run("timed", 0.9)
        self.assertGreater(self.runs.get(run.id).started_at, 0.0)
        self.runs.evaluate(run)
        self.runs.promote(run)
        self.assertGreater(self.runs.get(run.id).finished_at, 0.0)

    def test_failure_is_recorded_with_its_reason(self):
        run = self.runs.start("crashed", output_model="crashed-model")
        self.runs.fail(run, error="CUDA out of memory")
        saved = self.runs.get(run.id)
        self.assertEqual(saved.status, RunStatus.FAILED)
        self.assertIn("CUDA", saved.error)

    def test_reap_marks_crashed_runs_failed(self):
        run = self.runs.start("orphan", output_model="orphan-model")
        self.assertEqual(run.status, RunStatus.RUNNING)
        self.assertEqual(self.runs.reap_interrupted(), 1)
        self.assertEqual(self.runs.get(run.id).status, RunStatus.FAILED)
        self.assertEqual(self.runs.reap_interrupted(), 0)

    def test_stats_counts_promotions_and_rejections(self):
        good = self._run("good", 0.9)
        self.runs.evaluate(good)
        self.runs.promote(good)
        bad = self._run("bad", 0.1)
        self.runs.evaluate(bad)
        self.runs.promote(bad)
        stats = self.runs.stats()
        self.assertEqual(stats["promoted"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["gate_passed"], 1)

    def test_get_unknown_run_raises(self):
        with self.assertRaises(NotFound):
            self.runs.get("does-not-exist")


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-ds-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.registry = DatasetRegistry(self.context.db)
        self.directory = Path(self.home) / "data"

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_register_examples_materializes_and_counts(self):
        dataset = self.registry.register_examples("corpus", _examples(12), self.directory)
        self.assertEqual(dataset.rows, 12)
        self.assertGreater(dataset.bytes, 0)
        self.assertEqual(len(dataset.checksum), 64)
        self.assertTrue(Path(dataset.path).is_file())

    def test_empty_dataset_is_refused(self):
        with self.assertRaises(ValidationError):
            self.registry.register_examples("empty", [], self.directory)

    def test_checksum_detects_a_changed_file(self):
        dataset = self.registry.register_examples("corpus", _examples(5), self.directory)
        from nomorals.training.dataset import file_checksum

        Path(dataset.path).write_text('{"messages":[]}\n', encoding="utf-8")
        self.assertNotEqual(file_checksum(dataset.path), dataset.checksum)

    def test_examples_stream_back_out_of_the_file(self):
        dataset = self.registry.register_examples("corpus", _examples(8), self.directory)
        restored = list(dataset.examples())
        self.assertEqual(len(restored), 8)
        self.assertIn("batch number 0", restored[0].prompt)

    def test_chatml_corpus_is_renderable(self):
        dataset = self.registry.register_examples("corpus", _examples(3), self.directory)
        rendered = list(dataset.chatml_corpus())
        self.assertEqual(len(rendered), 3)
        self.assertIn("assistant", rendered[0])

    def test_register_missing_file_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.registry.register("ghost", self.directory / "nope.jsonl")

    def test_unknown_kind_is_rejected(self):
        path = self.directory / "x.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValidationError):
            self.registry.register("x", path, kind="nonsense")

    def test_get_by_name_and_by_id(self):
        dataset = self.registry.register_examples("named", _examples(4), self.directory)
        self.assertEqual(self.registry.get(dataset.id).name, "named")
        self.assertEqual(self.registry.get("named").id, dataset.id)

    def test_missing_file_raises_on_stream(self):
        dataset = self.registry.register_examples("corpus", _examples(2), self.directory)
        Path(dataset.path).unlink()
        with self.assertRaises(NotFound):
            list(dataset.examples())

    def test_delete_optionally_removes_the_file(self):
        dataset = self.registry.register_examples("doomed", _examples(3), self.directory)
        path = Path(dataset.path)
        self.registry.delete(dataset.id, remove_file=True)
        self.assertFalse(path.exists())
        with self.assertRaises(NotFound):
            self.registry.get(dataset.id)

    def test_stats_aggregate(self):
        self.registry.register_examples("a", _examples(5), self.directory)
        self.registry.register_examples("b", _examples(7), self.directory)
        stats = self.registry.stats()
        self.assertEqual(stats["datasets"], 2)
        self.assertEqual(stats["rows"], 12)


if __name__ == "__main__":
    unittest.main()
