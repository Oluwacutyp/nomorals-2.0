"""Wave 92: the C++ training kernel.

The phone's training backend (NativeTrainer) was pure Python — every
(context, target) pair paid interpreter overhead on each multiply-add.
Wave 92 adds ``nomorals/native/mlptrain.cpp``: one C ABI call per batch
that runs the SAME math (same operation order, same clamps) and applies
the same SGD step.  The pure-Python trainer is the reference — these
tests compare loss AND every gradient element, and run a full training
run on both kernels to prove the models come out the same.
"""

from __future__ import annotations

import array
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals import native  # noqa: E402
from nomorals.training import trainer as T  # noqa: E402
from nomorals.training.dataset import Example, Turn  # noqa: E402
from nomorals.training.tokenize import BPETokenizer  # noqa: E402

VOCAB = 24
HIDDEN = 8
CONTEXT = 4
SEED = 777


def _random_setup(rng_seed: int = SEED):
    """Random weights + a batch of token sequences, as flat 'd' buffers."""
    rng = random.Random(rng_seed)
    embedding = array.array("d", (rng.uniform(-0.5, 0.5) for _ in range(VOCAB * HIDDEN)))
    hidden_w = array.array("d", (rng.uniform(-0.2, 0.2) for _ in range(HIDDEN * HIDDEN)))
    hidden_b = array.array("d", (rng.uniform(-0.1, 0.1) for _ in range(HIDDEN)))
    out_w = array.array("d", (rng.uniform(-0.1, 0.1) for _ in range(HIDDEN * VOCAB)))
    out_b = array.array("d", (rng.uniform(-0.1, 0.1) for _ in range(VOCAB)))
    seqs = []
    for _ in range(6):
        length = rng.randint(CONTEXT, CONTEXT + 10)
        seqs.append([rng.randrange(VOCAB) for _ in range(length)])
    return embedding, hidden_w, hidden_b, out_w, out_b, seqs


def _python_reference(seq_lists, embedding, hidden_w, hidden_b, out_w, out_b):
    """Run the pure-Python batch math and return (loss, count, grads)."""
    trainer = T.NativeTrainer.__new__(T.NativeTrainer)
    trainer.config = T.TrainConfig(
        context_window=CONTEXT, hidden_size=HIDDEN, seed=SEED)
    # lists, exactly what the Python path consumes
    emb = [list(embedding[i * HIDDEN:(i + 1) * HIDDEN]) for i in range(VOCAB)]
    hw = [list(hidden_w[i * HIDDEN:(i + 1) * HIDDEN]) for i in range(HIDDEN)]
    hb = list(hidden_b)
    ow = [list(out_w[i * VOCAB:(i + 1) * VOCAB]) for i in range(HIDDEN)]
    ob = list(out_b)
    return trainer._batch_gradients(seq_lists, emb, hw, hb, ow, ob)


class NativeMlpBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not native.mlp_available():
            ok, msg = native.build(force=True)
            assert ok, msg

    def test_build_produces_both_libraries(self):
        self.assertTrue(native.lib_path().exists())
        self.assertTrue(native.mlp_lib_path().exists())

    def test_version_strings(self):
        self.assertIn(".", native.load_mlp().nm_mlp_version().decode())


class NativeMlpParityTest(unittest.TestCase):
    """C++ batch == pure-Python batch, element for element."""

    def test_gradients_and_loss_match(self):
        embedding, hw, hb, ow, ob, seq_lists = _random_setup()
        py_loss, ge, ghw, ghb, gow, gob, py_count = _python_reference(
            seq_lists, embedding, hw, hb, ow, ob)

        seqs = array.array("i")
        for s in seq_lists:
            seqs.extend(s)
        lens = array.array("i", (len(s) for s in seq_lists))
        c_loss, c_count, scratch = native.mlp_batch(
            seqs, lens, array.array("d", embedding), array.array("d", hw),
            array.array("d", hb), array.array("d", ow), array.array("d", ob),
            context=CONTEXT, lr=0.0)

        self.assertEqual(c_count, py_count)
        self.assertGreater(c_count, 0)
        self.assertAlmostEqual(c_loss, py_loss, delta=1e-12)

        # gradient slices: embed | hidden_w | hidden_b | out_w | out_b
        off_e, off_w, off_b = 0, VOCAB * HIDDEN, VOCAB * HIDDEN + HIDDEN * HIDDEN
        off_ow = off_b + HIDDEN
        off_ob = off_ow + HIDDEN * VOCAB
        slices = {
            "embed": (scratch[off_e:off_w], ge),
            "hidden_w": (scratch[off_w:off_b], ghw),
            "hidden_b": (scratch[off_b:off_ow], ghb),
            "out_w": (scratch[off_ow:off_ob], gow),
            "out_b": (scratch[off_ob:], gob),
        }
        for name, (c_vals, p_vals) in slices.items():
            if name in ("embed", "hidden_w", "out_w"):
                p_flat = [v for row in p_vals for v in row]
            else:
                p_flat = p_vals
            self.assertEqual(len(c_vals), len(p_flat), name)
            for index, (c_v, p_v) in enumerate(zip(c_vals, p_flat)):
                self.assertAlmostEqual(c_v, p_v, delta=1e-12,
                                       msg=f"{name}[{index}]: {c_v} vs {p_v}")

    def test_sgd_step_matches_python_axpy(self):
        embedding, hw, hb, ow, ob, seq_lists = _random_setup()
        lr, l2 = 0.05, 0.002 * 0.05
        # python: gradients (from the flat buffers), then _axpy/_add on lists
        _, ge, ghw, ghb, gow, gob, _ = _python_reference(
            seq_lists, embedding, hw, hb, ow, ob)
        emb_py = [list(embedding[i * HIDDEN:(i + 1) * HIDDEN]) for i in range(VOCAB)]
        hw_py = [list(hw[i * HIDDEN:(i + 1) * HIDDEN]) for i in range(HIDDEN)]
        hb_py = list(hb)
        ow_py = [list(ow[i * VOCAB:(i + 1) * VOCAB]) for i in range(HIDDEN)]
        ob_py = list(ob)
        T._axpy(emb_py, ge, -lr, l2)
        T._axpy(hw_py, ghw, -lr, l2)
        T._axpy(ow_py, gow, -lr, l2)
        T._add(hb_py, ghb, -lr)
        T._add(ob_py, gob, -lr)

        emb_c = array.array("d", embedding)
        hw_c = array.array("d", hw)
        hb_c = array.array("d", hb)
        ow_c = array.array("d", ow)
        ob_c = array.array("d", ob)
        seqs = array.array("i")
        for s in seq_lists:
            seqs.extend(s)
        lens = array.array("i", (len(s) for s in seq_lists))
        native.mlp_batch(seqs, lens, emb_c, hw_c, hb_c, ow_c, ob_c,
                         context=CONTEXT, lr=lr, l2=l2)

        for name, c_buf, p_list in (
            ("embed", emb_c, [v for row in emb_py for v in row]),
            ("hidden_w", hw_c, [v for row in hw_py for v in row]),
            ("hidden_b", hb_c, hb_py),
            ("out_w", ow_c, [v for row in ow_py for v in row]),
            ("out_b", ob_c, ob_py),
        ):
            for i, (c_v, p_v) in enumerate(zip(c_buf, p_list)):
                self.assertAlmostEqual(c_v, p_v, delta=1e-12, msg=f"{name}[{i}]")

    def test_short_and_tied_sequences(self):
        # sequences at exactly context length produce zero count; repeated
        # tokens exercise the mod-vocab embedding aliasing
        embedding = array.array("d", [0.1] * (VOCAB * HIDDEN))
        hw = array.array("d", [0.1] * (HIDDEN * HIDDEN))
        hb = array.array("d", [0.0] * HIDDEN)
        ow = array.array("d", [0.05] * (HIDDEN * VOCAB))
        ob = array.array("d", [0.0] * VOCAB)
        seq_lists = [[3] * CONTEXT, [1, 1, 1, 1, 1, 1]]
        py_loss, _ge, _ghw, _ghb, _gow, _gob, py_count = _python_reference(
            seq_lists, embedding, hw, hb, ow, ob)
        seqs = array.array("i")
        for s in seq_lists:
            seqs.extend(s)
        lens = array.array("i", (len(s) for s in seq_lists))
        c_loss, c_count, _ = native.mlp_batch(
            seqs, lens, embedding, hw, hb, ow, ob, context=CONTEXT)
        self.assertEqual(c_count, py_count)
        self.assertEqual(c_count, 2)  # only the length-6 seq contributes: 6-4 pairs
        self.assertAlmostEqual(c_loss, py_loss, delta=1e-12)

    def test_empty_batch_returns_zero(self):
        embedding = array.array("d", [0.0] * (VOCAB * HIDDEN))
        hw = array.array("d", [0.0] * (HIDDEN * HIDDEN))
        hb = array.array("d", [0.0] * HIDDEN)
        ow = array.array("d", [0.0] * (HIDDEN * VOCAB))
        ob = array.array("d", [0.0] * VOCAB)
        c_loss, c_count, _ = native.mlp_batch(
            array.array("i"), array.array("i"), embedding, hw, hb, ow, ob,
            context=CONTEXT)
        self.assertEqual((c_loss, c_count), (0.0, 0))


class FullTrainingParityTest(unittest.TestCase):
    """A whole training run: C++ kernel vs pure Python — same model."""

    def _corpus(self):
        rng = random.Random(4242)
        words = ["the", "cat", "sat", "on", "the", "mat", "a", "dog", "runs", "fast"]
        examples = []
        for i in range(60):
            sentence = " ".join(rng.choices(words, k=rng.randint(6, 14)))
            examples.append(Example(turns=[
                Turn(role="user", content=f"line {i}: {sentence}"),
                Turn(role="assistant", content=f"reply {i}: {sentence}"),
            ]))
        return examples[:45], examples[45:]

    def _train(self, force_python: bool):
        train, evals = self._corpus()
        texts = [e.to_chatml() for e in train + evals]
        tokenizer = BPETokenizer.train(texts, vocab_size=300, min_frequency=1)
        config = T.TrainConfig(
            context_window=6, hidden_size=10, epochs=3, batch_size=8,
            learning_rate=0.05, l2=0.002, seed=1234, max_examples=200,
        )
        trainer = T.NativeTrainer(tokenizer, config)
        with mock.patch.object(T.NativeTrainer, "_cpp_kernel",
                               return_value=not force_python):
            model, metrics = trainer.fit(train, evals)
        return model, metrics

    def test_same_seed_same_model(self):
        model_cpp, metrics_cpp = self._train(force_python=False)
        model_py, metrics_py = self._train(force_python=True)
        self.assertAlmostEqual(metrics_cpp.final_loss, metrics_py.final_loss,
                               delta=1e-9)
        self.assertAlmostEqual(metrics_cpp.best_eval_loss, metrics_py.best_eval_loss,
                               delta=1e-9)
        self.assertEqual(metrics_cpp.steps, metrics_py.steps)
        # every weight identical to float64 noise
        for name in ("input_weights", "hidden_weights", "output_weights"):
            a, b = getattr(model_cpp, name), getattr(model_py, name)
            for i, (ra, rb) in enumerate(zip(a, b)):
                for j, (va, vb) in enumerate(zip(ra, rb)):
                    self.assertAlmostEqual(va, vb, delta=1e-9, msg=f"{name}[{i}][{j}]")
        for i, (va, vb) in enumerate(zip(model_cpp.hidden_bias, model_py.hidden_bias)):
            self.assertAlmostEqual(va, vb, delta=1e-9)

    def test_cpp_model_saves_and_loads(self):
        model, _ = self._train(force_python=False)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = T.TrainedModel.load(tmp)
            self.assertEqual(loaded.vocab_size, model.vocab_size)
            self.assertEqual(loaded.input_weights, model.input_weights)

    def test_fallback_when_library_missing(self):
        train, evals = self._corpus()
        texts = [e.to_chatml() for e in train]
        tokenizer = BPETokenizer.train(texts, vocab_size=300, min_frequency=1)
        config = T.TrainConfig(context_window=6, hidden_size=8, epochs=1,
                               batch_size=8, learning_rate=0.05, seed=99)
        trainer = T.NativeTrainer(tokenizer, config)
        with mock.patch.object(native, "mlp_available", return_value=False):
            model, metrics = trainer.fit(train, evals)
        self.assertEqual(metrics.steps, 0 if not train else metrics.steps)
        self.assertGreater(metrics.steps, 0)  # still trained, pure python
        self.assertEqual(model.config.hidden_size, 8)


class NativeCliSurfaceTest(unittest.TestCase):
    def test_nm_native_reports_both_kernels(self):
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "native"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("native vector search: native-cpp", proc.stdout)
        self.assertIn("native mlp training: native-cpp", proc.stdout)


if __name__ == "__main__":
    unittest.main()
