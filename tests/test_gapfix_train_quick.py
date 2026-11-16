"""Gapfix R2-31: `nm train --quick` — the quick-training demo, wired.

The module (`nomorals/training/quick.py`) is a complete, real
implementation — seed corpus + collected pairs, BPE tokenizer, native
MLP, honest before/after loss, durable artifact — but nothing called it
and `nm train` had no `--quick` flag, so the docstring promise in
`training/trainer.py:214` was false. These tests pin the wiring:

* the flag parses, and defaults off;
* the names are exported from `nomorals.training`;
* the handler runs the real demo end-to-end (one run per module, shared
  by the assertions — the demo takes ~4 minutes on this box) and reports
  dataset size, loss before → after, backend, and the artifact dir;
* failure is honest (too few examples → ok=False with a note, rc 1);
* plain `nm train` behaves exactly as before.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from nomorals.cmdline import _parser
from nomorals.cmdline.commands.train import _cmd_train


def _args(**overrides: object) -> argparse.Namespace:
    ns = argparse.Namespace(
        backends=False, run=False, backend="", base_model="",
        json=False, quick=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _context(home: str):
    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings

    ctx = build_context(
        Settings(home=home),
        with_executor=False, with_router=False,
        with_memory=False, with_tools=False,
    )
    ctx.__enter__()
    return ctx


class QuickParserTests(unittest.TestCase):
    def test_train_quick_parses(self) -> None:
        args = _parser().parse_args(["train", "--quick"])
        self.assertTrue(args.quick)

    def test_train_without_quick_defaults_off(self) -> None:
        args = _parser().parse_args(["train"])
        self.assertFalse(args.quick)
        self.assertFalse(args.run)

    def test_train_quick_shows_in_help(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as caught, redirect_stdout(buf):
            _parser().parse_args(["train", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--quick", buf.getvalue())


class QuickExportTests(unittest.TestCase):
    def test_quick_names_exported(self) -> None:
        from nomorals.training import (
            QuickResult,
            collect_quick_dataset,
            run_quick_training,
        )

        self.assertTrue(callable(run_quick_training))
        self.assertTrue(callable(collect_quick_dataset))
        self.assertTrue(callable(QuickResult))

    def test_trainer_docstring_claim_is_now_true(self) -> None:
        # trainer.py:214 promises `nm train --quick` shows what the demo
        # model learned. The promise is only true if the flag exists.
        from nomorals.training.trainer import TrainedModel

        self.assertIn("nm train --quick", TrainedModel.generate.__doc__)
        self.assertTrue(_parser().parse_args(["train", "--quick"]).quick)


class QuickHandlerTests(unittest.TestCase):
    """One real demo run per module, shared by every assertion."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.home = tempfile.mkdtemp(prefix="nm-quick-handler-")
        cls.context = _context(cls.home)
        t0 = time.perf_counter()
        buf = io.StringIO()
        with redirect_stdout(buf):
            cls.rc = _cmd_train(_args(quick=True, json=True), cls.context)
        cls.wall = time.perf_counter() - t0
        cls.payload = json.loads(buf.getvalue())
        cls.addClassCleanup(cls.context.__exit__, None, None, None)

    def test_quick_returns_zero(self) -> None:
        self.assertEqual(self.rc, 0)
        self.assertTrue(self.payload["ok"])

    def test_quick_reports_dataset_and_learning(self) -> None:
        # Seed-only, zero prior use: the bundled corpus alone clears the
        # 12-example floor (probe: 85 examples).
        self.assertGreaterEqual(self.payload["dataset"]["examples"], 12)
        self.assertGreater(self.payload["dataset"]["train"], 0)
        self.assertGreater(self.payload["dataset"]["eval"], 0)
        # It actually learned: held-out loss went down.
        self.assertLess(
            self.payload["loss_after"], self.payload["loss_before"])

    def test_quick_artifact_is_real_and_loadable(self) -> None:
        artifact = Path(self.payload["artifact_dir"])
        self.assertTrue((artifact / "model.json").is_file())
        self.assertTrue((artifact / "tokenizer.json").is_file())
        metrics = json.loads((artifact / "metrics.json").read_text())
        # metrics.json keeps full precision; to_dict() rounds to 5 places.
        self.assertAlmostEqual(metrics["loss_before"],
                               self.payload["loss_before"], places=5)
        self.assertAlmostEqual(metrics["loss_after"],
                               self.payload["loss_after"], places=5)
        self.assertEqual(metrics["dataset"]["examples"],
                         self.payload["dataset"]["examples"])
        # The artifact loads as a real model.
        from nomorals.training.trainer import TrainedModel

        model = TrainedModel.load(artifact)
        self.assertIsNotNone(model)

    def test_quick_backend_is_honest(self) -> None:
        self.assertIn(self.payload["backend"],
                      ("pure-python", "native-cpp"))

    def test_quick_completes_in_bounded_time(self) -> None:
        # Module docstring: "a couple of minutes on a phone". Measured
        # 231s on this box; the bound is sane, not exact.
        self.assertLess(self.payload["seconds"], 600)
        self.assertLess(self.wall, 600)

    def test_quick_failure_is_honest(self) -> None:
        with mock.patch(
                "nomorals.training.quick.collect_quick_dataset",
                return_value=[]):
            from nomorals.training.quick import run_quick_training

            result = run_quick_training(self.context)
        self.assertFalse(result.ok)
        self.assertIn("12", result.note)

    def test_quick_failure_returns_nonzero(self) -> None:
        with mock.patch(
                "nomorals.training.quick.collect_quick_dataset",
                return_value=[]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_train(_args(quick=True), self.context)
        self.assertEqual(rc, 1)
        self.assertIn("12", buf.getvalue())

    def test_train_without_flags_is_unchanged(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_train(_args(), self.context)
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", buf.getvalue())

    def test_quick_wins_over_run(self) -> None:
        # --quick and --run are different jobs (a 3-epoch demo vs the full
        # pipeline). Together they run the quick demo only — never both.
        from nomorals.training.quick import QuickResult

        stub = QuickResult(ok=True, dataset={"examples": 85},
                           loss_before=5.0, loss_after=4.0, seconds=1.0,
                           artifact_dir=str(Path(self.home) / "stub"))
        with mock.patch(
                "nomorals.training.quick.run_quick_training",
                return_value=stub) as quick_fn, \
                mock.patch(
                    "nomorals.self_improvement.SelfImprovementJob") as job:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_train(_args(quick=True, run=True), self.context)
        self.assertEqual(rc, 0)
        quick_fn.assert_called_once()
        job.assert_not_called()


class QuickRenderTests(unittest.TestCase):
    def test_render_shows_learning_and_sample(self) -> None:
        from nomorals.training.quick import QuickResult

        result = QuickResult(
            ok=True, dataset={"examples": 85},
            loss_before=5.7038, loss_after=3.8010, seconds=231.0,
            artifact_dir="/tmp/quick-demo", backend="pure-python",
            sample="the quick brown fox jumps",
        )
        text = result.render()
        # card() uppercases the title.
        self.assertIn("QUICK TRAINING", text)
        self.assertIn("5.7038", text)
        self.assertIn("3.8010", text)
        self.assertIn("the quick brown fox jumps", text)
        self.assertIn("/tmp/quick-demo", text)

    def test_render_failure_is_plain(self) -> None:
        from nomorals.training.quick import QuickResult

        text = QuickResult(ok=False, note="need at least 12").render()
        self.assertIn("FAILED", text)
        self.assertIn("need at least 12", text)


if __name__ == "__main__":
    unittest.main()
