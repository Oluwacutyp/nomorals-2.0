"""Sweep tests for the training module upgrade (2026-10-10).

Covers the new behavior only: AdamW/schedule/clipping/early-stopping in the
native trainer, model generation, exact-first dedupe, strict quality
signals, leakage checks, Presidio-optional + Nigerian-ID scrubbing, the
miner diversity filter + judge hook, PSI drift policy, the hardened
promotion gate (min_gain, staged promotion, demote, approval, compare,
model cards), pairwise judging with position-swap, checkpoint pruning,
catalog search/recommend, tokenizer batch/stats, format bundles, the QLoRA
recipe, and the style layer.  Everything is hermetic (fake DBs, no GPU).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.training import style as T
from nomorals.training import (
    Theme, card, sparkline, progress_bar, render_run_card, render_gate_report,
)
from nomorals.training.backends.unsloth import dry_run
from nomorals.training.checkpoints import (
    checkpoint_manifest, prune_checkpoints, validate_checkpoint,
)
from nomorals.training.collect import (
    ConversationMiner, scrub_pii, scrub_report, trigram_overlap,
)
from nomorals.training.dataset import (
    Example, Turn, corpus_stats, write_format_bundles,
)
from nomorals.training.evaluate import (
    GoldenCase, build_gate_scores, default_multiturn_cases, grade_golden_set,
    grade_multiturn_set, pairwise_judge, render_eval_report,
)
from nomorals.training.finetune import (
    EVOLVE_OPERATIONS, QLORA_RECIPE, generate_persona_samples, qlora_recipe,
    write_colab_script,
)
from nomorals.training.free_datasets import recommend, search
from nomorals.training.policy import RetrainingPolicy, drift_band, psi
from nomorals.training.preprocess import (
    check_leakage, dedupe, quality_filter, refined_quality_signals,
)
from nomorals.training import registry as registry_mod
from nomorals.training.style import render_eval_report as themed_eval_report
from nomorals.training.tokenize import (
    BPETokenizer, encode_batch, tokenizer_corpus_stats,
)
from nomorals.training.trainer import (
    NativeTrainer, TrainConfig, TrainMetrics, clip_grads, global_grad_norm,
    lr_factor,
)


def _examples(n=24):
    texts = [
        "the cat sat on the mat and the dog ran in the park",
        "a quick brown fox jumps over the lazy dog near the river",
        "machine learning models learn patterns from data every day",
        "the sun rises in the east and sets in the west each morning",
    ]
    return [
        Example(turns=[Turn("user", f"say something number {i}"),
                       Turn("assistant", f"{texts[i % len(texts)]} (variant {i})")])
        for i in range(n)
    ]


def _tokenizer(examples):
    return BPETokenizer.train([e.to_chatml() for e in examples], vocab_size=300)


# ── style ─────────────────────────────────────────────────────────────────


class StyleTest(unittest.TestCase):
    def test_card_renders_title_and_rows(self):
        theme = Theme(color=False)
        out = card(theme, "gate", [("score", "0.81")])
        self.assertIn("GATE", out)
        self.assertIn("score", out)
        self.assertIn("0.81", out)
        self.assertIn("╭", out)

    def test_sparkline_tracks_direction(self):
        down = sparkline([3.0, 2.0, 1.0])
        self.assertNotEqual(down[0], down[-1])
        self.assertEqual(sparkline([]), "·" * 8)

    def test_progress_bar_midpoint(self):
        theme = Theme(color=False)
        bar = progress_bar(theme, 0.5, width=10)
        self.assertIn("50.0%", bar)
        self.assertIn("█" * 5, bar)

    def test_run_card_and_gate_report(self):
        theme = Theme(color=False)
        run_card = render_run_card(theme, {"name": "r", "status": "promoted",
                                           "backend": "native",
                                           "metrics": {"score": 0.9}})
        self.assertIn("PROMOTED", run_card)
        gate = render_gate_report(theme, {"challenger": "m2", "champion": "m1",
                                          "challenger_score": 0.85,
                                          "champion_score": 0.8,
                                          "margin": 0.05, "min_gain": 0.01,
                                          "passed": True, "reasons": []})
        self.assertIn("PASS", gate)

    def test_themed_eval_report(self):
        theme = Theme(color=False)
        out = themed_eval_report(theme, {"eval_loss": 2.0, "golden_score": 0.9,
                                         "by_category": {"reasoning": 1.0}})
        self.assertIn("EVALUATION", out)
        self.assertIn("reasoning", out)


# ── trainer: optimizer, schedule, clipping, early stopping, generate ──────


class TrainerUpgradeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = _examples()
        cls.tokenizer = _tokenizer(cls.examples)

    def test_adamw_decreases_loss(self):
        _, metrics = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=16, context_window=8, epochs=3,
                        batch_size=4, learning_rate=0.05, optimizer="adamw"),
        ).fit(self.examples[:16], self.examples[16:])
        self.assertLess(metrics.train_loss_history[-1],
                        metrics.train_loss_history[0])
        self.assertEqual(metrics.optimizer, "adamw")
        self.assertGreater(metrics.tokens_per_sec, 0)

    def test_sgd_still_works(self):
        _, metrics = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=8, context_window=8, epochs=2,
                        batch_size=4, optimizer="sgd", lr_schedule="constant"),
        ).fit(self.examples[:8], self.examples[8:12])
        self.assertTrue(metrics.train_loss_history)

    def test_bad_optimizer_rejected(self):
        with self.assertRaises(Exception):
            TrainConfig(optimizer="rmsprop").validate()

    def test_lr_factor_shapes(self):
        self.assertEqual(lr_factor(5, 100, 0, "constant"), 1.0)
        # warmup ramps up
        self.assertLess(lr_factor(0, 100, 10, "warmup_cosine"),
                        lr_factor(9, 100, 10, "warmup_cosine"))
        # cosine decays to ~0 at the end
        self.assertAlmostEqual(lr_factor(99, 100, 10, "warmup_cosine"),
                               0.0, places=2)
        # cosine without warmup starts at 1
        self.assertAlmostEqual(lr_factor(0, 100, 0, "cosine"), 1.0, places=6)

    def test_clip_grads_caps_norm(self):
        grads = {"w": [[3.0, 4.0], [0.0, 0.0]], "b": [0.0, 0.0]}
        before = global_grad_norm(grads)
        self.assertAlmostEqual(before, 5.0)
        clip_grads(grads, 2.5)
        self.assertAlmostEqual(global_grad_norm(grads), 2.5, places=6)

    def test_early_stopping_triggers_on_plateau(self):
        # lr ~ 0: eval cannot improve by 1e-4 → patience=1 stops after epoch 2
        _, metrics = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=8, context_window=8, epochs=10,
                        batch_size=4, learning_rate=1e-9,
                        early_stopping_patience=1),
        ).fit(self.examples[:8], self.examples[8:12])
        self.assertTrue(metrics.stopped_early)
        self.assertLess(metrics.epochs, 10)

    def test_generate_returns_prompt_continuation(self):
        model, _ = NativeTrainer(
            self.tokenizer,
            TrainConfig(hidden_size=8, context_window=8, epochs=1,
                        batch_size=4, optimizer="sgd", lr_schedule="constant"),
        ).fit(self.examples[:8])
        out = model.generate(self.tokenizer, "the cat", max_new_tokens=8,
                             temperature=0.0, seed=1)
        self.assertTrue(out.startswith("the cat"))
        self.assertGreater(len(out), len("the cat"))
        # deterministic at temperature 0
        again = model.generate(self.tokenizer, "the cat", max_new_tokens=8,
                               temperature=0.0, seed=1)
        self.assertEqual(out, again)

    def test_metrics_carry_new_fields(self):
        metrics = TrainMetrics()
        d = metrics.to_dict()
        for key in ("best_epoch", "stopped_early", "tokens_per_sec",
                    "optimizer", "lr_final"):
            self.assertIn(key, d)


# ── preprocess ────────────────────────────────────────────────────────────


class PreprocessUpgradeTest(unittest.TestCase):
    def test_dedupe_drops_exact_first(self):
        examples = _examples(8)
        kept = dedupe(examples + examples)
        self.assertEqual(len(kept), 8)

    def test_check_leakage_finds_eval_dupes(self):
        examples = _examples(12)
        report = check_leakage(examples[:8], examples[4:12])
        self.assertGreater(report["leaked"], 0)
        self.assertLessEqual(report["leak_rate"], 1.0)

    def test_strict_filter_drops_symbol_dense(self):
        good = Example(turns=[Turn("user", "explain gravity"),
                              Turn("assistant",
                                   "Gravity pulls objects toward each other.")])
        bad = Example(turns=[Turn("user", "explain gravity"),
                             Turn("assistant", "$$$ ### @@@ %%% &&& *** +++")])
        kept = quality_filter([good, bad], strict=True)
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], good)

    def test_refined_signals(self):
        signals = refined_quality_signals("Hello world. This is fine!")
        self.assertGreater(signals["alpha_ratio"], 0.5)
        self.assertEqual(signals["ends_with_terminal"], 1.0)
        noisy = refined_quality_signals("$$$ ### @@@")
        self.assertGreater(noisy["symbol_ratio"], 1.0)


# ── collect: scrubbing + miner ────────────────────────────────────────────


class ScrubTest(unittest.TestCase):
    def test_redacts_email_phone_card(self):
        out = scrub_pii("mail ada@example.com, call +2348012345678, "
                        "card 4111 1111 1111 1111")
        self.assertIn("[REDACTED_EMAIL]", out)
        self.assertIn("[REDACTED_PHONE]", out)
        self.assertIn("[REDACTED_CARD]", out)
        self.assertNotIn("ada@example.com", out)

    def test_nigerian_ids_redacted(self):
        out = scrub_pii("my NIN is 12345678901 and BVN 09876543211")
        self.assertIn("[REDACTED_NIN]", out)
        self.assertNotIn("12345678901", out)

    def test_plus_phone_not_a_card(self):
        out = scrub_pii("call +2348012345678")
        self.assertIn("[REDACTED_PHONE]", out)
        self.assertNotIn("[REDACTED_CARD]", out)

    def test_hash_mode_is_deterministic(self):
        first = scrub_pii("ada@example.com", mode="hash")
        self.assertEqual(first, scrub_pii("ada@example.com", mode="hash"))
        self.assertNotEqual(first, scrub_pii("bob@example.com", mode="hash"))
        self.assertTrue(first.startswith("[HASH_EMAIL_"))

    def test_scrub_report_counts_without_values(self):
        report = scrub_report("a@b.com and c@d.com, call 08031234567")
        self.assertEqual(report["email"], 2)
        self.assertNotIn("a@b.com", json.dumps(report))

    def test_trigram_overlap(self):
        self.assertAlmostEqual(
            trigram_overlap("the cat sat on the mat", "the cat sat on the mat"),
            1.0)
        self.assertLess(
            trigram_overlap("the cat sat on the mat",
                            "quantum physics is deeply strange"), 0.2)


class FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, sql, params=()):
        return self._rows


class MinerUpgradeTest(unittest.TestCase):
    def _miner(self, pairs):
        rows = []
        for i, (user, answer) in enumerate(pairs):
            rows.append({"conversation_id": f"c{i}", "role": "user",
                         "content": user})
            rows.append({"conversation_id": f"c{i}", "role": "assistant",
                         "content": answer})
        return ConversationMiner(FakeDB(rows))

    def test_diversity_filter_drops_near_identical(self):
        miner = self._miner([
            ("how do I bake a chocolate cake with frosting layers",
             "Here is a detailed chocolate cake recipe with many steps " * 4),
            ("how do I bake a chocolate cake with frosting layers please",
             "Here is a detailed chocolate cake recipe with many steps " * 4),
            ("explain quantum entanglement simply for a beginner today",
             "Quantum entanglement links particles across distance " * 6),
        ])
        result = miner.mine(min_score=0.0, diversity_overlap=0.7)
        users = [e.turns[0].content for e in result["examples"]]
        self.assertEqual(len(users), 2)
        self.assertGreater(result["stats"]["diversity_dropped"], 0)

    def test_judge_hook_blends_into_score(self):
        miner = self._miner([
            ("what is 2+2, please explain your reasoning fully here",
             "Four. Two plus two equals four, a basic arithmetic fact " * 3),
        ])
        plain = miner.mine(min_score=0.0)["scores"][0]
        judged = miner.mine(min_score=0.0, judge=lambda u, a: 1.0,
                            judge_weight=0.5)["scores"][0]
        self.assertGreaterEqual(judged, plain)


# ── policy: PSI drift ─────────────────────────────────────────────────────


class PolicyDriftTest(unittest.TestCase):
    def test_psi_identical_is_zero(self):
        self.assertAlmostEqual(
            psi([1, 2, 3, 4, 5] * 20, [1, 2, 3, 4, 5] * 20), 0.0, places=6)

    def test_psi_shifted_is_large(self):
        self.assertGreater(psi([1, 2, 3] * 40, [8, 9, 10] * 40), 0.2)

    def test_drift_bands(self):
        self.assertEqual(drift_band(0.05), "stable")
        self.assertEqual(drift_band(0.15), "warning")
        self.assertEqual(drift_band(0.3), "critical")

    def test_drift_triggers_with_reason_and_response(self):
        decision = RetrainingPolicy().decide(
            new_examples=5,
            drift_reference=[0.8] * 50,
            drift_current=[0.3] * 50,
        )
        self.assertTrue(decision.should_retrain)
        self.assertTrue(any("drift" in r for r in decision.reasons))
        self.assertIn(decision.response, {"retrain", "expand", "investigate"})
        self.assertEqual(decision.to_dict()["drift_band"], "critical")

    def test_no_drift_no_trigger(self):
        decision = RetrainingPolicy().decide(
            new_examples=5,
            drift_reference=[0.8] * 50,
            drift_current=[0.8] * 50,
        )
        self.assertFalse(any("drift" in r for r in decision.reasons))


# ── registry: hardened gate ───────────────────────────────────────────────


class FakeModels:
    def __init__(self):
        self.scores = {}
        self._active = None

    def register(self, name, **kw):
        self.scores.setdefault(name, {})

    def record_eval(self, name, scores):
        self.scores[name].update(scores)

    def by_name(self, name):
        record = MagicMock()
        record.eval_scores = self.scores.get(name, {})
        return record if name in self.scores else None

    def active(self):
        if not self._active:
            return None
        record = MagicMock()
        record.name = self._active
        return record

    def activate(self, name):
        self._active = name

    def beats_incumbent(self, candidate, metric="score", tolerance=0.0):
        current = self.active()
        if current is None:
            return True
        candidate_score = self.scores.get(candidate, {}).get(metric)
        current_score = self.scores.get(current.name, {}).get(metric)
        if candidate_score is None:
            return False
        if current_score is None:
            return True
        return candidate_score >= current_score - tolerance


class FakeRepo:
    def __init__(self, db, *args, **kwargs):
        self.db = db

    def create(self, row):
        self.db.rows[row["id"]] = dict(row)

    def update(self, rid, row):
        self.db.rows[rid] = dict(row)

    def get(self, rid):
        return self.db.rows.get(rid)

    def find_one(self, **kwargs):
        for row in self.db.rows.values():
            if all(row.get(k) == v for k, v in kwargs.items()):
                return row
        return None


class RegistryGateTest(unittest.TestCase):
    def setUp(self):
        real_repo = registry_mod.Repository
        registry_mod.Repository = FakeRepo
        self.addCleanup(setattr, registry_mod, "Repository", real_repo)
        self.db = SimpleNamespace(rows={})
        self.models = FakeModels()
        self.runs = registry_mod.TrainingRegistry(self.db, models=self.models)

    def _run(self, name, model, score):
        run = self.runs.start(name, base_model="b", output_model=model)
        self.runs.complete(run, metrics={"score": score}, output_path=f"/tmp/{model}")
        return run

    def test_min_gain_blocks_noise_win(self):
        champ = self._run("champ", "m1", 0.80)
        self.assertTrue(self.runs.evaluate(champ))
        self.assertTrue(self.runs.promote(champ))
        chall = self._run("chall", "m2", 0.80005)
        self.assertFalse(self.runs.evaluate(chall, min_gain=0.01))
        self.assertAlmostEqual(chall.metadata["gate"]["margin"], 0.00005,
                               places=6)

    def test_min_gain_passes_real_win(self):
        champ = self._run("champ", "m1", 0.80)
        self.runs.evaluate(champ)
        self.runs.promote(champ)
        chall = self._run("chall", "m2", 0.85)
        self.assertTrue(self.runs.evaluate(chall, min_gain=0.01))
        self.assertTrue(self.runs.promote(chall))
        self.assertEqual(chall.metadata["promoted_over"], "m1")

    def test_staged_promotion_shadow(self):
        run = self._run("r", "m1", 0.9)
        self.runs.evaluate(run)
        self.assertTrue(self.runs.promote(run, stage="shadow"))
        self.assertEqual(run.status, registry_mod.RunStatus.SHADOW)
        # champion keeps serving until full promotion
        self.assertIsNone(self.models.active())

    def test_demote_restores_champion(self):
        champ = self._run("champ", "m1", 0.80)
        self.runs.evaluate(champ)
        self.runs.promote(champ)
        chall = self._run("chall", "m2", 0.85)
        self.runs.evaluate(chall)
        self.runs.promote(chall)
        self.assertEqual(self.models.active().name, "m2")
        self.assertTrue(self.runs.demote(chall, reason="regression"))
        self.assertEqual(self.models.active().name, "m1")
        self.assertEqual(chall.status, registry_mod.RunStatus.DONE)
        self.assertTrue(chall.metadata["demoted"])

    def test_approval_flow(self):
        run = self._run("r", "m1", 0.9)
        self.runs.evaluate(run)
        self.runs.request_approval(run, approver="owner")
        self.assertEqual(run.status, registry_mod.RunStatus.AWAITING_APPROVAL)
        self.runs.approve(run, approver="owner")
        self.assertEqual(run.status, registry_mod.RunStatus.DONE)
        self.assertEqual(run.metadata["approval"]["decision"], "approved")

    def test_compare_recommends(self):
        champ = self._run("champ", "m1", 0.80)
        self.runs.evaluate(champ)
        self.runs.promote(champ)
        report = self.runs.compare("m1")
        self.assertIn("recommendation", report)
        self.assertEqual(report["champion"], "m1")
        self.assertIn("tie", report["recommendation"])

    def test_model_card_sections(self):
        run = self._run("r", "m1", 0.85)
        self.runs.evaluate(run, min_gain=0.01)
        card_text = self.runs.model_card(run)
        for section in ("# Model card", "## Metrics", "## Promotion gate",
                        "## Limitations"):
            self.assertIn(section, card_text)

    def test_gate_report_shape(self):
        run = self._run("r", "m1", 0.85)
        self.runs.evaluate(run)
        report = self.runs.gate_report(run)
        self.assertTrue(report["passed"])
        self.assertEqual(report["challenger"], "m1")


# ── evaluate: pairwise + multiturn ────────────────────────────────────────


class EvaluateUpgradeTest(unittest.TestCase):
    def test_pairwise_position_swap_cancels_bias(self):
        cases = [GoldenCase(prompt="q", description="d", category="general")]
        # a judge that ALWAYS answers "A": without swap A would win 100%;
        # with swap the verdicts disagree → tie (bias cancelled)
        result = pairwise_judge(cases, lambda p: "answer one",
                                lambda p: "answer two",
                                lambda prompt, a, b: "A",
                                swap_positions=True)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["wins_a"], 0)
        honest = pairwise_judge(cases, lambda p: "answer one",
                                lambda p: "answer two",
                                lambda prompt, a, b: "A",
                                swap_positions=False)
        self.assertEqual(honest["wins_a"], 1)

    def test_pairwise_reports_verbosity_delta(self):
        cases = [GoldenCase(prompt="q", description="d")]
        result = pairwise_judge(cases, lambda p: "x" * 200, lambda p: "y" * 10,
                                lambda prompt, a, b: "tie")
        self.assertGreater(result["mean_length_delta_a_minus_b"], 0)

    def test_multiturn_memory(self):
        result = grade_multiturn_set(
            default_multiturn_cases(),
            lambda p: "Adaeze" if "name" in p.lower() else "pounded yam")
        self.assertEqual(result["multiturn_score"], 1.0)

    def test_category_rollup(self):
        cases = [GoldenCase(prompt="q", description="d", must_contain=("y",),
                            category="reasoning")]
        result = grade_golden_set(cases, lambda p: "y")
        self.assertEqual(result["by_category"]["reasoning"], 1.0)

    def test_render_eval_report(self):
        out = render_eval_report({"eval_loss": 2.0, "golden_score": 0.9,
                                  "golden_passed": 9, "golden_total": 10})
        self.assertIn("golden", out)
        self.assertIn("90.00%", out)

    def test_gate_scores_still_blend(self):
        scores = build_gate_scores(train_loss=1.0, eval_loss=0.8,
                                   golden={"golden_score": 1.0})
        self.assertIn("score", scores)
        self.assertEqual(scores["score_basis"], "0.7*loss + 0.3*golden")


# ── checkpoints ───────────────────────────────────────────────────────────


class CheckpointUpgradeTest(unittest.TestCase):
    def _tree(self, root, steps, corpse_step=None):
        for step in steps:
            d = Path(root) / f"checkpoint-{step}"
            d.mkdir(parents=True)
            for name in ("trainer_state.json", "optimizer.bin",
                         "scheduler.pt", "adapter_model.safetensors"):
                (d / name).write_text("x" * 10)
            (d / "trainer_state.json").write_text(
                json.dumps({"global_step": step}))
        if corpse_step:
            d = Path(root) / f"checkpoint-{corpse_step}"
            d.mkdir(parents=True)
            (d / "trainer_state.json").write_text("{}")

    def test_prune_keeps_newest_and_kills_corpses(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp, [100, 200, 300], corpse_step=400)
            result = prune_checkpoints(tmp, keep=2)
            self.assertEqual(sorted(result["kept"]),
                             ["checkpoint-200", "checkpoint-300"])
            self.assertEqual(result["deleted"], ["checkpoint-100"])
            self.assertEqual(result["corpses"], ["checkpoint-400"])
            self.assertFalse((Path(tmp) / "checkpoint-400").exists())

    def test_manifest_reports_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp, [300])
            manifest = checkpoint_manifest(Path(tmp) / "checkpoint-300")
            self.assertTrue(manifest["valid"])
            self.assertEqual(manifest["step"], 300)
            self.assertTrue(all(len(f["sha256"]) == 64 for f in manifest["files"]))
            bad = checkpoint_manifest(Path(tmp) / "checkpoint-999")
            self.assertFalse(bad["valid"])


# ── misc module upgrades ──────────────────────────────────────────────────


class MiscUpgradeTest(unittest.TestCase):
    def test_catalog_search_and_recommend(self):
        hits = search("agent tool")
        self.assertTrue(hits)
        self.assertTrue(any("agent" in h["name"] for h in hits))
        recs = recommend("code")
        self.assertTrue(recs)
        self.assertTrue(all(r.get("id") for r in recs))

    def test_tokenizer_batch_and_stats(self):
        examples = _examples(8)
        tok = _tokenizer(examples)
        batch = encode_batch(tok, ["hello world", "goodbye"])
        self.assertEqual(len(batch), 2)
        stats = tokenizer_corpus_stats(tok, [e.to_chatml() for e in examples])
        self.assertGreater(stats["tokens_p95"], 0)
        self.assertGreater(stats["chars_per_token"], 1.0)

    def test_write_format_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "bundle")
            counts = write_format_bundles(base, _examples(6))
            self.assertEqual(counts, {"internal": 6, "alpaca": 6,
                                      "sharegpt": 6, "chatml": 6})
            for suffix in (".jsonl", ".alpaca.json", ".sharegpt.json",
                           ".chatml.jsonl"):
                self.assertTrue(Path(str(base) + suffix).is_file())
            alpaca = json.loads(Path(str(base) + ".alpaca.json").read_text())
            self.assertIn("instruction", alpaca[0])

    def test_corpus_stats(self):
        stats = corpus_stats(_examples(10))
        self.assertEqual(stats["rows"], 10)
        self.assertGreater(stats["chars_p95"], 0)

    def test_qlora_recipe(self):
        recipe = qlora_recipe()
        self.assertEqual(recipe["lora_r"], 32)
        self.assertEqual(recipe["lora_alpha"], 64)
        self.assertEqual(recipe["neftune_noise_alpha"], 5)
        self.assertTrue(recipe["packing"])
        self.assertTrue(recipe["assistant_only_loss"])
        custom = qlora_recipe(lora_r=16, lora_alpha=32)
        self.assertEqual(custom["lora_r"], 16)
        self.assertEqual(QLORA_RECIPE["lora_r"], 32)  # original untouched
        self.assertEqual(len(EVOLVE_OPERATIONS), 5)

    def test_colab_script_has_mined_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_colab_script(os.path.join(tmp, "mix"))
            text = Path(path).read_text()
        for needle in ("neftune_noise_alpha=5", "assistant_only_loss=True",
                       "packing=True", "warmup_ratio=0.1"):
            self.assertIn(needle, text)

    def test_generate_persona_samples_with_fake_router(self):
        payload = {"pairs": [{"question": "q1", "answer": "a1"},
                             {"question": "q2", "answer": "a2"}]}

        class FakeResponse:
            ok = True
            text = json.dumps(payload)

        class FakeRouter:
            def chat(self, *args, **kwargs):
                return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "samples.jsonl")
            result = generate_persona_samples(
                FakeRouter(), "persona", n=4, per_call=2, out_path=out)
            self.assertTrue(result["ok"])
            self.assertEqual(result["rows"], 4)
            rows = [json.loads(line) for line in
                    Path(out).read_text().splitlines()]
            self.assertEqual(rows[0]["messages"][0]["role"], "system")

    def test_unsloth_dry_run_checks(self):
        result = dry_run(base_model="some/model", train_rows=100)
        names = {c["name"] for c in result["checks"]}
        self.assertTrue({"base model set", "training rows",
                         "lora alpha ≈ 2× rank"} <= names)
        bad = dry_run(base_model="", train_rows=0)
        self.assertFalse(bad["ok"])


if __name__ == "__main__":
    unittest.main()
