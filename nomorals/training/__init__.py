"""L3 — training: the self-improvement loop.

Collect a corpus, clean it, train a model, evaluate it, and promote it *only if
it beats what is already live*. That last clause is the whole point: without the
gate, an automated finetune loop monotonically degrades the system, because every
run that "trains successfully" gets shipped regardless of whether it improved
anything.

    from nomorals.training import DatasetRegistry, TrainingRegistry, NativeTrainer

    datasets = DatasetRegistry(db)
    runs = TrainingRegistry(db)
    run = runs.start("personal-v1", base_model="dolphin-8b")
    try:
        model, metrics = NativeTrainer(tokenizer, config).fit(train, evaluation)
        runs.complete(run, metrics=metrics.as_scores(), output_path=str(path))
        if runs.evaluate(run):
            runs.promote(run)
    except Exception as exc:
        runs.fail(run, error=str(exc))
"""

from __future__ import annotations

from .dataset import (
    Dataset, DatasetRegistry, Example, Turn, corpus_stats, decode_example,
    example_stats, file_checksum, read_jsonl, to_alpaca, to_chatml,
    to_sharegpt, write_format_bundles, write_jsonl,
)
from .preprocess import (
    CleanStats, check_leakage, clean_text, dedupe, hamming, prepare,
    quality_filter, refined_quality_signals, simhash, split,
)
from .registry import PROMOTION_STAGES, RunStatus, TrainingRegistry, TrainingRun
from .style import (
    Theme, card, format_seconds, progress_bar, render_eval_report,
    render_gate_report, render_policy_card, render_run_card, sparkline, table,
)
from .tokenize import (
    BPETokenizer, HFTokenizer, build_tokenizer, encode_batch,
    tokenizer_corpus_stats,
)
from .continual import (
    ContinualUpdater, FixedEvalHarness, ReplayBuffer, confined_update,
    due_for_update,
)
from .distillation import (
    DistillationPair, DistillationStore, capture, capture_owner_kept,
    capture_owner_passed, judge_and_capture, periodic_distill,
    train_local_adapter,
)
from .preference import (
    DISAPPROVAL_WEIGHT, PreferenceAdapter, PreferencePair,
    build_preference_pairs, evaluate_on_held_out, gate_with_registry,
    production_gate, split_held_out, train_preference_adapter,
)
from .trainer import (
    NativeTrainer, TrainConfig, TrainMetrics, TrainedModel, clip_grads,
    global_grad_norm, lr_factor,
)

__all__ = [
    "BPETokenizer",
    "CleanStats",
    "ContinualUpdater",
    "Dataset",
    "DatasetRegistry",
    "DistillationPair",
    "DistillationStore",
    "Example",
    "FixedEvalHarness",
    "HFTokenizer",
    "NativeTrainer",
    "PreferenceAdapter",
    "PreferencePair",
    "ReplayBuffer",
    "PROMOTION_STAGES",
    "RunStatus",
    "Theme",
    "TrainConfig",
    "TrainMetrics",
    "TrainedModel",
    "TrainingRegistry",
    "TrainingRun",
    "Turn",
    "build_preference_pairs",
    "build_tokenizer",
    "capture",
    "capture_owner_kept",
    "capture_owner_passed",
    "card",
    "check_leakage",
    "clean_text",
    "clip_grads",
    "confined_update",
    "corpus_stats",
    "decode_example",
    "dedupe",
    "due_for_update",
    "encode_batch",
    "evaluate_on_held_out",
    "example_stats",
    "file_checksum",
    "format_seconds",
    "gate_with_registry",
    "global_grad_norm",
    "hamming",
    "judge_and_capture",
    "lr_factor",
    "periodic_distill",
    "prepare",
    "production_gate",
    "progress_bar",
    "quality_filter",
    "read_jsonl",
    "refined_quality_signals",
    "render_eval_report",
    "render_gate_report",
    "render_policy_card",
    "render_run_card",
    "simhash",
    "sparkline",
    "split",
    "split_held_out",
    "table",
    "to_alpaca",
    "to_chatml",
    "to_sharegpt",
    "tokenizer_corpus_stats",
    "train_local_adapter",
    "train_preference_adapter",
    "write_format_bundles",
    "write_jsonl",
]
