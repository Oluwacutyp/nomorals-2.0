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
from .trainer import (
    NativeTrainer, TrainConfig, TrainMetrics, TrainedModel, clip_grads,
    global_grad_norm, lr_factor,
)

__all__ = [
    "BPETokenizer",
    "CleanStats",
    "Dataset",
    "DatasetRegistry",
    "Example",
    "HFTokenizer",
    "NativeTrainer",
    "PROMOTION_STAGES",
    "RunStatus",
    "Theme",
    "TrainConfig",
    "TrainMetrics",
    "TrainedModel",
    "TrainingRegistry",
    "TrainingRun",
    "Turn",
    "build_tokenizer",
    "card",
    "check_leakage",
    "clean_text",
    "clip_grads",
    "corpus_stats",
    "decode_example",
    "dedupe",
    "encode_batch",
    "example_stats",
    "file_checksum",
    "format_seconds",
    "global_grad_norm",
    "hamming",
    "lr_factor",
    "prepare",
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
    "table",
    "to_alpaca",
    "to_chatml",
    "to_sharegpt",
    "tokenizer_corpus_stats",
    "write_format_bundles",
    "write_jsonl",
]
