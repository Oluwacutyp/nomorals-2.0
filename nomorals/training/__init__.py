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
from .continual_driver import (
    CONTINUAL_JOB,
    CONTINUAL_SPEC,
    CONTINUAL_TOOL,
    ContinualStore,
    ensure_continual_update_job,
    load_saved_adapter,
    run_continual_update_once,
)
from .distill_driver import (
    DISTILL_JOB,
    DISTILL_SPEC,
    DISTILL_TOOL,
    DistillState,
    ensure_distill_job,
    harvest_kept_turns,
    run_distill_once,
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
from .quick import QuickResult, collect_quick_dataset, run_quick_training
from .trainer import (
    NativeTrainer, TrainConfig, TrainMetrics, TrainedModel, clip_grads,
    global_grad_norm, lr_factor,
)
from .conditioning import (
    ConditionSpec, ConditioningConfig, ConditioningError,
    ConditioningUnavailable, NumpyTensorIO, PrecomputeReport,
    PrecomputedStore, Precomputer, RealEncoders, TorchTensorIO,
    decode_video_frames, get_condition_spec, load_condition_models,
    load_dataset, load_latent_models, post_latent_preparation,
    prepare_conditions, preprocess_video, probe_video, select_tensor_io,
    should_perform_precomputation,
)
from .export import (
    EXPORT_FORMAT_VERSION, ExportError, ExportReceipt, LoadedExport,
    Verification, diagnose_script_error, export_module, load_exported,
    receipt_path_for, resolve_device, verify_export,
)

__all__ = [
    "BPETokenizer",
    "CONTINUAL_JOB",
    "CONTINUAL_SPEC",
    "CONTINUAL_TOOL",
    "CleanStats",
    "ConditionSpec",
    "ConditioningConfig",
    "ConditioningError",
    "ConditioningUnavailable",
    "ContinualStore",
    "ContinualUpdater",
    "DISTILL_JOB",
    "DISTILL_SPEC",
    "DISTILL_TOOL",
    "Dataset",
    "DatasetRegistry",
    "DistillState",
    "DistillationPair",
    "DistillationStore",
    "Example",
    "EXPORT_FORMAT_VERSION",
    "ExportError",
    "ExportReceipt",
    "FixedEvalHarness",
    "HFTokenizer",
    "LoadedExport",
    "NativeTrainer",
    "NumpyTensorIO",
    "PrecomputeReport",
    "PrecomputedStore",
    "Precomputer",
    "PreferenceAdapter",
    "PreferencePair",
    "QuickResult",
    "RealEncoders",
    "ReplayBuffer",
    "PROMOTION_STAGES",
    "RunStatus",
    "Theme",
    "TorchTensorIO",
    "TrainConfig",
    "TrainMetrics",
    "TrainedModel",
    "TrainingRegistry",
    "TrainingRun",
    "Turn",
    "Verification",
    "build_preference_pairs",
    "build_tokenizer",
    "capture",
    "capture_owner_kept",
    "capture_owner_passed",
    "card",
    "check_leakage",
    "clean_text",
    "clip_grads",
    "collect_quick_dataset",
    "confined_update",
    "corpus_stats",
    "decode_example",
    "decode_video_frames",
    "dedupe",
    "diagnose_script_error",
    "due_for_update",
    "encode_batch",
    "ensure_continual_update_job",
    "ensure_distill_job",
    "evaluate_on_held_out",
    "example_stats",
    "export_module",
    "file_checksum",
    "format_seconds",
    "gate_with_registry",
    "get_condition_spec",
    "global_grad_norm",
    "hamming",
    "harvest_kept_turns",
    "judge_and_capture",
    "lr_factor",
    "load_condition_models",
    "load_dataset",
    "load_exported",
    "load_latent_models",
    "load_saved_adapter",
    "periodic_distill",
    "post_latent_preparation",
    "prepare",
    "prepare_conditions",
    "preprocess_video",
    "probe_video",
    "production_gate",
    "progress_bar",
    "quality_filter",
    "read_jsonl",
    "receipt_path_for",
    "refined_quality_signals",
    "render_eval_report",
    "render_gate_report",
    "render_policy_card",
    "render_run_card",
    "resolve_device",
    "run_continual_update_once",
    "run_distill_once",
    "run_quick_training",
    "select_tensor_io",
    "should_perform_precomputation",
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
    "verify_export",
    "write_format_bundles",
    "write_jsonl",
]
