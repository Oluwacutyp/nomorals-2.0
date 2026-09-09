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

from .dataset import Dataset, DatasetRegistry, Example, Turn, to_chatml
from .preprocess import CleanStats, clean_text, dedupe, prepare, quality_filter, simhash, split
from .registry import RunStatus, TrainingRegistry, TrainingRun
from .tokenize import BPETokenizer, HFTokenizer, build_tokenizer
from .trainer import NativeTrainer, TrainConfig, TrainMetrics, TrainedModel

__all__ = [
    "BPETokenizer",
    "CleanStats",
    "Dataset",
    "DatasetRegistry",
    "Example",
    "HFTokenizer",
    "NativeTrainer",
    "RunStatus",
    "TrainConfig",
    "TrainMetrics",
    "TrainedModel",
    "TrainingRegistry",
    "TrainingRun",
    "Turn",
    "build_tokenizer",
    "clean_text",
    "dedupe",
    "prepare",
    "quality_filter",
    "simhash",
    "split",
    "to_chatml",
]
