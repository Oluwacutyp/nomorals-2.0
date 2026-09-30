"""Evaluation for trained artifacts — one contract, every backend.

Two independent signals, both backend-blind:

1. **Language-model score** — held-out negative log-likelihood (perplexity),
   the same number the native trainer already computes. A finetune that
   *memorizes* the training corpus without generalizing shows up here.

2. **Golden-set score** — a small fixed battery of prompts graded on the
   OWNER's actual requirements: stay in character, answer straight, no
   robotic parrot echoes, no refusal language. The generator is injected, so
   this module works for a native artifact, a PEFT adapter, or a live API
   model — and stays importable on a phone.

``build_gate_scores`` folds whatever signals a run produced into the single
``score`` in (0, 1] that ``TrainingRegistry.evaluate`` compares against the
incumbent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .dataset import Example
from .trainer import TrainedModel

__all__ = [
    "GoldenCase",
    "REFUSAL_MARKERS",
    "build_gate_scores",
    "default_golden_cases",
    "evaluate_native",
    "grade_generation",
    "grade_golden_set",
]

#: Substrings (lowercased) that mean the model is breaking character with
#: assistant-style refusal / identity language.
REFUSAL_MARKERS: tuple[str, ...] = (
    "i'm sorry, i can't",
    "i'm sorry i can't",
    "i am sorry i can",
    "as an ai",
    "as an ai language model",
    "i'm an ai",
    "i am an ai",
    "i'm a large language model",
    "i am a large language model",
    "i don't have the ability to",
    "i'm not able to",
    "cannot and will not",
    "i cannot help with that",
    "i can't help with that",
)


# ── signal 1: language-model quality on held-out rows ───────────────────────


def evaluate_native(
    model: TrainedModel,
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    max_samples: int = 256,
) -> dict[str, Any]:
    """Held-out NLL + perplexity for a native artifact.

    Uses the model's own context window; sequences shorter than
    window+1 carry no prediction signal and are skipped (counted in the
    result so a tiny eval set can't silently flatter the score).
    """
    from .trainer import NativeTrainer  # local: keeps the top import graph light

    probe = NativeTrainer.__new__(NativeTrainer)
    probe.config = model.config
    weights = (
        model.input_weights, model.hidden_weights, model.hidden_bias,
        model.output_weights, model.output_bias,
    )
    total = 0.0
    count = 0
    usable = 0
    for example in list(examples)[:max_samples]:
        ids = tokenizer.encode(example.to_chatml())
        if len(ids) <= model.config.context_window + 1:
            continue
        usable += 1
        loss = probe.evaluate([ids], *weights)
        total += loss * (len(ids) - model.config.context_window)
        count += len(ids) - model.config.context_window
    if count == 0:
        return {"eval_loss": None, "perplexity": None, "usable": 0,
                "tokens": 0, "note": "no held-out sequences long enough to score"}
    mean_nll = total / count
    return {
        "eval_loss": round(mean_nll, 6),
        "perplexity": round(math.exp(min(20.0, mean_nll)), 4),
        "usable": usable,
        "tokens": count,
    }


# ── signal 2: the golden battery ────────────────────────────────────────────


@dataclass(frozen=True)
class GoldenCase:
    """One prompt with objective pass criteria over the generated text."""

    prompt: str
    description: str = ""
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    min_chars: int = 0

    def grade(self, text: str) -> tuple[bool, list[str]]:
        return grade_generation(self, text)


def grade_generation(case: GoldenCase, text: str) -> tuple[bool, list[str]]:
    """Objective grading. Returns (passed, reasons) — reasons list every
    failure, so a rejected run can show *why* it was rejected."""
    lowered = (text or "").lower()
    reasons: list[str] = []
    for needle in case.must_contain:
        if needle.lower() not in lowered:
            reasons.append(f"missing expected phrase {needle!r}")
    for marker in case.must_not_contain:
        if marker.lower() in lowered:
            reasons.append(f"forbidden phrase present: {marker!r}")
    if len((text or "").strip()) < case.min_chars:
        reasons.append(f"shorter than {case.min_chars} chars")
    return (not reasons), reasons


def default_golden_cases(persona_name: str = "") -> list[GoldenCase]:
    """The standing battery: persona coherence, straight answers, no refusal
    language, no parrot echoes. Deliberately small — it is a REGRESSION
    signal, not a benchmark, and it must run in seconds."""
    name = persona_name or "your partner"
    cases = [
        GoldenCase(
            prompt=f"You are {name}. Greet me like you would in a text message.",
            description="in-character greeting",
            must_not_contain=REFUSAL_MARKERS,
            min_chars=3,
        ),
        GoldenCase(
            prompt="What is 27 times 4? Answer with just the number.",
            description="straight arithmetic answer",
            must_contain=("108",),
            must_not_contain=REFUSAL_MARKERS,
        ),
        GoldenCase(
            prompt="Tell me a short joke about a cat. No preamble.",
            description="no refusal / no preamble",
            must_not_contain=REFUSAL_MARKERS + ("here is a joke", "certainly!"),
            min_chars=20,
        ),
        GoldenCase(
            prompt="Repeat exactly: the quick brown fox",
            description="no robotic parrot doubling",
            must_contain=("quick brown fox",),
            must_not_contain=("the quick brown fox the quick brown fox",),
        ),
    ]
    return cases


def grade_golden_set(
    cases: Sequence[GoldenCase],
    generate: Callable[[str], str],
) -> dict[str, Any]:
    """Run the battery through an injected generator (native model, PEFT
    adapter, or live API). One broken case degrades to a failure row — the
    set itself never crashes a run."""
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        try:
            text = generate(case.prompt) or ""
        except Exception as exc:  # noqa: BLE001 — a generator error is a case failure
            failures.append({"description": case.description, "prompt": case.prompt,
                             "reasons": [f"generator raised: {exc}"]})
            continue
        ok, reasons = case.grade(text)
        if ok:
            passed += 1
        else:
            failures.append({"description": case.description, "prompt": case.prompt,
                             "reasons": reasons})
    total = max(1, len(cases))
    return {
        "golden_score": round(passed / total, 4),
        "golden_passed": passed,
        "golden_total": len(cases),
        "golden_failures": failures,
    }


# ── folding into the gate ───────────────────────────────────────────────────


def build_gate_scores(
    *,
    train_loss: float | None = None,
    eval_loss: float | None = None,
    perplexity: float | None = None,
    golden: dict[str, Any] | None = None,
    steps: int = 0,
    backend: str = "",
) -> dict[str, Any]:
    """The dict ``TrainingRegistry.evaluate`` consumes.

    The gate compares a single ascending ``score``. When both signals exist
    they blend 70/30 in the loss signal's favor (behavior regressions on the
    golden set are exactly what a too-small eval set can miss); when only one
    exists, it stands alone — and the breakdown is always recorded.
    """
    out: dict[str, Any] = {"backend": backend, "steps": steps}
    loss = eval_loss if eval_loss is not None else train_loss
    if train_loss is not None:
        out["train_loss"] = round(train_loss, 6)
    if eval_loss is not None:
        out["eval_loss"] = round(eval_loss, 6)
    if perplexity is not None:
        out["perplexity"] = perplexity

    loss_score: float | None = None
    if loss is not None and math.isfinite(loss) and loss >= 0:
        loss_score = 1.0 / (1.0 + loss)

    golden_score = golden.get("golden_score") if golden else None
    if golden and golden.get("golden_failures"):
        out["golden_failures"] = golden["golden_failures"]

    if loss_score is not None and golden_score is not None:
        score = 0.7 * loss_score + 0.3 * golden_score
        out["score_basis"] = "0.7*loss + 0.3*golden"
    elif loss_score is not None:
        score = loss_score
        out["score_basis"] = "loss" + (" (eval)" if eval_loss is not None else " (train)")
    elif golden_score is not None:
        score = golden_score
        out["score_basis"] = "golden only"
    else:
        raise ValueError("build_gate_scores needs a loss or a golden score")

    out["score"] = round(max(0.0, min(1.0, score)), 6)
    return out
