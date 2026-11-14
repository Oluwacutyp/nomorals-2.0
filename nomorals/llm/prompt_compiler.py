"""Prompt compiler — DSPy-style self-improving prompt pipelines, Devon-built.

DSPy (Declarative Self-improving Python) is the framework for
_programming_ — not prompting — language models: write a declarative
signature (input fields -> output fields + an instruction) and let an
optimizer compile it into a prompt that scores well, measured on a
held-out set. The pattern, mined from DSPy's public docs and its
paper ("DSPy: Compiling Declarative Language Model Calls into
Self-Improving Pipelines", arXiv:2310.03714) — the pattern, not the
library: nothing vendored, no new dependency, stdlib only.

What is mined, and what it becomes here:

* **Signature** — a declarative input -> output spec + instruction.
  Here: :class:`PromptSignature` (task, instruction, input fields,
  output fields, per-task floors).
* **BootstrapFewShot** — run the program on labeled training inputs,
  keep the traces that score well as few-shot demonstrations. Here
  the program already ran: the journal holds the traces (the lines
  the brain actually produced, graded by the owner), so the compiler
  SELECTS demos from the graded record instead of re-running the
  model — the best-graded outputs in, the worst-graded as
  counter-examples where they teach.
* **MIPRO** — jointly optimize the instruction and the demonstrations
  against a metric. Here the instruction refinement is deterministic:
  the compiler compares high-grade vs low-grade outputs on measurable
  traits (length, line count, punctuation) and appends a note naming
  only the traits that separate the quartiles by a clear margin —
  derived from the examples, never invented.
* **Metric-driven held-out evaluation** — compile on a train slice,
  measure on a held-out slice. Here: :class:`CompileReport` carries
  the base prompt's measured grade (the mean grade of every recorded
  output — they were all produced by the base prompt) against the
  compiled prompt's demo-set grade, and — when the caller supplies a
  model runner and a metric — a real before/after: the base and
  compiled prompts are each run on held-out inputs and scored.

The compiler plugs into the loop that already exists
(:mod:`nomorals.llm.specialists` — journal -> grade -> retrain). It
consumes the SAME graded rows the learner jobs consume
(``{"features": ..., "label": grade, "line": text, ...}`` — see
:class:`GradedExample.from_row`) and it never reads a journal itself:
callers supply the examples, the way ``fit_if_ready`` takes rows.
A compiled prompt is a versioned artifact in :class:`PromptStore`
(one JSON file per version + a manifest, the same shape as
``ModelStore``), with the example ids and grades it was compiled
from recorded alongside it, so it can be re-compiled when more
grades arrive.

Promotion mirrors ``fit_if_ready`` exactly:

* below the task's floor -> ``insufficient`` (plain reason, the base
  prompt carries, nothing saved);
* validated and the compiled prompt does not beat the base on the
  held-out slice -> ``held`` (not saved);
* validated and it beats the base, or unvalidated and the selected
  demos grade above the task average -> ``compiled`` (saved +
  promoted to the task's current prompt).

Unvalidated compiles are labeled as such in the report (``validated:
False``): the demo-set grade is a SELECTION effect, not a proven
causal gain — the next grades, earned by the compiled prompt in
live use, are what prove it. The compiler never invents a grade and
never pads with made-up examples: too few graded rows is a plain
refusal, not a guess.

Layer: L3 (``llm``) — imports only stdlib + ``core``. The model
runner is injected (a ``(prompt) -> text`` callable, the ladder's
``SuggestFn`` shape), never imported: the brain lives in this same
layer and importing it would be a cycle.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

__all__ = [
    "PromptSignature", "GradedExample", "CompiledPrompt",
    "CompileReport", "CompileOutcome", "PromptStore",
    "NotEnoughData", "compile_prompt", "render_prompt",
    "render_base_prompt", "imitation_metric", "default_prompts_dir",
    "control_prompt",
    "DEFAULT_MIN_EXAMPLES", "DEFAULT_MAX_DEMOS",
    "DEFAULT_MAX_COUNTER_EXAMPLES",
]

_log = get_logger(__name__)

#: Floor: how few graded examples a task may compile from. Few-shot
#: selection needs enough rows to pick demos AND hold out a test
#: slice; below 8 there is nothing meaningful to select from. Lower
#: than the classical-ML floor (30) on purpose: the compiler SELECTS
#: recorded outputs, it does not fit an estimator — but the floor
#: still exists, and the held-out proof still applies. A task may
#: declare a higher floor in its signature.
DEFAULT_MIN_EXAMPLES = 8
#: Demos in one compiled prompt. Four keeps the prompt short enough
#: to stay useful as a few-shot block.
DEFAULT_MAX_DEMOS = 4
#: Counter-examples ("avoid") in one compiled prompt.
DEFAULT_MAX_COUNTER_EXAMPLES = 2
#: Held-out fraction of the examples, mirroring the specialists split.
DEFAULT_TEST_FRAC = 0.2
#: A single example's output is truncated past this in the rendered
#: prompt — a runaway journal line must not blow up the prompt.
MAX_EXAMPLE_CHARS = 2000
#: A compiled prompt is trimmed past this: counter-examples go first,
#: then the lowest-grade demo, until it fits.
MAX_PROMPT_CHARS = 20_000

#: Input keys a journal row may carry, in preference order. The NPC
#: journal stores the situation; the character journal stores the
#: context. ``GradedExample.from_row`` reads whichever is present.
_ROW_INPUT_KEYS = ("situation", "context")

#: Traits the instruction refinement measures on an output. Each is a
#: (name, extractor) pair; numeric traits compare means, boolean
#: traits compare rates.
_TRAIT_NAMES = ("chars", "words", "lines", "question", "exclamation",
                "ends_period")


class NotEnoughData(Exception):
    """Raised when a task's journal holds fewer rows than its floor."""


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def _traits(text: str) -> dict[str, float]:
    """Measurable traits of one output, for the instruction refinement."""
    body = text or ""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return {
        "chars": float(len(body)),
        "words": float(len(_words(body))),
        "lines": float(len(lines)),
        "question": 1.0 if "?" in body else 0.0,
        "exclamation": 1.0 if "!" in body else 0.0,
        "ends_period": 1.0 if body.rstrip().endswith((".", "!", "?")) else 0.0,
    }


@dataclass(frozen=True)
class PromptSignature:
    """One compilable task: what the prompt must do.

    ``instruction`` is the base instruction (the task's current
    prompt text, e.g. the NPC speak instruction). ``input_fields``
    name the fields each example — and each new input — carries
    (``("situation",)`` for NPC dialogue, ``("context",)`` for
    character dialogue); ``output_fields`` name what the model must
    produce (``("line",)`` for both). The three ``None``-able floors
    are per-task config: a task declares its own, and the compiler
    falls back to the ``[prompt_compiler]`` settings defaults, then
    to the module defaults.

    ``input_label`` / ``output_label`` are the labels rendering uses
    for the input / output lines, when they differ from the field
    names: the label is what the model actually completes after, so
    it must be the label the runtime prompt carries. A character
    dialogue render ends on ``"Situation: ..."`` / ``"{name}:"``
    (the speak path's turn label), while the fields stay
    ``"context"`` / ``"line"`` (the journal keys). ``None`` keeps the
    historical capitalized-field labels — NPC dialogue renders
    ``"Situation:"`` / ``"Line:"`` exactly as before.
    """

    task: str
    instruction: str
    input_fields: tuple[str, ...] = ("situation",)
    output_fields: tuple[str, ...] = ("line",)
    min_examples: int | None = None
    max_demos: int | None = None
    max_counter_examples: int | None = None
    input_label: str | None = None
    output_label: str | None = None

    def effective(self, settings: Any = None) -> PromptSignature:
        """The signature with every ``None`` floor resolved."""
        def _pick(own: int | None, key: str, default: int) -> int:
            if own is not None:
                return int(own)
            if settings is not None:
                value = getattr(settings, key, None)
                if value is not None:
                    return int(value)
            return default

        return PromptSignature(
            task=self.task, instruction=self.instruction,
            input_fields=self.input_fields,
            output_fields=self.output_fields,
            min_examples=_pick(self.min_examples,
                               "default_min_examples",
                               DEFAULT_MIN_EXAMPLES),
            max_demos=_pick(self.max_demos, "default_max_demos",
                            DEFAULT_MAX_DEMOS),
            max_counter_examples=_pick(
                self.max_counter_examples,
                "default_max_counter_examples",
                DEFAULT_MAX_COUNTER_EXAMPLES),
            input_label=self.input_label,
            output_label=self.output_label,
        )


@dataclass(frozen=True)
class GradedExample:
    """One graded journal example: inputs, the output, its grade.

    ``grade`` is on the journal's -1..1 scale (owner grades). ``id``
    is the journal entry id (provenance); ``source`` names where the
    grade came from (``"owner"`` for owner grades).
    """

    inputs: dict[str, str]
    output: str
    grade: float
    id: str = ""
    source: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GradedExample | None:
        """A specialists-style journal row -> an example, or None.

        Accepts the rows ``fit_if_ready`` consumes (``features`` /
        ``label`` / ``line`` plus a ``situation`` or ``context`` input)
        and rows already in example shape (``inputs`` / ``output`` /
        ``grade``). A row without a usable output or grade is not an
        example — it is skipped, never relabelled or padded.
        """
        if not isinstance(row, dict):
            return None
        if isinstance(row.get("inputs"), dict) and "output" in row:
            grade = row.get("grade")
            try:
                grade_f = float(grade) if grade is not None else None
            except (TypeError, ValueError):
                return None
            if grade_f is None or math.isnan(grade_f) or math.isinf(grade_f):
                return None
            inputs = {str(k): str(v) for k, v in row["inputs"].items()}
            return cls(inputs=inputs, output=str(row.get("output") or ""),
                       grade=max(-1.0, min(1.0, grade_f)),
                       id=str(row.get("id") or row.get("entry_id") or ""),
                       source=str(row.get("source") or ""))
        line = row.get("line")
        label = row.get("label")
        if line is None or label is None:
            return None
        try:
            grade_f = float(label)
        except (TypeError, ValueError):
            return None
        if math.isnan(grade_f) or math.isinf(grade_f):
            return None
        inputs: dict[str, str] = {}
        for key in _ROW_INPUT_KEYS:
            value = row.get(key)
            if value:
                inputs[key] = str(value)
        return cls(inputs=inputs, output=str(line),
                   grade=max(-1.0, min(1.0, grade_f)),
                   id=str(row.get("entry_id") or row.get("id") or ""),
                   source=str(row.get("source") or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": dict(self.inputs), "output": self.output,
                "grade": self.grade, "id": self.id,
                "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradedExample:
        return cls(inputs={str(k): str(v)
                           for k, v in (data.get("inputs") or {}).items()},
                   output=str(data.get("output") or ""),
                   grade=float(data.get("grade") or 0.0),
                   id=str(data.get("id") or ""),
                   source=str(data.get("source") or ""))


@dataclass
class CompileReport:
    """The before/after numbers for one compile — the proof."""

    n_examples: int
    n_train: int
    n_test: int
    baseline_grade: float  # mean grade of every example (the base prompt)
    demo_grade: float | None = None  # mean grade of the selected demos
    selection_lift: float | None = None  # demo_grade - baseline_grade
    heldout_base_score: float | None = None  # validation only
    heldout_compiled_score: float | None = None  # validation only
    beats_baseline: bool | None = None
    validated: bool = False
    status: str = ""  # "compiled" | "held" | "insufficient" | "disabled"
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples, "n_train": self.n_train,
            "n_test": self.n_test,
            "baseline_grade": round(self.baseline_grade, 4),
            "demo_grade": (round(self.demo_grade, 4)
                           if self.demo_grade is not None else None),
            "selection_lift": (round(self.selection_lift, 4)
                               if self.selection_lift is not None else None),
            "heldout_base_score": (
                round(self.heldout_base_score, 4)
                if self.heldout_base_score is not None else None),
            "heldout_compiled_score": (
                round(self.heldout_compiled_score, 4)
                if self.heldout_compiled_score is not None else None),
            "beats_baseline": self.beats_baseline,
            "validated": self.validated, "status": self.status,
            "reason": self.reason, "detail": dict(self.detail),
        }


@dataclass
class CompiledPrompt:
    """A compiled prompt: a versioned artifact with its provenance."""

    task: str
    version: int
    instruction: str
    instruction_note: str
    prompt_text: str
    demos: list[GradedExample] = field(default_factory=list)
    counter_examples: list[GradedExample] = field(default_factory=list)
    report: CompileReport | None = None
    compiled_at: float = 0.0
    example_ids: list[str] = field(default_factory=list)
    grades: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "version": self.version,
            "instruction": self.instruction,
            "instruction_note": self.instruction_note,
            "prompt_text": self.prompt_text,
            "demos": [d.to_dict() for d in self.demos],
            "counter_examples": [d.to_dict()
                                 for d in self.counter_examples],
            "report": self.report.to_dict() if self.report else {},
            "compiled_at": self.compiled_at,
            "example_ids": list(self.example_ids),
            "grades": list(self.grades),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompiledPrompt:
        raw_rep = data.get("report") or {}
        report = None
        if raw_rep:
            report = CompileReport(
                n_examples=int(raw_rep.get("n_examples", 0) or 0),
                n_train=int(raw_rep.get("n_train", 0) or 0),
                n_test=int(raw_rep.get("n_test", 0) or 0),
                baseline_grade=float(raw_rep.get("baseline_grade", 0.0)
                                     or 0.0),
                demo_grade=raw_rep.get("demo_grade"),
                selection_lift=raw_rep.get("selection_lift"),
                heldout_base_score=raw_rep.get("heldout_base_score"),
                heldout_compiled_score=raw_rep.get(
                    "heldout_compiled_score"),
                beats_baseline=raw_rep.get("beats_baseline"),
                validated=bool(raw_rep.get("validated", False)),
                status=str(raw_rep.get("status") or ""),
                reason=str(raw_rep.get("reason") or ""),
                detail=dict(raw_rep.get("detail") or {}))
        return cls(
            task=str(data.get("task") or ""),
            version=int(data.get("version", 0) or 0),
            instruction=str(data.get("instruction") or ""),
            instruction_note=str(data.get("instruction_note") or ""),
            prompt_text=str(data.get("prompt_text") or ""),
            demos=[GradedExample.from_dict(d)
                   for d in (data.get("demos") or [])],
            counter_examples=[GradedExample.from_dict(d)
                              for d in (data.get("counter_examples")
                                        or [])],
            report=report,
            compiled_at=float(data.get("compiled_at", 0.0) or 0.0),
            example_ids=[str(v) for v in (data.get("example_ids") or [])],
            grades=[float(v) for v in (data.get("grades") or [])])


@dataclass
class CompileOutcome:
    """What :func:`compile_prompt` decided, and why."""

    status: str  # "compiled" | "held" | "insufficient" | "disabled"
    compiled: CompiledPrompt | None = None
    reason: str = ""
    report: CompileReport | None = None
    n_examples: int = 0


# ── rendering ───────────────────────────────────────────────────────────

def _clip(text: str, limit: int = MAX_EXAMPLE_CHARS) -> str:
    body = text or ""
    if len(body) <= limit:
        return body
    return body[:limit] + " […]"


def _in_label(signature: PromptSignature) -> str:
    """The rendered input label: the signature's override when set,
    else the capitalized field name (the historical behavior)."""
    if signature.input_label:
        return signature.input_label
    in_field = signature.input_fields[0] if signature.input_fields else ""
    return in_field.capitalize()


def _out_label(signature: PromptSignature) -> str:
    """The rendered output label: the signature's override when set,
    else the capitalized field name (the historical behavior)."""
    if signature.output_label:
        return signature.output_label
    out_field = signature.output_fields[0] if signature.output_fields \
        else ""
    return out_field.capitalize()


def _demo_block(examples: list[GradedExample],
                signature: PromptSignature, label: str) -> str:
    """One few-shot block: each demo as input fields + output field."""
    if not examples:
        return ""
    parts = [label]
    in_field = signature.input_fields[0] if signature.input_fields else ""
    for ex in examples:
        given = ex.inputs.get(in_field, "")
        parts.append(f"{_in_label(signature)}: {_clip(given, 400)}")
        parts.append(f"{_out_label(signature)}: {_clip(ex.output)} "
                     f"(graded {ex.grade:+.2f})")
    return "\n".join(parts)


def _assemble(signature: PromptSignature,
              demos: list[GradedExample],
              counter_examples: list[GradedExample],
              instruction_note: str,
              inputs: dict[str, str]) -> str:
    """One render of the compiled prompt: instruction + note + demo
    blocks + the new input + the output label. A missing input field
    renders as a ``{field}`` placeholder, so the stored artifact reads
    as a template the caller fills at use time."""
    in_field = signature.input_fields[0] if signature.input_fields else ""
    chunks = [signature.instruction.strip()]
    if instruction_note:
        chunks.append("\nWhat the graded record shows:\n"
                      + instruction_note)
    demo_text = _demo_block(
        demos, signature,
        "Examples — lines that graded highest for this task:")
    if demo_text:
        chunks.append("\n" + demo_text)
    counter_text = _demo_block(
        counter_examples, signature,
        "Avoid — lines that graded lowest for this task:")
    if counter_text:
        chunks.append("\n" + counter_text)
    given = (inputs or {}).get(in_field) or (f"{{{in_field}}}" if in_field
                                             else "")
    chunks.append(f"\n{_in_label(signature)}: {given}\n"
                  f"{_out_label(signature)}:")
    return "\n".join(c for c in chunks if c)


def render_prompt(signature: PromptSignature,
                  demos: list[GradedExample],
                  counter_examples: list[GradedExample],
                  instruction_note: str,
                  inputs: dict[str, str]) -> str:
    """The compiled prompt for one new input: instruction + note +
    demos + counter-examples + the new input + the output label.

    The caller prepends its own persona block (the NPC's
    ``to_prompt()``, the character's persona block) at USE time —
    the compiled prompt is the task's instruction and demos, not the
    persona. The prompt ends on the output-field label so the model
    completes it.
    """
    text = _assemble(signature, demos, counter_examples,
                     instruction_note, inputs)
    return _fit_prompt(text, signature, demos, counter_examples,
                       instruction_note, inputs)


def _fit_prompt(text: str, signature: PromptSignature,
                demos: list[GradedExample],
                counter_examples: list[GradedExample],
                instruction_note: str,
                inputs: dict[str, str]) -> str:
    """Trim a rendered prompt past MAX_PROMPT_CHARS: counter-examples
    go first, then the lowest-grade demo, until it fits. The
    instruction itself is never cut."""
    if len(text) <= MAX_PROMPT_CHARS:
        return text
    counters = list(counter_examples)
    live_demos = list(demos)
    while len(text) > MAX_PROMPT_CHARS and (counters or live_demos):
        if counters:
            counters = counters[:-1]
        elif live_demos:
            live_demos = live_demos[:-1]
        text = _assemble(signature, live_demos, counters,
                         instruction_note, inputs)
    if len(text) > MAX_PROMPT_CHARS:
        text = signature.instruction.strip()[:MAX_PROMPT_CHARS]
    return text


def render_base_prompt(signature: PromptSignature,
                       inputs: dict[str, str]) -> str:
    """The base prompt for one input: instruction only, no demos —
    the comparison point for validation."""
    in_field = signature.input_fields[0] if signature.input_fields else ""
    given = (inputs or {}).get(in_field) or (f"{{{in_field}}}" if in_field
                                             else "")
    return (f"{signature.instruction.strip()}\n\n"
            f"{_in_label(signature)}: {given}\n"
            f"{_out_label(signature)}:")


# ── validation metric ───────────────────────────────────────────────────

def imitation_metric(output: str, example: GradedExample) -> float:
    """Default validation metric: resemblance to the recorded line.

    Token-overlap (Jaccard on word sets) between the fresh output and
    the held-out example's recorded output, scaled by that output's
    grade — resembling a high-graded line scores high, resembling a
    low-graded line scores low. A PROXY, not a quality judgment: it
    measures whether the compiled prompt elicits lines like the good
    recorded ones. A task with its own scorer (the NPC hand scorer,
    the character scorer) should pass that as the metric instead.
    """
    fresh = set(_words(output or ""))
    recorded = set(_words(example.output or ""))
    if not fresh or not recorded:
        return 0.0
    overlap = len(fresh & recorded) / len(fresh | recorded)
    return round(overlap * example.grade, 4)


# ── selection + instruction refinement ──────────────────────────────────

def _dedup(examples: list[GradedExample]) -> list[GradedExample]:
    """Drop exact (inputs, output) duplicates, keeping the highest
    grade — a duplicate must not eat two demo slots."""
    best: dict[tuple[str, str], GradedExample] = {}
    for ex in examples:
        key_in = "|".join(f"{k}={v}" for k, v in sorted(ex.inputs.items()))
        key = (key_in, " ".join((ex.output or "").split()))
        prev = best.get(key)
        if prev is None or ex.grade > prev.grade:
            best[key] = ex
    return list(best.values())


def _split(examples: list[GradedExample], seed: int = 0,
           test_frac: float = DEFAULT_TEST_FRAC
           ) -> tuple[list[GradedExample], list[GradedExample]]:
    """Deterministic train/test split (fixed seed)."""
    ordered = sorted(examples, key=lambda e: (e.id, e.output))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_test = max(1, min(n - 1, int(round(n * test_frac))))
    return ordered[n_test:], ordered[:n_test]


def _select_demos(train: list[GradedExample], max_demos: int,
                  max_counter: int
                  ) -> tuple[list[GradedExample], list[GradedExample]]:
    """Top-graded train examples as demos (grade must be positive —
    a demo is a GOOD example), bottom-graded as counter-examples
    (grade must be negative). Deterministic ties: grade, then id."""
    ranked = sorted(train, key=lambda e: (-e.grade, e.id))
    demos = [e for e in ranked if e.grade > 0][:max_demos]
    worst = sorted(train, key=lambda e: (e.grade, e.id))
    counters = [e for e in worst if e.grade < 0][:max_counter]
    return demos, counters


def _refine_instruction(train: list[GradedExample]) -> str:
    """Instruction notes from the graded outcomes, deterministically.

    Splits the train examples into top and bottom halves by grade
    and compares each output trait. A trait that separates the halves
    by a clear margin — numeric traits by >25% relative, boolean
    traits by >0.4 in rate — earns one plain-words note stating the
    measured numbers. No trait separating means no note: the
    instruction stands as-is, never padded with a guess. At most
    three notes, in fixed trait order.
    """
    if len(train) < 4:
        return ""
    ranked = sorted(train, key=lambda e: -e.grade)
    half = max(2, len(ranked) // 2)
    top = ranked[:half]
    bottom = ranked[-half:]
    if len(top) < 2 or len(bottom) < 2:
        return ""
    top_traits = [_traits(e.output) for e in top]
    bot_traits = [_traits(e.output) for e in bottom]

    def _mean(rows: list[dict[str, float]], key: str) -> float:
        return sum(r[key] for r in rows) / max(len(rows), 1)

    notes: list[str] = []
    for key in _TRAIT_NAMES:
        hi = _mean(top_traits, key)
        lo = _mean(bot_traits, key)
        if key in ("chars", "words", "lines"):
            if lo <= 0 or hi <= 0:
                continue
            rel = abs(hi - lo) / max(hi, lo)
            if rel <= 0.25:
                continue
            direction = "shorter" if hi < lo else "longer"
            unit = {"chars": "characters", "words": "words",
                    "lines": "lines"}[key]
            notes.append(
                f"Highest-graded outputs average {hi:.0f} {unit}; "
                f"lowest-graded average {lo:.0f} — {direction} outputs "
                f"grade higher.")
        else:
            if abs(hi - lo) <= 0.4:
                continue
            verb = {"question": "ask a question",
                    "exclamation": "use an exclamation",
                    "ends_period": "end with a full stop"}[key]
            often = "usually" if hi >= 0.6 else "sometimes"
            rarely = "rarely" if lo <= 0.4 else "sometimes"
            notes.append(
                f"Highest-graded outputs {often} {verb} "
                f"({hi:.0%} of them); lowest-graded {rarely} do "
                f"({lo:.0%} of them).")
        if len(notes) >= 3:
            break
    if not notes:
        return ""
    return "- " + "\n- ".join(notes)


# ── the compile gate ────────────────────────────────────────────────────

def compile_prompt(signature: PromptSignature,
                   examples: list[GradedExample], *,
                   suggest: Callable[[str], str] | None = None,
                   metric: Callable[[str, GradedExample], float] | None = None,
                   seed: int = 0,
                   store: PromptStore | None = None,
                   settings: Any = None,
                   bus: Any = None) -> CompileOutcome:
    """The full gate: floor -> split -> select -> refine -> render ->
    validate (when a runner + metric are supplied) -> promote or hold.

    ``suggest`` is the model runner (``(prompt) -> text``, the
    ladder's SuggestFn shape); ``metric`` scores a fresh output
    against a held-out example (``(output, example) -> float``).
    Validation runs ONLY when both are given — otherwise the compile
    is unvalidated and the report says so. A held compile is NOT
    saved; a compiled one is saved + promoted through ``store`` (a
    fresh :class:`PromptStore` when none is given).
    """
    sig = signature.effective(settings)
    if settings is not None and not getattr(settings, "enabled", True):
        report = CompileReport(
            n_examples=len(examples or []), n_train=0, n_test=0,
            baseline_grade=0.0, status="disabled",
            reason="prompt compiling is off in config [prompt_compiler]")
        _journal("prompt.disabled", sig.task, report, bus)
        return CompileOutcome(status="disabled", report=report,
                              reason=report.reason,
                              n_examples=len(examples or []))
    usable: list[GradedExample] = []
    skipped = 0
    for raw in (examples or []):
        ex = raw if isinstance(raw, GradedExample) else None
        if ex is None and isinstance(raw, dict):
            ex = GradedExample.from_row(raw)
        if ex is None or not (ex.output or "").strip():
            skipped += 1
            continue
        usable.append(ex)
    usable = _dedup(usable)
    n = len(usable)
    if n < (sig.min_examples or DEFAULT_MIN_EXAMPLES):
        report = CompileReport(
            n_examples=n, n_train=0, n_test=0, baseline_grade=0.0,
            status="insufficient",
            reason=(f"{sig.task}: {n}/{sig.min_examples} graded "
                    f"examples — below the floor; the base prompt "
                    f"carries"),
            detail={"skipped": skipped})
        _journal("prompt.insufficient", sig.task, report, bus)
        return CompileOutcome(status="insufficient", report=report,
                              reason=report.reason, n_examples=n)
    train, test = _split(usable, seed=seed)
    baseline = sum(e.grade for e in usable) / max(n, 1)
    demos, counters = _select_demos(
        train, sig.max_demos or DEFAULT_MAX_DEMOS,
        sig.max_counter_examples or DEFAULT_MAX_COUNTER_EXAMPLES)
    note = _refine_instruction(train)
    demo_grade = (sum(e.grade for e in demos) / len(demos)
                  if demos else None)
    report = CompileReport(
        n_examples=n, n_train=len(train), n_test=len(test),
        baseline_grade=round(baseline, 4),
        demo_grade=(round(demo_grade, 4)
                    if demo_grade is not None else None),
        selection_lift=(round(demo_grade - baseline, 4)
                        if demo_grade is not None else None),
        detail={"skipped": skipped,
                "instruction_notes": len(note.splitlines()) if note else 0,
                "demos": len(demos),
                "counter_examples": len(counters)})
    if not demos:
        report.status = "held"
        report.reason = (f"{sig.task}: no positively-graded example "
                         f"to show — the record has nothing good to "
                         f"imitate yet")
        _journal("prompt.held", sig.task, report, bus)
        return CompileOutcome(status="held", report=report,
                              reason=report.reason, n_examples=n)
    # Validation: base vs compiled on the held-out inputs, scored.
    validated = False
    if suggest is not None and metric is not None:
        base_scores: list[float] = []
        comp_scores: list[float] = []
        failed = 0
        for ex in test:
            try:
                base_out = suggest(render_base_prompt(sig, ex.inputs))
                comp_out = suggest(render_prompt(
                    sig, demos, counters, note, ex.inputs))
                base_scores.append(float(metric(base_out, ex)))
                comp_scores.append(float(metric(comp_out, ex)))
            except Exception:  # noqa: BLE001 - a failed run is skipped
                failed += 1
                _log.debug("prompt validation run failed", exc_info=True)
                continue
        report.detail["validation_failed"] = failed
        if base_scores and comp_scores:
            validated = True
            report.validated = True
            report.heldout_base_score = round(
                sum(base_scores) / len(base_scores), 4)
            report.heldout_compiled_score = round(
                sum(comp_scores) / len(comp_scores), 4)
            report.beats_baseline = bool(
                report.heldout_compiled_score
                > report.heldout_base_score)
        else:
            report.reason = ("validation ran but no held-out run "
                             "produced a score — compiled from the best "
                             "recorded outputs, not yet proven")
    if not validated:
        report.beats_baseline = bool(
            demo_grade is not None and demo_grade > baseline)
    prompt_text = render_prompt(sig, demos, counters, note, {})
    compiled = CompiledPrompt(
        task=sig.task, version=0, instruction=sig.instruction,
        instruction_note=note, prompt_text=prompt_text,
        demos=list(demos), counter_examples=list(counters),
        report=report, compiled_at=time.time(),
        example_ids=[e.id for e in usable],
        grades=[e.grade for e in usable])
    if report.beats_baseline:
        report.status = "compiled"
        report.reason = (
            f"{sig.task}: compiled from {n} graded examples "
            f"(base grade {report.baseline_grade:+.2f} -> demo grade "
            f"{report.demo_grade:+.2f})"
            + ("; validated on the held-out slice "
               f"(base {report.heldout_base_score:+.2f} -> compiled "
               f"{report.heldout_compiled_score:+.2f})"
               if validated else
               "; unvalidated — the next live grades will judge it"))
        store = store or PromptStore()
        compiled.version = store.save(sig.task, compiled)
        _journal("prompt.compiled", sig.task, report, bus)
        return CompileOutcome(status="compiled", compiled=compiled,
                              report=report, reason=report.reason,
                              n_examples=n)
    report.status = "held"
    if validated:
        report.reason = (
            f"{sig.task}: held-out {report.heldout_compiled_score:+.2f} "
            f"does not beat the base prompt "
            f"{report.heldout_base_score:+.2f} — the base prompt carries")
    else:
        report.reason = (
            f"{sig.task}: the best recorded outputs grade no higher "
            f"than the task average — there is nothing to select")
    _journal("prompt.held", sig.task, report, bus)
    return CompileOutcome(status="held", report=report,
                          reason=report.reason, n_examples=n)


def _journal(topic: str, task: str, report: CompileReport,
             bus: Any) -> None:
    """Record one compile on the existing event bus — counts and
    grades only, never example text or prompt text. Never raises."""
    try:
        target = global_bus if bus is None else bus
        if target is False:
            return
        target.publish(Event(topic=topic, data={
            "task": task, "status": report.status,
            "n_examples": report.n_examples,
            "baseline_grade": report.baseline_grade,
            "demo_grade": report.demo_grade,
            "validated": report.validated,
            "beats_baseline": report.beats_baseline,
        }))
    except Exception:  # noqa: BLE001 - journaling never kills a compile
        _log.debug("prompt journal failed", exc_info=True)


# ── versioned store ─────────────────────────────────────────────────────

def default_prompts_dir() -> Path:
    """Prompt home: ~/.nomorals/prompts (NM_PROMPTS_DIR overrides)."""
    override = (os.environ.get("NM_PROMPTS_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".nomorals" / "prompts"


class PromptStore:
    """Versioned on-disk store: one task, one JSON file per compiled
    prompt version, plus a manifest carrying the before/after numbers.
    Atomic tmp+replace writes; a corrupt manifest degrades to 'no
    prompt', never a crash. Mirrors ``ModelStore``'s shape so the two
    versioned artifacts read the same way."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (Path(base_dir) if base_dir is not None
                         else default_prompts_dir())

    def _task_dir(self, task: str) -> Path:
        safe = "".join(c for c in (task or "")
                       if c.isalnum() or c in "-_.")[:64] or "task"
        return self.base_dir / safe

    def _manifest_path(self, task: str) -> Path:
        return self._task_dir(task) / "manifest.json"

    def _read_manifest(self, task: str) -> dict[str, Any]:
        try:
            raw = self._manifest_path(task).read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - corrupt manifest = no prompt
            return {}

    def latest_version(self, task: str) -> int:
        return int(self._read_manifest(task).get("latest", 0) or 0)

    def list_tasks(self) -> list[str]:
        try:
            return sorted(p.name for p in self.base_dir.iterdir()
                          if p.is_dir()
                          and (p / "manifest.json").is_file())
        except Exception:  # noqa: BLE001 - a missing dir is no tasks
            return []

    def history(self, task: str) -> list[dict[str, Any]]:
        raw = self._read_manifest(task).get("history") or []
        return [dict(h) for h in raw if isinstance(h, dict)]

    def save(self, task: str, compiled: CompiledPrompt) -> int:
        """Persist ``compiled`` as the next version. Returns the version."""
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        version = self.latest_version(task) + 1
        compiled.version = version
        target = task_dir / f"{task_dir.name}-v{version}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(compiled.to_dict(), indent=1),
                       encoding="utf-8")
        tmp.replace(target)
        manifest = self._read_manifest(task)
        rep = compiled.report.to_dict() if compiled.report else {}
        manifest.update({
            "task": task, "latest": version,
            "compiled_at": compiled.compiled_at,
            "n_examples": compiled.report.n_examples
            if compiled.report else 0,
            "report": rep,
            "history": (manifest.get("history") or []) + [{
                "version": version,
                "compiled_at": compiled.compiled_at,
                "n_examples": compiled.report.n_examples
                if compiled.report else 0,
                "baseline_grade": rep.get("baseline_grade"),
                "demo_grade": rep.get("demo_grade"),
                "beats_baseline": rep.get("beats_baseline"),
                "validated": rep.get("validated"),
                "status": rep.get("status"),
            }],
        })
        m_tmp = self._manifest_path(task).with_suffix(".tmp")
        m_tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        m_tmp.replace(self._manifest_path(task))
        return version

    def load(self, task: str) -> CompiledPrompt | None:
        """The latest compiled prompt, or None (never raises)."""
        return self.load_version(task, self.latest_version(task))

    def load_version(self, task: str, version: int) -> CompiledPrompt | None:
        """One version, or None (never raises)."""
        if version <= 0:
            return None
        target = (self._task_dir(task)
                  / f"{self._task_dir(task).name}-v{version}.json")
        try:
            raw = target.read_text(encoding="utf-8")
            data = json.loads(raw)
            return CompiledPrompt.from_dict(data) if data else None
        except Exception:  # noqa: BLE001 - a corrupt prompt = no prompt
            _log.debug("prompt load failed for %s v%s", task, version,
                       exc_info=True)
            return None


# ── chat command ────────────────────────────────────────────────────────

def control_prompt(tail: str, settings: Any = None,
                   store: PromptStore | None = None) -> str:
    """The /prompt command. Owner-only (CONTROL_COMMANDS).

    /prompt — status of every compiled task · /prompt status [task] —
    one task's numbers · /prompt list — tasks + versions. List-only:
    compiling happens through ``nm prompt compile`` or the
    ``prompt_compile`` tool, never from chat.
    """
    store = store or PromptStore()
    parts = (tail or "").split()
    verb = parts[0].lower() if parts else "status"
    if verb == "compile":
        return ("Compiling happens through `nm prompt compile <task>` "
                "or the prompt_compile tool — not from chat.")
    if verb == "list":
        tasks = store.list_tasks()
        if not tasks:
            return ("Prompt compiler: nothing compiled yet — "
                    "run `nm prompt compile <task>` first.")
        lines = ["Prompt compiler — compiled tasks:"]
        for task in tasks:
            latest = store.latest_version(task)
            hist = store.history(task)
            lines.append(f"· {task} — v{latest} "
                         f"({len(hist)} version(s))")
        return "\n".join(lines)
    if verb not in ("status",):
        task = verb
    elif len(parts) > 1:
        task = parts[1]
    else:
        task = ""
    if not task:
        tasks = store.list_tasks()
        if not tasks:
            return ("Prompt compiler: nothing compiled yet — "
                    "run `nm prompt compile <task>` first.")
        lines = ["Prompt compiler — status:"]
        for name in tasks:
            lines.append("· " + _status_line(name, store))
        return "\n".join(lines)
    if store.latest_version(task) <= 0:
        return (f"Prompt compiler: no compiled prompt for {task!r} yet — "
                f"run `nm prompt compile {task}` first.")
    return "Prompt compiler — " + _status_line(task, store)


def _status_line(task: str, store: PromptStore) -> str:
    compiled = store.load(task)
    if compiled is None or compiled.report is None:
        return f"{task} — never compiled"
    rep = compiled.report
    base = rep.baseline_grade
    demo = rep.demo_grade
    demo_txt = f"{demo:+.2f}" if demo is not None else "—"
    validated = "validated" if rep.validated else "unvalidated"
    return (f"{task} — v{compiled.version} "
            f"({rep.n_examples} examples; base grade {base:+.2f} -> "
            f"demo grade {demo_txt}; {validated}; "
            f"{rep.status})")
