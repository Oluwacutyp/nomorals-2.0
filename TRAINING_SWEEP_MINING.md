# TRAINING module — external mining (sweep, 2026-10-10)

Mining for the 21-file `nomorals/training/` module: how the best implementations
outside this repo do each job, what this module lacks, and what to steal.
Written BEFORE any code, per sweep method.

## 1. Native pure-Python trainer (`trainer.py`) — NativeTrainer / TrainConfig / TrainMetrics

**The best tiny trainers:** Karpathy's micrograd (scalar autograd engine +
tiny MLP, ~150 lines) and the from-scratch MLP tradition it spawned. The
consistent lesson from those implementations: hand-rolled analytic gradients are
fine, but the *optimizer* is where real trainers live — micrograd loops all
use plain SGD with a fixed LR, and the reason tiny trainers stall is exactly
that.

**What this module's NativeTrainer lacks vs. the field:**
- **No AdamW.** Every modern training loop (nanoGPT, HF Trainer, unsloth)
  defaults to AdamW with decoupled weight decay. The module bakes L2 into the
  SGD step (`value + rate*grad - l2*value`) with a naive `lr/(1+0.01*step)`
  decay. AdamW is implementable in ~25 lines of pure Python (per-param
  first/second moments) and converges meaningfully better on the same step
  budget. STEAL: optimizer choice (`sgd` | `adamw`), decoupled weight decay.
- **No LR schedule.** Cosine decay + linear warmup is the standard
  (HF `get_cosine_schedule_with_warmup`). Current decay is a fixed hyperbola
  with no warmup. STEAL: `lr_schedule` (`constant`|`cosine`|`warmup_cosine`).
- **No gradient clipping.** Global-norm clipping (threshold ~1.0) is the
  cheapest anti-explosion insurance; micrograd-style loops omit it, real
  trainers never do. STEAL: `grad_clip`.
- **No early stopping / best-checkpoint restore.** The loop keeps only the
  final weights; best practice (HF Trainer `load_best_model_at_end`) restores
  the best-eval checkpoint. STEAL: `early_stopping_patience` + keep-best.
- **No generation.** The artifact can't *talk*: there is no sampling method,
  so `evaluate_native` exists but the golden battery can't run against a
  native model end-to-end. STEAL: `TrainedModel.generate()` with
  temperature + top-k sampling.
- **No throughput signal.** `TrainMetrics` lacks tokens/sec and a best-epoch
  record. STEAL: `tokens_per_sec`, `best_epoch`, `stopped_early`.

## 2. Retraining policy (`policy.py`) — RetrainingPolicy

**The field:** MLOps drift-detection practice (Evidently, Alibi Detect,
NannyML): two-tier alerting (warn → investigate, critical → act); PSI for
data drift (threshold ~0.2); prediction-drift as a label-free canary; concept
drift needs labels; and the cardinal rule — *drift triggers an investigation,
the investigation decides whether the fix is retraining, dataset expansion, or
a deployment patch* (retrain-on-every-alert is an anti-pattern).

**Gaps:** the policy watches dataset growth, interval, and reflection-score
decline — but has **no distribution-drift signal at all**. It also has no
concept of *why* a retrain is the right response. STEAL:
- a pure-stdlib **PSI (Population Stability Index)** over collected
  score/length distributions, with `drift_threshold` (~0.2) → new reason
  `"distribution drift detected"`;
- decision carries the *response recommendation* (retrain vs. expand dataset),
  not just a boolean.

## 3. Training registry + promotion gate (`registry.py`) — TrainingRegistry

**The field:** MLflow model registry practice — champion/challenger workflow,
promotion as a *gated pipeline* (reproducibility → schema → performance ≥
champion + **minimum gain** → slice/fairness → latency), never auto-promote,
shadow → canary → production progressive rollout, rollback tag always kept one
step away, approval gates with two-person rule, and **model cards** as the
standard attached document.

**Gaps in this module's gate:**
- `beats_incumbent` with `tolerance=0` lets a +0.0001 noise win promote.
  STEAL: `min_gain` — challenger must beat champion by a margin.
- No staged rollout: promotion is binary. STEAL: `stage` (`shadow`/`canary`/
  `full`) with distinct statuses, and `demote()` restoring the previous
  champion (rollback recorded).
- No approval flow. STEAL: `request_approval()` → `approve()`/`reject()`,
  approver + timestamp recorded.
- No head-to-head comparison artifact. STEAL: `compare(a, b)` → margin,
  winner, rendered card.
- No model card. STEAL: `model_card(run)` → markdown with lineage,
  dataset hash, metrics, gate history, limitations.

## 4. PII scrubbing (`collect.py`) — scrub_pii / scrub_example

**The best:** Microsoft Presidio — analyzer (NER via spaCy/transformers +
regex + contextual scoring) separate from anonymizer (replace/hash/encrypt
operators); allow-lists for known-safe tokens; score thresholds to control
false positives; the explicit vendor disclaimer that detection is a
*trip-wire, not a gate* (tune for recall). Production guides wire redaction at
*every* stage (input, retrieved chunks, output), and log stats but never raw
values.

**Gaps:**
- Regex-only misses names (PERSON entities) and locale-specific IDs; the
  owner is Nigerian — **NIN/BVN (11-digit national IDs), Nigerian 10-digit
  account numbers, and +234 phones** are not covered by the generic patterns.
- No Presidio integration when it IS installed (phone can't install spaCy,
  but a workstation can). STEAL: optional `presidio_scrub()` — use it when
  importable, fall back to regex; never a hard dependency.
- No allow-list, no operator choice (replace vs. deterministic hash for
  joinable pseudonyms), no per-entity redaction report.
  STEAL: `allow=()` param, `mode="replace"|"hash"`, `scrub_report()`.

## 5. Data quality (`preprocess.py`, `collect.ConversationMiner`) — quality_filter / _quality_score

**The field:** DEITA (complexity × quality scores, diversity via embedding
cosine filter), AlpaGasus (LLM-judge filtering), IFD (instruction-following
difficulty under the student model), Self-Instruct's load-bearing **diversity
filter** (ROUGE-L < 0.7 against everything kept — without it, synthetic data
mode-collapses), RefinedWeb/C4 heuristics (symbol-to-word ratio, terminal
punctuation, line lengths).

**Gaps:**
- `dedupe()` is simhash-only. RedPajama's pipeline is **exact-first (Bloom
  filter) then fuzzy**; exact is cheaper and catches the byte-identical rows
  that dominate real corpora. STEAL: exact sha256 pre-pass inside `dedupe()`.
- `quality_filter` lacks the RefinedWeb-style signals: alphabetic ratio,
  terminal punctuation, symbol density. STEAL: `strict=True` mode adding
  them.
- `_quality_score` is heuristic-only. DEITA/AlpaGasus show a judge hook is
  the real upgrade. STEAL: optional `judge=` callable blended into the
  score, plus the Self-Instruct trigram-overlap diversity filter in the
  miner (prevents the "same question 40 ways" slice).
- No train/eval **leakage check beyond the split itself**: SlimPajama's
  global-dedup finding matters — dedupe eval against train. STEAL:
  `check_leakage(train, eval)` reporting near-dupes across the boundary.

## 6. Tokenizer (`tokenize.py`) — BPETokenizer

**The field:** Karpathy's minbpe (byte-level BPE, GPT-2 regex pre-tokenizer);
the HF `tokenizers` crate (Rust). Notable: minbpe's special-token handling
and the observation that merge quality is all about the pre-tokenizer.

**Verdict:** this module's BPE is already honest and complete. Gaps are
small: no `encode_batch`, no corpus stats (`token_counts`/`compression`
report to size `max_seq_len` honestly), no `trim_vocab`. STEAL: batch encode
+ a `corpus_stats()` helper (median/mean tokens per row — feeds SEQ_LEN
honestly instead of guessing 512).

## 7. Checkpoints (`checkpoints.py`) — validate_checkpoint / pick_checkpoint

**The field:** HF Trainer's `save_total_limit` retention + DeepSpeed's
corruption-tolerant resume; the consistent lesson is that resume logic must
validate *before* trusting, which this module already does well.

**Gaps:** no retention/pruning (free-tier disks fill up — the generated
scripts save every 500 steps with no cap), no integrity manifest (sha of the
payload). STEAL: `prune_checkpoints(out_dir, keep=3)` (oldest valid deleted,
corpses always deleted) + `checkpoint_manifest()` with sizes and sha256.

## 8. Fine-tune mix + Colab artifacts (`finetune.py`)

**The field — QLoRA recipe consensus** (QLoRA ablations + 50-config
benchmark corpora): rank 16–32 (alpha ≈ 2× rank, NOT r/4), dropout 0.05,
LR ~2e-4 with 5–10% warmup + cosine, all-linear target modules, NF4
double-quant + paged AdamW-8bit, **NEFTune** (`neftune_noise_alpha=5`,
up to +10–25% instruction-following on AlpacaEval/MT-Bench), **sample
packing** (`packing=True`, big efficiency win on short rows),
**assistant-only loss** (`assistant_only_loss=True` — don't train the model
to predict user turns), eval on a real held-out split, early stopping.

**Gaps in the generated artifacts:**
- The Colab script's `SFTConfig` has no NEFTune, no packing, no
  assistant-only loss, no warmup — it scores full conversations including
  user turns. The notebook likewise. STEAL: add all four to both, plus a
  shared `QLORA_RECIPE` constant documenting the vetted defaults.
- `build_persona_mix` is solid; missing is a `mix_report()` human summary
  (the manifest is machine-shaped). Minor: add a readable summary.
- `generate_persona_samples` uses only the active model; mining says
  Evol-Instruct (complexity evolution: add constraints, deepen, concretize,
  reasoning steps) beats plain self-distillation for quality. STEAL:
  `evolve` mode in the generator with the five Evol-Instruct operations.

## 9. Evaluation (`evaluate.py`) — golden battery / judge

**The field:** MT-Bench (multi-turn, 8 categories, judge with
position-swapped calibration for position bias), AlpacaEval (pairwise vs.
reference, **length-controlled** win-rate — verbosity bias is real),
the MT-Bench paper's three judge modes: pairwise (most reliable),
single-answer grading, reference-guided grading; G-Eval (rubric-driven
pointwise); the pairwise-vs-baseline protocol as the promotion-grade
comparison.

**Gaps:**
- `judge_golden_set` is pointwise single-answer only — pairwise is the more
  reliable protocol and it maps directly onto the promotion gate
  (challenger vs. champion). STEAL: `pairwise_judge()` with A/B position
  swap + tie handling, reporting win-rate, length delta (verbosity-bias
  check), and per-category breakdown.
- No category taxonomy on cases. STEAL: `category` field on GoldenCase
  (writing/roleplay/reasoning/math/code/persona/language) + category
  rollup in the report.
- No multi-turn cases. STEAL: follow-up coherence cases.
- No rendered report. STEAL: `render_eval_report()` — one card with loss,
  golden, judge, pairwise, and the gate recommendation.

## 10. Backends (`backends/`) — unsloth / axolotl / mlx / llama_factory

**The field:** unsloth docs (fused CE, `use_gradient_checkpointing="unsloth"`,
`packing=True` supported, `assistant_only_loss` via trl); axolotl example
configs (sample_packing, NEFTune via `neftune_noise_alpha`); mlx-lm LoRA
(train with `--iters`, `--steps-per-eval`, adapter save); LLaMA-Factory
(its `train_on_assistant_only` / template handling).

**Gaps:**
- The unsloth backend hardcodes `packing=False` and full-conversation loss.
  STEAL: `packing=True` + `assistant_only_loss=True` + NEFTune when the trl
  version accepts it (probe like the existing liger probe).
- mlx backend: check `build_mlx_args` for eval cadence (`--steps-per-eval`,
  `--save-every`) and test-after-train eval — verify in code.
- All backends: no `dry_run()` that validates the config without training
  (catches bad base_model ids on the phone before a Colab session burns).

## 11. Free datasets (`free_datasets.py`) — catalog / fetch

**The field:** HF Datasets + dataset cards; the useful pattern is
task → dataset recommendation with license + size + verified date.

**Gaps:** the catalog is a flat list — no `recommend(task)` ("agent
tool-use", "yoruba", "code", "preference"), no text search, no
freshness signal. STEAL: `search()` + `recommend()` over the catalog.

## 12. Quick demo (`quick.py`) — run_quick_training

**The field:** the best CLIs (gh, railway) render one rich status card:
what ran, what it cost, what changed, what to do next. This module returns
a dataclass with a note.

**Gaps:** no styled presentation, no sample generation from the trained
model (the demo trains a model and never lets you *hear* it). STEAL: a
`render_run_card()` via the new style layer + a generated sample line in
the result.

## 13. Presentation (cross-cutting)

Nothing in the module has a presentation layer: registry stats, policy
decisions, eval results, and gate outcomes are all dicts. The god-tier bar
demands a **style layer**: themed cards (box-drawing, color-optional),
sparklines for loss histories, progress bars, tables. STEAL: a new
`nomorals/training/style.py` — `Theme`, `card()`, `table()`, `sparkline()`,
`progress_bar()`, `render_run_card()`, `render_gate_report()`. Zero
dependencies, color auto-disabled when not a TTY.

## What NOT to steal
- Presidio as a hard dependency (phone can't carry spaCy) — optional only.
- Full autograd engine (the analytic-gradients design is deliberate and
  faster to trust; micrograd-style graphs would slow the phone path).
- Paid APIs for judging — the judge stays injected/callable.
- Cross-module changes: this sweep touches only `nomorals/training/`,
  `tests/test_training_sweep.py`, and this report.

## Implementation order
1. `style.py` (new) — the presentation spine everything else uses.
2. `dataset.py` — real `write_format_bundles(base, examples)` (current one
   is a stub with the wrong signature vs. its caller).
3. `trainer.py` — AdamW, cosine+warmup, grad clip, early stopping +
   best-restore, `generate()`, richer metrics.
4. `preprocess.py` — exact-first dedupe, strict quality signals,
   `check_leakage`.
5. `collect.py` — Presidio-optional scrub, Nigerian IDs, allow-list,
   hash mode, scrub report; miner: trigram diversity filter + judge hook.
6. `policy.py` — PSI drift signal.
7. `registry.py` — min_gain, staged promotion, demote/rollback, approval
   flow, compare(), model_card().
8. `evaluate.py` — pairwise_judge with position swap, categories,
   multi-turn cases, render_eval_report().
9. `finetune.py` — NEFTune/packing/assistant-only-loss/warmup in both
   generated artifacts, QLORA_RECIPE, Evol-Instruct evolve mode.
10. `backends/unsloth.py` — packing + assistant-only loss + NEFTune probe.
11. `checkpoints.py` — prune + manifest.
12. `free_datasets.py` — search + recommend.
13. `tokenize.py` — encode_batch + corpus_stats.
14. `quick.py` — run card + sample generation.
15. `__init__.py` exports; `tests/test_training_sweep.py`; commit + push.
