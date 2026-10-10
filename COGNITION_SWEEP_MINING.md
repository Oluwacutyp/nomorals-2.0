# Cognition Sweep — External Mining

Date: 2026-10-10. Module: `nomorals/cognition/` (trajectory learning, failure clustering, representation-quality ledger).
Mined per class: TrajectoryStore (execution-outcome learning + routing), FailureKB (failure triage), RepresentationLedger (autonomous-action quality + distillation).
Rule followed: mine every real implementation first, even weak ones — cite real sources, no invented techniques.

---

## 1. Sentry — error grouping / fingerprinting (gold for failure clustering + normalization)

**Source:** Sentry docs, "Fingerprinting Rules" — https://github.com/getsentry/sentry/blob/master/src/sentry/grouping/fingerprinting/__init__.py (grammar) and https://docs.sentry.io product docs (mirrored in the sentry-docs forks above); practitioner writeup of a Sentry-like rebuild hashing the top-5 stack frames (filename:function:lineno) with SHA-256 to turn thousands of raw crashes into issues ("Before building this, I assumed grouping errors was some sophisticated ML thing. It's mostly just a hash of the stack trace. Simple, deterministic, and it works." — https://medium.com/@odetokuntreasure6/how-i-built-a-self-hostable-error-tracking-tool-like-sentry-5e482471385b).

**What Sentry does better than our `normalize_error`:**
- Default grouping = exception **type** + **normalized message** + **stack trace**. Our module only keeps the first message line and throws the class away. Sentry treats exception class as a first-class grouping axis.
- **Custom fingerprint rules** (`error.type:DatabaseUnavailable -> system-down`, glob syntax, first-match-wins) let operators merge/split groups deliberately. Our module has zero operator overrides.
- **Issue lifecycle states**: Sentry issues are new / resolved / regressed / escalating. A resolved issue that recurs becomes *Regressed* (issue states: https://docs.sentry.io/product/issues/states-triage/). Our FailureKB has no resolution state at all — a fixed bug keeps shouting forever on the "fix me" list.
- **Escalating issues**: abnormal volume growth detection, not raw count ranking. Our `repeated()` sorts by raw count only.

**Adopted:**
- Split signatures into `exception_class` + normalized message (Sentry's two axes).
- Per-store **fingerprint rules table** (operator overrides, first-match-wins, Sentry-style) → `add_fingerprint_rule` / applied in a new `fingerprint_for()`.
- Cluster `first_seen`, per-cluster **resolution workflow** (`resolve`/`reopen`), and **regressed** detection (failures seen after resolution) in FailureKB.
- **Priority scoring** for the fix-me list: count × recency decay (×1.0 + blast-radius factor), replacing raw-count sort. "Escalating" (recent growth) weighting folded into `triage()`.

## 2. Bandit-based LLM routers — UCB1, Thompson Sampling (gold for TrajectoryStore selection)

**Sources:**
- OrcaRouter (arxiv.org/html/2605.30736v1): production LLM router using **LinUCB** per-arm Gram matrices + **Linear Thompson Sampling** (sample θ~N(θ̂, v²A⁻¹), pick argmax) + ε-greedy + **round-robin warmup + UCB** (RR+UCB) to prevent single-arm collapse under sparse feedback.
- Industry survey (dev.to/alex_aslam, Oct 2026): **L7**, a Bayesian router using **Thompson Sampling over per-model Beta distributions**, demonstrated a **68.5% aggregate inference-cost reduction** vs static round-robin across 10,000 heterogeneous tasks — **zero training data**, uniform prior + online updates. MTRouter (joint history-model embeddings + outcome estimator, −58.7% cost on ScienceWorld), Planner-as-Router (tier assignment at plan time, −44% cost), RouterHGC (heterogeneous graph, +0.8–6.2% accuracy / −27.4% cost).
- UCB1 (Auer et al. 2002, per Kaufmann's bandit tutorial): UCBₐ(t) = μ̂ₐ + √(α·ln(t) / (2Nₐ)), α>2, logarithmic regret.

**What they do better than our `rank()` (mean + name tiebreak only):**
- Our store records **cost and latency but never uses them**. Every production router scores on cost-efficiency; we ignore the fields we collect.
- No **exploration**: unknown candidates sit at the 0.5 prior and sort after measured ones — a new, better model is never tried. Bandit policies (TS, UCB1, RR+UCB) solve exactly this.
- Recency decay is present (half-life) but there is no uncertainty accounting.

**Adopted:**
- New `recommend()` with strategies: `mean` (legacy behavior), `thompson` (Beta posterior per candidate over decay-weighted successes/failures, seeded RNG), `ucb1` (Auer index; unplayed arms score +∞ = forced exploration, the RR+UCB idea collapsed into one method), `cost` (Thompson sampling on efficiency = success-per-cost).
- `rank(..., objective=...)`: `success` (unchanged), `efficiency` (success rate per unit cost), `speed` (rate penalized by mean latency). No fabricated units — all relative scores.

## 3. Wilson score interval (gold for sparse success-rate honesty)

**Sources:** Evan Miller, "How Not To Sort By Average Rating" (http://www.evanmiller.org/how-not-to-sort-by-average-rating.html) — lower bound of Wilson score confidence interval as the ranking statistic; canonical implementations (msn0/wilson-score-interval npm; multiple gists). Formula: W(p,n) = (p + z²/2n − z√(p(1−p)/n + z²/4n²)) / (1 + z²/n), z=1.96 for 95%.

**Why it matters here:** our `success_rate` returns a raw weighted mean; 1 success in 1 trial ranks above 80/100. Wilson's lower bound is the industry-standard fix (used by Reddit comment sorting and friends). With exponential-decay weighting, exact Wilson is approximate — we use **effective sample size** n_eff = (Σw)²/Σw² as the n.

**Adopted:** exported `wilson_lower(successes, n, z=1.96)`; `slice_stats()` returns wilson bounds per slice; `recommend(..., strategy="wilson")` sorts by lower bound.

## 4. Reflexion — verbal reinforcement learning (gold for RepresentationLedger lessons)

**Sources:** Shinn et al., NeurIPS 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning" (arxiv 2303.11366, https://github.com/noahshinn024/reflexion): agent attempts → evaluator scores → **self-reflection model turns sparse feedback + trajectory into natural-language critique** → critique stored in an **episodic memory buffer** → prepended to actor's context on retry. 91% pass@1 on HumanEval (vs 80% GPT-4 baseline). Architecture triad: **Actor / Evaluator / Self-Reflection**; the only thing that changes between trials is the textual memory.

**What it does better than our ledger:** we store counterfactuals ("better action + why") but produce **nothing consumable** — the module docstring even names "the feed for skill distillation (#2)" and there is no such feed. Reflexion's whole point is that the *reflection text* is the learning artifact.

**Adopted:** `RepresentationLedger.distill()` — assembles per-action-type **lesson cards** (weak pattern → owner feedback snippets → counterfactuals → prompt-ready "when you do X, do Y instead" guidance) for injection into the brain's context. Plus `trend()` (recent-window vs prior-window score delta — Reflexion-style trajectory of improvement) and `inconsistencies()` (evaluator mismatch: success+downvote / failed+upvote — Reflexion's evaluator axis).

## 5. Voyager — skill library + automatic curriculum (gold for FailureKB lessons → skills)

**Sources:** Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (NeurIPS 2023, arxiv 2305.16291): automatic curriculum + **ever-growing skill library of verified executable code indexed by description embeddings** + iterative self-verification. "Skills (programs), not actions, as the unit of learning"; verified skill stored forever, retrieved top-k by embedding; 3.3× more unique items, tech-tree milestones 15.3× faster.

**Adopted:** `FailureKB.lessons()` returns Voyager-style **verified lesson cards** (cluster signature → evidence counts → operator notes → distillable guidance text), designed to be indexed by the skill system. Kept dependency-free (no embeddings here — cards carry the text; retrieval lives with the consumer), per the user's offline-capable build rule.

## 6. Letta / MemGPT — memory consolidation + portable artifacts (gold for export/retention)

**Sources:** MemGPT (UC Berkeley, now Letta): OS-like layered memory (core blocks / recall / archival), agent self-edits its memory blocks, **sleep-time agents** for asynchronous consolidation (rglaubitz/project-apex assessment; Letta release notes Dec 2025: skill learning from past experience). Competitor analysis (dwamianm/prism): "Portable artifact — copyable memory_pack/" as a differentiator; lifecycle Tentative → Stable → Superseded → Archived.

**Adopted:** `export_json()` / `import_json()` on both TrajectoryStore and RepresentationLedger (portable artifact pattern); failure-cluster resolution lifecycle (new → resolved → regressed) mirrors the tentative/stable/superseded lifecycle.

---

## Gap table → what shipped

| Class | Gap (asked: "what SHOULD this class have?") | External gold | Shipped |
|---|---|---|---|
| TrajectoryStore | cost/latency recorded but never used | MTRouter/L7 cost-aware routing | `rank(objective="efficiency"/"speed")`, `recommend(strategy="cost")` |
| TrajectoryStore | no exploration; 0.5 prior sorts new candidates after measured ones | UCB1 (Auer 2002), Thompson Sampling Beta (L7) | `recommend(strategy="thompson"/"ucb1")` |
| TrajectoryStore | raw mean dishonest for sparse data | Wilson lower bound (Evan Miller) | `wilson_lower()`, `slice_stats()`, `recommend(strategy="wilson")` |
| TrajectoryStore | no trend / no "getting worse?" | Sentry escalating issues | `trend()` (recent vs prior window) |
| TrajectoryStore | normalization = first line only, no class axis, no operator control | Sentry fingerprinting rules | `exception_class` split, `add_fingerprint_rule()` + `fingerprint_for()` |
| TrajectoryStore | cluster dicts lack first_seen / blast radius | Sentry issue fields | `first_seen`, `models_affected`, `task_kinds` in clusters |
| TrajectoryStore | no portability | Letta portable artifact | `export_json`/`import_json` |
| FailureKB | fix-me list = raw count sort, no states | Sentry issue lifecycle | `triage()` (priority score), `resolve()`/`reopen()`, `regressed()` |
| FailureKB | notes are write-only; nothing feeds the brain | Reflexion/Voyager lesson cards | `lessons()` → prompt-ready guidance cards |
| FailureKB | no note search/dedup | basic hygiene | `search_notes()`, duplicate-note dedup |
| FailureKB | operator surface weak | — | `report()` markdown fix-me list |
| RepresentationLedger | `_score_value` formula convoluted | — | simplified, documented, same semantics |
| RepresentationLedger | counterfactuals stored, never distilled | Reflexion verbal-RL | `distill()` → per-type lesson cards for the brain |
| RepresentationLedger | no improvement trajectory | Reflexion trials | `trend()` improving/stable/declining |
| RepresentationLedger | evaluator feedback can contradict outcomes silently | Reflexion evaluator axis | `inconsistencies()` |
| RepresentationLedger | no portability | Letta portable artifact | `export_json`/`import_json` |
