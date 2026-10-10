# RESEARCH module — external mining report

Module: `nomorals/research/` (8 files: `__init__`, `pipeline`, `autonomy`,
`citations`, `grounded`, `grounded_store`, `scheduler`, `costs`).
Swept 2026-10-10. Every significant class was compared against best-in-class
implementations OUTSIDE the repo (mined best AND trash).

## 1. pipeline.research_deep / decompose / synthesize (the deep-research loop)

**Best — Open Deep Research** (`dzhng/deep-research`, the canonical <500 LoC
open implementation of OpenAI's Deep Research): iterative loop with explicit
**breadth/depth parameters**. Each iteration produces two things from the
sources: **learnings** (distilled facts) AND **follow-up directions** (new
questions driven by what's missing). Depth decrements; at depth 0 it writes
the markdown report. The Devon `_reasoning_pass` only generates follow-up
*queries* — no distilled learnings accumulate, so refinement iterations
re-read raw snippets instead of compressed insight. **Adopt: learnings +
directions as first-class per-iteration artifacts, breadth/depth params.**

**Best — langchain-ai/local-deep-researcher (IterDRAG pattern):** decompose →
retrieve for sub-query 1 → *answer the sub-query* → reflect → identify
knowledge gaps → new query → **iteratively update a running summary** with
each wave. Devon collects all findings then synthesizes once. The IterDRAG
running-summary keeps context small and gaps explicit. **Adopt: per-subquery
answers + iteratively-updated working summary in DeepReport.**

**Best — samjanjua6/ai-research-assistant (LangGraph):** draft synthesis →
**self-reflection & gap detection** → autonomous refinement until a quality
threshold is met → final output is an **executive summary (TL;DR) + full
cited report + PDF/Markdown export**. Devon returns one synthesis string.
**Adopt: structured Report (TL;DR, sections, tables) + markdown export.**

**Best — agenticaiengineer/research-agent guide (best-practices doc):**
dual-layer verification (upstream guardrails + downstream atomic statement
checking); **cross-validate every key finding against ≥2 independent
sources**; abstain mechanisms; max iteration limits + visited-URL tracking +
termination criteria; token budgets. Devon's `_verify_cited_sentences` is
the downstream layer but has no ≥2-source rule and no abstain when the
corpus contradicts itself. **Adopt: independent-source corroboration,
contradiction detection, abstain-on-conflict.**

**Trash mined:** dozens of "deep research" wrappers that are one search +
one LLM call with no iteration, no verification, no budget — exactly what
Devon already exceeds. Also mined: over-engineered supervisor-agent
hierarchies (dair-ai/m2-deep-research) that add orchestration cost without
better evidence — skipped; Devon's flat loop is the right shape.

## 2. pipeline.assess_worth / run_job / deliver (the scheduled pipeline)

**Best — beyondseo reputation rubric:** source quality scored on dimensions
(evidence captured, topic relevance, relationship/independence, placement
context) with *attributable* judgments and an explicit "unknown earns no
points" rule. Devon's `assess_worth` is keyword-hit counting — no
independence dimension (a press release and an independent review score
the same). **Adopt: independence/editorial signals in worth scoring.**

**Best — dev.to "No citation, no claim" RAG backend:** pre-generation
relevance gate (weak retrieval → honest "not found", never papered over);
citations as first-class response fields, not prose afterthoughts.
Devon's pipeline has the downstream verify pass but no upstream
relevance floor on the fetched set. **Adopt: retrieval-quality gate before
assessment.**

**Missing presentation:** `format_finding` is a fixed plain-text template.
Nothing in the module produces styled output. **Adopt: delivery themes
(markdown, TL;DR-first, verbose) — god-tier formatting.**

## 3. citations.CitationManager / extract_claims / verify_claims

**Best — bluxo1/axiom-rag "no citation, no claim":** verification is a
**hard gate that runs BEFORE confidence routing on every path**; a
citation to a chunk not in the retrieved set is treated as a bug; low
confidence = visible badge; every routing decision logged. Devon's
`CitationManager` has registration/quote-verification/deterministic Works
Cited/audit trail — the infrastructure is there — but nothing runs a
hard verification gate over the final synthesis, and there is no
cross-reference matrix (claim → supporting sources) or conflict flagging
when sources disagree. **Adopt: claim↔source matrix, contradiction
detection, verification summary on reports.**

**Trash mined:** citation libs that just renumber markers (formatting, not
verification) — Devon's existing hash+timestamp audit trail already beats
them.

## 4. grounded.GroundedSession (NotebookLM pattern)

**Best — the honest-RAG pattern (dev.to "How I stop an LLM from
hallucinating in production"):** two rules do the work — (1) the model is
NEVER allowed to recall a fact: every fact must come from verbatim tool
output; (2) **entity-match verification**: before a record enters the
grounded context it is re-verified against the queried entity — confidently
wrong *with a citation* is worse than no citation. Devon's grounded
session has neither: retrieved chunks enter the prompt unchecked, and no
confidence pass follows.

**Best — dev.to "RAG Isn't Magic" (fixes that worked):** larger chunks with
**hierarchy preserved** (document title + section headers in every chunk);
metadata-aware retrieval filters; forced citation format (80% drop in
made-up citations); second-LLM **confidence scoring** with a human-review
queue for low scores. Devon's chunks are raw 1200-char windows with no
section context, and answers carry no confidence signal. **Adopt: hierarchical
chunk headers, relevance-threshold short-circuit, confidence badge.**

**Bugs found while mining (verified against repo):**
- `pipeline._router_llm_fn` line ~1103: `brain_for(self.context)` inside a
  **module-level function** — `self` is undefined → NameError at call time.
  Any brain-router deep-research run crashes. Fix: `router.complete(...)`.
- `grounded.GroundedSession._complete` line ~427: `brain_for(self.context)` —
  `GroundedSession` has **no `.context` attribute** → AttributeError whenever
  `llm_fn` is None and a context router is passed. Fix: `brain_for(context)`.

## 5. autonomy.ResearchOrgan (watches / knowledge gaps / source quality)

**Best — MAIL framework (initiative_gap):** detect gaps → generate
hypothesis → validate → **integrate** → loop. Devon's gap loop detects and
validates but the **integrate** step is thin: an event is emitted to the
brain, nothing is stored back into memory as learned knowledge.

**Best — aetherlabsai/rsiagent:** broad-then-deep exploration with an
independent **verifier** agent whose feedback grounds learning; verifier
can't read the actor's private reasoning. Devon's `_reasoning_pass` is a
self-critique, not an independent verifier. **Adopt: independent
verification of synthesis against sources (the verifier can't just agree).**

**Recorded-but-unused:** `source_quality` scores are written on every
assessment but **never read** — no reordering, no deprioritization, no
weak-domain filtering of search results. The docstring promises
"deprioritized" learning; the code never does it. **Adopt: domain-aware
result reordering (evidence-learned, never a hardcoded allowlist) and a
quality report surfaced per run.**

**Best — qwen_ai_scientist prompts:** gaps carry novelty/value/feasibility
scores; the standing question "have I confused 'nobody tried it' with
'it's worth trying'?" Devon dedupes gaps by exact string only. **Adopt:
gap priority scoring + fuzzy dedup.**

## 6. scheduler.ResearchScheduler / grounded_store

Scheduler is conservative and correct (DB-backed last-run, no stampede,
no hot-loop). Missing: **digest batching** (one digest per day instead of
N deliveries), watch groups. GroundedStore lifecycle is solid (TTL, reaping,
rehydration); missing: session stats (docs, questions asked, citations
resolved) and **answer export** (markdown of a grounded Q&A thread).

## What this module SHOULD have that it doesn't (build list)

1. **Learnings + follow-up directions** per iteration (Open Deep Research
   pattern) — not just raw queries.
2. **Per-subquery answers + running summary** (IterDRAG) inside
   `research_deep`, surfaced in `DeepReport`.
3. **Structured report output**: TL;DR, sections, comparison tables,
   markdown export; **delivery themes** for `format_finding` (god-tier
   presentation).
4. **≥2 independent sources** corroboration for key claims; **contradiction
   detection** with abstain-on-conflict.
5. **Retrieval-quality gate** before assessment (don't paper over weak
   retrieval); **entity-match verification** for grounded context.
6. **Confidence scoring** on grounded answers with a visible badge.
7. **Hierarchical chunk headers** (doc title + section in every chunk).
8. **Claim↔source cross-reference matrix** + verification summary on
   `DeepReport`.
9. **Domain scores actually used**: reorder/deprioritize search results by
   learned domain quality (evidence-based, no hardcoding).
10. **Gap priority + fuzzy dedup**; integrate answered gaps back into
    memory (the MAIL integrate step).
11. **Digest batching** for scheduled deliveries.
12. Fix the two `self.context` crash bugs.

## Sources

- https://github.com/agenticaiengineer/research-agent/blob/HEAD/deep-research-agent-guide.md
- https://github.com/FlyingSnowLikeMist/deep-research
- https://github.com/langchain-ai/local-deep-researcher
- https://github.com/samjanjua6/ai-research-assistant
- https://github.com/dair-ai/m2-deep-research
- https://dev.to/yasantha_hettiarachchi_72/no-citation-no-claim-building-a-rag-backend-that-refuses-to-hallucinate-1klp
- https://dev.to/cavitnation/how-i-stop-an-llm-from-hallucinating-in-production-rag-entity-match-mcp-2ndj
- https://dev.to/zeroshotanu/llm-reasoning-why-models-hallucinate-and-how-to-reduce-it-2joo
- https://medium.com/@developer_programmer/rag-isnt-magic-why-our-ai-chatbot-hallucinated-anyway-5245533bad2d
- https://github.com/llmsresearch/llm-flashcards/blob/HEAD/src/content/cards/citation-attribution.mdx
- https://github.com/bluxo1/axiom-rag/blob/HEAD/docs/Prompt.md
- https://github.com/beyondtahir/beyondseo/blob/HEAD/references/reputation.md
- https://github.com/harshalworkai64254-bit/initiative_gap
- https://github.com/aetherlabsai/rsiagent
- https://github.com/sodium-oxide/qwen_ai_scientist/blob/HEAD/agent_prompts_v3.md
- https://github.com/mskazemi/idkmesh/blob/HEAD/docs/findings/2026-08-28-repository-driven-community-growth.md
