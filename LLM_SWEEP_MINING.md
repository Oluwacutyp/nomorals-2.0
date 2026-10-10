# LLM module sweep — external mining report

Date: 2026-10-10. Every significant class in `nomorals/llm/` was compared
against best-in-class external implementations (and a few trash ones, which
still held ideas). Sources are cited per class. Gold adopted in this sweep
is marked with →.

## 1. `LLMRouter` (router.py)

**Mined:** LiteLLM router (weighted routing, RPM/TPM load balancing, cooldowns,
`cooldown_time`, per-virtual-key budgets, fallbacks), Bifrost Go gateway
(microsecond overhead, adaptive load balancer, **per-provider key pooling**,
circuit breaker), nexus-llm-router (GitHub, Francis1998 — `circuit-breaker-half-open-probe`
with probe budget=2, `latency-slo-shed` shedding providers whose rolling p95
exceeds `NEXUS_LATENCY_SLO_MS`, `adaptive-timeout` risk-adjusted p95 timeouts,
`complexity-tier` quality-for-cost ladder, `cascade` cheapest-first escalation,
`round-robin` with stable `request_id` hash), free-best-router (discover →
normalize → health-probe → score → rank → cooldown → **explore under-sampled**),
Portkey (`single`/`loadbalance`/`fallback` composable, nesting loadbalance of
fallback chains, `on_status_codes` limiting which errors trigger failover),
OpenRouter provider routing (price-weighted load balancing weight ∝ 1/price²,
`sort: price|throughput|latency`, `order`, `only`/`ignore`, `allow_fallbacks`,
`max_price`, p50–p99 thresholds, `require_parameters`), rinbarpen/llm-router
(tag-based routing, 3-level priority, circuit breaker + channel fallback),
laravel-llm-router (tenant-aware "sovereign" routing — force on-prem providers).

**Gold adopted →**
- *Latency SLO shedding*: skip providers whose rolling p95 exceeds a SLO when
  faster alternatives exist (nexus `latency-slo-shed`). → `latency_slo_ms` on
  `LLMRouter`; rolling p50/p95 on `ProviderHealth`.
- *Half-open probe budget*: cap concurrent probes into recovering providers at
  2 (nexus). → `half_open_probe_budget`.
- *Per-request routing knobs*: `route={"sort": "latency"|"price"|"throughput",
  "only": [...], "ignore": [...], "max_price": x, "allow_fallbacks": bool}`
  on chat/complete (OpenRouter).
- *Key pooling*: Bifrost retries a 429 with a **different key from the same
  provider** before failing over. → `api_keys` pool + `rotate_key()` on
  `OpenAICompatProvider`; router rotates on rate-limit before moving down the
  chain.
- *Terminal errors never fail over*: Portkey `on_status_codes`; our
  `failures.RECOVERY` already classifies → new `should_failover()` /
  `is_retryable()` thin wrappers expose the table as the single source of
  truth. Deliberate decision: AUTH/CONFIG keep `failover=True` in the
  table (each provider carries its own credentials — a dead Groq key
  must not kill the HF fallback); BUDGET is the new `failover=False`
  class. The old code's substring checks are gone in favor of the table.
- *Route trace on every response*: Langfuse-style per-attempt trace
  (provider, latency, error) → `LLMResponse.route_trace`, `cost_usd`.
- *Budgets*: gateways enforce budgets at 4 levels; → `daily_budget_usd` on the
  router with 80% bus warning (never raises; returns a typed budget error
  response like every other router failure).
- *Retries belong to one layer* (heaven999b lessons): no new retry layer —
  the router stays the sole retry owner; provider-level `retry_call` stays
  small and unchanged.

**Weak spots noted (not adopted):** tenant isolation has no caller yet;
RPM/TPM load balancing needs multi-key accounting not present — key pooling
covers the immediate need.

## 2. `ModelBroker` (broker.py)

**Mined:** OpenRouter `openrouter/auto` + `route: fallback` (best-model pick
server-side), OpenRouter maximizer skill (`best_of_n` — generate 5, return
best; provider-optimize per cost/speed/quality), rinbarpen tag routing
(`coding`, `vision`, `high-quality`, `fast`, `reasoning` tags), nexus
`complexity-tier` (cheapest model meeting a quality target — catalog-adaptive,
no thresholds to tune) and `cascade` (cheapest first, escalate one rung at a
time on failure), linkedin cost-per-task analysis (measure only **successful**
runs, sort by difficulty; per-token pricing hides the truth).

**Gold adopted →**
- *Tag-based routing*: `ModelCard.tags` + `broker.select_by_tags({"code","fast"})`
  (rinbarpen). Cards auto-derive tags from capabilities/owner/local.
- *Escalation ladder*: `broker.escalation_chain(capability)` returns
  cheapest→best ordered cards — the `cascade` pattern; `Brain.complete`
  can walk it.
- *Complexity-tier selection*: `select_for_quality(quality 0..1)` picks the
  cheapest card whose quality score meets the target (nexus); quality comes
  from live benchmark scores with card-level prior.
- *Explainable selection*: `broker.explain(card, capability)` + `format_ranking`
  god-tier text table — shows score components per candidate, not just the
  winner (operators distrust black-box picks).
- *Cost-per-task scoring*: `BenchmarkDB.cost_per_task(model_id, task_kind)` —
  total cost / successful tasks (failed runs excluded, per the linkedin
  analysis); `leaderboard(task_kind)` ranks models by success rate then cost.

## 3. `Brain` (brain.py)

**Mined:** cost-tracking MLOps guide (query classifier: easy→cheap, hard→
expensive — `route()` by complexity), OpenRouter maximizer `best_of_n`.

**Gold adopted →**
- `Brain.best_of_n(prompt, n, task_kind)` — fan out across n chain providers,
  adjudicate the winner (OpenRouter maximizer's headline feature, native).
- `quality=` kwarg on `complete`/`chat` → broker `select_for_quality`
  (complexity-tier, cheapest model meeting the bar).
- `_estimate_complexity(prompt)` heuristic (length, code fences, question
  words, multi-part) → auto quality suggestion logged, not forced.
- `Brain.escalate(...)` walks the broker escalation chain cheapest-first,
  returning the first success with the attempts annotated.

## 4. `adjudicate` / `Judge` (adjudicate.py)

**Mined:** LLM-as-a-Judge literature — Wikipedia survey (position bias ← swap
order, count win only if consistent both ways; length effects ← AlpacaEval
2.0 length-controlled win rate; pairwise > pointwise; reference answers with
explicit rubrics; multi-judge majority across families; CoT before verdict),
galileo.ai (T=0.01 → consistency ~1.0; binary verdicts for gating; 3–5
few-shot examples from real disagreements), medium practitioner checklist
(different-family judge, rubric like a spec, human calibration set).

**Gold adopted →**
- *Position-debiased pairwise*: `Judge.pairwise(q, a, b, reference="")` runs
  both orders, counts a win only when both agree, else tie (Wikipedia).
- *Length-controlled*: `Judgment` records `candidate_lengths` + length delta;
  `length_controlled=True` mode asks the judge to ignore verbosity.
- *Multi-judge panel*: `Judge.panel(question, candidates, asks)` — majority
  vote across judge callables (different families), disagreement surfaced as
  low confidence (galileo).
- *Reference-guided rubric*: `reference` + structured dimensions prompt;
  CoT **before** the verdict; judge temperature 0.1 → **0.0** (galileo).
- *Richer verdict*: `Judgment` gains `scores` (per-dimension),
  `confidence`, `method`, `ties`; `format_judgment()` god-tier card.

## 5. `CostLog` / `estimate_cost` / `cost_display` (router.py, cost_display.py)

**Mined:** Langfuse (cost inferred from model defs, **ingested values take
priority**, records reasoning + cached tokens, per-generation cost),
Helicone (`Helicone-Property-<Name>` headers → segment cost by property),
devops-daily (check **provider-reported usage incl. reasoning/cached tokens**,
not text estimates; billed-to-read ratios 14–50x), Portkey (budget limits in
front of the provider, metadata tags), mrsameerkhan MLOps (token-level JSON:
request_id, feature, input/output tokens, latency; aggregate per feature).

**Gold adopted →**
- `CostLog.record` gains `cached_tokens`, `reasoning_tokens`, `feature`,
  `request_id` fields (Langfuse/Helicone shape).
- `CostLog.breakdown(since)` → per-operation/per-provider aggregates
  (calls, tokens, cost, avg latency) + `budget_report(daily_budget)` with
  50/80/100% alert levels.
- `estimate_cost` prices cached input at 0.1× (Anthropic convention) and
  accepts reasoning tokens; ingested explicit `cost_usd` always wins.
- `cost_display.format_cost_table(breakdown)` god-tier table +
  `budget_alert_line(spend, budget)`.

## 6. `context_fit` strategies (context_fit.py)

**Mined:** Anthropic CCA chapter-16 (case-facts block — facts that cannot be
lost sent unmodified every turn; **Lost in the Middle** — refs at top,
question/instructions at bottom, +30% quality; XML structured tags; trim
verbose tool results before they enter context), X19 context-compression
(system + rolling 3-message cache breakpoints, 5m/1h TTL; stable system
prompt; model change invalidates cache), Claude Code context research
(compaction is lossy — externalize claims/citations to disk; progressive
summarization strips numbers/dates).

**Gold adopted →**
- `CaseFacts` strategy: pinned facts rendered verbatim at the top of every
  fit, immune to summarization (Anthropic case-facts block).
- `PrioritizeEnds` strategy: longest reference message → top (after system),
  last user message → bottom (Lost in the Middle).
- `trim_tool_results(messages, max_chars)` helper: cap verbose tool outputs
  before fitting.
- `with_cache_breakpoints(messages, window=3)` → provider-agnostic cache
  breakpoint markers (system + rolling window, X19 pattern).
- `fit_messages(..., pinned_facts=[...])` threads facts through every chain;
  `FitResult` reports `facts_kept`.

## 7. `HuggingFaceDownloader` (download.py)

**Mined:** huggingface_hub docs (`snapshot_download` with
`allow_patterns`/`ignore_patterns`, `resume_download`, etag verification,
token auth), gguf-org/trainer materials docs (resumable by default, detached
jobs with pid/log/status files, progress = bytes-on-disk vs Hub sizes,
statuses missing/partial/downloading/ready), opendawcpp model-management
SKILL (validate before rename: size ±5%, `GGUF` magic header), hfd gist
(single-line progress: % complete, files/total, GB, MB/s, ETA; aria2c
multi-thread).

**Gold adopted →**
- Prefer `huggingface_hub.snapshot_download` when installed (resume,
  etag, patterns, token) with graceful fallback to the existing HTTP path.
- `on_progress(downloaded, total)` callback + `format_progress()` one-line
  status (%, files, GB, MB/s, ETA — hfd style).
- Post-download validation: GGUF magic header check for `.gguf` files
  (opendawcpp), size sanity vs expected.
- `include`/`exclude` glob patterns on download.

## 8. `messages_to_text` / `Message` (base.py)

**Mined:** HF chat templating docs + mistral.rs + windere/hifp8 (never
hand-roll: use the model's native chat template; ChatML default, Llama-3
`<|start_header_id|>`, Mistral `[INST]`, Gemma, Qwen; wrong template costs
10–30% benchmark accuracy; template env carries `tools`, `enable_thinking`,
`date_string`; `add_generation_prompt=True` for inference).

**Gold adopted →**
- Native template registry: chatml, llama3, llama2, mistral, gemma, qwen,
  deepseek, phi3, vicuna, zephyr, openchat, alpaca.
- `detect_template(model_id)` — family detection by id fragments.
- `messages_to_text(..., template="auto", model="")` — auto picks the native
  template; explicit names still work; unknown → chatml fallback (unchanged
  default behaviour).

## 9. `prompts` (prompts.py)

**Mined:** Langfuse prompt management (versioned prompts pulled at runtime;
prompt change = dashboard op, not deploy), LLM-as-judge rubrics (rubric like
a spec; reference-guided).

**Gold adopted →**
- `PromptLibrary`: versioned registry — `register/get/history`, JSON-file
  persistence, built-ins seeded as v1. Partial rendering preserved.
- `rubric_prompt(dimensions, reference="")` — reference-guided judge rubric
  builder with CoT-before-verdict instruction.
- Judge system prompt upgraded to the reference-guided rubric form
  (temperature now 0.0 in adjudicate.py).

## 10. `failures` (failures.py)

**Mined:** tenacity (standard: declarative, async, jitter built in), AWS
full-jitter formula `delay = random(0, min(base*2^attempt, max))`,
thedailybyte retry guide (retry at **one** layer; never retry 400; honor
Retry-After; pair retries inside circuit breakers), tenaz/tiny-retry
(total_timeout wall-clock cap, `RetryExhausted.last_exception`).

**Gold adopted →**
- `backoff_delay(attempt, base=1.0, cap=60.0, jitter="full"|"equal"|"none")` —
  full jitter default (AWS).
- `is_retryable(failure_class)` / `should_failover(failure_class)` thin
  wrappers over the existing `RECOVERY` table (single source of truth).
- `RetryBudget(deadline_s)` — wall-clock budget helper with `remaining` and
  `exhausted`; `retry_after_s(error)` honoring server hints.
- No new retry layer: router remains the sole retry owner (per the
  "one layer" lesson).

## 11. `ModelCard` / capabilities (capabilities.py)

**Mined:** LiteLLM model cost map, OpenRouter `/models` (pricing, context,
`supported_parameters`).

**Gold adopted →**
- `ModelCard` gains `tags: frozenset[str]` (auto-derived:
  fast/local/owner/code/vision/long-context), `price_in`/`price_out` per 1M
  (OpenRouter shape), `quality: float` prior, `throughput_tps`.
- `cost_per_1k` kept (backward compat); `estimated_call_cost(prompt_toks,
  completion_toks)` helper.
- `BrokerConstraints` gains `min_quality`, `tags_any`, `max_price_per_1m`.

## 12. `BenchmarkDB` (benchmarks.py)

**Mined:** lm-evaluation-harness task conventions, Open LLM Leaderboard
ranking, cost-per-task analysis (linkedin).

**Gold adopted →**
- `cost_per_task(model_id, task_kind)` — cost of successful runs only.
- `leaderboard(task_kind, limit)` — success rate → score → cost ordering.
- Fixed a real bug: `benchmark_model` had **two sequential `return`
  statements** (dead second return with the richer dict) — merged into one.

## 13. Providers

**Mined:** OpenRouter provider-routing docs (`provider: {order, only,
ignore, sort, allow_fallbacks, max_price, require_parameters}`),
Anthropic prompt-caching docs.

**Gold adopted →**
- `OpenRouterProvider.chat(..., provider_prefs={...})` passes OpenRouter's
  native `provider` routing object through (sort/order/only/ignore/
  allow_fallbacks/max_price).
- `OpenAICompatProvider` gains `api_keys` pool + `rotate_key()` (Bifrost key
  pooling); router rotates on 429 before failing over.
- `defaults.ProviderSpec` gains `api_key_pool`; `specs_from_env` reads
  `<PREFIX>_API_KEYS` (comma-separated).

## 14. `registry` (registry.py)

**Mined:** OpenRouter `/models` catalog shape.

**Gold adopted →**
- `ModelRecord` gains `price_in`/`price_out`, `quality`;
  `ModelRegistry.cheapest_for(capability)` + `best_quality_for(capability)`.

## 15. Deliberately untouched

- `local_server.py` (GGUFServerManager — 855 lines, recently reworked; server
  flags/health are already thorough), `lifecycle.py`, `learning.py`
  (trajectory schema owned by cognition), `benchmark.py` (CLI runner),
  `model_catalog.py`, `power.py`, `providers/{anthropic,deepseek,gemini,groq,
  hf_serverless,llama_cpp,mock,ocr,ollama}` — no gaps found that beat
  existing capability; the OpenAI-compat spine carries the new features.
