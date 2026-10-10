# INFRA MINING — Section E: Events + Error Systems

Mined 2026-10-09 before building. Sources: real-world event bus specs (PgEventBus reliability spec, Azure Service Bus, EventBridge patterns, Kafka/EDA skills), resilience libraries (Polly v8, resilience4j, opossum, pybreaker), Sentry's error-grouping/fingerprinting system, rate-limiting algorithm literature (GCRA, token bucket, sliding window variants). Best AND trash builds considered; gold taken, dirt left.

## 1. EVENT BUS

### Gold (take)
- **At-least-once + idempotency is the only honest guarantee.** "Exactly-once" is a myth; at-most-once loses events. The winning combo: producers stamp every event with an idempotency key (event id), consumers dedupe on it, brokers redeliver. (forge contract-event-bus SKILL, distributed-systems SKILL)
- **Dead-letter queue is part of the queue, not a later project.** Retry with backoff (~5x), then DLQ. Alert on *DLQ non-empty*, not on "queue failing". DLQ needs inspect / replay / reject tooling or it's a write-only graveyard. (yunusemrejr distributed-systems SKILL, Azure Service Bus practice)
- **Ordering: per-key, never global.** Partition by entity id; accept "parallel across keys". Anyone promising global order has a throughput problem, not a queue problem. Azure Service Bus *sessions* = ordered processing per conversation/pipeline. (distributed-systems SKILL, Service Bus sessions)
- **Poison-message discipline:** transient failures get retried; permanent failures must not burn CPU forever — route to DLQ after bounded retries, never silently drop. (EventBridge DLQ pattern, c-sharpcorner)
- **Catch-up / replay:** subscribers resume from a checkpoint; events during a disconnect must not be lost forever. For an in-process bus: a persistent event journal + replay API is the equivalent. (PgEventBus spec: global sequence, checkpoint tracking, catch-up on subscribe/reconnect)
- **Metrics that mean something:** queue depth, consumer lag (age of oldest unprocessed), poison rate, dropped count. Lag is the user-visible latency of the async path.
- **Ack-after-processed:** crash between "did the work" and "marked done" = redelivery, which is exactly why consumers are idempotent. For the sync-handler path: a handler exception must never kill the publisher or the bus.
- **Backpressure is a policy, not an accident:** bounded queue + explicit on-full behavior (drop-oldest / drop-newest / block / raise) instead of silently losing events.

### Trash (leave)
- In-process buses that `put_nowait` into a bounded queue and just `pass` on Full — silent event loss with no metric, no DLQ, no replay.
- Buses that claim "ordering guarantees" from a single dispatcher thread but then let sync handlers interleave — or promise global ordering across unrelated topics.
- Fire-and-forget buses where a handler exception vanishes into a log line and the event is considered "delivered".
- "Exactly-once" claims without idempotency keys or dedup windows.

### What this means for our build
Our EventBus already has the right skeleton (sync/async modes, glob topics, dispatcher thread). It is missing: dead-letter queue + handler retry, dedup window on event_id, explicit per-topic FIFO ordering guarantee, journal/replay, on-full backpressure policy, and lag/depth stats. All get built.

## 2. ERROR INTELLIGENCE (the flagship)

### Gold (take)
- **Sentry fingerprinting:** group key = hash of (exception type + normalized message + stack-trace fingerprint). Normalization replaces dynamic values (UUIDs, timestamps, IPs, emails, user ids, hex hashes, memory addresses) with placeholders — so "Connection refused to 127.0.0.1:6379" and "…to 10.0.0.1:6379" become ONE issue, not 1000. (docs.sentry.io rollups, open-source grouping spec)
- **Error groups as first-class records:** fingerprint, count, first_seen, last_seen, status (new / acknowledged / resolved). Resolved groups RE-OPEN if the error recurs after a cooldown — regression detection for free.
- **Fingerprint rules:** user/operator-overridable grouping (`error.type:X -> group-name`), because automatic grouping is sometimes too coarse or too fine.
- **Breadcrumbs:** a trail of events leading up to the error (auth → navigation → payment initiated → crash) attached to the report. Debugging context, not just the stack.
- **Scoped context:** tags/context objects attached per operation without polluting global state.
- **Chained exceptions:** walk `__cause__` / `__context__` — the root cause is often two frames up the chain.
- **Learning loop:** the KB must record outcomes. A fix that worked gets associated with the fingerprint and boosted next time; a fix that failed gets down-weighted. Static pattern tables are the "trash" version of this.

### Trash (leave)
- Naive trackers that create one issue per identical error (1000 "Connection refused to Redis" issues, no grouping, no counts, no first/last seen).
- Regex-only classifiers with no normalization (every IP variation = new "known" error).
- Knowledge bases that never learn: same suggestions for a fix that demonstrably failed 12 times.
- Severity by vibes; no spike detection (a sudden 100x surge in a fingerprint is the real signal).

### What this means for our build
ErrorIntelligence gets: fingerprinting with message normalization, ErrorGroup tracking (counts, first/last seen, status lifecycle with re-open), a *learning* knowledge base persisted to disk (records fix outcomes, boosts/demotes learned fixes, spike detection on per-fingerprint rates), breadcrumbs, scoped context, chained-exception analysis. API stays backward-compatible (analyze/catch_and_analyze/analyze_error/ErrorKnowledgeBase.match).

## 3. ERROR DOCTOR (diagnosis)

Gold: it's already strong (AST-based UnboundLocal analysis, never-raises, live-locals inspection). Gaps to close: more analyzers for the errors that actually hit production — IndexError (show length vs index), ZeroDivisionError (show the zero divisor's value), RecursionError, FileNotFoundError (did-you-mean in the same directory), JSONDecodeError (point at the offending character with a snippet), UnicodeDecodeError, AssertionError (AST: show the failed expression), OSError errno names. Plus: diagnose the `__cause__` chain, not just the surface exception.

## 4. SELF-HEAL

Gold: the existing design is *correct* — only provably semantics-preserving auto-fixes, everything else diagnose-and-report, every fix git-committed for reversibility, never raises. That conservatism is the feature, not a limitation. The real weakness is **no feedback loop**: heal outcomes (fixed/failed) vanish. Cure: report every outcome into the learning KB so the error catcher gets smarter from every heal; register fix strategies in a registry (dynamic, not hardcoded if/elif); distinguish "timed out" probes from "not broken" (a hung command is a finding, not a clean bill of health).

## 5. RETRY / CIRCUIT BREAKER / RATE LIMIT (one dialect)

### Gold (take)
- **Policy composition (Polly v8):** retry + circuit breaker + timeout + fallback composed into one ResiliencePipeline, not three dialects. Order matters: timeout innermost, then retry, then breaker, then fallback outermost.
- **Jitter styles:** full jitter (AWS-recommended default), equal jitter, and **decorrelated jitter** (Marc Brooker / Polly's DecorrelatedJitterBackoffV2 — the production-grade choice for fleet-wide retries). Never fixed-interval retries (thundering herd).
- **Retry budget:** cap retries as a fraction of total traffic (e.g. ≤10%, token-bucket enforced). Per-call policies multiply load 3–5x fleet-wide under partial outage; the budget converts "every call retries independently" into "the system retries sustainably." (skillz loop-engineering)
- **Retry only transients:** timeouts, 5xx, connection refused. Never 4xx — a bad parameter never resolves itself on retry. Our `retryable` flag on framework errors already encodes this; the budget and pipeline must consult it.
- **Breaker = system protection, retry = call protection.** Retries protect the individual call; the breaker protects the system from everyone's retries compounding. Per-dependency state (closed → open → half-open probe trickle).
- **Timeout: every remote call gets one.** Fail fast and retry beats waiting forever. Bulkheads isolate resources per dependency so one bad neighbor can't sink the rest.
- **Fallback is a product decision:** cached value, default, "degraded" — deliberate, not a silent default.
- **Agent-loop lesson:** transport-level transients (429/500/timeout) get retried *inside* the wrapper, invisible to the agent; semantic failures (bad output, failed validation) are NOT retries — they're new attempts that must differ. Conflating the layers is the pathology.

### Rate limiting specifics
| Algorithm | Verdict |
|---|---|
| Token bucket | Keep. Burst-friendly, O(1), matches real API behavior. The app-code default. |
| GCRA (Generic Cell Rate Algorithm) | ADD. Token-bucket-equivalent rate meter, single timestamp (TAT), cheapest per-key state — the high-throughput choice. |
| Sliding window log | Keep (already have). Exact, for low-limit precision (login attempts). |
| Sliding window counter | ADD. ~99% accurate, 2 counters, no boundary burst — the best distributed default. |
| Fixed window | Trash for strict limits (2x boundary burst: 100 at 00:59.9 + 100 at 01:00.1). Skip it — our sliding variants cover the need. |
| Leaky bucket | As a *meter* it's mathematically equivalent to token bucket/GCRA; as a *shaper* it smooths egress. Our throttle() already covers shaping — document, don't duplicate. |
- **Cost-based limiting:** weight expensive operations (an expensive search costs more budget than a cheap lookup), not just count-based.
- **429 responses carry signals:** Retry-After + X-RateLimit-Limit/Remaining/Reset headers so well-behaved clients back off instead of tight-looping.
- **Cohesion fix:** one error dialect. `ratelimit.RateLimitExceeded` should BE `errors.RateLimited` (retry_after honored by retry's default predicate) instead of a parallel plain-Exception.

### What this means for our build
- `retry.py`: JitterStyle enum (none/full/equal/decorrelated), TimeoutPolicy, FallbackPolicy, RetryBudget (token-bucket-capped retries), ResiliencePipeline builder. BackoffPolicy gains jitter_style.
- `ratelimit.py`: GCRA limiter, SlidingWindowCounter, cost-based acquisition, header-bearing 429 decisions, RateLimitExceeded unified under errors.RateLimited.
- `errors.py`: error-code registry + from_dict round-trip; fill genuine gaps (NetworkError, AuthError, QuotaExceeded, DependencyError, StateError).

## 6. Standing correction honored
Per the 2026-10-09 correction: build everything at FULL capability. No designing down to phone/AWS/free-tier constraints — the profile-aware runtime gates deployment, not the library. Every class below is maxed out.
