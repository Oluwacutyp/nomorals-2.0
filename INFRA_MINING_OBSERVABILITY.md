# Infrastructure Mining — Section F: OBSERVABILITY

Research-first groundwork for the observability rebuild:
`nomorals/core/logging_setup.py`, `nomorals/core/observability.py`,
`nomorals/core/tasks.py`, `nomorals/core/clock.py`.
Sources mined: OpenTelemetry SDK design docs, structlog best-practice guides,
Kubernetes probe guidance, Celery/RQ/Dask/Temporal scheduler patterns,
and the homegrown-observability post-mortems (the "trash builds" that
taught the lessons).

## 1. Logging (structlog, stdlib, OTel Logs)

**Gold from the best builds:**
- **contextvars, not thread-locals, carry correlation context** (`structlog.contextvars.merge_contextvars` as the *first* processor). Correct across threads AND asyncio tasks. Two non-negotiable rules: merge first in the chain, and `clear_contextvars()` at the *start* of each request/job — stale IDs leaking between jobs on a reused worker is the classic bug.
- **Redact at the point of logging, not at the sink.** Sink-side filtering is too late: the value already exists in memory/crash dumps, traverses every shipper, and a new destination silently disables it. Back it with **field allowlists** for structured events, plus regex scrubbing for free-text. Pino's `redact.paths` pattern generalizes.
- **Structured JSON is the production wire format** — never plain text in production. One event per line, flat, greppable, with mandatory fields: `ts` (ISO-8601 UTC), `level`, `service`, `env`, `message`, `trace_id`, `span_id`.
- **Never block the hot path on logging.** `QueueHandler` + a listener thread: log calls enqueue and return; formatting and I/O happen off-thread. On a busy 2-vCPU box this is the difference between a log storm and a wedged event loop.
- Log the secret's *name*, never its value. Log the export's row *count*, not its contents.
- Foreign protocol libraries (Discord gateways, MTProto traces) drown the application log — gate their level at import, never after first contact with an incident.
- Redaction processors must never raise — logging must never crash the program.

**Gold from the trash builds (post-mortems):**
- "More logs = more observability" — unstructured prose spam makes incidents slower. Three queryable events beat three hundred prose lines.
- Metric-cardinality bombs: user ID / raw error message / raw URL as a metric label kills the backend. IDs belong in logs and traces, not metric dimensions.
- Latency tracked as an average with no percentiles hides the worst experiences.
- Alerts on causes (CPU %) paging humans while user-facing error rate is unmonitored → alert fatigue → ignored real pages. Symptom-based alerts page; cause-based alerts belong on dashboards.
- Sampling deliberately, and saying so; **keep errors unsampled** — the rare event is the one worth having.

**What this means for `logging_setup.py`:**
correlation via contextvars (trace_id/agent/mission/task auto-injected);
QueueHandler non-blocking option; extras-in-JSON; key-based + regex redaction;
sampling hooks; audit() helper for security events.

## 2. Metrics + Tracing (OpenTelemetry SDK, Prometheus, SigNoz patterns)

**Gold from the best builds:**
- API/SDK split: application code instruments against the thin API; the SDK (processors, samplers, exporters) is configured once at startup. Libraries never import the SDK.
- **Batch span processor in production** — never the simple (synchronous) processor. Batch: 512/batch, 5s schedule delay, 2048 queue.
- **Head + tail sampling**: `ParentBased(TraceIdRatioBased(0.1))` for volume control at the head; tail sampling always captures errors and slow requests. Without sampling, a busy agent is a memory bomb.
- `service.name` resource attribute is REQUIRED — without it, backends can't group anything.
- Span discipline: `record_exception()` for stack traces in the flamegraph; status Ok/Error; span events as timestamped annotations; semantic-convention attribute names; never PII/secrets as attributes; high-cardinality values in attributes, not names.
- Prometheus exposition is the cheapest interoperable wire format — one endpoint, scraped later at zero code cost. Exporters must include **label sets** (our existing `as_prometheus` dropped labels entirely — a real bug).
- Periodic snapshot export (JSONL) for headless boxes where nothing scrapes you: the log file you can `scp` home is the operator's window.

**Gold from the trash builds:**
- Silent span drops from queue overflow with no counter — the instrumentation lies by omission. Always count dropped spans.
- Context loss across asyncio/thread boundaries → broken traces. contextvars propagate through `await` and thread-pool `copy_context` only if you propagate them deliberately.
- Dashboards that answer no question. RED method (Rate, Errors, Duration) for request surfaces; USE (Utilization, Saturation, Errors) for resources.

**What this means for `observability.py`:**
labeled Prometheus export (fix); batching processor + background flush thread;
head sampling + always-keep-errors tail rule; span events + record_exception;
attribute redaction on spans; JSONL trace export; metrics snapshot exporter;
real HealthChecker engine (named checks, timeouts, liveness/readiness split,
TTL caching, flapping counters, never-raise).

## 3. Health checks (Kubernetes probe guidance, rebash/INFRAA literature)

- **Liveness ≠ readiness ≠ startup.** Liveness: process responsive, no deadlock — never checks external dependencies (a DB blip must not become a restart storm). Readiness: can serve — includes hard dependencies with timeouts. Startup: warm-up window for slow boots.
- Probes are cheap, bounded, side-effect-free, circuit-broken. `failureThreshold ≥ 3` on liveness. Timeouts sized from measured response times.
- Return 200 vs 503; structured per-component body (`checks` map) so an operator or autoscaler can identify the failing component mechanically. Never leak secrets/versions/stack traces in the public body.
- Alert on **readiness flapping** separately from restarts — different incident classes.

**What this means:** `HealthChecker` with `liveness()` (process-only: event loop responsive, no deadlock, memory sane) vs `readiness()` (dependency probes), check TTLs, transition tracking.

## 4. Task graphs / schedulers (Celery, RQ/Arq, Dask, FAANG scheduler designs)

**Gold from the best builds:**
- **Timeouts everywhere.** No timeout = no recovery; you hold resources forever. Outer deadline ≤ sum of inner + margin (deadline propagation).
- **Retry order:** call → timeout → retry-with-backoff+jitter (idempotent ops only) → retry budget → circuit breaker → fallback. Exponential backoff capped (`2^n`, capped, uniform jitter), bounded attempts (3) AND bounded budget (~20–30% of traffic is retry traffic).
- **Never retry non-idempotent writes.** Never retry 4xx except 408/429 (honor Retry-After). Validation errors are permanent — retrying them is a busy loop.
- **Poison tasks:** bounded retries, then a permanent, diagnosable failure state. Failed tasks must remain diagnosable — DLQ or equivalent with replay.
- **Idempotency keys**: pass IDs, not objects; JSON-serializable args; check-before-process / status field; exactly-once via WAL-outbox patterns for real writes.
- **Distributed scheduler core**: leader election tolerates a scheduling gap — due tasks wait because they're durably stored, not lost; per-dependency bulkheads so one slow dependency can't consume the whole pool.
- **Dependencies propagate downward on cancellation**: a cancelled parent must not leave orphans burning tokens. Failure cascade: dependents of a FAILED task are SKIPPED (visible), not silently dropped.

**Gold from the trash builds:**
- Instant retries hammering a dying external dependency (the thundering herd) — the #1 homegrown scheduler bug. Our current executor retries *immediately* with no backoff.
- Retry counting infrastructure death (broken process pool) as task failure — already handled well here via requeue.
- Cycle detection only at add-time with unhelpful errors; topological order for dry runs is gold (already have it).
- Scheduler that polls without `not_before` → busy-loop retry storms.

**What this means for `tasks.py`:**
`RetryPolicy` (exponential backoff + full jitter, caps, retryable-exception classes, honor-Retry-After);
`Task.not_before` delayed-readiness so `ready()` respects backoff windows;
`Task.idempotency_key`; DLQ collection with replay; `cascade_skip()` for dependents of failures;
`on_state_change` hooks for metrics wiring; `dry_run()` serial executor honoring the executor's contract;
`RetryBudget` (bounded retry spend per graph).

## 5. Clock (mine-before-build findings)

- Injectable time is the load-bearing trick: tests simulate months in microseconds; missions replay deterministically. Already solid in `clock.py`.
- What's missing for the scheduler work: an explicit `Deadline` (absolute, monotonic, with `remaining()` and `expired()`), a `Backoff` policy object that computes sleep durations (and is itself testable under FrozenClock), `sleep_until`, and a `timeout_after` context manager that converts wall-clock timeouts into monotonic deadlines.

## Deliberate non-goals (kept native on purpose)

- No Prometheus client dependency, no `opentelemetry-sdk` dependency: the module stays stdlib-only so the phone and the AWS box share one code path. The wire formats (Prometheus exposition, OTLP-shaped JSON) stay compatible so adopting the real backends later costs nothing.
- No separate collector process — the exporter thread inside the process plus JSONL files is the deployment model.
