# MESH sweep — external mining report

Module: `nomorals/mesh/` — device mesh: node presence, task dispatch (work-stealing), local + HTTP transports.
Sweep date: 2026-10-10. Every significant class was compared against best-in-class outside implementations.
Rule followed: summarize in own words, link sources, no verbatim copying.

---

## 1. MeshNode / NodeRegistry — "How does the best implementation of presence do it?"

### Kubernetes (kubelet → control plane)
- **Two-tier heartbeats**: cheap `Lease` objects renewed every ~10s (just a timestamp) + expensive full `NodeStatus` pushed rarely (~5m). Lesson: heartbeats should be cheap; rich status should be separate and infrequent.
- **Conditions, not just liveness**: nodes carry `Ready / MemoryPressure / DiskPressure / NetworkUnavailable` conditions. Presence is richer than "seen recently".
- **Labels + selectors**: nodes carry `key=value` labels; the scheduler matches pods via `nodeSelector`. This is the gold standard for capability-aware placement.
- **Grace periods**: `NodeMonitorGracePeriod` (~50s) between "missed heartbeat" and "treated as failed". Failure is a *process*, not a single threshold.
- Sources: https://github.com/ironcore-dev/ironcore-dev.github.io/blob/HEAD/docs/iaas/architecture/machine-pool-health.md · https://medium.com/@phoenixarjun007/the-six-minutes-that-decide-a-kubernetes-node-failure-68f132628704

### HashiCorp Serf / memberlist (SWIM gossip — Consul, Nomad)
- **Suspicion window**: a node that stops acking is first marked *suspect* (gossiped), and only declared *dead* after a refutation window in which it can clear itself. No false positives from transient blips.
- **Indirect probing**: if direct ping fails, ask k random peers to probe before suspecting.
- Lesson: binary fresh/stale presence is crude. A three-state `ready → suspect → gone` model is the proven shape.
- Sources: https://github.com/hashicorp/serf/blob/master/docs/internals/gossip.html.markdown · https://github.com/nicolasholanda/swim-gossip

### Current state of our NodeRegistry
- Single `last_seen` timestamp, one `stale_after` cutoff. No labels (only a flat `capabilities` list), no conditions, no suspicion state, no metadata on heartbeat, no graceful leave, no presence events.
- **Gaps to fill**: labels/selectors (k8s), suspect window (Serf), heartbeat-attached info (load/version — k8s NodeStatus-lite), graceful deregister, node-joined/node-left events, a human-readable presence view.

---

## 2. MeshTasks / MeshTask — "How does the best implementation of task dispatch do it?"

### Temporal (task queues + activities)
- **Activity heartbeats with checkpoint/resume**: long tasks heartbeat progress; the *next retry attempt receives the last heartbeat details* and resumes instead of restarting. Our leases expire silently — a crashed worker's partial progress is lost.
- **Declarative retry policies**: initial interval, backoff coefficient, max interval, max attempts, *non-retryable error types*. Our retry is a boolean + fixed max_attempts on the queue.
- **Timeout model**: schedule-to-start (picked up in time?) vs start-to-close (finished in time?) vs heartbeat timeout (still alive?). We have none of these at the mesh layer.
- **Task queues are pull-based with capacity awareness**: workers ask for work only when they have capacity. Our poll loop does this part right.
- **Idempotency**: workflow-ID reuse policy rejects duplicate submissions. We have no dedupe — double-dispatch double-runs.
- Sources: https://github.com/kruzzzzy/ai-kos/blob/HEAD/research/atq/atq-research-queues.md · https://github.com/az-said/interlock/blob/HEAD/research/temporal-bar.md

### Celery
- **Routing**: `task_routes` maps task names → queues; workers consume selected queues. Resource-based queue separation (CPU-heavy vs IO) prevents starvation.
- **Per-task rate limits** (`rate_limit='10/m'`), **priorities** (0–9), **soft+hard time limits** so hung tasks can't occupy a worker forever.
- **Fairness**: low `worker_prefetch_multiplier` + `acks_late` so work spreads evenly instead of piling onto whoever grabbed first.
- **Observability**: Flower — per-queue backlog, success/failure rates, worker utilization. We have zero queue introspection.
- Sources: https://docs.celeryq.dev/en/4.4.1/getting-started/first-steps-with-celery.html · https://reintech.io/blog/understanding-celery-task-routing-queues

### Dask / Cilk (work stealing)
- **Locality-aware stealing**: Dask's scheduler estimates start time from data locality + worker occupancy and steals from overloaded workers to underloaded ones. Our "work-stealing" is first-claimer-wins with no notion of load or fit.
- Lesson: at minimum, route to nodes whose *capabilities/labels* fit the task (k8s nodeSelector again); ideally track node load from heartbeat info.
- Sources: http://arXiv.org/pdf/2010.11105 · https://distributed.dask.org (work-stealing docs, via paper)

### Current state of our MeshTasks
- Solid: topic-per-node + broadcast topics, atomic leases, payload size guard, telemetry events on dispatch/complete/fail.
- **Gaps to fill**: task heartbeat/progress (Temporal), declarative RetryPolicy with backoff + non-retryable errors, dedupe keys (Temporal id reuse), task expiry/schedule-to-start (Temporal), cancel + result fetch + wait (WorkQueue already supports all three — mesh just never exposed them!), queue stats (Flower-lite), capability-aware routing (k8s selectors), batch dispatch (enqueue_many exists, unexposed), dead-task inspection/retry.

---

## 3. Transport (ABC) — "How does the best client interface look?"

- gRPC/Temporal clients expose: health/ping, deadlines per call, cancellation, streaming, graceful close.
- Our ABC is the minimal CRUD of mesh ops. Missing: `ping()` (is the hub alive + latency), long-poll wait on `poll()` (near-instant dispatch without hammering), `result()`/`cancel()`/`stats()` passthrough, `deregister()`, lifecycle (`close()`/context manager).

---

## 4. HttpTransport — "How does the best HTTP client behave?"

### AWS Architecture Blog — Exponential Backoff and Jitter
- Canonical formula (full jitter): `sleep = random(0, min(cap, base * 2^attempt))`. Equal jitter and decorrelated jitter are the other two standard flavors.
- Our transport uses **deterministic** `min(2^attempt, 8)` — the exact thundering-herd shape AWS warns about: every node that lost the hub retries in lockstep.
- Sources: https://dev.to/kashif_manzer/why-your-retries-need-jitter-the-thundering-herd-explained-3eje · https://lumigo.io/blog/amazon-builders-library-in-focus-1-timeouts-retries-and-backoff-with-jitter/

### Amazon Builders' Library — timeouts/retries
- Retry at a **single point** in the stack (don't multiply retries across layers).
- **Know which errors to retry**: 4xx never; network/timeout/5xx/429 yes. We retry network errors but NOT HTTP 5xx/429 — a 503 from the hub is raised immediately with no retry, and `Retry-After` is ignored.
- **Circuit breakers** give failing systems room to recover. We hammer a dead hub on every poll cycle.
- Sources: https://lumigo.io/blog/amazon-builders-library-in-focus-1-timeouts-retries-and-backoff-with-jitter/

### Resilience4j — circuit breaker
- Three states: CLOSED (normal) → OPEN (fail fast after failure-rate threshold over a sliding window) → HALF_OPEN (limited trial calls; success closes, failure re-opens). Plus slow-call detection.
- Gold to merge: a small stdlib circuit breaker around hub calls — CLOSED/OPEN/HALF_OPEN with configurable threshold + reset timeout, state exposed for observability.
- Sources: https://github.com/kavyasiddharthan/spring-boot-patterns/blob/HEAD/resilience/circuit-breaker/README.md

### Push patterns — polling vs long-poll vs SSE vs WebSocket
- Our remote poll is short-polling: latency up to the poll interval, or wasteful empty responses at high frequency.
- Industry default ladder: **long-polling** as the cheap upgrade (server holds until data or timeout), SSE for one-way server push, WebSocket for true bidirectional.
- Gold to merge: `poll(wait=...)` long-poll on the client side works against the *existing* hub (no server change): loop short polls until tasks arrive or the deadline hits. Document SSE as the hub-side future.
- Sources: https://github.com/saint-james-fr/my-skills/blob/HEAD/system-design-real-time-systems/SKILL.md · https://dev.to/vivekyadav200988/deep-dive-into-server-sent-events-sse-4oko

### HTTP client hygiene
- We open a fresh TCP+TLS connection per request (urllib, no pooling) — expensive on a phone. Gold: keep-alive connection reuse within the transport (stdlib `http.client` persistent connections).
- Missing: `User-Agent`, request IDs, per-call timeout override.

---

## 5. errors.py — "How do the best error models look?"

- **gRPC status codes**: 17 canonical codes + retryability is derivable from the code. **RFC 9457 problem details**: machine-readable `type/title/status/detail` on the wire.
- Our errors are three bare classes with string messages — no codes, no retryability hint, no wire serialization. The hub already sends `{"code": "node_unknown"}` but the client maps it ad hoc.
- Gold to merge: `code` + `retryable` on every mesh error, `to_dict()`/`from_dict()` wire form, and specific subclasses for the cases operators actually branch on: auth rejected, hub unreachable, task not found/expired, circuit open.

---

## What this sweep merges (the gold list)

| # | Gold (source) | Lands in |
|---|---|---|
| 1 | k8s labels + selectors for placement | `MeshNode.labels`, `NodeRegistry.select()` |
| 2 | Serf suspicion window (ready→suspect→gone) | `NodeRegistry.list_suspect()` |
| 3 | k8s NodeStatus-lite on heartbeat | `heartbeat(node_id, info={...})`, `MeshNode.info` |
| 4 | Graceful leave + presence events | `deregister()`, `mesh.node.joined/left` events |
| 5 | Temporal activity heartbeat + lease extension | `MeshTasks.heartbeat()` (progress + `extend_lease`) |
| 6 | Temporal declarative retry policy | `RetryPolicy` dataclass + per-dispatch policy |
| 7 | Temporal schedule-to-start timeout | `dispatch(expire_after=...)` + `reap_expired()` |
| 8 | Temporal idempotent re-dispatch | `dispatch(dedupe_key=...)` |
| 9 | WorkQueue's hidden surface | `cancel/result/wait_for_result/retry_dead/dead/reclaim/dispatch_many/stats` |
| 10 | Celery Flower-lite introspection | `MeshTasks.stats()` per-topic ready/leased/failed |
| 11 | k8s nodeSelector routing | `dispatch(target_capabilities=[...])` → registry select |
| 12 | AWS full-jitter backoff | jittered retry sleeps in `HttpTransport` |
| 13 | Builders' Library: retry 5xx/429, honor Retry-After | `_request` retry policy |
| 14 | Resilience4j circuit breaker | `_CircuitBreaker` (CLOSED/OPEN/HALF_OPEN) |
| 15 | Long-polling ladder | `poll(..., wait=)` client-side long-poll |
| 16 | HTTP keep-alive | persistent `http.client` connection pool in transport |
| 17 | gRPC/RFC-9457 error model | `code`, `retryable`, `to_dict()` on errors + new subclasses |
| 18 | God-tier presentation | `describe()` one-liners + `format_*_table()` + mesh status dashboard |
