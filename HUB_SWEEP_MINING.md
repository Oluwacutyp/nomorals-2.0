# HUB sweep — external mining

Module under the knife: `nomorals/hub/` (`server.py`, `__init__.py`) — the
device hub: one stdlib HTTP listener composing mesh (`LocalTransport`) and
sync (`LocalPeer`/`SyncStore`) behind JSON routes with bearer auth.

Method: mine every class's outside-the-repo best-in-class, then implement
the gold. No invented techniques — everything below is a real, verifiable
behavior from a real project, cited.

## 1. Syncthing relay server (`strelaysrv`) — github.com/syncthing/syncthing

What it is: the rendezvous/proxy box Syncthing devices meet through — the
closest real-world cousin of our hub (device registry, sessions, keepalive).

Mined gold:
- **Ping interval + pong/network timeout**: default 60s ping, 120s network
  timeout; a session with no traffic in the window is terminated. → We add
  per-connection `timeout` on the handler so a hung client can't park a
  thread forever, and cap long-poll `wait` so a shutdown isn't held hostage.
- **Global + per-session rate limits** (`-global-rate`, `-per-session-rate`).
  → We add a per-IP token-bucket rate limiter, 429 + `Retry-After`.
- **/status endpoint for operators** (`numActiveSessions`, `uptimeSeconds`,
  rolling throughput). → We add `/metrics` (Prometheus) and richer `/health`
  with uptime + component status.
- TLS everywhere (relay protocol is TLS). → We add optional TLS via stdlib
  `ssl.SSLContext` (no new deps).

Source: https://github.com/syncthing/syncthing/blob/main/cmd/strelaysrv/README.md

## 2. CouchDB `_changes` feed — apache/couchdb

The reference design for "give me what changed since X" over HTTP:

Mined gold:
- `since` + `limit` cursor paging (we have this), **`feed=longpoll`** with
  `timeout` (ms): the server holds the request open until a change arrives
  or the timeout expires — instead of the client busy-polling. → We add
  `timeout` to `/sync/pull`, implemented with `SyncStore.subscribe()` so a
  push wakes sleepers immediately (exactly how CouchDB's update notifier
  feeds longpoll).
- `heartbeat` (ms): empty keepalive line to hold proxies open on
  long-lived feeds. → Long-poll responses always arrive promptly on our
  bounded timeouts; documented.
- Response shape: `results` + **`last_seq`** (the store's head sequence) so
  the client can checkpoint without a second read. CouchDB also reports
  `pending`. → We add `last_seq` (from `SyncStore.max_seq()`) and `pending`
  (from `count_since_seq`) to `/sync/pull`.
- `_bulk_docs`: per-document results (`id`/`ok`/`error`), not one
  all-or-nothing verdict. → `/sync/push` now returns per-record results
  while keeping the `applied` count for compatibility.

Sources:
- https://github.com/cozy/couchdb-debian/blob/HEAD/apache-couchdb-1.6.1/share/doc/src/api/database/changes.rst
- https://github.com/cozy/couchdb-debian/blob/HEAD/apache-couchdb-1.6.1/share/doc/src/replication/protocol.rst

## 3. Matrix Client-Server `/sync` — spec.matrix.org

The other great long-poll sync design:

Mined gold:
- `since` token + `timeout` (ms): "the server holds the request open until
  there's something to return (or a timeout), and returns only the deltas
  plus a new token. This is efficient, resumable, and naturally
  offline-first." → Same pattern as our `/sync/pull?timeout=`.
- `sync_notifier`: holds requests until an event arrives or the timeout
  expires; the notifier runs so long-polling clients can't starve short
  requests. → We implement wake-on-notify; ThreadingHTTPServer already
  isolates long-polls on their own threads so short routes never starve.
- Device lists: `changed`/`left` are *deduped* — one entry per subject
  however many rows the range covers. → We don't ship device lists, but the
  lesson lands in `/sync/pull`: records are already unique by key in the
  store, so no extra work needed (documented, not reimplemented).

Sources:
- https://github.com/bm4321/devops-obsidian/blob/HEAD/19%20-%20Matrix/Matrix%20Client-Server%20API.md
- https://github.com/omg-software/merovingian/blob/HEAD/src/sync/AGENTS.md
- https://github.com/grpc/grpc/blob/master/doc/health-checking.md (health, below)

## 4. gRPC health checking protocol — grpc/grpc

The industry-standard health contract:

Mined gold:
- `Check(service) → SERVING | NOT_SERVING | SERVICE_UNKNOWN`, with empty
  service = overall health; **the RPC itself returns OK — the status is in
  the body**, so health consumers never confuse transport errors with
  "unhealthy". → Our `/health` keeps HTTP 200 always and reports per-component
  status (`mesh`, `sync`, `db`) with real checks (`LocalTransport.ping()`,
  `LocalPeer.ping()`) plus latency — the old `{"ok": true}` stays for
  compatibility.
- Kubernetes practice layered on top: separate **liveness** (is the process
  up?) from **readiness** (can it serve? → 200/503). → We add `/ready`
  returning 503 with the failing component list when degraded.

Source: https://github.com/grpc/grpc/blob/master/doc/health-checking.md

## 5. Prometheus exposition format — prometheus/docs

Mined gold:
- Text format `text/plain; version=0.0.4`, `# HELP` / `# TYPE` lines,
  counters for cumulative events, gauges for point-in-time values, unit
  suffixes (`_total`, `_seconds`), labels for dimensions. → We add
  `GET /metrics` rendering hub request counters (per route × status),
  error counters, and gauges (active nodes, queued/leased tasks, sync
  head seq, uptime) in exactly this format — stdlib-only rendering.

Source: https://github.com/prometheus/docs/blob/HEAD/docs/instrumenting/exposition_formats.md

## 6. RQ (python-rq) job registries — rq/rq

The best small task-queue observability model:

Mined gold:
- Named registries: queued / **started** / deferred / **finished** /
  **failed** / scheduled / canceled — every transition updates all
  applicable owners, and each registry is inspectable. → Our queue layer
  already has `stats()` (per-topic × status), `list_live()`, `dead()`;
  the hub exposed *none* of it. We add `GET /mesh/stats`, `GET /mesh/jobs`,
  `GET /mesh/dead` (the dead-letter queue).
- `FailedJobRegistry.requeue` / `Job.requeue`: replay a failed job after the
  operator fixes the cause. → `POST /mesh/retry-dead` over
  `MeshTasks.retry_dead`.
- Job cancellation retains the job for inspection and records the canceled
  state. → `POST /mesh/cancel`.
- Result TTL / `job.return_value()`: fetch a finished job's result.
  → `GET /mesh/job?job_id=` (progress + result + live status).

Sources:
- https://raw.githubusercontent.com/mikeabrahamsen/rq/master/CHANGES.md
- https://github.com/e2e-bmk/bmk-dev/blob/HEAD/tasks/python/rq-fullrepro-001/spec.md

## 7. Temporal workflow-ID reuse (via our own `MeshTasks.dispatch`)

`dedupe_key`: "makes dispatch idempotent (Temporal workflow-ID reuse
idea): while a live task holds the key, re-dispatch returns the existing
job id instead of queueing a duplicate." The queue supports it; the hub
route didn't pass it through. → `/mesh/dispatch` now accepts `dedupe_key`
(+ `delay`, `max_attempts`, `expire_after`, `target_capabilities`,
`target_labels` — the Kubernetes-nodeSelector routing our queue already
does), and `POST /mesh/dispatch-many` fans out via `dispatch_many`.

Source: `nomorals/mesh/tasks.py` (in-repo, mined as instructed — existing
gold surfaced to the HTTP layer).

## 8. stdlib TLS for `http.server` — Python docs + community practice

Mined gold: wrap the *listening* socket with
`ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)`, `load_cert_chain`, `minimum_version`
(TLS 1.2+), `ctx.wrap_socket(sock, server_side=True)` — the standard recipe
used by ESP OTA servers, lab servers, and Cloudflare's documented Python
PoC. Zero new dependencies. → Hub takes `certfile`/`keyfile` (+
`NM_HUB_CERT`/`NM_HUB_KEY` env), `url` flips to `https://`.

Sources:
- https://github.com/saikumar-mandaji/esp-ota-https/blob/HEAD/docs/hardware/BOM.md
- https://github.com/sukesh-ak/pythonwsssl
- https://github.com/xol1507/cloudflare-docs (managed-networks.mdx, Python PoC)

## 9. Rate-limit / hardening practice

- 429 **must include `Retry-After`**; per-IP GCRA/token-bucket, separate
  limits per connection type (the DDoS-protection writeup).
  → Implemented, `/health` + `/ready` exempt so probes never 429.
- strelaysrv `-message-timeout`: bound how long we wait for a relevant
  message → our long-poll caps.
- `hmac.compare_digest` for the bearer token — already present, kept.

Source: https://github.com/ambiguous-interactive/signal-fish-server/blob/HEAD/.llm/skills/ddos-protection/SKILL.md

## 10. Compatibility findings (bugs the mining exposed)

- `HttpTransport.deregister()` raises `MeshError("hub does not expose node
  deregistration")` — the client *wants* `POST /mesh/deregister` and the hub
  404s it. **Bug: route missing.** Added.
- `HttpTransport.cancel/result/stats` raise "hub does not expose …".
  Added all three routes.
- `HttpTransport.poll(wait>0)` long-polls *client-side* ("no server change
  needed") — one HTTP round-trip per second per worker. `LocalTransport.poll`
  already supports server-side `wait` (deadline loop, 250ms DB slices); we
  expose `?wait=` on `/mesh/poll` so one connection replaces the hammering.
- `HttpTransport.heartbeat(node_id, info)` — the hub dropped `info`.
  Accepted now.

## What we deliberately did NOT take

- **gzip**: the stdlib client (`_KeepAlivePool`) doesn't send
  `Accept-Encoding`, so compressing would be dead code. Skipped honestly.
- **WebSocket/SSE event stream**: hub telemetry already flows through the
  in-process event bus (`global_bus`); an SSE route would duplicate it for
  no consumer. Skipped.
- **Per-node scoped tokens / ACLs** (headscale-style): real gold, but the
  hub's auth model today is one bearer token; scoped tokens belong in a
  dedicated auth pass, not bolted on here. Noted as the honest next step.
