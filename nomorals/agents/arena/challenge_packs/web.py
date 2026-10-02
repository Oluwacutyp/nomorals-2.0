"""Web-domain challenge pack for the arena: ``chall_web``.

Eight categories, 12 challenges each (3 easy / 5 medium / 4 deep):

* ``web_fullstack`` — SSR/hydration, edge rendering, form pipelines, offline-first
* ``apis`` — REST/OpenAPI design, versioning, pagination, rate limiting,
  webhook delivery with retries
* ``auth`` — WebAuthn/passkeys, OIDC, session vs token, MFA flows,
  permission models
* ``realtime`` — WebSocket fan-out, SSE, CRDT collab editing, presence
* ``mobile`` — offline sync, push, deep links, app-size budgets
* ``termux`` — Android constraints: no root, wake locks, battery,
  storage paths, proot
* ``performance`` — profiling, flame graphs, p99 budgets, caching layers
* ``memory`` — allocators, pooling, leak detection, RSS budgets

Every entry is a concrete task with a one-line verifiable acceptance
criterion — no essay prompts.
"""

from __future__ import annotations

from ..challenges import challenge
from ..topics import register_topic_pack

_PACK = {
    "web_fullstack": [
        challenge(
            "Build a server-rendered page with a contact form that validates "
            "on the server and redisplays errors inline",
            1,
            "pytest passes: invalid submissions re-render with field errors, "
            "valid submission stores a row, no JS required",
            "code", "ssr", "forms", "validation"),
        challenge(
            "Add hydration-mismatch detection to a small SSR app: log a "
            "warning and re-render when client HTML differs from the server",
            1,
            "node script outputs exactly 1 mismatch warning on a deliberately "
            "diverged page and 0 on a matching page",
            "code", "ssr", "hydration", "debugging"),
        challenge(
            "Ship a static page as a PWA with a service worker that caches "
            "all assets and serves them offline",
            1,
            "check script passes: simulated-offline fetch of the page plus 3 "
            "assets all return 200 from cache",
            "build", "pwa", "offline-first", "service-worker"),
        challenge(
            "Implement a form pipeline with progressive enhancement: client "
            "pre-validation, server validation, and an idempotency key that "
            "prevents double-submit",
            2,
            "pytest passes: double-submit with the same idempotency key "
            "creates one row; JS disabled still validates server-side",
            "code", "forms", "idempotency", "progressive-enhancement"),
        challenge(
            "Benchmark edge vs origin rendering for 3 page types and document "
            "the traffic level where edge rendering pays off",
            2,
            "report shows p50/p99 for edge vs origin at 1k/100k/1M hits/day "
            "with a stated break-even crossover point",
            "research", "edge", "ssr", "cost"),
        challenge(
            "Implement streaming SSR with suspense-style boundaries: send the "
            "shell HTML first, then stream in deferred sections",
            2,
            "script measures first HTML chunk under 200ms and a complete page "
            "with 3 deferred sections fully hydrated",
            "code", "streaming", "ssr", "performance"),
        challenge(
            "Build an offline-first notes app: local-first writes, a "
            "background sync queue, and conflict resolution on reconnect",
            2,
            "script passes: 10 notes created offline all reach the server on "
            "reconnect, no duplicates, conflicts resolved last-writer-wins",
            "build", "offline-first", "sync", "conflicts"),
        challenge(
            "Implement incremental static regeneration: static pages "
            "revalidate in the background on a TTL with stale-while-revalidate",
            2,
            "test passes: page served stale within 5s of a content change, "
            "then fresh after the revalidate window",
            "code", "isr", "caching", "ssr"),
        challenge(
            "Design a full-stack observability pipeline: request-id tracing "
            "from edge through SSR to the DB, with a sampled trace-viewer UI",
            3,
            "artifact runs: one request produces an end-to-end trace with 4+ "
            "spans viewable in the local UI",
            "build", "observability", "tracing", "fullstack"),
        challenge(
            "Implement a resumable multi-part upload pipeline with checksum "
            "verification and progress that survives page reloads",
            3,
            "pytest passes: kill the upload at 60%, resume completes it, "
            "sha256 matches the source, zero completed chunks re-uploaded",
            "code", "uploads", "resumability", "checksums"),
        challenge(
            "Build a zero-JS-first storefront: cart, checkout, and search all "
            "working server-side, with JS layered on only as enhancement",
            3,
            "lighthouse-style script passes: all core flows work with JS "
            "disabled; total JS under 50KB gzipped",
            "build", "progressive-enhancement", "ecommerce", "performance"),
        challenge(
            "Implement per-user feature-flagged SSR: flag evaluation at render "
            "time with consistent variants across edge and client hydration",
            3,
            "test passes: 100 requests show a stable variant per user with no "
            "hydration mismatch for either variant",
            "code", "feature-flags", "ssr", "hydration"),
    ],
    "apis": [
        challenge(
            "Design a paginated REST list endpoint with cursor pagination and "
            "document the contract in OpenAPI 3.1",
            1,
            "pytest passes: 10k rows page through with cursors, no duplicates "
            "or gaps; spec validates cleanly",
            "code", "rest", "pagination", "openapi"),
        challenge(
            "Implement a webhook receiver that verifies HMAC signatures and "
            "responds within the provider timeout",
            1,
            "test passes: tampered payload rejected, valid payload processed "
            "under 2s, replayed signature rejected",
            "code", "webhooks", "security", "hmac"),
        challenge(
            "Add versioning to an existing API: v1/v2 via header and path, "
            "with a deprecation sunset header on v1",
            1,
            "test passes: both versions serve, v1 responses carry Sunset and "
            "Deprecation headers, v2 schema validates",
            "code", "versioning", "rest", "deprecation"),
        challenge(
            "Implement token-bucket rate limiting with Redis: per-key and "
            "per-endpoint limits, Retry-After headers, burst allowance",
            2,
            "test passes: 100-request burst allowed then 429 with Retry-After; "
            "limits refill at the configured rate",
            "code", "rate-limiting", "redis", "throttling"),
        challenge(
            "Build a webhook delivery system with retries: exponential "
            "backoff, jitter, and a dead-letter queue after N failures",
            2,
            "script passes: flaky endpoint receives all 50 events in order "
            "after retries; permanently-down endpoint lands in the DLQ",
            "build", "webhooks", "retries", "reliability"),
        challenge(
            "Design an idempotent POST API: client-supplied idempotency keys, "
            "24h key TTL, safe replay semantics",
            2,
            "test passes: replay of 100 requests yields exactly one side "
            "effect; expired keys rejected with 422",
            "code", "idempotency", "rest", "design"),
        challenge(
            "Implement API filtering, sorting, and sparse fieldsets with "
            "query-cost limits that reject abusive queries",
            2,
            "test passes: 20 query combos return correct subsets; a "
            "pathological query is rejected with a 400 cost error",
            "code", "rest", "filtering", "query-cost"),
        challenge(
            "Audit 5 public REST APIs for pagination, versioning, and "
            "error-format consistency and publish a scored comparison",
            2,
            "report scores 5 APIs on 8 criteria with a reproducible check "
            "script per criterion",
            "research", "rest", "audit", "standards"),
        challenge(
            "Implement a multi-tenant API gateway: tenant routing, per-tenant "
            "rate limits, request signing, and an audit log",
            3,
            "artifact passes: 3 tenants isolated, cross-tenant request "
            "rejected 403, audit log records all mutations",
            "build", "gateway", "multitenancy", "security"),
        challenge(
            "Design and implement a breaking-change-free API evolution "
            "strategy: additive changes, field deprecation, client capability "
            "negotiation",
            3,
            "test passes: v1 client works unchanged against a v3 server; "
            "deprecated fields warn; capability header selects the shape",
            "code", "versioning", "evolution", "compatibility"),
        challenge(
            "Build a webhook fan-out service: 10k subscribers, per-subscriber "
            "retry policy, at-least-once delivery with dedup keys",
            3,
            "load script passes: 10k deliveries with zero loss; dedup keys "
            "prevent double-processing on retry",
            "build", "webhooks", "fanout", "scale"),
        challenge(
            "Implement an OpenAPI-first codegen pipeline: spec to server "
            "stubs plus client SDK plus mock server, diff-gated on spec change",
            3,
            "CI script passes: a spec change regenerates stubs, SDK, and "
            "mocks; a breaking diff fails the build",
            "code", "openapi", "codegen", "ci"),
    ],
    "auth": [
        challenge(
            "Build a comparison harness: cookie session, JWT, and opaque token "
            "auth against the same endpoints, including revocation behavior",
            1,
            "pytest passes: all three authenticate; revocation test shows "
            "opaque/session revoked instantly while JWT waits for expiry",
            "code", "sessions", "jwt", "tokens"),
        challenge(
            "Build a passwordless email magic-link flow with single-use "
            "tokens and 15-minute expiry",
            1,
            "test passes: link works once then returns 410; expired link "
            "rejected; sends rate-limited to 3 per hour",
            "code", "passwordless", "magic-link", "email"),
        challenge(
            "Implement RBAC: roles, permissions, and middleware enforcing "
            "them across 10 routes",
            1,
            "pytest passes: 30 allow/deny cases correct; privilege-escalation "
            "attempt returns 403",
            "code", "rbac", "permissions", "middleware"),
        challenge(
            "Implement an OAuth2 authorization-code flow with PKCE against a "
            "stub IdP, including refresh-token rotation",
            2,
            "pytest passes: full code-to-token-to-refresh cycle, 20 runs, "
            "tokens validate, rotated refresh token invalidates the old one",
            "code", "oauth2", "pkce", "oidc"),
        challenge(
            "Build a TOTP MFA enrollment and verification flow with backup "
            "codes and account recovery",
            2,
            "test passes: enroll, verify with a generated code, burn one "
            "backup code, lockout after 5 bad attempts",
            "code", "mfa", "totp", "account-recovery"),
        challenge(
            "Implement OIDC login against a real provider, mapping claims to "
            "app roles",
            2,
            "artifact logs in: ID token signature verified, claims mapped to "
            "roles, session created, logout destroys it",
            "build", "oidc", "sso", "claims"),
        challenge(
            "Build an ABAC permission model: attribute-based rules evaluated "
            "per request through a policy engine",
            2,
            "test passes: 25 policy scenarios evaluate correctly, including "
            "time-based and ownership rules",
            "code", "abac", "policy", "permissions"),
        challenge(
            "Implement secure session management: rotating session ids, "
            "absolute plus idle timeouts, concurrent-session limits",
            2,
            "test passes: fixation attempt fails, idle timeout logs out at "
            "30m, a 3rd concurrent login evicts the oldest",
            "code", "sessions", "security", "hardening"),
        challenge(
            "Implement WebAuthn passkey registration and authentication using "
            "a virtual authenticator",
            3,
            "pytest passes with a virtual authenticator: register, login, and "
            "a cross-device attempt rejected; 50 cycles green",
            "code", "webauthn", "passkeys", "phishing-resistant"),
        challenge(
            "Build a step-up authentication system: risk-scored re-auth for "
            "sensitive actions like payments and email changes",
            3,
            "artifact passes: low-risk action proceeds, high-risk triggers "
            "step-up, 10 risk scenarios behave per policy",
            "build", "step-up", "risk", "mfa"),
        challenge(
            "Implement a multi-tenant SSO broker: SAML plus OIDC IdPs, "
            "per-tenant config, JIT provisioning",
            3,
            "artifact passes: two tenants with different IdPs, JIT creates "
            "users with correct roles, misconfigured tenant fails safely",
            "build", "sso", "saml", "multitenancy"),
        challenge(
            "Design and implement a token service: short-lived access tokens, "
            "refresh rotation, device binding, revocation list",
            3,
            "test passes: stolen refresh-token reuse is detected and the "
            "whole chain revoked; device mismatch rejected",
            "code", "tokens", "rotation", "revocation"),
    ],
    "realtime": [
        challenge(
            "Build a WebSocket echo server with heartbeat pings and a client "
            "that reconnects automatically with backoff",
            1,
            "test passes: server kill leads to client reconnect within 10s; "
            "heartbeat detects a dead peer in under 30s",
            "code", "websockets", "reconnect", "heartbeat"),
        challenge(
            "Implement server-sent events for a live feed with reconnect and "
            "Last-Event-ID resume",
            1,
            "test passes: dropped connection resumes from the last id; zero "
            "missed events over 500 messages",
            "code", "sse", "streaming", "resume"),
        challenge(
            "Build a presence system: online/away/offline states with "
            "heartbeat TTL in Redis",
            1,
            "test passes: presence correct for 100 users; stale heartbeat "
            "flips to offline within TTL plus 5s",
            "code", "presence", "redis", "heartbeat"),
        challenge(
            "Implement WebSocket fan-out to 10k subscribers: room sharding, "
            "backpressure, slow-consumer disconnect",
            2,
            "load script passes: 10k clients receive the broadcast; slow "
            "consumer disconnected at buffer cap; no head-of-line stall",
            "code", "websockets", "fanout", "backpressure"),
        challenge(
            "Build a collaborative text editor with CRDT semantics: "
            "concurrent edits converge",
            2,
            "script passes: 3 simulated clients with 200 concurrent edits "
            "each converge to identical text",
            "build", "crdt", "collaboration", "convergence"),
        challenge(
            "Implement a realtime leaderboard: sorted-set updates with top-N "
            "broadcast throttled to 4Hz",
            2,
            "test passes: 5k score updates keep the top-10 correct; broadcast "
            "rate never exceeds 4Hz",
            "code", "leaderboard", "redis", "throttling"),
        challenge(
            "Build a chat with typing indicators, read receipts, and message "
            "ordering guarantees",
            2,
            "test passes: 200 messages arrive ordered per sender; typing "
            "indicator times out; receipts ack each message",
            "build", "chat", "ordering", "receipts"),
        challenge(
            "Implement operational transform for a shared list: transform "
            "concurrent ops so replicas converge",
            2,
            "test passes: 1000 randomized concurrent op pairs converge to "
            "the same state on both replicas",
            "code", "ot", "collaboration", "convergence"),
        challenge(
            "Build a geo-distributed realtime relay: two regions, a "
            "cross-region message bus, under 150ms p99 user-to-user",
            3,
            "artifact passes: messages flow in both regions, measured p99 "
            "latency under 150ms, region failover keeps chat alive",
            "build", "geo", "relay", "latency"),
        challenge(
            "Implement end-to-end encrypted realtime messaging with a Double "
            "Ratchet over a simulated transport",
            3,
            "test passes: 500 messages decrypt correctly; compromise of old "
            "state cannot decrypt new messages",
            "code", "e2ee", "ratchet", "crypto"),
        challenge(
            "Build a realtime sync engine: offline edits queued, delta sync, "
            "conflict-free merge with vector clocks",
            3,
            "script passes: offline client with 50 edits syncs; vector clocks "
            "order causally; no lost updates",
            "build", "sync", "vector-clocks", "offline"),
        challenge(
            "Implement a scalable pub/sub broker: topic wildcards, retained "
            "messages, QoS 0/1/2 semantics",
            3,
            "test passes: wildcard routing correct; QoS1 delivers exactly "
            "once to the app layer; retained replay on subscribe",
            "code", "pubsub", "mqtt", "qos"),
    ],
    "mobile": [
        challenge(
            "Implement deep-link routing: parse 10 deep-link shapes into "
            "in-app routes with fallback to web",
            1,
            "test passes: all 10 links route correctly; malformed link falls "
            "back to home; universal-link file validates",
            "code", "deep-links", "routing", "universal-links"),
        challenge(
            "Implement push-notification handling for foreground, background, "
            "and killed states: badge counts and deep-link on tap",
            1,
            "test passes: tap opens the correct screen in all 3 app states; "
            "badge clears on open",
            "code", "push", "notifications", "deep-links"),
        challenge(
            "Set an app-size budget: measure a release APK/AAB, attribute "
            "size by module, fail CI over budget",
            1,
            "CI script passes: size report attributes 90%+ of bytes; a build "
            "over the 25MB budget fails",
            "code", "app-size", "ci", "budgets"),
        challenge(
            "Implement offline-first sync for a field-notes app: outbox "
            "queue, delta pull, conflict resolution",
            2,
            "script passes: 7 days of offline edits sync on reconnect; "
            "server and client converge; conflicts logged",
            "build", "offline-first", "sync", "outbox"),
        challenge(
            "Build a background-upload manager: chunked uploads that survive "
            "app kill and resume on relaunch",
            2,
            "test passes: kill at 70% then relaunch resumes from 70%; sha256 "
            "matches; battery-saver mode still progresses",
            "code", "uploads", "background", "resumability"),
        challenge(
            "Implement biometric-gated secure storage: keychain/keystore "
            "backed secrets bound to the device lock",
            2,
            "test passes: secret readable after biometric; unreadable after "
            "simulated device wipe; no plaintext on disk",
            "code", "biometrics", "keychain", "secrets"),
        challenge(
            "Build an OTA update flow: version check, delta download, staged "
            "rollout with a kill switch",
            2,
            "script passes: 5% rollout cohort updates; kill switch halts the "
            "rollout; rollback restores the prior bundle",
            "code", "ota", "rollout", "updates"),
        challenge(
            "Implement an adaptive image pipeline: responsive sizes, "
            "WebP/AVIF negotiation, placeholder blur-up",
            2,
            "test passes: 3 network profiles fetch appropriate sizes; LCP "
            "under budget on the 4G profile",
            "code", "images", "responsive", "performance"),
        challenge(
            "Build a cross-platform offline map tile cache: vector tiles, LRU "
            "eviction, offline routing for a city",
            3,
            "artifact passes: full city cached under 200MB; offline route "
            "computed; eviction keeps the working set",
            "build", "maps", "offline", "caching"),
        challenge(
            "Implement E2E-encrypted cloud backup: device key, encrypted "
            "blobs, restore on a new device via recovery phrase",
            3,
            "test passes: backup, wipe, restore yields identical data; server "
            "blobs indecipherable without the key",
            "code", "backup", "e2ee", "recovery"),
        challenge(
            "Build a battery-aware sync scheduler: JobScheduler/WorkManager "
            "constraints, exponential backoff, doze-mode compliance",
            3,
            "test passes: no sync during the doze window; batched sync on the "
            "maintenance window; 24h drain within budget",
            "code", "battery", "background", "android"),
        challenge(
            "Implement a mobile analytics pipeline with offline buffering and "
            "privacy-safe aggregation",
            3,
            "artifact passes: 10k events buffered offline flush in order; no "
            "PII leaves the device; aggregates match raw",
            "build", "analytics", "privacy", "offline"),
    ],
    "termux": [
        challenge(
            "Write a Termux bootstrap script: installs packages, sets up "
            "storage symlinks, verifies no-root operation",
            1,
            "script runs on a fresh Termux: all packages install, ~/storage "
            "accessible, zero su calls made",
            "build", "termux", "bootstrap", "android"),
        challenge(
            "Implement a Termux wake-lock helper: acquire during long jobs, "
            "release on exit, survive screen-off",
            1,
            "test passes: job completes with the screen off; lock released on "
            "SIGTERM; held window visible in battery stats",
            "code", "wake-lock", "battery", "android"),
        challenge(
            "Survey the Termux filesystem layout across 3 Android versions "
            "and publish a path-permission matrix",
            1,
            "report covers 3 Android versions: each path's read/write/exec "
            "status with a verification script per row",
            "research", "termux", "storage", "android"),
        challenge(
            "Run a persistent service under Termux:Boot plus termux-wake-lock "
            "with crash-restart via a watchdog script",
            2,
            "script passes: kill the service 5 times, watchdog restarts each "
            "within 30s, survives a simulated reboot hook",
            "build", "termux-boot", "watchdog", "persistence"),
        challenge(
            "Set up proot-distro Ubuntu in Termux: install, share storage, "
            "run a systemd-free service stack",
            2,
            "artifact passes: proot boots; nginx plus python serve from the "
            "shared dir; no root required anywhere",
            "build", "proot", "distro", "linux"),
        challenge(
            "Implement battery-aware scheduling in Termux: check battery "
            "level and charging state before heavy jobs",
            2,
            "test passes: job pauses below 20% unplugged, resumes on charger, "
            "full run completes without a drain alarm",
            "code", "battery", "scheduling", "termux-api"),
        challenge(
            "Build a Termux SSH server setup: key-only auth, a fail2ban "
            "equivalent, port-forwarded access from the LAN",
            2,
            "script passes: password auth refused, key auth works, 50 failed "
            "attempts trigger a temporary ban",
            "build", "ssh", "hardening", "termux"),
        challenge(
            "Implement Termux:API integration: battery, location, an "
            "SMS-send guard, and notification actions from scripts",
            2,
            "test passes: battery JSON parsed, location fix acquired, "
            "notification action triggers its callback",
            "code", "termux-api", "android", "scripting"),
        challenge(
            "Compile a native binary for Android aarch64 inside Termux: "
            "toolchain setup, cross-compile, strip, run",
            3,
            "test passes: binary runs on-device; `file` reports aarch64; "
            "stripped size under 2MB; no NDK needed",
            "code", "native", "compile", "aarch64"),
        challenge(
            "Build a Termux-hosted LLM inference setup: llama.cpp server, "
            "GGUF model, wake-locked serving on the LAN",
            3,
            "artifact passes: server answers prompts over the LAN; "
            "tokens/sec measured; survives screen-off via wake lock",
            "build", "llm", "llama-cpp", "inference"),
        challenge(
            "Implement full-disk-encrypted backup of Termux $HOME: encrypted "
            "tar to shared storage plus a restore script",
            3,
            "test passes: backup, wipe $HOME, restore; sha256 of 100 files "
            "match; wrong key makes decrypt fail",
            "code", "backup", "encryption", "termux"),
        challenge(
            "Build a Termux CI runner: git pull, test, notify via Termux:API "
            "on a schedule with boot persistence",
            3,
            "artifact passes: scheduled run executes the suite; failure sends "
            "a notification; survives reboot via the boot hook",
            "build", "ci", "automation", "termux-boot"),
    ],
    "performance": [
        challenge(
            "Profile a slow endpoint with cProfile/py-spy: find the hotspot "
            "and document the top-5 functions",
            1,
            "report names the top-5 functions with cumulative time; flame "
            "graph SVG attached; hotspot reproduced 3 times",
            "research", "profiling", "flame-graph", "python"),
        challenge(
            "Add HTTP caching headers: ETag/Last-Modified, 304 handling, "
            "Cache-Control policies per route",
            1,
            "test passes: second request returns 304; immutable assets carry "
            "1-year cache; private data carries no-store",
            "code", "http-caching", "etag", "headers"),
        challenge(
            "Implement a two-level cache: in-process LRU plus Redis, with "
            "stampede protection via singleflight",
            1,
            "test passes: 100 concurrent misses cause 1 upstream fetch; LRU "
            "eviction correct; TTLs honored",
            "code", "caching", "lru", "stampede"),
        challenge(
            "Write a load script that reports p50/p99 latency and throughput "
            "for 10k requests against a local server",
            2,
            "script outputs p50/p99/throughput for 10k requests; results "
            "reproducible within 10% across 3 runs",
            "code", "load-testing", "latency", "benchmarking"),
        challenge(
            "Optimize a DB query path: add a covering index, rewrite an N+1, "
            "measure before/after with EXPLAIN ANALYZE",
            2,
            "test passes: query time drops at least 5x; EXPLAIN shows an "
            "index-only scan; N+1 eliminated from 101 queries to 1",
            "code", "sql", "indexing", "n-plus-one"),
        challenge(
            "Implement connection pooling and keep-alive tuning for an HTTP "
            "client under 500 concurrent requests",
            2,
            "script passes: 500 concurrent requests complete; pool reuse at "
            "least 95%; no socket-exhaustion errors",
            "code", "pooling", "keep-alive", "concurrency"),
        challenge(
            "Build a CDN-style static asset pipeline: content hashing, "
            "gzip/brotli precompression, immutable URLs",
            2,
            "artifact passes: assets served with content hash; brotli smaller "
            "than gzip; cache hit rate 100% on repeat",
            "build", "cdn", "compression", "assets"),
        challenge(
            "Diagnose a memory-bound batch job: cut peak RSS 50% via "
            "streaming/chunking without changing the output",
            2,
            "script passes: output byte-identical; peak RSS drops at least "
            "50%; runtime within 2x of baseline",
            "code", "streaming", "rss", "optimization"),
        challenge(
            "Hit a p99 under 100ms budget on a 3-service request path: trace "
            "it, budget per hop, fix the violator",
            3,
            "load script passes: p99 under 100ms over 20k requests; per-hop "
            "budget table shows no violator",
            "build", "p99", "latency-budget", "tracing"),
        challenge(
            "Benchmark a numeric hot loop across CPython, PyPy, and a C "
            "extension and justify the winner by measurement",
            3,
            "report shows ns/op for all three with flame graphs; winner "
            "justified by measurement; 5 runs each",
            "research", "benchmarking", "jit", "numeric"),
        challenge(
            "Build a zero-downtime deploy pipeline: blue-green with "
            "health-gated cutover and instant rollback",
            3,
            "artifact passes: deploy during 1k rps load with zero failed "
            "requests; rollback restores in under 30s",
            "build", "deployment", "blue-green", "reliability"),
        challenge(
            "Eliminate GC pauses in a latency-sensitive loop: object reuse, "
            "__slots__, measure the pause distribution",
            3,
            "test passes: p99.9 GC pause under 5ms over 1M iterations; "
            "allocation rate drops at least 80%",
            "code", "gc", "latency", "python"),
    ],
    "memory": [
        challenge(
            "Measure RSS over time for a long-running script with tracemalloc "
            "and find the top-3 allocation sites",
            1,
            "report lists the top-3 sites with bytes and line numbers; "
            "snapshot diff over 1k iterations attached",
            "research", "tracemalloc", "rss", "profiling"),
        challenge(
            "Implement an object pool for a hot allocation: acquire/release, "
            "max size, metrics on hit rate",
            1,
            "test passes: pool hit rate at least 90% under load; "
            "release-after-use enforced; no use-after-release",
            "code", "pooling", "allocation", "reuse"),
        challenge(
            "Set an RSS budget with enforcement: monitor, log, and gracefully "
            "degrade past the limit",
            1,
            "test passes: RSS over budget triggers the degrade path; alert "
            "logged; process never OOM-killed in test",
            "code", "rss", "budgets", "oom"),
        challenge(
            "Hunt a reference-cycle leak: build a repro, fix it with "
            "weakrefs, prove flat RSS over 10k cycles",
            2,
            "test passes: RSS flat within 2% over 10k cycles after the fix; "
            "gc object count stable; weakref invalidation correct",
            "code", "leaks", "weakref", "gc"),
        challenge(
            "Implement a slab allocator for fixed-size records: free-list, "
            "alignment, a fragmentation metric",
            2,
            "test passes: 1M alloc/free cycles; fragmentation under 5%; "
            "use-after-free detected in debug mode",
            "code", "allocator", "slab", "fragmentation"),
        challenge(
            "Build a leak-detection CI gate: fail the build when a soak "
            "test's RSS grows beyond a threshold",
            2,
            "CI script passes: leaking branch fails the gate; fixed branch "
            "passes; threshold documented",
            "build", "leaks", "ci", "soak-test"),
        challenge(
            "Optimize a DataFrame pipeline: downcast dtypes, categoricals, "
            "chunked IO, measure peak RSS",
            2,
            "script passes: peak RSS drops at least 60%; results identical "
            "by hash; dtypes documented",
            "code", "pandas", "dtypes", "optimization"),
        challenge(
            "Implement copy-on-write sharing for large buffers between "
            "processes with mmap",
            2,
            "test passes: 1GB buffer shared; child writes do not affect the "
            "parent; RSS shows a single physical copy",
            "code", "mmap", "cow", "ipc"),
        challenge(
            "Write a custom malloc for an arena use-case: bump allocation, "
            "reset semantics, thread-local arenas",
            3,
            "test passes: 10M allocs; zero fragmentation within the arena; "
            "thread-local arenas race-free under TSan",
            "code", "allocator", "arena", "threads"),
        challenge(
            "Build a heap profiler: sample allocations, attribute them to "
            "call stacks, render a flame graph of bytes",
            3,
            "artifact passes: a known 100MB leak attributed to the correct "
            "stack; flame graph renders; overhead under 5%",
            "build", "profiler", "heap", "flame-graph"),
        challenge(
            "Tune generational GC for a workload: measure pause times across "
            "3 configurations and recommend one",
            3,
            "report compares 3 GC configs with pause histograms; recommended "
            "config justified by p99 pause data",
            "research", "gc", "tuning", "pauses"),
        challenge(
            "Design a memory budget for a 512MB container: per-component caps, "
            "OOM-score tuning, graceful shedding",
            3,
            "artifact passes: load to 600MB demand; shedding keeps RSS at or "
            "under 512MB; critical path never OOM-killed",
            "build", "containers", "oom", "budgets"),
    ],
}

register_topic_pack("chall_web", _PACK)
