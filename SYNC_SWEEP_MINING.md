# SYNC Sweep — External Mining Report

Module: `nomorals/sync/` (5 files: `__init__`, `store`, `engine`, `http_peer`, `errors`).
Date: 2026-10-10. Written BEFORE any implementation, per sweep method.

Every significant class was compared against the best implementations outside
this repo. "Best" = mined for gold to merge. "Trash" = mined to learn what NOT
to do. Each section ends with the gaps this sweep will fill.

---

## 1. `SyncStore` — versioned KV + tombstones + seq cursor

### How the best do it

**CRDT LWW-Element-Set literature (senicko/crdt, imatkovic/crdtkit, rebar):**
- The lookup rule is identical to ours: winner = max by `(timestamp, node_id)`
  — a total order, deterministic convergence. We're aligned here.
- **The gold we lack #1 — Hybrid Logical Clocks.** A dev.to field report
  ("The last-write-wins clock was wrong, and I wrote it", andystanly) documents
  the exact failure our store has: plain wall-clock `updated_at` breaks under
  phone clock skew — a device 9 hours behind loses every conflict even when its
  edit happened last. The fix is 3 rules: every row carries `(hlc_ts,
  hlc_count)`; local write = `tick` (advance counter, never move backward);
  remote write = `receive` (`ts = max(local, remote, now)`, counter bump on
  ties); compare `(ts, count)` lexicographically, `device_id` breaks ties.
  TopGun's HLC doc and the `byearlybird/crdt` (Starling) library use the same
  shape; declarative_sqlite builds its whole sync framework on HLCs.
  **This is the #1 correctness upgrade for this module** — our user syncs a
  Termux phone against a cloud hub; phone clocks drift and users change them by
  hand.
- **The gold we lack #2 — tombstone garbage collection.** Ours grow forever.
  crdtkit: time-based expiry with a conservative grace period (e.g. 90 days) +
  monitoring of the oldest unsynced replica is the production balance;
  coordinated GC (all replicas ack a checkpoint) is safe but defeats CRDT
  benefits; checkpoint-and-rebase forces stale replicas to full-sync.
  peeringdb-plus purges tombstones after a 30-day cycle with the cutoff bound
  as UTC; binstash's chunk-store GC runs dry-run by default, apply is explicit.
  **We need `gc_tombstones()` with dry-run default + a documented,
  conservative grace period.**
- **The gold we lack #3 — per-field LWW.** charlietap/crdt-demo's SQLite CRDT
  schema flattens per-field timestamps (`name`, `name_timestamp`, `phone`,
  `phone_timestamp`, `tombstone`, `tombstone_timestamp`) — each mutable value
  is its own LWW register. TopGun: field-level LWW means Alice renaming a task
  and Bob moving its start date BOTH win. mdiener21's kanvana plan: per-field
  HLCs in a single JSON "fat column" (`_clocks`) instead of side tables.
  **Our record-level LWW loses concurrent edits to different fields of one
  JSON value — the dominant real conflict for settings/memory-style KV.**

**CouchDB/PouchDB replication:** per-write sequence numbers as the changes-feed
cursor; checkpoints written per batch (resumable); `_revs_diffs` to avoid
shipping what the target already has. Our seq cursor already mirrors the
changes-feed idea; we lack per-batch checkpointing in the engine (see §2).

**AWS AppSync Delta Sync:** Base table (source of truth) + Delta journal table
with TTL; `_deleted` flag so offline clients learn about deletes; a global
catch-up query covers clients offline past the TTL. Our tombstones are the
`_deleted` flag; we lack the TTL/GC and the catch-up story.

**Anti-entropy (Dynamo/Cassandra via the system-design curriculum):** gossip +
  Merkle-tree range comparison + read repair + hinted handoff. CRDTs give a safe
  merge; anti-entropy delivers the updates. **We have no divergence detector**
  — a cheap `digest()` (hash over keyspace) + `diff_keys()` fills the "did we
  actually converge?" gap.

### Trash mined (what NOT to do)
- **Pure OR-Sets**: every element carries add/remove tags; needs aggressive GC.
  Too heavy for our shape.
- **2P-Set**: remove wins permanently — a deleted key can never come back.
  Wrong for a user-facing KV store.
- **Server-assigned versions only** (dev.to/liaqat_ali): simpler, but you lose
  the ability to order edits made while fully offline. We stay with HLC.

### Gaps → plan
1. Add `(hlc_ts, hlc_count)` columns + store-level HLC clock (`tick`/`receive`),
   conflict key becomes `(hlc_ts, hlc_count, device_id)` with legacy
   `(updated_at, device_id)` fallback for HLC-less rows.
2. Add `clocks` fat column (per-field HLC + field tombstones); `apply()` merges
   field-wise so concurrent edits to different fields both survive.
3. Add `gc_tombstones(older_than_s, apply=False)` — dry-run default, 90-day
   conservative default, honest docs about the resurrection caveat.
4. Add `digest()` + `diff_keys()` divergence tools.
5. Add `subscribe()`/`unsubscribe()` change notifications (feeds AutoSync),
   `put_many()`, `list_since_seq(..., limit)`, `count_since_seq()`.

---

## 2. `SyncEngine` — push/pull replication, per-peer progress

### How the best do it

**CouchDB replication protocol (protocol.rst):**
- Checkpoint = last source sequence ID, saved on the *target* in
  `_local/<unique-id>` — survives interruption, resumes exactly.
- **Per-batch checkpointing**: "After the group of revisions is stored on the
  target, save the new checkpoint." Our engine saves progress once at the end
  of `sync()` — a mid-push crash re-pushes everything (wasteful) and, worse,
  our cursor math on partial application can strand records.
- `_revs_diffs`: don't send what the target already has (bandwidth).

**PouchDB (`replicate`/`sync` options):**
- `live: true` continuous replication + `retry: true` auto-retry on failure;
  user-space exponential backoff with reset on success is the documented
  pattern; `back_off_function` is customizable.
- `batch_size` (default 100) / `batches_limit` — bounded work units.
- One-way modes: `replicate.to()` (push) / `replicate.from()` (pull). **We
  only have two-way `sync()`.**
- Rich events: `change`, `paused`, `active`, `denied`, `complete`, `error` —
  the surface live progress UIs are built on. We emit one `sync.completed`.

**Offline-first mobile skill (urlsandcodes/agent-skills-library):**
- Transport resilience = truncated exponential backoff + full jitter +
  reachability triggers + delta pagination. Our `HttpSyncPeer` retries inside
  one call, but the engine has no retry/backoff around a failed phase.

### Trash mined
- pouchdb-persist: re-implements retry outside the library; upstream verdict is
  "do not use — use live replication settings instead." Lesson: build
  retry/continuity INTO the engine, not as an external wrapper script.

### Gaps → plan
1. `direction` parameter: `"push" | "pull" | "both"` (PouchDB-style one-way).
2. Chunked push with per-chunk checkpoint save (CouchDB-style); chunked pull
   apply with periodic cursor save → crash-resume both directions.
3. Engine-level retry with exponential backoff + jitter on transient peer
   failures (configurable attempts).
4. `dry_run` / `preview()` — "what would sync do" before doing it.
5. `on_progress` callback + granular bus events (`sync.push_progress`,
   `sync.pull_progress`) so real progress UIs can be built.
6. `sync_history` table: every run recorded (`history()`, `last_run()`);
   `status()` gains last-run summary.
7. `AutoSync`: live/continuous mode — background thread, store-change trigger
   (debounced), interval polling, backoff on error, `start()`/`stop()`/
   `trigger()`/`stats()`.
8. `sync_all({peer_id: peer})` multi-peer convenience with per-peer error
   capture.

---

## 3. `HttpSyncPeer` — JSON-over-HTTP peer (stdlib urllib)

### How the best do it
- **EteSync/Etebase journals**: versioned change log, `last_sync_tag`
  cursors, per-entry actions (create/modify/delete) — the journal IS the wire
  protocol. Our `since_seq`/`since_ts` pull params already rhyme with this.
- **CouchDB `_changes?feed=longpoll`**: the pull side can block until changes
  arrive instead of polling. (Our hub server has no long-poll endpoint and
  lives in another module — out of scope; note as follow-up, don't fake it.)
- Standard HTTP hygiene the best peers all do: `Accept-Encoding: gzip`,
  `User-Agent` identification, per-operation timeouts, byte/transfer
  accounting, explicit health probes.

### Trash mined
- Peers that retry HTTP 4xx (auth/validation) as if they were transport
  blips. Ours already translates HTTP errors without retry — keep that, but
  sharpen the taxonomy (below).

### Gaps → plan
1. Error taxonomy: `SyncAuthError` (401), `SyncHubError` (other 4xx/5xx with
   `.status_code`/`.detail`), `SyncConnectionError` (transport exhausted) —
   all subclass `SyncError` (backward compatible).
2. `ping()` health probe (cheap `limit=1` pull).
3. gzip `Accept-Encoding` + transparent decode; `User-Agent: nomorals-sync`.
4. Transfer stats: bytes sent/received, requests, last latency.
5. `describe()` (base URL + redacted auth presence) and `__repr__`.

---

## 4. `errors.py` — one base class

Best practice (PouchDB, AppSync, offline-first skill): distinguish
**transient** (retry) from **permanent** (auth, bad request) failures at the
type level so retry policy can be correct. One flat `SyncError` forces string
matching. Plan: `SyncError` base + `SyncConnectionError` (transient),
`SyncAuthError` (permanent, 401), `SyncHubError` (permanent-ish, hub answered
with an error; carries `.status_code`, `.detail`). `SyncResult.ok`/`error`
surfaces failures without exceptions where appropriate.

---

## 5. Presentation — how sync should LOOK/FEEL

Best-in-class references: Syncthing GUI (per-folder state, per-device progress,
last-seen, conflicts), Dropbox menu-bar sync states, PouchDB `change` events
driving live progress bars. Our module's entire presentation surface is
`SyncResult.to_dict()` — functional, not god-tier.

Plan (new, all additive):
- `SyncResult.format(style="rich"|"plain"|"compact")`: themed multi-line
  report — pushed/pulled/conflicts, throughput (records/s), duration,
  per-phase lines, error rendering. Themes are glyph sets, not hardcoded
  aesthetics (per the user's style-agnostic architecture principle).
- `SyncEngine.format_status(peer_id, style)`: human status card — keys,
  pending push, cursors, last sync ("12s ago"), last run outcome.
- `SyncPreview.format(style)`: dry-run report — keys that would move each way.
- Progress callback + bus events at chunk granularity for live UIs.

---

## What this sweep will NOT do
- No long-poll/subscription server endpoint (hub server is another module's
  file; flagged as follow-up).
- No per-record ack protocol change (hub `/sync/push` returns a count; wire
  format frozen). Chunk-level checkpointing gives crash-resume without it.
- No deletions, no parallel systems: every upgrade extends the existing
  classes; `SyncRecord.wins` keeps its legacy fallback so old tests and the
  hub keep working.
