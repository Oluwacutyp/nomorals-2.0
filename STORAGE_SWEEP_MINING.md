# Storage module sweep — external mining report

Date: 2026-10-10. Every significant class in `nomorals/storage/` was compared
against best-in-class external implementations (and a few trash ones, which
still held ideas). Sources are cited per class. Gold adopted in this sweep
is marked with →.

The storage layer was already strong (thread-local connections, WAL,
checksummed migrations, content-addressed blobs, Litestream-style
replication). The gaps were all in *what the best tools do around the core*:
WAL health monitoring, Redis-style KV ergonomics, blob range reads/soft
delete, S3 presigned URLs + multipart, real k-means for IVF, FTS5
tokenizer/snippet options, borg-style retention, queue idempotency keys +
cron, point-in-time restore, query-builder joins/aggregates, artifact
aliases + search, DuckDB zero-copy ATTACH, repeatable migrations, and
presentation through the shared style layer.

---

## 1. `Database` (db.py)

**Mined:** kru-sqlite / ice-962464 sqlite-concurrency-safety SKILL.md
(single-writer serializer, `busy_timeout` on every connection,
`BEGIN IMMEDIATE` for writers, never hold read transactions across awaits),
dev.to "Why your SQLite WAL file never shrinks" (`journal_size_limit`
64 MiB cap, periodic `wal_checkpoint(TRUNCATE)` from a maintenance job,
`PRAGMA wal_checkpoint(PASSIVE)` polling + `log - checkpointed` alerting),
bateau84/loom python-database SKILL.md (offload blocking sqlite off the
async loop with `run_in_threadpool`/`asyncio.to_thread`; per-request
connections), salasjustin/kru-sqlite (STRICT tables, `journal_mode` verify).

**Gold adopted →**
- *WAL health monitoring*: `PRAGMA journal_size_limit = 67108864` on new
  connections (dev.to) → `journal_size_limit` constructor arg; new
  `wal_health()` returning `{wal_size_bytes, checkpoint(busy,log,checkpointed),
  lag_pages}` so a scheduler tick can alert when `log - checkpointed` grows.
- *Maintenance checkpoint*: `checkpoint_maintenance()` — TRUNCATE with busy
  retry, meant for the idle tick.
- *Read/write connection split*: `Database` already serializes writers via
  `_write_lock` (matches kru "single-writer policy enforced in app").
- *Async offload*: `AsyncDatabase` — thin wrapper routing every blocking call
  through `asyncio.to_thread` (loom SKILL.md). The thread-local connection
  design makes this safe.
- *Query plan introspection*: `explain(sql, params)` → `EXPLAIN QUERY PLAN`
  rows as dicts (needed for the slow-query story to be actionable).
- *Integrity helpers*: `foreign_key_check()` alongside `integrity_check()`.

**Weak spots noted (not adopted):** STRICT tables would break 86 existing
migrations (no rewrite); `VACUUM INTO` hot backup is covered by
`backup_to()` which is better (online API).

---

## 2. `KVStore` (kv.py)

**Mined:** Redis command surface (`GETEX`/`GETDEL`, `SET NX`, `GETSET`,
`APPEND`, `MGET`/`MSET`, `HSET`/`HGETALL`/`HINCRBY`, `SCAN` cursor,
`RENAME`, `TOUCH`, `EXPIRE`/`PERSIST`/`TTL`), Couchbase
`get_and_touch` (sliding expiry), DiskCache/df-diskcache (`touch`,
`prune` expired), kv-one (namespaces, typed errors), Cloudflare Workers
KV (metadata alongside values).

**Gold adopted →**
- *Sliding expiry*: `touch(key, ttl)` — extend TTL without reading content
  (Couchbase `get_and_touch`, df-diskcache `touch`); `get_and_touch(key, ttl)`.
- *Redis write guards*: `setnx(key, value, ttl)` (SET NX), `getset(key, value)`
  (atomic get+replace), `getdel(key)` (atomic get+delete).
- *String ops*: `append(key, suffix)` for text values.
- *Hashes*: `hset/hget/hgetall/hdel/hincrby/hkeys/hlen` — a JSON object stored
  under one key with per-field ops (Redis hashes), namespaced like everything.
- *Cache-aside*: `get_or_set(key, factory, ttl)` — compute-on-miss with the
  factory called at most once per miss.
- *SCAN cursor*: `scan_cursor(prefix, cursor, count)` returning
  `(next_cursor, [(key, value)])` — bounded pagination instead of LIMIT-only.
- *rename(src, dst)* (Redis RENAME).
- *`dbsize`-style*: `stats()` already existed; added `memory_usage()` estimate.

**Weak spots noted (not adopted):** Lua scripting / transactions across
keys beyond what `_tx()` gives; pub/sub is out of scope for a KV table.

---

## 3. `BlobStore` (blob.py)

**Mined:** git content-addressed object store (fan-out `ab/cd/<sha>`,
immutable objects), restic design doc (content-defined chunking with Rabin
fingerprints, 512 KiB–8 MiB blobs, dedup across snapshots), bup/borg
(chunk dedup), vaultfs (content-addressable dedup + versioning + delete
markers + presigned URLs), Docker layer tar+gzip.

**Gold adopted →**
- *Range reads*: `read_range(sha256, offset, length)` / `stream_range()` —
  HTTP Range semantics for media streaming without loading whole blobs.
- *Soft delete*: `trash(sha256)` → `.trash/` with timestamp,
  `restore_trash(sha256)`, `empty_trash(older_than_s)` — vaultfs delete
  markers; `purge` stays the hard delete.
- *Cross-store sync*: `sync_to(other)` — push missing blobs to another
  `BlobStore`/`S3BlobStore` (both expose the same API), with progress
  callback; the primitive backups and replication need.
- *Upload progress*: `put_file(..., progress=fn)` and `put_stream(...,
  progress=fn)` callbacks (Docker-layer style UX).
- *`describe()`*: one-dict health overview (counts, bytes, dedup ratio,
  trash size) for status surfaces.

**Weak spots noted (not adopted):** full CDC chunking across blob versions
(restic-style) — real value, but needs a chunks table + new migration and
doubles the read path; parked as a documented next step, not silently
dropped.

---

## 4. `S3BlobStore` (s3blob.py)

**Mined:** boto3 (`generate_presigned_url` GET/PUT, `generate_presigned_post`,
multipart via `Upload` with `partSize`, lifecycle rules,
`AbortIncompleteMultipartUpload`), minio-py / light-s3-client (lightweight
SigV4 clients, multipart: Create/UploadPart/Complete/Abort, ListParts),
nanio (presigned URL SigV4 verification, range requests, streaming uploads),
terminalskills s3-storage SKILL.md (lifecycle, versioning, CORS).

**Gold adopted →**
- *Presigned URLs*: `presigned_get(sha256, expires_in)` and
  `presigned_put(key_hint, expires_in)` — pure SigV4 query-string signing,
  no network needed. This is the single most-requested S3 feature
  (direct browser/phone uploads and downloads without proxying bytes).
- *Multipart upload*: `put_file` now switches to multipart
  (CreateMultipartUpload → UploadPart → CompleteMultipartUpload, 8 MiB
  parts, abort-on-failure) above `multipart_threshold` (default 64 MiB) —
  boto3 `Upload` behavior, stdlib only.
- *Range GET*: `get_range(sha256, start, end)` via HTTP Range (nanio).
- *`head_object` metadata*: `content_length` from HEAD already fed
  `BlobInfo.stored`; added `last_modified` passthrough in `info()` meta.
- *`ensure_bucket()`*: create-if-missing via PUT (idempotent setup).

**Weak spots noted (not adopted):** lifecycle/versioning/CORS management —
bucket administration, not blob storage; out of scope.

---

## 5. `VectorStore` (vectors.py)

**Mined:** FAISS paper §5.1 + "Guidelines to choose an index" (IVF:
`nlist ≈ sqrt(N)`, `nprobe` default 8, HNSW `M` 4–64 with `efSearch`
speed-accuracy knob, `IVF1024,PQ` needs re-rank stage; flat is exact and
right under ~100k), sqlite-vss / sqlite-vec (SQLite-native vec0), ChromaDB
(persistent HNSW), SingleStore IVF docs.

**Gold adopted →**
- *Real k-means training*: `build_index()` ran one assignment pass over
  random seeds. Now Lloyd's algorithm with k-means++ seeding and
  configurable iterations (default 10) — FAISS `index.train()` semantics.
  Empty-cluster reseeding included.
- *Centroid persistence*: index state (centroids + member lists) saved to
  `<table>_index` (CREATE TABLE IF NOT EXISTS, lazily) so a restart doesn't
  retrain; `load_index()` / `drop_index()`.
- *Recall measurement*: `evaluate_recall(queries, ground_truth, k)` —
  measures IVF recall@k vs brute force, the FAISS-benchmark workflow.
- *Batch search*: `search_many(vectors, limit, ...)` amortizes `_load()`.
- *Distance mode*: `score_mode="distance"` returns cosine distance
  alongside similarity.
- *`nprobe` validation*: clamped to `[1, nlist]` with a warning (SingleStore
  docs: nprobe cannot exceed nlist).
- *Multi-owner filter*: `owner_ids: set[str] | None` in `search()`.

**Weak spots noted (not adopted):** HNSW graph index — pure-Python HNSW is
slower than the numpy flat scan it would replace; PQ compression — the
embeddings table is small at agent scale. Both documented, not faked.

---

## 6. `FTSIndex` (fts.py)

**Mined:** SQLite FTS5 docs (tokenizers: `unicode61 remove_diacritics 2`,
`porter` stemming, `trigram`; `bm25()` with fixed k1=1.2/b=0.75;
`highlight()`, `snippet()`, column filters `col:term`, NEAR/phrase/prefix
syntax; `fts5vocab` for suggestions), ivan-magda/swift-claw FTS research
(`unicode61 remove_diacritics 2` default for conversational English,
`porter` for stemming, trigram only for CJK/fuzzy), uukjtisa/filet
(trigram <3 chars degrades to full scan — branch query planner),
naveen-devang/procure (per-column bm25 weights, UNINDEXED columns,
drift self-check: one index row per source row), susomejias/rembric
(boolean operators, column-specific search, highlight).

**Gold adopted →**
- *Tokenizer choice*: `FTSIndex(..., tokenizer="unicode61",
  tokenize_args="remove_diacritics 2")` — porter/trigram supported;
  `build_ddl()` generates the `CREATE VIRTUAL TABLE` statement so migrations
  stop hand-writing FTS DDL.
- *Column filters*: `build_match_query` now parses `col:term` facets
  (omnibus F1.1: `author:`, `tag:`) with unknown facets falling through as
  free text.
- *NEAR/phrase builders*: `near_query(terms, distance)`,
  `phrase_query(text)` alongside `build_match_query`.
- *`highlight()`*: per-hit highlighted text via FTS5 `highlight()`.
- *Suggestions*: `suggest(prefix, limit)` over `fts5vocab` (trigram overlap
  idea from the pglite research, stdlib edition).
- *Drift check*: `check_drift(source_count)` — one index row per source row
  (procure self-check).
- *Per-column bm25 weights*: `search(..., weights={col: w})` via
  `bm25(table, w0, w1, …)`.

**Weak spots noted (not adopted):** typo-tolerant search via spellfix1 —
not compiled into stdlib SQLite; trigram+vocab suggestion is the portable
fallback.

---

## 7. `BackupManager` (backup.py)

**Mined:** restic (`check` deep verification, `forget` retention policy,
`copy` between repos, snapshots always full + dedup via CDC, prune),
borgbackup (`--keep-daily/weekly/monthly` retention, `borg check`,
`borg list` formatted output, `borg diff`), Kopia (policies, compression),
kunalganglani SQLite guide (RPO/RTO thinking: periodic exports +
continuous replication as two layers).

**Gold adopted →**
- *Borg-style retention*: `rotate(policy=RetentionPolicy(keep_last,
  keep_daily, keep_weekly, keep_monthly))` — replaces keep-N+one-per-day
  with the full daily/weekly/monthly ladder.
- *`check`*: `verify_all()` — deep-check every backup (restic `check`).
- *Formatted listing*: `format_table()` — borg-`list`-style table through
  the shared style layer (name, age, size, blobs, schema version).
- *Dry run*: `rotate(..., dry_run=True)` reports what would be deleted.
- *Pre-restore safety*: `restore()` refuses to overwrite an existing target
  unless `overwrite=True`, and with `safety_copy=True` (default) snapshots
  the existing target aside first.
- *`estimate()`*: projected size of the next backup before taking it.

**Weak spots noted (not adopted):** encryption-at-rest (restic always
encrypts) — no AES in stdlib and fake crypto is worse than none;
documented as requiring `cryptography` or an encrypted filesystem.

---

## 8. `WorkQueue` (queue.py)

**Mined:** Procrastinate (`queueing_lock` — one queued job per key,
`lock` — one running, periodic tasks with PG advisory locks, retries with
exponential backoff, `job.result()`), TaskTiger (unique tasks, task locks,
periodic tasks, per-task forking), RQ (job timeouts, dead-letter handling),
huey (SQLite mode, periodic tasks), QueueForge (idempotency keys, DLQ
replay, per-type concurrency limits, graceful shutdown), PgQue
(`receive`/`ack`/`nack` with delay).

**Gold adopted →**
- *Idempotency / queueing lock*: `enqueue(..., queueing_lock="report-42")` —
  a second live job with the same lock returns the existing id instead of
  duplicating (Procrastinate `queueing_lock`, TaskTiger unique tasks).
- *Cron*: `schedule_recurring(topic, payload, cron, name)` with a minimal
  stdlib cron parser (minute/hour/dom/month/dow, `*/n`, lists, ranges) and
  `tick_recurring(now)` the worker loop calls — huey/Procrastinate periodic.
- *Lease heartbeat*: `extend_lease(job_id, worker, extra_seconds)` for
  long jobs (RQ timeout story).
- *Cancel*: `cancel(job_id)` moves live jobs to `cancelled`.
- *Result wait*: `wait_for(job_id, timeout)` polls for the stored result.
- *DLQ replay*: `retry_dead(job_id)` — QueueForge replayable dead-letter.
- *Queue depth guard*: `enqueue` raises `QueueFull` when a topic exceeds
  `max_depth` (backpressure instead of unbounded growth).

**Weak spots noted (not adopted):** per-type concurrency limits and
LISTEN/NOTIFY wake-ups — SQLite has no notify; polling stays.

---

## 9. `SqliteReplicator` (replication.py)

**Mined:** Litestream (continuous WAL shipping to S3, point-in-time restore
to any transaction, generations, `litestream replicate` as a sidecar,
restore init containers), LiteFS (single-writer, export + continuous as two
layers), kunalganglani guide (RPO = interval, restore = download + replay +
boot + verify).

**Gold adopted →**
- *Point-in-time restore*: `restore_at(timestamp)` picks the newest snapshot
  `<= t` (honest PITR at snapshot granularity — documented, not oversold).
- *Compressed snapshots*: `compress=True` gzip option (bandwidth is the
  phone's scarcest resource).
- *RPO tracking*: `lag_seconds()` = time since last snapshot; `status()`
  one-dict overview (generations, bytes, lag, last ship error).
- *Pre-restore safety*: `restore()` copies any existing target aside as
  `.pre-restore-<stamp>` before overwriting (Litestream restore-container
  caution).
- *`verify_all()`*: integrity-check every generation (restore must be
  trustworthy).

**Weak spots noted (not adopted):** true WAL-frame streaming (Litestream's
core) — needs a sidecar process tailing the WAL; snapshot granularity is
the documented honesty trade.

---

## 10. `Repository` / `Query` / `InsertBuilder` (repository.py)

**Mined:** Peewee query builder (composable `select().where().join().
group_by().having().order_by()`, `fn.COUNT`, `paginate()`, dynamic
piece-by-piece filters, `.dicts()/.tuples()/.namedtuples()` row shapes),
SQLAlchemy 2.x repository pattern (type-safe generic repos, filter DSLs),
Piccolo (typed `create`/`update`/`delete`).

**Gold adopted →**
- *Query*: `join()`, `group_by()`, `having()`, `distinct()`, `first()`,
  `pluck(column)`, `exists()` execution helpers, `where_in()`,
  `where_like()`, `where_between()`, `where_null()`/`where_not_null()`.
- *Repository*: `aggregate(fn, column, **filters)` (COUNT/SUM/AVG/MIN/MAX),
  `update_where(changes, clause, params)`, `touch(key)` (bump updated_at),
  `paginate_with_total(page, per_page)` (rows + total + pages),
  `first_or_create` / `update_or_create`, `bulk_update(rows, key_field)`,
  `upsert_many` (ON CONFLICT DO UPDATE), `stream(batch_size)` generator for
  large tables.

**Weak spots noted (not adopted):** full ORM relations/backrefs — the module
docstring's "deliberately not an ORM" stands; joins cover the 20%.

---

## 11. `ArtifactStore` (artifacts.py)

**Mined:** MLflow (artifact lineage per run, model registry with
versioning + **aliases** like `champion`/`latest`, tags, `mlflow artifacts
list`), DVC (`.dvc` metafiles, `dvc pull` reproducible fetch, pipeline
stages), WandB (artifact versioning + aliases + lineage graph).

**Gold adopted →**
- *Aliases*: `set_alias(name, artifact_id)` / `resolve_alias(name)` —
  MLflow-registry-style named pointers (`latest`, `champion`); new
  `artifact_aliases` table (migration 87).
- *Search*: `search(type=, creator=, mission_id=, task_id=, metadata={...},
  limit=)` — MLflow `search_runs` for artifacts.
- *Tags*: `tag(artifact_id, key, value)` / `untag` on metadata (MLflow tags).
- *Delete*: `delete(artifact_id, drop_blob=True)` — removes the row and
  decrements the blob refcount (DVC `gc` semantics).
- *Bundle export*: `export_bundle(artifact_ids, dest)` — tar.gz of artifact
  bytes + `manifest.json` (DVC `get`/fetch story).
- *`format_card(art)`*: one-artifact pretty card through the style layer.

**Weak spots noted (not adopted):** cross-artifact diff — content diffing
is a media-layer concern.

---

## 12. `AnalyticsBackend` (analytics.py)

**Mined:** DuckDB expert SKILL.md (lazy loading default, `ATTACH` for
federated SQLite/Postgres/S3-Parquet, `read_parquet` predicate pushdown,
`EXPLAIN ANALYZE`, threads/memory tuning), "From SQLite to DuckDB"
(`ATTACH 'db' (TYPE SQLITE)` — query SQLite directly, zero-copy sync),
DuckDB tricks (time_bucket, `COPY TO parquet`, Arrow zero-copy).

**Gold adopted →**
- *Zero-copy attach*: `DuckDBAnalytics.attach(db)` — `ATTACH '<path>'
  (TYPE SQLITE)` creates live views; no row copying, always fresh
  (the "SQLite + DuckDB blending" pattern). Falls back to copy-sync for
  `:memory:` databases.
- *`profile(sql)`*: `EXPLAIN ANALYZE` output for the slow-query story.
- *Result cache*: `query_cached(sql, params, ttl)` — memoize expensive
  analytical queries with TTL invalidation.
- *`describe(table)`*: column names/types/row count one-dict.
- *`sample(table, n, seed)`*: `USING SAMPLE` reservoir.
- *`histogram(table, column, bins)`*: `histogram()` aggregate helper.
- *`to_csv()`* alongside `to_parquet()`.
- *Streaming*: `iter_query(sql, params, batch)` yields batches for results
  bigger than RAM.

**Weak spots noted (not adopted):** dbt-style model management — a build
tool, not a storage primitive.

---

## 13. `MigrationRunner` / `Migration` (schema.py)

**Mined:** Flyway (V__ versioned + R__ **repeatable** migrations re-applied
when checksum changes, `validate` before every `migrate`, `repair`,
`baseline`, `clean`, `ignoreMigrationPatterns` for future versions,
`outOfOrder`), Liquibase (changesets, contexts), dbmate (plain SQL,
up/down), Alembic (autogenerate, branches).

**Gold adopted →**
- *Repeatable migrations*: `Migration(..., repeatable=True)` (or
  `R__`-style name) — re-applied whenever the source checksum changes,
  always after versioned ones (Flyway R__). For views/indexes/FTS DDL.
- *`target` version*: `apply_all(migrations, target=42)` — migrate up to a
  version (deploy staging).
- *Applied-but-missing detection*: `validate()` now also reports applied
  versions with no matching migration object (Flyway "resolved vs applied").
- *`out_of_order`*: allow applying a lower pending version after a higher
  one was applied (branchy workflows, opt-in).
- *`clean()`*: explicit, double-confirmed dev-only wipe (`CLEAN` typed
  confirmation — Flyway `clean`).

**Weak spots noted (not adopted):** autogenerate diffing (Alembic) — needs
a declared model layer; our schema is hand-written SQL.

---

## 14. router telemetry (router_telemetry.py)

**Mined:** OpenTelemetry metrics (counters, histograms, attributes),
Prometheus client (Counter/Gauge/Histogram, `rate()` over windows),
Langfuse (per-route traces with latency).

**Gold adopted →**
- *Generic instruments*: `record_counter(db, key, delta)`,
  `record_gauge(db, key, value)`, `record_timing(db, key, seconds)` —
  timings keep `{count, sum, min, max}` so `snapshot()` can report
  avg/p95-ish without storing every sample (Prometheus histogram idea,
  exact math on a tiny sketch).
- *Top routes*: `top_routes(db, limit)` sorted by count.
- *Pretty snapshot*: `format_snapshot(db)` — styled table of routes,
  model checks, last error with age (the `nm mind` surface).
- *`reset(db, key)`*: clear one instrument (tests/ops).

**Weak spots noted (not adopted):** true histograms with buckets — the
count/sum/min/max sketch answers avg and rough p95; full HDR is overkill
for router telemetry.

---

## 15. models (models.py)

**Mined:** SQLModel/Pydantic (typed rows, validation), attrs (slotted
classes), Peewee `.dicts()` row shapes.

**Gold adopted →**
- *`MissionRecord`*: the `missions` table had no typed mirror (agents and
  tasks did).
- *`to_json()`/`from_json()`* on all records — symmetric with `to_row()`.
- *`changed_fields(other)`*: field-level diff for update paths.
- *`AgentRecord.is_running`*, `TaskRecord.elapsed` convenience properties.

---

## 16. Presentation (all modules)

**Mined:** the repo's own `nomorals/core/style.py` (themes as named
palettes, semantic roles, `plain` disables color, magenta-not-red errors,
graceful tty/width degradation).

**Gold adopted →** every `stats_snapshot()`-shaped surface that a human
reads gains a `format_*` renderer through the shared style layer:
`Database.format_stats()`, `KVStore.format_stats()`,
`BlobStore.format_stats()`, `WorkQueue.format_status()`,
`BackupManager.format_table()`, `ArtifactStore.format_card()`,
`VectorStore.format_stats()`, `MigrationRunner.format_status()`,
`router_telemetry.format_snapshot()`. No inline ANSI anywhere; `plain`
theme support inherited.
