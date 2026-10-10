# INFRA MINING — PERSISTENCE (Section D)

Mined 2026-10-09 before touching `nomorals/storage/db.py`, `schema.py`,
`migrations.py`, and the kv_store layer. Sources: sqlite.org recipe, Litestream
tips, dev.to corruption post-mortem, dbwarden migration-locking docs,
aurral #912 (concurrent startup migrations), warrioriq + scooper-cms
concurrency commits, local-sql-agent StaticPool fix, codex sqlite skill cards.

## Gold from the best builds

1. **The open sequence is a fixed recipe, run on every connection:**
   `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`,
   `cache_size=-200000`, `temp_store=MEMORY`, `foreign_keys=ON`,
   `mmap_size=268435456`. Our `Database._configure` already does all of
   these — keep, never regress.
2. **`synchronous=NORMAL` under WAL is safe** (only exposure is losing the
   last commits on *power loss*, not app crash). Litestream recommends it.
3. **Busy is handled, never swallowed.** Ignoring `SQLITE_BUSY` is on
   SQLite's own corruption list. Our `_with_lock_retry` (bounded, jittered)
   matches the recipe; keep it under the busy_timeout (ours absorbs the
   multi-process case).
4. **Backups use the online backup API** (`sqlite3.Connection.backup`),
   never `shutil.copy` on a live WAL file. Our `backup.py` already does
   this. Gap found: `Database` itself has no `backup_to()` convenience,
   so ad-hoc callers reimplement file copies — add it.
5. **Validate the backup you just made.** `VACUUM INTO` is transactional on
   the source, but an unplanned shutdown can still leave the output
   incomplete. `backup.py` verifies (good); `Database.backup_to` must run
   `integrity_check` on the copy.
6. **Migration locking across processes:** dbwarden uses BEGIN IMMEDIATE as
   the SQLite-native advisory lock; aurral #912 rechecks schema version
   *inside* an immediate transaction so concurrent starters serialize and
   losers observe history instead of re-applying. Our `MigrationRunner`
   has NO cross-process lock: two `Database` objects on one file (bot +
   CLI, bot + cron) can both see "pending" and double-apply a data
   migration. FIX: file lock (`<db>.migrate.lock` via `fcntl.flock`) +
   re-check pending inside the lock.
7. **`PRAGMA optimize` is a stated recipe step:** short-lived connections
   run it before close; long-lived run `0x10002` at open + periodic, and
   ALWAYS after schema change / CREATE INDEX. We never run it — the query
   planner flies blind after 85 migrations of index creation. FIX: run
   after `migrate()`, expose `optimize()`.
8. **`wal_autocheckpoint` tuning** (recipe uses 1000). We don't set it;
   add the knob, default 1000.
9. **Restore hygiene (scooper-cms):** after restoring, delete stale
   `-wal`/`-shm` sidecars or they replay pre-backup data over the restore.
   Our `restore()` must guarantee this — verify.
10. **One writer discipline, documented.** Thread-local connections +
    in-process write lock is correct (the local-sql-agent StaticPool
    disaster — one shared connection across FastAPI threads — is exactly
    what our design avoids). Document the model, don't redesign it.
11. **Repair path for edited migrations.** The migration-84 incident
    (checksum mismatch on the local Devon DB, hand-repaired) proves the
    runner's only response to a mismatch is "refuse to boot". A refuse is
    correct as the *default*, but an operator needs a first-class
    `repair()` that records acceptance of the new checksum (logged,
    explicit) instead of hand-editing `schema_migrations`. FIX: add it.
12. **Checksum the real artifact.** dbwarden verifies SQL checksums before
    changes; our `Migration.checksum` hashes `fn.__doc__` for fn
    migrations — a docstring edit trips the guard without any behavior
    change (false positive class). FIX: hash `inspect.getsource(fn)`,
    stored in a new `source_checksum` column; legacy rows fall back to the
    doc-based check for backward compatibility.
13. **Baseline/stamp command.** zongsoft's `check`/`apply`/`status`
    lifecycle and the common `stamp` operation: marking existing
    migrations applied without running them (fresh code against a
    pre-existing prod DB). Our runner has no baseline — add it.
14. **Never touch sidecars while open.** Our `open_database` quarantine
    already moves `-wal`/`-shm` *before* closing (correct — closing first
    would let SQLite delete them). Keep this ordering.

## Gold from the trash builds (what NOT to do)

- **Shared connection across threads** (local-sql-agent): raced commits,
  `cannot commit -- no transaction is active`. Thread-local is the fix —
  we have it.
- **WAL attempted, not assumed** (warrioriq): on network filesystems WAL
  fails; they log once and fall back. Our `_configure` swallows the
  pragma failure silently at debug — at least *record* the actual
  journal_mode achieved. FIX: `Database.journal_mode()` introspection +
  warn when WAL requested but not active.
- **Copy-then-rename restores break replication** (clace/Litestream): do
  `VACUUM` in place; restores come from the replica. Our restore writes
  to target then replaces — acceptable for a single-node embedded DB,
  but must nuke sidecars (see #9).
- **`--watch` that wasn't a command / unverified "fixed" claims** —
  process gold, not code: every persistence claim below is verified by a
  test that actually runs.

## Gaps in our current layer (all cured in this build)

| # | Gap | Cure |
|---|-----|------|
| 1 | No KV abstraction: 31 files hand-roll `_kv_get/_kv_set` with inconsistent upserts, no TTL, no namespacing | `nomorals/storage/kv.py`: `KVStore` — namespaces, TTL, JSON typing, batch ops, CAS, counters, prefix scan/delete, lazy expiry + sweeper |
| 2 | `kv_store` table has no expiry column | Migration 86: `expires_at REAL` + index |
| 3 | Concurrent `migrate()` from two processes can double-apply | File-lock + re-check inside lock in `MigrationRunner.apply_all` |
| 4 | Checksum mismatch = refuse-to-boot with no operator repair path (migration-84) | `MigrationRunner.repair()` — explicit, logged checksum acceptance |
| 5 | fn-migration checksum hashes docstring (false positives) | `source_checksum` column hashing `inspect.getsource(fn)`; legacy fallback |
| 6 | No `PRAGMA optimize` anywhere | `Database.optimize()`; auto-run after `migrate()` |
| 7 | No `wal_autocheckpoint` knob | `Database(..., wal_autocheckpoint=1000)` |
| 8 | No ad-hoc backup entry on `Database` | `Database.backup_to(path)` via online backup API + integrity check of the copy |
| 9 | No slow-query observability | `slow_query_threshold_s`; warn-log + stat on breach |
| 10 | No baseline/stamp for existing DBs | `MigrationRunner.baseline(version)` |
| 11 | WAL request can silently fail (network fs) with only a debug log | warn + `journal_mode()` introspection |
| 12 | `stats_snapshot` lacks page/freelist/WAL size | add `page_count`, `freelist_count`, `wal_size_bytes` |

## Design decisions

- **Full capability, no deployment ceiling.** Profile-gating is the
  runtime's job; the storage layer is written once, at maximum.
- **Backward compatible by construction.** Raw-SQL kv callers keep
  working (new column is nullable; lazy expiry in KVStore only).
  Existing `schema_migrations` rows validate with the legacy checksum
  path — no forced re-baseline.
- **Forward-only remains the default.** `rollback()` stays explicit;
  `repair()` and `baseline()` are operator-invoked, never automatic.
