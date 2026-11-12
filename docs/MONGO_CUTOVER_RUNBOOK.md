# Mongo cutover runbook — Devon on EC2

Status: **READY to flip, never flipped.** The flip is a separate operator
run, gated on the parity record in §2. Nothing in this file flips
anything — every command below is a check, a dry run, or the flip itself
run by the owner's hand on EC2.

EC2 facts this runbook assumes (do not re-derive them here): Windows
Server 2022, 2 vCPU / 2 GB RAM, 50 GB disk; Devon home
`C:\Users\Administrator\.nomorals`; repo `C:\devon\nomorals-2.0`;
SQLite file `C:\Users\Administrator\.nomorals\data\nomorals.db`.
Run the `nm` commands from the repo directory. All PowerShell blocks
below run in an elevated PowerShell on EC2.

## 1. What the flip is

Devon's storage has a DocumentStore seam (`nomorals/storage/docstore.py`).
`storage.backend` selects the rung: `"sqlite"` (default — every domain on
the SQLite file) or `"mongo"` (ported domains serve from MongoDB when it
is reachable, degrading loudly to SQLite when it is not; boot never
raises on a bad Mongo). The flip is a config change
(`[storage] backend = "mongo"`), applied atomically by `flip_config` and
reloaded through `load_settings` with an effective-value assertion — an
env override that would defeat the file (`NM_DB_BACKEND`) is named by
the flip and must be cleared first.

What the flip does NOT change:

- **Pinned domains stay on SQLite whatever the backend** — games-arena
  (settlement is one atomic state graph; the seam offers atomic
  single-document ops only) and memory-recall (memories, memory_archive,
  memory_links, mood_state, mood_history, embeddings). The SQLite file
  therefore stays alive, migrated and backed up on every install.
- **Residue tables' live paths stay native SQL** — 55 tables are recorded
  in `carry.PORT_RESIDUE` with per-domain reasons (each ports with its
  own domain in a later pass). Their DATA still carries to Mongo via P2;
  only the live read/write path stays on SQLite.
- **The scheduler's live path stays native SQL** — its claim book is
  ported (`storage/schedule_claim.py`), but the live scheduler's claim
  is single-process by design and seven other modules read/write
  `schedule_jobs` / `schedule_runs` natively, so a claim-only seam
  wiring would fork the record between two stores.
- **Carry never writes the source.** Rollback is a config flip back to
  `"sqlite"`; the Mongo copy stays in place.

## 2. The parity criterion (recorded, never weakened)

From `carry.run_parity_gate` (`nomorals/storage/carry.py`, mapping
`carry-v1`): PASS requires, for **every mapped collection** — equal
counts, equal ordered canonical SHA-256 checksums, no divergent ids
from the comparator, and no field-level mismatch in a deterministic
sample (default 25). Collections over 50,000 docs on either side use
the streaming comparator (same verdict contract, bounded memory).
The verdict is persisted in `carry_parity_records`
(`parity_gate_status` is the exact check); **no record is not a pass**.

### Gate evidence, 2026-11-12 (this lane, current code)

- **Gate 1 (pre-flip): PASS — 124/124 collections**, mapping
  `carry-v1`, 124 carried, 30 excluded. Seed: KV (2 keys),
  `failures` via FailureAnalyzer (3 rows), conversations (1 + 1
  message), `news_items` via NewsAgent (2), one generic row in the
  residue table `goals`, one in the pinned table `memories`.
  Evidence: `~/workspace/mongo-cutover-verify/gate/gate_result.json`,
  script `~/workspace/mongo-cutover-verify/gate_run.py`.
- **Source untouched:** per-table SHA-256 checksums of `failures`,
  `kv_store`, `goals` identical before carry, after carry, and after
  re-carry.
- **Seam path live:** after attaching Mongo as active, 2 further
  failures recorded through FailureAnalyzer were visible through the
  analyzer on the Mongo rung (5 total), and the re-carry did NOT
  prune them.
- **P6 rehearsal:** `tests/test_mongo_p6.py` 28/28 green on current
  code, including `test_full_cutover_cycle` (carry → gate → flip →
  smoke → backup → restore → rollback, SQLite source proven
  untouched). Log:
  `~/workspace/mongo-cutover-verify/run2/p6.log`.
- **Failures conformance:** `tests/test_mongo_p3_conformance.py::
  test_failures_conformance` green on all three rungs (native /
  sqlite_doc / mongo), including the raw carry-v1 document shape.
  Log: `~/workspace/mongo-cutover-verify/run1/failures-conformance.log`.

### ⚠️ Do NOT re-run the gate after the flip

The gate compares a **frozen source against a live target**. After the
flip, new writes legitimately exist only in Mongo — a post-flip
re-gate MUST fail on any collection with new writes. This lane proved
it deliberately: gate 2 failed on `failures` only, source 3 vs target
5, `only_in_target` exactly the 2 seam-written ids, `differing` empty,
`field_mismatches` empty. That is the expected behaviour, not a defect
— and the programme's own P6 rehearsal never re-gates post-flip.
Post-flip proof is the **smoke test** (§5), never a second gate run.
Run the gate once, pre-flip; the recorded PASS is the flip's evidence.
(`parity_gate_status` reports the LATEST record, so a post-flip
re-gate would overwrite the PASS with an expected FAIL — do not run
it.)

## 3. EC2 readiness checklist

Settle each line before the flip. Anything marked NO is a blocker —
fix it first, do not flip around it.

### 3a. Is mongod installed, and which version?

```powershell
Get-Command mongod -ErrorAction SilentlyContinue
mongod --version
```

Expected: a `mongod` binary present (MongoDB 7.0.x). **NO → blocker:**
install MongoDB 7.0 on EC2 first (the lane's throwaway mongod lives
at `~/workspace/tools/mongodb/bin/mongod` on the build box — it is NOT
on EC2 and is not the install).

### 3b. Is mongod running, and on which port?

```powershell
Get-Process mongod -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 27017 -State Listen -ErrorAction SilentlyContinue
```

Expected: one `mongod` process, listening on 127.0.0.1:27017 (or the
port in `storage.mongo_uri` / `NM_MONGO_URI`). **NO → blocker:** start
it (see §3e for the launch shape) before any `nm db` command that
needs a live target.

### 3c. Standalone or replica set?

```powershell
mongosh --quiet --eval "rs.status().ok"
```

Expected: `1` — a single-node replica set named `rs0`. The preflight
refuses a standalone mongod (`replica_set` check): the carry needs
change-stream-free replica-set semantics and the flip targets a
replica-set URI (`mongodb://127.0.0.1:27017/?replicaSet=rs0`).
**`rs.status()` errors / no `rs0` → blocker:** initiate it:

```powershell
mongosh --quiet --eval "rs.initiate()"
mongosh --quiet --eval "rs.status().members[0].stateStr"
```

Wait for `PRIMARY` before proceeding. (If `mongosh` is not on EC2,
the preflight in §4 settles this question itself — it checks
`replica_set` directly.)

### 3d. Auth mode

```powershell
Get-CimInstance Win32_Service -Filter "Name LIKE '%mongo%'" | Select-Object Name, PathName
```

The service `PathName` shows the launch args / config file. Read the
config file's `security.authorization` value. Expected for this
install: **no auth** (local-only mongod on 127.0.0.1) — if auth IS
enabled, the URI in Devon's config must carry credentials, and the
flip must be run with them; do not flip with an unauthenticated URI
against an authenticated mongod (preflight's `target_reachable`
check will refuse, which is the safe outcome — read the refusal
before forcing anything).

### 3e. Data dir + disk free + WiredTiger cache cap

```powershell
# data dir: from the service PathName / config file dbPath (step 3d)
Get-PSDrive C | Select-Object Used, Free
(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace
```

Expected: data dir on C: with free space comfortably above the SQLite
file's size (the Mongo copy is roughly the same size as the mapped
tables) plus headroom. **Then the cache cap — this one matters:**
the preflight refuses unless the WiredTiger cache is capped at
**0.5 GiB** (`MAX_MONGO_CACHE_BYTES`). An uncapped mongod OOMs the
2 GB host. The launch must carry it, e.g.:

```powershell
mongod --dbpath C:\data\db --bind_ip 127.0.0.1 --port 27017 --replSet rs0 --wiredTigerCacheSizeGB 0.5
```

or the equivalent `storage.wiredTiger.engineConfig.cacheSizeGB: 0.5`
in the config file. **No cap → blocker:** stop mongod, relaunch
capped, re-check §3b–§3c.

### 3f. Does Devon's config point at it?

```powershell
Get-Content C:\Users\Administrator\.nomorals\config.toml | Select-String -Pattern "storage|backend|mongo" -Context 0,4
Get-ChildItem Env: | Where-Object { $_.Name -like "NM_MONGO*" -or $_.Name -eq "NM_DB_BACKEND" }
```

Expected before the flip: `[storage] backend = "sqlite"`, and either
no `mongo_uri` (the default `mongodb://127.0.0.1:27017` is used) or a
`mongo_uri` naming the mongod from §3b with `?replicaSet=rs0`.
**Env overrides defeat the file** — `NM_DB_BACKEND`, `NM_MONGO_URI`,
`NM_MONGO_DB`, `NM_MONGO_SHADOW` override the config; if any is set,
note it now, because the flip's effective-value assertion will name
it and the flip will not take effect until it is cleared or set to
the intended value.

### 3g. The db-stall remainder (unproven, from the 2026-11-12 incident)

The 300s SELECT dwell in `fetchall` was never reproduced off EC2
(indexed 0.02s, 1M rows 0.65s, lock contention raises at 31.99s).
Before the flip, record these on EC2 — they settle whether the
remainder is a stalled disk or watchdog misattribution:

```powershell
Get-Item C:\Users\Administrator\.nomorals\data\nomorals.db* | Select-Object Name, Length
python -c "import sqlite3; c=sqlite3.connect(r'C:\Users\Administrator\.nomorals\data\nomorals.db'); print('journal:', c.execute('PRAGMA journal_mode').fetchone()); print('news_items:', c.execute('SELECT COUNT(*) FROM news_items').fetchone())"
```

Plus, from a Python prompt with a 10s wall: open the DB and run
`BEGIN IMMEDIATE` — it must return promptly, not hang. A slow
`BEGIN IMMEDIATE`, a WAL/`-shm` far larger than the DB, or a
near-full disk points at the EC2 disk, not the code.

## 4. The flip sequence

Run in order. Each step prints its verdict — read it before the next.

```powershell
cd C:\devon\nomorals-2.0
nm db cutover status        # parity gate record + residue dispositions + recent runs
nm db cutover preflight     # MUST print READY; any FAIL line is a refusal, not a warning
nm db cutover run --dry-run # walks preflight + carry (dry) + informational gate; flips NOTHING
nm db cutover run           # the flip: carry + recorded gate + config flip + smoke
```

`nm db cutover run` refuses (exit 1) unless preflight is green and
the recorded gate passes. If it refuses, read `refused_at` and the
step table — do not re-run it with different flags to get around a
refusal.

Dispositions (already recorded in code for all 55 residue tables —
`carry.PORT_RESIDUE` — so `nm db cutover dispositions` should report
no missing dispositions; if it lists any, record it before the flip):

```powershell
nm db cutover dispositions
nm db parity --status       # latest recorded gate verdict
```

## 5. Post-flip proofs (before calling it done)

- `nm db cutover status` — parity gate: **PASS** (the pre-flip
  record from §4; see the §2 warning — do not re-run the gate).
- Storage status shows Mongo LIVE (the backend surface reports
  `active == "mongo"`; pinned domains still list games-arena and
  memory-recall as SQLite-pinned).
- KV round-trip: set a value through Devon's normal operation,
  restart Devon, read it back — it must survive (the KV port is the
  first domain on the seam).
- Conversations round-trip: send a message, check the history shows
  it after a restart.
- A failure recorded after the flip (any tool error) is visible in
  the failure ledger — it rides the seam now, not the SQLite file.
- Scheduler jobs still fire: the scheduler's live path is native
  SQL by design (§1) — a missed tick after the flip is a scheduler
  problem, not a Mongo problem; check the job log, not the store.
- Pinned proof: a fresh memory write lands in the SQLite file and
  the Mongo copy does not move (the live path never consults the
  seam for pinned domains).

## 6. Rollback

```powershell
nm db cutover rollback
```

Flips the config back to `"sqlite"`; the Mongo copy is left in place
(carry never wrote the source, so the SQLite file is the pre-flip
state plus whatever the native paths wrote while Mongo was active —
pinned and residue domains never left it). After rollback, restart
Devon and re-run the §5 proofs against the SQLite rung. A second flip
later is a fresh `nm db cutover run` (new carry + new gate record).

## 7. If the flip refuses

- `target_reachable` FAIL → §3b (mongod not running / wrong port).
- `replica_set` FAIL → §3c (standalone, or `rs0` not initiated).
- `writable_primary` FAIL → no PRIMARY elected — wait, then
  re-check `rs.status()`.
- `cache_capped` FAIL → §3e (relaunch capped at 0.5 GiB).
- Missing dispositions → `nm db cutover dispositions` lists them;
  record each with `--record STORE --as DISPOSITION` (the code
  already records all 55 — a missing one means the live DB has a
  residue store the code does not know).
- Gate FAIL → read the per-collection divergences in the parity
  record (`nm db parity --status`); a FAIL is a report, never a
  reason to weaken the gate — fix the divergence (usually a stale
  target: re-run the carry) and re-gate **pre-flip**.
