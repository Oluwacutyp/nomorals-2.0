# Phase 10 Mining Report: Cross-Cutting Systems

## Surveyed

### Storage (`nomorals/storage/`)
- `kv.py` (473 lines): KVStore with namespaces, TTL, CAS, atomic ops. 31 sites migrated. Solid.
- `migrations.py` (3139 lines): 86 migrations, hardened. 
- `db.py` (788): Database wrapper.
- `backup.py` (576): online backups.
- `blob.py`, `s3blob.py`: blob storage.
- `vectors.py` (445): vector storage.
- `queue.py` (372): persistent queue.
- `replication.py` (291): replication.
- `analytics.py`, `artifacts.py`, `repository.py`, `fts.py`: specialized stores.

**Gold:** The KVStore migration unified 31 call sites. The migration system is robust.
**Gaps:** None major. The storage layer is mature.

### Scheduler (`nomorals/scheduler/`)
- `scheduler.py` (2167): RFC 5545, concurrent dispatch, DLQ, missed-fire policies.
- `recurrence.py` (670): recurrence rules.

**Gold:** Full overhaul done. Concurrent dispatch, blackout dates, jitter.
**Gaps:** None major.

### Triggers (`nomorals/triggers/`)
- `engine.py` (667): TriggerEngine with bus attachment, scheduler wiring.
- `saved_search.py` (775): saved searches.
- `models.py`, `store.py`, `sources.py`, `actions.py`, `webhook.py`.

**Gold:** Engine attaches to bus at boot, fires on events.
**Gaps:** None major.

### Core (`nomorals/core/`)
- Error system: `error_system.py`, `error_intelligence.py`, `error_doctor.py`, `self_heal.py`, `selfheal.py`, `incidents.py`, `budgets.py`, `degradation.py`
- `observability.py` (850): metrics, tracing, health.
- `events.py` (786): event bus.
- `policy.py` (1362): capability system.
- `config.py` (1758): hot-reload config.

**Gold:** Comprehensive error intelligence with self-healing. Policy system is mature.
**Gaps:** None major.

### Security (`nomorals/security/`)
- `dnsleak.py`: DNS leak detection.
- `exif.py`: EXIF stripping.
- Tools registered in `nomorals/tools/security.py`.

**Gold:** Real implementations, not stubs.
**Gaps:** Could use: secret scanning, dependency audit. Minor.

### Connectors (`nomorals/connectors/`)
- `base.py`: Connector ABC with lifecycle, auth methods.
- `registry.py`: connector registration.
- 30+ connectors: plaid, exness, spotify, github, etc.

**Gold:** Unified framework, real implementations.
**Gaps:** None major.

### Native (`nomorals/native/`)
- C++ accelerators: vecsim, mlptrain, bpe, memextract.
- Pure-Python fallbacks with parity testing.

**Gold:** Phone-first design, graceful degradation.
**Gaps:** None major.

## Integration Gaps Found & Fixed

1. **Pulse catch-up** — `check_pulse_catchup` existed but never fired at boot. Only `ensure_pulse_job` ran. FIXED: now calls both.

2. **Improvement → UpgradeQueue** — improvement loop in approval mode held proposals internally but never filed to the UpgradeQueue tool. FIXED: now files to queue with source="improvement".

3. **Research → proposals** — VERIFIED WIRED: `propose_from_ticket` called from research_loop.

## Remaining Minor Gaps

- Security: secret scanning, dependency vulnerability audit (nice-to-have)
- Storage: cross-region replication is stubbed (acceptable — single-device primary)

## Verdict

The cross-cutting layer is mature. The real work was the integration gaps — systems that existed but weren't connected. Those are now wired.
