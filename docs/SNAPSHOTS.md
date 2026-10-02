# Snapshots, recovery, and self-update

Devon's live state (database, blob store, config) can be snapshotted,
restored transactionally, and the whole installation self-updates with
automatic rollback.

## Snapshots

```
nm snapshot create --label "before big change"
nm snapshot list
nm snapshot verify [id]        # checksums + PRAGMA integrity_check
nm snapshot restore [id]        # refuses a running/dirty system unless --force
nm snapshot delete <id>
```

A snapshot is `<state>/snapshots/<id>/` with `db.sqlite` (taken via
`VACUUM INTO`, consistent even while the DB is open), `blobs/`,
`config.json`, and `manifest.json` (checksums, schema version).

Restore is transactional: stage to a temp dir → verify → atomic swap.
If the swap fails partway, the previous files are moved back.  Restore
refuses to run over a live system (another process holds
`<state>/nomorals.lock`) or a dirty one (un-checkpointed WAL sidecars)
unless `--force` is passed.

## Recovery

```
nm recover [--yes] [snapshot-id]
```

Interactive by default (asks before touching live state); `--yes`
makes it non-interactive.  Picks the newest snapshot that verifies
clean, restores it, re-runs migrations, and reports health
(schema version, table count, integrity, import).

## Self-update

```
nm update [--no-pull] [--check-only] [--repo DIR] [--yes]
```

1. Pre-update snapshot (the rollback anchor).
2. `git pull --ff-only` (all-or-nothing; skipped with `--no-pull`).
3. Database migrations.
4. Post-update health checks: package import, layering subset,
   error scan, DB integrity.

Any failure in steps 2–4 restores the pre-update snapshot
automatically and reports what failed.  A failed `git pull` changes
nothing, so no rollback is needed for it.  `--check-only` runs just
the health checks.

## Golden missions

Deterministic end-to-end drills (plan → execute → verify → repair)
that run without any model.  Short versions are the unit suite;
`--long` runs the real multi-minute drill.

```
nm golden list
nm golden run research_write_verify [--long]
nm golden run build_test_fix [--long]
nm golden run audit_remediate_rescan [--long]
nm golden resume <mission-id>
```

The three missions:

* **research_write_verify** — collect facts → draft a report → verify
  every fact is present.  The long drill loses a fact on first
  collection; the repair step re-collects and re-drafts.
* **build_test_fix** — scaffold a module → run pytest → verify green.
  The long drill scaffolds a deterministic off-by-one bug; repair
  patches it and re-runs green.
* **audit_remediate_rescan** — generate configs with injected
  `INSECURE-DEFAULT` markers → audit → remediate → re-scan (the guard:
  remediation only counts when the second scan finds zero issues).

Every run records a row in the benchmark DB as
`model_id="golden:<key>"` (`source="golden"`), so drills are
comparable over time.  Runs are killable (`GoldenRunner.kill()`;
status becomes PAUSED) and resumable from the persisted step list.

## Chaos durability

`tests/test_chaos_durability.py` injects faults and asserts the guards
fire — same style as `tests/test_fault_injection.py`:

| fault | guard |
|---|---|
| queued row deleted (lost message) | `WorkQueue.reconcile(receipts)` names the missing job |
| same payload enqueued twice | `WorkQueue.detect_duplicates()` groups it |
| `Policy.check` verdict flipped post-audit | `Policy.verify_decision()` vs the audit trail |
| blob bytes corrupted after write | `ArtifactStore.verify()` read-back hash check |
| DAG task dies mid-run | executor marks FAILED/skips dependents; `TaskGraph.audit()` confirms honest bookkeeping |
