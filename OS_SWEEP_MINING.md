# OS Module Sweep — External Mining Report

**Date:** 2026-10-10 · Module: `nomorals/os/` (15 files)
**Method:** for every significant class, asked "how does the best implementation of X do it?" — mined best AND trash, from live sources.

---

## 1. kernel.py — `OSKernel` (process lifecycle)

**Best-in-class mined:** systemd unit semantics (Restart=on-failure + StartLimitIntervalSec/Burst start-rate guard), supervisord, s6/runit "always restart" supervision, Fly Machines `restart.policy` (no|on-failure|always + max_retries), systemd sd_notify readiness (READY=1 / STATUS / STOPPING), graceful shutdown with stop timeouts.

**What the best do that we don't:**
- Graceful shutdown path: signal handlers (SIGTERM/SIGINT), ordered teardown, shutdown hooks, stop timeout. We only have `stop()`.
- Restart policy per service + start-rate guard (5 failures in 60s pages a human instead of spinning).
- Readiness gating: dependents don't start until dependencies report ready (s6-rc `waits-for`).
- Service dependency ordering at start (topo sort) and reverse at stop.

**Gaps to fill:** `on_shutdown` hook registration, signal/atexit wiring, dependency-ordered service start/stop, service recycle (`restart_service`), readiness wait, kernel status dashboard rendering.

## 2. services.py — `ServiceRegistry` (service locator)

**Best-in-class mined:** `dependency-injector` (providers.Singleton/Factory/Resource, Configuration providers, `container.override()` in tests, qualifiers for multiple providers of one type), FastAPI DI, myfy `@provider(scope=…)` scopes, composition-root discipline ("a DI container used as composition root is fine; as service locator is an antipattern" — deviq/Seemann).

**What the best do that we don't:**
- Scopes: singleton vs factory (fresh instance per lookup) vs resource (with teardown).
- Lifecycle: start/stop hooks per service, started/stopped in dependency order.
- Test overrides: scoped `override(name, instance)` context manager.
- Qualifiers/tags: multiple providers of the same kind, tag-based discovery.
- Descriptor validation: name format, factory callability.

**Gaps to fill:** `scope` on descriptors, lifecycle start/stop, `depends_on` topological ordering, `override()` context, `tags`, `restart(name)` recycle, rich `describe_all()` output.

## 3. health.py — `HealthMonitor` (health checks)

**Best-in-class mined:** Kubernetes probe triad (startup/liveness/readiness — different failure actions: restart vs remove-from-traffic vs gate-startup), kazhuravlev/healthcheck (background checks with interval + timeout + initial delay + stale data; manual checks; Basic vs Background check types), readiness failure ≠ restart-worthy.

**What the best do that we don't:**
- Probe roles: liveness (restart-worthy) vs readiness (traffic-worthy) vs startup. We treat all checks identically.
- Background scheduling: checks run on an interval in a loop with per-check timeouts, and stale results are marked stale rather than trusted.
- Consecutive-failure thresholds: 3 transient blips ≠ down (flapping suppression); consecutive-success to clear.
- Manual checks: application-controlled state (e.g. "cache warming").
- More than binary: degraded/unknown states; check history.

**Gaps to fill:** `kind` (liveness/readiness/startup) per check, `start_background()` loop with interval + timeout, consecutive-failure/success thresholds, manual checks (`set_manual`), status history + flapping detection, `summary()` (ready? alive?), plain-text `render()` dashboard.

## 4. session.py — `Session` / `SessionStore`

**Best-in-class mined:** tmux (named sessions, attach/detach semantics, server keeps sessions alive with no client attached, `new -A` get-or-create, tree view of sessions, idle sessions persist).

**What the best do that we don't:**
- Attach/detach lifecycle: a session exists independently of whether anyone is looking. We have no attach count / last-detach.
- Idle expiry: stale sessions get reaped (tmux `set -g status`, session timeouts in servers).
- Rename, per-session activity feed, tree/presence views.

**Gaps to fill:** `attach_count`, `idle_seconds`, `purge_expired(max_idle_s)`, `rename()`, presence rendering (`render()` session table), session activity touch on message (already have touch; expose idle).

## 5. session_bridge.py — `SessionBridge`

**Best-in-class mined:** tmux attach-anywhere, chat platform session multiplexers (one identity across surfaces), handoff patterns (move conversation between devices/agents).

**What the best do that we don't:**
- Session handoff: move a conversation from one frontend/chat to another (phone → desktop) keeping history.
- Principal-level views: "all sessions for owner".

**Gaps to fill:** `handoff_session(session_id, platform, chat_id)`, `sessions_for_principal()`, `session_counts()`.

## 6. project.py — `Project` / `ProjectStore`

**Best-in-class mined:** Timeshift tags (daily/monthly tags on one snapshot copy), project templates in PM tools, archive-vs-delete retention.

**What the best do that we don't:**
- Tags on projects, archive convenience (beyond raw state string), project health summary (missions/sessions/artifacts counts — needs duck-typed inputs), search by name.

**Gaps to fill:** `tags` field + `tag()`/`untag()`, `archive()`, `search(name_substr)`, `summary(project, mission_store=None, session_store=None)` dashboard dict + `render()`.

## 7. timeline.py — `Timeline` (event log)

**Best-in-class mined:** Event Sourcing/CQRS (immutable append-only event store in SQLite; events as past-tense facts; aggregates/views/projections rebuilt from the log), event envelopes (aggregate id + sequence number + causation/correlation ids + actor for audit trails), Ouroboros audit design (append-only segments, hash-linked, sealed/rotated; async SQLite indexing from committed events; full-text indexing opt-in), FTS5 in SQLite.

**What the best do that we don't:**
- Envelopes: no sequence numbers, no causation/correlation ids, no actor attribution.
- Full-text search over payloads (FTS5) — we only have topic globs.
- Retention: no pruning, no archive/export. Logs grow forever.
- Export: JSONL dump/import for forensics and handoff.

**Gaps to fill:** FTS5 `search()` over data_json, `causation_id`/`correlation_id`/`actor` columns (auto-added via ALTER on existing tables), `prune(older_than)` / `prune_keep_latest(n)`, `export_jsonl()` / `import_jsonl()`, `topic_counts()` stats, `render()` one-line feed view.

## 8. update.py — `UpdateManager` (self-update)

**Best-in-class mined:** TUF (signed metadata chains), LumenUpdate/Sparkle (stage in isolation → atomic replace + backup → health gate → commit or rollback), vool signed-atomic-self-update design (install transaction state machine with append+fsync journal; crash recovery at startup — pre-swap aborts clean, post-swap rolls back; prior bundle never deleted until FINALIZED; destructive-work guard), Rudder last-known-good (pin last-known-good release + candidate; probation mode blocking writes until commit; failed-version quarantine), SUSE transactional-update (snapshot → update snapshot → mark active → reboot).

**What the best do that we don't:**
- Crash recovery: if the process dies mid-update, nothing recovers. No journal.
- Update journal: append-only record of steps outside the report object.
- Failed-version quarantine: never retry a version that already failed.
- Last-known-good pinning / update history across runs.
- Dry-run mode.

**Gaps to fill:** persistent update journal (JSONL), `recover_interrupted()` (call at boot; pre-dangerous-point → mark aborted, post → roll back), failed-version quarantine file, `history()` of past updates, `dry_run` mode, `UpdateReport.render()` pretty output.

## 9. verifiers.py — `VerifierRegistry` + verifiers

**Best-in-class mined:** pytest JUnit-XML output parsing (structured, not regex on text), verifier composition (all-of/any-of suites), pre-flight gates (lint, git-clean, disk-space) before destructive ops.

**What the best do that we don't:**
- Composite verifiers (a suite that runs N verifiers with all/any semantics and merged verdicts).
- Pre-update gate verifiers: lint (py_compile over tree), git tree clean, disk space.
- Structured evidence: verdicts with per-check breakdowns.

**Gaps to fill:** `CompositeVerifier` (all/any), `LintVerifier` (py_compile tree walk), `GitCleanVerifier`, `DiskSpaceVerifier`; `Verdict.render()` pretty output.

## 10. mission_state.py — mission state machine

**Best-in-class mined:** `transitions` (guards, callbacks, hierarchical states, graphviz export), `python-statemachine` / `lores228` StateChart (compound states, `cond=`/`unless=` guards with priority ordering, automatic callback parameter injection, `f"{sm:md}"` markdown diagram generation, PNG graph export).

**What the best do that we don't:**
- Guards: conditional transitions (e.g. RUNNING→COMPLETED only when acceptance passed).
- Callbacks: on_enter/on_exit/on_transition hooks.
- Diagram: machine-readable/visual description of the legal graph.
- Rich transition log entries (actor, guard results).

**Gaps to fill:** guard registry (`guard(from,to,fn)` raising `GuardFailed(InvalidTransition)`), callback registries (`on_enter`/`on_exit`/`on_transition` + `clear_*` for tests), `to_mermaid()` diagram, `describe()` text table, actor recorded in transition log.

## 11. replay.py — mission replay/redrive

**Best-in-class mined:** event-sourcing time-travel debugging (reconstruct state at any point), A/B run comparison in experiment trackers, HTML report exports.

**What the best do that we don't:**
- Compare two replays (diff: transitions added/removed, timing deltas, artifact deltas) — "why did run B behave differently from run A?"
- Export formats beyond text/JSON (HTML report).
- Duration analytics (time per state, total wall time).

**Gaps to fill:** `compare(a, b)` diff report, `to_html()`, per-state duration stats in `MissionReplay`.

## 12. snapshots.py — `SnapshotManager`

**Best-in-class mined:** Timeshift (rsync+hardlinks or btrfs; scheduled snapshots hourly→monthly; tag-based retention levels instead of rotating copies; exclude filters; offline restore from live USB), borg/restic (dedup, tags), snapper (pre/post snapshot pairs around package ops).

**What the best do that we don't:**
- Tags (Timeshift-style: one copy, tagged daily/pre-update/manual) and tag-based retention.
- Retention policy: prune to keep-last-N + keep-tagged.
- Diff between snapshots (what changed).
- Export/import as a portable tarball.
- Pre/post pairs around updates (Timeshift does pre/post around package manager runs).

**Gaps to fill:** tags on create + `tag()`/`untag()`, `prune(keep_last, keep_tags)`, `diff(a, b)`, `export_tar()`/`import_tar()`, `create_pair()` pre/post context manager, `render_list()` table.

## 13. resources.py — `ResourceManager`

**Best-in-class mined:** htop/btop (live gauges), Prometheus node_exporter (PSI + cgroup signals), lmkd (composite memory pressure scores), Android lowmemorykiller.

Already god-tier from prior work (PSI, cgroups, steal time, battery forecasting, degradation ladder, subsystem budgets, profile gating). Gold remaining is **presentation + alerting**.

**Gaps to fill:** human-readable gauge rendering (`render()` with unicode bars — god-tier feel, not functional), alert callbacks (`on_alert`), one-line `summary()` for status bars.

## 14. adapters.py — session adapters

**Best-in-class mined:** adapter pattern done right (defensive, never-raise) — already solid. Missing: a generic adapter for arbitrary dict payloads, and coverage summary.

**Gaps to fill:** `GenericSessionAdapter` (project any mapping), adapter registry listing.

---

## Cross-cutting style upgrades (all files)

- Every major report/summary object gets a `render()` plain-text dashboard (god-tier feel, consistent voice).
- Consistent emoji/symbol language for status: ✓ ok, ! degraded, ✗ failed, ? unknown, … pending.
- Docstrings stay honest about what's real vs best-effort.
