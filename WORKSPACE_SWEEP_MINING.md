# WORKSPACE module sweep — external mining report

Module: `nomorals/workspace/` (6 files: `__init__.py`, `profile.py`, `vcpu.py`, `workspace.py`, `inbox.py`, `rooms.py`)
Date: 2026-10-10. Mined before any code change, per sweep method.

## 1. VirtualCPU (`vcpu.py`) — the execution engine

**What it is:** one daemon worker thread + priority heap + pause/resume/drain + EWMA busy fraction + crash-restart (×3 then offline).

**External comparisons:**

- **Tokio multi-thread runtime** (Rust): per-worker local LIFO deque + global injector queue; idle workers **steal half** of a sibling's queue; local queue spills half to global at 256 tasks; cooperative budget per task turn. Gold: our dispatch is push-only — an idle VCPU never helps a busy sibling. Work-stealing across VCPUs is the missing piece. Sources: tokio runtime docs via dev.to/0xgosu.dev "Fast Tokio Is a Scheduling Budget, Not a Bag of Tricks"; hassard0/mighty `docs/internals/scheduler.md`.
- **Celery worker**: `--autoscale=max,min` pair; `worker_max_tasks_per_child` (restart workers to fight memory leaks); soft/hard **task time limits**; task routing by queue (`-Q cpu_heavy` / `-Q io_heavy` with different pool types); task events for observability. Gold: per-task timeouts, worker rebirth after N tasks, per-task naming/observability. Sources: dev.to "Mastering Celery"; AWS MWAA Airflow worker-pool guide.
- **Java ThreadPoolExecutor**: rejection policies (Abort / CallerRuns / Discard); `allowCoreThreadTimeOut`. Gold: our backpressure *blocks* (CallerRuns-ish); offering a configurable policy (block vs caller-runs vs fail-fast) is strictly more capable.
- **Erlang BEAM schedulers**: per-scheduler run-queues with migration. Confirms the work-stealing direction.

**Gaps found (what SHOULD it have):** work-stealing from idle siblings; per-task timeouts (soft limit); max-tasks-per-vcpu rebirth (memory-leak hygiene); batch submit; per-task names + rolling latency histogram (p50/p95) for observability; stats on stolen tasks.

## 2. Workspace (`workspace.py`) — the farm

**What it is:** profile-aware pool (min/target/max), affinity dispatch scoring, ±1 autoscale tick, scale-down drain, events, status dict.

**External comparisons:**

- **Kubernetes HPA**: `desiredReplicas = ceil(current × currentMetric/targetMetric)` — **proportional** scaling, not ±1; 10% tolerance band against flapping; **asymmetric stabilization windows** (fast up ~60s, slow down ~300s); multiple metrics (CPU *and* memory); scale policies cap pods-per-period. Gold: proportional scale math, tolerance band, multi-metric signals, rate-limited scale steps. Sources: medium.com "The Ultimate Guide to Kubernetes Autoscaling"; docs.sc.otc.t-systems.com; nikhil-sy/itd HPA intro.
- **KEDA**: scale-to-zero on events; 60+ event sources. Gold: parking the farm to near-zero when the agent sleeps (min envelope permitting); queue-depth as a scaling metric.
- **Celery autoscaler**: grows/shrinks pool processes on load. Same shape as ours but dumber — we already beat it on drain semantics.

**Gaps found:** proportional (multi-step) scaling instead of ±1; tolerance/deadband; queue-depth + wait-latency as scaling signals alongside busy fraction; predictive scale-up from queue growth rate; per-kind (io/cpu) dedicated lanes that grow with their own hotness; scale history log; a god-tier `render_status()` ASCII farm table (bars, per-core load); `pause_all`/`resume_all`; `map()` convenience.

## 3. Inbox (`inbox.py`) — the drop-in triage

**What it is:** drop file/link/note → directive-first classification → intent handlers (summarize/describe/file/remind/watch/research/transcribe/media/vision) → DB + crash recovery + quarantine security + room scoping + 2-min watcher.

**External comparisons:**

- **Hazel (macOS)**: rule engine — folders on the left, rules on the right; conditions (name/kind/date/Spotlight metadata, nested conditionals) → actions (move/rename/tag/archive/run script); rule templates; token-based rename patterns. Gold: a **user-defined rule layer** between directives and the fallback classifier — the inbox currently has zero user configurability. Sources: macworld.com Hazel reviews; github.com/hazel-mac-b/hazel-mac.
- **AI email triage (n8n + GPT-4o-mini, Shortwave-style skills)**: closed label sets; confidence-gated routing (high-confidence auto, low-confidence → review queue); **drafts only, never auto-send**; snippet-not-full-body to the model; per-run KB updates (blocklist learning). Gold: classification confidence with a review threshold; structured digest per sweep. Sources: dev.to n8n inbox triage; github ezjonline/ezj-automations `inbox-cleaner/SKILL.md`; blog.workhint.com production triage workflow.
- **PARA/Obsidian capture**: `00-Inbox/` as the unprocessed capture bucket; weekly review ritual; "email shouldn't be where projects live — file to projects" (move tasks OUT of the inbox). Gold: auto-route actionable items to rooms; a weekly-review report.
- **maildir / Notmuch**: content-hash dedup of duplicates. Gold: duplicate detection on intake (same file dropped twice).

**Gaps found:** Hazel-style rule engine (conditions → intent, stored in DB); duplicate detection via content hash; classifier confidence + low-confidence review threshold; item priority ordering in sweeps; `digest()` daily summary renderer; `search()` over items; batch ops (retry-all, archive-done); item cards (`render_item`); per-item owner notes (`annotate`).

## 4. Rooms (`rooms.py`) — persistent project rooms

**What it is:** per-goal room dir + ROOM.md (state JSON fence = rehydration contract) + DB index + sandbox + decisions/blockers log + dirty-room reconcile + goal/project tick + scheduler job.

**External comparisons:**

- **LangGraph checkpointing / Temporal durable execution**: checkpoint every super-step keyed by thread_id; `invoke(None)` resumes from last checkpoint; time-travel via state history; PostgresSaver for multi-instance. Gold: rooms already do crash-safe rehydration; missing = **forks/branches** (time-travel equivalent: branch a room from its current checkpoint) and per-step checkpoints in the activity log. Sources: dev.to checkpointing article; langgraph-tutorials PostgresSaver README.
- **PARA method (Tiago Forte)**: Projects/Areas/Resources/Archives by actionability; weekly review ritual ("is this still active? next action? archive?"); start with Projects + Archive only. Gold: a `review()` weekly-review report; room templates (Obsidian-style templates plugin: meeting/project/literature notes). Sources: medium.com PARA for Engineers; paddychief92/second-brain-gtd obsidian-mastery SKILL.md.
- **Obsidian**: templates, MOC (maps of content) hub notes, YAML frontmatter, backlinks. Gold: `rooms/INDEX.md` hub (MOC-style); room templates; tags on rooms.
- **tmux-resurrect / Claude Code session resume**: restore full session state incl. panes. Gold: `brief()` — "here's where you left off" one-pager on enter (last step, open blockers, tail of activity).

**Gaps found:** room templates (meeting/research/build/goal seeds); `brief()` session-recap renderer; `rooms/INDEX.md` MOC hub; `review()` weekly report (stale/blocked/dirty/active); `fork()` room branching with provenance; `render_card()` pretty room card; room tags; `read_log()` tail reader on RoomContext.

## 5. `__init__.py` / `profile.py`

Re-export surface. Gold: export the new render helpers so CLI/chat can reach them. `profile.py` stays a shim (core.profile owns it) — no change needed.

## Cross-cutting style gold

- Hazel's rule list UI, K8s `kubectl top`-style tables → `render_status()` farm table with load bars.
- Triage digests → `digest()` markdown sections per intent, one-liner per item.
- Room cards → bordered `render_card()` with status glyphs.
- All renderers must be pure functions returning strings (chat/CLI/test friendly), never print.
