# NoMorals Core — Architecture

**Codename:** NMC (NoMorals Core)
**Primary language:** Python 3.11+ (stdlib-first, optional accelerators)
**Design target:** a self-hosted, self-improving, multi-agent personal AI substrate that runs
anywhere from a phone (Termux) to a multi-GPU workstation, with no hard dependency on any
single model vendor.

---

## 0. Reading guide

| You want to…                       | Read                       |
| ---------------------------------- | -------------------------- |
| Understand the layering            | §1                         |
| See every directory and its job     | §2                         |
| Understand concurrency model       | §4                         |
| Understand persistence + backups   | §5, §6                     |
| Understand the model pipeline      | §7, §8                     |
| Understand capability governance   | §10                        |
| See what exists today vs. planned  | §12 (Build status)         |

---

## 1. Layer model

Strictly layered. Lower layers never import higher layers. Dependencies point downward only.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ L7  SURFACE        cli.py · api/server.py · api/routes.py · tui · builders/    │
│                    (scaffolding + app lifecycle: templates, run/serve/smoke,  │
│                    policy-gated install, export)                              │
├───────────────────────────────────────────────────────────────────────────┤
│ L6  MISSIONS       mission · planner · reflector · runner · checkpoints   │
├───────────────────────────────────────────────────────────────────────────┤
│ L5  AGENTS         orchestrator · supervisor · runtime · roles · spawn    │
│                    blackboard · tasks (DAG) · budgets · comms             │
├───────────────────────────────────────────────────────────────────────────┤
│ L4  CAPABILITIES   tools/* (vision, media, web, fs, shell, parsers, social)│
├───────────────────────────────────────────────────────────────────────────┤
│ L3  COGNITION      memory/manager.py (episodic, semantic, working, consol.)│
│                    llm/* (providers, router, registry, sampling)          │
│                    training/* (datasets, trainers, eval, model registry)  │
├───────────────────────────────────────────────────────────────────────────┤
│ L2  STATE          storage/* (db, schema, migrations, repo, fts, vectors, │
│                    blob, backup, queue)                                   │
├───────────────────────────────────────────────────────────────────────────┤
│ L1  KERNEL         core/* (config, events, errors, ids, result, retry,    │
│                    ratelimit, policy, logging, observability)             │
└───────────────────────────────────────────────────────────────────────────┘
```

**Rule enforced by `tests/test_layering.py`:** a module in layer *n* may import only from
layers `< n`. The test parses the AST of every module and fails the build on a violation.
This is the single most important structural guarantee in the codebase — it is what keeps a
100k-line system from turning into a ball of mud.

---

## 2. Directory structure

```
No-morals-ai/
├── ARCHITECTURE.md            ← this document
├── README.md
├── ROADMAP.md                 ← path to 100k+ lines, milestone by milestone
├── pyproject.toml             ← dependencies + optional extras (fast, media,
│                               media-edit, train, finetune, hub, dev, all);
│                               e.g. pip install ".[fast,media,train]"
├── .env.example
├── Makefile
├── nomorals/                  ← the package
│   ├── __init__.py
│   ├── __main__.py            ← python -m nomorals
│   ├── version.py
│   │
│   ├── core/                  L1 — kernel, zero intra-project deps
│   │   ├── errors.py          typed error hierarchy + Outcome interop
│   │   ├── result.py          Ok/Err monad, used as every boundary return type
│   │   ├── ids.py             ULID-style monotonic ids (stdlib only)
│   │   ├── clock.py           injectable monotonic/wall clock (testability)
│   │   ├── config.py          layered settings: defaults → TOML → env → overrides
│   │   ├── events.py          thread-safe pub/sub bus, glob topics, sync/async
│   │   ├── logging_setup.py   structured logging, rotating files, redaction
│   │   ├── retry.py           backoff policies, jitter, retryable predicates
│   │   ├── ratelimit.py       token bucket + per-key registry + cron-ish windows
│   │   ├── policy.py          capability/permission model + hard limits
│   │   ├── observability.py   metrics, spans, counters (Prometheus-ish, no deps)
│   │   └── text.py            tokenizers, chunkers, simhash, dedup, normalization
│   │
│   ├── storage/               L2
│   │   ├── db.py              connection pool, WAL, thread-local conns, txn ctx
│   │   ├── schema.py          migration engine (versioned, forward+checksum)
│   │   ├── migrations.py      ordered migration set (SQL as data)
│   │   ├── repository.py      generic typed CRUD + query builder
│   │   ├── fts.py             FTS5 index, bm25 ranking, highlighting
│   │   ├── vectors.py         vector store: pure-Python cosine + numpy/IVF fast path
│   │   ├── blob.py            content-addressed blob store (sha256, dedup, gzip)
│   │   ├── queue.py           durable work queue in SQLite (at-least-once, leases)
│   │   ├── backup.py          versioned backups + rotation + git push
│   │   └── models.py          dataclasses mirroring tables
│   │
│   ├── memory/                L3 — cognition
│   │   ├── base.py            MemoryRecord, decay, importance scoring, normalization
│   │   ├── embeddings.py      embed via provider, hashing fallback (offline-safe)
│   │   └── manager.py         unified façade: remember/recall/reflect/consolidate
│   │                          (episodic, semantic, working, and consolidation are
│   │                           stores inside manager.py — four cooperating stores
│   │                           did not need four files, only one façade)
│   │
│   ├── llm/                   L3
│   │   ├── base.py            LLMProvider protocol, messages, sampling params
│   │   ├── sampling.py        temperature/top-p/top-k/min-p/repetition control
│   │   ├── providers/
│   │   │   ├── mock.py        deterministic offline provider (drives all tests)
│   │   │   ├── openai_compat.py  OpenAI-compatible: vLLM, Ollama, LM Studio, OpenRouter
│   │   │   ├── hf_serverless.py  HF Inference API + Inference Endpoints
│   │   │   └── llama_cpp.py   local llama.cpp server / gguf
│   │   ├── router.py          health, fallback chains, latency/cost stats, hot-swap
│   │   ├── registry.py        model catalog (Dolphin & co), metadata, provenance
│   │   └── download.py        resumable HF download (hf_hub if present, else raw HTTP)
│   │
│   ├── training/              L3
│   │   ├── dataset.py         ChatML / Alpaca / ShareGPT / JSONL codecs
│   │   ├── collect.py         harvest conversations + multi-model distillation
│   │   ├── preprocess.py      clean, dedup, filter, split, tokenize, shard
│   │   ├── tokenize.py        BPE trained from scratch (stdlib) + HF tokenizer adapter
│   │   ├── trainer.py         trainer protocol + native pure-Python loop
│   │   ├── backends/
│   │   │   ├── native.py      tiny transformer, real autograd-free numeric loop
│   │   │   ├── unsloth.py     config+script generation for Unsloth
│   │   │   └── llama_factory.py  LLaMA-Factory config generation
│   │   ├── evaluate.py        perplexity, held-out loss, capability probes
│   │   └── registry.py        run tracking, artifacts, promotion → hot-swap
│   │
│   ├── tools/                 L4
│   │   ├── registry.py        tool schema, dispatch, permission gate, call log
│   │   ├── filesystem.py      read/write/glob/patch across encodings
│   │   ├── parsers.py         pdf·docx·xlsx·odt·epub·csv·html·md·image (stdlib)
│   │   ├── shell.py           sandboxed exec: rlimits, jail, timeout, no-net
│   │   ├── web.py             HTTP client, retries, robots, html→text, search
│   │   ├── vision.py          image understanding, describe/compare/OCR-ish
│   │   ├── media.py           video/audio download (yt-dlp + fallbacks), ffmpeg
│   │   └── codegen.py         generate → sandbox → test → repair loop
│   │
│   ├── social/                L4
│   │   ├── base.py            SocialPlatform protocol, post shape, audit log
│   │   ├── queue.py           content queue, scheduling, retry, per-platform budget
│   │   ├── platforms/         mastodon · bluesky · telegram · discord · x · reddit
│   │   └── analytics.py       engagement metrics → feedback into memory
│   │
│   ├── agents/                L5
│   │   ├── base.py            Agent ABC, lifecycle, capability declaration
│   │   ├── context.py         AgentContext: ids, budget, tools, memory, llm
│   │   ├── tasks.py           Task, TaskGraph DAG, dependencies, cancellation
│   │   ├── runtime.py         HybridExecutor: threads + processes + asyncio
│   │   ├── blackboard.py      shared scratch space w/ locking + versioning
│   │   ├── supervisor.py      watchdog, restart policy, budget enforcement
│   │   ├── spawn.py           dynamic sub-agent spawning + lineage tree
│   │   ├── comms.py           agent↔agent mailbox, broadcast, request/reply
│   │   ├── orchestrator.py    Master Orchestrator: plan→decompose→run→aggregate
│   │   └── roles/             planner · critic · research · coding · vision
│   │                          data_collection · training · memory · execution
│   │                          social · reflection
│   │
│   ├── missions/              L6
│   │   ├── mission.py         Mission spec, state machine, checkpoints
│   │   ├── planner.py         goal decomposition, dependency discovery
│   │   ├── reflector.py       self-evaluation, lesson extraction, policy tuning
│   │   ├── runner.py          long-running loop, resume, graceful shutdown
│   │   └── goals.py           goal graph, progress metrics, success criteria
│   │
│   └── api/                   L7
│       ├── server.py          ThreadingHTTPServer + routing + auth + SSE
│       └── routes.py          REST surface over the whole system
│
├── tests/                     ← stdlib unittest, runs with zero installs
│   ├── test_layering.py       architecture enforcement
│   ├── test_core_*.py
│   ├── test_storage_*.py
│   ├── test_memory_*.py
│   ├── test_agents_*.py
│   └── test_tools_*.py
│
├── scripts/                   bootstrap.sh · termux_setup.sh · backup_now.py
└── docs/                      deployment, termux, models, security
```

---

## 3. Kernel invariants

1. **Every fallible boundary returns `Outcome[T]`** (`Ok(v)` / `Err(e)`), never raises across
   a layer boundary. Exceptions are for programmer errors, not for "the network was down".
2. **No global mutable state.** Singletons live in one `SystemContext` object that is
   explicitly threaded through. This is what makes the whole thing testable in parallel.
3. **The clock is injected.** `core/clock.py` — no `time.time()` calls in logic paths.
   Every decay, backoff, and schedule is therefore unit-testable at arbitrary speed.
4. **No blocking call inside the event loop.** The runtime enforces this by running async
   work and blocking work on separate executors.

---

## 4. Concurrency model

Three execution substrates, one scheduler.

```
                    ┌────────────────────────┐
                    │   HybridExecutor       │
                    │  (agents/runtime.py)   │
                    └───────────┬────────────┘
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
  ThreadPoolExecutor     ProcessPoolExecutor      asyncio event loop
  I/O-bound tools        CPU-bound work           high-fanout I/O
  (HTTP, scraping,       (tokenizing, training,   (websockets, SSE,
   file reads, social)    embedding, dedup)        many concurrent calls)
        └───────────────────────┴────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  TaskGraph scheduler   │
                    │  DAG · deps · budgets  │
                    │  cancel · retry · join │
                    └────────────────────────┘
```

* **Tasks** declare their kind: `IO`, `CPU`, or `ASYNC`. The scheduler places them on the
  matching substrate automatically. This is the single rule that keeps a mixed
  thread/process/async system comprehensible.
* **Backpressure** is explicit: bounded semaphores per resource class (network, disk, GPU),
  so 500 sub-agents cannot open 500 sockets.
* **Cancellation** propagates through the DAG. A cancelled parent cancels unfinished
  children; running children get a cooperative cancel event and a hard deadline.
* **Budgets** are enforced by `Supervisor`: wall-clock, tokens, cost, and child count per
  agent. An agent that blows its budget is killed and reported, never silently runaway.
* **Processes** are used where the GIL is a real cost (tokenization, embedding batches,
  dataset dedup, training). Results cross the boundary as picklable `Outcome` payloads.
* **Sub-agent spawning** (`agents/spawn.py`) is dynamic: an agent may request children at
  runtime; the lineage tree is persisted, so a mission can be resumed after a crash.

---

## 5. Persistence

SQLite is the substrate — not as a compromise, but because it is the only database that
survives `pkg install` on a phone and scales to hundreds of GB with WAL. The design keeps
the door open to Postgres/DuckDB behind the `Repository` protocol.

| Concern        | Mechanism                                                        |
| -------------- | ---------------------------------------------------------------- |
| Durability     | WAL + `synchronous=NORMAL`, checkpointed on backup                |
| Concurrency    | Thread-local connections, `busy_timeout`, single-writer discipline|
| Schema         | Versioned migrations, checksummed, forward-only + explicit down   |
| Search         | FTS5, bm25 ranking, porter tokenizer                              |
| Similarity     | Vector table; pure-Python cosine, numpy/IVF when available        |
| Binary         | Content-addressed blob store, sha256, dedup, gzip tier            |
| Queues         | Durable SQLite queue with leases + visibility timeout             |
| Backups        | `sqlite3.Connection.backup()` → timestamped → gzip → rotate → git |

**Backup design (§6 detail):** backups never use file copy — they use the SQLite online
backup API so a backup taken mid-write is still a consistent snapshot. Each backup is
gzipped, checksummed, rotated to N kept, and optionally committed to a *separate* GitHub
repository so model weights/datasets never bloat the code repo.

---

## 6. Memory

Four cooperating stores — episodic, semantic, working, consolidation — live as
classes inside `memory/manager.py` behind one façade. They share a connection,
an embedder, and a recall merge, so splitting them across files would have meant
threading those three through every constructor.

```
        ┌─────────────── WorkingMemory ───────────────┐
        │  budgeted context assembly (tokens/$)        │
        └───────▲───────────────▲───────────────▲──────┘
                │               │               │
      ┌─────────┴─────┐ ┌───────┴───────┐ ┌─────┴──────────┐
      │  Episodic     │ │  Semantic     │ │ Vector index   │
      │  events, time │ │  facts, prov. │ │ embeddings     │
      │  salience     │ │  contradictions│ │ cosine/IVF    │
      └─────────┬─────┘ └───────┬───────┘ └─────┬──────────┘
                └───────────────┴───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Consolidation loop   │
                    │ summarize→distill→forget│
                    └────────────────────────┘
```

* **Recall is a merge, not a lookup:** recency decay × importance × semantic similarity ×
  FTS bm25, normalized, then re-ranked. Weights are configurable and *tuned by the
  reflector* — the system learns how to remember.
* **Consolidation** ("sleep cycle") runs on a schedule or when memory pressure crosses a
  threshold: episodes are summarized into semantic facts, low-salience records decay below
  the forget threshold and are compacted into tombstones.
* **Embeddings degrade gracefully:** provider embeddings when online, deterministic hashing
  embeddings offline. Same API, same store, so the system works on a phone with no network.

---

## 7. Model plane

```
   ┌────────────────────────── ModelRegistry ──────────────────────────┐
   │  catalog: dolphin, mistral, qwen, llama, phi … + personal finetunes│
   │  metadata: params, ctx, license, sha256, source, eval scores       │
   └───────────────┬───────────────────────────────────┬───────────────┘
                   ▼                                   ▼
        ┌─────────────────────┐              ┌─────────────────────┐
        │  download.py        │              │  training/registry  │
        │  resumable, sha256  │              │  runs, artifacts,   │
        │  hf_hub | raw HTTP  │              │  promote → active   │
        └─────────┬───────────┘              └──────────┬──────────┘
                  └──────────────┬──────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │      LLMRouter         │
                    │  fallback chain, health │
                    │  latency/cost stats     │
                    │  hot-swap active model  │
                    └───────────┬────────────┘
        ┌───────────────┬───────┴────────┬────────────────┐
        ▼               ▼                ▼                ▼
   hf_serverless   openai_compat     llama_cpp          mock
   (Inference API, (vLLM, Ollama,    (local gguf)   (offline, tests)
    Endpoints)      LM Studio, OR)
```

**Hot-swap** is a first-class operation: `router.set_active("personal/dolphin-v3")` swaps the
serving model for *new* calls without dropping in-flight ones, and the swap is persisted so a
restart restores it.

---

## 8. Self-improvement loop

This is the actual point of the system. It is a closed loop, not a feature.

```
   ① COLLECT          every conversation, tool call, and outcome is persisted
        │             (episodic memory + training/collect harvest)
        ▼
   ② CURATE           dedup (simhash) · quality filter · PII scrub · split
        ▼
   ③ TRAIN            native loop (anywhere) or Unsloth/LLaMA-Factory (GPU)
        ▼
   ④ EVALUATE         held-out perplexity + capability probes + regression gate
        ▼
   ⑤ PROMOTE          pass gate → register → router hot-swap; fail → quarantine
        ▼
   ⑥ REFLECT          mission outcomes scored, lessons extracted, policy tuned
        └────────────────────────── back to ① ──────────────────────────┘
```

Step ⑤ has a **regression gate**: a new checkpoint is only promoted if it does not regress
on the standing probe set. This is what prevents self-improvement from becoming
self-degradation — the failure mode every naive auto-finetune loop has.

---

## 9. Mission execution

```
Mission(goal, budget, success_criteria)
   │
   ├─ Planner     → GoalGraph (subgoals, deps, acceptance tests)
   ├─ Orchestrator→ TaskGraph, assigns roles, runs in parallel
   ├─ Supervisor  → budgets, restarts, cancels
   ├─ Checkpoint  → every state transition persisted → crash-resumable
   └─ Reflector   → scores outcome, writes lessons, tunes recall weights
```

A mission survives `kill -9`. On restart, `runner.py` reloads the last checkpoint and
continues from the first incomplete task.

---

## 10. Capability governance

"Unrestricted" here means **no vendor content filter sitting between you and your own
hardware** — you own the model, you own the policy. It does not mean "unaccountable", and
unaccountable systems fail operationally: an agent that can do anything to your disk will
eventually do something to your disk.

So the kernel has an explicit capability model (`core/policy.py`):

* Every tool declares required capabilities (`fs.write`, `net.out`, `exec.shell`, `db.write`,
  `social.post`).
* Every agent is granted a capability set at spawn. Sub-agents inherit the *intersection* of
  their parent's grant and their role's requirement — privilege can narrow, never widen.
* Every call is logged to an append-only audit table with actor, args digest, and outcome.
* Destructive operations (`rm -rf`, DROP TABLE, mass DM, bulk follow) require an explicit
  confirmation token that must be issued out-of-band by the operator.
* Platform automation uses **official APIs only**, with per-platform rate limits and a
  posting budget. No engagement farming, no ban evasion, no bot-network behavior — not
  because of a lecture, but because those accounts get suspended and the capability becomes
  worthless.

This costs nothing in capability for legitimate use and it is the difference between a tool
you can leave running unattended for a week and one you cannot.

---

## 11. Portability: desktop → Termux

Same package, three profiles, selected by config — no forked codebase.

| Profile    | Concurrency      | Storage | Models                 | Extras        |
| ---------- | ---------------- | ------- | ---------------------- | ------------- |
| `workstation` | threads+process+async | SQLite/DuckDB | 70B via API, local 7–13B | torch, CUDA |
| `laptop`   | threads+async    | SQLite  | 7B quantized local     | llama.cpp     |
| `termux`   | threads only     | SQLite  | API-first, 1–3B local  | no torch      |

The Termux profile disables the process pool (fork on Android is fragile), drops numpy/torch,
and routes inference to APIs. Because the clock, executor, and storage are all injected, this
is a config change — not a different build.

---

## 12. Build status

Tracked honestly in `ROADMAP.md`. Legend: ✅ implemented & tested · 🟡 implemented, partially
verified · ⬜ designed, not yet written.

| Layer | Module group        | Status |
| ----- | ------------------- | ------ |
| L1    | core/*              | ✅     |
| L2    | storage/*           | ✅     |
| L3    | memory/*            | ✅     |
| L3    | llm/* (providers)   | 🟡 offline-verified; live HF path unverifiable in sandbox |
| L3    | training/*          | 🟡 native loop ✅, GPU backends generate-and-handoff |
| L4    | tools/*             | ✅ (media needs yt-dlp at runtime) |
| L4    | social/*            | 🟡 adapter code written; needs live platform credentials |
| L5    | agents/*            | ✅     |
| L6    | missions/*          | 🟡     |
| L7    | cli, api            | ✅     |

---

## 13. Non-goals

* **Not a web app framework.** The API server exists so other things can drive the system.
* **Not a distributed cluster.** Multiprocessing on one host, not Ray. The queue is designed
  so distribution is a later swap, not a rewrite.
* **Not a GUI.** Terminal + HTTP API first.
