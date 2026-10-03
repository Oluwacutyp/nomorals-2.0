# NoMorals Core

A self-hosted, self-evolving personal AI substrate. Multi-agent orchestration,
long-term memory, model training, and real tool access — running on your own
hardware, with no vendor filter between you and your own models.

**Zero mandatory dependencies.** Pure Python 3.11+ and the standard library.
Everything else is optional and degrades to a working pure-Python fallback.

```
python3 -m nomorals doctor     # what can this machine actually do?
python3 -m nomorals tools      # 20 tools, each capability-gated
python3 -m nomorals ask "hi"   # chat with the active model
python3 -m nomorals run "research X and write a report"
python3 -m nomorals missions --start "long goal" --budget-wall 3600
python3 -m nomorals missions --resume-all   # after a crash or reboot
python3 -m nomorals tui        # interactive terminal UI
```

## Why this exists

Hosted assistants refuse, forget, and rate-limit. A personal AI should do none of
those: it should remember everything you tell it, run your own uncensored open
weights, execute code, read any file, download any video, and keep working on a
goal for hours without supervision.

"Unrestricted" here means **no vendor filter between you and hardware you own**.
It does not mean ungoverned — every action passes through
[`core/policy.py`](nomorals/core/policy.py): a capability model with narrowing
inheritance, an append-only audit log, and confirmation tokens for destructive
operations. See [`ARCHITECTURE.md` §10](ARCHITECTURE.md).

## What's built

| Layer | Contents | Status |
|-------|----------|--------|
| L1 Core | config, errors, retry, rate limits, policy, events, logging, HTTP | ✅ tested |
| L2 Storage | SQLite, 52 tables, migrations, FTS5, vectors, blobs, queue, backups | ✅ tested |
| L3 Cognition | memory, embeddings, LLM router, model registry, HF download, **training pipeline** | ✅ tested |
| L4 Capability | filesystem, sandboxed shell, web, parsers, vision, media, **social** | ✅ tested |
| L5 Agents | task DAG, thread/process/async runtime, budgets, supervisor, orchestrator | ✅ tested |
| L6 Missions | crash-resumable long-running goals, checkpoints, reflection | ✅ tested |
| L7 Surface | CLI · HTTP API · TUI | ✅ tested |

**~4,000 tests.** `python3 -m unittest discover -s tests -t .` — the suite runs offline by default (network calls are mocked); see `tests/taxonomy.py` for the unit/integration/live tier map and the `NM_RUN_INTEGRATION=1` gate for live tests.

## Architecture

Seven layers with a hard rule: **a module may import only from strictly lower
layers.** Enforced by [`tests/test_layering.py`](tests/test_layering.py), not by
convention.

```
L7 SURFACE     cli.py · api/server.py
L6 MISSIONS    long-running autonomous goals
L5 AGENTS      tasks.py · runtime.py · supervisor.py · orchestrator.py · roles/
L4 CAPABILITY  tools/ (filesystem, shell, web, parsers, vision, media)
L3 COGNITION   memory/ · llm/ · training/
L2 STORAGE     storage/ (db, migrations, fts, vectors, blob, queue, backup)
L1 CORE        core/ (config, errors, policy, retry, events, http, …)
```

Full design: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Parallelism

`agents/runtime.py` runs a task DAG across three pools at once and places each
task by declared kind — threads for I/O, processes for CPU, an event loop for
async. Getting this right is subtle and the failure mode is silent: a graph that
reports `done` while running serially looks identical to one that is genuinely
parallel.

Measured on the 2-CPU development sandbox:

| Workload | Serial | Parallel | Speedup |
|----------|--------|----------|---------|
| 8 × CPU-bound | 2.21 s | 1.11 s | 2.0× |
| 16 × CPU-bound | 4.21 s | 2.18 s | 1.9× |
| 20 × 50 ms I/O | 1.00 s | 0.16 s | 6.3× |

`ExecutionReport.speedup_vs_serial` is reported on every run, so a regression to
serial execution is visible rather than assumed away. Tests assert on wall time,
not just on results coming back.

Three defects were found by running this code rather than reading it, and all
three were silent:

- `ProcessPoolExecutor(max_tasks_per_child=64)` with the default `fork` context
  raised at pool creation. CPU tasks quietly fell back to threads and reported
  success with **1.0× speedup**.
- `submit()` pickles on a queue-management thread, so `except PicklingError`
  around `submit()` never fired. Now probed with `pickle.dumps` first.
- Waiting on in-flight futures after `cancel()` made a 0.2 s deadline take
  5.00 s. Cancellation now short-circuits the drain.

## Memory

Recall merges four normalized signals — recency (exponential decay, half-life ×8
for facts), importance (with bounded log-linear access reinforcement), semantic
(cosine), and lexical (BM25 from FTS5). Weights default to
`{recency .25, importance .30, semantic .30, lexical .15}` and the reflector may
retune them.

Embeddings fall back to deterministic feature hashing with a suffix-stripping
stemmer, so recall works with no model and no network. Stated `fact` and
`preference` records are never auto-forgotten.

## Models

`llm/router.py` keeps a health-scored fallback chain and hot-swaps at runtime:

```python
router.set_active("dolphin-8b")   # next call uses it
```

`llm/registry.py` holds a curated catalog of uncensored open-weight models and
enforces a **promotion gate**: a self-trained model is not activated unless
`beats_incumbent()` passes. Without that gate an auto-finetune loop monotonically
degrades the system — every run that "trains successfully" gets promoted
regardless of whether it improved anything.

## Security

Tools declare a capability; `ToolRegistry.call()` checks it against the caller's
grant before dispatch and audits actor, decision, argument digest, and duration
to the `tool_calls` table. Auditing failures never break a tool call.

Verified by test:

- `fs_read("../../etc/passwd")` raises rather than normalizing the path
- symlink escapes are refused after resolution
- `fs_delete` requires a single-use confirmation token bound to `fs.delete`
- a `memory.read` grant cannot invoke `exec.shell`
- denials are written to the audit log, not just allowed ones
- a timed-out shell command kills the whole process group — zero orphans left

Shell execution uses the strongest isolation available (`bwrap` → `unshare` →
`setrlimit`), with network disabled by default and a hard wall-clock kill.

## Optional dependencies

Everything has a fallback. `python3 -m nomorals doctor` prints what is present:

| Package | Enables | Fallback |
|---------|---------|----------|
| `numpy` | vector similarity, training | pure-Python cosine, ~30× slower |
| `yt-dlp` | download from ~1800 sites | direct-URL downloader |
| `torch` | GPU fine-tuning | pure-Python trainer, or generated external configs |
| `transformers` | HF tokenizers | built-in BPE in `training.tokenize` |
| `huggingface-hub` | cached resumable downloads | raw HTTP range requests |
| `pillow` | rich image decoding | built-in PNG/JPEG/GIF/BMP header parser |

## Portability

`ARCHITECTURE.md` §11 defines three profiles. `termux` forces process pools off
(`fork` is unreliable on Android), drops context to 4k, and excludes blobs from
backups.

## Honest status

This is a working, tested system — past the 100,000-line target it was designed
around. Roughly 187,000 lines of Python across 550+ modules, with ~4,000 tests.
around. Roughly 209,000 lines of Python across 540+ modules, with ~4,000 tests.
The count is reported as measured (blank lines and comments included); no filler
was ever added to hit a number — the tree grew because the feature list did:
agent swarm, self-improvement engine, watchers, project rooms, morning briefing,
universal media editing, voice loop with an expressive clone engine, vision,
cross-platform chat (Telegram bot + userbot, WhatsApp bridge), finance brain
with free market-data endpoints, and a full coding-agent toolchain.

Everything in ARCHITECTURE.md is implemented. Live integrations (Hugging Face
model fetches, market-data endpoints, news/proxy/weather feeds) are verified
against the real APIs from environments with network access; anything not yet
live-tested is marked as such in its commit notes.

See [`ROADMAP.md`](ROADMAP.md).

## Development

```bash
python3 -m pytest tests            # ~2000 tests
python3 -m nomorals doctor                   # environment report
```

## License

Your machine, your models, your data.
