# Roadmap

Status as of the current commit. Verified means covered by a test that runs
offline in CI, or executed by hand with the result recorded here.

## Done

### L1 Core — `nomorals/core/` (14 modules, 105 tests)
Config with profile inheritance and env override, typed error hierarchy with
retryability, retry with jittered backoff, token-bucket rate limiter, capability
policy with narrowing inheritance and confirmation tokens, event bus, structured
logging with secret redaction, ULID ids, monotonic clock, stdlib HTTP client with
resumable range downloads.

### L2 Storage — `nomorals/storage/` (86 tests)
Thread-local pooled connections, 7 forward-only checksummed migrations to 52
tables, FTS5 full-text search with query building and snippets, vector store with
pure-Python cosine and an IVF path above 512 rows, content-addressed blob store
with dedup and refcounting, durable work queue with leases and dead-lettering,
versioned gzip backups with git push and restore.

Verified: schema v7, `integrity_check()` clean, 13500 B blob → 159 B (ratio
0.012), backup gunzips to a valid `SQLite format 3` header and restores clean.

### L3 Cognition — `memory/`, `llm/` (57 tests)
Four-store memory behind one façade, feature-hashing embeddings with stemming,
provider router with health/cooldown/failover, model registry with promotion
gate, resumable SHA-256-verified HF downloader.

### L4 Capability — `nomorals/tools/` (39 tests)
20 tools across filesystem, sandboxed shell, web, parsers, vision, media. Stdlib
parsers for PDF, DOCX, XLSX, ODT, EPUB, HTML, CSV, JSON.

### L5 Agents — `nomorals/agents/` (44 tests)
Task DAG with cycle rejection, hybrid thread/process/async runtime, budgets with
halving inheritance, supervisor restart ladder, master orchestrator with plan
repair, 10 role agents.

### L6 Missions — `nomorals/missions/` (37 tests)
Crash-resumable long-running goals. State, plan, progress, and budget persist to
SQLite; checkpoints are append-only so a torn write cannot lose the previous
state. `resume_all()` continues every interrupted mission on startup.

Verified for real: a subprocess is SIGKILLed mid-mission, and a fresh process
resumes from the persisted checkpoint. The test asserts that steps completed
before the kill are **not** re-executed, and that two kills in a row still
converge to a terminal state.

The guarantee is at-least-once, not exactly-once: if the process dies after a
step completes but before the checkpoint lands, that step repeats. Steps must be
idempotent or record their own completion.

### L3 Training — `nomorals/training/` (54 tests)
Dataset codecs (ChatML/Alpaca/ShareGPT/OpenAI), a BPE tokenizer trained from
scratch in stdlib, simhash dedup and quality filtering, a native pure-Python
next-token trainer with analytic gradients, and a run registry wired to the
promotion gate.

Verified end to end: train loss falls monotonically (6.62 → 6.38 over 4 epochs,
eval 5.75 → 5.55), the gate rejects a model scoring 0.153 against an incumbent at
0.500 and leaves the incumbent live, and a forced promotion is recorded in the run
row.

The native trainer is a single-hidden-layer softmax, not a transformer. It exists
so the self-improvement loop is exercisable on a phone with nothing installed;
when torch is present it should be the backend instead.

### L4 Social — `nomorals/social/` (46 tests)
Multi-platform publishing on official APIs only. One call fans out across every
connected account on the L5 thread pool, with per-platform failure isolation.

Verified: four platforms at 0.15s each complete in under 0.45s wall (serial would
be 0.6s), so the fan-out is genuinely parallel. A failing platform does not stop
the others and does not lose the post.

Two rules it does not bend: official APIs only, and credentials are never stored.
`Account.credentials` holds `env:NAME`, resolved at call time, because this
database is backed up to a git repo. A test asserts the resolved token never
appears in the serialized row.

Bulk posting requires a confirmation token. `social.bulk` is a confirmable
capability, so a prompt-injected agent cannot blast every connected account on its
own authority; `publish()` now takes the token and a test asserts the refusal.

### L7 Surface — `cli.py`, `api/server.py`, `tui/` (46 TUI tests)
CLI, a threaded stdlib HTTP API with bearer auth, and a curses TUI. All three
executed end to end.

The TUI splits logic from drawing: `tui/model.py` holds state, wrapping, key
meaning, and layout as pure functions, and `tui/app.py` is a thin curses driver.
That split is why the TUI is testable at all — written directly against curses,
none of its layout or key handling could be verified without a terminal.

`nm tui` dispatches slash-commands (`/mem`, `/recall`, `/tools`, `/models`,
`/missions`, `/doctor`, `/clear`) and sends anything else to the active model.

## Not done

Ordered by value, not by effort.

### `storage/models.py`
The one L2 gap. Tables are defined in migrations and rows come back as
`sqlite3.Row`; dataclass mirrors would give typed access.

### L7 TUI
The CLI and API exist. An interactive terminal UI does not.

## Known unverified

Stated plainly, because the difference between written and verified matters:

- **Hugging Face integration is untested against the live API.** This sandbox
  cannot reach huggingface.co (TLS closed). `llm/download.py` and
  `providers/hf_serverless.py` are written to spec but have only been exercised
  offline. Do not assume they work until run against a real token.
- **Video download is unverified.** No `yt-dlp` and no `ffmpeg` are installed
  here. `tools/media.py` reports its own capability honestly via
  `media_capability()`, and the direct-HTTP fallback is the only path exercised.
- **Video/media tools are unexercised beyond the direct-HTTP path.**
- **`torch`, `numpy`, `transformers`, `pillow` paths are unexercised.** No
  third-party packages are installable here (PEP 668 externally-managed). Every
  pure-Python fallback is tested; every accelerated path is not.

## Conventions worth keeping

Learned the hard way; each one cost a debugging cycle.

- **Never introspect `dataclasses.fields()[i].type`** — with
  `from __future__ import annotations` it is a string. Use `get_type_hints`.
- **Never signal failure through a context manager's `__enter__` return value.**
  The body runs regardless.
- **Never call `executescript()` inside a transaction** — it implicitly commits.
- **Reset a cancellation flag in the entry point, not the worker.** Clearing it
  in `run()` discarded a `cancel()` requested from another thread, because
  `resume_all()` also calls `run()`.
- **An append-only table has no `updated_at`.** The Repository injects timestamps
  by default; overriding `timestamp_columns` is required for immutable tables.
- **`Repository.find()` has no `limit` parameter.** It turns every keyword into a
  `WHERE` clause, so `find(limit=50)` became `WHERE "limit" = 50` and silently
  matched nothing. Use `repo.query().limit(n).build()`. This bug shipped in three
  places before a test caught it.
- **Classify failures by exception type, not by substring-matching the message.**
  The supervisor retried a `BudgetExceeded` three times because its message said
  "out of tokens", not "budget".
- **A graph that reports success may be delivering zero parallelism.** Assert on
  wall time and on `stats_snapshot()["processes"] > 0`.
- **`ProcessPoolExecutor.submit()` pickles lazily** on a background thread. Probe
  with `pickle.dumps` before submitting if you want to catch it.
- **An audit path must never raise.** Auditing failures get logged, not
  propagated, or a logging bug takes down every tool call.
- **Write the two-step version.** Clever one-line conditionals in dataclass field
  defaults parse unexpectedly and index possibly-empty lists.
