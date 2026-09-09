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

### L7 Surface — partial
CLI (`nm` / `python3 -m nomorals`) and a threaded stdlib HTTP API with bearer
auth. Both executed end to end.

## Not done

Ordered by value, not by effort.

### L6 Missions — the biggest gap
`ARCHITECTURE.md` §9 specifies this; nothing is written. A mission is a
long-running autonomous goal that survives restarts: persistent state, checkpoint
and resume, progress reporting, budget caps across hours, and escalation to the
operator when stuck. The pieces exist — durable queue, blackboard, supervisor,
budgets — but nothing composes them into an entity that outlives a process.

This is what turns "run a task" into "keep working on this until it's done."

### L3 Training — `training/`
Designed in `ARCHITECTURE.md` §7, zero lines written. Dataset codecs
(ChatML/Alpaca/ShareGPT), distillation from multiple models, dedup and quality
filtering, BPE tokenizer trained from scratch, a pure-Python training loop for
tiny models, config generation for Unsloth and LLaMA-Factory, perplexity
evaluation.

The promotion gate in `llm/registry.py` is already built and tested — the loop
that feeds it is not.

### L4 Social — `social/`
Multi-platform posting, scheduling, engagement collection. Must be restricted to
official APIs with rate limits (`ARCHITECTURE.md` §10). Nothing written.

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
