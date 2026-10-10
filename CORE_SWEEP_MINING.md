# CORE SWEEP — External Mining Report

Module: `nomorals/core/` (41 files, ~24.5K lines). Every significant class was
compared against the best implementation found OUTSIDE the repo (best AND
trash). Findings below drive the implementation; each item names the source
and the concrete gap to close.

Constraint note: `core/__init__.py` forbids imports from other `nomorals`
subpackages, so all styling/upgrades stay inside `core/` (a new shared
`style.py` lives here, not in some other module).

---

## 1. Resilience: retry / breaker / budgets / degradation

### `retry.py` — BackoffPolicy / CircuitBreaker / ResiliencePipeline
- **Best: tenacity** (jd/tenacity). Feature checklist ours lacks: stop
  strategies composable with `|`/`&` (`stop_after_attempt`,
  `stop_after_delay`, `stop_before_delay`, `stop_when_event_set`);
  wait strategies incl. `wait_random_exponential`, `wait_chain`,
  `wait_incrementing`; result-based retry (`retry_if_result`); async-native
  support; `before_sleep` callbacks. **Gaps to close:** deadline-based stop
  (`max_elapsed`, tenacity's `stop_after_delay`), an event-based stop
  (`stop_when_event_set` — graceful-shutdown integration), async variants
  (`aretry_call`/`aretry` — tenacity retries async coroutines; our
  Discord/Telegram adapters are async and currently can't use `retry`).
- **Best: pybreaker** (danielfm/pybreaker). `CircuitBreakerListener` with
  `before_call` / `state_change` / `failure` / `success` hooks; Redis-backed
  state storage for cross-process breakers; thread-safety. **Gaps:** listener
  protocol (we only have `on_open`/`on_close` callbacks) and a pluggable
  state store.
- **Best: Polly v8** (.NET). Pipeline ordering
  `fallback → breaker → retry → hedging → timeout` — our
  `ResiliencePipeline` already mirrors this. **Gold already taken.**
- **Trash mined:** hand-rolled `for i in range(3): try/except: sleep(2)`
  loops in random gists — no jitter (thundering herd), no classification of
  retryable vs terminal, no budget. Our error-dialect classification
  (`retryable` flag on the error itself) is the correct design; keep it.

### `ratelimit.py` — TokenBucket / GCRA / SlidingWindow / SemaphorePool
- **Best: the algorithm literature + production limiters** (Cloudflare/Stripe
  writeups, throtto, ratelink, mansi75/rate-limiter). Consensus table:
  token bucket = right default (burst-tolerant, 2 numbers of state); GCRA =
  cheapest exact meter (1 timestamp, token-bucket-equivalent); sliding-window
  counter = best window default (~99%, O(1)); sliding-window log = exact but
  O(limit) memory; fixed window = footgun (2× boundary burst — correctly NOT
  implemented here); leaky bucket = traffic *shaper* (smooth output, queue).
  **Gaps:** (a) **LeakyBucket** as a pacer — we have meters but no shaper for
  outbound request pacing; (b) **string presets** (`"100/minute"`, throtto's
  DX); (c) a **store/backend protocol** so limiters can go distributed
  (Redis Lua) later — ratelink's decoupled backend design; (d) `Retry-After`
  / `X-RateLimit-*` headers — already have `RateLimitDecision.to_headers()`.
  **Gold already taken:** GCRA with burst proof, weighted sliding-window
  counter with exact `wait_time`, per-cost decisions.

### `budgets.py` — ErrorBudget / BurnRate
- **Best: Google SRE Workbook Ch.5** (multi-window, multi-burn-rate
  alerting). Canonical 30d config: page at 14.4× (5m+1h), page at 6×
  (30m+6h), ticket at 3× (2h+24h), ticket at 1× (6h+3d); two-window rule
  (short proves it's happening *now*, long proves it's sustained);
  burn-rate = observed error ratio ÷ (1 − SLO). **Gold already taken** —
  ours implements exactly this. **Gaps (from the same source):** the
  *policy* layer that makes budgets change behavior — "without a policy,
  the error budget is a decoration": soft freeze (<20% remaining → deploys
  need approval), hard freeze (exhausted → reliability work only). We emit
  warning/page alerts but expose no **decision API** (`policy_action()` →
  `ok|warn|freeze`). Also missing: budget **forecast** (time-to-exhaustion
  curve) and a styled status report for chat/dashboards.

### `degradation.py` — DegradationLadder
- **Best: Netflix/AWS graceful-degradation practice** (documented
  degradation ladders per subsystem, honesty clauses, auto-recovery probes).
  **Gold taken:** rung honesty requirement enforced in `__init__`, probe-based
  auto-climb. **Gaps:** (a) `LadderExhausted` is a bare `Exception` — breaks
  the "one error dialect" rule; make it a `NoMoralsError`; (b) no transition
  history / `on_transition` hook for observability; (c) no styled status
  report (`LadderManager.format_report()`).

## 2. Errors: errors / result / incidents / error_*

### `errors.py` — NoMoralsError hierarchy + classify
- **Best: Rust anyhow/thiserror** (the consensus: thiserror for libraries =
  typed matchable errors; anyhow for applications = context chaining).
  Our design already matches the thiserror half (typed codes, registry,
  `from_dict` round-trip). **Gap = the anyhow half:** no `.context()`-style
  breadcrumb chaining — bare `raise` loses *where/what* context. Add
  `with_context()` / `note()` accumulating `details["context"]` plus
  `cause_chain()` / `format_chain()` for printing the full chain.
- **Trash mined:** stringly-typed `raise Exception("...")` everywhere,
  errno-soup, and error classes that lose the original traceback. Our
  `classify()` + code registry is the fix; keep.

### `result.py` — Ok/Err
- **Best: Rust `Result` / Haskell `Either`.** We mirror the core API
  (`map`, `and_then`, `or_else`, `unwrap*`, `expect`, `partition`).
  **Gaps:** `map_err` (transform the error without unwrapping — Rust has
  it), `tap`/`inspect` (side-effect passthrough), `flatten` for nested
  outcomes, and a `collect()` that keeps *all* errors (vs `unwrap_all`'s
  fail-fast). Small, high-value.

### `incidents.py` — IncidentJournal
- **Best: incident-management practice** (Blameless postmortems, PagerDuty).
  We have signatures, causal chains, spike detection, heartbeat ratios.
  **Gaps:** MTTR/MTTA stats, `top_subsystems()` rollup, styled report.

### `error_doctor.py` / `error_intelligence.py` / `error_system.py` / `self_heal.py` / `selfheal.py`
- **Best: Sentry** (issue grouping by stack signature, suspect commits,
  flakiness); **best self-healing literature** (MAPE-K loop:
  Monitor-Analyze-Plan-Execute over a Knowledge base; automated remediation
  with blast-radius guards and dry-run-first). **Gold taken:** stack
  signatures, learned-fix KB with confidence/demotion, AST-based fix
  strategies with `_compiles()` guard. **Gaps:** (a) **flakiness detection**
  (error that appears *and* resolves repeatedly — Sentry flags these);
  (b) trend analysis (counts per bucket over time); (c) styled diagnosis
  output; (d) `self_heal` fixes should offer **diff preview** before
  writing (blast-radius guard).

## 3. Observability: logging_setup / observability / events / clock

### `logging_setup.py`
- **Best: structlog** (processor pipeline, immutable bound loggers,
  `contextvars` integration, dev console renderer vs prod JSON renderer).
  **Gold taken:** contextvars injection, redaction filter, deterministic
  sampling. **Gaps:** (a) **output themes** — structlog's
  `ConsoleRenderer(colors=…)`; we have one hard-coded palette (and a note
  the owner hates red — honor it in every theme); add named `LogTheme`s
  incl. a ninja/electric-blue theme matching the user's Termux theme;
  (b) a structlog-style **event API**: `log_event(logger, "order.created",
  order_id=…)` emitting stable `event` names + fields instead of f-strings.

### `observability.py` — Metrics / Tracer / HealthChecker
- **Best: Prometheus** (RED method, `_total`/`_seconds` naming, label
  discipline — route templates not raw paths) and **OpenTelemetry**
  (W3C `traceparent`/`tracestate` propagation, span kinds, semantic
  conventions, tail-based sampling keeping errors). **Gold taken:**
  Prometheus exposition, OTLP-shaped export, deterministic head sampling +
  always-keep-errors tail rule, traceparent parse/continue.
  **Gaps:** (a) `Span.inject(headers)` — W3C *injection* for outbound calls
  (we only extract); (b) `Tracer.current_traceparent()` convenience for log
  correlation (the OTel logging-correlation pattern).

### `events.py` — EventBus
- **Best: blinker** (named-signal registry, weak-ref auto-disconnect,
  `connect` with sender filtering, thread-safety) and durable-bus practice
  (DLQ, redelivery with backoff, dedup windows). **Gold taken:** priority
  subs, once-subs, sync/async dispatch, retries + DLQ + replay, dedup,
  journaling. **Gaps:** (a) `wait_for(pattern, timeout)` — block until the
  next matching event (orchestration primitive); (b) data-predicate
  filtering on subscribe (blinker filters by sender; we filter by topic
  only).

### `clock.py` — Clock / Deadline / Backoff
- **Best:** injectable-clock pattern (every timing path testable), deadline
  propagation (gRPC-style: pass *remaining* down, never re-timeout).
  **Gold taken.** **Gaps:** a `Throttle` (call at most once per interval —
  for "alert me at most every X" paths) and human `Deadline.__str__`.

## 4. Data & text: text / jsonutil / ids / diff / corpus / decoder / cookies

### `text.py`
- **Best: datasketch** (MinHash + LSH: 128 perms, k=5 shingles, banding for
  sub-linear candidate retrieval; SimHash for Hamming-space near-dupes).
  **Gold taken:** SimHash, shingling, Jaccard, Levenshtein. **Gap: MinHash**
  itself — datasketch's core primitive — plus `jaro_winkler` (record
  linkage standard) and `slugify`.

### `jsonutil.py`
- **Best:** LLM-JSON recovery practice (fence extraction, balanced-scan,
  then *lenient* repair). **Gap:** no lenient pass — trailing commas,
  `//` comments, single quotes all return None today. Add `parse_lenient()`.

### `ids.py` — ULID
- **Best: ULID spec** (ulid/spec): 26-char Crockford base32, 48-bit ms
  timestamp + 80-bit randomness, max `7ZZ…Z` (first char 0–7),
  case-insensitive, monotonic within the same ms by incrementing randomness.
  **Gold taken:** monotonic `_MonotonicRandom`. **Gaps:** strict
  `is_ulid()` validation per spec (reject first-char > 7 — a real
  interop bug class), `ulid_at(ms)` deterministic generation, bytes/UUID
  conversions, `ULID.next()`.

### `diff.py`
- **Best: unified-diff tooling** (patch(1), difflib). We only *apply*.
  **Gap:** generate diffs (`unified_diff`) and colorized `format_diff`.

### `corpus.py` — password mutation rules
- **Best: hashcat rule engine** (`:` no-op, `l`/`u`/`c` case, `r` reverse,
  `d` duplicate, `f` reflect, `p` pluralize, `$x`/`^x` append/prepend,
  `sxy` substitute, `T` toggle-at). Ours has leet/year rules.
  **Gap:** the classic single-char rules (reverse, duplicate, reflect,
  toggle, capitalize) — the highest-yield hashcat rules.

### `decoder.py`
- **Best:** CyberChef (the "magic" operation: try-everything decoding with
  confidence scoring). **Gold taken:** registry + `_bytes_hit` commonness
  gate + word scoring. **Gaps:** base85/ascii85, quoted-printable, ROT47 —
  all CyberChef staples we lack.

### `cookies.py` — CookieLab
- **Best:** `http.cookiejar` (RFC 6265 domain/path matching, expiry) +
  security-scanner cookie analysis (flags audit: Secure/HttpOnly/SameSite).
  **Gold taken:** classification, JWT/base64/gzip decoding, fingerprinting.
  **Gaps:** `Set-Cookie` *serialization* (`to_header()`) and a real
  `CookieJar` with domain matching for replay.

## 5. Media/codec: pdf / midi / barcode / cipher

### `pdf.py` — pure-Python PDF render + parse
- **Best: fpdf2** (tables with styled headings, header/footer callbacks,
  TOC/bookmarks, markdown-ish input) and **reportlab Platypus** (flowables:
  paragraphs, tables, page templates). **Gold taken:** styled blocks,
  pagination, xref generation, text extraction. **Gaps:** real **tables**
  (markdown `| a | b |` → ruled grid), **page numbers** (`Page X of Y`),
  document metadata (author/subject), TOC bookmarks.

### `midi.py` — theory + MIDI file builder
- **Best: mido** (message objects, 18 MIDI message types, ports, SYX,
  MIDI-over-TCP) and **pretty_midi** (analysis-friendly). Ours is a
  *composition* engine (generators) + file writer, which mido is not —
  different lane, keep it. **Gaps:** **GM instrument table** (128 program
  names — every MIDI tool has it), **GM drum map**, `quantize()`.

### `barcode.py`
- **Best: python-barcode** (code39, code128, ean13/8, upc, itf, isbn…),
  **pyzbar/zxing** (decode), **tapirscan** (modern reader).
  **Gold taken:** code128 + EAN/UPC-A encode/decode with check digits.
  **Gaps:** **Code39** (the other universal 1D code), **ITF** (logistics),
  **EAN-8** (retail small packages), and an ASCII **render** for chat
  previews. (QR/DataMatrix deliberately out of scope — needs 2D
  Reed-Solomon; honest omission.)

### `cipher.py` — pure-Python AES
- **Best:** `cryptography` (AEAD: AES-GCM/ChaCha20-Poly1305; **HKDF**
  RFC 5869 for key separation; constant-time compare). **Gold taken:**
  AES-CTR + HMAC tag, PBKDF2. **Gaps:** **HKDF** (derive separate
  enc/mac keys from one master — real KDF hygiene), high-level
  `seal()`/`unseal()` with associated data.

## 6. Platform/config: config / runtune / profile / platform / owner / tz / trust / verify / http / policy / shutdown / tasks

### `config.py` (1758 lines of settings dataclasses)
- **Best:** pydantic-settings / 12-factor config (env binding, validation,
  secret redaction in `repr`). **Gaps:** `redacted_dict()` (the user noted
  "~20 secrets is nothing compared to the loads of config" — secrets must
  never leak into logs), `bind_env(prefix)` 12-factor binding,
  `validate()` returning human issues.

### `runtune.py` / `profile.py` / `profiles.py` / `platform.py`
- **Best:** runtime capability detection (Termux/Android quirks, Docker,
  WSL) + profile-gated tuning tables. **Gaps:** Docker/WSL detection,
  styled `describe()` output, per-profile tune presets surfaced readably.

### `tz.py`
- **Best:** zoneinfo best practice + never-crash wrappers.
  **Gold taken** (the Termux-missing-tzdata fallback chain).
  **Gaps:** convenience (`now_in()`, `format_ts()`) — tiny.

### `trust.py` — SourceTrust
- **Best:** PageRank-style reputation + per-source learning (what ours
  does). **Gaps:** `explain(url)` returning the score *breakdown*
  (tier/learned/corroboration/staleness) — debuggability; explicit
  allow/block lists.

### `verify.py` — LiveVerifier
- **Best:** health-check practice (TCP/TLS/DNS/HTTP probes with latency).
  Ours checks *tools and tokens*. **Gaps:** generic network probes
  (`check_tcp`, `check_tls` incl. cert-expiry, `check_dns`,
  `check_http` with latency) + a pluggable `add_check()` registry.

### `http.py` — HttpClient
- **Best: httpx/requests** (streaming, conditional requests, content
  sniffing). **Gold taken:** SSRF guard, proxy/SOCKS, retry-after parsing,
  multipart. **Gaps:** streaming **`download()`** with progress callback
  and resume.

### `policy.py` — capability policy
- **Best:** object-capability literature (seL4/Caja: unforgeable,
  attenuatable references; "privilege narrows down the tree, never
  widens" — already our model). **Gaps:** `PolicyDecision.explain()` and
  `Policy.describe()` — a human-readable rendering of *why* a decision
  was made (audit-grade).

### `shutdown.py`
- **Best:** the 4-phase pattern — Signal → Stop Accepting → Drain →
  Teardown — with bounded grace periods and health-probe awareness.
  Ours has priority bands encoding exactly these phases.
  **Gaps:** named decorators per phase (`on_stop_accepting`, `on_drain`,
  `on_close`, `on_beacon`), a `shutdown_event()` for background loops,
  `wait_for_shutdown()`.

### `tasks.py` — TaskGraph
- **Best:** DAG schedulers (Airflow: topological levels for parallel
  waves; critical-path analysis). **Gaps:** `topological_order()`,
  `levels()` (parallel waves), `critical_path()`, `to_dot()` for
  visualization.

---

## Cross-cutting STYLE findings
- No shared presentation layer: every `*_report`/`status()` returns raw
  dicts; chat surfaces re-format ad hoc. **Fix:** new `core/style.py` —
  `Theme` (named palettes incl. ninja/electric-blue honoring "owner hates
  red"), `paint()`, `styled_table()`, `styled_box()`, `sparkline()`,
  `progress_bar()`. `logging_setup` themes plug into it.
- Status dicts are machine-first. Add `format_*()` human renderers on:
  budgets, incidents, error intelligence, degradation, tasks, ratelimit
  (`LimiterRegistry.describe()`), retry (`CircuitBreaker.describe()`).

---

## Implementation log — session 2 (2026-10-10)

Everything below was implemented, smoke-tested, and covered in
`tests/test_core_sweep.py` (50 tests, all green; existing
`test_core.py`/`test_cookies.py`/`test_policy_grant_h3.py` still green).

### ids.py
- **Mined:** ULID spec (ulid.spec.js) — monotonicity within the same ms,
  spec-strict validation (first char 0–7; the classic interop bug is
  accepting 26-char non-ULIDs), 128-bit binary layout.
- **Added:** `is_ulid()` (strict), `ulid_at(ms)`, `ULID.at()`,
  `ULID.next()` (monotonic increment), `ULID.to_bytes()/from_bytes()`,
  `ULID.to_uuid()/from_uuid()`.

### diff.py
- **Mined:** unified-diff as the universal patch interchange (git/LLM
  tooling all speak it).
- **Added:** `unified_diff()` (generation — the inverse of applying;
  round-trips through `apply_unified_diff`), `format_diff()` (colored:
  magenta deletions per the no-red house rule, green additions, blue
  hunk headers).

### cipher.py
- **Mined:** RFC 5869 HKDF (extract-then-expand); libsodium `crypto_secretbox`
  pattern (one key → separated subkeys, associated data bound into the tag).
- **Added:** `hkdf()` (verified against the RFC 5869 test vector),
  `seal()`/`unseal()` — HKDF-separated enc/mac subkeys, associated-data
  binding, `nmc2$` self-describing blobs.

### decoder.py
- **Mined:** CyberChef's decoder breadth.
- **Added:** `_Base85Decoder` (ascii85, Adobe `<~ ~>` + raw variants),
  `_Rot47Decoder` (wordiness-gated so plain text isn't "decoded").

### barcode.py
- **Mined:** python-barcode (installed as a *verification oracle*, not a
  dependency).
- **Added:** Code 39 (full 43-char table — **verified entry-by-entry**
  against python-barcode's charset; 41/43 matched from memory, `8`/`9`
  corrected to the reference; start/stop `*` == python-barcode's EDGE
  run), ITF (table verified; pattern-structure matches the reference
  exactly — width scale 3:1 vs their 5:2, both valid; decode is
  ratio-agnostic), EAN-8 (**bit-exact** match with python-barcode),
  `render_ascii()`, `analyze()` extended to all six symbologies.

### midi.py
- **Mined:** every MIDI tool ships the GM tables; DAW quantize with
  strength control.
- **Added:** full 128-program `GM_INSTRUMENTS`, `GM_FAMILIES`, complete
  35–81 `DRUM_MAP`, `program_name()`/`program_number()` (exact → prefix →
  unique substring), `gm_family()`, `quantize()` with strength blend.

### pdf.py
- **Mined:** fpdf2/reportlab feature surface (tables, document info dict,
  page X of Y footers).
- **Added:** markdown pipe-table parsing → pre-aligned Courier rows with
  dashed header rule and `:---`/`---:` alignment; `/Info` metadata dict
  (title/author/subject/keywords/creator + Producer/CreationDate) on both
  render paths; `{total}` footer substitution ("Page N of {total}").

### corpus.py
- **Mined:** hashcat rule engine (`r`/`d`/`f`/`t`/`T0`/`{`/`}`).
- **Added:** `reflect` (hashcat `f`), `toggle` (`t`), `toggle_each`
  (`T0`…`Tn` — positional), `rotate_l`/`rotate_r` (`{`/`}`); fixed a
  duplicated `{low}!` in `d1`.

### cookies.py
- **Mined:** `http.cookiejar` / requests' RequestsCookieJar.
- **Added:** `Cookie.to_header()`, `Cookie.is_expired()`
  (Max-Age/Expires), `cookies_to_header()`, `CookieJar` (update/header/
  get/names/clear/to_dict, expiry-aware). **Fixed a real parser bug:**
  `Cookie: a=1; b=2` request headers previously swallowed `b=2` as a flag.

### trust.py
- **Mined:** reputation systems need operator override + explainability.
- **Added:** `allow()`/`block()`/`unallow()`/`unblock()`/`allowed()`/
  `blocked()` (persisted in KV, in-memory fallback when no db),
  `score()` now honors pins via a shared `_breakdown()`, `explain()`
  with a one-line `summary`.

### tz.py
- **Added:** `now_in()`, `to_utc()`, `format_ts()`, `parse_ts()`
  (ISO 8601 + common spellings + epoch; naive → tz-aware).

### config.py
- **Added:** `redacted_dict()` (secret-name sniffing: token/secret/
  password/api_key/… → `***`; safe to log), `validate_settings()`
  (declared-type checks + negative-budget guards). `bind_env()` was
  *not* added — env binding already exists via `load_settings(env=…)`
  + `env_var_path()`; a second path would be a parallel system.

### verify.py
- **Mined:** uptime/health probers (status checks as data).
- **Added:** `check_tcp()`, `check_tls()` (handshake + cert-expiry days,
  warn/fail thresholds), `check_dns()`, `check_http()` — all returning
  `VerificationCheck`; `LiveVerifier.add_check()` for custom checks
  (sync or async) wired into `verify_all()`.

### policy.py
- **Added:** `PolicyDecision.explain()` (headline + gates + audit id),
  `Policy.describe()` (rule counts by effect, top rules, enforcement
  state, lifetime counters).

### incidents.py
- **Mined:** SRE incident analytics (MTTR is the headline reliability
  metric everywhere).
- **Added:** `top_subsystems()` (volume, share, worst severity),
  `mttr()` (mean/median to first *verified* recovery, open count,
  human durations), `_human_duration()`.

### error_doctor.py
- **Added:** `format_diagnosis()` — styled multi-section report (headline,
  root cause, evidence, suggested fix, optional frames + chained cause).

### error_intelligence.py
- **Mined:** Sentry-style issue analytics (flaky-test detection uses
  inter-arrival variance).
- **Added:** `flakiness()` (episode clustering with adaptive gap +
  coefficient of variation → flaky/intermittent/steady/burst),
  `trend()` (bucketed least-squares slope → rising/falling/stable).

### http.py
- **Added:** module-level `download()` (dir-or-file-or-cwd destination,
  optional progress printer), `HttpClient.download()` now accepts a
  directory (filename from URL).

### platform.py / profiles.py / runtune.py
- **Added:** `is_docker()`, `is_wsl()`, `Platform.describe()` (styled),
  `Platform.to_dict()` gains container/wsl flags, `format_profile()`,
  `RuntimeTune.describe()` (sectioned report with provenance).

### owner.py
- **Mined:** OWASP password-storage guidance (memory-hard/KDF with work
  factor; single-round SHA-256 falls to GPUs).
- **Added:** `v2$` seals — PBKDF2-HMAC-SHA256, 600k iterations;
  `make_seal()` now mints v2, `verify_passphrase()` verifies v1+v2
  (backward compatible), `make_seal_v1()` kept for migration tooling.

### error_system.py
- **Added:** `health()` now includes `top_subsystems_24h` + `mttr_24h`;
  `format_health()` styled report. **Fixed a real bug:** `health()`
  unpacked `top_failing()` dicts as tuples (would have raised).

### selfheal.py
- **Added:** `RecoveryOutcome.format_outcome()` (styled), per-strategy
  `strategy_stats` track record on the executor.

### self_heal.py
- **Added:** `diff_preview()` — real unified diffs of proposed source
  fixes (replacing the one-line summaries), wired into both fixers.

## What's still weak / not done
- `decoder.py`: base85 covers ascii85 only, not RFC 1924 alphabet.
- `barcode.py`: no QR/DataMatrix (needs 2D Reed-Solomon — out of scope,
  documented); no `recover_ean8` hole-enumeration (EAN-13's `recover_ean`
  doesn't cover it yet).
- `owner.py` v2 uses PBKDF2 not Argon2 (stdlib-only constraint; PBKDF2
  at 600k is the honest stdlib ceiling).
- `verify.py` probes are sync (consistent with the module's existing
  urllib-sync style).
- `tasks.py` (TaskGraph: topological_order/levels/critical_path/to_dot)
  was mined but **not implemented** — ran out of session; prime candidate
  for the next pass.
