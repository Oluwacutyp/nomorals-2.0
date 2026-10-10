# TOOLS MINING REPORT — `nomorals/tools/` sweep (2026-10-10)

External research for all 79 files. Method: for every significant class, ask
"How does the best implementation of X do it outside this repo?" then "What
features SHOULD this have that it doesn't?" and "How should this LOOK/FEEL?"
Sources are linked; nothing was copied, only design laws extracted.

---

## 1. ToolRegistry / ToolSpec (`registry.py`)

**Now:** register/dispatch with capability gating, confirmations, audit trail
(args digest), per-tool stats, health probes (opencapx pattern), ranked tool
listing, description sanitization (prompt-injection filter), aliases,
deprecation, builtin modules isolated per-import, failure ledger.

**Best outside:**
- **Lelu (Lelu-ai/lelu)** — authorization engine with FOUR decisions:
  `allow / deny / human_review / compute`. `human_review` pauses the agent
  until a human approves; `compute` redirects the action to a safer
  alternative or sandbox. Adds prompt-injection detection (5-layer),
  confidence gating, anomaly/reputation scoring, full audit log.
  We only have allow/deny — no pause-for-review tier, no safer-alternative
  redirect, no confidence or anomaly gating.
- **MCP spec (modelcontextprotocol.io)** — every tool gets behavioral
  annotations: `readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`, set honestly so hosts can gate consequential calls.
  Descriptions must answer: what it does, when to use it (and when NOT
  to), what it returns, side effects, whether it needs confirmation.
  Schemas are tight: `enum`, `minimum`/`maximum`, `maxLength`, exact
  `required` list, `additionalProperties: false`, `outputSchema`.
  Errors must TEACH: `isError` + what failed + what to try instead + a
  stable code, and an explicit note when output is truncated.
  Destructive work is two-step (preview + idempotency key against
  duplicate retries). Tool definitions are prompt engineering — reviewed
  like prompts.
- **mcp-aggregator (marimerllc)** — lazy discovery (`find_tools(query)`
  returns only matching typed tools instead of dumping 500 schemas),
  timeout/retry/error-hints on proxied calls, idle connection auto-close.
- **rusty-gateway** — one supervised long-lived child per upstream,
  health-check + restart with backoff, redacted audit log in SQLite.

**SHOULD HAVE (gaps):**
1. MCP-style annotations on ToolSpec (`read_only`, `destructive`,
   `idempotent`, `open_world`) — honest behavior flags hosts can gate on.
2. `title`, `output_schema`, `examples` on ToolSpec; `schema()` emits them.
3. Tight schemas: `additionalProperties: false` + computed `required` list.
4. Teachable errors: when a tool call fails, suggest the fix/next tool;
   structured error codes.
5. Idempotency keys: `call(..., idempotency_key=...)` returns the cached
   result for a repeated destructive call (MCP two-step rule).
6. Result truncation with an explicit "truncated" notice (MCP rule).
7. `human_review` decision tier (Lelu): pause → approve → resume, wired
   through `confirm`.
8. Confidence-gated calls + anomaly scoring (later phase).

**LOOK/FEEL:** `describe()`/`health_report()` output should render as a
rich panel/table; error dicts should read like a coach, not a traceback.

## 2. CodeExecutor (`code_executor.py`)

**Now:** plan mode (LLM produces JSON plan) / act mode, step executor with
per-step confirmation, automatic rollback on failure, final test run.

**Best outside (LangGraph plan-and-execute, the canonical pattern):**
`PlanStep{step_number, description, expected_output}` → Executor (one step at
a time) → **Replanner** with conditional edge: `ReplanDecision{action:
continue|replan|finish, revised_steps, reasoning, final_answer}` →
**Synthesizer** that combines step results into the final answer. Advanced
variants add: loop detection (break execution loops), failure analysis
(adjust strategy to avoid repeating errors), time-aware planning,
tool-aware planner (plans only steps the toolset can do).

**SHOULD HAVE:** We stop dead at the first failed step. Add StepResult
records, a `replan()` pass that asks the agent to revise remaining steps
after a failure (with reasoning), loop detection (same step failing 3× =
abort with diagnosis), `expected_output` per step, and a `synthesize()`
final summary. Plan prompt should include the available tool catalog so
plans stay executable.

**LOOK/FEEL:** plan renders as a numbered checklist with risk badges
(low/medium/high), execution shows live step status (pending → running →
ok/failed), final synthesis is a short narrative + changes list.

## 3. EditLoop / surgical edits (`edit_loop.py`)

**Now:** Read → diff → apply → test → iterate; `_apply_edits_atomic`
(exact-unique match, overlap rejection, format-preservation check),
`apply_patch` (unified diff via `patch` binary or pure-Python fallback),
`edit_file` (exact surgical), Python syntax check + revert.

**Best outside (Aider's edit formats — the most benchmarked system):**
- `diff` (SEARCH/REPLACE blocks, git-conflict-marker style) is the default
  and dominates leaderboards; design laws: FAMILIAR / SIMPLE / HIGH-LEVEL
  / FLEXIBLE.
- **Flexible patching is worth 9× on apply success**: normalize hunks,
  relative leading whitespace, sub-hunk splitting, flexible context
  windows. Removing it radically increases failures.
- Prompt rules: FULL path verbatim unquoted; SEARCH matches char-for-char;
  first-match-only; multiple small blocks > one giant block; **empty
  SEARCH = new file creation**.
- Measured failure modes to tolerate: whole-file dumped into SEARCH,
  lazy `# ... code here ...` placeholders (udiff cut laziness 3×),
  missing `+` markers, uniform outdenting, jumping hunks without `@@`.

**SHOULD HAVE:** We demand exact matches and reject empty old_text.
Add: whitespace/indentation-tolerant fallback matching, a
`parse_search_replace_blocks()` parser for the aider-native LLM format,
`apply_search_replace` as a first-class tool, empty-SEARCH creates the
file, anti-laziness placeholder detection (reject/repair
`# ... rest of code ...`), and failure feedback that teaches the model
what to fix (Aider's "highly informative feedback when edits fail").

**LOOK/FEEL:** diffs rendered with context lines and +/- coloring;
failures show the closest match found ("did you mean this block at line
N?") instead of "old_text not found".

## 4. RepoIndex (`repo_index.py`)

**Now:** AST-based symbol/import/caller extraction, `RepoMap`,
`find_symbol`, `who_imports`, `callers`, `pack_context` with token budget.

**Best outside (Aider repo-map — the most-copied technique):**
tree-sitter parse → symbol extraction (functions, classes, methods,
types, top-level vars **plus cross-file references**) → dependency graph
→ **PageRank personalized by the files currently in context** → budgeted
output (default 1024 tokens) showing **signatures, not bodies**. The
intuition: the model needs to know *what exists and how it connects*,
not how it's implemented. Also: universal-ctags for 100+ languages,
sqlite tables for symbols/refs/tests/traces/commits, git churn (SZZ
algorithm links bug-introducing commits by blaming fixed lines).

**SHOULD HAVE:** Add an Aider-style `render_repo_map()`: reference-count
ranking (our PageRank stand-in, no new deps), personalized boost for
focus files, token-budgeted signature-only output, ctags fallback for
non-Python languages, git-churn signal for "hot" files. `pack_context`
packs whole files; the map should come first as the cheap layer.

**LOOK/FEEL:** the map reads like a table of contents with
importance-ordered entries; hot files marked.

## 5. Web extraction (`web.py`)

**Now:** custom `readability_extract` (boilerplate strip + block scoring
by length penalized by link density), search-result parsers (Bing, DDG,
lite, Mojeek), `html_to_text`.

**Best outside (trafilatura):** "the best out-of-the-box article
extraction tool in the Python ecosystem" — F1 ~0.94 in independent
benchmarks, builds on Readability with many more heuristics, extracts
metadata (title/author/date) and tables, `favor_recall` knob,
`fetch_and_extract` one-shot, markdown output. Fallback ladder used in
the field: **trafilatura first, readability-lxml for recall,
markdownify/html2text only for structure-heavy pages**. newspaper3k is
stale (2020).

**SHOULD HAVE:** trafilatura as the primary extractor (optional import,
zero new hard deps), our custom scorer as fallback; metadata JSON
(title/author/date/tags); `favor_recall` option; markdown output option
for LLM ingestion; extraction quality signal (words/confidence) so
callers know when the page defeated the extractor.

**LOOK/FEEL:** extracted article keeps headings/lists/tables as markdown;
callers get a one-line quality note ("main content, 1,240 words" vs
"low-confidence extraction — page may be JS-rendered").

## 6. Captcha (`captcha.py`)

**Now:** detection (reCAPTCHA v2/v3/enterprise, hCaptcha, Turnstile,
GeeTest, Arkose, AWS WAF, image/audio), backends (detect/takeover/
service via 2captcha in.php/res.php shape, capmonster-compatible),
`SolverRateLimiter` (per-minute/per-day budgets, exponential backoff,
6h hard cooldown, JSON-persisted), friendly errors, audit trail,
proxy support, takeover owner-ping with dedupe.

**Best outside:**
- **2captcha modern API:** `createTask` → poll `getTaskResult` every
  5–10s → `solution.token`; `reportCorrect`/`reportIncorrect` quality
  feedback; don't change proxies mid-captcha session.
- **ghostmcp 3-layer pipeline:** Layer 1 avoidance (rotator + stealth,
  free) → Layer 2 self-solve (VLM, free) → Layer 3 paid API fallback.
  **Budget controls**: daily cap (default $0 = disabled until configured),
  monthly cap (default $5), auto-reset on day/month boundary, per-solve
  cost tracked, `get_budget_status()` for audit visibility.

**SHOULD HAVE:** We lack spend tracking and the layered pipeline. Add:
`SolveBudget` (daily/monthly $ caps, auto-reset, per-solve cost ledger,
`budget_status()`), `report_solve(task_id, good)` quality feedback,
modern `createTask`/`getTaskResult` JSON API in ServiceBackend, and an
`auto` pipeline backend: detect → takeover-hints → budget-gated service,
escalating only when cheaper layers fail.

**LOOK/FEEL:** solve attempts render as a pipeline trace (detect ✓ →
service … → token in 34s, $0.002); budget status is a small dashboard.

## 7. ProxyLab (`proxylab.py`, `proxy.py`, `ssh_socks.py`, `proxysources.py`)

**Now:** Proxy dataclass with decayed latency, `score_proxy`
(anonymity > speed, time-decayed latency, failure streaks, NG-first),
ProxyScraper, ProxyTester, ProxyStore, DomainAffinityManager,
ProxyRotationManager (round_robin/random/sticky/least_used, cooldown,
failover, DIRECT fallback, kv-persisted).

**Best outside (production proxy operations):**
- **Tier-to-target matching**: default cheap datacenter, escalate to
  residential/mobile only for targets whose defenses demand it, cached
  per target — never pay residential rates for sites that never checked.
- **Session coherence**: bind IP + cookies + fingerprint for the life of
  a logical session (sticky); rotate WHOLE identities, never the proxy
  alone mid-session (a network origin jumping between requests is the
  classic self-inflicted wound). 2captcha docs agree: don't change
  proxies mid-captcha session.
- **Pool hygiene**: health-check, quarantine failing, retire burned IPs,
  **track success rate per subnet/ASN** (bans cluster by subnet, not IP).
- **Rotate on the right trigger**: per session, per N requests, or on
  detected block; honor `Retry-After` on 429 (back off, don't hammer).
- **Ban detection**: classify every response DATA | BLOCK | CHALLENGE |
  THROTTLE; a BLOCK retires the burned identity and teaches the pool.
- **proxy-pool (kariemSeiam)**: REST tiers gold/silver/new_untested/dead,
  `/best` `/random(min_score)` `/stats`, tiered revalidation scheduler.
- **slopsearx**: escalating cooloff (base 120s, ×3 after 3 consecutive
  failures), `report_success`/`report_failure`, fail-open when all are
  cooling.

**SHOULD HAVE:** `ProxyIdentity` (proxy + cookies + UA + fingerprint,
sticky session binding), response classifier, per-subnet/ASN stats,
Retry-After backoff, block-triggered retirement, gold/silver/dead tiers
in pool stats. Our sticky strategy exists but without the identity
binding it's half the pattern.

**LOOK/FEEL:** pool stats render as a tier table (gold/silver/cooling/
dead) with per-subnet burn warnings.

## 8. OSINT (`osint.py`, `osint_people.py`)

**Now:** domain/ip/url/email sweeps over keyless sources (DoH DNS,
urlscan, urlhaus, threatfox, hackertarget, ipapi), per-source isolation
+ timings, threat bundle, username checks, breach check, ASCII graph.

**Best outside (theHarvester — the Kali-standard):**
- **Passive vs Active modules**: passive (default) never touches the
  target; active adds DNS resolution, brute-force, port scans.
- ETL shape: Information Retriever → Formatter Module (raw → cleaned →
  structured intelligence).
- Source breadth for subdomains: **crt.sh + certspotter (certificate
  transparency — free and huge)**, rapiddns, dnsdumpster, bufferoverrun,
  anubis, github-code, brave search. Our sweep has NO CT-log source —
  the single biggest free subdomain feed.
- Patterns: `-v` verify hostnames via DNS, screenshot web services,
  export HTML/XML/JSON for the next pipeline stage.

**SHOULD HAVE:** Add crt.sh (CT logs), certspotter, rapiddns to the
domain sweep; a `mode="passive"|"active"` flag (active = DNS-verify
discovered hosts, resolve IPs); JSON export of the merged report;
source quality notes (which sources are stale/blocked).

**LOOK/FEEL:** sweep report leads with the asset map (subdomains/IPs/
emails), then per-source sections with timing; ASCII graph already good.

## 9. Sandbox code (`sandbox_code.py`, `shell.py`)

**Now:** `CodeInterpreter` sessions, `run_sandboxed` with rlimits
(SandboxLimits), backend detection, kill-tree.

**Best outside (E2B patterns):** create → use → destroy with try/finally;
per-command timeouts; **keep-alive with `set_timeout` heartbeat**
between steps (never let a policy kill a build mid-flight); file
upload/download round-trips; handle partial execution (files generated
even when code errors); resource metrics (`get_metrics`) with
appropriate timeouts per task class (short 60–120s, analysis 300–600s).

**SHOULD HAVE:** heartbeat/keep-alive for long sessions, file
round-trip helpers (write inputs in, read artifacts out), partial-
execution artifact collection on error, resource metrics, per-task
timeout profiles. Our rlimit sandbox is the local equivalent of the
isolation half; the lifecycle half is missing.

**LOOK/FEEL:** execution result shows stdout/stderr separately,
artifacts produced, resource usage line.

## 10. Hashcrack (`hashcrack.py`)

**Now:** pure-Python MD4, auto-detect, wordlist mutation, MarkovChain,
Engine, benchmark, native C fast path for NTLM (verified against
oracle).

**Best outside (hashcat professional workflow):** strategy, not speed —
wordlist + rules is the best balance (best64.rule hugely improves hit
rate); **mask attacks encode human patterns** (`?u?l?l?l?l?d?s` —
uppercase-first is THE common shape; word+year via hybrid `-a 6`);
combinator (`-a 1`) and hybrid (`-a 7`) attacks; potfile `--show` for
already-cracked; per-mode benchmark; session restore. Attack order:
straight → rules → combinator → hybrid → mask.

**SHOULD HAVE:** An attack **strategy ladder** (auto-escalating
dictionary → rules → hybrid → mask) instead of one-shot attempts;
built-in rule mutations mirroring best64 (we have `mutate()` — extend
toward rule-file semantics); hybrid word+mask generation; potfile
(already-cracked cache); benchmark-driven time estimates per mode.

**LOOK/FEEL:** crack progress shows current attack stage in the ladder
("stage 2/5: rules(best64-like)…"), ETA from benchmark.

## 11. Macros (`macros.py`)

**Now:** explicit record (start/step/stop), auto-capture of tool calls,
durable store, saved macro becomes a live `macro_<name>` tool, run
history.

**Best outside:**
- **Playwright codegen**: records real interactions live; adds CHECKS —
  records not just what you did but what SHOULD be true (assertions).
- **demo-maker**: live record → YAML; **continue-from** (replay a prefix,
  then record more); `--payload` per-run overrides; import from Chrome
  DevTools Recorder JSON.
- **playwriter**: JSON event log + screencast frames, auto-stop after 20
  min, end goal is a replayable skill; dead ends filtered later.

**SHOULD HAVE:** `record_checkpoint(description)` (assertions on
replay), per-run payload overrides (`run_macro(overrides={...})`
already takes overrides str — make it structured), YAML/JSON
export-import, continue-from-macro, auto-stop on idle timeout, step
timing capture (replay pacing).

**LOOK/FEEL:** `show_macro` renders steps as a numbered recipe with
checkpoints marked; runs show a per-step ✓/✗ trace.

## 12. DeliverReport (`deliver_report.py`), formatters

**Now:** markdown → HTML/text report composition, destination
validation, size caps, delivery with failure messages.

**Best outside (rich + mcp-aggregator REST):** dual text/structured
output (`structuredContent` + human text block); `Console(record=True)`
+ `export_text()` for report capture; Tables/Panels/Progress;
graceful degradation when piped (no ANSI); CHARACTER_LIMIT truncation
with notice.

**SHOULD HAVE:** a shared stdlib-only style layer (no new hard deps)
with optional rich integration: status lines, tables, panels, progress
bars, ANSI-stripping when not a TTY, `NO_COLOR` respect. Wire it into
`format_report` (error_scan), `format_test_result` (pytest_runner),
`_summary` (giftcard), `format_chat_summary` (vision).

**LOOK/FEEL:** god-tier, not functional — themed headers (Devon ninja
electric-blue default, plain fallback), aligned tables, ✓/✗/⚠ status
glyphs, compact KV blocks.

## 13. CodeExecutor-adjacent: `agents.py`, `parsers.py`, `metadata.py`, `imagedb.py`

- **agents.py** (`RoleScopedRegistry`, `CodingRoleAgent`): best outside =
  LangGraph supervisor pattern (one supervisor routes to N specialists;
  workers report back). Gap: role scoping is static; add dynamic
  delegation with result synthesis. (Light touch this sweep.)
- **parsers.py** (pdf/docx/xlsx/odt/epub/csv): best outside = trafilatura
  JSON + `markitdown` (Microsoft's any-to-markdown). Gap: no markdown
  output for LLM ingestion — add `to_markdown` for office docs.
- **metadata.py / imagedb.py**: solid (EXIF, GPS, dhash, colors). Gap:
  perceptual duplicate search across a directory (dhash + hamming
  already there — add `find_duplicates(dir)`), reverse-image note.

## 14. Native loader (`native/loader.py`)

Verified-against-oracle C fast path with honest benchmark notes
(OpenSSL 3 removed MD4 → pure Python 24× slower; md5/sha are already C
so the ctypes trampoline loses). This is already the right design
(verify, fail open, measure). No change — the MCP-gateway mining
belongs to a future `mcp_bridge`, not this loader.

## 15. Thin bridges (errorsys, connectors, research, memory, workspace, autonomy, seer, gameplayer, directed, sceneintel, wisdom, social, whatsapp, characters, music, games, studio, accounts, security, deals, giftcard, commerce, services, trading, vision, audio, media*, traindata, side_chats, filesend, weather, finance, scriptgen, archive, compress, database, decoder, cipher, git, lint, network, ssh_socks, pytest_runner, build_app, attacker, browser, security)

These either bridge to owner modules (don't duplicate that logic here)
or are single-purpose and already complete for their contract. The
cross-cutting upgrades (annotations, teachable errors, idempotency,
truncation notices, style layer) apply to them automatically through
the registry and the shared formatter.

---

## Implementation plan (this sweep)

1. `registry.py` — ToolSpec: `title`, `annotations{read_only,
   destructive, idempotent, open_world}`, `output_schema`, `examples`;
   `schema()` tightened (`additionalProperties: false`, required list);
   `call(..., idempotency_key=...)` with result cache;
   `max_result_chars` truncation with explicit notice; teachable error
   wrapping.
2. `_style.py` (new, stdlib-only, optional rich) — theme, tables,
   panels, status lines, progress bars, TTY/NO_COLOR handling.
3. `captcha.py` — `SolveBudget` ($ caps, auto-reset, ledger,
   `budget_status()`), `report_solve()` quality feedback,
   createTask/getTaskResult JSON API, `AutoPipeline` backend
   (detect → hints → budget-gated service).
4. `osint.py` — crt.sh + certspotter + rapiddns sources; `mode`
   passive/active on sweep; JSON export.
5. `proxylab.py` — `ProxyIdentity` sticky sessions, response
   classifier (DATA|BLOCK|CHALLENGE|THROTTLE), per-subnet/ASN stats,
   Retry-After backoff, gold/silver/dead tiers, block retirement.
6. `edit_loop.py` — flexible matching, `parse_search_replace_blocks`,
   `apply_search_replace` tool, empty-SEARCH creates file,
   placeholder/laziness detection, teachable match failures.
7. `code_executor.py` — `StepResult`, `ReplanDecision`, `replan()`,
   loop detection, `expected_output`, `synthesize()`.
8. `web.py` — trafilatura-first `readability_extract` with fallback,
   metadata, `favor_recall`, markdown output.
9. `repo_index.py` — `render_repo_map()` (reference-ranked,
   personalized, token-budgeted, signature-only).
10. `macros.py` — `record_checkpoint`, structured payload overrides,
    YAML/JSON export-import, continue-from, idle auto-stop.
11. Wire style layer into `error_scan.format_report`,
    `pytest_runner.format_test_result`, `giftcard._summary`,
    `vision.format_chat_summary`.
12. Tests in `tests/test_tools_sweep.py`; commit `sweep(tools): …`;
    push `two main:main`.
