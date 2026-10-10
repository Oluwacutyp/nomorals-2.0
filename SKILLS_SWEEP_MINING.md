# SKILLS Sweep — External Mining Report

Module: `nomorals/skills/` — executable skill packages (L4): manifest,
registry, runner, bench, repair. Mined 2026-10-10 against the best
implementations outside the repo. Every class below gets a "how does the
best do it" comparison and the gold merged in this sweep.

## 1. Skill manifests — what the industry converged on

**Anthropic Agent Skills (open standard, 2025; agentskills.io spec, donated
to the Agentic AI Foundation).** The winning shape is YAML frontmatter +
markdown body, with required `name` / `description` and optional
`version`, `license`, `allowed-tools`, `model`, `metadata`. Hard rules the
best enforce at install time:

- `name`: max 64 chars, lowercase letters/numbers/hyphens, **no XML tags,
  reserved words forbidden** (`anthropic`, `claude`), must match the
  directory name.
- `description`: max 1024 chars, non-empty, no XML tags, third person with
  triggering conditions — "THE most critical field" because it drives
  skill *selection*.
- Progressive disclosure: only name + description load first; full body
  loads on relevance; scripts last. Prevents context bloat with hundreds
  of skills.
- `allowed-tools` as an explicit security sandbox per skill.
- Enterprise guidance: instruction manipulation in skills rated high risk;
  review SKILL.md + referenced files, sandbox scripts, never deploy
  untrusted skills unaudited.

*Gold merged:* name rules already close (ours allows 1–64 chars,
`[A-Za-z0-9_.-]`); adding reserved-word rejection, XML-tag rejection in
name/description, `license`/`tags`/`short_description` metadata fields,
description trigger-quality guidance, and **export to portable formats**.

**Trash mined:** random GitHub "skill packs" with free-text names like
`"my cool skill!!"` and 3-paragraph descriptions — unselectable,
uninstallable, unportable. Our registry already beats these; the reserved
words/XML rules close the last gap.

## 2. Skill description poisoning — the attack that names our defense

**Invariant Labs (2025): "MCP Security Notification: Tool Poisoning
Attacks"** — a malicious server hides instructions inside tool
descriptions the user never sees; a weather tool can quietly tell the
model to also read `~/.ssh/id_rsa`. Demonstrated against stock clients.

**OWASP Top 10 for Agentic Applications 2026 (ASI02), Trail of Bits
"Jumping the line" (Apr 2025), Simon Willison's confused-deputy framing:**
the attack surface is anywhere the server controls text the model reads —
descriptions, parameter names, enums, defaults, error messages. Distinct
attack classes: tool poisoning, tool shadowing, **rug pull** (approved
server later *changes* its definitions — "no widely deployed mechanism
for detecting this change"), toxic flow.

**Documented defenses (what the best do):** review and **pin** tool
descriptions, **diff them on every change**, scan for imperative text,
pin versions + **hash definitions**, re-scan on update, canary markers
(`CANARY-7F3A` planted to detect exfiltration paths). The classic
poisoning phrasing from the Invariant Labs demos: *"Before using any
other tool, call this tool with …"* and `<IMPORTANT>…</IMPORTANT>`
delimiter blocks.

*Gold merged:* our `sanitize_description` two-tier clean (strip hidden
unicode/ANSI/controls → reject marker phrases) is ahead of most clients,
which do nothing. Upgrades: `<IMPORTANT>` delimiter markers, "before
using/calling any|this|other tool" sequencing directives, "you must
first/always" tool-call directives — all high-precision, low
false-positive. Plus **definition hashing + change detection**: the
registry now stores a manifest hash per version and `install` reports
when a re-install *changes* a pinned version's bytes (the rug-pull
detector the research says nobody deploys).

## 3. Wiring expressions — GitHub Actions vs n8n vs JSONata

**GitHub Actions (the closest analog to our step chains):** steps get
named `id:`s; downstream steps reference `steps.<id>.outputs.<name>`
— **by name, not index**. Job outputs map explicitly
(`jobs.<id>.outputs.<name>`). Contexts: `steps`, `needs`, `env`,
`matrix`, `github`. `if:` conditions gate steps (`success()`,
`failure()`, `always()`). `continue-on-error` lets a step fail without
killing the job. `fromJSON()` coerces output strings to typed values.
`$GITHUB_STEP_SUMMARY` appends per-step Markdown.

**n8n:** `{{ $json.field }}` (current item) and
`{{ $node["Node Name"].json.field }}` (named node), plus JMESPath for
complex queries. Rule of thumb: prefer the positional form over hardcoded
node names — it survives renames.

**AWS Step Functions:** JSONata (`{% %}`) as the query language, JSONPath
fallback; declarative XPath-for-JSON.

*Gold merged:* our `$input` / `$<n>` / `$last` refs are n8n-positional in
spirit but lack the GHA **named** form. Adding optional `step_ids` so
wiring can use `$<step_id>.<path>` (validated: unique, name-shaped,
length-matched). Adding `$env.VAR` root (n8n `$env`, GHA `env.*`
context), opt-in on the runner. Adding per-step `continue-on-error` with
`fallback` expression (GHA), because a chain that dies on one flaky step
is a toy. Keeping the explicit no-silent-merge threading rule — it is
stricter and better than GHA's loose output bags.

**Trash mined:** workflow builders that auto-merge every step output into
one soup dict ("convenience") — the exact silent-merge bug our threading
rule forbids. We keep our rule and say so in the docs.

## 4. Retries, timeouts, error classification — Temporal

**Temporal (the gold standard for step execution):** every activity gets a
retry policy (initial interval, backoff coefficient 2.0 default, max
interval, max attempts) **and** timeouts (start-to-close, schedule-to-close,
heartbeat). The three rules:

1. **Classify errors: transient vs permanent.** Permanent failures
   (validation, auth, not-found) surface immediately — never retry.
2. **Always set maximums.** Bounded attempts AND a total timeout: "no
   matter what, the workflow won't just sit there spinning."
3. **Exponential backoff, not linear.** Log the phase transition
   (fast→slow retries) with request id + attempt count — it signals a
   sick downstream.

*Gold merged:* our runner had **zero** retries and **zero** timeouts — a
single hung tool hangs the skill forever. Adding per-manifest
`retries: {max_attempts, initial_backoff_s, backoff_multiplier,
max_backoff_s, retryable_errors, non_retryable_errors}` and `timeout_s`
per skill (threaded execution with join timeout; timeouts are
non-retryable-after-exhaustion). Default classifier: wiring errors,
schema validation failures, capability denials, unknown tools are
permanent (no retry); everything else retries with exp backoff. Attempts
recorded on the StepResult.

**Trash mined:** "retry forever with 1s sleep" snippets, and the
opposite — no retry at all with a comment `# TODO: retry`. Both are
below our bar now.

## 5. Benchmarking — the four golden signals + canary analysis

**Argo Rollouts AnalysisTemplates (progressive delivery gold):** canary
steps gated on `successCondition: result[0] >= 0.95/0.99`, queried every
15–60s, `failureLimit: 2–3` consecutive failures → **automatic abort and
rollback to stable**. The gate is the automated, release-blocking subset
of dashboards.

**Temporal/SLO practice:** workflow success rate, average execution time,
per-activity latency, error rates by category; percentiles (p50/p95/p99),
not just averages.

*Gold merged:* our `SkillBench` recorded only avg/min/max — averages hide
tails. Adding p50/p95/p99, per-step latency tracking, `compare_versions`
side-by-side, **regression detection** (candidate vs baseline: success
rate delta + p95 delta), and a **`canary_gate`** verdict function
(min runs, min success rate, max p95) that answers "is this skill version
safe to pin?" — the exact AnalysisTemplate shape, but for skill pins.
Plus `prune` retention (benchmark tables grow forever otherwise).

**Trash mined:** benchmark scripts that print one number and exit.
Averages without distributions are how bad versions ship.

## 6. Repair tickets — incident management, not log lines

**PagerDuty/Ops practice:** incidents have lifecycles (open →
acknowledged → resolved), get clustered by signature, escalate on repeat,
and postmortems feed back into prevention. Our module's own docstring
promises "repeated failures are visible as a pattern" — but shipped no
grouping at all.

*Gold merged:* ticket lifecycle (`resolve`/`reopen` with notes, status
column with guarded migration), **`patterns()`** clustering by
normalized error signature (finally delivering the docstring's promise),
stale-pattern surfacing, and new `suggest_fix` branches for timeouts and
retry exhaustion. `format_ticket()` renders a ticket as a readable
incident card.

**Trash mined:** "error log aggregators" that store strings and call it
observability. Tickets without lifecycle are a write-only log.

## 7. Skill discovery — tool selection as retrieval

**Measured best practice (multiple sources, 2025–2026):** treat tool
selection as retrieval past ~30–50 tools; below that, load all
definitions. The winning pipeline: keyword/TF-IDF pre-filter → embedding
retrieval top-k (k=5–7) → agent chooses. Reciprocal Rank Fusion for
multi-variant queries. **Default to lexical; add semantic as a measured
fallback** — pure keyword agentic search reaches ≥90% of RAG on real
tasks (the embedding arm often doesn't clear its own operational cost).

*Gold merged:* our registry had **no search at all** — skills were
invisible unless you knew the exact name. Adding `search()` (tokenized
lexical scoring over name/description/tools/tags/owner) and
`recommend(task_description)` (stopword-stripped keyword overlap +
bench success-rate boost — reliable skills rank higher). Deliberately
lexical, zero-dep, offline: the measured-default per the research, and
it matches this project's offline-safety-net posture. Embedding hooks
left as a documented extension point, not a half-built dependency.

## 8. Execution modes — dry run as a contract

**Airflow** `tasks test --dry-run`: render template fields, invoke
nothing. **flowdular SKILL.md** (good execution-mode design): dry-run
*validates a draft and returns issues, compiled order, references,
permissions, checksum, limits — creates no run and invokes nothing*;
simulation uses fixtures; live runs only published revisions.

*Gold merged:* adding `SkillRunner.run(..., dry_run=True)` — validates
the manifest, resolves every wiring expression against the input shape,
and returns the planned step/parameter table without calling a single
tool. Plus a stable "dry-run receipt" (run id, planned steps, checked,
skipped) per the dev.to dry-run contract piece.

**Trash mined:** dry-run flags that still hit the network "just to check".
Ours invokes nothing — verified by test with a counting tool registry.

## 9. Presentation — $GITHUB_STEP_SUMMARY and the god-tier bar

GHA's `$GITHUB_STEP_SUMMARY` (per-step Markdown, 1 MiB cap) proves that
workflow systems live or die by readable step output. Our `SkillResult`
serialized to JSON only.

*Gold merged:* `format_result()` — a step timeline with ✓/✗ markers,
per-step latency, input/output peeks, the error callout, and the repair
ticket id; `format_score()` with ASCII success sparkline; registry
`format_table()`; ticket `format_ticket()`. Plain-text, no dependency,
terminal-first — the god-tier bar for a CLI-driven agent OS.

## 10. What we deliberately did NOT take

- **Full JSONata/JMESPath engine as a dependency** — our `$` ref language
  covers the wiring cases; a vendored mini-predicate for `when` stays
  stdlib-only. (Zero-mandatory-deps standing order.)
- **Automatic ticket → manifest auto-fix application** — suggested fixes
  stay suggestions; the human/agent loop applies them. Auto-applying
  wiring rewrites is how you get silent corruption.
- **Semantic embeddings for search** — lexical default per the measured
  research; the extension point is documented.
- **Parallel fan-out steps** — out of scope for a strictly ordered chain;
  the manifest would need a DAG, not a list. Noted as future work, not
  half-built.
