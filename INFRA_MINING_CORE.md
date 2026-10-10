# INFRA MINING — Core Runtime (Section A)

Mined 2026-10-09 before touching code. Sources: production agent/sandbox builds
(agenticos SECURITY-SANDBOX, opencapx plugin protocol, proengineerdev wrongstack
H1 audit pattern, openclaw-android registry), the Google/Microsoft/OWASP agent
security writeups, agentsfleet's async-signal-safe shutdown design,
go-complete-notes' shutdown lifecycle, Spring Boot's SmartLifecycle ordering,
the pattern-agentic pydantic-settings hot-reload build, and the where-is-my-shit
config-consolidation research. Trash builds mined too (oppenheimer rbac-rules:
"nobody grants what they do not hold").

## Gold taken

### Config (hot reload + hygiene)
1. **watchfiles (Rust) with mtime-polling fallback** — uvicorn does exactly
   this: efficient event watching when available, poll fallback otherwise.
   Taken: optional `watchfiles` import, degrade to polling thread.
2. **Atomic writes (temp + os.replace)** — a config rewrite mid-read must
   never produce a half-parsed TOML. Watcher must tolerate it, and our own
   config writer must do it.
3. **Validate-before-swap** — reload builds a brand-new Settings, runs the
   full validator, and only then swaps the process-wide reference. A bad
   edit never poisons the running system; the old config stays live and the
   failure is logged with the reason.
4. **Redacted startup dump** — pattern-agentic prints all settings on startup
   with secrets masked. We already redact in `_asdict`; the watcher reports
   *which dotted paths changed* (a diff), not values.
5. **Debounce** — file saves come in bursts; coalesce to one reload.
6. **Precedence honesty** — the where-is-my-shit research caught a classic:
   documented precedence ≠ implemented precedence. Our docstring says
   profile(2) < TOML(3), but the code applies the profile preset AFTER TOML,
   so the profile silently wins. Fix the code to match the docs.

### Policy (expressiveness)
1. **agenticos evaluation sequence**: capability → parent policy → scope
   attenuation → approval requirement → resource binding → time/usage limits
   → runtime-issued grant → consume/revoke. We have most of it; we are
   missing **scope attenuation** (resource binding), **time-bounded grants**,
   and **revocation**.
2. **Invariants from the same doc**: capabilities must be *attenuable,
   time-bounded and revocable*. A stale/expired/consumed capability cannot be
   reused — our confirmations are single-use (good), but grants are eternal
   (bad). → timed grants with expiry + explicit revoke.
3. **"Nobody grants what they do not hold"** (oppenheimer rbac-rules):
   sub-agent grant derivation must be `role_preset ∩ parent_grant`, enforced
   as a helper, not left to call sites to remember.
4. **Conditional rules**: context is currently recorded into the audit log but
   never *evaluated*. Real expressiveness = `allow(cap, when=lambda ctx: …)`
   so e.g. `fs.write` can be scoped to the workspace directory per call.
5. **Google/Microsoft pattern**: approval bound to a *specific request*
   (we have that — capability-bound tokens), JIT credentials with TTL
   (minutes), fail-closed on engine errors (we have that).
6. **Proposals must survive restart** — an autonomous proposal pending owner
   approval dies with the process today. Pluggable `PolicyStore` (memory
   default; DB-backed when wired) fixes it without dragging storage into L1.

### Shutdown (ordering + safety)
1. **"No new work, then finish work, then clean up"** — ordering is a
   dependency graph, not a wish list (go-complete-notes). Our SIGTERM→
   KeyboardInterrupt mapping has no ordering at all.
2. **Async-signal-safe boundary** (agentsfleet): the signal handler does ONE
   thing — set an atomic flag. All real work happens on the main thread.
   Python signal handlers already run on the main thread, but raising
   KeyboardInterrupt *inside* a handler mid-critical-section is the same
   class of bug — we keep the mapping for compat, but the new coordinator
   runs ordered hooks instead of relying on exception unwinding.
3. **Spring SmartLifecycle ordering**: startup in ascending phase, shutdown
   in DESCENDING phase. Taken: hooks carry a priority; shutdown runs them
   in reverse.
4. **Every shutdown step gets a timeout** — force-close stragglers, never
   block a deploy forever (also from agentsfleet's join_deadline).
5. **Liveness vs readiness split** (py-ops skill): readiness fails *during
   shutdown* so load balancers stop sending traffic; liveness stays up until
   the process actually dies.

### Registry (health + introspection)
1. **Every tool must implement teardown() + health()** — the H1 audit
   pattern from wrongstack. For a tool *registry* this means: per-tool
   health probes, registered at registration time or after.
2. **opencapx**: `plugin.ping` with 5s timeout; 3 consecutive failures →
   `error` state. Taken verbatim: consecutive-failure counter, ok/degraded/
   down states.
3. **Registry record = identity + source + origin + status + diagnostics**
   (openclaw). Taken: `describe(name)` returns module/file, capability,
   confirm level, health, per-tool stats.
4. **setup() is idempotent** — re-registering a tool module must not
   accumulate state. Taken: `reload_module()` that re-imports and re-runs
   the register hook cleanly.
5. **Per-tool stats**, not just global stats — calls/errors/denied/seconds
   per tool, plus last-error capture. This is what lets routing demote what
   keeps breaking.

## Weaknesses found in audit (cured in the build, not listed after)

- **config.py**: no hot-reload; no validated `set(dotted, value)`; no diff
  between two Settings; precedence bug (profile beats TOML contrary to
  docstring); validation is a small hand list with no cross-field checks;
  no schema export for docs/`nm config`.
- **policy.py**: context recorded but never evaluated (no conditional
  rules); grants are eternal (no timed grants / revoke); no enforced
  child-grant narrowing helper; proposals die on restart (no store).
- **profile.py**: no cloud detection (AWS/GCP/Azure) — detection accuracy
  gap; `EnvironmentProfile` carries no cloud field.
- **profiles.py**: `get_profile_kind()` ignores the configured profile pin
  (reads only NM_PROFILE env + detection); runtune has the right
  resolution order, profiles.py doesn't.
- **shutdown.py**: 63 lines, one trick (SIGTERM→KeyboardInterrupt), no
  ordering, no hooks, no timeouts, no readiness signal.
- **runtune.py**: BUG — `RuntimeSettings.threads` defaults to 4, which
  `build_tune` treats as an *explicit override*, so auto-tuning of threads
  NEVER runs. Default must mean "unset" (0).
- **registry.py**: no health checks, no per-tool stats, no
  describe/introspection, no aliases/deprecation, no module reload.
