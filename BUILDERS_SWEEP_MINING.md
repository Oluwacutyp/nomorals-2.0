# BUILDERS SWEEP — External Mining Report

Module: `nomorals/builders/` (9 Python files + 6 template trees). Mined
2026-10-10. Written BEFORE any code changes. Every significant class was
compared against the best implementation found outside the repo — best AND
trash. Sources are summarized in my own words; no source text reproduced
beyond brief fair-use quotes.

## Method

For each class cluster: "How does the best implementation of X do it?",
then "What features SHOULD this have that it doesn't?" and "How should
this LOOK/FEEL?" All conclusions below became concrete build decisions.

---

## 1. `scaffold.py` — project templating

### Best: Cookiecutter (24k+ stars, the Python templating gold standard)

- **Typed prompts, not one variable**: `cookiecutter.json` declares
  string/number/bool/choice/dict variables with defaults, human-readable
  prompts, and input validation. Ours substitutes exactly one variable
  (`$PROJECT_NAME`) — no author, no description, no choices.
- **Hooks**: `pre_prompt` (env checks), `pre_gen_project` (input
  validation, runs inside the generated project), `post_gen_project`
  (git init, venv creation, pip install, initial commit). Hook failure
  stops generation and cleans up. Ours has zero hooks — a scaffolded
  project is never git-initialized, never committed.
- **Conditional files/dirs**: Jinja `{% if %}` in directory and file
  names lets one template adapt its shape to user choices. Ours is a
  static tree copy.
- **Remote + private templates**: `gh:`, `gl:`, `bb:` shorthands, zip
  URLs, private repos via `git+`. Ours only reads its own bundled dir.
- **Replay files** for reproducible generation.

### Best: Copier (the modern successor)

- **Template versioning and updates**: generated projects can be
  *updated* when the template changes (`copier update`). Ours is
  one-shot; template improvements never reach old projects.
- `_copy_without_render` (binary passthrough), `_templates_suffix`,
  composite templates, JSON-schema validation.

### Best: cargo-generate

- **Typed placeholders with prompts/defaults/choices** in
  `cargo-generate.toml`; conditional follow-up prompts (`prompt_if`).
- **Three hook phases** (init/pre/post) in Rhai scripts with a real
  variable API (file::exists, case filters snake/kebab).
- **`--test`**: runs the template's test suite against the *expanded*
  output — generation and verification are one command.

### Best: Backstage Scaffolder (enterprise self-service UX)

- Templates are **actions pipelines** (`fetch:template` →
  `publish:github` → `catalog:register`), each step named, shown in a
  progress UI, conditionally skipped (`if:`), with outputs referenced
  by later steps (`${{ steps.x.output.y }}`).
- **The catalog**: every template has metadata (title, description,
  owner, tags) browsable in a UI. Ours has `KINDS` — a bare tuple.
- `fetch:template` renders Nunjucks into **file and directory names**,
  not just file contents.

### Trash mined

Yeoman forks that are a single `fs.copy` with no prompts, no hooks, no
tests; "scaffolder" tutorials that `git clone` a repo and `sed` one
string. Also: templating engines that mutate the *source* template dir
in place (ours renders into a fresh copy — keep that).

### → Scaffold decisions

1. **Template catalog**: `list_templates()` / `describe(kind)` —
   title, description, file list, variables, test command (Backstage
   catalog, cheap).
2. **Variables**: `scaffold(..., variables={...})` merged over builtins
   (`PROJECT_NAME`, `PROJECT_SLUG`, `AUTHOR`, `YEAR`, `DATE`,
   `DESCRIPTION`) — Cookiecutter's multi-variable prompts, stdlib-only.
3. **`dry_run=True`**: render to a temp dir and report the file list
   without touching the destination (degit/cookiecutter rehearsal).
4. **`overwrite` flag** instead of only "non-empty dest = error".
5. **Post-scaffold hooks**: `git_init=True` — `git init`, initial
   commit (post_gen_project gold); failure recorded in the result,
   never fatal.

---

## 2. `app_builder.py` — multi-stack app generation + lifecycle

### Best: the AI app builders (v0 / Bolt / Lovable) — UX gold

- **Iterate, don't regenerate**: change requests patch the existing
  app (diff → apply → re-verify). Ours is build-once; fixing a typo
  means rebuilding from scratch.
- **Preview URL + file tree + one-click deploy** as first-class
  outputs. Ours has `serve`/`deploy` but no file listing API and no
  "what changed" story.
- **Duplicate/remix**: clone an app as a starting point.

### Best: cookiecutter-django — what a *production* template looks like

- Docker + docker-compose, CI workflow, pre-commit, pytest config,
  `.env` handling, Postgres/Redis wiring, docs. Ours generates a raw
  app with a README and nothing else — no Dockerfile, no CI, no
  `.dockerignore`. Every server stack we ship should be
  **containerizable and CI-ready in one command**.

### Best: PM2 — process management done right

- `restart` with **policies** (always / on-failure), `--watch`
  (restart on file change), `--max-memory-restart`, ecosystem file,
  `pm2 logs`, `pm2 save` (snapshot), startup scripts.
- Ours: `serve()` starts a detached process with a pid in kv and
  `stop()` kills it. No restart, no logs API (only a log *path*), no
  watch, no memory guard, no "what's the exit story".

### Trash mined

"App builder" repos that are a README of links; template repos whose
"validation" is a badge SVG; generators that shell out to `npm create`
and call the exit code verification.

### → AppBuilder decisions

1. **`dockerize(name)`**: per-stack Dockerfile + `.dockerignore`
   (production templates, cookiecutter-django gold).
2. **`ci(name)`**: per-stack GitHub Actions workflow (test/build).
3. **`patch(name, files)`**: overwrite/add files, re-validate, update
   manifest (Bolt-style iteration primitive).
4. **`duplicate(name, new_name)`** and **`remove(name)`**.
5. **`theme` spec** (`dark`/`light`/`neon`): the generated CSS is one
   hardcoded dark theme; themes make output feel designed, not
   default.
6. **`/readyz` readiness endpoints** next to `/health` (k8s probe
   semantics: liveness ≠ readiness).

---

## 3. `run.py` — run configs + process management

### Best: Kubernetes probes (the semantics gold standard)

- **Three probes, three jobs**: startup ("done initializing?", gates
  the others), liveness ("dead/wedged?" → restart), readiness
  ("send traffic now?" → route/don't route, no kill).
- Fields: `initialDelaySeconds`, `periodSeconds`, `timeoutSeconds`,
  `failureThreshold`, `successThreshold`. Defaults that encode real
  ops wisdom: period 10s, timeout 1s, failureThreshold 3.
- Ours: `serve()` waits for *TCP accept* only — a process can accept
  TCP while returning 500s on every route. No HTTP probe, no path
  configurability, no failure threshold.

### Best: PM2 (again, for the supervisor loop)

- Auto-restart on crash with backoff, watch mode, log capture,
  `restart` command. Our `ServeHandle` has `stop()` but no
  `restart()`, no restart policy, no uptime, no log accessor.

### Best: honcho/foreman (Procfile runners)

- One command runs the whole formation; `.env` loading; graceful
  shutdown with SIGTERM → SIGKILL escalation (which ours already does
  — keep).

### Trash mined

`subprocess.Popen` wrappers that never drain stderr (deadlock on a
chatty child — ours has `_Drainer`, keep), "process managers" that
`shell=True` everything.

### → Run decisions

1. **HTTP startup probe in `serve()`**: after TCP is ready, poll the
   configured `health_path` with `initial_delay` / `period` /
   `failure_threshold` semantics; raise `ServeError` carrying the
   probe transcript on failure (never a bare timeout).
2. **`HttpProbe` dataclass** (path, expected statuses, body match,
   timeout) + `TcpProbe`; `RunConfig.health_path` auto-detected.
3. **`ServeHandle.restart()`**, **`.logs(n)`**, **`.uptime`**,
   **`.restarts`**; `serve(..., restart_policy="on-failure"|"always"|"no",
   max_restarts=3, watch=False)` — watch polls mtimes and restarts on
   change (PM2 `--watch`, dev-mode gold).
4. **`stop_all()`** over a weakref handle registry.

---

## 4. `smoke.py` — smoke tests

### Best: Docker HEALTHCHECK + k8s probes

- `HEALTHCHECK CMD-SHELL "curl -f http://localhost/health || exit 1"`
  with `--interval --timeout --retries --start-period`: retries and a
  start period are what separate a smoke test from a flaky test.
- k8s `httpGet` accepts **2xx–3xx** as healthy (ours demands exactly
  200 — a 301 from a trailing-slash redirect would fail ours).
- Content assertions (`grep` on the body) are the norm in real
  health checks; ours only checks the status code.

### Trash mined

"Smoke tests" that are `curl -s -o /dev/null -w "%{http_code}"`
piped to `echo OK` unconditionally; test suites that assert on their
own hardcoded strings.

### → Smoke decisions

1. **`HttpExpectation(path, statuses=(200..399), body_contains=...)`**:
   multiple named expectations per target (k8s/Docker semantics).
2. **TCP probe** option (`tcp_socket` style).
3. Keep the CLI `--help` path; add `env` passthrough.

---

## 5. `install.py` — dependency installation

### Best: uv (the current Python packaging gold standard)

- `uv.lock` with per-package **hashes**; `uv sync --locked` errors on
  a stale lock; `uv pip compile --generate-hashes` produces a
  hash-pinned requirements file; `uv export --format
  requirements.txt|cyclonedx` for SBOMs.
- Ours: bare `pip install -r requirements.txt` — no hashes, no lock,
  no reproducibility, installs into whatever interpreter happens to
  run the agent.

### Best: nox (local pipeline runner)

- Sessions with **isolated venvs**, venv reuse, parameterized
  matrices, `--parallel`. Our install has no venv isolation at all
  (PEP 668 systems will refuse it).

### Trash mined

`curl | sh` installers; `pip install` with `--break-system-packages`
in Dockerfiles; "dependency managers" that are `os.system("pip
install " + user_input)`.

### → Install decisions

1. **`installer="auto"`**: use `uv` when present, else pip (uv is
   10–100x faster and hash-aware; pip stays the fallback).
2. **`venv=` param**: create `.venv` and install into it (PEP 668
   compliant, nox-style isolation).
3. **`generate_lock()`**: hash-pinned lockfile via `uv pip compile
   --generate-hashes` (uv) or a pinned `pip freeze`-style lock after
   resolve; **`verify_lock()`** freshness check.
4. **`dry_run`**: `pip install --dry-run` preview.
5. Result records `installer_used`, `venv`, `lockfile`.

---

## 6. `export.py` — packaging + tamper-evident archives

### Best: reproducible-builds.org discipline

- `tar --sort=name --mtime=@$SOURCE_DATE_EPOCH --owner=0 --group=0
  --numeric-owner`, `gzip -n`, normalized modes — **byte-identical
  rebuilds** verified by building twice and comparing hashes.
- Ours: `tarfile` with wall-clock mtimes, real uid/gid, gzip header
  mtime — the same project exported twice gives different bytes, so
  hash comparison is meaningless.

### Best: `git archive` / Python wheels

- Deterministic member metadata; wheels normalize permissions.
- SBOM-style manifests (CycloneDX) alongside the file manifest.

### Trash mined

Zips built with `shutil.make_archive` and no manifest at all; "backup"
tools that include `.env` and `__pycache__` by default (ours excludes
junk — keep).

### → Export decisions

1. **`reproducible=True`**: fixed mtime from `SOURCE_DATE_EPOCH`,
   uid/gid 0, normalized modes, gzip mtime 0, `sort_keys` manifest —
   plus **`verify_reproducible()`** (export twice, compare sha256).
2. `ExportResult.sha256`, `.file_count`, `.reproducible`; manifest
   gains `toolchain` (python version, platform).
3. Keep the sha256-per-file MANIFEST.json (already tamper-evident —
   good bones).

---

## 7. `deliver.py` — zip + chat delivery

### Best: artifact delivery (GitHub Actions artifacts, Telegram Bot API)

- `sendDocument` with chunked upload, captions, progress; Actions
  artifacts with retention and integrity.
- Retry with backoff on transient send failures is universal.

### Trash mined

"Send file" helpers that `open().read()` the whole file into RAM;
deliveries that re-zip on every retry (ours already zips once —
keep).

### → Deliver decisions

1. **`split_mb` in `zip_project`**: multi-part archives (`.partNN`)
   for chat size limits — a real gap (Telegram caps at 50MB).
2. **Retry with backoff** in `_send_zip` (`retries`, `attempts`
   recorded in the result).
3. **Caption templating** (`{name}`, `{kind}`, `{bytes}`).

---

## 8. `verify.py` — the lifecycle pipeline

### Best: GitHub Actions (the pipeline UX gold standard)

- Named steps with timing, per-step logs, artifacts, `if:`
  conditionals, `continue-on-error`, summary pages. Our
  `BuildReport.summary()` is plain text with `[ok]`/`[FAIL]` markers —
  functional, not god-tier.
- **Step selection** (`jobs.<id>.if`, workflow_dispatch inputs):
  ours runs all five steps always.

### Best: nox sessions

- Named, isolated, parameterizable steps with venv reuse. Our steps
  share the agent's interpreter.

### Trash mined

"CI" scripts that are one `&&`-chained shell line with `set -e` and a
single echo at the end.

### → Verify decisions

1. **Selectable steps**: `build_and_verify(..., steps=[...],
   skip=[...], fail_fast=False)` (Actions-style).
2. **New `validate` step**: `py_compile` every `.py` in the project
   (AppBuilder's validator, applied to scaffolded projects).
3. **`BuildReport.save()` / `.load()`**, **`.as_markdown()`**, and a
   **styled `fancy_summary()`** (see style.py).
4. `BuildStep.attempts` for retried steps.

---

## 9. Presentation — the god-tier bar

### Best: `rich` (terminal rendering gold)

- `Status` spinners, `Progress` bars, `Table`, `Panel`, `Tree`,
  `Syntax` — and graceful degradation when not a TTY. `rich` is not
  installed here, so the lesson is the *pattern*, reimplemented
  stdlib-only: ANSI themes, box drawing, spinner frames, and a
  `NO_COLOR`/`plain` fallback.

### Best: Backstage scaffolder UI

- Every step named, timed, and shown with live status; outputs linked
  from the summary. Our reports should read like that in a terminal.

### → Style decisions

New **`style.py`**: stdlib-only ANSI themes (`neon`, `minimal`,
`plain`), `banner()`, `render_steps()` (named steps + timing +
status glyphs), `render_kv()`, `spinner` frames. Wired into
`BuildReport.fancy_summary()` and `DeliveryReport.fancy_summary()`.
No new dependency — `rich` stays optional, never required.

---

## What stays untouched (already gold)

- `_Drainer` stderr draining (deadlock-proof child handling).
- Policy-gated `install_deps` (confirmable capability, explicit
  denial — the security posture is right).
- `build_and_verify` never raising on step failure (report, don't
  crash).
- Exclusion lists in export/zip (no `.env`/`__pycache__` leaks).
- `deploy()`'s real reverse proxy with TLS (out of scope for this
  sweep's changes, left as-is).
