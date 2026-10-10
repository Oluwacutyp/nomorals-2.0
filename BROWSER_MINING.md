# Browser Automation Mining Report

Written before build commits. The user's insight: "not everything needs to be api — she's got a browser just like yours."

## 1. What Devon already has (surveyed, real)

**`nomorals/browser/service.py` (3020 lines) — the real rendered browser:**
- `BrowserService` with Playwright-backed tabs: `navigate`, `click`, `fill`, `select`, `check`, `submit`, `upload`, `wait_for` (selector/url/text), `wait_for_load_state`, `wait_for_network_idle`, `screenshot`, `evaluate` (arbitrary JS), `cookies` via `storage_state` (persisted sessions), proxy support, pacing between actions, tab history, error tolerance.
- `_require_playwright_sync()` — honest `BrowserError` with exact install instructions when Playwright/Chromium is missing. Never imports at module load.
- Used by: trial flow (`agents/trial/flow.py`), connectors (jiji, konga), CLI (`cmdline/commands/browse.py`, `account.py`), the browser daemon (`browser/daemon.py` — socket IPC service).
- `browser/daemon.py` (919 lines): out-of-process browser service over a Unix socket — `daemon.sock/pid/lock/log` under `~/.nomorals/browser-service`.
- `browser/forms.py` (322 lines): form analysis/filling helpers. `browser/pacing.py`: human-like cadence. `browser/liveview.py`: action hooks. `browser/errors.py` (560 lines): error taxonomy.

**`nomorals/tools/browser.py` (1529 lines) — the spine tool, but HTTP-only:**
- Registered as `browser` in the tool registry (module list line 270). Capability `NET_BROWSER`.
- Actions: open/text/markdown/links/click/fill/submit/extract/walk/task/state/close/check_captcha.
- Built on **stdlib urllib + a hand-rolled HTML DOM parser**. No JavaScript. The docstring *mentions* Playwright (`NM_BROWSER_PLAYWRIGHT=1`) but `tools/browser.py` never imports it — the rendered path is aspirational text, not code.
- `task` action: multi-step programs (open/fill/submit/extract/click/back/wait/stop) — the right shape, wrong engine for JS sites.

**The gap:** the spine `browser` tool — the one the brain actually calls — cannot render JavaScript, cannot screenshot, cannot wait for dynamic content, cannot do what a real browser does. The real browser (`BrowserService`) is reachable from connectors, CLI, and trial flows, but **not from the spine**. The brain's browser is a scraper wearing a browser's name.

## 2. How the best browser agents work (web-mined, 2026)

**The loop shape (convergent across browser-use, WebVoyager, Playwright MCP, Skyvern, Magnitude):**
`observe → decide → act → verify`, with an independent verifier stage (Skyvern 2.0's Planner→Actor→Validator).

**Three grounding strategies (all production-proven):**
1. **Set-of-marks** — numbered bounding boxes baked into a screenshot + parallel text list (`[127]<h3>iPhone</h3>`). WebVoyager paper: 59.1% success vs 40.1% for a11y-tree-text-only. Best for vision-capable brains.
2. **Accessibility tree with stable refs** — Playwright MCP's `browser_snapshot` returns YAML a11y tree; agent calls `browser_click({ref: "e5"})`. Microsoft's production pattern (Playwright MCP, Mar 2025; first-class by Playwright 1.59+). Best for text-only brains, cheapest tokens.
3. **Pure vision** — screenshots only, VLM predicts coordinates. Works on canvas/shadow-DOM where DOM fails; most expensive.

**Action space (convergent):** navigate, click, type/fill, scroll, hover, press key, select, check, upload, wait (for selector/text/url/network-idle), screenshot, evaluate JS, extract text/structured, back/forward, cookies.

**Key engineering lessons:**
- **Page-stability waiting** before observation (DOM quietness + network idle) — acting on a half-loaded page is the #1 flake source.
- **Verify after act** — re-observe and confirm the UI reacted; on failure, hand context back to the planner to re-plan (not blind retry).
- **Session persistence** — cookies/storage_state across restarts; one identity per session.
- **Pacing** — human-like cadence between actions (anti-bot-detection, not just politeness).
- **Never auto-retry mutating steps** — idempotent fetches retry with backoff; form submits don't (Devon already does this right).

## 3. What to build (gap → fix)

1. **Rendered path on the spine `browser` tool** — add `engine="rendered"` (auto when Playwright present, honest error when not). Same action vocabulary the brain already knows, plus `screenshot`, `wait`, `scroll`, `hover`, `press`, `observe` (interactive-element listing = grounding). Merge, don't duplicate: extend `tools/browser.py`, don't create a parallel tool.
2. **OSINT browser adapters** — `search/osint_browser.py`: `SourceAdapter`s that drive the rendered browser for API-less sites (form-fill people search, result extraction). The paywalled aggregators the last task skipped get browser flows instead of being skipped.
3. **Research organ** — it already calls spine tools; verify the browser tool reaches it and document the pattern for API-less sources.
4. **Trial flows** — survey `agents/trial/flow.py` signup scenarios; extend where the rendered browser adds coverage.

## 4. Design decisions

- **A11y-tree grounding first** (cheapest, text-brain-compatible), screenshot on demand. Devon's brain is text-first; set-of-marks needs a vision model she doesn't always have.
- **In-process `BrowserService`**, not the daemon socket — the spine tool runs in-process; the daemon is for CLI/external use.
- **Profile-gated**: Playwright/Chromium is heavy (~170MB download, RAM-hungry). Termux/phone profile keeps the HTTP engine; workstation/laptop get rendered. The tool reports which engine served each call.
- **No stubs**: every action either works through the real tab or raises the honest "install playwright" error. No fake screenshots, no simulated clicks.
