# Browser Module Sweep — External Mining Report

Sweep of `nomorals/browser/` (7 files, ~5.3k lines): daemon.py, errors.py,
forms.py, liveview.py, pacing.py, service.py. Mined 2026-10-10 via targeted
web research across the best (and most instructive) outside implementations.

## Method

For each significant class I asked: "How does the best implementation of X
do it?" — and also mined weaker builds, because even trash holds something.
Sources: browser-use, Stagehand (Browserbase), Playwright MCP, puppeteer-extra
stealth + forks (zorilla), cloakbrowser-human / humanization-playwright,
crawlora-antibot / wafer / antibot-detect / anti-bot-sniffer (passive WAF
fingerprint catalogs), query-selector-shadow-dom, and agent-UX patterns
(Hark-style live agent windows).

---

## 1. RenderedTab (service.py) — playwright agent tab

### Accessibility-snapshot interaction model (BEST: browser-use, Playwright MCP)
- **What the best does:** browser-use and Playwright MCP are *accessibility-
  tree-first*. `browser_snapshot` returns the page's ARIA tree with stable
  `[ref=eN]` handles; the agent clicks/fills/hovers **by ref**, not by
  screenshots or CSS. Token-efficient (~200–400 tokens), deterministic,
  no vision model needed. Re-snapshot after DOM changes; every mutating
  command appends the fresh snapshot so the agent rarely issues a separate
  call. Playwright exposes it as `page.locator("body").aria_snapshot()`
  (Python: on Locator, not Page — `page.accessibility.snapshot()` is
  deprecated since 1.44). gptme's fix history is instructive: cap depth,
  never `networkidle`-wait before snapshotting (SPAs hang), `aria_snapshot`
  returns pre-formatted YAML-like text.
- **Our gap:** RenderedTab has `text()`, `links()`, `html()` — but no
  accessibility snapshot and no ref-based targeting. An agent driving it
  must guess selectors. **Action:** add `snapshot()` returning the aria
  tree (with graceful JS-fallback when the driver lacks it), and accept
  `aria-ref=eN` selectors in `click()`/`fill()` passthrough.

### Stagehand's observe/act split (BEST: Stagehand v4)
- **What the best does:** `observe("find the email input")` returns *real
  selectors* (credentials never reach the model); `act("click sign in")`
  self-heals when sites redesign. Discovery is split from action, so
  failures are diagnosable (no candidates = discovery problem; candidates
  but no effect = credentials/state problem). Also handles out-of-process
  iframes and closed shadow DOM natively.
- **Our gap:** our form resolver is heuristic scoring with no candidate
  reporting. **Action:** add `resolve_candidates()` returning the ranked
  list, so "field not found" errors can say "did you mean X?" — the same
  diagnosis split Stagehand uses.

### Stealth evasions (BEST: puppeteer-extra-plugin-stealth, zorilla fork, CRW)
- **What the best does:** 14 evasion techniques, not just webdriver-hide:
  `navigator.webdriver` deletion **plus** `--disable-blink-features=
  AutomationControlled` (the flag prevents Chrome creating the property at
  all — property deletion alone is detectable via `'webdriver' in
  navigator`), `window.chrome` runtime stub (`onMessage`, `sendMessage`,
  `connect`, `loadTimes`, `csi`), realistic `navigator.plugins` (Chrome PDF
  Plugin / Chrome PDF Viewer / Native Client), `navigator.languages`,
  WebGL vendor/renderer spoofing, media-codec patching, permissions
  overrides, iframe `contentWindow` patching, UA override with platform
  matching. rebrowser-playwright passes bot.sannysoft.com 30/30.
- **Our gap:** we do webdriver-hide + UA + viewport + locale + timezone +
  the blink flag. No chrome-runtime stub, no plugins, no languages, no
  permissions override. **Action:** extend the stealth init script set with
  chrome-runtime, plugins, languages, and permissions spoofing (all
  passive property stubs — no fingerprint *spoofing* rabbit holes, no
  canvas noise; keeping the "no deception machinery" line: we hide the
  automation tell, we don't forge a different device).

### Human-like behavior (BEST: cloakbrowser-human, humanization-playwright, emunium)
- **What the best does:** intercepts click/type/scroll: Bezier-curve mouse
  with easing + overshoot, realistic aim points (left-third for inputs,
  center for buttons), per-character typing with variable delays and
  occasional "thinking" pauses, typo-with-self-correction (2%), natural
  scroll (accelerate→cruise→decelerate micro-steps), idle micro-movements
  between actions, reading pauses proportional to content. Presets:
  `default` / `careful`. Originals stay accessible (`page._original`).
- **Our gap:** pacing.py only sleeps *between* actions; typing is instant
  `fill()`. No per-character typing, no scroll, no reading pause, no
  per-action overrides. **Action:** add `type()` (keyboard.type with
  per-char delay + thinking pauses), `scroll()`, `read()` pacing pause,
  per-action pacing overrides, `careful` preset, pacing stats. Keep it
  native (no new deps — sync API's `keyboard.type(text, delay=)` exists).

### Playwright MCP action coverage (BEST: Playwright MCP tool list)
- **What the best does:** navigate, snapshot, click, type, select, hover,
  press key, fill form, wait, screenshot, **console_messages,
  network_requests**, evaluate, handle_dialog, file_upload, tabs, resize,
  **pdf**, drag, close.
- **Our gap:** no `pdf()`, no console capture, no network-request log, no
  dialog policy (an unexpected alert hangs automation), no hover/dblclick/
  drag/press/reload/forward/element-screenshot on RenderedTab. **Action:**
  add all of these as real methods: listeners attached at page creation
  (console + request/response ring buffers), dialog auto-policy with
  recording, `page.pdf()`, keyboard press, hover, dblclick, drag, reload,
  forward, element screenshot.

### Shadow DOM (BEST: query-selector-shadow-dom; TRASH taught the rule)
- **What the best does:** Playwright CSS selectors pierce *open* shadow
  roots natively; the escape hatch for closed/nested roots is `evaluate()`
  walking `el.shadowRoot` recursively.
- **Our gap:** forms.py's resolver JS uses `document.querySelectorAll`,
  which stops at shadow boundaries — custom-element forms (Shoelace,
  Ionic, Material Web, Salesforce LWC) are invisible to it. **Action:**
  recursive deep traversal (document → shadow roots → same-origin iframes)
  in the resolver, describe, and marker JS.

---

## 2. errors.py — failure taxonomy + challenge detection

### Passive WAF fingerprint catalogs (BEST: crawlora-antibot, wafer, antibot-detect)
- **What the best does:** match header names (`cf-ray`, `x-datadome`,
  `x-iinfo` Imperva, `x-amzn-waf-action`, `x-kpsdk-ct` Kasada,
  `akamai-grn`), **Set-Cookie name prefixes** (`__cf_bm`/`cf_clearance`,
  `_abck`/`bm_sz` Akamai, `datadome`, `_px*`, `incap_ses_` Imperva),
  body/script markers trusted only on challenge-shaped responses.
  wafer catalogs 17 challenge types: Cloudflare, Akamai, DataDome,
  PerimeterX/HUMAN, Imperva/Incapsula (`___utmvc`, Reese84), Kasada,
  F5 Shape (`istlWasHere`), AWS WAF (`aws-waf-token`,
  `AwsWafIntegration`), Alibaba ACW (`acw_sc__v2`), TMD, Amazon, Arkose/
  FunCaptcha, GeeTest v4, hCaptcha, reCAPTCHA, Vercel, generic JS.
  anti-bot-doctor adds the *diagnosis→prescription* shape:
  symptom → cause → concrete fix.
- **Our gap:** we detect CF / PerimeterX / DataDome / generic captcha
  wall only; no cookie-name signals; no Akamai, Imperva, Kasada, F5,
  AWS WAF, Turnstile-as-distinct, Arkose, GeeTest. No `retryable` /
  `retry_after` guidance (429 with `Retry-After` header!). **Action:**
  add detections `akamai`, `imperva`, `kasada`, `f5-shape`, `aws-waf`,
  `turnstile`, `arkose`; accept `set_cookies`; honor `Retry-After`;
  add `retryable` + `retry_after_seconds` to every typed error; add a
  styled `summarize()` (the prescription card) for CLI output.

---

## 3. pacing.py — action pacing

### Behavioral mimicry timing (BEST: cloakbrowser-human, zeeeepa BehaviorMimicry)
- **What the best does:** pauses *between* actions are only half the
  story — reading pauses proportional to content length, click-hold
  durations (50–200ms), per-action speed profiles, `careful` preset for
  aggressive sites.
- **Our gap:** single global delay+jitter; no per-action overrides (a
  submit deserves more respect than a hover), no stats, no reading pause.
  **Action:** `per_action` overrides, `read()` pause, `careful` preset,
  `stats()` (total slept, count) — pacing becomes observable.

---

## 4. liveview.py — live agent window

### Trust UX (BEST: Hark-style live view + Playwright MCP "prefer snapshots")
- **What the best does:** one live message, in-place updates
  (`edit_media`), throttled, action descriptions that read like narration.
- **Our gap:** captions are bare text (`🔍 label…` / action). No action
  icons per verb, no elapsed time, no progress (x/25), no caption styles.
  **Action:** verb→icon map, elapsed clock, step counter, caption styles
  (`rich`/`compact`), finish/fail frames with timing summary. Keep the
  never-raises contract.

---

## 5. daemon.py — persistent browser daemon

### Agent-browser CLIs (BEST: superbereza/browser-skill, cyber-harness aiscan)
- **What the best does:** thin client + background daemon holding one
  connection; state in the daemon; auto-start on first use; idle auto-stop;
  session commands persist pages; stateless commands use fresh contexts.
- **Our gap (relative):** our daemon is solid (framing, pid liveness,
  escalation, event forwarding). Missing: ops for every new RenderedTab
  method (must mirror the CLI surface), a `stats` op, and idle reporting.
  **Action:** add ops for all new methods + `stats`; add new mutating ops
  to `_MUTATING_OPS`.

---

## 6. Tab / SessionHandle / BrowserService (service.py)

### Session/proxy patterns (BEST: linkedin-buddy ProfileManager, Browserbase)
- **What the best does:** dedicated profiles you sign into once; cookie
  transplant between profiles (storageState capture both cookies AND
  localStorage); proxy health probing at attach.
- **Our state:** already good (storage_state persists, pool probing at
  attach). **Action:** expose `service.stats()` (sessions/tabs/downloads/
  pacing), keep everything else.

### Style/UX mandate
- Add `nomorals/browser/styles.py`: output themes for browser surfaces —
  styled error cards (rich/plain), download/tab line formatters, snapshot
  headers. Presentation should feel god-tier, not functional.

---

## What this sweep will NOT do
- No fingerprint *forging* (canvas noise, WebGL renderer spoofing to fake
  GPUs, TLS/JA3 impersonation): hiding the automation tell is the line;
  forging a different device is deception machinery and out of scope.
- No challenge *solving* beyond the existing solver hooks — detection and
  honest reporting only.
- No new third-party deps (playwright-stealth etc. stay optional reading,
  not requirements); everything implemented natively.
