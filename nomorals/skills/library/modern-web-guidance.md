---
id: modern-web-guidance
title: Modern Web Guidance — web standards, accessibility, security, performance
summary: Google's Modern Web Guidance 1.0 as Devon's web router: search the vendored guides first for any HTML/CSS or client-side JS work on her web surfaces, retrieve the guide, verify the code against it, and disclose what was applied. Ported from Google's modern-web-guidance skill (Apache-2.0); the 209 guides are vendored under this skill's directory.
topics: web standards, html, css, javascript, accessibility, security, performance, baseline, web components, forms, passkeys, csp, sanitization, web vitals
triggers: html, css, web page, web surface, dialog, modal, popover, form, autofill, validation, dark mode, accessibility, aria, keyboard, focus, csp, content security policy, sanitize html, passkey, webauthn, web vitals, inp, cls, lcp, view transition, container query, anchor positioning, web component, shadow dom, install web app, pwa
layer: media
version: 1
---

# Modern Web Guidance

This skill is Devon's web router. It applies whenever she builds or
changes one of her **web surfaces** — the token-gated handoff link
page, the PHP pairing page, the computer organ, the motion-server
pages (`bridge/motion-server.mjs`, `nomorals/media/motion_rungs.py`),
`nomorals/api/server.py`, `nomorals/hub/server.py`, and the builders'
webapp templates (`nomorals/builders/templates/webapp/`). For that
work, the guides below are the standard: search them **first**,
before writing an ad-hoc solution or adding a dependency, because
web APIs move fast and training weights carry obsolete patterns.

**Scope — where it does and does not trigger.** It triggers for
HTML/CSS and client-side JS on her web surfaces: modals, dialogs,
popovers, tables, dark mode, anchor positioning, container queries,
view transitions, scroll-driven animation, forms, validation,
autofill, accessibility, custom selects, web vitals (LCP, INP, CLS),
fetch priority, image optimisation, security, passkeys, HTML
sanitisation, web components, dates/timezones, built-in AI,
WebAssembly. It does **not** trigger for her chat/media work, for
backend work (SQL, ORMs, API routes), for infra or scripts (CI/CD,
Docker, Python/Go), or for tooling and trivial edits (tests,
renames, copy). The motion skills (`motion-design`, `motion-web`)
own how things *feel*; this skill owns whether the platform idiom
is correct — prefer a native element over a JS reimplementation,
then make it move well.

## The guides (vendored, read from disk)

The 209 guides live under this skill's directory at
`nomorals/skills/library/modern-web-guidance/guides/<category>/`,
one plain-markdown file per guide, no frontmatter, each starting
with `# Title`. They are copied as-is from Google's Modern Web
Guidance 1.0 (Apache-2.0, LICENSE kept alongside them) — the
detail is the product, so they are never summarised down. The
skill-version pin is **`2026_11_09-5bcc021b`**; re-pin it when the
guides are updated, because the Baseline data inside them goes
stale as the platform moves.

| Category | Guides | What it covers |
|----------|--------|----------------|
| accessibility | 4 | Landmarks, headings, ARIA, names, language, keyboard/focus, alt text, live regions, colour, typography, motion, modals, testing — including the 52K discipline guide, reviewed with real assistive tech |
| built-in-ai | 5 | Chrome's on-device language detection, language model, prompt API, summariser, translator |
| css | 18 | Layout, fluid scaling, `:has()`, size-aware styling, responsive design |
| forms | 19 | Autofill (×4), validation after interaction, SMS OTP, custom select pickers, sliders, switches |
| html | 8 | Web components, custom elements, shadow DOM, form-associated custom elements |
| js | 5 | Event differentials, global calendar systems, high-precision timestamps |
| performance | 26 | INP causes, long tasks, image priority, task scheduling, SPA view transitions |
| privacy | 1 | Privacy |
| pwa | 2 | Install a web app, migrate a web-app origin |
| security | 17 | CSP, sanitising untrusted HTML, Trusted Types, passkeys (×6), outbound connections, subresources, origins |
| ui-atoms | 13 | Responsive tables, position-aware tooltips, scroll progress, per-component light/dark themes |
| ui-behaviors | 40 | Dialogs/popovers (top layer, light dismiss), scroll, swipe, drag-and-drop, transitions, physics-based easing |
| ui-components | 22 | Dialog, accordion, carousel, tabs, menu, navigation drawer, dark-mode toggle, copy button, progress, spinner |
| user-experience | 3 | Content-first noise reduction, form-factor adaptation, natural interactions |
| visual-design | 22 | Colour, dark mode, typography, contrast, WebGL shaders, font fallbacks |
| wasm | 1 | C++ on the web |
| webmcp | 3 | MCP-shaped tools exposed by web pages for agents (agentic forms, agentic JS tools) |

## Primary workflow

### Step 1. Search the vendored guides

Form a **2–5 word action-oriented query** for a single web
platform capability or UI pattern: `<verb> + <UI/platform
object>`, optionally paired with the target HTML/CSS/JS feature
name. Omit framework/library names and app/domain nouns. Split
multi-feature tasks into separate searches.

- Name the implied UI primitive when the request does not: "confirm
  first" means a confirmation modal dialog; "pick a date" means a
  date-picker input. Search `"confirmation modal dialog"` or
  `"date picker input"`, not the domain phrasing.
- Search `"light dismiss dialog"` or `"responsive data table"` —
  not `"dialog"` (too terse), not `"responsive table dark mode"`
  (compound — split it), not a framework-flavoured query.

There is no CLI in this phase: search by reading the guides from
disk. Pick the category from the table above, then grep the guide
files in it for the capability (titles first, then the body).
Judge matches by the guide's description — a guide whose topic
matches the task is a match; a guide that only shares a word is
not. If nothing matches after one rephrase with the concrete
HTML/CSS/UI component noun or synonym, proceed without retrieving
unrelated guides.

### Step 2. Retrieve the guide

Read the full guide file(s) — both the specific use-case guide
that matches the task and any broader discipline guide that
applies (`html`, `css`, `forms`, `accessibility`, `color`,
`performance`, `security`). Read the whole file; do not skim the
intro and stop. The discipline guides carry the rules the
specific guide assumes.

### Step 3. Verify guidance compliance

Before concluding, cross-check the implementation against the
retrieved guide:

- **Applicable guidance and fallbacks**: the modern pattern and
  the fallback the guide requires are both applied, without
  forcing unrequested features.
- **Task fulfilment**: the implementation fully satisfies the
  request.
- **Native first**: if the guide names a native element
  (`<dialog>`, `<select>`, popover, anchor positioning), use it
  rather than a hand-rolled JS equivalent, and adapt the guide's
  vanilla HTML/CSS/JS to Devon's surfaces without dropping the
  modern feature.

### Step 4. Disclose skill usage

When this skill shaped a change, end the response that delivers
those changes with a `### Modern Web Guidance` section:

- **When guides shaped the code**: one bullet (1–2 sentences,
  ≤75 words) per newly applied `<id>`, in first-person voice,
  explaining the non-obvious implementation choices, CSS/API
  overrides, or fallbacks `<id>` required and the pitfall it
  prevented, anchored by a short verbatim quote of that specific
  rule or gotcha (never the guide's intro overview). Never guess
  `<id>`s or quotes from memory — re-read the guide if needed —
  and credit only guide-backed changes from the current turn.
- **When guides were checked but none applied**: one first-person
  sentence naming which topics or guides were checked and why none
  applied (omit if any guide applied).

## Alternative flow: modernise an existing surface

Only when asked for a broad, open-ended modernisation or
legacy-pattern scan of one of Devon's web surfaces — where no
specific target feature or component is named — run the
modernisation flow. Do **not** run it when asked to add, update,
fix, or audit a specific feature or component.

1. **Scan**: find legacy web-platform patterns across the surface
   (the `legacy-scan` command is a Phase 2 rung and is not wired
   yet; until it lands, scan by hand against the guides' rules —
   the security, html, css and ui-behaviour guides name the
   legacy patterns they replace).
2. **Verify**: check each finding against its `keepWhen` /
   `dropWhen` rubric; drop false positives (fallbacks,
   cross-file helpers, vendor code, lookalikes).
3. **Report**: summarise verified findings (and the
   modernisation plan, if editing), noting that a scan only
   checks the subset of the catalog that has deterministic
   legacy rules.
4. **Modernise** (only when asked to modernise rather than only
   audit/report): retrieve each finding's guide (Step 2), apply
   the replacement (keeping existing libraries as fallbacks
   where the guide requires one), and re-scan the modified files
   to confirm resolution. Dropped findings and kept fallbacks
   still match static scans; do not edit safe code to force a
   clean scan.

## Interpreting browser support and fallbacks

- **Default**: all guides assume **Baseline Widely available**
  features are safe to use without fallbacks. For features that
  are not Baseline widely available, follow the fallback
  recommendations in the guide, unless the owner has specified a
  custom browser support policy.
- **Custom policies**: if the owner defines explicit browser
  support requirements, use the guide's compatibility data to
  decide whether a fallback can be ignored (for `Baseline YYYY`
  targets, a feature satisfies the target if its "Baseline
  since" date is `<= YYYY`).
- Devon's surfaces are served to modern browsers (her own
  handoff/computer pages and the owner's browser). No custom
  policy is set today; if a surface ever targets a restricted
  runtime, document the policy in the skill's notes first, then
  apply it.

## What Devon does not have yet (honest gaps)

- **Content Security Policy**: none of her web surfaces sets a
  CSP header today (`bridge/motion-server.mjs` sets only
  `Content-Type` + `Content-Length`). The `security` guides give
  the exact header — the first fix this skill should drive.
- **HTML sanitisation**: no sanitiser for web output. The
  `sanitize-untrusted-html` guide is the reference when one is
  built.
- **Web vitals**: no INP/CLS/LCP measurement anywhere. The
  `performance` guides are the reference when it is added.
- **Passkeys / WebAuthn**: mentions only, no code. The six
  passkey guides are the reference when it is built.
- The `legacy-scan` command and the CLI search/retrieve rung are
  Phase 2 work, not part of this skill port.

## Provenance

Ported from Google's Modern Web Guidance 1.0
(https://developer.chrome.com/blog/modern-web-guidance-v1,
https://github.com/GoogleChrome/modern-web-guidance, Apache-2.0).
The router body above is adapted into Devon's format and voice —
the search step reads vendored guides from disk instead of calling
the `npx modern-web-guidance` CLI, the scope is Devon's web
surfaces, and the disclosure step names Devon. The 209 guide files
are copied as-is (reference material, not Devon's voice) with the
Apache-2.0 LICENSE kept alongside them. The `chrome-extensions`
skill is deliberately not ported: Devon builds no Chrome
extensions. Skill-version pin: `2026_11_09-5bcc021b`.
