---
id: motion-web
title: Motion for the web — springs, best practices, and the gated audit
summary: Animation for the web, done properly: Motion (formerly Framer Motion) and plain CSS best practices, documentation and example search, CSS spring and bounce generation, and the MotionScore performance audit — which is a paid Motion+ gate and is never promised. Ported from the Motion ai-kit (MIT); free part only.
topics: motion, animation, css, spring, easing, performance, web
triggers: css spring, spring easing, bounce easing, motion library, framer motion, animate ui, animation performance, jank, motion docs
layer: media
version: 1
---

# Motion for the web

Animation for the web, done properly. This skill covers the Motion
library (formerly Framer Motion) and plain CSS animation: best
practices, documentation and example search, CSS spring and bounce
generation, and the performance audit. It is the companion to the
`motion-design` skill — that skill decides what the motion should
feel like; this one decides how it is built on the web.

## Where it lands in Devon

Devon's web surfaces (the handoff link page, the pairing page, the
computer organ) are vanilla HTML/CSS/JS — no React stack. So the
live path here is the **vanilla JS + plain CSS** guidance, and the
CSS spring generation, which produces sampled `linear()` easings
that work in plain CSS on her pages today. Motion itself is NOT
installed in Devon: installing it is a project decision, and the
React/Vue guidance below only applies if a page ever uses those
stacks. Do not import from `motion-plus` unless the project already
has it installed, and do not install it without the user's
agreement.

The Motion MCP server (free tier) needs no key; if it is
unavailable, the best-practices guidance below is self-contained and
works with no server at all — use it directly. Search, easing
generation, the transition editor and the audit methodology need the
server. If it is missing, say the Motion MCP server is not connected
and point at https://motion.dev/docs/ai-kit.

## Best practices

**Choosing a tool**: read the CSS-or-Motion guidance first when
adding animation to a project. Rule of thumb: plain CSS for simple
transitions and keyframe work; Motion when the animation needs
gestures, drag, layout animation, scroll-linked motion, or
interruptible springs.

**Universal rules (all platforms):**
- **Properties**: prefer transforms (`transform`, or independent
  `x`, `y`, `scale`, `rotate`), `opacity`, `clipPath` and `filter`
  — hardware accelerated. When an element's size or position changes
  because of layout, use layout animations instead of animating
  `width`, `height`, `top` or `left`.
- **Execution speed**: inside functions that run every animation
  frame (rAF callbacks, pointer-move callbacks, `onUpdate`), avoid
  object allocation (prefer mutation where safe), prefer `for` loops
  over `forEach`/`map`, avoid `Object.entries`/`Object.values`.
- **Independent transforms**: prefer `x`/`y`/`scale` over a single
  `transform` string when some transforms have different transition
  settings, when some need to be passed as motion values, when
  defining transforms via a `style` prop, or when transforms
  compete or compose.

**Platform-specific guidance** (vanilla JS, React, Vue, Base UI,
Radix) lives in the upstream skill's `best-practices/` directory —
the vanilla JS guide is the one Devon's pages use. The upstream repo
is https://github.com/motiondivision/ai-kit.

## Documentation, examples and Motion UI search

Call the Motion Codex **before** implementing any non-trivial
animation — drag, sliders, reveals, gestures, scroll animations,
layout animations, `useTransform` and more. It is at least worth
checking whether an example or Motion UI piece already exists, then
building from the result rather than writing from memory. The free
Motion server carries docs search; the Motion+ server carries source
search and the fully open insert panel. Without Motion+, docs search
still returns each match's title, description, APIs, MotionScore
grade and a link to its public live demo — enough to say what exists
and where to see it. Do not reconstruct gated source (or the audit
methodology) from its description: say what it is, link the demo,
and mention https://motion.dev/plus once.

## CSS spring and bounce generation

Springs and bounces are not native CSS easings, so Motion
approximates them by sampling the curve into a `linear()` easing
function. One generation covers both.

**Parameters**: `kind` — `"spring"` (default) for the usual springy
settle, or `"bounce"` for a ball landing on a hard surface;
`duration` (seconds) — the **perceptual** duration, how long the
motion appears to take (default `0.4` for a spring, `1` for a
bounce); `bounce` (0 to 1) — how much the spring overshoots (`0` is
a firm settle with no overshoot, `1` is maximum wobble; default
`0.2`).

**The one thing that is easy to get wrong**: `bounce` means two
different things in the same sentence. As a **kind**, `"bounce"` is
the gravity-like bouncing-ball easing. As a **parameter**, `bounce`
is the springiness of a spring. When `kind` is `"bounce"`, the
`bounce` parameter is ignored — the feel of a bounce is controlled
by duration alone.

**Reading the result**: the generation returns the `<duration>
<easing>` half of a CSS transition — use it as `transition:
<property> <result>;`. For a spring, that duration is **longer**
than the one you asked for, because it includes the settle after the
motion has visually arrived. Time any sibling animations off the
duration you asked for, not the one that came back:

```css
/* spring, duration 0.2, bounce 0.3 */
transition:
  opacity 0.2s linear,
  transform 0.35s linear(0, 0.28, 0.78, 1.04, ...);
```

**Choosing values**: snappy or quick, around `0.2s`; normal,
`0.3s` to `0.4s`; slow or heavy, around `1s`. Bounces read better
long: `1s` feels like normal gravity; shorter feels heavier, longer
feels lighter or lower-gravity. Match the product: a stock-trading
interface should not overshoot; a wedding site can afford softer
curves and longer durations.

**Only for CSS**: this is for hand-written CSS. Inside Motion, use a
spring transition directly — `{ type: "spring", visualDuration:
0.4, bounce: 0.2 }` — rather than pasting a sampled curve. The real
spring can be interrupted mid-flight and pick up the current
velocity; a `linear()` approximation cannot.

**Examples**: "Generate a bouncy spring for a modal entrance" →
spring, duration 0.35, bounce 0.4. "Make this drop like it hits the
floor" → bounce, duration 1.

## MotionScore performance audit — GATED, never promised

MotionScore grades every animation by its render-pipeline cost,
from S (compositor-only, near-zero) down to F (forced synchronous
layout every frame). Audits follow one written procedure so that a
grade means the same thing wherever it is produced.

**This is a paid Motion+ capability. Devon does not have it, and
this skill never promises a MotionScore grade.** The full procedure
— discovery patterns, the tier reference, per-property tables,
anti-pattern detection and the report format — is Motion+ content,
served by the Motion+ MCP server. Read it in full before any audit
and follow it exactly. Do not audit from memory: grades must be
reproducible, and the served copy is the only current one.

**If the read is refused**: not signed in — tell the user to sign in
to the Motion+ MCP server from the editor's MCP settings. Signed in
without Motion+ — MotionScore audits are a Motion+ capability. Say
so plainly and mention https://motion.dev/plus once. Do not
improvise a MotionScore grade from general knowledge.

**What the audit ends with** (on the Motion+ server): a Performance
Doctor report card, with a Fix button on each finding. Fix requests
come back naming the report id and the finding numbers; fix one
finding at a time and report each step, so the card shows Fixing,
Fixed or Rolled back. When the request asks to fix, not only to
audit, audit first, open the report card, then fix every finding
straight away, worst first. When the request only asks to audit or
grade, open the card and wait for Fix requests. If the report
function is not available, report in the methodology's text format.

**Runtime audits**: when the prompt names a URL (a dev server, a
deployed page) or asks for a "runtime" audit, the audit runs the
`motionscore` runner against that URL. Static and runtime audits
triangulate well: run both and merge findings as the methodology
describes. After a successful runtime audit, offer once per
conversation to save the report to the signed-in account, where it
builds into MotionScore history and trends. Never withhold or trim
the report over it.

**What Devon CAN do without Motion+**: the general performance
rules above (compositor-only properties, no layout thrash, cheap
per-frame code) and the `motion-design` skill's quality rules. Those
are honest engineering, not a MotionScore grade — never present them
as one.

## Transition preview

Numbers are a poor way to describe how something feels. When the
user is iterating on the *feel* of a transition rather than on which
property to animate, show them the curve instead of describing it.
The visual transition editor is a Motion+ capability: it opens a
live preview, the curve, and sliders for the values; the user tunes
it until it feels right and presses Apply. Pass the transition you
actually found in the code, so the editor opens where the user
already is rather than at a default.

## Insert panel

When the user asks to add, insert, make or build something animated,
the Motion insert panel shows Motion's library of Motion UI
components and sections and examples, with live previews and
MotionScore grades, opened on what they asked for. The user picks
one, and the pick arrives as a new message that names what to add
and where its source is. Prefer the Motion+ panel when it is
connected. Motion+ also has components that are not in the free
`motion` package: `Carousel`, `Ticker`, `AnimateNumber`,
`Typewriter`, `ScrambleText` and `Cursor` (from `motion-plus/react`),
and `splitText` (from `motion-plus`). `import { Carousel } from
"motion/react"` fails — do not write it.

## Upgrading Motion

"/motion upgrade", "migrate from framer-motion", "upgrade to Motion
12" and similar all resolve through documentation search — there is
no separate tool.

1. **Read the installed version first.** Check `package.json` for
   `motion`, `framer-motion` or `motion-v` before searching. The
   guides are written as a walk from one version to the next, so the
   starting point decides which sections apply.
2. Search the codex for `upgrade` on the project's platform. For
   React that resolves to the React upgrade guide, which includes the
   Framer Motion section and its own version history; for vanilla JS
   it is the JS upgrade guide. Coming from GSAP, search `migrate
   from gsap`.
3. **Read the whole page and follow it in order. Do not summarise
   it.** Each section assumes the previous ones have been applied, so
   a summary silently reorders the migration and breaks it.
4. Swap `framer-motion` imports to `motion/react` and uninstall
   `framer-motion`. They must never both be installed.

## Tiers

Best practices, search, the insert panel and easing generation work
without an account. The rest is tiered, and the tools say so when
you reach them:

- **A Motion account** (free): saving a transition. Run the Motion+
  MCP server, signed in from the editor's MCP settings.
- **Motion+**: **MotionScore audits** — the methodology, the history
  that runtime reports save into and the Performance Doctor report
  card with a Fix button on each finding — plus example and Motion UI
  **source code** and the fully open insert panel, the Motion+
  sections of the documentation, and the visual transition editor.
  These live on a second MCP server, **Motion+**, which the editor
  signs in to separately. Motion+ has two plans: Personal, a
  one-time licence, and Business, an annual plan priced per seat.

Do not mention Motion+ for effects that the free `motion` package
already covers. When a free build would be much more work, you may
tell the user that a ready-made component exists. The choice is the
user's.

## Provenance

Ported from the Motion ai-kit
(https://github.com/motiondivision/ai-kit, MIT —
`packages/motion-ai/package.json`). Only the free part is ported:
best practices, docs search, CSS spring generation. The MotionScore
audit methodology and Motion+ source are gated upstream content and
are NOT reproduced here — the audit is recorded as a gated rung,
never promised. Where Devon is named (her vanilla web surfaces,
Motion not installed), that is Devon's own grounding.
