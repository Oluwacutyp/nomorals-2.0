---
id: web-browsing-handoff
title: Web browsing — the ladder and the handoff
summary: How Devon browses: presentation ladder first, consent handled honestly, owner takeover last.
topics: browser, webnav, screenshots, webshot, handoff
triggers: browse, open page, screenshot, webshot, webrec, website
layer: browser
version: 1
---

# Web browsing

- Captures run through the webnav organ (`/webshot`, `/webrec`).
  Presentation rotates coherently — mobile first, then other coherent
  presentations; a wall is "hard" only after the whole ladder is
  exhausted.
- Rungs race in parallel; captures are concurrent; holds are
  non-blocking. Sequence is not selection.
- Consent walls are handled by the consent machinery, and the result
  reports what actually happened — a click that did not land is never
  reported as clicked.
- The owner's takeover is the LAST rung: the handoff link lets him drive
  the same session, and hand-back resumes her session where he left it.
- A robots refusal is answered in plain words with the reason — never a
  raw dump of internals.
- Signed-in state comes from cookie bundles (`/cookies import`) or the
  handoff login; the session persists in her browser profile.
