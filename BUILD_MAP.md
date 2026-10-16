# BUILD_MAP — recovered build-map numbering (partial)

## Verdict

**The full numbered build map is NOT recoverable from the tree, git
history, or docs.** No canonical plan document exists: there is no
`BUILD_MAP.md`, no numbered plan in `docs/`, no `*.md` plan file, and
no git commit whose message or tree ever contained one
(`git log --all --oneline -- "*BUILD_MAP*" "*build_map*"` → empty;
`git log --all --grep="build-map"` → empty; `grep -rln "build.map|BUILD_MAP"
docs/ *.md` → empty).

What survives is **scattered citations**: module/test docstrings and
comments of the form `build-map #N`. Those citations recover the
*meaning* of the numbers they mention — and nothing else. Every entry
below cites its evidence as `file:line`. Any number with no citation
is listed as **undefined** — not guessed.

Two non-numbered labels also appear and are not part of the `#N`
sequence: `build-map Phase 23` (marketing) and `build-map extension
#7` / `#9` (LLM cost display / one more extension). They are recorded
as found, not renumbered.

A `TEN_PACKAGES` manifest was also searched for (`grep -rn
"TEN_PACKAGES|ten_packages|ten packages" .` → no hits). No
ten-packages list exists in the tree. Do not invent one.

## Method

```sh
grep -rn "build-map #" nomorals/ tests/ --include="*.py"
git log --all --oneline -- "*BUILD_MAP*" "*build_map*"   # empty
git log --all --oneline --grep="build-map" -i            # empty
grep -rln "build.map\|BUILD_MAP" docs/ *.md              # empty
```

Highest cited number: **#107** (live audio rooms). That only proves
the plan *reached* at least #107 — it does not define #1–#107.

## The six requested numbers — recovered with citations

### #46 — Socratic tutoring engine

* `nomorals/learn/tutor.py:1` — "Socratic tutoring engine — Devon as tutor (build-map #46)."
* `nomorals/learn/tutor.py:626` — mistake recorded automatically "(build-map #46's promise)."
* `nomorals/learn/tutor.py:770` — "Record the mistake (build-map #46's promise)."
* `nomorals/learn/__init__.py:3` — "Build-map #46. `tutor.py` is the dialogue engine…"
* `nomorals/learn/flashcards.py:3` — "Build-map #46. Every tutoring mistake is recorded…"
* `nomorals/learn/curriculum.py:678` — "Quizzes feed the mistake notebook (#46)…"
* `nomorals/legal/research.py:11` — cross-ref: "#46's source-grounded tutoring".
* Test: `tests/test_tutor.py:1` — "Socratic tutoring + photo input + mistake notebook (build-map #46)."

### #53 — Ask-your-data health coaching

* `nomorals/health/coach.py:1` — "Ask-your-data coaching — HealthKit / Health Connect (build-map #53)."
* `nomorals/agents/morning_briefing.py:1016` — "Readiness (health coaching, build-map #53)".
* `nomorals/health/form.py:25` / `nomorals/health/training.py:19` — coaching output "passes `guard_coaching` (the #53 diagnostic-phrase ban)".
* Test: `tests/test_health_coach.py:1` — "Ask-your-data coaching — build-map #53."

### #76 — Single-contract review (letter grades + Nigerian playbooks)

* `tests/test_contracts.py:1` — "Build-map #76: letter-grade contract risk scores + Nigerian playbooks."
* Implementing module: `nomorals/legal/contracts.py` (`review_contract`, `format_review`).
* `nomorals/legal/portfolio.py:5` — "#76 reviews single contracts; this module ingests those reviews…" — i.e. #76 is the *single-contract* review; the portfolio ingesting it is a different number, **#79** (`tests/test_portfolio.py:1` — "build-map #79: contract portfolio").
* Note: no `build-map #76` literal exists in `nomorals/` source itself — the literal citation is in the test plus the `#76` references in `portfolio.py:339,387`. The mapping is still unambiguous.

### #80 — Regulatory change monitoring

* `nomorals/legal/regulatory.py:3` — "Build-map #80 (Thomson Reuters Regulatory Intelligence / Compliance.ai pattern)."
* Test: `tests/test_regulatory.py:1` — "Tests for build-map #80: regulatory change monitoring (CBN/SEC/NDPA)."

### #83 — Curated daily batches + mutual-preference matching

* `nomorals/matching/__init__.py:1` — "Curated daily batches + mutual-preference matching (build-map #83)."
* `nomorals/matching/batches.py:1` — "Curated daily batches — scarcity-as-ritual (#83)."
* `nomorals/matching/chat.py:1` — "Chat control for /match — daily batches + stable matching (#83)."
* `nomorals/matching/questionnaire.py:1` — "Importance-weighted preference capture (#83) — the OkCupid pattern."
* `nomorals/matching/stable.py:1` — "Stable matching mechanisms (#83)…"
* `nomorals/matching/__init__.py:8` — "#46 mentoring — daily learning picks, mentor ↔ mentee matching" (cross-ref: #46's matching surface lives under #83's machinery).
* Test: `tests/test_matching.py:1` — "Offline tests for build-map #83…"

### #87 — Photo-proof challenges + streak/squad mechanics

* `nomorals/health/challenges.py:1` — "Photo-proof challenges + streak/squad mechanics (build-map #87)."
* `nomorals/agents/partner/runtime.py:2706` — "Photo-proof challenges + streaks + squads (#87)".
* `nomorals/social/chat/control.py:363` — "/challenge … — photo-proof challenges + streaks + squads (#87)".
* Test: `tests/test_challenges.py:1` — "Tests for build-map #87…".

## Other cited numbers — recovered the same way

Each line is the first citation found; `(+N more)` = additional
citations of the same number. Descriptions are quoted/paraphrased
from the cited line only.

| # | First citation | What the citation says |
|---|---|---|
| #10 | `tests/test_relationship_cadence.py:1` | relationship cadence |
| #11 | `tests/test_email_triage.py:1` | AI email triage module |
| #14 | `tests/test_progression_unlocks.py:1` | features-as-loot (progression-gated commands) |
| #18 | `nomorals/agents/coding.py:663` (+12) | complexity-routed model selection |
| #19 | `nomorals/agents/partner/runtime.py:519` (+3) | SMS fallback surface |
| #20 | `nomorals/research/citations.py:1` (+1) | claim-level evidence and citation infrastructure |
| #22 | `nomorals/media_edit/models.py:1` (+1) | license-aware image model router |
| #23 | `tests/test_vision_stack.py:1` | vision utility stack |
| #26 | `nomorals/media_edit/video_models.py:1` (+1) | local text-to-video model router |
| #30 | `tests/test_dual_voice.py:1` | dual voice stack: XTTS private / Chatterbox public |
| #33 | `tests/test_liveview.py:1` | live agent window |
| #34 | `nomorals/agents/confidence.py:1` (+5) | calibrated confidence UX — anti-hallucination |
| #35 | `tests/test_representation.py:1` | representation-quality ledger |
| #36 | `nomorals/agents/partner/runtime.py:89` (+4) | two-tier memory |
| #37 | `tests/test_add_only_memory.py:1` | ADD-only memory + session episodic layer (Mem0 pattern) |
| #39 | `tests/test_typed_memory.py:1` | typed memory objects + hybrid retrieval |
| #40 | `nomorals/agents/morning_briefing.py:850` (+6) | spaced-repetition resurfacing |
| #41 | `tests/test_mcp_server.py:1` | MCP server |
| #43 | `nomorals/social/drafts.py:1` (+1) | agent-drafts with the explicit-override rule |
| #44 | `tests/test_creator_voice.py:1` | voice-trained brand voice + virality + evergreen + auto-plug |
| #45 | `nomorals/social/content_pipeline.py:1` (+1) | research → draft → schedule content pipeline |
| #47 | `tests/test_dialect.py:1` | Ekiti/Ilawe Ekiti dialect tutoring |
| #48 | `nomorals/learn/curriculum.py:1` (+1) | WAEC/JAMB curriculum alignment + course generation |
| #49 | `tests/test_money_guard.py:1` | conversational money primitive + pre-transaction warnings |
| #51 | `nomorals/health/coach.py:443` (+1) | health timeline (coaching insights land there) |
| #52 | `tests/test_previsit.py:1` | pre-visit layer: navigation, NOT diagnosis |
| #54 | `tests/test_nutrition.py:1` | photo meal logging + 2 questions |
| #55 | `nomorals/agents/morning_briefing.py:1024` (+1) | mood patterns |
| #56 | `nomorals/health/drift.py:1` (+1) | drift detection + recovery actions |
| #57 | `tests/test_ace_step.py:1` | ACE-Step 1.5 song-bed generation |
| #60 | `tests/test_distribute.py:1` | `nomorals/media/distribute.py` |
| #66 | `tests/test_cart_recovery.py:1` | `nomorals/commerce/cart_recovery.py` |
| #68 | `nomorals/social/whatsapp_cost.py:23` | WhatsApp cost transparency pitch |
| #71 | `nomorals/travel/__init__.py:1` (+1) | travel price watchers |
| #72 | `nomorals/travel/itinerary.py:1` (+1) | TripIt-style email forwarding → auto itinerary |
| #73 | `nomorals/travel/whitelabel.py:1` | white-label travel assistants (B2B2C) |
| #79 | `tests/test_portfolio.py:1` | contract portfolio (ingests #76 reviews) |
| #81 | `nomorals/social/profiles.py:1` (+1) | prompt-scaffolded profiles + element-level likes |
| #82 | `nomorals/social/voice_notes.py:1` (+1) | voice notes + the voice trust layer |
| #85 | `nomorals/health/training.py:1` (+1) | recovery-gated training plans + voice coaching |
| #86 | `nomorals/health/form.py:1` | Seer as form coach — video movement analysis |
| #90 | `tests/test_route.py:1` | Predict → Build → Solve routing pipeline |
| #92 | `tests/test_congestion.py:1` | multi-agent congestion prediction |
| #93 | `nomorals/property/scam.py:1` (+1) | Nigerian rental scam-detection engine |
| #94 | `nomorals/property/passport.py:1` (+1) | verify-once rental passport + true-cost calculator |
| #95 | `tests/test_value.py:1` | DIY AVM with confidence interval + comp-picker |
| #97 | `nomorals/marketing/aeo.py:1` (+1) | AEO/GEO visibility tracking |
| #98 | `nomorals/marketing/send.py:1` (+1) | self-hosted send layer |
| #99 | `nomorals/marketing/briefs.py:1` (+1) | brief-first content pipeline |
| #100 | `nomorals/social/leads.py:1` (+1) | comment → DM → lead social-sales loop |
| #101 | `nomorals/marketing/guardrails.py:1` (+1) | rule-based ad guardrails |
| #102 | `nomorals/marketing/competitor.py:1` (+1) | no-access competitor intel |
| #106 | `tests/test_audiobook.py:1` | EPUB → audiobook + disclosure |
| #107 | `tests/test_audio_rooms.py:1` | live audio rooms |

Non-`#N` labels found verbatim: `build-map Phase 23`
(`nomorals/marketing/__init__.py:1`, marketing automation), `build-map
extension #7` (`nomorals/llm/cost_display.py:1`, transparent cost
display), `build-map extension #9` (`tests/test_redteam.py:459`,
attack catalog + sandbox-first). They are recorded as found, not
renumbered.

## Undefined numbers — NOT recovered, NOT invented

Within 1–107, these numbers have **no citation anywhere** in
`nomorals/` or `tests/` and are therefore **undefined** in the
recovered map: 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 21,
24, 25, 27, 28, 29, 31, 32, 38, 42, 50, 58, 59, 61, 62, 63, 64, 65,
67, 69, 70, 74, 75, 77, 78, 84, 88, 89, 91, 96, 103, 104, 105.
(47 numbers.)

Also unrecoverable: the *ordering* of the plan (numbers are cited
independently; nothing says #46 precedes #53 in build order), any
phase grouping (except the single `Phase 23` label), and any status
(done / in-progress) per number — citations prove a module *claims*
a number, not that the number is complete.

## What would fully recover the map

The original plan document (if it ever existed outside this repo),
or the author's own numbering notes. Until one surfaces, this file
is the complete honest recovery: 60 cited numbers with evidence,
47 undefined numbers named as undefined, and no invented entries.
