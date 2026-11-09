---
id: content-cell
title: The content delivery cell — research, draft, fact-check, deliver, receipt
summary: Every cooperative content job runs as one delegation contract — researcher, drafter, skeptic, verifier, delivery steward — with a readable receipt per piece.
topics: content, drafts, shorts, weekly, publishing, receipts, series, calendar
triggers: weekly content, content pipeline, publish shorts, draft posts, receipt, content calendar, series, campaign
layer: agents
version: 1
---

# The content delivery cell

One pattern runs every cooperative content job (agents/content_cell.py):
a delegation contract with five roles and the owner as arbiter —
**researcher** (bounded, provenanced material) → **drafter** (the
owner's voice, per platform) → **skeptic** (challenges factual
claims; the standing gate is the number-trace — a number with no
source in the evidence never lands) → **verifier** (fit + sign-off)
→ **delivery steward** (lands the pieces, writes the receipts).
The day cell and the idea cell are instances of this pattern with
their own evidence shapes; the weekly pipeline and the shorts
publisher run through it directly.

Rules that make it honest:

- Claims carry evidence or they are labelled unverified; a blocked
  piece is challenged and resolved contradicted on the record,
  never silently dropped and never queued.
- Every landed piece gets a **receipt**: contract → draft → post →
  metric rows, readable end-to-end (social/series.py + /calendar
  show shape; agents/content_cell.receipt_chain joins the record).
  Internal ids ride structured data only — surfaces name pieces by
  platform, title and state.
- The cell produces drafts and scheduled deliveries for the
  owner's decision. Scheduling is not approval (#43 stands): a
  pipeline draft still goes through review; only the owner's own
  scheduling acts (his words, his queue) execute unattended.
- A series is a named, ordered run of pieces; a campaign is a
  series with an objective. Membership is metadata on the existing
  stores — never a parallel content store.
- Scheduled drafts post through the wired poster
  (social/drafts.wire_draft_poster, installed at boot). A fire
  with no poster wired fails loudly and the draft stays scheduled
  — silence is the one unacceptable outcome.
- Shorts publish through the factory's own publishers on the
  durable every-30-minutes sweep. A platform without credentials
  refuses in words and is recorded failed in the publish ledger —
  never faked, never retried blind.
