---
id: cbt-wellbeing-pack
title: CBT/wellbeing craft pack — structured exercises on the health shell
summary: The CBT exercise set in pack form: thought records, activation, restructuring, routines and relapse-prevention plans as structured intervention records, run as a cell on the shipped health shell.
topics: cbt, wellbeing, exercises, health, records
triggers: cbt pack, wellbeing, thought record, intervention record
layer: craft
kind: pack
domain: health
fixtures: tests/test_health_coach.py, tests/test_health_sweep.py, tests/test_d2223_stage_b.py
version: 1
---

# CBT/wellbeing craft pack

This pack carries the exercise craft of cognitive-behavioral
skills practice in Devon: the standard exercise set — thought
records, behavioral activation, cognitive restructuring, sleep
and activity routines, relapse-prevention plans, session prep —
as structured intervention records on the health shell that
already ships (the timeline, the mood and drift organs, the
previsit prep, the coach's no-diagnosis guard).

The pack is a skills-practice craft. The records are the person's
own exercise work, structured so it can be revisited, compared
and carried into a session — the shell tracks and structures; it
does not diagnose, and nothing in this pack changes that.

## Terminology

- **Intervention record** — one completed exercise, stored in
  its kind's structure: a thought record has a situation, the
  automatic thought, the emotion, the evidence both ways and
  the balanced thought — the fields are the exercise.
- **Exercise kind** — one of the seven structured forms:
  thought record, behavioral activation, cognitive
  restructuring, sleep routine, activity routine,
  relapse-prevention plan, session prep.
- **Completeness** — the record meets its kind's schema on
  read-back, not just at intake: what the store holds now,
  checked against the schema now.
- **Guard vocabulary** — the health shell's shipped list of
  diagnostic and prescriptive phrases (the coach guard's list):
  exercise records are scanned against it, and a record carrying
  such language is blocked, never delivered.

## Workflow

1. The person does the exercise — the words are theirs. The
   cell's work starts at structuring, never at inventing
   content.
2. Open the contract: the skills practice coach as worker, a
   skeptic, an independent verifier, the owner as arbiter;
   required evidence is the intervention record itself.
3. Structure the work into its kind's record and store it on
   the health shell; a record that does not meet the schema is
   refused with every missing or mis-shaped field named.
4. The skeptic scans the record's rendered text against the
   guard vocabulary — the standing question is whether the
   record stays inside the exercise's structure and the
   person's own words.
5. The verifier re-reads the stored record and re-validates it
   complete against the schema; only then is the evidence claim
   supported.
6. The run completes with a receipt derived from the record;
   a blocked or refused record is a failed run, recorded as
   one — never quietly completed over.

## Tools and formats

- The intervention record store (`health/interventions`): one
  sqlite home beside the health timeline's, owner-scoped like
  everything in the shell.
- The health timeline and drift organs — the person's logged
  words and mood history the exercises draw on, untouched by
  this pack.
- The coach guard's phrase list, reused verbatim as the scan
  vocabulary — one list, never a copy that can drift.
- Delegation contracts and the universal receipt schema, so a
  CBT run's record chains beside every other cell's.

## Quality rubric

- The record is stored, and re-validates complete against its
  kind's schema on read-back.
- The record carries no diagnostic or prescriptive language —
  the guard vocabulary scan is clean.
- The fields are the person's own exercise work; the structure
  adds shape, never content.
- Independence holds: the coach who structures the record
  never verifies it.

## Anti-patterns

- **Diagnosis in exercise clothing** — a record that names a
  condition or prescribes a treatment has left the craft; the
  scan exists to catch exactly this.
- **Invented content** — fields filled from the model's
  imagination instead of the person's work: a structured
  fiction is worse than no record.
- **Half-shaped records** — an exercise stored with its hard
  fields missing teaches the store to lie about completeness;
  the store refuses them, by name.
- **Silent blocking** — a blocked record the run reports as
  delivered; blocks are outcomes and are recorded as outcomes.

## Exemplars

- The Stage B CBT pilot: a thought record runs the full cell —
  stored, scanned, re-validated, receipted — and a second run
  carrying diagnostic language is blocked by the skeptic and
  fails in the open.
- The previsit prep: the shell's existing discipline of
  structuring the person's own history for a professional
  conversation — preparation as a record, in the same voice.

## Evaluators

The pack's fixtures are its regression scenarios: the health
coach suite (the guard's banned-phrase law over coach output),
the health sweep (the shell's organs end-to-end), and the
Stage B pilot suite (every exercise kind stored and validated,
the diagnostic block, the receipt chain). Across the fixture
set, the exercises produce zero diagnostic output — that is a
measured property of the fixtures, re-measured on every run.

## Failure modes

- **Schema refusal** — the exercise's fields do not meet the
  kind's schema; the store refuses with the problems named and
  the run records the refusal as its outcome.
- **Guard block** — diagnostic language in a record: the
  record is marked blocked, the claim contradicted, the run
  failed — the words stay the person's to revise.
- **Missing shell** — no health store available: the run says
  so as an absent organ rather than storing the record
  somewhere improvised.

## Locale variants

One locale ships with this pack (English, Nigerian operating
context). Exercise prompts and field labels translate through
the same lifecycle when the practice runs in another language —
the schemas and the guard discipline do not weaken per locale.

## Credential constraints

This pack's deliverables are the person's own structured
exercise records, produced and stored inside the owner's own
systems. Nothing the pack produces is a diagnosis, a treatment
plan or a clinical credential, and its receipts claim only the
outcomes they record — a completed exercise, stored completely,
in the person's own words.
