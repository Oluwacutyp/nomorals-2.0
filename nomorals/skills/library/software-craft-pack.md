---
id: software-craft-pack
title: Software craft pack — the cooperative delivery cell as a pack
summary: The software cooperative in pack form: the contract-first workflow, rubric, anti-patterns, evaluators and failure modes a delivery cell runs when it builds software.
topics: cooperative, craft, delivery cell, software engineering
triggers: craft pack, software cooperative, delivery cell
layer: craft
kind: pack
domain: build
fixtures: tests/test_delegation.py, tests/test_d18p2_run_ticket.py, tests/test_skillbook.py, tests/test_d2223_stage_b.py
version: 1
---

# Software craft pack

This pack describes how a cooperative cell builds software in Devon:
not as personas chatting, but as a contract with named roles, fenced
artifacts and receipts. It is the pack form of the machinery that
already runs — the delegation substrate, the specialist manifests and
the run-ticket projection — written down so any cell can load it.

## Terminology

- **Contract** — one unit of accountable work: a goal, a deliverable,
  a worker, an independent verifier, and the evidence the work must
  carry. Contracts live in the delegation store.
- **Cell** — the role set a contract binds: worker, witness, skeptic,
  verifier, integration owner, recovery supervisor, arbiter.
- **Claim ledger** — the contract's evidence trail: every claim names
  its producer and its validator, and carries a verdict.
- **Revision fence** — artifact ownership by revision and generation;
  a stale-base write is refused, never merged over.
- **Receipt** — the recorded outcome of a delivery, in the universal
  receipt shape, graded by the evidence attached to it.

## Workflow

1. Decompose the goal into steps small enough to verify one by one.
2. Open one contract per consequential step, naming worker and
   verifier — never the same actor — before work begins.
3. The worker produces the artifact inside the contract's scope and
   files its claims on the ledger with evidence references.
4. The skeptic attacks the claims: counterexamples, edge cases,
   untested paths. A challenged claim blocks completion until it is
   resolved on evidence.
5. The verifier signs off only against the contract's evidence
   duties; the completion gate refuses a contract with unresolved
   challenges or contradicted claims.
6. Delivery lands as a receipt chained to the contract, and the
   outcome is left where the steward can judge it later.

## Tools and formats

- Delegation contracts and the claim ledger (the accountability
  store); TaskGraph for step structure and acceptance criteria.
- Artifacts with provenance and supersede lineage — versioned, never
  overwritten in place.
- Skill docs and packs as flat Markdown files with frontmatter; the
  token router carries both, the lifecycle promotes both.
- Receipts in the universal schema, so a chain renders across cells
  in one shape.

## Quality rubric

- Every claim on the ledger cites evidence a verifier can follow.
- The deliverable matches the contract's output schema, field by
  field — a missing field is a defect, not a style choice.
- The acceptance checks named at contract time pass, and the run
  that proves it is recorded, not remembered.
- Independence holds: no actor verifies, validates or signs off
  their own work, at any point in the chain.

## Anti-patterns

- **Persona as contract** — a role defined only by a prompt's tone,
  with no output schema, rubric or verifier.
- **Self-verification** — the worker marking their own work checked.
- **Silent overwrite** — editing an artifact another contract owns,
  or writing from a stale revision and hoping.
- **Receipt-free delivery** — announcing done with no recorded
  outcome the steward can later judge.
- **Claimed capability** — presenting a draft pack's domain as
  proven before its fixtures have run and passed.

## Exemplars

- The content delivery cell: researcher, drafter, skeptic, verifier
  and delivery steward on one contract, the owner as arbiter, one
  receipt per delivered piece, chained contract → draft → post →
  metrics.
- A mission whose terminal gate refused DONE while a contract stayed
  open — the refusal is the system working, and the re-plan it forced
  is recorded on the same ledger.

## Evaluators

The pack's fixtures are its regression scenarios: the delegation
suite (independence, fencing, completion gates), the run-ticket
suite (the projection's cited-or-absent field law) and the skillbook
suite (routing, lifecycle, this pack's own schema). A fixture that
fails is a defect in the work or in the pack — either way it is
recorded, and the pack is patched only through the draft → prove →
promote lifecycle, never by editing a live claim.

## Failure modes

- **Stale revision** — a writer working from an old artifact base;
  the fence refuses the write and the worker re-reads.
- **Zombie generation** — work arriving from a superseded contract
  generation; refused by the generation check.
- **Unmapped evidence** — a claim citing an evidence kind the proof
  ladder does not map; it carries the unmapped reason, never a
  guessed grade.
- **Unproven pack** — a pack still in draft status being read as
  established capability; the draft label is the correction.

## Locale variants

One locale ships with this pack (English, Nigerian operating
context). Further variants are added as pack sections through the
same lifecycle when a cell works in another locale — a variant is a
translation of the craft, never a weaker rubric.

## Credential constraints

This pack's deliverables are working software, documents and
records, produced and verified inside the owner's own systems.
Nothing the pack produces confers a license, certification or
professional credential, and its receipts claim only the outcomes
they record — the work is the credential it has.
