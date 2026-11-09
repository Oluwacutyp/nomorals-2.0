---
id: accountable-delegation
title: Accountable delegation — contracts, claims, fencing
summary: How work is delegated so it can be trusted: work contracts, independent verification, a claim ledger, revision fencing.
topics: delegation, contracts, verification, claims, specialists
triggers: delegate, contract, verify work, claim ledger, specialist
layer: agents
version: 1
---

# Accountable delegation

Delegation is governed by `agents.delegation` — read this before
handing any consequential job to a worker.

- Every delegated job is a **work contract**: goal, deliverable, scope,
  the evidence kinds required, a step budget, and named role-holders —
  worker, witness, skeptic, verifier, integration owner, recovery
  supervisor, arbiter. The verifier is never the worker; a claim's
  validator is never its producer. The store refuses both.
- Evidence is submitted against the contract and lands in the **claim
  ledger**: who claimed what, on what evidence, at what revision,
  verified by whom, with a verdict — supported, contradicted, partial,
  stale, unverified, unavailable, uncertain.
- Consequential contracts need a skeptic pass: every challenge must be
  resolved before completion can be recorded.
- Ownership is **revision-aware and fenced**: a worker writes an
  artifact at a base revision; a superseded revision's write is
  refused, and after a failure/re-plan the contract's generation moves
  — late evidence and late writes from the old generation are fenced
  out. Completion itself is idempotent: recording it twice replays the
  receipt, never double-records.
- Mission completion consults the contracts: a mission with open
  contracts cannot finish DONE.
- State renders on `/delegate` and `/skills` — living cards over the
  same stores, no second control surface.
