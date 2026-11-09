---
id: memory-and-blackboard
title: Memory, blackboard and the world model
summary: Where shared knowledge lives: memory for the durable, the blackboard for the working, the world model for facts with provenance.
topics: memory, blackboard, world model, facts, provenance
triggers: remember, blackboard, world model, shared state, fact
layer: memory
version: 1
---

# Memory, blackboard and the world model

Three different stores, three different jobs — do not blur them.

- **Memory** (`nomorals.memory`) is the durable record: facts about the
  owner, decisions, history. It survives everything and is curated.
- **The blackboard** (`agents.blackboard.Blackboard`) is the working
  surface between agents: versioned key/value entries with an immutable
  audit trail, TTLs, and reactive links that fire when a key changes.
  Every post bumps the entry's version.
- **The world model** (`agents.worldmodel.WorldModel`) is the
  blackboard used with discipline: facts are asserted with a kind —
  observed, inferred, proposed, unverified — and a source. Readers get
  the fact's version with the value; invalidation is a journaled
  tombstone, never a silent delete. A decision that consumed a fact can
  name the exact version it read.

When a worker hands knowledge to another worker, it goes on the
blackboard with provenance — never as an unlabelled claim in a prompt.
