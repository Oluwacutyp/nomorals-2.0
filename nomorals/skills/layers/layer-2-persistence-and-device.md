---
id: layer-2-persistence-and-device
title: Layer 2 — Persistence & device — storage, accounts, native bridges
summary: Instruction file for architectural layer 2 (persistence-and-device).
topics: layers, architecture, layer 2
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 2 — Persistence & device — storage, accounts, native bridges

**Packages:** accounts, native, storage

## What this layer owns

Durable state (the Database and its schema), account and credential custody, and the native bridge processes that talk to platforms.

## What this layer may not do

Know about missions, agents or chat. Storage serves queries; it never decides policy. Applied migrations are frozen — new persistence is a runtime-created table or a JSON store.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
