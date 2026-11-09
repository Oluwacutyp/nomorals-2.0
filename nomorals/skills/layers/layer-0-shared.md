---
id: layer-0-shared
title: Layer 0 — Shared surface — importable by everyone, imports nothing
summary: Instruction file for architectural layer 0 (shared).
topics: layers, architecture, layer 0
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 0 — Shared surface — importable by everyone, imports nothing

**Packages:** compat, version

## What this layer owns

Compatibility shims and version identity: the tiny surface every layer may lean on without creating a dependency.

## What this layer may not do

Import any other nomorals package. If shared code needs a real dependency it belongs in core, not here.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
