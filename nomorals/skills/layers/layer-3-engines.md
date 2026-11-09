---
id: layer-3-engines
title: Layer 3 — Engines — memory, models, cognition, matching, security
summary: Instruction file for architectural layer 3 (engines).
topics: layers, architecture, layer 3
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 3 — Engines — memory, models, cognition, matching, security

**Packages:** cognition, i18n, llm, matching, memory, planning, security, ta, training

## What this layer owns

The deep engines: long-term memory, the model ladder and providers, cognition and planning, matching, technical analysis, training, and the security substrate (vault, capabilities, audit).

## What this layer may not do

Reach up into chat, missions or agents. An engine exposes capability; it never orchestrates a user surface.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
