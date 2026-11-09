---
id: layer-5-application
title: Layer 5 — Application — agents, missions, partner, context
summary: Instruction file for architectural layer 5 (application).
topics: layers, architecture, layer 5
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 5 — Application — agents, missions, partner, context

**Packages:** agents, autonomy, backup, books, codews, connectors, console, context, games, health, hub, integrations, mesh, missions, partner, power, search, stream, sync, travel, triggers, wisdom, workspace

## What this layer owns

The acting system: the agent roster and delegation, missions and their lifecycle, the partner runtime that speaks in chat, context assembly, autonomy, games, search and the integration organs.

## What this layer may not do

Import the control plane (os/tui) or entry points. The partner runtime receives OS hooks as injected callables; it never reaches up for them.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
