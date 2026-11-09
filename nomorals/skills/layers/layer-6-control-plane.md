---
id: layer-6-control-plane
title: Layer 6 — Control plane — OS services and the terminal UI
summary: Instruction file for architectural layer 6 (control-plane).
topics: layers, architecture, layer 6
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 6 — Control plane — OS services and the terminal UI

**Packages:** os, tui

## What this layer owns

Machine-level services: process and mission state machines, update/restart machinery, and the terminal interface.

## What this layer may not do

Be imported by the layers it controls. Control flows down through explicit seams, never up through imports.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
