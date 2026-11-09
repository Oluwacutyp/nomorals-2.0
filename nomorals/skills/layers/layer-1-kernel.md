---
id: layer-1-kernel
title: Layer 1 — Kernel — errors, events, tasks, text, time
summary: Instruction file for architectural layer 1 (kernel).
topics: layers, architecture, layer 1
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 1 — Kernel — errors, events, tasks, text, time

**Packages:** core

## What this layer owns

The primitives everything else is built from: error types, the event bus, task kinds, text/time helpers, configuration shape.

## What this layer may not do

Import any other nomorals package — core has zero intra-project imports, enforced by test. No organ logic lives here; core stays a kernel, not a junk drawer.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
