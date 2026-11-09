---
id: layer-7-entry-points
title: Layer 7 — Entry points — API, builders, CLI
summary: Instruction file for architectural layer 7 (entry-points).
topics: layers, architecture, layer 7
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 7 — Entry points — API, builders, CLI

**Packages:** api, builders, cmdline
**Top-level modules:** __main__, archives, bench, builders_proxy, cli, execbox, exporter, self_improvement

## What this layer owns

How the system is entered: the HTTP API, the builder harness, the `nm` command line, and the top-level single-file entry modules.

## What this layer may not do

Contain organ logic. An entry point parses, dispatches into the layers below, and renders the result — nothing more.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
