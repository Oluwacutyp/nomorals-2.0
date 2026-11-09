---
id: layer-4-services
title: Layer 4 — Services — tools, social, media, skills, domains
summary: Instruction file for architectural layer 4 (services).
topics: layers, architecture, layer 4
triggers: layering, import rules, which layer
layer: architecture
version: 1
---

# Layer 4 — Services — tools, social, media, skills, domains

**Packages:** audio, browser, characters, commerce, community, datasci, documents, finance, goals, learn, legal, marketing, media, media_edit, plugins, property, research, scheduler, skills, social, tools, vision, voice

## What this layer owns

User-facing capability organs: the tool registry, chat/social surfaces and living cards, media and voice, the browser, documents, the skills system, and the domain services (finance, legal, property, marketing, learning).

## What this layer may not do

Import layer-5 application organs (agents, missions, partner, context). A service that needs orchestration receives it as an injected callable.

## The rule that enforces this

Imports may only point at the same layer or a lower one; `tests/test_layering.py` scans every import in the repo (including lazy, function-level imports) and fails on an upward edge. This file is generated from `nomorals/skills/layers.py:LAYER_GROUPS`, which a consistency test pins to the test's own layer map — the two cannot drift silently.
