---
id: context-budgeting
title: Context budgeting — the prompt is assembled, not accumulated
summary: How working context is built: prioritized sections, honest truncation, skills loaded on demand.
topics: context, prompts, budget, tokens
triggers: context, prompt, too long, token budget, assemble
layer: context
version: 1
---

# Context budgeting

- Context is assembled per task by the context engine
  (`nomorals.context`) from prioritized sections — system, mission,
  skills, artifacts, tools, project, profile, memory, history — and
  trimmed to budget by priority, never by chopping the tail off a
  string.
- Skills are part of that assembly: the router scores the task text
  against the skill index and the matching skill bodies load on demand
  (capped), labelled with their id — and drafts are labelled unproven.
- Truncation is honest: a section that was cut says so in the assembled
  output. Silent loss is how wrong-but-confident answers happen.
- The legacy step-prompt path is byte-stable: new sections join the
  rich path only, so old behaviour is comparable in tests.
