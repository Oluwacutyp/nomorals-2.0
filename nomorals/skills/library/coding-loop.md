---
id: coding-loop
title: The coding loop — inspect, change, prove
summary: How code changes are made in this repo: read first, smallest real change, tests as proof.
topics: coding, editing, tests, verification
triggers: write code, fix bug, implement, refactor, patch
layer: agents
version: 1
---

# The coding loop

1. **Read the real code first.** Trace the path end to end — trigger →
   handler → organ → storage → reply — before touching anything. Repo
   truth beats memory and tutorials.
2. **Confirm signatures and registries.** Tools, commands and imports
   are discovered from source, never invented. A new chat command is
   registered in the control catalog with help text, or it does not
   exist.
3. **Smallest real change that is fully capable.** No stubs, no
   `NotImplementedError`, no TODO-as-delivery, no placeholder returns.
   Never delete a class, module, agent or tool — consolidate helpers,
   tighten wiring, extend the real organ.
4. **Respect the layers.** `tests/test_layering.py` is the enforcement;
   the layer instruction files beside it are the readable law. New
   persistence is a runtime-created table or a JSON store —
   `storage/migrations.py` is frozen and never edited.
5. **Prove it.** Run the new tests plus the adjoining suites and the
   layering test. Judge results from the log files on disk, and A/B any
   failure against the pristine base before calling it yours.
