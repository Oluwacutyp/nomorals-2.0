---
id: failure-triage
title: Failure triage — evidence before verdicts
summary: How failures are diagnosed: read the real record, suspect the environment first, never guess a cause.
topics: debugging, failures, triage, incidents
triggers: failing, broken, error, not working, diagnose
layer: agents
version: 1
---

# Failure triage

- Read the actual record first: the journal rows, the log file, the
  screenshot as it literally is. Name the cause only from evidence in
  hand — a plausible story is not a diagnosis.
- Suspect the environment before the code: a full /tmp tmpfs, a run
  under heavy parallel load, a stale deployment and a poisoned test
  binding have all produced "bug" clusters here that vanished on a
  quiet, correct rerun. Failure victims that vary run to run are an
  environment smell, not a code smell.
- OFF and BROKEN must never be indistinguishable: check the feature's
  declared state AND its live evidence before saying either.
- A/B against the pristine base before owning a test failure — the
  baseline noise list is long and every entry on it was proven, not
  assumed.
- When a fix ships, the verdict waits for live proof on the real
  surface. "The code is correct" is not the close criterion.
