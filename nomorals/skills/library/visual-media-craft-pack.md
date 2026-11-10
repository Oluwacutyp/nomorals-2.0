---
id: visual-media-craft-pack
title: Visual/media craft pack — edit graphs, rights and provenance
summary: The visual/media cooperative in pack form: a media job as a contract, the edit graph as the work, the rights manifest and validation recorded in the deliverable's provenance.
topics: media, craft, editing, provenance, images
triggers: media craft pack, visual media, edit graph, rights manifest
layer: craft
kind: pack
domain: interests
fixtures: tests/test_media_edit.py, tests/test_d18p1_durable_records.py, tests/test_d2223_stage_b.py
version: 1
---

# Visual/media craft pack

This pack describes how a cooperative cell makes visual media in
Devon: not as a generator call and a shrug, but as a media job on a
contract — a declared edit graph, executed by the shipped media
substrate, landing as a versioned artifact that carries where it
came from and what may be done with it.

## Terminology

- **Edit graph** — the ordered op chain that produced an image:
  each step names its operation and parameters. A deliverable
  without its graph is an orphan — nobody can say what was done
  to it or redo it.
- **Rights manifest** — the deliverable's ownership record: the
  source, who holds the rights, the licence the work is under,
  and any third-party material it contains. Recorded in the
  artifact's provenance metadata, additively — the artifact
  schema does not change to carry it.
- **Job record** — the durable record of the media job that ran
  (the doc-18 media job records): queued, running, done, failed —
  the run happened in the world, not just in the chat.
- **Validation** — the read-back: the output probes as the image
  the edit graph implies (format, dimensions, bytes) before
  anyone calls it delivered.

## Workflow

1. Take the source in as an artifact with its provenance — the
   source is a record before it is an input.
2. Declare the edit graph up front and validate it against the
   shipped op allowlist; an op the substrate does not have is a
   named refusal, not an improvisation.
3. Open the contract: editor, skeptic, verifier, the owner as
   arbiter; required evidence is the job record and the
   validation.
4. Run the job through the media job manager so its record is
   durable; the job's result names the output file it wrote.
5. Land the output as an artifact derived from the source, with
   the rights manifest and the full edit graph in its provenance
   metadata.
6. The verifier re-reads the artifact and re-probes the output;
   the skeptic's standing challenge — does the deliverable match
   the declared graph and carry its rights? — resolves on that
   read-back, or the run fails in the open.

## Tools and formats

- The media job manager and its durable records; the run-ticket
  projection reads the same records.
- The Pillow edit chain (`media_edit.images`): resize, crop,
  annotate, watermark, stack, grid and the rest of the allowlist —
  the deterministic rung that runs anywhere.
- Artifacts with provenance and supersede lineage; the rights
  manifest and edit graph ride the metadata, never a forked
  schema.
- Generation lanes (SDXL/FLUX/ComfyUI) where the profile carries
  them — a heavier rung of the same craft, not a different one.

## Quality rubric

- The job record stands done in the durable store — a run that
  only happened in memory is not a delivery.
- The output validates: it probes as the format and dimensions
  the edit graph implies.
- The rights manifest is recorded on the deliverable: source,
  rights holder, licence; third-party material named, never
  silently absorbed.
- The edit graph recorded on the deliverable matches the graph
  that ran, step for step.
- Independence holds: the editor never verifies their own edit.

## Anti-patterns

- **Untracked output** — an edited file with no job record, no
  artifact, no provenance: unrepeatable and unaccountable.
- **Rights by assumption** — delivering an image whose source
  and licence nobody recorded.
- **Graph drift** — the declared edit graph says one chain, the
  run executes another, and the record keeps the declaration.
- **Lane bluffing** — claiming a generation lane ran when the
  deterministic rung did the work; the run names the rung it
  used, and the lanes it did not use are absent-with-reason.

## Exemplars

- The Stage B media pilot: a durable job runs a three-op edit
  graph, the output derives from the source artifact with rights
  and graph in its metadata, and the receipt chain renders the
  run beside the other cells' receipts in one shape.
- The mix delivery receipts: a rendered mix carries its
  fingerprint and send record — the audio side of the same
  provenance discipline.

## Evaluators

The pack's fixtures are its regression scenarios: the media-edit
suite (the op chain, the plan, the job manager), the durable
records suite (media job records surviving as records), and the
Stage B pilot suite (the end-to-end run: job, validation,
provenance, receipt). A fixture that fails is a defect in the
work or in the pack — recorded either way, and the pack is
patched only through the draft → prove → promote lifecycle.

## Failure modes

- **Dead job** — the job record stands failed with its error;
  the run reports the failure, and no deliverable is derived
  from a file that was never written.
- **Corrupt source** — the probe refuses the input before any
  op runs; the refusal names the file.
- **Metadata mismatch** — the stored artifact's edit graph or
  rights differ from the declaration; the skeptic contradicts
  the claim and the contract cannot complete over it.

## Locale variants

One locale ships with this pack (English, Nigerian operating
context). Captions, annotations and typography follow the
typography service's locale handling where a job carries text —
a variant changes the words, never the provenance discipline.

## Credential constraints

This pack's deliverables are media artifacts produced inside the
owner's own systems, with their sources and licences recorded.
Nothing the pack produces transfers or invents rights in
third-party material: the manifest states what the record shows
— where a source's licence is unknown, the manifest says
unknown, and the deliverable is the owner's to judge.
