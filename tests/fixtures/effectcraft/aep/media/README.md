# Supporting media

Keep original media needed by the AEP fixtures here. This directory is empty in
the first batch: the footage links of the projects name media that are not included, and the
tests expect those footage items to be missing (documented placeholders, see
[`docs/aep/2026-11-03-initial-corpus.md`](../../docs/aep/2026-11-03-initial-corpus.md)).

For each file, include its generator or authoring recipe, an adjacent `.attribution` sidecar,
an entry in the root [asset index](../../ATTRIBUTION.md), and its hash in
[`SHA256SUMS`](../../SHA256SUMS). Record how project paths are resolved or relinked. Prefer
small generated images, audio and sequences over large clips.

Inspect embedded profiles and metadata in the media itself as well as in the project that
references it. Reference system fonts by name and document substitutions; do not copy fonts
or bundled application media into this directory.
