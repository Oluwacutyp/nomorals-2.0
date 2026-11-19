# AEP authoring inputs

Keep reviewed, original inputs used to generate the AEP fixtures here, including
contributor-created animation presets and starting projects. The first batch holds three presets (`curves-red-mid-down.ffx`, `curves-rgb-two-points.ffx`,
`meshwarp-3x3-center-right.ffx`), used by cases 963, 964 and 985; see
[`tools/aep/GENERATING.md`](../../tools/aep/GENERATING.md).

Finished test projects belong directly in [`aep/`](../), and supporting images, audio and
sequences belong in [`aep/media/`](../media/). This directory is for reusable authoring
inputs, not temporary generation output.

For each input:

- Add an adjacent `.attribution` sidecar, a row in the root
  [asset index](../../ATTRIBUTION.md), and a hash in [`SHA256SUMS`](../../SHA256SUMS).
- Document its creator, authoring application and version, creation steps, and the cases that
  use it. For a contributor-created preset, explain the settings saved in it and how it is
  applied; do not include vendor presets or other bundled application assets.
- Inspect and clean embedded payloads and identifying metadata before publication, as for
  the finished fixtures. Review the exact input bytes covered by the manifest.
- Link the generation instructions in [`docs/aep/`](../../docs/aep/) and make the tools'
  input paths configurable. Identify unavailable inputs explicitly.

Submit inputs with the fixture batch that uses them, through a pull request from a fork.
