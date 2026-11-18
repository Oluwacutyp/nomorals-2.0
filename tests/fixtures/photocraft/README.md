# PhotoCraft corpus fixtures (real PSDs)

Real Adobe Photoshop files from the PhotoCraft corpus
(`storytold/photocraft-corpus`, dual MIT OR Apache-2.0 —
"Copyright (c) 2026 ArtCraft Team and the PhotoCraft
contributors"; see `LICENSE-MIT` / `LICENSE-APACHE` in that
repo). The corpus README states: "PSDs made in Adobe Photoshop
by our own scripts. Each file stores Photoshop's rendering of
itself" — every file carries its own merged composite, which is
Photoshop's rendering of its layer stack, so the oracle can check
Devon's flatten against the vendor's rendering of the same file.

- `rgb8-baseline-layer-copy.psd` (RGB, 8-bit, 2 layers) — the
  flatten-exactness oracle: no adjustment layers, so Devon's
  normal-blend flatten must match the composite pixel for pixel.
- `rgb8-brightness-contrast.psd` (RGB, 8-bit) — carries a
  brightness/contrast adjustment layer: parsed and reported, its
  effect not yet rendered (a named remainder for a later lane).
- `cmyk8-baseline-layer-copy.psd` (CMYK, 8-bit, 2 layers) —
  mode-conversion oracle.
- `gray8-baseline-layer-copy.psd` (grayscale, 8-bit, 2 layers) —
  single-channel oracle.

The full 258-file corpus lives at
`~/workspace/echelon-mining/repos/storytold/photocraft-corpus/`
and `nomorals.media.photo_editor.oracle.run_oracle` runs over it
when present (env `NM_PHOTOCRAFT_CORPUS` overrides the path).
