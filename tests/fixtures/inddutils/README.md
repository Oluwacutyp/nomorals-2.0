# Indd-Utils fixtures

Real `.indd` files, downloaded from the locations upstream's own
`tests/fixtures/manifest.json` points at and SHA-256-checked against
that manifest. They are the fixtures the parity TOML and the tests
run against; a missing fixture is a failure, never a skip.

| File | Bytes | SHA-256 | Source | Licence |
| ---- | ----- | ------- | ------ | ------- |
| BlueSquare.indd | 757,760 | `5e4d1db2…e144e0` | Adobe XMP-Toolkit-SDK sample | BSD-3-Clause |
| Neddy.indd | 1,503,232 | `8f206d61…e3ed2` | openpreserve format-corpus | CC0-1.0 |
| minizine.indd | 1,507,328 | `189021fc…b8cd8` | lizdenys.com | CC0-1.0 |

(Full SHA-256 values are in upstream's manifest and in the download
log under `~/workspace/inddutils-verify/`.)

Upstream's repository policy is "never commit the files" — that is
their repo hygiene, not a licence bar: both licences permit
redistribution with attribution, which this file is. The files are
vendored here so the parity run works offline, with no network.

What they exercise, per the probe in `~/workspace/inddutils-verify/probe.py`:

- **BlueSquare** — big-endian v4.0, 182 database pages, 186 objects,
  0 document-listed stories (its one story-class object is the XML
  backing story), 3 pages (1 spread + 2 master), 15 colours, 4 font
  families, 36 languages, 3 styles.
- **Neddy** — big-endian v3.0, 364 database pages, 1401 objects, 32
  document-listed stories, 33 tracked-change records, 8 images, 12
  groups, 40 spline items.
- **minizine** — little-endian v20.3, 337 database pages, 330
  objects, 3 document-listed stories, 8 text frames.

The Python port's IDML packages for all three are byte-identical to
the packages the compiled upstream Rust converter emits
(`~/workspace/inddutils-verify/rust/`), SHA-256 equal whole-package.
