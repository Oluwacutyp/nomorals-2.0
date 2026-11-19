# AEP corpus

Reviewed, original After Effects projects used to test EffectCraft's `.aep` and `.aepx` reader
and importer. The first batch (17 files, case map in
[`docs/aep/CASES.md`](../docs/aep/CASES.md)) is described in
[`docs/aep/2026-11-03-initial-corpus.md`](../docs/aep/2026-11-03-initial-corpus.md).

## Layout

- Put `.aep` and `.aepx` fixtures directly in this directory. The importer test harness uses
  this directory through `EFFECTCRAFT_AEP_FIXTURES` and expects its named files here.
- Put an attribution sidecar beside every fixture, for example `example.aep.attribution`.
- Keep reviewed contributor-created presets and starting projects in [`inputs/`](inputs/),
  with their own sidecars, asset-index entries and checksums.
- Keep required original media in [`media/`](media/), with sidecars and generation recipes.
- Keep generators and cleanup tools under [`tools/aep/`](../tools/aep/).
- Record the case map, inputs, application versions and validation in
  [`docs/aep/`](../docs/aep/).

Combined projects should identify each case composition and its expected values. Explain
which tests only need the project structure and which also require external footage, fonts
or reference renders. Missing-footage placeholders must be intentional and documented.

## Before adding a batch

1. Review the project and every embedded payload. Replace vendor ICC data with original
   synthetic data where needed, then independently scan the saved results across raw,
   base64 and AEPX hex representations.
2. Document how each input was made, including contributor-created presets and manual steps.
   Identify unavailable inputs instead of claiming a complete rebuild recipe. Configure the
   generation tools to use the reviewed files in `inputs/` and `media/`; document any separate
   working directories they need.
3. Add the sidecars, the root asset-index entries and the SHA-256 manifest entries together.
4. Run the importing app's tests with `EFFECTCRAFT_AEP_FIXTURES_REQUIRED=1`; record the source
   revision, commands, counts and any ignored cases.

A description-only profile stub may preserve a working-space name for metadata tests. It
does not preserve a colour transform or establish rendering equivalence. Describe that
limitation in the affected cases.

The initial AEP integration is being developed in
[EffectCraft #2558](https://github.com/storytold/effectcraft/pull/2558), with export in
[#2594](https://github.com/storytold/effectcraft/pull/2594). The first fixture contribution
should record the exact app revision it was tested against.
