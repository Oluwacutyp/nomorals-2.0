# Suite runtime (measured, not guessed)

## Latest measured run

- **Date:** 2026-10-14
- **Command:** `python3 -u -m unittest discover -s tests -t .`
- **Result:** `Ran 13279 tests in 3524.894s` — **~59 minutes**,
  `FAILED (failures=56, errors=14, skipped=153)`
- **Tree:** T0.1–T0.3 + T0.4's `test_finance_commands` fixes, on the
  `main` branch of this clone. Full log:
  `~/workspace/suite-runs-t04/suite_t04b.log` (kept outside the repo).

The suite is **not green**: 70 failures/errors are pre-existing on this
tree and unrelated to T0.4 (scattered across routing, presence/typing,
decoder, exporter, and other areas; the two layering failures are the
226 known violations T1.5/T1.6 will remove, and the two docs failures
were fixed by the T0.3-gap commit). T0.4 records the runtime honestly
rather than claiming green.

## Stated budget

**90 minutes.** Enforced by `timeout-minutes: 90` on the CI suite step
(`.github/workflows/ci.yml`). A run exceeding 90 minutes fails the
budget — it is not merely slow. Measured 59 minutes on 2026-10-14
leaves ~50% headroom.

## Temp-disk requirement

The suite writes a lot of temp files (~175 MB during the measured run).
On machines where `/tmp` is a small tmpfs (this dev box: 512 MB), the
first full-suite attempt died from `/tmp` exhaustion with no summary
line (`database or disk is full`, log ending mid-line). Run with
`TMPDIR` pointed at a large disk:

```bash
mkdir -p ~/workspace/suite-runs-t04/tmpdir
TMPDIR=~/workspace/suite-runs-t04/tmpdir python3 -u -m unittest discover -s tests -t .
```

## What T0.4 actually found

The brief's assumed cause — "minutes-long ML training" in the finance
tests — did **not** reproduce in this tree: `tests/test_finance_commands.py`
runs in ~7 s. What was actually stale was the strategy registry: the
strategy zoo grew 9 → 16, so two tests asserting the old count failed.
T0.4 fixed the tests to derive from `list_strategies()` (set-based),
added the 7 missing strategy blurbs, and recorded this measured
runtime instead of inventing a "minutes" claim.
