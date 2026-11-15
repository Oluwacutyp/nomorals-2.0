"""Pass #217 pins: media/dj_desk.py — 1 exclusion, 0 converts.

- 2040: ``madmom_beat_track`` in-process import probe. madmom is
  tried in-process first (the fast rung); when it is not installed
  the ImportError is caught and the function falls through to the
  venv subprocess rung (``_madmom_venv_python`` + ``_MADMOM_DRIVER``).
  When neither rung is available the function raises RuntimeError
  with the named wall — never ImportError. The swallow is the
  correct behavior: a missing optional model is the designed
  fallback to the provisioned venv, and the function's contract
  already raises RuntimeError for "unavailable" on a non-error
  path.

media/dj_desk.py DOES call record_failure elsewhere
(dj_desk.detect_beat_grid, dj_desk.detect_downbeat), so
NOT-recording is pinned against the media ledger: media
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers call the real function with
madmom absent, never a swallow.
"""

import pytest

from nomorals.media import failures as media_fail
from nomorals.media.dj_desk import madmom_available, madmom_beat_track


def _snap():
    return list(media_fail.recent_failures(limit=200))


def _madmom_missing():
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        assert isinstance(exc, ImportError)
        return True
    return False


# ── L2040: a missing madmom falls through to the venv rung ─────────

def test_2040_missing_madmom_raises_runtime_error_not_import_error():
    # madmom is not installed on this box: the in-process import
    # raises ImportError, the function catches it, runs the venv
    # rung, and the venv rung fails with the named wall.
    if not _madmom_missing():
        pytest.skip("madmom is installed here; the probe cannot fail")
    before = _snap()
    with pytest.raises(RuntimeError, match="madmom beat tracking failed"):
        madmom_beat_track("/nonexistent/track.wav")
    assert _snap() == before


def test_2040_missing_madmom_is_reported_unavailable():
    # The availability probe agrees: madmom is not importable, and
    # the reason names the provision step instead of raising.
    if not _madmom_missing():
        pytest.skip("madmom is installed here; the probe cannot fail")
    ok, reason = madmom_available()
    assert ok is False
    assert reason  # a wall, not an empty string
