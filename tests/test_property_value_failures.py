"""Pass tail-6 pins: property/value failures stay honest, ledger intact.

Target: nomorals/property/value.py — 0 converts / 11 exclusions.
Every site below maps a failure to a genuine degraded value (the
documented miss/unknown value of that function) and the stack proceeds
honestly — estimates inform, humans commit; this module never triggers
a money move, so no failure here can drop money or a commitment.
None of them records, because none of them hides a user-visible
outcome: the degraded value itself (zero range, empty comps, False,
None, []) is what the consumer sees, and every consumer handles it
on an already-honest path (control_value's "no comparables" message,
pick_comps' "couldn't refine", trend's "need at least two").
Ledger REUSED: nomorals/property/failures.py (same package —
read-only for pins; no record call, no import added to the target,
which contains no record_failure call at all).

Each test pins the degraded behavior AND the NOT-recording property
(ledger empty after the trigger). No bare ``except: pass`` anywhere
in this file — the scanner counts test files too; the tests below
need no except blocks at all (stubs raise, the code under test
catches).
"""

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from nomorals.property import failures as pf
from nomorals.property import value as V
from nomorals.property.value import (
    ValueEstimate,
    ValueStore,
    _default_db,
    _estimate_from_comps,
    _parse_desc,
    estimate_value,
    pick_comps,
    seed_comp_source,
)


def _silent():
    assert pf.recent_failures() == []


def test_147_confidence_corrupt_fsd_is_zero():
    # Fail-closed: an uncomputable confidence claims 0%, never a
    # fabricated score. Trigger needs post-construction corruption:
    # __init__ itself coerces fsd through float().
    est = ValueEstimate(point_kobo=1000, low_kobo=900, high_kobo=1100,
                        fsd=0.12)
    est.fsd = "bad"
    assert est.confidence == 0
    _silent()


def test_209_parse_desc_unparseable_is_unknown():
    # Unknown area/bedrooms, not a guessed area: the estimate path
    # then yields no comps and control_value says so honestly.
    assert _parse_desc(123) == ("", "")
    assert _parse_desc(None) == ("", "")
    _silent()


def test_240_seed_source_corrupt_norms_is_empty():
    # Corrupt norm data degrades to no comps, never invented comps.
    with mock.patch.object(V, "AREA_NORMS", None):
        assert seed_comp_source("surulere", "2br") == []
    _silent()


def test_272_estimate_foreign_comps_is_zero_thin():
    # Foreign comp objects degrade to the documented zero/thin tuple
    # — the same tuple the legitimate empty-comps case returns.
    foreign = [SimpleNamespace(price_kobo="100", similarity=0.9,
                               source_trust=1.0)]
    assert _estimate_from_comps(foreign) == (0, 0, 0, 0.0, True)
    _silent()


def test_280_default_db_first_join_failure_is_fallback():
    # The fallback path only fires when the first join fails but the
    # second succeeds; the wrapper proves both calls ran.
    real_join = os.path.join
    calls = []

    def flaky_join(*args):
        calls.append(args)
        if len(args) >= 4 and "property" in args:
            raise OSError("join boom")
        return real_join(*args)

    with mock.patch("os.path.join", side_effect=flaky_join):
        out = _default_db()
    assert out == real_join(os.path.expanduser("~"), ".nomorals_values.db")
    assert any(len(c) >= 4 for c in calls)
    assert any(len(c) == 2 for c in calls)
    _silent()


def _db_down_store():
    with mock.patch("sqlite3.connect", side_effect=OSError("db boom")):
        store = ValueStore(os.path.join(tempfile.mkdtemp(), "v.db"))
    assert store._db is None
    return store


def test_321_save_db_down_is_false():
    assert _db_down_store().save(ValueEstimate()) is False
    _silent()


def test_343_get_db_down_is_none():
    assert _db_down_store().get("nope") is None
    _silent()


def test_376_history_db_down_is_empty():
    assert _db_down_store().history() == []
    _silent()


def test_401_trend_read_failure_is_honest_dict():
    # Not the "need at least two" path: history itself raises, so the
    # trend handler's own dict is what ships.
    store = ValueStore(os.path.join(tempfile.mkdtemp(), "v.db"))
    with mock.patch.object(ValueStore, "history",
                           side_effect=OSError("read boom")):
        assert store.trend() == {"ok": False,
                                 "reason": "couldn't compute trend"}
    _silent()


class _RaisingSaveStore:
    def save(self, est):
        raise OSError("save boom")


def test_428_estimate_survives_save_failure():
    # Best-effort persistence: the estimate itself is genuine and is
    # returned regardless; nothing claims the save succeeded (its
    # bool is ignored on every path, success included).
    est = estimate_value("2br flat in Surulere",
                         store=_RaisingSaveStore())
    assert est.point_kobo > 0 and len(est.comps) == 6
    _silent()


class _GetOkSaveBoomStore:
    def __init__(self, est):
        self._est = est

    def get(self, estimate_id):
        return self._est

    def save(self, est):
        raise OSError("save boom")


def test_466_pick_survives_save_failure():
    # Same best-effort persistence: the refinement is returned; a
    # later lookup miss surfaces honestly via pick_comps' None path.
    base = estimate_value("2br flat in Surulere")
    assert base.comps
    refined = pick_comps(base.id, ["1", "2"],
                         store=_GetOkSaveBoomStore(base))
    assert refined is not None and len(refined.comps) == 2
    _silent()
