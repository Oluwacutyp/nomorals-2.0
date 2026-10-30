"""Tail-60 pins for nomorals/property/cross_listing.py —
0 converts / 7 exclusions (66, 127, 197, 374, 411, 425, 440).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

cross_listing.py calls no failures-ledger record function
anywhere (grep-proven; its non-site _load/_save handlers log
via _log.debug instead), so NOT-recording is pinned against
the property package ledger (nomorals/property/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-60 SURVEY.md):
- 66: normalize_address failure -> ""; "" is the genuine
  no-address value and same_property/by_address treat it as
  no-match — the conservative direction. Pure in-memory op:
  a raise is a programming-defect class, not operational.
- 127: same_property failure -> False; False is the
  conservative no-merge answer, group_listings simply keeps
  the listings in separate groups. No false claim.
- 197: cross_listing_score failure -> the designed error
  dict (basis "error", verdict "could not score") — honest
  by construction, distinguishable from every real score.
- 374: store.create failure -> None, the genuine
  no-passport value (empty address returns None by design);
  nothing is stored, so None is literally true.
- 411: add_event failure -> False, the function's own
  honest failure signal (missing passport / empty kind
  return False by design); the trigger fires before any
  mutation, so False is literally true.
- 425: add_document failure -> False; same shape as 411.
- 440: add_viewing failure -> False; the float(when)
  conversion fires inside the Viewing construction, before
  the append, so a bad `when` adds nothing and False is
  literally true.
"""

from unittest import mock

import nomorals.property.cross_listing as CL
from nomorals.property.failures import recent_failures


def _snap():
    return list(recent_failures(200))


class _BadStr(str):
    """A str whose named method logs the attempt, then raises."""

    def __new__(cls, value, log, method):
        obj = super().__new__(cls, value)
        obj.log = log
        obj.method = method
        return obj

    def lower(self):
        if self.method == "lower":
            self.log.append("lower")
            raise RuntimeError("lower boom")
        return super().lower()

    def strip(self, *args):
        if self.method == "strip":
            self.log.append("strip")
            raise RuntimeError("strip boom")
        return super().strip(*args)


def test_66_normalize_address_failure_yields_empty():
    before = _snap()
    log = []
    out = CL.normalize_address(_BadStr("12 Allen Avenue", log, "lower"))
    assert log == ["lower"]  # the normalization was actually attempted
    assert out == ""
    assert CL.normalize_address("12, Allen Avenue, Ikeja") == \
        "12 allen avenue ikeja"  # control
    assert _snap() == before


def test_127_same_property_failure_yields_false():
    before = _snap()

    class BadPrice:
        def __init__(self):
            self.calls = 0

        def __float__(self):
            self.calls += 1
            raise TypeError("price boom")

    bp = BadPrice()
    a = CL.Listing(source="s1", address="12 Allen Avenue", price=bp)
    b = CL.Listing(source="s2", address="12 Allen Avenue", price=100.0)
    assert CL.same_property(a, b) is False
    assert bp.calls == 1  # the price comparison was actually attempted
    c = CL.Listing(source="s1", address="12 Allen Avenue", price=100.0)
    assert CL.same_property(c, b) is True  # control
    assert _snap() == before


def test_197_cross_listing_score_failure_yields_error_dict():
    before = _snap()

    class BadSource:
        def __init__(self):
            self.calls = 0

        def __str__(self):
            self.calls += 1
            raise RuntimeError("source boom")

    bs = BadSource()
    out = CL.cross_listing_score([CL.Listing(source=bs)], total_sources=3)
    assert bs.calls >= 1  # the source read was actually attempted
    assert out == {"sources": 0, "source_names": [], "listings": 0,
                   "total_sources": 0, "score": 0.0, "basis": "error",
                   "verdict": "could not score"}
    good = CL.cross_listing_score(
        [CL.Listing(source="a"), CL.Listing(source="b")], total_sources=3)
    assert good["sources"] == 2 and good["basis"] == "known-universe"
    assert _snap() == before


def test_374_create_failure_yields_none(tmp_path):
    before = _snap()
    store = CL.PropertyPassportStore(str(tmp_path / "pp.json"))
    ulog = []

    def bad_uuid():
        ulog.append(1)
        raise RuntimeError("uuid boom")

    with mock.patch.object(CL.uuid, "uuid4", bad_uuid):
        out = store.create("12 Allen Avenue")
    assert ulog == [1]  # passport construction was actually attempted
    assert out is None and store.list() == []
    ok = store.create("12 Allen Avenue")  # control
    assert ok is not None and ok.version == 1
    assert _snap() == before


def _store_with_passport(tmp_path):
    store = CL.PropertyPassportStore(str(tmp_path / "pp.json"))
    pp = store.create("5 Broad Street")
    assert pp is not None
    return store, pp


def test_411_add_event_failure_yields_false(tmp_path):
    before = _snap()
    store, pp = _store_with_passport(tmp_path)
    log = []
    out = store.add_event(pp.passport_id, _BadStr("listed", log, "strip"))
    assert log  # the kind normalization was actually attempted
    assert out is False and len(pp.history) == 0
    assert store.add_event(pp.passport_id, "listed", "seen") is True
    assert len(pp.history) == 1  # control
    assert _snap() == before


def test_425_add_document_failure_yields_false(tmp_path):
    before = _snap()
    store, pp = _store_with_passport(tmp_path)
    log = []
    out = store.add_document(pp.passport_id,
                             _BadStr("vault:1", log, "strip"))
    assert log  # the ref normalization was actually attempted
    assert out is False and len(pp.documents) == 0
    assert store.add_document(pp.passport_id, "vault:1", "Survey") is True
    assert len(pp.documents) == 1  # control
    assert _snap() == before


def test_440_add_viewing_failure_yields_false(tmp_path):
    before = _snap()
    store, pp = _store_with_passport(tmp_path)
    # float("not-a-time") raises inside the Viewing construction,
    # before the append — the conversion IS the attempted operation.
    out = store.add_viewing(pp.passport_id, "not-a-time")
    assert out is False and len(pp.viewings) == 0
    assert store.add_viewing(pp.passport_id, 123.0,
                             outcome="scheduled") is True
    assert len(pp.viewings) == 1  # control
    assert _snap() == before
