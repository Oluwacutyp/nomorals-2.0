"""Tail-53 pins for nomorals/travel/itinerary.py —
2 converts / 6 exclusions (495, 504 converted by amendment).

9 tests: 8 site tests + 1 runtime-surface test for the
companion guards in runtime_memory._control_trip. Each
test pins one site with a control half proving the guarded
path was actually exercised, asserting via the operation's
own call log / double that the failing operation itself
was attempted. The file contains no exception handlers
(raises are asserted via pytest.raises), so it adds zero
census sites.

itinerary.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the travel package ledger (nomorals/travel/failures.py):
recent_failures() / failure_count() snapshot before/after
every trigger.

Verdict notes (full grounds in the tail-53 SURVEY.md):
- 495: CONVERTED (parent ruling, tail-51 precedent) —
  get_trip read failure raises ItineraryStoreError;
  summary/timeline/packing return the unavailable string,
  not "no such trip."; missing id still returns None.
- 504: CONVERTED — list_trips failure returns None (not
  []); find_merge_candidate returns None ("cannot tell");
  genuine empty store still returns [].
- 563: delete_trip failure -> False, literally true (the
  trip was not deleted); also the genuine missing-id value.
- 842: _pdf_text pypdf ImportError -> pass falls THROUGH
  to the pdftotext fallback; not a terminal swallow.
- 854: pdftotext failure -> "", the genuine no-text value
  at the end of the fail-closed two-tool chain.
- 874: one undecodable Gmail part is skipped under the
  documented best-effort extraction; sibling parts survive.
- 883: malformed Gmail message dict -> "", the same value
  a part-less, snippet-less message yields healthily.
- 892: subject walk failure -> "", the genuine
  no-subject value.
"""

import base64
import subprocess
import sys
import types

import pytest

from nomorals.travel import failures as tf
from nomorals.travel import itinerary as I


def _snap():
    return (tf.failure_count(),
            [(f.site, f.exc_type, f.message)
             for f in tf.recent_failures(200)])


def _builder(tmp_path, tag):
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
    return I.ItineraryBuilder(db_path=str(d / "t.db"),
                              vault_dir=str(d / "vault"))


class _ExplodingDB:
    """DB double: execute() logs the SQL and raises."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params=()):
        self.calls.append(sql)
        raise RuntimeError("db boom")


def test_495_get_trip_failure_raises_store_error(tmp_path):
    b = _builder(tmp_path, "p495")
    b._save(I.Trip(id="trip_seed495", name="Seeded495"))
    assert b.get_trip("trip_seed495").name == "Seeded495"  # control
    assert b.get_trip("nope") is None  # genuine missing -> None
    assert b.summary("nope") == "no such trip."  # genuine missing
    ex = _ExplodingDB()
    b._db = ex
    before = _snap()
    with pytest.raises(I.ItineraryStoreError):
        b.get_trip("trip_seed495")
    assert ex.calls and "SELECT data_json FROM trips" in ex.calls[0]
    unavailable = ("trip store unavailable right now — "
                   "try again in a moment.")
    assert b.summary("trip_seed495") == unavailable
    assert b.timeline("trip_seed495") == unavailable
    assert b.packing("trip_seed495") == unavailable
    assert b.summary("trip_seed495") != "no such trip."
    # Propagating wrappers: no []/""-on-failure recreated.
    with pytest.raises(I.ItineraryStoreError):
        b.vault_docs("trip_seed495")
    with pytest.raises(I.ItineraryStoreError):
        b.conflicts("trip_seed495")
    with pytest.raises(I.ItineraryStoreError):
        b.export_ics("trip_seed495")
    assert _snap() == before


def test_504_list_trips_failure_returns_none(tmp_path):
    b = _builder(tmp_path, "p504")
    empty = _builder(tmp_path, "p504empty")
    assert empty.list_trips() == []  # genuine empty -> []
    b._save(I.Trip(id="trip_seed504", name="Seeded504",
                   flights=[I.Flight(flight_number="BA075",
                                     pnr="ABC123")]))
    assert [t.id for t in b.list_trips()] == ["trip_seed504"]  # control
    assert b.find_merge_candidate({"pnr": "ABC123"}) is not None
    ex = _ExplodingDB()
    b._db = ex
    before = _snap()
    assert b.list_trips() is None  # failure -> None, not []
    assert ex.calls and "SELECT id, data_json FROM trips" in ex.calls[0]
    # "Cannot tell": no candidate claimed on a failed read.
    assert b.find_merge_candidate({"pnr": "ABC123"}) is None
    assert _snap() == before


def test_495_504_runtime_control_trip_unavailable(tmp_path):
    from nomorals.agents.partner.runtime_memory import (
        RuntimeMemoryMixin)
    b = _builder(tmp_path, "prt")
    b._save(I.Trip(id="trip_seedrt", name="SeededRT"))
    ex = _ExplodingDB()
    b._db = ex
    mixin = RuntimeMemoryMixin()
    mixin._trip_builder = b
    mixin._gcal_instance = object()
    before = _snap()
    unavailable_frag = "trips unavailable right now"
    assert unavailable_frag in mixin._control_trip("")
    assert unavailable_frag in mixin._control_trip("docs trip_seedrt")
    assert unavailable_frag in mixin._control_trip(
        "calendar trip_seedrt")
    assert unavailable_frag in mixin._control_trip("trip_seedrt")
    assert _snap() == before


def test_563_delete_trip_failure_returns_false(tmp_path):
    b = _builder(tmp_path, "p563")
    b._save(I.Trip(id="trip_seed563", name="Seeded563"))
    real_db = b._db
    ex = _ExplodingDB()
    b._db = ex
    before = _snap()
    assert b.delete_trip("trip_seed563") is False
    assert ex.calls and "DELETE FROM trips" in ex.calls[0]
    assert _snap() == before
    b._db = real_db
    assert b.get_trip("trip_seed563") is not None  # really not deleted
    assert b.delete_trip("trip_seed563") is True  # control
    assert b.delete_trip("trip_seed563") is False  # genuine missing


def test_842_pdf_text_importerror_falls_through_to_pdftotext(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    run_calls = []

    def fake_run(args, **kw):
        run_calls.append(args)
        return types.SimpleNamespace(
            returncode=0, stdout="FALLBACK TEXT", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    before = _snap()
    out = I.ItineraryBuilder._pdf_text("/nonexistent/x.pdf")
    assert run_calls and run_calls[0][0] == "pdftotext"  # fell through
    assert out == "FALLBACK TEXT"
    assert _snap() == before
    fake = types.ModuleType("pypdf")

    class FakeReader:
        def __init__(self, path):
            self.pages = [types.SimpleNamespace(
                extract_text=lambda: "PAGE TEXT")]

    fake.PdfReader = FakeReader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    assert I.ItineraryBuilder._pdf_text("/whatever.pdf") == "PAGE TEXT"


def test_854_pdf_text_pdftotext_failure_returns_empty(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    run_calls = []

    def fake_run_boom(args, **kw):
        run_calls.append(args)
        raise FileNotFoundError("no pdftotext binary")

    monkeypatch.setattr(subprocess, "run", fake_run_boom)
    before = _snap()
    assert I.ItineraryBuilder._pdf_text("/nonexistent/x.pdf") == ""
    assert run_calls and run_calls[0] == [
        "pdftotext", "/nonexistent/x.pdf", "-"]
    assert _snap() == before

    def fake_run_ok(args, **kw):
        return types.SimpleNamespace(
            returncode=0, stdout="PDF TEXT", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    assert I.ItineraryBuilder._pdf_text("/x.pdf") == "PDF TEXT"  # control


def test_874_gmail_body_bad_part_skipped(tmp_path, monkeypatch):
    good = base64.urlsafe_b64encode(b"hello body").decode()
    full = {"payload": {"mimeType": "multipart/mixed", "parts": [
        {"mimeType": "text/plain", "body": {"data": "BADPART"}},
        {"mimeType": "text/plain", "body": {"data": good}},
    ]}}
    dec_calls = []
    real_dec = base64.urlsafe_b64decode

    def dec_spy(data, *a, **k):
        dec_calls.append(data)
        if data.startswith("BADPART"):
            raise RuntimeError("decode boom")
        return real_dec(data, *a, **k)

    monkeypatch.setattr(base64, "urlsafe_b64decode", dec_spy)
    before = _snap()
    assert I.ItineraryBuilder._gmail_body(full) == "hello body"
    assert any(c.startswith("BADPART") for c in dec_calls)  # attempted
    assert _snap() == before
    monkeypatch.undo()
    full2 = {"payload": {"parts": [
        {"mimeType": "text/plain", "body": {"data": good}},
        {"mimeType": "text/plain",
         "body": {"data": base64.urlsafe_b64encode(b"second").decode()}},
    ]}}
    assert I.ItineraryBuilder._gmail_body(full2) == "hello body\nsecond"


def test_883_gmail_body_malformed_message_returns_empty(tmp_path):
    class FullBoom:
        def __init__(self):
            self.calls = []

        def get(self, *a, **k):
            self.calls.append(a)
            raise RuntimeError("payload boom")

    fb = FullBoom()
    before = _snap()
    assert I.ItineraryBuilder._gmail_body(fb) == ""
    assert fb.calls and fb.calls[0][0] == "payload"  # attempted
    assert _snap() == before
    assert I.ItineraryBuilder._gmail_body(
        {"payload": {}, "snippet": "SNIP"}) == "SNIP"  # control


def test_892_gmail_subject_failure_returns_empty(tmp_path):
    class HeaderBoom:
        def __init__(self):
            self.calls = []

        def get(self, *a, **k):
            self.calls.append(a)
            raise RuntimeError("header boom")

    hb = HeaderBoom()
    before = _snap()
    assert I.ItineraryBuilder._gmail_subject(
        {"payload": {"headers": [hb]}}) == ""
    assert hb.calls and hb.calls[0][0] == "name"  # attempted
    assert _snap() == before
    assert I.ItineraryBuilder._gmail_subject(
        {"payload": {"headers": [
            {"name": "Subject", "value": "Your trip"}]}}) == "Your trip"
    assert I.ItineraryBuilder._gmail_subject(
        {"payload": {"headers": [
            {"name": "From", "value": "x@y.z"}]}}) == ""  # genuine
