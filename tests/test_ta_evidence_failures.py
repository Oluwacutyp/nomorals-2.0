"""Pass tail-118 pins — nomorals/ta/evidence.py (3 exclusions).

Sites pinned (all E104 "Never raises" read/verdict degradations). The
store's contract is printed on every sibling: summaries report absence
(``decayed_win_rate: None``), never an invented number; both record
entry doors (``record()`` and ``from_dict()``) float-coerce timestamps,
so the crash classes here need an out-of-band record in ``_records``.

- 222: ``list()`` crash (one record with a non-comparable timestamp)
  -> ``[]``; downstream summaries take the documented no-evidence
  shape. SILENT — unlike _load/_save/record/monitor, this handler
  records nothing (pinned via caplog).
- 248: ``weighted_results()`` crash (a record whose ``decay()`` raises)
  -> the documented empty summary ``{weights: {}, win_rate: None,
  records: 0}``.
- 301: ``retire()`` crash (``_save`` raises after the in-memory delete)
  -> False, the refusal shape; the durable record on disk is unharmed
  and a fresh store still reads it (pinned).
"""

from __future__ import annotations

import logging

import pytest

from nomorals.ta.evidence import EvidenceRecord, EvidenceStore


def _store(tmp_path):
    store = EvidenceStore(str(tmp_path / "ev.json"))
    win = store.record("qm", "H1", result="win")
    loss = store.record("qm", "H1", result="loss")
    assert win is not None and loss is not None
    return store, win, loss


def test_healthy_controls(tmp_path):
    store, win, _loss = _store(tmp_path)
    assert len(store.list()) == 2
    wr = store.weighted_results("qm")
    assert wr["records"] == 2 and wr["decayed_win_rate"] == 0.5
    assert store.retire(win.record_id) is True
    assert store.get(win.record_id) is None


def test_222_bad_record_empties_list_silently(tmp_path, caplog):
    store, _win, _loss = _store(tmp_path)
    bad = EvidenceRecord(setup="qm", timeframe="H1", result="win",
                         timestamp="not-a-float", record_id="ev_bad")
    store._records[bad.record_id] = bad
    with caplog.at_level(logging.DEBUG, logger="nomorals.ta.evidence"):
        assert store.list() == []
    # non-recording: the 222 handler emits nothing (siblings _log.debug)
    assert [r for r in caplog.records
            if r.name == "nomorals.ta.evidence"] == []
    wr = store.weighted_results("qm")
    assert wr["records"] == 0 and wr["decayed_win_rate"] is None


class _ExplodingDecay:
    setup = "qm"
    result = "win"
    timestamp = 1.0

    def decay(self, **k):
        raise RuntimeError("decay boom")


def test_248_decay_crash_returns_empty_summary(tmp_path):
    store = EvidenceStore(str(tmp_path / "ev2.json"))
    store._records["x"] = _ExplodingDecay()
    assert store.weighted_results("qm") == {
        "setup": "qm", "weights": {}, "decayed_win_rate": None,
        "records": 0}


def test_301_save_crash_retire_false_disk_intact(tmp_path, monkeypatch):
    store, _win, loss = _store(tmp_path)

    def save_boom(self):
        raise RuntimeError("save boom")

    monkeypatch.setattr(EvidenceStore, "_save", save_boom)
    assert store.retire(loss.record_id) is False
    monkeypatch.undo()
    fresh = EvidenceStore(str(tmp_path / "ev.json"))
    assert fresh.get(loss.record_id) is not None
