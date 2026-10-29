"""Tail-35 pins for nomorals/travel/whitelabel.py — 8 exclusions.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback). The file
itself contains no exception handlers, so it adds zero census sites.

whitelabel.py calls no record function anywhere (grep-proven: only
_log.debug), so NOT-recording is pinned with the tail-8 snapshot method:
recent_failures() before/after each trigger must be unchanged — a swallowed
failure leaves no trace anywhere. (The module's own debug log is not a
failure record.)

Site 293 (client_spend_report) is NOT pinned here: it is the batch's convert
(see tests/test_travel_whitelabel_spend_guard.py). The other 8 sites are
exclusions; see SURVEY.md for grounds.
"""

import os
import sqlite3
from pathlib import Path

import pytest

from nomorals.agents import failures as ag_fail
from nomorals.travel import whitelabel as W


def _snap():
    return list(ag_fail.recent_failures())


class RaisingDB:
    """Stand-in for store._db: every execute raises, call-logged."""

    def __init__(self, log):
        self.log = log

    def execute(self, *a, **k):
        self.log.append(a[0][:60] if a else "")
        raise sqlite3.OperationalError("boom")

    def commit(self):
        raise sqlite3.OperationalError("boom")


def test_102_double_init_absorbs_duplicate_column(tmp_path):
    db = tmp_path / "clients.db"
    W.TravelClientStore(db_path=str(db))
    before = _snap()
    s2 = W.TravelClientStore(db_path=str(db))  # 2nd init: ALTERs MUST fail
    assert _snap() == before
    # The pass's trigger, proven directly: the columns already exist.
    conn = sqlite3.connect(str(db))
    with pytest.raises(sqlite3.OperationalError) as ei:
        conn.execute("ALTER TABLE travel_clients ADD COLUMN channels_json TEXT")
    assert "duplicate column" in str(ei.value)
    # Control: the store is fully usable after the absorbed failure.
    c = s2.create("X", channels=["whatsapp", "telegram"],
                  languages=["en", "yo"])
    got = s2.get(c.id)
    assert got is not None and got.channels == ["whatsapp", "telegram"]
    assert got.languages == ["en", "yo"]


def test_169_get_broken_db_is_none(tmp_path):
    store = W.TravelClientStore(db_path=str(tmp_path / "c.db"))
    keep = store.create("Keep")
    log = []
    store._db = RaisingDB(log)
    before = _snap()
    assert store.get(keep.id) is None and len(log) == 1
    assert _snap() == before
    # Control: healthy store distinguishes found from missing.
    store2 = W.TravelClientStore(db_path=str(tmp_path / "c2.db"))
    k2 = store2.create("K2")
    assert store2.get(k2.id) is not None and store2.get("missing") is None


def test_181_list_broken_db_is_empty(tmp_path):
    store = W.TravelClientStore(db_path=str(tmp_path / "c.db"))
    store.create("A")
    log = []
    store._db = RaisingDB(log)
    before = _snap()
    assert store.list() == [] and len(log) == 1
    assert _snap() == before
    # Control: healthy store lists what it holds.
    store2 = W.TravelClientStore(db_path=str(tmp_path / "c2.db"))
    store2.create("A")
    store2.create("B")
    assert sorted(c.name for c in store2.list()) == ["A", "B"]


def test_192_deactivate_broken_db_is_false(tmp_path):
    store = W.TravelClientStore(db_path=str(tmp_path / "c.db"))
    keep = store.create("Keep")
    log = []
    store._db = RaisingDB(log)
    before = _snap()
    assert store.deactivate(keep.id) is False and len(log) == 1
    assert _snap() == before
    # Control: healthy store deactivates what exists, refuses the rest.
    store2 = W.TravelClientStore(db_path=str(tmp_path / "c2.db"))
    k2 = store2.create("K2")
    assert store2.deactivate(k2.id) is True
    assert store2.deactivate("missing") is False


def test_221_list_knowledge_corrupt_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_KNOWLEDGE_DIR", str(tmp_path))
    (tmp_path / "c9.docs.json").write_text("{garbage json", encoding="utf-8")
    before = _snap()
    assert W.list_knowledge("c9") == []
    assert _snap() == before
    # Control: valid registry reads; missing registry reads empty (fresh).
    import json
    (tmp_path / "c10.docs.json").write_text(
        json.dumps([{"title": "T", "chars": 5}]), encoding="utf-8")
    assert W.list_knowledge("c10") == [{"title": "T", "chars": 5}]
    assert W.list_knowledge("nobody") == []


def test_252_toc_write_failure_still_returns_true(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_KNOWLEDGE_DIR", str(tmp_path))
    os.makedirs(tmp_path / "c1.docs.json", exist_ok=True)  # write must fail
    before = _snap()
    assert W.add_knowledge("c1", "Beach Guide",
                           "Lagos beaches and suya spots.") is True
    assert _snap() == before
    # The injection itself succeeded: the doc grounds.
    res = W.answer("c1", "beaches")
    assert res["grounded"] is True and len(res["hits"]) > 0
    # ...while the table of contents never recorded it (stale listing).
    assert W.list_knowledge("c1") == []
    # Control: normal add is both injected AND listed.
    assert W.add_knowledge("c2", "Abuja Guide", "Aso Rock views.") is True
    assert any(d.get("title") == "Abuja Guide"
               for d in W.list_knowledge("c2"))


def test_267_answer_failure_is_ungrounded(tmp_path, monkeypatch):
    calls = []

    def boom_ck(client_id):
        calls.append(client_id)
        raise RuntimeError("index down")

    real_ck = W.client_knowledge
    monkeypatch.setattr(W, "_KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(W, "client_knowledge", boom_ck)
    before = _snap()
    assert W.answer("c1", "beaches") == {
        "client_id": "c1", "question": "beaches", "hits": [],
        "grounded": False} and calls == ["c1"]
    assert _snap() == before
    # Control: restore the real index; a healthy index grounds.
    monkeypatch.setattr(W, "client_knowledge", real_ck)
    assert W.add_knowledge("c2", "Abuja Guide", "Aso Rock views.") is True
    assert W.answer("c2", "Aso Rock")["grounded"] is True


def test_336_analytics_failure_is_no_analytics(tmp_path, monkeypatch):
    ctor_calls = []

    class BoomStore:
        def __init__(self, *a, **k):
            ctor_calls.append(1)
            raise RuntimeError("db down")

    monkeypatch.setattr(W, "TravelClientStore", BoomStore)
    before = _snap()
    assert W.client_analytics("clientX",
                              db_path=str(tmp_path / "x.db")) == (
        "no analytics for clientX.") and len(ctor_calls) == 1
    assert _snap() == before
    # Control: a fresh client gets the FULL zero report, never "no analytics".
    monkeypatch.undo()
    fresh = W.client_analytics("fresh", db_path=str(tmp_path / "ana.db"))
    assert fresh.startswith("📊 fresh") and "queries: 0" in fresh
    assert "no analytics" not in fresh
