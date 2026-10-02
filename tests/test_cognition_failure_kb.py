"""Tests for nomorals.cognition.failure_kb — fully offline, tmp SQLite."""

from nomorals.cognition import FailureKB, TrajectoryStore
from nomorals.cognition.trajectories import cluster_key_for


def make_kb(tmp_path):
    store = TrajectoryStore(db=str(tmp_path / "cog.db"))
    return FailureKB(store=store), store


def record_failures(store, task_kind, error, n):
    for _ in range(n):
        store.record(task_kind=task_kind, capability="general", model_id="m1",
                     success=False, error=error)


def test_empty_kb_is_sane(tmp_path):
    kb, _store = make_kb(tmp_path)
    assert kb.lookup("chat") == []
    assert kb.repeated() == []


def test_note_and_lookup_roundtrip(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "TimeoutError: upstream timeout", 3)
    key = cluster_key_for("chat", "TimeoutError: upstream timeout")
    note_id = kb.note(key, "Retry with backoff; flaky provider.")
    assert note_id
    clusters = kb.lookup("chat")
    assert len(clusters) == 1
    c = clusters[0]
    assert c["cluster_key"] == key
    assert c["count"] == 3
    assert len(c["notes"]) == 1
    assert c["notes"][0]["note_text"] == "Retry with backoff; flaky provider."


def test_notes_order_newest_first(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "Boom", 3)
    key = cluster_key_for("chat", "Boom")
    kb.note(key, "first note")
    kb.note(key, "second note")
    notes = kb.lookup("chat")[0]["notes"]
    assert [n["note_text"] for n in notes] == ["second note", "first note"]


def test_repeated_surfaces_count_gte_3(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "FlakyError: x", 3)
    record_failures(store, "chat", "RareError: y", 2)
    repeated = kb.repeated()
    sigs = [c["error_signature"] for c in repeated]
    assert "FlakyError: x" in sigs
    assert "RareError: y" not in sigs


def test_repeated_respects_limit_and_order(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "ErrA", 6)
    record_failures(store, "code", "ErrB", 4)
    record_failures(store, "code", "ErrC", 3)
    out = kb.repeated(limit=2)
    assert [c["error_signature"] for c in out] == ["ErrA", "ErrB"]
    assert len(kb.repeated(limit=100)) == 3


def test_note_for_error_convenience(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "ValueError: bad 0xabc", 3)
    kb.note_for_error("chat", "ValueError: bad 0xdef", "normalize-safe")
    out = kb.repeated()
    assert out[0]["notes"][0]["note_text"] == "normalize-safe"


def test_repeated_includes_notes(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "ErrA", 3)
    key = cluster_key_for("chat", "ErrA")
    kb.note(key, "Known issue; tracking upstream fix.")
    out = kb.repeated()
    assert out[0]["notes"][0]["note_text"] == "Known issue; tracking upstream fix."


def test_lookup_scoped_to_task_kind(tmp_path):
    kb, store = make_kb(tmp_path)
    record_failures(store, "chat", "ErrA", 3)
    record_failures(store, "code", "ErrB", 3)
    assert [c["error_signature"] for c in kb.lookup("code")] == ["ErrB"]
    assert [c["error_signature"] for c in kb.lookup("chat")] == ["ErrA"]


def test_kb_creates_own_store_from_path(tmp_path):
    kb = FailureKB(db=str(tmp_path / "own.db"))
    assert kb.lookup("chat") == []
    kb.note("chat::whatever", "orphan note — no failure row needed")
    assert kb.repeated() == []
