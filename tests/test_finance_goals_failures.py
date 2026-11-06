"""Tail-113 pins: nomorals/finance/goals.py census sites (93, 115, 123).

All three are held as exclusions:

93 — ``GoalStore._load`` typed guard (OSError, ValueError) -> ``{}``.
     Was a TARGETED-FIX CANDIDATE (data-loss shelf, top tier):
     add/remove/mark_done/snooze/unsnooze are all load-then-rewrite,
     so one corrupt byte made the next mutation overwrite the whole
     persisted goal registry with a single-goal dict (byte-proven
     below). CONVERTED by the data-loss targeted fix (Pass 1):
     "unreadable" is now distinguished from "empty" — the corrupt
     bytes are quarantined to ``goals.json.corrupt``, the failure
     is recorded in the core failures ledger, and a mutation on
     the poisoned instance raises ``StoreUnreadableError``. Reads
     (get/list) still assume {}-on-trouble. Same shape as
     finance/alerts 162 and core/trust 152.
115 — ``get``: a record whose ``Goal.from_dict`` raises -> None, the
     same answer as a missing id. Per-record parse guard.
123 — ``list``: the same guard in loop form -> the bad record is
     skipped, healthy records list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomorals.core.storeguard import StoreUnreadableError
from nomorals.finance.goals import Goal, GoalStore


def _store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals.json")


def test_load_corrupt_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.write_text("{corrupt json", encoding="utf-8")
    assert store._load() == {}
    assert store.list() == []
    assert store.get("anything") is None


def test_load_non_dict_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.write_text("[1, 2]", encoding="utf-8")
    assert store._load() == {}


def test_load_unreadable_then_add_wipes_registry(tmp_path: Path) -> None:
    # CONVERTED (data-loss Pass 1): the add that used to wipe the
    # registry now refuses; the corrupt bytes — the only copy of
    # the registry — sit quarantined, and a fresh instance recovers.
    store = _store(tmp_path)
    store.add(Goal(id="a", name="Alpha", target_kobo=100))
    assert [g.id for g in store.list()] == ["a"]
    store.path.write_text("{corrupt json", encoding="utf-8")
    with pytest.raises(StoreUnreadableError):
        store.add(Goal(id="b", name="Beta", target_kobo=200))
    assert not store.path.exists()
    assert (tmp_path / "goals.json.corrupt").read_text(
        encoding="utf-8") == "{corrupt json"
    fresh = _store(tmp_path)
    fresh.add(Goal(id="b", name="Beta", target_kobo=200))
    assert [g.id for g in fresh.list()] == ["b"]


def test_get_bad_record_returns_none_like_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.write_text(
        json.dumps({"bad": {"id": "bad", "name": "X",
                            "target_kobo": "not-a-number"}}),
        encoding="utf-8")
    assert store.get("bad") is None
    assert store.get("missing") is None  # identical answer


def test_list_skips_bad_record_keeps_good(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.write_text(
        json.dumps({
            "bad": {"id": "bad", "name": "X", "target_kobo": "abc"},
            "good": {"id": "good", "name": "G", "target_kobo": 5},
        }),
        encoding="utf-8")
    assert [g.id for g in store.list()] == ["good"]
