"""Pins for Pass tail-111: nomorals/browser/solved_sessions.py — 3 exclusions.

The module's printed contract is "best effort by design": a save/restore
failure must never break the flow that cleared the wall, and a failed
restore is not a clear — the caller falls through to the normal ladder.

210 E104 save_solved_session: tab.storage_state() raises -> falls back
    to the tab's cookies() ladder rung; the save still lands.
222 E104 save_solved_session: tab.cookies() raises -> [] -> returns 0
    (nothing saved, nothing claimed).
246 E104 restore_solved_session: store.load() raises -> returns 0; the
    challenge ladder (browser/challenge.py) proceeds un-restored.
All three guards are silent: no log record carries the failure (the
module DOES log store.save and add_cookies failures elsewhere — those
sites are not in this batch).
"""

from __future__ import annotations

import logging

from nomorals.browser.solved_sessions import (
    SolvedSessionStore,
    restore_solved_session,
    save_solved_session,
)

CK = [{"name": "a", "value": "1", "domain": "example.com"}]


class TabStateBoom:
    def storage_state(self):
        raise RuntimeError("tab gone")

    def cookies(self):
        return list(CK)


class TabBothBoom:
    def storage_state(self):
        raise RuntimeError("tab gone")

    def cookies(self):
        raise RuntimeError("tab gone")


class TabState:
    def storage_state(self):
        return {
            "cookies": list(CK)
            + [{"name": "b", "value": "2", "domain": "example.com"}],
            "origins": [{"origin": "https://example.com", "localStorage": []}],
        }


class TabBare:
    pass


class StoreBoom:
    def load(self, url):
        raise RuntimeError("disk gone")


class StoreFixed:
    def __init__(self, rec):
        self._rec = rec

    def load(self, url):
        return self._rec


class TabSink:
    def __init__(self):
        self.added = None

    def add_cookies(self, cookies):
        self.added = cookies


# ---------------------------------------------------------------- 210/222

def test_save_falls_back_to_cookies_when_storage_state_raises(tmp_path):
    store = SolvedSessionStore(tmp_path)
    assert save_solved_session(TabStateBoom(), "https://example.com/x", store) == 1
    assert store.sites() != []  # the fallback save actually persisted


def test_save_returns_zero_when_both_getters_raise(tmp_path, caplog):
    store = SolvedSessionStore(tmp_path)
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.browser.solved_sessions"):
        assert save_solved_session(TabBothBoom(), "https://example.com/x",
                                   store) == 0
    assert not any("tab gone" in r.getMessage() for r in caplog.records)


def test_save_healthy_paths(tmp_path):
    store = SolvedSessionStore(tmp_path)
    assert save_solved_session(TabState(), "https://example.com/y", store) == 2
    assert save_solved_session(TabBare(), "https://example.com/z", store) == 0


# ---------------------------------------------------------------- 246

def test_restore_load_crash_returns_zero(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.browser.solved_sessions"):
        assert restore_solved_session(TabSink(), "https://example.com/",
                                      StoreBoom()) == 0
    assert not any("disk gone" in r.getMessage() for r in caplog.records)


def test_restore_healthy_and_empty(tmp_path):
    sink = TabSink()
    rec = {"cookies": list(CK)}
    assert restore_solved_session(sink, "https://example.com/",
                                  StoreFixed(rec)) == 1
    assert sink.added and sink.added[0]["name"] == "a"
    assert restore_solved_session(TabSink(), "https://example.com/",
                                  StoreFixed(None)) == 0
