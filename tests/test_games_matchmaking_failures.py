"""Tail-88 pins: nomorals/games/matchmaking.py census sites.

The matchmaker's posture: ratings and pairing degrade to the
module's own neutral values rather than raising into game
flow — an unknown matchup is an even one, an unreadable queue
pairs nobody this sweep, and a pairing falls back to the
stored Elo the queue row already carries. Every site is held
as an exclusion:

223 — ``predict_win_probability`` crash -> 0.5. 0.5 is the
even/unknown value (two new players genuinely return it); the
sole production consumer is record_glicko2's upset flag, whose
thresholds (0.35 / 0.65) a 0.5 can never trip, and the prob_a
report field has no consumer. The inner glicko reads already
self-guard with debug logs.
627 — ``sweep`` queue-read crash -> 0. Zero matches is
literally true (none were made), the only caller
(maybe_sweep) discards the return, the rows persist, and the
20-second sweep retries. The one traceless guard in the
module — a bind+debug observability sketch is shelved.
639 — per-row QueuedPlayer construction crash -> continue.
Row-level isolation inside a batch sweep: one malformed queue
row cannot wedge pairing for everyone else. The bad row stays
queued (match_status still answers for that player) and is
skipped again next sweep.
660 — ``_pair_game`` ordinal read crash -> pass, return None.
None selects skill()'s documented fallback: the queue row's
stored Elo, the same path unrated players take by design. A
crash lands on the designed fallback, never on a claim.
708 — ``_make_match`` queue-DELETE crash -> pass. WEAKEST
exclusion of the batch, tension stated plainly: the delete is
post-success cleanup — the duel was created and both players
notified before it — so the match count stays honest (a match
WAS made), but the stale rows remain queued and the next
sweep can pair the same two players into a second duel
(probe-proven: both still show as queued after their match).
Held: no surface claims the rows were deleted, the failure
needs a db that accepted the pairing reads/writes yet fails
this one delete, and a sweep-side live-game guard is a design
change for a deliberate pass, not an error-handling fix.
"""
from __future__ import annotations

from types import SimpleNamespace

from nomorals.games import matchmaking as M
from nomorals.storage.db import Database


def _fresh_db():
    db = Database(":memory:")
    db.migrate()
    return db


def _raise_boom(*args, **kwargs):
    raise RuntimeError("boom")


class _FakeRelay:
    def __init__(self):
        self.invites = []
        self.accepted = []

    def create_invite(self, chat, player, game):
        inv = SimpleNamespace(code=f"code{len(self.invites)}")
        self.invites.append((chat, player, game))
        return inv

    def accept_invite(self, code, chat, player):
        self.accepted.append((code, chat, player))
        return True


def _engine(db):
    sent = []
    eng = SimpleNamespace(
        db=db, relay=_FakeRelay(),
        _send=lambda chat, text: sent.append((chat, text)),
        live=lambda chat: None)
    eng.sent = sent
    return eng


def _player(key, name):
    return SimpleNamespace(key=key, name=name, platform="local")


def _queued_pair(db):
    eng = _engine(db)
    mm = M.Matchmaker(eng)
    mm.enqueue(_player("a", "A"), "ca", "pvp")
    mm.enqueue(_player("b", "B"), "cb", "pvp")
    return eng, mm


# ── 223: win probability degrades to even ───────────────────────

def test_223_predict_crash_returns_even(monkeypatch):
    db = _fresh_db()
    monkeypatch.setattr(M, "get_glicko", _raise_boom)
    assert M.predict_win_probability(db, "pvp", "a", "b") == 0.5
    monkeypatch.undo()
    # control: two new players are genuinely even
    assert M.predict_win_probability(db, "pvp", "a", "b") == 0.5
    # control: a recorded win makes the winner the favourite
    M.record_glicko2(db, "pvp", "a", "b", 1.0)
    assert M.predict_win_probability(db, "pvp", "a", "b") > 0.5


# ── 627: sweep read crash pairs nobody ──────────────────────────

class _QueryBoomDb:
    def execute(self, sql, args=()):
        return None

    def query(self, sql, args=()):
        raise RuntimeError("db down")

    def query_one(self, sql, args=()):
        return None


def test_627_sweep_read_crash_returns_zero():
    mm = M.Matchmaker(_engine(_QueryBoomDb()))
    assert mm.sweep() == 0
    # control: an empty real queue also genuinely makes 0 matches
    mm2 = M.Matchmaker(_engine(_fresh_db()))
    assert mm2.sweep() == 0


# ── 639: a malformed queue row is skipped, not fatal ────────────

_GOOD_A = {"player_key": "a", "chat_key": "ca", "player_name": "A",
           "platform": "local", "game": "pvp", "elo": 1000,
           "enqueued_at": 1.0}
_GOOD_B = dict(_GOOD_A, player_key="b", chat_key="cb", player_name="B")


class _RowsDb:
    def execute(self, sql, args=()):
        return None

    def query(self, sql, args=()):
        return [{"game": "pvp"}, _GOOD_A, _GOOD_B]

    def query_one(self, sql, args=()):
        return None


def test_639_malformed_row_skipped_pair_still_made():
    eng = _engine(_RowsDb())
    mm = M.Matchmaker(eng)
    assert mm.sweep() == 1
    assert len(eng.sent) == 2


# ── 660: ordinal crash falls back to stored Elo ─────────────────

def test_660_ordinal_crash_falls_back_to_elo(monkeypatch):
    db = _fresh_db()
    _eng, mm = _queued_pair(db)
    monkeypatch.setattr(M, "get_glicko", _raise_boom)
    # pairing still happens, on the queue rows' stored Elo
    assert mm.sweep() == 1


# ── 708: post-match delete crash leaves the match standing ──────

def test_708_delete_crash_match_still_made(monkeypatch):
    db = _fresh_db()
    eng, mm = _queued_pair(db)
    orig_execute = db.execute

    def del_boom(sql, args=()):
        if sql.startswith("DELETE FROM match_queue"):
            raise RuntimeError("delete boom")
        return orig_execute(sql, args)

    monkeypatch.setattr(db, "execute", del_boom)
    assert mm.sweep() == 1
    assert len(eng.sent) == 2
    # the pinned consequence: both rows remain queued
    assert M.match_status(db, "a") is not None
    monkeypatch.undo()
    # control: a healthy delete clears both rows after the match
    db2 = _fresh_db()
    _eng2, mm2 = _queued_pair(db2)
    assert mm2.sweep() == 1
    assert M.match_status(db2, "a") is None
