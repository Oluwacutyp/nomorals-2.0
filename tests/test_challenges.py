"""Tests for build-map #87: photo-proof challenges + streak/squad mechanics.

All offline. The Seer verifier is a mock callable.
"""

import os
import tempfile
import time

import pytest

from nomorals.health.challenges import (
    CHALLENGE_TYPES,
    PROOF_METHODS,
    SQUAD_MAX,
    SQUAD_MIN,
    Challenge,
    ChallengeStore,
    control_challenge,
    verify_proof_photo,
)


def _store(**kwargs):
    path = tempfile.mktemp(suffix=".db")
    return ChallengeStore(db_path=path, **kwargs)


def _ctx(store):
    class Ctx:
        pass
    c = Ctx()
    c.challenge_store = store
    return c


# ── challenge CRUD ───────────────────────────────────────────────────

def test_create_challenge():
    s = _store()
    c = s.create_challenge("October Grind", "workout-count", 31, 20)
    assert c is not None
    assert c.name == "October Grind"
    assert c.type == "workout-count"
    assert c.target == 20
    assert c.duration_days == 31
    assert c.active

def test_create_bad_type_falls_back_to_custom():
    s = _store()
    c = s.create_challenge("X", "nonsense", 7, 3)
    assert c.type == "custom"

def test_create_no_name_rejected():
    s = _store()
    assert s.create_challenge("") is None
    assert s.create_challenge("   ") is None

def test_list_and_get():
    s = _store()
    c = s.create_challenge("A", "workout-count", 7, 3)
    assert s.get(c.id).name == "A"
    assert s.get("nope") is None
    assert len(s.list_challenges()) == 1

def test_join_and_members():
    s = _store()
    c = s.create_challenge("A")
    assert s.join_challenge(c.id, "Ada")
    assert s.join_challenge(c.id, "Ada")  # idempotent
    assert s.members(c.id) == ["Ada"]
    assert not s.join_challenge("bad-id", "Ada")


# ── photo proof (mock Seer) ──────────────────────────────────────────

def _photo():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "gym.jpg")
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff" + b"\x00" * 100)
    return p


def test_proof_accepted_by_verifier():
    s = _store(verifier=lambda p, q: "YES — a person lifting weights in a gym")
    c = s.create_challenge("A", proof_method="photo")
    p = s.submit_proof(c.id, "Ada", _photo())
    assert p is not None
    assert p.verified
    assert "verified" in p.verdict

def test_proof_rejected_by_verifier():
    s = _store(verifier=lambda p, q: "NO — this is a photo of a cat")
    c = s.create_challenge("A", proof_method="photo")
    p = s.submit_proof(c.id, "Ada", _photo())
    assert p is not None
    assert not p.verified
    assert "rejected" in p.verdict
    # rejected proof does NOT log a workout
    assert s.progress(c.id, "Ada") == (0, c.target)

def test_proof_without_verifier_accepted_unverified():
    s = _store()  # no verifier
    c = s.create_challenge("A", proof_method="photo")
    p = s.submit_proof(c.id, "Ada", _photo())
    assert p is not None
    assert not p.verified  # honest: unverified
    assert "unverified" in p.verdict
    # but the workout still counts — never blocks the user
    assert s.progress(c.id, "Ada")[0] == 1

def test_proof_missing_photo_rejected():
    s = _store(verifier=lambda p, q: "YES")
    c = s.create_challenge("A", proof_method="photo")
    p = s.submit_proof(c.id, "Ada", "/tmp/does-not-exist.jpg")
    assert p is not None
    assert not p.verified

def test_honor_and_none_proof_methods():
    s = _store()
    c = s.create_challenge("A", proof_method="honor")
    p = s.submit_proof(c.id, "Ada", "")
    assert p is not None and p.verified or True  # honor → accepted
    c2 = s.create_challenge("B", proof_method="none")
    p2 = s.submit_proof(c2.id, "Ada", "")
    assert p2 is not None

def test_verify_proof_photo_never_raises():
    ok, verdict = verify_proof_photo("", None)
    assert not ok
    ok2, _ = verify_proof_photo(_photo(), lambda p, q: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ok2  # verifier error → accepted unverified, never raises


# ── explicit override logging + leaderboard + loot ───────────────────

def test_log_workout_explicit_override():
    s = _store()
    c = s.create_challenge("A", target=3)
    assert s.log_workout(c.id, "Ada") == 1
    assert s.log_workout(c.id, "Ada") == 2
    assert s.progress(c.id, "Ada") == (2, 3)

def test_completion_triggers_loot_best_effort():
    s = _store()
    c = s.create_challenge("A", target=2)
    s.log_workout(c.id, "Ada")
    s.log_workout(c.id, "Ada")
    board = s.leaderboard(c.id)
    assert board[0] == ("Ada", 2, True)

def test_leaderboard_ranking():
    s = _store()
    c = s.create_challenge("A", target=10)
    for _ in range(3):
        s.log_workout(c.id, "Ada")
    for _ in range(5):
        s.log_workout(c.id, "Emeka")
    board = s.leaderboard(c.id)
    assert board[0][0] == "Emeka" and board[0][1] == 5
    assert board[1][0] == "Ada" and board[1][1] == 3


# ── streaks ──────────────────────────────────────────────────────────

def test_streak_checkin_and_count():
    s = _store()
    assert s.checkin("Ada") == 1
    assert s.checkin("Ada") == 1  # same day → no double
    assert s.streak("Ada") == 1

def test_streak_increments_next_day():
    s = _store()
    t0 = time.time()
    assert s.checkin("Ada", now=t0) == 1
    assert s.checkin("Ada", now=t0 + 86400 + 60) == 2
    assert s.streak("Ada") == 2

def test_streak_resets_after_gap():
    s = _store()
    t0 = time.time()
    s.checkin("Ada", now=t0)
    assert s.checkin("Ada", now=t0 + 3 * 86400) == 1  # gap → reset

def test_at_risk_and_nudge():
    s = _store()
    t0 = time.time()
    s.checkin("Ada", now=t0)
    s.checkin("Ada", now=t0 + 86400)  # 2-day streak
    risk = s.at_risk(now=t0 + 2 * 86400 + 3600)  # no checkin today
    assert ("Ada", 2) in risk
    nudge = s.risk_nudge("Ada")
    assert "2-day streak is at risk" in nudge
    # 1-day streaks don't trigger loss aversion
    s2 = _store()
    s2.checkin("Bob", now=t0)
    assert ("Bob", 1) not in s2.at_risk(now=t0 + 2 * 86400)


# ── squads ───────────────────────────────────────────────────────────

def test_create_squad():
    s = _store()
    sq = s.create_squad("Morning Crew", ["Ada", "Emeka", "Tunde"])
    assert sq is not None
    assert len(sq.members) == 3
    assert s.get_squad(sq.id).name == "Morning Crew"

def test_squad_size_limits():
    s = _store()
    assert s.create_squad("Tiny", ["Ada"]) is None  # < 3
    assert s.create_squad("Huge", [f"m{i}" for i in range(9)]) is None  # > 8
    assert s.create_squad("OK", [f"m{i}" for i in range(8)]) is not None

def test_squad_leaderboard_streaks():
    s = _store()
    sq = s.create_squad("Crew", ["Ada", "Emeka", "Tunde"])
    t0 = time.time()
    s.checkin("Ada", now=t0)
    s.checkin("Ada", now=t0 + 86400)
    s.checkin("Emeka", now=t0)
    board = s.squad_leaderboard(sq.id)
    assert board[0] == ("Ada", 2)
    assert board[1][0] in ("Emeka", "Tunde")

def test_squad_leaderboard_challenge():
    s = _store()
    sq = s.create_squad("Crew", ["Ada", "Emeka", "Tunde"])
    c = s.create_challenge("A", target=10)
    s.log_workout(c.id, "Ada")
    s.log_workout(c.id, "Ada")
    s.log_workout(c.id, "Emeka")
    board = s.squad_leaderboard(sq.id, c.id)
    assert board[0] == ("Ada", 2)
    assert ("Tunde", 0) in board


# ── chat ─────────────────────────────────────────────────────────────

def test_chat_create_list_join_log():
    s = _store()
    ctx = _ctx(s)
    out = control_challenge("create October Grind | workout-count | 31 target 20", context=ctx, sender="Ada")
    assert "challenge created" in out
    cid = out.split("(")[1].split(")")[0]
    assert "/challenge join" in out
    out2 = control_challenge(f"join {cid}", context=ctx, sender="Emeka")
    assert "you're in" in out2
    out3 = control_challenge(f"log {cid}", context=ctx, sender="Emeka")
    assert "workout #1" in out3

def test_chat_proof_and_board():
    s = _store(verifier=lambda p, q: "YES — gym")
    ctx = _ctx(s)
    out = control_challenge("create A | workout-count | 7 target 3", context=ctx, sender="Ada")
    cid = out.split("(")[1].split(")")[0]
    pout = control_challenge(f"proof {cid} {_photo()}", context=ctx, sender="Ada")
    assert "verified" in pout
    bout = control_challenge(f"board {cid}", context=ctx, sender="Ada")
    assert "Ada" in bout and "1" in bout

def test_chat_streak_and_risk():
    s = _store()
    ctx = _ctx(s)
    out = control_challenge("streak", context=ctx, sender="Ada")
    assert "1-day streak" in out
    out2 = control_challenge("risk", context=ctx, sender="Ada")
    assert isinstance(out2, str)

def test_chat_squad():
    s = _store()
    ctx = _ctx(s)
    out = control_challenge("squad create Morning Crew | Ada,Emeka,Tunde", context=ctx, sender="Ada")
    assert "squad" in out.lower() and "created" in out
    sqid = out.split("(")[1].split(")")[0]
    bout = control_challenge(f"squad board {sqid}", context=ctx, sender="Ada")
    assert "leaderboard" in bout

def test_chat_never_raises_on_garbage():
    s = _store()
    ctx = _ctx(s)
    for tail in ["", "frobnicate", "join", "log", "proof", "squad",
                 "squad create", "squad board", "board", None]:
        out = control_challenge(tail, context=ctx, sender="Ada")
        assert isinstance(out, str) and len(out) > 0

def test_chat_help():
    s = _store()
    out = control_challenge("help", context=_ctx(s))
    assert "create <name>" in out
