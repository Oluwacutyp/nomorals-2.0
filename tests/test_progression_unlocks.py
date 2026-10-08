"""Tests for build-map #14: features-as-loot (progression-gated commands).

Achievements unlock commands and capabilities; the policy consults the
progression callable as an additive grant source; the runtime gates locked
commands. All offline, temp SQLite.
"""
from __future__ import annotations

import sqlite3

import pytest

from nomorals.core.policy import Capability, CapabilitySet, Policy
from nomorals.games import achievements as A
from nomorals.games.unlocks import (
    COMMAND_UNLOCKS,
    command_unlock_required,
    locked_reply,
)


# ── test DB ────────────────────────────────────────────────────────────────


class _DB:
    """Minimal Database-interface shim over sqlite3 (mirrors test_arena_extras)."""

    def __init__(self):
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE achievements (player_key TEXT NOT NULL, "
            "achievement_id TEXT NOT NULL, unlocked_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, achievement_id))"
        )

    def execute(self, sql, params=()):
        cur = self._db.execute(sql, params)
        self._db.commit()
        return cur


@pytest.fixture()
def db():
    return _DB()


# ── catalog sanity ─────────────────────────────────────────────────────────


def test_unlock_catalog_references_real_achievements():
    catalog = {a.id for a in A.ACHIEVEMENTS}
    for aid in A.ACHIEVEMENT_UNLOCKS:
        assert aid in catalog, f"stale achievement id in ACHIEVEMENT_UNLOCKS: {aid}"
    assert A.ACHIEVEMENT_UNLOCKS  # not empty


def test_unlock_grant_format():
    for aid, grants in A.ACHIEVEMENT_UNLOCKS.items():
        for g in grants:
            assert g.startswith("unlock:command:") or g.startswith("unlock:capability:"), g
            assert len(g.split(":")) == 3, g


def test_command_unlocks_reference_real_grants():
    known = set()
    for grants in A.ACHIEVEMENT_UNLOCKS.values():
        known.update(grants)
    for cmd, required in COMMAND_UNLOCKS.items():
        assert required in known, f"/{cmd} requires unknown grant {required}"


# ── unlock persistence ─────────────────────────────────────────────────────


def test_unlock_achievement_persists_grants_on_new_unlock(db):
    assert A.unlock_achievement(db, "p1", "craps_high_roller") is True
    unlocks = A.get_unlocks(db, "p1")
    assert "unlock:command:predict" in unlocks


def test_unlock_achievement_no_double_persist_on_reunlock(db):
    A.unlock_achievement(db, "p1", "craps_high_roller")
    A.unlock_achievement(db, "p1", "craps_high_roller")  # already unlocked
    unlocks = A.get_unlocks(db, "p1")
    assert unlocks.count("unlock:command:predict") == 1


def test_unlock_achievement_without_grants_persists_nothing(db):
    A.unlock_achievement(db, "p1", "snake_50")  # no entry in ACHIEVEMENT_UNLOCKS
    assert A.get_unlocks(db, "p1") == []


def test_has_unlock(db):
    assert not A.has_unlock(db, "p1", "unlock:command:predict")
    A.unlock_achievement(db, "p1", "arena_raid_win")
    assert A.has_unlock(db, "p1", "unlock:command:predict")
    assert not A.has_unlock(db, "p2", "unlock:command:predict")


def test_get_unlocks_missing_table_no_crash():
    class _NoTable:
        def execute(self, sql, params=()):
            raise sqlite3.OperationalError("no such table")

    assert A.get_unlocks(_NoTable(), "p1") == []
    assert not A.has_unlock(_NoTable(), "p1", "unlock:command:predict")
    assert A.progression_capabilities(_NoTable(), "p1") == set()


# ── progression_capabilities ───────────────────────────────────────────────


def test_progression_capabilities_filters(db):
    A.unlock_achievement(db, "p1", "case_expert")       # capability grant
    A.unlock_achievement(db, "p1", "craps_high_roller")  # command grant
    caps = A.progression_capabilities(db, "p1")
    assert caps == {"research.advanced"}
    assert "predict" not in caps


def test_progression_capabilities_empty(db):
    assert A.progression_capabilities(db, "nobody") == set()


# ── policy progression grant ───────────────────────────────────────────────


def test_policy_grants_via_progression_callable():
    p = Policy(
        default_grant=CapabilitySet.none(),
        progression=lambda: {"research.advanced"},
    )
    d = p.check("research.advanced")
    assert d.allowed


def test_policy_denies_without_progression():
    p = Policy(default_grant=CapabilitySet.none())
    d = p.check("research.advanced")
    assert not d.allowed


def test_policy_progression_additive_not_subtractive():
    # The base grant still works; progression only adds.
    p = Policy(
        default_grant=CapabilitySet([Capability.FS_READ]),
        progression=lambda: {"research.advanced"},
    )
    assert p.check(Capability.FS_READ).allowed
    assert p.check("research.advanced").allowed
    assert not p.check("something.else").allowed


def test_policy_never_raises_on_exploding_callable():
    def _boom():
        raise RuntimeError("db is gone")

    p = Policy(default_grant=CapabilitySet.none(), progression=_boom)
    d = p.check("research.advanced")
    assert not d.allowed  # fail closed, no exception


def test_policy_progression_end_to_end(db):
    A.unlock_achievement(db, "tg:42", "sudoku_hard")
    p = Policy(
        default_grant=CapabilitySet.none(),
        progression=lambda: A.progression_capabilities(db, "tg:42"),
    )
    assert p.check("research.advanced").allowed
    # another player without the achievement is still denied
    p2 = Policy(
        default_grant=CapabilitySet.none(),
        progression=lambda: A.progression_capabilities(db, "tg:99"),
    )
    assert not p2.check("research.advanced").allowed


# ── command gating ─────────────────────────────────────────────────────────


def test_command_unlock_required():
    assert command_unlock_required("predict") == "unlock:command:predict"
    assert command_unlock_required("game") is None
    assert command_unlock_required("PREDICT") == "unlock:command:predict"


def test_locked_reply_names_achievement(db):
    msg = locked_reply("predict", db, "tg:42")
    assert msg is not None
    assert "/predict" in msg
    assert "High Roller" in msg  # craps_high_roller's display name


def test_locked_reply_none_when_unlocked(db):
    A.unlock_achievement(db, "tg:42", "craps_high_roller")
    assert locked_reply("predict", db, "tg:42") is None


def test_locked_reply_none_for_open_command(db):
    assert locked_reply("game", db, "tg:42") is None


def test_locked_reply_never_raises():
    class _Bad:
        def execute(self, sql, params=()):
            raise RuntimeError("nope")

    msg = locked_reply("predict", _Bad(), "tg:42")
    assert msg is not None and "/predict" in msg


# ── AgentContext wiring ────────────────────────────────────────────────────


def test_context_threads_progression_into_policy(db):
    from unittest.mock import MagicMock

    from nomorals.agents.context import AgentContext

    A.unlock_achievement(db, "tg:42", "case_expert")
    ctx = AgentContext(
        settings=MagicMock(), db=db, bus=MagicMock(),
        actor="tg:42", capabilities=CapabilitySet.none(),
    )
    policy = ctx.ensure_policy()
    assert policy.check("research.advanced").allowed
    assert not policy.check("something.else").allowed


def test_context_owner_unaffected_without_unlocks(db):
    from unittest.mock import MagicMock

    from nomorals.agents.context import AgentContext

    ctx = AgentContext(
        settings=MagicMock(), db=db, bus=MagicMock(),
        actor="owner", capabilities=CapabilitySet([Capability.FS_READ]),
    )
    policy = ctx.ensure_policy()
    # owner's own grant intact; no progression unlocks for the owner actor
    assert policy.check(Capability.FS_READ).allowed
    assert not policy.check("research.advanced").allowed
