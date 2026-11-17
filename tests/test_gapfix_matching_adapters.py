"""Gapfix R3-66 guard: the matching surface adapters are deleted, and
the capability they shimmed still lives in /match.

The three adapters (adapter_gig / adapter_community / adapter_learning)
were pure one-line delegations to matching.batches/stable/questionnaire
with zero production importers; the only live consumer, control_match
(/match), already calls the core modules directly per surface. Deleting
the shims removes no user-facing capability: /match still serves
gig/community/learning end to end.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from nomorals.matching import DailyBatchStore, Questionnaire, curate_daily
from nomorals.matching.chat import control_match

ADAPTERS = ("adapter_gig", "adapter_community", "adapter_learning")


def test_adapters_are_gone():
    """The three adapter modules must not be importable."""
    for name in ADAPTERS:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module(f"nomorals.matching.{name}")


def test_no_production_source_references_adapters():
    """No nomorals/ source file may mention an adapter module name.

    The guard test itself and historical deletion notes carrying the
    R3-66 marker are exempted.
    """
    root = Path(__file__).resolve().parents[1] / "nomorals"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in ADAPTERS:
            if name in text and "R3-66" not in text:
                offenders.append(f"{path.relative_to(root)}: {name}")
    assert not offenders, offenders


def test_matching_core_still_importable():
    """The core modules the adapters shimmed are untouched."""
    from nomorals.matching import (
        explain_pick,
        optimal_assignment,
        roommate_match,
        stable_match,
        stable_match_capacities,
        top_trading_cycles,
    )

    assert callable(curate_daily)
    assert callable(explain_pick)
    assert callable(stable_match)
    assert callable(stable_match_capacities)
    assert callable(roommate_match)
    assert callable(optimal_assignment)
    assert callable(top_trading_cycles)
    assert callable(DailyBatchStore)
    assert callable(Questionnaire)


def _ctx(tmp_path, surface="gig"):
    """A chat context wired to tmp stores, like control_match expects."""
    return types.SimpleNamespace(
        matching_batch_store=DailyBatchStore(db_path=str(tmp_path / "b.db")),
        matching_questionnaire=Questionnaire(
            owner="owner", db_path=str(tmp_path / "q.db")),
        matching_surface=surface,
    )


@pytest.mark.parametrize("surface", ["gig", "community", "learning"])
def test_match_daily_serves_each_surface(tmp_path, surface):
    """Daily picks work per surface through /match, not through an adapter."""
    ctx = _ctx(tmp_path, surface)
    assert "registered for" in control_match(
        f"add {surface}-1 | {surface} pick | python | quality=0.9",
        context=ctx)
    out = control_match(f"daily {surface}", context=ctx)
    assert f"{surface} pick" in out
    # Stable all day: a second call returns the same batch.
    assert control_match(f"daily {surface}", context=ctx) == out


def test_match_run_roommates_assign_ttc_still_work(tmp_path):
    """The two-sided mechanisms the adapters wrapped still run via /match."""
    ctx = _ctx(tmp_path)
    assert "ranking saved" in control_match(
        "rank proposer f1 c1", context=ctx)
    assert "ranking saved" in control_match(
        "rank reviewer c1 f1", context=ctx)
    out = control_match("run", context=ctx)
    assert "f1" in out and "c1" in out
    assert "no blocking pairs" in out

    ctx2 = _ctx(tmp_path)
    for who in ("a", "b", "c", "d"):
        control_match(f"rank proposer {who} " +
                      ",".join(x for x in "abcd" if x != who), context=ctx2)
    out = control_match("roommates a,b,c,d", context=ctx2)
    assert "a" in out and "b" in out

    ctx3 = _ctx(tmp_path)
    for who in ("p1", "p2"):
        control_match(f"rank proposer {who} r1,r2", context=ctx3)
    for who in ("r1", "r2"):
        control_match(f"rank reviewer {who} p1,p2", context=ctx3)
    assert "max-total assignment" in control_match("assign", context=ctx3)

    out = control_match("ttc a=i1 b=i2 -- a:i2,i1 b:i1,i2", context=_ctx(tmp_path))
    assert "top trading cycles" in out


def test_match_questionnaire_and_dealbreakers_still_work(tmp_path):
    """Questionnaire + deal-breakers still work via /match per surface."""
    ctx = _ctx(tmp_path)
    out = control_match("questionnaire gig", context=ctx)
    assert "answer more" in out or "all questions answered" in out
    assert "deal-breaker" in control_match(
        "dealbreaker budget rate le 100", context=ctx)
