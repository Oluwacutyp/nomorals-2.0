"""Failing-first tests: second-wave autonomy expansion.

The sibling agency build wired 11 systems (digestion, wisdom, research,
impossible, planner, deals, weather, news, social, leads, gigs). The
rest of the repo — media, code, books, games, commerce, health, travel,
property, legal, finance, learn, marketing, community, characters,
documents, datasci, plus source-health itself — must get the same deal:
a standing initiative, a rotation slot on the SHARED scheduler (no
parallel system), real-organ queue checks where the organ exposes one,
honest stand-down otherwise, and knowledge deposits on every turn.
"""

import time

import pytest

from nomorals.agency.expansion import (
    EXPANSION_SYSTEMS,
    EXPANSION_WEIGHTS,
    ExpansionSystems,
)
from nomorals.agency.initiatives import InitiativeStore
from nomorals.agency.knowledge import KnowledgeStore
from nomorals.agency.rotation import RotationScheduler
from nomorals.agency.sourcehealth import FetchResult, SourceHealth, SourceSpec
from nomorals.storage.db import Database


def _systems(**kw):
    db = Database(":memory:")
    return ExpansionSystems(db=db, **kw)


class _FakeOrgan:
    """An organ with a real queue + tick, like books/games would expose."""

    def __init__(self, pending=0):
        self._pending = pending
        self.ticks = 0

    def pending_count(self):
        return self._pending

    def tick(self):
        self.ticks += 1
        done = self._pending
        self._pending = 0
        return {"done": done}


def test_expansion_covers_the_rest_of_the_repo_without_overlap():
    from nomorals.agency.systems import SYSTEM_NAMES

    assert not set(EXPANSION_SYSTEMS) & set(SYSTEM_NAMES)
    for name in ("media", "code", "books", "games", "commerce", "health",
                 "travel", "property", "legal", "finance", "learn",
                 "marketing", "community", "characters", "documents",
                 "datasci", "sourcehealth"):
        assert name in EXPANSION_SYSTEMS
    assert set(EXPANSION_WEIGHTS) == set(EXPANSION_SYSTEMS)
    assert all(w > 0 for w in EXPANSION_WEIGHTS.values())


def test_seed_initiatives_idempotent_one_per_system():
    sys = _systems()
    first = sys.seed_initiatives(now=1000.0)
    second = sys.seed_initiatives(now=2000.0)
    assert len(first) == len(EXPANSION_SYSTEMS)
    assert len(second) == len(EXPANSION_SYSTEMS)
    ids_first = {it.id for it in first}
    ids_second = {it.id for it in second}
    assert ids_first == ids_second


def test_tick_without_organ_does_standing_watch_bookkeeping():
    sys = _systems()
    sys.seed_initiatives(now=1000.0)
    result = sys.tick("books", now=2000.0)
    assert result.useful is True
    assert "books" in result.work_key
    assert sys.knowledge.count() >= 1


def test_tick_with_no_initiative_stands_down():
    sys = _systems()
    assert sys.has_work("books") is False
    result = sys.tick("books", now=1000.0)
    assert result.useful is False


def test_tick_with_real_organ_reads_its_queue_and_runs_it():
    organ = _FakeOrgan(pending=3)
    sys = _systems(organs={"games": organ})
    sys.seed_initiatives(now=1000.0)
    assert sys.has_work("games") is True
    result = sys.tick("games", now=2000.0)
    assert result.useful is True
    assert organ.ticks == 1
    assert organ.pending_count() == 0


def test_has_work_false_when_organ_queue_empty_and_no_initiative():
    organ = _FakeOrgan(pending=0)
    sys = _systems(organs={"games": organ})
    assert sys.has_work("games") is False


def test_run_cycle_uses_shared_scheduler_and_counts_useful_turns():
    scheduler = RotationScheduler()
    sys = _systems(scheduler=scheduler, registry_collect=lambda: [])
    sys.seed_initiatives(now=1000.0)
    out = sys.run_cycle(budget_s=600.0, now=2000.0)
    assert out["systems_run"] == len(EXPANSION_SYSTEMS)
    assert out["useful_turns"] == len(EXPANSION_SYSTEMS)
    assert out["stood_down"] == 0
    assert scheduler.cycles_run == 1


def test_run_cycle_stands_down_systems_without_work():
    sys = _systems()
    # Only seed one system: the rest must stand down, consuming nothing.
    store = sys.store
    store.create(__import__(
        "nomorals.agency.initiatives", fromlist=["Initiative"]).Initiative(
        outcome="media: standing watch", kind="standing", objective="media",
        evidence="seeded", next_action="run media cycle",
        next_review_at=time.time() + 3600, dedupe_key="system:media"))
    out = sys.run_cycle(budget_s=600.0, now=time.time())
    assert out["systems_run"] == 1
    assert out["ran"] == ["media"]
    assert out["stood_down"] == len(EXPANSION_SYSTEMS) - 1


def test_sourcehealth_system_verifies_sources_and_deposits_summary():
    db = Database(":memory:")
    health = SourceHealth(db)

    def fetching(url, headers=None):
        if "bad" in url:
            return FetchResult(404, b"nf", {})
        return FetchResult(
            200,
            b'<?xml version="1.0"?><rss><channel>'
            b'<item><title>A</title><link>https://x/a</link></item>'
            b'</channel></rss>', {})

    specs = [
        SourceSpec(key="news:a", category="news", name="A",
                   url="https://ok.example/a", kind="rss"),
        SourceSpec(key="news:b", category="news", name="B",
                   url="https://bad.example/b", kind="rss"),
    ]
    sys = ExpansionSystems(db=db, source_health=health,
                           registry_collect=lambda: specs,
                           registry_fetcher=fetching)
    sys.seed_initiatives(now=1000.0)
    result = sys.tick("sourcehealth", now=2000.0)
    assert result.useful is True
    assert health.record("news:b").status == "dead"
    assert health.record("news:a").status == "verified-live"
    found = sys.knowledge.search("sourcehealth verified")
    assert found


def test_expansion_registers_on_shared_scheduler_alongside_core_systems():
    from nomorals.agency.systems import ActiveSystems

    db = Database(":memory:")
    scheduler = RotationScheduler()
    core = ActiveSystems(db=db, scheduler=scheduler)
    exp = ExpansionSystems(db=db, scheduler=scheduler,
                           store=core.store, knowledge=core.knowledge)
    names = scheduler.names()
    assert len(names) == len(set(names))
    assert "research" in names and "media" in names and "books" in names
