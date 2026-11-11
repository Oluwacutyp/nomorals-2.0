"""Bench wiring (Lane 4) — one composed bench, ready to run.

Lanes 0–3 shipped organs: the monitor, staging, the
integration queue, the verifier, the contract organ, the
verifier pool. This module is the composition the live system
actually constructs:

- :func:`build_bench` assembles a :class:`BenchRuntime` over
  one home ``Database`` and one integration repo: the staging
  manager, the contract organ, the monitor (with the contract
  organ's claim guard injected — contract suspension is
  therefore structural on every claim this monitor grants),
  the serial integrator, the verifier pool, and the read
  views.
- :meth:`BenchRuntime.cycle` is one full bench pass: publish
  contract generations for promotions already landed, verify
  what waits (pool), expire what sat too long, promote what
  the verdict gate clears (integrator), then publish again —
  a contract revision promoted in this pass rebases its bound
  tasks before the pass ends.
- **Mission registration**: a bench run registers as a mission
  (:func:`register_run_as_mission`) — a ``MissionStore`` record
  carrying the bench run id in its state/metadata and the
  ``bench`` tag — so /runs, milestones and the stall surfaces
  read bench work through the organs they already read. The
  registration is a *record*, never a second driver: the
  MissionRunner is never handed a bench mission (the monitor
  owns bench execution state — the dossier's Wall 1), and
  closing the bench run settles the mission record to match.

Layer: agents (L5) — composes the bench package, missions
(L5 peer) and core.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...missions.mission import MissionStatus, MissionStore
from .contracts import BenchContracts
from .integration import BenchIntegrator
from .monitor import (
    DEFAULT_LEASE_TTL_S,
    DEFAULT_STALL_AFTER_S,
    BenchMonitor,
)
from .pool import DEFAULT_VERIFIER_IDS, VerifierPool
from .staging import BenchViews, StagingManager

__all__ = [
    "BenchRuntime",
    "build_bench",
    "register_run_as_mission",
    "mission_for_run",
    "BENCH_MISSION_TAG",
]

#: The tag a bench run's mission registration carries.
BENCH_MISSION_TAG = "bench"

_STATE_KEY = "bench_run_id"


@dataclass
class BenchRuntime:
    """One composed bench: every organ, wired. Construct via
    :func:`build_bench`."""

    monitor: BenchMonitor
    staging: StagingManager
    contracts: BenchContracts
    integrator: BenchIntegrator
    pool: VerifierPool
    views: BenchViews

    @property
    def store(self) -> Any:
        return self.monitor.store

    def open_run(self, goal: str, base_revision: str, *,
                 now: float | None = None,
                 register_mission: bool = True) -> dict[str, Any]:
        """Open a bench run at ``base_revision`` and (by
        default) register it as a mission for the surfaces.
        The returned run dict carries ``mission_id`` when the
        registration landed."""
        run = self.monitor.open_run(goal, base_revision, now=now)
        if register_mission:
            run["mission_id"] = register_run_as_mission(
                self.monitor.db, run, now=now)
        return run

    def cycle(self, *, now: float | None = None) -> dict[str, Any]:
        """One full bench pass over every open run: contract
        publications, verification, expiry, promotion,
        publication again (see the module docstring)."""
        published: list[dict[str, Any]] = []
        promoted: list[dict[str, Any]] = []
        runs = self.store.list_runs(state="open")
        for run in runs:
            published.extend(
                self.contracts.sync_promotions(run["id"], now=now))
        pool_result = self.pool.cycle(now=now)
        for run in runs:
            promoted.extend(self.integrator.promote_all(
                run["id"], now=now))
        for run in runs:
            published.extend(
                self.contracts.sync_promotions(run["id"], now=now))
        return {"verified": pool_result["verified"],
                "expired": pool_result["expired"],
                "promoted": promoted, "published": published}

    def close_run(self, run_id: str, *,
                  now: float | None = None) -> dict[str, Any] | None:
        """Close a bench run and settle its mission
        registration to DONE — the record follows the bench,
        never the other way round."""
        run = self.monitor.close_run(run_id, now=now)
        mission_id = mission_for_run(self.monitor.db, run_id)
        if mission_id:
            # The bench close already landed; a mission record
            # that cannot be settled is named by the surfaces'
            # own readers, never a reason to un-close the run.
            with contextlib.suppress(Exception):
                MissionStore(self.monitor.db).set_status(
                    mission_id, MissionStatus.DONE,
                    note="bench run closed")
        return run


def build_bench(db: Any, repo: str | Path, root: str | Path, *,
                lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
                stall_after_s: float = DEFAULT_STALL_AFTER_S,
                verifier_ids: tuple[str, ...]
                = DEFAULT_VERIFIER_IDS,
                smoke: Any = None) -> BenchRuntime:
    """Compose the full bench over ``db`` (the home Database),
    ``repo`` (the integration repo — the promoted tree) and
    ``root`` (the bench dir: staging areas + verifier scratch
    live under it)."""
    staging = StagingManager(repo, root)
    contracts = BenchContracts(db, staging=staging)
    monitor = BenchMonitor(
        db, lease_ttl_s=lease_ttl_s, stall_after_s=stall_after_s,
        staging=staging, claim_guard=contracts.claim_guard)
    contracts.attach_monitor(monitor)
    integrator = BenchIntegrator(monitor, smoke=smoke)
    pool = VerifierPool(monitor, Path(root) / "verify")
    views = BenchViews(monitor.store, staging)
    return BenchRuntime(monitor=monitor, staging=staging,
                        contracts=contracts, integrator=integrator,
                        pool=pool, views=views)


def register_run_as_mission(db: Any, run: dict[str, Any], *,
                            now: float | None = None) -> str:
    """Register a bench run as a mission record and return the
    mission id. Idempotent per run: a run that already has a
    registration returns it. The mission is a surface record —
    PENDING, tagged ``bench``, carrying the run id — and is
    never handed to the MissionRunner (the monitor owns bench
    execution state)."""
    existing = mission_for_run(db, str(run["id"]))
    if existing:
        return existing
    store = MissionStore(db)
    mission = store.create_new(
        goal=str(run.get("goal") or "bench run"),
        name=("bench: " + str(run.get("goal") or "run"))[:60],
        state={_STATE_KEY: str(run["id"])},
        metadata={_STATE_KEY: str(run["id"])},
        tags=[BENCH_MISSION_TAG])
    return mission.id


def mission_for_run(db: Any, run_id: str) -> str | None:
    """The mission id registered for a bench run, or None.
    Read from the missions' own records (state/metadata carry
    the run id) — no parallel mapping is kept anywhere."""
    try:
        missions = MissionStore(db).list(limit=200)
    except Exception:  # noqa: BLE001 - unreadable = unregistered
        return None
    for mission in missions:
        state = mission.state or {}
        metadata = mission.metadata or {}
        if state.get(_STATE_KEY) == str(run_id) \
                or metadata.get(_STATE_KEY) == str(run_id):
            return mission.id
    return None
