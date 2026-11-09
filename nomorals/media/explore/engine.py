"""The sweep engine — one pass of the exploration loop.

A sweep re-observes every beat through every instrument and records
what comes back into the :class:`~.store.ExploreStore` as facts
(census §3.1). P1 ships the pipeline without autonomy: the trigger
is manual — the ``video_explore`` tool's ``run`` action, or any
caller of :func:`run_exploration`. P3's durable job will call the
same :func:`run_exploration` entry point; nothing about the sweep
itself changes when the scheduler arrives.

Bounds (census §7 posture): beats per sweep, results per
beat×instrument, and NEW candidates per sweep are hard-capped, and
a sweep fetches metadata only — exemplar byte fetches are zero in
P1 (they ride the downloader's own budgets when a later pass wants
them). Caps that bite are reported in the sweep result, never
silently absorbed.

Candidate intake (the content analogue of ``note_candidate``):
channels and out-of-beat topics surfacing in sweep results are noted
as watching candidates with the observation that evidenced them;
the owner's own download path notes what he actually fetched (see
:func:`note_owner_activity`, wired from the downloader's success
seam). Intake records discovery — it promotes nothing.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ...core.logging_setup import get_logger
from ..failures import record_failure
from .beats import load_beats
from .instruments import InstrumentError, default_instruments
from .store import OBS_EMPTY, OBS_FAILED, OBS_OK, ExploreStore

_log = get_logger(__name__)

__all__ = [
    "MAX_BEATS_PER_SWEEP",
    "MAX_CANDIDATES_PER_SWEEP",
    "MAX_RESULTS_PER_BEAT",
    "SweepEngine",
    "default_explore_store",
    "explore_for_context",
    "explore_status",
    "note_owner_activity",
    "run_exploration",
]

#: Hard caps (census §7 budget posture). Most sweeps fetch metadata
#: only; exemplar byte fetches per sweep are ZERO in P1 by design.
MAX_BEATS_PER_SWEEP = 12
MAX_RESULTS_PER_BEAT = 10
MAX_CANDIDATES_PER_SWEEP = 40
MAX_EXEMPLAR_FETCHES_PER_SWEEP = 0

#: Words that carry no topic signal when mining result titles for
#: candidate topics (scaffolding + the platform's own furniture).
_TOPIC_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "in", "into", "of",
    "on", "the", "to", "with", "video", "videos", "watch", "youtube",
    "official", "full", "new", "best", "top", "how", "what", "why",
    "vs", "ft", "feat",
})


def _topic_phrases(title: str, beat: str) -> list[str]:
    """Candidate topic phrases from a result title: capitalised
    runs and beat-adjacent phrases, minus scaffolding. Conservative
    on purpose — intake is cheap, false topics are noise P2 would
    have to shed."""
    title = (title or "").strip()
    if not title:
        return []
    phrases: list[str] = []
    for run in re.findall(r"(?:[A-Z][\w&']+(?:\s+|$)){2,}", title):
        phrase = run.strip(" -–—|")
        words = [w for w in phrase.split()
                 if w.lower() not in _TOPIC_STOPWORDS]
        if len(words) >= 2:
            phrases.append(" ".join(words))
    beat_words = {w.lower() for w in re.findall(r"[a-z0-9']+", beat.lower())
                  if len(w) > 2}
    title_words = [w for w in re.findall(r"[A-Za-z0-9']+", title)
                   if w.lower() not in _TOPIC_STOPWORDS and len(w) > 2]
    for i in range(len(title_words) - 1):
        pair = title_words[i:i + 2]
        if {w.lower() for w in pair} & beat_words:
            phrases.append(" ".join(pair))
    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen and phrase.lower() != beat.lower():
            seen.add(key)
            out.append(phrase)
    return out[:3]


class SweepEngine:
    """Runs sweeps over an ExploreStore with an instrument set.

    Instruments are injected (production: :func:`default_instruments`;
    tests: fakes) — the engine owns the loop, the caps, the
    observation bookkeeping and candidate intake; instruments own
    only their organ's call.
    """

    def __init__(self, store: ExploreStore, instruments: list[Any]) -> None:
        self.store = store
        self.instruments = list(instruments)

    def run_sweep(self, beats: list[dict[str, Any]], *,
                  trigger: str = "manual",
                  now: float = 0.0) -> dict[str, Any]:
        """One sweep: every beat × every instrument, recorded.

        Returns the sweep report (counts + what was capped). A beat
        whose instruments all fail still produces its failed
        observations — the sweep completes and says so."""
        started = now or time.time()
        beat_rows = [b for b in beats if (b.get("topic") or "").strip()]
        beats_deferred = max(0, len(beat_rows) - MAX_BEATS_PER_SWEEP)
        beat_rows = beat_rows[:MAX_BEATS_PER_SWEEP]
        sweep_id = self.store.begin_sweep(
            trigger=trigger,
            beats=[b["topic"] for b in beat_rows], at=started)
        totals = {"observations": 0, "failed": 0, "empty": 0, "results": 0}
        new_candidates = 0
        candidates_deferred = 0
        per_beat: list[dict[str, Any]] = []
        try:
            for beat in beat_rows:
                topic = beat["topic"]
                beat_results = 0
                urls_by_instrument: dict[str, set[str]] = {}
                beat_obs: list[dict[str, Any]] = []
                for instrument in self.instruments:
                    name = getattr(instrument, "name", "unknown")
                    query = topic
                    at = time.time()
                    try:
                        rows = instrument.search(
                            query, limit=MAX_RESULTS_PER_BEAT)
                    except InstrumentError as exc:
                        obs = {
                            "sweep_id": sweep_id, "beat": topic,
                            "instrument": name, "query": query, "at": at,
                            "status": OBS_FAILED, "error": str(exc),
                            "result_count": 0, "results": [],
                        }
                        self.store.record_observation(obs)
                        beat_obs.append(obs)
                        totals["observations"] += 1
                        totals["failed"] += 1
                        continue
                    except Exception as exc:  # noqa: BLE001 — contained
                        record_failure(
                            f"SweepEngine.instrument.{name}", exc)
                        obs = {
                            "sweep_id": sweep_id, "beat": topic,
                            "instrument": name, "query": query, "at": at,
                            "status": OBS_FAILED,
                            "error": f"{type(exc).__name__}: {exc}",
                            "result_count": 0, "results": [],
                        }
                        self.store.record_observation(obs)
                        beat_obs.append(obs)
                        totals["observations"] += 1
                        totals["failed"] += 1
                        continue
                    rows = list(rows or [])[:MAX_RESULTS_PER_BEAT]
                    urls_by_instrument[name] = {
                        str(r.get("url") or "") for r in rows
                        if r.get("url")}
                    obs = {
                        "sweep_id": sweep_id, "beat": topic,
                        "instrument": name, "query": query, "at": at,
                        "status": OBS_OK if rows else OBS_EMPTY,
                        "error": "",
                        "result_count": len(rows),
                        "results": rows,
                    }
                    self.store.record_observation(obs)
                    beat_obs.append(obs)
                    totals["observations"] += 1
                    totals["results"] += len(rows)
                    if not rows:
                        totals["empty"] += 1
                    beat_results += len(rows)
                # Cross-instrument corroboration: the same URL (or
                # video) surfacing through two instruments in this
                # beat is recorded on the sweep's beat row — the raw
                # input P2's promotion rule counts legs from.
                url_counts: dict[str, int] = {}
                for urls in urls_by_instrument.values():
                    for url in urls:
                        url_counts[url] = url_counts.get(url, 0) + 1
                corroborated = sum(1 for c in url_counts.values() if c >= 2)
                # Candidate intake from this beat's observations.
                for obs in beat_obs:
                    for row in obs.get("results") or []:
                        if new_candidates >= MAX_CANDIDATES_PER_SWEEP:
                            candidates_deferred += 1
                            continue
                        added = _intake_result(
                            self.store, row, beat=topic,
                            instrument=obs.get("instrument") or "",
                            sweep_id=sweep_id)
                        new_candidates += added
                self.store.record_delivery(
                    topic, sweep_id=sweep_id, delivered=beat_results)
                per_beat.append({
                    "beat": topic, "origin": beat.get("origin") or "",
                    "results": beat_results,
                    "corroborated": corroborated,
                })
        except Exception as exc:  # noqa: BLE001 — a dead sweep still records
            record_failure("SweepEngine.run_sweep", exc)
            self.store.finish_sweep(sweep_id, status="failed")
            raise
        self.store.note_sweep_candidates(sweep_id, new_candidates)
        record = self.store.finish_sweep(sweep_id) or {}
        return {
            "sweep_id": sweep_id,
            "trigger": trigger,
            "started_at": started,
            "finished_at": record.get("finished_at") or time.time(),
            "status": record.get("status") or "complete",
            "beats": per_beat,
            "observations": totals["observations"],
            "failed": totals["failed"],
            "empty": totals["empty"],
            "results": totals["results"],
            "new_candidates": new_candidates,
            "candidates_deferred": candidates_deferred,
            "beats_deferred": beats_deferred,
            "exemplar_fetches": 0,
        }


def _intake_result(store: ExploreStore, row: dict[str, Any], *,
                   beat: str, instrument: str,
                   sweep_id: str) -> int:
    """Note the candidates one result evidences: its channel, and
    topic phrases from its title that reach beyond the beat itself.
    Returns how many were NEW. Intake never promotes."""
    added = 0
    evidence = {
        "title": str(row.get("title") or ""),
        "url": str(row.get("url") or ""),
        "rank": row.get("rank"),
        "view_count": row.get("view_count"),
        "instrument": instrument,
        "beat": beat,
    }
    channel = str(row.get("channel") or "").strip()
    if channel and store.note_candidate(
            "channel", channel, via=f"sweep:{instrument}",
            platform=str(row.get("source") or ""), beat=beat,
            evidence=evidence, sweep_id=sweep_id):
        added += 1
    for phrase in _topic_phrases(str(row.get("title") or ""), beat):
        if store.note_candidate(
                "topic", phrase, via=f"sweep:{instrument}",
                beat=beat, evidence=evidence, sweep_id=sweep_id):
            added += 1
    return added


def note_owner_activity(store: ExploreStore, *, title: str = "",
                        channel: str = "", url: str = "",
                        via: str = "download",
                        now: float = 0.0) -> int:
    """Candidate intake from the owner's REAL path: a video he
    actually downloaded (or fetched through the chain) notes its
    channel and title topics as watching candidates — his own
    gravity is a seed signal (census §3.2). Wired from the
    downloader's success seam; bookkeeping only, never raises into
    the download path. Returns how many candidates were NEW."""
    added = 0
    try:
        evidence = {"title": title or "", "url": url or "",
                    "instrument": via, "beat": ""}
        if channel and store.note_candidate(
                "channel", channel, via=via, evidence=evidence, now=now):
            added += 1
        for phrase in _topic_phrases(title or "", ""):
            if store.note_candidate(
                    "topic", phrase, via=via, evidence=evidence, now=now):
                added += 1
    except Exception as exc:  # noqa: BLE001 — intake never breaks downloads
        record_failure("explore.note_owner_activity", exc)
    return added


# ── context factory (registry pattern) ─────────────────────────────

_DEFAULT_STORE: ExploreStore | None = None


def default_explore_store() -> ExploreStore:
    """The process-wide ExploreStore singleton (memory-only until
    bound to a context KV store by :func:`explore_for_context`)."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = ExploreStore()
    return _DEFAULT_STORE


def explore_for_context(context: Any = None, *,
                        store: ExploreStore | None = None,
                        profile: dict[str, Any] | None = None,
                        watches: list[str] | None = None,
                        settings: Any = None,
                        instruments: list[Any] | None = None,
                        ) -> tuple[ExploreStore, SweepEngine,
                                   list[dict[str, Any]]]:
    """Resolve (store, engine, beats) for a context.

    Mirrors ``betting_registry_for_context``: an injected store
    wins; otherwise the shared default store bound to the context's
    KV store — the same persisted state across callers and restarts.
    A db-less context (CLI, tests) keeps the unbound singleton.
    Beats resolve through :func:`~.beats.load_beats` (injected
    profile > persisted profile > seeds, + briefing topics +
    interests watches). Never raises."""
    store = store if store is not None else default_explore_store()
    db = getattr(context, "db", None) if context is not None else None
    try:
        if db is not None:
            from ...storage.kv import KVStore
            store.bind_store(KVStore(db))
    except Exception as exc:  # noqa: BLE001 — persistence is an enhancement
        record_failure("explore.explore_for_context.bind", exc)
    if settings is None and context is not None:
        settings = getattr(context, "settings", None)
    beats = load_beats(db, settings, profile=profile, watches=watches)
    engine = SweepEngine(store, instruments
                         if instruments is not None
                         else default_instruments(context))
    return store, engine, beats


def run_exploration(context: Any = None, *,
                    trigger: str = "manual",
                    store: ExploreStore | None = None,
                    beats: list[dict[str, Any]] | None = None,
                    profile: dict[str, Any] | None = None,
                    watches: list[str] | None = None,
                    settings: Any = None,
                    instruments: list[Any] | None = None,
                    now: float = 0.0) -> dict[str, Any]:
    """Run ONE exploration sweep and return its report.

    THE sweep entry point: the ``video_explore`` tool's ``run``
    action calls it today; P3's durable ``ensure_explore_jobs`` job
    will call it unchanged. ``beats`` overrides beat resolution
    entirely when given."""
    store, engine, resolved = explore_for_context(
        context, store=store, profile=profile, watches=watches,
        settings=settings, instruments=instruments)
    return engine.run_sweep(
        beats if beats is not None else resolved,
        trigger=trigger, now=now)


def explore_status(store: ExploreStore, *,
                   observation_limit: int = 20) -> dict[str, Any]:
    """The substrate read surface: what sweeps ran, what they saw,
    which candidates are being watched, per-beat baselines — every
    row with its provenance. This is the data P3's ``/explore``
    board renders; P1 ships it as the tool's read actions."""
    summary = store.status()
    return {
        "summary": summary,
        "sweeps": store.sweeps(limit=10),
        "baselines": store.beat_baselines(),
        "candidates": store.candidates(),
        "recent_observations": store.observations(limit=observation_limit),
        "claims": store.claims(),
    }
