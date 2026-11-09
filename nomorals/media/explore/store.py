"""ExploreStore — the persistent observation record for exploration.

WS4 Pass 1 (the exploration substrate). The census
(``~/workspace/ws4-video-census/CENSUS.md`` §3.1/§3.4) designs WS4 as
the source registry's proven loop aimed at *content*: seeds, an
intake for things seen on the real path, scheduled re-observation,
promotion by evidence. This module is the store that loop keeps its
memory in — the registry-pattern sibling of
``media/source_registry.py``: state persists through the injected KV
store (``KVStore`` over the context db), seeds are code and are never
written, and nothing in here ever raises into a caller.

What is stored (payload ``media.explore.v1``, version 1):

* **sweeps** — one record per sweep run: when it ran, what triggered
  it, which beats it covered, and its outcome counts (observations,
  failures, empties, results, new candidates). A sweep that failed
  still records as a sweep — the pass-#71 semantics the source health
  loop runs on.
* **observations** — APPEND-ONLY facts. One observation = one
  instrument answering one query for one beat at one moment, with its
  provenance (instrument, query, timestamp) and the raw evidence it
  returned (ranked results with the inputs any later scoring needs:
  rank, view counts where the platform exposes them, durations,
  cross-instrument corroboration). An observation is a FACT, never a
  conclusion: ``status`` is exactly one of ``ok`` (results came back),
  ``empty`` (the instrument answered and found nothing) or ``failed``
  (the instrument could not answer — the error is stored). Empty and
  failed are different facts and are never merged: a silent zero
  would poison every velocity series P2 builds on this store.
* **candidates** — content sources (channels, topics) seen on the
  real path, recorded UNTRUSTED with status ``watching``: they steer
  nothing and feed no pipeline until a later pass promotes them on
  evidence (P2's claim layer) or the owner acts. Intake mirrors
  ``SourceRegistry.note_candidate``: repeats bump sightings, a shed
  candidate is not re-intaken inside the re-intake window, the pool
  is capped with oldest-first eviction.
* **claims** — the digestion-``Claim``-shaped findings P2 writes.
  P1 stores and returns them verbatim; it authors none, and no
  vocabulary in this module calls anything a trend — a single
  observation in here is never promoted by storage.
* **delivery_log** — per-beat delivery counts (§3.4 contract 4): a
  delivery is a finding RECORDED into this store for a beat during a
  sweep. The WS2 interests ``watch_coverage`` provider measures
  coverage over exactly this log; P3's surfacing adds its own
  deliveries through the same method when it lands.

Ring caps keep the persisted payload bounded; eviction drops the
OLDEST records first and never rewrites what remains — history here
is append-only by construction.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ...core.logging_setup import get_logger
from ..failures import record_failure

_log = get_logger(__name__)

__all__ = ["ExploreStore", "PERSIST_KEY"]

#: KV key the exploration state persists under (injected store — the
#: ``KVStore`` over the context db, exactly the pattern
#: ``media/source_registry.py`` uses for its learned state).
PERSIST_KEY = "media.explore.v1"

#: Observation outcome vocabulary. ``empty`` and ``failed`` are kept
#: distinct on purpose — see the module docstring.
OBS_OK = "ok"
OBS_EMPTY = "empty"
OBS_FAILED = "failed"

#: Candidate lifecycle (P1 sets only ``watching``; promotion and
#: shedding are evidence verdicts a later pass renders — the store
#: carries the vocabulary so records never need a schema migration).
CANDIDATE_WATCHING = "watching"
CANDIDATE_PROMOTED = "promoted"
CANDIDATE_SHED = "shed"

_MAX_SWEEPS = 60
_MAX_OBSERVATIONS = 500
_MAX_CANDIDATES = 200
_MAX_SHED = 50
_MAX_CLAIMS = 300
_MAX_DELIVERY_LOG = 2000
_MAX_EVIDENCE_PER_CANDIDATE = 10
#: A shed candidate is not re-intaken for a week — no flap loop
#: (mirrors the source registry's shed discipline).
_SHED_REINTAKE_S = 7 * 86400.0


def _slug(text: str) -> str:
    """Stable key fragment for a candidate name (lowercase, dashed)."""
    out = []
    for ch in (text or "").lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


class ExploreStore:
    """The exploration loop's memory: sweeps, observations,
    candidates, claims, and the per-beat delivery log.

    Construct with the context's KV store (or None for memory-only,
    the CLI/test posture). ``bind_store`` mirrors the registry's
    re-binding contract for callers that resolve the store late.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store
        self._sweeps: list[dict[str, Any]] = []
        self._observations: list[dict[str, Any]] = []
        self._candidates: dict[str, dict[str, Any]] = {}
        self._shed: list[dict[str, Any]] = []
        self._claims: list[dict[str, Any]] = []
        self._delivery_log: list[dict[str, Any]] = []
        self._sweep_seq = 0
        self._obs_seq = 0
        if store is not None:
            self._load_persisted()

    # ── persistence (registry pattern) ─────────────────────────────
    def bind_store(self, store: Any) -> bool:
        """Bind (or re-bind) the KV store state persists through.
        Idempotent for the same store object; a new store merges its
        persisted state in (in-memory state, which is fresher, wins
        per record). Returns True when the binding changed. Never
        raises."""
        try:
            if store is None or self._store is store:
                return False
            self._store = store
            self._load_persisted(merge=True)
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("ExploreStore.bind_store", e)
            return False

    def _persist(self) -> None:
        """Write the exploration state to the bound store. Never
        raises."""
        if self._store is None:
            return
        try:
            payload = {
                "version": 1,
                "sweep_seq": self._sweep_seq,
                "obs_seq": self._obs_seq,
                "sweeps": self._sweeps,
                "observations": self._observations,
                "candidates": list(self._candidates.values()),
                "shed": self._shed,
                "claims": self._claims,
                "delivery_log": self._delivery_log,
            }
            raw = json.dumps(payload, ensure_ascii=False)
            if hasattr(self._store, "set_raw"):
                self._store.set_raw(PERSIST_KEY, raw, "json")
            else:
                self._store.set(PERSIST_KEY, payload)
        except Exception:  # noqa: BLE001
            _log.debug("explore store persist failed", exc_info=True)

    def _load_persisted(self, *, merge: bool = False) -> None:
        """Restore state from the bound store, AS RECORDED — nothing
        is re-interpreted on load. Never raises."""
        if self._store is None:
            return
        try:
            data = self._store.get(PERSIST_KEY)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                return
            if merge:
                known_sweeps = {s.get("sweep_id") for s in self._sweeps}
                for row in data.get("sweeps") or []:
                    if isinstance(row, dict) \
                            and row.get("sweep_id") not in known_sweeps:
                        self._sweeps.append(row)
                known_obs = {o.get("obs_id") for o in self._observations}
                for row in data.get("observations") or []:
                    if isinstance(row, dict) \
                            and row.get("obs_id") not in known_obs:
                        self._observations.append(row)
                for row in data.get("candidates") or []:
                    if isinstance(row, dict) and row.get("key") \
                            and row["key"] not in self._candidates:
                        self._candidates[row["key"]] = row
                known_claims = {c.get("claim_id") for c in self._claims}
                for row in data.get("claims") or []:
                    if isinstance(row, dict) \
                            and row.get("claim_id") not in known_claims:
                        self._claims.append(row)
                self._shed = (list(data.get("shed") or [])
                              + [s for s in self._shed
                                 if s not in (data.get("shed") or [])])
                self._delivery_log = self._delivery_log or list(
                    data.get("delivery_log") or [])
            else:
                self._sweeps = [r for r in data.get("sweeps") or []
                                if isinstance(r, dict)]
                self._observations = [
                    r for r in data.get("observations") or []
                    if isinstance(r, dict)]
                self._candidates = {
                    r["key"]: r for r in data.get("candidates") or []
                    if isinstance(r, dict) and r.get("key")}
                self._shed = [r for r in data.get("shed") or []
                              if isinstance(r, dict)]
                self._claims = [r for r in data.get("claims") or []
                                if isinstance(r, dict)]
                self._delivery_log = [
                    r for r in data.get("delivery_log") or []
                    if isinstance(r, dict)]
            self._sweep_seq = max(self._sweep_seq,
                                  int(data.get("sweep_seq") or 0))
            self._obs_seq = max(self._obs_seq,
                                int(data.get("obs_seq") or 0))
            self._sweeps = self._sweeps[-_MAX_SWEEPS:]
            self._observations = self._observations[-_MAX_OBSERVATIONS:]
            self._claims = self._claims[-_MAX_CLAIMS:]
            self._delivery_log = self._delivery_log[-_MAX_DELIVERY_LOG:]
            self._shed = self._shed[-_MAX_SHED:]
        except Exception:  # noqa: BLE001
            _log.debug("explore store load failed", exc_info=True)

    # ── sweeps ─────────────────────────────────────────────────────
    def begin_sweep(self, *, trigger: str = "manual",
                    beats: list[str] | None = None,
                    at: float = 0.0) -> str:
        """Open a sweep record; returns its sweep id. The record
        starts ``running`` and is closed by :meth:`finish_sweep` —
        a sweep whose engine died mid-run stays visible as running
        rather than vanishing."""
        at = at or time.time()
        self._sweep_seq += 1
        sweep_id = f"sweep-{self._sweep_seq}"
        self._sweeps.append({
            "sweep_id": sweep_id,
            "trigger": trigger or "manual",
            "started_at": at,
            "finished_at": 0.0,
            "status": "running",
            "beats": list(beats or []),
            "observations": 0,
            "failed": 0,
            "empty": 0,
            "results": 0,
            "new_candidates": 0,
        })
        self._sweeps = self._sweeps[-_MAX_SWEEPS:]
        self._persist()
        return sweep_id

    def finish_sweep(self, sweep_id: str, *, at: float = 0.0,
                     status: str = "complete") -> dict[str, Any] | None:
        """Close a sweep record with its final status. ``complete``
        means the sweep ran to the end (individual observations may
        still have failed — those facts live on the observations);
        ``failed`` means the sweep itself did not finish."""
        at = at or time.time()
        for row in self._sweeps:
            if row.get("sweep_id") == sweep_id:
                row["finished_at"] = at
                row["status"] = status
                self._persist()
                return dict(row)
        return None

    def note_sweep_candidates(self, sweep_id: str, new_count: int) -> None:
        """Credit a sweep record with newly intaken candidates."""
        if new_count <= 0:
            return
        for row in self._sweeps:
            if row.get("sweep_id") == sweep_id:
                row["new_candidates"] = int(row.get("new_candidates") or 0) \
                    + int(new_count)
                self._persist()
                return

    def sweeps(self, limit: int = 10) -> list[dict[str, Any]]:
        """Sweep records, newest first."""
        return [dict(r) for r in reversed(self._sweeps[-limit:])]

    def latest_sweep(self) -> dict[str, Any] | None:
        return dict(self._sweeps[-1]) if self._sweeps else None

    # ── observations (append-only) ─────────────────────────────────
    def record_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Append one observation fact. The store assigns its id and
        folds its counts into the owning sweep record. Existing
        observations are never rewritten — a later sweep's facts are
        new rows. Returns the stored row."""
        row = dict(observation)
        self._obs_seq += 1
        row["obs_id"] = f"obs-{self._obs_seq}"
        row.setdefault("at", time.time())
        self._observations.append(row)
        self._observations = self._observations[-_MAX_OBSERVATIONS:]
        sweep_id = row.get("sweep_id") or ""
        for sweep in self._sweeps:
            if sweep.get("sweep_id") == sweep_id:
                sweep["observations"] = int(sweep.get("observations") or 0) + 1
                if row.get("status") == OBS_FAILED:
                    sweep["failed"] = int(sweep.get("failed") or 0) + 1
                elif row.get("status") == OBS_EMPTY:
                    sweep["empty"] = int(sweep.get("empty") or 0) + 1
                sweep["results"] = int(sweep.get("results") or 0) \
                    + int(row.get("result_count") or 0)
                break
        self._persist()
        return dict(row)

    def observations(self, *, beat: str = "", sweep_id: str = "",
                     instrument: str = "", status: str = "",
                     limit: int = 50) -> list[dict[str, Any]]:
        """Observation rows, newest first, optionally filtered. Each
        row carries its full provenance + evidence snapshot."""
        out = []
        for row in reversed(self._observations):
            if beat and row.get("beat") != beat:
                continue
            if sweep_id and row.get("sweep_id") != sweep_id:
                continue
            if instrument and row.get("instrument") != instrument:
                continue
            if status and row.get("status") != status:
                continue
            out.append(dict(row))
            if len(out) >= limit:
                break
        return out

    def beat_baselines(self) -> dict[str, dict[str, Any]]:
        """Per-beat observation series — the raw material P2's
        velocity math reads. Counts and timestamps only; no verdict
        is computed here (a baseline is evidence, not a claim)."""
        beats: dict[str, dict[str, Any]] = {}
        for row in self._observations:
            beat = str(row.get("beat") or "")
            if not beat:
                continue
            entry = beats.setdefault(beat, {
                "beat": beat, "observations": 0, "failed": 0, "empty": 0,
                "sweeps": [], "result_counts": [], "last_result_count": 0,
                "last_swept_at": 0.0,
            })
            entry["observations"] += 1
            if row.get("status") == OBS_FAILED:
                entry["failed"] += 1
            elif row.get("status") == OBS_EMPTY:
                entry["empty"] += 1
            sid = row.get("sweep_id") or ""
            if sid and sid not in entry["sweeps"]:
                entry["sweeps"].append(sid)
            if row.get("status") != OBS_FAILED:
                entry["result_counts"].append(int(row.get("result_count") or 0))
                entry["last_result_count"] = int(row.get("result_count") or 0)
            entry["last_swept_at"] = max(entry["last_swept_at"],
                                         float(row.get("at") or 0))
        for entry in beats.values():
            counts = entry["result_counts"]
            entry["sweep_count"] = len(entry["sweeps"])
            entry["mean_result_count"] = (
                round(sum(counts) / len(counts), 3) if counts else 0.0)
        return beats

    # ── candidates (intake; mirror of note_candidate) ──────────────
    @staticmethod
    def candidate_key(kind: str, name: str, platform: str = "") -> str:
        kind = (kind or "topic").strip().lower()
        if kind == "channel":
            return f"channel:{(platform or 'web').strip().lower()}:{_slug(name)}"
        return f"topic:{_slug(name)}"

    def note_candidate(self, kind: str, name: str, *, via: str = "",
                       platform: str = "", beat: str = "",
                       evidence: dict[str, Any] | None = None,
                       sweep_id: str = "", now: float = 0.0) -> bool:
        """Record a discovered content source as a CANDIDATE.

        Candidates are untrusted by construction: they steer no
        sweep and feed no pipeline — they are *watched*, and a later
        evidence pass (P2) or the owner promotes them. Returns True
        when a NEW candidate was recorded; False when it is already
        known (sightings bumped instead), was shed within the
        re-intake window, or the name is not plausible. Never
        raises."""
        try:
            now = now or time.time()
            name = (name or "").strip()
            if not name or len(name) < 2:
                return False
            kind = (kind or "topic").strip().lower()
            if kind not in ("channel", "topic"):
                kind = "topic"
            key = self.candidate_key(kind, name, platform)
            existing = self._candidates.get(key)
            if existing is not None:
                existing["sightings"] = int(existing.get("sightings") or 0) + 1
                existing["last_seen"] = now
                if sweep_id and sweep_id not in existing["sweeps"]:
                    existing["sweeps"].append(sweep_id)
                if beat and beat not in existing["beats"]:
                    existing["beats"].append(beat)
                if evidence:
                    existing["evidence"].append(evidence)
                    existing["evidence"] = \
                        existing["evidence"][-_MAX_EVIDENCE_PER_CANDIDATE:]
                self._persist()
                return False
            for entry in self._shed:
                if entry.get("key") == key \
                        and now - float(entry.get("at") or 0) \
                        < _SHED_REINTAKE_S:
                    return False
            if len(self._candidates) >= _MAX_CANDIDATES:
                oldest_key = min(
                    self._candidates,
                    key=lambda k: float(
                        self._candidates[k].get("last_seen") or 0),
                    default=None)
                if oldest_key is not None:
                    self._candidates.pop(oldest_key, None)
            self._candidates[key] = {
                "key": key,
                "kind": kind,
                "name": name,
                "platform": (platform or "").strip().lower(),
                "status": CANDIDATE_WATCHING,
                "first_seen": now,
                "last_seen": now,
                "sightings": 1,
                "sweeps": [sweep_id] if sweep_id else [],
                "beats": [beat] if beat else [],
                "via": via or "sweep",
                "evidence": [evidence] if evidence else [],
            }
            _log.info("explore: candidate noted %s via %s",
                      key, via or "sweep")
            self._persist()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("ExploreStore.note_candidate", e)
            return False

    def candidates(self, *, status: str = "") -> list[dict[str, Any]]:
        """Candidate records, most-sighted first."""
        rows = [dict(r) for r in self._candidates.values()
                if not status or r.get("status") == status]
        rows.sort(key=lambda r: (-int(r.get("sightings") or 0),
                                 -float(r.get("last_seen") or 0)))
        return rows

    def candidate(self, key: str) -> dict[str, Any] | None:
        row = self._candidates.get(key)
        return dict(row) if row else None

    def shed_candidate(self, key: str, reason: str = "",
                       now: float = 0.0) -> bool:
        """Move a candidate out of the watched pool, recording why.
        The shed record enforces the re-intake window in
        :meth:`note_candidate`. (P1 sheds only when a caller rules
        it; sweep staleness verdicts are P2's.)"""
        now = now or time.time()
        row = self._candidates.pop(key, None)
        if row is None:
            return False
        self._shed.append({"key": key, "name": row.get("name") or "",
                           "reason": reason, "at": now})
        self._shed = self._shed[-_MAX_SHED:]
        self._persist()
        return True

    def update_candidate_verdict(self, key: str, *,
                                 status: str | None = None,
                                 standing: dict[str, Any] | None = None,
                                 reason: str = "",
                                 now: float = 0.0) -> bool:
        """Record an evidence verdict on a candidate (P2's write
        path — the trend evaluator's promotion / standing / shed
        decisions land here; P1 only intakes).

        ``standing`` is the evaluator's evidence snapshot (sighting
        counts, sweeps seen, source classes, corroboration, missed
        sweeps, score — inputs stored, per the census's rule that a
        score whose inputs aren't stored is not a score). A
        ``status`` of ``shed`` delegates to :meth:`shed_candidate`
        so the shed record + re-intake window stay single-sourced;
        any other status updates the row in place. Returns False
        when the key is unknown. Never raises."""
        try:
            if status == CANDIDATE_SHED:
                row = self._candidates.get(key)
                if row is not None and standing is not None:
                    row["standing"] = dict(standing)
                return self.shed_candidate(key, reason, now)
            row = self._candidates.get(key)
            if row is None:
                return False
            if status is not None:
                row["status"] = status
            if standing is not None:
                row["standing"] = dict(standing)
            self._persist()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("ExploreStore.update_candidate_verdict", e)
            return False

    # ── claims (P2 writes; P1 stores verbatim) ─────────────────────
    def record_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Store one claim record verbatim (digestion ``Claim``
        shape + backing observation ids, as P2 authors them). The
        store renders no verdict on it."""
        row = dict(claim)
        row.setdefault("recorded_at", time.time())
        self._claims.append(row)
        self._claims = self._claims[-_MAX_CLAIMS:]
        self._persist()
        return dict(row)

    def claims(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in reversed(self._claims[-limit:])]

    # ── delivery log (§3.4 contract: per-beat deliveries) ──────────
    def record_delivery(self, beat: str, *, sweep_id: str = "",
                        at: float = 0.0, delivered: int = 0,
                        channel: str = "explore-store") -> None:
        """Record findings delivered for a beat. In P1 a delivery is
        a finding recorded into this store during a sweep (the store
        is the delivery point until P3's surfaces exist); the WS2
        interests provider measures watch coverage over this log."""
        self._delivery_log.append({
            "beat": beat, "sweep_id": sweep_id, "at": at or time.time(),
            "delivered": int(delivered), "channel": channel,
        })
        self._delivery_log = self._delivery_log[-_MAX_DELIVERY_LOG:]
        self._persist()

    def deliveries(self, *, beat: str = "") -> list[dict[str, Any]]:
        return [dict(r) for r in self._delivery_log
                if not beat or r.get("beat") == beat]

    # ── status ─────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        watching = sum(1 for r in self._candidates.values()
                       if r.get("status") == CANDIDATE_WATCHING)
        latest = self.latest_sweep()
        return {
            "sweeps": len(self._sweeps),
            "observations": len(self._observations),
            "candidates": len(self._candidates),
            "candidates_watching": watching,
            "claims": len(self._claims),
            "deliveries": len(self._delivery_log),
            "latest_sweep": latest,
        }
