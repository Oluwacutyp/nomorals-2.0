"""Trend identification — WS4 Pass 2: observations become claims.

P1's substrate records facts and renders no verdicts. This module
is the verdict layer the census (``~/workspace/ws4-video-census/
CENSUS.md`` §3.2) specifies, and its rules are STRUCTURAL — encoded
in the evaluator's shape, not in caller discipline:

* One sweep = an *observation*. A **trend** claim requires two or
  more observations across two or more distinct sweeps, OR one
  sweep corroborated by a second source class (the registry's
  classes: the video organs are one class, the open web another;
  the owner's own activity is a third, stated as his gravity, never
  dressed up as field evidence).
* Anything below that bar is labelled what it is — a **spike**
  (a burst: notable volume against the subject's own quiet
  baseline, in a single sweep) or **unverified** (seen, once,
  uncorroborated). Those labels are claim kinds in their own
  right, not failed trends.
* Platform trending feeds are seed signals only: a sighting that
  arrives through a single source class in a single sweep
  contributes no corroboration leg by itself, and every claim
  whose legs are all one class carries ``single_source`` in its
  basis, named in its confidence basis text.
* Search velocity is relative to the subject's OWN baseline (the
  per-sweep result-count series; failed observations never enter
  it — a failure is not a zero). Velocity is the strongest signal
  and the basis says when a claim's evidence includes it, but
  velocity alone never upgrades a claim past the bar: no level
  upgrades itself (proof doctrine, census §7).

Claims are authored in the digestion ``Claim`` shape (agency is
L5, above media's L4, so the shape is mirrored as plain dicts —
exactly what P1's store persists verbatim) plus the backing the
census §3.4 requires: the observation ids / candidate key behind
the claim, a mandatory ``confidence_basis`` naming the evidence,
and a structured ``basis`` dict carrying the same legs in
machine-readable form (sweeps, instruments, source classes,
corroboration, velocity, single-source flag).

Freshness/decay: each kind carries ``freshness_s`` (trend 3 days,
spike 12 hours, unverified 1 day). A claim's standing is COMPUTED
at read time from its newest evidence timestamp — never a stored
boolean that can rot — so a claim read back after its window
reads ``stale`` wherever it surfaces.

Supersede/idempotency: P1's ``record_claim`` appends verbatim and
renders no verdict, so the evaluator owns claim identity: every
claim carries a stable ``claim_key`` (kind-independent), and a
re-evaluation writes a new row ONLY when the authored claim
materially differs from the latest stored row for that key (kind,
text, basis, freshness, newest-evidence time). Running the
evaluator twice over the same observations writes nothing the
second time. :func:`current_claims` folds the stored history to
the latest row per key — that fold is the current claim set P3's
board and the WS3 feed read; the history stays auditable in the
store. Claims for candidates that have left the pool (shed) stop
being current — a shed subject's claim never lingers as if live.

Candidate verdicts mirror the source registry's evidence shape
(``source_registry.verify_health``): promotion on an evidence
threshold, shedding after ``CANDIDATE_ABSENCE_SHED_AFTER``
consecutive evaluated sweeps that covered the candidate's beats
without sighting it, a sighting resets the absence streak.
Promotion here is standing in the EXPLORE store only — nothing
silently enters any production fetch path or niche vocabulary
(niche adoption is P4 + the owner's tap, census §3.3).

No manufactured certainty: a subject the evidence cannot carry
gets NO claim, and the evaluation report's ``declined`` list says
which subjects were declined and why. The evaluator never raises
into a caller.

Proof grades (doc 24 PG3): every claim ALSO carries a
``proof_grade`` — a :class:`~nomorals.core.proof_grade.ClaimGrade`
reading DERIVED from the evidence legs attached to the claim
(:func:`proof_grade_for_claim`), never assigned from the kind
label. The derivation agrees with PG1's kind mapping when the
kind's defining evidence is actually attached (trend with its
corroboration legs → G4; spike with its derivation → G2;
unverified with its intake record → G1) — and a claim whose
record lacks the legs grades at what its evidence earns, however
it is labelled. Media is L4, core is L1, so the machinery is
imported, not mirrored. The grade is stamped at authoring time,
re-derived at read time in :func:`current_claims` (so rows stored
before PG3 grade too), and rendered beside the kind on the board
and the trends text.
"""

from __future__ import annotations

import time
from typing import Any

from ...core.logging_setup import get_logger
from ...core.proof_grade import ClaimGrade, EvidenceRef
from ..failures import record_failure
from .store import (
    CANDIDATE_PROMOTED,
    CANDIDATE_SHED,
    CANDIDATE_WATCHING,
    OBS_FAILED,
    ExploreStore,
)

_log = get_logger(__name__)

__all__ = [
    "CANDIDATE_ABSENCE_SHED_AFTER",
    "CLAIM_KINDS",
    "CLAIM_SPIKE",
    "CLAIM_TREND",
    "CLAIM_UNVERIFIED",
    "FRESHNESS_S",
    "claim_standing",
    "current_claims",
    "evaluate_trends",
    "proof_grade_for_claim",
    "source_class_of",
]

#: Claim kinds — the census §3.2 labels. ``spike`` and
#: ``unverified`` are first-class kinds, not failed trends.
CLAIM_TREND = "trend"
CLAIM_SPIKE = "spike"
CLAIM_UNVERIFIED = "unverified"
CLAIM_KINDS = (CLAIM_TREND, CLAIM_SPIKE, CLAIM_UNVERIFIED)

#: Freshness windows per kind (digestion ``freshness_s``
#: semantics: after this, treat as stale). A corroborated trend
#: holds for days; a one-sweep spike is a same-day fact; an
#: unverified sighting is worth a day of attention, no more.
FRESHNESS_S = {
    CLAIM_TREND: 3 * 86400.0,
    CLAIM_SPIKE: 12 * 3600.0,
    CLAIM_UNVERIFIED: 86400.0,
}

#: Consecutive evaluated sweeps covering a candidate's beats
#: without a sighting, after which the candidate is shed —
#: mirrors ``source_registry.CANDIDATE_SHED_AFTER`` (3), the same
#: evidence-threshold shape the host loop sheds candidates on.
CANDIDATE_ABSENCE_SHED_AFTER = 3

#: Instrument name → source class (the registry's classes, census
#: §3.2: web, video, news). The two video organs are ONE class —
#: the same platform family agreeing with itself is not
#: cross-class corroboration. The owner's own activity is its own
#: class, always labelled as his gravity.
_SOURCE_CLASSES = {
    "youtube_search": "video",
    "video_finder": "video",
    "web_search": "web",
    "download": "owner",
    "owner": "owner",
    "owner_activity": "owner",
    "chat_inbound": "owner",
}

#: A burst against a quiet baseline: at least this many results in
#: the latest sweep AND at least SPIKE_RATIO× the baseline mean.
_SPIKE_MIN_RESULTS = 4
_SPIKE_RATIO = 3.0
#: A rise worth naming as velocity in a trend basis: latest at
#: least RISING_RATIO× the baseline mean and at least
#: RISING_MIN_DELTA more results than the baseline mean.
_RISING_RATIO = 2.0
_RISING_MIN_DELTA = 3


def source_class_of(instrument: str) -> str:
    """The source class an instrument belongs to. Unknown
    instruments are their own class each (``other:<name>``) — two
    different unknown organs never silently merge into one class,
    and one unknown organ never corroborates itself."""
    name = (instrument or "").strip()
    if name in _SOURCE_CLASSES:
        return _SOURCE_CLASSES[name]
    return f"other:{name}" if name else "other:unknown"


def _normalize(text: str) -> str:
    """Claim-identity normalization — the digestion ``key``
    semantics (Unicode-aware: letters/digits from every script are
    word characters; everything else is a separator). Mirrored
    here because agency is L5 above media's L4; identity is
    internal to the explore store, so the mirror only needs to be
    self-consistent."""
    lowered = (text or "").lower()
    chars = [ch if ch.isalnum() else " " for ch in lowered]
    return " ".join("".join(chars).split())


def proof_grade_for_claim(claim: dict[str, Any]) -> ClaimGrade:
    """The proof grade a claim's ATTACHED evidence earns (doc 24
    PG3), derived from its ``basis`` legs and ``backing`` — never
    from its kind label or its ``confidence_basis`` wording.

    The legs, weakest to strongest:

    * no basis at all → the claim itself is the only provenance:
      a ``generative_thought`` ref, G0.
    * a structured basis + the mandatory ``confidence_basis`` →
      the claim passed the evaluator's intake (the inexpensive
      filter): a ``logical_check`` ref, G1. Every authored claim
      earns at least this.
    * a spike whose defining derivation is attached — the
      velocity dict computed over the beat's own series, or a
      candidate's sighting count at the spike bar — earns a
      ``derivation`` ref, G2. Velocity on any other kind is a
      signal named in the basis text, never a grade leg: velocity
      does not upgrade a claim (WS4 doctrine, pinned).
    * corroboration legs actually attached — result-bearing
      observations across >=2 distinct sweeps, results
      corroborated across source classes, or a single sweep whose
      legs span >=2 source classes — earn ``empirical_source``
      refs naming the sweeps / the candidate record the sightings
      live on, G4.

    A claim LABELLED trend whose record lacks the corroboration
    legs therefore grades G1 (or G0 without a basis): labels do
    not grade, evidence does (the PG2 lesson)."""
    legs = claim.get("basis") or {}
    key = str(claim.get("claim_key") or claim.get("claim_id") or "")
    kind = str(claim.get("kind") or "")
    evidence: list[EvidenceRef] = []
    if legs and str(claim.get("confidence_basis") or "").strip():
        evidence.append(EvidenceRef("logical_check", f"claim:{key}"))
    if kind == CLAIM_SPIKE:
        velocity = legs.get("velocity")
        sightings = legs.get("sightings")
        if velocity:
            evidence.append(EvidenceRef(
                "derivation", f"claim:{key}:velocity"))
        elif sightings is not None and int(sightings or 0) >= 3:
            evidence.append(EvidenceRef(
                "derivation", f"claim:{key}:sightings"))
    sweeps = [str(s) for s in (legs.get("sweeps") or []) if s]
    classes = [str(c) for c in (legs.get("source_classes") or [])
               if c]
    corroborated = int(legs.get("corroborated_urls") or 0) >= 1 \
        or int(legs.get("corroborated_sightings") or 0) >= 1
    if len(sweeps) >= 2 or corroborated \
            or (len(sweeps) == 1 and len(classes) >= 2):
        for sweep_id in sweeps[:8]:
            evidence.append(EvidenceRef(
                "empirical_source", f"sweep:{sweep_id}"))
        if corroborated:
            backing = claim.get("backing") or {}
            candidate_key = str(backing.get("candidate_key") or "")
            ref = (f"candidate:{candidate_key}" if candidate_key
                   else f"claim:{key}:corroboration")
            evidence.append(EvidenceRef("empirical_source", ref))
    if not evidence:
        evidence.append(EvidenceRef(
            "generative_thought", f"claim:{key}"))
    return ClaimGrade(evidence=tuple(evidence))


def claim_standing(claim: dict[str, Any], now: float = 0.0) -> str:
    """``fresh`` or ``stale`` at ``now``, computed from the claim's
    newest evidence time and its freshness window (digestion
    ``_is_stale`` semantics: a non-positive window never stales,
    and ``now=0`` — a caller without a clock — never stales)."""
    try:
        freshness = float(claim.get("freshness_s") or 0)
        if freshness <= 0 or not now:
            return "fresh"
        age = float(now) - float(claim.get("observed_at") or 0)
    except (TypeError, ValueError):
        return "fresh"
    return "stale" if age > freshness else "fresh"


def current_claims(store: ExploreStore, *,
                   now: float = 0.0) -> list[dict[str, Any]]:
    """The current claim set: the latest stored row per
    ``claim_key``, each annotated with its computed ``standing``
    (and ``age_s`` when a clock is given). Claims whose subject
    candidate has left the pool (shed / evicted) are not current.
    Newest evidence first."""
    latest: dict[str, dict[str, Any]] = {}
    for row in store.claims(limit=10000):  # newest first: first wins
        key = row.get("claim_key") or ""
        if key and key not in latest:
            latest[key] = row
    live_candidates = {c.get("key") for c in store.candidates()}
    out: list[dict[str, Any]] = []
    for row in latest.values():
        if row.get("subject_kind") in ("channel", "topic") \
                and row.get("subject") not in live_candidates:
            continue
        annotated = dict(row)
        annotated["standing"] = claim_standing(row, now)
        # PG3: the grade is re-derived from the basis at read
        # time — the derivation governs, so rows stored before
        # PG3 (or with a stale stamp) read at the grade their
        # attached evidence earns.
        annotated["proof_grade"] = proof_grade_for_claim(row).to_dict()
        if now:
            annotated["age_s"] = max(
                0.0, float(now) - float(row.get("observed_at") or 0))
        out.append(annotated)
    out.sort(key=lambda r: -float(r.get("observed_at") or 0))
    return out


# ── evidence assembly ──────────────────────────────────────────────


def _sweep_times(store: ExploreStore) -> dict[str, dict[str, float]]:
    """sweep_id → {started_at, finished_at, status} from the
    store's sweep records (the ordering + completion facts)."""
    out: dict[str, dict[str, float]] = {}
    for row in store.sweeps(limit=1000):
        out[row.get("sweep_id") or ""] = {
            "started_at": float(row.get("started_at") or 0),
            "finished_at": float(row.get("finished_at") or 0),
            "status": str(row.get("status") or ""),
            "beats": list(row.get("beats") or []),
        }
    return out


def _beat_evidence(store: ExploreStore,
                   sweep_times: dict[str, dict[str, float]],
                   ) -> dict[str, dict[str, Any]]:
    """Per-beat evidence assembled from the observation history:
    the per-sweep result-count series (failed observations NEVER
    enter it), support sweeps (sweeps where the beat returned
    results), source classes, instruments, corroborated URLs and
    the backing observation ids."""
    beats: dict[str, dict[str, Any]] = {}
    for row in reversed(store.observations(limit=10000)):
        beat = str(row.get("beat") or "")
        if not beat:
            continue
        entry = beats.setdefault(beat, {
            "beat": beat,
            "answered": 0,         # non-failed observations
            "series": {},          # sweep_id → result_count (non-failed)
            "failed_sweeps": set(),
            "result_sweeps": set(),
            "instruments": set(),
            "classes": set(),
            "classes_by_sweep": {},  # sweep_id → set of classes (results)
            "url_instruments": {},   # (sweep_id, url) → set of instruments
            "obs_ids": [],
            "latest_evidence_at": 0.0,
        })
        sweep_id = row.get("sweep_id") or ""
        instrument = str(row.get("instrument") or "")
        if row.get("status") == OBS_FAILED:
            entry["failed_sweeps"].add(sweep_id)
            continue
        entry["answered"] += 1
        count = int(row.get("result_count") or 0)
        entry["series"][sweep_id] = \
            entry["series"].get(sweep_id, 0) + count
        if count <= 0:
            continue
        entry["result_sweeps"].add(sweep_id)
        entry["instruments"].add(instrument)
        cls = source_class_of(instrument)
        entry["classes"].add(cls)
        entry["classes_by_sweep"].setdefault(sweep_id, set()).add(cls)
        entry["obs_ids"].append(row.get("obs_id") or "")
        entry["latest_evidence_at"] = max(
            entry["latest_evidence_at"], float(row.get("at") or 0))
        for result in row.get("results") or []:
            url = str(result.get("url") or "")
            if url:
                entry["url_instruments"].setdefault(
                    (sweep_id, url), set()).add(instrument)
    for entry in beats.values():
        entry["corroborated_urls"] = sum(
            1 for instruments in entry["url_instruments"].values()
            if len({source_class_of(i) for i in instruments}) >= 2)
        entry["max_classes_one_sweep"] = max(
            (len(classes) for classes in
             entry["classes_by_sweep"].values()), default=0)
        ordered = sorted(
            entry["series"].items(),
            key=lambda kv: (sweep_times.get(kv[0], {}).get(
                "started_at", 0.0), kv[0]))
        entry["series_ordered"] = ordered
    return beats


def _velocity(series_ordered: list[tuple[str, int]],
              ) -> dict[str, Any] | None:
    """Velocity of the latest sweep against the beat's own
    baseline (the mean of every earlier sweep in the series).
    None when there is no prior sweep to baseline against."""
    if len(series_ordered) < 2:
        return None
    latest = series_ordered[-1][1]
    priors = [count for _sid, count in series_ordered[:-1]]
    baseline = sum(priors) / len(priors)
    ratio = (latest / baseline) if baseline > 0 else None
    rising = False
    if baseline > 0:
        rising = ratio is not None and ratio >= _RISING_RATIO \
            and (latest - baseline) >= _RISING_MIN_DELTA
    elif latest >= _SPIKE_MIN_RESULTS:
        rising = True  # a burst off a zero baseline
    return {
        "latest": latest,
        "baseline_mean": round(baseline, 3),
        "prior_sweeps": len(priors),
        "ratio": round(ratio, 3) if ratio is not None else None,
        "from_zero": baseline == 0 and latest > 0,
        "rising": rising,
    }


def _is_spike(velocity: dict[str, Any] | None) -> bool:
    """A burst: the latest sweep alone is loud against the beat's
    own quiet baseline. With no prior sweep at all there is no
    baseline, so a first sighting can never be a spike — it is
    unverified."""
    if velocity is None:
        return False
    latest = velocity["latest"]
    if latest < _SPIKE_MIN_RESULTS:
        return False
    baseline = velocity["baseline_mean"]
    if baseline == 0:
        return True
    return latest >= _SPIKE_RATIO * max(baseline, 1.0)


def _basis_text(legs: dict[str, Any]) -> str:
    """The mandatory confidence basis: one sentence naming exactly
    which legs hold the claim up — sweeps, instruments, source
    classes, corroboration, velocity, and the single-source flag
    when every leg is one class."""
    parts = []
    sweeps = legs.get("sweeps") or []
    parts.append(
        f"{len(sweeps)} sweep(s): {', '.join(sweeps)}"
        if sweeps else "no sweep evidence")
    instruments = legs.get("instruments") or []
    if instruments:
        parts.append("instruments: " + ", ".join(instruments))
    classes = legs.get("source_classes") or []
    if classes:
        parts.append("source classes: " + ", ".join(classes))
    if legs.get("corroborated_urls"):
        parts.append(
            f"{legs['corroborated_urls']} result(s) corroborated "
            "across source classes")
    if legs.get("corroborated_sightings"):
        parts.append(
            f"{legs['corroborated_sightings']} sighting(s) "
            "corroborated across source classes")
    velocity = legs.get("velocity")
    if velocity:
        if velocity.get("from_zero"):
            parts.append(
                f"velocity: burst from a zero baseline "
                f"(latest {velocity['latest']} results)")
        elif velocity.get("ratio") is not None:
            parts.append(
                f"velocity {velocity['ratio']}x the beat's own "
                f"baseline (latest {velocity['latest']} vs mean "
                f"{velocity['baseline_mean']} over "
                f"{velocity['prior_sweeps']} prior sweep(s))")
    if legs.get("owner_gravity"):
        parts.append("includes the owner's own download/play "
                     "gravity (his field, stated as such)")
    if legs.get("single_source"):
        parts.append("single-source: every leg is one source "
                     "class — a seed signal, not corroboration")
    return "; ".join(parts)


def _claim_row(*, key_seed: str, kind: str, subject: str,
               subject_kind: str, text: str, scope: str,
               legs: dict[str, Any], observed_at: float,
               backing: dict[str, Any],
               entities: list[str]) -> dict[str, Any]:
    claim_key = _normalize(key_seed)
    row = {
        "claim_id": f"claim:{claim_key.replace(' ', '-')}",
        "claim_key": claim_key,
        "kind": kind,
        "subject": subject,
        "subject_kind": subject_kind,
        "text": text,
        "source": "explore-trends",
        "observed_at": observed_at,
        "scope": scope,
        "confidence_basis": _basis_text(legs),
        "basis": legs,
        "freshness_s": FRESHNESS_S[kind],
        "published_at": None,
        "contradicts": "",
        "entities": entities,
        "backing": backing,
    }
    # PG3: stamp the derived grade at authoring time (the durable
    # record carries it; reads re-derive — see current_claims).
    row["proof_grade"] = proof_grade_for_claim(row).to_dict()
    return row


_MATERIAL_FIELDS = (
    "kind", "text", "confidence_basis", "basis", "freshness_s",
    "observed_at", "subject", "subject_kind", "scope", "entities",
    "backing", "proof_grade",
)


def _write_claim(store: ExploreStore, authored: dict[str, Any],
                 latest_by_key: dict[str, dict[str, Any]]) -> bool:
    """Supersede-by-key write: store the authored claim only when
    it materially differs from the latest stored row for its key.
    Returns True when a row was written."""
    prior = latest_by_key.get(authored["claim_key"])
    if prior is not None and all(
            prior.get(field) == authored.get(field)
            for field in _MATERIAL_FIELDS):
        return False
    store.record_claim(authored)
    latest_by_key[authored["claim_key"]] = authored
    return True


# ── candidate verdicts (registry shape) ────────────────────────────


def _candidate_legs(candidate: dict[str, Any]) -> dict[str, Any]:
    """The evidence legs behind a candidate: sweeps it surfaced
    in, source classes across its sighting evidence, sightings
    corroborated across classes (the same URL evidenced by
    instruments of two different classes)."""
    classes: set[str] = set()
    instruments: set[str] = set()
    url_classes: dict[str, set[str]] = {}
    owner_gravity = False
    for evidence in candidate.get("evidence") or []:
        instrument = str(evidence.get("instrument") or "")
        if not instrument:
            continue
        instruments.add(instrument)
        cls = source_class_of(instrument)
        classes.add(cls)
        if cls == "owner":
            owner_gravity = True
        url = str(evidence.get("url") or "")
        if url:
            url_classes.setdefault(url, set()).add(cls)
    corroborated = sum(1 for cls_set in url_classes.values()
                       if len(cls_set) >= 2)
    sweeps = list(candidate.get("sweeps") or [])
    return {
        "sweeps": sweeps,
        "sweep_count": len(sweeps),
        "sightings": int(candidate.get("sightings") or 0),
        "instruments": sorted(instruments),
        "source_classes": sorted(classes),
        "corroborated_sightings": corroborated,
        "owner_gravity": owner_gravity,
        "trend_bar": len(sweeps) >= 2 or corroborated >= 1,
    }


def _missed_sweeps(candidate: dict[str, Any],
                   sweep_times: dict[str, dict[str, float]]) -> int:
    """Consecutive completed sweeps that BEGAN after the
    candidate's last sighting and covered at least one of its
    beats without sighting it. (Began-after, not finished-after:
    sightings land mid-sweep, before the sweep closes, so the
    sweep a candidate was sighted in can never count as a miss.)
    A sighting resets the streak (``last_seen`` moves); sweeps
    that never covered its field are not misses — Devon not
    looking is not the candidate vanishing. Candidates with no
    recorded beat (owner-activity intake) are judged against
    every completed sweep."""
    last_seen = float(candidate.get("last_seen") or 0)
    beats = set(candidate.get("beats") or [])
    missed = 0
    for sweep_id, info in sweep_times.items():
        if info.get("status") != "complete":
            continue
        if info.get("started_at", 0.0) <= last_seen:
            continue
        if beats and not beats.intersection(info.get("beats") or []):
            continue
        missed += 1
    return missed


def _candidate_verdicts(store: ExploreStore,
                        sweep_times: dict[str, dict[str, float]],
                        now: float,
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply candidate standing verdicts (promotion / decay /
    shed) to the pool. Returns (summary, legs-by-key for the
    candidates that remain). Mirrors the registry's shape:
    promotion on an evidence threshold, shed after
    CANDIDATE_ABSENCE_SHED_AFTER consecutive misses."""
    summary: dict[str, Any] = {
        "promoted": [], "shed": [], "watching": 0, "standing": {}}
    legs_by_key: dict[str, Any] = {}
    for candidate in store.candidates():
        key = candidate.get("key") or ""
        legs = _candidate_legs(candidate)
        missed = _missed_sweeps(candidate, sweep_times)
        legs["missed_sweeps"] = missed
        score = round(
            legs["sightings"]
            + 2 * legs["corroborated_sightings"]
            + 2 * max(0, legs["sweep_count"] - 1)
            - 2 * missed, 3)
        standing = {
            "sightings": legs["sightings"],
            "sweeps_seen": legs["sweep_count"],
            "source_classes": legs["source_classes"],
            "corroborated_sightings": legs["corroborated_sightings"],
            "missed_sweeps": missed,
            "score": score,
            "evaluated_at": now,
        }
        status = str(candidate.get("status") or CANDIDATE_WATCHING)
        if missed >= CANDIDATE_ABSENCE_SHED_AFTER:
            reason = (
                f"absent from {missed} consecutive evaluated "
                "sweeps covering its beats (threshold "
                f"{CANDIDATE_ABSENCE_SHED_AFTER})")
            if store.update_candidate_verdict(
                    key, status=CANDIDATE_SHED, reason=reason,
                    standing=standing, now=now):
                summary["shed"].append(key)
            continue
        new_status = status
        if status == CANDIDATE_WATCHING and legs["trend_bar"]:
            new_status = CANDIDATE_PROMOTED
            summary["promoted"].append(key)
        if store.update_candidate_verdict(
                key, status=new_status, standing=standing, now=now):
            legs_by_key[key] = legs
            summary["standing"][key] = standing
            if new_status == CANDIDATE_WATCHING:
                summary["watching"] += 1
    return summary, legs_by_key


# ── the evaluator ──────────────────────────────────────────────────


def evaluate_trends(store: ExploreStore, *,
                    now: float = 0.0) -> dict[str, Any]:
    """Evaluate the store's observation history into claims +
    candidate verdicts. Idempotent: a second run over the same
    observations writes no new claim rows and re-renders no
    verdict. Returns the evaluation report — counts, the current
    claim set, candidate verdicts, and the ``declined`` list
    naming every subject the evidence could not carry and why.
    Never raises."""
    now = now or time.time()
    report: dict[str, Any] = {
        "evaluated_at": now,
        "claims_written": 0,
        "claims_current": [],
        "declined": [],
        "candidates": {"promoted": [], "shed": [], "watching": 0},
    }
    try:
        sweep_times = _sweep_times(store)
        latest_by_key: dict[str, dict[str, Any]] = {}
        for row in store.claims(limit=10000):
            key = row.get("claim_key") or ""
            if key and key not in latest_by_key:
                latest_by_key[key] = row

        verdicts, legs_by_key = _candidate_verdicts(
            store, sweep_times, now)
        report["candidates"] = {
            "promoted": verdicts["promoted"],
            "shed": verdicts["shed"],
            "watching": verdicts["watching"],
        }

        # ── beat claims ──
        for beat, entry in _beat_evidence(store, sweep_times).items():
            ordered = entry["series_ordered"]
            velocity = _velocity(ordered)
            result_sweeps = entry["result_sweeps"]
            if not result_sweeps:
                if entry["answered"] == 0 and entry["failed_sweeps"]:
                    report["declined"].append({
                        "subject": beat, "subject_kind": "beat",
                        "reason": "every instrument failed in every "
                                  "sweep — no evidence to claim from"})
                else:
                    report["declined"].append({
                        "subject": beat, "subject_kind": "beat",
                        "reason": "no results observed in any sweep — "
                                  "nothing to claim"})
                continue
            legs = {
                "sweeps": [sid for sid, _count in ordered
                           if sid in result_sweeps],
                "instruments": sorted(entry["instruments"]),
                "source_classes": sorted(entry["classes"]),
                "corroborated_urls": entry["corroborated_urls"],
                "velocity": velocity,
                "single_source": len(entry["classes"]) < 2,
            }
            trend_bar = len(result_sweeps) >= 2 \
                or entry["max_classes_one_sweep"] >= 2
            if trend_bar:
                kind = CLAIM_TREND
                if velocity and velocity["rising"]:
                    text = (f"{beat}: search interest rising — "
                            "sustained results across sweeps with "
                            "velocity above the beat's own baseline")
                elif len(result_sweeps) >= 2:
                    text = (f"{beat}: sustained search presence "
                            "across sweeps (a trend, not a one-off)")
                else:
                    text = (f"{beat}: corroborated across source "
                            "classes within one sweep (a trend)")
            elif _is_spike(velocity):
                kind = CLAIM_SPIKE
                text = (f"{beat}: spike — a burst of search results "
                        "in a single sweep, uncorroborated so far")
            else:
                kind = CLAIM_UNVERIFIED
                text = (f"{beat}: seen in a single sweep so far — "
                        "unverified")
            authored = _claim_row(
                key_seed=f"explore beat {beat}", kind=kind,
                subject=beat, subject_kind="beat", text=text,
                scope=beat, legs=legs,
                observed_at=entry["latest_evidence_at"],
                backing={"observations": entry["obs_ids"],
                         "candidate_key": ""},
                entities=[beat])
            if _write_claim(store, authored, latest_by_key):
                report["claims_written"] += 1

        # ── candidate claims (survivors of the verdicts) ──
        for candidate in store.candidates():
            key = candidate.get("key") or ""
            legs_full = legs_by_key.get(key)
            if legs_full is None:
                continue  # shed this run — no claim (see docstring)
            name = str(candidate.get("name") or key)
            cand_kind = str(candidate.get("kind") or "topic")
            legs = {
                "sweeps": legs_full["sweeps"],
                "instruments": legs_full["instruments"],
                "source_classes": legs_full["source_classes"],
                "corroborated_sightings":
                    legs_full["corroborated_sightings"],
                # PG3: the sighting count is the candidate spike's
                # defining evidence — it rides the basis so the
                # grade derivation can read it back.
                "sightings": legs_full["sightings"],
                "velocity": None,
                "owner_gravity": legs_full["owner_gravity"],
                "single_source":
                    len(legs_full["source_classes"]) < 2,
            }
            if legs_full["trend_bar"]:
                kind = CLAIM_TREND
                text = (f"{name} ({cand_kind}): recurring across "
                        "sweeps — a trend in the exploration field")
            elif legs_full["sightings"] >= 3:
                kind = CLAIM_SPIKE
                text = (f"{name} ({cand_kind}): a burst of sightings "
                        "inside one sweep — a spike, uncorroborated "
                        "across sweeps so far")
            else:
                kind = CLAIM_UNVERIFIED
                text = (f"{name} ({cand_kind}): sighted, not yet "
                        "corroborated — unverified")
            authored = _claim_row(
                key_seed=f"explore candidate {key}", kind=kind,
                subject=key, subject_kind=cand_kind, text=text,
                scope=", ".join(candidate.get("beats") or [])
                or "owner field",
                legs=legs,
                observed_at=float(candidate.get("last_seen") or 0),
                backing={"observations": [],
                         "candidate_key": key},
                entities=[name])
            if _write_claim(store, authored, latest_by_key):
                report["claims_written"] += 1

        report["claims_current"] = current_claims(store, now=now)
    except Exception as exc:  # noqa: BLE001 — the evaluator reports
        record_failure("explore.evaluate_trends", exc)
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report
