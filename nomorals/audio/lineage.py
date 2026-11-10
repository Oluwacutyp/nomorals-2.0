"""Sample / lineage analysis — SongRecon classes 4, 5 and 7.

Doc 24 SR4. The class-1 engine (``audio/fingerprint.py``) answers
"is this that *recording*" by collapsing all landmark evidence
into ONE whole-track offset vote. A sample defeats that vote by
construction: only a slice of the query aligns with the source,
at one offset, while the rest aligns elsewhere or not at all.
This module reads the same evidence windowed instead:

* **Windowed partial matching.** For every enrolled track, the
  query's landmark occurrences are grouped by reference-offset
  bin (the shipped 50 ms bins); inside one bin, votes are split
  into runs of consecutive query anchors. A run that clears the
  sample floors (span, votes, distinct landmark types, and the
  fraction of in-region query landmarks it explains) is a
  *sample region*: a contiguous stretch of the query that is a
  lifted piece of the enrolled recording, with both time ranges
  and its regional coherent fraction as the evidence.
* **Lineage (class 5).** The set of claimed regions, grouped by
  source work, is the lineage record: which enrolled works this
  work draws from, where, and how strongly each region holds.
  The query may be an upload or an enrolled track itself
  (``FingerprintDB.track_hashes``), so lineage exists between
  works in the catalog too.
* **Originality bands (class 7).** The union of claimed regions
  measures how much of the work is *accounted for* by enrolled
  sources. The band is a pure function of that measured share,
  computed at scan time — never stored, never assigned, and the
  unaccounted remainder is reported as *unaccounted*, never as
  "original": the scan covers only Devon's enrolled catalog, so
  absence of lineage means "no enrolled source found", full stop.
  A band is a measurement summary, never a copyright verdict.

Honesty rules, enforced by construction:

* a lineage entry without regions is not a claim — sources are
  only ever built from claimed regions;
* evidence that aligns but falls under a floor is reported as
  sub-floor evidence with the floor it missed, never silently
  dropped and never promoted to a claim;
* a query that IS an enrolled recording (class-1 acceptance) is
  excluded as its own source and the receipt says so;
* alignment is at the recording's own pitch/tempo — a
  pitch/tempo-shifted sample does not align in landmark space
  (that is class-2 territory, measured separately in SR2);
* regional landmark coherence cannot tell a lifted recording
  from a frame-exact re-performance whose landmarks survive the
  same quantization (the SR2 sibling problem, one level down):
  when the whole query is such a re-performance, the class-1
  identity check claims it first and lineage defers to it;
* reference fingerprints are the enrolled fingerprints *as
  stored* — enrollment keeps each recording's first
  ``_MAX_HASHES`` landmarks (the shipped window), so a sample
  drawn from beyond a source's stored window cannot be found.

Dependencies: stdlib + the fingerprint module's substrate. No
network, no external service.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .failures import record_failure
from .fingerprint import (
    FINGERPRINT_WINDOW_SECONDS,
    LOCAL_RIGHTS_NOTE,
    MATCH_THRESHOLD,
    AudioReadError,
    FingerprintDB,
    _media_duration,
    _query_fingerprint_hash,
    default_fingerprint_db,
    fingerprint,
)

_log = logging.getLogger(__name__)

__all__ = [
    "LINEAGE_SCHEMA",
    "MATCH_CLASS_SAMPLE_LINEAGE",
    "BAND_NONE_ACCOUNTED",
    "BAND_MINORITY_ACCOUNTED",
    "BAND_MAJORITY_ACCOUNTED",
    "BAND_NEAR_FULLY_ACCOUNTED",
    "BAND_LABELS",
    "MIN_REGION_SPAN_S",
    "MIN_REGION_VOTES",
    "MIN_REGION_TYPES",
    "MIN_REGION_FRACTION",
    "LineageReceipt",
    "originality_band",
    "band_from_receipt",
    "scan_lineage",
    "scan_lineage_db",
    "scan_lineage_hashes",
    "render_lineage_lines",
]

#: Lineage receipt schema (sibling of songrecon.receipt/v1).
LINEAGE_SCHEMA = "songrecon.lineage/v1"
#: Match class for every sample/lineage answer. Samples and the
#: derivative graph they form are classes 4+5; the class name
#: keeps them apart from class-1 identifications and class-3
#: hypotheses on every surface.
MATCH_CLASS_SAMPLE_LINEAGE = "class_4_sample_usage_and_lineage"

#: Offset-bin width of the shipped landmark voting (fingerprint.py).
OFFSET_BIN_MS = 50
#: Largest silence between consecutive coherent anchors inside one
#: region. Landmark anchors in real material land many times per
#: second; a gap beyond this is a different placement, not a pause
#: inside one lifted segment.
REGION_GAP_MS = 2000
#: Fragments of one placement land in neighbouring offset bins:
#: the query's STFT grid is offset from the reference's, which
#: flips some landmarks a frame (~186 ms) either way, so one
#: placement's votes form modes at the true offset ± k·frame.
#: Fragments whose query spans overlap and whose median offsets
#: agree within this window merge before floors are applied.
FRAGMENT_DELTA_TOLERANCE_MS = 450
FRAGMENT_SPAN_GAP_S = 0.5
#: Region edges are trimmed to the coherent core: edge buckets
#: (0.5 s) whose coherent fraction falls under this share of the
#: region's median bucket fraction are chained chance alignments,
#: not part of the lifted segment (measured: true interiors run
#: bucket fractions 0.4-0.9 while chained tails sit at 0.05-0.2).
#: A density trim at EDGE_DENSITY_SHARE of the median bucket vote
#: count follows for tails that are dense but incoherent.
EDGE_BUCKET_S = 0.5
EDGE_FRACTION_SHARE = 0.5
EDGE_DENSITY_SHARE = 0.25
EDGE_MIN_BUCKET_VOTES = 4
#: The lineage query fingerprint is extracted effectively
#: uncapped: the shipped 4,000-hash cap covers only a prefix of a
#: full work, and a lineage scan must see the whole work to state
#: an accounted share over it. (Reference fingerprints stay as
#: enrolled — see the module docstring.)
SCAN_MAX_HASHES = 1_000_000
#: Share of the analyzed work a class-1 winner's evidence window
#: must cover before the query counts as BEING that recording
#: (and is excluded as its own lineage source).
IDENTITY_WINDOW_SHARE = 0.80

#: Sample floors (doc 24 SR4). A merged region becomes a claimed
#: sample region only when it clears ALL of these; the values are
#: set from the SR4 synthesized-fixture battery (see
#: tests/test_d24sr4_lineage.py and the calibration log in the
#: pass's verify dir). On that battery, true embedded excerpts
#: (gains 0.6-1.0) measured regional fractions 0.39-0.85 and
#: regional type recalls 0.31-0.78, while the strongest structured
#: chance alignments (the fixture family's shared chord lattice
#: makes cross-song collisions coherent, not uniform) reached
#: fraction 0.45 but never type recall over 0.26 — the fraction
#: alone does not separate them, the type recall does: a lifted
#: segment explains most of the landmark TYPES present in its
#: span, a chance alignment only the shared-vocabulary subset.
#: Floors sit in the measured gaps. Region boundaries are the
#: first/last coherent anchor times, so a claimed region can sit
#: slightly inside the lifted segment's true edges and accounted
#: shares err conservative.
MIN_REGION_SPAN_S = 3.0
MIN_REGION_VOTES = 12
MIN_REGION_TYPES = 6
MIN_REGION_FRACTION = 0.30
MIN_REGION_TYPE_RECALL = 0.30

#: What a region's strength measures: inside the region, the share
#: of the query's own landmark hashes that agree on the region's
#: one reference offset. An alignment statistic, not a probability.
REGION_SCORE_NAME = "regional_coherent_fraction"
REGION_SCORE_UNIT = ("coherent landmark-hash votes per in-region "
                     "query landmark hash")
REGION_SCORE_BASIS = (
    "largest set of query landmark hashes anchored inside one "
    "time region agreeing on one 50 ms reference-offset bin, "
    "divided by the distinct query landmark hashes anchored "
    "inside that region"
)

# ---------------------------------------------------------------------------
# originality bands — a pure function of the measured accounted share
# ---------------------------------------------------------------------------

BAND_NONE_ACCOUNTED = "none_accounted"
BAND_MINORITY_ACCOUNTED = "minority_accounted"
BAND_MAJORITY_ACCOUNTED = "majority_accounted"
BAND_NEAR_FULLY_ACCOUNTED = "near_fully_accounted"

#: Band cut points on the accounted share (union of claimed sample
#: regions ÷ analyzed duration). Bands describe how much of the
#: work the enrolled catalog accounts for — nothing else.
BAND_MAJORITY_MIN = 0.50
BAND_NEAR_FULLY_MIN = 0.90

BAND_LABELS = {
    BAND_NONE_ACCOUNTED: (
        "no part of this work is accounted for by an enrolled "
        "source"),
    BAND_MINORITY_ACCOUNTED: (
        "a minority of this work is accounted for by enrolled "
        "sources"),
    BAND_MAJORITY_ACCOUNTED: (
        "a majority of this work is accounted for by enrolled "
        "sources"),
    BAND_NEAR_FULLY_ACCOUNTED: (
        "nearly all of this work is accounted for by enrolled "
        "sources"),
}

#: The scope sentence every lineage answer carries. "No lineage
#: found" is a statement about the enrolled catalog, never about
#: the work's absolute originality; a band is a measurement, never
#: a copyright verdict.
LINEAGE_SCOPE_NOTE = (
    "Scope: this scan covers only Devon's enrolled catalog. "
    "Audio no enrolled source accounts for is reported as "
    "unaccounted — that is not a finding that it is original, "
    "and an originality band is a measurement summary, never a "
    "copyright verdict."
)


def originality_band(accounted_share: float) -> str:
    """The band for one measured accounted share. Pure."""
    share = float(accounted_share or 0.0)
    if share <= 0.0:
        return BAND_NONE_ACCOUNTED
    if share < BAND_MAJORITY_MIN:
        return BAND_MINORITY_ACCOUNTED
    if share < BAND_NEAR_FULLY_MIN:
        return BAND_MAJORITY_ACCOUNTED
    return BAND_NEAR_FULLY_ACCOUNTED


def band_from_receipt(receipt: dict[str, Any]) -> str:
    """Re-derive a receipt's band from its measured seconds.

    Reads ONLY ``accounted_s`` and ``query_duration_s`` — a band
    label stored on (or planted into) the receipt is never read,
    so a receipt cannot carry a band its own measurements do not
    support.
    """
    accounted = float(receipt.get("accounted_s") or 0.0)
    duration = float(receipt.get("query_duration_s") or 0.0)
    if duration <= 0.0:
        return BAND_NONE_ACCOUNTED
    return originality_band(accounted / duration)


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageReceipt:
    """Typed evidence record for one sample/lineage scan.

    Every claimed source carries its regions (both time ranges,
    the regional coherent fraction with numerator/denominator);
    the band derives from the accounted seconds at scan time.
    """

    status: str
    schema: str = LINEAGE_SCHEMA
    source: str = "devon_local_fingerprint"
    source_label: str = "Devon's local fingerprint catalog"
    match_class: str = MATCH_CLASS_SAMPLE_LINEAGE
    catalog_version: str = ""
    catalog_track_count: int | None = None
    catalog_hash_count: int | None = None
    query_hash: str = ""
    query_fingerprint_count: int = 0
    query_duration_s: float = 0.0
    query_identity: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, Any], ...] = ()
    accounted_s: float = 0.0
    unaccounted_s: float = 0.0
    accounted_share: float = 0.0
    originality_band: str = BAND_NONE_ACCOUNTED
    unaccounted_regions: tuple[dict[str, Any], ...] = ()
    sub_floor_evidence: tuple[dict[str, Any], ...] = ()
    method: dict[str, Any] = field(default_factory=dict)
    rights_note: str = ""
    scope_note: str = LINEAGE_SCOPE_NOTE
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the receipt."""
        return {
            "schema": self.schema,
            "status": self.status,
            "source": self.source,
            "source_label": self.source_label,
            "match_class": self.match_class,
            "catalog_version": self.catalog_version,
            "catalog_track_count": self.catalog_track_count,
            "catalog_hash_count": self.catalog_hash_count,
            "query_hash": self.query_hash,
            "query_fingerprint_count": self.query_fingerprint_count,
            "query_duration_s": self.query_duration_s,
            "query_identity": dict(self.query_identity),
            "sources": [dict(s) for s in self.sources],
            "accounted_s": self.accounted_s,
            "unaccounted_s": self.unaccounted_s,
            "accounted_share": self.accounted_share,
            "originality_band": self.originality_band,
            "unaccounted_regions": [
                dict(r) for r in self.unaccounted_regions],
            "sub_floor_evidence": [
                dict(e) for e in self.sub_floor_evidence],
            "method": dict(self.method),
            "rights_note": self.rights_note,
            "scope_note": self.scope_note,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# windowed region analysis
# ---------------------------------------------------------------------------

def _region_strength(votes: int, in_region: int,
                     fraction: float) -> dict[str, Any]:
    """Name a region's strength by the quantity computed."""
    return {
        "name": REGION_SCORE_NAME,
        "value": round(float(fraction), 3),
        "numerator": int(votes),
        "denominator": int(in_region),
        "unit": REGION_SCORE_UNIT,
        "basis": REGION_SCORE_BASIS,
        "calibration": "uncalibrated — not a probability",
    }


def _runs_for_track(
        evidence: list[tuple[int, int, int, int]],
        ) -> list[dict[str, Any]]:
    """Split one track's offset evidence into per-bin fragments.

    ``evidence`` rows are (delta_ms, query_ms, reference_ms, hash)
    — the same rows class-1 voting consumes, kept per bin here.
    A fragment is one offset bin's worth of consecutive anchors;
    fragments are NOT regions yet — one placement's fragments
    merge in :func:`_merge_fragments` before floors apply.
    """
    bins: dict[int, dict[tuple[int, int], tuple[int, int, int, int]]] = {}
    for delta, q_ms, r_ms, h in evidence:
        slot = bins.setdefault(int(delta // OFFSET_BIN_MS), {})
        key = (h, q_ms)
        if key not in slot:
            slot[key] = (delta, q_ms, r_ms, h)
    fragments: list[dict[str, Any]] = []
    for slot in bins.values():
        votes = sorted(slot.values(), key=lambda v: v[1])
        start = 0
        for i in range(1, len(votes) + 1):
            if i < len(votes) and \
                    votes[i][1] - votes[i - 1][1] <= REGION_GAP_MS:
                continue
            run = votes[start:i]
            start = i
            deltas = sorted(v[0] for v in run)
            fragments.append({
                "q0_ms": run[0][1],
                "q1_ms": run[-1][1],
                "median_delta_ms": deltas[len(deltas) // 2],
                "vote_keys": {(v[3], v[1]) for v in run},
            })
    return fragments


def _edge_trim(keys: set[tuple[int, int]],
               keep: dict[int, bool]) -> set[tuple[int, int]]:
    """Drop vote keys in edge buckets ``keep`` marks False.

    Only the prefix/suffix runs of False buckets are dropped —
    interior False buckets (a quiet bar inside the segment) stay.
    """
    if not keys:
        return keys
    bucket_ms = int(EDGE_BUCKET_S * 1000)
    buckets = sorted({q // bucket_ms for _h, q in keys})
    first, last = buckets[0], buckets[-1]
    while first <= last and not keep.get(first, False):
        first += 1
    while last >= first and not keep.get(last, False):
        last -= 1
    if first > last:
        return keys
    return {(h, q) for h, q in keys
            if first * bucket_ms <= q < (last + 1) * bucket_ms}


def _trim_to_dense_core(keys: set[tuple[int, int]],
                        anchor_hist: dict[int, int]
                        ) -> set[tuple[int, int]]:
    """Trim a merged region's vote keys to its coherent core.

    Two edge trims, interior never touched: (1) fraction profile —
    an edge bucket whose coherent fraction (bucket votes ÷ bucket
    query landmarks) falls under :data:`EDGE_FRACTION_SHARE` of
    the region's median bucket fraction is chained chance;
    (2) density — an edge bucket under :data:`EDGE_DENSITY_SHARE`
    of the median bucket vote count is too sparse to be part of
    the segment's evidence.
    """
    if not keys:
        return keys
    bucket_ms = int(EDGE_BUCKET_S * 1000)
    counts: dict[int, int] = {}
    for _h, q in keys:
        counts[q // bucket_ms] = counts.get(q // bucket_ms, 0) + 1
    landmarks: dict[int, int] = {}
    for ms, n in anchor_hist.items():
        landmarks[ms // bucket_ms] = \
            landmarks.get(ms // bucket_ms, 0) + n
    fractions = {b: counts.get(b, 0) / landmarks[b]
                 for b in counts if landmarks.get(b, 0) > 0}
    if fractions:
        ordered = sorted(fractions.values())
        median_f = ordered[len(ordered) // 2]
        keep = {b: f >= EDGE_FRACTION_SHARE * median_f
                for b, f in fractions.items()}
        keys = _edge_trim(keys, keep)
    if not keys:
        return keys
    counts = {}
    for _h, q in keys:
        counts[q // bucket_ms] = counts.get(q // bucket_ms, 0) + 1
    nonzero = sorted(c for c in counts.values() if c > 0)
    median = nonzero[len(nonzero) // 2] if nonzero else 0
    threshold = max(EDGE_MIN_BUCKET_VOTES,
                    EDGE_DENSITY_SHARE * median)
    keep = {b: c >= threshold for b, c in counts.items()}
    return _edge_trim(keys, keep)


def _merge_fragments(fragments: list[dict[str, Any]],
                     anchor_hist: dict[int, int],
                     query_pairs: list[tuple[int, int]] | None = None,
                     ) -> list[dict[str, Any]]:
    """Merge one track's fragments into candidate regions and
    apply the sample floors.

    Two fragments merge when their query spans overlap (or nearly
    abut) AND their median offsets agree within
    :data:`FRAGMENT_DELTA_TOLERANCE_MS` — the signature of one
    placement split across neighbouring offset bins by the STFT
    grid offset. Merged statistics are recomputed from the union
    of vote keys, so a query landmark counts once however many
    fragments it voted in; the merged region is then trimmed to
    its dense core (:func:`_trim_to_dense_core`). Each returned
    region carries ``claimed`` + ``failed_floors`` verdicts.
    """
    ordered = sorted(fragments, key=lambda f: f["q0_ms"])
    merged: list[dict[str, Any]] = []
    for frag in ordered:
        target = None
        for cand in merged:
            if frag["q0_ms"] <= cand["q1_ms"] + \
                    int(FRAGMENT_SPAN_GAP_S * 1000) and \
                    abs(frag["median_delta_ms"]
                        - cand["median_delta_ms"]) \
                    <= FRAGMENT_DELTA_TOLERANCE_MS:
                target = cand
                break
        if target is None:
            merged.append({
                "q0_ms": frag["q0_ms"], "q1_ms": frag["q1_ms"],
                "median_delta_ms": frag["median_delta_ms"],
                "vote_keys": set(frag["vote_keys"]),
                "delta_votes": [(frag["median_delta_ms"],
                                 len(frag["vote_keys"]))],
            })
        else:
            target["vote_keys"] |= frag["vote_keys"]
            target["q0_ms"] = min(target["q0_ms"], frag["q0_ms"])
            target["q1_ms"] = max(target["q1_ms"], frag["q1_ms"])
            target["delta_votes"].append(
                (frag["median_delta_ms"], len(frag["vote_keys"])))
            total = sum(n for _d, n in target["delta_votes"])
            target["median_delta_ms"] = int(
                sum(d * n for d, n in target["delta_votes"])
                / max(1, total))
    regions: list[dict[str, Any]] = []
    for cand in merged:
        keys = _trim_to_dense_core(cand["vote_keys"], anchor_hist)
        if not keys:
            continue
        q0 = min(q for _h, q in keys)
        q1 = max(q for _h, q in keys)
        span_s = (q1 - q0) / 1000.0
        n_votes = len(keys)
        types = len({h for h, _q in keys})
        in_region = _in_span(anchor_hist, q0, q1)
        fraction = n_votes / max(1, in_region)
        in_region_types = len({h for q, h in (query_pairs or [])
                               if q0 <= q <= q1})
        type_recall = types / max(1, in_region_types)
        med_delta = cand["median_delta_ms"]
        failed: list[str] = []
        if span_s < MIN_REGION_SPAN_S:
            failed.append(
                f"span {span_s:.1f}s < {MIN_REGION_SPAN_S:.1f}s "
                "sample floor")
        if n_votes < MIN_REGION_VOTES:
            failed.append(
                f"{n_votes} coherent votes < {MIN_REGION_VOTES} "
                "sample floor")
        if types < MIN_REGION_TYPES:
            failed.append(
                f"{types} distinct landmark types < "
                f"{MIN_REGION_TYPES} sample floor")
        if fraction < MIN_REGION_FRACTION:
            failed.append(
                f"regional coherent fraction {fraction:.3f} < "
                f"{MIN_REGION_FRACTION:.3f} sample floor")
        if type_recall < MIN_REGION_TYPE_RECALL:
            failed.append(
                f"regional type recall {type_recall:.3f} < "
                f"{MIN_REGION_TYPE_RECALL:.3f} sample floor")
        regions.append({
            "query_start_s": round(q0 / 1000.0, 3),
            "query_end_s": round(q1 / 1000.0, 3),
            "reference_start_s": round(
                (q0 + med_delta) / 1000.0, 3),
            "reference_end_s": round(
                (q1 + med_delta) / 1000.0, 3),
            "duration_s": round(span_s, 3),
            "coherent_votes": n_votes,
            "distinct_types": types,
            "in_region_landmarks": in_region,
            "regional_type_recall": round(type_recall, 4),
            "vote_keys": keys,
            "match_strength": _region_strength(
                n_votes, in_region, fraction),
            "claimed": not failed,
            "failed_floors": failed,
        })
    return regions


def _in_span(anchor_hist: dict[int, int], q0: int, q1: int) -> int:
    """Distinct query landmarks anchored in [q0, q1].

    ``anchor_hist`` maps each query anchor ms to the number of
    distinct (hash, anchor) pairs at that anchor; this sums the
    window. (Kept O(window) — catalog-scale queries are small.)
    """
    return sum(count for ms, count in anchor_hist.items()
               if q0 <= ms <= q1)


def _union_seconds(spans: list[tuple[float, float]]) -> float:
    """Measure of the union of [start, end] spans."""
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def _complement(spans: list[tuple[float, float]], duration: float
                ) -> list[dict[str, Any]]:
    """The unaccounted spans of [0, duration] given claimed spans."""
    out: list[dict[str, Any]] = []
    cursor = 0.0
    for s, e in sorted(spans):
        if s - cursor >= 0.1:
            out.append({"start_s": round(cursor, 3),
                        "end_s": round(s, 3),
                        "duration_s": round(s - cursor, 3)})
        cursor = max(cursor, e)
    if duration - cursor >= 0.1:
        out.append({"start_s": round(cursor, 3),
                    "end_s": round(duration, 3),
                    "duration_s": round(duration - cursor, 3)})
    return out


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

def scan_lineage_hashes(
        hashes: Sequence[tuple[int, float]], *,
        duration_s: float,
        db: FingerprintDB | None = None,
        exclude_track_id: int | None = None,
        query_identity: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Lineage scan over a fingerprint (the scan's core).

    ``hashes`` is a query fingerprint — from a file via
    :func:`scan_lineage_db`, or an enrolled track's stored
    fingerprint for catalog-internal lineage. Never raises.
    """
    db = db or default_fingerprint_db()
    try:
        return _scan(hashes, duration_s=float(duration_s or 0.0),
                     db=db, exclude_track_id=exclude_track_id,
                     query_identity=dict(query_identity or {}))
    except Exception as exc:  # noqa: BLE001
        _log.debug("lineage scan failed", exc_info=True)
        reason = f"lineage scan failed: {exc}"
        receipt = LineageReceipt(
            status="unavailable", reason=reason,
            rights_note=LOCAL_RIGHTS_NOTE).to_dict()
        return {"ok": False, "status": "unavailable",
                "reason": reason, "sources": [],
                "accounted_s": 0.0, "unaccounted_s": 0.0,
                "accounted_share": 0.0,
                "originality_band": BAND_NONE_ACCOUNTED,
                "unaccounted_regions": [], "sub_floor_evidence": [],
                "query_identity": {}, "receipt": receipt}


def _scan(hashes: Sequence[tuple[int, float]], *,
          duration_s: float, db: FingerprintDB,
          exclude_track_id: int | None,
          query_identity: dict[str, Any]) -> dict[str, Any]:
    catalog = db.catalog_snapshot()
    query_hash = _query_fingerprint_hash(hashes)
    analyzed = min(duration_s, FINGERPRINT_WINDOW_SECONDS) \
        if duration_s > 0 else 0.0
    if analyzed <= 0 and hashes:
        analyzed = min(max(t for _h, t in hashes) + 1.0,
                       FINGERPRINT_WINDOW_SECONDS)
    method = {
        "name": "windowed landmark offset alignment",
        "basis": (
            "per enrolled track, query landmark hashes are grouped "
            "by 50 ms reference-offset bin; consecutive anchors "
            "within one bin form a fragment; fragments whose query "
            "spans overlap and whose median offsets agree within "
            "250 ms merge into one region; a region clearing the "
            "sample floors is a claimed sample region. Reference "
            "fingerprints are the enrolled fingerprints as stored "
            "(the shipped enrollment window per recording)"),
        "offset_bin_ms": OFFSET_BIN_MS,
        "region_gap_ms": REGION_GAP_MS,
        "floors": {
            "min_span_s": MIN_REGION_SPAN_S,
            "min_votes": MIN_REGION_VOTES,
            "min_distinct_types": MIN_REGION_TYPES,
            "min_regional_fraction": MIN_REGION_FRACTION,
            "min_regional_type_recall": MIN_REGION_TYPE_RECALL,
        },
    }
    if not hashes:
        reason = ("no fingerprints extracted — audio too quiet or "
                  "too short")
        receipt = LineageReceipt(
            status="unavailable", reason=reason,
            catalog_version=str(catalog.get("version") or ""),
            catalog_track_count=int(catalog.get("tracks") or 0),
            catalog_hash_count=int(catalog.get("hashes") or 0),
            query_hash=query_hash, query_duration_s=round(analyzed, 3),
            method=method, rights_note=LOCAL_RIGHTS_NOTE).to_dict()
        return {"ok": False, "status": "unavailable",
                "reason": reason, "sources": [],
                "accounted_s": 0.0, "unaccounted_s": 0.0,
                "accounted_share": 0.0,
                "originality_band": BAND_NONE_ACCOUNTED,
                "unaccounted_regions": [], "sub_floor_evidence": [],
                "query_identity": query_identity, "receipt": receipt}

    occurrences = db.hash_occurrences(hashes)
    tracks = {int(t["track_id"]): t for t in db.list_tracks()}

    # distinct query landmarks per anchor ms — the in-region
    # denominator basis.
    anchor_hist: dict[int, int] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for h, t in hashes:
        key = (int(h), int(t * 1000))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        anchor_hist[key[1]] = anchor_hist.get(key[1], 0) + 1
    query_pairs = sorted((q, h) for h, q in seen_pairs)

    evidence: dict[int, list[tuple[int, int, int, int]]] = {}
    for h, t in hashes:
        q_ms = int(t * 1000)
        for tid, r_ms in occurrences.get(int(h), ()):
            if exclude_track_id is not None \
                    and tid == exclude_track_id:
                continue
            evidence.setdefault(tid, []).append(
                (int(r_ms) - q_ms, q_ms, int(r_ms), int(h)))

    sources: list[dict[str, Any]] = []
    sub_floor: list[dict[str, Any]] = []
    all_spans: list[tuple[float, float]] = []
    for tid, rows in evidence.items():
        meta = tracks.get(tid) or {}
        runs = _merge_fragments(_runs_for_track(rows), anchor_hist,
                                query_pairs)
        claimed = []
        for run in runs:
            if not run["claimed"]:
                continue
            region = dict(run)
            region.pop("vote_keys", None)
            region.pop("claimed", None)
            region.pop("failed_floors", None)
            claimed.append(region)
        # One placement can surface as two claimed regions through
        # frame-flip offset modes a frame apart; where two claimed
        # regions of one source overlap by most of the smaller
        # span they are the same placement — keep the stronger.
        claimed.sort(key=lambda r: -int(r["coherent_votes"]))
        kept: list[dict[str, Any]] = []
        for region in claimed:
            span = region["query_end_s"] - region["query_start_s"]
            duplicate = False
            for other in kept:
                overlap = min(region["query_end_s"],
                              other["query_end_s"]) - max(
                                  region["query_start_s"],
                                  other["query_start_s"])
                other_span = other["query_end_s"] \
                    - other["query_start_s"]
                if overlap >= 0.6 * min(span, other_span):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(region)
        claimed = sorted(kept, key=lambda r: r["query_start_s"])
        for region in claimed:
            all_spans.append((region["query_start_s"],
                              region["query_end_s"]))
        if claimed:
            covered = _union_seconds(
                [(r["query_start_s"], r["query_end_s"])
                 for r in claimed])
            sources.append({
                "track_id": tid,
                "recording_id": f"local:{tid}",
                "title": str(meta.get("title") or ""),
                "artist": str(meta.get("artist") or ""),
                "source": str(meta.get("source") or ""),
                "covered_s": round(covered, 3),
                "regions": claimed,
            })
        failed = [r for r in runs if not r["claimed"]
                  and r["coherent_votes"] >= 4]
        if failed and not claimed:
            best = max(failed, key=lambda r: r["coherent_votes"])
            sub_floor.append({
                "track_id": tid,
                "recording_id": f"local:{tid}",
                "title": str(meta.get("title") or ""),
                "artist": str(meta.get("artist") or ""),
                "query_start_s": best["query_start_s"],
                "query_end_s": best["query_end_s"],
                "reference_start_s": best["reference_start_s"],
                "reference_end_s": best["reference_end_s"],
                "coherent_votes": best["coherent_votes"],
                "distinct_types": best["distinct_types"],
                "failed_floors": best["failed_floors"],
            })
    sources.sort(key=lambda s: (-float(s["covered_s"]),
                                str(s.get("title") or "").casefold()))
    sub_floor.sort(key=lambda e: -int(e["coherent_votes"]))

    accounted = _union_seconds(all_spans)
    accounted = min(accounted, analyzed) if analyzed > 0 else accounted
    share = accounted / analyzed if analyzed > 0 else 0.0
    band = originality_band(share)
    unaccounted_regions = _complement(all_spans, analyzed) \
        if analyzed > 0 else []
    status = "lineage_found" if sources else "no_lineage_found"
    if sources:
        reason = ""
    elif int(catalog.get("tracks") or 0) == 0:
        reason = ("the local catalog is empty — no enrolled source "
                  "could be found because none is enrolled")
    else:
        reason = (
            "no enrolled source found in this work (searched "
            f"{int(catalog.get('tracks') or 0)} enrolled track(s); "
            "no region of the work aligned with an enrolled "
            "recording above the sample floors)")
    receipt = LineageReceipt(
        status=status, reason=reason,
        catalog_version=str(catalog.get("version") or ""),
        catalog_track_count=int(catalog.get("tracks") or 0),
        catalog_hash_count=int(catalog.get("hashes") or 0),
        query_hash=query_hash,
        query_fingerprint_count=len(hashes),
        query_duration_s=round(analyzed, 3),
        query_identity=query_identity,
        sources=tuple(sources),
        accounted_s=round(accounted, 3),
        unaccounted_s=round(max(0.0, analyzed - accounted), 3),
        accounted_share=round(share, 4),
        originality_band=band,
        unaccounted_regions=tuple(unaccounted_regions),
        sub_floor_evidence=tuple(sub_floor),
        method=method,
        rights_note=LOCAL_RIGHTS_NOTE,
    ).to_dict()
    return {"ok": True, "status": status, "reason": reason,
            "sources": sources,
            "accounted_s": round(accounted, 3),
            "unaccounted_s": round(max(0.0, analyzed - accounted), 3),
            "accounted_share": round(share, 4),
            "originality_band": band,
            "unaccounted_regions": unaccounted_regions,
            "sub_floor_evidence": sub_floor,
            "query_identity": query_identity,
            "receipt": receipt}


def scan_lineage_db(path: str | os.PathLike[str], *,
                    db: FingerprintDB | None = None
                    ) -> dict[str, Any]:
    """Sample/lineage scan of one audio file. Never raises.

    If the file IS an enrolled recording (class-1 acceptance over
    the shipped voting), that recording is excluded as its own
    source and named on the receipt as the query identity — a work
    does not "sample" itself in a lineage record.
    """
    db = db or default_fingerprint_db()
    try:
        hashes = fingerprint(path, max_hashes=SCAN_MAX_HASHES)
    except AudioReadError as exc:
        reason = str(exc)
        receipt = LineageReceipt(
            status="unavailable", reason=reason,
            rights_note=LOCAL_RIGHTS_NOTE).to_dict()
        return {"ok": False, "status": "unavailable",
                "reason": reason, "sources": [],
                "accounted_s": 0.0, "unaccounted_s": 0.0,
                "accounted_share": 0.0,
                "originality_band": BAND_NONE_ACCOUNTED,
                "unaccounted_regions": [], "sub_floor_evidence": [],
                "query_identity": {}, "receipt": receipt}
    except Exception as exc:  # noqa: BLE001
        record_failure("lineage.scan_fingerprint", exc)
        reason = f"lineage scan failed: {exc}"
        receipt = LineageReceipt(
            status="unavailable", reason=reason,
            rights_note=LOCAL_RIGHTS_NOTE).to_dict()
        return {"ok": False, "status": "unavailable",
                "reason": reason, "sources": [],
                "accounted_s": 0.0, "unaccounted_s": 0.0,
                "accounted_share": 0.0,
                "originality_band": BAND_NONE_ACCOUNTED,
                "unaccounted_regions": [], "sub_floor_evidence": [],
                "query_identity": {}, "receipt": receipt}
    duration = _media_duration(path)
    exclude: int | None = None
    identity: dict[str, Any] = {}
    try:
        ranked = db.rank_candidates(path, limit=1)
        if ranked and FingerprintDB._clears_acceptance(
                ranked, MATCH_THRESHOLD):
            top = ranked[0]
            windows = top.get("evidence_windows") or []
            window_span = 0.0
            if windows:
                window_span = float(windows[0]["query_end_s"]) \
                    - float(windows[0]["query_start_s"])
            analyzed = min(duration, FINGERPRINT_WINDOW_SECONDS) \
                if duration > 0 else 0.0
            # Identity means the recording accounts for the work,
            # not merely that a winning bin exists: a host whose
            # first seconds are a long sample can clear whole-track
            # acceptance on its prefix alone (measured: a 30 s host
            # carrying two 6 s excerpts of one source was accepted
            # as that source). The winner's evidence window must
            # cover at least IDENTITY_WINDOW_SHARE of the work.
            if analyzed > 0 and window_span >= \
                    IDENTITY_WINDOW_SHARE * analyzed:
                exclude = int(top["track_id"])
                identity = {
                    "recording_id": str(
                        top.get("recording_id") or ""),
                    "title": str(top.get("title") or ""),
                    "artist": str(top.get("artist") or ""),
                    "note": ("the query is itself this enrolled "
                             "recording (class-1 identification "
                             "whose evidence window covers the "
                             "work); it is excluded as its own "
                             "lineage source"),
                }
    except Exception as exc:  # noqa: BLE001
        record_failure("lineage.query_identity", exc)
    return scan_lineage_hashes(
        hashes, duration_s=duration, db=db,
        exclude_track_id=exclude, query_identity=identity)


#: Backwards-friendly short name.
scan_lineage = scan_lineage_db


# ---------------------------------------------------------------------------
# rendering — one wording for every surface
# ---------------------------------------------------------------------------

def render_lineage_lines(result: dict[str, Any]) -> list[str]:
    """Render a lineage scan for chat/CLI surfaces.

    The wording is the doctrine: sources with their regions and
    the fraction in its own units; the unaccounted remainder
    named as unaccounted; the catalog scope on every answer; a
    band never rendered as a verdict.
    """
    lines: list[str] = []
    if not result.get("ok"):
        return [f"sample/lineage scan unavailable: "
                f"{result.get('reason') or 'unknown reason'}"]
    identity = result.get("query_identity") or {}
    if identity:
        lines.append(
            f"this audio is itself enrolled as {identity.get('artist')}"
            f" — {identity.get('title')} (excluded as its own source)")
    sources = result.get("sources") or []
    receipt = result.get("receipt") or {}
    duration = float(receipt.get("query_duration_s") or 0.0)
    if sources:
        for src in sources:
            regions = src.get("regions") or []
            lines.append(
                f"🧬 {src.get('artist')} — {src.get('title')}: "
                f"sampled in {len(regions)} region(s), "
                f"{src.get('covered_s')}s of this work")
            for region in regions:
                strength = region.get("match_strength") or {}
                lines.append(
                    f"  query {region.get('query_start_s')}–"
                    f"{region.get('query_end_s')}s ↔ source "
                    f"{region.get('reference_start_s')}–"
                    f"{region.get('reference_end_s')}s "
                    f"(regional coherent fraction "
                    f"{strength.get('value')} — "
                    f"{strength.get('numerator')} of "
                    f"{strength.get('denominator')} in-region query "
                    f"landmark hashes; uncalibrated, not a "
                    f"probability)")
    else:
        lines.append(
            "no enrolled source found in this work — no region of "
            "it aligned with Devon's enrolled catalog above the "
            "sample floors")
    for ev in result.get("sub_floor_evidence") or []:
        lines.append(
            f"below the sample floor (not claimed as a sample): "
            f"{ev.get('artist')} — {ev.get('title')} — aligned "
            f"evidence at query {ev.get('query_start_s')}–"
            f"{ev.get('query_end_s')}s "
            f"({'; '.join(ev.get('failed_floors') or [])})")
    band = str(result.get("originality_band") or "")
    lines.append(
        f"accounted for by enrolled sources: "
        f"{result.get('accounted_s')}s of {duration}s "
        f"({float(result.get('accounted_share') or 0.0):.0%}) — "
        f"originality band: {band.replace('_', ' ')} "
        f"({BAND_LABELS.get(band, '')}); unaccounted: "
        f"{result.get('unaccounted_s')}s")
    lines.append(LINEAGE_SCOPE_NOTE)
    return lines
