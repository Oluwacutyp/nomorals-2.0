"""Niche discovery + the shared niche overlay — WS4 Pass 4.

The census (``~/workspace/ws4-video-census/CENSUS.md`` §3.3, §4)
designs niche growth as a *proposal* discipline, because the
shorts pipeline publishes under the owner's name — its vocabulary
changes are his call:

* **Discovery** clusters the exploration field (candidates +
  trend claims, all stored records — membership is auditable,
  never a model's impression) over two glue facts: candidates
  sighted under the same beat, and candidates evidenced by the
  same result URL (a channel and the topic phrases one video
  evidenced arrive together). A cluster becomes a **niche
  proposal** only when its evidence clears the bar below; its
  record carries the evidence pack (members with their sighting
  inputs, backing claims, exemplar citations, the velocity series,
  the score WITH its inputs, and the nearest-existing-niche
  distance with the why-not-existing note). A cluster the evidence
  cannot carry produces NO proposal, and the discovery report's
  ``declined`` list names it with the reason — P2's discipline.
  Doc 24 PG4: the decline is ALSO preserved in the store as an
  eliminated branch (the report is per-run; the branch is the
  durable loser record — the cluster, its evidence snapshot, the
  reason, and the existing niche/proposal that superseded it
  where one did), graded from the cluster's OWN evidence.
* **The overlay** is the persisted proposal set in the
  ExploreStore beside the code registry (census §4): the seven
  hand-written plugins in ``media/contentops/niches/`` are the
  production FLOOR — always in force, never edited by this
  module. The effective niche set consumers read is
  ``floor ∪ adopted overlay entries`` (:func:`effective_niches`,
  the read seam WS3's content cell calls). A proposed niche is in
  NO effective set and steers NO fetch: proposal is not adoption.
* **Adoption is an owner act** (:func:`adopt_niche`): it flips
  the overlay record to ``adopted`` and scaffold-generates a
  DRAFT plugin through the niches package's own scaffold
  (``niches/scaffold.py``) into the drafts directory
  (:func:`drafts_dir`) — outside the package's importable modules
  and outside ``_SHIPPED_MODULES``, so the draft cannot silently
  become a production plugin. It goes live only when a human
  reviews it, fills its TODO sections, moves it into the niches
  package proper and adds its slug to ``_SHIPPED_MODULES``.
  Until then the adopted niche serves consumers through the
  overlay itself — :func:`overlay_plugin` materialises a
  data-driven generic plugin from the record (census §4's second
  materialisation), never through the draft file loading.
* **Dismissal sticks** (:func:`dismiss_niche`): a dismissed
  cluster is not re-proposed on the same evidence. Discovery
  re-proposes it only when the evidence has genuinely moved —
  a member sighted again after the dismissal, or a new member
  joining the cluster.

The evidence bar for a proposal (ALL must hold; the first failure
is the declined reason):

1. the cluster has at least ``MIN_MEMBERS`` member candidates;
2. at least one member is *promoted* (P2's evidence verdict) or a
   current *trend* claim backs the cluster;
3. total member sightings >= ``MIN_SIGHTINGS``;
4. members were sighted across >= ``MIN_SWEEPS`` distinct sweeps;
5. the cluster is not substantially an existing niche (derived
   slug equal to a floor/adopted slug, or token overlap with one
   >= ``COVERED_JACCARD``) — a cluster that IS ``finance_facts``
   is a finding about that niche, not a new one.

Nothing in here ever raises into a caller: discovery reports its
own failure in its report; the acts answer ``ok`` + a reason.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ..failures import record_failure
from .store import (
    BRANCH_NICHE_CLUSTER,
    CANDIDATE_PROMOTED,
    ELIM_DECLINED,
    NICHE_ADOPTED,
    NICHE_DISMISSED,
    NICHE_PROPOSED,
    ExploreStore,
)
from .trends import (
    current_claims,
    proof_grade_for_branch,
    source_class_of,
)

_log = get_logger(__name__)

__all__ = [
    "COVERED_JACCARD",
    "MIN_MEMBERS",
    "MIN_SIGHTINGS",
    "MIN_SWEEPS",
    "adopt_niche",
    "discover_niches",
    "dismiss_niche",
    "drafts_dir",
    "effective_niches",
    "floor_niches",
    "overlay_plugin",
    "proposal_board_order",
]

#: The proposal evidence bar (module docstring, §3.3 scoring:
#: cluster size, corroboration, velocity, distance, beat fit —
#: the inputs are stored on every proposal, per the census rule
#: that a score whose inputs aren't stored is not a score).
MIN_MEMBERS = 2
MIN_SIGHTINGS = 3
MIN_SWEEPS = 2
COVERED_JACCARD = 0.6

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "your",
    "you", "with", "that", "this", "it", "is", "how", "why", "what",
}


# ── the floor (the code registry, read-only here) ────────────────


def _code_registry() -> Any:
    """The hand-written niche registry — imported lazily so the
    explore package never pays (or risks) the plugin bootstrap at
    import time. This module READS it; it never writes to it."""
    from ..contentops.niches import registry as code_registry

    return code_registry


def floor_niches() -> list[dict[str, Any]]:
    """The production floor as records: the shipped hand-written
    plugins, always in force. Origin ``floor``, status ``live``."""
    reg = _code_registry()
    out: list[dict[str, Any]] = []
    for slug in reg.list_niches():
        plugin = reg.get(slug)
        out.append({
            "slug": slug,
            "thesis": plugin.thesis,
            "cadence": float(plugin.cadence),
            "origin": "floor",
            "status": "live",
        })
    return out


def effective_niches(store: ExploreStore | None = None,
                     ) -> list[dict[str, Any]]:
    """THE shared-registry read: the effective niche set =
    floor ∪ adopted overlay entries (floor first, floor wins any
    slug collision). This is the seam WS3's content cell and
    contentops consumers read niches through — a *proposed*
    niche appears nowhere in it, by construction."""
    out = floor_niches()
    seen = {row["slug"] for row in out}
    if store is not None:
        for rec in store.niche_proposals(status=NICHE_ADOPTED):
            slug = rec.get("slug") or ""
            if not slug or slug in seen:
                continue
            seen.add(slug)
            out.append({
                "slug": slug,
                "thesis": rec.get("thesis") or "",
                "cadence": float(rec.get("cadence") or 1.0),
                "origin": "overlay",
                "status": NICHE_ADOPTED,
            })
    return out


# ── text helpers ─────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(t) >= 3 and t not in _STOPWORDS}


def _slugify(text: str) -> str:
    """A niche slug (``a-z0-9_``) from free text — the code
    registry's slug grammar."""
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower())
    return slug.strip("_") or "untitled"


def _human_name(slug: str) -> str:
    return slug.replace("_", " ").strip().title()


# ── clustering over stored findings ──────────────────────────────


def _clusters(store: ExploreStore) -> list[list[dict[str, Any]]]:
    """Connected components of the candidate pool under the two
    glue facts (shared beat, shared evidence URL). Deterministic:
    entities and edges are visited in sorted key order, so the
    same store always yields the same components."""
    entities = store.candidates()
    by_key = {c.get("key") or "": c for c in entities}
    parent: dict[str, str] = {k: k for k in by_key}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_beat: dict[str, list[str]] = {}
    by_url: dict[str, list[str]] = {}
    for cand in entities:
        key = cand.get("key") or ""
        for beat in cand.get("beats") or []:
            if beat:
                by_beat.setdefault(beat, []).append(key)
        for ev in cand.get("evidence") or []:
            url = str(ev.get("url") or "")
            if url:
                by_url.setdefault(url, []).append(key)
    for index in (by_beat, by_url):
        for keys in index.values():
            for other in keys[1:]:
                union(keys[0], other)
    components: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(by_key):
        components.setdefault(find(key), []).append(by_key[key])
    return list(components.values())


def _cluster_claims(claims: list[dict[str, Any]],
                    members: list[dict[str, Any]],
                    member_keys: set[str],
                    member_beats: set[str],
                    ) -> list[dict[str, Any]]:
    """Current trend claims backing a cluster: claims whose
    backing candidate is a member, plus beat claims scoped to a
    member beat."""
    out = []
    for claim in claims:
        if claim.get("kind") != "trend":
            continue
        backing = claim.get("backing") or {}
        if backing.get("candidate_key") in member_keys or (
                claim.get("subject_kind") == "beat"
                and (claim.get("scope") in member_beats
                     or claim.get("subject") in member_beats)):
            out.append(claim)
    return out


def _cluster_evidence_refs(*, backing: list[dict[str, Any]],
                           sweeps: list[str], sightings: int,
                           slug: str) -> list[dict[str, str]]:
    """The PG1 evidence refs a niche cluster's OWN record earns
    (doc 24 PG4 — the refs an eliminated branch is graded from):
    backing trend claims are empirical sources; member sightings
    spread across two or more sweeps are empirical sources; any
    other evaluated cluster carries the evaluation itself — the
    inexpensive filter — as a logical check. A raw sighting COUNT
    is never a ref: a large single-sweep burst is weaker evidence
    than a small corroborated one, and the refs say so."""
    refs: list[dict[str, str]] = []
    for claim in backing or []:
        key = claim.get("claim_key") or ""
        if key:
            refs.append({"kind": "empirical_source",
                         "ref": f"claim:{key}"})
    if not refs and len(sweeps or []) >= 2:
        refs = [{"kind": "empirical_source", "ref": f"sweep:{s}"}
                for s in list(sweeps)[:8]]
    if not refs:
        refs = [{"kind": "logical_check",
                 "ref": f"niche-cluster:{slug}"}]
    return refs


def _nearest_existing(slug: str,
                      existing: list[dict[str, Any]],
                      ) -> tuple[str, float]:
    """(slug, token distance) of the nearest existing niche —
    1.0 is the same name. Distance is stored on the pack so the
    why-not-existing note is auditable, not asserted."""
    mine = set(slug.split("_"))
    best_slug, best = "", 0.0
    for row in existing:
        theirs = set((row.get("slug") or "").split("_"))
        if not mine or not theirs:
            continue
        jaccard = len(mine & theirs) / len(mine | theirs)
        if jaccard > best:
            best_slug, best = row.get("slug") or "", jaccard
    return best_slug, round(best, 3)


def _derive_slug(members: list[dict[str, Any]]) -> str:
    """The cluster's name: its dominant beat (the field it was
    sighted in), else the most-sighted member's name."""
    beat_counts: dict[str, int] = {}
    for member in members:
        for beat in member.get("beats") or []:
            if beat:
                beat_counts[beat] = beat_counts.get(beat, 0) + 1
    if beat_counts:
        dominant = sorted(
            beat_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return _slugify(dominant)
    top = sorted(members,
                 key=lambda m: (-int(m.get("sightings") or 0),
                                m.get("key") or ""))[0]
    return _slugify(str(top.get("name") or ""))


# ── discovery ────────────────────────────────────────────────────


def discover_niches(store: ExploreStore, *,
                    now: float = 0.0) -> dict[str, Any]:
    """Cluster the field and write the proposals the evidence
    carries. Idempotent in P2's supersede spirit: an existing
    *proposed* record for the same cluster is refreshed in place
    (evidence pack + score updated, created_at kept); an adopted
    one is left alone; a dismissed one is re-proposed ONLY on
    genuinely new evidence. Returns the discovery report —
    ``proposed`` / ``refreshed`` records written, and ``declined``
    naming every evaluated cluster the evidence could not carry,
    with the reason. Never raises."""
    now = now or time.time()
    report: dict[str, Any] = {
        "ok": True, "at": now, "proposed": [], "refreshed": [],
        "declined": [], "clusters": 0,
    }
    try:
        claims = current_claims(store, now=now)
        existing_effective = effective_niches(store)
        existing_slugs = {row["slug"] for row in existing_effective}
        baselines = store.beat_baselines()
        for members in _clusters(store):
            members = sorted(
                members,
                key=lambda m: (-int(m.get("sightings") or 0),
                               m.get("key") or ""))
            member_keys = {m.get("key") or "" for m in members}
            member_beats = {b for m in members
                            for b in (m.get("beats") or []) if b}
            backing = _cluster_claims(
                claims, members, member_keys, member_beats)
            promoted = [m for m in members
                        if m.get("status") == CANDIDATE_PROMOTED]
            evaluated = len(members) >= MIN_MEMBERS or promoted \
                or backing
            if not evaluated:
                continue
            report["clusters"] += 1
            slug = _derive_slug(members)
            sightings = sum(int(m.get("sightings") or 0)
                            for m in members)
            sweeps = sorted({s for m in members
                             for s in (m.get("sweeps") or []) if s})

            def decline(reason: str, _slug: str = slug,
                        _members: list = members,
                        _member_keys: set = member_keys,
                        _sightings: int = sightings,
                        _sweeps: list = sweeps,
                        _backing: list = backing,
                        superseded_by: Any = None) -> None:
                report["declined"].append({
                    "subject": _slug,
                    "subject_kind": "niche_cluster",
                    "members": [m.get("name") or ""
                                for m in _members],
                    "reason": reason,
                })
                # PG4: preserve the losing branch durably — the
                # report above is per-run and vanishes; the store
                # record keeps the cluster, its evidence snapshot,
                # its own derived grade, the reason, and the
                # winner that superseded it where one did.
                branch: dict[str, Any] = {
                    "branch_kind": BRANCH_NICHE_CLUSTER,
                    "subject": _slug,
                    "subject_key": _slug,
                    "subject_kind": "niche_cluster",
                    "kind": "niche_cluster",
                    "text": (f"niche cluster {_slug} — "
                             f"{len(_members)} member(s)"),
                    "members": [
                        {"key": m.get("key") or "",
                         "name": m.get("name") or ""}
                        for m in _members],
                    "member_keys": sorted(_member_keys),
                    "evidence": {
                        "sightings": _sightings,
                        "sweeps": list(_sweeps),
                        "backing_claims": [
                            c.get("claim_key") or ""
                            for c in _backing],
                    },
                    "evidence_refs": _cluster_evidence_refs(
                        backing=_backing, sweeps=_sweeps,
                        sightings=_sightings, slug=_slug),
                    "outcome": ELIM_DECLINED,
                    "reason": reason,
                    "superseded_by": superseded_by,
                    "eliminated_at": now,
                    "source": "explore-niches",
                }
                branch["proof_grade"] = \
                    proof_grade_for_branch(branch).to_dict()
                store.record_eliminated_branch(branch)

            if len(members) < MIN_MEMBERS:
                decline(
                    "a single member — a niche needs a cluster, "
                    "not one source")
                continue
            if not promoted and not backing:
                decline(
                    "no promoted member and no trend claim — "
                    "watching evidence only")
                continue
            if sightings < MIN_SIGHTINGS:
                decline(
                    f"only {sightings} sighting(s) across members "
                    f"(bar: {MIN_SIGHTINGS})")
                continue
            if len(sweeps) < MIN_SWEEPS:
                decline(
                    "sighted in a single sweep — corroboration "
                    "needs two")
                continue
            nearest, distance = _nearest_existing(
                slug, existing_effective)
            if slug in existing_slugs:
                decline(
                    f"already covered by the existing niche "
                    f"{slug} — a finding about it, not a new niche",
                    superseded_by={
                        "ref": slug, "kind": "niche",
                        "text": f"the existing niche {slug}"})
                continue
            if distance >= COVERED_JACCARD:
                decline(
                    f"too close to the existing niche {nearest} "
                    f"(name-token overlap {distance}) — a finding "
                    "about it, not a new niche",
                    superseded_by={
                        "ref": nearest, "kind": "niche",
                        "text": f"the existing niche {nearest}"})
                continue
            prior = store.niche_proposal(slug)
            if prior is not None \
                    and set(prior.get("member_keys") or []) \
                    != member_keys \
                    and prior.get("status") in (
                            NICHE_PROPOSED, NICHE_ADOPTED):
                decline(
                    f"the name {slug} is already a different "
                    "cluster's proposal — rename the field before "
                    "this one can be proposed",
                    superseded_by={
                        "ref": slug, "kind": "niche_proposal",
                        "text": f"the existing proposal {slug}"})
                continue
            if prior is not None \
                    and prior.get("status") == NICHE_ADOPTED:
                continue  # already his — nothing to propose
            if prior is not None \
                    and prior.get("status") == NICHE_DISMISSED \
                    and set(prior.get("member_keys") or []) \
                    == member_keys:
                dismissed_at = float(prior.get("dismissed_at") or 0)
                moved = any(
                    float(m.get("last_seen") or 0) > dismissed_at
                    for m in members)
                if not moved:
                    decline(
                        "dismissed by the owner and the evidence "
                        "has not moved since — not re-proposed")
                    continue
            record = _compose_proposal(
                slug=slug, members=members, backing=backing,
                sightings=sightings, sweeps=sweeps,
                baselines=baselines, nearest=nearest,
                distance=distance, prior=prior, now=now)
            store.record_niche_proposal(record)
            if prior is not None \
                    and prior.get("status") == NICHE_PROPOSED:
                report["refreshed"].append(record)
            else:
                report["proposed"].append(record)
    except Exception as exc:  # noqa: BLE001 — discovery reports
        record_failure("explore.discover_niches", exc)
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _compose_proposal(*, slug: str,
                      members: list[dict[str, Any]],
                      backing: list[dict[str, Any]],
                      sightings: int, sweeps: list[str],
                      baselines: dict[str, dict[str, Any]],
                      nearest: str, distance: float,
                      prior: dict[str, Any] | None,
                      now: float) -> dict[str, Any]:
    """The proposal record: the claim-shaped niche + its evidence
    pack. Every number in the pack traces to a stored record —
    member sightings, backing claims, evidence citations, the
    per-beat velocity series, the score with its inputs."""
    classes: set[str] = set()
    instruments: set[str] = set()
    exemplars: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for member in members:
        for ev in member.get("evidence") or []:
            instrument = str(ev.get("instrument") or "")
            if instrument:
                instruments.add(instrument)
                classes.add(source_class_of(instrument))
            url = str(ev.get("url") or "")
            if url and url not in seen_urls and len(exemplars) < 5:
                seen_urls.add(url)
                exemplar: dict[str, Any] = {
                    "title": str(ev.get("title") or ""),
                    "url": url,
                    "instrument": instrument,
                }
                if ev.get("artifact_id"):
                    exemplar["artifact_id"] = ev["artifact_id"]
                exemplars.append(exemplar)
    max_velocity: float | None = None
    for claim in backing:
        basis = claim.get("basis") or {}
        for cls in basis.get("source_classes") or []:
            classes.add(str(cls))
        velocity = basis.get("velocity") or {}
        ratio = velocity.get("ratio")
        if ratio is not None:
            max_velocity = max(max_velocity or 0.0, float(ratio))
    member_beats = sorted({b for m in members
                           for b in (m.get("beats") or []) if b})
    velocity_series = {}
    for beat in member_beats:
        base = baselines.get(beat) or {}
        velocity_series[beat] = {
            "result_counts": list(base.get("result_counts") or []),
            "mean_result_count": base.get("mean_result_count", 0.0),
            "sweeps": int(base.get("sweep_count") or 0),
        }
    score_inputs = {
        "members": len(members),
        "sightings": sightings,
        "sweeps": len(sweeps),
        "source_classes": sorted(classes),
        "trend_claims": len(backing),
        "max_velocity_ratio": max_velocity,
        "nearest_existing": nearest,
        "nearest_distance": distance,
    }
    score = round(
        2.0 * len(members)
        + 0.5 * min(sightings, 20)
        + 3.0 * len(classes)
        + 4.0 * len(backing)
        + min(max_velocity or 0.0, 5.0), 3)
    field = member_beats[0] if member_beats else "the open field"
    top_names = ", ".join(
        str(m.get("name") or "") for m in members[:3])
    if nearest:
        why_not = (
            f"nearest existing niche is {nearest} (name-token "
            f"distance {distance}) — this cluster's members and "
            "field are distinct from it")
    else:
        why_not = ("no existing niche name shares vocabulary with "
                   "this cluster")
    history = list((prior or {}).get("history") or [])
    history.append({"status": NICHE_PROPOSED, "at": now})
    return {
        "slug": slug,
        "name": _human_name(slug),
        "thesis": (
            f"Short-form built from the {field} field — "
            f"{top_names} — discovered by exploration; the "
            "evidence pack is on this record."),
        "cadence": 1.0,
        "status": NICHE_PROPOSED,
        "members": [{
            "key": m.get("key") or "",
            "kind": m.get("kind") or "",
            "name": m.get("name") or "",
            "platform": m.get("platform") or "",
            "sightings": int(m.get("sightings") or 0),
            "sweeps": list(m.get("sweeps") or []),
            "beats": list(m.get("beats") or []),
        } for m in members],
        "member_keys": sorted(
            m.get("key") or "" for m in members),
        "claims": [{
            "claim_id": c.get("claim_id") or "",
            "kind": c.get("kind") or "",
            "text": c.get("text") or "",
        } for c in backing],
        "evidence_pack": {
            "sightings": sightings,
            "sweeps": sweeps,
            "source_classes": sorted(classes),
            "instruments": sorted(instruments),
            "exemplars": exemplars,
            "velocity": velocity_series,
            "score": score,
            "score_inputs": score_inputs,
            "nearest_existing": nearest,
            "nearest_distance": distance,
            "why_not_existing": why_not,
        },
        "basis": (
            f"{len(members)} member(s), {sightings} sighting(s) "
            f"across {len(sweeps)} sweep(s); source classes: "
            f"{', '.join(sorted(classes)) or 'none recorded'}; "
            f"{len(backing)} trend claim(s) backing; score "
            f"{score} from the stored inputs"),
        "created_at": float((prior or {}).get("created_at") or now),
        "updated_at": now,
        "history": history,
    }


# ── the owner's acts ─────────────────────────────────────────────


def proposal_board_order(store: ExploreStore,
                         ) -> list[dict[str, Any]]:
    """The proposed records in BOARD order (score desc, then
    slug) — the one ordering the board renders and the act
    resolver reads, so ``#n`` names the same proposal in both."""
    rows = store.niche_proposals(status=NICHE_PROPOSED)
    rows.sort(key=lambda r: (
        -float((r.get("evidence_pack") or {}).get("score") or 0),
        r.get("slug") or ""))
    return rows


def _resolve_proposal(store: ExploreStore, ref: str,
                      ) -> tuple[dict[str, Any] | None, str]:
    """Resolve an act reference: ``#n`` / ``n`` (the board's
    numbering of the proposed list) or a proposal slug."""
    ref = (ref or "").strip()
    if not ref:
        return None, "no proposal named — /explore niches lists them"
    if ref.startswith("#"):
        ref = ref[1:]
    if ref.isdigit():
        rows = proposal_board_order(store)
        idx = int(ref) - 1
        if 0 <= idx < len(rows):
            return rows[idx], ""
        return None, (f"there is no proposal #{ref} — "
                      f"{len(rows)} awaiting")
    rec = store.niche_proposal(_slugify(ref))
    if rec is None:
        rec = store.niche_proposal(ref)
    if rec is None:
        return None, f"no niche proposal matches {ref!r}"
    return rec, ""


def drafts_dir() -> Path:
    """Where adoption writes DRAFT plugins: ``NM_NICHE_DRAFTS_DIR``
    when set, else the ``drafts/`` folder beside the niches
    package — inside the package tree for discoverability, but
    NOT an importable package module and NOT in
    ``_SHIPPED_MODULES``, so the code registry's bootstrap never
    imports it. A draft goes live only when a human reviews it,
    fills its TODO sections, moves it into the niches package
    proper and adds its slug to ``_SHIPPED_MODULES``."""
    env = os.environ.get("NM_NICHE_DRAFTS_DIR")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parent.parent
            / "contentops" / "niches" / "drafts")


def adopt_niche(store: ExploreStore, ref: str, *,
                dest_dir: str | Path | None = None,
                now: float = 0.0) -> dict[str, Any]:
    """THE owner's adoption act: flip a *proposed* record to
    adopted in the overlay AND scaffold-generate its draft
    plugin. The draft is generated FIRST — if generation fails,
    the record stays proposed and the answer says why (no
    half-adopted state). Adoption changes the effective set
    through the overlay only; the code registry is never touched
    and the draft file is never imported by this act."""
    now = now or time.time()
    rec, err = _resolve_proposal(store, ref)
    if rec is None:
        return {"ok": False, "reason": err}
    slug = rec.get("slug") or ""
    if rec.get("status") != NICHE_PROPOSED:
        return {"ok": False,
                "reason": f"niche {slug} is {rec.get('status')} — "
                          "only a proposed niche can be adopted"}
    dest = Path(dest_dir) if dest_dir is not None else drafts_dir()
    note = ""
    try:
        from ..contentops.niches.scaffold import scaffold

        dest.mkdir(parents=True, exist_ok=True)
        try:
            path = scaffold(
                slug, dest_dir=dest,
                thesis=str(rec.get("thesis") or ""), cadence=float(
                    rec.get("cadence") or 1.0))
        except FileExistsError:
            path = str(dest / f"{slug}.py")
            note = "a draft was already on disk at that path"
    except Exception as exc:  # noqa: BLE001 — the act answers
        record_failure("explore.adopt_niche.scaffold", exc)
        return {"ok": False,
                "reason": f"draft generation failed for {slug}: "
                          f"{type(exc).__name__}: {exc} — the "
                          "proposal stays proposed"}
    if not store.set_niche_proposal_status(
            slug, NICHE_ADOPTED, at=now,
            fields={"draft_path": path, "adopted_at": now}):
        return {"ok": False,
                "reason": f"the overlay record for {slug} vanished "
                          "before the adoption landed"}
    _log.info("explore: niche %s ADOPTED by the owner; draft at %s",
              slug, path)
    return {"ok": True, "slug": slug,
            "record": store.niche_proposal(slug),
            "draft_path": path, "note": note}


def dismiss_niche(store: ExploreStore, ref: str, *,
                  reason: str = "",
                  now: float = 0.0) -> dict[str, Any]:
    """The owner's other act: dismiss a proposal (or retire an
    adopted niche). The record keeps its member fingerprint +
    dismissal time, so discovery does not re-propose the same
    evidence (see the module docstring)."""
    now = now or time.time()
    rec, err = _resolve_proposal(store, ref)
    if rec is None:
        return {"ok": False, "reason": err}
    slug = rec.get("slug") or ""
    if rec.get("status") == NICHE_DISMISSED:
        return {"ok": False,
                "reason": f"niche {slug} is already dismissed"}
    if not store.set_niche_proposal_status(
            slug, NICHE_DISMISSED, at=now,
            fields={"dismissed_at": now,
                    "dismiss_reason": reason or "dismissed by the "
                                                "owner"}):
        return {"ok": False,
                "reason": f"the overlay record for {slug} vanished "
                          "before the dismissal landed"}
    _log.info("explore: niche %s dismissed by the owner", slug)
    return {"ok": True, "slug": slug,
            "record": store.niche_proposal(slug)}


# ── the generic materialisation (census §4, second route) ───────


def overlay_plugin(slug: str, store: ExploreStore) -> Any:
    """A data-driven :class:`NichePlugin` for an ADOPTED overlay
    record — how contentops consumers use an adopted niche
    before (or instead of) its draft plugin being finished and
    shipped. Returns None for a floor slug (the code registry
    serves those), an unadopted proposal, or an unknown slug.
    The plugin is constructed per call and NEVER registered into
    the code registry."""
    if store is None:
        return None
    rec = store.niche_proposal(slug)
    if rec is None or rec.get("status") != NICHE_ADOPTED:
        return None

    from ..contentops.niches.base import NichePlugin, VoiceSpec

    members = rec.get("members") or []
    topics = ", ".join(str(m.get("name") or "") for m in members[:4])
    thesis = str(rec.get("thesis") or "")

    class _OverlayNiche(NichePlugin):
        # Methods only in the class body (a class body cannot
        # read the enclosing function's locals); the record's
        # data lands as instance attributes below.
        def script_prompt(self, topic: str) -> str:
            lo, hi = self.word_budget()
            return (
                "You are a short-form scriptwriter for an adopted "
                f"exploration niche. THESIS: {thesis}\n\n"
                f"TOPIC: {topic}\n\n"
                f"Write a spoken-word script, {lo}–{hi} words, "
                "for a vertical short.\n"
                "RULES:\n"
                "- FIRST LINE is the hook — a curiosity gap or a "
                "bold claim. No greetings, no setup.\n"
                "- Every 3–5 seconds add a retention beat, then "
                "pay the open loop off near the end.\n"
                "- Plain spoken text only. No stage directions, "
                "no [brackets], no emojis.\n"
                "- One idea per script. End on a line worth "
                "rewatching or sharing.\n"
                "- The field this niche was discovered in: "
                f"{topics or 'the exploration field'} — stay "
                "inside it.\n\n"
                "Return ONLY the script text.")

        def visual_strategy(self, script: str) -> Any:
            return self._beats_to_scenes(
                script,
                ("the hook frame — the topic, visualised bold",
                 "escalation — the detail that deepens the topic",
                 "the payoff frame — the takeaway, visualised"),
                style_lock=("clean vertical short-form still, "
                            "one subject, high contrast"),
                motions=("zoom-in-slow", "pan-right-slow",
                         "zoom-out-slow"),
                notes=f"{slug} overlay visual plan")

    plugin = _OverlayNiche()
    plugin.name = slug
    plugin.thesis = thesis
    plugin.cadence = float(rec.get("cadence") or 1.0)
    plugin.target_seconds = (25, 55)
    plugin.title_template = "{topic}"
    plugin.description_template = "{topic}\n\n{script}"
    plugin.hashtags = ["#shorts"]
    plugin.voice_spec = VoiceSpec(preferred_voice="", mood="neutral",
                                  intensity=3, pace_wpm=150)
    plugin.ypp_rationale = (
        "Overlay niche adopted by the owner from exploration "
        "evidence; original scripts and AI visuals, "
        "transformative commentary.")
    return plugin
