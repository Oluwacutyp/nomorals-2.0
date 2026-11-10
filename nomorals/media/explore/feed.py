"""The WS3 contracts, WS4's side (WS4 P5): exploration feeds the
content cooperative, and content outcomes flow back.

The two censuses specify the same four contracts from opposite
sides (WS3 census §9; WS4 census §5); this module is the ONE
adapter where the mapping lives, so no caller smears it:

1. **Trend claims feed.** Exploration's current claims reach the
   weekly content pipeline as findings in the exact flat-dict
   shape its ``findings=`` seam already accepts (``title`` /
   ``snippet`` / ``url`` — the only keys its drafters read),
   projected from the digestion Claim shape exactly once, here.
2. **Shared niche registry.** Content consumers read niches
   through P4's :func:`~.niches.effective_niches` seam via this
   module's read API and plugin resolver — floor ∪ adopted
   overlay; a proposed or dismissed niche is invisible to
   drafting, and no consumer learns how the overlay is stored.
3. **Artifact-cited media.** When a claim's evidence carries
   exemplar media that lives in the WS1 artifact store, the
   finding carries the artifact citation (the store's own id +
   ``artifact://`` URI) — verified resolvable against the store
   at projection time. Media flows by citation, never by copying
   bytes into the social stores; a finding whose media is not
   artifact-backed carries no citation (never a fabricated one).
4. **One measurement spine.** When the spine (``social/metrics``)
   measures a post that descends from a claim, the outcome is
   recorded back here against the claim — claim key, post
   reference, measured engagement, at — through the same honest
   join discipline the spine uses for briefs (exact content +
   platform; an ambiguous join records nothing and is reported
   ambiguous; a claim with no descended post stays outcome-less).

Rules this adapter implements where the censuses leave the edge
to the builder, stated once and pinned by the P5 suite:

* **Freshness.** Only claims whose computed standing is *fresh*
  feed. A stale claim is excluded, not labelled-and-fed: the
  census's decay rule is that a stale claim "does not linger as
  if still true", and drafting is exactly where lingering would
  do harm. Fresh spike / unverified claims DO feed — labelled in
  the finding's title and snippet and in its ``claim_kind`` key,
  so neither the pipeline's drafters nor the owner can mistake a
  single-sweep burst for a corroborated trend.
* **Merge order.** The feed APPENDS to the weekly run's own
  research findings — exploration evidence widens the run's
  evidence base; it never silently displaces the research the
  run was asked to do. A caller that injects ``findings=``
  explicitly controls the evidence base and bypasses the feed.
* **Outcomes are evidence, not promotion.** A recorded outcome
  is a fact future evaluation may read. Nothing in P2's claim
  bars or P4's niche bars changes because a claim performed
  well; no outcome recorded here promotes, sheds, adopts or
  dismisses anything.

Layering: this module is media (L4), a peer of social (L4). The
weekly pipeline cannot import it (the pipeline's composition root
installs the feed callable — the factory-inversion discipline P6
used for the cell), and the spine reaches the outcome feed
through the same lazily-resolved seam metrics already uses for
its briefs store.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .niches import effective_niches, overlay_plugin
from .store import ExploreStore
from .trends import CLAIM_SPIKE, CLAIM_UNVERIFIED, current_claims

_log = logging.getLogger(__name__)

__all__ = [
    "claim_exemplars",
    "claim_to_finding",
    "claims_as_findings",
    "make_claims_feed",
    "drafting_niches",
    "niche_drafting_context",
    "drafting_plugin_resolver",
    "feed_claim_outcome",
]

#: Finding key carrying the verified WS1 artifact citation.
ARTIFACT_URI_PREFIX = "artifact://"

#: Title prefixes that keep a below-trend claim honest everywhere
#: the finding surfaces (draft prompts, review cards, reports).
_KIND_TITLE_PREFIX = {
    CLAIM_SPIKE: "Spike: ",
    CLAIM_UNVERIFIED: "Unverified: ",
}

_MAX_EXEMPLARS = 5
_DEFAULT_FEED_LIMIT = 10


# ── contract 1 + 3: claims → findings (with artifact citations) ──


def claim_exemplars(store: ExploreStore,
                    claim: dict[str, Any]) -> list[dict[str, Any]]:
    """The exemplar media a claim's evidence carries, in citation
    shape: ``{title, url, instrument}`` plus ``artifact_id`` when
    the recorded evidence names one.

    Two evidence sources, one shape: a candidate claim reads its
    candidate's stored evidence entries (the P4 exemplar shape);
    a beat claim reads the result rows of the observations that
    back it. Capped — a citation list is evidence, not a feed."""
    out: list[dict[str, Any]] = []
    backing = claim.get("backing") or {}
    candidate_key = str(backing.get("candidate_key") or "")
    if candidate_key:
        candidate = store.candidate(candidate_key)
        for entry in (candidate or {}).get("evidence") or []:
            if not isinstance(entry, dict):
                continue
            item = {
                "title": str(entry.get("title") or ""),
                "url": str(entry.get("url") or ""),
                "instrument": str(entry.get("instrument") or ""),
            }
            if entry.get("artifact_id"):
                item["artifact_id"] = str(entry["artifact_id"])
            out.append(item)
    obs_ids = {str(o) for o in backing.get("observations") or []}
    if obs_ids:
        for obs in store.observations(limit=10000):
            if str(obs.get("obs_id") or "") not in obs_ids:
                continue
            instrument = str(obs.get("instrument") or "")
            for result in obs.get("results") or []:
                if not isinstance(result, dict):
                    continue
                out.append({
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("url") or ""),
                    "instrument": instrument,
                })
    return out[:_MAX_EXEMPLARS]


def _verified_citation(exemplars: list[dict[str, Any]],
                       artifacts: Any) -> tuple[str, list[dict[str, Any]]]:
    """(artifact_id, exemplars) with the citation VERIFIED.

    The finding may cite an artifact only when the id the
    evidence names resolves through the artifact store the
    caller injected — the citation is the store's own handle, so
    it must name something real. An id that does not resolve is
    stripped from the exemplar copies (an unverified id must
    never ride along looking like a citation); with no store
    injected, no citation is issued at all."""
    cleaned: list[dict[str, Any]] = []
    cited = ""
    for item in exemplars:
        item = dict(item)
        artifact_id = str(item.get("artifact_id") or "")
        if artifact_id:
            resolved = None
            if artifacts is not None:
                try:
                    resolved = artifacts.get(artifact_id)
                except Exception:  # noqa: BLE001 - a read never raises
                    resolved = None
            if resolved is not None and not cited:
                cited = artifact_id
            else:
                item.pop("artifact_id", None)
        cleaned.append(item)
    return cited, cleaned


def claim_to_finding(claim: dict[str, Any], store: ExploreStore,
                     *, artifacts: Any = None) -> dict[str, Any]:
    """Project ONE current claim into the pipeline's finding
    shape — the single digestion-Claim ↔ finding mapping.

    The flat keys the pipeline's consumers read (``title`` /
    ``snippet`` / ``url``) come first; the claim's identity and
    basis ride as extra keys the consumers ignore but the fan-out
    stamps onto drafts, which is what lets contract 4 join a
    measured post back to this claim. Below-trend kinds are
    labelled in the title itself."""
    kind = str(claim.get("kind") or CLAIM_UNVERIFIED)
    text = str(claim.get("text") or "").strip()
    basis = str(claim.get("confidence_basis") or "").strip()
    exemplars = claim_exemplars(store, claim)
    cited, exemplars = _verified_citation(exemplars, artifacts)
    prefix = _KIND_TITLE_PREFIX.get(kind, "")
    snippet = f"Exploration {kind} claim"
    if claim.get("standing"):
        snippet += f" (standing: {claim['standing']})"
    snippet += f". {basis}" if basis else "."
    finding: dict[str, Any] = {
        "title": f"{prefix}{text}",
        "snippet": snippet,
        "url": next((e["url"] for e in exemplars if e["url"]), ""),
        "source": "explore",
        "claim_id": str(claim.get("claim_id") or ""),
        "claim_key": str(claim.get("claim_key") or ""),
        "claim_kind": kind,
        "standing": str(claim.get("standing") or ""),
        "confidence_basis": basis,
        "observed_at": float(claim.get("observed_at") or 0.0),
        "exemplars": exemplars,
    }
    if cited:
        finding["artifact_id"] = cited
        finding["artifact_uri"] = f"{ARTIFACT_URI_PREFIX}{cited}"
    return finding


def claims_as_findings(store: ExploreStore, *,
                       now: float | None = None,
                       artifacts: Any = None,
                       limit: int = _DEFAULT_FEED_LIMIT,
                       ) -> list[dict[str, Any]]:
    """The feed: the store's current claims as pipeline findings.

    Population rule (stated in the module docstring): the current
    claim set (P2's fold — latest row per key, shed subjects
    excluded), freshness applied, FRESH claims only; stale claims
    do not feed. Newest evidence first, bounded."""
    moment = float(now if now is not None else time.time())
    out: list[dict[str, Any]] = []
    for claim in current_claims(store, now=moment):
        if claim.get("standing") != "fresh":
            continue
        out.append(claim_to_finding(claim, store, artifacts=artifacts))
        if len(out) >= max(1, int(limit)):
            break
    return out


def make_claims_feed(store: ExploreStore, *,
                     artifacts: Any = None,
                     ) -> Callable[[str], list[dict[str, Any]]]:
    """The installed feed callable for the weekly pipeline.

    The composition root builds this over the context's explore
    store (+ the artifact store, for citation verification) and
    installs it into ``social.content_pipeline`` — the pipeline
    calls it with the run's niche name. Claims are field-wide
    evidence, not per-niche research, so the name does not filter
    the feed; it is accepted because the seam's shape is the
    pipeline's. A feed failure yields no findings — the run's own
    research still runs (the pipeline also guards the call)."""

    def _feed(niche: str) -> list[dict[str, Any]]:
        try:
            return claims_as_findings(store, artifacts=artifacts)
        except Exception as exc:  # noqa: BLE001 - a feed never gates
            _log.warning("explore claims feed failed: %s", exc)
            return []

    return _feed


# ── contract 2: the shared niche registry, consumer side ────────


def drafting_niches(store: ExploreStore | None) -> list[dict[str, Any]]:
    """The niches drafting may use, read through P4's seam: the
    floor plugins ∪ the ADOPTED overlay records. Proposed and
    dismissed records appear nowhere in this list — a niche the
    owner has not adopted is not drafting context."""
    return effective_niches(store)


def niche_drafting_context(store: ExploreStore | None,
                           slug: str) -> dict[str, Any] | None:
    """The effective record for one niche name, or None.

    The weekly run's niche is free text; it names a registered
    niche when it matches a slug exactly or in slugified form
    (``Gold Trading`` → ``gold_trading``). Floor and adopted
    records resolve; a merely-proposed (or dismissed, or unknown)
    name resolves to None — drafting proceeds without context
    rather than borrowing an unadopted thesis."""
    if not slug:
        return None
    wanted = {str(slug), str(slug).strip().lower().replace(" ", "_")}
    for record in effective_niches(store):
        if record.get("slug") in wanted:
            return record
    return None


def drafting_plugin_resolver(store: ExploreStore | None,
                             ) -> Callable[[str], Any]:
    """A ``get_niche``-shaped resolver for contentops consumers:
    floor plugin first (the code registry wins, as everywhere),
    then the ADOPTED overlay's data-driven plugin, then the
    shipped default (which shim-falls-back for ad-hoc names,
    exactly as an unwired pipeline behaves). A proposed niche
    resolves through the default — never to a plugin carrying
    its unadopted thesis."""

    def _resolve(name: str) -> Any:
        from ..contentops.niches import get_niche
        from ..contentops.niches import registry as code_registry

        try:
            return code_registry.get(name)
        except Exception:  # noqa: BLE001 - unknown to the floor
            pass
        plugin = overlay_plugin(name, store)
        if plugin is not None:
            return plugin
        return get_niche(name)

    return _resolve


# ── contract 4: measured outcomes flow back to the claim ────────


def feed_claim_outcome(store: ExploreStore, queue: Any, *,
                       content: str, platform: str,
                       post_ref: str, handle: str = "",
                       measured: dict[str, Any] | None = None,
                       at: float = 0.0) -> str:
    """Record one measured post's outcome against its claim.

    The join is the spine's own discipline, mirrored: publishing
    writes no claim id onto a post row, so the link is exact
    content + platform identity against the review queue's drafts
    (the fan-out stamps the claim identity onto the drafts it
    queues from a claim finding). Verdicts:

    * ``fed`` — exactly one posted draft descends from a claim;
      the outcome is recorded against that claim.
    * ``already_recorded`` — that (claim, post) pair already has
      its outcome of record; nothing is written twice.
    * ``ambiguous`` — more than one claim-carrying posted draft
      matches; NO outcome is recorded (never guess which claim a
      post descends from).
    * ``unlinked`` — no claim-carrying draft matches; the post
      did not descend from a claim, and every claim stays
      outcome-less for it.
    * ``feed_failed`` — the store refused the write; the caller
      leaves the post unstamped so a later cycle retries.

    The recorded engagement is the content system's canonical
    number for a metrics dict (``content_pipeline._engagement_of``
    — the same number the briefs feed reports and the window
    ranking reads), reported unscaled."""
    from ...social.content_pipeline import _engagement_of

    candidates = queue.find_by_content(content, platform=platform)
    linkable = [
        d for d in candidates
        if (d.metadata or {}).get("claim_key")
        and d.status == "posted"
    ]
    if not linkable:
        # No posted draft descends from a claim for this post. If
        # a claim-carrying draft matches the content and this post
        # already has its outcome recorded against that claim, the
        # determination was made once — say so; otherwise the post
        # simply did not descend from a claim.
        for d in candidates:
            key = str((d.metadata or {}).get("claim_key") or "")
            if key and any(
                    o.get("post_ref") == post_ref
                    for o in store.claim_outcomes(claim_key=key)):
                return "already_recorded"
        return "unlinked"
    if len(linkable) > 1:
        return "ambiguous"
    draft = linkable[0]
    meta = draft.metadata or {}
    record = {
        "claim_key": str(meta.get("claim_key") or ""),
        "claim_id": str(meta.get("claim_id") or ""),
        "platform": platform,
        "post_ref": post_ref,
        "handle": handle,
        "engagement": _engagement_of(measured or {}),
        "metrics": dict(measured or {}),
        "at": float(at or time.time()),
    }
    if not store.record_claim_outcome(record):
        # The store refuses an exact (claim, post) duplicate —
        # that is the already-recorded case, not a failure. A
        # write that failed for another reason looks identical
        # here, so confirm by reading back before naming it.
        existing = store.claim_outcomes(claim_key=record["claim_key"])
        if any(o.get("post_ref") == post_ref for o in existing):
            return "already_recorded"
        return "feed_failed"
    return "fed"
