"""The measurement spine (WS3 P1): post performance, measured for real.

Until this module, :meth:`PlatformAdapter.metrics` had zero production
callers. ``social_posts.metrics`` was written once — from whatever the
publish result happened to carry — and never refreshed, so
``content_pipeline.optimal_windows`` ranked windows on stale or empty
numbers and fell back to generic windows in practice, and the briefs
feedback loop (``BriefStore.record_outcome`` → ``score_feedback`` →
``_calibration``) had never once been fed. Three learning loops, all
open. This module is the one spine that closes them:

* a durable ``social metrics refresh`` job (registered in
  :mod:`nomorals.social.jobs`, run through the ``social_metrics_refresh``
  tool) re-reads every recently posted row's engagement from its own
  platform adapter and writes the measured dict back, with its fetch
  time, via :meth:`SocialManager.update_post_metrics`;
* on a post's first measured refresh at a fair age (>= 24h), the
  outcome is fed — once, stamped on the row — into the briefs loop
  when the post links deterministically to exactly one brief draft.

Honesty rules, enforced here:

* an adapter that does not override ``PlatformAdapter.metrics``
  cannot produce metrics: its platform is NAMED ``unavailable`` in the
  run's accounting and its rows are left untouched — never zeroed;
* an adapter that answers ``{}`` returned no data (the shipped
  adapters' own best-effort contract): the stored metrics are left
  untouched and the row is counted ``empty``, never silently skipped;
* a fetch that raises is counted and logged with the platform, the
  account handle and the post's external id — never swallowed;
* the briefs join is exact content + platform identity (publishing
  writes no brief-draft id onto a post row, so content is the only
  deterministic link): zero or several candidates means NO feed, and
  the verdict is stamped on the row so the determination is made once.

Deliberately NOT wired here: ``agency/digestion.record_outcome``. Its
contract serves the gig/deal estimate calibration (its one production
caller is the deals trigger in ``agency/systems.py``); for any other
kind it returns a constant ``{"adjustment": "none", "ratio": 1.0}``,
discarding expected/actual — a post outcome fed through it would be
wiring in form only, with the information thrown away. The briefs
``score_feedback`` loop is the content outcome store this spine feeds;
digestion's claims shape is the WS4 trend seam (§9), not this pass.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger
from .base import Account, PlatformAdapter

__all__ = ["adapter_supports_metrics", "refresh_post_metrics"]

_log = get_logger(__name__)

#: How far back a posted row stays worth measuring — the same 90-day
#: horizon ``optimal_windows`` reads.
WINDOW_DAYS = 90
#: A post younger than this has no meaningful numbers yet.
MIN_AGE_S = 3600.0
#: A row measured less than this ago waits for the next cycle; the
#: refresh job's own cadence is 6h, so a row is re-measured at most
#: once per cycle.
REFRESH_INTERVAL_S = 6 * 3600.0
#: The briefs outcome feed waits until a post is this old — a day of
#: engagement is a fair read; an hour's is noise that would be stamped
#: as the post's one outcome.
OUTCOME_MIN_AGE_S = 24 * 3600.0
#: Bounded work per run, freshest posts first.
DEFAULT_LIMIT = 200


def adapter_supports_metrics(adapter: PlatformAdapter) -> bool:
    """True when the adapter really implements metrics.

    The base class answers ``{}`` for every platform ("not every
    platform exposes them"), so capability is whether an override
    exists anywhere below :class:`PlatformAdapter` in the adapter's
    MRO — the shipped shape: Mastodon overrides it, Bluesky and Postiz
    do not.
    """
    return type(adapter).metrics is not PlatformAdapter.metrics


def _decode_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def _feed_brief_outcome(
    store: Any, row: dict[str, Any], measured: dict[str, Any]
) -> str:
    """Feed one post's measured outcome into the briefs loop.

    Returns the verdict to stamp on the row: ``fed``, ``unlinked``
    (no brief draft carries this content), ``ambiguous`` (more than
    one candidate — no guessing), or ``already_recorded`` (the one
    matching draft already carries an outcome). ``feed_failed`` means
    the store refused mid-write; the caller leaves the row unstamped
    so the next cycle retries.
    """
    from .content_pipeline import _engagement_of

    candidates = store.find_by_content(
        str(row.get("content") or ""), platform=str(row.get("platform") or ""))
    linkable = [
        d for d in candidates
        if d.brief_id and not d.published_at and not d.outcome_metrics
    ]
    if not linkable:
        return "already_recorded" if candidates else "unlinked"
    if len(linkable) > 1:
        return "ambiguous"
    draft = linkable[0]
    payload = dict(measured)
    # briefs' outcome contract takes one engagement figure; the content
    # system's canonical engagement number for a metrics dict is
    # content_pipeline._engagement_of — the same number optimal_windows
    # ranks by. Reported unscaled, never normalized into existence.
    payload["engagement_score"] = _engagement_of(measured)
    return "fed" if store.record_outcome(draft.draft_id, payload) \
        else "feed_failed"


def refresh_post_metrics(
    manager: Any,
    *,
    now: float | None = None,
    limit: int = DEFAULT_LIMIT,
    min_age_s: float = MIN_AGE_S,
    refresh_interval_s: float = REFRESH_INTERVAL_S,
    outcome_min_age_s: float = OUTCOME_MIN_AGE_S,
    window_days: int = WINDOW_DAYS,
    brief_store: Any = None,
    feed_outcomes: bool = True,
) -> dict[str, Any]:
    """Refresh measured metrics for posted rows; feed first outcomes.

    ``manager`` is a :class:`SocialManager` with adapters registered.
    Returns the run's accounting — every considered row lands in
    exactly one bucket, and every failure names its platform, account
    handle and the post's external id. Never raises for a per-row
    problem; a spine-level failure (no database) raises, because a
    scheduled run that could not run at all must record as failed.
    """
    started = time.perf_counter()
    moment = float(now if now is not None else manager._clock())
    report: dict[str, Any] = {
        "ok": True,
        "considered": 0,
        "refreshed": 0,
        "unavailable": 0,
        "unavailable_platforms": [],
        "empty": 0,
        "failed": 0,
        "skipped_recent": 0,
        "skipped_no_account": 0,
        "skipped_no_adapter": 0,
        "outcomes_fed": 0,
        "outcomes_unlinked": 0,
        "outcomes_ambiguous": 0,
        "outcomes_already_recorded": 0,
        "outcomes_failed": 0,
        "failures": [],
    }
    rows = manager.db.query(
        "SELECT * FROM social_posts WHERE status = 'posted' "
        "AND posted_at IS NOT NULL AND external_id != '' "
        "AND posted_at > ? AND posted_at <= ? "
        "ORDER BY posted_at DESC LIMIT ?",
        (moment - window_days * 86400.0, moment - min_age_s,
         max(1, int(limit))),
    )
    unavailable: set[str] = set()
    store: Any = brief_store
    store_state = "pending"  # pending | ready | broken
    for raw_row in rows:
        row = dict(raw_row)
        report["considered"] += 1
        post_id = str(row.get("id") or "")
        platform = str(row.get("platform") or "")
        external_id = str(row.get("external_id") or "")
        meta = _decode_json(row.get("metadata"))
        fetched_at = float(meta.get("metrics_fetched_at") or 0.0)
        if fetched_at and moment - fetched_at < refresh_interval_s:
            report["skipped_recent"] += 1
            continue
        account_row = manager.accounts.get(str(row.get("account_id") or ""))
        if account_row is None:
            report["skipped_no_account"] += 1
            _log.warning(
                "metrics refresh: post %s on %s has no connected account",
                external_id, platform)
            continue
        account = Account.from_row(account_row)
        adapter = manager.adapters.get(platform)
        if adapter is None:
            report["skipped_no_adapter"] += 1
            _log.warning(
                "metrics refresh: no adapter registered for %s (post %s)",
                platform, external_id)
            continue
        if not adapter_supports_metrics(adapter):
            report["unavailable"] += 1
            unavailable.add(platform)
            continue
        try:
            measured = adapter.metrics(account, external_id)
        except Exception as exc:  # noqa: BLE001 - a fetch failure is data
            report["failed"] += 1
            _log.warning(
                "metrics refresh failed for %s post %s (@%s): %s: %s",
                platform, external_id, account.handle,
                type(exc).__name__, exc)
            if len(report["failures"]) < 10:
                report["failures"].append({
                    "platform": platform, "handle": account.handle,
                    "external_id": external_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue
        if not measured:
            report["empty"] += 1
            _log.info(
                "metrics refresh: %s post %s (@%s) returned no data",
                platform, external_id, account.handle)
            continue
        manager.update_post_metrics(
            post_id, metrics=dict(measured), fetched_at=moment)
        report["refreshed"] += 1
        # First-outcome feed: once per post, at a fair age, stamped.
        if not feed_outcomes or meta.get("metrics_outcome"):
            continue
        if moment - float(row.get("posted_at") or 0.0) < outcome_min_age_s:
            continue
        if store_state == "pending":
            try:
                if store is None:
                    from ..marketing.briefs import BriefStore

                    store = BriefStore()
                store_state = "ready"
            except Exception as exc:  # noqa: BLE001 - feeding is best-effort
                store_state = "broken"
                _log.warning("metrics spine: brief store unavailable: %s",
                             exc)
        if store_state != "ready":
            report["outcomes_failed"] += 1
            continue
        try:
            verdict = _feed_brief_outcome(store, row, dict(measured))
        except Exception as exc:  # noqa: BLE001 - never sink the run
            verdict = "feed_failed"
            _log.warning(
                "metrics spine: outcome feed failed for %s post %s: %s",
                platform, external_id, exc)
        if verdict == "feed_failed":
            report["outcomes_failed"] += 1
            continue  # unstamped: the next cycle retries this post
        manager.update_post_metrics(post_id, outcome=verdict)
        report[{
            "fed": "outcomes_fed",
            "unlinked": "outcomes_unlinked",
            "ambiguous": "outcomes_ambiguous",
            "already_recorded": "outcomes_already_recorded",
        }[verdict]] += 1
    report["unavailable_platforms"] = sorted(unavailable)
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["text"] = (
        f"metrics refresh: {report['refreshed']} refreshed, "
        f"{report['unavailable']} unavailable "
        f"({', '.join(report['unavailable_platforms']) or 'none'}), "
        f"{report['empty']} no-data, {report['failed']} failed, "
        f"{report['outcomes_fed']} outcomes fed to briefs")
    return report
