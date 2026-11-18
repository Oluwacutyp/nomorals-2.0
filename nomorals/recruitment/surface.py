"""Recruitment surface: the /status board and vacancy views.

The presentation layer over :class:`~nomorals.recruitment.store.RecruitmentStore`,
shared by the chat ``/recruit`` surface and the ``recruit`` brain
tool.  Rendering is pure text over a store the caller supplies.

The board's honesty rule (the careers board's rule, applied to
recruiting): a vacancy whose last scrape failed is CALLED OUT with
the failure reason inline — a dead source never reads as an empty
queue — and a vacancy with no candidates says so, never a faked
line.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from .store import Candidate, RecruitmentStore, Vacancy

__all__ = [
    "render_status_board",
    "render_vacancy",
    "resolve_candidate",
    "resolve_vacancy",
    "unknown_candidate_text",
    "unknown_vacancy_text",
]


def _fmt_day(ts: float | None) -> str:
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")


def _days(n: float) -> str:
    days = int(n // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "1d"
    return f"{days}d"


def resolve_vacancy(
    store: RecruitmentStore, ref: str,
) -> tuple[Vacancy | None, list[Vacancy]]:
    """Resolve a chat vacancy reference to one vacancy.

    Accepts a vacancy id (exact or unique prefix), ``#n`` (1-based
    into the board order — the numbers the board prints), or a unique
    substring of "title employer".  Returns ``(vacancy, [])`` on a
    hit, ``(None, candidates)`` when the reference is ambiguous or
    matches nothing (mirrors the careers surface's resolve contract).
    """
    ref = (ref or "").strip()
    vacs = store.list_vacancies()
    if not ref:
        return None, vacs
    exact = store.get_vacancy(ref)
    if exact is not None:
        return exact, []
    if ref.startswith("#") and ref[1:].isdigit():
        idx = int(ref[1:]) - 1
        if 0 <= idx < len(vacs):
            return vacs[idx], []
        return None, vacs
    by_prefix = [v for v in vacs if v.id.startswith(ref)]
    if len(by_prefix) == 1:
        return by_prefix[0], []
    needle = ref.casefold()
    by_text = [v for v in vacs
               if needle in f"{v.title} {v.employer}".casefold()]
    if len(by_text) == 1:
        return by_text[0], []
    if len(by_text) > 1:
        return None, by_text
    return None, vacs


def unknown_vacancy_text(
    ref: str, candidates: list[Vacancy],
) -> str:
    """The honest unknown-ref reply: name the miss, show what exists."""
    lines = [f"no single vacancy matches {ref!r}."]
    if candidates:
        lines.append("vacancies on file:")
        for i, vac in enumerate(candidates[:10], 1):
            lines.append(
                f"  #{i} {vac.employer} — {vac.title} ({vac.status}) "
                f"· {vac.id}")
    else:
        lines.append(
            "no vacancies yet — /recruit add <title> | <employer> "
            "opens the first one.")
    return "\n".join(lines)


def resolve_candidate(
    store: RecruitmentStore, ref: str,
) -> tuple[Candidate | None, list[Candidate]]:
    """Resolve a candidate id (exact or unique prefix) or a unique
    substring of "name email" across every vacancy's queue."""
    ref = (ref or "").strip()
    if not ref:
        return None, []
    exact = store.get_candidate(ref)
    if exact is not None:
        return exact, []
    all_cands: list[Candidate] = []
    for vac in store.list_vacancies():
        all_cands.extend(store.list_candidates(vac.id))
    by_prefix = [c for c in all_cands if c.id.startswith(ref)]
    if len(by_prefix) == 1:
        return by_prefix[0], []
    needle = ref.casefold()
    by_text = [c for c in all_cands
               if needle in f"{c.name} {c.email}".casefold()]
    if len(by_text) == 1:
        return by_text[0], []
    if len(by_text) > 1:
        return None, by_text
    return None, all_cands


def unknown_candidate_text(
    ref: str, candidates: list[Candidate],
) -> str:
    lines = [f"no single candidate matches {ref!r}."]
    if candidates:
        lines.append("candidates on file:")
        for cand in candidates[:10]:
            lines.append(
                f"  {cand.name} ({cand.email or 'no contact'}) "
                f"· tier {cand.tier or '?'} · `{cand.id}`")
    else:
        lines.append("no candidates yet — intake one with "
                     "/recruit candidate <vacancy> | <name> | <email>.")
    return "\n".join(lines)


def _vacancy_lines(
    store: RecruitmentStore, vac: Vacancy, index: int, *, now: float,
) -> list[str]:
    stats = store.stats(vac.id)
    tiers = stats["tier_counts"]
    tier_bits = " ".join(
        f"{band} {tiers[band]}" for band in ("A", "B", "C", "D")
        if tiers[band])
    lines = [
        f"  #{index} **{vac.employer} — {vac.title}** ({vac.status})",
        f"    queue {stats['queue_depth']}"
        + (f" · tiers {tier_bits}" if tier_bits else " · no candidates yet")
        + f" · scraped {stats['scraped_count']}",
    ]
    scrape = stats["last_scrape_result"]
    if scrape == "failed":
        lines.append(
            f"    ⛔ last scrape failed {_fmt_day(stats['last_scrape_at'])}: "
            f"{stats['last_scrape_error']}")
    elif scrape == "ok":
        lines.append(
            f"    last scrape ok {_fmt_day(stats['last_scrape_at'])}")
    else:
        lines.append("    never scraped"
                     + (" — no source URL set" if not vac.source_url
                        else ""))
    if vac.deadline:
        due = "overdue" if vac.deadline < now else f"due {_fmt_day(vac.deadline)}"
        lines.append(f"    deadline {due}")
    lines.append(f"    `{vac.id}`")
    return lines


def render_status_board(
    store: RecruitmentStore, *, now: float | None = None,
) -> str:
    """The recruitment board: every vacancy with queue depth, tier
    counts and scrape state, live from the store.  An empty module
    renders a designed empty state, never a bare blank."""
    now = time.time() if now is None else now
    vacs = store.list_vacancies()
    if not vacs:
        return (
            "🧲 **Recruitment**\n"
            "  🗒 no vacancies yet — this is where Devon recruits: "
            "one independent job per vacancy, each with its own "
            "queue, scraper, stats and CV flow.\n"
            "  open one: /recruit add <title> | <employer>")
    lines = [f"🧲 **Recruitment** — {len(vacs)} vacanc"
             f"{'y' if len(vacs) == 1 else 'ies'}", ""]
    for i, vac in enumerate(vacs, 1):
        lines.extend(_vacancy_lines(store, vac, i, now=now))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_vacancy(
    store: RecruitmentStore, vac: Vacancy, *,
    now: float | None = None,
) -> str:
    """One vacancy, whole: requirements, stats, candidates by tier."""
    now = time.time() if now is None else now
    stats = store.stats(vac.id)
    lines = [
        f"🧲 **{vac.employer} — {vac.title}** ({vac.status})",
        f"  queue {stats['queue_depth']} · "
        f"scraped {stats['scraped_count']} · "
        f"last scrape {stats['last_scrape_result']} "
        f"{_fmt_day(stats['last_scrape_at'])}",
    ]
    reqs = []
    if vac.required_skills:
        reqs.append("skills: " + ", ".join(vac.required_skills))
    if vac.min_years:
        reqs.append(f"min {vac.min_years:g}y")
    if vac.location:
        reqs.append(f"location: {vac.location}")
    if vac.deadline:
        reqs.append(f"deadline {_fmt_day(vac.deadline)}")
    if reqs:
        lines.append("  " + " · ".join(reqs))
    if vac.source_url:
        lines.append(f"  source: {vac.source_url}")
    lines.append("")
    cands = store.list_candidates(vac.id)
    if not cands:
        lines.append("  no candidates yet — intake one with "
                     f"/recruit candidate {vac.id} | <name> | <email>, "
                     "or run a scrape.")
        return "\n".join(lines)
    by_tier: dict[str, list[Candidate]] = {}
    for cand in cands:
        by_tier.setdefault(cand.tier or "?", []).append(cand)
    for band in ("A", "B", "C", "D"):
        group = by_tier.get(band)
        if not group:
            continue
        lines.append(f"**Tier {band}**")
        for cand in group:
            contact = cand.email or cand.phone or "no contact"
            top = cand.tier_reasons[0] if cand.tier_reasons else ""
            lines.append(
                f"  {cand.name} · score {cand.score:g} · {contact}"
                + (f" · {top}" if top else "")
                + f" · `{cand.id}`")
        lines.append("")
    return "\n".join(lines).rstrip()
