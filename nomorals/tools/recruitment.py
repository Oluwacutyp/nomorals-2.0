"""Recruitment tool: ``recruit`` — the brain's handle on the
recruitment module.

One tool, action-dispatched, over :mod:`nomorals.recruitment`; the
organ owns the vacancies, the queues, the tiering rules and the
scraper — this module is only the surface.

``action=status`` boards every vacancy (``vacancy=<ref>`` for one
vacancy's detail); ``action=list`` lists vacancies compactly;
``action=add`` opens a vacancy (``title=``, ``employer=``,
``source_url=``, ``skills=`` comma list, ``min_years=``,
``deadline=`` YYYY-MM-DD); ``action=scrape`` runs one vacancy's
scrape now (``vacancy=<ref>``); ``action=candidate`` intakes one
candidate (``vacancy=<ref>``, ``name=``, ``email=``, ``phone=``,
``years=``, ``skills=``, ``cv=``); ``action=tier`` shows one
candidate's tier, score and reasons (``candidate=<ref>``).  An
unknown vacancy or candidate is answered honestly as nothing on
record, never invented; a scrape that cannot fetch says so with
the reason, never a faked intake.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..recruitment import surface as _surface_mod
from ..recruitment.scraper import scrape_vacancy
from ..recruitment.store import (
    RecruitmentError,
    RecruitmentStore,
    recruitment_path,
)

__all__ = ["register"]


def _store(context: Any) -> RecruitmentStore:
    settings = getattr(context, "settings", None)
    return RecruitmentStore(recruitment_path(settings))


def _parse_deadline(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(
            hour=12, tzinfo=UTC).timestamp()
    except ValueError:
        return None


def register(registry: Any) -> None:
    """Attach the ``recruit`` tool."""
    context = registry.context

    @registry.register(
        "recruit",
        description=(
            "Recruitment module: per-vacancy independent jobs, each "
            "with its own candidate queue, scraper, stats and CV flow. "
            "action=status boards every vacancy with queue depth, tier "
            "counts and scrape state (vacancy=<ref> for one vacancy). "
            "action=list lists vacancies. action=add opens a vacancy "
            "(title=, employer=, source_url=, skills=, min_years=, "
            "deadline= YYYY-MM-DD). action=scrape runs one vacancy's "
            "scrape now (vacancy=<ref>). action=candidate intakes one "
            "candidate (vacancy=<ref>, name=, email=, phone=, years=, "
            "skills=, cv=). action=tier shows one candidate's tier, "
            "score and reasons (candidate=<ref>). Unknown vacancies or "
            "candidates are answered as nothing on record, never "
            "invented; a scrape that cannot fetch says so with the "
            "reason, never a faked intake."
        ),
        capability="owner",
    )
    def recruit_tool(
        action: str = "status",
        vacancy: str = "",
        candidate: str = "",
        title: str = "",
        employer: str = "",
        source_url: str = "",
        skills: str = "",
        min_years: str = "",
        deadline: str = "",
        name: str = "",
        email: str = "",
        phone: str = "",
        years: str = "",
        cv: str = "",
    ) -> dict[str, Any]:
        store = _store(context)
        action = str(action or "status").strip().lower()
        if action not in ("status", "list", "add", "scrape",
                          "candidate", "tier"):
            return {"ok": False,
                    "error": f"unknown action {action!r}"}
        try:
            if action == "status":
                if (vacancy or "").strip():
                    found, cands = _surface_mod.resolve_vacancy(
                        store, vacancy)
                    if found is None:
                        return {"ok": False, "error":
                                _surface_mod.unknown_vacancy_text(
                                    vacancy, cands)}
                    return {"ok": True, "text":
                            _surface_mod.render_vacancy(store, found)}
                return {"ok": True, "text":
                        _surface_mod.render_status_board(store)}
            if action == "list":
                vacs = store.list_vacancies()
                return {"ok": True, "vacancies": [
                    {"id": v.id, "title": v.title,
                     "employer": v.employer, "status": v.status,
                     "queue_depth": store.stats(v.id)["queue_depth"]}
                    for v in vacs]}
            if action == "add":
                if not (title or "").strip() \
                        or not (employer or "").strip():
                    return {"ok": False,
                            "error": "add needs title= and employer="}
                try:
                    min_y = float(min_years) if str(min_years).strip() \
                        else 0.0
                except (TypeError, ValueError):
                    min_y = 0.0
                vac = store.add_vacancy(
                    title, employer,
                    source_url=source_url,
                    required_skills=skills,
                    min_years=min_y,
                    deadline=_parse_deadline(deadline))
                return {"ok": True, "vacancy_id": vac.id,
                        "text": f"opened {vac.employer} — {vac.title}"}
            if action == "scrape":
                found, cands = _surface_mod.resolve_vacancy(
                    store, vacancy)
                if found is None:
                    return {"ok": False, "error":
                            _surface_mod.unknown_vacancy_text(
                                vacancy, cands)}
                out = scrape_vacancy(store, found)
                return {"ok": bool(out["ok"]), **out}
            if action == "candidate":
                found, cands = _surface_mod.resolve_vacancy(
                    store, vacancy)
                if found is None:
                    return {"ok": False, "error":
                            _surface_mod.unknown_vacancy_text(
                                vacancy, cands)}
                if not (name or "").strip():
                    return {"ok": False,
                            "error": "candidate needs name="}
                try:
                    yrs = float(years) if str(years).strip() else 0.0
                except (TypeError, ValueError):
                    yrs = 0.0
                cand, is_new = store.add_candidate(
                    found.id, name=name, email=email, phone=phone,
                    years=yrs, skills=skills, cv_text=cv,
                    source="tool")
                return {"ok": True, "candidate_id": cand.id,
                        "is_new": is_new, "tier": cand.tier,
                        "score": cand.score,
                        "reasons": cand.tier_reasons}
            # action == "tier"
            found_c, cands = _surface_mod.resolve_candidate(
                store, candidate)
            if found_c is None:
                return {"ok": False, "error":
                        _surface_mod.unknown_candidate_text(
                            candidate, cands)}
            return {"ok": True, "candidate_id": found_c.id,
                    "name": found_c.name, "tier": found_c.tier,
                    "score": found_c.score,
                    "reasons": found_c.tier_reasons}
        except RecruitmentError as exc:
            return {"ok": False, "error": str(exc)}
