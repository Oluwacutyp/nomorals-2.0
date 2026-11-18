"""The recruitment store — per-vacancy jobs, durable.

The recruiter's side of hiring, the complement of the owner's own
career pipeline (:mod:`nomorals.careers`): where careers tracks the
applications *the owner* files, this store tracks the vacancies
*Devon recruits for* — each vacancy an independent job with its own
candidate queue, its own scraper state, its own stats and its own
CV flow.  A candidate belongs to exactly one vacancy; queues never
mix, and a vacancy's stats are computed from its own queue only.

Persistence follows the careers store — the repo's shape model for
a personal tracker: one JSON file, read through the storeguard
family guard (unreadable bytes are quarantined and saves then
refuse rather than overwriting them), written atomically via a
tmp file + replace.  The store lives at
``<home>/recruitment/vacancies.json`` (see :func:`recruitment_path`,
the ``finance_paths`` pattern).

Candidate history is append-only: intake and every tiering are
facts with timestamps, never rewritten — the /status board and any
later audit read what actually happened to a candidate, not a
recomputed present.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.storeguard import StoreUnreadableError, read_json_file
from .tiering import tier_candidate

_log = get_logger("nomorals.recruitment")

__all__ = [
    "Candidate",
    "RecruitmentError",
    "RecruitmentStore",
    "Vacancy",
    "VACANCY_STATUSES",
    "recruitment_path",
]

#: A vacancy is live, paused (keeps its queue, takes no new
#: candidates), or closed (finished — never takes candidates again).
VACANCY_STATUSES: tuple[str, ...] = ("open", "paused", "closed")

#: Scrape outcomes recorded on the vacancy.
SCRAPE_RESULTS: tuple[str, ...] = ("never", "ok", "failed")


class RecruitmentError(ValueError):
    """A recruitment operation refused in words (unknown vacancy,
    closed vacancy, duplicate intake, bad status).  A ValueError so
    surfaces handle it like the careers store's input errors."""


def recruitment_path(settings: Any = None) -> Path:
    """The recruitment store path, honoring the runtime settings home.

    Mirrors :func:`nomorals.careers.store.careers_path`: the settings
    home when one is in force, else ``~/.nomorals``.
    """
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "recruitment" / "vacancies.json"
    return Path.home() / ".nomorals" / "recruitment" / "vacancies.json"


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _clean_skills(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        return []
    return [p.strip() for p in parts if p.strip()]


def _dedupe_key(email: str, phone: str, cv_url: str,
                name: str) -> str:
    """What makes two intakes the same candidate in one vacancy:
    the email first, then the phone, then the CV link, then the
    name — each casefolded / digit-stripped so a re-scrape that
    re-spells the same person does not queue them twice."""
    if email:
        return "email:" + email.strip().casefold()
    digits = "".join(c for c in phone if c.isdigit())
    if digits:
        return "phone:" + digits
    if cv_url:
        return "cv:" + cv_url.strip().casefold()
    if name:
        return "name:" + name.strip().casefold()
    return ""


@dataclass
class Vacancy:
    """One vacancy Devon recruits for — an independent job."""

    id: str
    title: str
    employer: str
    deadline: float | None = None
    source_url: str = ""
    source_kind: str = "auto"      # auto | json | html
    required_skills: list[str] = field(default_factory=list)
    min_years: float = 0.0
    location: str = ""
    status: str = "open"
    created_at: float = 0.0
    updated_at: float = 0.0
    last_scrape_at: float | None = None
    last_scrape_result: str = "never"
    last_scrape_error: str = ""
    scraped_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "employer": self.employer,
            "deadline": self.deadline,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "required_skills": list(self.required_skills),
            "min_years": self.min_years,
            "location": self.location,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_scrape_at": self.last_scrape_at,
            "last_scrape_result": self.last_scrape_result,
            "last_scrape_error": self.last_scrape_error,
            "scraped_count": self.scraped_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vacancy:
        status = str(data.get("status", "open") or "open")
        if status not in VACANCY_STATUSES:
            status = "open"
        kind = str(data.get("source_kind", "auto") or "auto")
        if kind not in ("auto", "json", "html"):
            kind = "auto"
        result = str(data.get("last_scrape_result", "never") or "never")
        if result not in SCRAPE_RESULTS:
            result = "never"
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            employer=str(data.get("employer", "")),
            deadline=(float(data["deadline"])
                      if data.get("deadline") else None),
            source_url=_clean_str(data.get("source_url")),
            source_kind=kind,
            required_skills=_clean_skills(data.get("required_skills")),
            min_years=float(data.get("min_years", 0) or 0),
            location=_clean_str(data.get("location")),
            status=status,
            created_at=float(data.get("created_at", 0) or 0),
            updated_at=float(data.get("updated_at", 0) or 0),
            last_scrape_at=(float(data["last_scrape_at"])
                            if data.get("last_scrape_at") else None),
            last_scrape_result=result,
            last_scrape_error=_clean_str(data.get("last_scrape_error")),
            scraped_count=int(data.get("scraped_count", 0) or 0),
        )


@dataclass
class Candidate:
    """One candidate in one vacancy's queue."""

    id: str
    vacancy_id: str
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    years: float = 0.0
    skills: list[str] = field(default_factory=list)
    cv_text: str = ""
    cv_artifact_id: str = ""
    cv_url: str = ""
    source: str = "manual"        # manual | scrape
    score: float = 0.0
    tier: str = ""
    tier_reasons: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vacancy_id": self.vacancy_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "years": self.years,
            "skills": list(self.skills),
            "cv_text": self.cv_text,
            "cv_artifact_id": self.cv_artifact_id,
            "cv_url": self.cv_url,
            "source": self.source,
            "score": self.score,
            "tier": self.tier,
            "tier_reasons": list(self.tier_reasons),
            "history": list(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(
            id=str(data.get("id", "")),
            vacancy_id=str(data.get("vacancy_id", "")),
            name=str(data.get("name", "")),
            email=_clean_str(data.get("email")),
            phone=_clean_str(data.get("phone")),
            location=_clean_str(data.get("location")),
            years=float(data.get("years", 0) or 0),
            skills=_clean_skills(data.get("skills")),
            cv_text=str(data.get("cv_text", "") or ""),
            cv_artifact_id=_clean_str(data.get("cv_artifact_id")),
            cv_url=_clean_str(data.get("cv_url")),
            source=str(data.get("source", "manual") or "manual"),
            score=float(data.get("score", 0) or 0),
            tier=str(data.get("tier", "") or ""),
            tier_reasons=[str(r) for r in data.get("tier_reasons") or []],
            history=[dict(h) for h in data.get("history") or []
                     if isinstance(h, dict)],
            created_at=float(data.get("created_at", 0) or 0),
            updated_at=float(data.get("updated_at", 0) or 0),
        )


class RecruitmentStore:
    """Persistent vacancy registry.  JSON file, careers-shape."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = recruitment_path()
        self.path = Path(path)
        #: Set when the last _load found the file unreadable; while
        #: set, _save refuses rather than overwriting the quarantined
        #: bytes (data-loss family guard, the careers precedent).
        self._unreadable: BaseException | None = None

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> dict[str, Any]:
        data, exc = read_json_file(
            self.path, site="recruitment.store._load", logger=_log)
        if exc is not None:
            self._unreadable = exc
            return {"vacancies": {}, "candidates": {}}
        if data is not None:
            self._unreadable = None
            vacancies = data.get("vacancies")
            candidates = data.get("candidates")
            return {
                "vacancies": vacancies if isinstance(vacancies, dict)
                             else {},
                "candidates": candidates if isinstance(candidates, dict)
                              else {},
            }
        return {"vacancies": {}, "candidates": {}}

    def _save(self, data: dict[str, Any]) -> None:
        unreadable = getattr(self, "_unreadable", None)
        if unreadable is not None:
            raise StoreUnreadableError(
                f"recruitment store {self.path} is unreadable "
                f"({unreadable}); original bytes quarantined to "
                f"{self.path.name}.corrupt — refusing to overwrite them")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": 1, **data}, indent=1),
            encoding="utf-8")
        tmp.replace(self.path)

    def _mutate_vacancy(self, vac: Vacancy) -> Vacancy:
        data = self._load()
        data["vacancies"][vac.id] = vac.to_dict()
        self._save(data)
        return vac

    def _mutate_candidate(self, cand: Candidate) -> Candidate:
        data = self._load()
        data["candidates"][cand.id] = cand.to_dict()
        self._save(data)
        return cand

    def _require_vacancy(self, vacancy_id: str) -> Vacancy:
        vac = self.get_vacancy(vacancy_id)
        if vac is None:
            raise RecruitmentError(f"unknown vacancy {vacancy_id!r}")
        return vac

    # ── vacancies ────────────────────────────────────────────────
    def add_vacancy(
        self,
        title: str,
        employer: str,
        *,
        deadline: float | None = None,
        source_url: str = "",
        source_kind: str = "auto",
        required_skills: Any = None,
        min_years: float = 0.0,
        location: str = "",
        now: float | None = None,
    ) -> Vacancy:
        """Open a vacancy — its own job, its own queue from birth."""
        title = _clean_str(title)
        employer = _clean_str(employer)
        if not title:
            raise RecruitmentError("a vacancy needs a title")
        if not employer:
            raise RecruitmentError("a vacancy needs an employer")
        kind = _clean_str(source_kind).lower() or "auto"
        if kind not in ("auto", "json", "html"):
            raise RecruitmentError(
                f"unknown source kind {kind!r} (auto | json | html)")
        ts = time.time() if now is None else now
        vac = Vacancy(
            id="vac_" + uuid.uuid4().hex[:12],
            title=title,
            employer=employer,
            deadline=deadline,
            source_url=_clean_str(source_url),
            source_kind=kind,
            required_skills=_clean_skills(required_skills),
            min_years=float(min_years or 0),
            location=_clean_str(location),
            status="open",
            created_at=ts,
            updated_at=ts,
        )
        return self._mutate_vacancy(vac)

    def get_vacancy(self, vacancy_id: str) -> Vacancy | None:
        raw = self._load()["vacancies"].get(vacancy_id or "")
        if not raw:
            return None
        try:
            return Vacancy.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None

    def list_vacancies(self) -> list[Vacancy]:
        """Every vacancy, opened order."""
        out = []
        for raw in self._load()["vacancies"].values():
            try:
                out.append(Vacancy.from_dict(raw))
            except Exception:  # noqa: BLE001
                continue
        return sorted(out, key=lambda v: v.created_at)

    def remove_vacancy(self, vacancy_id: str) -> bool:
        data = self._load()
        if vacancy_id not in data["vacancies"]:
            return False
        del data["vacancies"][vacancy_id]
        for cid in [c for c, raw in data["candidates"].items()
                    if isinstance(raw, dict)
                    and raw.get("vacancy_id") == vacancy_id]:
            del data["candidates"][cid]
        self._save(data)
        return True

    def set_vacancy_status(
        self, vacancy_id: str, status: str, *,
        now: float | None = None,
    ) -> Vacancy:
        if status not in VACANCY_STATUSES:
            raise RecruitmentError(
                f"unknown status {status!r} "
                f"(statuses: {', '.join(VACANCY_STATUSES)})")
        vac = self._require_vacancy(vacancy_id)
        vac.status = status
        vac.updated_at = time.time() if now is None else now
        return self._mutate_vacancy(vac)

    def set_source(
        self, vacancy_id: str, source_url: str, *,
        source_kind: str = "auto",
        now: float | None = None,
    ) -> Vacancy:
        """Point a vacancy at its applicant source (URL + kind)."""
        vac = self._require_vacancy(vacancy_id)
        kind = _clean_str(source_kind).lower() or "auto"
        if kind not in ("auto", "json", "html"):
            raise RecruitmentError(
                f"unknown source kind {kind!r} (auto | json | html)")
        vac.source_url = _clean_str(source_url)
        vac.source_kind = kind
        vac.updated_at = time.time() if now is None else now
        return self._mutate_vacancy(vac)

    def record_scrape(
        self, vacancy_id: str, result: str, *, error: str = "",
        new_count: int = 0, now: float | None = None,
    ) -> Vacancy:
        """Stamp a vacancy's scrape state — per vacancy, never global."""
        if result not in ("ok", "failed"):
            raise RecruitmentError(
                f"unknown scrape result {result!r} (ok | failed)")
        vac = self._require_vacancy(vacancy_id)
        ts = time.time() if now is None else now
        vac.last_scrape_at = ts
        vac.last_scrape_result = result
        vac.last_scrape_error = _clean_str(error)
        if result == "ok":
            vac.scraped_count += int(new_count or 0)
        vac.updated_at = ts
        return self._mutate_vacancy(vac)

    # ── candidates ───────────────────────────────────────────────
    def add_candidate(
        self,
        vacancy_id: str,
        *,
        name: str,
        email: str = "",
        phone: str = "",
        location: str = "",
        years: float = 0.0,
        skills: Any = None,
        cv_text: str = "",
        cv_artifact_id: str = "",
        cv_url: str = "",
        source: str = "manual",
        now: float | None = None,
    ) -> tuple[Candidate, bool]:
        """Intake one candidate into one vacancy's queue.

        Runs the tiering rules at intake and records the tier + its
        reasons on the candidate, with intake/tiered history.  A
        duplicate (same contact/CV in this vacancy) returns the
        existing candidate with ``is_new=False`` — the queue never
        holds the same person twice.  A closed or paused vacancy
        refuses the intake in words.
        """
        vac = self._require_vacancy(vacancy_id)
        if vac.status != "open":
            raise RecruitmentError(
                f"{vac.employer} — {vac.title} is {vac.status}; "
                "it takes no new candidates")
        name = _clean_str(name)
        if not name:
            raise RecruitmentError("a candidate needs a name")
        email = _clean_str(email)
        phone = _clean_str(phone)
        cv_url = _clean_str(cv_url)
        key = _dedupe_key(email, phone, cv_url, name)
        ts = time.time() if now is None else now
        data = self._load()
        if key:
            for raw in data["candidates"].values():
                if not isinstance(raw, dict):
                    continue
                if raw.get("vacancy_id") != vacancy_id:
                    continue
                other = Candidate.from_dict(raw)
                if _dedupe_key(other.email, other.phone,
                               other.cv_url, other.name) == key:
                    return other, False
        cand = Candidate(
            id="cand_" + uuid.uuid4().hex[:12],
            vacancy_id=vacancy_id,
            name=name,
            email=email,
            phone=phone,
            location=_clean_str(location),
            years=float(years or 0),
            skills=_clean_skills(skills),
            cv_text=str(cv_text or ""),
            cv_artifact_id=_clean_str(cv_artifact_id),
            cv_url=cv_url,
            source=_clean_str(source) or "manual",
            history=[{"event": "intake", "at": ts,
                      "note": _clean_str(source) or "manual"}],
            created_at=ts,
            updated_at=ts,
        )
        result = tier_candidate(cand, vac)
        cand.score = result.score
        cand.tier = result.tier
        cand.tier_reasons = list(result.reasons)
        cand.history.append({"event": "tiered", "at": ts,
                             "tier": result.tier,
                             "score": result.score})
        data["candidates"][cand.id] = cand.to_dict()
        self._save(data)
        return cand, True

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        raw = self._load()["candidates"].get(candidate_id or "")
        if not raw:
            return None
        try:
            return Candidate.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None

    def list_candidates(self, vacancy_id: str) -> list[Candidate]:
        """One vacancy's queue, intake order — never another's."""
        out = []
        for raw in self._load()["candidates"].values():
            try:
                cand = Candidate.from_dict(raw)
            except Exception:  # noqa: BLE001
                continue
            if cand.vacancy_id == vacancy_id:
                out.append(cand)
        return sorted(out, key=lambda c: c.created_at)

    # ── derived views ────────────────────────────────────────────
    def stats(self, vacancy_id: str) -> dict[str, Any]:
        """One vacancy's numbers from its own queue only."""
        self._require_vacancy(vacancy_id)
        cands = self.list_candidates(vacancy_id)
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for cand in cands:
            if cand.tier in counts:
                counts[cand.tier] += 1
        vac = self.get_vacancy(vacancy_id)
        assert vac is not None
        return {
            "queue_depth": len(cands),
            "tier_counts": counts,
            "scraped_count": vac.scraped_count,
            "last_scrape_at": vac.last_scrape_at,
            "last_scrape_result": vac.last_scrape_result,
            "last_scrape_error": vac.last_scrape_error,
        }
