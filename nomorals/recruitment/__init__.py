"""Recruitment — the recruiter's organ (B42).

Devon recruits for vacancies: each vacancy is an independent job
with its own candidate queue, its own scraper, its own stats and
its own CV flow.  The complement of :mod:`nomorals.careers` (the
owner's own job hunt) — this package is the hiring side, the
Jobilate recruiter's domain built as Devon's own module, not a
wrapper around the separate recruiter bot.

* :mod:`nomorals.recruitment.store` — the durable vacancy +
  candidate registry (careers-shape JSON store);
* :mod:`nomorals.recruitment.tiering` — the tiering rules, pure
  functions in code (CV + skills + experience + contact + location);
* :mod:`nomorals.recruitment.scraper` — the per-vacancy fetcher
  (JSON feeds + applicant pages through the house HttpClient seam);
* :mod:`nomorals.recruitment.surface` — the /status board and
  vacancy views over the store.
"""

from __future__ import annotations

from .store import (
    VACANCY_STATUSES,
    Candidate,
    RecruitmentError,
    RecruitmentStore,
    Vacancy,
    recruitment_path,
)
from .tiering import TierResult, score_candidate, tier_candidate

__all__ = [
    "VACANCY_STATUSES",
    "Candidate",
    "RecruitmentError",
    "RecruitmentStore",
    "TierResult",
    "Vacancy",
    "recruitment_path",
    "score_candidate",
    "tier_candidate",
]
