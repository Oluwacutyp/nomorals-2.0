"""Durable scheduler registration for the download-source health loop
(pass #86).

Follows the same ``ensure_*`` pattern as every other job family
(``finance/jobs.py``, ``agents/news_jobs.py``): extracted so the
registration can be enumerated — and binding-tested — without booting
the whole runtime.

Job:

* ``download source health`` — every 6h, re-verify every registered
  download source and every pending discovery candidate through the
  source registry's own health loop: verified candidates are
  promoted, dead ones shed, failing sources stand down.  Since
  Pass #88 the music search class lives in the same registry, so
  this one job also re-verifies the /play chain sources and can
  promote the candidates the music path discovers — no second job,
  no second binding.  Since Pass #89 the book/story class lives in
  the same registry too, so this one job also re-verifies the story
  sites and can promote the candidates the book path discovers —
  still no second job, no second binding.  Since Pass #90 the
  web-search class lives in the same registry too, so this one job
  also re-verifies the search engines — keyless engines through
  their real search endpoints (one bounded probe query each),
  keyed APIs by host root only, spending no search quota — still
  no second job, no second binding.  Since Pass #92 the image
  class (reverse image search — the shared Bing/Yandex host
  descriptors) lives in the same registry too, so this one job
  also re-verifies those hosts — still no second job, no second
  binding.  Since Pass #93 the film class (the film search
  chain — Internet Archive, NetNaija movies, Nkiri, FzMovies)
  lives in the same registry too, so this one job also
  re-verifies the film sources — through their real search
  endpoints (one bounded probe query each) — and can promote
  the candidates the film path discovers — still no second
  job, no second binding.  Since Pass #94 the wisdom archive
  class (the WisdomKeeper hunt's backends — Gutendex,
  Wikisource, archive.org, DuckDuckGo Lite) lives in the same
  registry too, so this one job also re-verifies the archive
  backends — the two new seeds through their real search
  endpoints (one bounded probe query each) — and can promote
  the archive candidates the wisdom hunt and the ingest fetch
  path discover — still no second job, no second binding.  Since
  Pass 8 (directed) the betting class (the betting desk's books,
  bustabit, the keyed odds/football-data APIs, the keyless ESPN
  feed) lives in the same registry too — the desk resolves the
  registry bound to the same KV store, so this one job also
  re-verifies the betting sources: ESPN through its real
  scoreboard endpoint, books at their roots (a bot-wall reads
  bot-blocked, never dead), the keyed APIs by root with the
  non-live verdict downgraded to unverifiable — still no second
  job, no second binding.
  Marked
  ``routine`` deliberately: the loop is self-reporting bookkeeping —
  its results land in the registry's status surface (``/download
  status``), not in owner alerts — but a FAILED run still records as
  a failed run (pass #71 semantics: routine silences success noise,
  never failure records).

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_source_health_jobs", "SOURCE_HEALTH_JOB"]

SOURCE_HEALTH_JOB = "download source health"

_JOBS = (
    (SOURCE_HEALTH_JOB, "every 6h",
     {"tool": "download_sources", "args": {"action": "health"}}),
)


def ensure_source_health_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the download-source health loop job.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
