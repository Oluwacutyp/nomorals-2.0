"""Durable scheduler registration for the story chapter discovery
sweep (novel pass, 2026-11-07).

Follows the same ``ensure_*`` pattern as every other job family
(``media/source_jobs.py``, ``agents/news_jobs.py``): extracted so
the registration can be enumerated — and binding-tested — without
booting the whole runtime.

Job:

* ``story chapter discovery`` — every 30m (every 1h on termux: the
  politeness profile fetches less, slower), run the
  ``story_discovery_sweep`` tool: compare each followed story's
  live chapter list against its library and publish
  ``story.chapters_discovered`` for the stories holding new
  chapters.

Why a poll at all: a story source is a web archive — it has no push
channel, no webhook, no feed of per-story chapter events.  SOMETHING
must look.  But the poll is a DISCOVERY trigger only: it does no
fetching and no reading itself.  The reading engine
(:mod:`nomorals.books.reading_engine`) consumes the discovery event
and runs the read immediately — the work never waits for a tick.
That split is deliberate, and it is the whole answer to the owner's
standing objection to schedule-queued work: what the schedule holds
is a *question* ("did anything new arrive?"), never the work
itself.  Marked ``routine``: the sweep's results land in the
reading engine's own pings (only when it actually read something
new) — but a FAILED run still records as a failed run.

Idempotent by name; no-ops when the job already exists.
"""

from __future__ import annotations

from typing import Any

from ..core.profiles import get_profile_kind

__all__ = ["ensure_story_jobs", "STORY_DISCOVERY_JOB"]

STORY_DISCOVERY_JOB = "story chapter discovery"


def _jobs() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    spec = "every 1h" if get_profile_kind() == "termux" else "every 30m"
    return (
        (STORY_DISCOVERY_JOB, spec,
         {"tool": "story_discovery_sweep", "args": {}}),
    )


def ensure_story_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the story chapter discovery job.

    Idempotent by name. Returns one status dict per job.
    """
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    for name, spec, payload in _jobs():
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(name, spec, "tool", payload, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out
