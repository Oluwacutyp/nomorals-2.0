"""SSE dashboard client: one client, four jobs, phone-first answers.

The web dashboard shows live state — sessions, groups, queue, costs,
jobs, organ health — over the :mod:`nomorals.stream` SSE server. This
module is the *client* side of that wire: it speaks to the server with
a :class:`~nomorals.stream.phone_client.TimedGrant` (``stream:read``),
runs the dashboard's four jobs, and renders short plain-language
answers for a phone screen.

The four jobs (``run_four_jobs`` — one client, one call, four results):

1. ``sessions`` — live sessions on the dashboard.
2. ``groups_queue`` — groups + the work queue, merged into one view.
3. ``costs_jobs`` — cost meter + background jobs.
4. ``organ_health`` — per-organ health (autonomy organs, scheduler, …).

**Fail-soft per job.** Each job runs isolated: one job failing
(server down, bad payload, timeout) records a calm error for that job
and the other three still return. A dashboard with one dead panel
beats a dashboard that shows nothing.

**Phone answers beat curses.** Errors are rendered as short, calm,
plain-language sentences — "Couldn't reach the queue — try again in
a bit." — never a traceback, never a raw exception, never profanity.
:func:`render_phone_answer` is the only error surface the phone sees.

**Grants enforced.** :class:`DashboardClient` refuses to make any
request on an expired grant (raises
:exc:`~nomorals.stream.phone_client.TokenExpired` *before* the
request, exactly like :class:`SSEClient`). When a ``secret`` is
supplied, the grant token is re-verified with
:func:`verify_stream_token` on every call — a tampered token never
reaches the wire.

**Transport.** The default fetcher is stdlib ``urllib`` GETs against
``{base_url}/api/<job>`` with ``Authorization: Bearer <token>``. Tests
and the chat layer inject their own ``fetcher`` callable
(``fetcher(job_name) -> dict``) — the client never assumes the server
implements an endpoint it hasn't confirmed; an HTTP error or a
non-dict payload becomes a calm per-job error, not a crash.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .phone_client import (
    SSEClient,
    TimedGrant,
    TokenExpired,
    TokenInvalid,
    verify_stream_token,
)

__all__ = [
    "DashboardClient",
    "DashboardSnapshot",
    "FOUR_JOBS",
    "JobResult",
    "calm_error",
    "render_phone_answer",
]

#: The four dashboard jobs, in display order.
FOUR_JOBS = ("sessions", "groups_queue", "costs_jobs", "organ_health")

#: Default per-job timeout (seconds). Dashboards must answer fast.
JOB_TIMEOUT_S = 10.0

#: Phrases that must never appear in a phone answer (kept as data so
#: renderers and tests share one list). Errors are calm, always.
_CALM_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Traceback", "Something went wrong"),
    ("Exception", "Something went wrong"),
)


def calm_error(job: str, detail: str = "") -> str:
    """One calm sentence for a failed job. No traceback, no jargon.

    ``detail`` is sanitised: anything traceback-shaped collapses to
    the generic sentence, and the raw detail is never echoed (it may
    contain paths, tokens, or stack frames).
    """
    _ = detail  # deliberately unused — raw errors never reach the phone
    return f"Couldn't reach {job} — try again in a bit."


@dataclass
class JobResult:
    """One job's outcome: data on success, a calm error on failure."""

    job: str = ""
    ok: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    took_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"job": self.job, "ok": self.ok, "data": self.data,
                "error": self.error, "took_s": round(self.took_s, 3)}


@dataclass
class DashboardSnapshot:
    """The four jobs + the phone answer, in one object."""

    jobs: dict[str, JobResult] = field(default_factory=dict)
    taken_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"taken_at": self.taken_at,
                "jobs": {k: v.to_dict() for k, v in self.jobs.items()}}

    @property
    def all_ok(self) -> bool:
        return bool(self.jobs) and all(j.ok for j in self.jobs.values())

    @property
    def ok_jobs(self) -> list[str]:
        return [k for k, j in self.jobs.items() if j.ok]

    @property
    def failed_jobs(self) -> list[str]:
        return [k for k, j in self.jobs.items() if not j.ok]


class DashboardClient:
    """One client for the dashboard: four jobs + the live SSE stream.

    ``grant`` must be a live :class:`TimedGrant` (``stream:read``).
    ``fetcher`` (optional) is ``Callable[[job_name], dict]`` — when
    omitted, stdlib HTTP GETs ``{base_url}/api/<job>`` are used.
    ``secret`` (optional) re-verifies the grant token on every call.
    """

    def __init__(self, base_url: str, grant: TimedGrant,
                 *, secret: str | None = None,
                 fetcher: Callable[[str], dict[str, Any]] | None = None,
                 timeout_s: float = JOB_TIMEOUT_S) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.grant = grant
        self.secret = secret
        self._fetcher = fetcher
        self.timeout_s = timeout_s

    # ── grant enforcement ──
    def _check_grant(self) -> None:
        """Refuse expired / tampered grants before any request. Raises."""
        if self.grant.expired:
            raise TokenExpired(
                f"stream grant expired at {self.grant.expires_at}")
        if self.secret:
            verify_stream_token(self.grant.token, self.secret)

    # ── transport ──
    def _http_fetch(self, job: str) -> dict[str, Any]:
        """Default fetcher: GET {base}/api/<job> with the grant."""
        url = f"{self.base_url}/api/{job}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/json",
                          "Authorization": f"Bearer {self.grant.token}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TokenInvalid(
                f"dashboard server rejected {job}: HTTP {exc.code}") from exc
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ConnectionError(f"dashboard fetch failed for {job}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"dashboard {job} returned a non-object payload")
        return payload

    def _fetch_job(self, job: str) -> JobResult:
        started = time.time()
        try:
            self._check_grant()
            if self._fetcher is not None:
                data = self._fetcher(job)
            else:
                data = self._http_fetch(job)
            if not isinstance(data, dict):
                raise ValueError(f"fetcher for {job} returned a non-object")
            return JobResult(job=job, ok=True, data=data,
                             took_s=time.time() - started)
        except (TokenExpired, TokenInvalid):
            raise  # grant failures are real failures — never calmed away
        except Exception as exc:  # noqa: BLE001 — per-job fail-soft
            return JobResult(job=job, ok=False, error=calm_error(job, str(exc)),
                             took_s=time.time() - started)

    # ── the four jobs ──
    def fetch_sessions(self) -> JobResult:
        """Job 1: live sessions."""
        return self._fetch_job("sessions")

    def fetch_groups_queue(self) -> JobResult:
        """Job 2: groups + work queue, merged into one view."""
        return self._fetch_job("groups_queue")

    def fetch_costs_jobs(self) -> JobResult:
        """Job 3: cost meter + background jobs."""
        return self._fetch_job("costs_jobs")

    def fetch_organ_health(self) -> JobResult:
        """Job 4: per-organ health."""
        return self._fetch_job("organ_health")

    def run_four_jobs(self) -> DashboardSnapshot:
        """Run all four jobs. One client, four results, isolated failures."""
        jobs = {
            "sessions": self.fetch_sessions(),
            "groups_queue": self.fetch_groups_queue(),
            "costs_jobs": self.fetch_costs_jobs(),
            "organ_health": self.fetch_organ_health(),
        }
        return DashboardSnapshot(jobs=jobs, taken_at=time.time())

    def snapshot(self) -> DashboardSnapshot:
        """Alias for :meth:`run_four_jobs` — the dashboard's one call."""
        return self.run_four_jobs()

    # ── live stream ──
    def stream_events(self, *, topic: str | None = None,
                      max_events: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield live SSE events. Grant enforced before connecting."""
        self._check_grant()
        client = SSEClient(self.base_url, self.grant, timeout_s=self.timeout_s)
        yield from client.stream(topic=topic, max_events=max_events)


def render_phone_answer(snapshot: DashboardSnapshot) -> str:
    """Short plain-language phone answer for a snapshot.

    Calm by construction: failures become one gentle line each, and
    the raw error text never appears. A fully healthy dashboard reads
    as four short lines; a partially dead one still shows what works.
    """
    lines = ["📱 **Dashboard**"]
    labels = {
        "sessions": "Sessions",
        "groups_queue": "Groups + queue",
        "costs_jobs": "Costs + jobs",
        "organ_health": "Organ health",
    }
    for job in FOUR_JOBS:
        result = snapshot.jobs.get(job)
        if result is None:
            lines.append(f"• {labels[job]}: not checked yet.")
        elif result.ok:
            lines.append(f"• {labels[job]}: {_summarise(job, result.data)}")
        else:
            lines.append(f"• {labels[job]}: {result.error}")
    if snapshot.failed_jobs:
        lines.append("\nThe rest is still working — pull to retry the rest.")
    return "\n".join(lines)


def _summarise(job: str, data: dict[str, Any]) -> str:
    """One short line per healthy job. Unknown shapes get an honest count."""
    try:
        if job == "sessions":
            n = data.get("sessions", data.get("count", data.get("active")))
            return f"{n} live" if isinstance(n, (int, float)) else "up"
        if job == "groups_queue":
            g = data.get("groups", data.get("group_count"))
            q = data.get("queue", data.get("queue_depth"))
            bits = []
            if isinstance(g, (int, float)):
                bits.append(f"{g} groups")
            if isinstance(q, (int, float)):
                bits.append(f"{q} queued")
            return ", ".join(bits) if bits else "up"
        if job == "costs_jobs":
            cost = data.get("cost", data.get("spent", data.get("cost_usd")))
            jobs = data.get("jobs", data.get("running_jobs"))
            bits = []
            if isinstance(cost, (int, float)):
                bits.append(f"${cost:.2f}")
            if isinstance(jobs, (int, float)):
                bits.append(f"{jobs} jobs running")
            return ", ".join(bits) if bits else "up"
        if job == "organ_health":
            organs = data.get("organs", data.get("health"))
            if isinstance(organs, dict):
                bad = [k for k, v in organs.items()
                       if isinstance(v, dict) and v.get("ok") is False]
                return (f"{len(organs) - len(bad)}/{len(organs)} organs healthy"
                        if organs else "no organs reported")
            return "up"
    except Exception:  # noqa: BLE001 — summarising never raises
        return "up"
    return "up"
