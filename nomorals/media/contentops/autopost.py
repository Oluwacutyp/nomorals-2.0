"""Animato Studio auto-post mission (port step 2) — the paid tier, free in Devon.

A series that re-runs the spark → build → share loop on a schedule and
posts each finished video itself. It reads a :class:`StudioProject`'s
``schedule`` (``"daily"``, ``"weekly"``, or an ISO time) and the shorts
factory's :class:`ContentCalendar`, runs
:class:`StudioRunner.spark` once per due slot (consuming the project's
topic queue where present, else the next due calendar post in the
project's niche), and publishes through the YouTube publisher.

Posture is the standing autonomy rule: once the mission is started she
runs the schedule herself — the connected YouTube channel auth is the
gate, there is no per-video review step. A failed publish still hands
back the video file (the episode stays ``ready`` for a manual
``animato post``), and the slot is consumed: one attempt per slot, so a
persistent failure cannot hammer the render every 30 minutes. Every
run is journaled (what posted, what failed and why) in
``<studio_root>/autopost.json``, visible through
``animato autopost status``.

Entry points:

* :func:`run_autopost` — the sweep itself; the single function every
  trigger calls.
* the ``animato_autopost`` agent tool (``tools/animato.py``) +
  :func:`ensure_animato_autopost_job` — the durable path: every 30
  minutes the runtime's scheduler fires the tool, exactly like the
  shorts publisher's durable job.
* ``nm animato autopost start|stop|status|run`` — the CLI surface,
  in the same ``animato`` command as step 1.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from .studio import StudioProject, StudioRunner

_log = get_logger(__name__)

__all__ = [
    "AUTOPOST_JOB",
    "MAX_JOURNAL_RUNS",
    "AutopostStore",
    "autopost_status",
    "ensure_animato_autopost_job",
    "is_due",
    "parse_schedule",
    "run_autopost",
]

AUTOPOST_JOB = "animato autopost"

#: Sweep cadence for the durable job — same as the shorts publisher:
#: often enough that a slot fires promptly, sparse enough that a
#: render (minutes long) never overlaps itself.
SWEEP_SPEC = "every 30m"

#: How many journaled runs the state file keeps; older runs drop off
#: the tail (the episodes themselves stay on the project manifest).
MAX_JOURNAL_RUNS = 200


def _utcnow() -> str:
    # NOTE: timezone.utc, not datetime.UTC — this file imports the
    # datetime CLASS, and the UTC alias lives on the datetime MODULE.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # noqa: UP017


def parse_schedule(value: str) -> dict[str, Any]:
    """Parse a StudioProject ``schedule`` string.

    ``"daily"`` → one slot per UTC day; ``"weekly"`` → one slot per
    7 days; an ISO datetime → one slot at that time; ``""`` → no
    schedule. Anything else raises ValueError naming the value —
    an unparseable schedule is journaled as a skip, never guessed.
    """
    raw = (value or "").strip()
    if not raw:
        return {"kind": "none", "at": None}
    low = raw.lower()
    if low == "daily":
        return {"kind": "daily", "at": None}
    if low == "weekly":
        return {"kind": "weekly", "at": None}
    try:
        return {"kind": "once", "at": _parse_iso(raw)}
    except ValueError:
        raise ValueError(
            f"unparseable schedule {value!r} — use 'daily', 'weekly', "
            "or an ISO time") from None


def is_due(schedule: dict[str, Any], last_attempt_at: str,
           now: datetime | None = None) -> bool:
    """True when the schedule has an open slot right now.

    ``last_attempt_at`` is the ISO time of the project's last mission
    attempt (any outcome — one attempt per slot). ``now`` defaults to
    this instant, UTC.
    """
    kind = schedule.get("kind")
    if kind == "none":
        return False
    now = now or datetime.now(timezone.utc)  # noqa: UP017
    last: datetime | None = None
    if last_attempt_at:
        try:
            last = _parse_iso(last_attempt_at)
        except ValueError:
            last = None
    if kind == "daily":
        if last is None:
            return True
        return last.date() < now.date()
    if kind == "weekly":
        if last is None:
            return True
        return (now - last) >= timedelta(days=7)
    if kind == "once":
        at = schedule.get("at")
        if at is None or now < at:
            return False
        return last is None or last < at
    return False


class AutopostStore:
    """Mission state + run journal under ``<studio_root>/autopost.json``.

    ``enabled`` is the start/stop switch (off until the owner starts
    the mission — starting it is the authorization). ``projects``
    records the last attempt per project slug. ``runs`` is the
    journal: one entry per run, newest last, capped at
    :data:`MAX_JOURNAL_RUNS`.
    """

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "autopost.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        self.enabled = False
        self.projects: dict[str, dict[str, str]] = {}
        self.runs: list[dict[str, Any]] = []
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log.warning("autopost: corrupt %s (%s) — starting empty",
                         self.path, exc)
            return
        if not isinstance(raw, dict):
            return
        self.enabled = bool(raw.get("enabled", False))
        projects = raw.get("projects") or {}
        if isinstance(projects, dict):
            self.projects = {str(k): dict(v) for k, v in projects.items()
                             if isinstance(v, dict)}
        runs = raw.get("runs") or []
        if isinstance(runs, list):
            self.runs = [dict(r) for r in runs if isinstance(r, dict)]

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"enabled": self.enabled, "projects": self.projects,
             "runs": self.runs[-MAX_JOURNAL_RUNS:]},
            indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def set_enabled(self, enabled: bool) -> bool:
        """Flip the start/stop switch; returns the new state."""
        self.enabled = bool(enabled)
        self._save()
        return self.enabled

    def last_attempt(self, slug: str) -> str:
        """ISO time of the project's last mission attempt ("" = never)."""
        return str((self.projects.get(slug) or {}).get("last_run_at") or "")

    def record_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Journal one run and update the project's last attempt."""
        entry = {"at": _utcnow(), **run}
        self.runs.append(entry)
        self.runs = self.runs[-MAX_JOURNAL_RUNS:]
        slug = str(run.get("project") or "")
        if slug:
            self.projects[slug] = {
                "last_run_at": entry["at"],
                "last_status": str(run.get("status") or ""),
                "last_episode_id": str(run.get("episode_id") or ""),
            }
        self._save()
        return entry

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Newest journal entries first."""
        return list(reversed(self.runs[-max(0, int(limit)):]))


def _calendar_due_for(pipeline: Any, niche: str,
                      now: datetime) -> Any | None:
    """Next due calendar post in this niche (None when there is none).

    The calendar is the owner's queue of planned shorts — a project
    with an empty topic queue takes its next episode idea from here
    instead of inventing one. A missing/unreadable calendar is not
    fatal: the topic queue is the primary source.
    """
    calendar = getattr(pipeline, "calendar", None)
    if calendar is None:
        return None
    try:
        due = calendar.due(now)
    except Exception as exc:  # noqa: BLE001 — calendar is a bonus source
        _log.debug("autopost: calendar.due failed: %s", exc)
        return None
    for post in due or []:
        if getattr(post, "niche", "") == niche:
            return post
    return None


def _calendar_mark(pipeline: Any, post: Any, status: str,
                   job_id: str = "") -> None:
    calendar = getattr(pipeline, "calendar", None)
    if calendar is None or post is None:
        return
    try:
        calendar.mark(post.id, status, job_id=job_id)
    except Exception as exc:  # noqa: BLE001 — the journal is the record
        _log.debug("autopost: calendar.mark failed: %s", exc)


def run_autopost(*, runner: StudioRunner | None = None,
                 store: AutopostStore | None = None,
                 now: datetime | None = None,
                 limit: int = 5,
                 project: str = "",
                 force: bool = False) -> dict[str, Any]:
    """Run one auto-post sweep: every due project posts one episode.

    ``force`` runs due projects even when the mission is stopped
    (the manual ``autopost run``); the durable job calls with
    ``force=False``, so stopping the mission stops the schedule.
    ``project`` narrows the sweep to one slug; combined with
    ``force`` it re-runs that project even when its slot is not open.
    Returns ``{"ok", "enabled", "ran", "posted", "failed",
    "skipped", "outcomes"}`` — one outcome dict per project touched.
    """
    if runner is None:
        runner = StudioRunner()
    if store is None:
        store = AutopostStore(runner.store.root)
    moment = now or datetime.now(timezone.utc)  # noqa: UP017
    projects = runner.store.list()
    if project:
        projects = [p for p in projects if p.slug == project]
        if not projects:
            return {"ok": False, "enabled": store.enabled, "ran": 0,
                    "posted": 0, "failed": 0, "skipped": 0,
                    "outcomes": [],
                    "reason": f"unknown studio project {project!r}"}
    if not store.enabled and not force:
        return {"ok": True, "enabled": False, "ran": 0, "posted": 0,
                "failed": 0, "skipped": 0, "outcomes": [],
                "note": "autopost is off — `animato autopost start` "
                        "to run the schedule"}
    outcomes: list[dict[str, Any]] = []
    for proj in projects:
        if len(outcomes) >= max(0, int(limit)):
            break
        outcomes.append(_run_one_project(runner, store, proj, moment,
                                         force=force))
    posted = sum(1 for o in outcomes if o.get("status") == "posted")
    failed = sum(1 for o in outcomes if o.get("status") == "failed")
    skipped = sum(1 for o in outcomes if o.get("status") == "skipped")
    return {"ok": True, "enabled": store.enabled,
            "ran": len(outcomes), "posted": posted, "failed": failed,
            "skipped": skipped, "outcomes": outcomes}


def _run_one_project(runner: StudioRunner, store: AutopostStore,
                     project: StudioProject, now: datetime,
                     *, force: bool) -> dict[str, Any]:
    """One project's slot: due check → topic → spark → publish → journal."""
    slug = project.slug
    try:
        schedule = parse_schedule(project.schedule)
    except ValueError as exc:
        outcome = {"project": slug, "topic": "", "episode_id": "",
                   "status": "skipped", "video_path": "", "video_id": "",
                   "reason": str(exc)}
        store.record_run(outcome)
        return outcome
    if not force and not is_due(schedule, store.last_attempt(slug), now):
        return {"project": slug, "topic": "", "episode_id": "",
                "status": "not-due", "video_path": "", "video_id": "",
                "reason": f"schedule {project.schedule!r} — no open slot"}
    topic = ""
    cal_post: Any = None
    if project.topic_queue:
        topic = project.topic_queue[0]
    else:
        cal_post = _calendar_due_for(runner.pipeline, project.niche, now)
        if cal_post is not None:
            topic = str(getattr(cal_post, "topic", "") or "")
    if not topic:
        outcome = {"project": slug, "topic": "", "episode_id": "",
                   "status": "skipped", "video_path": "", "video_id": "",
                   "reason": "no topic queued — queue one "
                             f"(`animato queue {slug} \"<topic>\") or add "
                             "a planned post to the shorts calendar"}
        store.record_run(outcome)
        return outcome
    if cal_post is not None:
        _calendar_mark(runner.pipeline, cal_post, "rendering")
    result = runner.spark(project, topic, publish="youtube")
    episode_id = str(result.get("episode_id") or "")
    video_path = str(result.get("video_path") or "")
    job_id = str(result.get("job_id") or "")
    if result.get("ok") and result.get("published"):
        video_id = str(result["published"].get("youtube") or "")
        outcome = {"project": slug, "topic": topic,
                   "episode_id": episode_id, "status": "posted",
                   "video_path": video_path, "video_id": video_id,
                   "reason": ""}
        store.record_run(outcome)
        if cal_post is not None:
            _calendar_mark(runner.pipeline, cal_post, "posted",
                           job_id=job_id)
        return outcome
    # Failed render or failed publish — the reason names it, the
    # video (when one exists) is kept for a manual `animato post`.
    if result.get("ok"):
        reason = "; ".join(result.get("notes", []) or ["not posted"])
    else:
        reason = "; ".join(result.get("notes", []) or ["spark failed"])
    outcome = {"project": slug, "topic": topic,
               "episode_id": episode_id, "status": "failed",
               "video_path": video_path, "video_id": "",
               "reason": reason}
    store.record_run(outcome)
    if cal_post is not None:
        _calendar_mark(runner.pipeline, cal_post, "failed", job_id=job_id)
    return outcome


def autopost_status(*, runner: StudioRunner | None = None,
                    store: AutopostStore | None = None,
                    now: datetime | None = None) -> dict[str, Any]:
    """Mission state for `autopost status`: switch, per-project slots, recent runs."""
    if runner is None:
        runner = StudioRunner()
    if store is None:
        store = AutopostStore(runner.store.root)
    moment = now or datetime.now(timezone.utc)  # noqa: UP017
    rows = []
    for proj in runner.store.list():
        try:
            schedule = parse_schedule(proj.schedule)
            due = is_due(schedule, store.last_attempt(proj.slug), moment)
            sched_note = proj.schedule or "(not set)"
        except ValueError as exc:
            due = False
            sched_note = f"{proj.schedule!r} — {exc}"
        last = store.projects.get(proj.slug) or {}
        rows.append({
            "project": proj.slug, "name": proj.name,
            "schedule": sched_note, "due": due,
            "last_run_at": str(last.get("last_run_at") or ""),
            "last_status": str(last.get("last_status") or ""),
            "last_episode_id": str(last.get("last_episode_id") or ""),
            "queued_topics": len(proj.topic_queue),
        })
    return {"ok": True, "enabled": store.enabled, "projects": rows,
            "recent": store.recent(10)}


# ── durable job (the runtime's scheduler) ─────────────────────────────

def ensure_animato_autopost_job(scheduler: Any) -> list[dict[str, Any]]:
    """Register the durable auto-post job (idempotent by name, the
    shorts-publisher pattern): every 30 minutes, fire the
    ``animato_autopost`` tool's ``run`` action. Routine: outcomes
    surface through the mission journal and `animato autopost
    status`, not job alerts — a failed sweep still records as
    failed in the journal."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    if AUTOPOST_JOB in have:
        return [{"name": AUTOPOST_JOB, "already_scheduled": True}]
    job = scheduler.add(
        AUTOPOST_JOB, SWEEP_SPEC, "tool",
        {"tool": "animato_autopost", "args": {"action": "run"}},
        missed_fire_policy="skip", overlap_policy="skip",
        run_timeout_s=1800, routine=True)
    out.append({"name": AUTOPOST_JOB, "scheduled": True,
                "job_id": job.get("id")})
    return out
