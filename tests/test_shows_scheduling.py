"""Shows, scheduled on their own clock.

The shows lane (audit gap R1-4) left automatic episode scheduling
as an open call in shows/library.json: a show's schedule was free
text and no scheduler read it. Owner order 2026-11-16: an
autonomous system does everything itself — the schedule field is
REAL now. Every test here drives the tick handler directly
(``run_show_schedule_tick_once``) with fakes — no live network,
no live scheduler thread — plus the schedule parser and the
durable job registration.

Rules pinned here:
- a show with "daily 06:00" gets a draft episode when the job
  runs past 06:00 with no episode for that slot;
- a second job run for the same slot creates NOTHING (the
  durable last-slot watermark holds);
- a show with no schedule, or a schedule the parser cannot read,
  is skipped, never crashed — an unparseable schedule is noted
  on the show row once per distinct text;
- the created episode's segments come from the morning-pulse and
  research sources through the composer's own grammar
  (script/voice/bed/ad), plus the show's own bed track as a bed
  segment;
- the job NEVER publishes: a scheduled episode is a draft until
  the owner publishes it with /show publish (the publicface
  gate stays the only way to a public link).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from nomorals.shows.composer import Show
from nomorals.shows.scheduling import (
    SHOW_SCHEDULE_JOB,
    SHOW_SCHEDULE_SPEC,
    SHOW_SCHEDULE_TOOL,
    due_slots,
    ensure_show_schedule_job,
    parse_schedule,
    run_show_schedule_tick_once,
)
from nomorals.shows.store import ShowStore

#: WAT — Africa/Lagos, UTC+1 year-round, no DST. The scheduling
#: module uses a fixed +01:00 offset for the same reason (exact
#: for Lagos, no tzdata dependency).
WAT = timezone(timedelta(hours=1))

MON_0700 = datetime(2026, 11, 16, 7, 0, tzinfo=WAT)  # a Monday
MON_0530 = datetime(2026, 11, 16, 5, 30, tzinfo=WAT)
TUE_0700 = datetime(2026, 11, 17, 7, 0, tzinfo=WAT)


def _store(tmp_path, shows=()):
    path = Path(tmp_path) / "shows.json"
    store = ShowStore(path)
    for show in shows:
        store.put_show(show.to_dict())
    return store


def _pulse():
    return [
        {"title": "Overnight", "text": "Markets rose overnight."},
        {"title": "News", "text": "Two headlines worth hearing."},
    ]


def _research():
    return [{"title": "Trend: video essays",
             "text": "Video essays are moving in your field."}]


# ── schedule grammar ────────────────────────────────────────────────


def test_parse_daily():
    parsed = parse_schedule("daily 06:00")
    assert parsed is not None and parsed.kind == "daily"
    assert (parsed.hour, parsed.minute) == (6, 0)


def test_parse_daily_variants():
    for text in ("every day 06:00", "every day 06:00 WAT",
                 "daily 6:00", "Daily 06:00"):
        parsed = parse_schedule(text)
        assert parsed is not None and parsed.kind == "daily", text
        assert (parsed.hour, parsed.minute) == (6, 0), text


def test_parse_weekly():
    parsed = parse_schedule("weekly Mon 06:00")
    assert parsed is not None and parsed.kind == "weekly"
    assert parsed.weekday == 0
    assert (parsed.hour, parsed.minute) == (6, 0)
    assert parse_schedule("weekly Monday 18:30").weekday == 0
    assert parse_schedule("weekly fri 09:00").weekday == 4


def test_parse_interval():
    parsed = parse_schedule("every 6h")
    assert parsed is not None and parsed.kind == "interval"
    assert parsed.interval_seconds == 6 * 3600
    assert parse_schedule("every 90m").interval_seconds == 90 * 60
    assert parse_schedule("every 2 hours").interval_seconds == 2 * 3600


def test_parse_garbage_refuses():
    for text in ("banana", "daily", "daily 25:00", "weekly 06:00",
                 "every 0h", "every 1m", "daily 06:00 extra words"):
        assert parse_schedule(text) is None, text


# ── due slots ───────────────────────────────────────────────────────


def test_due_daily_before_time_is_empty():
    parsed = parse_schedule("daily 06:00")
    assert due_slots(parsed, MON_0530, None) == []


def test_due_daily_after_time_is_today_slot():
    parsed = parse_schedule("daily 06:00")
    slots = due_slots(parsed, MON_0700, None)
    assert len(slots) == 1
    assert slots[0] == datetime(2026, 11, 16, 6, 0, tzinfo=WAT)


def test_due_daily_watermark_holds():
    parsed = parse_schedule("daily 06:00")
    last = datetime(2026, 11, 16, 6, 0, tzinfo=WAT)
    assert due_slots(parsed, MON_0700, last) == []
    later = datetime(2026, 11, 17, 7, 0, tzinfo=WAT)
    slots = due_slots(parsed, later, last)
    assert slots == [datetime(2026, 11, 17, 6, 0, tzinfo=WAT)]


def test_due_weekly():
    parsed = parse_schedule("weekly Mon 06:00")
    assert due_slots(parsed, MON_0700, None) == [
        datetime(2026, 11, 16, 6, 0, tzinfo=WAT)]
    assert due_slots(parsed, TUE_0700, None) == []


def test_due_interval_first_run_is_now():
    parsed = parse_schedule("every 6h")
    slots = due_slots(parsed, MON_0700, None)
    assert len(slots) == 1
    assert slots[0] == datetime(2026, 11, 16, 7, 0, tzinfo=WAT)


def test_due_interval_after_watermark():
    parsed = parse_schedule("every 6h")
    last = datetime(2026, 11, 16, 7, 0, tzinfo=WAT)
    assert due_slots(parsed, datetime(2026, 11, 16, 10, 0,
                                      tzinfo=WAT), last) == []
    slots = due_slots(parsed, datetime(2026, 11, 16, 13, 0,
                                       tzinfo=WAT), last)
    assert slots == [datetime(2026, 11, 16, 13, 0, tzinfo=WAT)]


# ── the tick ────────────────────────────────────────────────────────


def test_tick_creates_draft_episode_from_pulse_and_research(tmp_path):
    show = Show(title="Morning Pulse", schedule="daily 06:00",
                bed_track="lofi.mp3")
    store = _store(tmp_path, [show])
    summary = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert summary["ok"] is True, summary
    assert summary["created"], summary
    episodes = store.list_episodes(show.id)
    assert len(episodes) == 1
    ep = episodes[0]
    assert "2026-11-16 06:00" in ep["title"]
    assert ep.get("published") is not True  # draft until /show publish
    assert "Markets rose overnight." in ep["script_text"]
    assert "Video essays are moving" in ep["script_text"]
    kinds = [s["kind"] for s in ep["segments"]]
    assert kinds.count("script") == 3  # two pulse + one research
    assert kinds.count("bed") == 1
    assert ep["segments"][-1] == {"kind": "bed", "ref": "lofi.mp3"}
    row = store.get_show(show.id)
    assert row["sched_last_slot"] == "2026-11-16T06:00:00+01:00"
    assert not row.get("sched_note")


def test_tick_second_run_same_slot_creates_nothing(tmp_path):
    show = Show(title="Morning Pulse", schedule="daily 06:00")
    store = _store(tmp_path, [show])
    run_show_schedule_tick_once(store=store, now=MON_0700,
                                pulse=_pulse, research=_research)
    summary = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert summary["created"] == []
    assert len(store.list_episodes(show.id)) == 1


def test_tick_skips_empty_schedule_silently(tmp_path):
    show = Show(title="No Schedule", schedule="")
    store = _store(tmp_path, [show])
    summary = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert summary["created"] == []
    assert store.list_episodes(show.id) == []
    assert not store.get_show(show.id).get("sched_note")


def test_tick_unparseable_schedule_noted_once_not_crashed(tmp_path):
    show = Show(title="Bad Schedule", schedule="banana o'clock")
    store = _store(tmp_path, [show])
    first = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert first["created"] == []
    assert store.list_episodes(show.id) == []
    note = store.get_show(show.id).get("sched_note") or ""
    assert "banana o'clock" in note
    assert "daily 06:00" in note  # the note names a usable shape
    second = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert second["created"] == []
    assert store.get_show(show.id).get("sched_note") == note


def test_tick_no_material_skips_slot_and_retries_later(tmp_path):
    show = Show(title="Morning Pulse", schedule="daily 06:00")
    store = _store(tmp_path, [show])
    summary = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=lambda: [],
        research=lambda: [])
    assert summary["created"] == []
    assert store.list_episodes(show.id) == []
    row = store.get_show(show.id)
    assert "no material" in (row.get("sched_note") or "")
    assert not row.get("sched_last_slot")  # watermark NOT advanced
    later = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=_pulse, research=_research)
    assert len(later["created"]) == 1
    assert len(store.list_episodes(show.id)) == 1


def test_tick_one_bad_show_never_kills_the_pass(tmp_path):
    good = Show(title="Good", schedule="daily 06:00")
    bad = Show(title="Bad", schedule="daily 06:00")
    store = _store(tmp_path, [good, bad])

    def flaky_pulse():
        raise RuntimeError("pulse exploded")

    # The good show's segments come from research only when the
    # pulse reader raises — the pass must still compose it.
    summary = run_show_schedule_tick_once(
        store=store, now=MON_0700, pulse=flaky_pulse,
        research=_research)
    assert summary["ok"] is True
    assert len(store.list_episodes(good.id)) == 1


def test_tick_store_survives_reload(tmp_path):
    show = Show(title="Morning Pulse", schedule="daily 06:00")
    store = _store(tmp_path, [show])
    run_show_schedule_tick_once(store=store, now=MON_0700,
                                pulse=_pulse, research=_research)
    reloaded = ShowStore(Path(tmp_path) / "shows.json")
    assert len(reloaded.list_episodes(show.id)) == 1
    assert reloaded.get_show(show.id)["sched_last_slot"]


# ── job registration ────────────────────────────────────────────────


class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"id": f"job-{len(self.jobs) + 1}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


def test_ensure_job_registers_and_is_idempotent():
    sched = _FakeScheduler()
    first = ensure_show_schedule_job(sched)
    assert first is not None
    assert first["name"] == SHOW_SCHEDULE_JOB
    assert first["spec"] == SHOW_SCHEDULE_SPEC
    assert first["payload"]["tool"] == SHOW_SCHEDULE_TOOL
    second = ensure_show_schedule_job(sched)
    assert second["id"] == first["id"]
    assert len(sched.jobs) == 1


def test_ensure_job_none_scheduler_is_none():
    assert ensure_show_schedule_job(None) is None


def test_ensure_job_replaces_stale_spec():
    sched = _FakeScheduler()
    sched.add(SHOW_SCHEDULE_JOB, "every 1h", "tool",
              {"tool": SHOW_SCHEDULE_TOOL, "args": {}})
    job = ensure_show_schedule_job(sched)
    assert job["spec"] == SHOW_SCHEDULE_SPEC
    assert len(sched.jobs) == 1
