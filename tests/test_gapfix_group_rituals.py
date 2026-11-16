"""Group digest + rituals (audit gap R2-29).

The organ (nomorals/community/digest.py) is finished pure code —
build_digest / render_digest / RitualScheduler / DEFAULT_RITUALS —
with zero production callers. These tests pin the wiring: the
/cgroup digest verb (digest over a themed group's own feed), the
/cgroup rituals verb (list / run / enable / disable), and the
durable "group ritual tick" job that posts due rituals into the
group feed on their own clock.

Seams copied from the neighbouring lanes: the mirror lane's
_cgroup_setup (monkeypatched gmod._DATA_DIR + gid parsed from the
create reply) and the WhatsApp-digest lane's fake-scheduler job test
and DecoRegistry tool test.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from nomorals.community import digest as dmod
from nomorals.community import groups as gmod
from nomorals.community.digest import DEFAULT_RITUALS, Ritual
from nomorals.community.groups import GroupStore, control_cgroup

FRIDAY_0700 = datetime(2026, 10, 16, 7, 0, tzinfo=UTC)
THURSDAY_0700 = datetime(2026, 10, 15, 7, 0, tzinfo=UTC)


def _setup(tmp_path, monkeypatch):
    """One themed group (creator Ada, member Bob) under a tmp data dir."""
    monkeypatch.setattr(gmod, "_DATA_DIR", tmp_path / "themed")
    created = control_cgroup("new Testers | testing", context=None,
                             chat=None, sender="Ada", sender_id="Ada")
    m = re.search(r"`(grp_[^`]+)`", created)
    assert m, created
    gid = m.group(1)
    joined = control_cgroup(f"join {gid}", context=None, chat=None,
                            sender="Bob", sender_id="Bob")
    assert "Welcome" in joined, joined
    return gid


def _seed_feed(gid):
    store = GroupStore()
    assert store.post(gid, "Ada",
                      "Should we move the meetup to Saturday?") is not None
    assert store.post(gid, "Bob",
                      "What do you think about the new logo?") is not None
    assert store.post(gid, "Bob",
                      "@Ada can you confirm the venue?") is not None
    assert store.post(gid, "Ada", "Venue confirmed ✅") is not None
    assert store.post(gid, "Ada", "Lunch at noon") is not None


def _seed_last_delivered(gid, when):
    store = GroupStore()
    rituals = [Ritual(name=r.name, kind=r.kind, rrule=r.rrule,
                      template=r.template, group_id=gid, enabled=True,
                      last_delivered=when.timestamp())
               for r in DEFAULT_RITUALS]
    store.save_rituals(gid, rituals)
    return rituals


# ── /cgroup digest ─────────────────────────────────────────────────────


def test_digest_ranks_decisions_and_flags_unanswered(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    _seed_feed(gid)
    out = control_cgroup(f"digest {gid}", context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "While you were away" in out
    assert "Mentioned you" in out
    assert "Top 3 decisions" in out
    assert "Unanswered (3)" in out
    # A resolved message that was never a decision is not rendered;
    # neither is small talk.
    assert "Venue confirmed" not in out
    assert "Lunch at noon" not in out


def test_digest_empty_feed_is_honest(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"digest {gid}", context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "Nothing needs you — all quiet." in out
    assert "0 messages scanned" in out


def test_digest_nonmember_refused(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"digest {gid}", context=None, chat=None,
                         sender="Zed", sender_id="Zed")
    assert "only members" in out.lower()
    assert "While you were away" not in out


def test_digest_unknown_group(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = control_cgroup("digest grp_nope", context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "No group" in out


# ── /cgroup rituals ────────────────────────────────────────────────────


def test_rituals_list_shows_defaults_and_next_due(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid}", context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    for name in ("Daily verse", "Good morning", "Market open",
                 "Friday vibes", "Weekly poll"):
        assert name in out, out
    assert "next" in out.lower()


def test_rituals_run_posts_to_feed_and_marks_delivered(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} run Good morning", context=None,
                         chat=None, sender="Ada", sender_id="Ada")
    assert "Good morning" in out
    store = GroupStore()
    feed = store.feed(gid, limit=5)
    assert feed and feed[0].author == "Devon"
    assert "Good morning" in feed[0].text
    rituals = store.load_rituals(gid)
    morning = [r for r in rituals if r.name == "Good morning"][0]
    assert morning.last_delivered is not None


def test_rituals_run_nonadmin_refused(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} run Good morning", context=None,
                         chat=None, sender="Bob", sender_id="Bob")
    assert "only" in out.lower() and "admin" in out.lower()
    assert GroupStore().feed(gid) == []


def test_rituals_unknown_name_lists_names(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} run Not a ritual", context=None,
                         chat=None, sender="Ada", sender_id="Ada")
    assert "Good morning" in out
    assert "No ritual" in out


def test_rituals_disable_then_enable(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    out = control_cgroup(f"rituals {gid} disable Good morning",
                         context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "disabled" in out.lower()
    listed = control_cgroup(f"rituals {gid}", context=None, chat=None,
                            sender="Ada", sender_id="Ada")
    assert "Good morning" in listed
    assert "disabled" in listed.lower()
    out = control_cgroup(f"rituals {gid} enable Good morning",
                         context=None, chat=None,
                         sender="Ada", sender_id="Ada")
    assert "enabled" in out.lower()


# ── ritual tick driver ─────────────────────────────────────────────────


def _tick(store, now):
    return dmod.run_group_ritual_tick_once(store=store, now=now)


def test_tick_delivers_due_rituals_once(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    _seed_last_delivered(gid, THURSDAY_0700)
    store = GroupStore()
    result = _tick(store, FRIDAY_0700)
    assert result["ok"] is True
    names = sorted(d["ritual"] for d in result["delivered"])
    # Market open is due too: with last delivery Thu 07:00, its next
    # occurrence is Thu 09:30 — in the past, never delivered.
    assert names == ["Daily verse", "Good morning", "Market open"], result
    feed = store.feed(gid, limit=5)
    assert sorted(p.author for p in feed) == ["Devon"] * 3
    # Second tick at the same instant: nothing due, nothing re-posted.
    result2 = _tick(store, FRIDAY_0700)
    assert result2["delivered"] == []
    assert len(store.feed(gid, limit=10)) == 3


def test_tick_nothing_due_fires_nothing(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    _seed_last_delivered(gid, FRIDAY_0700)
    store = GroupStore()
    result = _tick(store, FRIDAY_0700)
    assert result["ok"] is True
    assert result["delivered"] == []
    assert store.feed(gid) == []


def test_tick_respects_disabled(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    rituals = _seed_last_delivered(gid, THURSDAY_0700)
    for r in rituals:
        if r.name == "Good morning":
            r.enabled = False
    GroupStore().save_rituals(gid, rituals)
    result = _tick(GroupStore(), FRIDAY_0700)
    assert [d["ritual"] for d in result["delivered"]] == \
        ["Daily verse", "Market open"]


def test_tick_fresh_group_catches_up_once(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    store = GroupStore()
    result = _tick(store, FRIDAY_0700)
    names = sorted(d["ritual"] for d in result["delivered"])
    assert names == ["Daily verse", "Good morning", "Market open"], result
    result2 = _tick(store, FRIDAY_0700)
    assert result2["delivered"] == []


def test_tick_one_bad_group_never_kills_the_pass(tmp_path, monkeypatch):
    gid_a = _setup(tmp_path, monkeypatch)
    store = GroupStore()
    bad = Ritual(name="Broken", kind="custom",
                 rrule="FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
                 template="{nope} broke", group_id=gid_a)
    store.save_rituals(gid_a, [bad])
    created = control_cgroup("new Second | more", context=None, chat=None,
                             sender="Ada", sender_id="Ada")
    gid_b = re.search(r"`(grp_[^`]+)`", created).group(1)
    _seed_last_delivered(gid_b, THURSDAY_0700)
    result = _tick(store, FRIDAY_0700)
    assert result["ok"] is True
    assert gid_a in [f["group"] for f in result["failed"]], result
    assert sorted(d["ritual"] for d in result["delivered"]) == \
        ["Daily verse", "Good morning", "Market open"]


def test_tick_corrupt_rituals_file_degrades_to_defaults(tmp_path, monkeypatch):
    gid = _setup(tmp_path, monkeypatch)
    store = GroupStore()
    path = store.data_dir / f"rituals_{gid}.json"
    path.write_text("{not json", encoding="utf-8")
    result = _tick(store, FRIDAY_0700)
    assert result["ok"] is True
    assert sorted(d["ritual"] for d in result["delivered"]) == \
        ["Daily verse", "Good morning", "Market open"]


# ── durable job + tool ─────────────────────────────────────────────────


def test_ensure_ritual_job_idempotent_and_stale_replaced():
    from nomorals.community.digest import (
        RITUAL_JOB,
        RITUAL_SPEC,
        RITUAL_TOOL,
        ensure_group_ritual_job,
    )

    class _Sched:
        def __init__(self):
            self.jobs = []

        def list_jobs(self):
            return list(self.jobs)

        def add(self, name, spec, kind, payload, routine=False):
            job = {"name": name, "spec": spec, "kind": kind,
                   "payload": payload, "routine": routine}
            self.jobs.append(job)
            return job

        def remove(self, job_id):
            self.jobs = [j for j in self.jobs
                         if j.get("id") != job_id]

    sched = _Sched()
    first = ensure_group_ritual_job(sched)
    assert first["name"] == RITUAL_JOB
    assert first["spec"] == RITUAL_SPEC
    assert first["payload"]["tool"] == RITUAL_TOOL
    assert first["routine"] is True
    second = ensure_group_ritual_job(sched)
    assert second == first
    assert len(sched.jobs) == 1
    third = ensure_group_ritual_job(sched, spec="every 10m")
    assert third is not first and len(sched.jobs) == 1
    assert third["spec"] == "every 10m"


def test_ritual_tick_tool_registered_and_never_raises(tmp_path, monkeypatch):
    from nomorals.tools.autonomy import register

    monkeypatch.setattr(gmod, "_DATA_DIR", tmp_path / "themed")
    registered = {}

    class DecoRegistry:
        context = None

        def register(self, name, fn=None, **kw):
            def deco(f):
                registered[name] = f
                return f
            if fn is not None:
                return deco(fn)
            return deco

    register(DecoRegistry())
    assert "__group_ritual_tick__" in registered
    result = registered["__group_ritual_tick__"]()
    assert result["ok"] is True
    assert result["delivered"] == []


# ── isolation guard (passes before and after; the organ must stay L4) ──


def test_digest_module_has_no_hard_groups_import():
    import ast
    from pathlib import Path

    # Module level only: the tick driver lazily imports GroupStore
    # inside the function (the documented cycle-avoidance pattern —
    # groups.py imports this module for the /cgroup verbs), which is
    # not a hard import.
    tree = ast.parse(Path(dmod.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "groups" not in node.module, \
                f"digest.py hard-imports {node.module}"
        elif isinstance(node, ast.Import):
            assert not any("community.groups" in a.name
                           for a in node.names)
