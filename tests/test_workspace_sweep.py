"""Sweep tests: workspace module upgrade (vcpu / workspace / inbox / rooms).

Covers the mined-then-built additions:
- VirtualCPU: named tasks, soft timeouts, batch submit, rebirth, stealing,
  latency percentiles
- Workspace: proportional autoscale, rebalance, kind lanes, scale history,
  render_status, map
- Inbox: rule engine, duplicate detection, confidence gating, priority
  sweeps, digest/cards, search/annotate, batch retry
- Rooms: templates, brief, index, review, fork, tags, cards
"""
import tempfile
import threading
import time
from pathlib import Path

import pytest

from nomorals.workspace.vcpu import VirtualCPU, VcpuStatus
from nomorals.workspace.workspace import Workspace
from nomorals.workspace.inbox import (
    ActionResult, Inbox, InboxRule, rule_matches,
)
from nomorals.workspace.rooms import RoomManager, ROOM_TEMPLATES
from nomorals.storage.db import Database


# ── helpers ──────────────────────────────────────────────────────────────

class FakeNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title, body="", critical=False):
        self.published.append((kind, title, body, critical))


def make_inbox(tmp, **kw):
    kw.setdefault("notifier", FakeNotifier())
    kw.setdefault("scheduler", object())  # never used by these tests
    return Inbox(tmp, **kw)


def wait_for(pred, timeout=8.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ── VirtualCPU ───────────────────────────────────────────────────────────

def test_vcpu_named_task_and_latency():
    cpu = VirtualCPU(0)
    try:
        fut = cpu.submit(lambda: 42, name="the-answer")
        assert fut.result(timeout=5) == 42
        assert cpu.latency_stats()["count"] >= 1
        assert cpu.latency_stats()["p50_ms"] >= 0
        d = cpu.to_dict()
        assert "current_task" in d and "latency" in d
        assert "orphan_threads" in d
    finally:
        cpu.stop()


def test_vcpu_soft_timeout():
    cpu = VirtualCPU(0)
    try:
        fut = cpu.submit(lambda: time.sleep(5), name="slowpoke", timeout=0.2)
        with pytest.raises(TimeoutError):
            fut.result(timeout=5)
        assert cpu.stats["tasks_timed_out"] == 1
        # the core itself survives the timeout and keeps working
        assert cpu.submit(lambda: "alive").result(timeout=5) == "alive"
    finally:
        cpu.stop()


def test_vcpu_submit_many():
    cpu = VirtualCPU(0)
    try:
        futs = cpu.submit_many([
            {"fn": lambda x: x * 2, "args": (i,), "name": f"dbl-{i}"}
            for i in range(6)
        ])
        assert [f.result(timeout=5) for f in futs] == [i * 2 for i in range(6)]
        assert cpu.stats["tasks_queued"] == 6
    finally:
        cpu.stop()


def test_vcpu_rebirth_after_max_tasks():
    cpu = VirtualCPU(0, max_tasks=3)
    try:
        futs = [cpu.submit(lambda: None) for _ in range(3)]
        for f in futs:
            f.result(timeout=5)
        assert wait_for(lambda: cpu.stats["rebirths"] >= 1), \
            "worker thread should rebirth after max_tasks"
        # replacement thread still serves work
        assert cpu.submit(lambda: "reborn").result(timeout=5) == "reborn"
    finally:
        cpu.stop()


def test_vcpu_work_stealing():
    donor = VirtualCPU(1)
    thief = VirtualCPU(2)
    gate = threading.Event()
    try:
        # deterministic: the donor's worker is blocked on the gate task,
        # so the 4 follow-ups pile up in its queue (no pause race)
        blocker = donor.submit(lambda: gate.wait(10), name="gate")
        queued = [donor.submit(lambda: "w", name=f"w{i}") for i in range(4)]
        assert wait_for(lambda: donor.queue_depth() == 4), \
            "donor should hold 4 queued tasks behind the gate"
        moved = thief.steal_from(donor)
        assert moved == 2, f"expected half of 4 stolen, got {moved}"
        assert donor.queue_depth() == 2
        assert thief.queue_depth() == 2
        assert donor.stats["tasks_stolen_out"] == 2
        assert thief.stats["tasks_stolen_in"] == 2
        gate.set()
        assert blocker.result(timeout=10) is True
        assert [f.result(timeout=10) for f in queued] == ["w"] * 4
    finally:
        donor.stop()
        thief.stop()


# ── Workspace ────────────────────────────────────────────────────────────

def test_workspace_proportional_autoscale_up():
    ws = Workspace(vcpus=2, min_vcpus=1, max_vcpus=6,
                   autoscale_interval=5, auto_start=False)
    try:
        ws.autoscale_interval = 0.01
        ws._last_scale = 0.0
        # saturate: sequential pinned tasks (no queue wait) → EWMA busy → ~0.9
        names = ["vcpu0", "vcpu1"]
        for i in range(16):
            _, fut = ws.submit(lambda: time.sleep(0.15),
                               vcpu=names[i % 2])
            fut.result(timeout=10)
        assert ws._fleet_busy() > 0.7
        assert ws.autoscale_tick() == ""  # first hot sample arms the timer
        time.sleep(0.05)
        action = ws.autoscale_tick()
        assert "scaled up" in action, f"expected scale-up, got {action!r}"
        assert len(ws.vcpus()) > 2
        assert ws.scale_history(), "scale decision should be logged"
    finally:
        ws.shutdown()


def test_workspace_rebalance_steals_work():
    ws = Workspace(vcpus=2, min_vcpus=1, max_vcpus=4, auto_start=False)
    gate = threading.Event()
    try:
        v0 = ws.get("vcpu0")
        futs = [ws.submit(lambda: gate.wait(5), vcpu="vcpu0")[1]
                for _ in range(6)]
        assert wait_for(lambda: v0.queue_depth() >= 4)
        moved = ws.rebalance()
        assert moved > 0, "idle vcpu1 should steal from busy vcpu0"
        assert ws.get("vcpu1").queue_depth() > 0
        gate.set()
        for f in futs:
            f.result(timeout=10)
    finally:
        ws.shutdown()


def test_workspace_kind_lanes():
    ws = Workspace(vcpus=2, min_vcpus=1, max_vcpus=5, auto_start=False)
    try:
        counts = ws.ensure_kind("io", 2)
        assert counts.get("io") == 2
        # affinity dispatch prefers the io lane
        name, fut = ws.submit(lambda: "x", affinity="io")
        assert ws.get(name).kind == "io"
        assert fut.result(timeout=5) == "x"
        counts = ws.ensure_kind("io", 0)
        assert counts.get("io", 0) == 0
        assert len(ws.vcpus()) >= ws.min_vcpus
    finally:
        ws.shutdown()


def test_workspace_render_status_and_map():
    ws = Workspace(vcpus=2, min_vcpus=1, max_vcpus=4, auto_start=False)
    try:
        ws.scale_to(3, reason="test")
        table = ws.render_status()
        assert "vcpu0" in table and "workspace farm" in table
        assert "▲" in table  # scale history entry rendered
        futs = ws.map(lambda x: x * 3, [1, 2, 3])
        assert [f.result(timeout=5) for f in futs] == [3, 6, 9]
        line = ws.summary_line()
        assert "queued" in line and "kinds(" in line
    finally:
        ws.shutdown()


# ── Inbox: rule engine ───────────────────────────────────────────────────

def test_rule_matches_conditions():
    from nomorals.workspace.inbox import InboxItem
    item = InboxItem(id="x", name="Invoice-2026.pdf", path="/tmp/a",
                     kind="file", mime="application/pdf", size_bytes=9000)
    assert rule_matches(item, {"ext": [".pdf", ".docx"]})
    assert rule_matches(item, {"ext": "pdf", "size_lt": 10000})
    assert rule_matches(item, {"name_re": r"invoice.*2026"})
    assert rule_matches(item, {"mime_contains": "pdf", "kind": "file"})
    assert not rule_matches(item, {"ext": ".png"})
    assert not rule_matches(item, {"size_gt": 100000})
    assert not rule_matches(item, {"bogus_key": 1})


def test_inbox_rule_priority_and_crud(tmp_path):
    ib = make_inbox(str(tmp_path / "ib1"))
    r1 = ib.add_rule("pdfs", {"ext": ".pdf"}, "summarize", priority=1)
    r2 = ib.add_rule("docs by name", {"name_contains": "doc"}, "file",
                     priority=5)
    assert isinstance(r1, InboxRule)
    assert [r.name for r in ib.list_rules()] == ["docs by name", "pdfs"]
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF fake")
    item = ib.intake(p)
    assert ib.classify(item) == "file"  # higher-priority rule wins
    assert ib.set_rule_enabled(r2.id, False)
    assert ib.classify(item) == "summarize"
    assert ib.remove_rule(r1.id) and ib.remove_rule(r2.id)
    assert ib.list_rules() == []


def test_inbox_duplicate_detection(tmp_path):
    ib = make_inbox(str(tmp_path / "ib2"))
    p1 = tmp_path / "a.txt"
    p1.write_text("same content here")
    first = ib.intake(p1)
    assert first.status == "pending"
    p2 = tmp_path / "b.txt"
    p2.write_text("same content here")
    dup = ib.intake(p2)
    assert dup.status == "done"
    assert "duplicate of" in dup.result_summary
    assert first.id in dup.result_summary
    assert not p2.exists(), "dropped duplicate copy is discarded"


def test_inbox_confidence_gating(tmp_path):
    ib = make_inbox(str(tmp_path / "ib3"),
                    classifier=lambda i, it: ("summarize", 0.1))
    p = tmp_path / "n.txt"
    p.write_text("hello")
    item = ib.intake(p)
    assert ib.classify(item) == "needs_input"
    assert "low confidence" in item.error

    ib2 = make_inbox(str(tmp_path / "ib4"),
                     classifier=lambda i, it: ("summarize", 0.99))
    p2 = tmp_path / "m.txt"
    p2.write_text("hello")
    assert ib2.classify(ib2.intake(p2)) == "summarize"


def test_inbox_priority_sweep_order_and_batch(tmp_path):
    ib = make_inbox(str(tmp_path / "ib5"))
    order = []

    def rec(inbox, item):
        order.append(item.name)
        return ActionResult(summary="ok", disposition="processed")

    ib.handlers["summarize"] = rec
    pa = tmp_path / "aaa.txt"
    pa.write_text("first in, low priority")
    pb = tmp_path / "zzz.txt"
    pb.write_text("second in, high priority")
    ia, iz = ib.intake(pa), ib.intake(pb)
    ib.set_priority(iz.id, 10)
    report = ib.sweep()
    assert report["swept"] and order == ["zzz.txt", "aaa.txt"]

    # batch retry
    bad = ib._fail(ia, "summarize", "boom")
    assert bad.status == "failed"
    assert ib.retry_all("failed") == 1
    assert ib.get_item(ia.id).status == "pending"


def test_inbox_search_annotate_digest_render(tmp_path):
    ib = make_inbox(str(tmp_path / "ib6"))

    def rec(inbox, item):
        return ActionResult(summary="summarized fine", notify="done",
                            disposition="processed")

    ib.handlers["summarize"] = rec
    p = tmp_path / "notes.txt"
    p.write_text("some notes")
    item = ib.intake(p)
    ib.annotate(item.id, "owner: this one matters")
    assert "matters" in ib.get_item(item.id).owner_note
    assert ib.search("matters")[0].id == item.id
    ib.sweep()
    got = ib.get_item(item.id)
    assert got.status == "done"
    card = ib.render_item(got)
    assert "notes.txt" in card and "[done]" in card and "matters" in card
    digest = ib.digest(hours=1)
    assert "inbox digest" in digest and "notes.txt" in digest


# ── Rooms ────────────────────────────────────────────────────────────────

def make_manager(tmp_path):
    db = Database(tmp_path / "rooms.db")
    return RoomManager(str(tmp_path / "wsroot"), db=db)


def test_room_template_create(tmp_path):
    m = make_manager(tmp_path)
    assert "build" in ROOM_TEMPLATES
    r = m.create("Ship v2", template="build", tags=["Ship", " v2 "])
    assert r.kind == "project"
    assert r.current_step == "scope"
    assert r.tags == ["ship", "v2"]
    plan = (m.rooms_dir / r.slug / "plan.md").read_text()
    assert "Scope + acceptance criteria" in plan
    # explicit args beat the template
    r2 = m.create("Ad hoc", template="build", kind="ad_hoc")
    assert r2.kind == "ad_hoc"


def test_room_brief_and_card(tmp_path):
    m = make_manager(tmp_path)
    r = m.create("Deep work", template="research")
    with m.enter(r.slug) as ctx:
        ctx.set_step("gather sources")
        ctx.decide("use sqlite", "simpler")
        ctx.log("gather", "found 3 papers")
    brief = m.brief(r.slug)
    assert "Deep work" in brief and "gather sources" in brief
    assert "use sqlite" in brief and "Next:" in brief
    card = m.render_card(m.get(r.slug))
    assert "Deep work" in card and r.slug in card and "gather sources" in card


def test_room_index_review_fork(tmp_path):
    m = make_manager(tmp_path)
    r1 = m.create("Alpha", template="goal", tags=["q4"])
    r2 = m.create("Beta")
    with m.enter(r1.slug) as ctx:
        (ctx.path("files", "spec.md")).write_text("spec")
        ctx.decide("go left", "reasons")
    idx = m.index()
    text = idx.read_text()
    assert idx.name == "INDEX.md" and r1.slug in text and r2.slug in text
    review = m.review(days=7)
    assert "Active" in review and "Alpha" in review
    forked = m.fork(r1.slug, "Alpha risky")
    assert forked.slug != r1.slug
    assert forked.state.get("forked_from") == r1.slug
    assert (m.rooms_dir / forked.slug / "files" / "spec.md").exists()
    assert forked.tags == ["q4"]


def test_room_tag_search(tmp_path):
    m = make_manager(tmp_path)
    r = m.create("Tagged room")
    m.tag(r.slug, "Urgent", "home")
    assert m.get(r.slug).tags == ["home", "urgent"]
    assert any(h["slug"] == r.slug for h in m.search("urgent"))
    m.untag(r.slug, "home")
    assert m.get(r.slug).tags == ["urgent"]
