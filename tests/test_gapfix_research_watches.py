"""Gapfix R1-12 — the out-of-the-box research watches never run.

Audit gap 12: ``research.scheduler.default_jobs()`` (money-watch +
builder-watch, 24h each) had zero production callers; the
``ResearchScheduler`` itself is never started anywhere. The
``ResearchOrgan`` IS live (idle-coordinator tick + delivery via
``pipeline.deliver`` to the owner's DMs) — only the two default
watches were never installed on it.

Pins: boot (``compose_agency``) installs both defaults as organ
watches, idempotently; an owner-removed default is NOT silently
re-installed; a due default fires through the organ tick and
delivers through the existing delivery path; the stored
max_results/fetch_top are honoured (builder-watch fetches top 1,
not the profile default 2).
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.research import autonomy as ra
from nomorals.research.pipeline import Assessment, ResearchFinding
from nomorals.research.scheduler import default_jobs
from nomorals.storage.db import Database


def _db(path):
    db = Database(str(path))
    db.migrate()
    return db


def _organ(tmp_path, name="organ.db"):
    db = _db(tmp_path / name)
    ctx = SimpleNamespace(db=db, tools=None, memory=None,
                          gateway=None, router=None)
    return ra.ResearchOrgan(ctx), db


def _by_topic(organ):
    return {w["topic"]: w for w in organ.list_watches()}


def test_install_adds_both_default_watches(tmp_path):
    organ, _db = _organ(tmp_path)
    result = organ.install_default_watches()
    assert set(result["installed"]) == {
        w_id for w_id in _expected_ids()}
    watches = _by_topic(organ)
    assert len(watches) == 2
    money = watches["AI-training gigs and expert platforms hiring Nigerians"]
    builder = watches["Tools and releases relevant to the owner's builds"]
    assert money["enabled"] is True
    assert money["cadence_hours"] == 24.0
    assert money["queries"] == [
        "AI training jobs hiring Nigeria 2026",
        "Outlier AI expert roles coding",
        "Mindrift hiring AI trainer",
    ]
    assert money["goals"] == ["money"]
    assert money["max_results"] == 6
    assert money["fetch_top"] == 2
    assert builder["enabled"] is True
    assert builder["cadence_hours"] == 24.0
    assert builder["goals"] == ["devon", "virtual_cam"]
    assert builder["max_results"] == 6
    assert builder["fetch_top"] == 1


def _expected_ids():
    from nomorals.research.autonomy import watch_id_for
    return [watch_id_for(j.topic, j.queries) for j in default_jobs()]


def test_install_is_idempotent(tmp_path):
    organ, _db = _organ(tmp_path)
    first = organ.install_default_watches()
    second = organ.install_default_watches()
    assert len(first["installed"]) == 2
    assert second["installed"] == []
    assert set(second["already_present"]) == set(first["installed"])
    assert len(organ.list_watches()) == 2


def test_owner_removed_default_is_not_reinstalled(tmp_path):
    organ, _db = _organ(tmp_path)
    organ.install_default_watches()
    money_id = _expected_ids()[0]
    assert organ.remove_watch(money_id) is True
    result = organ.install_default_watches()
    assert result["installed"] == []
    watches = {w["id"]: w for w in organ.list_watches()}
    assert watches[money_id]["enabled"] is False


def test_watch_id_is_stable(tmp_path):
    organ, _db = _organ(tmp_path)
    first = organ.add_watch("Some topic", ["q one", "q two"])
    second = organ.add_watch("Some topic", ["q one", "q two"])
    assert first == second
    assert first == ra.watch_id_for("Some topic", ["q one", "q two"])


def test_builder_watch_keeps_fetch_top_one(tmp_path):
    organ, _db = _organ(tmp_path)
    organ.install_default_watches()
    watches = _by_topic(organ)
    builder = watches["Tools and releases relevant to the owner's builds"]
    job = organ._job_for_watch(builder)
    assert job.fetch_top == 1
    assert job.max_results == 6
    money = watches["AI-training gigs and expert platforms hiring Nigerians"]
    assert organ._job_for_watch(money).fetch_top == 2


def test_due_default_watch_fires_through_tick_and_delivers(tmp_path):
    organ, db = _organ(tmp_path)
    organ.install_default_watches()
    delivered = []

    def _fake_run(job, rctx, **kw):
        return [ResearchFinding(
            job_id=job.id, title="Gig: AI trainer",
            url="https://example.com/gig-1",
            snippet="pays well")]

    def _fake_assess(finding, rctx):
        return Assessment(worth=True, score=0.9, reasons=["money"])

    def _fake_deliver(finding, assessment, rctx):
        delivered.append(finding.job_id)
        return ["telegram:owner"]

    with mock.patch("nomorals.research.pipeline.run_job",
                    side_effect=_fake_run), \
            mock.patch("nomorals.research.pipeline.assess_worth",
                       side_effect=_fake_assess), \
            mock.patch("nomorals.research.pipeline.deliver",
                       side_effect=_fake_deliver):
        report = organ.tick()
    assert report.watches_run == 2
    assert report.delivered == 2
    assert sorted(delivered) == sorted(_expected_ids())
    # Runs are recorded: a second immediate tick must not hot-loop.
    with mock.patch("nomorals.research.pipeline.run_job",
                    side_effect=_fake_run), \
            mock.patch("nomorals.research.pipeline.assess_worth",
                       side_effect=_fake_assess), \
            mock.patch("nomorals.research.pipeline.deliver",
                       side_effect=_fake_deliver):
        second = organ.tick()
    assert second.watches_run == 0
    assert len(delivered) == 2


class _FakeGateway:
    def __init__(self):
        self.sent = []

    def send(self, platform, contact, body):
        self.sent.append((platform, contact, body))
        return SimpleNamespace(ok=True)


class _FakeTools:
    def __init__(self):
        self.calls = []

    def call(self, name, **kw):
        self.calls.append((name, kw))
        return SimpleNamespace(ok=True, error=None,
                               unwrap=lambda: {"results": []})


def _boot_ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    settings = SimpleNamespace(
        workspace_dir=str(ws), resolve=lambda rel: ws / rel)
    return SimpleNamespace(
        settings=settings,
        extras={"gateway": _FakeGateway()},
        tools=_FakeTools(),
        db=_db(ws / "db.sqlite"))


def test_compose_agency_installs_defaults_at_boot(tmp_path, monkeypatch):
    from nomorals.agency.wiring import compose_agency

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ctx = _boot_ctx(tmp_path)
    comp = compose_agency(ctx)
    assert comp["research_organ"] is not None
    organ = ra.ResearchOrgan(ctx)
    watches = _by_topic(organ)
    assert set(watches) == {
        "AI-training gigs and expert platforms hiring Nigerians",
        "Tools and releases relevant to the owner's builds",
    }
    # A second boot (second composition) must not duplicate them.
    comp2 = compose_agency(ctx)
    assert comp2["research_organ"] is not None
    assert len(ra.ResearchOrgan(ctx).list_watches()) == 2
