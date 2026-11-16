"""Pins for the §3.5 follow-up feature sketches (tails 66–91
observability family), recovered from the day logs + per-tail
SURVEY.md shelf sections and landed in one pass.

Each test pins the LANDED behavior: the failure the sketch named
is now recorded (log line / ledger entry / status key / new
metrics series / stderr note) while the return contract stays
byte-identical to the exclusion the tails pinned.  Every test
fails on the pre-change code (the record is absent there).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest


def _msgs(caplog, needle):
    return [r for r in caplog.records if needle in r.getMessage()]


# ── S1 tail-66: acp/mcp principal-resolver visibility ──────────

def _boom_resolver():
    raise RuntimeError("resolver down")


@pytest.mark.parametrize("mod_name", [
    "nomorals.api.acp", "nomorals.api.mcp_server"])
def test_s1_resolver_failure_warns_once_per_identity(
        mod_name, caplog, monkeypatch):
    import importlib
    mod = importlib.import_module(mod_name)
    monkeypatch.setattr(mod, "_RESOLVER_FAILURE_WARNED", set())
    cls = mod.ACPServer if mod_name.endswith("acp") else mod.MCPServer
    srv = cls.__new__(cls)
    srv._principal_tls = SimpleNamespace()
    srv._principal_resolver = _boom_resolver
    srv._principal = "configured-principal"
    with caplog.at_level(logging.DEBUG):
        assert srv.principal == "configured-principal"
        assert srv.principal == "configured-principal"
    warns = _msgs(caplog, "principal resolver failed")
    assert len(warns) == 1  # once per resolver identity, not per call
    assert warns[0].levelno == logging.WARNING


# ── S2 tail-71: events 181 predicate failure is logged ─────────

def test_s2_broken_predicate_logs_debug(caplog):
    from nomorals.core.events import Subscription

    def bad(_event):
        raise ValueError("predicate boom")

    sub = Subscription(pattern="x", handler=lambda e: None, predicate=bad)
    with caplog.at_level(logging.DEBUG):
        assert sub.accepts(SimpleNamespace(topic="x")) is False
    assert _msgs(caplog, "subscription predicate failed")


# ── S3 tail-73: drift 584/598 debug lines ──────────────────────

def test_s3_drift_failures_log_debug(caplog, monkeypatch):
    import time as _time

    import nomorals.health.drift as drift

    mon = drift.DriftMonitor.__new__(drift.DriftMonitor)

    class BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db gone")

        def query_one(self, *a, **k):
            raise RuntimeError("db gone")

    mon._db = BoomDB()
    with caplog.at_level(logging.DEBUG):
        assert mon.consecutive_act_days() == 0
    assert _msgs(caplog, "consecutive_act_days failed")

    # escalation_note's own guard: the streak query succeeds (3 act
    # days) and the coaching guard downstream raises.
    today = _time.time()
    rows = [{"day": _time.strftime(
        "%Y-%m-%d", _time.localtime(today - i * 86400)),
        "severity": "act"} for i in range(3)]

    class RowDB:
        def query(self, *a, **k):
            return rows

    mon._db = RowDB()
    monkeypatch.setattr(
        drift, "guard_coaching",
        lambda text: (_ for _ in ()).throw(RuntimeError("guard boom")))
    with caplog.at_level(logging.DEBUG):
        assert mon.escalation_note() == ""
    assert _msgs(caplog, "escalation_note failed")


# ── S4 tail-74: twin 818 mqtt fallback debug line ──────────────

def test_s4_twin_mqtt_import_fallback_logs_debug(caplog, monkeypatch):
    import types as _types

    import nomorals.integrations.digital_twin as twin_mod

    broken = _types.ModuleType("nomorals.integrations.mqtt_client")

    def _raise(name):
        raise RuntimeError("mqtt lib broken")

    broken.__getattr__ = _raise
    monkeypatch.setitem(
        sys.modules, "nomorals.integrations.mqtt_client", broken)
    seen = {}
    bridge = SimpleNamespace(
        subscribe=lambda topic, cb: seen.setdefault("cb", cb))
    twin = SimpleNamespace(ingest=lambda *a, **k: None)
    with caplog.at_level(logging.DEBUG):
        twin_mod.attach_twin_to_mqtt(bridge, twin)
    assert _msgs(caplog, "twin MQTT ids use the local fallback")
    assert "cb" in seen  # the fallback still wires the handler


# ── S5 tail-74: hub scrape-errors counter series ───────────────

def test_s5_hub_scrape_errors_counter(tmp_path, monkeypatch):
    from nomorals.hub.server import HubServer
    from nomorals.storage.db import Database

    db = Database(str(tmp_path / "hub.db"))
    db.migrate()
    srv = HubServer(db)
    monkeypatch.setattr(
        srv.transport, "stats",
        lambda: (_ for _ in ()).throw(RuntimeError("transport down")))
    text = srv._metrics_text().body.decode()
    assert 'nomorals_hub_scrape_errors_total{source="mesh_tasks"} 1' \
        in text
    text2 = srv._metrics_text().body.decode()
    assert 'nomorals_hub_scrape_errors_total{source="mesh_tasks"} 2' \
        in text2


# ── S6 tail-75: payment fiat observability + unknown marker ────

def test_s6_fiat_failure_logs_and_marks_unknown(caplog, monkeypatch):
    import nomorals.integrations.market_data as md
    import nomorals.integrations.payment_integration as pay

    monkeypatch.setattr(
        md, "quote",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("feed down")))
    inst = pay.PaymentIntegration.__new__(pay.PaymentIntegration)
    with caplog.at_level(logging.DEBUG):
        value, known = asyncio.run(inst._fiat_quote("BTC", 2.0, "USD"))
        assert asyncio.run(inst.fiat_value("BTC", 2.0, "USD")) == 0.0
    assert value == 0.0 and known is False
    assert _msgs(caplog, "fiat_value feed failed")

    async def _bal(wallet="main", currency="BTC"):
        return 1.5

    async def _addr(currency, wallet="main"):
        return "bc1qexampleaddress0123456789"

    inst.get_balance = _bal
    inst.receive_crypto = _addr
    card = asyncio.run(inst.format_wallet())
    assert "price unavailable" in card
    assert "≈ 0.00" not in card


# ── S7 tail-76: style kv fallback debug sink ───────────────────

def test_s7_style_kv_failure_logs_debug(caplog):
    from nomorals.media import style

    class BadStr:
        def __str__(self):
            raise RuntimeError("no str for you")

    with caplog.at_level(logging.DEBUG):
        assert style.kv([("k", BadStr())]) == ""
    assert _msgs(caplog, "kv render failed")


# ── S8 tail-77: ltx native attempt names its cause ─────────────

def test_s8_ltx_native_failure_names_cause(caplog, monkeypatch):
    import types as _types

    import nomorals.media.videogen.ltx_backend as ltx

    fake_diffusers = _types.ModuleType("diffusers")

    class _BoomPipe:
        @staticmethod
        def from_pretrained(*a, **k):
            raise RuntimeError("lora file corrupt")

    class _CondPipe:
        @staticmethod
        def from_pretrained(*a, **k):
            pipe = SimpleNamespace(
                enable_model_cpu_offload=lambda: None,
                to=lambda dev: None)
            return pipe

    fake_diffusers.LTXImageToVideoPipeline = _BoomPipe
    fake_diffusers.LTXConditionPipeline = _CondPipe
    fake_torch = _types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    cond_mod = _types.ModuleType(
        "diffusers.pipelines.ltx.pipeline_ltx_condition")
    cond_mod.LTXVideoCondition = object
    utils_mod = _types.ModuleType("diffusers.utils")
    utils_mod.export_to_video = lambda *a, **k: None
    utils_mod.load_image = lambda *a, **k: None
    utils_mod.load_video = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.pipelines.ltx.pipeline_ltx_condition", cond_mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils_mod)
    monkeypatch.setattr(ltx, "diffusers_available", lambda: True)
    backend = ltx.LTXBackend.__new__(ltx.LTXBackend)
    backend._i2v_pipe = None
    backend.model_id = "fake-model"
    backend._apply_loras = lambda pipe: pipe
    with caplog.at_level(logging.DEBUG):
        result = backend._load_i2v()
    assert result[1] == "condition"
    hits = _msgs(caplog, "native I2V attempt failed")
    assert hits and "lora file corrupt" in hits[0].getMessage()


# ── S9 tail-78: backup inventory debug lines ───────────────────

def test_s9_backup_inventory_failures_log_debug(
        tmp_path, caplog, monkeypatch):
    import nomorals.memory.backup as backup
    import nomorals.memory.types as mem_types

    class BadSettings:
        @property
        def home_path(self):
            raise RuntimeError("settings gone")

    manager = SimpleNamespace(
        context=SimpleNamespace(settings=BadSettings()), db=None)
    with caplog.at_level(logging.DEBUG):
        assert backup._settings_home(manager) is None
    assert _msgs(caplog, "settings home resolution failed")

    monkeypatch.setattr(
        mem_types, "entity_home",
        lambda s: (_ for _ in ()).throw(RuntimeError("no entities")))
    good = SimpleNamespace(
        context=SimpleNamespace(
            settings=SimpleNamespace(home_path=tmp_path)),
        db=None)
    with caplog.at_level(logging.DEBUG):
        files = backup._memory_files(good)
    assert isinstance(files, dict)
    assert _msgs(caplog, "entities inventory failed")


# ── S10 tail-79: mission enumeration debug lines ───────────────

def test_s10_mission_enumeration_failures_log_debug(caplog):
    from nomorals.missions.mission import MissionStore

    store = MissionStore.__new__(MissionStore)
    store.get = lambda mission_id: SimpleNamespace(id=mission_id)

    class BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db gone")

    store.db = BoomDB()
    with caplog.at_level(logging.DEBUG):
        assert store.children("m1") == []
    assert _msgs(caplog, "children query failed")

    class RowDB:
        def query(self, *a, **k):
            return [{"bogus": "row"}]

    store.db = RowDB()
    with caplog.at_level(logging.DEBUG):
        assert store.children("m1") == []
        assert store.list() == []
    assert _msgs(caplog, "skipping unreadable row in children()")
    assert _msgs(caplog, "skipping unreadable row in list()")


# ── S11 tail-79: repetition remove() debug line ────────────────

def test_s11_repetition_remove_failure_logs_debug(caplog):
    from nomorals.memory.repetition import RepetitionScheduler

    store = RepetitionScheduler.__new__(RepetitionScheduler)

    class BoomDB:
        def execute(self, *a, **k):
            raise RuntimeError("db gone")

        def commit(self):
            raise RuntimeError("db gone")

    store._db = BoomDB()
    store._broken = lambda: False
    with caplog.at_level(logging.DEBUG):
        assert store.remove("mem-1") is False
    assert _msgs(caplog, "repetition remove failed")


# ── S12 tail-80: kernel stop() debug lines ─────────────────────

def test_s12_kernel_stop_failures_log_debug(caplog):
    import threading

    from nomorals.os.kernel import OSKernel

    def boom(*a, **k):
        raise RuntimeError("teardown boom")

    kern = OSKernel.__new__(OSKernel)
    kern._lock = threading.Lock()
    kern._started = True
    kern._ready = True
    kern._shutdown_hooks = [("bad-hook", boom)]
    kern.services = SimpleNamespace(stop_all=boom)
    kern.health = SimpleNamespace(stop_background=boom)
    kern.bus = SimpleNamespace(stop=boom)
    with caplog.at_level(logging.DEBUG):
        kern.stop(timeout_s=1.0)
    assert _msgs(caplog, "shutdown hook bad-hook failed")
    assert _msgs(caplog, "services stop_all failed")
    assert _msgs(caplog, "health stop_background failed")
    assert _msgs(caplog, "bus stop failed")


# ── S13 tail-80: chat_profile get_profile debug line ───────────

def test_s13_chat_profile_read_failure_logs_debug(caplog):
    from nomorals.partner.chat_profile import get_profile

    class BoomDB:
        def query_one(self, *a, **k):
            raise RuntimeError("db gone")

        def execute(self, *a, **k):
            raise RuntimeError("db gone")

    with caplog.at_level(logging.DEBUG):
        assert get_profile(BoomDB(), "chat-1") == {}
    assert _msgs(caplog, "chat profile read failed")


# ── S14 tail-81: research gap_stats debug line ─────────────────

def test_s14_gap_stats_failure_logs_debug(caplog):
    import nomorals.research.autonomy as ra

    organ = ra.ResearchOrgan.__new__(ra.ResearchOrgan)

    class BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db gone")

    organ.db = BoomDB()
    organ.context = SimpleNamespace()
    with caplog.at_level(logging.DEBUG):
        assert organ.gap_stats() == {
            "open": 0, "resolved": 0, "stalled": 0, "top_open": []}
    assert _msgs(caplog, "gap_stats failed")


# ── S15 tail-82: captcha ceiling unreadable-store warning ──────

def test_s15_captcha_ceiling_corrupt_store_warns(tmp_path, caplog):
    from nomorals.tools.captcha_flow import OriginCostCeiling

    store = tmp_path / "ceiling.json"
    store.write_text("{not json", encoding="utf-8")
    ceiling = OriginCostCeiling(store)
    with caplog.at_level(logging.DEBUG):
        data = ceiling._load()
    assert data["origins"] == {}
    warns = _msgs(caplog, "captcha ceiling store unreadable")
    assert warns and warns[0].levelno == logging.WARNING

    missing = OriginCostCeiling(tmp_path / "absent.json")
    with caplog.at_level(logging.DEBUG):
        caplog.clear()
        assert missing._load()["origins"] == {}
    assert not _msgs(caplog, "captcha ceiling store unreadable")


# ── S16 tail-83: context per-model lifecycle skip warning ──────

def test_s16_lifecycle_model_skip_warns(tmp_path, caplog, monkeypatch):
    import nomorals.llm.lifecycle as lifecycle
    from nomorals.agents.context import build_router
    from nomorals.core.config import Settings
    from nomorals.core.events import EventBus

    class FakeLifecycle:
        def __init__(self, db):
            self.primary = None

        def list(self):
            return [SimpleNamespace(
                id="m-skip", provider="other", status="ok")]

    monkeypatch.setattr(lifecycle, "ModelLifecycle", FakeLifecycle)
    monkeypatch.setattr(
        lifecycle, "card_for_model",
        lambda model: (_ for _ in ()).throw(RuntimeError("bad card")))
    from nomorals.storage.db import Database
    db = Database(str(tmp_path / "ctx.db"))
    db.migrate()
    settings = Settings(home=tmp_path / "home")
    with caplog.at_level(logging.DEBUG):
        build_router(settings, EventBus(), db=db)
    assert _msgs(caplog, "lifecycle model m-skip skipped")


# ── S17 tail-84: motivation status error keys ──────────────────

def test_s17_motivation_status_surfaces_errors(monkeypatch):
    import nomorals.agents.motivation as motivation
    import nomorals.agents.scheduler as scheduler_mod
    import nomorals.agents.teachings as teachings_mod

    class BoomScheduler:
        def __init__(self, context):
            pass

        def list_jobs(self):
            raise RuntimeError("scheduler gone")

    class BoomLedger:
        def __init__(self, context):
            pass

        def stats(self, now=None):
            raise RuntimeError("ledger gone")

    monkeypatch.setattr(scheduler_mod, "Scheduler", BoomScheduler)
    monkeypatch.setattr(teachings_mod, "TeachingsLedger", BoomLedger)
    out = motivation.motivation_status(SimpleNamespace())
    assert out["scheduler_error"].startswith("RuntimeError")
    assert out["ledger_error"].startswith("RuntimeError")
    assert out["windows"]  # the report still renders


# ── S18 tail-85: flow gateway failure hits the ledger ──────────

def test_s18_flow_gateway_failure_recorded():
    from nomorals.agents.trial import failures
    from nomorals.agents.trial.flow import active_delivery_platforms

    failures.clear_failures()

    class BoomGateway:
        def status(self):
            raise RuntimeError("gateway down")

    assert active_delivery_platforms(BoomGateway()) == []
    sites = [f.site for f in failures.recent_failures()]
    assert "flow.active_delivery_platforms" in sites


# ── S20 tail-86: reader corrupt-story warning ──────────────────

def test_s20_reader_corrupt_story_warns(tmp_path, caplog):
    from nomorals.books.reader import ReaderError, StoryReader

    reader = StoryReader.__new__(StoryReader)
    reader.reader_dir = lambda: tmp_path
    bad = tmp_path / "broken-story"
    bad.mkdir()
    (bad / "story.json").write_text("{corrupt", encoding="utf-8")
    with caplog.at_level(logging.DEBUG), pytest.raises(ReaderError):
        reader.resolve_slug("anything")
    hits = _msgs(caplog, "skipping corrupt story at broken-story")
    assert hits and hits[0].levelno == logging.WARNING


# ── S21 tail-87: challenge detector debug lines ────────────────

def test_s21_challenge_detector_failures_log_debug(caplog, monkeypatch):
    import nomorals.browser.challenge as challenge
    import nomorals.browser.question_solver as question
    import nomorals.browser.slider_solver as slider

    def boom(*a, **k):
        raise RuntimeError("detector boom")

    monkeypatch.setattr(slider, "detect_slider_challenge", boom)
    monkeypatch.setattr(question, "detect_question_challenge", boom)
    with caplog.at_level(logging.DEBUG):
        signal = challenge.detect_challenge_signal(
            "<html></html>", "https://example.test/")
    assert signal is None
    assert _msgs(caplog, "slider challenge detection failed")
    assert _msgs(caplog, "question challenge detection failed")


# ── S22 tail-87: browse restore failure notes stderr ───────────

def test_s22_browse_restore_failure_notes_stderr(capsys, monkeypatch):
    import nomorals.browser as browser_pkg
    import nomorals.cmdline.commands.browse as browse_cmd

    class FakeService:
        def __init__(self, data_dir=None):
            pass

        def restore(self):
            raise RuntimeError("state file corrupt")

    monkeypatch.setattr(browser_pkg, "BrowserService", FakeService)
    browse_cmd._service(SimpleNamespace(), SimpleNamespace(db=None))
    err = capsys.readouterr().err
    assert "could not be restored" in err
    assert "state file corrupt" in err


# ── S23 tail-88: matchmaking sweep read debug line ─────────────

def test_s23_matchmaking_sweep_read_failure_logs_debug(caplog):
    from nomorals.games.matchmaking import Matchmaker

    mm = Matchmaker.__new__(Matchmaker)

    class BoomDB:
        def execute(self, *a, **k):
            return None

        def query(self, *a, **k):
            raise RuntimeError("queue gone")

    mm.db = BoomDB()
    with caplog.at_level(logging.DEBUG):
        assert mm.sweep() == 0
    assert _msgs(caplog, "match sweep read failed")


# ── S24 tail-89: smarthome unsubscribe send debug line ─────────

def test_s24_smarthome_unsubscribe_send_failure_logs_debug(
        caplog, monkeypatch):
    import nomorals.integrations.smarthome_integration as smart

    cls = None
    for name in dir(smart):
        obj = getattr(smart, name)
        if isinstance(obj, type) and hasattr(obj, "unsubscribe_trigger"):
            cls = obj
            break
    assert cls is not None
    inst = cls.__new__(cls)
    inst._trigger_subs = {7: SimpleNamespace(msg_id=7)}
    inst._writer = SimpleNamespace(is_closing=lambda: False)

    def boom(*a, **k):
        raise RuntimeError("socket gone")

    monkeypatch.setattr(smart, "_ws_send_text", boom)
    with caplog.at_level(logging.DEBUG):
        assert inst.unsubscribe_trigger(7) is True
    assert _msgs(caplog, "trigger unsubscribe send failed")


# ── S25 tail-90: learning atexit debug lines ───────────────────

def test_s25_learning_atexit_failures_log_debug(caplog):
    import nomorals.llm.learning as learning

    class FakeAttachment:
        def __init__(self):
            self._worker = None

        @property
        def _stop(self):
            raise RuntimeError("stop boom")

    fake = FakeAttachment()
    learning._live_attachments.add(fake)
    try:
        with caplog.at_level(logging.DEBUG):
            learning._stop_all_learning_workers()
    finally:
        learning._live_attachments.discard(fake)
    assert _msgs(caplog, "atexit stop signal failed")


# ── S26 tail-90: tutor syllabus debug lines ────────────────────

def test_s26_tutor_syllabus_failures_log_debug(caplog, monkeypatch):
    import nomorals.learn.curriculum as curriculum
    import nomorals.learn.tutor as tutor_mod

    def boom(*a, **k):
        raise RuntimeError("curriculum boom")

    session = tutor_mod.TutorSession.__new__(tutor_mod.TutorSession)
    session.mastery = SimpleNamespace(snapshot=lambda: {})
    session.mode = "direct"
    session.citations = []
    session._teach_direct = lambda: "teaching"

    # Guard 1: a syllabus string whose find_topic raises stays a
    # string, and the failure is on the record.
    monkeypatch.setattr(curriculum, "find_topic", boom)
    session.syllabus = "some topic"
    with caplog.at_level(logging.DEBUG):
        turn = session.start()
    assert turn.prompt.startswith("teaching")
    assert _msgs(caplog, "syllabus resolution failed")

    # Guard 2: a resolved (non-str) syllabus whose code render
    # raises loses the code line, and the failure is on the record.
    monkeypatch.setattr(
        curriculum, "find_topic", lambda s: SimpleNamespace(title="t"))
    monkeypatch.setattr(curriculum, "syllabus_code", boom)
    session.syllabus = "some topic"
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        turn = session.start()
    assert turn.prompt.startswith("teaching")
    assert _msgs(caplog, "syllabus code render failed")


# ── S27 tail-91: library extraction debug lines ────────────────

def test_s27_library_extraction_failures_log_debug(
        tmp_path, caplog, monkeypatch):
    import nomorals.media.highlight_model.features as features
    import nomorals.media.library as library
    import nomorals.media.scene_intel.segment as segment

    def boom(*a, **k):
        raise RuntimeError("extractor boom")

    monkeypatch.setattr(segment, "segment_movie", boom)
    monkeypatch.setattr(features, "extract_features", boom)
    index = library.ClipIndex()
    path = str(tmp_path / "clip.mp4")
    with caplog.at_level(logging.DEBUG):
        entries = index.index_file(path, transcript=[])
    assert entries and entries[0]["start"] == 0.0
    assert _msgs(caplog, "segmentation failed")
    assert _msgs(caplog, "feature extraction failed")
