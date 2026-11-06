"""Infra/autonomy organs pass (AA-03/09/11/14/15/28/34/35).

Each test drives the PRODUCTION path for one dormant organ — the real
caller/consumer, not the organ in isolation:

- AA-03: ``nm doctor --live`` runs core.verify.LiveVerifier.
- AA-09: the AssemblyAI / OpenAI Whisper API engines live inside
  voice.stt.UniversalSTT (ported from the superseded integrations.stt).
- AA-11: the partner /routine + /home surfaces and the deals tool
  construct integrations through integrations.expose.
- AA-14: DELIBERATE PRESERVED-BEHAVIOR PIN — the exporter was already
  wired (tools registry fallback reaches nomorals.exporter.register);
  this pins the full registry → tool → dataset path so it cannot
  silently unwire. It passes on the pristine tree by design.
- AA-15: PRESERVED-BEHAVIOR PIN — reflections rows decode through
  storage.models.Reflection from both production readers
  (MissionStore.reflections and the training collector) with the same
  results the hand-rolled decoders produced.
- AA-28: OS projects are served by the API through SessionBridge's
  ProjectStore (create → list → summary with session counts).
- AA-34: LLMRouter consent-checks every provider attempt when a
  ConsentLedger is attached; build_router arms enforcement from the
  ledger file; ``nm consent`` records the grants.
- AA-35: autonomy.wiring assembles the REAL satellites around
  UnifiedScheduler; ``nm why`` reads and runs through that stack.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from types import SimpleNamespace

import pytest

# ── AA-03: doctor --live ─────────────────────────────────────────────


def _doctor_settings(tmp_path):
    return SimpleNamespace(
        db_path=str(tmp_path / "doc.db"), profile="test",
        concurrency=SimpleNamespace(processes=None, use_processes=False))


class _StubVerifier:
    """Stands in for LiveVerifier (its environment checks vary by
    machine); the doctor plumbing around it is the real code."""

    result = None

    def __init__(self, config=None):
        self.config = config

    async def verify_all(self):
        return type(self).result


def _run_doctor(tmp_path, capsys, monkeypatch, *, live, result=None,
                json_mode=True):
    from nomorals.cmdline.commands import doctor as doc
    from nomorals.core import verify as verify_mod
    from nomorals.core.verify import VerificationCheck, VerificationResult

    if result is None:
        result = VerificationResult(checks=[
            VerificationCheck(name="Network", passed=True,
                              message="reachable"),
        ])
    _StubVerifier.result = result
    monkeypatch.setattr(verify_mod, "LiveVerifier", _StubVerifier)
    args = argparse.Namespace(json=json_mode, build_native=False,
                              live=live)
    rc = doc._cmd_doctor(args, _doctor_settings(tmp_path))
    return rc, capsys.readouterr().out


def test_aa03_doctor_live_failure_fails(tmp_path, capsys, monkeypatch):
    from nomorals.core.verify import VerificationCheck, VerificationResult

    result = VerificationResult(checks=[
        VerificationCheck(name="Network", passed=False,
                          message="no route"),
        VerificationCheck(name="Disk", passed=True, message="fine"),
    ])
    rc, out = _run_doctor(tmp_path, capsys, monkeypatch,
                          live=True, result=result)
    assert rc == 1
    payload = json.loads(out)
    assert payload["live"]["failed"] == 1
    assert payload["live"]["all_passed"] is False
    names = {c["name"] for c in payload["live"]["checks"]}
    assert names == {"Network", "Disk"}


def test_aa03_doctor_without_live_has_no_live_section(
        tmp_path, capsys, monkeypatch):
    rc, out = _run_doctor(tmp_path, capsys, monkeypatch, live=False)
    payload = json.loads(out)
    assert "live" not in payload


def test_aa03_doctor_live_text_mode_prints_checks(
        tmp_path, capsys, monkeypatch):
    from nomorals.core.verify import VerificationCheck, VerificationResult

    result = VerificationResult(checks=[
        VerificationCheck(name="Network", passed=False,
                          message="no route"),
    ])
    rc, out = _run_doctor(tmp_path, capsys, monkeypatch,
                          live=True, result=result, json_mode=False)
    assert rc == 1
    assert "live verification:" in out
    assert "Network: no route" in out


# ── AA-09: ported cloud STT backends ─────────────────────────────────


def _fake_assemblyai():
    mod = types.ModuleType("assemblyai")
    mod.settings = SimpleNamespace(api_key="")
    mod.TranscriptStatus = SimpleNamespace(error="error")

    class _Transcriber:
        def transcribe(self, path):
            return SimpleNamespace(
                status="completed",
                text="hello from assembly",
                utterances=[SimpleNamespace(start=0, end=1500,
                                            text=" hello "),
                            SimpleNamespace(start=1500, end=2000,
                                            text="world")],
                confidence=0.92,
                audio_duration=2.0,
            )

    mod.Transcriber = _Transcriber
    return mod


def _fake_openai():
    mod = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, **kw):
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(
                create=self._create))

        def _create(self, **kw):
            assert kw["response_format"] == "verbose_json"
            return SimpleNamespace(
                text="hello from whisper api",
                language="en",
                segments=[SimpleNamespace(start=0.0, end=1.25,
                                          text=" hello ")])

    mod.OpenAI = _OpenAI
    return mod


def test_aa09_assemblyai_backend_through_universal_stt(
        tmp_path, monkeypatch):
    from nomorals.voice.stt import UniversalSTT

    monkeypatch.setitem(sys.modules, "assemblyai", _fake_assemblyai())
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    stt = UniversalSTT(backend="assemblyai")
    out = stt.transcribe(str(clip), language="en")
    assert out["backend"] == "assemblyai"
    assert out["text"] == "hello from assembly"
    assert out["segments"][0] == {"start": 0.0, "end": 1.5,
                                  "text": "hello"}
    assert out["confidence"] == pytest.approx(0.92)


def test_aa09_whisper_api_backend_through_universal_stt(
        tmp_path, monkeypatch):
    from nomorals.voice.stt import UniversalSTT

    monkeypatch.setitem(sys.modules, "openai", _fake_openai())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    stt = UniversalSTT(backend="whisper-api")
    out = stt.transcribe(str(clip), language="en")
    assert out["backend"] == "whisper-api"
    assert out["text"] == "hello from whisper api"
    assert out["segments"][0]["end"] == 1.25


def test_aa09_paid_backends_need_keys_and_rank_last(
        tmp_path, monkeypatch):
    from nomorals.voice import stt as stt_mod

    monkeypatch.setitem(sys.modules, "assemblyai", _fake_assemblyai())
    monkeypatch.setitem(sys.modules, "openai", _fake_openai())
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    with pytest.raises(RuntimeError, match="ASSEMBLYAI_API_KEY"):
        stt_mod.UniversalSTT(backend="assemblyai").transcribe(str(clip))
    available = stt_mod.available_stt_backends()
    assert available[-2:] == ["assemblyai", "whisper-api"]


# ── AA-11: expose facade consumers ───────────────────────────────────


def test_aa11_make_home_twin_default_is_the_durable_twin():
    from nomorals.integrations import digital_twin, expose

    twin = expose.make_home_twin()
    assert twin.db_path == digital_twin.DEFAULT_DB
    throwaway = expose.make_home_twin(":memory:")
    assert throwaway.db_path == ":memory:"


def test_aa11_home_surface_reads_through_facade(tmp_path):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    from nomorals.integrations.digital_twin import HomeTwin

    mixin = object.__new__(RuntimeMemoryMixin)
    mixin._home_twin = HomeTwin(str(tmp_path / "twin.db"))
    out = mixin._control_home("what changed 2")
    assert isinstance(out, str) and out


def test_aa11_routine_surface_flows_through_facade():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    from nomorals.integrations import routines

    class _Memory:
        def recall(self, query, **kw):
            if "device alias" in str(query):
                return [SimpleNamespace(
                    text="device alias: kitchen lights -> light.kitchen")]
            return []

    routines._DRAFTS.clear()
    mixin = object.__new__(RuntimeMemoryMixin)
    mixin._memory = _Memory()
    reply = mixin._control_routine(
        "every morning at 7, turn on kitchen lights")
    match = re.search(r"/routine confirm ([0-9a-f]+)", reply)
    assert match, reply
    confirmed = mixin._control_routine(f"confirm {match.group(1)}")
    assert "routine validated:" in confirmed
    assert "smart-home backend not connected" in confirmed
    routines._DRAFTS.clear()


def test_aa11_deals_engine_built_through_facade(monkeypatch):
    from nomorals.integrations.naija_shopping import NaijaShoppingEngine
    from nomorals.storage.db import Database
    from nomorals.tools import deals

    class _Browser:
        def __init__(self, name=""):
            self.name = name

    monkeypatch.setattr("nomorals.tools.browser.BrowserSession", _Browser)
    deals._engine = None
    deals._lazy_context = SimpleNamespace(db=Database(":memory:"))
    try:
        engine = deals._get_engine()
        assert isinstance(engine, NaijaShoppingEngine)
    finally:
        deals._engine = None
        deals._lazy_context = None


def test_aa11_facade_is_on_the_package_surface():
    import nomorals.integrations as integrations

    assert integrations.make_home_twin is not None
    assert integrations.make_naija_shopping is not None
    assert integrations.expose is not None


# ── AA-14: exporter production path (preserved-behavior pin) ─────────


class _FakeExportClient:
    dialogs: list = []
    messages: dict = {}
    started = False

    def __init__(self, *args, **kwargs):
        pass

    async def start(self):
        type(self).started = True

    async def disconnect(self):
        pass

    def iter_dialogs(self):
        dialogs = type(self).dialogs

        async def _gen():
            for dialog in dialogs:
                yield dialog

        return _gen()

    def iter_messages(self, entity, limit=None, reverse=False):
        messages = type(self).messages.get(entity, [])

        async def _gen():
            for msg in messages:
                yield msg

        return _gen()


def test_aa14_export_tool_through_real_registry(tmp_path, monkeypatch):
    """PIN (passes on pristine too — the wiring already existed):
    the built-in tool registry reaches exporter.register, and calling
    export_training_data through the registry produces a registered,
    PII-scrubbed dataset."""
    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings
    from nomorals.training.dataset import DatasetRegistry

    fake = types.ModuleType("telethon")
    fake.TelegramClient = _FakeExportClient
    monkeypatch.setitem(sys.modules, "telethon", fake)
    _FakeExportClient.started = False
    _FakeExportClient.dialogs = [
        SimpleNamespace(id=101, title="Alice", entity="e101"),
    ]
    _FakeExportClient.messages = {
        "e101": [
            SimpleNamespace(raw_text="hey, call me at 08034567890",
                            out=False),
            SimpleNamespace(raw_text="will call you tonight", out=True),
            SimpleNamespace(raw_text="thanks, see you", out=False),
            SimpleNamespace(raw_text="see you then", out=True),
        ],
    }
    with build_context(Settings(home=str(tmp_path)),
                       with_executor=False, with_router=False,
                       with_memory=False, with_tools=True) as context:
        context.settings.chat.telegram_api_id = "123456"
        context.settings.chat.telegram_api_hash = "abcdef0123456789"
        registry = context.tools
        assert "export_training_data" in registry.names()
        assert registry.failed_modules() == []
        outcome = registry.call("export_training_data")
        assert outcome.ok, outcome.error
        value = outcome.value
        assert value["ok"], value
        assert value["dataset_id"]
        dataset = DatasetRegistry(context.db).get(value["dataset_id"])
        assert dataset is not None
        assert dataset.rows == 1
        assert _FakeExportClient.started


# ── AA-15: typed reflection decode (preserved-behavior pin) ──────────


def test_aa15_mission_store_reflections_decode_typed():
    from nomorals.missions.mission import MissionStore
    from nomorals.storage.db import Database
    from nomorals.storage.models import Reflection

    db = Database(":memory:")
    db.migrate()
    store = MissionStore(db)
    rid = store.record_reflection(
        "m1", score=0.75, summary="went fine",
        lessons=["stay steady"], weights={"patience": 0.5})
    rows = store.reflections("m1")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["lessons"] == ["stay steady"]
    assert row["weights"] == {"patience": 0.5}
    assert row["score"] == 0.75
    # The returned dict is exactly the Reflection field set.
    assert set(row) == set(Reflection.__dataclass_fields__)


def test_aa15_training_collector_reads_reflections_typed():
    from nomorals.storage.db import Database
    from nomorals.training.collect import CollectionStats, TrainingCollector

    db = Database(":memory:")
    db.migrate()
    db.insert("reflections", {
        "id": "r1", "mission_id": "m1", "score": 0.5,
        "summary": "learned things", "lessons": json.dumps(["stay steady"]),
        "weights": "{}", "created_at": 1000.0,
    })
    collector = TrainingCollector(db)
    stats = CollectionStats()
    out = collector._reflection_examples(stats, since=0.0, limit=10)
    assert stats.reflections == 1
    assert len(out) == 1


# ── AA-28: OS projects through the API ───────────────────────────────


def _api_server(tmp_path):
    from nomorals.api.server import APIServer, Principal
    from nomorals.core.policy import CapabilitySet
    from nomorals.storage.db import Database
    from nomorals.tools.registry import ToolRegistry

    db = Database(":memory:")
    db.migrate()
    context = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir=str(tmp_path), workspace_dir=str(tmp_path)),
        tools=ToolRegistry(), bus=None, memory=None)
    api = APIServer(
        context, token="owner-token",
        principals={"owner-token": Principal(
            "owner", CapabilitySet.all())},
        max_body_bytes=64 * 1024)
    owner = api.resolve_principal("Bearer owner-token")
    return api, owner


def test_aa28_projects_api_round_trip(tmp_path):
    api, owner = _api_server(tmp_path)
    status, payload = api.dispatch(
        "POST", "/projects",
        {"name": "Trading Desk", "mission_ids": ["m1"],
         "tags": ["gold"]}, {}, principal=owner)
    assert status == 200, payload
    project = payload["project"]
    assert project["name"] == "Trading Desk"
    assert project["mission_ids"] == ["m1"]

    status, payload = api.dispatch("GET", "/projects", {}, {},
                                   principal=owner)
    assert status == 200
    assert [p["id"] for p in payload["projects"]] == [project["id"]]

    status, payload = api.dispatch(
        "GET", "/projects", {}, {"id": project["id"]}, principal=owner)
    assert status == 200
    summary = payload["project"]
    assert summary["missions"]["total"] == 1
    assert summary["sessions"] == 0

    status, payload = api.dispatch(
        "GET", "/projects", {}, {"id": "proj_missing"}, principal=owner)
    assert status == 400


def test_aa28_bridge_shares_one_db_for_sessions_and_projects(tmp_path):
    from nomorals.os.session_bridge import SessionBridge
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    bridge = SessionBridge(db=db)
    project = bridge.projects.create("Scoped")
    session = bridge.store.create(
        frontend="cli", principal="owner", project_id=project.id)
    summary = bridge.project_summary(project.id)
    assert summary is not None
    assert summary["sessions"] == 1
    assert session.project_id == project.id


# ── AA-34: consent guard on the router ───────────────────────────────


def _consent_router(ledger):
    from nomorals.llm.providers.mock import MockProvider
    from nomorals.llm.router import LLMRouter

    calls = {"groq": 0}

    class _Groq(MockProvider):
        def chat(self, messages, params=None, **kw):
            calls["groq"] += 1
            return super().chat(messages, params, **kw)

    router = LLMRouter(cooldown_seconds=60.0, failure_threshold=2,
                       consent_ledger=ledger)
    router.add(_Groq(model="g"), name="groq", primary=True)
    router.add(MockProvider(model="m"), name="mock")
    return router, calls


def test_aa34_unconsented_hosted_provider_is_skipped():
    from nomorals.core.consent import ConsentLedger
    from nomorals.llm.base import Message

    router, calls = _consent_router(ConsentLedger())
    response = router.chat([Message.user("hi")])
    assert response.ok
    assert calls["groq"] == 0  # never touched: no byte reached it
    assert router.consent_skips == 1


def test_aa34_grant_lets_the_provider_serve():
    from nomorals.core.consent import ConsentLedger
    from nomorals.llm.base import Message

    ledger = ConsentLedger()
    ledger.grant(destination="groq", data_kinds=["chat_history"],
                 granted_by="owner")
    router, calls = _consent_router(ledger)
    response = router.chat([Message.user("hi")])
    assert response.ok
    assert calls["groq"] == 1
    assert router.consent_skips == 0


def test_aa34_all_hosted_denied_is_an_honest_error():
    from nomorals.core.consent import ConsentLedger
    from nomorals.llm.base import Message
    from nomorals.llm.providers.mock import MockProvider
    from nomorals.llm.router import LLMRouter

    router = LLMRouter(cooldown_seconds=60.0, consent_ledger=ConsentLedger())
    router.add(MockProvider(model="g"), name="groq", primary=True)
    response = router.chat([Message.user("hi")])
    assert not response.ok
    assert "consent" in (response.error or "")


def test_aa34_embed_uses_its_own_capability_and_kind():
    from nomorals.core.consent import ConsentLedger
    from nomorals.core.errors import ProviderUnavailable
    from nomorals.llm.providers.mock import MockProvider
    from nomorals.llm.router import LLMRouter

    ledger = ConsentLedger()
    ledger.grant(destination="groq", data_kinds=["chat_history"],
                 granted_by="owner")  # chat grant must NOT cover embed
    router = LLMRouter(cooldown_seconds=60.0, consent_ledger=ledger)
    router.add(MockProvider(model="g"), name="groq", primary=True)
    with pytest.raises(ProviderUnavailable) as excinfo:
        router.embed(["hello"])
    assert "consent" in str(excinfo.value)
    ledger.grant(destination="groq", data_kinds=["embeddings_text"],
                 capability="llm.embed", granted_by="owner")
    assert router.embed(["hello"])


def test_aa34_build_router_arms_only_with_a_ledger_file(tmp_path):
    from nomorals.agents.context import build_router
    from nomorals.core.config import Settings
    from nomorals.core.consent import ConsentLedger
    from nomorals.core.events import EventBus

    settings = Settings(home=str(tmp_path))
    router = build_router(settings, EventBus())
    assert router.consent_ledger is None

    ledger = ConsentLedger(tmp_path / "consent.json")
    ledger.grant(destination="groq", data_kinds=["chat_history"],
                 granted_by="owner")
    router = build_router(settings, EventBus())
    assert router.consent_ledger is not None
    decision = router.consent_ledger.check(
        destination="groq", data_kind="chat_history")
    assert decision.allowed


def test_aa34_consent_command_records_and_revokes(tmp_path, capsys):
    from nomorals.cmdline.commands.consent import _cmd_consent

    settings = SimpleNamespace(home=str(tmp_path))

    def run(**kw):
        args = argparse.Namespace(
            action=kw.get("action", "status"), target=kw.get("target", ""),
            data_kind=kw.get("data_kind", ""),
            capability=kw.get("capability", ""),
            ttl_hours=0.0, json=False)
        rc = _cmd_consent(args, settings)
        return rc, capsys.readouterr().out

    rc, out = run()
    assert rc == 0 and "not armed" in out
    rc, out = run(action="grant", target="groq")
    assert rc == 0 and "consent granted" in out
    assert (tmp_path / "consent.json").is_file()
    rc, out = run()
    assert "armed" in out and "groq" in out
    record_id = re.search(r"  (\w{16})  groq", out).group(1)
    rc, out = run(action="check", target="groq")
    assert rc == 0 and "allowed" in out
    rc, out = run(action="revoke", target=record_id)
    assert rc == 0
    rc, out = run(action="check", target="groq")
    assert rc == 1 and "DENIED" in out
    rc, out = run(action="grant", target="no-such-provider")
    assert rc == 2


# ── AA-35: autonomy wiring + nm why ──────────────────────────────────


def test_aa35_stack_constructs_the_real_satellites(tmp_path):
    from nomorals.autonomy import wiring
    from nomorals.autonomy.budget import BudgetGovernor
    from nomorals.autonomy.preemption import PreemptionLog
    from nomorals.autonomy.rhythm import OwnerRhythm
    from nomorals.autonomy.usage import UsageTracker
    from nomorals.autonomy.why import WhyStore

    stack = wiring.build_stack(tmp_path)
    assert isinstance(stack.why_store, WhyStore)
    assert isinstance(stack.preemption_log, PreemptionLog)
    assert isinstance(stack.rhythm, OwnerRhythm)
    assert isinstance(stack.budget, BudgetGovernor)
    assert isinstance(stack.usage, UsageTracker)
    # The scheduler holds THE SAME objects, not lookalikes.
    assert stack.scheduler.why_store is stack.why_store
    assert stack.scheduler.rhythm is stack.rhythm
    assert stack.scheduler.budget is stack.budget
    assert stack.scheduler.preemption is stack.preemption_log


def test_aa35_rhythm_gates_and_why_store_explains(tmp_path):
    import datetime

    from nomorals.autonomy import wiring

    # 03:00 local — inside OwnerRhythm's default 22:00–07:00 quiet
    # window no matter the machine timezone (built from local parts).
    quiet_ts = datetime.datetime(2026, 1, 15, 3, 0).timestamp()
    stack = wiring.build_stack(tmp_path, clock=lambda: quiet_ts)
    stack.scheduler.register_handler(
        "noop", lambda mission, step: "done")
    mission_id = stack.schedule_mission(
        {"goal": "nightly tidy", "plan": [{"name": "noop"}],
         "priority": "ambient", "due_at": quiet_ts - 10})
    results = stack.scheduler.run_due(quiet_ts)
    assert results[0]["status"] == "deferred"
    assert "rhythm" in results[0]["reason"]
    assert len(stack.why_store) == 0  # deferred ≠ fired: nothing to explain
    # The prep was usage-tracked by the composition root.
    assert len(stack.usage._items) == 1

    stack.rhythm.set_quiet_hours(0, 0)  # no quiet window
    results = stack.scheduler.run_due(quiet_ts)
    assert results[0]["status"] == "done"
    assert len(stack.why_store) == 1
    record = stack.why_store.why_last()
    assert record["found"] is True
    assert record["what_fired"] == f"mission:{mission_id}"


def test_aa35_nm_why_reads_the_stack(tmp_path, capsys):
    from nomorals.autonomy import wiring
    from nomorals.cmdline.commands.why import _cmd_why

    stack = wiring.build_stack(tmp_path)
    stack.scheduler.register_handler("noop", lambda m, s: "ok")
    stack.schedule_mission({"goal": "demo", "plan": [{"name": "noop"}]})
    context = SimpleNamespace(settings=SimpleNamespace(home=str(tmp_path)))

    args = argparse.Namespace(firing=["run"], json=False)
    rc = _cmd_why(args, context)
    out = capsys.readouterr().out
    assert rc == 0 and "1 firing(s)" in out

    args = argparse.Namespace(firing=[], json=False)
    rc = _cmd_why(args, context)
    out = capsys.readouterr().out
    assert rc == 0 and "why:" in out and "demo" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
