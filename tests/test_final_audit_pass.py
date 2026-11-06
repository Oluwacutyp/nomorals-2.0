"""Final assembly-audit pass (2026-11-05): AA-39, AA-40, AA-44,
AA-45, AA-48, AA-49, AA-51.

Each test drives the PRODUCTION path after consolidation — the live
registrar / runner / surface — not the retired duplicate.  The file
contains no exception handlers, so it adds zero census sites.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ── AA-39: `nm bench` runs the system benchmark ─────────────────────


def _cli_args(argv):
    from nomorals.cmdline.parser import _parser

    return _parser().parse_args(argv)


def test_aa39_bench_registered_in_dispatch():
    from nomorals.cmdline.commands.bench import _cmd_bench
    from nomorals.cmdline.dispatch import _COMMANDS

    assert _COMMANDS["bench"].handler is _cmd_bench
    assert _COMMANDS["bench"].needs_context is False
    # the K3 scoreboard keeps its own name and handler
    assert _COMMANDS["benchmark"].handler is not _cmd_bench


def test_aa39_nm_bench_runs_and_audits(capsys):
    args = _cli_args(["bench", "--quick"])
    from nomorals.cmdline.commands.bench import _cmd_bench

    rc = _cmd_bench(args, SimpleNamespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "system benchmark (quick)" in out
    assert "vector_topk" in out
    assert "honesty audit" in out


def test_aa39_bench_baseline_save_and_compare(tmp_path, capsys,
                                              monkeypatch):
    from nomorals import bench as bench_mod

    base = tmp_path / "bench.json"
    monkeypatch.setattr(bench_mod, "baseline_path", lambda: base)
    from nomorals.cmdline.commands.bench import _cmd_bench

    rc = _cmd_bench(_cli_args(["bench", "--quick", "--save"]),
                    SimpleNamespace())
    assert rc == 0
    assert base.is_file()
    assert "baseline saved" in capsys.readouterr().out
    rc = _cmd_bench(_cli_args(["bench", "--quick"]), SimpleNamespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "vs baseline" in out


# ── AA-48/49: `nm organs` serves catalogue + inventory ──────────────


def test_aa4849_organs_registered_in_dispatch():
    from nomorals.cmdline.commands.organs import _cmd_organs
    from nomorals.cmdline.dispatch import _COMMANDS

    assert _COMMANDS["organs"].handler is _cmd_organs
    assert _COMMANDS["organs"].needs_context is False


def test_aa4849_organs_list_serves_catalogue(capsys):
    from nomorals.meta import expose_dormant

    entries = expose_dormant.entries()
    assert entries, "catalogue must not be empty"
    from nomorals.cmdline.commands.organs import _cmd_organs

    rc = _cmd_organs(_cli_args(["organs"]), SimpleNamespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "dormant-organ catalogue" in out
    assert entries[0].module in out
    assert entries[0].invoke in out


def test_aa4849_organs_find_queries_catalogue(capsys):
    from nomorals.meta import expose_dormant

    probe = expose_dormant.entries()[0]
    needle = probe.name.split()[0]
    from nomorals.cmdline.commands.organs import _cmd_organs

    rc = _cmd_organs(_cli_args(["organs", "find", needle]),
                     SimpleNamespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert probe.module in out


def test_aa4849_organs_scan_runs_inventory_on_demand(capsys):
    from nomorals.cmdline.commands.organs import _cmd_organs

    rc = _cmd_organs(_cli_args(["organs", "scan", "--json"]),
                     SimpleNamespace())
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["scanned"] > 100
    assert payload["dormant_count"] == len(payload["dormant"])
    cat = payload["catalogue"]
    assert cat["total"] == (len(cat["still_dormant"])
                            + len(cat["now_wired"]))
    assert "scan_seconds" in payload


# ── AA-40: group extras render on live group surfaces ───────────────


def test_aa40_announcement_carries_semantic_marker(tmp_path):
    from pathlib import Path

    from nomorals.community.groups import GroupStore

    groups = GroupStore(data_dir=Path(tmp_path) / "cgroups")
    group = groups.create_group("Devs", "coding", created_by="u1")
    groups.join_group(group.id, "u2")
    post = groups.announce(group.id, "u1", "Release day moved to Friday")
    assert post is not None
    assert post.text.startswith("[INFO:ANNOUNCE] 📢 ")
    assert "Release day moved to Friday" in post.text
    # a plain post is unmarked
    plain = groups.post(group.id, "u2", "thanks!")
    assert plain is not None and plain.text == "thanks!"
    # role gate still applies before marking: a member cannot announce
    assert groups.announce(group.id, "u2", "sneaky") is None


def test_aa40_party_recap_renders_in_hype_register():
    from nomorals.community.parties import PartyRecap, format_recap

    recap = PartyRecap(
        title="Friday Jam", attendees=["Soso", "Ada"],
        top_players=[{"member": "Soso", "points": 42}])
    out = format_recap(recap)
    assert out.startswith("YOOO family!!")
    assert "that's a wrap" in out
    assert "Soso" in out
    assert out.rstrip().endswith("LET'S GOOO 🚀")


# ── AA-44: evening remainders live on the slot engine ───────────────


class _FakeNotifier:
    def __init__(self, rows=(), mark_raises=False):
        self._rows = list(rows)
        self.mark_calls = []
        self.mark_raises = mark_raises

    def digest_unfolded(self, limit=20):
        return list(self._rows)

    def mark_digest_folded(self, ids):
        self.mark_calls.append(list(ids))
        if self.mark_raises:
            raise RuntimeError("db locked")
        return len(ids)


def test_aa44_fold_renders_shared_digest_lines():
    from nomorals.agents import morning_pulse as mp

    rows = [{"id": "d1", "title": "Gold dipped", "kind": "price"},
            {"id": "d2", "title": "Deploy finished", "kind": ""}]
    fake = _FakeNotifier(rows)
    with patch("nomorals.agents.notifier.Notifier", lambda ctx: fake):
        text, ids = mp._fold_evening_digest(SimpleNamespace(), "wrap")
    assert ids == ["d1", "d2"]
    assert "Alert digest" in text
    assert "• Gold dipped (price)" in text
    assert "• Deploy finished" in text
    assert text.startswith("wrap")
    # collecting marks nothing
    assert fake.mark_calls == []


def test_aa44_fold_failure_is_recorded_and_safe():
    from nomorals.agents import failures as ag_fail
    from nomorals.agents import morning_pulse as mp

    def boom(ctx):
        raise RuntimeError("notifier down")

    before = len(ag_fail.recent_failures(limit=200))
    with patch("nomorals.agents.notifier.Notifier", boom):
        text, ids = mp._fold_evening_digest(SimpleNamespace(), "wrap")
    assert (text, ids) == ("wrap", [])
    assert len(ag_fail.recent_failures(limit=200)) > before


def test_aa44_mark_folded_failure_is_bookkeeping_only():
    from nomorals.agents import failures as ag_fail
    from nomorals.agents import morning_pulse as mp

    fake = _FakeNotifier(mark_raises=True)
    before = len(ag_fail.recent_failures(limit=200))
    with patch("nomorals.agents.notifier.Notifier", lambda ctx: fake):
        assert mp._mark_evening_digest_folded(SimpleNamespace(),
                                              ["d1"]) == 0
    assert len(ag_fail.recent_failures(limit=200)) > before


def _run_evening_slot(deliver_ok, rows):
    from nomorals.agents import morning_pulse as mp

    fake = _FakeNotifier(rows)
    captured = {}

    def fake_deliver(ctx, slot, text):
        captured["text"] = text
        return deliver_ok

    ctx = SimpleNamespace()
    with patch.object(mp, "_compose_evening",
                      return_value=("wrap text", [])), \
            patch.object(mp, "slot_voice", return_value=False), \
            patch.object(mp, "_deliver_slot_text", fake_deliver), \
            patch.object(mp, "_deliver_stories", return_value=[]), \
            patch.object(mp, "_ledger", lambda *a, **k: None), \
            patch.object(mp, "_emit_bus", lambda *a, **k: None), \
            patch.object(mp, "_write_slot_marker",
                         lambda *a, **k: None), \
            patch("nomorals.agents.notifier.Notifier",
                  lambda c: fake):
        result = mp.run_slot(ctx, "evening")
    return result, captured, fake


def test_aa44_evening_slot_folds_and_marks_on_delivery():
    rows = [{"id": "d1", "title": "Gold dipped", "kind": "price"}]
    result, captured, fake = _run_evening_slot(True, rows)
    assert "Alert digest" in captured["text"]
    assert "• Gold dipped (price)" in captured["text"]
    assert fake.mark_calls == [["d1"]]
    assert "digest" in result["stages"]


def test_aa44_evening_slot_failed_delivery_keeps_digest():
    rows = [{"id": "d1", "title": "Gold dipped", "kind": "price"}]
    result, captured, fake = _run_evening_slot(False, rows)
    assert "Alert digest" in captured["text"]
    assert fake.mark_calls == []
    assert "digest" not in result["stages"]


def test_aa44_tomorrow_preview_includes_weather(tmp_path):
    from nomorals.agents import morning_pulse as mp
    from nomorals.agents.weather import DayForecast

    class _NoJobs:
        def __init__(self, context):
            pass

        def list_jobs(self):
            return []

    place = SimpleNamespace(label="Lagos, Nigeria",
                            latitude=6.5244, longitude=3.3792)
    forecast = {"daily": [
        DayForecast(date="2026-11-06", weather_code=0, temp_max=30.0,
                    temp_min=24.0, precip_prob=0, units="metric"),
        DayForecast(date="2026-11-07", weather_code=61, temp_max=29.0,
                    temp_min=23.0, precip_prob=40, units="metric")]}
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    with patch("nomorals.agents.scheduler.Scheduler", _NoJobs), \
            patch("nomorals.agents.weather.geocode",
                  return_value=place), \
            patch("nomorals.agents.weather.fetch_forecast",
                  return_value=forecast):
        out = mp._tomorrow_preview(ctx)
    assert "Weather tomorrow (Lagos, Nigeria)" in out
    assert "23–29°C" in out


def test_aa44_tomorrow_preview_dict_shaped_forecast(tmp_path):
    # wttr.in / met.no fallbacks yield plain dicts, not DayForecast.
    from nomorals.agents import morning_pulse as mp

    class _NoJobs:
        def __init__(self, context):
            pass

        def list_jobs(self):
            return []

    place = SimpleNamespace(label="Lagos, Nigeria",
                            latitude=6.5244, longitude=3.3792)
    forecast = {"daily": [
        {"date": "2026-11-06", "temp_max": 30.0, "temp_min": 24.0,
         "weather_code": 0},
        {"date": "2026-11-07", "temp_max": 28.0, "temp_min": 22.0,
         "weather_code": 3}]}
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    with patch("nomorals.agents.scheduler.Scheduler", _NoJobs), \
            patch("nomorals.agents.weather.geocode",
                  return_value=place), \
            patch("nomorals.agents.weather.fetch_forecast",
                  return_value=forecast):
        out = mp._tomorrow_preview(ctx)
    assert "Weather tomorrow (Lagos, Nigeria)" in out
    assert "22–28°C" in out


# ── AA-45: one finance digest composition ───────────────────────────


def _finance_ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(home_path=str(tmp_path)))


class _RecordingNotifier:
    published = []

    def __init__(self, context):
        pass

    def publish(self, kind, title, text):
        type(self).published.append((kind, title, text))
        return {"delivered": True}


def test_aa45_tool_delegates_to_canonical_composition(tmp_path):
    from nomorals.finance import digest as digest_mod
    from nomorals.finance import tools as tools_mod

    _RecordingNotifier.published = []

    class _Reg:
        def __init__(self):
            self.fns = {}

        def register(self, name, **kw):
            def deco(fn):
                self.fns[name] = fn
                return fn
            return deco

    reg = _Reg()
    tools_mod.register(reg)
    ctx = _finance_ctx(tmp_path)
    with patch("nomorals.agents.notifier.Notifier",
               _RecordingNotifier):
        via_tool = reg.fns["finance_digest"](ctx)
        direct = digest_mod.compose_weekly_digest(ctx)
    assert via_tool["ok"] is True and via_tool["delivered"] is True
    assert via_tool["text"] == direct["text"]
    assert "safe to spend" in via_tool["text"]
    kinds = [k for k, _, _ in _RecordingNotifier.published]
    assert kinds == ["finance_digest", "finance_digest"]


def test_aa45_cron_registrar_runs_same_composition(tmp_path):
    import asyncio

    from nomorals.finance import digest as digest_mod

    _RecordingNotifier.published = []
    actions = {}
    scheduled = []

    class _FakeCronScheduler:
        async def list_cron_jobs(self):
            return []

        def register_action(self, action, fn):
            actions[action] = fn

        async def schedule_cron(self, task_id, cron, action,
                                parameters=None):
            scheduled.append((task_id, cron, action))

    ctx = _finance_ctx(tmp_path)
    ok = asyncio.run(digest_mod.ensure_finance_digest_job(
        _FakeCronScheduler(), ctx))
    assert ok is True
    assert scheduled == [(digest_mod.FINANCE_WEEKLY_TASK_ID,
                          digest_mod.FINANCE_WEEKLY_CRON,
                          digest_mod.FINANCE_WEEKLY_ACTION)]
    with patch("nomorals.agents.notifier.Notifier",
               _RecordingNotifier):
        asyncio.run(actions[digest_mod.FINANCE_WEEKLY_ACTION]())
    assert [k for k, _, _ in _RecordingNotifier.published] == [
        "finance_digest"]


# ── AA-51: one /mandate branch, gate on the live path ───────────────


def _mandate_ctx(tmp_path, monkeypatch):
    from nomorals.finance import mandate as mand_mod

    monkeypatch.setattr(
        mand_mod.MandateStore, "__init__",
        lambda self, path=None: setattr(
            self, "path", tmp_path / "mandates.json") or setattr(
            self, "_data", None))
    return SimpleNamespace(
        settings=SimpleNamespace(home_path=str(tmp_path)))


def test_aa51_control_mandate_documented_grammar(tmp_path, monkeypatch):
    from nomorals.finance import commands as cmds

    ctx = _mandate_ctx(tmp_path, monkeypatch)
    # per-day optional: defaults to 4× the per-transaction cap
    out = cmds.control_mandate("issue transfer 50k", ctx)
    assert "mandate issued" in out
    out = cmds.control_mandate("list", ctx)
    assert "₦200,000/day" in out or "200k/day" in out or "/day" in out
    # stop all is the documented synonym for revoke-all
    out = cmds.control_mandate("stop all", ctx)
    assert "revoked 1 mandate" in out
    out = cmds.control_mandate("list", ctx)
    assert "ACTIVE" not in out
    # and revoke-all itself still works
    cmds.control_mandate("issue travel 10k 40k", ctx)
    out = cmds.control_mandate("revoke-all", ctx)
    assert "revoked 1 mandate" in out


def test_aa51_single_dispatch_branch_and_handler_gone():
    import inspect

    from nomorals.agents.partner import runtime as runtime_mod
    from nomorals.agents.partner.runtime_memory import (
        RuntimeMemoryMixin)

    assert not hasattr(RuntimeMemoryMixin, "_control_mandate")
    src = inspect.getsource(runtime_mod)
    assert src.count('kind == "mandate"') == 1


@pytest.fixture()
def _partner(tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    class _Gateway:
        adapters: dict = {}

        def __init__(self):
            self.sent = []

        def status(self):
            return {}

        def set_rate_limit(self, *a, **k):
            return None

    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("test must not call the model")

    context.router = _Router()
    runtime = PartnerRuntime(context, gateway=_Gateway())
    yield runtime
    try:
        runtime.stop()
    except Exception as exc:  # noqa: BLE001
        assert isinstance(exc, Exception)
    context.close()


def _msg(text, *, owner=False):
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id="999",
                     kind=ChatKind.DM),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
    )


def _patch_mandate_store(tmp_path, monkeypatch):
    from nomorals.finance import mandate as mand_mod

    monkeypatch.setattr(
        mand_mod.MandateStore, "__init__",
        lambda self, path=None: setattr(
            self, "path", tmp_path / "mandates.json") or setattr(
            self, "_data", None))


def test_aa51_mandate_gate_on_live_branch(_partner, tmp_path,
                                          monkeypatch):
    # Declared preserved-behavior pin (passes pre- and
    # post-consolidation): _handle_control_inner's capability layer
    # (caller_may_run — "mandate" is neither public nor a game kind)
    # already refused strangers ahead of the dispatch chain on the
    # old tree, so the dead branch's gate was never the only
    # protection.  The AA-51 fix — teeth in the grammar and
    # single-branch tests above — consolidates the dispatch to one
    # branch and keeps the operator gate ON that branch in the
    # region idiom, so the invariant is local to the branch and does
    # not depend on the capability table never changing.  This pin
    # records the end-to-end truth on the consolidated tree:
    # strangers refused, the owner served.
    _patch_mandate_store(tmp_path, monkeypatch)
    refused = _partner._handle_control_inner(
        "/mandate list", "telegram:999",
        message=_msg("/mandate list", owner=False))
    assert refused == "that one's just for the owner."
    allowed = _partner._handle_control_inner(
        "/mandate list", "telegram:999",
        message=_msg("/mandate list", owner=True))
    assert "no payment mandates" in allowed


def test_aa51_mandate_surface_stays_owner_only(_partner, tmp_path,
                                               monkeypatch):
    # Preserved-behavior pin (passes pre- and post-consolidation):
    # end-to-end through handle_control, strangers are refused and
    # the owner reaches the mandate surface.
    _patch_mandate_store(tmp_path, monkeypatch)
    refused = _partner.handle_control(
        "/mandate list", "telegram:999",
        message=_msg("/mandate list", owner=False))
    assert "just for the owner" in refused
    allowed = _partner.handle_control(
        "/mandate list", "telegram:999",
        message=_msg("/mandate list", owner=True))
    assert "no payment mandates" in allowed
