"""Assembly-audit false-success remainder (2026-11-05): AA-66/69/70/
73/74/75/78/47 behavior tests.

Doctrine (owner law): any swallow that turns a failure into a clean,
empty, healthy, or "none" report is a defect; equally, a ledger must
not log failure for what succeeded. Each test below pins the honest
signal reaching the surface that consumes it — exit code, status
dict, report field, returned record — not just an internal flag.

AA-72 is pinned in tests/test_agents_morning_briefing_failures.py
(test_731, converted there) and AA-69's JSON contract in
tests/test_cmdline_commands_doctor_failures.py (test_83, converted
there); this file adds AA-69's text surface. AA-76 is not here: the
contradiction check and its only production caller both live inside
nomorals/memory/persona.py, which this pass must not touch, and no
caller/wrapper layer outside that file can observe the swallowed
recall failure — reported as a deviation instead of faked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

# ── AA-66: `nm memory curate` exit code tells the truth ─────────────

def _curate_rc(monkeypatch, report) -> int:
    from nomorals.cmdline.commands import memory as mem_cmd
    from nomorals.memory.persona import MemoryCurator

    monkeypatch.setattr(MemoryCurator, "scheduled_pass",
                        lambda self: report)
    args = argparse.Namespace(memory_action="curate", json=True)
    ctx = SimpleNamespace(memory=SimpleNamespace(db=None))
    return mem_cmd._cmd_memory_action(args, ctx)


def test_aa66_curate_healthy_pass_exits_0(monkeypatch):
    report = {"dedup": {"ok": True, "merged": 0, "kept": 3},
              "forgotten": 0}
    assert _curate_rc(monkeypatch, report) == 0


def test_aa66_curate_failed_dedup_exits_1(monkeypatch):
    report = {"dedup": {"ok": False, "merged": 0, "kept": 3,
                        "failures": [{"id": "r1", "error": "boom"}]},
              "forgotten": 0}
    assert _curate_rc(monkeypatch, report) == 1


def test_aa66_curate_error_report_exits_1(monkeypatch):
    report = {"dedup": {"ok": True, "merged": 0, "kept": 3},
              "forgotten": 0, "error": "forget pass died"}
    assert _curate_rc(monkeypatch, report) == 1


# ── AA-69: doctor text surface names a failed config check ──────────

def test_aa69_doctor_text_prints_config_check_failure(tmp_path, capsys,
                                                      monkeypatch):
    import nomorals.core.config as cfg
    from nomorals.cmdline.commands import doctor as doc

    monkeypatch.setattr(cfg, "config_env_status",
                        mock.Mock(side_effect=RuntimeError("cfg down")))
    settings = SimpleNamespace(
        db_path=str(tmp_path / "doc.db"), profile="test",
        concurrency=SimpleNamespace(processes=None,
                                    use_processes=False))
    rc = doc._cmd_doctor(
        argparse.Namespace(json=False, build_native=False), settings)
    out = capsys.readouterr().out
    assert rc == 1
    assert "config keys:" in out
    assert "check failed" in out
    assert "cfg down" in out


# ── AA-70: research-loop status distinguishes probe failure ─────────

def _rl_ctx():
    return SimpleNamespace(db=None, settings=None)


def test_aa70_failed_probes_are_named_in_errors():
    from nomorals.agents import research_loop as rl
    from nomorals.agents.scheduler import Scheduler
    from nomorals.agents.upgrade_queue import UpgradeQueue

    with patch.object(rl, "_in_quiet_hours_now",
                      side_effect=RuntimeError("clock down")), \
            patch.object(Scheduler, "list_jobs",
                         side_effect=RuntimeError("sched down")), \
            patch.object(UpgradeQueue, "list",
                         side_effect=RuntimeError("queue down")):
        out = rl.status(_rl_ctx())
    # Defaults stay (they are the fallback values) — but they are no
    # longer indistinguishable from measurements.
    assert out["job"]["scheduled"] is False
    assert out["gates"]["quiet_hours"] is False
    assert out["pending_proposals"] == 0
    assert set(out["errors"]) == {"quiet_hours", "scheduler",
                                  "upgrade_queue"}
    assert "clock down" in out["errors"]["quiet_hours"]
    assert "sched down" in out["errors"]["scheduler"]
    assert "queue down" in out["errors"]["upgrade_queue"]


def test_aa70_healthy_probes_leave_errors_empty():
    from nomorals.agents import research_loop as rl
    from nomorals.agents.scheduler import Scheduler
    from nomorals.agents.upgrade_queue import UpgradeQueue

    with patch.object(rl, "_in_quiet_hours_now", return_value=False), \
            patch.object(Scheduler, "list_jobs", return_value=[]), \
            patch.object(UpgradeQueue, "list", return_value=[]):
        out = rl.status(_rl_ctx())
    assert out["errors"] == {}


def test_aa70_research_cli_surfaces_probe_failures(capsys, monkeypatch):
    from nomorals.agents import research_loop as rlmod
    from nomorals.cmdline.commands import research as research_cmd

    canned = {
        "job": {"scheduled": False, "enabled": False,
                "interval_hours": 6},
        "gates": {"feature_research": False, "proactive_master": True,
                  "quiet_hours": False},
        "last_run": None, "pending_proposals": 0,
        "topics": [], "signals": {},
        "errors": {"scheduler": "RuntimeError: sched down"},
    }
    monkeypatch.setattr(rlmod, "status", lambda ctx: canned)
    args = argparse.Namespace(action="status", json=False)
    rc = research_cmd._cmd_research_loop(args, SimpleNamespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "probes failed" in out
    assert "scheduler" in out


# ── AA-73: wisdom blob-index corruption surfaces ────────────────────

def _bare_ingestor(tmp_path: Path):
    from nomorals.wisdom.ingestor import ArchiveIngestor

    ing = ArchiveIngestor.__new__(ArchiveIngestor)
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    ing._blobs = blobs
    ing._index_path = blobs / "index.json"
    ing._index = None
    ing._http = None
    return ing


def test_aa73_non_dict_index_is_corruption_not_emptiness(tmp_path):
    from nomorals.core import failures as core_failures

    ing = _bare_ingestor(tmp_path)
    ing._index_path.write_text("[1, 2, 3]", encoding="utf-8")
    before = core_failures.failure_count()
    assert ing._load_index() == {}
    assert core_failures.failure_count() > before
    quarantined = ing._index_path.with_name("index.json.corrupt")
    assert quarantined.read_text(encoding="utf-8") == "[1, 2, 3]"


def test_aa73_missing_index_is_a_quiet_empty(tmp_path):
    from nomorals.core import failures as core_failures

    ing = _bare_ingestor(tmp_path)
    before = core_failures.failure_count()
    assert ing._load_index() == {}
    # "No index yet" is a normal state: no record, no quarantine.
    assert core_failures.failure_count() == before
    assert not ing._index_path.with_name(
        "index.json.corrupt").exists()


# ── AA-74: heal_all marks a partial run as partial ──────────────────

def test_aa74_loop_failure_marks_summary_truncated():
    from nomorals.core import self_heal as sh

    def gen():
        yield "fine"
        raise RuntimeError("iterable died")

    summary = sh.heal_all(lambda *a, **k: "ok", gen())
    assert summary["total"] == 1
    assert summary["truncated"] is True
    assert "iterable died" in summary["error"]


def test_aa74_clean_run_carries_no_truncation_marks():
    from nomorals.core import self_heal as sh

    summary = sh.heal_all(lambda *a, **k: "ok", [])
    assert summary["total"] == 0
    assert "error" not in summary
    assert "truncated" not in summary


# ── AA-75: note_result's return tells the truth about persistence ───

class _SaveFailingDB:
    """Delegates everything except the health-record INSERT."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, sql, params=()):
        if sql.lstrip().upper().startswith(
                "INSERT INTO AGENCY_SOURCE_HEALTH"):
            raise RuntimeError("disk full (injected)")
        return self._inner.execute(sql, params)

    def query_one(self, sql, params=()):
        return self._inner.query_one(sql, params)

    def query(self, sql, params=()):
        return self._inner.query(sql, params)


def _health_spec():
    from nomorals.agency.sourcehealth import RSS, SourceSpec

    return SourceSpec(key="aa75-feed", category="news", name="Feed",
                      url="https://feeds.example/rss.xml", kind=RSS)


def test_aa75_failed_save_marks_record_unpersisted_and_records():
    from nomorals.agency import failures as agency_failures
    from nomorals.agency.sourcehealth import SourceHealth
    from nomorals.storage.db import Database

    failing = SourceHealth(_SaveFailingDB(Database(":memory:")))
    before = list(agency_failures.recent_failures())
    rec = failing.note_result(_health_spec(), ok=True, items=5,
                              now=1000.0)
    # The run is not broken (note_result never raises) and the
    # in-memory record is the computed one — but it must not claim
    # a persistence that did not happen.
    assert rec.status == "verified-live"
    assert rec.persisted is False
    new_sites = [f.site for f in agency_failures.recent_failures()
                 if f not in before]
    assert "SourceHealth.note_result" in new_sites


def test_aa75_successful_save_is_persisted():
    from nomorals.agency.sourcehealth import SourceHealth
    from nomorals.storage.db import Database

    health = SourceHealth(Database(":memory:"))
    rec = health.note_result(_health_spec(), ok=True, items=5,
                             now=1000.0)
    assert rec.persisted is True
    assert health.record(_health_spec().key).status == "verified-live"


# ── AA-78: platforms.py capability claim matches reality ────────────

def test_aa78_whatsapp_buttons_claim_matches_platform_limits():
    from nomorals.social.chat import platforms
    from nomorals.social.chat.platform_limits import for_platform

    assert "No inline buttons" not in (platforms.__doc__ or "")
    assert for_platform("whatsapp").supports_buttons is True


# ── AA-47: Plaid defaults to production; sandbox is opt-in ──────────

def test_aa47_default_env_is_production(monkeypatch):
    from nomorals.connectors.plaid import PlaidConnector

    monkeypatch.delenv("PLAID_ENV", raising=False)
    assert PlaidConnector._normalize_env(None) == "production"
    assert PlaidConnector._normalize_env("") == "production"
    assert PlaidConnector._normalize_env("  ") == "production"


def test_aa47_sandbox_remains_explicit_opt_in(monkeypatch):
    from nomorals.connectors.plaid import PlaidConnector

    assert PlaidConnector._normalize_env("sandbox") == "sandbox"
    monkeypatch.setenv("PLAID_ENV", "sandbox")
    assert PlaidConnector._normalize_env(None) == "sandbox"
    # An explicit argument still beats the env var, both ways.
    assert PlaidConnector._normalize_env("production") == "production"
    monkeypatch.setenv("PLAID_ENV", "production")
    assert PlaidConnector._normalize_env("sandbox") == "sandbox"
