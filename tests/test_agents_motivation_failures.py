"""Tail-84 pins for nomorals/agents/motivation.py — 5 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken (trigger == fallback AND control !=
fallback, or the control proving the fallback EQUALS the module's own
designed no-data value). The file itself contains no exception
handlers, so it adds zero census sites.

motivation.py calls no failures-ledger record function anywhere
(grep-proven: only _log), so NOT-recording is pinned with the snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Verdict notes (full grounds in SURVEY.md):
- 308: _marker_path failure -> None, byte-identical to the genuine
  no-workspace value (the unguarded `if not ws: return None` path);
  both consumers (_read_marker, _write_marker) treat None as the
  first-class "no marker" state.
- 318: _read_marker failure -> {}, the designed no-usable-marker
  value (non-dict JSON yields {} by the isinstance check inside the
  same try). Consumers read per-window entries off it; {} reads as
  "not delivered", the catch-up function's own working assumption.
- 654: Scheduler.list_jobs failure -> jobs = {} in motivation_status;
  every window reports job_id/next_run None, the shape of a genuinely
  unscheduled window. Display-only surface (the tool dispatch returns
  it to the owner; nothing branches on it). Weakest exclusion of the
  file — see SURVEY.md.
- 680: TeachingsLedger.stats failure -> out["ledger"] = {} while the
  rest of the status stays intact. A healthy empty ledger returns a
  zeroed dict, so {} is the "no data" shape, never a counts claim.
- 693: TeachingsLedger.served_items failure -> [], the genuine
  empty-history value of motivation_history.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.scheduler as sched_mod
import nomorals.agents.teachings as teach_mod
from nomorals.agents import motivation as mot


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _ctx(ws=""):
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=ws))


class _BoomSettings:
    @property
    def settings(self):
        raise RuntimeError("settings boom")


def test_308_marker_path_failure_is_none():
    before = _snap()
    assert mot._marker_path(_BoomSettings()) is None
    # control: a normal context resolves the real path
    assert mot._marker_path(_ctx("/tmp/ws")) == \
        Path("/tmp/ws/motivation_last_run.json")
    # and None is also the genuine no-workspace value
    assert mot._marker_path(_ctx("")) is None
    assert _snap() == before


def test_318_read_marker_failure_is_empty_dict(tmp_path):
    before = _snap()
    ctx = _ctx(str(tmp_path))
    marker = tmp_path / "motivation_last_run.json"
    marker.write_text("{corrupt", encoding="utf-8")
    assert mot._read_marker(ctx) == {}
    # control: a valid marker round-trips
    marker.write_text(json.dumps(
        {"morning": {"lagos_date": "2026-11-01", "delivered": True}}),
        encoding="utf-8")
    assert mot._read_marker(ctx)["morning"]["delivered"] is True
    assert _snap() == before


class _BoomScheduler:
    def __init__(self, context):
        pass

    def list_jobs(self):
        raise RuntimeError("scheduler boom")


def test_654_status_jobs_failure_reports_unscheduled(monkeypatch):
    before = _snap()
    monkeypatch.setattr(sched_mod, "Scheduler", _BoomScheduler)
    out = mot.motivation_status(_ctx())
    assert out["windows"], "windows present"
    for w in out["windows"].values():
        assert w["job_id"] is None and w["next_run_utc"] is None
    assert out["backstop"] == {"job_id": None, "next_run": None}
    assert _snap() == before


class _BoomStatsLedger:
    def __init__(self, context):
        pass

    def stats(self, now=None):
        raise RuntimeError("ledger boom")


def test_680_status_ledger_failure_is_empty_dict(monkeypatch):
    before = _snap()
    monkeypatch.setattr(teach_mod, "TeachingsLedger", _BoomStatsLedger)
    out = mot.motivation_status(_ctx())
    assert out["ledger"] == {}
    assert "windows" in out  # the rest of the status is intact
    assert _snap() == before


class _BoomServedLedger:
    def __init__(self, context):
        pass

    def served_items(self, surface_prefix=""):
        raise RuntimeError("ledger boom")


def test_693_history_failure_is_empty_list(monkeypatch):
    before = _snap()
    monkeypatch.setattr(teach_mod, "TeachingsLedger", _BoomServedLedger)
    assert mot.motivation_history(_ctx()) == []
    assert _snap() == before
