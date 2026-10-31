"""Tail-74 pins for nomorals/hub/server.py —
0 converts / 6 exclusions (750, 754, 758, 804, 815, 824).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED (healthy fake returns a distinctive value and the
surface shows it), then the failure half asserting the pinned
degradation. The file contains no exception handlers, so it
adds zero census sites.

The ops surfaces are never-raise by contract (module docstring:
/health always answers 200; every guard is commented "health /
metrics must not raise"). The health VERDICT lives in
_component_status — not a census site; its guards bind and
record errors in the payload. These six are the informational
fields around the verdict; NOT-recording is pinned as caplog
silence on the module logger (nomorals.hub.server), which none
of the six sites writes to.

Verdict notes (full grounds in the tail-74 SURVEY.md):
- 750: stats() failure -> mesh_tasks {} — byte-identical to a
  healthy idle mesh (real stats() returns totals {}), stated
  plainly; held because the verdict fields (ok/components) are
  computed from the honest ping guards and nothing decides on
  mesh_tasks.
- 754 / 758: failure -> -1, a sentinel no healthy value can
  take (counts and seqs are >= 0; empty store max_seq is 0).
- 804: failure -> the -1 sentinel is emitted into the metrics
  exposition itself.
- 815: stats() failure -> mesh_tasks series absent, identical
  to healthy-idle on that family; weakest exclusion of the
  batch (shelf: scrape-error counter feature).
- 824: max_seq() failure -> value line absent; healthy ALWAYS
  emits it, so the gap is visible — missing, not false.
"""

import logging
from types import SimpleNamespace

from nomorals.hub.server import HubServer
from nomorals.storage.db import Database

HUB_LOG = "nomorals.hub.server"


def _hub(tmp_path):
    db = Database(str(tmp_path / "hub.db"))
    db.migrate()
    return HubServer(db)


def _boom(*a, **k):
    raise RuntimeError("transport down")


def _transport(*, stats=None, active_nodes=None):
    return SimpleNamespace(
        ping=lambda: 0.001,
        stats=stats if stats is not None
        else (lambda: {"totals": {"done": 3}}),
        active_nodes=active_nodes if active_nodes is not None
        else (lambda: ["n1", "n2"]))


def _peer(*, max_seq=None):
    return SimpleNamespace(
        ping=lambda: True,
        store=SimpleNamespace(
            max_seq=max_seq if max_seq is not None else (lambda: 7)))


def _silent(caplog):
    return [r for r in caplog.records if r.name == HUB_LOG]


def test_750_stats_failure_degrades_mesh_tasks_not_the_verdict(
        tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    healthy = hub._health()
    assert healthy["mesh_tasks"] == {"done": 3}  # attempt proven
    assert healthy["ok"] is True
    hub.transport = _transport(stats=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        broken = hub._health()
    assert broken["mesh_tasks"] == {}
    # the verdict surface stays honest: components are the
    # ping guards' own (recorded) results, untouched by 750
    assert broken["ok"] is True
    assert broken["components"]["mesh"]["status"] == "SERVING"
    assert _silent(caplog) == []


def test_754_active_nodes_failure_is_minus_one_not_a_count(
        tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    assert hub._health()["nodes_active"] == 2  # attempt proven
    hub.transport = _transport(active_nodes=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        broken = hub._health()
    assert broken["nodes_active"] == -1
    assert _silent(caplog) == []


def test_758_max_seq_failure_is_minus_one_not_a_sequence(
        tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    assert hub._health()["sync_head_seq"] == 7  # attempt proven
    hub.sync_peer = _peer(max_seq=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        broken = hub._health()
    assert broken["sync_head_seq"] == -1
    assert _silent(caplog) == []


def test_804_metrics_nodes_failure_emits_minus_one(tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    text = hub._metrics_text().body.decode()
    assert "nomorals_hub_nodes_active 2" in text  # attempt proven
    hub.transport = _transport(active_nodes=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        text = hub._metrics_text().body.decode()
    assert "nomorals_hub_nodes_active -1" in text
    assert _silent(caplog) == []


def test_815_metrics_stats_failure_omits_mesh_tasks_series(
        tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    text = hub._metrics_text().body.decode()
    assert 'nomorals_hub_mesh_tasks{status="done"} 3' in text
    hub.transport = _transport(stats=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        text = hub._metrics_text().body.decode()
    value_lines = [ln for ln in text.splitlines()
                   if ln.startswith("nomorals_hub_mesh_tasks{")]
    assert value_lines == []
    # HELP/TYPE headers for the family are still emitted; only
    # the values are absent (identical to a healthy idle mesh)
    assert "# TYPE nomorals_hub_mesh_tasks gauge" in text
    assert _silent(caplog) == []


def test_824_metrics_max_seq_failure_omits_the_value_line(
        tmp_path, caplog):
    hub = _hub(tmp_path)
    hub.transport = _transport()
    hub.sync_peer = _peer()
    text = hub._metrics_text().body.decode()
    assert "nomorals_hub_sync_head_seq 7" in text  # attempt proven
    hub.sync_peer = _peer(max_seq=_boom)
    with caplog.at_level(logging.DEBUG, logger=HUB_LOG):
        caplog.clear()
        text = hub._metrics_text().body.decode()
    value_lines = [ln for ln in text.splitlines()
                   if ln.startswith("nomorals_hub_sync_head_seq ")]
    assert value_lines == []
    assert "# TYPE nomorals_hub_sync_head_seq gauge" in text
    assert _silent(caplog) == []
