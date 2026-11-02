"""Tail-92 pins: nomorals/os/project.py census sites.

Two families, both held as exclusions:

58/63 — ``_ensure_project_columns``, the schema self-heal run
at ProjectStore construction. 58: the PRAGMA probe raises ->
return without altering (comment: "table may not exist yet" —
the DDL has just run, so a fresh table is current by
construction). 63: the ALTER for the legacy ``tags`` column
raises -> swallowed. Neither hides a lasting failure: if the
tags column truly is missing afterwards, the first real INSERT
in create() names the column and fails loudly at use, not
silently at startup.
270/280/286 — ``ProjectStore.summary`` degradation labels. The
docstring contracts them in advance: the dashboard counts
"whatever is passed in" and reports the rest as unknown. A
mission lookup failure is counted under the literal status
"unknown"; session and artifact count failures land on the -1
unknown sentinel — the same value an absent store reports.
The failure is VISIBLE in the returned dict, not success-shaped:
no fabricated counts are ever produced.

NOT-recording is pinned with the snapshot method: this module
calls no failures-ledger record function anywhere, so both
ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.os import project as PJ


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


class _BoomQueryDb:
    def __init__(self):
        self.execute_calls = []

    def query(self, *args, **kwargs):
        raise RuntimeError("pragma boom")

    def execute(self, *args, **kwargs):
        self.execute_calls.append(args)


class _BoomAlterDb:
    def __init__(self):
        self.execute_calls = []

    def query(self, *args, **kwargs):
        return [{"name": "id"}, {"name": "name"}]  # no "tags" column

    def execute(self, *args, **kwargs):
        self.execute_calls.append(args)
        raise RuntimeError("alter boom")


class _BoomMissionStore:
    def get(self, mission_id):
        raise RuntimeError("mission store boom")


class _BoomSessionStore:
    def list_for_project(self, project_id):
        raise RuntimeError("session store boom")


class _ExplodingArtifactStore:
    def for_mission(self, mission_id):
        yield SimpleNamespace(id="a1")
        raise RuntimeError("artifact iteration boom")


# ── 58: pragma probe failure -> no alter attempted ──────────────

def test_58_column_probe_failure_returns_without_altering():
    before = _snap()
    db = _BoomQueryDb()
    assert PJ._ensure_project_columns(db) is None
    assert db.execute_calls == []
    assert _snap() == before


# ── 63: legacy-table ALTER failure is best-effort ───────────────

def test_63_alter_failure_swallowed_after_being_attempted():
    before = _snap()
    db = _BoomAlterDb()
    assert PJ._ensure_project_columns(db) is None
    assert len(db.execute_calls) == 1  # the ALTER was really tried
    assert _snap() == before


# ── 270/280/286: summary reports failures as unknown, not zero ──

def _store_with_missions():
    store = PJ.ProjectStore()
    project = store.create("dash", mission_ids=["m1", "m2"])
    return store, project


def test_270_mission_lookup_failure_counts_as_unknown():
    before = _snap()
    store, project = _store_with_missions()
    out = store.summary(project.id, mission_store=_BoomMissionStore())
    assert out["missions"]["total"] == 2
    assert out["missions"]["by_status"] == {"unknown": 2}
    assert _snap() == before


def test_280_session_failure_reports_unknown_sentinel():
    before = _snap()
    store, project = _store_with_missions()
    out = store.summary(project.id, session_store=_BoomSessionStore())
    assert out["sessions"] == -1  # unknown — never a fabricated count
    assert _snap() == before


def test_286_artifact_failure_reports_unknown_sentinel():
    before = _snap()
    store, project = _store_with_missions()
    out = store.summary(project.id,
                        artifact_store=_ExplodingArtifactStore())
    assert out["artifacts"] == -1
    assert _snap() == before
