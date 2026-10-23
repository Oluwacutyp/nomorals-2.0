"""Pass #98 — the motion-studio render ledger becomes a brain tool.

Pre-fix behaviour (the fail-before state of this suite on the pristine
base): ``ledger_status()`` in ``media/motion_studio/_core.py`` tells the
ledger's whole truth — totals, active window, archived count, rotation
and legacy-migration meta — but only the ``nm video`` CLI and an internal
read inside the partner runtime's ``/video list`` handler consume it.
The tool registry has no name for it, so the partner brain / tool_loop
cannot answer plain-language questions like "how many studio renders
are in the ledger?" or "has anything been archived / rotated out?".

Post-fix contract:
- a read-only tool ``motion_ledger_status`` is registered by the
  builtin tool assembly (via ``tools/media_gen.py``, the generation
  family whose engines write the ledger) and callable through
  ``ToolRegistry.call`` — the same public path the brain uses;
- the tool returns the core's truth verbatim (plus ``ok``), never a
  re-implementation: totals/active/archived/rotation/migration fields
  match a direct ``ledger_status()`` call on the same ledger;
- an optional ``recent`` argument includes the newest N entries
  (``read_ledger(limit=N)``, clamped to 100 like ``incident_history``);
  the default call is status-only;
- the tool never writes: repeated calls leave the ledger untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.motion_studio import _core
from nomorals.media.motion_studio._core import ledger_status, record_ledger
from nomorals.tools.registry import ToolRegistry

WINDOW = _core.LEDGER_WINDOW_SIZE  # 500


@pytest.fixture(scope="module")
def registry():
    """The real builtin registry, assembled the production way."""
    context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="."),
        workspace=".", db=None, router=None)
    reg = ToolRegistry(context)
    reg.register_builtins()
    return reg


@pytest.fixture()
def ledger_home(tmp_path, monkeypatch):
    """Point the ledger at a scratch dir (archive/meta derive from it)."""
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(_core, "LEDGER_PATH", path)
    monkeypatch.setattr(_core, "MEDIA_ROOT", tmp_path)
    return path


def _call(registry, **kwargs):
    outcome = registry.call("motion_ledger_status", actor="test", **kwargs)
    assert outcome.ok, f"motion_ledger_status failed: {outcome.error}"
    return outcome.value


def _write(n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        record_ledger({"kind": "test", "path": f"/renders/r{i:05}.mp4",
                       "seq": i})


# ---------------------------------------------------------------------------
# registration: the brain can find the tool through the real registry
# ---------------------------------------------------------------------------

def test_tool_registered_in_builtins(registry):
    assert "motion_ledger_status" in registry.names()
    spec = registry.get("motion_ledger_status")
    assert spec is not None
    assert spec.capability == "fs.read"          # read-only, file-backed
    params = spec.schema()["parameters"]["properties"]
    assert "recent" in params                    # the one optional argument
    assert set(params) == {"recent"}             # nothing else, no writes


# ---------------------------------------------------------------------------
# the empty ledger: honest zeros, no crash
# ---------------------------------------------------------------------------

def test_empty_ledger_honest_zeros(registry, ledger_home):
    out = _call(registry)
    assert out["ok"] is True
    assert out["total_entries"] == 0
    assert out["active_entries"] == 0
    assert out["archived_entries"] == 0
    assert out["oldest_entry"] is None
    assert out["oldest_at"] is None
    assert out["newest_at"] is None
    assert out["rotated_out_total"] == 0
    assert out["migrated_from_legacy"] is False
    assert out["window_size"] == WINDOW
    assert out["ledger_path"] == str(ledger_home)  # the isolated ledger


# ---------------------------------------------------------------------------
# a populated ledger: the tool reports the core's truth, verbatim
# ---------------------------------------------------------------------------

def test_populated_ledger_matches_core_truth(registry, ledger_home):
    _write(3)
    out = _call(registry)
    truth = ledger_status()
    assert out == {"ok": True, **truth}          # no re-implementation drift
    assert out["total_entries"] == 3
    assert out["active_entries"] == 3
    assert out["archived_entries"] == 0
    assert out["oldest_entry"]["seq"] == 0
    assert out["newest_at"] == truth["newest_at"]


def test_rotation_meta_surfaced(registry, ledger_home):
    _write(WINDOW + 25)
    out = _call(registry)
    assert out["total_entries"] == WINDOW + 25
    assert out["active_entries"] == WINDOW        # live window stays bounded
    assert out["archived_entries"] == 25          # rotated out, not lost
    assert out["rotated_out_total"] == 25
    assert out["first_rotated_out_at"] is not None
    assert out["oldest_entry"]["seq"] == 0        # history still whole


def test_legacy_migration_meta_surfaced(registry, ledger_home):
    legacy = [{"kind": "kenburns", "path": f"/old/r{i:05}.mp4", "seq": i}
              for i in range(WINDOW)]             # old shape: no ts/archive
    Path(ledger_home).write_text(json.dumps(legacy, indent=1))
    out = _call(registry)
    assert out["total_entries"] == WINDOW
    assert out["migrated_from_legacy"] is True
    assert out["migrated_entries"] == WINDOW


# ---------------------------------------------------------------------------
# the optional recent-entries view
# ---------------------------------------------------------------------------

def test_recent_entries_opt_in(registry, ledger_home):
    _write(3)
    status_only = _call(registry)
    assert "recent" not in status_only            # default: status only

    out = _call(registry, recent=2)
    assert [e["seq"] for e in out["recent"]] == [1, 2]   # newest, in order
    assert out["total_entries"] == 3                    # status still whole

    out = _call(registry, recent=50)              # more than exists: all of it
    assert [e["seq"] for e in out["recent"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# read-only: asking never changes the ledger
# ---------------------------------------------------------------------------

def test_tool_is_read_only(registry, ledger_home):
    _write(4)
    archive = ledger_home.with_name("ledger.archive.jsonl")
    before = (ledger_home.read_bytes(), archive.read_bytes())
    _call(registry)
    _call(registry, recent=4)
    _call(registry)
    after = (ledger_home.read_bytes(), archive.read_bytes())
    assert before == after
    assert ledger_status()["total_entries"] == 4
