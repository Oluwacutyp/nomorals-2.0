"""Tail-106 pins: nomorals/autonomy/unified.py census sites
(122, 130, 143).

All three sites are held as EXCLUSIONS:

122 — ``atomic_write_json``'s finally cleanup: when a write
fails mid-way (the real error is already propagating), removing
the leftover temp file can itself fail (OSError -> pass). The
guard's job is that the cleanup failure never masks the real
write error: the caller sees the replace failure, not an unlink
failure. The honest cost is pinned below: the stranded
``.tmp.<pid>`` file stays on disk; the next write uses a fresh
pid-named temp and is unaffected. On a successful write the temp
was renamed away, so this rung never fires. FLAGGED-ACCEPTED: no
noqa on the except, so the site stays in the flagged scan as an
accepted remainder.

130 — the directory fsync guard: the module header prints the
durability contract (temp file -> flush + fsync -> atomic rename
-> directory fsync) and the except comment states the exception
("directory fsync is best-effort on some filesystems"). The
payload's durability comes from the file fsync + atomic rename,
which happen before this block and raise honestly on failure.
Pinned by making ``os.open`` raise: the write still lands
complete.

143 — ``normalize_mission``'s duck-typing cascade: a mission
object whose ``to_dict()`` raises falls back to reading the
object's attributes directly (id/goal/state.plan and friends) —
the fallback consumes the SAME object's real fields through the
other door, then runs the identical normalization (defaults for
what is genuinely absent). Nothing is fabricated: the record
either comes from ``to_dict()`` or from the attributes, never
from thin air. Pinned with a raising ``to_dict`` plus working
``to_dict`` and dict controls.

Non-recording: this module keeps no failure ledger for these
paths; the pins assert behavior (and, for 122, the exact debris
cost) rather than an absence in a ledger that does not exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from nomorals.autonomy import unified as un


# ── 122: cleanup failure never masks the real write error ────────

def test_122_cleanup_failure_never_masks_write_error(tmp_path):
    target = tmp_path / "q.json"
    real_unlink = Path.unlink

    def _bad_replace(*args, **kwargs):
        raise OSError("replace boom")

    def _bad_unlink(self, *args, **kwargs):
        if ".tmp." in self.name:
            raise OSError("unlink boom")
        return real_unlink(self, *args, **kwargs)

    with mock.patch.object(os, "replace", _bad_replace), \
            mock.patch.object(Path, "unlink", _bad_unlink):
        with pytest.raises(OSError) as err:
            un.atomic_write_json(target, {"b": 2})
    # the caller sees the WRITE error, not the cleanup error
    assert "replace boom" in str(err.value)
    # the honest cost: the temp file is stranded on disk
    debris = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert len(debris) == 1


def test_122_control_failed_write_with_healthy_cleanup(tmp_path):
    target = tmp_path / "q.json"

    def _bad_replace(*args, **kwargs):
        raise OSError("replace boom")

    with mock.patch.object(os, "replace", _bad_replace):
        with pytest.raises(OSError) as err:
            un.atomic_write_json(target, {"b": 2})
    assert "replace boom" in str(err.value)
    assert [p.name for p in tmp_path.iterdir()] == []  # temp removed


def test_122_control_clean_write_leaves_no_debris(tmp_path):
    target = tmp_path / "q.json"
    un.atomic_write_json(target, {"c": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"c": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["q.json"]


# ── 130: directory fsync is best-effort, payload already durable ─

def test_130_dir_fsync_failure_leaves_payload_intact(tmp_path):
    target = tmp_path / "q.json"

    def _bad_open(*args, **kwargs):
        raise OSError("no dir fsync on this filesystem")

    with mock.patch.object(os, "open", _bad_open):
        assert un.atomic_write_json(target, {"a": 1}) is None
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


# ── 143: to_dict failure falls back to the object's attributes ───

class _BadDoc:
    id = "m1"
    goal = "g1"
    state = {"plan": [{"name": "s", "handler": "h"}]}

    def to_dict(self):
        raise RuntimeError("to_dict boom")


class _GoodDoc:
    def to_dict(self):
        return {"id": "m2", "goal": "g2", "plan": ["step1"]}


def test_143_to_dict_failure_uses_attributes():
    rec = un.normalize_mission(_BadDoc(), now=1000.0)
    assert rec["id"] == "m1"
    assert rec["goal"] == "g1"
    assert rec["plan"] == [{"name": "s", "handler": "h"}]
    assert rec["status"] == "pending"


def test_143_to_dict_control_uses_dict_form():
    rec = un.normalize_mission(_GoodDoc(), now=1000.0)
    assert rec["id"] == "m2"
    assert rec["goal"] == "g2"
    assert rec["plan"] == [{"name": "step1", "handler": "step1"}]


def test_143_dict_control():
    rec = un.normalize_mission({"id": "m3", "goal": "g3"}, now=1000.0)
    assert rec["id"] == "m3"
    assert rec["goal"] == "g3"
    assert rec["plan"] == []
