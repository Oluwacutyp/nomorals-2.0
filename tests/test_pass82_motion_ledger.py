"""Pass #82 — the motion-studio ledger stops eating history.

Pre-fix behaviour (proven by this suite against the old code): the
ledger file is rewritten as ``ledger[-500:]`` on every record, so entry
#1 silently vanishes the moment entry #501 lands — the store IS the
recent window and there is no archive, no count, no notice.

Post-fix contract:
- every recorded entry is preserved in an append-only archive
  (JSONL, the download-kit IntegrityLedger pattern) next to the live
  window file;
- ``read_ledger()`` returns FULL history; ``read_ledger(limit=N)``
  returns the newest N — the window is a read choice, not storage;
- ``ledger_status()`` reports the truth: total, active (in the live
  window), archived (rotated out of the window), oldest entry;
- a legacy ledger file (old shape, at the cap) migrates losslessly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomorals.media.motion_studio import _core
from nomorals.media.motion_studio._core import read_ledger, record_ledger

WINDOW = 500


@pytest.fixture()
def ledger_home(tmp_path, monkeypatch):
    """Point the ledger at a scratch dir (archive/meta derive from it)."""
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(_core, "LEDGER_PATH", path)
    monkeypatch.setattr(_core, "MEDIA_ROOT", tmp_path)
    return path


def _write(n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        record_ledger({"kind": "test", "path": f"/renders/r{i:05}.mp4",
                       "seq": i})


def _seqs(entries) -> list[int]:
    return [e["seq"] for e in entries]


# ---------------------------------------------------------------------------
# (a) beyond the cap: everything survives; the window is a read choice
# ---------------------------------------------------------------------------

def test_beyond_cap_full_history_survives(ledger_home):
    _write(WINDOW + 25)
    full = read_ledger()
    assert len(full) == WINDOW + 25
    assert _seqs(full) == list(range(WINDOW + 25))  # nothing dropped, in order


def test_window_read_returns_newest_in_order(ledger_home):
    _write(WINDOW + 25)
    recent = read_ledger(limit=20)
    assert _seqs(recent) == list(range(WINDOW + 5, WINDOW + 25))
    one = read_ledger(limit=1)
    assert _seqs(one) == [WINDOW + 24]


def test_live_window_file_stays_bounded(ledger_home):
    _write(WINDOW + 25)
    on_disk = json.loads(Path(ledger_home).read_text())
    assert len(on_disk) == WINDOW                 # compat window, same shape
    assert on_disk[-1]["seq"] == WINDOW + 24       # newest present


# ---------------------------------------------------------------------------
# (b) status tells the whole truth
# ---------------------------------------------------------------------------

def test_status_at_zero_cap_and_beyond(ledger_home):
    from nomorals.media.motion_studio._core import ledger_status

    st = ledger_status()
    assert st["total_entries"] == 0
    assert st["archived_entries"] == 0
    assert st["oldest_entry"] is None

    _write(WINDOW)
    st = ledger_status()
    assert st["total_entries"] == WINDOW
    assert st["active_entries"] == WINDOW
    assert st["archived_entries"] == 0

    _write(25, start=WINDOW)
    st = ledger_status()
    assert st["total_entries"] == WINDOW + 25
    assert st["active_entries"] == WINDOW
    assert st["archived_entries"] == 25
    assert st["oldest_entry"]["seq"] == 0
    assert st["oldest_at"] == st["oldest_entry"]["ts"]  # stamped at record
    assert st["window_size"] == WINDOW
    assert st["rotated_out_total"] == 25


# ---------------------------------------------------------------------------
# (c) persistence across reopen — no dupes, no loss
# ---------------------------------------------------------------------------

def test_reopen_keeps_everything_no_dupes(ledger_home):
    _write(WINDOW + 10)
    first = read_ledger()
    second = read_ledger()  # stateless module: re-reading IS reopening
    assert _seqs(first) == _seqs(second) == list(range(WINDOW + 10))
    assert len({e["seq"] for e in second}) == WINDOW + 10  # no dupes
    archive = ledger_home.with_name("ledger.archive.jsonl")
    lines = [ln for ln in archive.read_text().splitlines() if ln.strip()]
    assert len(lines) == WINDOW + 10
    _write(5, start=WINDOW + 10)  # recording after reopen continues cleanly
    assert _seqs(read_ledger()) == list(range(WINDOW + 15))


# ---------------------------------------------------------------------------
# (d) legacy file at the cap migrates whole
# ---------------------------------------------------------------------------

def test_legacy_ledger_at_cap_migrates_lossless(ledger_home):
    legacy = [{"kind": "kenburns", "path": f"/old/r{i:05}.mp4", "seq": i}
              for i in range(WINDOW)]  # old shape: no ts, no archive yet
    Path(ledger_home).write_text(json.dumps(legacy, indent=1))

    from nomorals.media.motion_studio._core import ledger_status

    full = read_ledger()
    assert _seqs(full) == list(range(WINDOW))     # all 500 intact
    st = ledger_status()
    assert st["total_entries"] == WINDOW
    assert st["migrated_from_legacy"] is True

    record_ledger({"kind": "test", "path": "/new/one.mp4", "seq": WINDOW})
    full = read_ledger()
    assert len(full) == WINDOW + 1                  # pre-fix: still 500, seq 0 gone
    assert full[0]["seq"] == 0
    assert full[-1]["seq"] == WINDOW


# ---------------------------------------------------------------------------
# (e) nothing dropped at any size — 499 / 500 / 501
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [499, 500, 501])
def test_no_entry_dropped_at_boundary_sizes(ledger_home, n):
    _write(n)
    full = read_ledger()
    assert len(full) == n
    assert _seqs(full) == list(range(n))
    assert full[0]["seq"] == 0


# ---------------------------------------------------------------------------
# consumer surface: `nm video list` shows totals, not just the window
# ---------------------------------------------------------------------------

def test_cli_list_reports_totals(ledger_home, capsys):
    import argparse

    from nomorals.cmdline.commands.video import _cmd_list

    _write(WINDOW + 7)
    rc = _cmd_list(argparse.Namespace(json=False), None)
    out = capsys.readouterr().out
    assert rc == 0
    assert str(WINDOW + 7) in out                  # the true total is surfaced
    assert f"r{WINDOW + 6:05}" in out              # newest render listed
