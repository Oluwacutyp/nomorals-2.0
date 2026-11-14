"""Pass #216 pins: social/chat/output_router.py — 2 exclusions, 0 converts.

- 253: ``store_text`` disk write ``except OSError: pass``. The
  disk write is a persistence optimization; on failure the
  in-memory map still serves this process (the comment documents
  it: "the in-memory map below still serves this process"). The
  content is not lost — it is in memory, and the manifest is
  returned.
- 399: ``_record`` journal write ``except OSError: pass``. The
  journal is observability; on failure the delivery proceeds (the
  comment documents it: "journaling never breaks a delivery").
  The delivery is not affected.

social/chat/output_router.py calls no failures-ledger record
function anywhere, so NOT-recording is pinned with the agency
snapshot method: agency recent_failures() before/after each
trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Triggers use a failing disk
write and a failing journal open, instead of swallowing.
"""

from pathlib import Path
from unittest.mock import patch

from nomorals.agents import failures as ag_fail
from nomorals.social.chat import output_router as orr


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L253: disk write failure still serves from memory ─────────────

def test_253_store_text_returns_manifest_and_serves(tmp_path):
    # A normal store_text returns the manifest and the text is
    # retrievable via text_for.
    before = _snap()
    router = orr.OutputRouter(tmp_path / "store")
    text = "hello, this is a test document " * 10
    manifest = router.store_text(text)
    assert manifest.document_id
    assert router.text_for(manifest.document_id) == text
    assert _snap() == before


def test_253_disk_write_failure_still_serves_from_memory(tmp_path):
    # When the disk write itself raises OSError, store_text still
    # returns the manifest and the in-memory map serves the text.
    before = _snap()
    router = orr.OutputRouter(tmp_path / "store")
    text = "hello, this is a test document " * 10
    with patch.object(Path, "write_bytes",
                      side_effect=OSError(13, "Permission denied")):
        manifest = router.store_text(text)
    assert manifest.document_id
    assert router.text_for(manifest.document_id) == text
    assert _snap() == before


# ── L399: journal write failure never breaks a delivery ───────────

def test_399_journal_write_failure_does_not_raise(tmp_path):
    # A route journal that cannot be written does not raise; the
    # record is simply not persisted.
    before = _snap()
    router = orr.OutputRouter(tmp_path / "routes")
    decision = orr.RouteDecision(
        kind="send", reason="test", chars=10, chunks=1,
        title="t", document_id="d1")
    with patch.object(Path, "open",
                      side_effect=OSError(13, "Permission denied")):
        router._record(decision, platform="local", reader_key="r1")
    assert _snap() == before


def test_399_journal_write_persists_row(tmp_path):
    # Control: a working journal write persists the row.
    before = _snap()
    router = orr.OutputRouter(tmp_path / "routes")
    decision = orr.RouteDecision(
        kind="send", reason="test", chars=10, chunks=1,
        title="t", document_id="d1")
    router._record(decision, platform="local", reader_key="r1")
    rows = (tmp_path / "routes" / "routes.jsonl").read_text(
        encoding="utf-8")
    assert '"document_id": "d1"' in rows
    assert _snap() == before
