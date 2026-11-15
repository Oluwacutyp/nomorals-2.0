"""Pass #217 pins: media/typography/catalogue.py — 1 exclusion, 0 converts.

- 146: ``_load_index`` read/parse failure. The font index is a
  cache: a missing file (first run — no index yet), an unreadable
  file, or a corrupt JSON payload all return ``{}`` and the caller
  rebuilds the index from the font directories. The swallow is the
  correct behavior: a stale or absent cache is not a failure, and
  the function's contract already produces ``{}`` for "no index"
  on a non-error path.

media/typography/catalogue.py calls no failures-ledger record
function anywhere, so NOT-recording is pinned with the agency
snapshot method: agency recent_failures() before/after each trigger
must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use real files in a temp
dir, never a swallow.
"""

import json

from nomorals.agents import failures as ag_fail
from nomorals.media.typography.catalogue import _load_index


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L146: a missing / unreadable / corrupt index degrades to {} ─────

def test_146_missing_index_file_returns_empty(tmp_path):
    before = _snap()
    assert _load_index(tmp_path / "no-such-index.json") == {}
    assert _snap() == before


def test_146_corrupt_index_returns_empty(tmp_path):
    p = tmp_path / "index.json"
    p.write_text("{not json at all", encoding="utf-8")
    before = _snap()
    assert _load_index(p) == {}
    assert _snap() == before


def test_146_wrong_version_returns_empty(tmp_path):
    # A version-stamped payload from another version is not trusted.
    p = tmp_path / "index.json"
    p.write_text(json.dumps({"version": 999, "files": {"a": 1}}),
                 encoding="utf-8")
    before = _snap()
    assert _load_index(p) == {}
    assert _snap() == before


def test_146_valid_index_returns_the_files(tmp_path):
    files = {"DejaVuSans.ttf": {"family": "DejaVu Sans"}}
    p = tmp_path / "index.json"
    from nomorals.media.typography.catalogue import _INDEX_VERSION
    p.write_text(json.dumps({"version": _INDEX_VERSION, "files": files}),
                 encoding="utf-8")
    before = _snap()
    assert _load_index(p) == files
    assert _snap() == before


def test_146_unreadable_index_returns_empty(tmp_path):
    # A directory where the file should be: read_text raises
    # IsADirectoryError (an OSError), caught the same way.
    d = tmp_path / "subdir"
    d.mkdir()
    before = _snap()
    assert _load_index(d) == {}
    assert _snap() == before
