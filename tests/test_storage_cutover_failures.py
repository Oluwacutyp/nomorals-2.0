"""Pass #218 pins: storage/cutover.py — 1 exclusion, 0 converts.

- 382: ``_env_culprits`` .env read probe. The install's ``.env`` file
  is read to see whether it sets the storage key; when the read raises
  OSError the code falls through and returns the culprits found so far
  (from the process environment, or empty). The read is best-effort:
  an unreadable file is honestly "no evidence from the file", not an
  error — the caller (flip_config's verification) still re-reads the
  effective backend and reports its own fallback when nothing is found.

storage/cutover.py calls no failures-ledger record function anywhere,
so NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a directory in place of
the .env file (read_text raises IsADirectoryError), never a swallow.
"""

from nomorals.agents import failures as ag_fail
from nomorals.storage.cutover import _env_culprits


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L382: the unreadable .env file yields no culprits from the file ───


def test_382_no_env_file_and_no_environ_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("NM_DB_BACKEND", raising=False)
    before = _snap()
    assert _env_culprits(tmp_path, "NM_DB_BACKEND") == []
    assert _snap() == before


def test_382_environ_override_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_DB_BACKEND", "mongo")
    before = _snap()
    culprits = _env_culprits(tmp_path, "NM_DB_BACKEND")
    assert len(culprits) == 1
    assert "NM_DB_BACKEND" in culprits[0]
    assert "process environment" in culprits[0]
    assert _snap() == before


def test_382_env_file_override_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv("NM_DB_BACKEND", raising=False)
    (tmp_path / ".env").write_text("NM_DB_BACKEND=mongo\n",
                                   encoding="utf-8")
    before = _snap()
    culprits = _env_culprits(tmp_path, "NM_DB_BACKEND")
    assert len(culprits) == 1
    assert "NM_DB_BACKEND" in culprits[0]
    assert str(tmp_path / ".env") in culprits[0]
    assert _snap() == before


def test_382_unreadable_env_file_is_swallowed_not_raised(tmp_path,
                                                         monkeypatch):
    # A directory named ".env": exists() is True, read_text raises
    # IsADirectoryError (an OSError) — the probe passes and the
    # function returns the culprits found so far (none here).
    monkeypatch.delenv("NM_DB_BACKEND", raising=False)
    (tmp_path / ".env").mkdir()
    before = _snap()
    assert _env_culprits(tmp_path, "NM_DB_BACKEND") == []
    assert _snap() == before


def test_382_unreadable_env_file_keeps_the_environ_culprit(tmp_path,
                                                           monkeypatch):
    # The file probe fails but the environ culprit still lands —
    # the swallow only drops the file half, never the whole answer.
    monkeypatch.setenv("NM_DB_BACKEND", "mongo")
    (tmp_path / ".env").mkdir()
    before = _snap()
    culprits = _env_culprits(tmp_path, "NM_DB_BACKEND")
    assert len(culprits) == 1
    assert "process environment" in culprits[0]
    assert _snap() == before


def test_382_env_file_without_the_key_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("NM_DB_BACKEND", raising=False)
    (tmp_path / ".env").write_text("OTHER_KEY=value\n", encoding="utf-8")
    before = _snap()
    assert _env_culprits(tmp_path, "NM_DB_BACKEND") == []
    assert _snap() == before
