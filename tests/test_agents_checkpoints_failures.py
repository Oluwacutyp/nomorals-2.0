"""Tail-83 pins for nomorals/agents/checkpoints.py — 5 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving
the except path was taken (trigger == fallback AND control !=
fallback). The file itself contains no exception handlers, so it adds
zero census sites.

checkpoints.py calls no failures-ledger record function anywhere
(grep-proven: it logs via core.logging_setup.get_logger only), so
NOT-recording is pinned with the snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Verdict notes (full grounds in SURVEY.md):
- 60: _json_safe's pass is the pass-through of a coercion ladder —
  after the serializability test fails, dict/list values are coerced
  recursively and anything else becomes str(value). Best-effort by
  docstring; no failure value is returned as if it were the input.
- 154: verify()'s git-check failure is RECORDED in the returned
  problems list ("could not check X commit") and flips ok to False —
  the module's "a saved checkpoint you can't restore is a lie"
  contract operating as designed. Fail-closed and loud, not a swallow.
- 205: diff() failure returns an explicit {"error": "diff failed"}
  marker, distinguishable from every real diff (which carries the
  scope/untracked lists and no "error" key). Docstring: never raises.
- 389: _read_index's guarded read is rung 1 of a recovery ladder — a
  corrupt/missing index rebuilds from the metadata files on disk.
  Only a failed rebuild yields [] (the separate 395 guard), consumed
  as absence, never as a claim about a specific checkpoint.
- 591: rewind_to's HEAD-check guard covers only the advisory
  "HEAD moved" warning; the restore runs identically either way and
  reports its real counts. Suppressing an advisory whose evidence is
  unavailable changes no outcome.
"""

import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import checkpoints as ck
from nomorals.agents import failures as ag_fail


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _ckpt(**kw):
    base = dict(id="x", ts=0.0, label="", head_hash="",
                stash_hash=None, workdir="")
    base.update(kw)
    return ck.Checkpoint(**base)


def test_60_json_safe_falls_through_to_coercion():
    marker = object()
    before = _snap()
    out = ck._json_safe({"a": marker, "b": [1, marker]})
    assert out == {"a": str(marker), "b": [1, str(marker)]}
    assert _snap() == before
    plain = {"a": 1, "b": ["x", True]}
    assert ck._json_safe(plain) == plain


def test_154_verify_git_failure_records_problem(tmp_path):
    ckpt = _ckpt(id="ckpt_t", workdir=str(tmp_path), code_captured=True,
                 head_hash="a" * 40, stash_hash="b" * 40)

    def raising_git(args, cwd, timeout=60):
        raise OSError("git gone")

    def ok_git(args, cwd, timeout=60):
        return subprocess.CompletedProcess(args, 0, "", "")

    def make_store(name, git_impl):
        store = ck.CheckpointStore(base_dir=tmp_path / name)
        store.base.mkdir(parents=True, exist_ok=True)
        (store.base / "ckpt_t.json").write_text("{}", encoding="utf-8")
        store._git = staticmethod(git_impl)
        store._is_repo = lambda workdir: True
        return store

    before = _snap()
    out = ckpt.verify(make_store("s1", raising_git))
    assert out["ok"] is False
    assert any("could not check HEAD commit" in p for p in out["problems"])
    assert any("could not check stash commit" in p for p in out["problems"])
    assert _snap() == before
    out2 = ckpt.verify(make_store("s2", ok_git))
    assert out2 == {"ok": True, "problems": []}


def test_205_diff_failure_returns_error_marker():
    a = _ckpt(id="a")
    before = _snap()
    out = a.diff(SimpleNamespace(id="b"))  # no scope_files -> guard
    assert out["error"] == "diff failed"
    assert out["from"] == "a" and out["to"] == "b"
    assert _snap() == before
    b = _ckpt(id="b", ts=1.0, head_hash="h1", scope_files=["x.py"],
              untracked_files=["u.txt"])
    real = a.diff(b)
    assert "error" not in real
    assert real["scope_added"] == ["x.py"]
    assert real["head_moved"] is True


def test_389_corrupt_index_rebuilds_from_meta_files(tmp_path):
    store = ck.CheckpointStore(base_dir=tmp_path)
    store.base.mkdir(parents=True, exist_ok=True)
    (tmp_path / "index.json").write_text("{corrupt", encoding="utf-8")
    (tmp_path / "ckpt_20260101_000000_ab.json").write_text(
        "{}", encoding="utf-8")
    (tmp_path / "ckpt_20260102_000000_cd.json").write_text(
        "{}", encoding="utf-8")
    before = _snap()
    ids = store._read_index()
    assert ids == ["ckpt_20260101_000000_ab",
                   "ckpt_20260102_000000_cd"]
    assert _snap() == before
    (tmp_path / "index.json").write_text('["ckpt_x"]', encoding="utf-8")
    assert store._read_index() == ["ckpt_x"]


def test_591_rewind_head_check_failure_skips_warning_only(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir(parents=True, exist_ok=True)
    store = ck.CheckpointStore(base_dir=tmp_path / "store")
    files_dir = store._files_dir("ckpt_r")
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "new.txt").write_bytes(b"captured")
    ckpt = _ckpt(id="ckpt_r", workdir=str(workdir), code_captured=True,
                 head_hash="a" * 40, stash_hash="c" * 40,
                 scope_files=["tracked.py"], untracked_files=["new.txt"])

    def dispatch(head_impl):
        def _git(args, cwd, timeout=60):
            if args[:2] == ["rev-parse", "--is-inside-work-tree"]:
                return subprocess.CompletedProcess(args, 0, "true", "")
            if args[:2] == ["rev-parse", "HEAD"]:
                return head_impl(args)
            if args and args[0] == "checkout":
                return subprocess.CompletedProcess(args, 0, "", "")
            if args and args[0] == "status":
                return subprocess.CompletedProcess(args, 0, "", "")
            return subprocess.CompletedProcess(args, 0, "", "")
        return _git

    def raising_head(args):
        raise OSError("git crashed")

    before = _snap()
    # _is_repo reads the CLASS-level _git, so patch the class.
    with patch.object(ck.CheckpointStore, "_git",
                      staticmethod(dispatch(raising_head))):
        out = store.rewind_to(ckpt)
    assert "HEAD moved" not in out
    assert "restored 1 tracked file(s), 1 untracked file(s)" in out
    assert (workdir / "new.txt").read_bytes() == b"captured"
    assert _snap() == before

    def moved_head(args):
        return subprocess.CompletedProcess(args, 0, "d" * 40 + "\n", "")

    with patch.object(ck.CheckpointStore, "_git",
                      staticmethod(dispatch(moved_head))):
        out2 = store.rewind_to(ckpt)
    assert "HEAD moved since this checkpoint" in out2
    shutil.rmtree(tmp_path / "store", ignore_errors=True)
