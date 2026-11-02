"""Tail-99 pins: nomorals/books/branches.py census sites.

The what-if branch surface is read-mostly and per-item by
design: one story's settings, one branch's file, one branch's
choices may each fail without taking the whole surface down,
and every failure value is an absence — a fallback root, a None
overlap, a skipped listing entry, an empty child list — never
an invented fact. All four sites held as exclusions:

35 — ``_books_root()``: ``Settings()`` crash -> the documented
fallback ``<cwd>/workspace/books/fiction``. The docstring
prints the contract ("settings unavailable (bare import)");
this is the same resolution FictionWriter itself performs,
degrading to the conventional workspace when no settings can
be built.
255 — ``StoryBranch.diff()``: a branch chapter file that
cannot be read (OSError) -> ``branch_text`` stays "" and the
row reports ``text_overlap: None``. The comparison never
fabricates an overlap number for unread text; canon words are
still counted from the canon file that did read. (Flagged-scan
remainder: the narrow OSError handler carries no noqa and
stays visible in the flagged census — accepted.)
307 — ``list_branches()``: a branch whose branch.json cannot
be parsed (or lacks the summary keys) is skipped; the healthy
branches still list. The skipped branch's data is untouched
on disk — an omission in a listing, not a claim about it.
(Flagged-scan remainder, same as 255 — accepted.)
344 — ``tree()``: a branch whose choices cannot be expanded
(here: a choice entry with no ``id``) keeps its branch node
but gains no choice children. The tree is a rendering;
a degraded limb is drawn bare rather than invented.

NOT-recording is pinned with the snapshot method: branches.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.books.branches as BR
import nomorals.core.config as cfg


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _ctx(root: Path):
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(root)))


# ── 35: Settings crash -> cwd workspace fallback ────────────────

def test_35_settings_crash_falls_back_to_cwd_workspace(tmp_path):
    before = _snap()
    with mock.patch.object(cfg, "Settings", _raise):
        assert BR._books_root() == (
            Path.cwd() / "workspace" / "books" / "fiction")

    class _Settings:
        def __init__(self):
            self.workspace_dir = tmp_path

    # control: resolvable settings win over the fallback
    with mock.patch.object(cfg, "Settings", _Settings):
        assert BR._books_root() == tmp_path / "books" / "fiction"
    assert _snap() == before


# ── 255: unreadable branch chapter -> overlap None ──────────────

def _story_with_branch(tmp_path: Path) -> BR.StoryBranch:
    story = tmp_path / "books" / "fiction" / "tale"
    (story / "chapters").mkdir(parents=True)
    (story / "chapters" / "0001.md").write_text(
        "alpha beta gamma delta epsilon")
    br = BR.StoryBranch("tale", "whatif", context=_ctx(tmp_path))
    assert br.fork(0, "what if")["ok"] is True
    meta = json.loads((br._dir / "branch.json").read_text())
    meta["chapters"] = [{"n": 1, "title": "B1", "words": 5}]
    (br._dir / "branch.json").write_text(json.dumps(meta))
    return br


def test_255_missing_branch_chapter_reports_no_overlap(tmp_path):
    before = _snap()
    br = _story_with_branch(tmp_path)
    row = br.diff()["chapters"][0]
    assert row["canon_words"] == 5
    assert row["text_overlap"] is None
    assert row["diverged"] is False
    # control: the chapter file present with canon's text -> 1.0
    (br._dir / "chapter_001.md").write_text(
        "alpha beta gamma delta epsilon")
    row = br.diff()["chapters"][0]
    assert row["text_overlap"] == 1.0
    assert row["diverged"] is False
    assert _snap() == before


# ── 307: corrupt branch.json -> branch skipped in listing ───────

def test_307_corrupt_branch_meta_is_skipped(tmp_path):
    before = _snap()
    bdir = tmp_path / "books" / "fiction" / "tale" / "branches"
    (bdir / "good").mkdir(parents=True)
    (bdir / "good" / "branch.json").write_text(json.dumps({
        "branch": "good", "forked_from_chapter": 2, "premise": "p",
        "chapters": [], "status": "active"}))
    (bdir / "bad").mkdir(parents=True)
    (bdir / "bad" / "branch.json").write_text("{corrupt")
    names = [b["branch"] for b in
             BR.list_branches("tale", context=_ctx(tmp_path))]
    assert names == ["good"]
    # the skipped branch's file is untouched by the listing
    assert (bdir / "bad" / "branch.json").read_text() == "{corrupt"
    # control: repaired meta -> the branch lists again
    (bdir / "bad" / "branch.json").write_text(json.dumps({
        "branch": "bad", "forked_from_chapter": 1, "premise": "p",
        "chapters": [], "status": "active"}))
    names = [b["branch"] for b in
             BR.list_branches("tale", context=_ctx(tmp_path))]
    assert names == ["bad", "good"]
    assert _snap() == before


# ── 344: unexpandable choices -> bare branch node in tree ───────

def test_344_choice_without_id_renders_bare_node(tmp_path):
    before = _snap()
    bdir = tmp_path / "books" / "fiction" / "tale" / "branches" / "b1"
    bdir.mkdir(parents=True)
    meta = {"branch": "b1", "forked_from_chapter": 1, "premise": "p",
            "chapters": [], "status": "active",
            "choices": [{"prompt": "What now?", "options": ["a", "b"]}]}
    (bdir / "branch.json").write_text(json.dumps(meta))
    node = BR.tree("tale", context=_ctx(tmp_path))["children"][0]
    assert node["name"].endswith("b1")
    assert node["children"] == []
    # control: a well-formed choice renders as a child node
    meta["choices"] = [{"id": 1, "prompt": "What now?",
                         "options": ["a", "b"]}]
    (bdir / "branch.json").write_text(json.dumps(meta))
    node = BR.tree("tale", context=_ctx(tmp_path))["children"][0]
    assert len(node["children"]) == 1
    assert "choice #1" in node["children"][0]["name"]
    assert _snap() == before
