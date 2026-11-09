"""WS1-P1 — the Archivist is THE archive organ, wired to the brain.

Before this pass, ``nomorals/archives.py``'s ``register()`` had ZERO
call sites: the brain got only the zip-only ``tools/archive`` module,
while the full engine (magic-byte identification, tar family, 7z/rar,
digest) sat on the shelf behind the typed ``/zip`` command.  The two
organs also disagreed on safety: the Archivist skips + reports bad
entries; ``tools/archive`` rejected the whole archive.

This suite pins the unified state:

(a) the production registry (built the way production builds it)
    carries the ``archive`` tool, backed by ``nomorals.archives``;
(b) brain-path create → list → extract round-trips on zip AND tar.gz
    through ``registry.call``;
(c) the chat send-fusion callables (``zip_send`` / ``unzip_send``)
    still work end to end, riding the Archivist underneath;
(d) traversal / bomb guards hold on the brain path, with the
    skip-report ALWAYS surfaced and a whole-archive refusal only
    when nothing could be extracted;
(e) ``digest`` works through the tool surface.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.archives import Archivist
from nomorals.core.errors import ToolError
from nomorals.core.policy import CapabilitySet
from nomorals.tools import archive as A
from nomorals.tools.registry import ToolRegistry


def _context(tmp_path: Path, **over: Any) -> SimpleNamespace:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        extras={},
        router=None,
    )
    for key, value in over.items():
        setattr(ctx, key, value)
    return ctx


class FakeGateway:
    """Minimal stand-in for the live chat gateway's file-send path."""

    def __init__(self) -> None:
        self.adapters = {"telegram": object(), "whatsapp": object()}
        self.sent: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat_id: str, path: str,
                  caption: str = "", kind: str = "",
                  view_once: bool = False) -> SimpleNamespace:
        self.sent.append({"platform": platform, "chat": chat_id,
                          "path": path, "caption": caption})
        return SimpleNamespace(ok=True, message_id="m-ws1")


def _tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("alpha")
    (root / "sub" / "b.txt").write_text("beta")


def _registry(ctx: Any) -> ToolRegistry:
    return ToolRegistry(ctx).register_builtins()


def _call(reg: ToolRegistry, name: str, **kwargs: Any) -> Any:
    out = reg.call(name, capabilities=CapabilitySet.all(), **kwargs)
    assert out.ok, f"{name} failed: {out.error}"
    return out.unwrap()


# ── (a) the production registry carries the Archivist ───────────────────


def test_production_registry_has_archive_tool_backed_by_archivist(
        tmp_path: Path) -> None:
    reg = _registry(_context(tmp_path))
    spec = reg.get("archive")
    assert spec is not None, "the `archive` tool is not registered"
    # Backed by the Archivist organ itself, not a re-implementation:
    # the registered callable is the closure archives.register made.
    assert spec.fn.__module__ == "nomorals.archives"
    failed = {f["module"] for f in reg.failed_modules()}
    assert "archives" not in failed
    # The full verb set is on the description the brain reads.
    for verb in ("info", "list", "extract", "digest", "create",
                 "compress"):
        assert verb in spec.description
    # The send-fusion surface survives alongside it.
    for name in ("zip_create", "zip_extract", "zip_send", "unzip_send"):
        assert reg.get(name) is not None, f"{name} lost in unification"


# ── (b) brain-path round-trips: zip and tar.gz ──────────────────────────


def test_brain_round_trip_zip(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _tree(tmp_path / "docs")
    reg = _registry(ctx)
    created = _call(reg, "archive", action="create", path="docs",
                    dest="bundle.zip")
    assert created["format"] == "zip"
    assert created["entries"] == 2
    listing = _call(reg, "archive", action="list", path="bundle.zip")
    assert {e["name"] for e in listing["entries"]} == {
        "docs/a.txt", "docs/sub/b.txt"}
    out = _call(reg, "archive", action="extract", path="bundle.zip",
                dest="unzipped")
    assert out["extracted"] == 2
    assert out["skipped"] == []
    assert (tmp_path / "unzipped" / "docs" / "a.txt").read_text() == "alpha"
    assert (tmp_path / "unzipped" / "docs" / "sub" / "b.txt").read_text() \
        == "beta"


def test_brain_round_trip_tar_gz(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _tree(tmp_path / "docs")
    reg = _registry(ctx)
    created = _call(reg, "archive", action="create", path="docs",
                    dest="bundle.tar.gz", fmt="tar.gz")
    assert created["format"] == "tar.gz"
    assert created["entries"] >= 2  # dir entries ride along; files = 2
    out = _call(reg, "archive", action="extract", path="bundle.tar.gz",
                dest="unzipped")
    assert out["format"] == "tar.gz"
    assert out["extracted"] == 2
    assert (tmp_path / "unzipped" / "docs" / "a.txt").read_text() == "alpha"


def test_brain_info_identifies_by_magic_not_extension(
        tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _tree(tmp_path / "docs")
    reg = _registry(ctx)
    _call(reg, "archive", action="create", path="docs",
          dest="real.tar.gz", fmt="tar.gz")
    (tmp_path / "fake.zip").write_bytes(
        (tmp_path / "real.tar.gz").read_bytes())
    info = _call(reg, "archive", action="info", path="fake.zip")
    assert info["format"] == "tar.gz"  # magic bytes win over the name


# ── (c) send-fusion callables ride the Archivist ────────────────────────


def test_zip_send_end_to_end_and_workspace_relative_names(
        tmp_path: Path) -> None:
    gw = FakeGateway()
    ctx = _context(tmp_path, extras={"gateway": gw})
    _tree(tmp_path / "docs")
    out = A.zip_send(ctx, ["docs"], "telegram", "123", name="bundle")
    assert out["ok"] and out["sent"]
    assert len(gw.sent) == 1
    assert gw.sent[0]["path"].endswith("bundle.zip")
    # The archive was written by the Archivist with this surface's
    # workspace-relative member names, exactly as before.
    with zipfile.ZipFile(out["path"]) as zf:
        assert sorted(zf.namelist()) == ["docs/a.txt", "docs/sub/b.txt"]


def test_unzip_send_reports_unsafe_skips(tmp_path: Path) -> None:
    gw = FakeGateway()
    ctx = _context(tmp_path, extras={"gateway": gw})
    mixed = tmp_path / "mixed.zip"
    with zipfile.ZipFile(mixed, "w") as zf:
        zf.writestr("good.txt", "hello")
        zf.writestr("../escape.txt", "pwned")
    out = A.unzip_send(ctx, "mixed.zip", "telegram", "123")
    assert out["ok"]
    assert [s["path"] for s in out["sent"]] == ["good.txt"]
    assert out["unsafe_skipped"] == ["../escape.txt"]
    assert out["unsafe_skipped_count"] == 1
    assert not (tmp_path / "escape.txt").exists()


# ── (d) guards on the brain path: skip + report, refusal last ──────────


def _mixed_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("good.txt", "hello")
        zf.writestr("../escape.txt", "pwned")
        zf.writestr("..\\win.txt", "pwned")


def test_brain_extract_skips_traversal_and_reports(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _mixed_zip(tmp_path / "mixed.zip")
    reg = _registry(ctx)
    out = _call(reg, "archive", action="extract", path="mixed.zip",
                dest="mx")
    assert out["extracted"] == 1
    assert out["skipped_count"] == 2
    assert "../escape.txt" in out["skipped"]
    assert "..\\win.txt" in out["skipped"]
    assert (tmp_path / "mx" / "good.txt").read_text() == "hello"
    # Nothing escaped the destination — on either separator shape,
    # and no backslash-named file was left inside it either.
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "win.txt").exists()
    assert [p.name for p in (tmp_path / "mx").iterdir()] == ["good.txt"]


def test_zip_extract_adapter_same_unified_semantics(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _mixed_zip(tmp_path / "mixed.zip")
    out = A.zip_extract(ctx, "mixed.zip")
    assert out["files"] == 1
    assert out["skipped_count"] == 2
    assert (Path(out["destination"]) / "good.txt").read_text() == "hello"


def test_all_unsafe_archive_is_refused_on_both_surfaces(
        tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
        zf.writestr("/absolute.txt", "pwned")
    with pytest.raises(ToolError) as ei:
        A.zip_extract(ctx, "evil.zip")
    assert "zip-slip" in str(ei.value)
    reg = _registry(ctx)
    outcome = reg.call("archive", action="extract", path="evil.zip",
                       capabilities=CapabilitySet.all())
    assert not outcome.ok  # a refusal, not a silent empty success
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "absolute.txt").exists()


def test_bomb_caps_refuse_on_the_brain_path(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as zf:
        for i in range(3):
            zf.writestr(f"f{i}.txt", "x")
    with mock.patch.object(A, "MAX_EXTRACT_FILES", 2), \
            pytest.raises(ToolError) as ei:
        A.zip_extract(ctx, "many.zip")
    assert "zip-bomb" in str(ei.value)
    # The engine's own live budget (the one the tool path inherits)
    # refuses mid-stream when a cap is tighter than the payload.
    big = tmp_path / "big.zip"
    with zipfile.ZipFile(big, "w") as zf:
        zf.writestr("blob.bin", "x" * 500)
    with pytest.raises(ToolError) as ei2:
        Archivist(ctx).extract("big.zip", dest="big-out", max_bytes=100)
    assert "zip-bomb" in str(ei2.value)


# ── (e) digest through the tool surface ─────────────────────────────────


def test_brain_digest_curates_documents(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = tmp_path / "notes.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("notes.md", "# Hello\nsome curated text")
        zf.writestr("data.bin", "\x00\x01\x02")
    reg = _registry(ctx)
    out = _call(reg, "archive", action="digest", path="notes.zip")
    assert out["format"] == "zip"
    assert out["text_files"] >= 1
    assert "notes.md" in out["sample"]
