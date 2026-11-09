"""WS1-P2 — Archivist depth: passwords, nesting, append, honest
partials, external-path budgets, filename encoding.

P1 wired the Archivist as the brain's ``archive`` tool with unified
skip+report semantics.  This suite pins the depth P2 adds on top:

(a) passwords — a protected zip opens with the caller's password
    (API + tool surface + digest); wrong/missing passwords fail with
    NAMED errors, never tracebacks, never silent empties, and the
    password never appears in any result or error;
(b) nesting — archives inside archives open recursively to a depth
    cap, under bomb budgets that apply CUMULATIVELY across levels,
    with the nesting reported member by member;
(c) append/remove — members added to existing zip + tar archives
    (additions and replacements named), honest refusals for formats
    that cannot take members;
(d) honest partials — a corrupt entry no longer aborts the run:
    good entries land, bad ones are named with their reason;
(e) external budgets — the 7z/rar path pre-lists and refuses over
    budget, enforces landed bytes after the fact, and classifies
    password failures by name;
(f) filename encoding — zip names follow the spec (UTF-8 flag else
    cp437 fallback), the fallback is surfaced, undecidable names
    fail by name, and unwritable names are skipped with a reason.

NOTE on (e): this box has NO 7z / unrar binary installed, so the
external paths are pinned with the binary FAKED AT THE SEAM — a
script standing in for the real tool, driven by a manifest file,
exercising the engine's real argument construction, listing parse,
budget checks and error classification.  The stdlib paths (a)–(d),
(f) run against real archives throughout.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nomorals.archives import Archivist
from nomorals.core.errors import ToolError
from nomorals.core.policy import CapabilitySet
from nomorals.tools import archive as A
from nomorals.tools.registry import ToolRegistry

ZIP_CLI = shutil.which("zip")
needs_zip_cli = pytest.mark.skipif(
    ZIP_CLI is None,
    reason="Info-ZIP `zip` CLI not installed (encrypted-zip fixture)")


def _context(tmp_path: Path, **over: Any) -> SimpleNamespace:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        extras={},
        router=None,
    )
    for key, value in over.items():
        setattr(ctx, key, value)
    return ctx


def _registry(ctx: Any) -> ToolRegistry:
    return ToolRegistry(ctx).register_builtins()


def _call(reg: ToolRegistry, name: str, **kwargs: Any) -> Any:
    out = reg.call(name, capabilities=CapabilitySet.all(), **kwargs)
    assert out.ok, f"{name} failed: {out.error}"
    return out.unwrap()


def _archivist(ctx: Any, **bins: str) -> Archivist:
    a = Archivist(ctx)
    for key, value in bins.items():
        setattr(a, key, value)
    return a


# ── (a) passwords ───────────────────────────────────────────────────────


def _make_encrypted_zip(tmp_path: Path, password: str = "pw123",
                        name: str = "enc.zip") -> Path:
    (tmp_path / "secret.txt").write_text("top secret")
    out = subprocess.run(
        [ZIP_CLI, "-P", password, "-j", str(tmp_path / name),
         str(tmp_path / "secret.txt")],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return tmp_path / name


@needs_zip_cli
def test_password_zip_right_wrong_missing(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _make_encrypted_zip(tmp_path)
    a = Archivist(ctx)
    assert a.info("enc.zip")["encrypted"] is True
    with pytest.raises(ToolError) as ei:
        a.extract("enc.zip", dest="o1")
    assert "password required" in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.extract("enc.zip", dest="o2", password="nope")
    assert "password incorrect" in str(ei.value)
    # The wrong password is never echoed back in the error.
    assert "nope" not in str(ei.value)
    out = a.extract("enc.zip", dest="o3", password="pw123")
    assert out["extracted"] == 1
    assert (tmp_path / "o3" / "secret.txt").read_text() == "top secret"
    # …and the right password never lands in a result either.
    assert "pw123" not in json.dumps(out)


@needs_zip_cli
def test_password_zip_through_the_tool_surface(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _make_encrypted_zip(tmp_path)
    reg = _registry(ctx)
    outcome = reg.call("archive", action="extract", path="enc.zip",
                       dest="t1", capabilities=CapabilitySet.all())
    assert not outcome.ok
    assert "password required" in str(outcome.error)
    out = _call(reg, "archive", action="extract", path="enc.zip",
                dest="t2", password="pw123")
    assert out["extracted"] == 1
    # digest rides the same password through to the curated text.
    (tmp_path / "note.md").write_text("# Classified\nread me")
    subprocess.run(
        [ZIP_CLI, "-P", "pw123", "-j", str(tmp_path / "enc.zip"),
         str(tmp_path / "note.md")], check=True, capture_output=True)
    dug = _call(reg, "archive", action="digest", path="enc.zip",
                password="pw123")
    assert dug["text_files"] >= 1
    assert "note.md" in dug["sample"]


@needs_zip_cli
def test_mixed_zip_partial_without_password(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = tmp_path / "mixed.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("plain.txt", "in the clear")
    (tmp_path / "secret.txt").write_text("top secret")
    subprocess.run(
        [ZIP_CLI, "-P", "pw123", "-j", str(bundle),
         str(tmp_path / "secret.txt")], check=True, capture_output=True)
    a = Archivist(ctx)
    out = a.extract("mixed.zip", dest="m1")
    assert out["extracted"] == 1
    assert (tmp_path / "m1" / "plain.txt").read_text() == "in the clear"
    details = {d["name"]: d["reason"] for d in out["skipped_details"]}
    assert details["secret.txt"] == "password required"
    out = a.extract("mixed.zip", dest="m2", password="pw123")
    assert out["extracted"] == 2
    assert (tmp_path / "m2" / "secret.txt").read_text() == "top secret"


# ── (b) nesting ─────────────────────────────────────────────────────────


def _chain(tmp_path: Path, levels: int) -> Path:
    cur = tmp_path / f"L{levels - 1}.zip"
    with zipfile.ZipFile(cur, "w") as zf:
        zf.writestr("hello.txt", "bottom")
    for i in range(levels - 2, -1, -1):
        nxt = tmp_path / f"L{i}.zip"
        with zipfile.ZipFile(nxt, "w") as zf:
            zf.write(cur, cur.name)
        cur = nxt
    return cur


def test_nested_archives_open_recursively(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep.txt", "deep content")
    with zipfile.ZipFile(tmp_path / "mid.zip", "w") as zf:
        zf.write(inner, "inner.zip")
        zf.writestr("mid.txt", "mid")
    a = Archivist(ctx)
    a.create(["mid.zip"], "outer.tar.gz", fmt="tar.gz")
    out = a.extract("outer.tar.gz", dest="nest")
    deep = (tmp_path / "nest" / "mid.zip.extracted"
            / "inner.zip.extracted" / "deep.txt")
    assert deep.read_text() == "deep content"
    assert out["extracted"] == 4  # mid.zip, inner.zip, mid.txt, deep.txt
    by_member = {r["member"]: r for r in out["nested"]}
    assert by_member["mid.zip"]["opened"] is True
    assert by_member["mid.zip"]["format"] == "zip"
    assert by_member["mid.zip.extracted/inner.zip"]["extracted"] == 1


def test_nested_depth_cap_stops_and_reports(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    top = _chain(tmp_path, levels=5)
    a = Archivist(ctx)
    out = a.extract(top.name, dest="chain")
    # Default cap 3: L1..L3 open, the L4 archive is left unopened
    # and REPORTED as capped — hello.txt at the bottom never lands.
    assert "hello.txt" not in " ".join(out["files"])
    capped = [r for r in out["nested"] if not r.get("opened", True)]
    assert len(capped) == 1
    assert capped[0]["member"].endswith("L4.zip")
    assert "depth cap" in capped[0]["reason"]
    # max_depth=0 turns nesting off entirely.
    out = a.extract(top.name, dest="chain0", max_depth=0)
    assert out["extracted"] == 1
    assert "hello.txt" not in " ".join(out["files"])


def test_nested_budgets_are_cumulative(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("x1.txt", "x")
        zf.writestr("x2.txt", "x")
    with zipfile.ZipFile(tmp_path / "outer.zip", "w") as zf:
        for i in range(3):
            zf.write(inner, f"i{i}.zip")
    a = Archivist(ctx)
    # 3 inner archives + 2 files each = 9 landings against a cap of
    # 5: the budget is shared across levels, so the bomb dies even
    # though no single level exceeds it.
    with pytest.raises(ToolError) as ei:
        a.extract("outer.zip", dest="bomb", max_files=5)
    assert "zip-bomb" in str(ei.value)


@needs_zip_cli
def test_nested_password_archive_reports_not_sinks(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _make_encrypted_zip(tmp_path)
    with zipfile.ZipFile(tmp_path / "outer.zip", "w") as zf:
        zf.write(tmp_path / "enc.zip", "enc.zip")
        zf.writestr("plain.txt", "clear")
    a = Archivist(ctx)
    out = a.extract("outer.zip", dest="n1")
    assert (tmp_path / "n1" / "plain.txt").read_text() == "clear"
    rec = {r["member"]: r for r in out["nested"]}["enc.zip"]
    assert "password required" in rec["error"]
    out = a.extract("outer.zip", dest="n2", password="pw123")
    secret = (tmp_path / "n2" / "enc.zip.extracted" / "secret.txt")
    assert secret.read_text() == "top secret"


# ── (c) append / remove ─────────────────────────────────────────────────


def test_append_zip_add_and_replace(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / "c.txt").write_text("gamma")
    a = Archivist(ctx)
    a.create(["a.txt", "b.txt"], "bundle.zip")
    out = a.append("bundle.zip", ["c.txt"])
    assert out["added"] == ["c.txt"] and out["replaced"] == []
    assert out["entries"] == 3
    (tmp_path / "a.txt").write_text("ALPHA-2")
    out = a.append("bundle.zip", ["a.txt"])
    assert out["added"] == [] and out["replaced"] == ["a.txt"]
    assert out["entries"] == 3
    with zipfile.ZipFile(tmp_path / "bundle.zip") as zf:
        assert zf.read("a.txt") == b"ALPHA-2"
        assert sorted(zf.namelist()) == ["a.txt", "b.txt", "c.txt"]


def test_append_tar_rebuild_and_remove(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    a = Archivist(ctx)
    a.create(["a.txt"], "bundle.tar.gz", fmt="tar.gz")
    out = a.append("bundle.tar.gz", ["b.txt"])
    assert out["added"] == ["b.txt"] and out["entries"] == 2
    (tmp_path / "a.txt").write_text("ALPHA-2")
    out = a.append("bundle.tar.gz", ["a.txt"])
    assert out["replaced"] == ["a.txt"] and out["entries"] == 2
    out = a.remove_members("bundle.tar.gz", ["b.txt", "ghost.txt"])
    assert out["removed"] == ["b.txt"] and out["missing"] == ["ghost.txt"]
    assert out["entries"] == 1
    out = a.extract("bundle.tar.gz", dest="check")
    assert (tmp_path / "check" / "a.txt").read_text() == "ALPHA-2"


def test_append_refusals_are_named(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    a = Archivist(ctx)
    a.compress("a.txt")  # → a.txt.gz, a single-file stream
    with pytest.raises(ToolError) as ei:
        a.append("a.txt.gz", ["a.txt"])
    assert "single-file" in str(ei.value)
    (tmp_path / "mystery.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError) as ei:
        a.append("mystery.bin", ["a.txt"])
    assert "not a recognized archive" in str(ei.value)


def test_append_remove_through_the_tool_surface(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    reg = _registry(ctx)
    spec = reg.get("archive")
    assert spec is not None
    for verb in ("append", "remove", "password"):
        assert verb in spec.description
    _call(reg, "archive", action="create", path="a.txt",
          dest="bundle.zip")
    out = _call(reg, "archive", action="append", path="bundle.zip",
                paths="b.txt")
    assert out["added"] == ["b.txt"]
    out = _call(reg, "archive", action="remove", path="bundle.zip",
                paths="b.txt|ghost.txt")
    assert out["removed"] == ["b.txt"]
    assert out["missing"] == ["ghost.txt"]


@needs_zip_cli
def test_append_rebuild_over_encrypted_member_needs_password(
        tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _make_encrypted_zip(tmp_path)  # enc.zip holds secret.txt, encrypted
    a = Archivist(ctx)
    (tmp_path / "new.txt").write_text("new")
    out = a.append("enc.zip", ["new.txt"])  # no collision: fine
    assert out["added"] == ["new.txt"]
    assert "unencrypted" in out["note"]
    # Replacing new.txt forces a rebuild, and a rebuild must COPY
    # the kept encrypted member — copying is what needs the
    # password.  (Replacing the encrypted member itself would not:
    # its bytes are never read.)
    (tmp_path / "new.txt").write_text("new v2")
    with pytest.raises(ToolError) as ei:
        a.append("enc.zip", ["new.txt"])
    assert "password required" in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.append("enc.zip", ["new.txt"], password="nope")
    assert "password incorrect" in str(ei.value)
    out = a.append("enc.zip", ["new.txt"], password="pw123")
    assert out["replaced"] == ["new.txt"]
    out = a.extract("enc.zip", dest="final", password="pw123")
    assert (tmp_path / "final" / "secret.txt").read_text() == \
        "top secret"
    assert (tmp_path / "final" / "new.txt").read_text() == "new v2"


def test_create_with_password_refusals_without_7z(
        tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    a = Archivist(ctx)
    if a._bin7z:
        pytest.skip("7z installed here — refusal path not exercisable")
    with pytest.raises(ToolError) as ei:
        a.create(["a.txt"], "enc.zip", password="pw123")
    assert "7z" in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.create(["a.txt"], "enc.tar.gz", fmt="tar.gz",
                 password="pw123")
    assert "only available for zip" in str(ei.value)


# ── (d) honest partials ─────────────────────────────────────────────────


def _corrupt_crc_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w",
                         compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("good.txt", "good")
        zf.writestr("bad.txt", "BAD-SENTINEL-DATA")
    raw = bytearray(path.read_bytes())
    i = raw.find(b"BAD-SENTINEL-DATA")
    assert i > 0
    raw[i] ^= 0xFF
    path.write_bytes(bytes(raw))


def test_corrupt_zip_entry_partial(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _corrupt_crc_zip(tmp_path / "corr.zip")
    a = Archivist(ctx)
    out = a.extract("corr.zip", dest="c1")
    assert out["extracted"] == 1
    assert (tmp_path / "c1" / "good.txt").read_text() == "good"
    assert out["skipped"] == ["bad.txt"]
    details = {d["name"]: d["reason"] for d in out["skipped_details"]}
    assert "corrupt entry" in details["bad.txt"]
    # The corrupt bytes are NOT left on disk as a "successful" file.
    assert not (tmp_path / "c1" / "bad.txt").exists()


def test_all_corrupt_refusal_names_the_reason(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bad = tmp_path / "only-bad.zip"
    with zipfile.ZipFile(bad, "w",
                         compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("bad.txt", "BAD-SENTINEL-DATA")
    raw = bytearray(bad.read_bytes())
    raw[raw.find(b"BAD-SENTINEL-DATA")] ^= 0xFF
    bad.write_bytes(bytes(raw))
    reg = _registry(ctx)
    outcome = reg.call("archive", action="extract", path="only-bad.zip",
                       dest="x", capabilities=CapabilitySet.all())
    assert not outcome.ok
    assert "corrupt entry" in str(outcome.error)
    # The send-fusion adapter names the corruption too — it must not
    # call a corrupt archive a zip-slip.
    with pytest.raises(ToolError) as ei:
        A.zip_extract(ctx, "only-bad.zip")
    assert "zip-slip" not in str(ei.value)
    assert "corrupt entry" in str(ei.value)


def test_corrupt_tar_header_partial(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    path = tmp_path / "broken.tar"
    with tarfile.open(path, "w") as tf:
        for name, data in [("one.txt", b"1" * 512),
                           ("two.txt", b"2" * 10),
                           ("three.txt", b"3" * 10)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = bytearray(path.read_bytes())
    # Member 2's header block sits at 1024 (header 512 + data 512);
    # mangling a name byte breaks its checksum.  tarfile ends
    # iteration silently at a bad header, so the engine proves the
    # loss from the tail: iteration stopped on a non-zero block
    # instead of the zero end-marker.
    raw[1024 + 5] ^= 0xFF
    path.write_bytes(bytes(raw))
    a = Archivist(ctx)
    out = a.extract("broken.tar", dest="t1")
    assert out["extracted"] == 1
    assert (tmp_path / "t1" / "one.txt").read_bytes() == b"1" * 512
    reasons = " ".join(d["reason"] for d in out["skipped_details"])
    assert "ended early" in reasons
    # A healthy tar of the same shape reports no such loss.
    good = tmp_path / "good.tar"
    with tarfile.open(good, "w") as tf:
        for name, data in [("one.txt", b"1" * 512),
                           ("two.txt", b"2" * 10)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = a.extract("good.tar", dest="t2")
    assert out["extracted"] == 2 and out["skipped_details"] == []


def test_truncated_tar_gz_partial(tmp_path: Path) -> None:
    import random

    ctx = _context(tmp_path)
    path = tmp_path / "cut.tar.gz"
    rng = random.Random(42)
    with tarfile.open(path, "w:gz") as tf:
        for name in ("one.bin", "two.bin", "three.bin"):
            data = rng.randbytes(20_000)  # incompressible: offsets track
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = path.read_bytes()
    path.write_bytes(raw[: int(len(raw) * 0.55)])  # cut inside member 2
    a = Archivist(ctx)
    out = a.extract("cut.tar.gz", dest="c1")
    assert out["extracted"] == 1
    assert (tmp_path / "c1" / "one.bin").stat().st_size == 20_000
    reasons = " ".join(d["reason"] for d in out["skipped_details"])
    assert "partway" in reasons or "corrupt entry" in reasons


def test_unreadable_archive_fails_named_not_raw(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    (tmp_path / "junk.zip").write_bytes(b"PK\x03\x04 not really a zip")
    a = Archivist(ctx)
    with pytest.raises(ToolError) as ei:
        a.extract("junk.zip", dest="j1")
    assert "unreadable zip archive" in str(ei.value)


# ── (e) external paths, binary faked at the seam ────────────────────────

FAKE7Z = r'''
import os, sys
from pathlib import Path

def log(msg):
    logp = os.environ.get("FAKE7Z_LOG")
    if logp:
        with open(logp, "a") as fh:
            fh.write(msg + "\n")

def manifest(path):
    files, pw, extract_error, partial = [], None, None, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("!password "):
            pw = line.split(None, 1)[1]
        elif line.startswith("!extract-error "):
            parts = line.split(None, 2)
            extract_error = (int(parts[1]),
                             parts[2] if len(parts) > 2 else "error")
        elif line.startswith("!partial "):
            partial = line.split(None, 1)[1]
        elif line.startswith("file "):
            _t, name, size = line.split(None, 2)
            files.append((name, int(size)))
    return files, pw, extract_error, partial

def given_pw(argv):
    for a in argv:
        if a.startswith("-p") and len(a) > 2:
            return a[2:]
    return None

def main():
    argv = sys.argv[1:]
    mode = argv[0]
    log(mode + " " + " ".join(argv[1:]))
    if mode == "a":
        out_path = argv[4]
        Path(out_path).write_bytes(b"FAKE-AES-ZIP")
        return 0
    files, pw, extract_error, partial = manifest(argv[-1])
    if mode == "l":
        if pw is not None and given_pw(argv) != pw:
            print("Wrong password", file=sys.stderr)
            return 2
        for name, size in files:
            print(f" 1 2 3 {size} {name}")
        return 0
    if mode == "x":
        outdir = next(a[2:] for a in argv if a.startswith("-o"))
        if pw is not None and given_pw(argv) != pw:
            print("Wrong password", file=sys.stderr)
            return 2
        if extract_error is not None:
            code, text = extract_error
            if partial:
                for name, size in files:
                    if name == partial:
                        p = Path(outdir) / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"x" * size)
            print(text, file=sys.stderr)
            return code
        for name, size in files:
            p = Path(outdir) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * size)
        return 0
    return 0

sys.exit(main())
'''

FAKEUNRAR = r'''
import sys
from pathlib import Path

def manifest(path):
    files = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("file "):
            _t, name, size = line.split(None, 2)
            files.append((name, int(size)))
    return files

def main():
    argv = sys.argv[1:]
    mode = argv[0]
    if mode == "lt":
        for name, size in manifest(argv[-1]):
            print(f"{name} 2024-01-01 1 {size}")
        return 0
    if mode == "x":
        dest, arc = argv[-1], argv[-2]
        for name, size in manifest(arc):
            p = Path(dest) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * size)
        return 0
    return 0

sys.exit(main())
'''


def _fake_tool(tmp_path: Path, name: str, body: str) -> str:
    tool = tmp_path / "fakes" / name
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/usr/bin/env python3\n" + body)
    tool.chmod(0o755)
    return str(tool)


def _manifest(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_7z_budget_refusal_before_any_extraction(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    fake = _fake_tool(tmp_path, "7z", FAKE7Z)
    log = tmp_path / "fakes" / "calls.log"
    monkeypatch.setenv("FAKE7Z_LOG", str(log))
    _manifest(tmp_path, "big.7z",
              "file a.bin 2000\nfile b.bin 2000\nfile c.bin 1\n")
    a = _archivist(ctx, _bin7z=fake)
    with pytest.raises(ToolError) as ei:
        a.extract("big.7z", dest="b1", max_bytes=100)
    assert "refusing to extract" in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.extract("big.7z", dest="b2", max_files=2)
    assert "refusing to extract" in str(ei.value)
    calls = log.read_text()
    assert " l " in calls or calls.startswith("l ")
    assert " x " not in calls  # extraction was never invoked


def test_7z_extract_and_password_via_fake(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    fake = _fake_tool(tmp_path, "7z", FAKE7Z)
    _manifest(tmp_path, "vault.7z",
              "!password s3cret\nfile doc.txt 12\nfile pic.bin 30\n")
    a = _archivist(ctx, _bin7z=fake)
    with pytest.raises(ToolError) as ei:
        a.list("vault.7z")
    assert "password required" in str(ei.value)
    assert "s3cret" not in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.list("vault.7z", password="bad")
    assert "password incorrect" in str(ei.value)
    out = a.extract("vault.7z", dest="v1", password="s3cret")
    assert out["extracted"] == 2
    assert out["bytes"] == 42
    assert (tmp_path / "v1" / "doc.txt").stat().st_size == 12


def test_7z_partial_error_is_an_honest_partial(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    fake = _fake_tool(tmp_path, "7z", FAKE7Z)
    _manifest(tmp_path, "half.7z",
              "!extract-error 1 CRC error on b.txt\n!partial a.txt\n"
              "file a.txt 3\nfile b.txt 3\n")
    a = _archivist(ctx, _bin7z=fake)
    out = a.extract("half.7z", dest="h1")
    assert out["extracted"] == 1
    reasons = " ".join(d["reason"] for d in out["skipped_details"])
    assert "partial extraction" in reasons


def test_rar_budget_refusal_via_fake(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    fake = _fake_tool(tmp_path, "unrar", FAKEUNRAR)
    _manifest(tmp_path, "big.rar", "file a.bin 5000\nfile b.bin 5000\n")
    a = _archivist(ctx, _bin_unrar=fake)
    with pytest.raises(ToolError) as ei:
        a.extract("big.rar", dest="r1", max_bytes=100)
    assert "refusing to extract" in str(ei.value)
    out = a.extract("big.rar", dest="r2")
    assert out["extracted"] == 2
    assert (tmp_path / "r2" / "a.bin").stat().st_size == 5000


def test_create_encrypted_zip_via_fake_7z(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    fake = _fake_tool(tmp_path, "7z", FAKE7Z)
    log = tmp_path / "fakes" / "calls.log"
    monkeypatch.setenv("FAKE7Z_LOG", str(log))
    (tmp_path / "a.txt").write_text("alpha")
    a = _archivist(ctx, _bin7z=fake)
    a.create(["a.txt"], "enc.zip", password="pw123")
    assert (tmp_path / "enc.zip").read_bytes() == b"FAKE-AES-ZIP"
    calls = log.read_text()
    assert "-mem=AES256" in calls
    assert "-ppw123" in calls  # the flag reached the tool


# ── (f) filename encoding ───────────────────────────────────────────────


def _byte_patch_zip(tmp_path: Path, name: str,
                    pairs: list[tuple[bytes, bytes]]) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        for raw, _ in pairs:
            zf.writestr(raw.decode("ascii"), "data")
    raw = p.read_bytes()
    for old, new in pairs:
        assert len(old) == len(new)
        assert old in raw
        raw = raw.replace(old, new)
    p.write_bytes(raw)
    return p


def test_cp437_fallback_is_used_and_surfaced(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    # Member name bytes are cp437 (0x82 = é) with NO UTF-8 flag: the
    # spec fallback decodes them as cp437.
    _byte_patch_zip(tmp_path, "legacy.zip",
                    [(b"cafX.txt", b"caf\x82.txt")])
    a = Archivist(ctx)
    assert a.info("legacy.zip")["filename_encoding"] == "cp437"
    listing = a.list("legacy.zip")
    assert listing["filename_encoding"] == "cp437"
    assert listing["entries"][0]["name"] == "café.txt"
    out = a.extract("legacy.zip", dest="l1")
    assert out["filename_encoding"] == "cp437"
    assert (tmp_path / "l1" / "café.txt").read_text() == "data"


def test_utf8_names_report_utf8(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    with zipfile.ZipFile(tmp_path / "modern.zip", "w") as zf:
        zf.writestr("café.txt", "yum")
    a = Archivist(ctx)
    assert a.info("modern.zip")["filename_encoding"] == "utf-8"


def test_undecidable_name_fails_by_name(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    p = _byte_patch_zip(tmp_path, "broken.zip",
                        [(b"invXlid.txt", b"inv\xfflid.txt")])
    # …and set the UTF-8 flag (0x800) in both headers, so the name
    # CLAIMS utf-8 but carries an invalid byte: undecidable.
    raw = bytearray(p.read_bytes())
    raw[7] |= 0x08                      # local header flag high byte
    central = raw.find(b"PK\x01\x02")
    assert central > 0
    raw[central + 9] |= 0x08            # central header flag high byte
    p.write_bytes(bytes(raw))
    a = Archivist(ctx)
    with pytest.raises(ToolError) as ei:
        a.list("broken.zip")
    assert "undecidable member name encoding" in str(ei.value)
    with pytest.raises(ToolError) as ei:
        a.extract("broken.zip", dest="u1")
    assert "undecidable member name encoding" in str(ei.value)
    # info() reports instead of raising, in its own style.
    assert "unreadable" in a.info("broken.zip")["note"]


def test_unwritable_name_is_skipped_with_reason(tmp_path: Path) -> None:
    # A member name no filesystem here can hold (300 chars > the
    # 255-byte component limit): skipped with its reason while its
    # sibling lands.  (NUL-byte names never reach the engine at
    # all — the stdlib zip reader truncates member names at the
    # first NUL when parsing the directory; the engine still guards
    # for one, but there is nothing to pin against this reader.)
    ctx = _context(tmp_path)
    long_name = "x" * 300 + ".txt"
    with zipfile.ZipFile(tmp_path / "names.zip", "w") as zf:
        zf.writestr("good.txt", "good")
        zf.writestr(long_name, "bad")
    a = Archivist(ctx)
    out = a.extract("names.zip", dest="n1")
    assert out["extracted"] == 1
    details = {d["name"]: d["reason"] for d in out["skipped_details"]}
    assert "cannot write entry" in details[long_name]
