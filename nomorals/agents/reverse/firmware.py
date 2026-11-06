"""The firmware rung (Pass 3) — census + bounded extraction.

REA drives Binwalk/Unblob as caller-supplied one-shots. Devon's
version stands on the same split this organ uses everywhere: a
**built-in census** that always answers (a signature scan over the
image's own bytes — filesystems, containers, wrappers, with the
uImage / SquashFS / cpio structures actually parsed, CRCs
actually checked), and a **self-provisioned extraction engine**
(unblob, pinned through PyPI's own published digests — see
:mod:`.provision`) that carves and extracts what the census
found. binwalk is probe-only here, stated plainly: its publisher
ships source-only releases (every GitHub release tag carries
zero binary assets) and a crates.io source build is a build
system, not a pinnable install artifact — the readelf precedent.

Extraction discipline, non-negotiable:

* extracted content is **never executed** — it is inventoried,
  hashed and read as bytes, nothing more (executing targets is
  the gated dynamic rung, Pass 4, and stays off);
* extraction lands in a bounded work dir under the RE tools
  root, never in the repo;
* bombs are guarded: file-count and byte caps are enforced on
  the extracted tree — past the cap the overflow is deleted and
  the refusal is recorded (failure ledger + a residual unknown),
  never silently absorbed;
* a string that *looks* like a credential is an observation,
  flagged with its limits on the record — never reported as a
  proven secret, and the value itself is never copied into
  evidence (a redacted shape and the path are the record; the
  path is the pointer).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tarfile
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import provision
from .failures import record_failure
from .inspectors import detect_format

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "ENGINE_PROVIDER_ID",
    "FirmwareError", "census", "is_firmware_image",
    "parse_cpio", "extract_native", "probe_binwalk",
    "parse_unblob_report", "run_unblob", "build_inventory",
    "scan_credentials", "fold_evidence", "LIMITATIONS",
    "MAX_EXTRACT_FILES", "MAX_EXTRACT_BYTES",
]

PROVIDER_ID = "devon.firmware-census"
PROVIDER_NAME = "Devon built-in firmware census"
PROVIDER_VERSION = "1"
ENGINE_PROVIDER_ID = "devon.unblob"
ENGINE_PROVIDER_NAME = "unblob (self-provisioned by Devon)"

RUN_TIMEOUT_S = 600.0
#: Extraction-bomb guards, enforced on the extracted tree.
MAX_EXTRACT_FILES = 20_000
MAX_EXTRACT_BYTES = 1 << 30
#: What one evidence record will list vs merely count.
MAX_INVENTORY_LISTED = 500
MAX_TEXT_SCAN_BYTES = 256 * 1024
MAX_GZIP_PEEK = 4 * 1024 * 1024

LIMITATIONS = (
    "static census and extraction only — nothing extracted was "
    "executed, so these records describe what the image contains, "
    "never what the firmware does when it runs",
    "the inventory lists a bounded sample of the extracted tree; "
    "counts are complete, listings are capped",
)

Runner = Callable[..., tuple[int, str]]


class FirmwareError(RuntimeError):
    """The firmware rung cannot answer; the message says why."""


def _run(argv: list[str], *, timeout: float = RUN_TIMEOUT_S,
         env: dict[str, str] | None = None) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


# ── the built-in signature census ───────────────────────────────────

#: (kind, magic, where) — "zero" magics anchor at offset 0, "scan"
#: magics are searched for across the image.
_SIGNATURES: tuple[tuple[str, bytes, str], ...] = (
    ("uimage", b"\x27\x05\x19\x56", "zero"),
    ("squashfs", b"hsqs", "scan"),
    ("squashfs-be", b"sqsh", "scan"),
    ("cpio", b"070701", "scan"),
    ("cpio-crc", b"070702", "scan"),
    ("trx", b"HDR0", "scan"),
    ("cramfs", b"\x28\xcd\x3d\x45", "zero"),
    ("cramfs-be", b"\x45\x3d\xcd\x28", "zero"),
    ("romfs", b"-rom1fs-", "zero"),
    ("ubi", b"UBI#", "scan"),
    ("fit-dtb", b"\xd0\x0d\xfe\xed", "zero"),
    ("zip", b"PK\x03\x04", "scan"),
    ("elf", b"\x7fELF", "scan"),
)

_FIRMWARE_KINDS = frozenset({
    "uimage", "squashfs", "squashfs-be", "cpio", "cpio-crc",
    "trx", "cramfs", "cramfs-be", "romfs", "ubi", "fit-dtb",
    "jffs2", "extfs", "tar",
})

_UIMAGE_OS = {5: "Linux"}
_UIMAGE_ARCH = {2: "ARM", 3: "i386", 5: "MIPS", 7: "PowerPC",
                12: "SH", 22: "AArch64", 23: "x86_64"}
_UIMAGE_TYPE = {1: "standalone", 2: "OS kernel", 3: "RAMDisk",
                4: "multi-file", 5: "firmware", 6: "script",
                7: "filesystem"}
_UIMAGE_COMP = {0: "none", 1: "gzip", 2: "bzip2", 3: "lzma",
                4: "lzo", 5: "lz4", 6: "zstd"}
_SQUASHFS_COMP = {1: "gzip", 2: "lzma", 3: "lzo", 4: "xz",
                  5: "lz4", 6: "zstd"}


def _parse_uimage(data: bytes) -> dict[str, Any]:
    (magic, hcrc, stamp, size, load, entry, dcrc, os_id, arch,
     type_id, comp, raw_name) = struct.unpack_from(">7I4B32s", data, 0)
    header = bytearray(data[:64])
    struct.pack_into(">I", header, 4, 0)
    facts: dict[str, Any] = {
        "name": raw_name.split(b"\x00")[0].decode("utf-8", "replace"),
        "image_size": size,
        "load_address": hex(load),
        "entry_point": hex(entry),
        "os": _UIMAGE_OS.get(os_id, f"os-{os_id}"),
        "arch": _UIMAGE_ARCH.get(arch, f"arch-{arch}"),
        "image_type": _UIMAGE_TYPE.get(type_id, f"type-{type_id}"),
        "compression": _UIMAGE_COMP.get(comp, f"comp-{comp}"),
        "header_crc_ok": (zlib.crc32(bytes(header)) & 0xFFFFFFFF) == hcrc,
        "payload_offset": 64,
    }
    payload = data[64:64 + size]
    if len(payload) == size:
        facts["data_crc_ok"] = \
            (zlib.crc32(payload) & 0xFFFFFFFF) == dcrc
    return facts


def _parse_squashfs(data: bytes, offset: int,
                    big_endian: bool) -> dict[str, Any]:
    en = ">" if big_endian else "<"
    try:
        (_magic, inodes, _mkfs, block_size, fragments, comp,
         _blog, _flags, _ids, major, minor, _root_inode,
         bytes_used) = struct.unpack_from(en + "IIIIIHHHHHHQQ", data,
                                          offset)
    except struct.error:
        return {}
    return {"inodes": inodes, "block_size": block_size,
            "fragments": fragments,
            "compression": _SQUASHFS_COMP.get(comp, f"comp-{comp}"),
            "version": f"{major}.{minor}", "bytes_used": bytes_used}


def parse_cpio(data: bytes, offset: int = 0, *,
               max_entries: int = 100_000) -> dict[str, Any]:
    """Parse a cpio (newc / newc-crc) archive at ``offset``.

    Returns {entries: [{name, size, mode}], end_offset, complete}.
    Names and sizes only — no content is read by the census.
    """
    entries: list[dict[str, Any]] = []
    pos = offset
    complete = False
    while pos + 110 <= len(data) and len(entries) < max_entries:
        header = data[pos:pos + 110]
        if header[:6] not in (b"070701", b"070702"):
            break
        try:
            fields = [int(header[6 + i * 8:14 + i * 8], 16)
                      for i in range(13)]
        except ValueError:
            break
        (_ino, mode, _uid, _gid, _nlink, _mtime, filesize, _maj,
         _min, _rmaj, _rmin, namesize, _check) = fields
        name_raw = data[pos + 110:pos + 110 + namesize - 1]
        name = name_raw.decode("utf-8", "replace")
        # header + name pad to a 4-byte boundary from the
        # archive's own start (the newc rule)
        pos += 110 + namesize
        pos = offset + ((pos - offset + 3) & ~3)
        if name == "TRAILER!!!":
            complete = True
            break
        entries.append({"name": name, "size": filesize,
                        "mode": oct(mode)})
        pos += filesize
        pos = offset + ((pos - offset + 3) & ~3)
    return {"entries": entries, "end_offset": pos,
            "complete": complete}


def _gzip_peek(data: bytes, offset: int) -> dict[str, Any]:
    """Decompress the head of a gzip stream and sniff what it wraps."""
    out: dict[str, Any] = {"kind": "gzip"}
    try:
        obj = zlib.decompressobj(31)
        head = obj.decompress(data[offset:offset + MAX_GZIP_PEEK + 64],
                              MAX_GZIP_PEEK)
    except zlib.error:
        out["readable"] = False
        return out
    out["readable"] = True
    out["inner"] = _sniff_inner(head)
    return out


def _sniff_inner(head: bytes) -> str:
    if head[:6] in (b"070701", b"070702"):
        return "cpio"
    if head[:4] == b"hsqs":
        return "squashfs"
    if len(head) > 262 and head[257:262] == b"ustar":
        return "tar"
    if head[:4] == b"\x7fELF":
        return "elf"
    return "unknown"


def census(data: bytes) -> dict[str, Any]:
    """The built-in firmware census over one image's bytes.

    Returns {is_firmware, primary, signatures, parsed} — signatures
    are regions [{kind, offset}] (bounded per kind), ``parsed``
    carries the deep-parsed facts for the primary structure
    (uImage header with CRC verdicts, SquashFS superblock, cpio
    entry census) and for gzip wrappers (what they wrap).
    """
    signatures: list[dict[str, Any]] = []
    for kind, magic, where in _SIGNATURES:
        starts = [0] if where == "zero" else None
        hits: list[int] = []
        if starts is not None:
            if data[:len(magic)] == magic:
                hits = [0]
        else:
            pos = data.find(magic)
            while pos >= 0 and len(hits) < 32:
                hits.append(pos)
                pos = data.find(magic, pos + 1)
        for off in hits:
            signatures.append({"kind": kind, "offset": off})
    # jffs2 node magic walks halfword-aligned — count, don't list all
    jffs2_hits = 0
    for magic in (b"\x85\x19", b"\x19\x85"):
        jffs2_hits += data.count(magic)
    if jffs2_hits >= 8:
        signatures.append({"kind": "jffs2", "offset": 0,
                           "node_magic_count": jffs2_hits})
    if len(data) > 1082 and data[1080:1082] == b"\x53\xef":
        signatures.append({"kind": "extfs", "offset": 1024})
    if len(data) > 262 and data[257:262] == b"ustar":
        signatures.append({"kind": "tar", "offset": 0})
    gzip_regions: list[dict[str, Any]] = []
    pos = data.find(b"\x1f\x8b")
    while pos >= 0 and len(gzip_regions) < 16:
        peek = _gzip_peek(data, pos)
        gzip_regions.append({"kind": "gzip", "offset": pos, **peek})
        pos = data.find(b"\x1f\x8b", pos + 1)
    signatures.extend(gzip_regions)
    signatures.sort(key=lambda s: (s["offset"], s["kind"]))

    parsed: dict[str, Any] = {}
    primary: str | None = None
    for sig in signatures:
        if sig["kind"] in _FIRMWARE_KINDS:
            primary = sig["kind"]
            break
    if primary is None:
        for reg in gzip_regions:
            if reg.get("inner") in _FIRMWARE_KINDS:
                primary = "gzip"
                break
    if any(s["kind"] == "uimage" and s["offset"] == 0
           for s in signatures):
        parsed["uimage"] = _parse_uimage(data)
    for sig in signatures:
        if sig["kind"] in ("squashfs", "squashfs-be"):
            parsed.setdefault("squashfs", _parse_squashfs(
                data, sig["offset"], sig["kind"].endswith("-be")))
            break
    for sig in signatures:
        if sig["kind"] in ("cpio", "cpio-crc"):
            cpio = parse_cpio(data, sig["offset"])
            parsed.setdefault("cpio", {
                "entries": len(cpio["entries"]),
                "complete": cpio["complete"],
                "names": [e["name"] for e in cpio["entries"][:32]],
            })
            break
    return {
        "is_firmware": primary is not None,
        "primary": primary,
        "signatures": signatures,
        "parsed": parsed,
    }


def is_firmware_image(data: bytes) -> bool:
    return census(data)["is_firmware"]


# ── native extraction (cpio / tar shapes, incl. wrapped) ────────────

def _safe_target(dest: Path, name: str) -> Path | None:
    if not name or name.startswith("/") or name == "TRAILER!!!":
        return None
    target = (dest / name).resolve()
    root = dest.resolve()
    if target != root and root not in target.parents:
        return None
    return target


def extract_native(data: bytes, dest: Path, *,
                   max_files: int = MAX_EXTRACT_FILES,
                   max_bytes: int = MAX_EXTRACT_BYTES) -> dict[str, Any]:
    """Extract the natively-supported shapes (cpio, tar, and
    uImage/gzip wrappers of them) into ``dest`` — data only,
    nothing is executed, links are written as plain files or
    skipped, and every member path is escape-checked.

    Returns {extracted: bool, shape, files, bytes, error?}.
    """
    dest.mkdir(parents=True, exist_ok=True)
    blob = data
    shape = ""
    facts = census(data)
    if any(s["kind"] == "uimage" and s["offset"] == 0
           for s in facts["signatures"]):
        parsed = facts["parsed"].get("uimage") or {}
        start = int(parsed.get("payload_offset") or 64)
        size = int(parsed.get("image_size") or len(blob))
        blob = blob[start:start + size]
        shape = "uimage+"
        if parsed.get("compression") == "gzip":
            try:
                blob = gzip.decompress(blob)
            except OSError as exc:
                return {"extracted": False, "shape": shape,
                        "error": f"uImage gzip payload unreadable: "
                                 f"{exc}"}
            shape += "gzip+"
    elif blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except OSError as exc:
            return {"extracted": False, "shape": "gzip",
                    "error": f"gzip payload unreadable: {exc}"}
        shape = "gzip+"
    written = 0
    total = 0
    if blob[:6] in (b"070701", b"070702"):
        pos = 0
        # walk the raw archive with content
        while pos + 110 <= len(blob):
            header = blob[pos:pos + 110]
            if header[:6] not in (b"070701", b"070702"):
                break
            try:
                fields = [int(header[6 + i * 8:14 + i * 8], 16)
                          for i in range(13)]
            except ValueError:
                break
            (_ino, mode, _u, _g, _n, _m, filesize, _a, _b, _c, _d,
             namesize, _chk) = fields
            name = blob[pos + 110:pos + 110 + namesize - 1] \
                .decode("utf-8", "replace")
            pos += 110 + namesize
            pos = (pos + 3) & ~3
            content = blob[pos:pos + filesize]
            pos += filesize
            pos = (pos + 3) & ~3
            if name == "TRAILER!!!":
                break
            target = _safe_target(dest, name)
            if target is None:
                continue
            if mode & 0o170000 == 0o040000:
                target.mkdir(parents=True, exist_ok=True)
                continue
            if mode & 0o170000 == 0o120000:
                continue  # links are inventoried, never materialized
            if written >= max_files or total + filesize > max_bytes:
                return {"extracted": True, "shape": shape + "cpio",
                        "files": written, "bytes": total,
                        "refused": "native extraction hit the cap"}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            written += 1
            total += filesize
        return {"extracted": True, "shape": shape + "cpio",
                "files": written, "bytes": total}
    if len(blob) > 262 and blob[257:262] == b"ustar":
        try:
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                for member in tf.getmembers():
                    if written >= max_files \
                            or total + member.size > max_bytes:
                        return {"extracted": True,
                                "shape": shape + "tar",
                                "files": written, "bytes": total,
                                "refused": "native extraction hit "
                                           "the cap"}
                    if not member.isfile():
                        continue
                    target = _safe_target(dest, member.name)
                    if target is None:
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    content = fh.read()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    written += 1
                    total += len(content)
        except tarfile.TarError as exc:
            return {"extracted": False, "shape": shape + "tar",
                    "error": f"tar unreadable: {exc}"}
        return {"extracted": True, "shape": shape + "tar",
                "files": written, "bytes": total}
    return {"extracted": False, "shape": shape,
            "error": "no natively-extractable shape (cpio/tar) in "
                     "this image"}


# ── binwalk: probe-only, with the reason on the record ──────────────

def probe_binwalk(runner: Runner | None = None) -> dict[str, Any]:
    """binwalk is never provisioned by Devon — its publisher ships
    source-only releases with no binary artifact to pin (checked
    across the v2.1–v3.1 release line, 2026-11-06). A system
    binwalk is probed and reported for the record; the extraction
    work belongs to the pinned unblob rung.
    """
    path = shutil.which("binwalk")
    reason = ("binwalk is probe-only for Devon: ReFirmLabs ships "
              "source-only releases (no binary assets on the "
              "publisher's release channel) and a crates.io source "
              "build is a build system, not a pinnable install "
              "artifact — the pinned unblob rung does the work")
    if not path:
        return {"available": False, "path": "", "version": "",
                "detail": "no system binwalk on PATH. " + reason}
    run = runner or _run
    try:
        _rc, out = run([path, "--version"], timeout=10.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "path": path, "version": "",
                "detail": f"binwalk at {path} did not answer: "
                          f"{exc}. " + reason}
    version = (out or "").strip().splitlines()[0] if out else ""
    return {"available": True, "path": path, "version": version,
            "detail": f"system binwalk probed at {path} "
                      f"({version}); reported, not driven — "
                      + reason}


# ── unblob: the provisioned extraction engine ───────────────────────

def parse_unblob_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse an unblob ``--report`` JSON document (a list of task
    rows) into the fold's facts. Pure — the shapes pinned here
    are unblob 26.x's real report vocabulary (ChunkReport,
    UnknownChunkReport, StatReport, HashReport,
    ExtractorDependencyNotFoundReport), recorded from real runs.
    """
    chunks: list[dict[str, Any]] = []
    unknown_chunks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for row in rows or []:
        task = row.get("task") or {}
        in_path = str(task.get("path") or "")
        by_path: dict[str, dict[str, Any]] = {}
        for rep in row.get("reports") or []:
            kind = rep.get("__typename__")
            if kind == "ChunkReport":
                chunk_errors = []
                for sub in rep.get("extraction_reports") or []:
                    entry = {
                        "report": sub.get("__typename__"),
                        "severity": sub.get("severity"),
                        "missing": list(sub.get("dependencies")
                                        or []),
                        "handler": rep.get("handler_name"),
                    }
                    chunk_errors.append(entry)
                    if entry["missing"] or "Error" in str(
                            entry["report"]):
                        errors.append(entry)
                chunks.append({
                    "handler": rep.get("handler_name"),
                    "start": rep.get("start_offset"),
                    "end": rep.get("end_offset"),
                    "size": rep.get("size"),
                    "encrypted": bool(rep.get("is_encrypted")),
                    "in": in_path,
                    "extraction_errors": chunk_errors,
                })
            elif kind == "UnknownChunkReport":
                unknown_chunks.append({
                    "start": rep.get("start_offset"),
                    "end": rep.get("end_offset"),
                    "size": rep.get("size"),
                    "randomness": rep.get("randomness"),
                    "in": in_path,
                })
            elif kind == "StatReport":
                by_path.setdefault(str(rep.get("path") or ""), {
                    "path": str(rep.get("path") or ""),
                    "size": rep.get("size"),
                    "is_dir": bool(rep.get("is_dir")),
                })
            elif kind == "HashReport":
                target = by_path.setdefault(in_path, {
                    "path": in_path, "size": None, "is_dir": False})
                target["sha256"] = rep.get("sha256")
        files.extend(by_path.values())
    return {"chunks": chunks, "unknown_chunks": unknown_chunks,
            "extraction_errors": errors, "files": files}


def run_unblob(target: str | Path, work_dir: str | Path, *,
               root: Path | None = None,
               runner: Runner | None = None) -> dict[str, Any]:
    """Run the provisioned unblob rung over one image.

    Returns the parsed report facts plus {extract_root, rc}. A
    non-zero exit is NOT automatically a failure — unblob exits
    non-zero when a chunk's extractor is missing while still
    carving and censusing the rest; the report is the verdict.
    Raises :class:`FirmwareError` when the engine is unavailable
    or no readable report lands.
    """
    state = provision.unblob_state(root, runner=runner)
    if not state["available"]:
        raise FirmwareError("unblob rung unavailable: "
                            + state["detail"])
    exe = state["path"]
    work = Path(work_dir)
    extract_root = work / "unblob"
    report_path = work / "unblob-report.json"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if state["source"] == "provisioned":
        bindir = str(Path(exe).parent)
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    argv = [exe, "-e", str(extract_root), "-f", "--depth", "6",
            "--report", str(report_path), str(target)]
    run = runner or _run
    try:
        rc, out = run(argv, timeout=RUN_TIMEOUT_S, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FirmwareError(f"unblob run failed: {exc}") from exc
    if not report_path.is_file():
        tail = " · ".join((out or "").strip().splitlines()[-3:])
        raise FirmwareError(
            f"unblob produced no report (exit {rc}): {tail}")
    try:
        rows = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FirmwareError(f"unblob report unreadable: {exc}") \
            from exc
    facts = parse_unblob_report(rows)
    facts["extract_root"] = str(extract_root)
    facts["rc"] = rc
    facts["version"] = state["version"]
    return facts


# ── inventory + credential-shaped flags ─────────────────────────────

def _kind_of(path: Path, head: bytes) -> str:
    fmt = detect_format(head)
    if fmt != "unknown":
        return fmt
    if head[:2] == b"\x1f\x8b" or head[:6] in (b"070701", b"070702") \
            or head[:4] == b"PK\x03\x04":
        return "archive"
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return "data"
    printable = sum(1 for ch in text if ch.isprintable()
                    or ch in "\n\t")
    return "text" if text and printable / len(text) > 0.92 else "data"


_NOTABLE = (
    re.compile(r"(^|/)etc/(passwd|shadow)$"),
    re.compile(r"(^|/)etc/config/"),
    re.compile(r"authorized_keys$"),
    re.compile(r"\.(key|pem)$"),
    re.compile(r"(^|/)etc/(inittab|resolv\.conf)$"),
    re.compile(r"(hostapd|wpa_supplicant)\.conf$"),
    re.compile(r"(^|/)\.ssh/"),
)

_CRED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Assignment in either syntax: ``name = value`` / ``name: value``
    # or the config-file shape ``option name 'value'`` (space + quote).
    ("password-assignment",
     re.compile(r"(?i)(?:password|passwd|pwd)[\w-]*\s*"
                r"(?:[:=]\s*['\"]?|\s+['\"])(?P<v>[^\s'\"]{2,})")),
    ("secret-assignment",
     re.compile(r"(?i)(?:api[_-]?key|aws[_-]?key|access[_-]?key|"
                r"secret|token|private[_-]?key)[\w-]*\s*"
                r"(?:[:=]\s*['\"]?|\s+['\"])(?P<v>[^\s'\"]{2,})")),
    ("aws-access-key-id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key-block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("credential-in-url",
     re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@")),
)


def scan_credentials(files: list[tuple[str, Path]]) -> dict[str, Any]:
    """Scan extracted text files for credential-SHAPED strings.

    A flag is an observation that a string of a credential's
    shape exists at a path — the value is never copied (the
    record carries a redacted shape), and nothing here proves
    the string is a live credential: firmware ships defaults,
    placeholders and test values by the thousand.
    """
    flags: list[dict[str, Any]] = []
    scanned = 0
    for rel, path in files:
        try:
            if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
                continue
            raw = path.read_bytes()
        except OSError as exc:
            record_failure("reverse.firmware.cred_scan_read", exc)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for line in text.splitlines()[:2000]:
            for name, pattern in _CRED_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                value = match.groupdict().get("v")
                if value:
                    start = match.start("v")
                    redacted = (line[:start] + value[:2] + "…")
                else:
                    redacted = match.group()[:6] + "…"
                flags.append({"path": rel, "pattern": name,
                              "redacted": redacted[:120]})
                break
    return {"flags": flags[:100], "flag_count": len(flags),
            "scanned_files": scanned}


def build_inventory(extract_root: str | Path, *,
                    subject_prefix: str = "",
                    max_files: int = MAX_EXTRACT_FILES,
                    max_bytes: int = MAX_EXTRACT_BYTES
                    ) -> dict[str, Any]:
    """Walk an extracted tree into an inventory, enforcing the
    extraction caps: files past the cap are deleted and the
    refusal is returned as data (the caller records it), never
    silently absorbed. Kinds are sniffed per file; credential
    scanning runs over the text-shaped members.
    """
    root = Path(extract_root)
    listed: list[dict[str, Any]] = []
    text_files: list[tuple[str, Path]] = []
    total_files = 0
    total_bytes = 0
    refusal: dict[str, Any] | None = None
    removed = 0
    if root.is_dir():
        all_files = sorted(
            (p for p in root.rglob("*") if p.is_file()),
            key=lambda p: str(p))
        for path in all_files:
            size = path.stat().st_size
            total_files += 1
            total_bytes += size
            if refusal is None and (total_files > max_files
                                    or total_bytes > max_bytes):
                refusal = {
                    "reason": (f"extracted tree crossed the cap "
                               f"({max_files} files / {max_bytes} "
                               "bytes) — the overflow was deleted "
                               "and the inventory is partial"),
                    "cap_files": max_files, "cap_bytes": max_bytes,
                }
                record_failure(
                    "reverse.firmware.extraction_cap",
                    ValueError(refusal["reason"]))
            if refusal is not None:
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:
                    record_failure(
                        "reverse.firmware.cap_delete", exc)
                continue
            try:
                with open(path, "rb") as fh:
                    head = fh.read(8192)
            except OSError as exc:
                record_failure("reverse.firmware.inventory_read",
                               exc)
                head = b""
            kind = _kind_of(path, head)
            rel = str(path.relative_to(root))
            entry = {"path": rel, "size": size, "kind": kind}
            if kind in ("elf", "pe", "macho", "macho-fat"):
                entry["sha256"] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
            if len(listed) < MAX_INVENTORY_LISTED:
                listed.append(entry)
            if kind == "text":
                text_files.append((rel, path))
    census_counter = Counter(e["kind"] for e in listed)
    notable = [e["path"] for e in listed
               if any(p.search(e["path"]) for p in _NOTABLE)]
    carved = [e for e in listed
              if e["kind"] in ("elf", "pe", "macho", "macho-fat")]
    creds = scan_credentials(text_files)
    out: dict[str, Any] = {
        "files_total": total_files,
        "bytes_total": total_bytes,
        "files_listed": listed,
        "listing_capped": total_files > len(listed),
        "type_census": dict(census_counter),
        "notable_paths": notable[:64],
        "carved_binaries": carved[:64],
        "credential_flags": creds,
        "extract_root": str(root),
    }
    if refusal is not None:
        refusal["files_removed"] = removed
        out["refusal"] = refusal
    return out


# ── the fold ────────────────────────────────────────────────────────

def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, subject_size: int,
                  subject_path: str,
                  census_facts: dict[str, Any],
                  unblob_facts: dict[str, Any] | None = None,
                  inventory: dict[str, Any] | None = None,
                  extraction_source: str = "",
                  link_to: str = "") -> dict[str, Any]:
    """Fold one firmware analysis into Evidence + unknowns.

    Chain: census (built-in) → chunks (unblob, when it ran) →
    inventory → credential-shaped flags. Every gap the engines
    named — a missing extractor, an unextracted region, an
    encrypted chunk, a crossed cap — becomes a residual unknown
    naming what closes it.
    """
    evidence_ids: list[str] = []
    unknowns: list[dict[str, Any]] = []
    common = {"subject_sha256": subject_sha256,
              "subject_name": subject_name,
              "subject_size": subject_size}
    links = [link_to] if link_to else []

    census_rec = store.record(
        predicate_type="firmware.census", operation="signature.scan",
        provider_id=PROVIDER_ID, provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        parameters={"scanner": "builtin-signatures"},
        result={"primary": census_facts.get("primary"),
                "signatures": census_facts.get("signatures") or [],
                "parsed": census_facts.get("parsed") or {}},
        limitations=list(LIMITATIONS),
        locations=[{"type": "artifact-path", "value": subject_path}],
        links=links, **common)
    evidence_ids.append(census_rec.evidence_id)
    anchor = census_rec.evidence_id

    if unblob_facts is not None:
        chunk_rec = store.record(
            predicate_type="firmware.chunks",
            operation="unblob.extract",
            provider_id=ENGINE_PROVIDER_ID,
            provider_name=ENGINE_PROVIDER_NAME,
            provider_version=str(unblob_facts.get("version") or ""),
            parameters={"engine": "unblob"},
            result={
                "chunks": unblob_facts.get("chunks") or [],
                "unknown_chunks":
                    unblob_facts.get("unknown_chunks") or [],
                "extraction_errors":
                    unblob_facts.get("extraction_errors") or [],
                "exit_code": unblob_facts.get("rc"),
            },
            limitations=list(LIMITATIONS),
            locations=[{"type": "artifact-path",
                        "value": subject_path}],
            links=[anchor], **common)
        evidence_ids.append(chunk_rec.evidence_id)
        anchor = chunk_rec.evidence_id
        for err in unblob_facts.get("extraction_errors") or []:
            missing = ", ".join(err.get("missing") or []) or "?"
            unknowns.append(store.open_unknown(
                f"extraction of a chunk by the "
                f"{err.get('handler') or 'unknown'} handler is "
                f"unestablished — the extractor(s) it needs "
                f"({missing}) are not on this machine; a probed "
                f"system install of the named extractor closes "
                f"this",
                subject_sha256=subject_sha256, raised_by=anchor,
                supporting=[anchor],
                required_authority="shipped-artifact",
                required_confidence="derived").to_dict())
        for unk in unblob_facts.get("unknown_chunks") or []:
            randomness = unk.get("randomness")
            rand_note = ""
            if isinstance(randomness, dict):
                mean = ((randomness.get("shannon") or {})
                        .get("mean"))
                if isinstance(mean, (int, float)):
                    rand_note = (f" (shannon randomness "
                                 f"{mean:.1f}%)")
            unknowns.append(store.open_unknown(
                f"the region {unk.get('start')}–{unk.get('end')} "
                f"of the image matched no known format"
                f"{rand_note} — its content is unestablished",
                subject_sha256=subject_sha256, raised_by=anchor,
                supporting=[anchor],
                required_authority="shipped-artifact",
                required_confidence="derived").to_dict())
        for chunk in unblob_facts.get("chunks") or []:
            if chunk.get("encrypted"):
                unknowns.append(store.open_unknown(
                    f"the {chunk.get('handler')} chunk at "
                    f"{chunk.get('start')}–{chunk.get('end')} is "
                    f"encrypted — its content is opaque to static "
                    f"extraction",
                    subject_sha256=subject_sha256,
                    raised_by=anchor, supporting=[anchor],
                    required_authority="controlled-replay",
                    required_confidence="derived").to_dict())

    if inventory is not None:
        inv_rec = store.record(
            predicate_type="firmware.inventory",
            operation="extraction.inventory",
            provider_id=(ENGINE_PROVIDER_ID if extraction_source
                         == "unblob" else PROVIDER_ID),
            provider_name=(ENGINE_PROVIDER_NAME
                           if extraction_source == "unblob"
                           else PROVIDER_NAME),
            provider_version=(str(unblob_facts.get("version") or "")
                              if extraction_source == "unblob"
                              else PROVIDER_VERSION),
            parameters={"extraction": extraction_source},
            result={
                "files_total": inventory["files_total"],
                "bytes_total": inventory["bytes_total"],
                "files": inventory["files_listed"],
                "listing_capped": inventory["listing_capped"],
                "type_census": inventory["type_census"],
                "notable_paths": inventory["notable_paths"],
                "carved_binaries": inventory["carved_binaries"],
                "extract_root": inventory["extract_root"],
                **({"refusal": inventory["refusal"]}
                   if inventory.get("refusal") else {}),
            },
            limitations=list(LIMITATIONS),
            locations=[{"type": "artifact-path",
                        "value": inventory["extract_root"]}],
            links=[anchor], **common)
        evidence_ids.append(inv_rec.evidence_id)
        anchor = inv_rec.evidence_id
        creds = inventory.get("credential_flags") or {}
        if creds.get("flag_count"):
            flag_rec = store.record(
                predicate_type="firmware.credential_flags",
                operation="content.scan",
                provider_id=PROVIDER_ID,
                provider_name=PROVIDER_NAME,
                provider_version=PROVIDER_VERSION,
                parameters={"scan": "credential-shapes"},
                result={"flags": creds["flags"],
                        "flag_count": creds["flag_count"],
                        "scanned_files": creds["scanned_files"]},
                confidence="derived",
                limitations=[
                    "a flagged string is an observation that a "
                    "credential-SHAPED string exists at the path — "
                    "not proof the credential is real, live or "
                    "valid; firmware ships default, placeholder "
                    "and test values, and the value itself is "
                    "never copied into evidence (redacted shape "
                    "only; the path is the pointer)",
                    *LIMITATIONS],
                locations=[{"type": "artifact-path",
                            "value": inventory["extract_root"]}],
                links=[anchor], **common)
            evidence_ids.append(flag_rec.evidence_id)
        if inventory.get("refusal"):
            unknowns.append(store.open_unknown(
                "the extracted content past the enforced cap is "
                "unestablished — extraction crossed the file/byte "
                "cap and the overflow was deleted unexamined; a "
                "larger cap (or a narrower target) closes this",
                subject_sha256=subject_sha256, raised_by=anchor,
                supporting=[anchor],
                required_authority="shipped-artifact",
                required_confidence="derived").to_dict())
    return {"evidence_ids": evidence_ids, "unknowns": unknowns}
