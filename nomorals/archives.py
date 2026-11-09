"""Archive system — create and crack open any common archive.

Formats, by capability:
* **stdlib (always)** — zip, tar, tar.gz, tar.bz2, tar.xz, and single-file
  gzip / bzip2 / xz.
* **external (when installed)** — 7z / 7z / 7za, rar (``unrar`` / ``unar``).

Every archive is first *identified* by magic bytes (not just extension),
so a renamed ``.zip`` that's actually tar.gz still works, and ``info``
reports what the file really is.  Extraction is traversal-safe: absolute
paths and ``..`` escapes are skipped and reported, never followed.

Depth (WS1-P2): password-protected archives open with a caller-supplied
``password`` (never logged or persisted; a wrong/missing one fails with
a named error, never a traceback); extraction recurses into archives
inside archives up to a depth cap under the *same* cumulative bomb
budgets; a corrupt member yields an honest partial (good entries land,
bad ones are named with their reason) instead of aborting the run;
members can be appended to / removed from existing zip and tar
archives; and zip member names follow the spec's encoding rule (UTF-8
flag, else the cp437 fallback) with the fallback surfaced, never
silent.

    from nomorals.archives import Archivist
    a = Archivist(context)
    a.info("workspace/data.zip")        # real format + entries + sizes
    a.list("workspace/data.zip")
    a.extract("workspace/data.zip")     # → workspace/data.zip.extracted/
    a.extract("workspace/secret.zip", password="…")
    a.create(["workspace/a.txt", "workspace/b.txt"], "bundle.tar.gz")
    a.append("workspace/bundle.zip", ["workspace/c.txt"])
    a.compress("workspace/big.bin")     # → big.bin.gz

Registered as the ``archive`` tool.
"""

from __future__ import annotations

import bz2
import contextlib
import glob
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

from .core.errors import ToolError
from .core.logging_setup import get_logger
from .core.policy import Capability

_log = get_logger(__name__)

__all__ = ["Archivist", "detect_format", "register"]

_MAX_LIST = 500
_MAX_EXTRACT_FILES = 20_000
_MAX_EXTRACT_BYTES = 2_000_000_000  # 2 GB guard against zip bombs
#: how many levels of archive-inside-archive extraction opens before
#: it stops and reports the rest unopened (budgets still apply at
#: every level, cumulatively)
_MAX_NEST_DEPTH = 3

#: formats that hold members (and so can nest / be appended to)
_CONTAINER_FORMATS = ("zip", "tar", "tar.gz", "tar.bz2", "tar.xz",
                      "7z", "rar")


class _BombError(ToolError):
    """A bomb-budget trip inside extraction.  Always a whole refusal
    — even when raised from a nested level, a bomb is a bomb at any
    depth — so the nesting recursion re-raises it instead of folding
    it into a per-archive error note."""


def _scrub(text: str, *secrets: str | None) -> str:
    """Remove secret values (passwords) from external-tool output
    before any of it can reach an error message or a log line."""
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def _passwordish(text: str) -> bool:
    """Does this external-tool output read as a credential failure?"""
    low = text.lower()
    return "password" in low or "encrypted" in low


def _silent_unlink(path: Path) -> None:
    """Best-effort cleanup of a partial output file.  Even unlink
    can fail on the names this guards against (an overlong member
    name raises ENAMETOOLONG on the unlink itself) — a cleanup
    failure must never mask the real, named error."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _zip_name_encoding(infos: list[zipfile.ZipInfo]) -> str:
    """How the member names were decoded, per the zip spec: the
    UTF-8 flag when set, else the cp437 fallback.  ``"cp437"`` in a
    result means at least one non-ASCII name rode the legacy
    fallback — the mojibake-risk case, surfaced rather than silent.
    ``"ascii"`` means every name is plain ASCII (encoding moot)."""
    fallback = False
    utf8 = False
    for i in infos:
        if i.flag_bits & 0x800:
            utf8 = True
        elif any(ord(c) > 127 for c in i.filename):
            fallback = True
    if fallback:
        return "cp437"
    return "utf-8" if utf8 else "ascii"


def detect_format(path: str | Path) -> str:
    """Identify an archive by magic bytes (extension only as a hint)."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(262)
    except OSError:
        head = b""
    name = p.name.lower()
    # tarballs before their container magics
    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if name.endswith((".tar.bz2", ".tbz2")):
        return "tar.bz2"
    if name.endswith((".tar.xz", ".txz")):
        return "tar.xz"
    if name.endswith(".tar"):
        return "tar"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    if head.startswith(b"\x1f\x8b"):
        return _sniff_tarball(p, "gzip")
    if head.startswith(b"BZh"):
        return _sniff_tarball(p, "bzip2")
    if head.startswith(b"\xfd7zXZ\x00"):
        return _sniff_tarball(p, "xz")
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if head.startswith(b"Rar!\x1a\x07"):
        return "rar"
    ext = p.suffix.lower()
    if ext == ".gz":
        return "gzip"
    if ext == ".bz2":
        return "bzip2"
    if ext == ".xz":
        return "xz"
    if ext == ".7z":
        return "7z"
    if ext == ".zip":
        return "zip"
    if ext == ".rar":
        return "rar"
    return "unknown"


def _which(names: tuple[str, ...]) -> str:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return ""


def _sniff_tarball(p: Path, fmt: str) -> str:
    """A gzip/bzip2/xz stream may wrap a tarball: peek at the
    decompressed head for tar's ``ustar`` marker and report the tar
    format when it's there.  Identification by content — a tarball
    renamed to ``.zip`` (or anything else) is still a tarball, which
    is what the module docstring has always claimed.  An unreadable
    stream keeps the plain container format."""
    import gzip

    openers = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open}
    try:
        with openers[fmt](p, "rb") as fh:
            head = fh.read(512)
    except Exception:  # noqa: BLE001 — sniff failure is not a refusal
        return fmt
    if len(head) >= 262 and head[257:262] == b"ustar":
        return {"gzip": "tar.gz", "bzip2": "tar.bz2",
                "xz": "tar.xz"}[fmt]
    return fmt


class Archivist:
    """Detect, list, create, and safely extract archives."""

    role = "archivist"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._bin7z = _which(("7z", "7za", "7zr"))
        self._bin_unrar = _which(("unrar", "unar"))

    # ── resolve a workspace path ─────────────────────────────────────────
    def _path(self, ref: str, *, must_exist: bool = True) -> Path:
        from .tools.filesystem import safe_path

        return safe_path(self.context, ref, must_exist=must_exist)

    # ── info / list ───────────────────────────────────────────────────────
    def info(self, path: str, *,
             password: str | None = None) -> dict[str, Any]:
        p = self._path(path)
        fmt = detect_format(p)
        size = p.stat().st_size
        out: dict[str, Any] = {"path": str(p), "format": fmt,
                               "compressed_bytes": size, "entries": 0,
                               "total_bytes": 0}
        try:
            if fmt == "zip":
                with zipfile.ZipFile(p) as zf:
                    infos = zf.infolist()
                    out["entries"] = len(infos)
                    out["total_bytes"] = sum(i.file_size for i in infos)
                    out["encrypted"] = any(
                        i.flag_bits & 0x1 for i in infos)
                    out["filename_encoding"] = _zip_name_encoding(infos)
            elif fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
                with tarfile.open(p) as tf:
                    members = tf.getmembers()
                    out["entries"] = len(members)
                    out["total_bytes"] = sum(m.size for m in members)
            elif fmt in ("gzip", "bzip2", "xz"):
                out["entries"] = 1
            elif fmt == "7z" and self._bin7z:
                out.update(self._list7z(p, password=password))
            elif fmt == "rar" and self._bin_unrar:
                out.update(self._list_rar(p, password=password))
            elif fmt in ("7z", "rar"):
                out["note"] = (f"needs {'7z' if fmt == '7z' else 'unrar'} "
                               "installed to inspect")
            else:
                out["note"] = "not a recognized archive"
        except ToolError as exc:
            # A named failure (e.g. password required to list a
            # header-encrypted archive) is the honest note here —
            # info() reports, it does not raise.
            out["note"] = str(exc)
        except (OSError, EOFError, UnicodeError, zipfile.BadZipFile,
                tarfile.TarError, lzma.LZMAError) as exc:
            out["note"] = f"unreadable: {exc}"
        if out.get("total_bytes") and size > 0:
            out["ratio"] = round(out["total_bytes"] / size, 2)
        return out

    def list(self, path: str, *, limit: int = _MAX_LIST,
             password: str | None = None) -> dict[str, Any]:
        p = self._path(path)
        fmt = detect_format(p)
        names: list[dict[str, Any]] = []
        extra: dict[str, Any] = {}
        if fmt == "zip":
            try:
                with zipfile.ZipFile(p) as zf:
                    infos = zf.infolist()
            except UnicodeDecodeError as exc:
                raise ToolError(
                    f"undecidable member name encoding in {p.name}: "
                    f"{exc}") from exc
            extra["filename_encoding"] = _zip_name_encoding(infos)
            extra["encrypted"] = any(i.flag_bits & 0x1 for i in infos)
            for i in infos[:limit]:
                names.append({"name": i.filename,
                              "size": i.file_size,
                              "is_dir": i.filename.endswith("/")})
        elif fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
            with tarfile.open(p) as tf:
                for m in tf.getmembers()[:limit]:
                    names.append({"name": m.name, "size": m.size,
                                  "is_dir": m.isdir()})
        elif fmt in ("gzip", "bzip2", "xz"):
            names.append({"name": p.stem, "size": p.stat().st_size,
                          "is_dir": False})
        elif fmt == "7z" and self._bin7z:
            names = self._list7z(p, password=password).get(
                "entries_list", [])[:limit]
        elif fmt == "rar" and self._bin_unrar:
            names = self._list_rar(p, password=password).get(
                "entries_list", [])[:limit]
        else:
            raise ToolError(f"can't list a {fmt} archive here")
        return {"path": str(p), "format": fmt, "count": len(names),
                "entries": names, **extra}

    def _list7z(self, p: Path, *,
                password: str | None = None) -> dict[str, Any]:
        args = [self._bin7z, "l"]
        if password is not None:
            args.append(f"-p{password}")
        args.append(str(p))
        out = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60)
        entries = []
        for line in out.stdout.splitlines():
            m = re.match(r"^\s+\d+\s+\S+\s+\S+\s+(\d+|-)\s+(.+)$", line)
            if m and m.group(2):
                try:
                    entries.append({"name": m.group(2).strip(),
                                    "size": int(m.group(1))
                                    if m.group(1) != "-" else 0,
                                    "is_dir": False})
                except ValueError:
                    continue
        if out.returncode != 0 and not entries:
            blob = out.stdout + out.stderr
            if _passwordish(blob):
                raise ToolError(
                    "password required to list this 7z archive"
                    if password is None else
                    "password incorrect for this 7z archive")
            raise ToolError(
                f"7z could not list {p.name}: "
                f"{_scrub(blob, password)[:300]}")
        return {"entries": len(entries), "total_bytes":
                sum(e["size"] for e in entries),
                "entries_list": entries}

    def _list_rar(self, p: Path, *,
                  password: str | None = None) -> dict[str, Any]:
        if os.path.basename(self._bin_unrar) == "unar":
            args = ["-l", str(p)]
            if password is not None:
                args += ["-password", password]
        else:
            args = ["lt"]
            if password is not None:
                args.append(f"-p{password}")
            args.append(str(p))
        out = subprocess.run(
            [self._bin_unrar, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60)
        entries = []
        for line in out.stdout.splitlines():
            m = re.match(r"^\s*(\S.*?)\s+(\d+[-]\d+[-]\d+)\s+(\d+)\s+(\d+)",
                         line)
            if m:
                entries.append({"name": m.group(1).strip(),
                                "size": int(m.group(4)), "is_dir": False})
        if out.returncode not in (0, 1) and not entries:
            blob = out.stdout + out.stderr
            if _passwordish(blob):
                raise ToolError(
                    "password required to list this rar archive"
                    if password is None else
                    "password incorrect for this rar archive")
            raise ToolError(
                f"unrar could not list {p.name}: "
                f"{_scrub(blob, password)[:300]}")
        return {"entries": len(entries), "total_bytes":
                sum(e["size"] for e in entries),
                "entries_list": entries}

    # ── create ────────────────────────────────────────────────────────────
    def create(self, paths: list[str], dest: str,
               fmt: str = "zip", *, arc_base: str = "",
               password: str | None = None) -> dict[str, Any]:
        """Create an archive from workspace paths.

        Default member names: a file rides under its base name, a
        directory under its own name (relative to its parent).  With
        ``arc_base`` set, every member name is instead relative to
        that base — the workspace-relative convention the chat
        send-fusion (tools/archive) has always used, so a zipped
        ``docs/a.txt`` extracts back as ``docs/a.txt``.

        ``password`` creates an AES-256-encrypted zip via the 7z
        tool — the only encryption path (the stdlib zip writer
        cannot encrypt, and the tar family has no encryption at
        all); without 7z installed, or for a non-zip format, the
        refusal is named rather than the protection faked.
        """
        fmt = (fmt or "zip").lower().lstrip(".")
        if fmt not in ("zip", "tar.gz", "tar", "tar.bz2", "tar.xz"):
            raise ToolError(f"unsupported create format {fmt!r} "
                            "(zip|tar|tar.gz|tar.bz2|tar.xz)")
        resolved = [self._path(p) for p in paths]
        base = Path(arc_base).resolve() if arc_base else None
        dest_p = self._path(dest, must_exist=False)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        if password is not None:
            if fmt != "zip":
                raise ToolError(
                    "password protection is only available for zip "
                    "archives (the tar family has no encryption); "
                    f"got fmt={fmt!r}")
            if not self._bin7z:
                raise ToolError(
                    "encrypted zip creation needs the 7z tool "
                    "installed (the stdlib zip writer cannot encrypt)")
            return self._create_encrypted_zip(paths, arc_base, dest_p,
                                              password)
        tmp_dest = dest_p.with_suffix(dest_p.suffix + f".{int(time.time())}")
        try:
            if fmt == "zip":
                with zipfile.ZipFile(tmp_dest, "w",
                                     zipfile.ZIP_DEFLATED,
                                     compresslevel=9) as zf:
                    for p in resolved:
                        if p.is_dir():
                            for f in sorted(p.rglob("*")):
                                if f.is_file():
                                    zf.write(f, str(f.relative_to(
                                        base or p.parent)))
                        else:
                            zf.write(p, str(p.relative_to(base))
                                     if base else p.name)
            else:
                mode = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2",
                        "tar.xz": "w:xz"}[fmt]
                with tarfile.open(tmp_dest, mode) as tf:
                    for p in resolved:
                        tf.add(p, arcname=str(p.relative_to(base))
                               if base else p.name,
                               recursive=True)
            tmp_dest.rename(dest_p)
        except Exception:
            tmp_dest.unlink(missing_ok=True)
            raise
        return self.info(str(dest_p))

    # ── single-file compression ───────────────────────────────────────────
    def compress(self, path: str, fmt: str = "gz") -> dict[str, Any]:
        fmt = (fmt or "gz").lower().lstrip(".")
        p = self._path(path)
        dest_p = self._path(str(p) + ("." + fmt if fmt not in
                                      ("gzip", "bzip2", "xz")
                                      else {
                                          "gzip": ".gz", "bzip2": ".bz2",
                                          "xz": ".xz"}[fmt]),
                            must_exist=False)
        opener = {"gz": gzip_open, "bz2": bz2_open, "xz": lzma_open}.get(fmt)
        if opener is None:
            raise ToolError(f"unsupported compress format {fmt!r} "
                            "(gz|bz2|xz)")
        with open(p, "rb") as src, opener(dest_p) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        return self.info(str(dest_p))

    # ── append / remove members ───────────────────────────────────────────

    def _member_sources(self, paths: list[str], arc_base: str = ""
                        ) -> list[tuple[Path, str]]:
        """Resolve addable files to (source, member-name) pairs,
        deduplicated by member name (last source wins), using
        create()'s naming convention: a file rides under its base
        name (or its path relative to ``arc_base``); a directory
        contributes every file under its own name."""
        resolved = [self._path(p) for p in paths]
        base = Path(arc_base).resolve() if arc_base else None
        by_name: dict[str, Path] = {}
        for p in resolved:
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        by_name[str(f.relative_to(
                            base or p.parent)).replace("\\", "/")] = f
            else:
                by_name[str(p.relative_to(base)).replace("\\", "/")
                        if base else p.name] = p
        return [(src, arc) for arc, src in by_name.items()]

    def append(self, path: str, paths: list[str], *,
               arc_base: str = "",
               password: str | None = None) -> dict[str, Any]:
        """Add files to an EXISTING zip or tar archive.

        The result names exactly what happened: ``added`` (new
        members) and ``replaced`` (members whose name already
        existed — the old content is superseded, and it is listed
        here, never silent).  Formats that cannot take members
        refuse in words: a single-file stream holds exactly one
        file, and 7z/rar are extract-only here.
        """
        p = self._path(path)
        fmt = detect_format(p)
        if fmt in ("gzip", "bzip2", "xz"):
            raise ToolError(
                f"can't append to {p.name}: a single-file {fmt} "
                "stream holds exactly one file, not members — "
                "create a zip or tar archive instead")
        if fmt in ("7z", "rar"):
            raise ToolError(
                f"can't append to {p.name}: {fmt} is extract-only "
                "here — extract it and re-create as zip or tar")
        if fmt not in _CONTAINER_FORMATS:
            raise ToolError(
                f"can't append to {p.name}: not a recognized archive")
        additions = self._member_sources(paths, arc_base)
        if not additions:
            raise ToolError("append needs at least one file")
        if fmt == "zip":
            return self._append_zip(p, additions, password=password)
        return self._append_tar(p, fmt, additions)

    def _append_zip(self, p: Path,
                    additions: list[tuple[Path, str]], *,
                    password: str | None) -> dict[str, Any]:
        try:
            with zipfile.ZipFile(p) as zf:
                infos = zf.infolist()
        except zipfile.BadZipFile as exc:
            raise ToolError(
                f"unreadable zip archive {p.name}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ToolError(
                f"undecidable member name encoding in {p.name}: "
                f"{exc}") from exc
        existing = {i.filename for i in infos}
        srcmap = {arc: src for src, arc in additions}
        replaced = sorted(set(srcmap) & existing)
        added = sorted(set(srcmap) - existing)
        out: dict[str, Any] = {"path": str(p), "format": "zip",
                               "added": added, "replaced": replaced}
        if not replaced:
            # Native append: the archive grows in place, history
            # untouched.
            with zipfile.ZipFile(p, "a", zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                for src, arc in additions:
                    zf.write(src, arc)
            if any(i.flag_bits & 0x1 for i in infos):
                out["note"] = ("existing entries are encrypted; the "
                               "appended entries are stored "
                               "unencrypted (the stdlib zip writer "
                               "cannot encrypt)")
        else:
            # A name collision means a rebuild (zip cannot replace
            # in place): every kept member is copied across, the
            # replacements land where their predecessors sat, and
            # the new file atomically replaces the old.
            tmp = p.with_suffix(
                p.suffix + f".append-{int(time.time())}")
            pwd = (password.encode("utf-8")
                   if password is not None else None)
            try:
                with zipfile.ZipFile(p) as zin, \
                        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                                        compresslevel=9) as zout:
                    written: set[str] = set()
                    for info in zin.infolist():
                        if info.filename in written:
                            continue  # duplicate name: first copy wins
                        written.add(info.filename)
                        if info.filename in srcmap:
                            zout.write(srcmap[info.filename],
                                       info.filename)
                            continue
                        if info.is_dir():
                            zout.writestr(info.filename, b"")
                            continue
                        data = self._read_member(zin, info, pwd, p)
                        zout.writestr(
                            zipfile.ZipInfo(info.filename,
                                            info.date_time),
                            data,
                            compress_type=zipfile.ZIP_DEFLATED)
                    for src, arc in additions:
                        if arc not in existing:
                            zout.write(src, arc)
                tmp.rename(p)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        out["entries"] = self.info(str(p))["entries"]
        return out

    def _append_tar(self, p: Path, fmt: str,
                    additions: list[tuple[Path, str]]) -> dict[str, Any]:
        """Tar archives cannot grow in place (compressed ones least
        of all), so appending is a rebuild: every existing member is
        copied across, replacements land where their predecessors
        sat, additions go last, and the new file atomically replaces
        the old."""
        rmode = {"tar": "r", "tar.gz": "r:gz", "tar.bz2": "r:bz2",
                 "tar.xz": "r:xz"}[fmt]
        wmode = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2",
                 "tar.xz": "w:xz"}[fmt]
        try:
            with tarfile.open(p, rmode) as tf:
                existing = {m.name for m in tf.getmembers()}
        except (tarfile.TarError, EOFError, OSError,
                lzma.LZMAError) as exc:
            raise ToolError(
                f"unreadable tar archive {p.name}: {exc}") from exc
        srcmap = {arc: src for src, arc in additions}
        replaced = sorted(set(srcmap) & existing)
        added = sorted(set(srcmap) - existing)
        tmp = p.with_suffix(p.suffix + f".append-{int(time.time())}")
        try:
            with tarfile.open(p, rmode) as tin, \
                    tarfile.open(tmp, wmode) as tout:
                placed: set[str] = set()
                for m in tin.getmembers():
                    if m.name in srcmap:
                        if m.name not in placed:
                            tout.add(srcmap[m.name], arcname=m.name)
                            placed.add(m.name)
                        continue
                    if m.isfile():
                        tout.addfile(m, tin.extractfile(m))
                    else:
                        tout.addfile(m)
                for src, arc in additions:
                    if arc not in existing:
                        tout.add(src, arcname=arc)
            tmp.rename(p)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return {"path": str(p), "format": fmt, "added": added,
                "replaced": replaced,
                "entries": self.info(str(p))["entries"]}

    @staticmethod
    def _read_member(zin: zipfile.ZipFile, info: zipfile.ZipInfo,
                     pwd: bytes | None, archive: Path) -> bytes:
        """Read one member during a zip rebuild, translating a
        credential failure into its named ToolError."""
        try:
            return zin.read(info, pwd=pwd)
        except RuntimeError as exc:
            if info.flag_bits & 0x1 and pwd is None:
                raise ToolError(
                    f"password required to rewrite {archive.name}: "
                    f"member {info.filename!r} is encrypted") from exc
            if info.flag_bits & 0x1:
                raise ToolError(
                    f"password incorrect for {archive.name}") from exc
            raise ToolError(
                f"cannot copy member {info.filename!r} from "
                f"{archive.name}: {exc}") from exc

    def remove_members(self, path: str, names: list[str], *,
                       password: str | None = None) -> dict[str, Any]:
        """Remove named members from an existing zip or tar archive
        (a rebuild; the file is replaced atomically).  Names that
        are not members are reported under ``missing`` — never
        silently ignored."""
        p = self._path(path)
        fmt = detect_format(p)
        if fmt in ("gzip", "bzip2", "xz"):
            raise ToolError(
                f"can't remove members from {p.name}: a single-file "
                f"{fmt} stream holds exactly one file, not members")
        if fmt in ("7z", "rar"):
            raise ToolError(
                f"can't remove members from {p.name}: {fmt} is "
                "extract-only here — extract it and re-create as "
                "zip or tar")
        if fmt not in _CONTAINER_FORMATS:
            raise ToolError(
                f"can't remove members from {p.name}: not a "
                "recognized archive")
        if not names:
            raise ToolError("remove needs at least one member name")
        drop = set(names)
        tmp = p.with_suffix(p.suffix + f".remove-{int(time.time())}")
        removed: list[str] = []
        try:
            if fmt == "zip":
                pwd = (password.encode("utf-8")
                       if password is not None else None)
                try:
                    zin = zipfile.ZipFile(p)
                except UnicodeDecodeError as exc:
                    raise ToolError(
                        f"undecidable member name encoding in "
                        f"{p.name}: {exc}") from exc
                except zipfile.BadZipFile as exc:
                    raise ToolError(
                        f"unreadable zip archive {p.name}: "
                        f"{exc}") from exc
                with zin, zipfile.ZipFile(
                        tmp, "w", zipfile.ZIP_DEFLATED,
                        compresslevel=9) as zout:
                    for info in zin.infolist():
                        if info.filename in drop:
                            removed.append(info.filename)
                            continue
                        if info.is_dir():
                            zout.writestr(info.filename, b"")
                            continue
                        data = self._read_member(zin, info, pwd, p)
                        zout.writestr(
                            zipfile.ZipInfo(info.filename,
                                            info.date_time),
                            data,
                            compress_type=zipfile.ZIP_DEFLATED)
            else:
                rmode = {"tar": "r", "tar.gz": "r:gz",
                         "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[fmt]
                wmode = {"tar": "w", "tar.gz": "w:gz",
                         "tar.bz2": "w:bz2", "tar.xz": "w:xz"}[fmt]
                with tarfile.open(p, rmode) as tin, \
                        tarfile.open(tmp, wmode) as tout:
                    for m in tin.getmembers():
                        if m.name in drop:
                            removed.append(m.name)
                            continue
                        if m.isfile():
                            tout.addfile(m, tin.extractfile(m))
                        else:
                            tout.addfile(m)
            tmp.rename(p)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return {"path": str(p), "format": fmt, "removed": removed,
                "missing": sorted(drop - set(removed)),
                "entries": self.info(str(p))["entries"]}

    def _create_encrypted_zip(self, paths: list[str], arc_base: str,
                              dest_p: Path, password: str
                              ) -> dict[str, Any]:
        """Encrypted zip creation rides the 7z tool (AES-256): the
        stdlib writer cannot encrypt at all, and pretending
        otherwise would be worse than refusing.  Sources are staged
        under their member names so the archive layout matches
        create()'s exactly."""
        additions = self._member_sources(paths, arc_base)
        if not additions:
            raise ToolError("create needs at least one file")
        stage = dest_p.parent / \
            f".enc-stage-{int(time.time())}-{os.getpid()}"
        tmp_out = dest_p.with_suffix(
            dest_p.suffix + f".enc-{int(time.time())}")
        try:
            for src, arc in additions:
                staged = stage / arc
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, staged)
            tops = sorted({arc.split("/")[0] for _, arc in additions})
            out = subprocess.run(
                [self._bin7z, "a", "-tzip", "-mem=AES256",
                 f"-p{password}", str(tmp_out), *tops],
                cwd=str(stage), capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=600)
            if out.returncode != 0 or not tmp_out.exists():
                raise ToolError(
                    "7z could not create the encrypted zip: "
                    f"{_scrub(out.stdout + out.stderr, password)[:300]}")
            tmp_out.rename(dest_p)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            tmp_out.unlink(missing_ok=True)
        return self.info(str(dest_p))

    # ── extract (traversal-safe, recursive, honest partials) ─────────────
    def extract(self, path: str, dest: str = "", *,
                max_files: int = _MAX_EXTRACT_FILES,
                max_bytes: int = _MAX_EXTRACT_BYTES,
                password: str | None = None,
                max_depth: int = _MAX_NEST_DEPTH) -> dict[str, Any]:
        """Extract with skip-and-report safety semantics.

        A member whose name would escape the destination (absolute,
        ``..`` — with backslashes treated as separators, so a
        Windows-shaped traversal is caught on every host) is SKIPPED
        and named in the result's ``skipped`` list; the good members
        still land.  The same honest-partial rule covers corrupt
        members: a bad entry is named in ``skipped_details`` with its
        reason while its readable siblings extract; a whole refusal
        happens only when nothing could be extracted or a bomb guard
        trips.  The bomb guards: more than ``max_files`` members or
        more than ``max_bytes`` live-streamed bytes aborts with
        ToolError.

        ``password`` opens protected archives (zip natively, incl.
        list/digest; 7z/rar via their binaries' password flags).  A
        missing password on a fully-protected archive, and a wrong
        password anywhere, fails with a named ToolError — never a raw
        traceback, never a silent empty result.  The password is
        never logged, journaled, or persisted.

        Archives inside the archive are opened recursively, up to
        ``max_depth`` levels; the bomb budgets apply CUMULATIVELY
        across levels, and the result's ``nested`` list reports which
        member opened into what (and which were left unopened at the
        depth cap).
        """
        p = self._path(path)
        fmt = detect_format(p)
        dest_p = (self._path(dest, must_exist=False) if dest
                  else p.parent / (p.name + ".extracted"))
        dest_p.mkdir(parents=True, exist_ok=True)
        dest_root = dest_p.resolve()
        state: dict[str, Any] = {
            "written": [], "paths": [], "skipped": [], "details": [],
            "bytes": 0, "count": 0, "nested": [],
        }
        self._extract_level(p, fmt, dest_root, dest_root, state,
                            password=password, max_files=max_files,
                            max_bytes=max_bytes, depth=0,
                            max_depth=max_depth)
        skipped = state["skipped"]
        out: dict[str, Any] = {
            "path": str(p), "format": fmt, "dest": str(dest_root),
            "extracted": state["count"], "bytes": state["bytes"],
            "skipped": skipped[:50], "skipped_count": len(skipped),
            "skipped_details": state["details"][:50],
            "nested": state["nested"],
            "files": state["written"][:200]}
        if fmt == "zip":
            out["filename_encoding"] = state.get("encoding", "ascii")
        return out

    # ── extraction internals ─────────────────────────────────────────────

    @staticmethod
    def _within(root: Path, target: Path) -> bool:
        resolved = (target if target.is_absolute()
                    else root / target).resolve()
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _skip(state: dict[str, Any], name: str, reason: str) -> None:
        state["skipped"].append(name)
        state["details"].append({"name": name, "reason": reason})

    @staticmethod
    def _note(state: dict[str, Any], reason: str) -> None:
        """A failure that is not one member's skip (e.g. a stream
        dying partway) — recorded in the details only, so the
        skipped-NAMES list stays exactly the members that did not
        land."""
        state["details"].append({"name": "", "reason": reason})

    def _land(self, state: dict[str, Any], top_root: Path,
              target: Path) -> None:
        state["paths"].append(target)
        state["written"].append(str(target.relative_to(top_root)))
        state["count"] += 1

    def _extract_level(self, p: Path, fmt: str, level_root: Path,
                       top_root: Path, state: dict[str, Any], *,
                       password: str | None, max_files: int,
                       max_bytes: int, depth: int,
                       max_depth: int) -> None:
        """Extract ONE archive into ``level_root``, folding counts
        into the shared ``state`` (budgets cumulative across levels),
        then recurse into archive members while depth lasts."""
        start = len(state["paths"])
        if fmt == "zip":
            self._extract_zip(p, level_root, top_root, state,
                              password=password, max_files=max_files,
                              max_bytes=max_bytes,
                              top_level=depth == 0)
        elif fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
            self._extract_tar(p, fmt, level_root, top_root, state,
                              max_files=max_files, max_bytes=max_bytes)
        elif fmt == "7z" and self._bin7z:
            self._extract_external(p, "7z", level_root, top_root, state,
                                   password=password,
                                   max_files=max_files,
                                   max_bytes=max_bytes)
        elif fmt == "rar" and self._bin_unrar:
            self._extract_external(p, "rar", level_root, top_root,
                                   state, password=password,
                                   max_files=max_files,
                                   max_bytes=max_bytes)
        elif fmt in ("gzip", "bzip2", "xz"):
            self._extract_single(p, fmt, level_root, top_root, state,
                                 max_bytes=max_bytes)
        else:
            raise ToolError(f"can't extract format {fmt!r} here"
                            + ("" if fmt not in ("7z", "rar")
                               else " (install 7z / unrar)"))
        landed = list(state["paths"][start:])
        for f in landed:
            if not f.is_file():
                continue
            nfmt = detect_format(f)
            if nfmt not in _CONTAINER_FORMATS:
                continue
            rel = str(f.relative_to(top_root))
            if depth >= max_depth:
                state["nested"].append(
                    {"member": rel, "format": nfmt, "opened": False,
                     "reason": f"depth cap ({max_depth}) reached"})
                continue
            sub = f.parent / (f.name + ".extracted")
            sub.mkdir(parents=True, exist_ok=True)
            rec: dict[str, Any] = {
                "member": rel, "format": nfmt,
                "into": str(sub.relative_to(top_root)), "opened": True,
                "depth": depth + 1}
            c0, s0 = state["count"], len(state["skipped"])
            try:
                self._extract_level(f, nfmt, sub, top_root, state,
                                    password=password,
                                    max_files=max_files,
                                    max_bytes=max_bytes,
                                    depth=depth + 1,
                                    max_depth=max_depth)
                rec["extracted"] = state["count"] - c0
                rec["skipped_count"] = len(state["skipped"]) - s0
            except _BombError:
                raise
            except ToolError as exc:
                # A nested archive that refuses (password, unreadable)
                # does not sink its parent: the member file itself
                # landed, and the refusal is reported on its record.
                rec["error"] = str(exc)
            state["nested"].append(rec)

    def _extract_zip(self, p: Path, level_root: Path, top_root: Path,
                     state: dict[str, Any], *, password: str | None,
                     max_files: int, max_bytes: int,
                     top_level: bool) -> None:
        try:
            zf = zipfile.ZipFile(p)
        except UnicodeDecodeError as exc:
            # The central directory is parsed at construction: a
            # name that claims UTF-8 but carries invalid bytes
            # surfaces here, not at infolist().
            raise ToolError(
                f"undecidable member name encoding in {p.name}: "
                f"{exc}") from exc
        except zipfile.BadZipFile as exc:
            raise ToolError(
                f"unreadable zip archive {p.name}: {exc}") from exc
        with zf:
            infos = zf.infolist()
            if top_level:
                state["encoding"] = _zip_name_encoding(infos)
            if any(i.compress_type == 99 for i in infos):
                # AES-encrypted entries (method 99): the stdlib
                # reader cannot open them at all.  With the 7z tool
                # installed the archive is handed to it whole (it
                # reads zip natively); without it the AES entries
                # are skipped below, named — never silently mangled.
                if self._bin7z:
                    self._extract_external(
                        p, "7z", level_root, top_root, state,
                        password=password, max_files=max_files,
                        max_bytes=max_bytes)
                    return
                aes_note = ("AES-encrypted entry — the stdlib zip "
                            "reader cannot open AES; install 7z")
            else:
                aes_note = ""
            file_infos = [i for i in infos
                          if not i.filename.endswith("/")]
            encrypted = [i for i in file_infos if i.flag_bits & 0x1]
            if encrypted and password is None \
                    and len(encrypted) == len(file_infos):
                raise ToolError(
                    f"password required to extract {p.name} "
                    "(every entry is password-protected)")
            pwd = (password.encode("utf-8")
                   if password is not None else None)
            for info in infos:
                raw = info.filename
                # Backslashes are separators too: a member named
                # ``..\evil`` is a traversal on the Windows host even
                # when this process runs on POSIX, so the safety
                # decision (and the on-disk layout) uses the
                # normalized name on every platform; the skipped
                # report keeps the name as archived.
                name = raw.replace("\\", "/")
                if "\x00" in name:
                    self._skip(state, raw,
                               "member name contains a NUL byte — "
                               "not writable on a filesystem")
                    continue
                if not self._within(level_root, Path(name)):
                    self._skip(state, raw,
                               "path traversal (escapes the "
                               "destination)")
                    continue
                target = level_root / name
                if name.endswith("/"):
                    try:
                        target.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        self._skip(state, raw,
                                   f"cannot create directory: {exc}")
                    continue
                if info.compress_type == 99:
                    self._skip(state, raw, aes_note)
                    continue
                is_enc = bool(info.flag_bits & 0x1)
                if is_enc and pwd is None:
                    self._skip(state, raw, "password required")
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, pwd=pwd) as src, \
                            open(target, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            state["bytes"] += len(chunk)
                            if state["bytes"] > max_bytes:
                                raise _BombError(
                                    "extraction aborted: exceeds "
                                    f"{max_bytes} bytes "
                                    "(zip-bomb guard)")
                            dst.write(chunk)
                except _BombError:
                    _silent_unlink(target)
                    raise
                except RuntimeError as exc:
                    _silent_unlink(target)
                    if is_enc and pwd is not None:
                        raise ToolError(
                            f"password incorrect for {p.name}: "
                            f"{exc}") from exc
                    self._skip(state, raw, f"unreadable entry: {exc}")
                    continue
                except (zipfile.BadZipFile, EOFError, zlib.error,
                        lzma.LZMAError) as exc:
                    _silent_unlink(target)
                    if is_enc and pwd is not None and \
                            isinstance(exc, zipfile.BadZipFile):
                        # The password check-byte passed by luck,
                        # then the integrity check failed: the
                        # password is wrong, not the archive.
                        raise ToolError(
                            f"password incorrect for {p.name} "
                            f"(entry {raw!r} failed its integrity "
                            "check)") from exc
                    self._skip(state, raw, f"corrupt entry: {exc}")
                    continue
                except (OSError, ValueError,
                        NotImplementedError) as exc:
                    _silent_unlink(target)
                    self._skip(state, raw,
                               f"cannot write entry: {exc}")
                    continue
                self._land(state, top_root, target)
                if state["count"] > max_files:
                    raise _BombError(
                        "extraction aborted: more than "
                        f"{max_files} files (zip-bomb guard)")

    def _extract_tar(self, p: Path, fmt: str, level_root: Path,
                     top_root: Path, state: dict[str, Any], *,
                     max_files: int, max_bytes: int) -> None:
        level_start = state["count"]
        current = ""
        try:
            with tarfile.open(p) as tf:
                for m in tf:
                    current = m.name
                    name = m.name.replace("\\", "/")
                    if "\x00" in name:
                        self._skip(state, m.name,
                                   "member name contains a NUL byte — "
                                   "not writable on a filesystem")
                        continue
                    if not self._within(level_root, Path(name)):
                        self._skip(state, m.name,
                                   "path traversal (escapes the "
                                   "destination)")
                        continue
                    target = level_root / name
                    if m.isdir():
                        try:
                            target.mkdir(parents=True, exist_ok=True)
                        except OSError as exc:
                            self._skip(state, m.name,
                                       f"cannot create directory: {exc}")
                        continue
                    if m.issym() or m.islnk():
                        # link targets must stay inside
                        if m.issym() and not self._within(
                                level_root,
                                Path(m.linkname.replace("\\", "/"))):
                            self._skip(state, m.name,
                                       "link target escapes the "
                                       "destination")
                            continue
                        try:
                            target.parent.mkdir(parents=True,
                                                exist_ok=True)
                            tf.extract(m, level_root, filter="data")
                        except TypeError:  # Python < 3.12
                            try:
                                tf.extract(m, level_root)
                            except Exception as exc:  # noqa: BLE001
                                self._skip(state, m.name,
                                           f"link failed: {exc}")
                        except Exception as exc:  # noqa: BLE001
                            self._skip(state, m.name,
                                       f"link failed: {exc}")
                        continue
                    try:
                        fobj = tf.extractfile(m)
                    except (tarfile.TarError, EOFError, OSError,
                            zlib.error, lzma.LZMAError) as exc:
                        self._skip(state, m.name,
                                   f"corrupt entry: {exc}")
                        continue
                    if fobj is None:
                        self._skip(state, m.name,
                                   "member has no readable data")
                        continue
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with open(target, "wb") as dst:
                            while chunk := fobj.read(1 << 20):
                                state["bytes"] += len(chunk)
                                if state["bytes"] > max_bytes:
                                    raise _BombError(
                                        "extraction aborted: exceeds "
                                        f"{max_bytes} bytes "
                                        "(zip-bomb guard)")
                                dst.write(chunk)
                    except _BombError:
                        _silent_unlink(target)
                        raise
                    except (tarfile.TarError, EOFError, zlib.error,
                            lzma.LZMAError) as exc:
                        _silent_unlink(target)
                        self._skip(state, m.name,
                                   f"corrupt entry: {exc}")
                        continue
                    except (OSError, ValueError) as exc:
                        _silent_unlink(target)
                        self._skip(state, m.name,
                                   f"cannot write entry: {exc}")
                        continue
                    self._land(state, top_root, target)
                    if state["count"] > max_files:
                        raise _BombError(
                            "extraction aborted: more than "
                            f"{max_files} files (zip-bomb guard)")
                if fmt == "tar":
                    # A corrupt header mid-file ends tarfile's
                    # iteration SILENTLY (it reads as end-of-archive)
                    # — but a clean tar ends on zero blocks, so for
                    # a plain seekable tar the loss is checkable: if
                    # the block where iteration stopped is not the
                    # zero end-marker, members were lost, and that
                    # is reported rather than silent.
                    try:
                        with open(p, "rb") as fh:
                            fh.seek(tf.offset)
                            block = fh.read(512)
                    except (OSError, AttributeError):
                        block = b""
                    if block and any(block):
                        self._note(
                            state,
                            "archive stream ended early: unread "
                            "non-empty data after the last member "
                            "(corrupt or truncated tail)")
        except _BombError:
            raise
        except (tarfile.TarError, EOFError, OSError, zlib.error,
                lzma.LZMAError) as exc:
            # The stream itself died mid-iteration (a corrupt
            # header, a truncated compressed stream): what
            # already landed at this level stays landed and the
            # failure is named on the member being read.  Only a
            # level that produced nothing is a whole refusal.
            if state["count"] == level_start:
                raise ToolError(
                    f"unreadable tar archive {p.name}: "
                    f"{exc}") from exc
            self._note(state,
                       "archive stream failed partway (members "
                       f"after {current!r} unread): {exc}")

    def _extract_single(self, p: Path, fmt: str, level_root: Path,
                        top_root: Path, state: dict[str, Any], *,
                        max_bytes: int) -> None:
        opener = {"gzip": gzip_open_r, "bzip2": bz2_open_r,
                  "xz": lzma_open_r}[fmt]
        out_p = level_root / p.stem
        try:
            with open(p, "rb") as src, open(out_p, "wb") as dst:
                stream = opener(src)
                while chunk := stream.read(1 << 20):
                    state["bytes"] += len(chunk)
                    if state["bytes"] > max_bytes:
                        raise _BombError(
                            "extraction aborted: exceeds "
                            f"{max_bytes} bytes (zip-bomb guard)")
                    dst.write(chunk)
        except _BombError:
            _silent_unlink(out_p)
            raise
        except (EOFError, OSError, lzma.LZMAError,
                zlib.error) as exc:
            _silent_unlink(out_p)
            raise ToolError(
                f"unreadable {fmt} stream {p.name}: {exc}") from exc
        self._land(state, top_root, out_p)

    def _extract_external(self, p: Path, engine: str,
                          level_root: Path, top_root: Path,
                          state: dict[str, Any], *,
                          password: str | None, max_files: int,
                          max_bytes: int) -> None:
        """Extract through the external tool (7z / unrar) under the
        same bomb budgets the stdlib paths enforce: the archive is
        pre-listed and refused in words when the listing alone
        exceeds a budget, and the landed output is re-measured
        afterwards, since a listing can lie."""
        if engine == "7z":
            listing = self._list7z(p, password=password)
        else:
            listing = self._list_rar(p, password=password)
        entries = listing.get("entries_list", [])
        if state["count"] + len(entries) > max_files:
            raise ToolError(
                f"{engine} archive lists {len(entries)} files — over "
                f"the {max_files} file cap (zip-bomb guard); "
                "refusing to extract")
        total = int(listing.get("total_bytes") or 0)
        if state["bytes"] + total > max_bytes:
            raise ToolError(
                f"{engine} archive would extract {total} bytes — "
                f"over the {max_bytes}-byte cap (zip-bomb guard); "
                "refusing to extract")
        before = {f for f in level_root.rglob("*")}
        if engine == "7z":
            cmd = [self._bin7z, "x", f"-o{level_root}", "-y"]
            if password is not None:
                cmd.append(f"-p{password}")
            cmd.append(str(p))
            ok_codes = (0,)
        elif os.path.basename(self._bin_unrar) == "unar":
            cmd = [self._bin_unrar, "-o", str(level_root)]
            if password is not None:
                cmd += ["-password", password]
            cmd.append(str(p))
            ok_codes = (0,)
        else:
            cmd = [self._bin_unrar, "x", "-y"]
            if password is not None:
                cmd.append(f"-p{password}")
            cmd += [str(p), f"{level_root}/"]
            ok_codes = (0, 1)  # 1 = warnings (ok)
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        landed: list[Path] = []
        for f in level_root.rglob("*"):
            if f in before:
                continue
            if f.is_symlink():
                try:
                    f.resolve().relative_to(level_root)
                except ValueError:
                    f.unlink(missing_ok=True)
                    self._skip(state, str(f.relative_to(top_root)),
                               "symlink escapes the destination — "
                               "removed")
                    continue
            if f.is_file() and not f.is_symlink():
                landed.append(f)
        landed_bytes = sum(f.stat().st_size for f in landed)
        if state["count"] + len(landed) > max_files or \
                state["bytes"] + landed_bytes > max_bytes:
            for f in landed:
                f.unlink(missing_ok=True)
            raise _BombError(
                "extraction aborted: external tool extracted "
                f"{len(landed)} files / {landed_bytes} bytes, over "
                "the budget (zip-bomb guard)")
        state["bytes"] += landed_bytes
        for f in sorted(landed):
            self._land(state, top_root, f)
        if out.returncode not in ok_codes:
            blob = out.stdout + out.stderr
            if _passwordish(blob):
                raise ToolError(
                    f"password required to extract this {engine} "
                    "archive" if password is None else
                    f"password incorrect for this {engine} archive")
            if not landed:
                raise ToolError(
                    f"{engine} extraction failed: "
                    f"{_scrub(blob, password)[:300]}")
            self._note(state,
                       f"{engine} reported errors; partial "
                       f"extraction: {_scrub(blob, password)[:200]}")


    def digest(self, path: str, *, max_files: int = 200,
               max_bytes: int = 500_000,
               password: str | None = None) -> dict[str, Any]:
        """Crack the archive open and feed every text document into the
        knowledge graph + long-term memory.  Returns counts + what was
        curated.  Binary/media files are counted but not parsed."""
        p = self._path(path)
        fmt = detect_format(p)
        dest = self._path(f".digest-{int(time.time() * 1000)}-{os.getpid()}",
                          must_exist=False)
        n = 1
        while dest.exists():
            dest = self._path(f".digest-{int(time.time() * 1000)}-{os.getpid()}-{n}",
                              must_exist=False)
            n += 1
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if fmt in ("gzip", "bzip2", "xz"):
                opener = {"gzip": gzip_open_r, "bzip2": bz2_open_r,
                          "xz": lzma_open_r}[fmt]
                with open(p, "rb") as src, open(dest / p.stem, "wb") as dst:
                    shutil.copyfileobj(opener(src), dst, length=1 << 20)
                extracted_root = dest
            else:
                self.extract(str(p), dest=str(dest), password=password)
                extracted_root = dest
            files = _extract_text_files(extracted_root, max_files=max_files,
                                        max_bytes=max_bytes)
        finally:
            # keep the extraction on disk (it's inside the workspace and the
            # owner may want the files); record where they landed
            pass
        text_blob = "\n\n".join(f"=== {rel} ===\n{t[:200_000]}"
                                 for rel, t in files)
        out: dict[str, Any] = {
            "path": str(p), "format": fmt,
            "extracted_to": str(dest),
            "text_files": len(files),
            "chars": len(text_blob),
            "sample": [rel for rel, _ in files[:20]],
        }
        self._ingest_files(out, files, label=os.path.basename(str(p)),
                           what=(f"Archive {os.path.basename(str(p))} "
                                 f"({fmt})"),
                           extra={"path": str(p), "format": fmt})
        return out

    def _ingest_files(self, out: dict[str, Any],
                      files: list[tuple[str, str]], *, label: str,
                      what: str,
                      extra: dict[str, Any] | None = None) -> None:
        """Shared tail of every digest (archive or directory): the text
        files become KG nodes/links + a long-term memory episode.  Mutates
        ``out`` with kg / kg_error / memory_stored."""
        text_blob = "\n\n".join(f"=== {rel} ===\n{t[:200_000]}"
                                for rel, t in files)
        out["chars"] = len(text_blob)
        out["text_files"] = len(files)
        out["sample"] = [rel for rel, _ in files[:20]]
        # knowledge graph
        kg_added = {"added_nodes": 0, "added_links": 0}
        db = getattr(self.context, "db", None)
        if db is not None and text_blob.strip():
            try:
                from .agents.kg import KnowledgeGraph

                res = KnowledgeGraph(db).curate_from_text(
                    text_blob, source=f"digest:{label}", max_items=60)
                kg_added = {"added_nodes": res.get("added_nodes", 0),
                            "added_links": res.get("added_links", 0)}
            except Exception as exc:  # noqa: BLE001
                out["kg_error"] = str(exc)
        out["kg"] = kg_added
        # long-term memory (episode: what was in this corpus)
        mem = getattr(self.context, "memory", None)
        if mem is not None and files:
            try:
                preview = " | ".join(f[0] for f in files[:8])
                metadata = {"files": len(files)}
                metadata.update(extra or {})
                mem.remember(
                    f"{what}: {len(files)} text files, "
                    f"{len(text_blob):,} chars. Files: {preview}",
                    kind="episode", importance=0.5,
                    source="archive-digest", metadata=metadata)
                out["memory_stored"] = True
            except Exception:  # noqa: BLE001
                out["memory_stored"] = False
        else:
            out["memory_stored"] = False

    def digest_directory(self, path: str, *, max_files: int = 500,
                         max_bytes: int = 500_000) -> dict[str, Any]:
        """Walk a WHOLE DIRECTORY (not just archives) and feed every text
        file into the knowledge graph + long-term memory.  Skips binary
        and non-text files (see _TEXT_EXTS); returns the same shape as
        :meth:`digest` with format='directory'."""
        p = self._path(path)
        if not p.is_dir():
            raise ToolError(
                f"{p} is not a directory (use digest for archives, "
                "or point at a folder)")
        files = _extract_text_files(p, max_files=max_files,
                                    max_bytes=max_bytes)
        out: dict[str, Any] = {"path": str(p), "format": "directory",
                               "extracted_to": ""}
        self._ingest_files(out, files, label=p.name,
                           what=f"Directory {p.name}",
                           extra={"path": str(p), "kind": "directory"})
        return out


#: extensions counted as readable text for digest()
_TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".py", ".js",
              ".sh", ".yml", ".yaml", ".toml", ".html", ".htm", ".xml",
              ".log", ".ini", ".cfg", ".conf", ".c", ".h", ".cpp", ".rs",
              ".go", ".rb", ".pl", ".php", ".sql", ".ipynb"}

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", text)


#: directory names never worth digesting
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              "build", "dist", ".next", ".nuxt", ".output", ".cache",
              ".local", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
              ".ruff_cache", ".turbo", ".parcel-cache", "coverage",
              "target", "out", ".svelte-kit"}


def _extract_text_files(root: Path, *, max_files: int = 200,
                        max_bytes: int = 500_000) -> list[tuple[str, str]]:
    """(relative path, text) for every readable text file under root
    (skipping VCS/dependency/cache directories)."""
    out: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file() or p.suffix.lower() not in _TEXT_EXTS:
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if p.suffix.lower() in (".html", ".htm", ".xml"):
            text = _strip_html(text)
        out.append((str(p.relative_to(root)), text))
        if len(out) >= max_files:
            break
    return out


def gzip_open(path: Path):
    import gzip

    return gzip.open(path, "wb")


def bz2_open(path: Path):
    return bz2.open(path, "wb")


def lzma_open(path: Path):
    return lzma.open(path, "wb")


def gzip_open_r(path):
    import gzip

    return gzip.open(path, "rb")


def bz2_open_r(path):
    return bz2.open(path, "rb")


def lzma_open_r(path):
    return lzma.open(path, "rb")


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "archive",
        description=(
            "Universal zip/unzip: identify (by magic bytes), list, create "
            "(zip/tar/tar.gz/tar.bz2/tar.xz), compress single files "
            "(gz/bz2/xz), and safely extract zip/tar*/gz/bz2/xz (+7z/rar "
            "when the tools are installed). Extraction skips unsafe or "
            "corrupt entries and reports them in the result with their "
            "reasons (never silently); an archive is refused only when "
            "nothing in it can be extracted or a bomb guard trips. "
            "Extraction recurses into nested archives up to a depth "
            "cap under the same cumulative budgets. Password-protected "
            "archives open with password= (never stored); a wrong or "
            "missing password fails with a named error. action=info "
            "(path) | list (path) | extract (path, dest, password) | "
            "digest (path, password) - extract an archive (or walk a "
            "whole DIRECTORY) and curate every text document into the "
            "knowledge graph + memory | create (paths, dest, fmt, "
            "password) | append (path=archive, paths=files) - add "
            "members to an existing zip/tar archive, naming what was "
            "added and what was replaced | remove (path=archive, "
            "paths=member names) | compress (path, fmt)."
        ),
        capability=Capability.FS_WRITE,
    )
    def archive(action: str = "info", path: str = "", dest: str = "",
                paths: str = "", fmt: str = "zip",
                password: str = "") -> dict[str, Any]:
        a = Archivist(context)
        pwd = password or None
        if action == "info":
            return a.info(path, password=pwd)
        if action == "list":
            return a.list(path, password=pwd)
        if action == "extract":
            out = a.extract(path, dest=dest, password=pwd)
            if not out.get("extracted") and out.get("skipped_count"):
                # Unified semantics (WS1-P1): bad entries are
                # skipped + reported, and the whole archive is
                # refused only when nothing in it could be
                # extracted — the same rule the tools/archive
                # send-fusion surface applies.  The refusal names
                # the recorded reasons, not just the entries.
                details = out.get("skipped_details") or []
                if details:
                    shown = "; ".join(
                        f"{d['name']!r}: {d['reason']}"
                        for d in details[:5])
                else:
                    shown = ", ".join(
                        repr(s) for s in out["skipped"][:5])
                raise ToolError(
                    "archive extract: nothing could be extracted "
                    f"({out['skipped_count']} skipped: {shown})")
            return out
        if action == "digest":
            from .tools.filesystem import safe_path

            p = safe_path(context, path, must_exist=True)
            if p.is_dir():
                return a.digest_directory(path)
            return a.digest(path, password=pwd)
        if action == "create":
            items = [x for x in (path, *paths.split("|")) if x.strip()]
            if not dest:
                raise ToolError("archive create needs dest=")
            return a.create(items, dest, fmt=fmt, password=pwd)
        if action == "append":
            items = [x for x in paths.split("|") if x.strip()]
            if not items:
                raise ToolError(
                    "archive append needs paths= (files to add, "
                    "|-separated)")
            return a.append(path, items, password=pwd)
        if action == "remove":
            names = [x for x in paths.split("|") if x.strip()]
            if not names:
                raise ToolError(
                    "archive remove needs paths= (member names, "
                    "|-separated)")
            return a.remove_members(path, names, password=pwd)
        if action == "compress":
            return a.compress(path, fmt=fmt)
        raise ToolError(f"unknown archive action {action!r}")


# ── Tier 3: magic-byte detection + traversal-safe extraction helpers ──────
# Added without touching existing APIs.  ``detect_format()`` above mixes
# extension hints with magic bytes; the helpers below are the pure
# layers a downloader / source adapter should call: identify bytes by
# magic alone, validate member names before extraction, and extract
# through a pre-flight safety check that refuses unsafe archives
# instead of skipping members mid-stream.

#: (magic bytes, format, offset) — pure magic-byte identification.
#: tar's ``ustar`` lives at offset 257; everything else is at 0.
MAGIC_SIGNATURES: tuple[tuple[bytes, str, int], ...] = (
    (b"PK\x03\x04", "zip", 0),
    (b"PK\x05\x06", "zip", 0),          # empty zip
    (b"\x1f\x8b", "gzip", 0),
    (b"BZh", "bzip2", 0),
    (b"\xfd7zXZ\x00", "xz", 0),
    (b"7z\xbc\xaf\x27\x1c", "7z", 0),
    (b"Rar!\x1a\x07\x00", "rar", 0),
    (b"Rar!\x1a\x07\x01\x00", "rar", 0),
    (b"\x28\xb5\x2f\xfd", "zstd", 0),
    (b"ustar", "tar", 257),
)


def detect_magic_bytes(data: bytes) -> str:
    """Identify bytes by magic alone — no filename, no extension.

    Returns the format (``"zip"`` / ``"gzip"`` / ``"tar"`` / ...) or
    ``"unknown"``.  A gzip stream is reported as ``"gzip"`` even when
    it wraps a tarball: distinguishing ``.tar.gz`` needs the filename,
    which is :func:`detect_file_magic`'s job.
    """
    head = bytes(data or b"")[:512]
    for magic, fmt, offset in MAGIC_SIGNATURES:
        if len(head) >= offset + len(magic) \
                and head[offset:offset + len(magic)] == magic:
            return fmt
    return "unknown"


def detect_file_magic(path: str | Path) -> str:
    """Identify a file by magic bytes, with the extension only as the
    tarball hint (``.tar.gz`` / ``.tgz`` / ``.tar.bz2`` / ``.tar.xz``).
    Reads at most 512 bytes.  ``"unknown"`` when unreadable."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(512)
    except OSError:
        return "unknown"
    fmt = detect_magic_bytes(head)
    name = p.name.lower()
    if fmt == "gzip" and name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if fmt == "bzip2" and name.endswith((".tar.bz2", ".tbz2")):
        return "tar.bz2"
    if fmt == "xz" and name.endswith((".tar.xz", ".txz")):
        return "tar.xz"
    return fmt


def is_safe_member_name(name: str) -> tuple[bool, str]:
    """Validate one archive member name before extraction.

    Returns ``(True, "")`` when safe, ``(False, reason)`` when the
    member must never be written: absolute paths, drive prefixes,
    ``..`` escapes, NUL bytes, empty names, and — for the pre-flight
    check — symlink-style trickery is caught by the extractor itself;
    this gate covers the name layer.
    """
    if not name:
        return False, "empty member name"
    if "\x00" in name:
        return False, "NUL byte in member name"
    if name.startswith(("/", "\\")):
        return False, f"absolute path: {name!r}"
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return False, f"drive-prefixed path: {name!r}"
    parts = name.replace("\\", "/").split("/")
    if ".." in parts:
        return False, f"path traversal: {name!r}"
    return True, ""


def validate_archive_members(names: list[str]) -> dict[str, Any]:
    """Pre-flight check for a member-name list. Returns
    ``{"safe": [...], "unsafe": [{"name", "reason"}]}`` — the caller
    refuses the whole archive when ``unsafe`` is non-empty."""
    safe: list[str] = []
    unsafe: list[dict[str, str]] = []
    for name in names:
        ok, reason = is_safe_member_name(name)
        if ok:
            safe.append(name)
        else:
            unsafe.append({"name": name, "reason": reason})
    return {"safe": safe, "unsafe": unsafe, "total": len(names)}


def extract_for_downloader(archivist: "Archivist", path: str,
                           dest: str = "") -> dict[str, Any]:
    """Wire helper for the downloader + source adapters: identify an
    archive by magic, pre-flight its member names, and extract only
    when every member is safe.

    Unlike :meth:`Archivist.extract` (skip-and-report per member), this
    helper refuses the whole archive when ANY member is unsafe — a
    downloaded archive with one traversal entry is treated as hostile,
    not partially extracted.  Returns the extraction dict with an added
    ``safety`` block (``{"checked": n, "unsafe": [...]}``), or a blocked
    dict with ``extracted: 0`` when refused.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    fmt = detect_file_magic(p)
    try:
        listing = archivist.list(path)
        names = [e.get("name", "") for e in listing.get("entries", [])]
    except Exception as exc:  # noqa: BLE001 — listing failure is a refusal
        return {"path": str(p), "format": fmt, "extracted": 0,
                "safety": {"checked": 0,
                           "unsafe": [{"name": "",
                                       "reason": f"could not list archive: {exc}"}]},
                "blocked": True,
                "reason": f"could not list archive members: {exc}"}
    verdict = validate_archive_members(names)
    if verdict["unsafe"]:
        return {"path": str(p), "format": fmt, "extracted": 0,
                "safety": {"checked": len(names),
                           "unsafe": verdict["unsafe"]},
                "blocked": True,
                "reason": (f"refused: {len(verdict['unsafe'])} unsafe "
                           f"member(s), e.g. "
                           f"{verdict['unsafe'][0]['name']!r}")}
    result = archivist.extract(path, dest=dest)
    result["safety"] = {"checked": len(names), "unsafe": []}
    result["blocked"] = False
    return result


__all__ += ["MAGIC_SIGNATURES", "detect_magic_bytes", "detect_file_magic",
            "is_safe_member_name", "validate_archive_members",
            "extract_for_downloader"]


# ── bundle seam (WS1-P5) ─────────────────────────────────────────────
#
# ONE bundle format + writer/reader for every family that packs
# files into a single transportable unit: artifact export
# (storage/artifacts), database-backup blob sidecars
# (storage/backup), OS snapshot export (os/snapshots) and backup
# generations (backup/).  Before P5 each family hand-rolled its own
# tar/zip writer and its own manifest shape, sharing no code path.
#
# A bundle is a zip or tar-family archive whose members are joined
# by a manifest member (``bundle.manifest.json``): JSON, versioned,
# carrying every member's name, size and SHA-256 plus who made the
# bundle and when.  A family's own domain manifest may ride along
# as an ordinary member (the artifact store's ``manifest.json``);
# the bundle manifest is the cross-family contract — the shape a
# foreign reader (e.g. GoldenSwing harvest bundles, WS1-P7)
# targets.
#
# Backward compatibility is a hard rule: bundles written before
# the seam exist (no bundle manifest, each family's legacy layout)
# and MUST still open.  The reader therefore treats the bundle
# manifest as an overlay: present, it is authoritative; absent,
# the member list is synthesized from the container itself, and a
# family manifest (``manifest.json``) when one is visible rides
# along in the result as ``family_manifest``.  Directory bundles —
# a snapshot dir, a backup generation dir — open through the same
# reader, their family manifests synthesized the same way.

BUNDLE_FORMAT = "nomorals-bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle.manifest.json"

_BUNDLE_TAR_WRITE = {"tar": "w", "tar.gz": "w:gz",
                     "tar.bz2": "w:bz2", "tar.xz": "w:xz"}
_BUNDLE_CONTAINERS = ("zip", "tar", "tar.gz", "tar.bz2", "tar.xz")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file — the seam's one hash helper."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_manifest(members: Any, *, created_by: str, kind: str = "",
                    created_at: float | None = None,
                    extra: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Build the standard bundle manifest dict.

    ``members`` is an iterable of dicts with ``name`` / ``size`` /
    ``sha256`` (``is_dir`` optional).  The shape is the contract:
    ``format`` + ``version`` identify it, ``created_by`` /
    ``created_at`` / ``kind`` say who made it and what family it
    belongs to, ``members`` is the per-file truth a verifier checks
    bytes against.  ``extra`` may add top-level keys but never
    override the core ones.
    """
    entries = [{
        "name": str(m["name"]),
        "size": int(m.get("size", 0)),
        "sha256": str(m.get("sha256", "")),
        "is_dir": bool(m.get("is_dir", False)),
    } for m in members]
    out: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_FORMAT_VERSION,
        "created_by": str(created_by),
        "created_at": float(created_at if created_at is not None
                            else time.time()),
        "kind": str(kind),
        "members": entries,
    }
    if extra:
        for key, value in extra.items():
            if key not in out:
                out[key] = value
    return out


def create_bundle(dest: str | Path, members: Any, *,
                  created_by: str, kind: str = "",
                  container: str = "tar.gz",
                  created_at: float | None = None) -> dict[str, Any]:
    """Write a bundle to ``dest`` and return its manifest.

    ``members`` is an iterable of ``(name, source)`` pairs: source
    is a file path (its bytes ride), a directory path (a directory
    entry rides — how an empty directory survives a round trip),
    or raw bytes.  Member names must be safe (the Archivist's
    member-name gate), unique, and may not claim the reserved
    manifest name.  The bundle manifest is written LAST, so a
    reader that sees it can trust it describes the whole archive;
    the file is published by atomic rename, like
    :meth:`Archivist.create`.
    """
    dest_p = Path(dest)
    container = (container or "tar.gz").lower().lstrip(".")
    if container == "tgz":
        container = "tar.gz"
    if container not in _BUNDLE_CONTAINERS:
        raise ToolError(f"unsupported bundle container {container!r} "
                        f"(one of {', '.join(_BUNDLE_CONTAINERS)})")
    prepared: list[tuple[str, bool, int, str, Any]] = []
    seen: set[str] = set()
    for name, source in members:
        norm = str(name).replace("\\", "/").rstrip("/")
        ok, reason = is_safe_member_name(norm)
        if not ok:
            raise ToolError(f"refusing bundle member {name!r}: {reason}")
        if norm == BUNDLE_MANIFEST_NAME:
            raise ToolError(f"bundle member name {norm!r} is reserved "
                            "for the bundle manifest")
        if norm in seen:
            raise ToolError(f"duplicate bundle member {norm!r}")
        seen.add(norm)
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
            prepared.append((norm, False, len(data),
                             _sha256_bytes(data), data))
            continue
        src = Path(source)
        if src.is_dir():
            prepared.append((norm, True, 0, _sha256_bytes(b""), src))
        elif src.is_file():
            prepared.append((norm, False, src.stat().st_size,
                             sha256_file(src), src))
        else:
            raise ToolError(f"bundle member source missing: {src}")
    manifest = bundle_manifest(
        [{"name": n, "size": s, "sha256": h, "is_dir": d}
         for n, d, s, h, _src in prepared],
        created_by=created_by, kind=kind, created_at=created_at)
    manifest_bytes = json.dumps(manifest, indent=2,
                                ensure_ascii=False).encode("utf-8")
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_p.with_suffix(
        dest_p.suffix + f".bundle-{int(time.time())}-{os.getpid()}")
    try:
        if container == "zip":
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                for name, is_dir, _s, _h, src in prepared:
                    if is_dir:
                        zf.writestr(zipfile.ZipInfo(name + "/"), b"")
                    elif isinstance(src, bytes):
                        zf.writestr(name, src)
                    else:
                        zf.write(src, name)
                zf.writestr(BUNDLE_MANIFEST_NAME, manifest_bytes)
        else:
            with tarfile.open(tmp, _BUNDLE_TAR_WRITE[container]) as tf:
                for name, is_dir, _s, _h, src in prepared:
                    if is_dir:
                        info = tarfile.TarInfo(name)
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tf.addfile(info)
                    elif isinstance(src, bytes):
                        info = tarfile.TarInfo(name)
                        info.size = len(src)
                        tf.addfile(info, io.BytesIO(src))
                    else:
                        info = tf.gettarinfo(str(src), arcname=name)
                        with open(src, "rb") as fh:
                            tf.addfile(info, fh)
                info = tarfile.TarInfo(BUNDLE_MANIFEST_NAME)
                info.size = len(manifest_bytes)
                tf.addfile(info, io.BytesIO(manifest_bytes))
        tmp.rename(dest_p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return manifest


class _BundleArchive:
    """Member reader over one zip/tar container file (seam-internal).

    Normalizes both stdlib readers to one shape: ``entries()``
    yields ``{"name", "size", "is_dir", "kind"}`` dicts (kind is
    ``"file"`` / ``"dir"`` / ``"other"`` — a tar link or device is
    "other", and extraction refuses it), ``read`` / ``stream``
    fetch a member's bytes by normalized name.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.format = detect_format(self.path)
        self._zip: zipfile.ZipFile | None = None
        self._tar: tarfile.TarFile | None = None
        self._raw: dict[str, Any] = {}
        if self.format == "zip":
            try:
                self._zip = zipfile.ZipFile(self.path)
            except (zipfile.BadZipFile, UnicodeDecodeError,
                    OSError) as exc:
                raise ToolError(
                    f"unreadable bundle {self.path.name}: {exc}"
                ) from exc
            for info in self._zip.infolist():
                is_dir = info.filename.endswith("/")
                name = info.filename.rstrip("/")
                self._raw[name] = info
        elif self.format in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
            try:
                self._tar = tarfile.open(self.path, "r:*")
                members = self._tar.getmembers()
            except (tarfile.TarError, EOFError, OSError) as exc:
                raise ToolError(
                    f"unreadable bundle {self.path.name}: {exc}"
                ) from exc
            for m in members:
                self._raw[m.name.rstrip("/")] = m
        else:
            raise ToolError(
                f"not a bundle container: {self.path.name} "
                f"(format {self.format!r})")

    def __enter__(self) -> "_BundleArchive":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
        if self._tar is not None:
            self._tar.close()

    def entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, raw in self._raw.items():
            if self._zip is not None:
                is_dir = raw.filename.endswith("/")
                out.append({"name": name, "size": raw.file_size,
                            "is_dir": is_dir,
                            "kind": "dir" if is_dir else "file"})
            else:
                kind = ("dir" if raw.isdir() else
                        "file" if raw.isfile() else "other")
                out.append({"name": name, "size": raw.size,
                            "is_dir": raw.isdir(), "kind": kind})
        return out

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self._raw[name])
        fh = self._tar.extractfile(self._raw[name])  # type: ignore[union-attr]
        if fh is None:
            raise ToolError(f"unreadable bundle member {name!r}")
        return fh.read()

    def stream(self, name: str) -> Any:
        if self._zip is not None:
            return self._zip.open(self._raw[name])
        fh = self._tar.extractfile(self._raw[name])  # type: ignore[union-attr]
        if fh is None:
            raise ToolError(f"unreadable bundle member {name!r}")
        return fh


def _parse_bundle_manifest(raw: bytes, origin: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError(
            f"bundle manifest in {origin} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict) \
            or parsed.get("format") != BUNDLE_FORMAT:
        raise ToolError(
            f"bundle manifest in {origin} is not a {BUNDLE_FORMAT} "
            "manifest")
    if int(parsed.get("version", 0)) != BUNDLE_FORMAT_VERSION:
        raise ToolError(
            f"unsupported bundle format version "
            f"{parsed.get('version')!r} in {origin} "
            f"(this reader speaks {BUNDLE_FORMAT_VERSION})")
    return parsed


def open_bundle(path: str | Path) -> dict[str, Any]:
    """Open a bundle — archive file or directory — through the seam.

    Returns ``{"path", "format", "legacy", "manifest",
    "family_manifest", "members"}``.  ``manifest`` is the standard
    bundle manifest when the bundle carries one (``legacy`` False)
    and ``members`` then comes from it; on a legacy bundle
    ``manifest`` is None, ``legacy`` is True, and ``members`` is
    synthesized from the container itself.  A family's own
    ``manifest.json``, when visible and parseable, rides along as
    ``family_manifest`` either way.
    """
    p = Path(path)
    if p.is_dir():
        return _open_bundle_dir(p)
    if not p.is_file():
        raise ToolError(f"bundle not found: {p}")
    with _BundleArchive(p) as arc:
        entries = arc.entries()
        names = {e["name"] for e in entries}
        manifest = None
        if BUNDLE_MANIFEST_NAME in names:
            manifest = _parse_bundle_manifest(
                arc.read(BUNDLE_MANIFEST_NAME), p.name)
        family = None
        if "manifest.json" in names:
            try:
                family = json.loads(
                    arc.read("manifest.json").decode("utf-8"))
            except Exception:  # noqa: BLE001 - family data, best effort
                family = None
        fmt = arc.format
    members = (manifest.get("members", []) if manifest is not None
               else entries)
    return {"path": str(p), "format": fmt,
            "legacy": manifest is None, "manifest": manifest,
            "family_manifest": family, "members": members}


def _open_bundle_dir(p: Path) -> dict[str, Any]:
    manifest = None
    family = None
    members: list[dict[str, Any]] | None = None
    bm = p / BUNDLE_MANIFEST_NAME
    mf = p / "manifest.json"
    if bm.is_file():
        manifest = _parse_bundle_manifest(
            bm.read_bytes(), str(bm))
        members = list(manifest.get("members", []))
    elif mf.is_file():
        try:
            family = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - family data, best effort
            family = None
        if isinstance(family, dict):
            block = family.get("bundle")
            if isinstance(block, dict) \
                    and block.get("format") == BUNDLE_FORMAT:
                manifest = _parse_bundle_manifest(
                    json.dumps(block).encode("utf-8"), str(mf))
                members = list(manifest.get("members", []))
            elif isinstance(family.get("files"), dict):
                # OS snapshot shape: {name: {sha256, bytes}}.
                members = [{
                    "name": name,
                    "size": int(meta.get("bytes", 0)),
                    "sha256": str(meta.get("sha256", "")),
                    "is_dir": (p / name).is_dir(),
                } for name, meta in sorted(family["files"].items())
                    if isinstance(meta, dict)]
            elif isinstance(family.get("files"), list):
                # Backup generation shape: [{rel, size, sha256, ...}].
                members = [{
                    "name": str(e.get("rel", "")),
                    "size": int(e.get("size", 0)),
                    "sha256": str(e.get("sha256", "")),
                    "is_dir": False,
                } for e in family["files"] if isinstance(e, dict)]
    if members is None:
        members = [{
            "name": f.relative_to(p).as_posix(),
            "size": f.stat().st_size, "sha256": "", "is_dir": False,
        } for f in sorted(p.rglob("*"))
            if f.is_file() and f.name not in (BUNDLE_MANIFEST_NAME,
                                              "manifest.json")]
    return {"path": str(p), "format": "directory",
            "legacy": manifest is None, "manifest": manifest,
            "family_manifest": family, "members": members}


def list_bundle(path: str | Path) -> list[dict[str, Any]]:
    """The member list of a bundle (manifest view when carried)."""
    return open_bundle(path)["members"]


def verify_bundle(path: str | Path) -> dict[str, Any]:
    """Verify a bundle against its manifest.

    Manifest-carrying bundles: every recorded member is checked
    for presence, size and SHA-256 — a mismatch names the member —
    and any archive member the manifest does not record is itself
    a problem.  Legacy bundles have no recorded hashes to check
    against: verification is structural (the bundle opens, member
    names are safe) plus, for directory bundles whose family
    manifest records per-file hashes (snapshots), those hashes are
    checked.  Returns ``{"path", "ok", "legacy", "checked",
    "problems"}``.
    """
    p = Path(path)
    info = open_bundle(p)
    problems: list[str] = []
    checked = 0
    manifest = info["manifest"]
    if manifest is not None:
        recorded = {str(m["name"]): m
                    for m in manifest.get("members", [])}
        if p.is_dir():
            for name, m in recorded.items():
                checked += 1
                target = p / name
                if m.get("is_dir"):
                    if not target.is_dir():
                        problems.append(f"missing member: {name}")
                    continue
                if not target.is_file():
                    problems.append(f"missing member: {name}")
                    continue
                actual_size = target.stat().st_size
                if actual_size != int(m.get("size", -1)):
                    problems.append(
                        f"size mismatch: {name} (manifest "
                        f"{m.get('size')}, actual {actual_size})")
                    continue
                if m.get("sha256") \
                        and sha256_file(target) != m["sha256"]:
                    problems.append(f"sha256 mismatch: {name}")
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(p).as_posix()
                    if rel not in recorded and rel not in (
                            BUNDLE_MANIFEST_NAME, "manifest.json"):
                        problems.append(f"unmanifested member: {rel}")
        else:
            with _BundleArchive(p) as arc:
                present = {e["name"]: e for e in arc.entries()}
                for name, m in recorded.items():
                    checked += 1
                    entry = present.get(name)
                    if entry is None:
                        problems.append(f"missing member: {name}")
                        continue
                    if bool(m.get("is_dir")) != bool(entry["is_dir"]):
                        problems.append(
                            f"member kind mismatch: {name}")
                        continue
                    if entry["is_dir"]:
                        continue
                    if int(entry["size"]) != int(m.get("size", -1)):
                        problems.append(
                            f"size mismatch: {name} (manifest "
                            f"{m.get('size')}, bundle {entry['size']})")
                        continue
                    if m.get("sha256"):
                        digest = hashlib.sha256()
                        with arc.stream(name) as fh:
                            for chunk in iter(
                                    lambda: fh.read(1 << 20), b""):
                                digest.update(chunk)
                        if digest.hexdigest() != m["sha256"]:
                            problems.append(f"sha256 mismatch: {name}")
                for name in present:
                    if name not in recorded \
                            and name != BUNDLE_MANIFEST_NAME:
                        problems.append(f"unmanifested member: {name}")
    else:
        for entry in info["members"]:
            name = entry.get("name", "")
            if not name:
                continue
            ok, reason = is_safe_member_name(name)
            if not ok:
                problems.append(
                    f"unsafe member name: {name} ({reason})")
        if p.is_dir():
            for entry in info["members"]:
                if entry.get("sha256") and not entry.get("is_dir"):
                    checked += 1
                    target = p / entry["name"]
                    if not target.is_file():
                        problems.append(
                            f"missing member: {entry['name']}")
                    elif sha256_file(target) != entry["sha256"]:
                        problems.append(
                            f"sha256 mismatch: {entry['name']}")
    return {"path": str(p), "ok": not problems,
            "legacy": info["legacy"], "checked": checked,
            "problems": problems}


def extract_bundle(path: str | Path, dest: str | Path, *,
                   max_files: int = _MAX_EXTRACT_FILES,
                   max_bytes: int = _MAX_EXTRACT_BYTES
                   ) -> dict[str, Any]:
    """Extract a bundle file into ``dest`` under the Archivist's
    safety semantics.

    Bundles are machine-made, so the contract is refuse-whole (the
    ``extract_for_downloader`` rule, not the chat extractor's
    skip-and-report): any unsafe member name, any non-file member
    that is not a directory (a tar link, a device), or a bomb
    budget trip (member count / live-streamed bytes) refuses the
    whole bundle with the offender named.  Directory bundles are
    already on disk and refuse.  Returns ``{"path", "dest",
    "format", "extracted", "bytes", "files", "manifest",
    "legacy"}``.
    """
    p = Path(path)
    if p.is_dir():
        raise ToolError(f"{p} is a directory bundle — its files "
                        "are already on disk, nothing to extract")
    if not p.is_file():
        raise ToolError(f"bundle not found: {p}")
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    root = dest_p.resolve()
    written: list[str] = []
    total = 0
    manifest = None
    with _BundleArchive(p) as arc:
        entries = arc.entries()
        verdict = validate_archive_members(
            [e["name"] for e in entries if e["name"]])
        if verdict["unsafe"]:
            first = verdict["unsafe"][0]
            raise ToolError(
                f"refused: bundle {p.name} carries unsafe member "
                f"{first['name']!r} ({first['reason']}) — bundles "
                "are machine-made, so one bad name refuses the "
                "whole bundle")
        for entry in entries:
            if entry["kind"] == "other":
                raise ToolError(
                    f"refused: bundle {p.name} member "
                    f"{entry['name']!r} is not a regular file or "
                    "directory (links and devices never ride a "
                    "bundle)")
        if len(entries) > max_files:
            raise _BombError(
                f"bundle {p.name} holds {len(entries)} members, "
                f"over the {max_files} extraction budget")
        for entry in entries:
            target = root / entry["name"]
            if not Archivist._within(root, target):
                raise ToolError(
                    f"refused: bundle member {entry['name']!r} "
                    "escapes the destination")
            if entry["is_dir"]:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with arc.stream(entry["name"]) as src, \
                    open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise _BombError(
                            f"bundle {p.name} exceeds the "
                            f"{max_bytes}-byte extraction budget")
                    out.write(chunk)
            written.append(entry["name"])
        if BUNDLE_MANIFEST_NAME in {e["name"] for e in entries}:
            manifest = _parse_bundle_manifest(
                arc.read(BUNDLE_MANIFEST_NAME), p.name)
        fmt = arc.format
    return {"path": str(p), "dest": str(root), "format": fmt,
            "extracted": len(written), "bytes": total,
            "files": written, "manifest": manifest,
            "legacy": manifest is None}


def read_bundle_members(path: str | Path, *,
                        max_bytes: int = _MAX_EXTRACT_BYTES
                        ) -> dict[str, bytes]:
    """Read every file member of a bundle file into memory.

    For consumers that work on member bytes directly (the artifact
    store's import).  Budget-capped like extraction; directory
    members are skipped; insertion order is archive order.
    """
    p = Path(path)
    if p.is_dir():
        raise ToolError(f"{p} is a directory bundle — read its "
                        "files from disk instead")
    if not p.is_file():
        raise ToolError(f"bundle not found: {p}")
    out: dict[str, bytes] = {}
    total = 0
    with _BundleArchive(p) as arc:
        for entry in arc.entries():
            if entry["is_dir"]:
                continue
            data = arc.read(entry["name"])
            total += len(data)
            if total > max_bytes:
                raise _BombError(
                    f"bundle {p.name} exceeds the {max_bytes}-byte "
                    "read budget")
            out[entry["name"]] = data
    return out


__all__ += ["BUNDLE_FORMAT", "BUNDLE_FORMAT_VERSION",
            "BUNDLE_MANIFEST_NAME", "bundle_manifest", "create_bundle",
            "open_bundle", "list_bundle", "verify_bundle",
            "extract_bundle", "read_bundle_members", "sha256_file"]
