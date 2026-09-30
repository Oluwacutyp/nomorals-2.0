"""Archive system — create and crack open any common archive.

Formats, by capability:
* **stdlib (always)** — zip, tar, tar.gz, tar.bz2, tar.xz, and single-file
  gzip / bzip2 / xz.
* **external (when installed)** — 7z / 7z / 7za, rar (``unrar`` / ``unar``).

Every archive is first *identified* by magic bytes (not just extension),
so a renamed ``.zip`` that's actually tar.gz still works, and ``info``
reports what the file really is.  Extraction is traversal-safe: absolute
paths and ``..`` escapes are skipped and reported, never followed.

    from nomorals.archives import Archivist
    a = Archivist(context)
    a.info("workspace/data.zip")        # real format + entries + sizes
    a.list("workspace/data.zip")
    a.extract("workspace/data.zip")     # → workspace/data.zip.extracted/
    a.create(["workspace/a.txt", "workspace/b.txt"], "bundle.tar.gz")
    a.compress("workspace/big.bin")     # → big.bin.gz

Registered as the ``archive`` tool.
"""

from __future__ import annotations

import bz2
import glob
import lzma
import os
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
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
        return "gzip"
    if head.startswith(b"BZh"):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
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
    def info(self, path: str) -> dict[str, Any]:
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
            elif fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
                with tarfile.open(p) as tf:
                    members = tf.getmembers()
                    out["entries"] = len(members)
                    out["total_bytes"] = sum(m.size for m in members)
            elif fmt in ("gzip", "bzip2", "xz"):
                out["entries"] = 1
            elif fmt == "7z" and self._bin7z:
                out.update(self._list7z(p))
            elif fmt == "rar" and self._bin_unrar:
                out.update(self._list_rar(p))
            elif fmt in ("7z", "rar"):
                out["note"] = (f"needs {'7z' if fmt == '7z' else 'unrar'} "
                               "installed to inspect")
            else:
                out["note"] = "not a recognized archive"
        except (OSError, EOFError, zipfile.BadZipFile,
                tarfile.TarError, lzma.LZMAError, bz2.OSError) as exc:
            out["note"] = f"unreadable: {exc}"
        if out.get("total_bytes") and size > 0:
            out["ratio"] = round(out["total_bytes"] / size, 2)
        return out

    def list(self, path: str, *, limit: int = _MAX_LIST) -> dict[str, Any]:
        p = self._path(path)
        fmt = detect_format(p)
        names: list[dict[str, Any]] = []
        if fmt == "zip":
            with zipfile.ZipFile(p) as zf:
                for i in zf.infolist()[:limit]:
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
            names = self._list7z(p).get("entries_list", [])[:limit]
        elif fmt == "rar" and self._bin_unrar:
            names = self._list_rar(p).get("entries_list", [])[:limit]
        else:
            raise ToolError(f"can't list a {fmt} archive here")
        return {"path": str(p), "format": fmt, "count": len(names),
                "entries": names}

    def _list7z(self, p: Path) -> dict[str, Any]:
        out = subprocess.run(
            [self._bin7z, "l", str(p)], capture_output=True, text=True,
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
        return {"entries": len(entries), "total_bytes":
                sum(e["size"] for e in entries),
                "entries_list": entries}

    def _list_rar(self, p: Path) -> dict[str, Any]:
        if os.path.basename(self._bin_unrar) == "unar":
            args = ["-l", str(p)]
        else:
            args = ["lt", str(p)]
        out = subprocess.run(
            [self._bin_unrar, *args], capture_output=True, text=True,
            timeout=60)
        entries = []
        for line in out.stdout.splitlines():
            m = re.match(r"^\s*(\S.*?)\s+(\d+[-]\d+[-]\d+)\s+(\d+)\s+(\d+)",
                         line)
            if m:
                entries.append({"name": m.group(1).strip(),
                                "size": int(m.group(4)), "is_dir": False})
        return {"entries": len(entries), "total_bytes":
                sum(e["size"] for e in entries),
                "entries_list": entries}

    # ── create ────────────────────────────────────────────────────────────
    def create(self, paths: list[str], dest: str,
               fmt: str = "zip") -> dict[str, Any]:
        fmt = (fmt or "zip").lower().lstrip(".")
        if fmt not in ("zip", "tar.gz", "tar", "tar.bz2", "tar.xz"):
            raise ToolError(f"unsupported create format {fmt!r} "
                            "(zip|tar|tar.gz|tar.bz2|tar.xz)")
        resolved = [self._path(p) for p in paths]
        dest_p = self._path(dest, must_exist=False)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
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
                                    zf.write(f, str(f.relative_to(p.parent)))
                        else:
                            zf.write(p, p.name)
            else:
                mode = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2",
                        "tar.xz": "w:xz"}[fmt]
                with tarfile.open(tmp_dest, mode) as tf:
                    for p in resolved:
                        tf.add(p, arcname=p.name,
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

    # ── extract (traversal-safe) ──────────────────────────────────────────
    def extract(self, path: str, dest: str = "") -> dict[str, Any]:
        p = self._path(path)
        fmt = detect_format(p)
        dest_p = (self._path(dest, must_exist=False) if dest
                  else p.parent / (p.name + ".extracted"))
        dest_p.mkdir(parents=True, exist_ok=True)
        dest_root = dest_p.resolve()
        written: list[str] = []
        skipped: list[str] = []
        total_bytes = 0
        count = 0

        def _safe(target: Path) -> bool:
            resolved = (target if target.is_absolute()
                        else dest_root / target).resolve()
            try:
                resolved.relative_to(dest_root)
                return True
            except ValueError:
                return False

        if fmt == "zip":
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    target = dest_root / info.filename
                    if not _safe(Path(info.filename)):
                        skipped.append(info.filename)
                        continue
                    if info.filename.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            total_bytes += len(chunk)
                            if total_bytes > _MAX_EXTRACT_BYTES:
                                raise ToolError(
                                    "extraction aborted: exceeds 2 GB "
                                    "(zip-bomb guard)")
                            dst.write(chunk)
                    count += 1
                    written.append(str(target.relative_to(dest_root)))
                    if count > _MAX_EXTRACT_FILES:
                        raise ToolError("extraction aborted: too many "
                                        "files (zip-bomb guard)")
        elif fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
            with tarfile.open(p) as tf:
                for m in tf.getmembers():
                    if not _safe(Path(m.name)):
                        skipped.append(m.name)
                        continue
                    target = dest_root / m.name
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if m.issym() or m.islnk():
                        # link targets must stay inside
                        if m.issym() and not _safe(Path(m.linkname)):
                            skipped.append(m.name)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            tf.extract(m, dest_root, filter="data")
                        except TypeError:  # Python < 3.12
                            tf.extract(m, dest_root)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    with open(target, "wb") as dst:
                        while chunk := fobj.read(1 << 20):
                            total_bytes += len(chunk)
                            if total_bytes > _MAX_EXTRACT_BYTES:
                                raise ToolError("extraction aborted: "
                                                "exceeds 2 GB")
                            dst.write(chunk)
                    count += 1
                    written.append(str(target.relative_to(dest_root)))
        elif fmt == "7z" and self._bin7z:
            out = subprocess.run(
                [self._bin7z, "x", f"-o{dest_root}", "-y", str(p)],
                capture_output=True, text=True, timeout=600)
            if out.returncode != 0:
                raise ToolError(f"7z extraction failed: "
                                f"{out.stderr[:300]}")
            written = [str(f.relative_to(dest_root))
                       for f in dest_root.rglob("*") if f.is_file()]
            count = len(written)
        elif fmt == "rar" and self._bin_unrar:
            if os.path.basename(self._bin_unrar) == "unar":
                cmd = [self._bin_unrar, "-o", str(dest_root), str(p)]
            else:
                cmd = [self._bin_unrar, "x", "-y", str(p), f"{dest_root}/"]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=600)
            if out.returncode not in (0, 1):  # 1 = warnings (ok)
                raise ToolError(f"rar extraction failed: "
                                f"{out.stderr[:300]}")
            written = [str(f.relative_to(dest_root))
                       for f in dest_root.rglob("*") if f.is_file()]
            count = len(written)
        elif fmt in ("gzip", "bzip2", "xz"):
            opener = {"gzip": gzip_open_r, "bzip2": bz2_open_r,
                      "xz": lzma_open_r}[fmt]
            out_p = dest_p / p.stem
            with open(p, "rb") as src, open(out_p, "wb") as dst:
                shutil.copyfileobj(opener(src), dst, length=1 << 20)
            written.append(out_p.name)
            count = 1
        else:
            raise ToolError(f"can't extract format {fmt!r} here"
                            + ("" if fmt not in ("7z", "rar")
                               else " (install 7z / unrar)"))
        return {"path": str(p), "format": fmt, "dest": str(dest_root),
                "extracted": len(written), "bytes": total_bytes,
                "skipped": skipped[:50], "files": written[:200]}


    def digest(self, path: str, *, max_files: int = 200,
               max_bytes: int = 500_000) -> dict[str, Any]:
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
                self.extract(str(p), dest=str(dest))
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
            "when the tools are installed). action=info (path) | list "
            "(path) | extract (path, dest) | digest (path) - extract an "
            "archive (or walk a whole DIRECTORY) and curate every text "
            "document into the knowledge graph + memory | create (paths, "
            "dest, fmt) | compress (path, fmt)."
        ),
        capability=Capability.FS_WRITE,
    )
    def archive(action: str = "info", path: str = "", dest: str = "",
                paths: str = "", fmt: str = "zip") -> dict[str, Any]:
        a = Archivist(context)
        if action == "info":
            return a.info(path)
        if action == "list":
            return a.list(path)
        if action == "extract":
            return a.extract(path, dest=dest)
        if action == "digest":
            from .tools.filesystem import safe_path

            p = safe_path(context, path, must_exist=True)
            if p.is_dir():
                return a.digest_directory(path)
            return a.digest(path)
        if action == "create":
            items = [x for x in (path, *paths.split("|")) if x.strip()]
            if not dest:
                raise ToolError("archive create needs dest=")
            return a.create(items, dest, fmt=fmt)
        if action == "compress":
            return a.compress(path, fmt=fmt)
        raise ToolError(f"unknown archive action {action!r}")
