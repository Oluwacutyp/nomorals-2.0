"""fontpack — build, stock and verify the FontLibrary pack.

This is the provisioning half of the FontLibrary (the registry
half is :mod:`fontlib`).  Stocking is a ONE-TIME, whole-library
act in Devon's self-provisioning idiom: build the pack from the
google/fonts source at a pinned commit, verify it, stock it
into a home's data estate — after which every render runs off
the stock, offline, forever.  Runtime code never imports the
network paths here; rendering performs NO network access.

The pack (census S1/S2):

* every family in the google/fonts ``ofl/``, ``apache/`` and
  ``ufl/`` buckets — TTF faces as served, variable fonts kept
  as variable (one file replacing a static weight range is a
  weights curation, not a library cut);
* every family's licence file carried inside the pack, its
  licence id read from the family's own ``METADATA.pb`` —
  never assumed;
* exclusions NAMED in the manifest, never silent: any other
  bucket (``cc-by-sa`` carries share-alike obligations the
  other three buckets do not), families with no TTF faces,
  files that fail to download;
* a ``MANIFEST.json``: per family — name, category, licence
  id, licence file, Reserved Font Name flag, subsets; per
  face — file, weight, weight range, style, variable flag,
  script coverage, SHA-256.

``python -m nomorals.media.typography.fontpack build|verify|stock|core``
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ...core.inflight import run_inflight
from .fontlib import MANIFEST_NAME, estate_library_dir

__all__ = [
    "DEFAULT_BATCH_FAMILIES",
    "DEFAULT_SOURCE_COMMIT",
    "STOCK_BUCKETS",
    "STOCK_STATE_NAME",
    "FontPackError",
    "build_pack",
    "build_pack_resumable",
    "core_subset",
    "ensure_font_pack",
    "estate_build_dir",
    "parse_metadata_pb",
    "stock_pack",
    "stock_status",
    "verify_pack",
]

#: The google/fonts commit this build of Devon pins.  The pack
#: manifest records the commit it was actually built from; a
#: rebuild at a newer commit is a deliberate act, never drift.
DEFAULT_SOURCE_COMMIT = "c30b221beb0c1da59d563feb91ec8656a1fa972a"
SOURCE_REPO = "google/fonts"
STOCK_BUCKETS = ("ofl", "apache", "ufl")

#: The per-run state file inside a resumable build dir: which
#: families are done, which are named exclusions, what the last
#: fatal attempt said.  The build dir is never the library
#: itself; :func:`stock_pack` refuses to stock it until the
#: manifest is whole, and never copies this file into the stock.
STOCK_STATE_NAME = "stock-state.json"

#: Families fetched per batch of a resumable build — bounded so
#: one run (one boot, one CLI invocation) never has to swallow
#: the whole ~2,000-family catalogue to make progress, and a
#: killed run resumes from the state file instead of restarting.
DEFAULT_BATCH_FAMILIES = 200

#: A fatal attempt (the repo tree cannot be enumerated, the disk
#: is short) is not retried for this long unless forced — a
#: dead network at 03:00 must not become a retry loop.  A single
#: family's fetch failure is NOT fatal: the family stays pending
#: and the next batch retries it.
STOCK_RETRY_COOLDOWN_S = 3600

#: Mean bytes per family in the real pack (~2.47 GB over 2,067
#: families): the per-batch disk estimate.  A batch is refused
#: when free space cannot cover it plus margin — the refusal is
#: reported, never worked around by shrinking the pack.
_EST_BYTES_PER_FAMILY = 1_200_000
_BATCH_DISK_MARGIN_BYTES = 256 * 1024 * 1024
_STOCK_DISK_MARGIN_BYTES = 512 * 1024 * 1024

_LICENSE_IDS = {"OFL": "OFL-1.1", "APACHE2": "Apache-2.0",
                "UFL": "UFL-1.0"}
_CATEGORY_NAMES = {
    "SANS_SERIF": "Sans Serif", "SERIF": "Serif",
    "DISPLAY": "Display", "HANDWRITING": "Handwriting",
    "MONOSPACE": "Monospace",
}
_LICENSE_FILE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENSE",
                       "LICENSE.md", "Apache.txt", "LICENCE.txt",
                       "LICENCE")
_USER_AGENT = "nomorals-fontpack/1.0"


class FontPackError(RuntimeError):
    """A pack build / stock / verify step failed, named."""


# ── METADATA.pb ──────────────────────────────────────────────

_FONT_BLOCK = re.compile(r"fonts\s*\{(.*?)\n\}", re.DOTALL)
_AXIS_BLOCK = re.compile(r"axes\s*\{(.*?)\n\}", re.DOTALL)


def _pb_str(block: str, field: str) -> str:
    match = re.search(rf'^\s*{field}:\s*"([^"]*)"', block,
                      re.MULTILINE)
    return match.group(1) if match else ""


def _pb_num(block: str, field: str) -> float | None:
    match = re.search(rf"^\s*{field}:\s*([0-9.]+)", block,
                      re.MULTILINE)
    return float(match.group(1)) if match else None


class _FamilyExcluded(Exception):
    """One family cannot be stocked, with its reason — the
    build records it as a NAMED exclusion and carries on."""

    def __init__(self, fam_key: str, reason: str):
        super().__init__(f"{fam_key}: {reason}")
        self.fam_key = fam_key
        self.reason = reason


def _license_id_from_text(text: str) -> str:
    """The licence id a licence FILE names for itself — used
    for the legacy families that ship no METADATA.pb."""
    if "Apache License" in text:
        return "Apache-2.0"
    if "SIL Open Font License" in text:
        return "OFL-1.1"
    if "Ubuntu Font License" in text:
        return "UFL-1.0"
    return ""


def parse_metadata_pb(text: str) -> dict[str, Any]:
    """The fields the pack needs out of a family's METADATA.pb
    (protobuf text format, parsed as data — no protobuf lib)."""
    head = text.split("fonts {", 1)[0]
    fonts: list[dict[str, Any]] = []
    for block in _FONT_BLOCK.findall(text):
        fonts.append({
            "filename": _pb_str(block, "filename"),
            "style": _pb_str(block, "style") or "normal",
            "weight": int(_pb_num(block, "weight") or 400),
        })
    axes: dict[str, tuple[float, float]] = {}
    for block in _AXIS_BLOCK.findall(text):
        tag = _pb_str(block, "tag")
        lo, hi = _pb_num(block, "min_value"), _pb_num(block, "max_value")
        if tag and lo is not None and hi is not None:
            axes[tag] = (lo, hi)
    license_raw = _pb_str(head, "license")
    category_raw = _pb_str(head, "category")
    return {
        "name": _pb_str(head, "name"),
        "license_raw": license_raw,
        "license": _LICENSE_IDS.get(license_raw, license_raw),
        "category": _CATEGORY_NAMES.get(
            category_raw, category_raw.title().replace("_", " ")),
        "subsets": re.findall(r'^subsets:\s*"([^"]*)"', text,
                              re.MULTILINE),
        "fonts": fonts,
        "axes": axes,
    }


# ── fetching ─────────────────────────────────────────────────

def _fetch_bytes(url: str, *, attempts: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried, then named
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise FontPackError(f"fetch failed after {attempts} tries: "
                        f"{url}: {last}")


def _open_raw_connection():
    """One HTTPS connection to raw.githubusercontent.com,
    honouring the environment's proxy the way urllib does
    (an egress proxy is how some hosts reach the internet at
    all — a direct connection there hangs, it does not fail).
    """
    import base64
    import http.client
    from urllib.parse import unquote, urlsplit

    proxy = urllib.request.getproxies().get("https", "")
    if not proxy:
        return http.client.HTTPSConnection(
            "raw.githubusercontent.com", timeout=120)
    parts = urlsplit(proxy if "//" in proxy else f"//{proxy}")
    conn = http.client.HTTPSConnection(
        parts.hostname, parts.port or 8080, timeout=120)
    headers: dict[str, str] = {}
    if parts.username:
        creds = (f"{unquote(parts.username)}:"
                 f"{unquote(parts.password or '')}").encode()
        headers["Proxy-Authorization"] = \
            "Basic " + base64.b64encode(creds).decode()
    conn.set_tunnel("raw.githubusercontent.com", 443, headers)
    return conn


class _RawFetcher:
    """Per-thread persistent HTTPS connections to
    raw.githubusercontent.com.  The pack is ~4,000 small files;
    a fresh TLS handshake per file (urllib's shape) wastes more
    time than the bytes take.  One connection per worker
    thread, rebuilt on any error."""

    def __init__(self, commit: str):
        import threading

        self._commit = commit
        self._local = threading.local()

    def _connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _open_raw_connection()
            self._local.conn = conn
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        self._local.conn = None

    def fetch(self, relpath: str) -> bytes:
        path = f"/{SOURCE_REPO}/{self._commit}/{relpath}"
        last: Exception | None = None
        for attempt in range(3):
            try:
                conn = self._connection()
                conn.request("GET", path,
                             headers={"User-Agent": _USER_AGENT})
                resp = conn.getresponse()
                blob = resp.read()
                if resp.status != 200:
                    raise FontPackError(
                        f"HTTP {resp.status} for {relpath}")
                return blob
            except Exception as exc:  # noqa: BLE001 - retried, named
                last = exc
                self._drop()
                if attempt + 1 < 3:
                    time.sleep(1.0 * (attempt + 1))
        raise FontPackError(f"fetch failed after 3 tries: "
                            f"{relpath}: {last}")


def _tree_listing(commit: str) -> list[dict[str, Any]]:
    url = (f"https://api.github.com/repos/{SOURCE_REPO}/git/trees/"
           f"{commit}?recursive=1")
    data = json.loads(_fetch_bytes(url).decode("utf-8"))
    if data.get("truncated"):
        raise FontPackError(
            "the GitHub tree listing came back truncated; the "
            "pack cannot claim completeness from it")
    return list(data.get("tree", []))


def _raw_url(commit: str, path: str) -> str:
    return (f"https://raw.githubusercontent.com/{SOURCE_REPO}/"
            f"{commit}/{path}")


def _endpoint_family_count() -> int | None:
    """The catalogue's own family count (the census's counting
    endpoint), as a cross-check for the manifest.  Best-effort:
    None when the endpoint cannot be read from here."""
    try:
        raw = _fetch_bytes("https://fonts.google.com/metadata/fonts")
        text = raw.decode("utf-8", "replace")
        if text.startswith(")]}'"):
            text = text.split("\n", 1)[1]
        data = json.loads(text)
        return len(data.get("familyMetadataList", []))
    except Exception:  # noqa: BLE001 - a cross-check, never a gate
        return None


# ── enumeration ──────────────────────────────────────────────

def _enumerate_remote(commit: str):
    """(families, exclusions, fetch) from the pinned repo tree.

    ``families`` maps ``bucket/family-dir`` → sorted repo paths
    under it; ``fetch(relpath) -> bytes`` reads one repo file.
    """
    tree = _tree_listing(commit)
    buckets_seen: dict[str, int] = {}
    families: dict[str, list[str]] = {}
    for entry in tree:
        path = str(entry.get("path", ""))
        parts = path.split("/")
        if len(parts) < 2:
            continue
        buckets_seen[parts[0]] = buckets_seen.get(parts[0], 0) + 1
        if parts[0] in STOCK_BUCKETS and len(parts) >= 3:
            families.setdefault("/".join(parts[:2]), []).append(path)
    exclusions = []
    for bucket, count in sorted(buckets_seen.items()):
        if bucket not in STOCK_BUCKETS:
            exclusions.append({
                "bucket": bucket, "entries": count,
                "reason": "not a stockable family bucket "
                          "(licence terms differ or not a "
                          "font-family tree)"})
    fetcher = _RawFetcher(commit)
    return families, exclusions, fetcher.fetch


def _enumerate_local(source_dir: Path):
    if not source_dir.is_dir():
        raise FontPackError(f"source checkout not found: {source_dir}")
    families: dict[str, list[str]] = {}
    for bucket in STOCK_BUCKETS:
        root = source_dir / bucket
        if not root.is_dir():
            continue
        for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rels = []
            for dirpath, _dirs, filenames in os.walk(fam_dir):
                for name in filenames:
                    rels.append(str(
                        (Path(dirpath) / name).relative_to(source_dir)))
            families[f"{bucket}/{fam_dir.name}"] = sorted(rels)
    exclusions = [{
        "bucket": name, "entries": 0,
        "reason": "bucket not present in the local checkout or "
                  "not stockable"}
        for name in ("cc-by-sa",)
        if (source_dir / name).is_dir()]

    def fetch(relpath: str) -> bytes:
        return (source_dir / relpath).read_bytes()
    return families, exclusions, fetch


# ── one family ───────────────────────────────────────────────

def _stock_family(fam_key: str, rels: list[str], fetch,
                  out_dir: Path) -> dict[str, Any]:
    """Download/copy + hash + parse one family into the pack.

    Returns the manifest family record.  Raises FontPackError
    naming the family when anything it must carry is missing —
    a partial family is never stocked quietly.  Face facts come
    from the family's own METADATA.pb here; the build's parse
    pass (:func:`_parse_pass`) then checks every file against
    the real artifact and corrects the record.
    """
    by_name = {Path(r).name: r for r in rels}
    metadata_rel = by_name.get("METADATA.pb")
    metadata_less = metadata_rel is None
    if metadata_less:
        # Legacy / early-access families ship no METADATA.pb.
        # They still stock — named and licensed from the
        # artifacts themselves (the licence file's own text,
        # the family name out of the font file in the parse
        # pass).  A family whose licence cannot be determined
        # from its own files is a named exclusion instead.
        meta = {"name": "", "license": "", "category": "",
                "subsets": [], "fonts": [], "axes": {}}
    else:
        meta = parse_metadata_pb(fetch(metadata_rel).decode("utf-8"))
        if not meta["name"]:
            raise FontPackError(
                f"{fam_key}: METADATA.pb names no family")
    meta_by_file = {f["filename"]: f for f in meta["fonts"]}
    wght_axis = meta["axes"].get("wght")
    weight_range = (int(wght_axis[0]), int(wght_axis[1])) \
        if wght_axis else None

    license_rel = ""
    for candidate in _LICENSE_FILE_NAMES:
        if candidate in by_name:
            license_rel = by_name[candidate]
            break
    reserved = False
    if license_rel:
        blob = fetch(license_rel)
        _write(out_dir / license_rel, blob)
        reserved = b"Reserved Font Name" in blob
        license_file = license_rel
        if metadata_less:
            meta["license"] = _license_id_from_text(
                blob.decode("utf-8", "replace"))
            if not meta["license"]:
                raise _FamilyExcluded(
                    fam_key, "no METADATA.pb and its licence "
                             "file names no known licence")
    else:
        license_file = ""
        if metadata_less:
            raise _FamilyExcluded(
                fam_key, "no METADATA.pb and no licence file — "
                         "the family cannot be licensed from "
                         "its own files")

    faces: list[dict[str, Any]] = []
    for rel in sorted(r for r in rels if r.lower().endswith(".ttf")):
        blob = fetch(rel)
        dest = out_dir / rel
        _write(dest, blob)
        digest = hashlib.sha256(blob).hexdigest()
        declared = meta_by_file.get(Path(rel).name, {})
        record: dict[str, Any] = {
            "file": rel,
            "weight": int(declared.get("weight", 400) or 400),
            "style": str(declared.get("style", "normal") or "normal"),
            "variable": "[" in Path(rel).name,
            "sha256": digest,
            "bytes": len(blob),
            "scripts": [],
        }
        if record["variable"] and weight_range:
            record["weight_range"] = list(weight_range)
        faces.append(record)
    if not faces:
        raise _FamilyExcluded(
            fam_key, "no TTF faces in the repo directory — "
                     "nothing to stock")
    unlisted = [f["file"] for f in faces
                if Path(f["file"]).name not in meta_by_file]
    return {
        "name": meta["name"],
        "category": meta["category"],
        "license": meta["license"],
        "license_file": license_file,
        "reserved_font_name": reserved,
        "dir": fam_key,
        "subsets": meta["subsets"],
        "faces": faces,
        "_unlisted_faces": unlisted,
        "_metadata_less": metadata_less,
    }


def _parse_pass(out_dir: Path, records: list[dict[str, Any]],
                log=lambda *_a: None) -> int:
    """Check every stocked file against the real artifact
    (fontparse): script coverage onto the face record, and
    where the parsed weight/style disagrees with METADATA.pb,
    the FILE wins and the disagreement is counted.  Returns
    the disagreement count."""
    from .fontparse import parse_font_file

    disagreements = 0
    done = 0
    total = sum(len(r["faces"]) for r in records)
    for record in records:
        for face_rec in record["faces"]:
            done += 1
            if done % 500 == 0:
                log(f"fontpack: parsed {done}/{total} faces")
            try:
                parsed = parse_font_file(out_dir / face_rec["file"])
            except Exception:  # noqa: BLE001 - declared data stays
                continue
            if not parsed:
                continue
            face = parsed[0]
            if not record["name"]:
                # A metadata-less family is named by its font.
                record["name"] = face.family
            if int(face.weight) != face_rec["weight"] or \
                    bool(face.italic) != \
                    (face_rec["style"] == "italic"):
                disagreements += 1
                face_rec["weight"] = int(face.weight)
                face_rec["style"] = "italic" if face.italic \
                    else "normal"
            if face.axes:
                face_rec["variable"] = True
                axes = {a.tag: a for a in face.axes}
                if "wght" in axes:
                    face_rec["weight_range"] = [
                        int(axes["wght"].minimum),
                        int(axes["wght"].maximum)]
            face_rec["scripts"] = sorted(
                name for name, cov in face.scripts.items()
                if cov >= 0.5)
    return disagreements


def _write(dest: Path, blob: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(blob)
    os.replace(tmp, dest)


# ── build ────────────────────────────────────────────────────

def build_pack(out_dir: Any, *, commit: str = DEFAULT_SOURCE_COMMIT,
               source_dir: Any = None, workers: int = 24,
               log=lambda *_a: None) -> dict[str, Any]:
    """Build the whole pack into ``out_dir`` and return its
    manifest.  A download failure anywhere fails the build,
    naming the family — a silently partial library is exactly
    what this programme exists to replace."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if source_dir:
        families, exclusions, fetch = _enumerate_local(
            Path(source_dir).expanduser())
        source_commit = str(commit)
    else:
        families, exclusions, fetch = _enumerate_remote(commit)
        source_commit = str(commit)
    log(f"fontpack: {len(families)} family directories to stock")
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    def job(item):
        fam_key, rels = item
        return _stock_family(fam_key, rels, fetch, out)

    items = sorted(families.items())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_job_guard(job), items))
    for (fam_key, _rels), result in zip(items, results,
                                        strict=True):
        if isinstance(result, _FamilyExcluded):
            exclusions.append({"dir": result.fam_key,
                               "reason": result.reason})
        elif isinstance(result, Exception):
            failures.append(f"{fam_key}: {result}")
        else:
            records.append(result)
    if failures:
        raise FontPackError(
            "pack build incomplete — " + "; ".join(failures[:12])
            + (f" … and {len(failures) - 12} more"
               if len(failures) > 12 else ""))
    disagreements = _parse_pass(out, records, log)
    records.sort(key=lambda r: r["name"].casefold())
    unlisted = {}
    metadata_less_names = []
    for record in records:
        extra = record.pop("_unlisted_faces", None)
        if extra:
            unlisted[record["name"]] = extra
        if record.pop("_metadata_less", False):
            metadata_less_names.append(record["name"])
    file_count = sum(len(r["faces"]) for r in records)
    total_bytes = sum(f["bytes"] for r in records
                      for f in r["faces"])
    endpoint_count = _endpoint_family_count()
    manifest: dict[str, Any] = {
        "format": 1,
        "kind": "full",
        "source": {"repo": SOURCE_REPO, "commit": source_commit},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family_count": len(records),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "universe": {
            "repo_family_directories": len(families),
            "metadata_endpoint_families": endpoint_count,
        },
        "exclusions": exclusions,
        "metadata_less_families": sorted(metadata_less_names),
        "families_without_license_file": sorted(
            r["name"] for r in records if not r["license_file"]),
        "unlisted_faces": unlisted,
        "weight_disagreements": disagreements,
        "families": records,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=False) + "\n",
        encoding="utf-8")
    check = verify_pack(out, sample=25)
    if not check["ok"]:
        raise FontPackError(f"fresh pack failed verification: "
                            f"{check['bad'][:5]}")
    log(f"fontpack: stocked {len(records)} families, "
        f"{file_count} files, {total_bytes} bytes")
    return manifest


def _job_guard(job):
    def run(item):
        try:
            return job(item)
        except Exception as exc:  # noqa: BLE001 - collected, named
            return exc
    return run


# ── verify / stock ───────────────────────────────────────────

def verify_pack(pack_dir: Any, *, full: bool = False,
                sample: int = 0) -> dict[str, Any]:
    """Check a pack against its own manifest: counts always,
    hashes in full (``full=True``) or on a deterministic sample.
    Returns a report dict; ``ok`` is the verdict."""
    root = Path(pack_dir).expanduser()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return {"ok": False, "bad": [f"no {MANIFEST_NAME} in {root}"],
                "families": 0, "files": 0, "checked": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    faces = [(fam["name"], face)
             for fam in manifest.get("families", [])
             for face in fam.get("faces", [])]
    bad: list[str] = []
    if len(manifest.get("families", [])) != \
            int(manifest.get("family_count", -1)):
        bad.append("manifest family_count disagrees with its "
                   "families list")
    if len(faces) != int(manifest.get("file_count", -1)):
        bad.append("manifest file_count disagrees with its "
                   "faces list")
    if full:
        todo = faces
    elif sample > 0:
        step = max(1, len(faces) // max(1, sample))
        todo = faces[::step][:sample]
    else:
        todo = []
    checked = 0
    for fam_name, face in todo:
        path = root / face["file"]
        if not path.is_file():
            bad.append(f"{fam_name}: missing file {face['file']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if digest != face.get("sha256"):
            bad.append(f"{fam_name}: hash mismatch on {face['file']}")
    return {"ok": not bad, "bad": bad,
            "families": len(manifest.get("families", [])),
            "files": len(faces), "checked": checked}


def stock_pack(pack_dir: Any, *, home: Any = None,
               log=lambda *_a: None) -> dict[str, Any]:
    """Stock a verified pack into a home's data estate.

    The source pack is FULLY hash-verified first; the copy
    lands in a staging sibling and swaps in only when whole,
    then the stocked copy is verified again.  A pack that
    fails verification is never stocked — and a PARTIAL pack
    (a resumable build that still has families pending) is
    refused by name, so a half library can never be swapped
    in as the library.  The build state file is never copied
    into the stock."""
    src = Path(pack_dir).expanduser()
    manifest_path = src / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            kind = manifest.get("kind", "")
        except (OSError, ValueError):
            manifest = {}
            kind = ""
        if kind == "partial":
            pending = int(manifest.get("pending_families", -1))
            raise FontPackError(
                "refusing to stock a partial pack: "
                f"{pending} families still pending — finish the "
                "build first (nm fonts stock --auto)")
    check = verify_pack(src, full=True)
    if not check["ok"]:
        raise FontPackError(f"refusing to stock an unverified "
                            f"pack: {check['bad'][:5]}")
    dest = estate_library_dir(home)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(dest.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        src, staging,
        ignore=shutil.ignore_patterns(STOCK_STATE_NAME))
    staged = verify_pack(staging, full=True)
    if not staged["ok"]:
        shutil.rmtree(staging, ignore_errors=True)
        raise FontPackError(f"staged copy failed verification: "
                            f"{staged['bad'][:5]}")
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(staging, dest)
    final = verify_pack(dest, full=True)
    if not final["ok"]:
        raise FontPackError(f"stocked copy failed verification: "
                            f"{final['bad'][:5]}")
    log(f"fontpack: stocked {final['families']} families at {dest}")
    return {"dest": str(dest), "families": final["families"],
            "files": final["files"], "checked": final["checked"]}


# ── resumable batched stocking ───────────────────────────────

def estate_build_dir(home: Any = None) -> Path:
    """Where a resumable pack build lives: the sibling of the
    stocked library, never the library itself."""
    return estate_library_dir(home).parent / "library.build"


def _read_state(build_dir: Path) -> dict[str, Any]:
    path = build_dir / STOCK_STATE_NAME
    if not path.is_file():
        return {"commit": "", "done": [], "excluded": [],
                "failures": {}, "weight_disagreements": 0,
                "metadata_less": [], "unlisted": {},
                "last_error": "", "last_fatal_attempt": 0.0,
                "batches": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"commit": "", "done": [], "excluded": [],
                "failures": {}, "weight_disagreements": 0,
                "metadata_less": [], "unlisted": {},
                "last_error": "", "last_fatal_attempt": 0.0,
                "batches": 0}
    if not isinstance(data, dict):
        return {"commit": "", "done": [], "excluded": [],
                "failures": {}, "weight_disagreements": 0,
                "metadata_less": [], "unlisted": {},
                "last_error": "", "last_fatal_attempt": 0.0,
                "batches": 0}
    data.setdefault("done", [])
    data.setdefault("excluded", [])
    data.setdefault("failures", {})
    data.setdefault("weight_disagreements", 0)
    data.setdefault("metadata_less", [])
    data.setdefault("unlisted", {})
    data.setdefault("last_error", "")
    data.setdefault("last_fatal_attempt", 0.0)
    data.setdefault("batches", 0)
    return data


def _write_state(build_dir: Path, state: dict[str, Any]) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    tmp = build_dir / (STOCK_STATE_NAME + ".part")
    tmp.write_text(json.dumps(state, indent=1) + "\n",
                   encoding="utf-8")
    os.replace(tmp, build_dir / STOCK_STATE_NAME)


def _write_manifest(out_dir: Path,
                    manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (MANIFEST_NAME + ".part")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def _manifest_from_records(
        records: list[dict[str, Any]], *,
        source_commit: str, kind: str, pending: int,
        exclusions: list[dict[str, Any]],
        universe_dirs: int | None, endpoint_count: int | None,
        disagreements: int, metadata_less: list[str],
        unlisted: dict[str, Any]) -> dict[str, Any]:
    """The manifest for the records stocked so far — the same
    shape :func:`build_pack` writes, plus the pending count
    that says whether this manifest is the whole library."""
    return {
        "format": 1,
        "kind": kind,
        "source": {"repo": SOURCE_REPO, "commit": source_commit},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                  time.gmtime()),
        "family_count": len(records),
        "file_count": sum(len(r["faces"]) for r in records),
        "total_bytes": sum(f["bytes"] for r in records
                           for f in r["faces"]),
        "pending_families": pending,
        "universe": {
            "repo_family_directories": universe_dirs,
            "metadata_endpoint_families": endpoint_count,
        },
        "exclusions": exclusions,
        "metadata_less_families": sorted(metadata_less),
        "families_without_license_file": sorted(
            r["name"] for r in records if not r["license_file"]),
        "unlisted_faces": unlisted,
        "weight_disagreements": disagreements,
        "families": records,
    }


def build_pack_resumable(
        build_dir: Any, *, commit: str = DEFAULT_SOURCE_COMMIT,
        source_dir: Any = None,
        batch_families: int = DEFAULT_BATCH_FAMILIES,
        max_batches: int | None = None, workers: int = 12,
        log=lambda *_a: None) -> dict[str, Any]:
    """Stock the pack in bounded, resumable batches.

    One call stocks up to ``max_batches`` batches of
    ``batch_families`` families (``None`` = until the pack is
    whole or a fatal error stops the run) into ``build_dir``,
    persisting progress in ``stock-state.json`` after every
    batch.  A killed or failed run resumes where it stopped:
    families already stocked are never re-fetched.  A family
    whose fetch fails stays pending and is retried next run —
    it is never silently dropped and never becomes an
    exclusion.  Only :class:`_FamilyExcluded` (no TTF faces,
    no determinable licence) becomes a NAMED exclusion.

    The manifest in the build dir is rewritten after every
    batch: kind ``partial`` with ``pending_families`` while
    work remains, kind ``full`` only when nothing is pending.
    Returns a report dict; ``complete`` is the verdict.  A
    fatal error (the repo tree cannot be enumerated, the disk
    is short) raises :class:`FontPackError` AFTER recording
    itself in the state file, so the next caller can say what
    happened instead of guessing.
    """
    out = Path(build_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state = _read_state(out)
    if state.get("commit") and state["commit"] != str(commit):
        raise FontPackError(
            f"build dir holds a pack from commit "
            f"{state['commit'][:12]}; rebuild from a fresh "
            f"directory to build commit {str(commit)[:12]}")
    manifest_path = out / MANIFEST_NAME
    records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            prior = json.loads(
                manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = None
        if isinstance(prior, dict):
            if str((prior.get("source") or {}).get(
                    "commit", "")) not in ("", str(commit)):
                raise FontPackError(
                    "build dir holds a pack from a different "
                    "source commit; rebuild from a fresh "
                    "directory")
            records = [r for r in prior.get("families", [])
                       if isinstance(r, dict)]
    done: set[str] = set(state.get("done") or [])
    excluded_keys: set[str] = {
        str(e.get("dir", "")) for e in state.get("excluded") or []
        if isinstance(e, dict)}
    # A family the state calls done but whose files are gone
    # (a killed run, a deleted file) is NOT done: drop it from
    # the records and the done set so this run re-fetches it.
    # The manifest never claims a family whose files are absent.
    # (When nothing is pending the loop below simply does not
    # run and the call returns complete — the no-op path.)
    kept: list[dict[str, Any]] = []
    for record in records:
        faces = record.get("faces", [])
        if faces and all((out / f["file"]).is_file()
                         for f in faces
                         if isinstance(f, dict)):
            kept.append(record)
        else:
            done.discard(str(record.get("dir", "")))
    records = kept
    try:
        if source_dir:
            families, bucket_exclusions, fetch = _enumerate_local(
                Path(source_dir).expanduser())
        else:
            families, bucket_exclusions, fetch = _enumerate_remote(
                commit)
    except Exception as exc:  # noqa: BLE001 - fatal, recorded
        state["commit"] = str(commit)
        state["last_error"] = str(exc)
        state["last_fatal_attempt"] = time.time()
        _write_state(out, state)
        raise FontPackError(f"cannot enumerate the font source: "
                            f"{exc}") from exc
    state["commit"] = str(commit)
    pending = sorted(
        key for key in families
        if key not in done and key not in excluded_keys)
    total = len(families)
    families_this_run = 0
    failures: dict[str, str] = {}
    excluded_this_run: list[dict[str, Any]] = []
    batches_run = 0
    endpoint_count: int | None = None
    while pending and (max_batches is None
                       or batches_run < max_batches):
        batch = pending[:max(1, batch_families)]
        need = (len(batch) * _EST_BYTES_PER_FAMILY
                + _BATCH_DISK_MARGIN_BYTES)
        free = _free_bytes(out)
        if free < need:
            reason = (f"disk short for the next batch: "
                      f"{free} bytes free, about {need} needed "
                      f"for {len(batch)} families")
            state["last_error"] = reason
            state["last_fatal_attempt"] = time.time()
            _write_state(out, state)
            return {
                "complete": False, "build_dir": str(out),
                "families_done": len(done),
                "families_total": total,
                "families_pending": len(pending),
                "families_this_run": families_this_run,
                "files": sum(len(r["faces"]) for r in records),
                "bytes": sum(f["bytes"] for r in records
                             for f in r["faces"]),
                "failures": failures,
                "excluded_this_run": excluded_this_run,
                "manifest_kind": "partial",
                "blocked_reason": reason,
                "batches_run": batches_run,
            }
        log(f"fontpack: batch {batches_run + 1}: {len(batch)} "
            f"families ({len(pending)} pending)")

        def job(item):
            fam_key, rels = item
            return _stock_family(fam_key, rels, fetch, out)

        items = [(key, families[key]) for key in batch]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(_job_guard(job), items))
        new_records: list[dict[str, Any]] = []
        for (fam_key, _rels), result in zip(items, results,
                                            strict=True):
            if isinstance(result, _FamilyExcluded):
                entry = {"dir": result.fam_key,
                         "reason": result.reason}
                state.setdefault("excluded", []).append(entry)
                excluded_keys.add(result.fam_key)
                excluded_this_run.append(entry)
            elif isinstance(result, Exception):
                failures[fam_key] = str(result)
            else:
                new_records.append(result)
                done.add(fam_key)
                families_this_run += 1
        state["failures"] = dict(failures)
        disagreements = _parse_pass(out, new_records, log)
        state["weight_disagreements"] = int(
            state.get("weight_disagreements", 0)) + disagreements
        for record in new_records:
            extra = record.pop("_unlisted_faces", None)
            if extra:
                state.setdefault("unlisted", {})[record["name"]] = \
                    extra
            if record.pop("_metadata_less", False) and \
                    record["name"] not in \
                    state.setdefault("metadata_less", []):
                state["metadata_less"].append(record["name"])
        records.extend(new_records)
        records.sort(key=lambda r: r["name"].casefold())
        pending = sorted(
            key for key in families
            if key not in done and key not in excluded_keys)
        complete = not pending
        if complete and endpoint_count is None:
            endpoint_count = _endpoint_family_count()
        manifest = _manifest_from_records(
            records, source_commit=str(commit),
            kind="full" if complete else "partial",
            pending=len(pending),
            exclusions=list(bucket_exclusions)
            + list(state.get("excluded") or []),
            universe_dirs=total, endpoint_count=endpoint_count,
            disagreements=int(state.get("weight_disagreements", 0)),
            metadata_less=list(state.get("metadata_less") or []),
            unlisted=dict(state.get("unlisted") or {}))
        _write_manifest(out, manifest)
        state["done"] = sorted(done)
        state["batches"] = int(state.get("batches", 0)) + 1
        if complete:
            state["last_error"] = ""
        _write_state(out, state)
        batches_run += 1
        if complete:
            break
    check = verify_pack(out, sample=25)
    if not check["ok"]:
        raise FontPackError(
            f"batch-built pack failed verification: "
            f"{check['bad'][:5]}")
    return {
        "complete": not pending, "build_dir": str(out),
        "families_done": len(done), "families_total": total,
        "families_pending": len(pending),
        "families_this_run": families_this_run,
        "files": sum(len(r["faces"]) for r in records),
        "bytes": sum(f["bytes"] for r in records
                     for f in r["faces"]),
        "failures": failures,
        "excluded_this_run": excluded_this_run,
        "manifest_kind": "full" if not pending else "partial",
        "blocked_reason": "",
        "batches_run": batches_run,
    }


def stock_status(home: Any = None, *,
                 build_dir: Any = None) -> dict[str, Any]:
    """The honest stocking facts, read from disk only — no
    network, never raises.  The pack tier as stocked, the
    build in progress (if any), and one plain sentence saying
    where the library stands."""
    pack: dict[str, Any] = {"stocked": False}
    build: dict[str, Any] = {"started": False}
    try:
        dest = estate_library_dir(home)
        manifest_path = dest / MANIFEST_NAME
        if manifest_path.is_file():
            data = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                pack = {
                    "stocked": data.get("kind") == "full",
                    "kind": str(data.get("kind", "")),
                    "families": int(data.get("family_count", 0)
                                    or 0),
                    "files": int(data.get("file_count", 0) or 0),
                    "bytes": int(data.get("total_bytes", 0) or 0),
                    "source_commit": str(
                        (data.get("source") or {}).get(
                            "commit", "")),
                }
        build_root = Path(build_dir).expanduser() \
            if build_dir else estate_build_dir(home)
        state = _read_state(build_root)
        if state.get("commit") or state.get("done") or \
                (build_root / MANIFEST_NAME).is_file():
            done = state.get("done") or []
            manifest = None
            if (build_root / MANIFEST_NAME).is_file():
                try:
                    manifest = json.loads(
                        (build_root / MANIFEST_NAME).read_text(
                            encoding="utf-8"))
                except (OSError, ValueError):
                    manifest = None
            pending = None
            total = None
            if isinstance(manifest, dict):
                pending = int(manifest.get("pending_families", 0)
                              or 0)
                universe = manifest.get("universe") or {}
                total = universe.get("repo_family_directories")
            build = {
                "started": True,
                "build_dir": str(build_root),
                "families_done": len(done),
                "families_pending": pending,
                "families_total": total,
                "batches": int(state.get("batches", 0) or 0),
                "failures": dict(state.get("failures") or {}),
                "last_error": str(state.get("last_error") or ""),
                "last_attempt": float(
                    state.get("last_fatal_attempt", 0.0) or 0.0),
            }
    except Exception as exc:  # noqa: BLE001 - status never raises
        return {"pack": {"stocked": False},
                "build": {"started": False},
                "summary": f"font stock status could not be read: "
                           f"{exc}"}
    if pack["stocked"]:
        summary = (f"full pack stocked: {pack['families']} "
                   f"families · {pack['files']} files")
    elif build["started"] and build.get("families_total"):
        summary = (
            f"full pack not stocked yet — "
            f"{build['families_done']} of "
            f"{build['families_total']} families fetched so far")
        if build.get("families_pending"):
            summary += (f" ({build['families_pending']} still to "
                        f"fetch)")
        if build.get("last_error"):
            summary += (f"; last attempt failed: "
                        f"{build['last_error']}")
    elif build["started"]:
        summary = ("full pack not stocked yet — a stocking build "
                   "is in progress")
        if build.get("last_error"):
            summary += (f"; last attempt failed: "
                        f"{build['last_error']}")
    else:
        summary = ("full pack not stocked — stocking starts on "
                   "its own at startup, or run "
                   "nm fonts stock --auto")
    return {"pack": pack, "build": build, "summary": summary}


def ensure_font_pack(
        home: Any = None, *, commit: str = DEFAULT_SOURCE_COMMIT,
        source_dir: Any = None,
        batch_families: int = DEFAULT_BATCH_FAMILIES,
        max_batches: int | None = None, workers: int = 12,
        force: bool = False,
        log=lambda *_a: None) -> dict[str, Any]:
    """Stock the full pack, or say honestly why it is not
    stocked.  The auto-stock driver: a no-op when the pack
    tier is already whole and verifies; otherwise resumable
    batches until the pack is whole (or ``max_batches`` is
    spent), then the stock swap.  Concurrent callers in this
    process attach to the one in-flight run (self-provisioning
    law §3.9) instead of starting parallel multi-gigabyte
    builds.  Never raises for a failed fetch or a short disk
    — the returned status dict carries the reason.  Never
    claims ``stocked`` on a partial pack."""
    dest = estate_library_dir(home)
    manifest_path = dest / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            data = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and \
                    data.get("kind") == "full":
                check = verify_pack(dest, sample=25)
                if check["ok"]:
                    return {
                        "status": "stocked",
                        "families": int(data.get("family_count",
                                                 0) or 0),
                        "files": int(data.get("file_count", 0)
                                     or 0),
                        "bytes": int(data.get("total_bytes", 0)
                                     or 0),
                        "detail": "full pack already stocked",
                        "batches": 0,
                    }
        except (OSError, ValueError):
            pass

    def _run() -> dict[str, Any]:
        build_root = estate_build_dir(home)
        state = _read_state(build_root)
        if not force and state.get("last_fatal_attempt"):
            age = time.time() - float(
                state["last_fatal_attempt"])
            if age < STOCK_RETRY_COOLDOWN_S:
                return {
                    "status": "deferred",
                    "families_done": len(state.get("done") or []),
                    "families_pending": None,
                    "detail": (
                        "last stocking attempt failed "
                        f"{int(age)}s ago: "
                        f"{state.get('last_error', '')} — "
                        "retrying after the cooldown, or run "
                        "with --force"),
                    "batches": 0,
                }
        try:
            report = build_pack_resumable(
                build_root, commit=commit, source_dir=source_dir,
                batch_families=batch_families,
                max_batches=max_batches, workers=workers, log=log)
        except FontPackError as exc:
            return {
                "status": "failed",
                "families_done": len(
                    _read_state(build_root).get("done") or []),
                "families_pending": None,
                "detail": str(exc),
                "batches": 0,
            }
        if report.get("blocked_reason"):
            return {
                "status": "blocked",
                "families_done": report["families_done"],
                "families_pending": report["families_pending"],
                "detail": str(report["blocked_reason"]),
                "batches": 0,
            }
        if not report["complete"]:
            return {
                "status": "partial",
                "families_done": report["families_done"],
                "families_total": report["families_total"],
                "families_pending": report["families_pending"],
                "failures": report["failures"],
                "detail": (
                    f"{report['families_done']} of "
                    f"{report['families_total']} families "
                    f"fetched; {report['families_pending']} "
                    f"still pending — the next run resumes "
                    f"where this one stopped"),
                "batches": report.get("batches_run", 0),
            }
        total_bytes = int(report["bytes"])
        free = _free_bytes(build_root)
        if free < total_bytes + _STOCK_DISK_MARGIN_BYTES:
            reason = (f"disk short for the stock swap: {free} "
                      f"bytes free, about "
                      f"{total_bytes + _STOCK_DISK_MARGIN_BYTES} "
                      f"needed")
            return {"status": "blocked",
                    "families_done": report["families_done"],
                    "families_pending": 0, "detail": reason,
                    "batches": 0}
        try:
            stocked = stock_pack(build_root, home=home, log=log)
        except FontPackError as exc:
            return {"status": "failed",
                    "families_done": report["families_done"],
                    "families_pending": 0, "detail": str(exc),
                    "batches": 0}
        return {
            "status": "stocked",
            "families": stocked["families"],
            "files": stocked["files"],
            "bytes": total_bytes,
            "detail": (f"full pack stocked: "
                       f"{stocked['families']} families · "
                       f"{stocked['files']} files"),
            "batches": report.get("batches_run", 0),
        }

    return run_inflight("fontpack-stock", _run)


# ── the vendored core subset ─────────────────────────────────

#: The bare-clone floor: core families across the category
#: spread (text sans/serif, display, script, mono, Latin-Ext
#: coverage for Yoruba diacritics).  Faces: variable files
#: whole, else the regular/bold/italic/bold-italic quad.
#: Noto Color Emoji is deliberately NOT vendored — its single
#: file is ~25 MB; the stocked pack carries it.
CORE_FAMILIES: tuple[str, ...] = (
    "Inter", "Roboto", "Open Sans", "Lato", "Noto Sans",
    "Source Sans 3", "Work Sans", "Montserrat", "Poppins",
    "Oswald", "Bebas Neue", "Anton",
    "Playfair Display", "Lora", "Noto Serif", "Source Serif 4",
    "PT Serif", "Merriweather",
    "Dancing Script", "Great Vibes", "Caveat", "Pacifico",
    "Shadows Into Light",
    "JetBrains Mono", "Roboto Mono", "IBM Plex Mono",
    "Space Mono",
)
_CORE_QUAD = {(400, "normal"), (700, "normal"),
              (400, "italic"), (700, "italic")}


def core_subset(pack_dir: Any, out_dir: Any, *,
                families: tuple[str, ...] = CORE_FAMILIES
                ) -> dict[str, Any]:
    """Carve the vendored core subset out of a built pack."""
    src = Path(pack_dir).expanduser()
    manifest = json.loads(
        (src / MANIFEST_NAME).read_text(encoding="utf-8"))
    by_name = {fam["name"]: fam for fam in manifest["families"]}
    out = Path(out_dir).expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in families:
        fam = by_name.get(name)
        if fam is None:
            missing.append(name)
            continue
        faces = [f for f in fam["faces"] if f.get("variable")]
        if not faces:
            faces = [f for f in fam["faces"]
                     if (int(f["weight"]), f["style"]) in _CORE_QUAD]
        if not faces:
            faces = list(fam["faces"][:4])
        copied = []
        for face in faces:
            blob = (src / face["file"]).read_bytes()
            _write(out / face["file"], blob)
            copied.append(dict(face))
        if fam.get("license_file"):
            _write(out / fam["license_file"],
                   (src / fam["license_file"]).read_bytes())
        records.append({
            "name": fam["name"], "category": fam["category"],
            "license": fam["license"],
            "license_file": fam["license_file"],
            "reserved_font_name": fam["reserved_font_name"],
            "dir": fam["dir"], "subsets": fam.get("subsets", []),
            "faces": copied,
        })
    if missing:
        raise FontPackError(
            f"core families absent from the pack: {missing}")
    core_manifest: dict[str, Any] = {
        "format": 1,
        "kind": "core",
        "source": dict(manifest.get("source") or {}),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                  time.gmtime()),
        "family_count": len(records),
        "file_count": sum(len(r["faces"]) for r in records),
        "total_bytes": sum(f["bytes"] for r in records
                           for f in r["faces"]),
        "families": records,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(core_manifest, indent=1) + "\n", encoding="utf-8")
    return core_manifest


# ── CLI ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fontpack",
        description="build / verify / stock the FontLibrary pack")
    sub = parser.add_subparsers(dest="action", required=True)
    p_build = sub.add_parser("build", help="build the pack")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--commit", default=DEFAULT_SOURCE_COMMIT)
    p_build.add_argument("--source-dir", default="")
    p_build.add_argument("--workers", type=int, default=24)
    p_verify = sub.add_parser("verify", help="verify a pack")
    p_verify.add_argument("--pack", required=True)
    p_verify.add_argument("--full", action="store_true")
    p_stock = sub.add_parser("stock", help="stock a pack into a home")
    p_stock.add_argument("--from", dest="pack", required=True)
    p_stock.add_argument("--home", default="")
    p_core = sub.add_parser("core", help="carve the core subset")
    p_core.add_argument("--pack", required=True)
    p_core.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            manifest = build_pack(
                args.out, commit=args.commit,
                source_dir=args.source_dir or None,
                workers=args.workers, log=print)
            print(f"families={manifest['family_count']} "
                  f"files={manifest['file_count']} "
                  f"bytes={manifest['total_bytes']}")
        elif args.action == "verify":
            report = verify_pack(args.pack, full=args.full,
                                 sample=0 if args.full else 25)
            print(json.dumps(report, indent=1))
            return 0 if report["ok"] else 1
        elif args.action == "stock":
            report = stock_pack(args.pack, home=args.home or None,
                                log=print)
            print(json.dumps(report, indent=1))
        elif args.action == "core":
            manifest = core_subset(args.pack, args.out)
            print(f"core families={manifest['family_count']} "
                  f"files={manifest['file_count']} "
                  f"bytes={manifest['total_bytes']}")
    except FontPackError as exc:
        print(f"fontpack: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
