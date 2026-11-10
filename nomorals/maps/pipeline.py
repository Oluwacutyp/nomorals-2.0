"""The basemap pipeline (docs 19+21 M2): acquire → build → package
→ verify → install.

Doc 21 §3's data operation as code.  The pipeline is generic over
the pack definitions in :mod:`nomorals.maps.definitions`; each
stage is a separate function so the ordering — and the verify
gate — are testable offline with fake fetchers/runners:

* **acquire** downloads a definition's sources into the build
  root and enforces the hash discipline the source registry
  declares (pinned SHA-256 refused by name on mismatch; rolling
  extracts verified against the publisher's MD5 sidecar and
  pin-after-first-fetch recorded).  Downloads run through the
  system ``curl``/``wget`` as a subprocess — resumable, bounded,
  and the maps package keeps M1's no-sockets-in-package pin.
* **build** turns sources into tiles in a staging directory, via
  the definition's builder: Planetiler (external toolchain, MVT)
  or Devon's own Natural Earth rasterizer (PNG).
* **package** pins the staging directory into an M1 manifest.
* **verify** runs M1's ``verify_pack_dir`` — a pack that fails
  is NOT installed and mints no install receipt.
* **install** moves the pack into the packs directory, writes
  the release-registry entry (signature state as the manifest
  *earned* it), and mints the ``pack_install`` receipt.

Subprocess calls take argument lists, never shell strings, run
under explicit timeouts, and their output lands in a build log.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger
from .definitions import (
    BUILDER_NE_RASTER,
    BUILDER_PLANETILER,
    PLANETILER_AUX_SOURCE_IDS,
    MapSource,
    PackDefinition,
    get_pack,
    get_source,
)
from .manifest import build_manifest, load_manifest, write_manifest
from .mbtiles import extract_mbtiles
from .neraster import build_ne_tiles
from .packs import verify_pack_dir
from .receipts import OP_PACK_INSTALL, OUTCOME_OK, MapReceipt, MapReceiptStore
from .registry import ReleaseEntry, ReleaseRegistry

_log = get_logger(__name__)

__all__ = [
    "AcquireResult", "BuildReport", "InstallOutcome", "PipelineError",
    "acquire_source", "package_pack", "planetiler_argv", "run_build",
    "verify_and_install",
]

#: Fetcher protocol: (url, dest_path, timeout_s) → None or raise.
Fetcher = Callable[[str, Path, float], None]

#: Command runner protocol: (argv, timeout_s) → (returncode, output).
CommandRunner = Callable[[list, float], tuple]

#: Builder protocol: (definition, inputs, staging, ctx) → tile count.
BuilderFn = Callable[..., int]


class PipelineError(RuntimeError):
    """A pipeline stage failed; the message names the stage's truth."""


@dataclass(frozen=True)
class AcquireResult:
    """What acquire proved about one downloaded source."""

    source_id: str
    path: str
    sha256: str
    size: int
    pinned: bool
    md5_verified: bool = False


@dataclass(frozen=True)
class InstallOutcome:
    """The verify gate + install, as one honest result."""

    ok: bool
    pack_dir: str = ""
    digest: str = ""
    signature_state: str = ""
    error: str = ""
    verify_receipt_id: str = ""
    install_receipt_id: str = ""


@dataclass
class BuildReport:
    """One ``run_build`` outcome: stages reached, pack facts, error."""

    pack_id: str
    ok: bool = False
    stages: list[str] = field(default_factory=list)
    tiles: int = 0
    pack_dir: str = ""
    digest: str = ""
    signature_state: str = ""
    source_sha256: str = ""
    error: str = ""
    verify_receipt_id: str = ""
    install_receipt_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"pack_id": self.pack_id, "ok": self.ok,
                "stages": list(self.stages), "tiles": self.tiles,
                "pack_dir": self.pack_dir, "digest": self.digest,
                "signature_state": self.signature_state,
                "source_sha256": self.source_sha256,
                "error": self.error,
                "verify_receipt_id": self.verify_receipt_id,
                "install_receipt_id": self.install_receipt_id}


# ── acquire ──────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 — publisher sidecar checksum
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_fetcher(url: str, dest: Path, timeout_s: float) -> None:
    """Download via system curl (resumable) or wget, arg-list only."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        argv = ["curl", "-fL", "--retry", "3", "--connect-timeout",
                "30", "-C", "-", "-o", str(dest), url]
    elif shutil.which("wget"):
        argv = ["wget", "-c", "-O", str(dest), url]
    else:
        raise PipelineError(
            "acquire needs curl or wget on PATH — neither is "
            "available to download " + url)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=float(timeout_s), check=False)
        if proc.returncode == 33 and "-C" in argv:
            # The server refuses byte ranges (curl rc 33): the
            # resume posture is impossible, so restart the
            # download cleanly rather than refuse the artifact.
            argv = [item for item in argv if item not in ("-C", "-")]
            proc = subprocess.run(argv, capture_output=True,
                                  text=True, timeout=float(timeout_s),
                                  check=False)
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(
            "download timed out after " + str(int(timeout_s))
            + "s: " + url) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        raise PipelineError(
            "download failed (rc " + str(proc.returncode) + "): "
            + url + " — " + tail)


def acquire_source(source: MapSource, dest_dir: Path | str, *,
                   fetcher: Fetcher | None = None,
                   timeout_s: float = 1800.0) -> AcquireResult:
    """Download (or reuse) one source and enforce its hash discipline.

    A pinned SHA-256 that does not match is refused by name.  An
    unpinned rolling source is checked against the publisher MD5
    sidecar when one is declared, and its fetched SHA-256 is
    returned for the build record (pin-after-first-fetch).
    """
    fetch = fetcher if fetcher is not None else _default_fetcher
    dest = Path(dest_dir) / source.artifact_name()
    if not (dest.is_file() and dest.stat().st_size > 0):
        fetch(source.url, dest, float(timeout_s))
    if not dest.is_file() or dest.stat().st_size == 0:
        raise PipelineError(
            "acquire produced no artifact for source "
            + source.source_id + ": " + str(dest))
    sha = _sha256_file(dest)
    if source.sha256 and sha != source.sha256:
        raise PipelineError(
            "hash mismatch for source " + source.source_id
            + ": registry pins sha256 " + source.sha256
            + ", fetched " + sha + " — refusing the artifact")
    md5_verified = False
    if source.md5_url and not source.sha256:
        sidecar = dest.with_name(dest.name + ".md5")
        if not (sidecar.is_file() and sidecar.stat().st_size > 0):
            fetch(source.md5_url, sidecar,
                  min(float(timeout_s), 120.0))
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        actual = _md5_file(dest)
        if actual != expected:
            raise PipelineError(
                "publisher MD5 mismatch for source " + source.source_id
                + ": sidecar says " + expected + ", artifact is "
                + actual + " — refusing the artifact")
        md5_verified = True
    return AcquireResult(
        source_id=source.source_id, path=str(dest), sha256=sha,
        size=dest.stat().st_size, pinned=bool(source.sha256),
        md5_verified=md5_verified)


# ── build ────────────────────────────────────────────────────────


def _run_command(argv: list[str], *, timeout_s: float,
                 runner: CommandRunner | None = None,
                 log_path: Path | None = None) -> str:
    """Run one bounded command; nonzero exit is a named failure."""
    if runner is not None:
        rc, output = runner(list(argv), float(timeout_s))
    else:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=float(timeout_s), check=False)
            rc, output = proc.returncode, (proc.stdout or "") \
                + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(
                "command timed out after " + str(int(timeout_s))
                + "s: " + argv[0]) from exc
        except OSError as exc:
            raise PipelineError(
                "command could not start: " + argv[0] + ": "
                + str(exc)) from exc
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    if rc != 0:
        tail = output.strip()[-500:]
        raise PipelineError(
            "command failed (rc " + str(rc) + "): " + argv[0]
            + " — " + tail)
    return output


def _tools_dir(env: Mapping[str, str]) -> Path:
    named = env.get("NM_MAPS_TOOLS_DIR", "").strip()
    if named:
        return Path(named)
    return Path.home() / "workspace" / "tools"


def _find_java(env: Mapping[str, str]) -> str:
    """Java for Planetiler: env, PATH, then this host's toolchain."""
    named = env.get("NM_MAPS_JAVA", "").strip()
    if named:
        return named
    on_path = shutil.which("java")
    if on_path:
        return on_path
    candidate = _tools_dir(env) / "jre21" / "bin" / "java"
    if candidate.is_file():
        return str(candidate)
    raise PipelineError(
        "planetiler build needs a Java runtime (17+): set "
        "NM_MAPS_JAVA, put java on PATH, or install a JRE under "
        + str(_tools_dir(env) / "jre21"))


def _find_planetiler_jar(env: Mapping[str, str], *,
                         fetcher: Fetcher | None = None,
                         timeout_s: float = 600.0) -> str:
    """The Planetiler jar: env, tools dir, or self-provisioned.

    The jar is a registered source (``planetiler-jar``), so when
    it is absent the acquire stage fetches it into the tools dir
    with its pinned hash verified — self-provisioning, not a
    silent download inside the build.
    """
    named = env.get("NM_MAPS_PLANETILER_JAR", "").strip()
    if named and Path(named).is_file():
        return named
    candidate = _tools_dir(env) / "planetiler" / "planetiler.jar"
    if candidate.is_file():
        return str(candidate)
    if fetcher is not None:
        source = get_source("planetiler-jar")
        result = acquire_source(source, candidate.parent,
                                fetcher=fetcher, timeout_s=timeout_s)
        return result.path
    raise PipelineError(
        "planetiler jar not found: set NM_MAPS_PLANETILER_JAR or "
        "place planetiler.jar under "
        + str(_tools_dir(env) / "planetiler"))


def _jvm_proxy_args(env: Mapping[str, str]) -> list[str]:
    """JVM proxy properties from the standard proxy env vars.

    Host and port only: any userinfo in the proxy URL is stripped
    and never reaches the command line (or the build log) — a
    proxy that demands authentication fails Planetiler's own
    downloads loudly in the log, and pre-seeding the download
    directory through the acquire fetcher is the workaround.
    With no proxy env (a directly-connected host), no properties
    are set at all.
    """
    raw = ""
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy",
                "HTTP_PROXY"):
        raw = str(env.get(key, "") or "").strip()
        if raw:
            break
    if not raw:
        return []
    hostport = raw.split("://", 1)[-1].rsplit("@", 1)[-1]
    hostport = hostport.split("/", 1)[0]
    host, _sep, port = hostport.partition(":")
    if not host:
        return []
    args = ["-Dhttp.proxyHost=" + host, "-Dhttps.proxyHost=" + host]
    if port:
        args += ["-Dhttp.proxyPort=" + port,
                 "-Dhttps.proxyPort=" + port]
    return args


def planetiler_argv(java: str, jar: str, pbf: Path, mbtiles: Path,
                    definition: PackDefinition, data_dir: Path, *,
                    xmx: str = "2g", threads: int = 2,
                    proxy_args: tuple[str, ...] = ()) -> list[str]:
    """The one place the Planetiler command line is constructed."""
    bbox = definition.bbox
    bounds = ",".join(str(v) for v in bbox.as_list())
    data_path = Path(data_dir)
    return [
        java, *proxy_args, "-Xmx" + xmx, "-jar", jar,
        "--osm-path=" + str(pbf),
        "--output=" + str(mbtiles),
        "--bounds=" + bounds,
        "--minzoom=" + str(int(definition.zoom_min)),
        "--maxzoom=" + str(int(definition.zoom_max)),
        "--download-dir=" + str(data_dir),
        # The tile-weights file is pre-seeded by the acquire stage
        # (planetiler-aux-tileweights): Planetiler otherwise tries
        # to download it mid-run, which a proxied JVM cannot do.
        # The path is pinned explicitly — the default is relative
        # to the JVM's working directory.  tmpdir is pinned into
        # the build directory for the same reason (temp DBs belong
        # to the build, never to the caller's cwd).
        "--tile-weights=" + str(data_path / "tile_weights.tsv.gz"),
        "--tmpdir=" + str(data_path.parent / "planetiler-tmp"),
        "--download",
        "--threads=" + str(int(threads)),
        "--force",
    ]


def _build_planetiler(definition: PackDefinition,
                      inputs: dict[str, Path], staging: Path, *,
                      runner: CommandRunner | None,
                      fetcher: Fetcher | None,
                      env: Mapping[str, str],
                      timeout_s: float, xmx: str,
                      threads: int, log_path: Path) -> int:
    """OSM extract → Planetiler MBTiles → pack-layout MVT tiles."""
    java = _find_java(env)
    jar = _find_planetiler_jar(env, fetcher=fetcher)
    work = staging.parent
    # Planetiler's auxiliary inputs live in the build root's
    # SHARED downloads directory, alongside the extracts: acquired
    # once through the pipeline's own fetcher (which honours the
    # host's proxy env — a JVM on a proxied host cannot fetch them
    # itself), under the exact filenames Planetiler expects, and
    # reused by every pack build, never refetched.
    data_dir = work.parent / "downloads"
    for aux_id in PLANETILER_AUX_SOURCE_IDS:
        acquire_source(get_source(aux_id), data_dir,
                       fetcher=fetcher, timeout_s=timeout_s)
    mbtiles = work / (definition.pack_id + ".mbtiles")
    argv = planetiler_argv(
        java, jar, inputs[definition.source_id], mbtiles, definition,
        data_dir, xmx=xmx, threads=threads,
        proxy_args=tuple(_jvm_proxy_args(env)))
    _run_command(argv, timeout_s=timeout_s, runner=runner,
                 log_path=log_path)
    return extract_mbtiles(mbtiles, staging, definition.tile_format)


def _build_ne_raster(definition: PackDefinition,
                     inputs: dict[str, Path], staging: Path, *,
                     runner: CommandRunner | None,
                     fetcher: Fetcher | None,
                     env: Mapping[str, str],
                     timeout_s: float, xmx: str,
                     threads: int, log_path: Path) -> int:
    """Natural Earth vectors → Devon-painted PNG overview tiles."""
    return build_ne_tiles(
        land_path=inputs[definition.source_id],
        lakes_path=inputs.get("ne-lakes", ""),
        lines_path=inputs.get("ne-admin0-lines", ""),
        out_dir=staging, bbox=definition.bbox,
        zoom_min=definition.zoom_min, zoom_max=definition.zoom_max)


BUILDERS: dict[str, BuilderFn] = {
    BUILDER_PLANETILER: _build_planetiler,
    BUILDER_NE_RASTER: _build_ne_raster,
}


# ── package / verify / install ───────────────────────────────────


def _combined_attribution(definition: PackDefinition) -> str:
    """Every contributing source's attribution, in order.

    For Planetiler builds that includes the auxiliary inputs —
    Planetiler embeds Natural Earth data at low zooms, so the
    pack's credit line names it alongside the OSM credit.
    """
    seen: list[str] = []
    source_ids = list(definition.source_ids())
    if definition.builder == BUILDER_PLANETILER:
        source_ids += list(PLANETILER_AUX_SOURCE_IDS)
    for source_id in source_ids:
        credit = get_source(source_id).attribution
        if credit not in seen:
            seen.append(credit)
    return " · ".join(seen)


def package_pack(definition: PackDefinition, staging: Path | str, *,
                 built_at: str = "") -> str:
    """Pin the staging directory into an M1 manifest; returns digest."""
    manifest = build_manifest(
        staging, pack_id=definition.pack_id, name=definition.name,
        bbox=definition.bbox, zoom_min=definition.zoom_min,
        zoom_max=definition.zoom_max,
        attribution=_combined_attribution(definition),
        built_at=built_at, tile_format=definition.tile_format)
    write_manifest(staging, manifest)
    return manifest.digest


def verify_and_install(definition: PackDefinition,
                       staging: Path | str, packs_dir: Path | str, *,
                       store: MapReceiptStore | None = None,
                       source_sha256: str = "",
                       tile_count: int = 0) -> InstallOutcome:
    """The verify gate, then install + registry + install receipt.

    A pack that fails M1 verification is not installed, earns no
    registry entry and mints no ``pack_install`` receipt — the
    failed ``pack_verify`` receipt is the only record it leaves.
    """
    verification, verify_receipt = verify_pack_dir(staging, store=store)
    if not verification.ok:
        return InstallOutcome(
            ok=False, digest=verification.digest,
            signature_state=verification.signature_state,
            error="verify failed: " + verification.summary(),
            verify_receipt_id=verify_receipt.receipt_id)
    packs_root = Path(packs_dir)
    dest = packs_root / definition.pack_id
    if dest.exists():
        shutil.rmtree(dest)
    packs_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(dest))
    manifest = load_manifest(dest)
    source = get_source(definition.source_id)
    entry = ReleaseRegistry(packs_root).record(ReleaseEntry(
        pack_id=definition.pack_id, digest=manifest.digest,
        signature_state=verification.signature_state,
        built_at=manifest.built_at, source=source.name,
        source_sha256=source_sha256,
        tile_format=manifest.tile_format,
        zoom_min=manifest.zoom_min, zoom_max=manifest.zoom_max))
    install_receipt_id = ""
    if store is not None:
        receipt = MapReceipt.mint(
            OP_PACK_INSTALL, OUTCOME_OK,
            pack_refs=(definition.pack_id,),
            pack_digests=(manifest.digest,),
            detail="installed " + str(tile_count) + " tiles · source "
            + source.name + " · signature " + entry.signature_state)
        if store.append(receipt):
            install_receipt_id = receipt.receipt_id
    return InstallOutcome(
        ok=True, pack_dir=str(dest), digest=manifest.digest,
        signature_state=entry.signature_state,
        verify_receipt_id=verify_receipt.receipt_id,
        install_receipt_id=install_receipt_id)


# ── the run ──────────────────────────────────────────────────────


def run_build(pack_id: str, *, build_root: Path | str,
              packs_dir: Path | str,
              store: MapReceiptStore | None = None,
              fetcher: Fetcher | None = None,
              runner: CommandRunner | None = None,
              env: Mapping[str, str] | None = None,
              acquire_timeout_s: float = 1800.0,
              build_timeout_s: float = 7200.0,
              xmx: str = "2g", threads: int = 2) -> BuildReport:
    """Run the whole pipeline for one registered pack definition.

    Never raises for a stage failure: the report carries the
    stages reached and the named error.  (Programming errors in
    the caller — an unknown pack id — still raise KeyError.)
    """
    definition = get_pack(pack_id)
    environ: Mapping[str, str] = os.environ if env is None else env
    report = BuildReport(pack_id=definition.pack_id)
    root = Path(build_root) / definition.pack_id
    # Downloads are shared across packs under the build root: two
    # packs built from one extract download it once.
    downloads = Path(build_root) / "downloads"
    staging = root / "pack"
    try:
        inputs: dict[str, Path] = {}
        for source_id in definition.source_ids():
            acquired = acquire_source(
                get_source(source_id), downloads, fetcher=fetcher,
                timeout_s=acquire_timeout_s)
            inputs[source_id] = Path(acquired.path)
            if source_id == definition.source_id:
                report.source_sha256 = acquired.sha256
            report.stages.append("acquire:" + source_id)
        builder = BUILDERS.get(definition.builder)
        if builder is None:
            raise PipelineError(
                "no builder registered for " + repr(definition.builder))
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        report.tiles = builder(
            definition, inputs, staging, runner=runner,
            fetcher=fetcher, env=environ, timeout_s=build_timeout_s,
            xmx=xmx, threads=threads, log_path=root / "build.log")
        report.stages.append("build:" + definition.builder)
        report.digest = package_pack(definition, staging)
        report.stages.append("package")
        outcome = verify_and_install(
            definition, staging, packs_dir, store=store,
            source_sha256=report.source_sha256,
            tile_count=report.tiles)
        report.stages.append("verify")
        report.verify_receipt_id = outcome.verify_receipt_id
        report.install_receipt_id = outcome.install_receipt_id
        report.signature_state = outcome.signature_state
        if not outcome.ok:
            report.error = outcome.error
            return report
        report.stages.append("install")
        report.pack_dir = outcome.pack_dir
        report.ok = True
        return report
    except Exception as exc:  # noqa: BLE001 — the report is the surface
        record_failure("maps.pipeline_build", exc)
        _log.debug("maps: build failed for %s", definition.pack_id,
                   exc_info=True)
        report.error = type(exc).__name__ + ": " + str(exc)
        return report
