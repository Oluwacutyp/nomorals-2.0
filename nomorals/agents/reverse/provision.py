"""Self-provisioning for the RE organ's external engines (Pass 2).

REA's sharpest gap is Devon's home ground: REA refuses to install
its engines and makes the human fetch them — Ghidra, the JDK under
it, everything is caller-supplied via PATH/env, and its doctor is a
read-only audit. Devon provisions her own. This module is the
ApkForge toolchain pattern generalized to the RE rungs (pinned
downloads from the publisher's own channel, sha256 verified on
every install, probe-executed before an engine is called ready),
with the pairing ladder's attempts discipline on top: every route
tried is recorded by name, a failed pin is a hard stop rather than
a quiet install, and an engine that cannot be landed is reported
unavailable *with the attempts attached* — never dressed up.

Pinned here:

* **Temurin JDK 25.0.5+8** (Eclipse Adoptium) — Ghidra 12.2's own
  GettingStarted requires a JDK 25; the checksums are Adoptium's
  published ones, read from the Adoptium release API.
* **Ghidra 12.2** (NSA, ``ghidra_12.2_PUBLIC_20261016.zip``) — the
  sha256 is the one published in the official release notes.
* **unblob 26.6.4** (Pass 3, the firmware rung) — a Python engine,
  so its pin channel is PyPI itself: the install set lives in
  ``pins/unblob_26.6.4.txt`` with every wheel hash read from the
  PyPI JSON API (the publisher's own published digests), and the
  install runs ``pip install --require-hashes`` into a dedicated
  venv — a wheel that matches no published digest installs
  nothing. Linux profiles only: unblob's native dependencies
  publish no Windows wheels.
* **Frida 17.26.2** (dynamic rung) — the same pip-venv discipline
  as unblob (``pins/frida_17.26.2.txt``): one self-contained abi3
  wheel per platform, no dependencies except typing_extensions
  on Python < 3.11 (pinned alongside it, since --require-hashes
  holds on every package pip installs).
* **GDB** (dynamic rung) — a system debugger, so the channel is
  the distro package manager (apt on Debian/Ubuntu): the drive
  runs ``apt-get update`` + ``apt-get install -y gdb`` as root
  and trusts the install only after the version probe answers;
  the exact installed version is recorded in the state ledger,
  which is the pin for a distro package. No apt channel (Windows,
  non-Debian Linux without apt) → unavailable with the reason.

Installs live under the managed tools root (``~/workspace/tools``
by house convention, ``NM_RE_TOOLS_DIR`` overrides) in ``re/`` —
never in the repo tree. The state ledger (``re/state.json``) keeps
each engine's version, pin, source and full attempt history.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .failures import record_failure

__all__ = [
    "ProvisionError", "DESCRIPTORS", "tools_root", "re_root",
    "platform_key", "jdk_state", "ghidra_state", "unblob_state",
    "frida_state", "gdb_state", "engine_states", "provision_engine",
    "provision_all", "unblob_pins_path", "frida_pins_path",
]

#: Download budget for one engine archive (the Ghidra zip is
#: ~570 MB; the JDK ~210 MB — generous but bounded).
DOWNLOAD_TIMEOUT_S = 1200.0
#: Probe budget for one version/launch check.
PROBE_TIMEOUT_S = 30.0
#: Budget for the post-install analyzeHeadless launch smoke probe.
LAUNCH_PROBE_TIMEOUT_S = 180.0

_MARKER = ".pinned-sha256"


class ProvisionError(RuntimeError):
    """A provisioning drive failed; carries the attempts by name."""

    def __init__(self, engine: str, attempts: list[str]) -> None:
        self.engine = engine
        self.attempts = attempts
        super().__init__(
            f"{engine} could not be provisioned. Attempts: "
            + " | ".join(attempts))


# ── pinned descriptors ──────────────────────────────────────────────
#
# sha256 provenance: the JDK digests are Adoptium's published
# checksums (Adoptium release API, feature release 25.0.5+8); the
# Ghidra digest is the SHA-256 published in the official Ghidra
# 12.2 release notes. Publisher channels only — never a mirror.

_JDK_VERSION = "25.0.5+8"
_JDK_BASE = ("https://github.com/adoptium/temurin25-binaries/"
             "releases/download/jdk-25.0.5%2B8")
DESCRIPTORS: dict[str, dict[str, Any]] = {
    "jdk": {
        "version": _JDK_VERSION,
        "required_major": 25,
        "artifacts": {
            "linux-x64": {
                "url": f"{_JDK_BASE}/"
                       "OpenJDK25U-jdk_x64_linux_hotspot_25.0.5_8"
                       ".tar.gz",
                "sha256": "68982529869257d0ee6b824ec5aa270ff8b118bb"
                          "438670da0b97ebdab178d7fb",
                "kind": "tar.gz",
            },
            "linux-aarch64": {
                "url": f"{_JDK_BASE}/"
                       "OpenJDK25U-jdk_aarch64_linux_hotspot_25.0.5_8"
                       ".tar.gz",
                "sha256": "bb5b2ded252f21c6090cf780b0a408ad0d83bbb557"
                          "e702955643557f13231880",
                "kind": "tar.gz",
            },
            "windows-x64": {
                "url": f"{_JDK_BASE}/"
                       "OpenJDK25U-jdk_x64_windows_hotspot_25.0.5_8"
                       ".zip",
                "sha256": "636bdc13dc649ddc7d5921c5c7d23cca6e44fd19d662b"
                          "45cc9c02135a6c96c42",
                "kind": "zip",
            },
        },
    },
    "ghidra": {
        "version": "12.2",
        "artifacts": {
            "any": {
                "url": "https://github.com/NationalSecurityAgency/"
                       "ghidra/releases/download/Ghidra_12.2_build/"
                       "ghidra_12.2_PUBLIC_20261016.zip",
                "sha256": "2c2cd776db3c1f6a097ba418922378ac3649cadd5"
                          "e844a6f5f088ae628a7be67",
                "kind": "zip",
            },
        },
    },
    "unblob": {
        "version": "26.6.4",
        "kind": "pip-venv",
        "package": "unblob",
        "pins_file": "unblob_26.6.4.txt",
        "platforms": ("linux-x64", "linux-aarch64"),
    },
    "frida": {
        "version": "17.26.2",
        "kind": "pip-venv",
        "package": "frida",
        "pins_file": "frida_17.26.2.txt",
        "platforms": ("linux-x64", "linux-aarch64", "windows-x64"),
    },
    "gdb": {
        "kind": "apt",
        "package": "gdb",
        "platforms": ("linux-x64", "linux-aarch64"),
    },
}


# ── paths ───────────────────────────────────────────────────────────

def tools_root() -> Path:
    """The managed tools root (env-overridable, house convention)."""
    override = os.environ.get("NM_RE_TOOLS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "workspace" / "tools"


def re_root(root: Path | None = None) -> Path:
    """Where the RE engines live under the tools root."""
    return (root if root is not None else tools_root()) / "re"


def platform_key() -> str:
    """``linux-x64`` / ``linux-aarch64`` / ``windows-x64`` / …"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        machine = "x64"
    elif machine in ("aarch64", "arm64"):
        machine = "aarch64"
    return f"{system}-{machine}"


def _state_path(root: Path | None) -> Path:
    return re_root(root) / "state.json"


def _load_state(root: Path | None) -> dict[str, Any]:
    try:
        raw = _state_path(root).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(root: Path | None, state: dict[str, Any]) -> None:
    path = _state_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1, sort_keys=True),
                        encoding="utf-8")
    except OSError as exc:
        record_failure("reverse.provision.save_state", exc)


def _record_attempts(root: Path | None, engine: str,
                     attempts: list[str], **fields: Any) -> None:
    state = _load_state(root)
    engines = state.setdefault("engines", {})
    row = engines.setdefault(engine, {})
    history = row.setdefault("attempts", [])
    history.extend(attempts)
    row.update(fields)
    row["updated_at"] = time.time()
    _save_state(root, state)


def recorded_attempts(root: Path | None, engine: str) -> list[str]:
    row = (_load_state(root).get("engines") or {}).get(engine) or {}
    return list(row.get("attempts") or [])


# ── download / verify / extract ─────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "devon-reverse-provision"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if target != dest and dest not in target.parents:
            raise ProvisionError("extract", [
                f"zip entry escapes the install dir: {info.filename}"])
        zf.extract(info, dest)
        # zipfile drops the unix mode; the Ghidra zip's launch
        # scripts (analyzeHeadless and friends) are dead weight
        # without their exec bits, so restore them from the
        # archive's own external attributes.
        if info.create_system == 3 and not info.is_dir():
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                try:
                    os.chmod(target, mode)
                except OSError as exc:
                    record_failure("reverse.provision.chmod", exc)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if target != dest and dest not in target.parents:
            raise ProvisionError("extract", [
                f"tar entry escapes the install dir: {member.name}"])
        if member.islnk() or member.issym():
            link = (target.parent / member.linkname).resolve()
            if link != dest and dest not in link.parents:
                raise ProvisionError("extract", [
                    f"tar link escapes the install dir: {member.name}"])
    tf.extractall(dest, filter="data")


def _extract(archive: Path, kind: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest)
    elif kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as tf:
            _safe_extract_tar(tf, dest)
    else:  # pragma: no cover - descriptors are closed
        raise ProvisionError("extract", [f"unknown archive kind {kind}"])


def _installed_sha(dest: Path) -> str:
    try:
        return (dest / _MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _mark_installed(dest: Path, sha: str) -> None:
    (dest / _MARKER).write_text(sha + "\n", encoding="utf-8")


# ── probes ──────────────────────────────────────────────────────────

def _run(argv: list[str], *, env: dict[str, str] | None = None,
         timeout: float = PROBE_TIMEOUT_S) -> tuple[int, str]:
    done = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout,
        env=env)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


Runner = Callable[..., tuple[int, str]]


def _java_major(output: str) -> int | None:
    match = re.search(r'version "(\d+)', output or "")
    return int(match.group(1)) if match else None


def _java_version(java: Path, runner: Runner | None) -> tuple[int | None, str]:
    """Probe one java binary → (major, full first version line)."""
    run = runner or _run
    try:
        _rc, out = run([str(java), "-version"], timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    line = ""
    for candidate in (out or "").splitlines():
        if "version" in candidate:
            line = candidate.strip().strip('"')
            break
    return _java_major(out), line


def _find_java_home(base: Path) -> Path | None:
    """The JDK home under an install dir (archive top-level layout)."""
    exe = "java.exe" if os.name == "nt" else "java"
    direct = base / "bin" / exe
    if direct.is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if (child / "bin" / exe).is_file():
                return child
    return None


def jdk_state(root: Path | None = None, *,
              runner: Runner | None = None) -> dict[str, Any]:
    """Probe the JDK rung: pinned install → system java → none.

    Ready requires a probed major >= the pinned descriptor's
    required major (Ghidra 12.2 refuses anything older, so an older
    java is reported present-but-unusable, never as ready).
    """
    required = int(DESCRIPTORS["jdk"]["required_major"])
    base = re_root(root) / "jdk"
    attempts: list[str] = []
    home = _find_java_home(base) if base.is_dir() else None
    if home is not None:
        pin = DESCRIPTORS["jdk"]["artifacts"].get(
            platform_key(), {}).get("sha256", "")
        exe = "java.exe" if os.name == "nt" else "java"
        major, line = _java_version(home / "bin" / exe, runner)
        if _installed_sha(base) != pin:
            attempts.append(
                "pinned JDK install present but its install marker "
                "does not match the pin — re-provision replaces it")
        elif major is not None and major >= required:
            return {"engine": "jdk", "state": "ready",
                    "available": True, "source": "provisioned",
                    "version": line or DESCRIPTORS["jdk"]["version"],
                    "path": str(home), "attempts": attempts,
                    "detail": f"pinned Temurin JDK at {home} "
                              f"(probe: {line})"}
        else:
            attempts.append(
                f"pinned JDK at {home} probed at major {major} — "
                f"Ghidra 12.2 requires {required}+")
    system_java = shutil.which("java")
    if system_java:
        major, line = _java_version(Path(system_java), runner)
        if major is not None and major >= required:
            return {"engine": "jdk", "state": "ready",
                    "available": True, "source": "system",
                    "version": line, "path": system_java,
                    "attempts": attempts,
                    "detail": f"system java probes at {line} "
                              f"(>= {required}, usable by Ghidra)"}
        attempts.append(
            f"system java at {system_java} probes at major {major} "
            f"— Ghidra 12.2 requires {required}+")
    key = platform_key()
    provisionable = key in DESCRIPTORS["jdk"]["artifacts"]
    return {"engine": "jdk",
            "state": "provisionable" if provisionable else "unavailable",
            "available": False, "source": "",
            "version": DESCRIPTORS["jdk"]["version"], "path": "",
            "attempts": attempts + recorded_attempts(root, "jdk"),
            "detail": ("no JDK >= " + str(required) + " on this "
                       "machine yet — the pinned Temurin "
                       + DESCRIPTORS["jdk"]["version"]
                       + (" provisions on demand"
                          if provisionable else
                          f"; no pinned artifact for {key}"))}


def _ghidra_home(base: Path) -> Path | None:
    script = "analyzeHeadless.bat" if os.name == "nt" \
        else "analyzeHeadless"
    if (base / "support" / script).is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if (child / "support" / script).is_file():
                return child
    return None


def _ghidra_version(home: Path) -> str:
    try:
        text = (home / "Ghidra" / "application.properties").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("application.version="):
            return line.split("=", 1)[1].strip()
    return ""


def ghidra_state(root: Path | None = None, *,
                 runner: Runner | None = None) -> dict[str, Any]:
    """Probe the Ghidra rung: pinned install + a usable JDK."""
    base = re_root(root) / "ghidra"
    pin = DESCRIPTORS["ghidra"]["artifacts"]["any"]["sha256"]
    home = _ghidra_home(base) if base.is_dir() else None
    attempts: list[str] = []
    if home is None:
        return {"engine": "ghidra", "state": "provisionable",
                "available": False, "source": "",
                "version": DESCRIPTORS["ghidra"]["version"],
                "path": "",
                "attempts": recorded_attempts(root, "ghidra"),
                "detail": "Ghidra is not provisioned on this "
                          "machine yet — the pinned Ghidra "
                          + DESCRIPTORS["ghidra"]["version"]
                          + " provisions on demand"}
    if _installed_sha(base) != pin:
        return {"engine": "ghidra", "state": "provisionable",
                "available": False, "source": "",
                "version": _ghidra_version(home)
                           or DESCRIPTORS["ghidra"]["version"],
                "path": str(home),
                "attempts": recorded_attempts(root, "ghidra"),
                "detail": "a Ghidra install is present but its "
                          "install marker does not match the pin — "
                          "re-provision replaces it before it is "
                          "trusted"}
    jdk = jdk_state(root, runner=runner)
    if not jdk["available"]:
        attempts.append("Ghidra is installed but no usable JDK "
                        "resolves: " + jdk["detail"])
        return {"engine": "ghidra", "state": "unavailable",
                "available": False, "source": "provisioned",
                "version": _ghidra_version(home), "path": str(home),
                "attempts": attempts
                            + recorded_attempts(root, "ghidra"),
                "detail": attempts[0]}
    version = _ghidra_version(home) or DESCRIPTORS["ghidra"]["version"]
    return {"engine": "ghidra", "state": "ready", "available": True,
            "source": "provisioned", "version": version,
            "path": str(home), "attempts": attempts,
            "detail": f"pinned Ghidra {version} at {home} "
                      f"(JDK: {jdk['detail']})"}


# ── unblob: the pip-venv engine (Pass 3) ────────────────────────────

def _pins_path(engine: str) -> Path:
    """The pinned install set for a ``pip-venv`` engine."""
    return (Path(__file__).parent / "pins"
            / DESCRIPTORS[engine]["pins_file"])


def unblob_pins_path() -> Path:
    """The pinned install set for the pip-venv engine."""
    return _pins_path("unblob")


def frida_pins_path() -> Path:
    """The pinned install set for the Frida dynamic rung."""
    return _pins_path("frida")


def _pins_sha(engine: str) -> str:
    """The pin identity of a pip-venv engine: the sha256 of its
    pinned requirements file — the hash set IS the pin."""
    try:
        return _sha256(_pins_path(engine))
    except OSError:
        return ""


def _venv_bin(base: Path) -> Path:
    return base / "venv" / ("Scripts" if os.name == "nt" else "bin")


def _unblob_probe(exe: Path, bindir: Path,
                  runner: Runner | None) -> str:
    """Probe one unblob executable → its version line, or ""."""
    run = runner or _run
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    try:
        _rc, out = run([str(exe), "--version"], timeout=PROBE_TIMEOUT_S,
                       env=env)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in (out or "").splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            return line
    return ""


def unblob_state(root: Path | None = None, *,
                 runner: Runner | None = None) -> dict[str, Any]:
    """Probe the firmware rung's engine: pinned venv → system → none.

    Ready requires the install marker to match the pinned
    requirements file's own sha256 AND the executable to answer
    its version probe — an install whose pin set changed under it
    is reported provisionable, never trusted.
    """
    version = DESCRIPTORS["unblob"]["version"]
    base = re_root(root) / "unblob"
    bindir = _venv_bin(base)
    exe = bindir / ("unblob.exe" if os.name == "nt" else "unblob")
    attempts: list[str] = []
    if exe.is_file():
        if _installed_sha(base) != _pins_sha("unblob"):
            attempts.append(
                "pinned unblob venv present but its install marker "
                "does not match the pinned requirements — "
                "re-provision replaces it")
        else:
            probed = _unblob_probe(exe, bindir, runner)
            if probed:
                return {"engine": "unblob", "state": "ready",
                        "available": True, "source": "provisioned",
                        "version": probed, "path": str(exe),
                        "attempts": attempts,
                        "detail": f"pinned unblob {probed} at {exe} "
                                  "(pip --require-hashes venv)"}
            attempts.append(
                f"pinned unblob at {exe} did not answer its "
                "version probe — the install is not trusted")
            return {"engine": "unblob", "state": "unavailable",
                    "available": False, "source": "provisioned",
                    "version": version, "path": str(exe),
                    "attempts": attempts
                    + recorded_attempts(root, "unblob"),
                    "detail": attempts[0]}
    system = shutil.which("unblob")
    if system:
        probed = _unblob_probe(Path(system), Path(system).parent,
                               runner)
        if probed:
            return {"engine": "unblob", "state": "ready",
                    "available": True, "source": "system",
                    "version": probed, "path": system,
                    "attempts": attempts,
                    "detail": f"system unblob probed at {system} "
                              f"({probed})"}
        attempts.append(f"system unblob at {system} did not "
                        "answer its version probe")
    key = platform_key()
    provisionable = key in DESCRIPTORS["unblob"]["platforms"]
    return {"engine": "unblob",
            "state": "provisionable" if provisionable
                     else "unavailable",
            "available": False, "source": "", "version": version,
            "path": "",
            "attempts": attempts + recorded_attempts(root, "unblob"),
            "detail": ("unblob is not provisioned on this machine "
                       "yet — the pinned unblob " + version
                       + " (pip --require-hashes, PyPI digests) "
                         "provisions on demand"
                       if provisionable else
                       "unblob's native dependencies publish no "
                       f"wheels for {key} — the firmware extraction "
                       "rung is Linux-profile")}


def _frida_probe(exe: Path, bindir: Path,
                  runner: Runner | None) -> str:
    """Probe one frida python → its version, or "".

    The probe imports frida in the candidate interpreter — an
    install whose import fails (or answers no version) is not
    trusted, however the venv looks on disk.
    """
    run = runner or _run
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    try:
        _rc, out = run(
            [str(exe), "-c",
             "import frida; print(frida.__version__)"],
            timeout=PROBE_TIMEOUT_S, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in (out or "").splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            return line
    return ""


def frida_state(root: Path | None = None, *,
                runner: Runner | None = None) -> dict[str, Any]:
    """Probe the Frida dynamic rung: pinned venv → system → none.

    Ready requires the install marker to match the pinned
    requirements file's own sha256 AND the interpreter to answer
    the frida import probe — an install whose pin set changed
    under it is reported provisionable, never trusted.
    """
    version = DESCRIPTORS["frida"]["version"]
    base = re_root(root) / "frida"
    bindir = _venv_bin(base)
    exe = bindir / ("python.exe" if os.name == "nt" else "python")
    attempts: list[str] = []
    if exe.is_file():
        if _installed_sha(base) != _pins_sha("frida"):
            attempts.append(
                "pinned frida venv present but its install marker "
                "does not match the pinned requirements — "
                "re-provision replaces it")
        else:
            probed = _frida_probe(exe, bindir, runner)
            if probed:
                return {"engine": "frida", "state": "ready",
                        "available": True, "source": "provisioned",
                        "version": probed, "path": str(exe),
                        "attempts": attempts,
                        "detail": f"pinned frida {probed} at {exe} "
                                  "(pip --require-hashes venv)"}
            attempts.append(
                f"pinned frida at {exe} did not answer its "
                "import probe — the install is not trusted")
            return {"engine": "frida", "state": "unavailable",
                    "available": False, "source": "provisioned",
                    "version": version, "path": str(exe),
                    "attempts": attempts
                    + recorded_attempts(root, "frida"),
                    "detail": attempts[0]}
    probed = _frida_probe(Path(sys.executable), Path(sys.executable).parent,
                          runner)
    if probed:
        return {"engine": "frida", "state": "ready",
                "available": True, "source": "system",
                "version": probed, "path": sys.executable,
                "attempts": attempts,
                "detail": f"system python already imports frida "
                          f"({probed})"}
    key = platform_key()
    provisionable = key in DESCRIPTORS["frida"]["platforms"]
    return {"engine": "frida",
            "state": "provisionable" if provisionable
                     else "unavailable",
            "available": False, "source": "", "version": version,
            "path": "",
            "attempts": attempts + recorded_attempts(root, "frida"),
            "detail": ("frida is not provisioned on this machine "
                       "yet — the pinned frida " + version
                       + " (pip --require-hashes, PyPI digests) "
                         "provisions on demand"
                       if provisionable else
                       "frida publishes no wheel for "
                       f"{key} — the Frida rung cannot land here")}


def _gdb_version(exe: str, runner: Runner | None) -> str:
    """Probe one gdb binary → its version number, or ""."""
    run = runner or _run
    try:
        _rc, out = run([exe, "--version"], timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    first = ((out or "").splitlines() or [""])[0]
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", first)
    return match.group(1) if match else ""


def _apt_available() -> bool:
    """Is there an apt channel on this machine at all?"""
    return (platform.system().lower() == "linux"
            and shutil.which("apt-get") is not None)


def gdb_state(root: Path | None = None, *,
              runner: Runner | None = None) -> dict[str, Any]:
    """Probe the GDB dynamic rung: the system gdb, or none.

    GDB is a system debugger, not a downloadable archive — the
    probe is the binary on PATH answering ``--version``. There is
    no provisioned-venv form; the provision drive installs it
    through apt (see :func:`provision_engine`).
    """
    exe = shutil.which("gdb")
    attempts: list[str] = []
    if exe:
        version = _gdb_version(exe, runner)
        if version:
            return {"engine": "gdb", "state": "ready",
                    "available": True, "source": "system",
                    "version": version, "path": exe,
                    "attempts": attempts,
                    "detail": f"system gdb {version} at {exe}"}
        attempts.append(
            f"a gdb binary is present at {exe} but it did not "
            "answer its version probe — the install is not trusted")
        return {"engine": "gdb", "state": "unavailable",
                "available": False, "source": "system",
                "version": "", "path": exe,
                "attempts": attempts
                + recorded_attempts(root, "gdb"),
                "detail": attempts[0]}
    key = platform_key()
    provisionable = (key in DESCRIPTORS["gdb"]["platforms"]
                     and _apt_available())
    return {"engine": "gdb",
            "state": "provisionable" if provisionable
                     else "unavailable",
            "available": False, "source": "", "version": "",
            "path": "",
            "attempts": attempts + recorded_attempts(root, "gdb"),
            "detail": ("gdb is not installed on this machine yet — "
                       "it provisions on demand through apt "
                       "(Debian/Ubuntu)"
                       if provisionable else
                       "no gdb on this machine and no apt channel "
                       f"to install it from ({key}) — install GDB "
                       "via MSYS2 (pacman -S gdb) or MinGW")}


def _provision_pip_venv(engine: str, *, root: Path | None,
                        force: bool,
                        runner: Runner | None) -> dict[str, Any]:
    """The provisioning drive for a ``pip-venv`` descriptor.

    Create the venv → ``pip install --require-hashes`` the pinned
    requirements (pip itself refuses any wheel whose digest is not
    in the pinned set) → mark the install with the pin set's sha256
    → probe-execute. Every failure is a named attempt; nothing is
    installed by a step whose verification failed.
    """
    descriptor = DESCRIPTORS[engine]
    state_fn = {"unblob": unblob_state, "frida": frida_state}[engine]
    attempts: list[str] = []
    if not force:
        current = state_fn(root, runner=runner)
        if current["available"]:
            attempts.append(f"{engine} already provisioned "
                            f"({current['detail']})")
            _record_attempts(root, engine, attempts,
                             version=current["version"],
                             source=current["source"])
            return {**current, "attempts":
                    attempts + list(current.get("attempts") or [])}
    key = platform_key()
    if key not in descriptor["platforms"]:
        attempts.append(f"no pinned {engine} install set for "
                        f"platform {key} (Linux-profile engine)")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    pins = _pins_path(engine)
    if not pins.is_file():
        attempts.append(f"pinned requirements file missing: {pins}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    base = re_root(root) / engine
    venv = base / "venv"
    run = runner or _run
    try:
        if venv.exists():
            shutil.rmtree(venv)
        base.mkdir(parents=True, exist_ok=True)
        rc, out = run([sys.executable, "-m", "venv", str(venv)],
                      timeout=300.0)
        if rc != 0 or not _venv_bin(base).is_dir():
            tail = " · ".join((out or "").strip().splitlines()[-3:])
            attempts.append(f"venv creation failed (exit {rc}): "
                            f"{tail}")
            _record_attempts(root, engine, attempts)
            raise ProvisionError(engine, attempts)
    except ProvisionError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempts.append(f"venv creation failed: {exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    attempts.append(f"created the engine venv at {venv}")
    bindir = _venv_bin(base)
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    try:
        rc, out = run([str(py), "-m", "pip", "install",
                       "--require-hashes", "-r", str(pins)],
                      timeout=DOWNLOAD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempts.append(f"pip install of the pinned set failed: "
                        f"{exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    if rc != 0:
        tail = " · ".join((out or "").strip().splitlines()[-3:])
        attempts.append(
            f"pip install --require-hashes failed (exit {rc}) — "
            f"nothing is trusted from a failed pinned install: "
            f"{tail}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append(
        "pip installed the pinned set under --require-hashes "
        "(every wheel matched a PyPI-published digest)")
    _mark_installed(base, _pins_sha(engine))
    attempts.append(f"installed {engine} {descriptor['version']} "
                    f"under {base}")
    state = state_fn(root, runner=runner)
    if not state["available"]:
        attempts.append("post-install probe failed: "
                        + state["detail"])
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append("post-install probe passed")
    _record_attempts(root, engine, attempts,
                     version=descriptor["version"],
                     sha256=_pins_sha(engine), source="provisioned")
    return {**state, "attempts": attempts}


def _provision_apt(engine: str, *, root: Path | None,
                   force: bool,
                   runner: Runner | None) -> dict[str, Any]:
    """The provisioning drive for an ``apt`` descriptor (GDB).

    GDB is a system debugger: the channel is the distro package
    manager, not a pinned archive. The drive checks root (apt
    needs it), runs ``apt-get update`` then ``apt-get install -y
    <package>``, and trusts the install only after the version
    probe answers — the exact installed version is recorded in
    the state ledger, which is the pin for a distro package.
    Every failure is a named attempt.
    """
    descriptor = DESCRIPTORS[engine]
    state_fn = {"gdb": gdb_state}[engine]
    attempts: list[str] = []
    if not force:
        current = state_fn(root, runner=runner)
        if current["available"]:
            attempts.append(f"{engine} already provisioned "
                            f"({current['detail']})")
            _record_attempts(root, engine, attempts,
                             version=current["version"],
                             source=current["source"])
            return {**current, "attempts":
                    attempts + list(current.get("attempts") or [])}
    key = platform_key()
    if key not in descriptor["platforms"] or not _apt_available():
        attempts.append(
            f"no apt channel for {engine} on {key} — install GDB "
            "via MSYS2 (pacman -S gdb) or MinGW")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    try:
        euid = os.geteuid()
    except AttributeError:  # pragma: no cover - no geteuid on Windows
        euid = -1
    if euid != 0:
        attempts.append(
            "apt install needs root (this process runs as uid "
            f"{euid}) — run the provision as root or install gdb "
            "through the system package manager")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    run = runner or _run
    try:
        rc, out = run(["apt-get", "update"], timeout=600.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempts.append(f"apt-get update failed: {exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    if rc != 0:
        tail = " · ".join((out or "").strip().splitlines()[-3:])
        attempts.append(f"apt-get update failed (exit {rc}): {tail}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append("apt-get update succeeded")
    try:
        rc, out = run(["apt-get", "install", "-y",
                       descriptor["package"]], timeout=DOWNLOAD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        attempts.append(f"apt-get install failed: {exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    if rc != 0:
        tail = " · ".join((out or "").strip().splitlines()[-3:])
        attempts.append(
            f"apt-get install -y {descriptor['package']} failed "
            f"(exit {rc}): {tail}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append(
        f"apt-get install -y {descriptor['package']} succeeded")
    state = state_fn(root, runner=runner)
    if not state["available"]:
        attempts.append("post-install probe failed: "
                        + state["detail"])
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append("post-install probe passed")
    _record_attempts(root, engine, attempts,
                     version=state["version"], source="system")
    return {**state, "attempts": attempts}


def engine_states(root: Path | None = None, *,
                  runner: Runner | None = None) -> dict[str, Any]:
    return {"jdk": jdk_state(root, runner=runner),
            "ghidra": ghidra_state(root, runner=runner),
            "unblob": unblob_state(root, runner=runner),
            "frida": frida_state(root, runner=runner),
            "gdb": gdb_state(root, runner=runner)}


# ── the provisioning drive ──────────────────────────────────────────

def _launch_probe(engine: str, home: Path, root: Path | None,
                  runner: Runner | None) -> str | None:
    """Post-install smoke probe. Returns None on success, else the
    named failure. JDK: ``java -version`` must parse at the required
    major. Ghidra: analyzeHeadless must launch under the resolved
    JDK and print its usage (exit code is meaningless there — the
    usage text is the proof of launch)."""
    run = runner or _run
    if engine == "jdk":
        exe = "java.exe" if os.name == "nt" else "java"
        major, line = _java_version(home / "bin" / exe, runner)
        required = int(DESCRIPTORS["jdk"]["required_major"])
        if major is None or major < required:
            return (f"installed JDK probed at major {major} "
                    f"({line or 'no version output'}) — requires "
                    f"{required}+")
        return None
    jdk = jdk_state(root, runner=runner)
    if not jdk["available"]:
        return "no usable JDK after install: " + jdk["detail"]
    script = "analyzeHeadless.bat" if os.name == "nt" \
        else "analyzeHeadless"
    env = dict(os.environ)
    jdk_home = jdk["path"]
    if jdk["source"] == "system":
        jdk_home = str(Path(jdk_home).parent.parent)
    env["JAVA_HOME"] = jdk_home
    try:
        _rc, out = run([str(home / "support" / script)],
                       env=env, timeout=LAUNCH_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"analyzeHeadless launch probe failed: {exc}"
    if "analyzeHeadless" not in (out or ""):
        tail = " · ".join((out or "").strip().splitlines()[-3:])
        return f"analyzeHeadless launched without its usage text: {tail}"
    return None


def provision_engine(engine: str, *, root: Path | None = None,
                     force: bool = False,
                     downloader: Callable[[str, Path], None] | None
                     = None,
                     runner: Runner | None = None) -> dict[str, Any]:
    """Provision one engine under the pin discipline.

    Resolve → (already ready? stop) → download the pinned artifact
    → sha256 verify (a mismatch installs NOTHING) → extract beside
    the target and swap in → mark with the pin → probe-execute.
    Every step that fails is a named attempt; total failure raises
    :class:`ProvisionError` carrying them, after they are recorded
    in the state ledger.
    """
    if engine not in DESCRIPTORS:
        raise ProvisionError(engine, [f"no pinned descriptor for "
                                      f"{engine!r}"])
    descriptor = DESCRIPTORS[engine]
    if descriptor.get("kind") == "pip-venv":
        # a Python engine: pip does the downloading, under the
        # --require-hashes discipline of the pinned set (the
        # ``downloader`` seam does not apply)
        return _provision_pip_venv(engine, root=root, force=force,
                                   runner=runner)
    if descriptor.get("kind") == "apt":
        # a system debugger: the distro package manager is the
        # channel (the ``downloader`` seam does not apply)
        return _provision_apt(engine, root=root, force=force,
                              runner=runner)
    state_fn = jdk_state if engine == "jdk" else ghidra_state
    attempts: list[str] = []
    if not force:
        current = state_fn(root, runner=runner)
        if current["available"]:
            attempts.append(f"{engine} already provisioned "
                            f"({current['detail']})")
            _record_attempts(root, engine, attempts,
                             version=current["version"],
                             source=current["source"])
            return {**current, "attempts":
                    attempts + list(current.get("attempts") or [])}
    artifact = descriptor["artifacts"].get(platform_key()) \
        or descriptor["artifacts"].get("any")
    if artifact is None:
        attempts.append(f"no pinned {engine} artifact for platform "
                        f"{platform_key()}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    base = re_root(root)
    dl_dir = base / "dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    archive = dl_dir / Path(artifact["url"]).name
    download = downloader or _download_file
    try:
        download(artifact["url"], archive)
        attempts.append(f"downloaded {artifact['url']}")
    except Exception as exc:  # noqa: BLE001 - named attempt
        attempts.append(f"download of {artifact['url']} failed: {exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    got = _sha256(archive)
    if got != artifact["sha256"]:
        attempts.append(
            f"sha256 mismatch for {archive.name}: got {got[:16]}…, "
            f"pin is {artifact['sha256'][:16]}… — nothing installed")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append(f"sha256 verified against the pin ({got[:16]}…)")
    dest = base / engine
    staging = base / f".staging-{engine}"
    try:
        if staging.exists():
            shutil.rmtree(staging)
        _extract(archive, artifact["kind"], staging)
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    except ProvisionError:
        raise
    except Exception as exc:  # noqa: BLE001 - named attempt
        attempts.append(f"extract/install failed: {exc}")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts) from exc
    _mark_installed(dest, artifact["sha256"])
    attempts.append(f"installed {engine} {descriptor['version']} "
                    f"under {dest}")
    home = _find_java_home(dest) if engine == "jdk" \
        else _ghidra_home(dest)
    if home is None:
        attempts.append("installed tree does not contain the "
                        "expected executable layout")
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    problem = _launch_probe(engine, home, root, runner)
    if problem is not None:
        attempts.append("post-install probe failed: " + problem)
        _record_attempts(root, engine, attempts)
        raise ProvisionError(engine, attempts)
    attempts.append("post-install probe passed")
    _record_attempts(root, engine, attempts,
                     version=descriptor["version"],
                     sha256=artifact["sha256"], source="provisioned")
    return {**state_fn(root, runner=runner), "attempts": attempts}


def provision_all(*, root: Path | None = None, force: bool = False,
                  downloader: Callable[[str, Path], None] | None
                  = None,
                  runner: Runner | None = None) -> dict[str, Any]:
    """Provision the deep-rung set: JDK first (Ghidra stands on
    it), then Ghidra. One engine's failure is reported in its own
    row with its attempts; it never masquerades as the other's."""
    report: dict[str, Any] = {"engines": {}, "ok": True}
    for engine in ("jdk", "ghidra"):
        try:
            row = provision_engine(engine, root=root, force=force,
                                   downloader=downloader,
                                   runner=runner)
        except ProvisionError as exc:
            row = {"engine": engine, "state": "unavailable",
                   "available": False, "source": "",
                   "version": DESCRIPTORS[engine]["version"],
                   "path": "", "attempts": exc.attempts,
                   "detail": str(exc)}
            record_failure(f"reverse.provision.{engine}", exc)
        report["engines"][engine] = row
        if not row.get("available"):
            report["ok"] = False
    return report
