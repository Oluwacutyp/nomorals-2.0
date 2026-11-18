"""Antivirus / malware scanning — one detection point for the engines
on this box, wired into the profile runtime (register §11(1), owner
scope: everything Devon can reach on every machine, profile runtime
sorts per-machine reach).

Every organ used to have no answer at all for "is this file safe" —
``payload_inspection.py`` names this scanner as future work ("the
future file/media malware scanner (register §11 — same extractors)").
This module answers it once, and the profile runtime carries the
answer:

* ``AntivirusInfo`` — which engines exist on this box, in precedence
  order, plus the primary engine and a human-readable detail line.
* ``detect_antivirus()`` — precedence: explicit ``pin`` argument →
  ``NM_AV`` env var → probe (Windows Defender via MpCmdRun, then
  ClamAV via ``clamscan``) → ``builtin`` floor rung. Detection never
  raises and never pretends: a pinned engine that is absent fails
  closed (no engine), and the builtin engine says plainly what it
  is — the EICAR test file only, never a full antivirus pass.
* ``scan_file()`` / ``scan_path()`` — scan one file, or a file or
  directory (walked, aggregate verdict). Verdicts are ``clean`` /
  ``infected`` / ``unknown``; ``unknown`` means "no determination"
  (no engine, file missing, engine error) and is NEVER reported as
  clean. A hit is quarantined (moved out of reach, sidecar record
  + journal line) when a quarantine dir is given.
* ``ScanVerdict`` — one scan answer, JSON-plain via ``to_dict()``.

Engine contracts (verified from the vendors' own docs, not guessed):

* Defender (Microsoft Learn, MpCmdRun): ``MpCmdRun.exe -Scan
  -ScanType 3 -File <path>`` — ScanType 3 is the custom scan;
  ``-File`` is valid only for custom scans. Return code 0 means no
  malware found OR malware found and successfully remediated; 2 means
  malware found and not remediated, user action required, or scanning
  errors. Both codes are ambiguous, so stdout is parsed for threat
  lines as well.
* ClamAV (clamscan man page): 0 = no virus found; 1 = virus(es)
  found; 2 = an error occurred — the scan did NOT produce a
  determination, and is never read as clean or infected.
* Builtin: hashes the file and checks the EICAR standard test string
  + its sha256 (derived at runtime from the EICAR string constant —
  no magic hash from memory). The EICAR string is assembled from
  parts so this file never contains the contiguous signature.

Probing is cached per process; ``refresh=True`` re-probes. Tests
that fake the probes must restore the cache to the real box
afterwards — see ``tests/test_core_antivirus.py``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AV_ENGINES",
    "AV_PIN_ENV",
    "AV_QUARANTINE_ENV",
    "AV_TIMEOUT_CLAMAV_ENV",
    "AV_TIMEOUT_DEFENDER_ENV",
    "EICAR_THREAT",
    "AntivirusInfo",
    "ScanVerdict",
    "detect_antivirus",
    "engine_timeouts",
    "quarantine_dir_for",
    "quarantine_file",
    "scan_file",
    "scan_path",
]

#: Engine names, in precedence order: the best engine on the box wins.
AV_ENGINES: tuple[str, ...] = ("defender", "clamav", "builtin")

#: Verdict values. "unknown" is a real answer — no determination —
#: never a synonym for clean.
VERDICTS: tuple[str, ...] = ("clean", "infected", "unknown")

#: Env vars: NM_AV pins the engine (auto|defender|clamav|builtin|none);
#: NM_AV_QUARANTINE overrides the quarantine directory (the tool layer
#: reads it; the organ takes the dir as an argument).
#: NM_AV_TIMEOUT_DEFENDER_S / NM_AV_TIMEOUT_CLAMAV_S set the per-engine
#: scan timeout in seconds (a positive number; anything else — unset,
#: unparseable, zero or negative — falls back to _DEFAULT_TIMEOUT_S).
#: An explicit timeout_s argument to scan_file/scan_path still wins.
AV_PIN_ENV = "NM_AV"
AV_QUARANTINE_ENV = "NM_AV_QUARANTINE"
AV_TIMEOUT_DEFENDER_ENV = "NM_AV_TIMEOUT_DEFENDER_S"
AV_TIMEOUT_CLAMAV_ENV = "NM_AV_TIMEOUT_CLAMAV_S"

#: The EICAR standard antivirus test string, assembled from parts so
#: this file never contains the contiguous signature (a scanner run
#: over the repo must not flag the scanner itself). Harmless by
#: design — it exists to be detected.
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
EICAR_STRING = _EICAR_A + _EICAR_B
EICAR_THREAT = "EICAR-Test-File"

#: Built-in known-bad hashes: the EICAR test file only. Derived at
#: runtime from the string above — no magic hash from memory.
_KNOWN_BAD: dict[str, str] = {
    hashlib.sha256(EICAR_STRING.encode("ascii")).hexdigest(): EICAR_THREAT,
}

#: Chunk size for streaming file reads (hash + content check in one pass).
_CHUNK = 1024 * 1024

#: Default per-engine-call timeout (Defender custom scans on large
#: files are slow; clamscan is usually fast). The per-engine values
#: below default to this; NM_AV_TIMEOUT_DEFENDER_S /
#: NM_AV_TIMEOUT_CLAMAV_S override one engine at a time.
_DEFAULT_TIMEOUT_S = 300.0

#: Default cap on files in one directory scan (a huge tree must not
#: hang the caller; the verdict says when it stopped early).
_DEFAULT_MAX_FILES = 1000


def _engine_timeout_s(engine: str) -> float:
    """Timeout for one engine's scan call.

    Env override for that engine (a positive number of seconds);
    anything else — unset, unparseable, zero or negative — falls
    back to the shared default. Never raises.
    """
    if engine == "defender":
        env_var = AV_TIMEOUT_DEFENDER_ENV
    elif engine == "clamav":
        env_var = AV_TIMEOUT_CLAMAV_ENV
    else:
        return _DEFAULT_TIMEOUT_S
    try:
        value = float((os.environ.get(env_var) or "").strip())
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


def engine_timeouts() -> dict[str, float]:
    """The live per-engine scan timeouts, in seconds.

    ``{"defender": ..., "clamav": ...}`` — the builtin engine runs
    in-process and takes no timeout. ``nm av status`` surfaces this.
    """
    return {"defender": _engine_timeout_s("defender"),
            "clamav": _engine_timeout_s("clamav")}


def _run(args: list[str], timeout: int | None = None,
         cwd: str | None = None) -> Any:
    """Subprocess seam (same shape as core.gpu._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd)


def _is_windows() -> bool:
    """Platform seam (tests fake Windows on this Linux box)."""
    return sys.platform == "win32"


def _find_mpcmdrun() -> str:
    """Locate MpCmdRun.exe. "" when Defender is not installed."""
    if not _is_windows():
        return ""
    found = shutil.which("MpCmdRun.exe")
    if found:
        return found
    program_data = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
    pattern = os.path.join(
        program_data, "Microsoft", "Windows Defender", "Platform",
        "*", "MpCmdRun.exe")
    try:
        candidates = sorted(glob.glob(pattern), reverse=True)
    except Exception:  # noqa: BLE001 - a failed glob is "not found"
        return ""
    return candidates[0] if candidates else ""


def _find_clamscan() -> str:
    """Locate clamscan. "" when ClamAV is not installed."""
    return shutil.which("clamscan") or ""


@dataclass(frozen=True)
class AntivirusInfo:
    """One antivirus-capability answer: which engines exist here."""

    engines: tuple[str, ...] = ()   # available, precedence order
    primary: str = ""               # engines[0], "" when none
    has_engine: bool = False
    detail: str = ""                # human-readable: what was found
    defender_path: str = ""         # MpCmdRun.exe, "" when absent
    clamav_path: str = ""           # clamscan, "" when absent

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines": list(self.engines),
            "primary": self.primary,
            "has_engine": self.has_engine,
            "detail": self.detail,
            "defender_path": self.defender_path,
            "clamav_path": self.clamav_path,
        }


#: Process-wide detection cache: probing spawns subprocesses, so
#: detection pays and the rest reuse it. refresh=True re-probes.
_AV_CACHE: AntivirusInfo | None = None


def _probe(pin: str | None) -> AntivirusInfo:
    """Probe the box once. Never raises — a failed probe is "absent"."""
    pin = (pin or "").strip().lower()
    if not pin or pin == "auto":
        pin = ""
    env_pin = (os.environ.get(AV_PIN_ENV) or "").strip().lower()
    if not pin and env_pin and env_pin != "auto":
        pin = env_pin
    if pin and pin not in AV_ENGINES and pin != "none":
        detail = (f"unknown {AV_PIN_ENV} pin {pin!r}, treated as auto")
        pin = ""
    else:
        detail = ""
    try:
        defender = _find_mpcmdrun()
    except Exception:  # noqa: BLE001 - a failed probe is "absent"
        defender = ""
    try:
        clamav = _find_clamscan()
    except Exception:  # noqa: BLE001 - a failed probe is "absent"
        clamav = ""
    available: list[str] = []
    if defender:
        available.append("defender")
    if clamav:
        available.append("clamav")
    available.append("builtin")
    if pin == "none":
        engines: tuple[str, ...] = ()
        bits = ["scanning disabled by pin (none)"]
    elif pin:
        engines = tuple(e for e in available if e == pin)
        if not engines:
            bits = [f"pinned engine {pin!r} not found on this box"]
        else:
            bits = [f"pinned to {pin}"]
    else:
        engines = tuple(available)
        bits = []
    if defender:
        bits.append(f"Defender via {defender}")
    if clamav:
        bits.append(f"ClamAV via {clamav}")
    if "builtin" in engines:
        bits.append("builtin signature set: EICAR test file only")
    if detail:
        bits.insert(0, detail)
    return AntivirusInfo(
        engines=engines,
        primary=engines[0] if engines else "",
        has_engine=bool(engines),
        detail="; ".join(bits) if bits else "no antivirus engine found",
        defender_path=defender,
        clamav_path=clamav,
    )


def detect_antivirus(*, pin: str | None = None,
                     refresh: bool = False) -> AntivirusInfo:
    """Which antivirus engines exist on this box.

    Precedence: explicit ``pin`` argument → ``NM_AV`` env var → probe
    (Defender, then ClamAV) → ``builtin`` floor rung. An explicit pin
    always wins: a pinned engine that is absent fails closed (no
    engine), it never silently falls back. ``pin="none"`` disables
    scanning entirely. Never raises.
    """
    global _AV_CACHE
    if not refresh and _AV_CACHE is not None and pin is None:
        return _AV_CACHE
    try:
        info = _probe(pin)
    except Exception:  # noqa: BLE001 - detection never raises
        info = AntivirusInfo(
            engines=("builtin",), primary="builtin", has_engine=True,
            detail="builtin signature set: EICAR test file only "
                   "(detection probe failed)")
    if pin is None:
        _AV_CACHE = info
    return info


def quarantine_dir_for(context: Any) -> Path | None:
    """Where infected files go. Env override first, then the Devon
    home's ``quarantine/`` dir, then the tool workspace's. None means
    report-only: hits are reported, never moved."""
    override = (os.environ.get(AV_QUARANTINE_ENV) or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    settings = getattr(context, "settings", None)
    if settings is not None:
        try:
            return settings.resolve("quarantine")
        except Exception:  # noqa: BLE001 - fall through to workspace
            pass
    workspace = getattr(context, "workspace", None)
    if workspace:
        return Path(workspace) / "quarantine"
    return None


# ── scanning ─────────────────────────────────────────────────────

@dataclass
class ScanVerdict:
    """One scan answer: verdict, engine, threat, and what happened."""

    verdict: str                    # clean | infected | unknown
    engine: str = ""                # defender | clamav | builtin | ""
    threat: str = ""                # threat name, "" when none
    sha256: str = ""                # file hash, "" when unread
    path: str = ""                  # scanned path as given
    quarantined_to: str = ""        # quarantine path, "" when not moved
    detail: str = ""                # one human line: why this verdict
    scanned_at: float = 0.0         # epoch seconds
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "engine": self.engine,
            "threat": self.threat,
            "sha256": self.sha256,
            "path": self.path,
            "quarantined_to": self.quarantined_to,
            "detail": self.detail,
            "scanned_at": self.scanned_at,
            "results": list(self.results),
        }


def _sha256_of(path: Path) -> str:
    """Streaming sha256. "" when the file cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _builtin_inspect(path: Path) -> tuple[str, str]:
    """Hash + EICAR content check in one streaming pass.

    Returns (sha256, threat-or-""). The EICAR marker is checked with
    a sliding overlap so it is caught even across chunk boundaries.
    """
    marker = EICAR_STRING.encode("ascii")
    digest = hashlib.sha256()
    threat = ""
    overlap = b""
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                window = overlap + chunk
                if marker in window:
                    threat = EICAR_THREAT
                overlap = window[-(len(marker) - 1):] if chunk else b""
    except OSError:
        return "", ""
    if not threat and digest.hexdigest() in _KNOWN_BAD:
        threat = _KNOWN_BAD[digest.hexdigest()]
    return digest.hexdigest(), threat


#: Malware-family prefixes MpCmdRun/clamscan name threats with.
_THREAT_PREFIXES = ("trojan:", "virus:", "worm:", "backdoor:",
                    "ransomware:", "spyware:", "adware:", "pua:",
                    "hacktool:", "exploit:", "malware:", "rootkit:",
                    "keylogger:", "dropper:", "eicar")

#: Negations that turn a "threat" mention into a clean report —
#: Defender's own clean line is "No threats found".
_THREAT_NEGATIONS = ("no threat", "no malware", "not found",
                     "0 threat", "zero threat")


def _threat_line(output: str) -> str:
    """First output line that names a found threat (best-effort
    threat name). "" when the engine named nothing.

    A bare "threat" substring is not enough — Defender's clean
    report itself reads "No threats found" — so negated lines are
    skipped and a line must carry a malware-family prefix or an
    explicit "threat detected/found".
    """
    for line in (output or "").splitlines():
        low = line.lower()
        if any(neg in low for neg in _THREAT_NEGATIONS):
            continue
        if any(p in low for p in _THREAT_PREFIXES):
            return line.strip()[:200]
        if "threat detected" in low or "threat found" in low:
            return line.strip()[:200]
    return ""


def _scan_with_defender(path: Path, mpcmdrun: str,
                        timeout_s: float) -> ScanVerdict:
    """Custom Defender scan of one file via MpCmdRun.

    Return code 0 means no malware OR malware found and remediated;
    2 means malware not remediated, user action required, or a
    scanning error — so stdout is parsed for threat lines as well,
    and a 2 with no threat named is "unknown", never a guess.
    """
    now = time.time()
    try:
        proc = _run([mpcmdrun, "-Scan", "-ScanType", "3", "-File",
                     str(path)], timeout=int(timeout_s))
    except Exception as exc:  # noqa: BLE001 - engine failure is data
        return ScanVerdict(
            verdict="unknown", engine="defender", path=str(path),
            detail=f"Defender scan failed to run: {exc}",
            scanned_at=now)
    out = (getattr(proc, "stdout", "") or "") + "\n" + (
        getattr(proc, "stderr", "") or "")
    threat_line = _threat_line(out)
    rc = getattr(proc, "returncode", -1)
    if rc == 0 and not threat_line:
        return ScanVerdict(
            verdict="clean", engine="defender", path=str(path),
            detail="Defender custom scan (MpCmdRun): no threats found",
            scanned_at=now)
    if rc == 0:
        return ScanVerdict(
            verdict="infected", engine="defender",
            threat=threat_line, path=str(path),
            detail="Defender found a threat and remediated it (exit 0)",
            scanned_at=now)
    if rc == 2 and threat_line:
        return ScanVerdict(
            verdict="infected", engine="defender",
            threat=threat_line, path=str(path),
            detail="Defender found a threat (exit 2, not remediated)",
            scanned_at=now)
    return ScanVerdict(
        verdict="unknown", engine="defender", path=str(path),
        detail=f"Defender returned {rc} with no threat named — "
               "scanning error, no determination",
        scanned_at=now)


def _scan_with_clamav(path: Path, timeout_s: float) -> ScanVerdict:
    """One file via clamscan. Exit 2 is an engine error — "I don't
    know" — and is never read as clean or infected."""
    now = time.time()
    try:
        proc = _run(["clamscan", "--no-summary", str(path)],
                    timeout=int(timeout_s))
    except Exception as exc:  # noqa: BLE001 - engine failure is data
        return ScanVerdict(
            verdict="unknown", engine="clamav", path=str(path),
            detail=f"clamscan failed to run: {exc}",
            scanned_at=now)
    out = getattr(proc, "stdout", "") or ""
    rc = getattr(proc, "returncode", -1)
    if rc == 0:
        return ScanVerdict(
            verdict="clean", engine="clamav", path=str(path),
            detail="clamscan: no virus found",
            scanned_at=now)
    if rc == 1:
        threat = ""
        for line in out.splitlines():
            if " FOUND" in line:
                threat = line.rsplit(":", 1)[-1].replace(
                    " FOUND", "").strip()[:200]
                break
        return ScanVerdict(
            verdict="infected", engine="clamav",
            threat=threat or "unknown (clamscan exit 1)",
            path=str(path),
            detail="clamscan: virus found",
            scanned_at=now)
    return ScanVerdict(
        verdict="unknown", engine="clamav", path=str(path),
        detail=f"clamscan returned {rc} — engine error, "
               "no determination",
        scanned_at=now)


def _scan_with_builtin(path: Path) -> ScanVerdict:
    """Hash + EICAR check. A clean verdict here is explicitly
    limited — it means no known-bad signature matched, never a full
    antivirus pass."""
    now = time.time()
    sha, threat = _builtin_inspect(path)
    if not sha:
        return ScanVerdict(
            verdict="unknown", engine="builtin", path=str(path),
            detail="builtin scan could not read the file",
            scanned_at=now)
    if threat:
        return ScanVerdict(
            verdict="infected", engine="builtin", threat=threat,
            sha256=sha, path=str(path),
            detail=f"matched built-in signature {threat}",
            scanned_at=now)
    return ScanVerdict(
        verdict="clean", engine="builtin", sha256=sha, path=str(path),
        detail="builtin signature set: EICAR test file only — no "
               "known-bad signature matched; this is not a full "
               "antivirus pass",
        scanned_at=now)


def _journal(quarantine_dir: Path, record: dict[str, Any]) -> None:
    """Append one scan record to the journal. Best-effort — a failed
    journal write never breaks the scan it records."""
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with open(quarantine_dir / "scan-journal.jsonl", "a",
                  encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        _log.debug("antivirus journal write failed", exc_info=True)


def quarantine_file(path: Path, verdict: ScanVerdict,
                    quarantine_dir: Path) -> str:
    """Move an infected file out of reach. Returns the quarantine
    path. Raises OSError when the move fails — the caller records
    that honestly and the file stays where it was (never deleted
    without a successful move)."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    digest = verdict.sha256 or _sha256_of(path)
    stem = f"{digest[:12]}_{path.name}" if digest else path.name
    dest = quarantine_dir / stem
    n = 2
    while dest.exists():
        dest = quarantine_dir / f"{stem}-{n}"
        n += 1
    shutil.move(str(path), str(dest))
    sidecar = Path(str(dest) + ".json")
    sidecar.write_text(json.dumps({
        "verdict": verdict.verdict,
        "engine": verdict.engine,
        "threat": verdict.threat,
        "sha256": verdict.sha256 or digest,
        "original_path": str(path),
        "quarantined_to": str(dest),
        "scanned_at": verdict.scanned_at,
        "detail": verdict.detail,
    }, indent=1), encoding="utf-8")
    return str(dest)


def scan_file(path: str | Path, *, engine: str | None = None,
             quarantine_dir: str | Path | None = None,
             timeout_s: float | None = None) -> ScanVerdict:
    """Scan one file. Never raises for a scan outcome — failures are
    ``unknown`` verdicts with the reason in ``detail``.

    ``timeout_s`` None (the default) means the engine's own timeout
    (``engine_timeouts()`` — env override per engine, 300s default);
    an explicit value overrides both engines.
    """
    target = Path(path)
    now = time.time()
    qdir = Path(quarantine_dir) if quarantine_dir else None
    if not target.exists():
        return ScanVerdict(
            verdict="unknown", engine=engine or "",
            path=str(target), detail="file not found",
            scanned_at=now)
    if target.is_dir():
        return ScanVerdict(
            verdict="unknown", engine=engine or "",
            path=str(target),
            detail="is a directory — use scan_path for directories",
            scanned_at=now)
    info = detect_antivirus()
    chosen = (engine or "").strip().lower() or info.primary
    if not chosen:
        return ScanVerdict(
            verdict="unknown", path=str(target),
            detail="no antivirus engine available on this box",
            scanned_at=now)
    if chosen not in AV_ENGINES:
        return ScanVerdict(
            verdict="unknown", path=str(target),
            detail=f"unknown engine {chosen!r}",
            scanned_at=now)
    if chosen == "defender":
        if not info.defender_path:
            return ScanVerdict(
                verdict="unknown", path=str(target),
                detail="Defender is not installed on this box",
                scanned_at=now)
        if timeout_s is None:
            timeout_s = _engine_timeout_s("defender")
        verdict = _scan_with_defender(target, info.defender_path,
                                      timeout_s)
    elif chosen == "clamav":
        if not info.clamav_path:
            return ScanVerdict(
                verdict="unknown", path=str(target),
                detail="ClamAV is not installed on this box",
                scanned_at=now)
        if timeout_s is None:
            timeout_s = _engine_timeout_s("clamav")
        verdict = _scan_with_clamav(target, timeout_s)
    else:
        verdict = _scan_with_builtin(target)
    if not verdict.sha256:
        verdict.sha256 = _sha256_of(target)
    if verdict.verdict == "infected" and qdir is not None:
        try:
            verdict.quarantined_to = quarantine_file(
                target, verdict, qdir)
        except OSError as exc:
            verdict.detail += (f"; quarantine failed: {exc} — file "
                               "left in place")
    if qdir is not None:
        record = verdict.to_dict()
        record["scope"] = "file"
        _journal(qdir, record)
    return verdict


def scan_path(target: str | Path, *, engine: str | None = None,
              quarantine_dir: str | Path | None = None,
              timeout_s: float | None = None,
              max_files: int = _DEFAULT_MAX_FILES) -> ScanVerdict:
    """Scan a file or a directory (walked, sorted, aggregate verdict).

    Infected if any file is infected; unknown if any file is unknown
    and none is infected; clean otherwise. Files under the quarantine
    dir itself are never re-scanned. Stops after ``max_files`` files
    and says so in the detail. ``timeout_s`` None means each file
    takes its engine's own timeout (see ``scan_file``).
    """
    root = Path(target)
    now = time.time()
    qdir = Path(quarantine_dir) if quarantine_dir else None
    if not root.exists():
        return ScanVerdict(
            verdict="unknown", path=str(root), detail="not found",
            scanned_at=now)
    if root.is_file():
        return scan_file(root, engine=engine, quarantine_dir=qdir,
                         timeout_s=timeout_s)
    qdir_resolved = qdir.resolve() if qdir is not None else None
    files: list[Path] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        if qdir_resolved is not None:
            try:
                here.resolve().relative_to(qdir_resolved)
                dirnames[:] = []
                continue
            except ValueError:
                pass
            dirnames[:] = [d for d in dirnames
                           if (here / d).resolve() != qdir_resolved]
        for name in sorted(filenames):
            files.append(here / name)
            if len(files) >= max_files:
                truncated = True
                break
        if truncated:
            break
    results: list[ScanVerdict] = []
    for f in files:
        results.append(scan_file(f, engine=engine,
                                 quarantine_dir=qdir,
                                 timeout_s=timeout_s))
    verdicts = [r.verdict for r in results]
    if "infected" in verdicts:
        verdict = "infected"
    elif "unknown" in verdicts:
        verdict = "unknown"
    else:
        verdict = "clean"
    threats = sorted({r.threat for r in results if r.threat})
    detail = f"scanned {len(results)} file(s)"
    if truncated:
        detail += f" (stopped at the {max_files}-file cap)"
    if threats:
        detail += f"; threats: {', '.join(threats)}"
    out = ScanVerdict(
        verdict=verdict,
        engine=results[0].engine if results else "",
        threat=threats[0] if threats else "",
        path=str(root),
        detail=detail,
        scanned_at=now,
        results=[r.to_dict() for r in results],
    )
    if qdir is not None:
        record = out.to_dict()
        record["scope"] = "path"
        _journal(qdir, record)
    return out
