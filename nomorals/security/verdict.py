"""Multi-engine verdict — the VirusTotal shape, Devon-local
(52-websites #24).

One artefact in → many independent engines → a verdict record
with per-engine results.  The engines are Devon's own local
rungs, composed here instead of a second scanner (SOUL §5):

* ``hash-reputation`` — the sha256 against the known-bad hash
  set (the EICAR test hash, derived at runtime from the AV
  organ's string — the set is small and the report says so).
* ``local-av`` — ``core.antivirus.scan_file`` (Defender, ClamAV,
  or the builtin EICAR-only rung), mapped clean → clean,
  infected → flagged, unknown → unknown.
* ``pattern-battery`` — the secret-pattern battery
  (``security.audit_patterns``) over the content as text, hits
  reported redacted, never in full.
* ``structure`` — magic bytes vs declared extension
  (``core.decoder``), double-extension and type-mismatch lies.
* ``sandbox`` — detonation of script content in the sandbox.
  The runner is INJECTED (``sandbox_fn``): this organ is L3 and
  the sandbox lives in ``tools`` (L4).  When nothing is injected
  the engine reports ``unknown`` — "not wired on this call
  path" — never a silent pass.

Hash-first: a subject whose sha256 already has a recorded
verdict returns it without re-scanning content (``from_cache``).
Content stays on the box unless the owner sends it out — the
URL rung fetches bytes in and scans them locally; nothing is
uploaded anywhere.  The report states the engine count
honestly: Devon runs five local engines, not VirusTotal's 70+
commercial ones.  Her compensating edge is locality: a
confidential file can be scanned at all, which VirusTotal's
sharing model forbids.

Every engine reports ``clean`` / ``flagged`` / ``unknown``.
An engine that errors reports ``unknown``, never a silent pass.
Aggregate: flagged if any engine flags, unknown if any engine
is unknown, clean only when every engine that could measure
says clean.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.antivirus import EICAR_STRING, EICAR_THREAT, scan_file
from ..core.decoder import (
    extension_for_magic,
    identify_magic,
    looks_like_text,
)
from ..core.http import HttpClient, url_filename
from ..core.logging_setup import get_logger
from .audit_patterns import entropy_hits, scan_text_for_secrets

_log = get_logger(__name__)

__all__ = [
    "EngineResult",
    "ScanSubject",
    "VerdictReport",
    "VerdictStore",
    "format_verdict_reply",
    "verdict_for",
]

#: Verdict values, shared with the AV organ. "unknown" is a real
#: answer — no determination — never a synonym for clean.
VERDICTS: tuple[str, ...] = ("clean", "flagged", "unknown")

#: The engines, in run order. The count is part of the honest
#: report — the reply names it, never "70+".
ENGINE_NAMES: tuple[str, ...] = (
    "hash-reputation",
    "local-av",
    "pattern-battery",
    "structure",
    "sandbox",
)

#: Content is loaded fully only up to this size; above it the
#: hash + local-av (both streaming) still run and the content
#: engines report unknown with the size named.
MAX_CONTENT_BYTES = 64 * 1024 * 1024

#: How much text the pattern battery reads (the audit organs
#: scan page source and bundles the same way).
TEXT_SCAN_CAP = 2 * 1024 * 1024

#: Bytes the structure engine sniffs for magic.
MAGIC_SNIFF_BYTES = 4096

#: URL fetch ceiling + timeout.
URL_TIMEOUT_S = 30.0

#: A hash subject is a hex digest of one of these lengths.
_HASH_HEX = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{40}$"
                       r"|^[0-9a-fA-F]{64}$|^[0-9a-fA-F]{128}$")

#: Extensions that mark executable content for the sandbox rung.
SCRIPT_EXTS: frozenset[str] = frozenset({
    ".py", ".pyw", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs",
    ".bat", ".cmd", ".ps1", ".pl", ".rb", ".php",
})

#: Extensions the structure engine treats as executable for the
#: double-extension lie (the real type is the LAST extension).
EXECUTABLE_EXTS: frozenset[str] = frozenset({
    ".exe", ".bat", ".cmd", ".scr", ".msi", ".com", ".apk", ".dmg",
    ".ps1", ".vbs", ".js", ".jar",
})

#: stderr markers of a network attempt while detonated offline.
_NETWORK_ATTEMPT_MARKERS: tuple[str, ...] = (
    "gaierror",
    "Name or service not known",
    "Temporary failure in name resolution",
    "Network is unreachable",
    "Connection refused",
    "Connection timed out",
    "getaddrinfo failed",
)

#: Known-bad hashes, derived at runtime from the AV organ's
#: EICAR string — no magic hash from memory. The set is the
#: EICAR test hash only, and the hash-reputation engine says so.
_KNOWN_BAD: dict[str, str] = {
    hashlib.sha256(EICAR_STRING.encode("ascii")).hexdigest():
        EICAR_THREAT,
}

STORE_FILENAME = "verdicts.json"


def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


# ── shapes ────────────────────────────────────────────────────

@dataclass
class ScanSubject:
    """What to verdict: a file path, a URL, or a bare hash."""

    kind: str                       # file | url | hash
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in ("file", "url", "hash"):
            raise ValueError(
                f"kind must be file|url|hash, got {self.kind!r}")
        if not (self.ref or "").strip():
            raise ValueError("ref is required")
        if self.kind == "hash" and not _HASH_HEX.match(
                self.ref.strip()):
            raise ValueError(
                f"{self.ref!r} is not a hex hash "
                "(md5/sha1/sha256/sha512 length)")

    @classmethod
    def from_string(cls, value: str, *,
                    kind: str | None = None) -> ScanSubject:
        """Guess the kind from the string: an existing path is a
        file, an http(s) URL is a URL, a bare hex digest is a
        hash.  An explicit ``kind`` wins; anything else is a file
        path (a missing file reports unknown, never a guess)."""
        value = (value or "").strip()
        if not value:
            raise ValueError("subject is required")
        if kind:
            return cls(kind=kind.strip().lower(), ref=value)
        if value.startswith(("http://", "https://")):
            return cls(kind="url", ref=value)
        if _HASH_HEX.match(value):
            return cls(kind="hash", ref=value.lower())
        return cls(kind="file", ref=value)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref}


@dataclass
class EngineResult:
    """One engine's answer: name, verdict, and the reason."""

    name: str
    verdict: str                    # clean | flagged | unknown
    detail: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {VERDICTS}, "
                f"got {self.verdict!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verdict": self.verdict,
                "detail": self.detail}


@dataclass
class VerdictReport:
    """The aggregator's answer for one subject."""

    subject: ScanSubject
    sha256: str = ""
    engines: list[EngineResult] = field(default_factory=list)
    aggregate: str = "unknown"
    scanned_at: float = 0.0
    from_cache: bool = False

    @property
    def engine_count(self) -> int:
        return len(ENGINE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject.to_dict(),
            "sha256": self.sha256,
            "engines": [e.to_dict() for e in self.engines],
            "aggregate": self.aggregate,
            "scanned_at": self.scanned_at,
            "from_cache": self.from_cache,
            "engine_count": self.engine_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerdictReport:
        subject = ScanSubject(**data.get("subject") or {})
        engines = [EngineResult(**e)
                   for e in data.get("engines") or []]
        return cls(
            subject=subject,
            sha256=str(data.get("sha256") or ""),
            engines=engines,
            aggregate=str(data.get("aggregate") or "unknown"),
            scanned_at=float(data.get("scanned_at") or 0.0),
            from_cache=bool(data.get("from_cache") or False),
        )


def _aggregate(engines: list[EngineResult]) -> str:
    if any(e.verdict == "flagged" for e in engines):
        return "flagged"
    if any(e.verdict == "unknown" for e in engines):
        return "unknown"
    return "clean"


# ── durable store (hash-first cache) ──────────────────────────

class VerdictStore:
    """Recorded verdicts by sha256: ``<home>/data/verdicts.json``.

    Atomic writes (temp file + replace). A corrupt store is
    quarantined aside and the store starts empty — recorded
    verdicts are derived data, rebuilt by re-scanning, never
    silently merged with a torn file."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.path = _default_home(home) / "data" / STORE_FILENAME
        self.quarantined = False
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("reports"), dict):
                raise ValueError("verdict store shape is wrong")
            self._data = raw["reports"]
        except (OSError, ValueError):
            stamp = int(time.time())
            try:
                self.path.replace(
                    self.path.with_suffix(f".corrupt-{stamp}"))
            except OSError:
                _log.debug("verdict store quarantine failed",
                           exc_info=True)
            self._data = {}
            self.quarantined = True

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"reports": self._data},
                                      indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            _log.debug("verdict store save failed", exc_info=True)

    def get(self, sha256: str) -> VerdictReport | None:
        row = self._data.get(sha256 or "")
        if not isinstance(row, dict):
            return None
        try:
            report = VerdictReport.from_dict(row)
        except (TypeError, ValueError):
            return None
        report.from_cache = True
        return report

    def put(self, report: VerdictReport) -> None:
        if not report.sha256:
            return
        self._data[report.sha256] = report.to_dict()
        self._save()


# ── content resolution ────────────────────────────────────────

@dataclass
class _ScanInput:
    """Everything the engines measure, resolved once."""

    sha256: str = ""
    data: bytes | None = None       # None when unread / too big
    path: Path | None = None        # local file to scan, if any
    filename: str = ""
    size: int = 0
    too_big: bool = False
    fetch_error: str = ""           # why content is absent, if any
    temp_path: Path | None = None   # fetched bytes, removed by caller


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _default_fetcher(url: str, *, timeout_s: float,
                     max_bytes: int) -> bytes:
    """Fetch a URL's bytes for local scanning. The bytes come IN
    and are scanned here — nothing is uploaded anywhere."""
    client = HttpClient()
    resp = client.get(url, timeout=timeout_s)
    if not resp.ok:
        raise ValueError(
            f"fetch failed: HTTP {resp.status} for {url}")
    if len(resp.body) > max_bytes:
        raise ValueError(
            f"fetched {len(resp.body)} bytes, over the "
            f"{max_bytes}-byte cap")
    return resp.body


def _resolve(subject: ScanSubject, *, fetcher: Any | None,
             max_bytes: int, timeout_s: float) -> _ScanInput:
    """Resolve a subject to measurable content. Never raises for
    a resolution outcome — an absent file or a failed fetch is
    data (``fetch_error``), and the engines report unknown."""
    if subject.kind == "hash":
        return _ScanInput(sha256=subject.ref.lower(),
                          filename=subject.ref.lower())
    if subject.kind == "url":
        fetch = fetcher or _default_fetcher
        try:
            data = fetch(subject.ref, timeout_s=timeout_s,
                         max_bytes=max_bytes)
        except Exception as exc:  # noqa: BLE001 - a failed fetch
            return _ScanInput(                  # is data
                filename=url_filename(subject.ref),
                fetch_error=f"could not fetch {subject.ref}: {exc}")
        if not isinstance(data, (bytes, bytearray)):
            return _ScanInput(
                filename=url_filename(subject.ref),
                fetch_error="fetcher did not return bytes")
        data = bytes(data)
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    prefix="verdict-url-", delete=False) as fh:
                fh.write(data)
            tmp = Path(fh.name)
        except OSError as exc:
            return _ScanInput(
                sha256=_sha256_of_bytes(data), data=data,
                filename=url_filename(subject.ref),
                size=len(data),
                fetch_error=f"could not stage fetched bytes: {exc}")
        return _ScanInput(
            sha256=_sha256_of_bytes(data), data=data, path=tmp,
            filename=url_filename(subject.ref), size=len(data),
            temp_path=tmp)
    path = Path(subject.ref)
    filename = path.name
    if not path.exists():
        return _ScanInput(path=path, filename=filename,
                          fetch_error=f"file not found: {path}")
    if path.is_dir():
        return _ScanInput(path=path, filename=filename,
                          fetch_error=f"is a directory: {path} "
                          "— verdict scans one file, use av_scan "
                          "for folders")
    size = 0
    with suppress(OSError):
        size = path.stat().st_size
    sha = _sha256_of_file(path)
    if size > max_bytes:
        return _ScanInput(sha256=sha, path=path, filename=filename,
                          size=size, too_big=True)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _ScanInput(sha256=sha, path=path, filename=filename,
                          size=size,
                          fetch_error=f"could not read {path}: "
                          f"{exc}")
    return _ScanInput(sha256=sha, data=data, path=path,
                      filename=filename, size=size)


def _cleanup(inp: _ScanInput) -> None:
    if inp.temp_path is None:
        return
    try:
        inp.temp_path.unlink()
    except OSError:
        _log.debug("verdict temp cleanup failed", exc_info=True)


# ── engines ───────────────────────────────────────────────────

#: A sandbox detonation result: outcome completed | timeout |
#: error, plus what the run observed. The tool layer builds this
#: from ``tools.shell.run_sandboxed``; the organ only reads it.
SandboxResult = dict[str, Any]

#: Injected detonation: bytes + filename -> SandboxResult.
SandboxFn = Callable[[bytes, str], SandboxResult]

#: Injected engine override (tests): name -> fn(input) result.
EngineFns = dict[str, Callable[[_ScanInput], EngineResult]]


def _engine_hash_reputation(inp: _ScanInput) -> EngineResult:
    if not inp.sha256:
        return EngineResult(
            name="hash-reputation", verdict="unknown",
            detail="no hash available — nothing to look up")
    threat = _KNOWN_BAD.get(inp.sha256.lower())
    if threat:
        return EngineResult(
            name="hash-reputation", verdict="flagged",
            detail=f"hash matches known-bad {threat}")
    return EngineResult(
        name="hash-reputation", verdict="clean",
        detail="hash not in the known-bad set (EICAR test hash "
               "only — the reputation set is small and this "
               "report says so)")


def _engine_local_av(inp: _ScanInput) -> EngineResult:
    if inp.path is None:
        return EngineResult(
            name="local-av", verdict="unknown",
            detail="no local file to scan (hash-only subject)")
    verdict = scan_file(inp.path)
    if verdict.verdict == "infected":
        return EngineResult(
            name="local-av", verdict="flagged",
            detail=f"{verdict.engine or 'antivirus'} found "
                   f"{verdict.threat or 'malware'}: "
                   f"{verdict.detail}")
    if verdict.verdict == "clean":
        return EngineResult(
            name="local-av", verdict="clean",
            detail=f"{verdict.engine or 'antivirus'}: "
                   f"{verdict.detail}")
    return EngineResult(
        name="local-av", verdict="unknown",
        detail=verdict.detail or "no determination from the "
               "local antivirus engine")


def _engine_pattern_battery(inp: _ScanInput) -> EngineResult:
    if inp.data is None:
        reason = inp.fetch_error or (
            "file too large to load for content engines"
            if inp.too_big else "no content available")
        return EngineResult(
            name="pattern-battery", verdict="unknown",
            detail=reason)
    text = inp.data[:TEXT_SCAN_CAP].decode("utf-8", errors="replace")
    hits = scan_text_for_secrets(text) + entropy_hits(text)
    seen: set[str] = set()
    named: list[str] = []
    for hit in hits:
        key = f"{hit.pattern}:{hit.redacted}"
        if key in seen:
            continue
        seen.add(key)
        named.append(f"{hit.pattern} ({hit.redacted})")
    if not named:
        return EngineResult(
            name="pattern-battery", verdict="clean",
            detail="no secret patterns matched in the scanned "
                   "text")
    shown = ", ".join(named[:5])
    if len(named) > 5:
        shown += f", +{len(named) - 5} more"
    return EngineResult(
        name="pattern-battery", verdict="flagged",
        detail=f"secret patterns matched: {shown} — values are "
               "redacted, never shown in full")


def _is_script(filename: str, data: bytes | None) -> bool:
    ext = Path(filename or "").suffix.lower()
    if ext in SCRIPT_EXTS:
        return True
    if data:
        first = data.split(b"\n", 1)[0][:64]
        if first.startswith(b"#!"):
            return True
    return False


def _engine_structure(inp: _ScanInput) -> EngineResult:
    if inp.data is None:
        reason = inp.fetch_error or (
            "file too large to load for content engines"
            if inp.too_big else "no content available")
        return EngineResult(
            name="structure", verdict="unknown", detail=reason)
    name = inp.filename or ""
    suffixes = Path(name).suffixes
    if len(suffixes) >= 2 \
            and suffixes[-1].lower() in EXECUTABLE_EXTS:
        return EngineResult(
            name="structure", verdict="flagged",
            detail=f"double extension {''.join(suffixes)} — the "
                   f"real type is {suffixes[-1]}, the earlier "
                   "extension is a disguise")
    magic = identify_magic(inp.data[:MAGIC_SNIFF_BYTES])
    ext = Path(name).suffix.lower().lstrip(".")
    expected = extension_for_magic(magic).lstrip(".") if magic else ""
    if magic and expected and ext and expected != ext:
        return EngineResult(
            name="structure", verdict="flagged",
            detail=f"declared .{ext} but content is {magic} "
                   f"(.{expected}) — a structural lie about what "
                   "the file is")
    if magic:
        claimed = f"declared .{ext}" if ext else "no extension claimed"
        return EngineResult(
            name="structure", verdict="clean",
            detail=f"content is {magic}; {claimed} — structure "
                   "matches")
    text = inp.data[:65536].decode("utf-8", errors="replace")
    if looks_like_text(text):
        return EngineResult(
            name="structure", verdict="clean",
            detail="plain text — no binary structure to verify")
    return EngineResult(
        name="structure", verdict="unknown",
        detail="unrecognised binary format — cannot verify "
               "structure")


def _engine_sandbox(inp: _ScanInput,
                    sandbox_fn: SandboxFn | None) -> EngineResult:
    if inp.data is None:
        reason = inp.fetch_error or (
            "file too large to load for content engines"
            if inp.too_big else "no content available")
        return EngineResult(
            name="sandbox", verdict="unknown", detail=reason)
    if not _is_script(inp.filename, inp.data):
        return EngineResult(
            name="sandbox", verdict="clean",
            detail="not executable content — nothing to detonate")
    if sandbox_fn is None:
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail="sandbox detonation not wired on this call "
                   "path — the profile decides")
    result = sandbox_fn(inp.data, inp.filename)
    if not isinstance(result, dict):
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail="sandbox result unreadable — no determination")
    outcome = str(result.get("outcome") or "")
    detail = str(result.get("detail") or "")
    if outcome == "timeout":
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail="detonation timed out — no determination")
    if outcome == "error":
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail="detonation failed: "
                   f"{detail or 'no reason given'} — no determination")
    if outcome != "completed":
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail=f"detonation outcome {outcome!r} is not one "
                   "this engine understands — no determination")
    if result.get("network_attempt"):
        return EngineResult(
            name="sandbox", verdict="unknown",
            detail="the script tried to reach the network while "
                   "detonated offline — observation incomplete, "
                   "no determination")
    code = result.get("returncode")
    return EngineResult(
        name="sandbox", verdict="clean",
        detail=f"detonated in the sandbox: exit {code}, no "
               "hostile behaviour observed"
               + (f" — {detail}" if detail else ""))


def _run_engine(name: str, fn: Callable[[], EngineResult]
                ) -> EngineResult:
    """One engine, never raising: an engine that errors reports
    unknown, never a silent pass."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - an engine error
        return EngineResult(          # is data, not a crash
            name=name, verdict="unknown",
            detail=f"engine error: {exc}")


# ── the aggregator ────────────────────────────────────────────

def verdict_for(subject: ScanSubject | str, *,
                home: str | Path | None = None,
                store: VerdictStore | None = None,
                sandbox_fn: SandboxFn | None = None,
                fetcher: Any | None = None,
                engine_fns: EngineFns | None = None,
                max_bytes: int = MAX_CONTENT_BYTES,
                timeout_s: float = URL_TIMEOUT_S) -> VerdictReport:
    """Run the engines over one subject and return the report.

    Hash-first: a sha256 with a recorded verdict returns it
    without re-scanning content.  ``engine_fns`` replaces
    individual engines by name (tests); ``fetcher`` replaces
    the URL fetch (tests); ``sandbox_fn`` wires detonation.
    Never raises for a verdict outcome — the worst case is a
    report of unknowns with the reason in the details."""
    if isinstance(subject, str):
        subject = ScanSubject.from_string(subject)
    now = time.time()
    inp = _resolve(subject, fetcher=fetcher, max_bytes=max_bytes,
                   timeout_s=timeout_s)
    try:
        own_store = store if store is not None else VerdictStore(home)
        if inp.sha256:
            cached = own_store.get(inp.sha256)
            if cached is not None:
                return cached
        overrides = engine_fns or {}

        def run(name: str, fn: Callable[[], EngineResult]
                ) -> EngineResult:
            if name in overrides:
                return _run_engine(
                    name, lambda: overrides[name](inp))
            return _run_engine(name, fn)

        engines = [
            run("hash-reputation",
                lambda: _engine_hash_reputation(inp)),
            run("local-av", lambda: _engine_local_av(inp)),
            run("pattern-battery",
                lambda: _engine_pattern_battery(inp)),
            run("structure", lambda: _engine_structure(inp)),
            run("sandbox",
                lambda: _engine_sandbox(inp, sandbox_fn)),
        ]
        report = VerdictReport(
            subject=subject, sha256=inp.sha256, engines=engines,
            aggregate=_aggregate(engines), scanned_at=now)
        if inp.sha256:
            own_store.put(report)
        return report
    finally:
        _cleanup(inp)


# ── reply ─────────────────────────────────────────────────────

def format_verdict_reply(report: VerdictReport) -> str:
    """The chat reply: aggregate first, per-engine lines, then
    the honest count + locality notes. Unknowns are named as
    unknowns, never smoothed over."""
    lines: list[str] = []
    mark = {"clean": "clean", "flagged": "FLAGGED",
            "unknown": "unknown"}[report.aggregate]
    ref = report.subject.ref
    lines.append(f"Verdict on {ref}: {mark}")
    if report.sha256:
        lines.append(f"sha256: {report.sha256[:16]}…")
    for engine in report.engines:
        lines.append(
            f"  {engine.name}: {engine.verdict} — "
            f"{engine.detail}")
    if report.from_cache:
        lines.append("  (from a recorded verdict for this hash — "
                     "content was not re-scanned)")
    lines.append(
        f"Engines: {report.engine_count} local engines, not "
        "VirusTotal's 70+ commercial ones — Devon runs five "
        "local rungs and says so.")
    lines.append(
        "Nothing left this box: a confidential file can be "
        "scanned at all, which VirusTotal's sharing model "
        "forbids.")
    if report.aggregate == "flagged":
        lines.append(
            "Use av_scan to quarantine it, or delete it — do not "
            "open it.")
    return "\n".join(lines)
