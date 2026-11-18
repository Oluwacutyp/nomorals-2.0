"""Security-posture spine tools — DNS leak check + EXIF stripping +
malware scanning + multi-engine verdicts.

- dns_leak_check: "is my DNS leaking?" — nonce-probe technique.
- net_leak_check: "is my real IP leaking?" — STUN srflx + IPv6 egress.
- strip_exif: remove GPS/camera/timestamps from an image before sending.
- exif_report: what metadata an image carries (awareness, not removal).
- av_scan: scan a file or folder for malware (Windows Defender, ClamAV,
  or the built-in EICAR-only signature check) and quarantine hits.
- file_verdict: multi-engine verdict on a file, URL or hash — the
  VirusTotal shape, Devon-local (five local engines, per-engine
  lines, aggregate; unknown means no determination, never clean).
- tor_status / onion_validate / onion_fetch: Tor presence, v3 .onion
  validation, fetch through Tor (SOCKS5h) — the owner's onion organ,
  wired here like dnsleak/netleak; a missing Tor is named, never faked.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def _quarantine_dir(context: Any) -> Path | None:
    """Where infected files go — the organ's answer, shared with the
    intake scan and the sweep (one implementation, not three)."""
    from ..core.antivirus import quarantine_dir_for

    return quarantine_dir_for(context)


def _home_of(context: Any) -> str | None:
    """Devon's home from the tool context (same idiom as the burn-note
    tool). None means the verdict organ's own default."""
    settings = getattr(context, "settings", None)
    home = (getattr(settings, "home", None)
            or getattr(settings, "home_path", None))
    return str(home) if home else None


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


def _sandbox_detonate(data: bytes, filename: str) -> dict[str, Any]:
    """Detonate script bytes in the sandbox and report what the run
    observed. Network is off; the script runs in a throwaway dir
    that is removed afterwards. Returns the organ's SandboxResult
    shape: outcome completed | timeout | error, returncode,
    network_attempt, detail."""
    from ..tools.shell import run_sandboxed

    ext = Path(filename or "").suffix.lower()
    if ext in (".py", ".pyw"):
        run_name, cmd = "run.py", sys.executable
    elif ext in (".sh", ".bash", ".zsh"):
        run_name, cmd = "run.sh", "/bin/sh"
    elif ext in (".js", ".mjs", ".cjs"):
        node = shutil.which("node")
        if not node:
            return {"outcome": "error",
                    "detail": "no node interpreter on this box"}
        run_name, cmd = "run.js", node
    else:
        return {"outcome": "error",
                "detail": "no interpreter wired for "
                          f"{ext or 'this script type'}"}
    argv = f"{shlex.quote(cmd)} {shlex.quote(run_name)}"
    workdir = Path(tempfile.mkdtemp(prefix="verdict-detonate-"))
    try:
        (workdir / run_name).write_bytes(data)
        try:
            out = run_sandboxed(argv, cwd=workdir, timeout=30.0,
                                network=False)
        except Exception as exc:  # noqa: BLE001 - a sandbox
            return {"outcome": "error",      # failure is data
                    "detail": f"sandbox failed to run: {exc}"}
        stderr = str(out.get("stderr") or "")
        if out.get("timed_out"):
            return {"outcome": "timeout",
                    "detail": "detonation hit the 30s window"}
        network_attempt = any(m in stderr
                              for m in _NETWORK_ATTEMPT_MARKERS)
        detail = ""
        if stderr.strip():
            detail = stderr.strip().splitlines()[-1][:200]
        return {
            "outcome": "completed",
            "returncode": out.get("exit_code"),
            "network_attempt": network_attempt,
            "detail": detail,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "dns_leak_check",
        description=(
            "Check whether DNS queries leak outside the VPN/tunnel. "
            "('is my DNS leaking?', 'dns leak test'). Nonce-probe "
            "technique via bash.ws with a local fallback. No API key."
        ),
        capability=Capability.NET_OUT,
    )
    def dns_leak_check(*, expected_asn: str = "") -> dict[str, Any]:
        from ..security.dnsleak import check_dns_leak
        try:
            rep = check_dns_leak(expected_asn=expected_asn)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **rep.to_dict()}

    @registry.register(
        "net_leak_check",
        description=(
            "Check whether the real IP leaks via WebRTC/STUN or IPv6 "
            "egress. ('is my IP leaking?', 'webrtc leak test', 'am I "
            "leaking my real IP'). STUN Binding Request + IPv6 egress "
            "probe, pure stdlib. No API key. Result carries verdict "
            "'leak' / 'clean' / 'inconclusive' — inconclusive means the "
            "check could not measure the path, never clean."
        ),
        capability=Capability.NET_OUT,
    )
    def net_leak_check(*, expected_ip: str = "") -> dict[str, Any]:
        from ..security.netleak import check_net_leak
        try:
            rep = check_net_leak(expected_ip=expected_ip)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **rep.to_dict()}

    @registry.register(
        "strip_exif",
        description=(
            "Remove EXIF metadata (GPS, camera model, timestamps) from an "
            "image file. ('strip metadata from this photo', 'clean exif'). "
            "Keeps a .bak. Use before sending sensitive photos."
        ),
        capability=Capability.FS_WRITE,
    )
    def strip_exif(path: str) -> dict[str, Any]:
        from ..security.exif import strip_exif as _strip
        try:
            return {"ok": True, **_strip(path)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "exif_report",
        description=(
            "Report what EXIF metadata an image carries (GPS? camera? "
            "timestamps?). Awareness only — use strip_exif to remove."
        ),
        capability=Capability.FS_READ,
    )
    def exif_report(path: str) -> dict[str, Any]:
        from ..security.exif import exif_summary
        try:
            return {"ok": True, **exif_summary(path)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "av_scan",
        description=(
            "Scan a file or folder for malware and quarantine hits. "
            "('scan this file for viruses', 'is this download safe', "
            "'check this folder for malware'). Uses the best engine on "
            "the box — Windows Defender, ClamAV, or the built-in "
            "signature check (EICAR test file only) — and names the "
            "engine in the result. A hit is moved into the quarantine "
            "dir with a record, unless quarantine=false. A result of "
            "unknown means no determination, never clean."
        ),
        capability=Capability.FS_WRITE,
    )
    def av_scan(path: str, *, engine: str = "",
                quarantine: bool = True) -> dict[str, Any]:
        from ..core.antivirus import scan_path
        if not (path or "").strip():
            return {"ok": False, "error": "path is required"}
        try:
            out = scan_path(
                path, engine=engine or None,
                quarantine_dir=_quarantine_dir(context)
                if quarantine else None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **out.to_dict()}

    @registry.register(
        "file_verdict",
        description=(
            "Multi-engine verdict on a file, URL or hash — 'is this "
            "safe?', 'is this file safe', 'check this link'. Runs "
            "Devon's five local engines (hash reputation, local "
            "antivirus, secret-pattern battery, file-structure "
            "check, sandbox detonation where wired) and reports "
            "per-engine lines plus an aggregate verdict. A verdict "
            "of unknown means no determination, never clean. "
            "Nothing leaves the box: confidential files can be "
            "scanned at all. Read-only — it never quarantines or "
            "deletes; use av_scan to quarantine a flagged file."
        ),
        capability=Capability.FS_READ,
    )
    def file_verdict(subject: str, *, kind: str = "",
                     sandbox: bool = True) -> dict[str, Any]:
        from ..security.verdict import (
            ScanSubject,
            format_verdict_reply,
            verdict_for,
        )
        if not (subject or "").strip():
            return {"ok": False, "error": "subject is required"}
        try:
            sub = ScanSubject.from_string(
                subject, kind=kind.strip() or None)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            report = verdict_for(
                sub, home=_home_of(context),
                sandbox_fn=_sandbox_detonate if sandbox else None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **report.to_dict(),
                "reply": format_verdict_reply(report)}

    # Tor / .onion (owner's security/onion.py): presence, v3 validation,
    # SOCKS5h fetch. Wired here like dnsleak/netleak — the organ lives
    # in security/, the registry entry lives in this tools module.
    from ..security import onion as _onion

    _onion.register(registry)
