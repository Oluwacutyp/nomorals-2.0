"""Security-posture spine tools — DNS leak check + EXIF stripping +
malware scanning.

- dns_leak_check: "is my DNS leaking?" — nonce-probe technique.
- net_leak_check: "is my real IP leaking?" — STUN srflx + IPv6 egress.
- strip_exif: remove GPS/camera/timestamps from an image before sending.
- exif_report: what metadata an image carries (awareness, not removal).
- av_scan: scan a file or folder for malware (Windows Defender, ClamAV,
  or the built-in EICAR-only signature check) and quarantine hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _quarantine_dir(context: Any) -> Path | None:
    """Where infected files go — the organ's answer, shared with the
    intake scan and the sweep (one implementation, not three)."""
    from ..core.antivirus import quarantine_dir_for

    return quarantine_dir_for(context)


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
