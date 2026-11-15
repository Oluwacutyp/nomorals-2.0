"""Security-posture spine tools — DNS leak check + EXIF stripping +
malware scanning.

- dns_leak_check: "is my DNS leaking?" — nonce-probe technique.
- strip_exif: remove GPS/camera/timestamps from an image before sending.
- exif_report: what metadata an image carries (awareness, not removal).
- av_scan: scan a file or folder for malware (Windows Defender, ClamAV,
  or the built-in EICAR-only signature check) and quarantine hits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _quarantine_dir(context: Any) -> Path | None:
    """Where infected files go. Env override first, then the Devon
    home's ``quarantine/`` dir, then the tool workspace's. None means
    report-only: hits are reported, never moved."""
    override = (os.environ.get("NM_AV_QUARANTINE") or "").strip()
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
