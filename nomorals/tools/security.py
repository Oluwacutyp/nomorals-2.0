"""Security-posture spine tools — DNS leak check + EXIF stripping.

- dns_leak_check: "is my DNS leaking?" — nonce-probe technique.
- strip_exif: remove GPS/camera/timestamps from an image before sending.
- exif_report: what metadata an image carries (awareness, not removal).
"""

from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    from ..core.policy import Capability

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
