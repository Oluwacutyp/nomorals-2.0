"""Security module — leak detection + metadata stripping.

- :mod:`nomorals.security.dnsleak` — DNS leak check (bash.ws nonce technique).
- :mod:`nomorals.security.netleak` — WebRTC-style (STUN) + IPv6 leak checks.
- :mod:`nomorals.security.exif` — lossless image metadata stripping.
"""

from . import dnsleak as _dnsleak_mod
from . import netleak as _netleak_mod
from .dnsleak import (
    DnsLeakReport,
    check_dns_leak,
    configured_resolvers,
    detect_smhnr_risk,
)
from .exif import (
    clean_copy,
    exif_summary,
    format_summary,
    scan_metadata,
    secure_delete,
    strip_exif,
    strip_exif_to_bytes,
    strip_metadata_bytes,
    verify_clean,
)
from .netleak import (
    NetLeakReport,
    check_net_leak,
    ipv6_egress,
    local_interface_ips,
    stun_public_ip,
)

dnsleak_format_report = _dnsleak_mod.format_report
netleak_format_report = _netleak_mod.format_report

__all__ = [
    "dnsleak_format_report",
    "netleak_format_report",
    "DnsLeakReport",
    "check_dns_leak",
    "configured_resolvers",
    "detect_smhnr_risk",
    "NetLeakReport",
    "check_net_leak",
    "ipv6_egress",
    "local_interface_ips",
    "stun_public_ip",
    "clean_copy",
    "exif_summary",
    "format_summary",
    "scan_metadata",
    "secure_delete",
    "strip_exif",
    "strip_exif_to_bytes",
    "strip_metadata_bytes",
    "verify_clean",
]
