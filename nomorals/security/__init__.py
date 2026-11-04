"""Security module — leak detection + metadata stripping.

- :mod:`nomorals.security.dnsleak` — DNS leak check (bash.ws nonce technique).
- :mod:`nomorals.security.netleak` — WebRTC-style (STUN) + IPv6 leak checks.
- :mod:`nomorals.security.exif` — lossless image metadata stripping.
- :mod:`nomorals.security.payload_inspection` — structural
  anticrash inspection of chat payloads + learned signatures.
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

# Tier 3: security -> egress auditing for sandbox L2.
from .egress_audit import EgressAuditor, EgressDecision, EgressRecord

__all__ += ["EgressAuditor", "EgressDecision", "EgressRecord"]

# Tier 3: security -> the audit class (owner, 2026-11-03):
# exposure scanning of web apps + repo code/security review +
# the disclosure desk.  Evidence packs, severity rubric, the
# durable book — see the audit_* module docstrings.
from .audit_evidence import (
    DEFAULT_SAMPLE_CAP, OWN_ASSET_SAMPLE_CAP, build_pack,
    field_sensitivity, mask_record, mask_value, pack_for_storage,
    schema_of,
)
from .audit_patterns import (
    SECRET_PATTERNS, SecretHit, entropy_hits, jwt_payload,
    redact_secret, scan_text_for_secrets, shannon_entropy,
)
from .audit_report import (
    SEVERITY_ORDER, AuditBook, AuditStoreError, default_book,
    discover_contacts_repo, discover_contacts_web,
    draft_disclosure, grade_finding, grade_findings,
    render_report_md, summarize,
)
from .audit_repo import (
    INSECURE_RULES, SEED_ADVISORIES, RepoAuditor, audit_repo,
    compare_versions, parse_lockfiles,
)
from .audit_service import AuditService, run_job
from .audit_web import (
    ExposureScanner, Fetcher, FetchResult, ScanBudgetExceeded,
    scan_url,
)

__all__ += [
    "DEFAULT_SAMPLE_CAP", "OWN_ASSET_SAMPLE_CAP", "build_pack",
    "field_sensitivity", "mask_record", "mask_value",
    "pack_for_storage", "schema_of",
    "SECRET_PATTERNS", "SecretHit", "entropy_hits", "jwt_payload",
    "redact_secret", "scan_text_for_secrets", "shannon_entropy",
    "SEVERITY_ORDER", "AuditBook", "AuditStoreError",
    "default_book", "discover_contacts_repo",
    "discover_contacts_web", "draft_disclosure", "grade_finding",
    "grade_findings", "render_report_md", "summarize",
    "INSECURE_RULES", "SEED_ADVISORIES", "RepoAuditor",
    "audit_repo", "compare_versions", "parse_lockfiles",
    "AuditService", "run_job",
    "ExposureScanner", "Fetcher", "FetchResult",
    "ScanBudgetExceeded", "scan_url",
]
