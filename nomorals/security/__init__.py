"""Security module — leak detection + metadata stripping.

- :mod:`nomorals.security.dnsleak` — DNS leak check (bash.ws nonce technique).
- :mod:`nomorals.security.netleak` — WebRTC-style (STUN) + IPv6 leak checks.
- :mod:`nomorals.security.exif` — lossless image metadata stripping.
- :mod:`nomorals.security.onion` — Tor / .onion: v3 validation,
  SOCKS5h fetch, circuit rotation (owner's onion.py).
- :mod:`nomorals.security.payload_inspection` — structural
  anticrash inspection of chat payloads + learned signatures.
- :mod:`nomorals.security.checkers` — service adapters for combo
  checking: verdicts from cookies/redirects/status (owner's batch).
- :mod:`nomorals.security.combo_checker` — email:password combos
  against many services at once (owner's batch).
- :mod:`nomorals.security.social_enum` — username / email
  enumeration across social platforms (owner's batch).
- :mod:`nomorals.security.recon` — passive surface discovery: CT,
  DNS over HTTPS, AXFR, takeover leads (owner's batch).
- :mod:`nomorals.security.subbrute` — DNS subdomain brute with
  wildcard suppression + permutation rounds (owner's batch).
- :mod:`nomorals.security.evidence` — hash-chained evidence packs,
  secrets masked at rest (owner's batch).
- :mod:`nomorals.security.disclose` — severity rubric, RFC 9116
  contact discovery, disclosure draft (owner's batch).
- :mod:`nomorals.security.report` — engagement report from a case's
  chain + inventory (owner's batch).
- :mod:`nomorals.security.seed_sources` — keyless seed sources for
  data harvesting: passive DNS, URL history, scan indexes, host
  intel, email reputation (owner's batch).
- :mod:`nomorals.security.cards` — BIN intel, Luhn generation,
  structural and gateway card checks (owner's batch).
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

# Tier 3: security -> the multi-engine verdict organ
# (52-websites #24, VirusTotal shape, Devon-local): one
# artefact in, five local engines, a verdict record with
# per-engine results.  See security/verdict.py.
from .verdict import (  # noqa: E402 - section import, this file's convention
    EngineResult,
    ScanSubject,
    VerdictReport,
    VerdictStore,
    format_verdict_reply,
    verdict_for,
)

__all__ += [
    "EngineResult",
    "ScanSubject",
    "VerdictReport",
    "VerdictStore",
    "format_verdict_reply",
    "verdict_for",
]

# Tor / .onion organ (owner's onion.py): v3 address validation,
# SOCKS5h fetch, circuit rotation. See security/onion.py.
from .onion import (  # noqa: E402 - section import, this file's convention
    TorControl,
    onion_fetch,
    tor_status,
    validate_onion,
)

__all__ += [
    "TorControl",
    "onion_fetch",
    "tor_status",
    "validate_onion",
]

# The owner's security batch (2026-11-18): service adapters + combo
# checking, social enumeration, passive recon, subdomain brute,
# hash-chained evidence, disclosure desk, engagement reports.
# See security/checkers.py, combo_checker.py, social_enum.py,
# recon.py, subbrute.py, evidence.py, disclose.py, report.py.
# The owner's security batch, second wave (2026-11-18): keyless seed
# sources + card tooling. See security/seed_sources.py, cards.py.
# No bare name here collides with a submodule: `harvest`, `wordlist`,
# `bin_check`, `generate` etc. are functions, never module names —
# the recon lesson (never export a bare function whose name IS a
# submodule) does not apply to either module.
from .cards import (  # noqa: E402
    BRANDS,
    Card,
    bin_lookup,
    brand_of,
    check_gateway,
    check_structural,
    generate,
    luhn_complete,
    luhn_ok,
    spam,
)
from .checkers import (  # noqa: E402 - section import, this file's convention
    ADAPTERS,
    EbayAdapter,
    FormAdapter,
    GoogleAdapter,
    NetflixAdapter,
    Probe,
    ServiceAdapter,
    Verdict,
    auth_cookie_names,
    get_adapter,
)
from .combo_checker import (  # noqa: E402
    ComboChecker,
    ComboResult,
    parse_combo,
)
from .disclose import (  # noqa: E402
    Tier,
    draft,
    find_contacts,
    score,
)
from .evidence import (  # noqa: E402
    EvidenceChain,
    Finding,
    mask,
    verify_chain,
)
from .recon import (  # noqa: E402
    axfr,
    crt_sh,
    dns_lookup,
    takeover_check,
)
from .report import (  # noqa: E402
    VERDICT_KIND,
    EngagementReport,
)
from .report import (  # noqa: E402
    build as build_engagement_report,
)
from .seed_sources import (  # noqa: E402
    KEYED,
    SOURCES,
    bin_check,
    harvest,
    wordlist,
)
from .social_enum import (  # noqa: E402
    SITES,
    enum_email,
    enum_username,
)
from .subbrute import (  # noqa: E402
    SEED_WORDS,
    BruteHit,
    SubBrute,
    permutations,
)

__all__ += [
    "ADAPTERS",
    "EbayAdapter",
    "FormAdapter",
    "GoogleAdapter",
    "NetflixAdapter",
    "Probe",
    "ServiceAdapter",
    "Verdict",
    "auth_cookie_names",
    "get_adapter",
    "ComboChecker",
    "ComboResult",
    "parse_combo",
    "Tier",
    "draft",
    "find_contacts",
    "score",
    "EvidenceChain",
    "Finding",
    "mask",
    "verify_chain",
    "axfr",
    "crt_sh",
    "dns_lookup",
    "takeover_check",
    "EngagementReport",
    "VERDICT_KIND",
    "build_engagement_report",
    "SITES",
    "enum_email",
    "enum_username",
    "SEED_WORDS",
    "BruteHit",
    "SubBrute",
    "permutations",
    "BRANDS",
    "Card",
    "bin_lookup",
    "brand_of",
    "check_gateway",
    "check_structural",
    "generate",
    "luhn_complete",
    "luhn_ok",
    "spam",
    "KEYED",
    "SOURCES",
    "bin_check",
    "harvest",
    "wordlist",
]
