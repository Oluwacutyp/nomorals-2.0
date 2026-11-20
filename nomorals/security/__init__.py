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
- :mod:`nomorals.security.bughunter` — authorized bug-bounty
  hunting (Claude-BugHunter port): scope rules, skill map, the
  7-Question Gate, coverage-closure ledger, engagement engine over
  Devon's HttpClient / brain / audit book / evidence chain.
- :mod:`nomorals.security.session_forge` — forge authenticated
  sessions with the auth plugins (security/plugins/): login, persist
  the cookie jar + headers (0600), verify before reuse, browser
  escalation when the HTTP flow fails.
- :mod:`nomorals.security.payloads` — client-side payload organ:
  decode bombs (PNG/GIF/SVG/PDF), keylogger page, cookie stealer.
- :mod:`nomorals.security.plugins` — the auth plugin registry
  (12 site plugins + the WebPlugin contract).
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
- :mod:`nomorals.security.scope` — scope files (include/exclude/
  allow_all) + the Governor: global budget, per-host spend,
  snapshot; default-deny, no probe fires without a scope
  (owner's probe batch, 2026-11-19).
- :mod:`nomorals.security.diff` — response fingerprinting +
  differential comparison: the ONE fingerprint primitive for the
  probe organs, including the login verdict (owner's probe batch).
- :mod:`nomorals.security.params` — parameter mining from a case
  corpus, then confirmation over HttpClient (owner's probe batch).
- :mod:`nomorals.security.inject` — injection probing over
  confirmed params, with a differential-oracle judge (owner's
  probe batch).
- :mod:`nomorals.security.jwt` — JWT attacks: alg-none, weak HMAC,
  kid traversal, jku redirect (owner's probe batch).
- :mod:`nomorals.security.graphql` — GraphQL introspection, then
  alias amplification (owner's probe batch).
- :mod:`nomorals.security.race` — N identical requests in one wave
  (owner's probe batch).
- :mod:`nomorals.security.cors_csrf` — CORS probe + CSRF PoC
  generation (owner's probe batch).
- :mod:`nomorals.security.planner` — hypothesis ledger (JSONL on
  disk) + next-action selection (owner's probe batch).
- :mod:`nomorals.security.leaks` — leak / credential-exposure
  sources: keyless Cavalier + ProxyNova, keyed HIBP / DeHashed
  (owner's probe batch).
- :mod:`nomorals.security.dossier` — one subject, every surface,
  one score (owner's probe batch).
- :mod:`nomorals.security.bans` — report / state tracking / appeal
  drafts for banned handles (owner's probe batch).
- :mod:`nomorals.security.report_amp` — capture one real report
  request in a browser, then replay-amplify it (owner's probe
  batch). One replay path: amplify() delegates to replay.
- :mod:`nomorals.security.replay` — request mutation and replay:
  coherent device minting, declarative mutation, drift bisection
  (owner's probe batch, 2026-11-20). The shared replay engine.
- :mod:`nomorals.security.case` — one case home: scope file,
  planner ledger, findings and the attacker checkpoint in one
  folder under Devon's home; the single Governor construction
  point; the L4 proxy-pool seam.
- :mod:`nomorals.security.attack_chain` — planner drives the
  attack: a confirmed login-form hypothesis becomes an attacker
  run, its outcome resolves the hypothesis, found credentials land
  in the case dossier.
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
from . import bughunter as _bughunter  # noqa: E402 - section import, this file's convention

BugHunterEngine = _bughunter.Engine
BugHunterScope = _bughunter.Scope
bughunter_classify_url = _bughunter.classify_url
bughunter_gate_finding = _bughunter.gate_finding
bughunter_render_report = _bughunter.render_report
bughunter_run_engagement = _bughunter.run_engagement
bughunter_run_parity = _bughunter.run_parity
bughunter_verify_attribution = _bughunter.verify_attribution
bughunter_worklist_from_audit = _bughunter.worklist_from_audit

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
    "BugHunterEngine",
    "BugHunterScope",
    "bughunter_classify_url",
    "bughunter_gate_finding",
    "bughunter_render_report",
    "bughunter_run_engagement",
    "bughunter_run_parity",
    "bughunter_verify_attribution",
    "bughunter_worklist_from_audit",
]

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

# The owner's security probe batch (2026-11-19/20): scope + Governor,
# fingerprinting, param mining + confirmation, injection probing, JWT,
# GraphQL, race waves, CORS/CSRF, hypothesis planner, leak harvest,
# dossiers, ban reports, report capture + amplification, request
# mutation + replay, one case home, and the planner→attacker→dossier
# chain. See security/scope.py, diff.py, params.py, inject.py,
# jwt.py, graphql.py, race.py, cors_csrf.py, planner.py, leaks.py,
# dossier.py, bans.py, report_amp.py, replay.py, case.py,
# attack_chain.py. Aliases where a bare name would collide: `amplify`
# exists in both graphql and report_amp; inject's Finding would
# shadow evidence.Finding; replay's `replay` function shares its
# module's name.
from . import replay as _replay_mod  # noqa: E402 - section import, this file's convention
from .bans import (  # noqa: E402
    PLATFORMS,
    STATES,
    appeal_text,
    mass_appeal,
    report_all,
    report_one,
    set_state,
)
from .case import (  # noqa: E402
    DEFAULT_SCOPE_NAME,
    attacker_paths,
    case_dir,
    default_scope_path,
    governor_for,
    home_of,
    init_case,
    pool_from_factory,
    resolve_scope,
    scope_path,
    set_pool_factory,
    slugify_case,
)
from .cors_csrf import (  # noqa: E402
    ORIGINS,
    CorsResult,
    cors_probe,
    csrf_poc,
)
from .diff import (  # noqa: E402
    Delta,
    Fingerprint,
    compare,
    fingerprint,
    login_delta,
    similarity,
)
from .dossier import (  # noqa: E402
    Dossier,
    build_dossier,
    render,
    save_dossier,
)
from .graphql import (  # noqa: E402
    INTROSPECTION,
    graphql_probe,
)
from .graphql import amplify as graphql_amplify  # noqa: E402
from .inject import (  # noqa: E402
    PAYLOADS,
    probe_param,
)
from .inject import Finding as InjectFinding  # noqa: E402
from .jwt import (  # noqa: E402
    WEAK_SECRETS,
    alg_none,
    attack,
    decode,
    encode,
    jku_redirect,
    kid_traversal,
    weak_hmac,
)
from .leaks import (  # noqa: E402
    KEYS,
    Exposure,
    harvest_exposure,
)
from .params import (  # noqa: E402
    PARAM_SEEDS,
    Param,
    confirm,
    mine,
)
from .planner import (  # noqa: E402
    CLASS_COST,
    CLASS_WEIGHT,
    Hypothesis,
    Ledger,
)
from .race import (  # noqa: E402
    RaceResult,
    wave,
)
from .replay import (  # noqa: E402
    FIELD_KINDS,
    Device,
    Mutation,
    diagnose,
    mint_device,
    mutate,
    set_signer_factory,
)
from .report_amp import (  # noqa: E402
    TEMPLATES,
    Template,
    capture,
    load_template,
    set_capture_driver,
)
from .report_amp import amplify as report_amplify  # noqa: E402
from .scope import (  # noqa: E402
    BudgetExhausted,
    Governor,
    OutOfScope,
    Scope,
    load_scope,
)

replay_request = _replay_mod.replay

from .attack_chain import (  # noqa: E402
    chain_for_case,
    run_chain,
)

__all__ += [
    "Scope",
    "Governor",
    "BudgetExhausted",
    "OutOfScope",
    "load_scope",
    "Fingerprint",
    "Delta",
    "fingerprint",
    "compare",
    "similarity",
    "login_delta",
    "PARAM_SEEDS",
    "Param",
    "mine",
    "confirm",
    "PAYLOADS",
    "InjectFinding",
    "probe_param",
    "WEAK_SECRETS",
    "decode",
    "encode",
    "alg_none",
    "weak_hmac",
    "kid_traversal",
    "jku_redirect",
    "attack",
    "INTROSPECTION",
    "graphql_probe",
    "graphql_amplify",
    "RaceResult",
    "wave",
    "ORIGINS",
    "CorsResult",
    "cors_probe",
    "csrf_poc",
    "Hypothesis",
    "Ledger",
    "CLASS_COST",
    "CLASS_WEIGHT",
    "KEYS",
    "Exposure",
    "harvest_exposure",
    "Dossier",
    "build_dossier",
    "render",
    "save_dossier",
    "PLATFORMS",
    "STATES",
    "report_one",
    "report_all",
    "set_state",
    "appeal_text",
    "mass_appeal",
    "TEMPLATES",
    "Template",
    "capture",
    "report_amplify",
    "load_template",
    "set_capture_driver",
    "FIELD_KINDS",
    "Mutation",
    "Device",
    "mint_device",
    "mutate",
    "replay_request",
    "diagnose",
    "set_signer_factory",
    "DEFAULT_SCOPE_NAME",
    "case_dir",
    "slugify_case",
    "scope_path",
    "default_scope_path",
    "resolve_scope",
    "governor_for",
    "attacker_paths",
    "init_case",
    "home_of",
    "set_pool_factory",
    "pool_from_factory",
    "run_chain",
    "chain_for_case",
]
