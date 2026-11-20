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
- combo_check: email:password combos against service adapters
  (eBay/Netflix/Google/Amazon/Oracle/ElevenLabs) — verdicts from
  cookies/redirects/status, never body text (owner's combo checker).
- social_enum: username availability / email registration across
  social platforms, 404-or-marker discipline (owner's social_enum).
- recon_domain: passive surface discovery — CT logs, DNS over HTTPS,
  AXFR attempt, wildcard suppression, takeover leads (owner's recon).
- subbrute: DNS subdomain brute with wildcard suppression and
  permutation rounds (owner's subbrute).
- evidence_add / evidence_verify / evidence_export: hash-chained
  evidence packs, secrets masked at rest (owner's evidence).
- severity_score / disclose_contacts / disclose_draft: deterministic
  severity rubric, RFC 9116 contact discovery, disclosure draft from
  a verified chain (owner's disclose).
- report_build: engagement report from a case's chain + inventory;
  refuses an unverifiable chain (owner's report).
- seed_harvest / seed_wordlist / ip_intel: keyless seed sources —
  passive DNS, URL history, scan indexes, host intel, email
  reputation; harvested URLs into brute words (owner's seed_sources).
- bin_lookup / bin_generate / card_check / card_spam: BIN intel,
  Luhn generation, structural and gateway card checks (owner's cards).
- bughunter_scope_check / bughunter_classify / bughunter_recon /
  bughunter_surface / bughunter_triage / bughunter_report /
  bughunter_engage / bughunter_hackerone: authorized bug-bounty
  hunting (Claude-BugHunter port) — scope rules, skill map,
  7-Question Gate, closure ledger, engagement engine over the
  security organ's own audit/evidence/disclosure rungs.
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


def _playwright_capture_driver(platform: str, url_hint: str = "",
                               timeout: float = 240.0) -> dict[str, Any]:
    """Drive a real headed browser and capture one report request.

    The sync_playwright block the owner's capture() carried, moved
    here to the L4 layer so it can self-provision through the real
    browser rung (browser/provision.ensure_playwright) instead of
    failing on a bare ImportError."""
    import time
    import urllib.parse

    from ..browser.provision import ensure_playwright
    from ..core.errors import ToolError
    from ..security import report_amp

    if not ensure_playwright():
        raise ToolError(
            "capture needs Playwright + Chromium: the browser rung "
            "could not provision them on this machine")
    from playwright.sync_api import sync_playwright
    spec = report_amp.TEMPLATES.get(platform)
    if spec is None:
        raise ToolError(f"unknown platform {platform!r}")
    needle = url_hint or urllib.parse.urlsplit(spec["url"]).path
    hits: list[dict[str, Any]] = []

    def on_request(req: Any) -> None:
        if needle not in req.url:
            return
        hits.append({"url": req.url, "method": req.method,
                     "headers": dict(req.headers),
                     "body": req.post_data or ""})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        ctx.on("request", on_request)
        page = ctx.new_page()
        page.goto(spec["url"], wait_until="domcontentloaded", timeout=45_000)
        print(f"  file one report on {platform} by hand — capturing "
              f"{needle}")
        deadline = time.time() + timeout
        while not hits and time.time() < deadline:
            page.wait_for_timeout(500)
        browser.close()
    if not hits:
        raise ToolError("no matching request captured")
    return hits[-1]


def _playwright_signer_factory(url: str) -> Any:
    """Build a browser signer for a page URL, provisioning the
    browser rung first."""
    from ..browser.provision import ensure_playwright
    from ..core.errors import ToolError
    from ..security import replay as _replay

    if not ensure_playwright():
        raise ToolError(
            "browser signing needs Playwright + Chromium: the browser "
            "rung could not provision them on this machine")
    return _replay._Signer(url)


def _proxy_pool_factory(context: Any) -> Any:
    """Build the rotation pool from the registry context.

    The owner's report_amp/replay imported `tools.proxy_rotate` — a
    module that does not exist (the pool lives in
    `tools.proxy_rotation`); the ImportError was swallowed by a bare
    except, so the rotated egress never engaged and every replay ran
    direct. This is the real pool, by its real name."""
    from .proxy_rotation import pool_from_context

    return pool_from_context(context)


def ensure_secprobe_seams() -> dict[str, bool]:
    """Plug the L4 seams into the security organs (idempotent). The
    wave calls this at registration; the CLI calls it before
    `nm sec capture` / `nm sec replay` so those verbs work outside
    the brain too."""
    from ..security import case as _case
    from ..security import replay as _replay
    from ..security import report_amp as _report_amp

    _report_amp.set_capture_driver(_playwright_capture_driver)
    _replay.set_signer_factory(_playwright_signer_factory)
    _case.set_pool_factory(_proxy_pool_factory)
    return {"capture_driver": True, "signer_factory": True,
            "pool_factory": True}


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

    # The owner's security batch (2026-11-18): combo checking, social
    # enumeration, passive recon, subdomain brute, hash-chained
    # evidence, disclosure desk, engagement reports. Each organ lives
    # in security/ and registers its own tools here, like onion.
    from ..security import combo_checker as _combo_checker
    from ..security import disclose as _disclose
    from ..security import evidence as _evidence
    from ..security import report as _report
    from ..security import social_enum as _social_enum
    from ..security import subbrute as _subbrute

    # NB: import recon's register() by dotted path, not through the
    # package — a bare `recon` name in the package namespace would
    # shadow the submodule (the recon() function is deliberately NOT
    # exported bare from nomorals/security for exactly this reason).
    from ..security.recon import register as _register_recon

    _combo_checker.register(registry)
    _social_enum.register(registry)
    _register_recon(registry)
    _subbrute.register(registry)
    _evidence.register(registry)
    _disclose.register(registry)
    _report.register(registry)

    # The owner's security batch, second wave (2026-11-18): keyless
    # seed sources + card tooling. Each organ lives in security/ and
    # registers its own tools here, like the first wave.
    from ..security import cards as _cards
    from ..security import seed_sources as _seed_sources

    _seed_sources.register(registry)
    _cards.register(registry)

    # BugHunter (Claude-BugHunter port, 2026-11-19): authorized
    # bug-bounty hunting — scope rules, skill map, 7-Question Gate,
    # closure ledger, engagement engine. The organ lives in
    # security/bughunter/ and registers its own tools here, like
    # the other security organs.
    from ..security.bughunter import surface as _bughunter

    _bughunter.register(registry)

    # Session forge + payloads (owner's auth plugins, 2026-11-19):
    # log in with a plugin and persist the cookie jar; verify a
    # stored session; client-side payload generators. The forge and
    # the payloads live in security/ proper — their imports only
    # resolve from there — and the plugin registry lives in
    # security/plugins/__init__.py.
    from ..security import payloads as _payloads
    from ..security import session_forge as _session_forge

    _session_forge.register(registry)
    _payloads.register(registry)

    # The owner's security probe batch, third wave (2026-11-19/20):
    # scope + Governor, param mining + confirmation, injection
    # probing, JWT attacks, GraphQL probe + amplify, race waves,
    # CORS/CSRF, hypothesis planner, leak harvest, dossiers, ban
    # reports, report capture + amplification, request mutation +
    # replay, and the planner→attacker→dossier chain. Each organ
    # lives in security/ and registers its own tools here, like the
    # earlier waves. The L4 seams (browser driver, browser signer,
    # proxy pool) are plugged in below — security/ is L3 and cannot
    # import them itself.
    from ..security import attack_chain as _attack_chain
    from ..security import bans as _bans
    from ..security import cors_csrf as _cors_csrf
    from ..security import dossier as _dossier
    from ..security import graphql as _graphql
    from ..security import inject as _inject
    from ..security import jwt as _jwt
    from ..security import leaks as _leaks
    from ..security import params as _params
    from ..security import planner as _planner
    from ..security import race as _race
    from ..security import replay as _replay
    from ..security import report_amp as _report_amp
    from ..security import scope as _scope

    # The attacker runner is L4 (tools/attacker.py) and the chain is
    # L3 (security/): the runner is injected here, never imported
    # from inside security/ — the layering scan flags even a lazy
    # import. (tools/attacker.py imports security.diff/scope, the
    # other direction, so no cycle.)
    from .attacker import run_attack

    _scope.register(registry)
    _params.register(registry)
    _inject.register(registry)
    _jwt.register(registry)
    _graphql.register(registry)
    _race.register(registry)
    _cors_csrf.register(registry)
    _planner.register(registry)
    _leaks.register(registry)
    _dossier.register(registry)
    _bans.register(registry)
    _report_amp.register(registry)
    _replay.register(registry)
    _attack_chain.register(registry, run_attack)

    ensure_secprobe_seams()
