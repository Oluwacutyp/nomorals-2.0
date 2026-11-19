"""Skill routing tables for the BugHunter engagement engine.

Data ported verbatim from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma): ``engine/skill_map.py``
(``CLASS_SKILL`` / ``TECH_SKILL`` / ``CLASS_PROBE``) and
``engine/engine.py`` (``CLASS_WEIGHT``). The tables are data —
they live together here instead of split across two upstream
files, so the parity runner can pin them in one place.

Presence filtering is Devon's: upstream resolved the skills dir
from the repo checkout / ``~/.claude/skills`` / ``$CBH_SKILLS_DIR``.
Devon resolves it to the vendored library
``nomorals/skills/library/bughunter/skills/`` (override with
``$DEVON_BUGHUNTER_SKILLS_DIR`` for tests). A mapping that names
a skill that is not vendored is filtered out, so the engine never
points at a skill that is not there — and when the directory
cannot be read, nothing is named rather than everything assumed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "CLASS_WEIGHT",
    "CLASS_SKILL",
    "TECH_SKILL",
    "CLASS_PROBE",
    "SKILLS_DIR",
    "skills_for",
    "tech_skills",
    "probe_for",
    "rank_priority",
    "skill_map_selftest",
]

#: Deterministic priority by class (impact-ish); used by rank.
#: Keep in sync with CLASS_SKILL so every routed class gets a
#: sensible (non-default) weight.
CLASS_WEIGHT = {
    "rce": 100, "sqli": 90, "deserialization": 88, "ssti": 88,
    "jwt-crypto": 88, "ato": 88, "ssrf": 85, "auth-bypass": 85,
    "idor": 80, "lfi": 78, "k8s": 78, "xxe": 75,
    "forgot-password": 75, "cache-poison": 72, "cicd": 72,
    "host-header": 70, "http-smuggling": 70, "fintech-graphql": 70,
    "subdomain-takeover": 70, "llm-ai": 70, "mfa": 70, "nosqli": 68,
    "file-upload": 68, "cloud-misconfig": 65, "graphql": 65,
    "saml": 65, "oauth": 62, "websocket": 60, "brute-force": 60,
    "ldap": 60, "race-condition": 58, "business-logic": 58,
    "xss": 55, "spa-api": 55, "rag-vector": 55, "shadow-api": 52,
    "csrf": 50, "captcha-bypass": 48, "cors": 45,
    "open-redirect": 40, "clickjacking": 38, "secret": 35,
    "info-leak": 30, "tls-network": 30, "html-injection": 30,
    "exceptional-conditions": 30, "ntlm-info": 25,
}

#: Attack class -> hunt skill(s) in the bundle.
#: NOTE: hunt-dispatch is an internal loader (for the hunt
#: orchestrator) and is intentionally not listed here — it routes
#: no attack class of its own.
CLASS_SKILL = {
    "sqli": ["hunt-sqli"], "nosqli": ["hunt-nosqli"],
    "xss": ["hunt-xss", "hunt-dom"],
    "ssrf": ["hunt-ssrf"], "idor": ["hunt-idor"],
    "open-redirect": ["hunt-open-redirect"],
    "lfi": ["hunt-lfi"], "ssti": ["hunt-ssti"], "rce": ["hunt-rce"],
    "xxe": ["hunt-xxe"],
    "auth-bypass": ["hunt-auth-bypass", "hunt-session"],
    "ato": ["hunt-ato", "hunt-forgot-password"],
    "llm-ai": ["hunt-llm-ai"],
    "rag-vector": ["hunt-rag-vector", "hunt-llm-ai"],
    "saml": ["hunt-saml"], "oauth": ["hunt-oauth"],
    "mfa": ["hunt-mfa-bypass"],
    "graphql": ["hunt-graphql"],
    "fintech-graphql": ["hunt-fintech-graphql", "hunt-graphql"],
    "csrf": ["hunt-csrf"], "cors": ["hunt-cors"],
    "clickjacking": ["hunt-clickjacking"],
    "info-leak": ["hunt-source-leak", "hunt-api-misconfig"],
    "secret": ["hunt-source-leak"],
    "shadow-api": ["hunt-shadow-api", "hunt-source-leak"],
    "spa-api": ["hunt-spa-api", "hunt-source-leak"],
    "deserialization": ["hunt-deserialization"],
    "file-upload": ["hunt-file-upload"],
    "host-header": ["hunt-host-header"],
    "http-smuggling": ["hunt-http-smuggling"],
    "cache-poison": ["hunt-cache-poison"],
    "race-condition": ["hunt-race-condition"],
    "business-logic": ["hunt-business-logic"],
    "brute-force": ["hunt-brute-force"],
    "captcha-bypass": ["hunt-captcha-bypass"],
    "forgot-password": ["hunt-forgot-password", "hunt-ato"],
    "jwt-crypto": ["hunt-jwt-crypto", "hunt-ato"],
    "websocket": ["hunt-websocket"],
    "html-injection": ["hunt-html-injection", "hunt-xss"],
    "exceptional-conditions": ["hunt-exceptional-conditions",
                               "hunt-source-leak"],
    "cicd": ["hunt-cicd"], "k8s": ["hunt-k8s"],
    "cloud-misconfig": ["hunt-cloud-misconfig"],
    "ldap": ["hunt-ldap"], "ntlm-info": ["hunt-ntlm-info"],
    "tls-network": ["hunt-tls-network"],
    "subdomain-takeover": ["hunt-subdomain"],
}

#: Detected tech -> tech-specific skill(s).
TECH_SKILL = {
    "next.js": ["hunt-nextjs"], "node.js": ["hunt-nodejs"],
    "react": ["hunt-dom"],
    "wordpress": ["hunt-sqli", "hunt-idor"],
    "laravel": ["hunt-laravel"], "spring": ["hunt-springboot"],
    "asp.net": ["hunt-aspnet"], "sharepoint": ["hunt-sharepoint"],
    "graphql": ["hunt-graphql"],
    "grpc": ["hunt-grpc"], "kubernetes": ["hunt-k8s"],
    "docker": ["hunt-k8s"],
    "websocket": ["hunt-websocket"], "socket.io": ["hunt-websocket"],
    "jenkins": ["hunt-cicd"], "teamcity": ["hunt-cicd"],
    "drone": ["hunt-cicd"],
    "argo": ["hunt-cicd"], "github actions": ["hunt-cicd"],
    "gitlab ci": ["hunt-cicd"],
}

#: Curl-first starter probe per class.
#: {u} = url with FUZZ->1, {ur} = injection prefix (before the value).
CLASS_PROBE = {
    "sqli": "curl -s \"{u}\"   # then  ' OR '1'='1'--  and time-based"
            "  ' AND SLEEP(5)--  (Burp Intruder for blind)",
    "nosqli": "curl -s \"{u}\"   # then  [$ne]=1 / {\"$gt\":\"\"}  in the param",
    "xss": "curl -s \"{ur}z1z<svg/onload=alert(1)>\"   "
           "# check the marker comes back UNencoded",
    "open-redirect": "curl -sI \"{ur}https://evil.example\"   "
                     "# inspect the Location: response header",
    "ssrf": "curl -s \"{ur}http://169.254.169.254/latest/meta-data/\"   "
            "# Burp Collaborator for blind OOB",
    "idor": "curl -s \"{u}\"   # swap/increment the id across two "
            "identities; diff the bodies",
    "lfi": "curl -s \"{ur}../../../../etc/passwd\"   "
           "# also php://filter/convert.base64-encode/resource=",
    "ssti": "curl -s \"{ur}{{7*7}}\"   # look for 49 (or ${7*7}); "
            "confirm engine before RCE",
    "rce": "curl -s \"{u}\"   # only with explicit authorization; "
           "start with benign id;sleep markers",
    "xxe": "curl -s -X POST \"{u}\" -d '<?xml ...>'   "
           "# OOB via Collaborator if blind",
    "auth-bypass": "curl -s \"{u}\"   # no token / expired token / "
                     "role swap / forced browse",
    "llm-ai": "curl -s -X POST \"{u}\" -H 'Content-Type: application/json'"
              " -d '{\"messages\":[{\"role\":\"user\",\"content\":\"...\"}]}'   "
              "# prompt-injection / system-prompt extraction",
    "graphql": "curl -s -X POST \"{u}\" "
               "-d '{\"query\":\"{__schema{types{name}}}\"}'   # introspection",
    "csrf": "curl -s \"{u}\"   # check for SameSite / CSRF token on "
            "state-changing POST",
    "info-leak": "curl -s \"{u}\"   # inspect for secrets / verbose "
                 "errors / source / stack traces",
    "secret": "curl -s \"{u}\"   # confirm the exposed secret is present "
              "(read-only — do NOT exercise it)",
    "host-header": "curl -s \"{u}\" -H 'Host: evil.example'   "
                   "# check for reflection / cache / pw-reset poisoning",
    "saml": "curl -s \"{u}\"   # inspect SAML/SSO config endpoints; "
            "test XSW / signature-stripping / IdP confusion (needs auth)",
    "oauth": "curl -s \"{u}\"   # check redirect_uri validation, state, "
             "PKCE, token leakage",
    "ato": "curl -s \"{u}\"   # reset/email-change flows: host-header "
           "poisoning, token in body/referer, no-expiry/reuse",
    "forgot-password": "curl -s \"{u}\"   # diff valid vs invalid email "
                         "responses; check token in response/referer, "
                         "replay, rate limit",
    "brute-force": "curl -s -X POST \"{u}\" -d 'user=a&pass=b'   "
                   "# measure lockout/throttle; try X-Forwarded-For bypass",
    "captcha-bypass": "curl -s -X POST \"{u}\"   # omit/blank/replay the "
                      "captcha field; test cross-endpoint acceptance",
    "jwt-crypto": "curl -s \"{u}\"   # decode the JWT; test alg:none / "
                  "RS256->HS256 key confusion",
    "websocket": "curl -s \"{u}\" -H 'Upgrade: websocket' "
                 "-H 'Connection: Upgrade' "
                 "-H 'Origin: https://evil.example'   "
                 "# check Origin validation",
    "html-injection": "curl -s \"{ur}<b>marker</b>\"   "
                      "# raw HTML reflected (no script exec) -> escalate "
                      "to XSS if JS runs",
    "cache-poison": "curl -s \"{u}\" "
                    "-H 'X-Forwarded-Host: evil.example'   "
                    "# check if unkeyed input is cached and served to others",
    "shadow-api": "curl -s \"{u}\"   # enumerate v1/v2/beta/legacy paths; "
                  "diff Wayback OpenAPI for auth regressions",
    "spa-api": "curl -s \"{u}\"   # pull JS bundle -> backend route map "
               "-> test for missing auth middleware",
    "exceptional-conditions": "curl -s \"{u}\"   # malformed/unexpected "
                             "input -> verbose stack-trace / ORM / path leak",
    "cicd": "curl -s \"{u}\"   # exposed CI dashboard / "
            "workflow-injection surface (pull_request_target, "
            "self-hosted runner)",
    "k8s": "curl -s \"{u}\"   # API anonymous access, kubelet /run, "
           "etcd unauth",
    "cloud-misconfig": "curl -s \"{u}\"   # anonymous read on S3/GCS/Blob; "
                       "PutObjectAcl public-write",
    "ldap": "curl -s \"{ur}*)(objectClass=*\"   "
            "# LDAP search-filter injection in auth/search",
    "ntlm-info": "curl -sI \"{u}\"   # WWW-Authenticate: NTLM -> leak "
                 "domain/forest/computer name",
    "tls-network": "curl -sI \"{u}\"   # HSTS / weak cipher / "
                     "SPF-DKIM-DMARC / DNS AXFR",
    "subdomain-takeover": "curl -sI \"{u}\"   # dangling CNAME -> check "
                          "claimable provider",
    "rag-vector": "curl -s \"{u}\"   # shared knowledge base / vector-DB "
                  "port; cross-tenant query",
    "fintech-graphql": "curl -s -X POST \"{u}\" "
                     "-d '{\"query\":\"{__schema{types{name}}}\"}'   "
                     "# money-movement mutations, idempotency-key "
                     "double-spend",
    "clickjacking": "curl -sI \"{u}\"   # missing X-Frame-Options / CSP "
                    "frame-ancestors -> confirm framing in a browser",
    "cors": "curl -sD - -o /dev/null \"{u}\" "
            "-H 'Origin: https://evil.example'   "
            "# GET, not HEAD: look for Access-Control-Allow-Origin "
            "echoing that origin WITH Allow-Credentials: true",
    "deserialization": "curl -s \"{u}\"   # hunt for rO0AB (Java) / O:8: "
                       "(PHP) / gAS (pickle) / __VIEWSTATE, then replay a "
                       "tampered blob",
    "file-upload": "curl -s -X POST \"{u}\" "
                   "-F 'file=@probe.jpg;type=image/jpeg'   "
                   "# then fetch it back: is it served, and does a "
                   ".php/.svg extension survive",
    "http-smuggling": "curl -s --http1.1 \"{u}\" "
                      "-H 'Transfer-Encoding: chunked' "
                      "-H 'Content-Length: 6'   "
                      "# desync probe; confirm with Burp's HTTP Request "
                      "Smuggler",
    "mfa": "curl -s \"{u}\"   # after the password step: reach the "
           "post-login URL directly, replay an OTP, or omit the MFA field",
    "race-condition": "curl -s -X POST \"{u}\"   # fire N copies in "
                      "parallel (Turbo Intruder single-packet); >1 success "
                      "means the limit is racy",
    "business-logic": "curl -s \"{u}\"   # replay the same action twice / "
                       "skip a step / try qty 0 or negative; diff the "
                       "resulting state",
}


def _resolve_skills_dir() -> Path:
    """Locate the vendored skills/ directory.

    Resolution order: ``$DEVON_BUGHUNTER_SKILLS_DIR`` (explicit
    override, e.g. tests), else the vendored library pack shipped
    with Devon. Unlike upstream there is no ``~/.claude/skills``
    fallback — Devon never reads another agent's home.
    """
    env = os.environ.get("DEVON_BUGHUNTER_SKILLS_DIR")
    if env:
        return Path(env).expanduser()
    return (Path(__file__).resolve().parents[2] / "skills" / "library"
            / "bughunter" / "skills")


SKILLS_DIR = _resolve_skills_dir()


def _present() -> set[str] | None:
    """Names in SKILLS_DIR, or None when it cannot be read.

    None is deliberately distinct from an empty set: an unreadable
    skills directory means presence cannot be *verified*, and
    ``_filter`` must not read that as "assume everything is
    installed".
    """
    try:
        return set(os.listdir(SKILLS_DIR))
    except OSError:
        return None


def _filter(names: list[str]) -> list[str]:
    present = _present()
    if present is None:
        # Unverifiable. Name only skills that are actually
        # installed, so name none rather than every one of them.
        return []
    return [s for s in names if s in present]


def skills_for(vuln_class: str, tech: Any = None) -> list[str]:
    """Class-specific skill(s) for a vuln class."""
    out = _filter(CLASS_SKILL.get(vuln_class, []))
    return out or _filter(["hunt-misc"])


def tech_skills(tech: Any) -> list[str]:
    """Tech-stack skill(s) for the detected service tech."""
    out: list[str] = []
    for t in (tech or []):
        for s in _filter(TECH_SKILL.get((t or "").lower(), [])):
            if s not in out:
                out.append(s)
    return out


def probe_for(vuln_class: str, url: str, param: str | None = None) -> str:
    """Curl-first starter probe string for a (class, url)."""
    tmpl = CLASS_PROBE.get(vuln_class, "curl -s \"{u}\"")
    u = url.replace("FUZZ", "1")
    if "FUZZ" in url:
        ur = url.split("FUZZ", 1)[0]
    elif param:
        ur = url + ("&" if "?" in url else "?") + param + "="
    else:
        ur = url + ("&" if "?" in url else "?") + "x="
    return tmpl.replace("{u}", u).replace("{ur}", ur)


def rank_priority(item: dict[str, Any]) -> int:
    """Upstream rank rule: class weight +5 for params, +55 for secrets."""
    base = CLASS_WEIGHT.get(str(item.get("vuln_class", "")), 20)
    if item.get("param"):
        base += 5
    if item.get("source") == "secret":
        base += 55
    return base


def skill_map_selftest() -> dict[str, Any]:
    """Self-check the maps against the vendored skills.

    Errors are conditions that make the routing wrong or
    unverifiable: the skills directory cannot be read, a mapping
    names a skill that is not vendored, or a vendored hunt-*
    skill is unreachable through every mapping. Warnings are
    coverage gaps: the class still routes to the right skill,
    but no tailored starter probe is defined for it.
    """
    present = _present()
    errors, warnings = [], []

    if present is None:
        errors.append(
            f"skills directory not readable: {SKILLS_DIR} — presence "
            "cannot be verified")
        present = set()

    referenced = {s for skills in
                  list(CLASS_SKILL.values()) + list(TECH_SKILL.values())
                  for s in skills}

    if present:
        missing = sorted(referenced - present)
        if missing:
            errors.append(
                f"mapped but NOT vendored: {', '.join(missing)}")

        # Coverage: every vendored hunt-* skill should be reachable
        # through CLASS_SKILL or TECH_SKILL (hunt-dispatch is an
        # internal loader and hunt-misc is the explicit fallback —
        # both exempt).
        mapped = referenced | {"hunt-misc"}
        unreachable = sorted(
            {s for s in present if s.startswith("hunt-")
             and s != "hunt-dispatch"} - mapped)
        if unreachable:
            errors.append(
                "vendored hunt-* skills NOT reachable through any "
                f"mapping: {', '.join(unreachable)}")

    no_probe = sorted(set(CLASS_SKILL) - set(CLASS_PROBE))
    if no_probe:
        warnings.append(
            "classes with a mapping but no tailored starter probe: "
            + ", ".join(no_probe))

    return {"class_mappings": len(CLASS_SKILL),
            "tech_mappings": len(TECH_SKILL),
            "probes": len(CLASS_PROBE),
            "skills_vendored": len(present),
            "errors": errors, "warnings": warnings,
            "ok": not errors}
