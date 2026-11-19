"""Recon -> manifest -> ranked surface, SPA-aware.

Port of the deterministic recon half of Claude-BugHunter (MIT,
Copyright (c) 2026 Sachin Sharma): ``cbh/cli.py``
(``build_manifest``, ``_rank_host``) and ``engine/recon.py``
(``classify`` / ``classify_all`` / ``classes_for_param``,
SPA mining, platform recon, OpenAPI recon, auth-enforcement
sweep, param gathering, inertness pruning).

Devon differences from upstream:

- All HTTP goes through Devon's :class:`nomorals.core.http.HttpClient`
  (the single TLS policy) instead of ``requests`` / ``urllib`` /
  ``curl`` subprocesses. There is no Burp-proxy / ``CERT_NONE``
  mode: an explicit insecure proxy is a named, logged opt-in
  only, and this module never opts into it.
- Secret scanning reuses Devon's own battery
  (``security/audit_patterns.py`` ``scan_text_for_secrets``) instead
  of upstream's ``SECRET_PATTERNS`` subset — one battery, not two.
  The return shape stays upstream's ``[{type, url, redacted}]``.
- ``gau`` / ``katana`` stay optional external binaries; when they
  are absent ``gather_params`` returns no historical URLs and logs
  it, instead of failing the run.
- The manifest schema (v1.0) is kept verbatim — it is the
  recon->hunt handoff contract.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from ..audit_patterns import redact_secret, scan_text_for_secrets
from .classify import classify_url
from .scope import Scope

__all__ = [
    "build_manifest",
    "classify",
    "classify_all",
    "classes_for_param",
    "classes_for_openapi",
    "auth_enforcement_sweep",
    "platform_recon",
    "openapi_recon",
    "spa_recon",
    "gather_params",
    "prune_inert_params",
    "scan_js_secrets",
]

_log = get_logger(__name__)

PRODUCER = "devon-bughunter/1.0"

#: Subdomain keywords -> triage priority + rationale for ranked_surface.
_P1_HINTS = ("api.", "api-", "graphql", "auth.", "sso.", "login.",
             "account.", "admin.", "internal.", "intranet.",
             "staging.", "stage.", "dev.", "test.", "uat.", "qa.",
             "gateway.", "gw.", "vpn.", "portal.")
_KILL_HINTS = ("cdn.", "static.", "assets.", "img.", "images.",
               "media.", "fonts.")

# ── SPA mining regexes (verbatim upstream) ───────────────────
JS_SRC = re.compile(r'<script[^>]+src=["\']([^"\']+\.js[^"\']*)["\']',
                    re.I)
REL_API = re.compile(
    r'''["'`](/(?:api|rest|graphql|v\d[\w.]*|auth|oauth|users?|account|admin|internal|webhook|'''
    r'''upload|download|files?|proxy|fetch|search|query|email|chat|ai|llm|gateway|service|'''
    r'''session|token|export|import|report|invoice|order)[A-Za-z0-9/_.\-]*)["'`]''')
ABS_URL = re.compile(r'https?://[A-Za-z0-9.\-]+(?:/[A-Za-z0-9/_.\-]*)?')
NEXT_DATA = re.compile(r'/_next/data/[A-Za-z0-9/_.\-]+')
FORM_BLOCK = re.compile(r'<form[\s\S]{0,600}?</form>', re.I)
FORM_ACTION = re.compile(r'action=["\']([^"\']*)["\']', re.I)
INPUT_NAME = re.compile(r'name=["\']([^"\']+)["\']', re.I)

#: Quoted URLs carrying a query string, e.g. "/api/account?id=1".
#: Devon's addition (the verbatim REL_API/ABS_URL above cannot match
#: a URL with a query — their character class stops at `?`, so a
#: link like `/api/account?id=1` mines as nothing at all and the
#: `id` parameter, the whole IDOR surface, is silently lost).
#: The pass below turns each quoted `path?name=` into a
#: (route-without-query, param) work unit.
_QUOTED_QUERY_URL = re.compile(r"""["'`]([^"'`\s]*\?[^"'`\s]*)["'`]""")
_PARAM_NAME = re.compile(r"([A-Za-z0-9_\-.]+)=")

#: Third-party / framework noise to exclude from "absolute backend host" mining.
NOISE = re.compile(
    r'w3\.org|schema\.org|googleapis|gstatic|fonts?\.|sentry|'
    r'google-?analytics|googletagmanager|cloudflare|jsdelivr|unpkg|'
    r'npmjs|github\.com|mozilla|react\.dev|nextjs\.org|vimeo|'
    r'example\.(com|org)|doubleclick|hotjar|segment|intercom|'
    r'facebook|linkedin|twitter|youtube', re.I)

#: An item is real attack surface (vs. a marketing/content page)
#: if its path looks like an API/auth/file route, OR it carries a
#: query parameter, OR it's a form submission.
API_HINT = re.compile(
    r'/(api|rest|graphql|gql|v\d[\w.]*|internal|webhook|auth|oauth|'
    r'token|sso|saml|login|signin|register|upload|download|admin|'
    r'proxy|fetch|gateway|service|session|export|import|search|'
    r'query|user|users|account|order|invoice|file|files|report|'
    r'_next/data)(/|$)', re.I)

OPENAPI_PATHS = ["/openapi.json", "/swagger.json", "/api/openapi.json",
                 "/v1/openapi.json", "/api-docs",
                 "/swagger/v1/swagger.json", "/api/v1/openapi.json"]

WELLKNOWN = ["/__/firebase/init.json",
             "/.well-known/oauth-authorization-server",
             "/.well-known/oauth-protected-resource",
             "/.well-known/openid-configuration",
             "/.well-known/security.txt", "/security.txt", "/robots.txt"]

#: Tracking / cache params with no attack value — dropped from the surface.
PARAM_NOISE = {"utm_source", "utm_medium", "utm_campaign",
               "utm_content", "utm_term", "gclid", "fbclid", "msclkid",
               "mc_cid", "mc_eid", "ref", "trk", "v", "ver", "_", "ts",
               "cache", "cb", "rnd", "version"}

LogFn = Callable[[str], None]


def _rank_host(host: str) -> tuple[str, str]:
    h = host.lower()
    if any(k in h for k in _P1_HINTS):
        return "P1", "high-value surface (api/auth/admin/non-prod)"
    if any(h.startswith(k) for k in _KILL_HINTS):
        return "KILL", "static/CDN host — low yield"
    return "P2", "standard web surface"


def build_manifest(target: str, subs: Any, resolved: dict[str, Any],
                   live: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the recon->hunt manifest (schema v1.0, verbatim).

    Devon fills assets + ranked_surface; the OSINT skill's deeper
    probes append secrets[] and identity_fabric{}.
    """
    live_by_host = {r["host"]: r for r in live}
    assets = []
    for host in sorted(live_by_host):
        r = live_by_host[host]
        assets.append({
            "host": host, "ips": resolved.get(host, []),
            "url": r.get("url"), "status": r.get("code"),
            "server": r.get("server", ""), "title": r.get("title", ""),
            "tech": r.get("tech", []), "source": "crtsh+httpx",
        })
    for host in sorted(resolved):  # DNS-only (resolved, not HTTP-live)
        if host not in live_by_host:
            assets.append({
                "host": host, "ips": resolved[host], "url": None,
                "status": None, "server": "", "title": "", "tech": [],
                "source": "crtsh+dns",
            })

    ranked = []
    for host in sorted(live_by_host):
        r = live_by_host[host]
        prio, why = _rank_host(host)
        bug_classes = sorted(classify_url(r["url"])["matches"].keys()) \
            if r.get("url") else []
        ranked.append({"url": r.get("url"), "host": host,
                       "bug_classes": bug_classes, "priority": prio,
                       "rationale": why})
    _order = {"P1": 0, "P2": 1, "KILL": 2}
    ranked.sort(key=lambda x: (_order.get(x["priority"], 9), x["host"]))

    return {
        "schema_version": "1.0",
        "target": target,
        # timezone.utc, not datetime.UTC: EC2 runs Python <= 3.11 and
        # this box's 3.12.3 build lacks datetime.UTC (verified).
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc  # noqa: UP017
        ).isoformat(timespec="seconds"),
        "producers": [PRODUCER],
        "counts": {"subdomains": len(subs), "resolved": len(resolved),
                   "live": len(live)},
        "assets": assets,
        "ranked_surface": ranked,
        # Filled in by the offensive-osint skill's deeper probes:
        "secrets": [],            # {pattern, severity, category, source}
        "identity_fabric": {},    # {idp, tenant, domains, ...}
    }


# ── deterministic vuln-class classification (verbatim upstream) ──
def classify(item: dict[str, Any]) -> str:
    """Cheap heuristic -> likely vuln class (no LLM). Param-name signals win first."""
    url = item.get("url", "").lower()
    p = (item.get("param", "") or "").lower()
    PARAM = {
        "open-redirect": {"url", "uri", "next", "redirect", "return",
                          "returnurl", "return_url", "callback", "goto",
                          "dest", "destination", "continue", "u", "link",
                          "target"},
        "lfi": {"file", "path", "page", "template", "include", "doc",
                "document", "filename", "filepath", "load"},
        "ssrf": {"image", "imageurl", "image_url", "fetch", "webhook",
                 "proxy", "feed", "host", "port", "ip", "endpoint"},
        "idor": {"id", "uid", "user", "userid", "user_id", "account",
                 "accountid", "order", "orderid", "invoice", "docid",
                 "pid", "oid", "ref"},
        "sqli": {"q", "query", "search", "keyword", "s", "term",
                 "filter", "sort"},
    }
    for cls, names in PARAM.items():
        if p in names:
            return cls
    rules = [
        (("ws://", "wss://", "/socket.io", "websocket"), "websocket"),
        (("/graphql",), "graphql"),
        (("/chat", "/ai", "/llm", "completion", "/assistant", "/agent",
          "prompt", "/copilot"), "llm-ai"),
        (("redirect", "returnurl", "callback", "goto", "/go"),
         "open-redirect"),
        (("/upload", "filename", "filepath", "/download", "/file"),
         "lfi"),
        (("/account", "/users", "/user/", "userid", "/profile", "/order",
          "/invoice", "/report", "/export"), "idor"),
        (("/search", "/query", "keyword"), "sqli"),
        (("/login", "/signin", "/register", "/auth", "/oauth", "/token",
          "/session"), "auth-bypass"),
        (("/proxy", "/fetch", "webhook"), "ssrf"),
        (("/jenkins", "/teamcity", "/drone", "/argo", "/.gitlab-ci",
          "/.github/workflows"), "cicd"),
        (("kubernetes", "/apis/", "/api/v1/namespaces", "kubelet"),
         "k8s"),
        (("s3.amazonaws", "storage.googleapis", "blob.core.windows",
          "cloudfront.net"), "cloud-misconfig"),
        (("/forgot-password", "/forgot", "/reset-password",
          "/password-reset", "/recover"), "forgot-password"),
        (("captcha", "recaptcha"), "captcha-bypass"),
        (("jwt",), "jwt-crypto"),
    ]
    for keys, cls in rules:
        if any(k in url for k in keys):
            return cls
    return "idor" if p else "info-leak"


def classify_all(item: dict[str, Any], max_classes: int = 3) -> list[str]:
    """Ranked list of plausible vuln classes for an endpoint (multi-class hunting).

    A preset class (e.g. a secret finding) stays single-class —
    don't multi-test a confirmed info-leak.
    """
    if item.get("source") == "secret" and item.get("vuln_class"):
        return [item["vuln_class"]]
    classes = [classify(item)]
    url = item.get("url", "").lower()
    p = (item.get("param", "") or "").lower()
    extras: list[str] = []
    if any(k in url for k in ("/chat", "/ai", "/llm", "/assistant",
                              "/agent", "/copilot")):
        extras += ["llm-ai", "ssrf", "info-leak"]
    if any(k in url for k in ("/search", "/query")) or p in (
            "q", "query", "search", "s", "keyword", "filter"):
        extras += ["xss", "sqli", "ssti"]
    if "redirect" in url or p in ("url", "uri", "next", "redirect",
                                  "return", "callback", "dest", "u"):
        extras += ["open-redirect", "ssrf"]
    if "/graphql" in url:
        extras += ["graphql", "info-leak"]
    if any(k in url for k in ("/api", "/rest", "/v1", "/v2")):
        extras += ["info-leak", "idor", "auth-bypass"]
    if any(k in url for k in ("ws://", "wss://", "/socket.io",
                              "websocket")):
        extras += ["websocket"]
    if re.search(r"/v\d+/", url):
        extras += ["shadow-api"]
    if p:  # any parameter is a generic injection / IDOR surface
        extras += ["idor", "sqli", "xss"]
    for e in extras:
        if e not in classes:
            classes.append(e)
    return classes[:max_classes]


def classes_for_param(name: str, url: str = "") -> list[str]:
    """Deterministic per-parameter attack categorization (name-driven). Returns ranked classes."""
    n = (name or "").lower()
    cls: list[str] = []

    def add(*cs: str) -> None:
        for c in cs:
            if c not in cls:
                cls.append(c)

    if n in ("url", "uri", "next", "dest", "destination", "u", "link",
             "target", "return", "continue") \
            or any(k in n for k in ("redirect", "returnurl", "return_to",
                                    "callback", "goto", "forward")):
        add("open-redirect", "ssrf")
    if any(k in n for k in ("image", "fetch", "webhook", "proxy",
                            "feed", "remote", "load_url")):
        add("ssrf")
    if (any(k in n for k in ("file", "path", "template", "include",
                             "document", "folder", "dir"))
            and not n.endswith("id")):
        add("lfi")
    if n in ("id", "uid", "pid", "oid", "gid") or n.endswith("_id") \
            or n.endswith("id") \
            or any(k in n for k in ("user", "account", "order",
                                    "invoice", "node", "post", "item",
                                    "doc", "page", "cat", "tag",
                                    "profile")):
        add("idor", "sqli")
    if n in ("q", "s", "query", "search", "keyword", "filter", "term",
             "sort", "name", "email", "cat", "tag", "lang"):
        add("sqli", "xss")
    if any(k in n for k in ("auth", "token", "session", "login", "sso",
                            "saml", "oauth")):
        add("auth-bypass")
    if any(k in n for k in ("otp", "mfa", "2fa", "totp", "captcha",
                            "recaptcha")):
        add("mfa", "brute-force", "captcha-bypass")
    if any(k in n for k in ("reset", "recover", "forgot")):
        add("forgot-password", "ato")
    if "jwt" in n:
        add("jwt-crypto", "auth-bypass")
    if any(k in n for k in ("websocket", "socket")):
        add("websocket")
    if not cls:
        # Any reflected/used param is a generic injection/IDOR surface.
        add("sqli", "xss", "idor")
    return cls[:3]


# ── HTTP through Devon's HttpClient ──────────────────────────
def _fetch(client: HttpClient, url: str, timeout: float = 20.0
           ) -> tuple[int, dict[str, str], str, str]:
    """GET via Devon's HttpClient -> (status, headers, body, final_url).

    A failed fetch returns (0, {}, "", url) — the caller treats it
    as "no data", never as "no surface".
    """
    try:
        resp = client.get(url, timeout=timeout)
        return (resp.status,
                {k.lower(): v for k, v in resp.headers.items()},
                resp.text, resp.url or url)
    except Exception as exc:  # noqa: BLE001 - fetch failure is data
        _log.debug("bughunter fetch failed %s: %s", url, exc)
        return 0, {}, "", url


def _probe(client: HttpClient, url: str, timeout: float = 12.0
           ) -> tuple[int, str]:
    """GET url, return (status_code, body). Status-aware."""
    code, _headers, body, _final = _fetch(client, url, timeout=timeout)
    return code, body


def scan_js_secrets(text: str, url: str) -> list[dict[str, Any]]:
    """Secrets exposed in client-side text, via Devon's own battery.

    Returns upstream's shape ``[{type, url, redacted}]`` with values
    redacted — Devon's ``scan_text_for_secrets`` already redacts
    through ``redact_secret`` (head+tail only).
    """
    out, seen = [], set()
    for hit in scan_text_for_secrets(text or ""):
        value = hit.value
        if not value or value in seen:
            continue
        seen.add(value)
        label = str(hit.pattern) or "secret"
        out.append({"type": label, "url": url,
                    "redacted": redact_secret(value)})
    return out


def _mine(text: str, found_rel: set, found_abs: set,
          base: str = "") -> None:
    # found_rel stores (base, relpath) so each route is later
    # resolved against the document it was mined from — not a
    # single shared seed origin (fixes multi-seed).
    for m in REL_API.findall(text):
        found_rel.add((base, m))
    for m in NEXT_DATA.findall(text):
        found_rel.add((base, m))
    for m in ABS_URL.findall(text):
        if not NOISE.search(m):
            found_abs.add(m.rstrip('.,);"\''))


def _mine_query_params(text: str, base: str) -> list[tuple[str, str]]:
    """(route, param) pairs from quoted URLs carrying a query string.

    REL_API/ABS_URL cannot match these — the verbatim upstream
    regexes stop at `?` — so without this pass a link like
    `/api/account?id=1` mines as nothing at all and the parameter,
    the whole IDOR surface, is silently lost. The returned URL is
    the route WITHOUT the query; the param travels in `param`,
    which is what classify/probe_for consume.
    """
    found: list[tuple[str, str]] = []
    for raw in _QUOTED_QUERY_URL.findall(text):
        path, _, query = raw.partition("?")
        if not query:
            continue
        names = _PARAM_NAME.findall(query)
        if not names:
            continue
        url = path if path.startswith("http") else urljoin(
            (base or "") + "/", path.lstrip("/"))
        for name in dict.fromkeys(names):
            found.append((url, name))
    return found


def _testable(item: dict[str, Any]) -> bool:
    u = item.get("url", "")
    return bool(API_HINT.search(urlparse(u).path)) or "?" in u or \
        item.get("source") == "form" or bool(item.get("param"))


def spa_recon(scope: Scope, client: HttpClient, seeds: Any = None,
              max_js: int = 20, interesting_only: bool = True,
              log: LogFn | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministically map the in-scope surface.

    Fetches seed HTML + robots/sitemap, pulls same-origin JS
    bundles, regex-mines endpoints (relative /api routes, absolute
    backend hosts, forms, Next.js data), scope-filters. Returns
    (surface_items, out_of_scope_hosts).
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    found_rel: set = set()
    found_abs: set = set()
    forms: list = []
    js_urls: set = set()
    query_params: list[tuple[str, str]] = []
    secrets: list[dict[str, Any]] = []
    seed_base = ""
    for seed in (seeds or scope.seeds):
        if not scope.in_scope_host(seed):
            log(f"recon: seed {seed} not in scope — skip")
            continue
        _status, hdrs, html, final = _fetch(client, seed)
        base = (f"{urlparse(final).scheme}://{urlparse(final).netloc}")
        if not seed_base:
            seed_base = base
        log(f"recon: fetched {seed} ({len(html)} bytes, "
            f"server={hdrs.get('server', '?')})")
        for p in ("/robots.txt", "/sitemap.xml"):
            _status, _hdrs, t, _final = _fetch(client, base + p)
            for a, b in re.findall(
                    r'(?:Allow|Disallow):\s*(\S+)|<loc>([^<]+)</loc>', t):
                pp = a or b
                if pp:
                    found_rel.add(
                        (base, pp if pp.startswith('/')
                         else (urlparse(pp).path or '/')))
        _mine(html, found_rel, found_abs, base)
        query_params += _mine_query_params(html, base)
        secrets += scan_js_secrets(html, final)
        for m in JS_SRC.findall(html):
            ju = urljoin(base, m)
            if scope.in_scope_host(ju):
                js_urls.add(ju)
        for fm in FORM_BLOCK.findall(html):
            act = FORM_ACTION.search(fm)
            forms.append({
                "action": urljoin(base, act.group(1)) if act else seed,
                "params": INPUT_NAME.findall(fm)})

    for ju in list(js_urls)[:max_js]:
        _status, _hdrs, js, _final = _fetch(client, ju)
        if js:
            js_base = (f"{urlparse(ju).scheme}://{urlparse(ju).netloc}")
            _mine(js, found_rel, found_abs, js_base)
            query_params += _mine_query_params(js, js_base)
            secrets += scan_js_secrets(js, ju)
    # Dedup secrets by (type, redacted).
    _seen = set()
    secrets = [s for s in secrets
               if (s["type"], s["redacted"]) not in _seen
               and not _seen.add((s["type"], s["redacted"]))]
    log(f"recon: mined {len(found_rel)} route(s), {len(found_abs)} "
        f"absolute url(s), {len(forms)} form(s), {len(secrets)} "
        f"secret(s) from {len(js_urls)} JS bundle(s)")
    for s in secrets:
        log(f"recon:   SECRET {s['type']} = {s['redacted']} in {s['url']}")

    items: list[dict[str, Any]] = []
    oos: set[str] = set()
    for base_i, rel in sorted(found_rel):
        b = base_i or seed_base  # resolve against the route's own doc
        url = rel if rel.startswith('http') else urljoin(
            (b or "") + '/', rel.lstrip('/'))
        if scope.in_scope_host(url):
            items.append({"url": url, "param": "", "source": "js/html"})
    for au in sorted(found_abs):
        if scope.in_scope_host(au):
            items.append({"url": au, "param": "", "source": "absolute"})
        else:
            oos.add(urlparse(au).netloc)
    for f in forms:
        if scope.in_scope_host(f["action"]):
            for p in (f["params"] or [""]):
                items.append({"url": f["action"], "param": p,
                              "source": "form", "method": "POST"})
    for url, param in query_params:
        if scope.in_scope_host(url):
            items.append({"url": url, "param": param, "source": "js/html"})
        else:
            oos.add(urlparse(url).netloc)

    seen, uniq = set(), []
    for it in items:
        k = (it["url"], it.get("param", ""))
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    if interesting_only:
        testable = [it for it in uniq if _testable(it)]
        log(f"recon: {len(uniq)} route(s) total -> {len(testable)} "
            f"testable endpoint(s) ({len(uniq) - len(testable)} "
            f"marketing/content pages filtered out)")
        uniq = testable
    # Secrets are findings in themselves — always kept as info-leak items.
    for s in secrets:
        uniq.append({"url": s["url"], "param": "", "vuln_class": "info-leak",
                     "source": "secret",
                     "note": f"{s['type']} exposed in client-side JS",
                     "evidence": f"{s['type']} = {s['redacted']} "
                                 f"(in {s['url']})"})
    return uniq, sorted(oos)


# ── platform / OpenAPI recon ─────────────────────────────────
def classes_for_openapi(path: str, method: str, op: dict[str, Any],
                        secured: bool) -> list[str]:
    """Categorize an OpenAPI operation into attack classes (name + shape driven)."""
    pl = path.lower()
    cls: list[str] = []

    def add(*cs: str) -> None:
        for c in cs:
            if c not in cls:
                cls.append(c)

    if "{" in path:  # path parameter -> object reference (IDOR/BOLA)
        add("idor")
    if any(k in pl for k in ("/admin", "permission", "role", "/users",
                             "/orgs", "deactivate", "reactivate",
                             "/members")):
        add("auth-bypass")  # broken access control / privilege escalation
    if any(k in pl for k in ("sso", "saml")):
        add("saml")
    if any(k in pl for k in ("oauth", "keycloak", "/auth", "/login",
                             "token")):
        add("auth-bypass")
    has_param = any(pr.get("in") in ("query", "path")
                    for pr in op.get("parameters", []))
    if has_param or method in ("post", "put", "patch"):
        add("sqli")
    if not cls:
        add("info-leak")
    return cls[:3]


def auth_enforcement_sweep(spec: dict[str, Any], base: str,
                           client: HttpClient,
                           log: LogFn | None = None,
                           junk: str = "zzz-pentest-nonexistent-00000",
                           *, allow_intrusive: bool = False
                           ) -> list[dict[str, Any]]:
    """Probe declared-secured operations unauthenticated.

    By default, probe only GET. Report responses that reach the
    handler or request validation without authentication.
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    # A global `security: []` means "no auth for the whole API" —
    # treat as NOT secured, else public ops are wrongly swept as
    # broken-auth.
    glob = spec.get("security") not in (None, [])
    broken, n, skipped_writes = [], 0, 0
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for m, op in methods.items():
            if m not in ("get", "post", "put", "patch", "delete"):
                continue
            if not (("security" in op and op["security"] != [])
                    or ("security" not in op and glob)):
                continue
            if m != "get" and not allow_intrusive:
                skipped_writes += 1
                continue
            url = base + path
            for pp in op.get("parameters", []):
                if pp.get("in") == "path":
                    url = url.replace("{" + pp["name"] + "}", junk)
            url = url.replace("{", "").replace("}", "")
            q = [f"{pp['name']}={junk}" for pp in op.get("parameters", [])
                 if pp.get("in") == "query" and pp.get("required")]
            if q:
                url += ("&" if "?" in url else "?") + "&".join(q)
            if m in ("post", "put", "patch"):
                try:
                    resp = client.request(
                        m.upper(), url, data=b"{}",
                        headers={"Accept": "application/json",
                                 "Content-Type": "application/json"},
                        timeout=10.0)
                    code = resp.status
                except Exception:  # noqa: BLE001 - probe failure is data
                    code = 0
            else:
                code, _body = _probe(client, url, timeout=10.0)
            n += 1
            # Reached the handler/validation unauth = NOT enforced.
            if code in (200, 201, 202, 204, 422):
                broken.append({
                    "url": url, "param": "", "vuln_class": "auth-bypass",
                    "source": "auth-sweep",
                    "note": f"{m.upper()} {path} -> {code} UNAUTH "
                            "(declared-secured, NOT enforced)"})
            time.sleep(0.03)
    if skipped_writes:
        log(f"auth-sweep: skipped {skipped_writes} declared-secured "
            "write operation(s) in read-only mode")
    if broken:
        log(f"auth-sweep: {len(broken)}/{n} declared-secured op(s) NOT "
            "auth-enforced (BROKEN function-level auth)")
    else:
        log(f"auth-sweep: {n}/{n} declared-secured op(s) correctly "
            "enforce auth (401/403) — verified")
    return broken


def platform_recon(scope: Scope, client: HttpClient,
                   log: LogFn | None = None) -> list[dict[str, Any]]:
    """Probe platform/.well-known config endpoints — cheap, high-signal.

    Firebase config disclosure, OAuth/OIDC discovery (reveals
    authorize/token/register), robots disallows.
    Returns surface items.
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    items, seen = [], set()
    for seed in scope.seeds:
        u = urlparse(seed if "://" in seed else "https://" + seed)
        base = f"{u.scheme or 'https'}://{u.hostname}"
        if base in seen:
            continue
        seen.add(base)
        for wk in WELLKNOWN:
            code, body = _probe(client, base + wk)
            if code != 200:
                continue
            b = body.lstrip()
            if "firebase/init" in wk and b.startswith("{"):
                try:
                    cfg = json.loads(body)
                    log(f"platform: {base}{wk} -> Firebase config "
                        f"(projectId={cfg.get('projectId', '')}, "
                        f"bucket={cfg.get('storageBucket', '')})")
                    items.append({
                        "url": base + wk, "param": "",
                        "vuln_class": "info-leak", "source": "platform",
                        "note": "Firebase config disclosed "
                                f"(projectId={cfg.get('projectId', '')})"})
                except (ValueError, TypeError):
                    pass
            elif ("oauth" in wk or "openid" in wk) and b.startswith("{"):
                try:
                    meta = json.loads(body)
                    log(f"platform: {base}{wk} -> OAuth/OIDC discovery "
                        f"(issuer={meta.get('issuer', '')})")
                    items.append({
                        "url": base + wk, "param": "",
                        "vuln_class": "info-leak", "source": "platform",
                        "note": "OAuth/OIDC discovery exposed"})
                    for k in ("authorization_endpoint", "token_endpoint",
                              "registration_endpoint"):
                        if meta.get(k):
                            items.append({
                                "url": meta[k], "param": "",
                                "vuln_class": "oauth", "source": "platform",
                                "note": f"OAuth {k}"})
                except (ValueError, TypeError):
                    pass
            elif "robots" in wk:
                dis = [ln.split(":", 1)[1].strip()
                       for ln in body.splitlines()
                       if ln.lower().startswith("disallow:")
                       and ln.split(":", 1)[1].strip() not in ("", "/")]
                if dis:
                    log(f"platform: {base}/robots.txt disallows "
                        f"{len(dis)} path(s): "
                        f"{', '.join(sorted(set(dis))[:6])}")
                    for d in sorted(set(dis))[:10]:
                        items.append({
                            "url": base + d, "param": "",
                            "vuln_class": "info-leak",
                            "source": "robots-disallow",
                            "note": "robots-disallowed path"})
            elif "security.txt" in wk:
                log(f"platform: {base}{wk} -> security.txt present")
    return items


def openapi_recon(scope: Scope, client: HttpClient,
                  log: LogFn | None = None,
                  *, allow_intrusive: bool = False
                  ) -> list[dict[str, Any]]:
    """If the target exposes an OpenAPI schema, parse it into categorized work units.

    The ideal deterministic recon for an API host. Returns
    already-expanded surface items.
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    items = []
    for seed in scope.seeds:
        host = seed.rstrip("/")
        if "://" not in host:
            host = "https://" + host
        for op_path in OPENAPI_PATHS:
            code, body = _probe(client, host + op_path)
            if code != 200 or not body.lstrip().startswith("{"):
                continue
            try:
                spec = json.loads(body)
            except ValueError:
                continue
            paths = spec.get("paths")
            if not paths:
                continue
            glob_sec = spec.get("security") not in (None, [])
            title = spec.get("info", {}).get("title", "API")
            noauth = 0
            for p, methods in paths.items():
                if not isinstance(methods, dict):
                    continue
                for m, opn in methods.items():
                    if m not in ("get", "post", "put", "patch", "delete"):
                        continue
                    secured = (("security" in opn
                                and opn["security"] != [])
                               or ("security" not in opn and glob_sec))
                    if not secured:
                        noauth += 1
                    for cls in classes_for_openapi(p, m, opn, secured):
                        items.append({
                            "url": f"{host}{p}", "param": "",
                            "method": m.upper(), "vuln_class": cls,
                            "source": "openapi",
                            "note": f"{m.upper()} {p} "
                                    f"[{'AUTH' if secured else 'NO-AUTH'}]"})
            log(f"openapi: {host}{op_path} -> '{title}', {len(paths)} "
                f"path(s) ({noauth} no-auth op(s)) -> {len(items)} "
                "categorized unit(s)")
            items += auth_enforcement_sweep(
                spec, host, client, log,
                allow_intrusive=allow_intrusive)
            break
    return items


# ── historical parameters (gau / katana, optional binaries) ──
def _gau_urls(scope: Scope, log: LogFn) -> dict[str, str]:
    """Passive: gau historical parameterized URLs (in-scope). url -> source."""
    if not shutil.which("gau"):
        log("params: gau not installed — skipping historical param discovery")
        return {}
    urls: dict[str, str] = {}
    apexes = set()
    for s in scope.seeds:
        h = urlparse(s if "://" in s else "//" + s).hostname
        if h:
            apexes.add(".".join(h.split(".")[-2:]))
    for apex in sorted(apexes):
        try:
            out = subprocess.run(["gau", apex], capture_output=True,
                                 text=True, timeout=120).stdout
        except Exception:  # noqa: BLE001 - a dead gau is "no gau"
            out = ""
        for u in out.splitlines():
            u = u.strip()
            if u and "?" in u and scope.in_scope_host(u):
                urls.setdefault(u, "gau-historical")
    log(f"params: gau -> {len(urls)} in-scope parameterized historical "
        f"URL(s)"
        + ("" if urls else " (0 — gau likely rate-limited/transient "
             "this run; katana below is independent)"))
    return urls


def _katana_urls(scope: Scope, log: LogFn, depth: int = 2,
                 timeout: int = 150) -> dict[str, str]:
    """Active live crawl (katana, GET-only) -> in-scope URLs. Independent of gau."""
    if not shutil.which("katana"):
        log("params: katana not installed — skipping live crawl")
        return {}
    urls, crawled = {}, 0
    for seed in scope.seeds:
        try:
            out = subprocess.run(
                ["katana", "-u", seed, "-silent", "-jc", "-d",
                 str(depth), "-kf", "all", "-c", "10", "-timeout", "10",
                 "-rl", "100"],
                capture_output=True, text=True,
                timeout=timeout).stdout
        except Exception:  # noqa: BLE001 - a dead katana is "no katana"
            out = ""
        for u in out.splitlines():
            u = u.strip()
            if u and scope.in_scope_host(u):
                crawled += 1
                if "?" in u:
                    urls.setdefault(u, "katana-crawl")
    log(f"params: katana live-crawl -> {crawled} in-scope URL(s), "
        f"{len(urls)} parameterized")
    return urls


def prune_inert_params(items: list[dict[str, Any]], client: HttpClient,
                       log: LogFn | None = None) -> list[dict[str, Any]]:
    """Drop param items whose ENDPOINT is dead (>=400) or whose PARAM is inert.

    Deterministic: marker-normalized body-diff vs the no-param
    baseline — removing the marker first neutralizes the
    canonical/og:url reflection trap. Keeps only live, processable
    params. (Active: ~1 request per unique endpoint + 1 per param.)
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    kept, dead, inert, failed, base_cache = [], 0, 0, 0, {}
    marker = "zq9k7xp"
    for it in items:
        base_path = it.get("url", "").split("?", 1)[0]
        param = it.get("param", "")
        if base_path not in base_cache:
            base_cache[base_path] = _probe(client, base_path)
        bcode, bbody = base_cache[base_path]
        if bcode == 0:
            # Probe FAILED (network/client) — can't judge; keep, don't prune.
            kept.append(it)
            failed += 1
            continue
        if bcode >= 400:
            dead += 1
            continue  # genuinely dead endpoint (real HTTP >= 400)
        tcode, tbody = _probe(client, f"{base_path}?{param}={marker}")
        if tcode == 0:
            kept.append(it)  # param probe failed — keep, not judged dead
            failed += 1
            continue
        if tcode >= 400:
            dead += 1
            continue
        norm = tbody.replace(marker, "")  # strip the param's own echo
        if abs(len(norm) - len(bbody)) <= 8 \
                and norm[:6000] == bbody[:6000]:
            inert += 1
            continue  # param ignored — no effect on response
        kept.append(it)
    if dead or inert or failed:
        log(f"params: inertness-prune dropped {dead} dead-endpoint + "
            f"{inert} inert param(s); kept {failed} unprobeable (probe "
            f"failed, not judged dead) -> {len(kept)} remain")
    return kept


def gather_params(scope: Scope, client: HttpClient,
                  log: LogFn | None = None,
                  max_items: int = 80) -> list[dict[str, Any]]:
    """Deterministic parameter discovery from TWO independent sources.

    gau (passive historical) and katana (active live crawl) —
    merged, scope/noise-filtered into (endpoint,param) pairs, then
    inertness-pruned so only live/processable params reach the map.
    When neither binary is installed, returns [] and logs it.
    """
    log = log or (lambda m: _log.info("bughunter %s", m))
    src: dict[str, str] = {}
    for u, s in _gau_urls(scope, log).items():
        src.setdefault(u, s)
    for u, s in _katana_urls(scope, log).items():
        src.setdefault(u, s)
    items, seen = [], set()
    for u in sorted(src):
        base = u.split("?", 1)[0]
        if base.count("://") != 1 or re.search(
                r'https?:/', urlparse(u).path or ""):
            continue  # skip malformed entries (double-scheme, etc.)
        for p in parse_qs(urlparse(u).query):
            pl = p.lower()
            if pl in PARAM_NOISE or pl.startswith("utm_"):
                continue
            key = (base, pl)
            if key in seen:
                continue
            seen.add(key)
            items.append({"url": f"{base}?{p}=FUZZ", "param": p,
                          "method": "GET", "source": src[u]})
    log(f"params: {len(items)} unique (endpoint,param) pair(s) from "
        "gau+katana after noise filter")
    return prune_inert_params(items[:max_items], client, log)


def recon_selftest() -> bool:
    """Upstream's offline regex self-test, verbatim in cases."""
    rel, ab = set(), set()
    _mine('a "/api/completion" b fetch("/api/v1/users") '
          '<a href="https://api.acme.com/x"> "/graphql" '
          '"https://www.googleapis.com/y"', rel, ab)
    rels = {r for _, r in rel}
    assert "/api/completion" in rels and "/api/v1/users" in rels \
        and "/graphql" in rels, rel
    assert "https://api.acme.com/x" in ab, ab
    assert not any("googleapis" in a for a in ab), "noise not filtered"
    assert classify({"url": "/api/completion"}) == "llm-ai"
    assert classify({"url": "/api/account", "param": "id"}) == "idor"
    assert classify({"url": "/go", "param": "url"}) == "open-redirect"
    assert classify({"url": "/x", "param": "file"}) == "lfi"
    return True
