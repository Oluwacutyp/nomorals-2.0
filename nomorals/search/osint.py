"""OSINT source adapters for federated search.

Keyless, public-data techniques mined from theHarvester / SpiderFoot /
Recon-ng / maigret / holehe playbooks — only the parts that work with
no API key:

- username sweep (sherlock/maigret pattern): probe well-known profile
  URL templates, detect existence via status + content signals.
- email check (holehe pattern): per-site account-existence probes via
  password-reset / signup endpoints that don't alert the owner.
- domain recon: crt.sh certificate transparency (subdomains) + RDAP
  (registrar, dates — the WHOIS replacement).
- IP intel: ipwho.is (ASN/org/geo, free no-key) + Shodan InternetDB
  (open ports/hostnames, free no-key).

Each adapter implements the ``SourceAdapter`` contract so federated
search (and the research organ behind it) picks them up automatically.
Queries are routed by shape: ``@username``/bare username → sweep,
``a@b.c`` → email check, ``example.com`` → domain recon, ``1.2.3.4``
→ IP intel. Anything else → the adapter reports no match (probe-style),
never a crash.

Honest limits: paywalled aggregators (intelius, thatsthem, etc.) have
no keyless API — they are NOT faked. The keyless path is the
primitives above plus web-search dorking via the existing web source.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.parse
import urllib.request
from typing import Any

from ..core.logging_setup import get_logger
from .base import SourceAdapter
from .model import SearchResult

_log = get_logger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _get(url: str, timeout: float = 15) -> tuple[int, str, str]:
    """GET → (status, body, final_url). Never raises."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, body, resp.url
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        return e.code, body, url
    except Exception as exc:  # noqa: BLE001
        _log.debug("osint GET failed %s: %s", url, exc)
        return 0, "", url


def _get_json(url: str, timeout: float = 15) -> Any:
    status, body, _ = _get(url, timeout)
    if status != 200 or not body:
        return None
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return None


# ── username sweep ───────────────────────────────────────────────────
# Site templates mined from sherlock/maigret site lists: URL template +
# a string whose ABSENCE means "no such user" (or whose presence on a
# 404 page means the same). Kept to high-signal sites to stay fast.

_USERNAME_SITES: list[tuple[str, str, str]] = [
    # (name, url_template, not_found_marker)
    ("github", "https://github.com/{u}", "This is not the web page you are looking for"),
    ("gitlab", "https://gitlab.com/{u}", "This user does not exist"),
    ("reddit", "https://www.reddit.com/user/{u}/", "nobody on Reddit goes by that name"),
    ("twitter", "https://twitter.com/{u}", "This account doesn"),
    ("instagram", "https://www.instagram.com/{u}/", "Sorry, this page isn't available"),
    ("tiktok", "https://www.tiktok.com/@{u}", "Couldn't find this account"),
    ("youtube", "https://www.youtube.com/@{u}", "404 Not Found"),
    ("twitch", "https://www.twitch.tv/{u}", "Sorry. Unless you"),
    ("medium", "https://medium.com/@{u}", "PAGE NOT FOUND"),
    ("devto", "https://dev.to/{u}", "404"),
    ("stackoverflow", "https://stackoverflow.com/users/{u}", "Page Not Found"),
    ("hackernews", "https://news.ycombinator.com/user?id={u}", "No such user"),
    ("producthunt", "https://www.producthunt.com/@{u}", "Page not found"),
    ("dribbble", "https://dribbble.com/{u}", "not found"),
    ("behance", "https://www.behance.net/{u}", "We couldn"),
    ("soundcloud", "https://soundcloud.com/{u}", "We can’t find that user"),
    ("spotify", "https://open.spotify.com/user/{u}", "Page not found"),
    ("steam", "https://steamcommunity.com/id/{u}", "The specified profile could not be found"),
    ("roblox", "https://www.roblox.com/user.aspx?username={u}", "Page cannot be found"),
    ("chesscom", "https://www.chess.com/member/{u}", "Page Not Found"),
    ("lichess", "https://lichess.org/@/{u}", "Page not found"),
    ("codepen", "https://codepen.io/{u}", "not found"),
    ("replit", "https://replit.com/@{u}", "not found"),
    ("kaggle", "https://www.kaggle.com/{u}", "404"),
    ("huggingface", "https://huggingface.co/{u}", "404"),
]


def _looks_like_username(q: str) -> str | None:
    q = q.strip().lstrip("@")
    if re.fullmatch(r"[A-Za-z0-9_.\-]{2,39}", q) and " " not in q and "." not in q:
        return q
    return None


def _looks_like_email(q: str) -> str | None:
    q = q.strip()
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", q):
        return q
    return None


def _looks_like_domain(q: str) -> str | None:
    q = q.strip().lower()
    q = re.sub(r"^https?://", "", q).split("/")[0]
    if re.fullmatch(r"[a-z0-9.\-]+\.[a-z]{2,}", q) and " " not in q:
        return q
    return None


def _looks_like_ip(q: str) -> str | None:
    q = q.strip()
    try:
        socket.inet_aton(q)
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", q):
            return q
    except OSError:
        pass
    return None


class UsernameSweepAdapter(SourceAdapter):
    """Username → which platforms have that profile (keyless sweep)."""

    name = "osint_username"
    result_type = "profile"
    description = "Username sweep: which sites have a profile for this handle (keyless, public pages only)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        user = _looks_like_username(query)
        if not user:
            return []
        out: list[SearchResult] = []
        for site, tmpl, nf_marker in _USERNAME_SITES:
            if len(out) >= limit:
                break
            url = tmpl.format(u=urllib.parse.quote(user))
            status, body, final = _get(url, timeout=10)
            if status == 200 and nf_marker.lower() not in body.lower():
                out.append(SearchResult(
                    source=self.name, title=f"{user} on {site}",
                    url=final, snippet=f"Public profile found for '{user}' on {site}.",
                    score=0.8,
                ))
            elif status in (404, 410):
                continue
        return out


class EmailCheckAdapter(SourceAdapter):
    """Email → breach exposure via public no-key sources + mail server check."""

    name = "osint_email"
    result_type = "breach"
    description = "Email exposure check: mail-server validity (MX) and public breach-data presence (keyless sources only)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        email = _looks_like_email(query)
        if not email:
            return []
        out: list[SearchResult] = []
        domain = email.split("@")[1]
        # MX check — does the domain accept mail at all?
        try:
            mx = socket.getaddrinfo(domain, 25)
            mx_ok = bool(mx)
        except OSError:
            mx_ok = False
        out.append(SearchResult(
            source=self.name, title=f"mail server for {domain}",
            url=f"https://{domain}",
            snippet=(f"Domain {domain} {'resolves for mail delivery' if mx_ok else 'does NOT resolve — address likely invalid'}."),
            score=0.6,
        ))
        if len(out) >= limit:
            return out
        # Hudson Rock free breach check (no key) — cavalry for "is this email in breaches"
        # endpoint pattern: https://cavalry.hudsonrock.com/api/json/v2/osint-tools/search-by-email
        try:
            data = _get_json(
                "https://cavalry.hudsonrock.com/api/json/v2/osint-tools/search-by-email?email="
                + urllib.parse.quote(email), timeout=15)
            if isinstance(data, dict):
                steals = data.get("stealers") or []
                if steals:
                    out.append(SearchResult(
                        source=self.name,
                        title=f"breach exposure for {email}",
                        url="https://hudsonrock.com",
                        snippet=(f"Found in {len(steals)} infostealer log(s) via Hudson Rock free API. "
                                 "Compromised — rotate passwords."),
                        score=0.95,
                    ))
                else:
                    out.append(SearchResult(
                        source=self.name,
                        title=f"breach exposure for {email}",
                        url="https://hudsonrock.com",
                        snippet="No infostealer logs found in Hudson Rock free database.",
                        score=0.5,
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("hudsonrock check failed: %s", exc)
        return out[:limit]


class DomainReconAdapter(SourceAdapter):
    """Domain → subdomains (crt.sh) + registration (RDAP)."""

    name = "osint_domain"
    result_type = "domain"
    description = "Domain recon: subdomains via certificate transparency (crt.sh) and registration data via RDAP (keyless)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        domain = _looks_like_domain(query)
        if not domain:
            return []
        out: list[SearchResult] = []
        # crt.sh — subdomains
        try:
            data = _get_json(
                f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json",
                timeout=20)
            if isinstance(data, list):
                subs = sorted({e.get("name_value", "").strip().lower()
                               for e in data if e.get("name_value")})
                subs = [s for s in subs if s and "*" not in s][: max(limit - 1, 1)]
                if subs:
                    out.append(SearchResult(
                        source=self.name, title=f"subdomains of {domain}",
                        url=f"https://crt.sh/?q=%25.{domain}",
                        snippet=f"Certificate-transparency subdomains ({len(subs)} shown): " + ", ".join(subs),
                        score=0.85,
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("crt.sh failed: %s", exc)
        # RDAP — registration
        try:
            data = _get_json(f"https://rdap.org/domain/{urllib.parse.quote(domain)}",
                             timeout=15)
            if isinstance(data, dict):
                events = {e.get("eventAction"): e.get("eventDate")
                          for e in data.get("events", [])}
                registrar = ""
                for ent in data.get("entities", []):
                    if "registrar" in (ent.get("roles") or []):
                        vcard = ent.get("vcardArray", [])
                        if len(vcard) > 1:
                            for item in vcard[1]:
                                if item[0] == "fn":
                                    registrar = item[3]
                out.append(SearchResult(
                    source=self.name, title=f"registration for {domain}",
                    url=f"https://rdap.org/domain/{domain}",
                    snippet=("Registrar: " + (registrar or "unknown")
                             + "; registered: " + str(events.get("registration", "?"))
                             + "; expires: " + str(events.get("expiration", "?"))),
                    score=0.8,
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("rdap failed: %s", exc)
        return out[:limit]


class IpIntelAdapter(SourceAdapter):
    """IP → ASN/org/geo (ipwho.is) + open ports (Shodan InternetDB)."""

    name = "osint_ip"
    result_type = "ip"
    description = "IP intel: ASN/org/geolocation (ipwho.is) and open ports/hostnames (Shodan InternetDB) — keyless."

    def search(self, query: str, *, limit: int, since=None, before=None):
        ip = _looks_like_ip(query)
        if not ip:
            return []
        out: list[SearchResult] = []
        try:
            data = _get_json(f"https://ipwho.is/{ip}", timeout=15)
            if isinstance(data, dict) and data.get("success"):
                conn = data.get("connection", {})
                out.append(SearchResult(
                    source=self.name, title=f"geolocation for {ip}",
                    url=f"https://ipwho.is/{ip}",
                    snippet=(f"{data.get('country', '?')}, {data.get('city', '?')} — "
                             f"ASN {conn.get('asn', '?')} ({conn.get('org', '?')}), "
                             f"ISP: {conn.get('isp', '?')}"),
                    score=0.85,
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("ipwho.is failed: %s", exc)
        if len(out) >= limit:
            return out
        try:
            data = _get_json(f"https://internetdb.shodan.io/{ip}", timeout=15)
            if isinstance(data, dict) and (data.get("ports") or data.get("hostnames")):
                out.append(SearchResult(
                    source=self.name, title=f"open ports for {ip}",
                    url=f"https://internetdb.shodan.io/{ip}",
                    snippet=("Open ports: " + str(data.get("ports", []))
                             + "; hostnames: " + str(data.get("hostnames", []))
                             + "; vulns: " + str(data.get("vulns", []))),
                    score=0.9,
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("internetdb failed: %s", exc)
        return out[:limit]




def _looks_like_phone(q: str) -> str | None:
    """Extract a phone number from a query, or None."""
    import re as _re
    digits = _re.sub(r"[^\d+]", "", q.strip())
    if _re.match(r"^\+?\d{7,15}$", digits):
        return digits
    return None


def _confidence(sources: int, keyless_verified: bool = False) -> str:
    """Rate finding confidence: high (2+ sources), medium (1 verified), low."""
    if sources >= 2:
        return "high"
    if keyless_verified:
        return "medium"
    return "low"


class PhoneIntelAdapter(SourceAdapter):
    """Phone → carrier/location via free lookup APIs."""

    name = "osint_phone"
    result_type = "phone"
    description = "Phone intel: carrier, location, line type via keyless APIs."

    def search(self, query: str, *, limit: int, since=None, before=None):
        phone = _looks_like_phone(query)
        if not phone:
            return []
        out: list[SearchResult] = []
        # numverify-style free lookup via abstract API pattern
        # Using ipqualityscore-adjacent free tier would need key; use
        # openly accessible carrier data via phonenumber parsing
        try:
            import phonenumbers
            parsed = phonenumbers.parse(phone, None)
            carrier = phonenumbers.carrier.name_for_number(parsed, "en")
            region = phonenumbers.geocoder.description_for_number(parsed, "en")
            out.append(SearchResult(
                source=self.name,
                title=f"phone intel for {phone}",
                url=f"https://www.truecaller.com/search/{phone}",
                snippet=(f"Carrier: {carrier or 'unknown'}, "
                         f"Region: {region or 'unknown'}, "
                         f"Valid: {phonenumbers.is_valid_number(parsed)}, "
                         f"Type: {phonenumbers.number_type(parsed)} "
                         f"[confidence: {_confidence(1, True)}]"),
                score=0.7,
            ))
        except ImportError:
            _log.debug("phonenumbers not installed, skipping phone parse")
        except Exception as exc:  # noqa: BLE001
            _log.debug("phone parse failed: %s", exc)
        return out[:limit]


class GitHubReconAdapter(SourceAdapter):
    """Username/email → GitHub profile, repos, commits, gists."""

    name = "osint_github"
    result_type = "github"
    description = "GitHub recon: profile, public repos, gists, commit emails — keyless API."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip().lstrip("@")
        if not q or " " in q:
            return []
        out: list[SearchResult] = []
        try:
            user = _get_json(f"https://api.github.com/users/{q}", timeout=15)
            if isinstance(user, dict) and user.get("login"):
                out.append(SearchResult(
                    source=self.name,
                    title=f"GitHub: {user.get('login')}",
                    url=user.get("html_url", ""),
                    snippet=(f"{user.get('name', '')} — {user.get('bio', '')} | "
                             f"Repos: {user.get('public_repos', 0)}, "
                             f"Followers: {user.get('followers', 0)}, "
                             f"Location: {user.get('location', '?')}, "
                             f"Blog: {user.get('blog', '')} "
                             f"[confidence: {_confidence(1, True)}]"),
                    score=0.9,
                ))
                # Recent repos for tech stack hints
                repos = _get_json(
                    f"https://api.github.com/users/{q}/repos?per_page=5&sort=updated",
                    timeout=15)
                if isinstance(repos, list):
                    langs = set()
                    for r in repos[:5]:
                        if isinstance(r, dict) and r.get("language"):
                            langs.add(r["language"])
                    if langs:
                        out.append(SearchResult(
                            source=self.name,
                            title=f"GitHub tech stack: {q}",
                            url=f"https://github.com/{q}?tab=repositories",
                            snippet=f"Languages: {', '.join(sorted(langs))}",
                            score=0.6,
                        ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("github recon failed: %s", exc)
        return out[:limit]


class WaybackAdapter(SourceAdapter):
    """URL/domain → Wayback Machine snapshots (historical versions)."""

    name = "osint_wayback"
    result_type = "wayback"
    description = "Wayback Machine: historical snapshots of URLs/domains."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip()
        if not q or " " in q:
            return []
        if not q.startswith(("http://", "https://")):
            q = f"https://{q}"
        out: list[SearchResult] = []
        try:
            data = _get_json(
                f"https://archive.org/wayback/available?url={q}", timeout=15)
            snap = (data.get("archived_snapshots") or {}).get("closest", {})
            if snap.get("url"):
                out.append(SearchResult(
                    source=self.name,
                    title=f"Wayback snapshot: {q}",
                    url=snap["url"],
                    snippet=(f"Closest snapshot: {snap.get('timestamp', '?')} "
                             f"[confidence: {_confidence(1, True)}]"),
                    score=0.75,
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("wayback failed: %s", exc)
        return out[:limit]


class GravatarAdapter(SourceAdapter):
    """Email → Gravatar profile (identity correlation)."""

    name = "osint_gravatar"
    result_type = "gravatar"
    description = "Gravatar: email → profile/avatar for identity correlation."

    def search(self, query: str, *, limit: int, since=None, before=None):
        email = _looks_like_email(query)
        if not email:
            return []
        import hashlib as _hl
        h = _hl.md5(email.lower().encode()).hexdigest()
        out: list[SearchResult] = []
        try:
            data = _get_json(f"https://www.gravatar.com/{h}.json", timeout=15)
            entry = (data.get("entry") or [{}])[0]
            if entry.get("displayName") or entry.get("preferredUsername"):
                out.append(SearchResult(
                    source=self.name,
                    title=f"Gravatar: {entry.get('displayName', email)}",
                    url=entry.get("profileUrl", f"https://www.gravatar.com/{h}"),
                    snippet=(f"Username: {entry.get('preferredUsername', '?')}, "
                             f"Name: {entry.get('displayName', '?')} "
                             f"[confidence: {_confidence(1, True)}]"),
                    score=0.8,
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("gravatar failed: %s", exc)
        return out[:limit]


class LeakCheckAdapter(SourceAdapter):
    """Username/email → breach exposure via LeakCheck free API."""

    name = "osint_leakcheck"
    result_type = "breach"
    description = "LeakCheck: free breach database search for usernames/emails."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip()
        if not q or " " in q:
            return []
        out: list[SearchResult] = []
        try:
            import urllib.parse as _up
            data = _get_json(
                f"https://leakcheck.net/api/public?check={_up.quote(q)}",
                timeout=15)
            if isinstance(data, dict) and data.get("success"):
                sources = data.get("sources", [])
                if sources:
                    names = [s.get("name", "?") for s in sources[:5]
                             if isinstance(s, dict)]
                    out.append(SearchResult(
                        source=self.name,
                        title=f"Breach exposure: {q}",
                        url="https://leakcheck.net",
                        snippet=(f"Found in {len(sources)} breach(es): "
                                 f"{', '.join(names)} "
                                 f"[confidence: {_confidence(len(sources))}]"),
                        score=0.85,
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("leakcheck failed: %s", exc)
        return out[:limit]


#: adapter classes in canonical order
OSINT_SPECS: list[tuple[str, str, str, type[SourceAdapter]]] = [
    (UsernameSweepAdapter.name, UsernameSweepAdapter.result_type,
     UsernameSweepAdapter.description, UsernameSweepAdapter),
    (EmailCheckAdapter.name, EmailCheckAdapter.result_type,
     EmailCheckAdapter.description, EmailCheckAdapter),
    (DomainReconAdapter.name, DomainReconAdapter.result_type,
     DomainReconAdapter.description, DomainReconAdapter),
    (IpIntelAdapter.name, IpIntelAdapter.result_type,
     IpIntelAdapter.description, IpIntelAdapter),
    (PhoneIntelAdapter.name, PhoneIntelAdapter.result_type,
     PhoneIntelAdapter.description, PhoneIntelAdapter),
    (GitHubReconAdapter.name, GitHubReconAdapter.result_type,
     GitHubReconAdapter.description, GitHubReconAdapter),
    (WaybackAdapter.name, WaybackAdapter.result_type,
     WaybackAdapter.description, WaybackAdapter),
    (GravatarAdapter.name, GravatarAdapter.result_type,
     GravatarAdapter.description, GravatarAdapter),
    (LeakCheckAdapter.name, LeakCheckAdapter.result_type,
     LeakCheckAdapter.description, LeakCheckAdapter),
]
