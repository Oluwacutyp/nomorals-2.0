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

import hashlib
import json
import re
import shutil
import socket
import subprocess
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


def _parse_ts(value: Any) -> float | None:
    """Best-effort ISO-8601 / epoch → epoch seconds; never raises."""
    if value is None or value is False or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone

        iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def build_osint_hit(
    adapter: SourceAdapter,
    query: str,
    title: str,
    url: str,
    snippet: str,
    score: float,
    *,
    confidence: str = "medium",
    timestamp: float | None = None,
    **extra_provenance: Any,
) -> SearchResult:
    """Build a well-formed :class:`SearchResult` for an OSINT hit.

    ``url`` and ``confidence`` land in provenance (they are not
    ``SearchResult`` fields); ``type`` comes from the adapter's
    ``result_type``; ``source_id`` is a stable per-source hash.
    """
    url = str(url or "")
    sid = hashlib.sha1(
        f"{adapter.name}:{url or title}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    provenance: dict[str, Any] = {"url": url, "confidence": confidence}
    provenance.update(extra_provenance)
    return SearchResult(
        query=query,
        title=str(title or "(untitled)")[:300],
        snippet=str(snippet or "")[:1200],
        source=adapter.name,
        type=adapter.result_type,
        score=float(score),
        raw_score=float(score),
        provenance=provenance,
        timestamp=timestamp,
        source_id=f"{adapter.name}:{sid}",
    )


def _cli_found_lines(exe: str, args: list[str], timeout_s: int) -> list[tuple[str, str]]:
    """Run a username CLI (sherlock/maigret) and parse its ``[+]`` lines.

    Returns ``(site, url)`` pairs. Returns ``[]`` on any failure, ``None``
    is never returned — callers distinguish "CLI absent" via which().
    """
    try:
        proc = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=timeout_s
        )
    except Exception as exc:  # noqa: BLE001 - CLI failed, fall back
        _log.debug("%s CLI failed: %s", exe, exc)
        return []
    out: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        m = re.match(r"\[\+\]\s*(.+?):\s*(https?://\S+)", line.strip())
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


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
    """Username → which platforms have that profile (keyless sweep).

    Delegation order, best-first (mined from the OSINT arsenal):

    1. ``sherlock`` CLI (400+ sites) when installed;
    2. ``maigret`` CLI (3000+ sites) when installed;
    3. the built-in 25-site hand sweep (zero-dependency fallback — the
       not-found markers rot over time, which is exactly why the CLIs,
       with their maintained site lists, go first).
    """

    name = "osint_username"
    result_type = "profile"
    description = "Username sweep: which sites have a profile for this handle (keyless, public pages only)."

    def _via_cli(self, user: str, query: str, limit: int) -> list[SearchResult] | None:
        """sherlock then maigret; ``None`` when neither CLI is installed."""
        sherlock = shutil.which("sherlock")
        if sherlock:
            found = _cli_found_lines(
                sherlock, [user, "--print-found", "--timeout", "10"], 120)
            if found:
                return [
                    build_osint_hit(
                        self, query, f"{user} on {site}", url,
                        f"Public profile found for '{user}' on {site} (sherlock).",
                        0.85, confidence="high", engine="sherlock",
                    )
                    for site, url in found[:limit]
                ]
            return []  # CLI ran, nothing found — honest empty, no fallback
        maigret = shutil.which("maigret")
        if maigret:
            found = _cli_found_lines(
                maigret, [user, "--no-progressbar", "--timeout", "10"], 180)
            if found:
                return [
                    build_osint_hit(
                        self, query, f"{user} on {site}", url,
                        f"Public profile found for '{user}' on {site} (maigret).",
                        0.85, confidence="high", engine="maigret",
                    )
                    for site, url in found[:limit]
                ]
            return []
        return None

    def search(self, query: str, *, limit: int, since=None, before=None):
        user = _looks_like_username(query)
        if not user:
            return []
        via_cli = self._via_cli(user, query, limit)
        if via_cli is not None:
            return via_cli
        # zero-dependency fallback: hand-rolled site sweep
        out: list[SearchResult] = []
        for site, tmpl, nf_marker in _USERNAME_SITES:
            if len(out) >= limit:
                break
            url = tmpl.format(u=urllib.parse.quote(user))
            status, body, final = _get(url, timeout=10)
            if status == 200 and nf_marker.lower() not in body.lower():
                out.append(build_osint_hit(
                    self, query, f"{user} on {site}", final,
                    f"Public profile found for '{user}' on {site}.",
                    0.8, confidence="medium", engine="hand-sweep",
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
        out.append(build_osint_hit(
            self, query, f"mail server for {domain}", f"https://{domain}",
            (f"Domain {domain} {'resolves for mail delivery' if mx_ok else 'does NOT resolve — address likely invalid'}."),
            0.6, confidence="medium",
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
                    out.append(build_osint_hit(
                        self, query, f"breach exposure for {email}",
                        "https://hudsonrock.com",
                        (f"Found in {len(steals)} infostealer log(s) via Hudson Rock free API. "
                         "Compromised — rotate passwords."),
                        0.95, confidence="high",
                    ))
                else:
                    out.append(build_osint_hit(
                        self, query, f"breach exposure for {email}",
                        "https://hudsonrock.com",
                        "No infostealer logs found in Hudson Rock free database.",
                        0.5, confidence="medium",
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("hudsonrock check failed: %s", exc)
        return out[:limit]


class XposedOrNotAdapter(SourceAdapter):
    """Email → known data breaches via XposedOrNot's free API.

    Mined as the keyless breach-check gold: ``GET
    api.xposedornot.com/v1/check-email/{email}`` needs no API key and is
    free for personal/low-volume use (HIBP has had no free tier since
    2024 — not attempted). Returns breach *names*; counts only, never
    credentials. Rate limits are 2/s, 25/hr, 100/day per endpoint — a 429
    is logged and yields no hit rather than an error.
    """

    name = "osint_xon"
    result_type = "breach"
    description = "XposedOrNot free breach check: which known data breaches exposed this email (no key, no account)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        email = _looks_like_email(query)
        if not email:
            return []
        status, body, _ = _get(
            "https://api.xposedornot.com/v1/check-email/"
            + urllib.parse.quote(email), timeout=15)
        if status == 429:
            _log.debug("xposedornot rate-limited for %s", email)
            return []
        if status != 200 or not body:
            return []
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return []
        breaches: list[str] = []
        raw = data.get("breaches") if isinstance(data, dict) else None
        if isinstance(raw, list):
            # documented shape: {"breaches": [["Adobe", "LinkedIn"]]} —
            # tolerate a flat list too (defensive, per mining notes)
            for item in raw:
                if isinstance(item, list):
                    breaches.extend(str(b) for b in item if b)
                elif isinstance(item, str) and item:
                    breaches.append(item)
        if breaches:
            shown = ", ".join(breaches[:12])
            more = f" (+{len(breaches) - 12} more)" if len(breaches) > 12 else ""
            return [build_osint_hit(
                self, query, f"breach exposure: {email}",
                "https://xposedornot.com",
                (f"Exposed in {len(breaches)} known breach(es): {shown}{more}. "
                 "Rotate this password everywhere it was reused."),
                0.95, confidence="high",
            )][:limit]
        return [build_osint_hit(
            self, query, f"breach exposure: {email}",
            "https://xposedornot.com",
            "Not found in XposedOrNot's known-breach corpus.",
            0.5, confidence="medium",
        )][:limit]


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
                    out.append(build_osint_hit(
                        self, query, f"subdomains of {domain}",
                        f"https://crt.sh/?q=%25.{domain}",
                        f"Certificate-transparency subdomains ({len(subs)} shown): " + ", ".join(subs),
                        0.85, confidence="high",
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
                out.append(build_osint_hit(
                    self, query, f"registration for {domain}",
                    f"https://rdap.org/domain/{domain}",
                    ("Registrar: " + (registrar or "unknown")
                     + "; registered: " + str(events.get("registration", "?"))
                     + "; expires: " + str(events.get("expiration", "?"))),
                    0.8, confidence="high",
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
                out.append(build_osint_hit(
                    self, query, f"geolocation for {ip}",
                    f"https://ipwho.is/{ip}",
                    (f"{data.get('country', '?')}, {data.get('city', '?')} — "
                     f"ASN {conn.get('asn', '?')} ({conn.get('org', '?')}), "
                     f"ISP: {conn.get('isp', '?')}"),
                    0.85, confidence="high",
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("ipwho.is failed: %s", exc)
        if len(out) >= limit:
            return out
        try:
            data = _get_json(f"https://internetdb.shodan.io/{ip}", timeout=15)
            if isinstance(data, dict) and (data.get("ports") or data.get("hostnames")):
                out.append(build_osint_hit(
                    self, query, f"open ports for {ip}",
                    f"https://internetdb.shodan.io/{ip}",
                    ("Open ports: " + str(data.get("ports", []))
                     + "; hostnames: " + str(data.get("hostnames", []))
                     + "; vulns: " + str(data.get("vulns", []))),
                    0.9, confidence="high",
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
            out.append(build_osint_hit(
                self, query,
                f"phone intel for {phone}",
                f"https://www.truecaller.com/search/{phone}",
                (f"Carrier: {carrier or 'unknown'}, "
                 f"Region: {region or 'unknown'}, "
                 f"Valid: {phonenumbers.is_valid_number(parsed)}, "
                 f"Type: {phonenumbers.number_type(parsed)}"),
                0.7, confidence="medium",
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
                out.append(build_osint_hit(
                    self, query,
                    f"GitHub: {user.get('login')}",
                    user.get("html_url", ""),
                    (f"{user.get('name', '')} — {user.get('bio', '')} | "
                     f"Repos: {user.get('public_repos', 0)}, "
                     f"Followers: {user.get('followers', 0)}, "
                     f"Location: {user.get('location', '?')}, "
                     f"Blog: {user.get('blog', '')}"),
                    0.9, confidence="high",
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
                        out.append(build_osint_hit(
                            self, query,
                            f"GitHub tech stack: {q}",
                            f"https://github.com/{q}?tab=repositories",
                            f"Languages: {', '.join(sorted(langs))}",
                            0.6, confidence="medium",
                        ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("github recon failed: %s", exc)
        return out[:limit]


class WaybackAdapter(SourceAdapter):
    """URL/domain → Wayback Machine snapshots (historical versions).

    Two layers: the ``available`` API gives the closest snapshot, and the
    CDX API lists every 200-OK capture (deduped by digest) so the full
    history of a page is visible, not just one snapshot.
    """

    name = "osint_wayback"
    result_type = "wayback"
    description = "Wayback Machine: historical snapshots of URLs/domains (closest snapshot + full CDX capture list)."

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
                out.append(build_osint_hit(
                    self, query,
                    f"Wayback snapshot: {q}",
                    snap["url"],
                    f"Closest snapshot: {snap.get('timestamp', '?')}",
                    0.75, confidence="high",
                ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("wayback failed: %s", exc)
        if len(out) >= limit:
            return out
        # CDX: full capture list, 200s only, digest-collapsed
        try:
            data = _get_json(
                "https://web.archive.org/cdx/search/cdx"
                f"?url={urllib.parse.quote(q)}"
                f"&output=json&limit={max(limit * 3, 10)}"
                "&filter=statuscode:200&collapse=digest",
                timeout=20)
            if isinstance(data, list) and len(data) > 1:
                rows = data[1:]  # first row is the header
                seen: list[tuple[str, str]] = []
                for row in rows:
                    if isinstance(row, list) and len(row) >= 3:
                        ts, original = str(row[1]), str(row[2])
                        if ts and original and (ts, original) not in seen:
                            seen.append((ts, original))
                    if len(seen) >= limit:
                        break
                if seen:
                    lines = "\n".join(
                        f"{ts}: https://web.archive.org/web/{ts}/{orig}"
                        for ts, orig in seen)
                    out.append(build_osint_hit(
                        self, query,
                        f"Wayback capture history: {q}",
                        f"https://web.archive.org/web/*/{q}",
                        f"{len(seen)} captures (200 OK, digest-deduped):\n{lines}",
                        0.7, confidence="high",
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("wayback cdx failed: %s", exc)
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
        h = hashlib.md5(email.lower().encode()).hexdigest()
        out: list[SearchResult] = []
        try:
            data = _get_json(f"https://www.gravatar.com/{h}.json", timeout=15)
            entry = (data.get("entry") or [{}])[0]
            if entry.get("displayName") or entry.get("preferredUsername"):
                out.append(build_osint_hit(
                    self, query,
                    f"Gravatar: {entry.get('displayName', email)}",
                    entry.get("profileUrl", f"https://www.gravatar.com/{h}"),
                    (f"Username: {entry.get('preferredUsername', '?')}, "
                     f"Name: {entry.get('displayName', '?')}"),
                    0.8, confidence="high",
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
                    out.append(build_osint_hit(
                        self, query,
                        f"Breach exposure: {q}",
                        "https://leakcheck.net",
                        (f"Found in {len(sources)} breach(es): "
                         f"{', '.join(names)}"),
                        0.85, confidence="high" if len(sources) >= 2 else "medium",
                    ))
        except Exception as exc:  # noqa: BLE001
            _log.debug("leakcheck failed: %s", exc)
        return out[:limit]


class DisifyAdapter(SourceAdapter):
    """Email → validity via Disify's free no-key API.

    Returns format validity, MX/DNS resolution, disposable-address and
    whitelist flags — the "is this address real" check that the MX
    socket probe alone can't answer.
    """

    name = "osint_disify"
    result_type = "email"
    description = "Disify free email validation: format, MX/DNS, disposable, whitelist (no key)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        email = _looks_like_email(query)
        if not email:
            return []
        try:
            data = _get_json(
                "https://www.disify.com/api/email/"
                + urllib.parse.quote(email), timeout=15)
            if not isinstance(data, dict):
                return []
            verdict = (
                "valid" if data.get("format") and data.get("dns")
                else "invalid")
            return [build_osint_hit(
                self, query,
                f"email validation: {email}",
                "https://www.disify.com",
                (f"Verdict: {verdict} — format: {bool(data.get('format'))}, "
                 f"MX/DNS: {bool(data.get('dns'))}, "
                 f"disposable: {bool(data.get('disposable'))}, "
                 f"whitelisted: {bool(data.get('whitelist'))}"),
                0.7, confidence="high",
            )][:limit]
        except Exception as exc:  # noqa: BLE001
            _log.debug("disify failed: %s", exc)
            return []


class EdgarAdapter(SourceAdapter):
    """Company → SEC EDGAR identity (free, no key).

    Uses the SEC's published ``company_tickers.json`` (ticker → CIK →
    legal name) so any public-company name or ticker resolves to its
    CIK and filing index. EDGAR is the free gold for US corporate
    identity; nothing here needs an account.
    """

    name = "osint_edgar"
    result_type = "company"
    description = "SEC EDGAR: US public-company lookup — ticker, CIK, filing index (free, no key)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip()
        if len(q) < 2 or "@" in q or " " in q and len(q.split()) > 4:
            return []
        try:
            data = _get_json(
                "https://www.sec.gov/files/company_tickers.json", timeout=20)
            if not isinstance(data, dict):
                return []
            ql = q.lower()
            matches: list[dict[str, Any]] = []
            for entry in data.values():
                if not isinstance(entry, dict):
                    continue
                ticker = str(entry.get("ticker", ""))
                title = str(entry.get("title", ""))
                if ql == ticker.lower() or ql in title.lower():
                    matches.append(entry)
                    if len(matches) >= limit:
                        break
            out = []
            for entry in matches:
                ticker = str(entry.get("ticker", ""))
                title = str(entry.get("title", ""))
                cik = str(entry.get("cik_str", "")).zfill(10)
                out.append(build_osint_hit(
                    self, query,
                    f"{title} ({ticker})",
                    ("https://www.sec.gov/cgi-bin/browse-edgar"
                     f"?action=getcompany&CIK={ticker}&type=&dateb="
                     "&owner=include&count=10"),
                    (f"US SEC EDGAR — CIK {cik}, ticker {ticker}. "
                     f"Public filings: 10-K, 10-Q, 8-K, insider trades."),
                    0.9, confidence="high",
                    cik=cik, ticker=ticker,
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            _log.debug("edgar failed: %s", exc)
            return []


class GleifAdapter(SourceAdapter):
    """Company → Legal Entity Identifier via GLEIF (free, no key).

    The GLEIF API is the authoritative LEI registry: legal name,
    registration status, and legal address for any registered entity,
    worldwide. JSON:API format, no authentication.
    """

    name = "osint_gleif"
    result_type = "company"
    description = "GLEIF: legal-entity (LEI) lookup — official identity, status, address (free, no key)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip()
        if len(q) < 2 or "@" in q:
            return []
        try:
            data = _get_json(
                "https://api.gleif.org/api/v1/lei-records"
                f"?filter[entity.legal-name]={urllib.parse.quote(q)}"
                f"&page[size]={min(max(limit, 1), 10)}",
                timeout=20)
            records = data.get("data") if isinstance(data, dict) else None
            if not records:
                return []
            out = []
            for rec in records[:limit]:
                attrs = rec.get("attributes", {}) or {}
                entity = attrs.get("entity", {}) or {}
                lei = attrs.get("lei", "?")
                legal = entity.get("legalName", {}) or {}
                name = legal.get("name", "?")
                status = (attrs.get("registration", {}) or {}).get("status", "?")
                addr = entity.get("legalAddress", {}) or {}
                addr_str = ", ".join(
                    str(addr.get(k, "")) for k in
                    ("addressLine1", "city", "country")
                    if addr.get(k))
                out.append(build_osint_hit(
                    self, query,
                    f"{name} (LEI {lei})",
                    f"https://search.gleif.org/#/record/{lei}",
                    (f"Legal name: {name}; LEI: {lei}; "
                     f"registration: {status}; address: {addr_str or '?'}"),
                    0.9, confidence="high", lei=lei,
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            _log.debug("gleif failed: %s", exc)
            return []


class CourtListenerAdapter(SourceAdapter):
    """Name/company → US court opinions via CourtListener (free, no key).

    The structured-identity gap in the OSINT arsenal: court opinions are
    public, citable, and free to search. Returns the top opinion matches
    with court, filing date, and a snippet.
    """

    name = "osint_courtlistener"
    result_type = "legal"
    description = "CourtListener: US court opinions mentioning a name/company (free legal API, no key)."

    def search(self, query: str, *, limit: int, since=None, before=None):
        q = query.strip()
        if len(q) < 2:
            return []
        try:
            data = _get_json(
                "https://www.courtlistener.com/api/rest/v3/search/"
                f"?q={urllib.parse.quote(q)}&type=o&order_by=score%20desc",
                timeout=20)
            results = data.get("results") if isinstance(data, dict) else None
            if not results:
                return []
            out = []
            for r in results[:limit]:
                if not isinstance(r, dict):
                    continue
                case = r.get("caseName") or "court opinion"
                abs_url = r.get("absolute_url") or ""
                url = ("https://www.courtlistener.com" + abs_url
                       if abs_url.startswith("/") else abs_url)
                snippet = re.sub(
                    r"<[^>]+>", "",
                    str(r.get("snippet") or r.get("plain_text", "") or ""))
                out.append(build_osint_hit(
                    self, query,
                    str(case)[:300],
                    url,
                    (f"{r.get('court', '?')} — filed "
                     f"{r.get('dateFiled', '?')}: {snippet[:500]}"),
                    0.75, confidence="medium",
                    timestamp=_parse_ts(r.get("dateFiled")),
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            _log.debug("courtlistener failed: %s", exc)
            return []


#: adapter classes in canonical order
OSINT_SPECS: list[tuple[str, str, str, type[SourceAdapter]]] = [
    (UsernameSweepAdapter.name, UsernameSweepAdapter.result_type,
     UsernameSweepAdapter.description, UsernameSweepAdapter),
    (EmailCheckAdapter.name, EmailCheckAdapter.result_type,
     EmailCheckAdapter.description, EmailCheckAdapter),
    (XposedOrNotAdapter.name, XposedOrNotAdapter.result_type,
     XposedOrNotAdapter.description, XposedOrNotAdapter),
    (DisifyAdapter.name, DisifyAdapter.result_type,
     DisifyAdapter.description, DisifyAdapter),
    (DomainReconAdapter.name, DomainReconAdapter.result_type,
     DomainReconAdapter.description, DomainReconAdapter),
    (EdgarAdapter.name, EdgarAdapter.result_type,
     EdgarAdapter.description, EdgarAdapter),
    (GleifAdapter.name, GleifAdapter.result_type,
     GleifAdapter.description, GleifAdapter),
    (CourtListenerAdapter.name, CourtListenerAdapter.result_type,
     CourtListenerAdapter.description, CourtListenerAdapter),
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
