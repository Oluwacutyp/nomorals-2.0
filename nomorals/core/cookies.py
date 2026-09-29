"""Cookie analysis & handling — the combined CookieLab (wave 76).

The universal decoder can *parse* a Cookie / Set-Cookie header.  The
CookieLab goes further: it structures every cookie (value + all flags),
classifies it (session / auth / csrf / tracking / preference / jwt /
encoded / unknown), fingerprints the platforms and services the cookie
set implies, decodes opaque values (URL-encoding, base64, JWT, hex,
gzip-compressed session blobs), and can ingest the findings into the
knowledge graph.

This is the one class that combines what used to be scattered:

* ``parse``      — Cookie / Set-Cookie headers (multi-line, multi-cookie)
* ``classify``   — what kind of cookie it is, and which service made it
* ``decode_value`` — try to read an opaque value
* ``fingerprint`` — platforms/services implied by the whole set
* ``report``     — the full structured analysis (what ``nm cookies`` shows)
* ``ingest``     — entities → knowledge graph + decode report archive

Everything is deterministic and offline-safe; the model is never needed.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["CookieLab", "parse_cookies", "cookie_report"]

# ── known-cookie fingerprints: name → (service, note) ───────────────────────
# Ordered: first match wins.  Keys are matched case-insensitively.
_SERVICE_TABLE: tuple[tuple[str, str, str], ...] = (
    ("PHPSESSID", "php", "PHP built-in session"),
    ("JSESSIONID", "java-servlet", "JVM servlet container session"),
    ("ASP.NET_SessionId", "asp.net", "ASP.NET session"),
    ("ASP.NET_SessionState", "asp.net", "ASP.NET session state"),
    ("__cf_bm", "cloudflare", "Cloudflare bot-management cookie"),
    ("cf_clearance", "cloudflare", "Cloudflare clearance (challenge passed)"),
    ("__cf_chl*", "cloudflare", "Cloudflare challenge cookie"),
    ("AWSALB", "aws-elastic-load-balancer", "AWS ALB sticky-session cookie"),
    ("AWSALBCORS", "aws-elastic-load-balancer", "AWS ALB CORS cookie"),
    ("BIGipServer", "f5-big-ip", "F5 BIG-IP persistence cookie"),
    ("_ga", "google-analytics", "Google Analytics visitor id"),
    ("_gid", "google-analytics", "Google Analytics session id"),
    ("_gat", "google-analytics", "Google Analytics throttle"),
    ("_fbp", "facebook", "Facebook pixel browser id"),
    ("fr", "facebook", "Facebook ad cookie"),
    ("__gads", "google-ads", "Google Ads user data"),
    ("__gpi", "google-ads", "Google Ads user id"),
    ("__hbcfg", "hubspot", "HubSpot tracking config"),
    ("_ttwid", "tiktok", "TikTok web id"),
    ("_ym_uid", "yandex-metrica", "Yandex Metrica user id"),
    ("_ym_d", "yandex-metrica", "Yandex Metrica daily id"),
    ("_clck", "clicky", "Clicky analytics"),
    ("_clsk", "clicky", "Clicky session"),
    ("_pin_unauth", "pinterest", "Pinterest unauthenticated id"),
    ("_pin_session", "pinterest", "Pinterest session"),
    ("__datadog", "datadog", "DataDog browser rum"),
    ("ak_bmsc", "akamai-bot-manager", "Akamai bot manager"),
    ("bm_sz", "akamai-bot-manager", "Akamai bot manager signature"),
    ("bm_sv", "akamai-bot-manager", "Akamai bot manager signature"),
    ("visid_incap_*", "imperva-incapsula", "Imperva bot detection"),
    ("incap_ses_*", "imperva-incapsula", "Imperva session"),
    ("__stripe_mid", "stripe", "Stripe merchant id"),
    ("__stripe_sid", "stripe", "Stripe session id"),
    ("connect.sid", "nodejs-socketio", "Node.js socket.io session"),
    ("express:sess", "nodejs-express", "Express session-store cookie"),
    ("_session_id", "ruby-rails", "Ruby on Rails session"),
    ("rack.session", "ruby-rack", "Rack session"),
    ("laravel_session", "php-laravel", "Laravel session"),
    ("XSRF-TOKEN", "laravel-csrf", "Laravel CSRF token"),
    ("_csrf", "generic-csrf", "CSRF token"),
    ("csrftoken", "generic-csrf", "CSRF token"),
    ("__RequestVerificationToken", "asp.net-csrf", "ASP.NET anti-forgery token"),
    ("__waf", "waf", "Web-application-firewall cookie"),
    ("waf_id", "waf", "Web-application-firewall cookie"),
    ("_waf_cookie", "waf", "Web-application-firewall cookie"),
    ("SSO_LANGID", "sso", "Enterprise SSO language"),
    ("ssoSessionId", "sso", "Enterprise SSO session"),
    ("SPID", "azure-ad", "Azure AD session"),
    ("WLSSPI", "azure-ad", "Azure AD session"),
    ("FedAuth", "saml-sso", "SAML federation session"),
    ("APJC*", "sso", "SSO federation cookie"),
    ("_next_session", "nextjs", "Next.js auth session"),
    ("__session", "nextjs", "Next.js session"),
)

#: cookie name → classification (checked before value-based rules)
_CSRF_NAMES = {"_csrf", "csrftoken", "xsrf-token",
               "__requestverificationtoken", "csrf", "_xsrf",
               "x-csrf-token", "csrf_token"}
_TRACKING_NAMES = {"_ga", "_gid", "_gat", "_fbp", "fr", "__gads", "__gpi",
                   "__hbcfg", "_ym_uid", "_ym_d", "_clck", "_clsk",
                   "__datadog", "_tt_anonymous_id"}
_PREF_NAMES = {"theme", "lang", "locale", "language", "currency", "tz",
               "timezone", "pref", "preferences", "ui", "dark_mode",
               "color_scheme"}
_SESSION_HINT = re.compile(
    r"(session|ssid|_sid$|^sid$|sess|connect\.sid|jar)", re.I)
_AUTH_HINT = re.compile(
    r"(auth|token|_jwt|access|credential|identity|principal)", re.I)

_FLAG_KEYS = {"path", "domain", "expires", "max-age", "secure", "httponly",
              "samesite", "partitioned", "priority", "version", "comment"}


@dataclass
class Cookie:
    """One parsed cookie, structured."""

    name: str
    value: str
    flags: dict[str, str] = field(default_factory=dict)
    kind: str = ""            # session|auth|csrf|tracking|preference|jwt|encoded|unknown
    service: str = ""         # fingerprinted platform/service ("" = none)
    decoded_value: Any = None  # set when decode_value() read the value
    decode_via: str = ""       # url|base64|base64-json|jwt|hex|gzip|""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "value": self.value,
                               "kind": self.kind, "service": self.service}
        if self.flags:
            out["flags"] = dict(self.flags)
        if self.decoded_value is not None:
            out["decoded_value"] = self.decoded_value
            out["decode_via"] = self.decode_via
        return out


# ─────────────────────────────────────────────────────────────────────────────

def _split_cookie_part(part: str) -> tuple[str, str] | None:
    """Split one ``name=value`` cookie part (values may contain '=')."""
    if "=" not in part:
        return None
    name, _, value = part.partition("=")
    name = name.strip()
    if not name or name.lower() in _FLAG_KEYS:
        return None
    return name, value.strip()


def _parse_flags(rest: str) -> dict[str, str]:
    """Parse the attribute tail of a Set-Cookie value into a dict."""
    flags: dict[str, str] = {}
    for part in rest.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            flags[k.strip().lower()] = v.strip()
        else:
            flags[part.lower()] = ""
    return flags


def _service_for(name: str) -> tuple[str, str]:
    """Fingerprint one cookie name → (service, note)."""
    low = (name or "").lower()
    for key, service, note in _SERVICE_TABLE:
        if key.endswith("*"):
            if low.startswith(key[:-1]):
                return service, note
        elif low == key.lower():
            return service, note
        elif low.startswith(key.lower() + "_") or \
                low.startswith(key.lower() + "-"):
            return service, note
    return "", ""


def _classify(name: str, value: str) -> str:
    """Classify one cookie: session|auth|csrf|tracking|preference|jwt|
    encoded|unknown."""
    low = (name or "").lower()
    if low in _CSRF_NAMES or "csrf" in low or "xsrf" in low:
        return "csrf"
    if low in _TRACKING_NAMES:
        return "tracking"
    if low in _PREF_NAMES or low.endswith(("_pref", "_prefs", "_theme")):
        return "preference"
    if (value or "").strip().startswith("eyJ"):
        return "jwt"
    if _SESSION_HINT.search(name):
        return "session"
    if _AUTH_HINT.search(name):
        return "auth"
    # an opaque base64-ish value on an unclassified cookie
    v = (value or "").strip()
    if len(v) >= 16 and re.fullmatch(r"[A-Za-z0-9+/=_\-]{16,}", v):
        return "encoded"
    return "unknown"


def _try_jwt(value: str) -> dict[str, Any] | None:
    parts = (value or "").strip().split(".")
    if len(parts) < 2:
        return None
    out: dict[str, Any] = {"header": None, "payload": None, "signed": len(parts) >= 3}
    for i, key in ((0, "header"), (1, "payload")):
        try:
            seg = parts[i]
            seg += "=" * (-len(seg) % 4)
            data = base64.urlsafe_b64decode(seg.encode())
            out[key] = json.loads(data.decode("utf-8", "ignore"))
        except Exception:  # noqa: BLE001
            return None
    return out


def _try_base64(value: str) -> tuple[Any, str] | None:
    v = (value or "").strip()
    if len(v) < 12 or len(v) % 4 in (1,):
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=_\-]{12,}", v):
        return None
    padded = v + "=" * (-len(v) % 4)
    for alphabet in (lambda s: s, lambda s: s.replace("-", "+").replace("_", "/")):
        try:
            raw = base64.b64decode(alphabet(padded), validate=True)
        except (binascii.Error, ValueError):
            continue
        if not raw:
            continue
        text = raw.decode("utf-8", "ignore")
        printable = sum(1 for c in text if c.isprintable() or c in "\n\t") / max(1, len(text))
        if printable >= 0.90:
            # JSON?
            try:
                return json.loads(text), "base64-json"
            except (ValueError, TypeError):
                return text, "base64"
        # not text — but maybe it's a compressed session blob
        return raw, "base64"
    return None


def _try_gzip_bytes(raw: bytes) -> bytes | None:
    if raw[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(raw)
        except (OSError, zlib.error):
            return None
    if raw[:2] in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return None
    return None


class CookieLab:
    """The combined cookie analysis & handling system."""

    # ── parse ──────────────────────────────────────────────────────────────
    def parse(self, text: str) -> list[Cookie]:
        """Parse a Cookie / Set-Cookie header (or several) into structured
        :class:`Cookie` objects with flags, kind, and service fingerprint.
        """
        out: list[Cookie] = []
        if not (text or "").strip():
            return out
        # normalize: strip header prefixes, keep cookie parts + flags
        lines = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r"^(?:set-)?cookie\s*:\s*(.+)$", line, re.I)
            if m:
                line = m.group(1).strip()
            # also handle bare "Name=value; Name2=value2" lines
            lines.append(line)
        joined = "\n".join(lines)
        # Split into individual Set-Cookie lines: a new cookie starts after
        # a flag-only segment or a newline.  We process line by line and,
        # within a line, split on ';' but keep the flag tail per cookie.
        for line in lines:
            # If the line is a single cookie with flags, split name=value
            # off the front, then the rest is flags.
            first_cookie: list[Cookie] = []
            remainder = line
            if "=" in remainder:
                name, _, rest = remainder.partition("=")
                name = name.strip()
                if name and name.lower() not in _FLAG_KEYS:
                    value, *flag_parts = rest.split(";", 1)
                    value = value.strip()
                    c = Cookie(name=name, value=value)
                    if flag_parts:
                        c.flags = _parse_flags(flag_parts[0])
                    first_cookie.append(c)
                    remainder = ""
            # Any extra "k=v" separated by ';' that are NOT flags are
            # additional cookies (rare in one line, but handle it).
            for piece in (line.split(";") if not first_cookie else []):
                piece = piece.strip()
                parsed = _split_cookie_part(piece)
                if parsed:
                    n, v = parsed
                    c = Cookie(name=n, value=v)
                    first_cookie.append(c)
            for c in first_cookie:
                c.kind = _classify(c.name, c.value)
                svc, _note = _service_for(c.name)
                c.service = svc
                out.append(c)
        return out

    # ── classify / fingerprint ──────────────────────────────────────────────
    def fingerprint(self, cookies: list[Cookie] | None = None,
                    text: str = "") -> list[str]:
        """The platforms/services implied by a cookie set (deduped,
        order-preserving)."""
        if cookies is None:
            cookies = self.parse(text)
        seen: list[str] = []
        for c in cookies:
            if c.service and c.service not in seen:
                seen.append(c.service)
        return seen

    # ── decode value ────────────────────────────────────────────────────────
    def decode_value(self, cookie: Cookie) -> Cookie:
        """Try to read an opaque cookie value in place (mutates + returns).
        Order: JWT → URL-unquote → base64(/JSON) → gzip blob → hex."""
        v = (cookie.value or "").strip()
        if not v:
            return cookie
        # JWT
        if v.startswith("eyJ"):
            jwt = _try_jwt(v)
            if jwt is not None:
                cookie.decoded_value = jwt
                cookie.decode_via = "jwt"
                cookie.kind = "jwt"
                return cookie
        # URL-encoding
        if "%" in v and re.search(r"%[0-9a-fA-F]{2}", v):
            from urllib.parse import unquote
            unq = unquote(v)
            if unq != v:
                # JSON after unquote?
                try:
                    cookie.decoded_value = json.loads(unq)
                    cookie.decode_via = "url-json"
                except (ValueError, TypeError):
                    cookie.decoded_value = unq
                    cookie.decode_via = "url"
                return cookie
        # base64 (plain or url-safe), then JSON, then gzip
        b64 = _try_base64(v)
        if b64 is not None:
            payload, via = b64
            if via == "base64" and isinstance(payload, (bytes, bytearray)):
                unz = _try_gzip_bytes(bytes(payload))
                if unz is not None:
                    text = unz.decode("utf-8", "ignore")
                    try:
                        cookie.decoded_value = json.loads(text)
                        cookie.decode_via = "gzip-json"
                    except (ValueError, TypeError):
                        cookie.decoded_value = text
                        cookie.decode_via = "gzip"
                else:
                    cookie.decoded_value = bytes(payload)
                    cookie.decode_via = via
            else:
                cookie.decoded_value = payload
                cookie.decode_via = via
            if cookie.kind == "unknown":
                cookie.kind = "encoded"
            return cookie
        # hex
        if re.fullmatch(r"[0-9a-fA-F]{8,}", v.replace(" ", "")) and \
                len(v.replace(" ", "")) % 2 == 0:
            try:
                raw = binascii.unhexlify(v.replace(" ", ""))
                text = raw.decode("utf-8", "ignore")
                printable = sum(1 for c in text
                                if c.isprintable() or c in "\n\t") / max(1, len(text))
                if printable >= 0.9:
                    try:
                        cookie.decoded_value = json.loads(text)
                        cookie.decode_via = "hex-json"
                    except (ValueError, TypeError):
                        cookie.decoded_value = text
                        cookie.decode_via = "hex"
                    if cookie.kind == "unknown":
                        cookie.kind = "encoded"
            except (binascii.Error, ValueError):
                pass
        return cookie

    # ── report ──────────────────────────────────────────────────────────────
    def report(self, text: str, *, decode: bool = True) -> dict[str, Any]:
        """The full structured cookie analysis for a header / cookie set.

        Returns ``{"cookies": [...], "count": n, "services": [...],
        "kinds": {...}, "flags": {...}, "security": {...}}``.
        """
        cookies = self.parse(text)
        if decode:
            for c in cookies:
                self.decode_value(c)
        kinds: dict[str, int] = {}
        for c in cookies:
            kinds[c.kind or "unknown"] = kinds.get(c.kind or "unknown", 0) + 1
        flags: dict[str, int] = {}
        for c in cookies:
            for f in c.flags:
                flags[f] = flags.get(f, 0) + 1
        security = {
            "with_httponly": sum(1 for c in cookies
                                 if "httponly" in c.flags),
            "with_secure": sum(1 for c in cookies
                               if "secure" in c.flags),
            "samesite": {c.flags.get("samesite", "(none)")
                         for c in cookies if c.flags.get("samesite")},
            # present (even as a bare flag: HttpOnly sets value "")
            "plaintext_auth": [c.name for c in cookies
                               if c.kind in {"auth", "jwt", "session"}
                               and "httponly" not in c.flags
                               and "secure" not in c.flags],
        }
        return {
            "cookies": [c.to_dict() for c in cookies],
            "count": len(cookies),
            "services": self.fingerprint(cookies),
            "kinds": kinds,
            "flags": flags,
            "security": security,
            "analyzed_at": time.time(),
        }

    # ── ingest ──────────────────────────────────────────────────────────────
    def ingest(self, context: Any, text: str, *, source: str = "cookies") -> dict[str, Any]:
        """Feed the cookie findings into the knowledge graph and archive
        the report.  Best-effort — returns what it did, never raises."""
        rep = self.report(text)
        ingested: dict[str, Any] = {"services": rep["services"],
                                    "nodes": 0, "report_id": ""}
        try:
            from ..agents.kg import KnowledgeGraph
            graph = KnowledgeGraph(context.db)
            source_node = graph.upsert_node(source, type="entity",
                                            properties={"kind": "cookie-source"})
            for svc in rep["services"]:
                node = graph.upsert_node(svc, type="entity",
                                         properties={"kind": "service"})
                graph.link(source_node.id, node.id, "uses_service")
                ingested["nodes"] += 1
            # named auth/session cookies become nodes (the actual identities)
            for c in rep["cookies"]:
                if c.get("kind") in {"session", "auth", "jwt"}:
                    node = graph.upsert_node(
                        c["name"], type="entity",
                        properties={"kind": c["kind"],
                                    "service": c.get("service", "")})
                    graph.link(source_node.id, node.id, "carries")
                    ingested["nodes"] += 1
                    # JWT claims → person/claim nodes
                    dv = c.get("decoded_value")
                    if isinstance(dv, dict):
                        payload = dv.get("payload") if isinstance(dv.get("payload"), dict) else dv
                        for claim_key in ("sub", "email", "name", "user", "user_id"):
                            val = payload.get(claim_key)
                            if isinstance(val, str) and val:
                                ctype = "person" if claim_key in ("sub", "email", "name") else "entity"
                                n2 = graph.upsert_node(val, type=ctype,
                                                       properties={"claim": claim_key})
                                graph.link(node.id, n2.id, "asserts")
                                ingested["nodes"] += 1
        except Exception as exc:  # noqa: BLE001
            _log.debug("cookie ingest failed: %s", exc)
        # archive the report via the decoder report store
        try:
            from .decoder import save_report
            from dataclasses import dataclass as _dc

            @_dc
            class _Rep:
                best: Any = None
                hits: Any = None
                forensics: Any = None
                hash: Any = None
                tokens: Any = None
                cookies: Any = None
                jwt: Any = None

                def to_dict(self):
                    return rep

            from . import decoder as _dec
            rid = _dec.save_report(
                context.db,
                _dec.DecodeReport(best=None, hits=[],
                                  forensics={"cookies": rep["count"]},
                                  hash=None, tokens=[], cookies=rep["cookies"],
                                  jwt=None),
                source=source, kind="cookie",
                input_head=(text or "")[:200],
                best_name="cookie-lab",
                summary=f"{rep['count']} cookies, services={rep['services']}")
            if rid:
                ingested["report_id"] = rid
        except Exception as exc:  # noqa: BLE001
            _log.debug("cookie report archive failed: %s", exc)
        return ingested


# ── module-level conveniences ────────────────────────────────────────────────

_DEFAULT = CookieLab()


def parse_cookies(text: str) -> list[Cookie]:
    return _DEFAULT.parse(text)


def cookie_report(text: str, *, decode: bool = True) -> dict[str, Any]:
    return _DEFAULT.report(text, decode=decode)
