"""Self-hosted send layer — Devon owns the mailgun (build-map #98).

A clean-room Python send engine that lives inside nomorals:

* SQLite **queue** — ``enqueue()`` stages sends; ``process()`` drains them.
* **Sliding-window rate limiting** per provider (N sends per window).
* **Retry with exponential backoff** — transient failures come back later,
  permanent ones don't.
* **Bounce processing** — DSN text parsed into hard/soft bounces;
  hard bounces disable/blocklist the address automatically.
* **Templates** with conditional content blocks (``{% if x %}`` … ``{% endif %}``)
  and ``{{ var }}`` substitution — a tiny renderer, no external deps.
* **Provider registry** — ``smtp`` (stdlib ``smtplib``), ``ses`` (stdlib
  SigV4 + ``urllib``), ``twilio`` (stdlib ``urllib``), plus any injected
  callable. Keys are read at call time only; never stored or logged.
* **Cost-awareness (#68)** — every bulk send routes through
  ``CostTracker``/``CostAwareSender`` semantics: per-provider cost,
  campaign estimates, budget blocks. WhatsApp legs are priced with the
  real WhatsApp rates.

Profile-gating: termux gets a light batch size and single-threaded
draining; laptop/workstation get bigger batches.

Everything is injectable and offline-testable. Every public entry point
never raises. This is a clean-room pattern implementation — it does NOT
port or copy any Listmonk (AGPL-3.0) code; only the queue / throttle /
bounce *architecture* it demonstrates is re-implemented here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

try:
    from ..core.profile import resolve_profile as _resolve_profile
except Exception as e:  # noqa: BLE001 — profile is a nicety, not a requirement
    record_failure("send.import._resolve_profile", e)
    _resolve_profile = None  # type: ignore[assignment]

try:
    from ..social.whatsapp_cost import CostTracker as _CostTracker
except Exception as e:  # noqa: BLE001
    record_failure("send.import._CostTracker", e)
    _CostTracker = None  # type: ignore[assignment]


def _default_db() -> str:
    try:
        home = __import__("os").path.expanduser("~")
        d = __import__("os").path.join(home, ".nomorals", "marketing")
        __import__("os").makedirs(d, exist_ok=True)
        return __import__("os").path.join(d, "send_queue.db")
    except Exception as e:  # noqa: BLE001
        record_failure("send._default_db", e)
        return ":memory:"


def _profile_kind(profile: str = "") -> str:
    if profile:
        return profile.strip().lower()
    try:
        if _resolve_profile is not None:
            return (_resolve_profile().kind or "pc").lower()
    except Exception as e:  # noqa: BLE001
        record_failure("send._profile_kind", e)
        pass
    return "pc"


#: Batch / concurrency envelopes per profile kind.
PROFILE_BATCH = {
    "termux": 10,
    "embedded": 10,
    "vps": 50,
    "pc": 200,
    "workstation": 1000,
}

#: Planning-estimate cost per message (kobo) by provider kind.
#: Real billing varies; these bound a campaign, not invoice it.
COST_PER_MSG_KOBO = {
    "smtp": 40,       # ~₦0.40 / email (typical self-hosted amortized)
    "ses": 15,        # ~₦0.15 / email (AWS SES $0.10/1k)
    "twilio": 2500,   # ~₦25 / SMS segment (planning estimate)
    "whatsapp": 1400, # ~₦14 / service message (#68 rate)
    "telegram": 0,
    "mock": 0,
}

#: Max attempts before a queued send is declared dead.
MAX_ATTEMPTS = 5

#: Base backoff between attempts (seconds); exponential: base * 2**attempt.
BACKOFF_BASE_S = 60.0


# ── template renderer ─────────────────────────────────────────────────────

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}")
_IF_RE = re.compile(r"\{%\s*if\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s*%\}")
_ENDIF_RE = re.compile(r"\{%\s*endif\s*%\}")
_ELSE_RE = re.compile(r"\{%\s*else\s*%\}")


def _lookup(vars_: dict, path: str):
    cur: object = vars_
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def render_template(body: str, vars_: dict | None) -> str:
    """Render ``{{ var }}`` and ``{% if x %}…{% else %}…{% endif %}``.

    Tiny, dependency-free, and deterministic. Never raises.
    """
    try:
        vars_ = dict(vars_ or {})
        text = body or ""

        # Conditionals first (supports one optional {% else %} per block).
        out: list[str] = []
        i = 0
        while True:
            m = _IF_RE.search(text, i)
            if not m:
                out.append(text[i:])
                break
            out.append(text[i:m.start()])
            var = m.group(1)
            # find matching endif (no nesting support — keep it simple/honest)
            e = _ENDIF_RE.search(text, m.end())
            if not e:
                out.append(text[m.start():])
                break
            inner = text[m.end():e.start()]
            else_m = _ELSE_RE.search(inner)
            if else_m:
                true_part, false_part = inner[:else_m.start()], inner[else_m.end():]
            else:
                true_part, false_part = inner, ""
            out.append(true_part if _lookup(vars_, var) else false_part)
            i = e.end()
        text = "".join(out)

        def _sub(mo: re.Match) -> str:
            val = _lookup(vars_, mo.group(1))
            return "" if val is None else str(val)

        return _VAR_RE.sub(_sub, text)
    except Exception:  # noqa: BLE001
        _log.debug("template render failed", exc_info=True)
        return body or ""


# ── data classes ──────────────────────────────────────────────────────────

@dataclass
class Provider:
    name: str
    kind: str = "smtp"  # smtp | ses | twilio | whatsapp | telegram | mock
    rate_per_minute: int = 30
    cost_kobo: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        self.kind = (self.kind or "smtp").lower()
        if self.cost_kobo <= 0:
            self.cost_kobo = COST_PER_MSG_KOBO.get(self.kind, 0)


@dataclass
class QueuedSend:
    send_id: str = ""
    to: str = ""
    template: str = ""
    vars: dict = field(default_factory=dict)
    provider: str = ""
    subject: str = ""
    status: str = "queued"  # queued | sending | sent | failed | dead | blocked
    attempts: int = 0
    next_attempt_at: float = 0.0
    last_error: str = ""
    created_at: float = 0.0


# ── provider callables (stdlib, keys at call time only) ───────────────────

def _smtp_send(to: str, subject: str, body: str) -> bool:
    """stdlib smtplib send. Config via env at call time; raises on failure."""
    import os
    import smtplib
    from email.message import EmailMessage

    host = os.environ.get("DEVON_SMTP_HOST", "")
    port = int(os.environ.get("DEVON_SMTP_PORT", "587") or 587)
    user = os.environ.get("DEVON_SMTP_USER", "")
    pw = os.environ.get("DEVON_SMTP_PASSWORD", "")
    sender = os.environ.get("DEVON_SMTP_FROM", user)
    if not host or not to:
        raise RuntimeError("SMTP not configured (DEVON_SMTP_HOST)")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject or ""
    # One-click unsubscribe (RFC 2369/8089) — easy, immediate opt-out.
    unsub = os.environ.get("DEVON_UNSUBSCRIBE_URL", "").strip()
    if unsub:
        msg["List-Unsubscribe"] = f"<{unsub}>"
        msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        if user:
            s.login(user, pw)
        s.send_message(msg)
    return True


def _ses_send(to: str, subject: str, body: str) -> bool:
    """AWS SES SendEmail with stdlib SigV4. Raises on failure."""
    import datetime
    import json
    import os
    import urllib.request

    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    region = os.environ.get("AWS_REGION", "eu-west-1")
    sender = os.environ.get("DEVON_SES_FROM", "")
    if not key or not secret or not sender or not to:
        raise RuntimeError("SES not configured (AWS keys / DEVON_SES_FROM)")

    payload = {
        "Source": sender,
        "Destination": {"ToAddresses": [to]},
        "Message": {
            "Subject": {"Data": subject or "", "Charset": "UTF-8"},
            "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
        },
    }
    data = json.dumps(payload).encode()
    amz = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    day = amz[:8]
    service, host = "ses", f"email.{region}.amazonaws.com"
    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": amz,
        "Host": host,
    }

    def _h(k: bytes, msg: str) -> bytes:
        return hmac.new(k, msg.encode(), hashlib.sha256).digest()

    k_date = _h(("AWS4" + secret).encode(), day)
    k_region = _h(k_date, region)
    k_service = _h(k_region, service)
    k_sign = _h(k_service, "aws4_request")
    signed = "content-type;host;x-amz-date"
    canon_headers = (f"content-type:{headers['Content-Type']}\n"
                     f"host:{host}\n"
                     f"x-amz-date:{amz}\n")
    canon = ("POST\n/\n\n" + canon_headers + "\n" + signed + "\n"
             + hashlib.sha256(data).hexdigest())
    scope = f"{day}/{region}/{service}/aws4_request"
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(canon.encode()).hexdigest()}"
    sig = hmac.new(k_sign, string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={key}/{scope}, "
        f"SignedHeaders={signed}, Signature={sig}"
    )
    req = urllib.request.Request(f"https://{host}/v2/email/outbound-emails",
                                 data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"SES HTTP {resp.status}")
    return True


def _twilio_send(to: str, subject: str, body: str) -> bool:
    """Twilio SMS via stdlib urllib. Raises on failure."""
    import os
    import urllib.parse
    import urllib.request

    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    sender = os.environ.get("TWILIO_FROM", "")
    if not sid or not token or not sender or not to:
        raise RuntimeError("Twilio not configured (TWILIO_* env)")
    creds = base64.b64encode(f"{sid}:{token}".encode()).decode()
    data = urllib.parse.urlencode(
        {"From": sender, "To": to, "Body": body[:1600]}).encode()
    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=data,
        headers={"Authorization": f"Basic {creds}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Twilio HTTP {resp.status}")
    return True


#: Built-in provider callables by kind: ``fn(to, subject, body) -> bool``.
BUILTIN_PROVIDERS = {
    "smtp": _smtp_send,
    "ses": _ses_send,
    "twilio": _twilio_send,
    "mock": lambda to, subject, body: True,  # tests / dry runs
}


# ── bounce processing ─────────────────────────────────────────────────────

_DSN_STATUS_RE = re.compile(r"\b([245])\.(\d{1,3})\.(\d{1,3})\b")
_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

#: Hard-bounce class codes → permanent failure, blocklist the address.
HARD_BOUNCE_CLASSES = {"5.1.1", "5.1.2", "5.1.6", "5.1.10", "5.2.1", "5.4.1", "5.7.1"}


def parse_dsn(dsn_text: str) -> dict:
    """Parse a DSN/bounce report into ``{addresses, kind, code}``.

    kind: ``hard`` | ``soft`` | ``unknown``. Never raises.
    """
    try:
        text = dsn_text or ""
        addresses = sorted(set(_ADDR_RE.findall(text)))
        kind, code = "unknown", ""
        for m in _DSN_STATUS_RE.finditer(text):
            code = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            cls = f"{m.group(1)}.{m.group(2)}"
            short = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            if short in HARD_BOUNCE_CLASSES or m.group(1) == "5":
                kind = "hard"
            elif m.group(1) == "4":
                kind = "soft"
            else:
                kind = "unknown"
            break
        low = text.lower()
        if kind == "unknown":
            if any(w in low for w in ("user unknown", "mailbox unavailable",
                                      "address rejected", "does not exist",
                                      "undeliverable", "permanent failure")):
                kind = "hard"
            elif any(w in low for w in ("temporarily", "deferred", "retry",
                                        "mailbox full", "greylisted")):
                kind = "soft"
        return {"addresses": addresses, "kind": kind, "code": code}
    except Exception:  # noqa: BLE001
        return {"addresses": [], "kind": "unknown", "code": ""}


# ── the engine ────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS send_templates (
    name TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS send_providers (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    rate_per_minute INTEGER NOT NULL DEFAULT 30,
    cost_kobo INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS send_queue (
    send_id TEXT PRIMARY KEY,
    recipient TEXT NOT NULL,
    template TEXT NOT NULL,
    vars TEXT NOT NULL DEFAULT '{}',
    provider TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    sent_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS send_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    send_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS send_blocklist (
    address TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS send_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    cost_kobo INTEGER NOT NULL,
    at REAL NOT NULL
);
-- sweep upgrade: A/B variants, suppression list, warmup state
CREATE TABLE IF NOT EXISTS send_variants (
    variant_id TEXT PRIMARY KEY,
    template TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    body TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS send_suppressions (
    address TEXT PRIMARY KEY,
    reason TEXT NOT NULL,          -- unsubscribe | complaint | manual
    detail TEXT NOT NULL DEFAULT '',
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS send_warmup (
    provider TEXT PRIMARY KEY,
    day0 REAL NOT NULL,
    daily_target INTEGER NOT NULL DEFAULT 100
);
"""

#: The seam DDL: the native DDL plus the final-shape columns the
#: in-file ALTER sweep adds on the native path (campaign_id,
#: variant) and the two seam-only claim columns (claim_token,
#: claim_until). The native path NEVER executes this string — it
#: runs _CREATE_SQL + the guarded ALTER sweep — so native rows never
#: carry claim fields and the native path stays byte-for-byte. The
#: seam mapping is derived from this string, so the claim fields ride
#: in the send_queue document exactly like the queue.py lease fields
#: ride in the job document (the extra-columns precedent).
_SEAM_DDL = (
    _CREATE_SQL
    + "ALTER TABLE send_queue ADD COLUMN campaign_id TEXT DEFAULT '';\n"
    + "ALTER TABLE send_queue ADD COLUMN variant TEXT DEFAULT '';\n"
    + "ALTER TABLE send_queue ADD COLUMN claim_token TEXT DEFAULT '';\n"
    + "ALTER TABLE send_queue ADD COLUMN claim_until REAL DEFAULT 0;\n"
)

#: Every live table, in the order the port imports them.
_TABLES = ("send_queue", "send_log", "send_providers", "send_templates",
           "send_variants", "send_suppressions", "send_blocklist",
           "send_costs", "send_warmup")

#: How long one engine's claim on a queued row lives (seconds). The
#: residual double-send window — a claim expiring under a
#: still-running attempt — is bounded by this, a strictly smaller
#: window than the native design's unbounded one (the native code
#: has no claim at all). >= the longest provider call the engine
#: tolerates; recorded beside the registry entry.
CLAIM_TTL = 300.0


#: Spam-trigger patterns (Mail-Tester pattern, offline heuristic).
_SPAM_TRIGGERS = (
    (r"\bfree\b.{0,20}(money|gift|prize|winner|cash)", 12, "free-money phrasing"),
    (r"\b(buy now|act now|click here|limited time|order now)\b", 10, "pushy CTA phrasing"),
    (r"\b(guarantee[ds]?|no risk|risk free|100% (free|safe))\b", 10, "guarantee phrasing"),
    (r"\b(congratulations|you('ve| have) (won|been selected)|dear friend)\b", 12, "lottery-scam phrasing"),
    (r"\$\$\$|!!!+", 8, "excessive !!! / $$$"),
    (r"\b(make money|earn \$|get rich|double your)\b", 12, "get-rich phrasing"),
    (r"\b(viagra|cialis|pharmacy|pills)\b", 15, "pharma spam terms"),
)


def spam_score(subject: str, body: str) -> dict:
    """0–100 spam-likelihood heuristic + fix suggestions. Never raises.

    Mail-Tester pattern, fully offline: trigger phrases, caps ratio,
    exclamation density, link count. Aim < 30 before sending bulk.
    """
    try:
        text = f"{subject or ''}\n{body or ''}"
        low = text.lower()
        score = 0
        issues: list[str] = []
        for pat, weight, label in _SPAM_TRIGGERS:
            if re.search(pat, low):
                score += weight
                issues.append(label)
        words = re.findall(r"[A-Za-z]{2,}", text)
        if words:
            caps = sum(1 for w in words if w.isupper())
            if caps / len(words) > 0.25:
                score += 10
                issues.append(f"{caps/len(words):.0%} ALL-CAPS words — tone it down")
        excl = text.count("!")
        if excl > 3:
            score += min(12, (excl - 3) * 3)
            issues.append(f"{excl} exclamation marks")
        links = len(re.findall(r"https?://", low))
        if links > 3:
            score += min(10, (links - 3) * 4)
            issues.append(f"{links} links — trim to the essential")
        if len(body or "") < 50:
            score += 8
            issues.append("very short body — looks like spam")
        score = min(100, score)
        verdict = ("likely spam-folder" if score >= 60 else
                   "risky — clean it up" if score >= 30 else "clean")
        return {"score": score, "verdict": verdict, "issues": issues}
    except Exception as e:  # noqa: BLE001
        record_failure("send.spam_score", e)
        return {"score": 0, "verdict": "unknown", "issues": []}


def warmup_plan(daily_target: int, days: int = 21) -> list[dict]:
    """Gradual warmup ramp: start small, ~1.35x/day, cap at target.

    Deliverability canon: never blast a fresh domain. Never raises.
    """
    try:
        target = max(10, int(daily_target or 100))
        days = max(3, min(60, int(days or 21)))
        start = min(25, target // 4 or 10)
        plan, cap = [], float(start)
        for d in range(1, days + 1):
            plan.append({"day": d, "cap": int(min(target, round(cap)))})
            cap *= 1.35
        plan[-1]["cap"] = target
        return plan
    except Exception as e:  # noqa: BLE001
        record_failure("send.warmup_plan", e)
        return []


def _dns_txt(name: str, timeout: float = 4.0) -> list[str]:
    """Minimal DNS TXT lookup over UDP (stdlib only). Never raises.

    Used for SPF (v=spf1) / DMARC checks without dnspython.
    """
    out: list[str] = []
    try:
        import random
        import socket
        import struct

        qid = random.randint(0, 65535)
        header = struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0)
        qname = b"".join(
            bytes([len(p)]) + p.encode("ascii", "ignore")
            for p in name.split(".") if p) + b"\x00"
        pkt = header + qname + struct.pack(">HH", 16, 1)  # TXT IN

        def _skip(nb: bytes, off: int) -> int:
            while off < len(nb):
                ln = nb[off]
                if ln == 0:
                    return off + 1
                if ln & 0xC0:  # compression pointer
                    return off + 2
                off += 1 + ln
            return off

        for resolver in ("8.8.8.8", "1.1.1.1"):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(timeout)
                s.sendto(pkt, (resolver, 53))
                data, _ = s.recvfrom(2048)
                s.close()
            except Exception:  # noqa: BLE001
                continue
            try:
                if len(data) < 12 or struct.unpack(">H", data[:2])[0] != qid:
                    continue
                off = _skip(data, 12) + 4  # question
                ancount = struct.unpack(">H", data[6:8])[0]
                for _ in range(ancount):
                    off = _skip(data, off)
                    rtype, _cls, _ttl, rdlen = struct.unpack(">HHIH", data[off:off + 10])
                    rdata = data[off + 10:off + 10 + rdlen]
                    off += 10 + rdlen
                    if rtype == 16:  # TXT
                        i = 0
                        txt = ""
                        while i < len(rdata):
                            ln = rdata[i]
                            txt += rdata[i + 1:i + 1 + ln].decode("utf-8", "replace")
                            i += 1 + ln
                        if txt:
                            out.append(txt)
                if out:
                    return out
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception as e:  # noqa: BLE001
        record_failure("send._dns_txt", e)
        return []


def deliverability_check(domain: str) -> dict:
    """Grade a sending domain: SPF, DMARC, MX/A reachability. Never raises.

    'Fix your DNS first' — misconfigured SPF/DKIM/DMARC is the #1 reason
    self-hosted mail lands in spam. DKIM can't be checked without knowing
    the selector, so it's reported as manual.
    """
    result: dict = {"domain": (domain or "").strip().lower(),
                    "score": 0, "grade": "F", "checks": []}
    try:
        dom = result["domain"]
        if not dom or "." not in dom:
            return result
        checks: list[dict] = []
        score = 0

        txts = _dns_txt(dom)
        spf = [t for t in txts if t.lower().startswith("v=spf1")]
        if spf:
            checks.append({"name": "SPF", "ok": True,
                           "detail": spf[0][:80]})
            score += 35
        elif txts:
            checks.append({"name": "SPF", "ok": False,
                           "detail": "TXT records exist but no v=spf1 — add one"})
        else:
            checks.append({"name": "SPF", "ok": None,
                           "detail": "no TXT records found (DNS unreachable?)"})

        dmarc = _dns_txt(f"_dmarc.{dom}")
        if any(t.lower().startswith("v=dmarc1") for t in dmarc):
            checks.append({"name": "DMARC", "ok": True,
                           "detail": dmarc[0][:80]})
            score += 35
        else:
            checks.append({"name": "DMARC", "ok": False,
                           "detail": "no _dmarc TXT — publish at least p=none"})

        try:
            import socket as _s
            _s.gethostbyname(dom)
            checks.append({"name": "domain resolves", "ok": True,
                           "detail": "A record present"})
            score += 15
        except Exception:  # noqa: BLE001
            checks.append({"name": "domain resolves", "ok": False,
                           "detail": "no A record"})

        checks.append({"name": "DKIM", "ok": None,
                       "detail": "needs your selector — check with "
                                 "mail-tester.com after sending a seed"})
        checks.append({"name": "sending subdomain", "ok": None,
                       "detail": "send from mail." + dom + " to isolate the "
                                 "main domain's reputation"})
        result["checks"] = checks
        result["score"] = min(100, score)
        g = result["score"]
        result["grade"] = ("A" if g >= 80 else "B" if g >= 60 else
                           "C" if g >= 40 else "D" if g >= 20 else "F")
        return result
    except Exception as e:  # noqa: BLE001
        record_failure("send.deliverability_check", e)
        return result


class SendEngine:
    """Self-hosted bulk send engine: queue, throttle, retry, bounce.

    Profile-gated batch sizes; every method never raises.

    When a document store is resolved (Mongo program P5 — an explicit
    ``store`` or the boot ambient), the same API runs over the seam
    instead (registered in storage.sidecars as marketing-send): the
    nine tables are documents in ``marketing-send__<table>``, the
    AUTOINCREMENT log/cost rows ride allocated ids via put_raw with
    the counters seeded by the one-time lazy import, and the file's
    existing rows are imported once, through the same mapping, at
    first construction. The queue itself is a coordination port: on
    the seam, ``process()`` claims each due row with ONE cas_row
    fenced on {status, attempts, next_attempt_at, claim_token
    as-read}, writing {claim_token, claim_until}; every outcome
    write is fenced on the claim token this attempt holds, and a
    lost fence fails closed — the write is refused and journaled,
    the row is left for its claim to expire. The claim fields are
    seam-only document fields (the _SEAM_DDL columns): native rows
    never carry them and the native path stays byte-for-byte. A
    mixed fleet — one native engine and one seam engine over the
    same logical queue — is NOT a supported composition: backends
    flip per install, per the whole programme.
    """

    #: Seam port, None on the native path. A class attribute (not
    #: only an __init__ assignment) so engines built via __new__
    #: — the failure-injection suites' construction — read None
    #: instead of raising AttributeError on the seam guards.
    _port: Any = None

    def __init__(self, db_path: str = "", *, profile: str = "",
                 store: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        try:
            path = db_path or _default_db()
            from ..storage.sidecars import sidecar_port

            port = sidecar_port("marketing-send", _SEAM_DDL, _TABLES,
                                path, store=store)
            if port is not None:
                self._import_into(port)
                self._port = port
            else:
                self._db = sqlite3.connect(path)
                self._db.row_factory = sqlite3.Row
                self._db.executescript(_CREATE_SQL)
                self._db.commit()
                # sweep migration: campaign tracking columns on older
                # queue tables (native-path concern only — the seam
                # DDL already carries the final shape)
                try:
                    for _col, _ddl in (("campaign_id", "TEXT DEFAULT ''"),
                                       ("variant", "TEXT DEFAULT ''")):
                        try:
                            self._db.execute(
                                f"ALTER TABLE send_queue ADD COLUMN {_col} {_ddl}")
                        except Exception as e:  # noqa: BLE001
                            # Duplicate column is the probe answer
                            # (column already migrated); any other
                            # ALTER failure is a real failure and
                            # records.
                            if "duplicate column" not in str(e):
                                record_failure(
                                    "send.SendEngine.__init__.sweep_column", e)
                    self._db.commit()
                except Exception as e:  # noqa: BLE001
                    record_failure("send.SendEngine.__init__.sweep", e)
        except Exception:  # noqa: BLE001
            # THE debug-honest DB handler: a dead database (native
            # connect, sidecar resolution, or the one-time import)
            # degrades the engine silently — debug-logged, never
            # recorded — exactly as the native path always did.
            _log.debug("send engine DB unavailable", exc_info=True)
            self._db = None
            self._port = None
        kind = _profile_kind(profile)
        self.profile = kind
        self.batch_size = PROFILE_BATCH.get(kind, PROFILE_BATCH["pc"])
        self._callables: dict[str, object] = {}
        self._tracker = None
        try:
            if _CostTracker is not None:
                self._tracker = _CostTracker()
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.__init__.tracker", e)
            self._tracker = None

    def _import_into(self, port: Any) -> None:
        """One-time lazy import of send_queue.db into the seam.

        The seven TEXT-PK tables port as-is through their RowTables
        (a file row that predates the ALTER sweep takes campaign_id /
        variant / claim_token / claim_until from the seam DDL
        defaults); the AUTOINCREMENT log/cost rows keep their file
        ids via put_raw, and the id counters are seeded from the
        imported maxima so seam-allocated ids continue the file's
        sequences instead of colliding with them. Runs through
        ensure_imported: one-time, idempotent — a live store is
        adopted, never overwritten."""
        max_ids = {"send_log": 0, "send_costs": 0}

        def _make(table: Any) -> Any:
            def write(rows: list[dict[str, Any]]) -> int:
                for row in rows:
                    table.put(row)
                return len(rows)
            return write

        def _write_ids(table_name: str) -> Any:
            def write(rows: list[dict[str, Any]]) -> int:
                for row in rows:
                    eid = int(row["id"])
                    max_ids[table_name] = max(max_ids[table_name], eid)
                    port.put_raw(table_name, str(eid), dict(row))
                return len(rows)
            return write

        def seed_counters(_counts: dict[str, int]) -> None:
            doc = port.store.get(port.meta_collection, "counters") or {}
            merged = dict(doc)
            for name, top in max_ids.items():
                if int(merged.get(name) or 0) < top:
                    merged[name] = top
            if merged != doc:
                port.store.put(port.meta_collection, "counters", merged)

        writers = {name: _make(port.tables[name]) for name in
                   ("send_queue", "send_providers", "send_templates",
                    "send_variants", "send_suppressions", "send_blocklist",
                    "send_warmup")}
        writers["send_log"] = _write_ids("send_log")
        writers["send_costs"] = _write_ids("send_costs")
        port.ensure_imported(writers, after_import=seed_counters)

    def _qt(self, table: str) -> Any:
        """The seam RowTable for ``table`` (port is not None here)."""
        return self._port.tables[table]

    # ── templates ──

    def save_template(self, name: str, body: str, subject: str = "") -> bool:
        """Save/update a template. Never raises."""
        try:
            name = (name or "").strip()
            if not name or (self._db is None and self._port is None):
                return False
            if self._port is not None:
                self._qt("send_templates").put({
                    "name": name, "body": body or "",
                    "subject": subject or "", "updated_at": time.time()})
                return True
            self._db.execute(
                "INSERT INTO send_templates (name, body, subject, updated_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET"
                " body=excluded.body, subject=excluded.subject,"
                " updated_at=excluded.updated_at",
                (name, body or "", subject or "", time.time()))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.save_template", e)
            return False

    def list_templates(self) -> list[dict]:
        try:
            if self._db is None and self._port is None:
                return []
            if self._port is not None:
                rows = sorted(self._qt("send_templates").all(),
                              key=lambda r: r["name"])
                return [{"name": r["name"], "subject": r["subject"],
                         "updated_at": r["updated_at"]} for r in rows]
            return [dict(r) for r in self._db.execute(
                "SELECT name, subject, updated_at FROM send_templates ORDER BY name")]
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.list_templates", e)
            return []

    def get_template(self, name: str) -> dict | None:
        try:
            if self._db is None and self._port is None:
                return None
            if self._port is not None:
                row = self._qt("send_templates").get((name or "").strip())
                return dict(row) if row else None
            row = self._db.execute(
                "SELECT * FROM send_templates WHERE name = ?",
                ((name or "").strip(),)).fetchone()
            return dict(row) if row else None
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.get_template", e)
            return None

    # ── providers ──

    def register_provider(self, name: str, *, kind: str = "smtp",
                          rate_per_minute: int = 30,
                          sender=None, enabled: bool = True) -> bool:
        """Register a provider. ``sender`` overrides the built-in callable.

        sender: ``fn(to, subject, body) -> bool`` (may raise on failure).
        """
        try:
            name = (name or "").strip()
            kind = (kind or "smtp").lower()
            if not name or (self._db is None and self._port is None):
                return False
            cost = COST_PER_MSG_KOBO.get(kind, 0)
            if self._port is not None:
                self._qt("send_providers").put({
                    "name": name, "kind": kind,
                    "rate_per_minute": max(1, int(rate_per_minute or 30)),
                    "cost_kobo": cost, "enabled": 1 if enabled else 0})
            else:
                self._db.execute(
                    "INSERT INTO send_providers (name, kind, rate_per_minute,"
                    " cost_kobo, enabled) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(name) DO UPDATE SET kind=excluded.kind,"
                    " rate_per_minute=excluded.rate_per_minute,"
                    " cost_kobo=excluded.cost_kobo, enabled=excluded.enabled",
                    (name, kind, max(1, int(rate_per_minute or 30)),
                     cost, 1 if enabled else 0))
                self._db.commit()
            if sender is not None:
                self._callables[name] = sender
            elif kind in BUILTIN_PROVIDERS:
                self._callables[name] = BUILTIN_PROVIDERS[kind]
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.register_provider", e)
            return False

    def list_providers(self) -> list[dict]:
        try:
            if self._db is None and self._port is None:
                return []
            if self._port is not None:
                return sorted((dict(r) for r in
                               self._qt("send_providers").all()),
                              key=lambda r: r["name"])
            return [dict(r) for r in self._db.execute(
                "SELECT * FROM send_providers ORDER BY name")]
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.list_providers", e)
            return []

    # ── queue ──

    def enqueue(self, to: str, template: str, vars_: dict | None = None,
                provider: str = "", *, subject: str = "",
                campaign_id: str = "", variant: str = "",
                send_at: float = 0.0) -> str:
        """Stage a send. Returns the send_id ("" on failure). Never raises.

        ``campaign_id``/``variant`` tag A/B sends; ``send_at`` schedules
        the send for a future time (process() picks it up when due).
        Suppressed (unsubscribed/complained) addresses are refused.
        """
        try:
            to = (to or "").strip()
            template = (template or "").strip()
            provider = (provider or "").strip()
            if not to or not template or not provider \
                    or (self._db is None and self._port is None):
                return ""
            if self.is_blocklisted(to):
                self._log("", provider, "blocked", f"blocklisted: {to}")
                return ""
            if self.is_suppressed(to):
                self._log("", provider, "suppressed",
                          f"suppressed (unsub/complaint): {to}")
                return ""
            import json
            send_id = "snd_" + uuid.uuid4().hex[:10]
            if self._port is not None:
                self._qt("send_queue").put({
                    "send_id": send_id, "recipient": to,
                    "template": template, "vars": json.dumps(vars_ or {}),
                    "provider": provider, "subject": subject or "",
                    "status": "queued", "attempts": 0,
                    "next_attempt_at": float(send_at or 0.0),
                    "last_error": "", "created_at": time.time(),
                    "sent_at": 0, "campaign_id": campaign_id or "",
                    "variant": variant or "", "claim_token": "",
                    "claim_until": 0})
                return send_id
            self._db.execute(
                "INSERT INTO send_queue (send_id, recipient, template, vars,"
                " provider, subject, status, created_at, campaign_id, variant,"
                " next_attempt_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                (send_id, to, template, json.dumps(vars_ or {}),
                 provider, subject or "", time.time(),
                 campaign_id or "", variant or "",
                 float(send_at or 0.0)))
            self._db.commit()
            return send_id
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.enqueue", e)
            return ""

    def queue_depth(self, status: str = "queued") -> int:
        try:
            if self._db is None and self._port is None:
                return 0
            if self._port is not None:
                return self._qt("send_queue").count({"status": status})
            row = self._db.execute(
                "SELECT COUNT(*) c FROM send_queue WHERE status = ?",
                (status,)).fetchone()
            return int(row["c"]) if row else 0
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.queue_depth", e)
            return 0

    # ── rate limiting ──

    def _rate_ok(self, provider: str, limit: int) -> bool:
        """Sliding-window: < limit sends in the last 60s for this provider."""
        try:
            if self._db is None and self._port is None:
                return True
            cutoff = time.time() - 60.0
            if self._port is not None:
                sent = sum(
                    1 for r in self._qt("send_log").find(
                        {"provider": provider, "event": "sent"})
                    if float(r["at"]) >= cutoff)
                return sent < max(1, limit)
            row = self._db.execute(
                "SELECT COUNT(*) c FROM send_log"
                " WHERE provider = ? AND event = 'sent' AND at >= ?",
                (provider, cutoff)).fetchone()
            return int(row["c"]) < max(1, limit)
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._rate_ok", e)
            return True

    # ── processing ──

    def process(self, limit: int = 0) -> dict:
        """Drain the queue: render → rate-limit → cost-check → send → retry.

        Returns ``{sent, failed, deferred, blocked}``. Never raises.
        """
        if self._port is not None:
            return self._process_seam(limit)
        stats = {"sent": 0, "failed": 0, "deferred": 0, "blocked": 0}
        try:
            if self._db is None:
                return stats
            n = max(1, int(limit or self.batch_size))
            now = time.time()
            rows = self._db.execute(
                "SELECT * FROM send_queue WHERE status = 'queued'"
                " AND next_attempt_at <= ? ORDER BY created_at LIMIT ?",
                (now, n)).fetchall()
            for r in rows:
                outcome = self._attempt(dict(r))
                stats[outcome] = stats.get(outcome, 0) + 1
            return stats
        except Exception:  # noqa: BLE001
            _log.debug("send process failed", exc_info=True)
            return stats

    def _process_seam(self, limit: int = 0) -> dict:
        """The seam drain: claim, then attempt, per row.

        Selection mirrors the native query (queued + due, created_at
        order, equal created_at ties fall back to send_id order) with
        one addition: a row is a candidate only while it is
        unclaimed or its claim has expired. Each candidate is claimed
        with ONE cas_row fenced on {status, attempts,
        next_attempt_at, claim_token as-read}; a lost cas skips the
        row — another engine owns it. Never raises."""
        stats = {"sent": 0, "failed": 0, "deferred": 0, "blocked": 0}
        try:
            n = max(1, int(limit or self.batch_size))
            now = time.time()
            queue = self._qt("send_queue")
            rows = [r for r in queue.find({"status": "queued"})
                    if float(r["next_attempt_at"]) <= now
                    and (not (r.get("claim_token") or "")
                         or float(r.get("claim_until") or 0) <= now)]
            rows.sort(key=lambda r: (float(r["created_at"]), r["send_id"]))
            for row in rows[:n]:
                token = "clm_" + uuid.uuid4().hex[:16]
                new_row = {**row, "claim_token": token,
                           "claim_until": time.time() + CLAIM_TTL}
                expected = {"status": "queued",
                            "attempts": row["attempts"],
                            "next_attempt_at": row["next_attempt_at"],
                            "claim_token": row.get("claim_token") or ""}
                if not queue.cas_row(row["send_id"], expected, new_row):
                    continue  # another engine claimed it first
                outcome = self._attempt_seam(new_row, token)
                stats[outcome] = stats.get(outcome, 0) + 1
            return stats
        except Exception:  # noqa: BLE001
            _log.debug("send process failed", exc_info=True)
            return stats

    def _attempt_seam(self, row: dict, token: str) -> str:
        """One claimed send attempt → 'sent' | 'failed' | 'deferred'
        | 'blocked'. The seam mirror of :meth:`_attempt`: every
        outcome write is fenced on the claim token this attempt
        holds, and a lost fence fails closed (the write is refused
        and journaled; the row is left for its claim to expire).
        Deferrals release the claim, so the row stays immediately
        re-attemptable exactly as on the native path."""
        send_id = row["send_id"]
        try:
            prov = self._qt("send_providers").get(row["provider"])
            if prov is None or not prov["enabled"]:
                self._set_status_seam(row, token, "failed",
                                      "unknown/disabled provider")
                return "failed"
            if self.is_blocklisted(row["recipient"]):
                self._set_status_seam(row, token, "blocked",
                                      "recipient blocklisted")
                return "blocked"
            if self.is_suppressed(row["recipient"]):
                self._set_status_seam(row, token, "blocked",
                                      "recipient suppressed (unsub/complaint)")
                return "blocked"
            if not self._rate_ok(row["provider"], int(prov["rate_per_minute"])):
                self._release_claim_seam(row, token)
                return "deferred"  # leave queued; window will clear
            if not self._warmup_ok(row["provider"]):
                self._release_claim_seam(row, token)
                return "deferred"  # warmup cap for today reached
            # Cost gate (#68): budget check before spend.
            if not self._cost_ok(prov):
                self._release_claim_seam(row, token)
                return "deferred"
            tpl = self.get_template(row["template"])
            variant = (row.get("variant") or "").strip()
            if variant:
                # A/B: variant body overrides the base template.
                var = self.get_variant(row["template"], variant)
                if var is not None:
                    tpl = {"body": var.get("body", ""),
                           "subject": var.get("subject", "")}
            if tpl is None:
                self._set_status_seam(row, token, "failed",
                                      "template not found")
                return "failed"
            import json
            vars_ = {}
            try:
                vars_ = json.loads(row["vars"] or "{}")
            except Exception as e:  # noqa: BLE001
                record_failure("send.SendEngine._attempt.vars", e)
                vars_ = {}
            body = render_template(tpl["body"], vars_)
            subject = row["subject"] or render_template(tpl["subject"], vars_)
            sender = self._callables.get(row["provider"])
            if sender is None:
                self._set_status_seam(row, token, "failed",
                                      "no sender for provider")
                return "failed"
            try:
                ok = bool(sender(row["recipient"], subject, body))
            except Exception as exc:  # provider raised → retry path
                return self._retry_or_dead_seam(row, token, str(exc))
            if ok:
                if self._set_status_seam(row, token, "sent"):
                    self._log(send_id, row["provider"], "sent")
                    self._record_cost(prov)
                    return "sent"
                # The send happened but its record was refused: the
                # stat says failed (the delivery was not recorded)
                # and the fence loss is journaled; the row is left
                # for its claim to expire.
                return "failed"
            return self._retry_or_dead_seam(row, token,
                                            "provider returned False")
        except Exception as exc:  # noqa: BLE001
            return self._retry_or_dead_seam(row, token,
                                            f"attempt error: {exc}")

    def _set_status_seam(self, row: dict, token: str, status: str,
                         error: str = "",
                         attempts: int | None = None) -> bool:
        """A terminal outcome write, fenced on this attempt's claim.

        The cas expects the row still queued under this claim token;
        a lost fence returns False and journals — the write is
        refused, never force-written. A landed write clears the
        claim (the row is no longer queued). Never raises."""
        try:
            new_row = dict(row)
            new_row["status"] = status
            new_row["last_error"] = error[:300]
            if status == "sent":
                new_row["sent_at"] = time.time()
            if attempts is not None:
                new_row["attempts"] = attempts
            new_row["claim_token"] = ""
            new_row["claim_until"] = 0
            ok = self._qt("send_queue").cas_row(
                row["send_id"],
                {"status": "queued", "claim_token": token}, new_row)
            if not ok:
                record_failure(
                    "send.SendEngine._set_status.fence",
                    f"claim fence lost for {row['send_id']}")
            return ok
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._set_status", e)
            return False

    def _release_claim_seam(self, row: dict, token: str) -> bool:
        """Give a claimed row back unattempted (deferral path).

        Fenced on the claim token, exactly like an outcome write:
        a lost fence journals and the row is left for its claim to
        expire. Never raises."""
        try:
            ok = self._qt("send_queue").cas_row(
                row["send_id"],
                {"status": "queued", "claim_token": token},
                {**row, "claim_token": "", "claim_until": 0})
            if not ok:
                record_failure(
                    "send.SendEngine._release_claim.fence",
                    f"claim fence lost for {row['send_id']}")
            return ok
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._release_claim", e)
            return False

    def _retry_or_dead_seam(self, row: dict, token: str,
                            error: str) -> str:
        """Exponential backoff; dead after MAX_ATTEMPTS. The seam
        mirror of :meth:`_retry_or_dead`: the retry write and the
        dead write are both fenced on this attempt's claim, and a
        lost fence fails closed (returns 'failed', journaled).
        Returns a stat key."""
        try:
            attempts = int(row.get("attempts", 0)) + 1
            if attempts >= MAX_ATTEMPTS:
                if self._set_status_seam(row, token, "dead", error[:300],
                                         attempts=attempts):
                    self._log(row["send_id"], row["provider"], "dead",
                              error[:200])
                return "failed"
            backoff = BACKOFF_BASE_S * (2 ** (attempts - 1))
            new_row = {**row, "attempts": attempts,
                       "next_attempt_at": time.time() + backoff,
                       "last_error": error[:300]}
            ok = self._qt("send_queue").cas_row(
                row["send_id"],
                {"status": "queued", "claim_token": token}, new_row)
            if not ok:
                record_failure(
                    "send.SendEngine._retry_or_dead.fence",
                    f"claim fence lost for {row['send_id']}")
                return "failed"
            self._log(row["send_id"], row["provider"], "retry",
                      f"attempt {attempts}: {error[:120]}")
            return "deferred"
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._retry_or_dead", e)
            return "deferred"

    def _attempt(self, row: dict) -> str:
        """One send attempt → 'sent' | 'failed' | 'deferred' | 'blocked'."""
        send_id = row["send_id"]
        try:
            prov = self._db.execute(
                "SELECT * FROM send_providers WHERE name = ?",
                (row["provider"],)).fetchone()
            if prov is None or not prov["enabled"]:
                self._set_status(send_id, "failed", "unknown/disabled provider")
                return "failed"
            if self.is_blocklisted(row["recipient"]):
                self._set_status(send_id, "blocked", "recipient blocklisted")
                return "blocked"
            if self.is_suppressed(row["recipient"]):
                self._set_status(send_id, "blocked", "recipient suppressed (unsub/complaint)")
                return "blocked"
            if not self._rate_ok(row["provider"], int(prov["rate_per_minute"])):
                return "deferred"  # leave queued; window will clear
            if not self._warmup_ok(row["provider"]):
                return "deferred"  # warmup cap for today reached
            # Cost gate (#68): budget check before spend.
            if not self._cost_ok(prov):
                return "deferred"
            tpl = self.get_template(row["template"])
            variant = (row.get("variant") or "").strip()
            if variant:
                # A/B: variant body overrides the base template.
                var = self.get_variant(row["template"], variant)
                if var is not None:
                    tpl = {"body": var.get("body", ""), "subject": var.get("subject", "")}
            if tpl is None:
                self._set_status(send_id, "failed", "template not found")
                return "failed"
            import json
            vars_ = {}
            try:
                vars_ = json.loads(row["vars"] or "{}")
            except Exception as e:  # noqa: BLE001
                record_failure("send.SendEngine._attempt.vars", e)
                vars_ = {}
            body = render_template(tpl["body"], vars_)
            subject = row["subject"] or render_template(tpl["subject"], vars_)
            sender = self._callables.get(row["provider"])
            if sender is None:
                self._set_status(send_id, "failed", "no sender for provider")
                return "failed"
            try:
                ok = bool(sender(row["recipient"], subject, body))
            except Exception as exc:  # provider raised → retry path
                return self._retry_or_dead(row, str(exc))
            if ok:
                self._set_status(send_id, "sent")
                self._log(send_id, row["provider"], "sent")
                self._record_cost(prov)
                return "sent"
            return self._retry_or_dead(row, "provider returned False")
        except Exception as exc:  # noqa: BLE001
            return self._retry_or_dead(row, f"attempt error: {exc}")

    def _retry_or_dead(self, row: dict, error: str) -> str:
        """Exponential backoff; dead after MAX_ATTEMPTS. Returns a stat key."""
        try:
            attempts = int(row.get("attempts", 0)) + 1
            if attempts >= MAX_ATTEMPTS:
                self._set_status(row["send_id"], "dead", error[:300],
                                 attempts=attempts)
                self._log(row["send_id"], row["provider"], "dead", error[:200])
                return "failed"
            backoff = BACKOFF_BASE_S * (2 ** (attempts - 1))
            self._db.execute(
                "UPDATE send_queue SET attempts = ?,"
                " next_attempt_at = ?, last_error = ? WHERE send_id = ?",
                (attempts, time.time() + backoff, error[:300],
                 row["send_id"]))
            self._db.commit()
            self._log(row["send_id"], row["provider"], "retry",
                      f"attempt {attempts}: {error[:120]}")
            return "deferred"
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._retry_or_dead", e)
            return "deferred"

    def _set_status(self, send_id: str, status: str, error: str = "",
                    attempts: int | None = None) -> None:
        try:
            if attempts is None:
                self._db.execute(
                    "UPDATE send_queue SET status = ?, last_error = ?,"
                    " sent_at = CASE WHEN ? = 'sent' THEN ? ELSE sent_at END"
                    " WHERE send_id = ?",
                    (status, error[:300], status, time.time(), send_id))
            else:
                self._db.execute(
                    "UPDATE send_queue SET status = ?, last_error = ?,"
                    " attempts = ? WHERE send_id = ?",
                    (status, error[:300], attempts, send_id))
            self._db.commit()
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._set_status", e)
            pass

    def _log(self, send_id: str, provider: str, event: str,
             detail: str = "") -> None:
        try:
            if self._port is not None:
                eid = self._port.next_id("send_log")
                self._port.put_raw("send_log", str(eid), {
                    "id": eid, "send_id": send_id, "provider": provider,
                    "event": event, "detail": detail[:300],
                    "at": time.time()})
                return
            self._db.execute(
                "INSERT INTO send_log (send_id, provider, event, detail, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (send_id, provider, event, detail[:300], time.time()))
            self._db.commit()
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._log", e)
            pass

    # ── cost routing (#68) ──

    def _cost_ok(self, prov) -> bool:
        """Budget gate: True when we may spend this provider's per-msg cost."""
        try:
            if self._tracker is None:
                return True
            cost = int(prov["cost_kobo"] or 0)
            if cost <= 0:
                return True
            may_send, _spent, _budget = self._tracker.check_budget("marketing")
            return bool(may_send)
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._cost_ok", e)
            return True

    def _record_cost(self, prov) -> None:
        try:
            cost = int(prov["cost_kobo"] or 0)
            if self._port is not None:
                cid = self._port.next_id("send_costs")
                self._port.put_raw("send_costs", str(cid), {
                    "id": cid, "provider": prov["name"],
                    "cost_kobo": cost, "at": time.time()})
            else:
                self._db.execute(
                    "INSERT INTO send_costs (provider, cost_kobo, at)"
                    " VALUES (?, ?, ?)",
                    (prov["name"], cost, time.time()))
                self._db.commit()
            # #68 cost-awareness: also log into the WhatsApp cost tracker
            # (mapped to the closest valid category) when available.
            if self._tracker is not None and cost > 0:
                cat = "marketing" if prov["kind"] in ("ses", "smtp", "twilio") else "service"
                try:
                    self._tracker.track("campaign", cat, client="marketing",
                                        purpose=f"send:{prov['name']}")
                except Exception as e:  # noqa: BLE001
                    record_failure("send.SendEngine._record_cost.tracker", e)
                    pass
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._record_cost", e)
            pass

    def estimate_campaign(self, count: int, provider: str = "") -> dict:
        """Price a campaign before the owner approves it. Never raises."""
        try:
            count = max(0, int(count or 0))
            per = 0
            if provider and (self._db is not None or self._port is not None):
                if self._port is not None:
                    row = self._qt("send_providers").get(provider)
                else:
                    row = self._db.execute(
                        "SELECT cost_kobo FROM send_providers WHERE name = ?",
                        (provider,)).fetchone()
                per = int(row["cost_kobo"]) if row else 0
            if per <= 0:
                per = COST_PER_MSG_KOBO.get("smtp", 40)
            total = count * per
            return {"count": count, "per_kobo": per, "total_kobo": total,
                    "text": (f"this campaign will cost ₦{total/100:,.0f}"
                             f" for {count} recipient{'s' if count != 1 else ''}"
                             f" ({provider or 'default'}, ₦{per/100:,.2f} each)")}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.estimate_campaign", e)
            return {"count": 0, "per_kobo": 0, "total_kobo": 0, "text": ""}

    # ── A/B testing ──

    def save_variant(self, template: str, variant_name: str,
                     body: str, subject: str = "") -> bool:
        """Save a named variant of a template for A/B tests. Never raises."""
        try:
            template = (template or "").strip()
            variant_name = (variant_name or "").strip()
            if not template or not variant_name \
                    or (self._db is None and self._port is None):
                return False
            vid = f"var_{template}_{variant_name}".replace(" ", "_")[:80]
            if self._port is not None:
                self._qt("send_variants").put({
                    "variant_id": vid, "template": template,
                    "variant_name": variant_name, "body": body or "",
                    "subject": subject or "", "created_at": time.time()})
                return True
            self._db.execute(
                "INSERT OR REPLACE INTO send_variants "
                "(variant_id, template, variant_name, body, subject, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (vid, template, variant_name, body or "", subject or "",
                 time.time()))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.save_variant", e)
            return False

    def list_variants(self, template: str = "") -> list[dict]:
        try:
            if self._db is None and self._port is None:
                return []
            if self._port is not None:
                rows = self._qt("send_variants").all()
                if (template or "").strip():
                    rows = [r for r in rows
                            if r["template"] == template.strip()]
                    rows.sort(key=lambda r: r["variant_name"])
                else:
                    rows.sort(key=lambda r: (r["template"],
                                             r["variant_name"]))
                return [{"template": r["template"],
                         "variant_name": r["variant_name"],
                         "subject": r["subject"],
                         "created_at": r["created_at"]} for r in rows]
            if (template or "").strip():
                rows = self._db.execute(
                    "SELECT template, variant_name, subject, created_at"
                    " FROM send_variants WHERE template = ? ORDER BY variant_name",
                    (template.strip(),)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT template, variant_name, subject, created_at"
                    " FROM send_variants ORDER BY template, variant_name").fetchall()
            return [dict(r) for r in rows]
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.list_variants", e)
            return []

    def get_variant(self, template: str, variant_name: str) -> dict | None:
        try:
            if self._db is None and self._port is None:
                return None
            if self._port is not None:
                rows = self._qt("send_variants").find({
                    "template": (template or "").strip(),
                    "variant_name": (variant_name or "").strip()})
                return dict(rows[0]) if rows else None
            row = self._db.execute(
                "SELECT * FROM send_variants WHERE template = ? AND variant_name = ?",
                ((template or "").strip(), (variant_name or "").strip())).fetchone()
            return dict(row) if row else None
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.get_variant", e)
            return None

    def create_ab_campaign(self, name: str, template: str,
                           variant_names: list[str], recipients: list[str],
                           provider: str, vars_fn=None) -> dict:
        """Split recipients across variants (round-robin), enqueue all.

        Returns ``{campaign_id, per_variant}``. Never raises.
        """
        try:
            variants = [v for v in (variant_names or []) if v and v.strip()]
            recips = [r.strip() for r in (recipients or []) if r and r.strip()]
            if not variants or not recips or not (provider or "").strip():
                return {"campaign_id": "", "per_variant": {}}
            campaign_id = "cmp_" + uuid.uuid4().hex[:8]
            per_variant: dict[str, int] = {}
            for i, recip in enumerate(recips):
                variant = variants[i % len(variants)]
                vv = vars_fn(recip, variant) if vars_fn else {}
                sid = self.enqueue(recip, template, vv, provider,
                                   campaign_id=campaign_id, variant=variant)
                if sid:
                    per_variant[variant] = per_variant.get(variant, 0) + 1
            return {"campaign_id": campaign_id, "per_variant": per_variant,
                    "name": name or campaign_id}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.create_ab_campaign", e)
            return {"campaign_id": "", "per_variant": {}}

    def ab_report(self, campaign_id: str) -> dict:
        """Per-variant funnel for an A/B campaign. Never raises."""
        out: dict = {"campaign_id": campaign_id or "", "variants": {}}
        try:
            if (self._db is None and self._port is None) or not campaign_id:
                return out
            if self._port is not None:
                rows = self._qt("send_queue").find(
                    {"campaign_id": campaign_id})
            else:
                rows = self._db.execute(
                    "SELECT variant, status, COUNT(*) c FROM send_queue"
                    " WHERE campaign_id = ? GROUP BY variant, status",
                    (campaign_id,)).fetchall()
            for r in rows:
                if self._port is not None:
                    variant, status, count = (r.get("variant") or "",
                                              r["status"], 1)
                else:
                    variant, status, count = (r["variant"], r["status"],
                                              int(r["c"]))
                v = out["variants"].setdefault(variant or "control",
                                               {"sent": 0, "failed": 0,
                                                "dead": 0, "queued": 0,
                                                "blocked": 0})
                v[status] = v.get(status, 0) + count
            for v, stats in out["variants"].items():
                done = stats.get("sent", 0) + stats.get("failed", 0) + stats.get("dead", 0)
                stats["fail_rate"] = round(
                    (stats.get("failed", 0) + stats.get("dead", 0)) / max(1, done), 3)
            return out
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.ab_report", e)
            return out

    def ab_winner(self, campaign_id: str) -> dict:
        """Winning variant = lowest fail rate (ties → most sent).

        Engagement-based winners need an injected signal — wire your
        open/click webhook into send_log with event='engaged' and this
        prefers the variant with the most engagement. Never raises.
        """
        try:
            rep = self.ab_report(campaign_id)
            variants = rep.get("variants", {})
            if not variants:
                return {"winner": "", "reason": "no data"}
            # Prefer engagement when the signal exists.
            eng: dict[str, int] = {}
            if self._db is not None or self._port is not None:
                try:
                    if self._port is not None:
                        # the log⋈queue JOIN, re-expressed over finds
                        by_id = {r["send_id"]: r for r in
                                 self._qt("send_queue").find(
                                     {"campaign_id": campaign_id})}
                        for lr in self._qt("send_log").find(
                                {"event": "engaged"}):
                            q = by_id.get(lr["send_id"])
                            if q is not None:
                                key = q.get("variant") or "control"
                                eng[key] = eng.get(key, 0) + 1
                    else:
                        for r in self._db.execute(
                                "SELECT q.variant, COUNT(*) c FROM send_log l"
                                " JOIN send_queue q ON q.send_id = l.send_id"
                                " WHERE q.campaign_id = ? AND l.event = 'engaged'"
                                " GROUP BY q.variant", (campaign_id,)):
                            eng[r["variant"] or "control"] = int(r["c"])
                except Exception as e:  # noqa: BLE001
                    record_failure("send.SendEngine.ab_winner.engagement", e)
            if eng:
                winner = max(eng.items(), key=lambda kv: kv[1])[0]
                return {"winner": winner, "reason": f"most engagement ({eng[winner]})",
                        "by": "engagement"}
            ranked = sorted(variants.items(),
                            key=lambda kv: (kv[1].get("fail_rate", 1),
                                            -kv[1].get("sent", 0)))
            w, stats = ranked[0]
            return {"winner": w, "reason": f"lowest fail rate ({stats['fail_rate']:.1%})",
                    "by": "fail_rate"}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.ab_winner", e)
            return {"winner": "", "reason": "unavailable"}

    # ── scheduled campaigns + seeds ──

    def schedule_campaign(self, template: str, recipients: list[str],
                          provider: str, send_at: float,
                          vars_fn=None, subject: str = "") -> dict:
        """Enqueue a campaign for a future time. Never raises."""
        try:
            recips = [r.strip() for r in (recipients or []) if r and r.strip()]
            campaign_id = "cmp_" + uuid.uuid4().hex[:8]
            queued = 0
            for r in recips:
                vv = vars_fn(r) if vars_fn else {}
                if self.enqueue(r, template, vv, provider, subject=subject,
                                campaign_id=campaign_id,
                                send_at=float(send_at or 0.0)):
                    queued += 1
            return {"campaign_id": campaign_id, "queued": queued,
                    "send_at": float(send_at or 0.0)}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.schedule_campaign", e)
            return {"campaign_id": "", "queued": 0, "send_at": 0.0}

    def send_seed(self, template: str, seeds: list[str],
                  provider: str, vars_: dict | None = None) -> dict:
        """'Always send a test email first' (Listmonk rule) — seed inboxes
        get a [SEED] copy before the real campaign. Never raises."""
        try:
            seeds = [s.strip() for s in (seeds or []) if s and s.strip()]
            sent = [self.enqueue(s, template, vars_, provider,
                                 subject="[SEED] test send")
                    for s in seeds]
            return {"seeded": sum(1 for s in sent if s), "of": len(seeds)}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.send_seed", e)
            return {"seeded": 0, "of": 0}

    def campaign_report(self, campaign_id: str) -> dict:
        """Funnel: queued → sent → failed → dead → blocked, per provider.
        Never raises."""
        out: dict = {"campaign_id": campaign_id or "", "total": 0,
                     "by_status": {}, "by_provider": {}, "spent_kobo": 0}
        try:
            if (self._db is None and self._port is None) or not campaign_id:
                return out
            if self._port is not None:
                rows = self._qt("send_queue").find(
                    {"campaign_id": campaign_id})
                costs = {r["name"]: int(r["cost_kobo"] or 0) for r in
                         self._qt("send_providers").all()}
            else:
                rows = self._db.execute(
                    "SELECT status, provider, COUNT(*) c FROM send_queue"
                    " WHERE campaign_id = ? GROUP BY status, provider",
                    (campaign_id,)).fetchall()
            for r in rows:
                if self._port is not None:
                    status, provider, c = r["status"], r["provider"], 1
                else:
                    status, provider, c = r["status"], r["provider"], int(r["c"])
                out["total"] += c
                out["by_status"][status] = out["by_status"].get(status, 0) + c
                p = out["by_provider"].setdefault(provider, {})
                p[status] = p.get(status, 0) + c
            # Spend: this campaign's SENT rows priced at each provider's
            # cost_kobo — the same basis _record_cost charges the daily
            # budget on. (send_costs rows carry no campaign_id, so the
            # per-provider rate is the honest source.)
            if self._port is None:
                costs = {r["name"]: int(r["cost_kobo"] or 0) for r in
                         self._db.execute(
                             "SELECT name, cost_kobo FROM send_providers"
                         ).fetchall()}
            out["spent_kobo"] = sum(
                (1 if self._port is not None else int(r["c"]))
                * costs.get(r["provider"], 0)
                for r in rows if r["status"] == "sent")
            return out
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.campaign_report", e)
            return out

    def spam_check_template(self, name: str) -> dict:
        """Spam-score a saved template before it goes anywhere. Never raises."""
        try:
            tpl = self.get_template(name or "")
            if not tpl:
                return {"score": 0, "verdict": "template not found", "issues": []}
            return spam_score(tpl.get("subject", ""), tpl.get("body", ""))
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.spam_check_template", e)
            return {"score": 0, "verdict": "unknown", "issues": []}

    # ── suppression (unsubscribe / complaints — distinct from bounces) ──

    def unsubscribe(self, address: str, reason: str = "unsubscribe",
                    detail: str = "") -> bool:
        """One-click unsubscribe / complaint handling. Immediate, permanent.
        Never raises."""
        try:
            address = (address or "").strip().lower()
            reason = (reason or "unsubscribe").strip().lower()
            if not address or "@" not in address \
                    or (self._db is None and self._port is None):
                return False
            if reason not in ("unsubscribe", "complaint", "manual"):
                reason = "unsubscribe"
            if self._port is not None:
                self._qt("send_suppressions").put({
                    "address": address, "reason": reason,
                    "detail": (detail or "")[:200], "at": time.time()})
                return True
            self._db.execute(
                "INSERT OR REPLACE INTO send_suppressions"
                " (address, reason, detail, at) VALUES (?, ?, ?, ?)",
                (address, reason, (detail or "")[:200], time.time()))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.unsubscribe", e)
            return False

    def is_suppressed(self, address: str) -> bool:
        try:
            if self._db is None and self._port is None:
                return False
            if self._port is not None:
                return self._qt("send_suppressions").get(
                    (address or "").strip().lower()) is not None
            row = self._db.execute(
                "SELECT 1 FROM send_suppressions WHERE address = ?",
                ((address or "").strip().lower(),)).fetchone()
            return row is not None
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.is_suppressed", e)
            return False

    def suppression_count(self) -> int:
        try:
            if self._db is None and self._port is None:
                return 0
            if self._port is not None:
                return self._qt("send_suppressions").count()
            row = self._db.execute("SELECT COUNT(*) c FROM send_suppressions").fetchone()
            return int(row["c"]) if row else 0
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.suppression_count", e)
            return 0

    # ── warmup enforcement ──

    def set_warmup(self, provider: str, daily_target: int) -> bool:
        """Start/restart a warmup ramp for a provider. Never raises."""
        try:
            provider = (provider or "").strip()
            if not provider or (self._db is None and self._port is None):
                return False
            if self._port is not None:
                self._qt("send_warmup").put({
                    "provider": provider, "day0": time.time(),
                    "daily_target": max(10, int(daily_target or 100))})
                return True
            self._db.execute(
                "INSERT OR REPLACE INTO send_warmup (provider, day0, daily_target)"
                " VALUES (?, ?, ?)",
                (provider, time.time(), max(10, int(daily_target or 100))))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.set_warmup", e)
            return False

    def warmup_status(self, provider: str) -> dict:
        """Today's warmup cap for a provider ({} = no warmup). Never raises."""
        try:
            if self._db is None and self._port is None:
                return {}
            if self._port is not None:
                row = self._qt("send_warmup").get((provider or "").strip())
            else:
                row = self._db.execute(
                    "SELECT * FROM send_warmup WHERE provider = ?",
                    ((provider or "").strip(),)).fetchone()
            if not row:
                return {}
            day = int((time.time() - float(row["day0"])) / 86400.0) + 1
            plan = warmup_plan(int(row["daily_target"]), 21)
            cap = plan[min(day, len(plan)) - 1]["cap"] if plan else int(row["daily_target"])
            if self._port is not None:
                used = sum(
                    1 for r in self._qt("send_log").find(
                        {"provider": row["provider"], "event": "sent"})
                    if float(r["at"]) >= time.time() - 86400.0)
            else:
                sent_today = self._db.execute(
                    "SELECT COUNT(*) c FROM send_log WHERE provider = ?"
                    " AND event = 'sent' AND at >= ?",
                    (row["provider"], time.time() - 86400.0)).fetchone()
                used = int(sent_today["c"]) if sent_today else 0
            return {"provider": row["provider"], "day": day, "cap": cap,
                    "used": used, "remaining": max(0, cap - used)}
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.warmup_status", e)
            return {}

    def _warmup_ok(self, provider: str) -> bool:
        try:
            st = self.warmup_status(provider)
            if not st:
                return True
            return st.get("remaining", 1) > 0
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine._warmup_ok", e)
            return True

    # ── bounces / blocklist ──

    def process_bounce(self, dsn_text: str) -> dict:
        """Parse a DSN; hard bounces disable the address. Never raises."""
        result = {"addresses": [], "kind": "unknown", "blocklisted": []}
        try:
            parsed = parse_dsn(dsn_text)
            result.update(parsed)
            for addr in parsed["addresses"]:
                if parsed["kind"] == "hard":
                    self.blocklist(addr, f"hard bounce ({parsed['code']})")
                    result["blocklisted"].append(addr)
            return result
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.process_bounce", e)
            return result

    def blocklist(self, address: str, reason: str = "") -> bool:
        try:
            address = (address or "").strip().lower()
            if not address or (self._db is None and self._port is None):
                return False
            if self._port is not None:
                self._qt("send_blocklist").insert_ignore({
                    "address": address, "reason": (reason or "")[:200],
                    "at": time.time()})
                return True
            self._db.execute(
                "INSERT OR IGNORE INTO send_blocklist (address, reason, at)"
                " VALUES (?, ?, ?)",
                (address, (reason or "")[:200], time.time()))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.blocklist", e)
            return False

    def unblocklist(self, address: str) -> bool:
        try:
            if self._db is None and self._port is None:
                return False
            if self._port is not None:
                self._qt("send_blocklist").delete(
                    (address or "").strip().lower())
                return True
            self._db.execute("DELETE FROM send_blocklist WHERE address = ?",
                             ((address or "").strip().lower(),))
            self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.unblocklist", e)
            return False

    def is_blocklisted(self, address: str) -> bool:
        try:
            if self._db is None and self._port is None:
                return False
            if self._port is not None:
                return self._qt("send_blocklist").get(
                    (address or "").strip().lower()) is not None
            row = self._db.execute(
                "SELECT 1 FROM send_blocklist WHERE address = ?",
                ((address or "").strip().lower(),)).fetchone()
            return row is not None
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.is_blocklisted", e)
            return False

    # ── status ──

    def status(self) -> dict:
        """Queue stats, provider health, spend. Never raises."""
        out: dict = {"queued": 0, "sent_24h": 0, "failed": 0, "dead": 0,
                     "blocklisted": 0, "suppressed": 0, "spent_24h_kobo": 0,
                     "providers": []}
        try:
            if self._db is None and self._port is None:
                return out
            day = time.time() - 86400.0
            for s in ("queued", "failed", "dead"):
                out[s] = self.queue_depth(s)
            if self._port is not None:
                out["sent_24h"] = sum(
                    1 for r in self._qt("send_log").find({"event": "sent"})
                    if float(r["at"]) >= day)
                out["blocklisted"] = self._qt("send_blocklist").count()
                out["suppressed"] = self.suppression_count()
                out["spent_24h_kobo"] = sum(
                    int(r["cost_kobo"] or 0)
                    for r in self._qt("send_costs").all()
                    if float(r["at"]) >= day)
            else:
                row = self._db.execute(
                    "SELECT COUNT(*) c FROM send_log WHERE event = 'sent'"
                    " AND at >= ?", (day,)).fetchone()
                out["sent_24h"] = int(row["c"]) if row else 0
                row = self._db.execute(
                    "SELECT COUNT(*) c FROM send_blocklist").fetchone()
                out["blocklisted"] = int(row["c"]) if row else 0
                out["suppressed"] = self.suppression_count()
                row = self._db.execute(
                    "SELECT COALESCE(SUM(cost_kobo), 0) s FROM send_costs"
                    " WHERE at >= ?", (day,)).fetchone()
                out["spent_24h_kobo"] = int(row["s"] or 0) if row else 0
            out["providers"] = self.list_providers()
            return out
        except Exception as e:  # noqa: BLE001
            record_failure("send.SendEngine.status", e)
            return out


def get_engine(db_path: str = "") -> SendEngine:
    """Process-local singleton. Never raises."""
    global _ENGINE
    try:
        if _ENGINE is None:
            _ENGINE = SendEngine(db_path)
        return _ENGINE
    except Exception as e:  # noqa: BLE001
        record_failure("send.get_engine", e)
        return SendEngine(db_path)


_ENGINE: SendEngine | None = None


# ── chat ──────────────────────────────────────────────────────────────────

SEND_CRON_ACTION = "marketing.send.process_due"


def _usage() -> str:
    return ("usage:\n"
            "  /send template add <name> | <body> — save a template\n"
            "  /send template list — list templates\n"
            "  /send variant add <template> <name> | <body> — A/B variant\n"
            "  /send provider add <name> <kind> [rate/min] — register a provider\n"
            "  /send campaign <template> to <a@b.com, c@d.com> [via <provider>] — enqueue\n"
            "  /send ab <template> <v1,v2> to <a@b.com, c@d.com> via <provider> — A/B test\n"
            "  /send abreport <campaign_id> — per-variant results + winner\n"
            "  /send seed <template> to <a@b.com> via <provider> — test send first\n"
            "  /send spamcheck <template> — spam-score before bulk\n"
            "  /send deliver <domain> — SPF/DMARC deliverability audit\n"
            "  /send warmup <provider> <daily_target> — start a warmup ramp\n"
            "  /send process — drain the queue now\n"
            "  /send queue — pending sends\n"
            "  /send status — providers, blocklist, suppression, spend\n"
            "  /send bounce <address> — manually blocklist (hard bounce)\n"
            "  /send unsub <address> — one-click unsubscribe (suppression)\n"
            "kinds: smtp, ses, twilio, whatsapp, telegram, mock (test).")


def control_send(tail: str, context=None, chat=None, **kwargs) -> str:
    """/send — self-hosted send layer. Owner-only; never raises."""
    try:
        eng = kwargs.get("engine") or get_engine()
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("template"):
            body = rest[8:].strip()
            if body.lower().startswith("add"):
                rest2 = body[3:].strip()
                if "|" not in rest2:
                    return "usage: /send template add <name> | <body>"
                name, tpl_body = rest2.split("|", 1)
                ok = eng.save_template(name.strip(), tpl_body.strip())
                return f"template '{name.strip()}' saved." if ok else "couldn't save that template."
            if body.lower().startswith("list"):
                tpls = eng.list_templates()
                if not tpls:
                    return "no templates yet — /send template add <name> | <body>"
                return "templates:\n" + "\n".join(f"• {t['name']}" for t in tpls)
            return _usage()

        if low.startswith("provider"):
            body = rest[8:].strip()
            if body.lower().startswith("add"):
                parts = body[3:].strip().split()
                if len(parts) < 2:
                    return "usage: /send provider add <name> <kind> [rate/min]"
                pname, pkind = parts[0], parts[1]
                rate = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30
                ok = eng.register_provider(pname, kind=pkind, rate_per_minute=rate)
                return (f"provider '{pname}' ({pkind}, {rate}/min) registered."
                        if ok else "couldn't register that provider.")
            return _usage()

        if low.startswith("campaign"):
            body = rest[8:].strip()
            # /send campaign <template> to <a, b> [via <provider>]
            m = re.match(r"(?P<tpl>\S+)\s+to\s+(?P<recips>[^;]+?)(?:\s+via\s+(?P<prov>\S+))?\s*$",
                         body, re.IGNORECASE)
            if not m:
                return "usage: /send campaign <template> to <a@b.com, c@d.com> [via <provider>]"
            tpl_name = m.group("tpl")
            recips = [r.strip() for r in m.group("recips").split(",") if r.strip()]
            prov = m.group("prov") or ""
            if not recips:
                return "no recipients — give me comma-separated addresses/numbers."
            if not prov:
                provs = eng.list_providers()
                prov = provs[0]["name"] if provs else ""
            if not prov:
                return "no providers registered — /send provider add <name> <kind> first."
            est = eng.estimate_campaign(len(recips), prov)
            queued = sum(1 for r in recips if eng.enqueue(r, tpl_name, {}, prov))
            return (f"📨 campaign queued: {queued}/{len(recips)} via {prov}.\n"
                    f"{est['text']}\n"
                    f"/send process to start sending.")

        if low.startswith("process"):
            stats = eng.process()
            return (f"📨 processed — sent {stats['sent']}, failed {stats['failed']}, "
                    f"deferred {stats['deferred']}, blocked {stats['blocked']}.")

        if low.startswith("queue"):
            n = eng.queue_depth()
            return f"📥 {n} send{'s' if n != 1 else ''} waiting in the queue."

        if low.startswith("status"):
            st = eng.status()
            lines = [f"📊 send status — queued {st['queued']} · sent(24h) {st['sent_24h']} · "
                     f"failed {st['failed']} · dead {st['dead']} · "
                     f"blocklisted {st['blocklisted']} · suppressed {st['suppressed']} · "
                     f"spent(24h) ₦{st['spent_24h_kobo']/100:,.0f}"]
            for p in st["providers"]:
                lines.append(f"• {p['name']} ({p['kind']}, {p['rate_per_minute']}/min, "
                             f"{'on' if p['enabled'] else 'off'})")
            return "\n".join(lines)

        if low.startswith("bounce"):
            addr = rest[6:].strip()
            if not addr:
                return "usage: /send bounce <address>"
            eng.blocklist(addr, "manual hard bounce")
            return f"{addr} blocklisted — no more sends to it."

        if low.startswith("unsub"):
            addr = rest[5:].strip()
            if not addr:
                return "usage: /send unsub <address>"
            ok = eng.unsubscribe(addr)
            return (f"🚫 {addr} unsubscribed — suppressed immediately, "
                    f"permanently." if ok else "couldn't suppress that address.")

        if low.startswith("variant"):
            body = rest[7:].strip()
            if body.lower().startswith("add"):
                rest2 = body[3:].strip()
                if "|" not in rest2:
                    return "usage: /send variant add <template> <name> | <body>"
                head, vbody = rest2.split("|", 1)
                parts = head.strip().split(None, 1)
                if len(parts) < 2:
                    return "usage: /send variant add <template> <name> | <body>"
                ok = eng.save_variant(parts[0], parts[1], vbody.strip())
                return (f"variant '{parts[1]}' of '{parts[0]}' saved."
                        if ok else "couldn't save that variant.")
            if body.lower().startswith("list"):
                vs = eng.list_variants(body[4:].strip())
                if not vs:
                    return "no variants yet."
                return "variants:\n" + "\n".join(
                    f"• {v['template']} / {v['variant_name']}" for v in vs)
            return _usage()

        if low.startswith("abreport"):
            cid = rest[8:].strip()
            if not cid:
                return "usage: /send abreport <campaign_id>"
            rep = eng.ab_report(cid)
            if not rep.get("variants"):
                return f"no data for campaign {cid}."
            lines = [f"🧪 A/B report — {cid}"]
            for v, s in sorted(rep["variants"].items()):
                lines.append(
                    f"• {v}: sent {s.get('sent', 0)} · failed {s.get('failed', 0)} · "
                    f"fail rate {s.get('fail_rate', 0):.1%}")
            win = eng.ab_winner(cid)
            if win.get("winner"):
                lines.append(f"🏆 winner: {win['winner']} ({win.get('reason', '')})")
            return "\n".join(lines)

        if low.startswith("ab"):
            # /send ab <template> <v1,v2> to <a,b> via <provider>
            m = re.match(r"(?P<tpl>\S+)\s+(?P<vars>\S+)\s+to\s+(?P<recips>[^;]+?)\s+via\s+(?P<prov>\S+)\s*$",
                         rest[2:].strip(), re.IGNORECASE)
            if not m:
                return "usage: /send ab <template> <v1,v2> to <a@b.com, c@d.com> via <provider>"
            variants = [v.strip() for v in m.group("vars").split(",") if v.strip()]
            recips = [r.strip() for r in m.group("recips").split(",") if r.strip()]
            res = eng.create_ab_campaign("chat-ab", m.group("tpl"), variants,
                                         recips, m.group("prov"))
            if not res.get("campaign_id"):
                return "couldn't start that A/B test — check template/variants."
            pv = ", ".join(f"{k}: {v}" for k, v in res["per_variant"].items())
            return (f"🧪 A/B test {res['campaign_id']} started — {pv}.\n"
                    f"/send abreport {res['campaign_id']} for results.")

        if low.startswith("seed"):
            m = re.match(r"(?P<tpl>\S+)\s+to\s+(?P<recips>[^;]+?)\s+via\s+(?P<prov>\S+)\s*$",
                         rest[4:].strip(), re.IGNORECASE)
            if not m:
                return "usage: /send seed <template> to <a@b.com> via <provider>"
            recips = [r.strip() for r in m.group("recips").split(",") if r.strip()]
            res = eng.send_seed(m.group("tpl"), recips, m.group("prov"))
            return (f"🌱 seed test sent to {res['seeded']}/{res['of']} — "
                    f"check the inbox, then launch the campaign.")

        if low.startswith("spamcheck"):
            name = rest[9:].strip()
            if not name:
                return "usage: /send spamcheck <template>"
            sc = eng.spam_check_template(name)
            lines = [f"🔍 spam score for '{name}': {sc['score']}/100 — {sc['verdict']}"]
            for i in sc.get("issues", [])[:6]:
                lines.append(f"  • {i}")
            if sc["score"] < 30:
                lines.append("clean enough for bulk. 🌱 send a seed first anyway.")
            return "\n".join(lines)

        if low.startswith("deliver"):
            domain = rest[7:].strip()
            if not domain:
                return "usage: /send deliver <domain>"
            res = deliverability_check(domain)
            lines = [f"📬 deliverability — {res['domain']}  "
                     f"grade {res['grade']} ({res['score']}/100)"]
            for c in res.get("checks", []):
                icon = "✅" if c.get("ok") else "❌" if c.get("ok") is False else "❔"
                lines.append(f"{icon} {c['name']}: {c.get('detail', '')}")
            lines.append("fix DNS first — authentication beats content tweaks.")
            return "\n".join(lines)

        if low.startswith("warmup"):
            parts = rest[6:].strip().split()
            if len(parts) < 2:
                return "usage: /send warmup <provider> <daily_target>"
            try:
                target = int(parts[1])
            except ValueError:
                return "usage: /send warmup <provider> <daily_target>"
            ok = eng.set_warmup(parts[0], target)
            if not ok:
                return "couldn't start that warmup."
            plan = warmup_plan(target)
            preview = ", ".join(f"d{p['day']}:{p['cap']}" for p in plan[:5])
            return (f"🔥 warmup started for {parts[0]} → {target}/day over 21 days.\n"
                    f"ramp: {preview} …\n"
                    f"the engine caps daily sends until the ramp completes.")

        return _usage()
    except Exception:  # noqa: BLE001
        _log.warning("send control failed", exc_info=True)
        return "send hit a snag — try again."
