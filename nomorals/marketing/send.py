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

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

try:
    from ..core.profile import resolve_profile as _resolve_profile
except Exception:  # noqa: BLE001 — profile is a nicety, not a requirement
    _resolve_profile = None  # type: ignore[assignment]

try:
    from ..social.whatsapp_cost import CostTracker as _CostTracker
except Exception:  # noqa: BLE001
    _CostTracker = None  # type: ignore[assignment]


def _default_db() -> str:
    try:
        home = __import__("os").path.expanduser("~")
        d = __import__("os").path.join(home, ".nomorals", "marketing")
        __import__("os").makedirs(d, exist_ok=True)
        return __import__("os").path.join(d, "send_queue.db")
    except Exception:  # noqa: BLE001
        return ":memory:"


def _profile_kind(profile: str = "") -> str:
    if profile:
        return profile.strip().lower()
    try:
        if _resolve_profile is not None:
            return (_resolve_profile().kind or "pc").lower()
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return result


class SendEngine:
    """Self-hosted bulk send engine: queue, throttle, retry, bounce.

    Profile-gated batch sizes; every method never raises.
    """

    def __init__(self, db_path: str = "", *, profile: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(_CREATE_SQL)
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("send engine DB unavailable", exc_info=True)
            self._db = None
        kind = _profile_kind(profile)
        self.profile = kind
        self.batch_size = PROFILE_BATCH.get(kind, PROFILE_BATCH["pc"])
        self._callables: dict[str, object] = {}
        self._tracker = None
        try:
            if _CostTracker is not None:
                self._tracker = _CostTracker()
        except Exception:  # noqa: BLE001
            self._tracker = None
        # sweep migration: campaign tracking columns on older queue tables
        try:
            if self._db is not None:
                for _col, _ddl in (("campaign_id", "TEXT DEFAULT ''"),
                                   ("variant", "TEXT DEFAULT ''")):
                    try:
                        self._db.execute(
                            f"ALTER TABLE send_queue ADD COLUMN {_col} {_ddl}")
                    except Exception:  # noqa: BLE001 — already there
                        pass
                self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    # ── templates ──

    def save_template(self, name: str, body: str, subject: str = "") -> bool:
        """Save/update a template. Never raises."""
        try:
            name = (name or "").strip()
            if not name or self._db is None:
                return False
            self._db.execute(
                "INSERT INTO send_templates (name, body, subject, updated_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET"
                " body=excluded.body, subject=excluded.subject,"
                " updated_at=excluded.updated_at",
                (name, body or "", subject or "", time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_templates(self) -> list[dict]:
        try:
            if self._db is None:
                return []
            return [dict(r) for r in self._db.execute(
                "SELECT name, subject, updated_at FROM send_templates ORDER BY name")]
        except Exception:  # noqa: BLE001
            return []

    def get_template(self, name: str) -> dict | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM send_templates WHERE name = ?",
                ((name or "").strip(),)).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
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
            if not name or self._db is None:
                return False
            cost = COST_PER_MSG_KOBO.get(kind, 0)
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
        except Exception:  # noqa: BLE001
            return False

    def list_providers(self) -> list[dict]:
        try:
            if self._db is None:
                return []
            return [dict(r) for r in self._db.execute(
                "SELECT * FROM send_providers ORDER BY name")]
        except Exception:  # noqa: BLE001
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
            if not to or not template or not provider or self._db is None:
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
        except Exception:  # noqa: BLE001
            return ""

    def queue_depth(self, status: str = "queued") -> int:
        try:
            if self._db is None:
                return 0
            row = self._db.execute(
                "SELECT COUNT(*) c FROM send_queue WHERE status = ?",
                (status,)).fetchone()
            return int(row["c"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── rate limiting ──

    def _rate_ok(self, provider: str, limit: int) -> bool:
        """Sliding-window: < limit sends in the last 60s for this provider."""
        try:
            if self._db is None:
                return True
            cutoff = time.time() - 60.0
            row = self._db.execute(
                "SELECT COUNT(*) c FROM send_log"
                " WHERE provider = ? AND event = 'sent' AND at >= ?",
                (provider, cutoff)).fetchone()
            return int(row["c"]) < max(1, limit)
        except Exception:  # noqa: BLE001
            return True

    # ── processing ──

    def process(self, limit: int = 0) -> dict:
        """Drain the queue: render → rate-limit → cost-check → send → retry.

        Returns ``{sent, failed, deferred, blocked}``. Never raises.
        """
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
            except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            pass

    def _log(self, send_id: str, provider: str, event: str,
             detail: str = "") -> None:
        try:
            self._db.execute(
                "INSERT INTO send_log (send_id, provider, event, detail, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (send_id, provider, event, detail[:300], time.time()))
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    # ── cost routing (#68) ──

    def _cost_ok(self, prov: sqlite3.Row) -> bool:
        """Budget gate: True when we may spend this provider's per-msg cost."""
        try:
            if self._tracker is None:
                return True
            cost = int(prov["cost_kobo"] or 0)
            if cost <= 0:
                return True
            may_send, _spent, _budget = self._tracker.check_budget("marketing")
            return bool(may_send)
        except Exception:  # noqa: BLE001
            return True

    def _record_cost(self, prov: sqlite3.Row) -> None:
        try:
            cost = int(prov["cost_kobo"] or 0)
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
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    def estimate_campaign(self, count: int, provider: str = "") -> dict:
        """Price a campaign before the owner approves it. Never raises."""
        try:
            count = max(0, int(count or 0))
            per = 0
            if provider and self._db is not None:
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
        except Exception:  # noqa: BLE001
            return {"count": 0, "per_kobo": 0, "total_kobo": 0, "text": ""}

    # ── A/B testing ──

    def save_variant(self, template: str, variant_name: str,
                     body: str, subject: str = "") -> bool:
        """Save a named variant of a template for A/B tests. Never raises."""
        try:
            template = (template or "").strip()
            variant_name = (variant_name or "").strip()
            if not template or not variant_name or self._db is None:
                return False
            vid = f"var_{template}_{variant_name}".replace(" ", "_")[:80]
            self._db.execute(
                "INSERT OR REPLACE INTO send_variants "
                "(variant_id, template, variant_name, body, subject, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (vid, template, variant_name, body or "", subject or "",
                 time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_variants(self, template: str = "") -> list[dict]:
        try:
            if self._db is None:
                return []
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
        except Exception:  # noqa: BLE001
            return []

    def get_variant(self, template: str, variant_name: str) -> dict | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM send_variants WHERE template = ? AND variant_name = ?",
                ((template or "").strip(), (variant_name or "").strip())).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return {"campaign_id": "", "per_variant": {}}

    def ab_report(self, campaign_id: str) -> dict:
        """Per-variant funnel for an A/B campaign. Never raises."""
        out: dict = {"campaign_id": campaign_id or "", "variants": {}}
        try:
            if self._db is None or not campaign_id:
                return out
            rows = self._db.execute(
                "SELECT variant, status, COUNT(*) c FROM send_queue"
                " WHERE campaign_id = ? GROUP BY variant, status",
                (campaign_id,)).fetchall()
            for r in rows:
                v = out["variants"].setdefault(r["variant"] or "control",
                                               {"sent": 0, "failed": 0,
                                                "dead": 0, "queued": 0,
                                                "blocked": 0})
                v[r["status"]] = v.get(r["status"], 0) + int(r["c"])
            for v, stats in out["variants"].items():
                done = stats.get("sent", 0) + stats.get("failed", 0) + stats.get("dead", 0)
                stats["fail_rate"] = round(
                    (stats.get("failed", 0) + stats.get("dead", 0)) / max(1, done), 3)
            return out
        except Exception:  # noqa: BLE001
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
            if self._db is not None:
                try:
                    for r in self._db.execute(
                            "SELECT q.variant, COUNT(*) c FROM send_log l"
                            " JOIN send_queue q ON q.send_id = l.send_id"
                            " WHERE q.campaign_id = ? AND l.event = 'engaged'"
                            " GROUP BY q.variant", (campaign_id,)):
                        eng[r["variant"] or "control"] = int(r["c"])
                except Exception:  # noqa: BLE001
                    pass
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return {"seeded": 0, "of": 0}

    def campaign_report(self, campaign_id: str) -> dict:
        """Funnel: queued → sent → failed → dead → blocked, per provider.
        Never raises."""
        out: dict = {"campaign_id": campaign_id or "", "total": 0,
                     "by_status": {}, "by_provider": {}, "spent_kobo": 0}
        try:
            if self._db is None or not campaign_id:
                return out
            rows = self._db.execute(
                "SELECT status, provider, COUNT(*) c FROM send_queue"
                " WHERE campaign_id = ? GROUP BY status, provider",
                (campaign_id,)).fetchall()
            for r in rows:
                c = int(r["c"])
                out["total"] += c
                out["by_status"][r["status"]] = out["by_status"].get(r["status"], 0) + c
                p = out["by_provider"].setdefault(r["provider"], {})
                p[r["status"]] = p.get(r["status"], 0) + c
            return out
        except Exception:  # noqa: BLE001
            return out

    def spam_check_template(self, name: str) -> dict:
        """Spam-score a saved template before it goes anywhere. Never raises."""
        try:
            tpl = self.get_template(name or "")
            if not tpl:
                return {"score": 0, "verdict": "template not found", "issues": []}
            return spam_score(tpl.get("subject", ""), tpl.get("body", ""))
        except Exception:  # noqa: BLE001
            return {"score": 0, "verdict": "unknown", "issues": []}

    # ── suppression (unsubscribe / complaints — distinct from bounces) ──

    def unsubscribe(self, address: str, reason: str = "unsubscribe",
                    detail: str = "") -> bool:
        """One-click unsubscribe / complaint handling. Immediate, permanent.
        Never raises."""
        try:
            address = (address or "").strip().lower()
            reason = (reason or "unsubscribe").strip().lower()
            if not address or "@" not in address or self._db is None:
                return False
            if reason not in ("unsubscribe", "complaint", "manual"):
                reason = "unsubscribe"
            self._db.execute(
                "INSERT OR REPLACE INTO send_suppressions"
                " (address, reason, detail, at) VALUES (?, ?, ?, ?)",
                (address, reason, (detail or "")[:200], time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def is_suppressed(self, address: str) -> bool:
        try:
            if self._db is None:
                return False
            row = self._db.execute(
                "SELECT 1 FROM send_suppressions WHERE address = ?",
                ((address or "").strip().lower(),)).fetchone()
            return row is not None
        except Exception:  # noqa: BLE001
            return False

    def suppression_count(self) -> int:
        try:
            if self._db is None:
                return 0
            row = self._db.execute("SELECT COUNT(*) c FROM send_suppressions").fetchone()
            return int(row["c"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── warmup enforcement ──

    def set_warmup(self, provider: str, daily_target: int) -> bool:
        """Start/restart a warmup ramp for a provider. Never raises."""
        try:
            provider = (provider or "").strip()
            if not provider or self._db is None:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO send_warmup (provider, day0, daily_target)"
                " VALUES (?, ?, ?)",
                (provider, time.time(), max(10, int(daily_target or 100))))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def warmup_status(self, provider: str) -> dict:
        """Today's warmup cap for a provider ({} = no warmup). Never raises."""
        try:
            if self._db is None:
                return {}
            row = self._db.execute(
                "SELECT * FROM send_warmup WHERE provider = ?",
                ((provider or "").strip(),)).fetchone()
            if not row:
                return {}
            day = int((time.time() - float(row["day0"])) / 86400.0) + 1
            plan = warmup_plan(int(row["daily_target"]), 21)
            cap = plan[min(day, len(plan)) - 1]["cap"] if plan else int(row["daily_target"])
            sent_today = self._db.execute(
                "SELECT COUNT(*) c FROM send_log WHERE provider = ?"
                " AND event = 'sent' AND at >= ?",
                (row["provider"], time.time() - 86400.0)).fetchone()
            used = int(sent_today["c"]) if sent_today else 0
            return {"provider": row["provider"], "day": day, "cap": cap,
                    "used": used, "remaining": max(0, cap - used)}
        except Exception:  # noqa: BLE001
            return {}

    def _warmup_ok(self, provider: str) -> bool:
        try:
            st = self.warmup_status(provider)
            if not st:
                return True
            return st.get("remaining", 1) > 0
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return result

    def blocklist(self, address: str, reason: str = "") -> bool:
        try:
            address = (address or "").strip().lower()
            if not address or self._db is None:
                return False
            self._db.execute(
                "INSERT OR IGNORE INTO send_blocklist (address, reason, at)"
                " VALUES (?, ?, ?)",
                (address, (reason or "")[:200], time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def unblocklist(self, address: str) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute("DELETE FROM send_blocklist WHERE address = ?",
                             ((address or "").strip().lower(),))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def is_blocklisted(self, address: str) -> bool:
        try:
            if self._db is None:
                return False
            row = self._db.execute(
                "SELECT 1 FROM send_blocklist WHERE address = ?",
                ((address or "").strip().lower(),)).fetchone()
            return row is not None
        except Exception:  # noqa: BLE001
            return False

    # ── status ──

    def status(self) -> dict:
        """Queue stats, provider health, spend. Never raises."""
        out: dict = {"queued": 0, "sent_24h": 0, "failed": 0, "dead": 0,
                     "blocklisted": 0, "suppressed": 0, "spent_24h_kobo": 0,
                     "providers": []}
        try:
            if self._db is None:
                return out
            day = time.time() - 86400.0
            for s in ("queued", "failed", "dead"):
                out[s] = self.queue_depth(s)
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
        except Exception:  # noqa: BLE001
            return out


def get_engine(db_path: str = "") -> SendEngine:
    """Process-local singleton. Never raises."""
    global _ENGINE
    try:
        if _ENGINE is None:
            _ENGINE = SendEngine(db_path)
        return _ENGINE
    except Exception:  # noqa: BLE001
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
