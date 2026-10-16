"""Phone-friendly web client for :mod:`nomorals.stream`.

Pure client + HTML render — **no server change**. The existing
:class:`~nomorals.stream.server.StreamServer` keeps serving
``GET /stream`` (SSE) and ``GET /health`` exactly as before; this
module is the other side of that wire, shaped for a phone browser:

* :class:`TimedGrant` — a ``stream:read`` token with an expiry.
  Tokens are HMAC-signed (``HMAC-SHA256(secret, payload)``); the
  client refuses to use an expired token and the verifier rejects
  expired tokens, tampered tokens, and tokens for another capability.
  A token is minted with :func:`mint_stream_grant` and checked with
  :func:`verify_stream_token`. There is no server-side session to
  revoke — expiry *is* the revocation (keep TTLs short, minutes).
* :func:`parse_sse_text` / :class:`SSEClient` — a stdlib SSE client:
  parses ``event:`` / ``data:`` / ``id:`` / ``retry:`` frames
  (multi-line data, ``: comment`` heartbeats skipped), resumes with
  ``Last-Event-ID``, and sends the grant as
  ``Authorization: Bearer <token>``.
* Live autonomy view — :func:`live_autonomy_view` turns raw mission
  dicts into one JSON view (current missions + status + cost meter);
  :func:`render_mission_board`, :func:`render_cost_meter`, and
  :func:`render_phone_page` render that view as thumb-friendly HTML
  (viewport meta, large tap targets, no JavaScript required for the
  static board; a tiny ``EventSource`` snippet wires the live stream
  when the page is served with a valid grant).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

__all__ = [
    "COST_WARN_PCT",
    "DEFAULT_GRANT_TTL_S",
    "SSEClient",
    "TimedGrant",
    "TokenError",
    "TokenExpired",
    "TokenInvalid",
    "cost_meter",
    "live_autonomy_view",
    "mint_stream_grant",
    "parse_sse_text",
    "render_cost_meter",
    "render_mission_board",
    "render_phone_page",
    "verify_stream_token",
]

#: Capability this module's grants cover. Nothing else is granted.
STREAM_READ_CAPABILITY = "stream:read"

#: Default grant lifetime (5 minutes — expiry is the revocation).
DEFAULT_GRANT_TTL_S = 300.0

#: Cost meter turns amber at/above this fraction of budget.
COST_WARN_PCT = 0.8


class TokenError(ValueError):
    """Base class for stream-grant token failures."""


class TokenExpired(TokenError):
    """The token's expiry has passed — it must be rejected."""


class TokenInvalid(TokenError):
    """Malformed, tampered, or wrong-capability token."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


@dataclass(frozen=True)
class TimedGrant:
    """A time-boxed ``stream:read`` grant.

    ``token`` is ``<payload>.<signature>`` where payload is
    base64url JSON ``{cap, exp, iat, scope}`` and signature is
    ``HMAC-SHA256(secret, payload)``. The secret never travels in the
    token; verification recomputes the signature.
    """

    capability: str = STREAM_READ_CAPABILITY
    token: str = ""
    expires_at: float = 0.0
    issued_at: float = 0.0
    scope: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "token": self.token,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "scope": self.scope,
        }


def mint_stream_grant(secret: str, *, ttl_s: float = DEFAULT_GRANT_TTL_S,
                      scope: str = "", now: float | None = None) -> TimedGrant:
    """Mint a ``stream:read`` grant expiring ``ttl_s`` from ``now``.

    ``secret`` is the shared stream secret (server-side config — never
    embedded in the page). ``ttl_s`` must be positive; a non-positive
    TTL mints an already-expired token (rejected everywhere).
    """
    if not secret:
        raise ValueError("mint_stream_grant needs a non-empty secret")
    at = now if now is not None else time.time()
    payload = {"cap": STREAM_READ_CAPABILITY, "iat": at,
               "exp": at + float(ttl_s), "scope": scope or ""}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(str(secret).encode(), raw, hashlib.sha256).digest()
    token = f"{_b64url_encode(raw)}.{_b64url_encode(sig)}"
    return TimedGrant(capability=STREAM_READ_CAPABILITY, token=token,
                      expires_at=at + float(ttl_s), issued_at=at,
                      scope=scope or "")


def verify_stream_token(token: str, secret: str, *,
                        now: float | None = None) -> TimedGrant:
    """Verify a grant token. Raises :exc:`TokenExpired` / :exc:`TokenInvalid`.

    **Expired tokens are rejected** — the check is ``now >= exp``,
    evaluated here, not trusted from the client. Tampered payloads
    fail the HMAC comparison (constant-time). Tokens for another
    capability are rejected even when the signature is valid.
    """
    if not secret:
        raise ValueError("verify_stream_token needs a non-empty secret")
    at = now if now is not None else time.time()
    try:
        payload_b64, sig_b64 = str(token).split(".", 1)
        payload_raw = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
        payload = json.loads(payload_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenInvalid(f"malformed stream token: {exc}") from exc
    expected = hmac.new(str(secret).encode(), payload_raw,
                        hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise TokenInvalid("stream token signature mismatch")
    if not isinstance(payload, dict) or payload.get("cap") != STREAM_READ_CAPABILITY:
        raise TokenInvalid(
            f"token capability is not {STREAM_READ_CAPABILITY!r}")
    try:
        exp = float(payload["exp"])
        iat = float(payload.get("iat", exp))
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenInvalid(f"token missing expiry: {exc}") from exc
    if at >= exp:
        raise TokenExpired(
            f"stream token expired at {exp} (now {at})")
    return TimedGrant(capability=STREAM_READ_CAPABILITY, token=str(token),
                      expires_at=exp, issued_at=iat,
                      scope=str(payload.get("scope") or ""))


# ── SSE parsing / client ─────────────────────────────────────────────

def parse_sse_text(text: str) -> list[dict[str, Any]]:
    """Parse raw SSE text into ``[{event, data, id, retry}]``.

    ``: comment`` heartbeat lines are skipped. Multi-line ``data:``
    lines are joined with ``\\n``. A blank line terminates an event.
    ``data`` is JSON-decoded when it parses, else kept as a string.
    """
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {"event": None, "data": [],
                               "id": None, "retry": None}
    has_content = False

    def flush() -> None:
        nonlocal current, has_content
        if not has_content:
            return
        raw = "\n".join(current["data"])
        try:
            data: Any = json.loads(raw) if raw else ""
        except ValueError:
            data = raw
        events.append({"event": current["event"], "data": data,
                       "id": current["id"], "retry": current["retry"]})
        current = {"event": None, "data": [], "id": None, "retry": None}
        has_content = False

    for line in str(text).splitlines():
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue  # comment / heartbeat — ignored by EventSource
        if ":" in line:
            field_name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
        else:
            field_name, value = line, ""
        if field_name == "event":
            current["event"] = value or None
            has_content = True
        elif field_name == "data":
            current["data"].append(value)
            has_content = True
        elif field_name == "id":
            current["id"] = value or None
            has_content = True
        elif field_name == "retry":
            try:
                current["retry"] = int(value)
            except ValueError:
                current["retry"] = None
            has_content = True
        # Unknown fields are ignored per the WHATWG stream rules.
    flush()
    return events


class SSEClient:
    """Minimal stdlib SSE client for the stream server.

    ``grant`` must be a live :class:`TimedGrant` — an expired grant
    raises :exc:`TokenExpired` *before* any request is made, so an
    expired token never reaches the wire. ``last_event_id`` resumes
    the stream (``Last-Event-ID`` header).
    """

    def __init__(self, base_url: str, grant: TimedGrant,
                 *, timeout_s: float = 30.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.grant = grant
        self.timeout_s = timeout_s
        self.last_event_id: str | None = None

    def _check_grant(self) -> None:
        if self.grant.expired:
            raise TokenExpired(
                f"stream grant expired at {self.grant.expires_at}")

    def stream(self, *, topic: str | None = None,
               on_event: Callable[[dict[str, Any]], None] | None = None,
               max_events: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE events from ``GET /stream``.

        ``on_event`` (if given) is called for each event as it arrives;
        events are also yielded. Stops after ``max_events`` when set.
        """
        self._check_grant()
        url = f"{self.base_url}/stream"
        if topic:
            from urllib.parse import quote
            url += f"?topic={quote(topic)}"
        req = urllib.request.Request(
            url, headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.grant.token}",
                **({"Last-Event-ID": self.last_event_id}
                   if self.last_event_id else {}),
            })
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise TokenInvalid(
                f"stream server rejected request: HTTP {exc.code}") from exc
        count = 0
        buffer = ""
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for event in parse_sse_text(frame + "\n\n"):
                        if event.get("id") is not None:
                            self.last_event_id = str(event["id"])
                        if on_event is not None:
                            on_event(event)
                        yield event
                        count += 1
                        if max_events is not None and count >= max_events:
                            return
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass


# ── live autonomy view + cost meter ──────────────────────────────────

def _mission_status(mission: dict[str, Any]) -> str:
    for key in ("status", "state", "phase"):
        value = mission.get(key)
        if value:
            return str(value)
    return "unknown"


def cost_meter(spent: float = 0.0, budget: float = 0.0,
               *, currency: str = "$") -> dict[str, Any]:
    """Cost meter: spent vs budget, with warn/over flags.

    ``pct`` is ``spent / budget`` (``None`` when no budget is set —
    no budget means no meter, not 0%). ``over`` is True past 100%,
    ``warn`` at/above :data:`COST_WARN_PCT`.
    """
    spent_f, budget_f = max(0.0, float(spent or 0.0)), float(budget or 0.0)
    pct = (spent_f / budget_f) if budget_f > 0 else None
    return {
        "spent": round(spent_f, 4),
        "budget": round(budget_f, 4),
        "currency": currency,
        "pct": round(pct, 4) if pct is not None else None,
        "remaining": round(max(0.0, budget_f - spent_f), 4)
        if budget_f > 0 else None,
        "warn": bool(pct is not None and pct >= COST_WARN_PCT),
        "over": bool(pct is not None and pct > 1.0),
    }


def live_autonomy_view(missions: list[dict[str, Any]] | None = None, *,
                       spent: float = 0.0, budget: float = 0.0,
                       now: float | None = None) -> dict[str, Any]:
    """One live autonomy view: current missions + status + cost meter.

    ``missions`` are raw mission dicts (any shape — ``status`` /
    ``state`` / ``phase`` and ``name`` / ``title`` / ``id`` are read
    defensively). The view normalizes them to
    ``{id, name, status}`` and counts by status, so the phone board
    never has to know the mission schema.
    """
    at = now if now is not None else time.time()
    normalized = []
    by_status: dict[str, int] = {}
    for m in missions or []:
        if not isinstance(m, dict):
            continue
        status = _mission_status(m)
        normalized.append({
            "id": str(m.get("id") or m.get("mission_id") or ""),
            "name": str(m.get("name") or m.get("title")
                        or m.get("id") or "mission"),
            "status": status,
        })
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "ts": at,
        "missions": normalized,
        "mission_count": len(normalized),
        "by_status": by_status,
        "cost": cost_meter(spent, budget),
    }


def render_cost_meter(spent: float = 0.0, budget: float = 0.0, *,
                      currency: str = "$") -> str:
    """Cost meter as a small HTML bar (green → amber → red)."""
    meter = cost_meter(spent, budget, currency=currency)
    if meter["pct"] is None:
        return (f'<div class="cost">Cost: {html.escape(currency)}'
                f'{meter["spent"]:.2f} <span class="muted">'
                f"(no budget set)</span></div>")
    pct = min(1.0, meter["pct"])
    color = "#c0392b" if meter["over"] else (
        "#e67e22" if meter["warn"] else "#27ae60")
    return (
        f'<div class="cost"><div class="cost-row">'
        f'<span>Cost: {html.escape(currency)}{meter["spent"]:.2f} / '
        f'{html.escape(currency)}{meter["budget"]:.2f}</span>'
        f'<span class="{"over" if meter["over"] else "warn" if meter["warn"] else "ok"}">'
        f'{meter["pct"] * 100:.0f}%</span></div>'
        f'<div class="bar"><div class="fill" style="width:{pct * 100:.0f}%;'
        f'background:{color}"></div></div>'
        f'<div class="muted">remaining: {html.escape(currency)}'
        f'{meter["remaining"]:.2f}</div></div>')


def render_mission_board(missions: list[dict[str, Any]] | None = None) -> str:
    """Mission board as thumb-friendly HTML (one card per mission)."""
    view = live_autonomy_view(missions)
    if not view["missions"]:
        return ('<div class="board"><p class="muted">'
                "No missions running.</p></div>")
    cards = []
    for m in view["missions"]:
        status = html.escape(m["status"])
        cards.append(
            f'<div class="mission"><div class="m-name">'
            f'{html.escape(m["name"])}</div>'
            f'<span class="badge">{status}</span></div>')
    return '<div class="board">' + "".join(cards) + "</div>"


def render_phone_page(missions: list[dict[str, Any]] | None = None, *,
                      spent: float = 0.0, budget: float = 0.0,
                      stream_url: str = "", grant: TimedGrant | None = None,
                      title: str = "Devon — live") -> str:
    """Full phone page: autonomy view + mission board + cost meter.

    Static HTML — no JavaScript is required to read the board. When
    ``stream_url`` and a live ``grant`` are given, a small
    ``EventSource`` snippet appends live events to ``#live``; an
    expired grant is refused (the snippet is omitted and the page
    notes the grant expired) rather than wiring a dead stream.
    """
    view = live_autonomy_view(missions, spent=spent, budget=budget)
    grant_ok = grant is not None and not grant.expired
    live_note = ""
    script = ""
    if grant is not None and not grant_ok:
        live_note = ('<p class="muted">Live stream paused: '
                     "grant expired — refresh for a new grant.</p>")
    elif stream_url and grant is not None:
        script = (
            '<script>var es=new EventSource('
            + json.dumps(stream_url)
            + ');es.onmessage=function(e){var d=document.getElementById'
            + '("live");var p=document.createElement("p");p.textContent=e.data;'
            + 'd.prepend(p);};es.onerror=function(){es.close();};</script>')
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;margin:0;padding:12px;"
        "background:#111;color:#eee}h1{font-size:20px}"
        ".mission{display:flex;justify-content:space-between;align-items:center;"
        "padding:12px;margin:8px 0;background:#222;border-radius:10px}"
        ".m-name{font-size:16px}.badge{background:#333;padding:4px 10px;"
        "border-radius:999px;font-size:13px}"
        ".bar{height:10px;background:#333;border-radius:5px;overflow:hidden}"
        ".fill{height:100%}.cost{margin:12px 0}.cost-row{display:flex;"
        "justify-content:space-between}.muted{color:#999;font-size:13px}"
        ".over{color:#e74c3c}.warn{color:#e67e22}.ok{color:#2ecc71}"
        "#live p{font-size:13px;word-break:break-word}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='muted'>{view['mission_count']} mission(s) · "
        f"updated {time.strftime('%H:%M:%S', time.gmtime(view['ts']))} UTC</p>"
        + render_mission_board(missions)
        + render_cost_meter(spent, budget)
        + live_note
        + "<div id='live'></div>" + script + "</body></html>")
