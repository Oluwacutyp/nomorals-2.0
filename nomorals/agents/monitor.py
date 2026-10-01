"""MonitorAgent — watch URLs and files, alert on real change.

A durable, low-noise change detector. Each monitor watches one target:

* **url**  — fetched through the proxy-aware, robots-respecting HttpClient;
  the body is hashed (SHA-256).
* **file** — a workspace file is read and hashed.

On every ``tick()`` each *due* monitor (``interval_s`` elapsed) is re-checked.
When the hash changes the owner is alerted through the Notifier with a short
unified diff (text targets) or a size delta (binary).  A monitor that errors
3 ticks in a row is flagged once, then goes quiet again until it recovers —
no alert storms from a flaky feed.

Persistence lives in the ``monitors`` table (migration 23); nothing is lost on
restart.  Fully offline-capable: file monitors need no network at all.

    from nomorals.agents.monitor import MonitorAgent
    mon = MonitorAgent(context)
    mon.add("https://example.com/status")          # url, every 5 min
    mon.add("workspace/config.toml", interval=60)  # file, every 60 s
    changes = mon.tick()                            # run all due checks
    changes["changed"]                              # [{target, diff, ...}]

Registered as the ``monitor`` tool.  The autonomy loop calls ``tick()`` each
cycle, so watches keep running while the agent is alive.
"""

from __future__ import annotations

import difflib
import hashlib
import hmac
import json
import time
from typing import Any

from ..core.ids import new_id
from ..core.policy import Capability

__all__ = [
    "MonitorAgent",
    "register",
    # Extracted fetch/hash primitives (Prompt 03): reused by the general
    # watcher system instead of being duplicated.  MonitorAgent's methods
    # delegate to these, so its public API is unchanged.
    "fetch_url_bytes",
    "fetch_file_bytes",
    "fetch_page_bytes",
    "hash_bytes",
    "unified_content_diff",
]

#: how much previous content to keep around so a change can be diffed
_MAX_STORED_CONTENT = 100_000
#: a fetch body larger than this is hashed but not stored/diffed
_MAX_FETCH_BYTES = 1_000_000
#: consecutive errors before a "watch is down" alert (and until recovery)
_ERROR_ALERT_STREAK = 3


def _hmac_hex(secret: str, body: bytes) -> str:
    """Slack-style request signature over the exact JSON body."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# ── extracted fetch/hash primitives (Prompt 03) ─────────────────────────────
# These were MonitorAgent._fetch / _fetch_page / _diff.  They are module-level
# now so the watcher system (nomorals/agents/watchers.py) can reuse the
# exact same fetching, rendering and hashing behaviour without duplicating it.


def fetch_url_bytes(url: str, *,
                    max_bytes: int = _MAX_FETCH_BYTES) -> bytes:
    """Fetch a URL's raw body through the proxy-aware HttpClient."""
    from ..core.http import HttpClient

    client = HttpClient()
    resp = client.get(url)
    if getattr(resp, "status", 200) >= 400:
        raise RuntimeError(f"HTTP {resp.status}")
    body = getattr(resp, "body", b"") or b""
    if not body and getattr(resp, "text", ""):
        body = resp.text.encode("utf-8", "ignore")
    return body[:max_bytes]


def fetch_file_bytes(context: Any, path: str, *,
                     max_bytes: int = _MAX_FETCH_BYTES) -> bytes:
    """Read a workspace file's bytes (sandboxed through ``safe_path``)."""
    from ..tools.filesystem import safe_path

    target = safe_path(context, path, must_exist=True)
    with target.open("rb") as fh:
        return fh.read(max_bytes)


def fetch_page_bytes(target: str, *, volatile: str = "",
                     max_bytes: int = _MAX_FETCH_BYTES) -> bytes:
    """Render a page headless (DOM → markdown, boilerplate stripped) and
    return the normalized visible text.

    A change in what a visitor can ACTUALLY SEE is what hashes — not a nonce,
    a comment, or a byte of markup churn.  ``volatile`` is newline-separated
    regexes stripped from the rendered text before hashing (rotating nonces,
    timestamps, CSRF tokens in visible text); bad patterns are skipped, never
    fatal.
    """
    from ..tools.browser import BrowserSession, parse_html, node_to_markdown

    session = BrowserSession()
    try:
        result = session._fetch(target)  # status + raw html (+cookie jar)
        status = result.get("status", 200)
        html = result.get("text", "") or ""
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
    if status >= 400:
        raise RuntimeError(f"HTTP {status}")
    if not html:
        return b""
    dom = parse_html(html)
    rendered = node_to_markdown(dom)
    volatile = (volatile or "").strip()
    if volatile:
        import re as _re

        # newline-separated patterns (a comma belongs to regex
        # syntax, so it can never be the delimiter)
        for pat in volatile.splitlines():
            pat = pat.strip()
            if not pat:
                continue
            try:
                rendered = _re.sub(
                    pat, "", rendered, flags=_re.IGNORECASE)
            except _re.error:
                continue
    # collapse whitespace runs (trailing spaces, repeated blank lines)
    # so pure reflow / markup churn is not a "change"
    lines = [ln.rstrip() for ln in rendered.splitlines()]
    tight: list[str] = []
    for ln in lines:
        if ln == "" and tight and tight[-1] == "":
            continue
        tight.append(ln)
    while tight and tight[-1] == "":
        tight.pop()
    rendered = "\n".join(tight)
    return rendered.encode("utf-8", "ignore")[:max_bytes]


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def unified_content_diff(old: str, new: str, limit: int = 4000) -> str:
    """Short unified diff between two text snapshots (truncated)."""
    diff = "".join(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="before", tofile="after", lineterm=""))
    return diff[:limit] + ("\n… (truncated)" if len(diff) > limit else "")


class MonitorAgent:
    """Durable change detection over URLs and files."""

    role = "monitor"

    def __init__(self, context: Any, notifier: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.notifier = notifier
        if self.db is None:
            raise RuntimeError("MonitorAgent needs a context with a database")

    # ── persistence ─────────────────────────────────────────────────────────
    def add(
        self,
        target: str,
        *,
        kind: str = "",
        watch: str = "content",
        interval: float = 300.0,
        webhook: str = "",
        secret: str = "",
        min_gap: float = 60.0,
        auto_decode: bool = True,
        volatile: str = "",
    ) -> dict[str, Any]:
        target = (target or "").strip()
        if not target:
            raise ValueError("monitor target is empty")
        if not kind:
            kind = "url" if target.lower().startswith(("http://", "https://")) \
                else "file"
        if kind not in ("url", "file", "page"):
            raise ValueError(f"kind must be url|file|page, got {kind!r}")
        if kind == "page" and not target.lower().startswith(
                ("http://", "https://")):
            raise ValueError("a page watch needs an http(s) URL")
        watch = watch if watch in ("content", "size") else "content"
        interval = max(30.0, float(interval or 300.0))
        webhook = (webhook or "").strip()
        if webhook and not webhook.lower().startswith(("http://", "https://")):
            raise ValueError("webhook must be an http(s) URL")
        secret = (secret or "").strip()
        min_gap = max(0.0, float(min_gap))
        # volatile regexes (page watches): NEWLINE-separated (a comma is
        # part of regex syntax, e.g. {2,} — it can't be the delimiter).
        # Validate now, fail fast on a bad pattern instead of at the
        # first tick.
        volatile = (volatile or "").strip()
        if volatile:
            import re as _re

            for pat in volatile.splitlines():
                pat = pat.strip()
                if pat:
                    _re.compile(pat, flags=_re.IGNORECASE)

        existing = self.db.query_one(
            "SELECT id FROM monitors WHERE target=?", (target,))
        if existing:
            self.db.execute(
                "UPDATE monitors SET kind=?, watch=?, interval_s=?, "
                "webhook_url=?, webhook_secret=?, min_alert_gap_s=?, "
                "auto_decode=?, volatile=?, enabled=1 WHERE target=?",
                (kind, watch, interval, webhook, secret, min_gap,
                 1 if auto_decode else 0, volatile, target))
            mid = existing["id"]
            created = False
        else:
            mid = new_id()
            self.db.execute(
                "INSERT INTO monitors (id, target, kind, watch, interval_s, "
                "enabled, created_at, last_ts, webhook_url, webhook_secret, "
                "min_alert_gap_s, auto_decode, volatile) "
                "VALUES (?,?,?,?,?,1,?,0,?,?,?,?,?)",
                (mid, target, kind, watch, interval, time.time(),
                 webhook, secret, min_gap, 1 if auto_decode else 0,
                 volatile))
            created = True
        return {"id": mid, "target": target, "kind": kind, "watch": watch,
                "interval_s": interval, "webhook_url": webhook,
                "webhook_secret_set": bool(secret),
                "min_alert_gap_s": min_gap,
                "auto_decode": bool(auto_decode), "volatile": volatile,
                "created": created}

    def set_alerting(self, ref: str, *, webhook: str | None = None,
                     secret: str | None = None,
                     min_gap: float | None = None,
                     auto_decode: bool | None = None) -> dict[str, Any] | None:
        """Change delivery on a watch: where alerts go (webhook), with what
        signature (secret), how often (min_gap) and whether the response is
        auto-decoded into the OSINT graph (auto_decode)."""
        row = self._find(ref)
        if row is None:
            return None
        wh = row.get("webhook_url", "") if webhook is None else webhook.strip()
        if wh and not wh.lower().startswith(("http://", "https://")):
            raise ValueError("webhook must be an http(s) URL (or '')")
        sec = (row.get("webhook_secret", "") if secret is None
               else secret.strip())
        gap = (float(row.get("min_alert_gap_s") or 0.0) if min_gap is None
               else max(0.0, float(min_gap)))
        decode = (1 if row.get("auto_decode", 1) else 0) if auto_decode is None \
            else (1 if auto_decode else 0)
        self.db.execute(
            "UPDATE monitors SET webhook_url=?, webhook_secret=?, "
            "min_alert_gap_s=?, auto_decode=? WHERE id=?",
            (wh, sec, gap, decode, row["id"]))
        return self._row(row["id"])

    def webhook_test(self, ref: str) -> dict[str, Any] | None:
        """Fire a one-off test alert to the watch's webhook right now —
        the fastest way to verify delivery before it's needed."""
        row = self._find(ref)
        if row is None:
            return None
        if not (row.get("webhook_url") or "").strip():
            return {"ok": False, "error": "no webhook configured on "
                    f"{row['target']!r} — set one with nm monitor alert "
                    "<ref> --webhook URL"}
        now = time.time()
        result = self._fire_webhook(row, "test", {"note": "manual test"}, now)
        return {"ok": bool(result.get("ok")), "target": row["target"],
                "result": result}

    def remove(self, ref: str) -> bool:
        row = self._find(ref)
        if row is None:
            return False
        self.db.execute("DELETE FROM monitors WHERE id=?", (row["id"],))
        return True

    def set_enabled(self, ref: str, enabled: bool) -> dict[str, Any] | None:
        row = self._find(ref)
        if row is None:
            return None
        self.db.execute("UPDATE monitors SET enabled=? WHERE id=?",
                        (1 if enabled else 0, row["id"]))
        return self._row(row["id"])

    def list(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM monitors ORDER BY created_at DESC")
        return [self._public(r) for r in rows]

    def status(self) -> dict[str, Any]:
        rows = self.list()
        return {
            "total": len(rows),
            "enabled": sum(1 for r in rows if r["enabled"]),
            "monitors": rows,
        }

    def _find(self, ref: str) -> dict[str, Any] | None:
        ref = (ref or "").strip()
        if not ref:
            return None
        row = self.db.query_one("SELECT * FROM monitors WHERE id=?", (ref,))
        if row is None:
            row = self.db.query_one("SELECT * FROM monitors WHERE target=?",
                                    (ref,))
        return row

    def _row(self, mid: str) -> dict[str, Any]:
        r = self.db.query_one("SELECT * FROM monitors WHERE id=?", (mid,))
        return self._public(r) if r else {}

    @staticmethod
    def _public(r: dict[str, Any]) -> dict[str, Any]:
        out = {
            "id": r["id"],
            "target": r["target"],
            "kind": r["kind"],
            "watch": r["watch"],
            "interval_s": r["interval_s"],
            "enabled": bool(r["enabled"]),
            "created_at": r["created_at"],
            "last_ts": r["last_ts"],
            "last_size": r["last_size"],
            "last_ok": r["last_ok"],
            "last_change_ts": r["last_change_ts"],
            "error_streak": r["error_streak"],
            "webhook_url": r.get("webhook_url", "") or "",
            "webhook_secret_set": bool(r.get("webhook_secret", "")),
            "min_alert_gap_s": r.get("min_alert_gap_s", 0.0) or 0.0,
            "last_alert_ts": r.get("last_alert_ts", 0.0) or 0.0,
            "auto_decode": bool(r.get("auto_decode", 1)),
            "volatile": r.get("volatile", "") or "",
        }
        return out

    # ── the check loop ───────────────────────────────────────────────────────
    def tick(self, *, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        rows = self.db.query(
            "SELECT * FROM monitors WHERE enabled=1 "
            "AND (last_ts=0 OR last_ts + interval_s <= ?)", (now,))
        changed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        checked = 0
        for row in rows:
            checked += 1
            try:
                content = self._fetch(row)
            except Exception as exc:  # noqa: BLE001 — one bad watch must not
                #                    #  kill the whole tick
                self._note_error(row, str(exc), now=now)
                errors.append({"target": row["target"], "error": str(exc)})
                continue
            self._note_check(row, content, now, changed)
        return {"checked": checked, "changed": changed, "errors": errors,
                "now": now}

    def _fetch(self, row: dict[str, Any]) -> bytes:
        # Delegates to the module-level primitives (Prompt 03 extraction)
        # so watchers reuse the exact same behaviour.
        if row["kind"] == "file":
            return fetch_file_bytes(self.context, row["target"])
        if row["kind"] == "page":
            return fetch_page_bytes(row["target"],
                                    volatile=row.get("volatile") or "")
        return fetch_url_bytes(row["target"])

    def _fetch_page(self, row: dict[str, Any]) -> bytes:
        """Thin wrapper kept for backward compatibility; see
        :func:`fetch_page_bytes`."""
        return fetch_page_bytes(row["target"],
                                volatile=row.get("volatile") or "")

    def _note_check(self, row: dict[str, Any], content: bytes,
                    now: float, changed: list[dict[str, Any]]) -> None:
        digest = hashlib.sha256(content).hexdigest()
        prev = row["last_hash"]
        is_change = bool(prev) and digest != prev
        stored = ""
        watch = row["watch"]
        if watch == "content" and len(content) <= _MAX_STORED_CONTENT:
            try:
                stored = content.decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                stored = ""

        self.db.execute(
            "UPDATE monitors SET last_ts=?, last_hash=?, last_size=?, "
            "last_ok=1, error_streak=0, last_content=?, last_change_ts=? "
            "WHERE id=?",
            (now, digest, len(content), stored,
             now if is_change else row["last_change_ts"], row["id"]))

        if is_change:
            entry = {
                "id": row["id"],
                "target": row["target"],
                "kind": row["kind"],
                "old_size": row["last_size"],
                "new_size": len(content),
                "watch": watch,
            }
            if watch == "content" and stored and row["last_content"]:
                entry["diff"] = self._diff(row["last_content"], stored)
            # per-monitor throttling: a flapping target can still change
            # every check; the owner gets at most one alert per min_gap
            throttled = self._throttled(row, now)
            entry["throttled"] = throttled
            if not throttled:
                if self.notifier is not None:
                    try:
                        self.notifier.publish(
                            kind="monitor",
                            title=f"change: {row['target']}",
                            body=entry.get("diff") or
                            f"size {row['last_size']} → {len(content)} bytes",
                        )
                    except Exception:  # noqa: BLE001 — alerting must not
                        # break the tick
                        pass
                entry["webhook"] = self._fire_webhook(
                    row, "change", entry, now)
                self.db.execute(
                    "UPDATE monitors SET last_alert_ts=? WHERE id=?",
                    (now, row["id"]))
            changed.append(entry)

        # wave 75 pipeline: monitors auto-decode their content (baseline
        # fetch + every change) — cookies/JWTs become person + domain
        # entities in the identity graph, and the report lands in the
        # decoder archive.  Fail-soft and bounded: analysis must never
        # break the watch.  page watches are skipped: their content is
        # rendered markdown (no raw cookies/JWTs to decode).
        if row.get("auto_decode", 1) and row["kind"] != "page":
            feed = self._decode_feed(row, content, now)
            if is_change and feed:
                entry["decoded"] = feed

    def _decode_feed(self, row: dict[str, Any], content: bytes,
                     now: float) -> dict[str, Any] | None:
        """Decode a fetched response into the OSINT graph + report archive.

        Returns ``{osint, report_id, persons, domains}`` or None when the
        decode found nothing / could not run.  Never raises."""
        try:
            text = content.decode("utf-8", "ignore")
            if len(text) > 200_000:
                text = text[:200_000]
            from ..agents.osint_graph import IdentityGraph
            from ..core.decoder import analyze, save_report

            report = analyze(text, max_depth=2)
            src = f"monitor:{row['target']}"
            stats = IdentityGraph(self.context).ingest_decoder_findings(
                report, source=src)
            rid = save_report(self.db, report, source=src,
                              kind=row["kind"], input_text=text)
            if (not stats.get("persons_found")
                    and not stats.get("domains_found")):
                return None
            return {"osint": stats, "report_id": rid,
                    "persons": stats.get("persons_found", 0),
                    "domains": stats.get("domains_found", 0)}
        except Exception:  # noqa: BLE001 — enrichment, never a failure
            return None

    def _throttled(self, row: dict[str, Any], now: float) -> bool:
        gap = float(row.get("min_alert_gap_s") or 0.0)
        if gap <= 0:
            return False
        last = float(row.get("last_alert_ts") or 0.0)
        return (now - last) < gap

    def _fire_webhook(self, row: dict[str, Any], kind: str,
                      entry: dict[str, Any], now: float) -> dict[str, Any]:
        """POST the alert as JSON to the monitor's webhook URL.

        Hardened delivery (wave 75):
        * **HMAC signing** — when the watch has a secret, the body is signed
          ``X-NoMorals-Signature: sha256=<hmac-hex>`` (Slack-style), so the
          receiver can verify the alert really came from this system.
        * **retry with backoff** — connection errors and 5xx responses are
          retried (0.5s / 1.5s backoff, max 3 attempts); 4xx is a permanent
          rejection and is not retried.  Total wait is bounded so a sick
          endpoint can't stall the tick.

        Never raises — a dead endpoint must not break the tick; the result
        (status/ok/attempts/error) rides along in the tick output."""
        url = (row.get("webhook_url") or "").strip()
        if not url:
            return {"fired": False, "reason": "no webhook configured"}
        payload = {
            "source": "nomorals-monitor",
            **{k: v for k, v in entry.items()
               if k not in ("webhook", "kind", "watch", "target")},
            "event": kind,                      # change | watch_down | test
            "monitor_id": row["id"],
            "target": row["target"],
            "watch_kind": row["kind"],          # url | file
            "watch": row["watch"],              # content | size
            "at": now,
        }
        headers = {"Content-Type": "application/json"}
        secret = (row.get("webhook_secret") or "").strip()
        if secret:
            body = json.dumps(payload, default=str).encode("utf-8")
            headers["X-NoMorals-Signature"] = "sha256=" + _hmac_hex(secret,
                                                                    body)
        from ..core.http import HttpClient

        last_status = 0
        last_error = ""
        for attempt in range(1, 4):
            try:
                resp = HttpClient().post_json(url, payload, headers=headers,
                                              timeout=10)
                last_status = int(getattr(resp, "status", 0) or 0)
                if 400 <= last_status < 500:
                    return {"fired": True, "status": last_status,
                            "ok": False, "attempts": attempt,
                            "error": f"HTTP {last_status}"}
                if 0 < last_status < 500:
                    return {"fired": True, "status": last_status,
                            "ok": True, "attempts": attempt}
                last_error = f"HTTP {last_status}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)[:200]
            if attempt < 3:
                time.sleep(0.5 if attempt == 1 else 1.5)
        return {"fired": True, "status": last_status, "ok": False,
                "attempts": 3, "error": last_error[:300]}

    def _note_error(self, row: dict[str, Any], error: str,
                    now: float | None = None) -> None:
        streak = int(row["error_streak"] or 0) + 1
        self.db.execute(
            "UPDATE monitors SET error_streak=?, last_ok=0 WHERE id=?",
            (streak, row["id"]))
        if streak == _ERROR_ALERT_STREAK:
            now = now if now is not None else time.time()
            throttled = self._throttled(row, now)
            if not throttled and self.notifier is not None:
                try:
                    self.notifier.publish(
                        kind="monitor",
                        title=f"watch down: {row['target']}",
                        body=f"3 consecutive check failures, "
                             f"latest: {error[:200]}",
                    )
                except Exception:  # noqa: BLE001
                    pass
            if not throttled:
                self._fire_webhook(row, "watch_down",
                                   {"error": error[:300],
                                    "error_streak": streak}, now)
                self.db.execute(
                    "UPDATE monitors SET last_alert_ts=? WHERE id=?",
                    (now, row["id"]))

    @staticmethod
    def _diff(old: str, new: str, limit: int = 4000) -> str:
        # Delegates to the extracted primitive (Prompt 03); behaviour
        # identical, kept as a method for backward compatibility.
        return unified_content_diff(old, new, limit)


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "monitor",
        description=(
            "Watch a URL, a workspace file, or a rendered web PAGE and "
            "alert when it changes. action=add (target, interval, kind, "
            "volatile, webhook, secret, min_gap, auto_decode) | alert "
            "(ref, webhook, secret, min_gap, auto_decode) | webhook_test "
            "(ref) | list | remove (ref) | enable (ref, on) | tick (run "
            "all due checks now) | status. kind=url hashes the raw body; "
            "kind=page renders the page headless (DOM → markdown, "
            "scripts/styles stripped) and diffs what a visitor actually "
            "sees — use page for any site where markup churn should not "
            "alert; volatile= newline-separated regexes stripped from the "
            "rendered text first (rotating nonces, timestamps, tokens). "
            "Changes are delivered through the notifier with a short diff "
            "and, when set, POSTed as JSON to the monitor's webhook URL "
            "(optionally HMAC-signed with secret). min_gap throttles "
            "alerts per monitor (seconds between alerts); auto_decode (url "
            "monitors) feeds the response into the OSINT identity graph + "
            "decoder archive."
        ),
        capability=Capability.MEM_WRITE,
    )
    def monitor(action: str = "list", target: str = "", ref: str = "",
                interval: float = 300.0, kind: str = "", volatile: str = "",
                on: bool = True, webhook: str = "", secret: str = "",
                min_gap: float = -1.0,
                auto_decode: int = -1) -> dict[str, Any]:
        # min_gap < 0 and auto_decode < 0 = "not given" — keep existing
        from .notifier import Notifier

        ad = None if auto_decode < 0 else bool(auto_decode)
        agent = MonitorAgent(context,
                             notifier=Notifier(context,
                                               getattr(context, "gateway",
                                                      None)))
        if action == "add":
            return agent.add(target, kind=kind, interval=interval,
                             volatile=volatile,
                             webhook=webhook, secret=secret,
                             min_gap=max(0.0, min_gap),
                             auto_decode=True if ad is None else ad)
        if action == "alert":
            row = agent.set_alerting(ref,
                                     webhook=webhook or None,
                                     secret=secret or None,
                                     min_gap=(min_gap if min_gap >= 0
                                              else None),
                                     auto_decode=ad)
            return {"monitor": row, "found": row is not None}
        if action == "webhook_test":
            res = agent.webhook_test(ref)
            if res is None:
                return {"ok": False, "error": f"no monitor {ref!r}"}
            return res
        if action == "remove":
            return {"removed": agent.remove(ref)}
        if action == "enable":
            row = agent.set_enabled(ref, bool(on))
            return {"monitor": row, "found": row is not None}
        if action == "tick":
            return agent.tick()
        if action == "status":
            return agent.status()
        return agent.status() if action != "list" else {"monitors": agent.list()}
