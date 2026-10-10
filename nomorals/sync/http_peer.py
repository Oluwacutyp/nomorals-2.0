"""HTTP sync peer: replicate with a remote hub over JSON-over-HTTP.

:class:`LocalPeer` wraps another :class:`SyncStore` on the same machine;
:class:`HttpSyncPeer` talks to a ``nomorals.hub`` server exposing the sync
endpoints. Same :class:`~nomorals.sync.engine.SyncPeer` interface — the
:class:`~nomorals.sync.engine.SyncEngine` runs unchanged, so a device
syncs with a cloud hub exactly the way it syncs with a local store.

Push batches are size-bounded (the hub rejects oversized bodies, and a
runaway record should not balloon one request). Pulls page through the
hub's cursor so a first sync against a large hub cannot blow the device's
memory in a single response.

Stdlib only (``urllib``) — no mandatory dependencies, Termux-safe.

Error discipline: HTTP 401 becomes :class:`SyncAuthError` (permanent —
never retried); other hub error statuses become :class:`SyncHubError`
with ``retryable`` set for 5xx; transport failure after all retries
becomes :class:`SyncConnectionError` (transient — the engine backs off
and retries).
"""

from __future__ import annotations

import gzip
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..core.logging_setup import get_logger
from .engine import SyncPeer
from .errors import SyncAuthError, SyncConnectionError, SyncError, SyncHubError
from .store import SyncRecord

__all__ = ["HttpSyncPeer", "DEFAULT_TIMEOUT", "DEFAULT_RETRIES"]

_log = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3

#: Max JSON bytes per push request — mirrors the mesh payload discipline:
#: records are envelopes, large blobs belong in the artifact store.
PUSH_MAX_BYTES = 512 * 1024

#: Records per pull page.
PULL_LIMIT = 500

USER_AGENT = "nomorals-sync/1.0"


def _auth_headers(token: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept-Encoding": "gzip",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class HttpSyncPeer(SyncPeer):
    """A sync peer reached over HTTP (the remote hub).

    ``base_url`` is the hub root, e.g. ``http://192.168.1.5:8861``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        pull_limit: int = PULL_LIMIT,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.pull_limit = max(1, int(pull_limit))
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "requests": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "retries": 0,
            "last_latency_s": 0.0,
            "last_error": "",
        }

    def __repr__(self) -> str:
        auth = "token" if self.token else "none"
        return f"HttpSyncPeer({self.base_url!r}, auth={auth})"

    def describe(self) -> dict[str, Any]:
        """Non-sensitive description (the token itself never leaves)."""
        return {
            "base_url": self.base_url,
            "auth": "bearer" if self.token else "none",
            "timeout": self.timeout,
            "retries": self.retries,
            "pull_limit": self.pull_limit,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Transfer accounting: requests, bytes, retries, latency."""
        with self._stats_lock:
            return dict(self._stats)

    #: Stats keys that accumulate; everything else is a gauge (set).
    _COUNTER_KEYS = frozenset(
        {"requests", "bytes_sent", "bytes_received", "retries"})

    def _bump(self, **kw: Any) -> None:
        with self._stats_lock:
            for k, v in kw.items():
                if k in self._COUNTER_KEYS:
                    self._stats[k] = self._stats.get(k, 0) + v
                else:
                    self._stats[k] = v

    # ── wire plumbing ────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        qs = {k: v for k, v in params.items() if v is not None}
        if qs:
            url += "?" + urllib.parse.urlencode(qs)
        req = urllib.request.Request(
            url, method="GET", headers=_auth_headers(self.token))
        return self._send(req, "GET", path)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST", headers=_auth_headers(self.token))
        req.add_header("Content-Length", str(len(body)))
        self._bump(bytes_sent=len(body))
        return self._send(req, "POST", path)

    def _send(self, req: urllib.request.Request, method: str,
              path: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip" and raw:
                        raw = gzip.decompress(raw)
                    self._bump(requests=1, bytes_received=len(raw),
                               last_latency_s=time.time() - started,
                               last_error="")
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                # HTTP errors are answers, not transport failures: translate
                # once, never retry here (the engine retries 5xx itself).
                self._bump(requests=1,
                           last_latency_s=time.time() - started)
                err = self._translate_http_error(exc)
                self._bump(last_error=str(err))
                raise err from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                self._bump(retries=1 if attempt else 0)
                _log.warning(
                    "hub %s %s failed (attempt %d/%d): %s",
                    method, path, attempt + 1, self.retries, exc)
                if attempt + 1 < self.retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        err = SyncConnectionError(
            f"hub unreachable at {self.base_url} after {self.retries} "
            f"attempt(s): {last_exc}")
        self._bump(last_error=str(err))
        raise err from last_exc

    @staticmethod
    def _translate_http_error(exc: urllib.error.HTTPError) -> SyncError:
        code = exc.code
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - best-effort error detail
            body = {}
        detail = str(body.get("detail") or body.get("error") or "")
        if code == 401:
            return SyncAuthError(
                f"hub auth rejected (401){': ' + detail if detail else ''}",
                status_code=401, detail=detail)
        if code in (400, 413):
            label = "push batch too large" if code == 413 \
                else "hub rejected request"
            return SyncHubError(
                f"{label} ({code}){': ' + detail if detail else ''}",
                status_code=code, detail=detail, retryable=False)
        return SyncHubError(
            f"hub error {code}{': ' + detail if detail else ''}",
            status_code=code, detail=detail,
            retryable=500 <= code < 600)

    # ── SyncPeer interface ───────────────────────────────────────────────

    def ping(self) -> bool:
        """Liveness probe: one cheap pull page. Raises on failure."""
        data = self._get("/sync/pull", {"since_seq": 0, "limit": 1})
        return bool(data.get("ok", True))

    def push_records(self, records: list[SyncRecord]) -> int:
        """Push records in size-bounded batches. Returns count applied."""
        applied = 0
        batch: list[dict[str, Any]] = []
        batch_bytes = 0
        for rec in records:
            doc = rec.to_dict()
            size = len(json.dumps(doc, default=str))
            if batch and batch_bytes + size > PUSH_MAX_BYTES:
                applied += self._push_batch(batch)
                batch, batch_bytes = [], 0
            batch.append(doc)
            batch_bytes += size
        if batch:
            applied += self._push_batch(batch)
        return applied

    def _push_batch(self, docs: list[dict[str, Any]]) -> int:
        data = self._post("/sync/push", {"records": docs})
        return int(data.get("applied") or 0)

    def fetch_since_seq(self, seq: int) -> list[SyncRecord]:
        """Records after cursor ``seq``; pages until a short page."""
        return self._pull({"since_seq": seq})

    def fetch_since(self, since: float) -> list[SyncRecord]:
        """Legacy timestamp-cursor pull (kept for pre-seq peers)."""
        return self._pull({"since_ts": since})

    def _pull(self, cursor: dict[str, Any]) -> list[SyncRecord]:
        out: list[SyncRecord] = []
        while True:
            params = {**cursor, "limit": self.pull_limit}
            data = self._get("/sync/pull", params)
            docs = data.get("records") or []
            out.extend(SyncRecord.from_dict(d) for d in docs)
            if len(docs) < self.pull_limit:
                break
            # Advance the cursor from the last record so pagination makes
            # progress even if the hub ignores `limit` semantics.
            last = docs[-1]
            if "since_seq" in cursor:
                cursor = {"since_seq": int(last.get("seq") or 0)}
            else:
                cursor = {"since_ts": float(last.get("updated_at") or 0.0)}
        return out
