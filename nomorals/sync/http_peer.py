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
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..core.logging_setup import get_logger
from .engine import SyncPeer
from .errors import SyncError
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


def _auth_headers(token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
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
        return self._send(req, "POST", path)

    def _send(self, req: urllib.request.Request, method: str,
              path: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                # HTTP errors are answers, not transport failures: translate
                # once, never retry.
                raise self._translate_http_error(exc) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                _log.warning(
                    "hub %s %s failed (attempt %d/%d): %s",
                    method, path, attempt + 1, self.retries, exc)
                if attempt + 1 < self.retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise SyncError(
            f"hub unreachable at {self.base_url} after {self.retries} "
            f"attempt(s): {last_exc}") from last_exc

    @staticmethod
    def _translate_http_error(exc: urllib.error.HTTPError) -> Exception:
        code = exc.code
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - best-effort error detail
            body = {}
        detail = str(body.get("detail") or body.get("error") or "")
        if code == 401:
            return SyncError(f"hub auth rejected (401): {detail}")
        if code == 400:
            return SyncError(f"hub rejected request (400): {detail}")
        if code == 413:
            return SyncError(f"push batch too large (413): {detail}")
        return SyncError(f"hub error {code}: {detail}")

    # ── SyncPeer interface ───────────────────────────────────────────────

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
