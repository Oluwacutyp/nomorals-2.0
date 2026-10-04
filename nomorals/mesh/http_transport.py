"""HTTP mesh transport: a device mesh across the network.

:class:`LocalTransport` shares one database; :class:`HttpTransport` talks
to a remote hub (``nomorals.hub``) exposing the same operations as JSON
over HTTP. Same abstract interface — node code built on
:class:`~nomorals.mesh.transport.Transport` needs no changes to join a
hub instead of a shared volume.

Auth is a shared bearer secret (``NM_HUB_TOKEN``) presented as
``Authorization: Bearer <token>``, the same posture as the main API
server. The wire errors map back to the exceptions the local transport
raises: unknown node → :class:`NodeUnknown`, bad request →
:exc:`ValueError`, everything unreachable → :class:`TransportError`.

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
from .errors import NodeUnknown, TransportError
from .node import MeshNode
from .tasks import MeshTask
from .transport import Transport

__all__ = ["HttpTransport", "DEFAULT_TIMEOUT", "DEFAULT_RETRIES"]

_log = get_logger(__name__)

DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 3


def _auth_headers(token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class HttpTransport(Transport):
    """Mesh transport over HTTP to a remote hub.

    ``base_url`` is the hub root, e.g. ``http://192.168.1.5:8861``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.retries = max(1, int(retries))

    # ── wire plumbing ────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
        body = None
        if payload is not None:
            body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method=method, headers=_auth_headers(self.token))
        if body is not None:
            req.add_header("Content-Length", str(len(body)))

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # HTTP errors are answers, not transport failures: translate
                # them once, never retry.
                raise self._translate_http_error(exc) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                _log.warning(
                    "hub %s %s failed (attempt %d/%d): %s",
                    method, path, attempt + 1, self.retries, exc)
                if attempt + 1 < self.retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise TransportError(
            f"hub unreachable at {self.base_url} after {self.retries} "
            f"attempt(s): {last_exc}") from last_exc

    @staticmethod
    def _translate_http_error(exc: urllib.error.HTTPError) -> Exception:
        """Map a hub HTTP status to the exception the local path raises."""
        code = exc.code
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - best-effort error detail
            body = {}
        detail = str(body.get("detail") or body.get("error") or "")
        error_code = str(body.get("code") or "")
        if code == 401:
            return TransportError(f"hub auth rejected (401): {detail}")
        if code == 404 and error_code == "node_unknown":
            return NodeUnknown(detail or "node not registered")
        if code == 400:
            return ValueError(detail or "bad request")
        return TransportError(f"hub error {code}: {detail}")

    # ── Transport interface ──────────────────────────────────────────────

    def register(
        self, name: str, platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        data = self._request("POST", "/mesh/register", payload={
            "name": name,
            "platform": platform,
            "capabilities": capabilities or [],
            "node_id": node_id,
        })
        return MeshNode.from_dict(data.get("node") or {})

    def heartbeat(self, node_id: str) -> None:
        self._request("POST", "/mesh/heartbeat",
                      payload={"node_id": node_id})

    def active_nodes(self) -> list[MeshNode]:
        data = self._request("GET", "/mesh/nodes")
        return [MeshNode.from_dict(n) for n in (data.get("nodes") or [])]

    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        origin_node: str,
        target_node: str | None = None,
        priority: int = 0,
    ) -> str:
        data = self._request("POST", "/mesh/dispatch", payload={
            "task_type": task_type,
            "payload": payload or {},
            "origin_node": origin_node,
            "target_node": target_node,
            "priority": priority,
        })
        job_id = str(data.get("job_id") or "")
        if not job_id:
            raise TransportError("hub dispatch returned no job_id")
        return job_id

    def poll(self, node_id: str, *, batch: int = 5) -> list[MeshTask]:
        data = self._request("POST", "/mesh/poll", payload={
            "node_id": node_id,
            "batch": batch,
        })
        return [MeshTask.from_dict(t) for t in (data.get("tasks") or [])]

    def complete(self, job_id: str, result: Any = None) -> None:
        self._request("POST", "/mesh/complete", payload={
            "job_id": job_id,
            "result": result,
        })

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        self._request("POST", "/mesh/fail", payload={
            "job_id": job_id,
            "error": error,
            "retry": retry,
        })
