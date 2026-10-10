"""The device hub: one HTTP endpoint for the whole multi-device story.

The hub is the cloud (or always-on home box) side of
:class:`nomorals.mesh.http_transport.HttpTransport` and
:class:`nomorals.sync.http_peer.HttpSyncPeer`. It composes the
production-grade local implementations — :class:`LocalTransport` for mesh
(node registry + task queue) and :class:`LocalPeer` over a hub-owned
:class:`SyncStore` — behind JSON-over-HTTP routes with bearer-token auth.

Not to be confused with the ``nm hub`` MediaHub console: this is the
*device* hub — the rendezvous point a phone (Termux), a laptop, and a
cloud worker sync state and exchange work through.

Routes (all JSON; POST bodies are JSON objects). ``/``, ``/health`` and
``/ready`` need no auth; everything else needs ``Authorization: Bearer``:

* ``GET  /``                        → service index + route list
* ``GET  /health``                  → per-component SERVING status, uptime
* ``GET  /ready``                   → 200/503 readiness probe
* ``GET  /metrics``                 → Prometheus text exposition
* ``POST /mesh/register``           ``{"name","platform","capabilities","node_id?","labels?"}``
* ``POST /mesh/heartbeat``          ``{"node_id","info?"}``
* ``POST /mesh/deregister``         ``{"node_id"}``
* ``GET  /mesh/nodes``              → ``{"nodes": [...]}`` (``?all=1``: stale too)
* ``POST /mesh/dispatch``           full queue opts: ``dedupe_key``, ``delay``,
  ``max_attempts``, ``expire_after``, ``target_capabilities``,
  ``target_labels``
* ``POST /mesh/dispatch-many``      ``{"task_type","payloads":[...],...}``
* ``POST /mesh/poll``               ``{"node_id","batch","wait?"}`` (long-poll)
* ``POST /mesh/complete``           ``{"job_id","result?"}``
* ``POST /mesh/fail``               ``{"job_id","error?","retry?"}``
* ``POST /mesh/cancel``             ``{"job_id"}``
* ``POST /mesh/progress``           ``{"job_id","node_id","detail"}``
* ``GET  /mesh/job?job_id=``        → status + progress + result
* ``GET  /mesh/jobs``               → live tasks (``?node=&limit=``)
* ``GET  /mesh/dead``               → dead-letter queue
* ``POST /mesh/retry-dead``         ``{"job_id","delay?"}``
* ``POST /mesh/reclaim``            → requeue expired leases
* ``POST /mesh/reap-expired``       → fail schedule-expired tasks
* ``GET  /mesh/stats``              → queue stats
* ``GET  /mesh/wait?job_id=&timeout=`` → block for a result
* ``POST /sync/push``               ``{"records": [...]}`` → ``{"applied": n, "results": [...]}``
* ``GET  /sync/pull?since_seq=N``   → ``{"records","last_seq","pending"}``
  (``since_ts=T`` supported; ``timeout=`` long-polls)

Hardening: per-IP rate limiting (429 + ``Retry-After``), optional TLS
(``certfile``/``keyfile``), optional CORS, env config via ``NM_HUB_*``.

Stdlib only (``http.server.ThreadingHTTPServer``), same as the stream
server — no mandatory dependencies.
"""

from __future__ import annotations

from .server import (
    DEFAULT_RATE_LIMIT,
    MAX_BODY_BYTES,
    MAX_WAIT_SECONDS,
    PULL_MAX_LIMIT,
    HubServer,
    serve,
)

__all__ = [
    "HubServer",
    "serve",
    "MAX_BODY_BYTES",
    "PULL_MAX_LIMIT",
    "MAX_WAIT_SECONDS",
    "DEFAULT_RATE_LIMIT",
]
