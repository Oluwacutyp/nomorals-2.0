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

Routes (all JSON; POST bodies are JSON objects):

* ``GET  /health``                  → ``{"ok": true, "version": ...}`` (no auth)
* ``POST /mesh/register``           ``{"name","platform","capabilities","node_id?"}``
* ``POST /mesh/heartbeat``          ``{"node_id"}``
* ``GET  /mesh/nodes``              → ``{"nodes": [...]}``
* ``POST /mesh/dispatch``           ``{"task_type","payload","origin_node","target_node?","priority"}``
* ``POST /mesh/poll``               ``{"node_id","batch"}``
* ``POST /mesh/complete``           ``{"job_id","result?"}``
* ``POST /mesh/fail``               ``{"job_id","error?","retry?"}``
* ``POST /sync/push``               ``{"records": [...]}`` → ``{"applied": n}``
* ``GET  /sync/pull?since_seq=N``   → ``{"records": [...]}`` (or ``since_ts=T``)

Stdlib only (``http.server.ThreadingHTTPServer``), same as the stream
server — no mandatory dependencies.
"""

from __future__ import annotations

from .server import HubServer, serve

__all__ = ["HubServer", "serve"]
