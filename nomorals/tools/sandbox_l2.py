"""Sandbox L2: L1 offline sandbox + token-gated network, same API.

L1 (:func:`nomorals.tools.shell.run_sandboxed`) runs with network off.
L2 is the same call, same result shape
(``stdout``, ``stderr``, ``exit_code``, ``timed_out``, ``seconds``,
``backend``) plus ``level`` and ``egress`` — with network ON **only**
when an :class:`~nomorals.core.egress.EgressToken` is presented, and
then only through the egress proxy sidecar:

* **No token → no network.** The call degrades to plain L1.
* **Token → live L2.** A local :class:`EgressProxy` (127.0.0.1, ephemeral
  port) is started, the child gets ``http_proxy``/``https_proxy`` pointed
  at it, and every HTTP(S) request the child makes is validated by the
  token + post-resolution checks in :mod:`nomorals.core.egress`
  (allowlist, IP-literal refusal, private-IP refusal, per-hop redirect
  re-validation, budgets, kill switch). Absolute-form proxy requests
  are fetched through :class:`EgressClient`; ``CONNECT`` tunnels are
  opened only to a checked IP of an allowlisted host.
* **Replay mode** (``cassette_mode="replay"``) runs true L1
  (``network=False``, no proxy): shell code gets no live network, and
  recorded traffic is served through :class:`EgressClient` in-process —
  L2 traffic recorded, replayed in L1, zero network I/O.

Honest scope: the proxy enforces HTTP(S). A child that opens a raw
socket directly bypasses the proxy — the *enforced* network API is
:class:`EgressClient`; the shell proxy is the enforcement point for
well-behaved HTTP clients (curl, wget, urllib, requests). Raw-socket
egress from shell would need a network-namespace firewall, which this
host cannot guarantee without root.
"""

from __future__ import annotations

import http.server
import select
import socket
import socketserver
import threading
from typing import Any

from ..core.cassette import GenericCassetteStore
from ..core.egress import (
    EgressClient,
    EgressController,
    EgressDenied,
    EgressToken,
    check_host_allowed,
    default_controller,
)
from ..core.logging_setup import get_logger
from .shell import SandboxLimits, run_sandboxed

_log = get_logger(__name__)

__all__ = ["EgressProxy", "run_sandboxed_l2"]

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade", "proxy-authenticate",
})


def _egress_summary(token: EgressToken | None, *, mode: str,
                    network: bool, replayed: int = 0) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mode": mode,
        "network": network,
        "token_id": token.token_id if token else None,
        "task_id": token.task_id if token else None,
        "requests": token.requests_used if token else 0,
        "bytes": token.bytes_used if token else 0,
        "denied": token.denied if token else 0,
        "replayed": replayed,
    }
    return summary


def run_sandboxed_l2(
    command: str,
    *,
    cwd: str | Any,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    limits: SandboxLimits | None = None,
    backend: str = "auto",
    max_output: int = 1_000_000,
    stdin: str = "",
    egress_token: EgressToken | None = None,
    cassette: GenericCassetteStore | None = None,
    cassette_mode: str = "record",
    controller: EgressController | None = None,
) -> dict[str, Any]:
    """Run ``command`` in the sandbox. Same result shape as L1, plus ``level``/``egress``.

    * ``egress_token=None`` → L1 (network off), exactly as
      :func:`run_sandboxed` with ``network=False``.
    * ``egress_token`` + ``cassette_mode="replay"`` → L1 (network off);
      recorded traffic is replayed through :class:`EgressClient`, not
      through the shell.
    * ``egress_token`` + live mode → L2: egress proxy sidecar + network on.
    """
    if egress_token is None:
        result = run_sandboxed(
            command, cwd=cwd, timeout=timeout, env=env, network=False,
            limits=limits, backend=backend, max_output=max_output, stdin=stdin)
        result["level"] = "L1"
        result["egress"] = _egress_summary(None, mode="L1", network=False)
        return result

    ctl = controller or default_controller()
    ctl.register(egress_token)
    ctl.check(egress_token)  # fail closed: dead token / global kill → raise

    if cassette_mode == "replay":
        result = run_sandboxed(
            command, cwd=cwd, timeout=timeout, env=env, network=False,
            limits=limits, backend=backend, max_output=max_output, stdin=stdin)
        result["level"] = "L1"
        result["egress"] = _egress_summary(
            egress_token, mode="L1-replay", network=False)
        return result

    child_env = dict(env or {})
    with EgressProxy(token=egress_token, controller=ctl, cassette=cassette,
                     cassette_mode=cassette_mode) as proxy:
        child_env.update({
            "http_proxy": proxy.proxy_url,
            "https_proxy": proxy.proxy_url,
            "HTTP_PROXY": proxy.proxy_url,
            "HTTPS_PROXY": proxy.proxy_url,
            "ALL_PROXY": proxy.proxy_url,
            # Empty no_proxy: even localhost goes through the proxy, so
            # the token's allowlist (and allow_private_ips) governs it.
            "no_proxy": "",
            "NO_PROXY": "",
        })
        result = run_sandboxed(
            command, cwd=cwd, timeout=timeout, env=child_env, network=True,
            limits=limits, backend=backend, max_output=max_output, stdin=stdin)
    result["level"] = "L2"
    result["egress"] = _egress_summary(
        egress_token, mode="L2", network=True)
    return result


# ── egress proxy sidecar ───────────────────────────────────────────────────

class _ProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class EgressProxy:
    """Local HTTP(S) proxy enforcing one :class:`EgressToken`.

    Context manager: ``with EgressProxy(...) as proxy:`` serves on
    ``proxy.proxy_url`` until exit. Absolute-form requests are fetched
    via :class:`EgressClient` (full per-hop checks + cassettes);
    ``CONNECT`` opens a tunnel only to a checked IP of an allowlisted
    host, and the tunnel dies on kill-switch / token revocation.
    """

    def __init__(
        self,
        *,
        token: EgressToken,
        controller: EgressController | None = None,
        cassette: GenericCassetteStore | None = None,
        cassette_mode: str = "record",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.token = token
        self.controller = controller or default_controller()
        self.client = EgressClient(
            controller=self.controller, cassette=cassette,
            cassette_mode=cassette_mode)
        self._server = _ProxyServer((host, port), _ProxyHandler)
        self._server.proxy = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
            name="egress-proxy", daemon=True)
        self.proxy_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "EgressProxy":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
        try:
            self._server.server_close()
        except Exception:  # noqa: BLE001 - close is best-effort
            pass


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NoMoralsEgress/1.0"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence stdlib logging
        pass

    @property
    def _proxy(self) -> EgressProxy:
        return self.server.proxy  # type: ignore[attr-defined]

    # ── absolute-form requests ──
    def _handle_proxy_request(self, method: str) -> None:
        proxy = self._proxy
        try:
            body = self._read_body()
            # The proxy form is an absolute URL. Origin-form ("/path")
            # has no host to check — refuse it.
            if self.path.startswith("/"):
                raise EgressDenied(
                    "egress denied: origin-form request has no host to check")
            forward: dict[str, str] = {}
            for key, value in self.headers.items():
                lowered = key.lower()
                if lowered in _HOP_BY_HOP or lowered == "host":
                    continue
                forward[lowered] = value
            response = proxy.client.fetch(
                proxy.token, method, self.path,
                headers=forward, data=body or None)
            self._send_relay(response.status, response.headers, response.body)
        except EgressDenied as exc:
            self._send_text(403, f"egress denied: {exc}\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - the proxy never crashes a client
            _log.debug("egress proxy error: %s", exc, exc_info=True)
            try:
                self._send_text(502, f"egress proxy error: {exc}\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is not None:
            try:
                n = int(length)
            except ValueError:
                return b""
            if n <= 0:
                return b""
            return self.rfile.read(min(n, 64 * 1024 * 1024))
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            chunks: list[bytes] = []
            while True:
                size_line = self.rfile.readline(65536).strip()
                try:
                    size = int(size_line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline(65536)  # trailing CRLF
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)  # CRLF after the chunk
            return b"".join(chunks)
        return b""

    def _send_text(self, status: int, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _send_relay(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() in _HOP_BY_HOP or key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._handle_proxy_request("GET")

    def do_POST(self) -> None:
        self._handle_proxy_request("POST")

    def do_PUT(self) -> None:
        self._handle_proxy_request("PUT")

    def do_PATCH(self) -> None:
        self._handle_proxy_request("PATCH")

    def do_DELETE(self) -> None:
        self._handle_proxy_request("DELETE")

    def do_HEAD(self) -> None:
        self._handle_proxy_request("HEAD")

    def do_OPTIONS(self) -> None:
        self._handle_proxy_request("OPTIONS")

    # ── CONNECT tunnel ──
    def do_CONNECT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler name
        proxy = self._proxy
        self.close_connection = True  # tunnels are one-shot; no keep-alive
        target = self.path or ""
        try:
            if ":" in target:
                host, port_s = target.rsplit(":", 1)
                try:
                    port = int(port_s)
                except ValueError as exc:
                    raise EgressDenied(f"bad CONNECT target {target!r}") from exc
            else:
                host, port = target, 443
            proxy.controller.check(proxy.token)
            # Full token + post-resolution check BEFORE connecting.
            proxy.token.check_url(f"https://{host}/")
            ips = check_host_allowed(
                host, allow_private_ips=proxy.token.allow_private_ips)
            upstream: socket.socket | None = None
            last_error: OSError | None = None
            for ip in ips:
                try:
                    upstream = socket.create_connection(
                        (ip, port), timeout=proxy.token.timeout)
                    break
                except OSError as exc:
                    last_error = exc
                    upstream = None
            if upstream is None:
                raise EgressDenied(
                    f"CONNECT to {host} failed: {last_error or 'no address'}")
            client_sock = self.connection
            proxy.controller.track_socket(client_sock)
            proxy.controller.track_socket(upstream)
            try:
                self._send_connect_ok()
                self._tunnel(client_sock, upstream, proxy)
            finally:
                proxy.controller.untrack_socket(client_sock)
                proxy.controller.untrack_socket(upstream)
                # The tunnel ran the sockets non-blocking; restore
                # blocking mode so the handler teardown behaves.
                for sock in (client_sock, upstream):
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
                try:
                    upstream.close()
                except OSError:
                    pass
        except EgressDenied as exc:
            try:
                self._send_text(403, f"egress denied: {exc}\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - the proxy never crashes a client
            try:
                self._send_text(502, f"egress proxy error: {exc}\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_connect_ok(self) -> None:
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()

    def _tunnel(self, client_sock: socket.socket, upstream: socket.socket,
                proxy: EgressProxy) -> None:
        """Bidirectional forward until either side closes or the kill switch fires."""
        for sock in (client_sock, upstream):
            sock.setblocking(False)
        while True:
            if proxy.controller.killed or proxy.token.revoked or proxy.token.expired:
                return
            try:
                readable, _, _ = select.select([client_sock, upstream], [], [], 0.25)
            except (OSError, ValueError):
                return
            for src in readable:
                dst = upstream if src is client_sock else client_sock
                try:
                    data = src.recv(65536)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    return
                if not data:
                    return
                try:
                    dst.sendall(data)
                except OSError:
                    return
