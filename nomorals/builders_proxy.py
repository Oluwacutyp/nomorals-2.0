"""A small, real HTTP reverse proxy used by ``AppBuilder.deploy``.

It is launched as a **detached subprocess** (``python -m
nomorals.builders_proxy``) so the public URL keeps living after the CLI
turn exits.  It forwards every request to the app's own server, optionally
under a path prefix and behind a domain name — the same behaviour a
reverse proxy (nginx/Caddy) would give you, in pure stdlib.

    python -m nomorals.builders_proxy --backend 127.0.0.1:8002 \
        --listen 0.0.0.0:9000 [--prefix /myapp] [--domain example.com] \
        [--tls [--cert C --key K]]

``--tls`` serves HTTPS.  With no ``--cert/--key`` a self-signed
certificate is generated on the fly (``cryptography`` when installed,
else the ``openssl`` binary) and cached next to the app so restarts
reuse the same identity — the honest way to give a deployment a TLS
terminator without a public CA.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import http.client
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["ReverseProxyHandler", "run", "main", "ensure_self_signed_cert"]

#: response headers we never forward verbatim (hop-by-hop / recomputed)
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "upgrade",
    "transfer-encoding", "content-length",
}


def _split(spec: str, default_port: int) -> tuple[str, int]:
    host, _, port = spec.rpartition(":")
    if not host:
        host, port = spec, default_port
    try:
        return host, int(port or default_port)
    except ValueError:
        return host, default_port


# ── TLS (wave 77) ───────────────────────────────────────────────────────────
def ensure_self_signed_cert(cert_path: str, key_path: str,
                            common_name: str = "localhost",
                            days: int = 825) -> tuple[str, str]:
    """Make sure ``cert_path``/``key_path`` hold a usable self-signed pair.

    Reuses an existing pair when the files are present; otherwise
    generates one — preferring the ``cryptography`` package (pure Python,
    no external binary) and falling back to the ``openssl`` CLI.  The
    certificate carries the domain as CN + SAN so browsers name-check it
    (they will still flag it self-signed — that is the honest state of a
    locally-issued identity, and clients verify with ``verify_tls=False``).

    Returns ``(cert_path, key_path)``.  Raises RuntimeError when neither
    generator is available — no fake TLS, no silent downgrade.
    """
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
    os.makedirs(os.path.dirname(os.path.abspath(cert_path)), exist_ok=True)
    _cn = common_name or "localhost"
    crypto_err = ""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, _cn),
        ])
        now = _dt.datetime.now(_dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1))
            .not_valid_after(now + _dt.timedelta(days=days))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName(_cn), x509.DNSName("localhost")]),
                critical=False)
            .add_extension(x509.BasicConstraints(ca=False, pathlen=None),
                           critical=True)
            .sign(key, hashes.SHA256())
        )
        with open(key_path, "wb") as fh:
            fh.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption()))
        with open(cert_path, "wb") as fh:
            fh.write(cert.public_bytes(serialization.Encoding.PEM))
        return cert_path, key_path
    except ImportError:  # noqa: E103 - cryptography optional, openssl fallback follows
        pass
    except Exception as exc:  # noqa: BLE001 — fall through to openssl
        crypto_err = str(exc)
    # openssl fallback
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", key_path, "-out", cert_path, "-days", str(days),
             "-nodes", "-subj", f"/CN={_cn}",
             "-addext", f"subjectAltName=DNS:{_cn},DNS:localhost"],
            check=True, capture_output=True, timeout=60)
        return cert_path, key_path
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = f"; cryptography: {crypto_err}" if crypto_err else ""
        raise RuntimeError(
            "could not generate a self-signed certificate: neither the "
            f"'cryptography' package nor the 'openssl' binary is usable "
            f"({exc}{detail}). Install one (pip install cryptography) or "
            "pass --cert/--key to the proxy.")


class ReverseProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NoMoralsProxy/1.0"

    # configured on the server instance
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    prefix: str = ""          # e.g. "/myapp" — stripped before forwarding
    domain: str = ""          # when set, the public domain (Host rewrite)

    # ── plumbing ─────────────────────────────────────────────────────────
    def _target_path(self) -> str:
        path = self.path
        if self.prefix and (path == self.prefix or
                            path.startswith(self.prefix + "/")):
            path = path[len(self.prefix):] or "/"
        return path

    def _forward_headers(self) -> dict[str, str]:
        fwd: dict[str, str] = {}
        for key, val in self.headers.items():
            lk = key.lower()
            if lk in _HOP_HEADERS:
                continue
            if lk == "host":
                continue
            fwd[key] = val
        # rewrite Host so the backend sees the domain it is "behind"
        if self.domain:
            fwd["Host"] = self.domain
        return fwd

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        try:
            n = int(length) if length else 0
        except ValueError:
            n = 0
        if n <= 0:
            return b""
        return self.rfile.read(n)

    @staticmethod
    def _rewrite_location(val: str, public_base: str) -> str:
        """Point backend redirects at the public origin when we know it."""
        if not public_base or not val:
            return val
        v = val.strip()
        if v.startswith("/"):            # relative → public base + path
            return public_base + (v if v.startswith("/") else "/" + v)
        if v.startswith(("http://", "https://")):
            host = v.split("://", 1)[1].split("/", 1)[0]
            if host in ("127.0.0.1", "localhost") or ":" in host and \
                    host.rsplit(":", 1)[1].isdigit():
                return public_base + ("/" + v.split("/", 3)[3]
                                      if len(v.split("/", 3)) > 3 else "")
        return val

    def _do(self) -> None:
        method = self.command
        target_path = self._target_path()
        body = self._read_body()
        headers = self._forward_headers()
        conn = None
        try:
            conn = http.client.HTTPConnection(
                self.backend_host, self.backend_port, timeout=120)
            conn.request(method, target_path, body=body, headers=headers)
            resp = conn.getresponse()
            payload = resp.read()  # http.client decodes chunked for us
            self.send_response(resp.status, resp.reason)
            public_base = ""
            if self.domain:
                public_base = ("https" if self.headers.get(
                    "X-Forwarded-Proto") == "https" else "http") \
                    + "://" + self.domain + self.prefix
            for key, val in resp.getheaders():
                lk = key.lower()
                if lk in _HOP_HEADERS:
                    continue
                if lk == "location" and resp.status in (301, 302, 303,
                                                         307, 308):
                    val = self._rewrite_location(val, public_base)
                self.send_header(key, val)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if method != "HEAD" and payload:
                self.wfile.write(payload)
        except (ConnectionRefusedError, OSError) as exc:
            self.send_error(502, f"backend unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001 — never kill the thread
            try:
                self.send_error(502, f"proxy error: {exc}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = \
        _do

    def log_message(self, fmt, *args):  # noqa: N802 — keep it quiet
        pass


def _make_handler(backend_host: str, backend_port: int, prefix: str,
                  domain: str):
    # (a class body can't see the enclosing function's locals, so the
    # configured values are assigned on the subclass afterwards)
    class _Handler(ReverseProxyHandler):
        pass

    _Handler.backend_host = backend_host
    _Handler.backend_port = backend_port
    _Handler.prefix = prefix
    _Handler.domain = domain
    return _Handler


def run(listen_host: str, listen_port: int, backend_host: str,
        backend_port: int, prefix: str = "", domain: str = "",
        tls: bool = False, cert: str = "", key: str = ""):
    """Blocking proxy serve (used by the detached subprocess).

    With ``tls`` the listener terminates TLS: the supplied (or a
    generated self-signed) certificate/key are loaded into an
    ``ssl.SSLContext`` and wrapped over the server socket, so every
    connection is HTTPS to the backend's plain HTTP.
    """
    import ssl as _ssl

    prefix = "/" + prefix.strip("/") if prefix.strip("/") else ""
    handler = _make_handler(backend_host, backend_port, prefix, domain)
    httpd = ThreadingHTTPServer((listen_host, listen_port), handler)
    httpd.daemon_threads = True
    scheme = "http"
    cert_used = ""
    if tls:
        # resolve the cert/key (generating a self-signed pair when missing)
        if not cert or not key:
            cert_dir = os.path.join(
                os.path.expanduser("~"), ".nomorals", "tls")
            tag = (domain or listen_host or "localhost").strip("/.") \
                .replace(":", "_") or "localhost"
            cert = cert or os.path.join(cert_dir, f"{tag}.crt")
            key = key or os.path.join(cert_dir, f"{tag}.key")
        cert, key = ensure_self_signed_cert(cert, key,
                                            common_name=domain or "localhost")
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert, keyfile=key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
        cert_used = cert
    # announce for the supervisor to health-check
    print(json.dumps({"proxy": "ready", "listen": listen_host,
                      "port": listen_port, "backend":
                      f"{backend_host}:{backend_port}",
                      "prefix": prefix, "domain": domain,
                      "scheme": scheme, "cert": cert_used}),
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E103, E106 - deliberate top-level shutdown hook
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nomorals.builders_proxy")
    ap.add_argument("--listen", required=True,
                    help="host:port to bind (e.g. 0.0.0.0:9000)")
    ap.add_argument("--backend", required=True,
                    help="host:port of the app server (e.g. 127.0.0.1:8002)")
    ap.add_argument("--prefix", default="", help="public path prefix")
    ap.add_argument("--domain", default="", help="domain the app is behind")
    ap.add_argument("--tls", action="store_true",
                    help="terminate TLS on the listener (HTTPS)")
    ap.add_argument("--cert", default="",
                    help="TLS cert (PEM); with --tls and no cert a "
                         "self-signed one is generated + cached")
    ap.add_argument("--key", default="", help="TLS key (PEM)")
    a = ap.parse_args(argv)
    lh, lp = _split(a.listen, 9000)
    bh, bp = _split(a.backend, 8000)
    run(lh, lp, bh, bp, prefix=a.prefix, domain=a.domain,
        tls=a.tls, cert=a.cert, key=a.key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
