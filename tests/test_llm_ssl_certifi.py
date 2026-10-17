"""LLM HTTPS must verify against certifi's CA bundle, never the bare platform store.

Regression: on Windows Server the system OpenSSL store is missing/stale, so
Groq/OpenRouter failed with SSL CERTIFICATE_VERIFY_FAILED while the Telegram
adapter (certifi-backed) worked. The fix: HttpClient builds an explicit
verifying context (explicit SSL_CERT_FILE > certifi > platform default),
and run_chat_bot.py sets SSL_CERT_FILE to certifi's bundle at import time.
Verification is never disabled on the verified path.
"""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
from pathlib import Path

import pytest

certifi = pytest.importorskip("certifi")

from nomorals.core import http as http_mod
from nomorals.core.http import HttpClient


@pytest.fixture(autouse=True)
def _clean_ssl_env(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    yield


def test_http_client_uses_certifi_bundle():
    client = HttpClient()
    assert client.tls_cafile == certifi.where()
    assert client._tls_context is not None
    # Verification stays ON: hostname checked, certs required.
    assert client._tls_context.verify_mode == ssl.CERT_REQUIRED
    assert client._tls_context.check_hostname is True


def test_http_client_verify_false_has_no_verifying_context():
    client = HttpClient(verify_tls=False)
    assert client.tls_cafile is None
    assert client._tls_context is None


def test_ssl_cert_file_env_wins(tmp_path, monkeypatch):
    # A real (copied) bundle at a different path: proves the explicit env
    # value wins over certifi's default location.
    custom = tmp_path / "my-ca.pem"
    custom.write_bytes(Path(certifi.where()).read_bytes())
    assert str(custom) != certifi.where()
    monkeypatch.setenv("SSL_CERT_FILE", str(custom))
    assert HttpClient().tls_cafile == str(custom)


def test_tls_cafile_ignores_missing_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/no-ca.pem")
    assert http_mod.tls_cafile() == certifi.where()


def test_certifi_missing_falls_back_to_system_default(monkeypatch):
    monkeypatch.setattr(http_mod, "_certifi_cafile", lambda: None)
    client = HttpClient()
    assert client.tls_cafile is None
    # Still a verifying context (platform default store), never CERT_NONE.
    assert client._tls_context is not None
    assert client._tls_context.verify_mode == ssl.CERT_REQUIRED


def test_http_client_context_actually_used_for_requests(monkeypatch):
    client = HttpClient()
    monkeypatch.setattr(HttpClient, "_check_ssrf", lambda self, url: None)
    monkeypatch.setattr(HttpClient, "_effective_proxy", lambda self, t: "")
    captured: dict = {}

    class _FakeRaw:
        status = 200
        headers = {}

        def geturl(self):
            return "https://api.groq.com/openai/v1/models"

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeOpener:
        def open(self, request, timeout=None):
            return _FakeRaw()

    def _fake_no_proxy_opener(opener_kwargs):
        captured.update(opener_kwargs)
        return _FakeOpener()

    monkeypatch.setattr(http_mod, "_no_proxy_opener", _fake_no_proxy_opener)
    resp = client.get("https://api.groq.com/openai/v1/models")
    assert resp.status == 200
    assert captured.get("context") is client._tls_context


def test_run_chat_bot_sets_ssl_cert_file():
    repo_root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
    code = (
        "import sys, os; sys.path.insert(0, 'scripts');"
        "import run_chat_bot; print(os.environ.get('SSL_CERT_FILE', ''))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == certifi.where()
