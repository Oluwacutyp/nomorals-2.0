"""Pass #30 — connector framework honesty fixes.

Bug 1 (AudD lifecycle fake-positives): ``AudDConnector._api_post``
flattened transport failures into ``AudDError(error_code=0)``, and every
lifecycle method read "any non-auth error" as "token accepted":

* ``connect()`` vaulted an *unvalidated* token and returned ``ok=True``
  with the network completely dead;
* ``test_connection()`` returned True on a dead network;
* ``status()`` never probed at all — a stored credential alone meant
  ``connected=True``, even for a revoked token (every other connector
  live-checks in ``status()``; the fleet health snapshot believed it).

Bug 2 (Google OAuth drift): ``GoogleOAuth._google_exchange_code_loopback``
returned the raw HTTP response object instead of the parsed token dict
and never raised on an HTTP error status, unlike its sibling
``_google_exchange_code`` — ``google_loopback_flow`` documents that it
returns the token dict.
"""

from __future__ import annotations

import unittest
from typing import Any

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import create_connector
from nomorals.connectors._google_oauth import GoogleOAuth
from nomorals.connectors.base import ConnectorError
from nomorals.core.http import RequestError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class AudDFakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class AudDFakeHttp:
    """Scripted AudD endpoint: one behaviour per instance."""

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour
        self.calls = 0

    def post_multipart(self, url: str, **kw: Any) -> AudDFakeResponse:
        self.calls += 1
        if self.behaviour == "dead":
            raise RequestError("network unreachable (fake)")
        if self.behaviour == "auth_error":
            return AudDFakeResponse({
                "status": "error",
                "error": {"error_code": 900,
                          "error_message": "invalid api_token"},
            })
        if self.behaviour == "fingerprint_error":
            # Token accepted; the (deliberately file-less) request is
            # malformed — the established cheap-validation signal.
            return AudDFakeResponse({
                "status": "error",
                "error": {"error_code": 300,
                          "error_message": "no audio to fingerprint"},
            })
        return AudDFakeResponse({"status": "success", "result": None})


def _audd(behaviour: str, *, with_cred: bool = False):
    conn = create_connector("audd", _vault(), http=AudDFakeHttp(behaviour))
    if with_cred:
        conn._store_credential(
            "audd-user", "some-token", credential_type="api_key")
    return conn


class AudDTransportFailureTests(unittest.TestCase):
    def test_connect_dead_network_raises_and_vaults_nothing(self):
        conn = _audd("dead")
        with self.assertRaises(ConnectorError):
            conn.connect(api_key="totally-bogus-token")
        self.assertIsNone(conn._load_credential())

    def test_test_connection_dead_network_is_false(self):
        conn = _audd("dead", with_cred=True)
        self.assertFalse(conn.test_connection())

    def test_status_dead_network_is_not_connected(self):
        conn = _audd("dead", with_cred=True)
        st = conn.status()
        self.assertFalse(st.connected)


class AudDProbeSemanticsTests(unittest.TestCase):
    def test_auth_error_means_not_connected(self):
        conn = _audd("auth_error", with_cred=True)
        self.assertFalse(conn.test_connection())
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("reject", st.detail.lower())

    def test_fingerprint_error_means_token_accepted(self):
        conn = _audd("fingerprint_error", with_cred=True)
        self.assertTrue(conn.test_connection())
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertEqual(st.account, "audd-user")

    def test_success_payload_means_connected(self):
        conn = _audd("success", with_cred=True)
        self.assertTrue(conn.test_connection())
        self.assertTrue(conn.status().connected)

    def test_connect_with_auth_error_raises_and_vaults_nothing(self):
        conn = _audd("auth_error")
        with self.assertRaises(ConnectorError):
            conn.connect(api_key="bad-token")
        self.assertIsNone(conn._load_credential())

    def test_connect_with_fingerprint_error_vaults_token(self):
        conn = _audd("fingerprint_error")
        result = conn.connect(api_key="good-token")
        self.assertTrue(result.ok)
        self.assertIsNotNone(conn._load_credential())


class _LoopbackResp:
    def __init__(self, payload: Any, *, ok: bool = True,
                 status: int = 200) -> None:
        self._payload = payload
        self.ok = ok
        self.status = status
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


class _LoopbackHttp:
    def __init__(self, resp: _LoopbackResp) -> None:
        self._resp = resp

    def post_form(self, url: str, form: Any) -> _LoopbackResp:
        return self._resp


class _LoopbackHost(GoogleOAuth):
    def __init__(self, resp: _LoopbackResp) -> None:
        self.http = _LoopbackHttp(resp)


class LoopbackExchangeTests(unittest.TestCase):
    def test_http_error_raises_connector_error(self):
        host = _LoopbackHost(_LoopbackResp(
            {"error": "invalid_grant"}, ok=False, status=400))
        with self.assertRaises(ConnectorError):
            host._google_exchange_code_loopback(
                "cid", "secret", "code123", "http://127.0.0.1:9/")

    def test_success_returns_token_dict(self):
        tokens = {"access_token": "at", "refresh_token": "rt",
                  "expires_in": 3600}
        host = _LoopbackHost(_LoopbackResp(tokens))
        out = host._google_exchange_code_loopback(
            "cid", "secret", "code123", "http://127.0.0.1:9/")
        self.assertEqual(out, tokens)


if __name__ == "__main__":
    unittest.main()
