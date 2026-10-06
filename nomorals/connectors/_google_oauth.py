"""Shared OAuth 2.0 plumbing for Google APIs (Gmail, Drive).

Both connectors use the same installed-app authorization-code flow:

1. The owner creates an OAuth client (Desktop app) in the Google Cloud
   console — human step, guided by the connector.
2. The connector prints an authorization URL; the owner opens it, picks
   their Google account, grants the scopes, and pastes back the code.
3. The connector exchanges the code for access + refresh tokens at
   ``https://oauth2.googleapis.com/token``. The refresh token is the
   vaulted credential; the short-lived access token rides in encrypted
   vault metadata and is refreshed automatically with a 60-second leeway.

Secrets: client_id / client_secret arrive via ``connect()`` arguments or
the ``GOOGLE_CLIENT_ID`` / ``GOOGLE_CLIENT_SECRET`` env vars — the same
client can serve Gmail and Drive, so the env vars are shared. Refresh
tokens are vault-stored under each connector's own service namespace.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from .base import Connector, ConnectorError

__all__ = [
    "GOOGLE_AUTH_URL",
    "GOOGLE_CLIENT_ID_ENV",
    "GOOGLE_CLIENT_SECRET_ENV",
    "GOOGLE_TOKEN_URL",
    "GoogleOAuth",
]

_log = get_logger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CLIENT_ID_ENV = "GOOGLE_CLIENT_ID"
GOOGLE_CLIENT_SECRET_ENV = "GOOGLE_CLIENT_SECRET"

#: Desktop-app OAuth clients use this fixed redirect for the manual flow.
#: NOTE: Google deprecated the OOB flow (urn:ietf:wg:oauth:2.0:oob) on
#: 2023-02-01. Use the loopback redirect flow instead — see
#: ``_loopback_redirect_server`` below.
_OOB_REDIRECT = "urn:ietf:wg:oauth:2.0:oob"

#: Refresh the access token this far ahead of expiry.
_REFRESH_LEEWAY = 60.0


class GoogleOAuth:
    """OAuth2 authorization-code mixin for Google connectors.

    Expected on the host class: ``_require_credential``,
    ``_store_credential``, ``self.http``, and ``self.vault`` (all provided
    by :class:`Connector`), plus ``_google_default_scopes()``.
    """

    # ── flow ─────────────────────────────────────────────────────

    def _google_default_scopes(self) -> list[str]:
        raise NotImplementedError

    def _google_client_id(self, client_id: str | None) -> str:
        cid = (client_id or "").strip() or os.environ.get(
            GOOGLE_CLIENT_ID_ENV, ""
        ).strip()
        if not cid:
            raise ConnectorError(
                "no Google OAuth client id — create a Desktop-app OAuth "
                "client in the Google Cloud console "
                "(https://console.cloud.google.com/apis/credentials) and "
                f"pass client_id=... or set {GOOGLE_CLIENT_ID_ENV}"
            )
        return cid

    def _google_client_secret(self, client_secret: str | None) -> str:
        return (client_secret or "").strip() or prompt_secret(
            "Google OAuth client secret", env_var=GOOGLE_CLIENT_SECRET_ENV
        )

    def google_authorize_url(
        self, client_id: str, scopes: list[str],
        redirect_uri: str = "",
    ) -> str:
        """The URL the owner opens to grant access."""
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri or _OOB_REDIRECT,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",  # ask for a refresh token
            "prompt": "consent",  # re-issue the refresh token every time
        }
        return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def google_loopback_flow(
        self, client_id: str, client_secret: str, scopes: list[str],
        *,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """OAuth flow via loopback redirect (replaces deprecated OOB).

        Spins up a temporary ``http://127.0.0.1:<port>/`` listener,
        returns the auth URL for the owner to open, waits for Google's
        redirect with the authorization code, then exchanges it for tokens.

        Returns the token dict from ``_google_exchange_code``.
        """
        import http.server
        import threading

        code_holder: dict[str, str] = {}
        error_holder: dict[str, str] = {}

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                if "code" in params:
                    code_holder["code"] = params["code"][0]
                    self._respond(200, "Authorization complete — you can close this tab.")
                elif "error" in params:
                    error_holder["error"] = params["error"][0]
                    self._respond(400, f"Authorization failed: {params['error'][0]}")
                else:
                    self._respond(400, "Missing authorization code.")

            def _respond(self, status: int, message: str):
                body = message.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # quiet

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/"

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            auth_url = self.google_authorize_url(client_id, scopes, redirect_uri=redirect_uri)
            print(f"\nOpen this URL to authorize:\n{auth_url}\n")
            print(f"Waiting for authorization (timeout {timeout:.0f}s)...")

            deadline = time.time() + timeout
            while time.time() < deadline:
                if "code" in code_holder:
                    break
                if "error" in error_holder:
                    raise ConnectorError(
                        f"Google authorization failed: {error_holder['error']}"
                    )
                time.sleep(0.5)
            else:
                raise ConnectorError(
                    "Timed out waiting for Google authorization. "
                    "Make sure you opened the URL and granted access."
                )

            return self._google_exchange_code_loopback(
                client_id, client_secret, code_holder["code"], redirect_uri,
            )
        finally:
            server.shutdown()
            server.server_close()

    def _google_exchange_code_loopback(
        self,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Swap an authorization code for tokens (loopback redirect)."""
        code = (code or "").strip()
        if not code:
            raise ConnectorError("empty authorization code from loopback redirect")
        try:
            resp = self.http.post_form(  # type: ignore[attr-defined]
                GOOGLE_TOKEN_URL,
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
        except Exception as exc:
            raise ConnectorError(f"Google token exchange failed: {exc}") from exc
        return resp

    def _google_exchange_code(
        self,
        client_id: str,
        client_secret: str,
        code: str,
    ) -> dict[str, Any]:
        """Swap an authorization code for access + refresh tokens."""
        code = (code or "").strip()
        if not code:
            raise ConnectorError(
                "empty authorization code — open the authorization URL, "
                "grant access, and paste back the code"
            )
        try:
            resp = self.http.post_form(  # type: ignore[attr-defined]
                GOOGLE_TOKEN_URL,
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": _OOB_REDIRECT,
                },
            )
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise ConnectorError(
                f"google token exchange failed: {exc}"
            ) from exc
        if not resp.ok:
            raise ConnectorError(
                "google rejected the authorization code "
                f"({resp.status}): {resp.text[:200]} — the code may have "
                "expired; generate a fresh one from the authorization URL"
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise ConnectorError(
                "google token exchange returned an unexpected response"
            )
        return data

    def _google_refresh(
        self, client_id: str, client_secret: str, refresh_token: str
    ) -> dict[str, Any]:
        try:
            resp = self.http.post_form(  # type: ignore[attr-defined]
                GOOGLE_TOKEN_URL,
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise ConnectorError(
                f"google token refresh failed: {exc}"
            ) from exc
        if not resp.ok:
            raise ConnectorError(
                "google rejected the refresh token "
                f"({resp.status}): {resp.text[:200]} — revoke and reconnect "
                "with a fresh authorization code"
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise ConnectorError(
                "google token refresh returned an unexpected response"
            )
        return data

    def _store_google_tokens(
        self: Connector,
        username: str,
        client_id: str,
        tokens: dict[str, Any],
        *,
        account: str = "",
        scopes: list[str] | None = None,
    ) -> None:
        """Vault the refresh token; cache the access token in metadata."""
        refresh = str(tokens.get("refresh_token", ""))
        access = str(tokens.get("access_token", ""))
        if not refresh:
            raise ConnectorError(
                "google did not return a refresh token — re-run the "
                "authorization URL with prompt=consent (already set) and "
                "use a fresh code"
            )
        if not access:
            raise ConnectorError(
                "google did not return an access token on exchange"
            )
        expires_in = float(tokens.get("expires_in", 3600) or 3600)
        self._store_credential(
            username,
            refresh,
            credential_type="oauth_token",
            scopes=scopes,
            metadata={
                "client_id": client_id,
                "account": account,
                "access_token": access,
                "access_expires_at": time.time() + expires_in,
            },
        )
        _log.info(
            "%s: google OAuth tokens stored for %s", self.id, username
        )

    def _google_access_token(self: Connector) -> str:
        """A fresh access token, refreshing when within the leeway."""
        cred = self._require_credential()  # type: ignore[attr-defined]
        meta = cred.metadata or {}
        client_id = str(meta.get("client_id", ""))
        access = str(meta.get("access_token", ""))
        expires_at = float(meta.get("access_expires_at", 0) or 0)
        if access and client_id and expires_at - time.time() > _REFRESH_LEEWAY:
            return access
        secret = self._google_client_secret(None)
        tokens = self._google_refresh(client_id, secret, cred.password)
        new_access = str(tokens.get("access_token", ""))
        if not new_access:
            raise ConnectorError(
                "google did not return an access token on refresh"
            )
        # Refresh tokens are long-lived; keep the stored one unless Google
        # rotated it.
        merged = dict(tokens)
        merged.setdefault("refresh_token", cred.password)
        self._store_google_tokens(
            cred.username,
            client_id,
            merged,
            account=str(meta.get("account", "")),
            scopes=list(meta.get("scopes", [])),
        )
        _log.info("%s: google access token refreshed", self.id)
        return new_access

    def _google_connect_guide(
        self, client_id: str, scopes: list[str]
    ) -> str:
        url = self.google_authorize_url(client_id, scopes)
        return "\n".join([
            "Connect your Google account (only you can grant this):",
            "1. Create a Desktop-app OAuth client in the Google Cloud",
            "   console if you haven't: "
            "https://console.cloud.google.com/apis/credentials",
            "   (enable the Gmail API / Drive API for the project first).",
            "2. Open this URL in your browser and grant access:",
            f"   {url}",
            "3. Google shows an authorization code. Hand it to Devon:",
            "   connect(..., code=<paste the code here>).",
            "The refresh token lands in the encrypted vault; Devon never",
            "sees your Google password.",
        ])
