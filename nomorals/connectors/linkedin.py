"""LinkedIn API connector.

Drives LinkedIn over the REST API (https://learn.microsoft.com/linkedin)
with the framework HttpClient.

Auth: an OAuth2 access token (``OAUTH2``) with the ``openid`` scope for
identity and ``w_member_social`` for posting. ``connect(token=...)``
takes a pre-obtained token; without one the grant guide prints and the
flow fails fast (LinkedIn has no device flow).

Honest scope note: there is NO consumer home-feed read endpoint — the
old social-feed APIs are gone. What the API really offers for reading is
``GET /rest/posts/{id}`` on posts the authenticated member can see, plus
identity. ``get_feed`` is therefore NOT implemented (the task called for
honesty over fakery); ``get_post`` covers own-post reads.

Posting uses ``POST /rest/posts`` with the member's URN
(``urn:li:person:{id}``) taken from ``/v2/userinfo`` at connect time. If
the stored credential lacks it, ``post`` fails fast and tells the owner
to reconnect.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.errors import NoMoralsError, RateLimited
from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["LinkedInConnector", "LinkedInError"]

_log = get_logger(__name__)

API_BASE = "https://api.linkedin.com"
TOKEN_ENV = "LINKEDIN_ACCESS_TOKEN"
DOCS_URL = "https://learn.microsoft.com/linkedin"

REQUIRED_SCOPES = ("openid", "profile", "w_member_social")


class LinkedInError(ConnectorError):
    """A LinkedIn API call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        service_error_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.service_error_code = service_error_code


@register_connector
class LinkedInConnector(Connector):
    """Devon's LinkedIn API adapter."""

    id = "linkedin"
    name = "LinkedIn"
    description = (
        "LinkedIn REST API: identify the member, publish posts to the "
        "feed, and read back own posts. No consumer home-feed read — the "
        "API does not offer one."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an OAuth access token and vault it.

        Takes a pre-obtained access token (the LinkedIn OAuth dance
        happens in the owner's browser). Without ``token`` the grant guide
        prints and the flow fails fast.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "linkedin is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        try:
            secret = (token or "").strip() or prompt_secret(
                "LinkedIn OAuth access token", env_var=TOKEN_ENV
            )
        except ConnectorError as exc:
            raise ConnectorError(
                "no access token given. Create a LinkedIn app at "
                "https://developer.linkedin.com, add the Sign In with "
                "LinkedIn and Share on LinkedIn products, authorize with "
                f"scopes {', '.join(REQUIRED_SCOPES)} — then pass "
                "token=<access token>."
            ) from exc
        info = self._api("GET", "/v2/userinfo", token=secret)
        person_id = str(info.get("sub", ""))
        name = str(info.get("name", "")) or person_id
        if not person_id:
            raise LinkedInError(
                "linkedin userinfo returned no subject (sub) — the token "
                "is missing the openid scope"
            )
        self._store_credential(
            name,
            secret,
            credential_type="oauth2",
            scopes=list(REQUIRED_SCOPES),
            metadata={
                "person_id": person_id,
                "person_urn": f"urn:li:person:{person_id}",
                "email": info.get("email", ""),
            },
        )
        _log.info("linkedin connected as %s", name)
        return ConnectResult(
            ok=True,
            account=name,
            scopes=list(REQUIRED_SCOPES),
            message=(
                f"connected as LinkedIn member {name} "
                f"(urn:li:person:{person_id}). The access token is in the "
                "encrypted vault. Posting uses /rest/posts with your "
                "member URN."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name linkedin`",
            )
        try:
            info = self._api("GET", "/v2/userinfo", token=cred.password)
        except LinkedInError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): it expired or was "
                       "revoked — reconnect with a fresh token",
            )
        return ConnectorStatus(
            connected=True,
            account=str(info.get("name", cred.username)),
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"member sub {info.get('sub')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/v2/userinfo", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── LinkedIn API ─────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """Member identity (``GET /v2/userinfo``) — also the verifier."""
        data = self._api("GET", "/v2/userinfo")
        return data if isinstance(data, dict) else {}

    def post(
        self,
        text: str,
        *,
        visibility: str = "PUBLIC",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Publish a text post (``POST /rest/posts``).

        Needs the member URN stored at connect time; if it is missing the
        call fails fast and asks for a reconnect. ``visibility`` is
        ``PUBLIC`` or ``CONNECTIONS``. Consequential: gated behind explicit
        owner confirmation.
        """
        text = text or ""
        if not text.strip():
            raise ConnectorError("refusing to post an empty update")
        if len(text) > 3000:
            raise ConnectorError(
                f"post is {len(text)} chars; LinkedIn caps commentary at "
                "3000 — shorten it first"
            )
        if visibility not in ("PUBLIC", "CONNECTIONS"):
            raise ConnectorError(
                f"invalid visibility {visibility!r}: use 'PUBLIC' or "
                "'CONNECTIONS'"
            )
        person_urn = (self._require_credential().metadata or {}).get(
            "person_urn"
        )
        if not person_urn:
            raise ConnectorError(
                "no member URN on the stored credential — reconnect so the "
                "identity can be resolved from /v2/userinfo"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="post",
            title=f"Post to LinkedIn ({visibility}, {len(text)} chars)",
            instructions="\n".join([
                "Devon wants to publish this post on your LinkedIn.",
                "Review it — posting is public and final.",
                f"Visibility: {visibility}",
                "",
                text,
            ]),
            resume_state={"text": text, "visibility": visibility},
        )
        payload = {
            "author": person_urn,
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {},
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        data = self._api("POST", "/rest/posts", payload=payload)
        return data if isinstance(data, dict) else {}

    def get_post(self, post_id: str) -> dict[str, Any]:
        """Read one post (``GET /rest/posts/{id}``).

        ``post_id`` is the URN id, e.g.
        ``urn:li:share:12345`` or just ``12345``.
        """
        post_id = (post_id or "").strip()
        if not post_id:
            raise ConnectorError("post_id is required")
        data = self._api("GET", f"/rest/posts/{post_id}")
        return data if isinstance(data, dict) else {}

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "linkedin is not connected — run "
                "`nm connectors connect --name linkedin` first"
            )
        return cred

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        token: str | None = None,
    ) -> Any:
        """One LinkedIn API call; failures become LinkedInError.

        LinkedIn errors arrive as
        ``{"status", "serviceErrorCode", "message"}`` — unwrapped below.
        """
        secret = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {secret}",
            "LinkedIn-Version": "202406",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise LinkedInError(
                "linkedin rate limit hit (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise LinkedInError(f"linkedin request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise LinkedInError(f"linkedin request failed: {exc}") from exc
        if resp.status == 401:
            detail, code = self._error_detail(resp)
            raise LinkedInError(
                "linkedin rejected the access token (401): "
                f"{detail or 'expired or revoked'} — reconnect with a "
                "fresh token",
                status_code=401,
                service_error_code=code,
            )
        if resp.status == 403:
            detail, code = self._error_detail(resp)
            raise LinkedInError(
                "linkedin refused (403): "
                f"{detail or 'missing product access or scope'} — posting "
                "needs the Share on LinkedIn product + w_member_social",
                status_code=403,
                service_error_code=code,
            )
        if resp.status == 429:
            raise LinkedInError(
                "linkedin rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            detail, code = self._error_detail(resp)
            raise LinkedInError(
                f"linkedin {method} {path} failed ({resp.status}): "
                f"{detail or resp.text[:200]}",
                status_code=resp.status,
                service_error_code=code,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise LinkedInError(
                f"linkedin {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, int]:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200], 0
        if isinstance(body, dict):
            message = str(body.get("message", ""))[:200]
            try:
                code = int(body.get("serviceErrorCode", 0))
            except (TypeError, ValueError):
                code = 0
            return message or resp.text[:200], code
        return resp.text[:200], 0
