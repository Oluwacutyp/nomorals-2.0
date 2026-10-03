"""X (Twitter) API v2 connector.

Drives X over the v2 REST API (https://docs.x.com) with the framework
HttpClient — no tweepy dependency.

Auth: one bearer token (``API_KEY``) in the ``Authorization: Bearer``
header. For read-only access any project app's bearer token works; posting
tweets needs a paid tier (Basic or higher) — the free tier is read-only and
heavily rate-limited (search is capped, posting is rejected).

Honest limits: the v2 API cannot read the home timeline of the
authenticated user (there is no consumer home-timeline endpoint), cannot
post media without a separate chunked upload, and free-tier search returns
at most 100 posts per request with strict monthly caps.
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

__all__ = ["XConnector", "XError"]

_log = get_logger(__name__)

API_BASE = "https://api.x.com/2"
TOKEN_ENV = "X_BEARER_TOKEN"
DOCS_URL = "https://docs.x.com"
MAX_TWEET_CHARS = 280


class XError(ConnectorError):
    """An X API v2 call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        x_error_code: int = 0,
        title: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.x_error_code = x_error_code
        self.title = title


@register_connector
class XConnector(Connector):
    """Devon's X (Twitter) API v2 adapter."""

    id = "x"
    name = "X"
    description = (
        "X API v2: identify the app user, post tweets, read a user's "
        "timeline, and search recent posts. Authenticates with a bearer "
        "token. Posting requires a paid tier (Basic+); free is read-only."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a bearer token against GET /users/me and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "x is already connected — one account per service. "
                "Disconnect first to switch tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "X bearer token", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError("empty bearer token: nothing to connect with")
        body = self._api("GET", "/users/me", token=secret)
        me = body.get("data") if isinstance(body, dict) else {}
        if not isinstance(me, dict):
            raise XError("x /users/me returned an unexpected response shape")
        username = str(me.get("username", ""))
        label = f"@{username}" if username else f"user {me.get('id', '?')}"
        self._store_credential(
            label,
            secret,
            credential_type="api_key",
            scopes=["tweet.read", "users.read", "tweet.write"],
            metadata={"user_id": me.get("id"), "username": username},
        )
        _log.info("x connected as %s", label)
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["tweet.read", "users.read", "tweet.write"],
            message=(
                f"connected as X user {label} (id {me.get('id')}). The "
                "token is in the encrypted vault. Note: posting tweets "
                "needs a paid X API tier (Basic or higher); a free-tier "
                "bearer token is read-only and heavily rate-limited."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name x`",
            )
        try:
            me = self._api("GET", "/users/me", token=cred.password)
        except XError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"bearer token rejected ({exc}): regenerate it in "
                       "the X developer portal and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=f"@{me.get('username', cred.username)}",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"user id {me.get('id')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/users/me", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── API v2 ───────────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """The authenticated user's profile (``GET /users/me``)."""
        data = self._api("GET", "/users/me")
        user = data.get("data") if isinstance(data, dict) else None
        return user if isinstance(user, dict) else {}

    def post_tweet(
        self,
        text: str,
        *,
        reply_to_tweet_id: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Post a tweet (``POST /tweets``).

        X caps tweets at 280 characters — longer input is rejected, not
        truncated. Needs a paid API tier; free-tier tokens are read-only.
        Consequential: gated behind explicit owner confirmation.
        """
        text = text or ""
        if not text.strip():
            raise ConnectorError("refusing to post an empty tweet")
        if len(text) > MAX_TWEET_CHARS:
            raise ConnectorError(
                f"tweet is {len(text)} chars; X caps tweets at "
                f"{MAX_TWEET_CHARS} — shorten it first"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="post_tweet",
            title=f"Post tweet ({len(text)} chars)",
            instructions="\n".join([
                "Devon wants to post this tweet on your X account.",
                "Review it — posting is public and final.",
                "",
                text,
            ]),
            resume_state={"text": text,
                          "reply_to_tweet_id": reply_to_tweet_id},
        )
        payload: dict[str, Any] = {"text": text}
        if reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to_tweet_id}
        data = self._api("POST", "/tweets", payload=payload)
        tweet = data.get("data") if isinstance(data, dict) else None
        return tweet if isinstance(tweet, dict) else {}

    def get_timeline(
        self,
        user_id: str = "",
        *,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """A user's recent tweets (``GET /users/{id}/tweets``).

        Defaults to the authenticated user. The v2 API has no consumer
        home-timeline endpoint — this reads one user's posts, not the
        follower feed.
        """
        if not user_id:
            user_id = str(self.get_me().get("id", ""))
        if not user_id:
            raise ConnectorError(
                "could not determine the user id — pass user_id explicitly"
            )
        params = {
            "max_results": max(5, min(max_results, 100)),
            "tweet.fields": "created_at,public_metrics",
        }
        data = self._api("GET", f"/users/{user_id}/tweets", params=params)
        tweets = data.get("data") if isinstance(data, dict) else None
        return tweets if isinstance(tweets, list) else []

    def search_recent(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search recent posts (``GET /tweets/search/recent``).

        Free tier: strict monthly post cap and max 100 results per
        request — search is the first thing to be throttled.
        """
        query = (query or "").strip()
        if not query:
            raise ConnectorError("refusing to search with an empty query")
        params = {
            "query": query,
            "max_results": max(10, min(max_results, 100)),
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        data = self._api("GET", "/tweets/search/recent", params=params)
        tweets = data.get("data") if isinstance(data, dict) else None
        return tweets if isinstance(tweets, list) else []

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "x is not connected — run `nm connectors connect --name x` "
                "first"
            )
        return cred

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        """One X API v2 call; failures become XError.

        X returns errors as ``{"title", "detail", "status"}`` or
        ``{"errors": [{"message", "code"}]}``; both shapes are unwrapped.
        """
        secret = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        headers = {"Authorization": f"Bearer {secret}"}
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=headers
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise XError(
                "x rate limit hit (429): free-tier quotas are tiny — back "
                f"off and retry after ~{exc.retry_after:.0f}s",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise XError(f"x request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise XError(f"x request failed: {exc}") from exc
        if resp.status == 401:
            title, detail, code = self._error_detail(resp)
            raise XError(
                "x rejected the bearer token (401): "
                f"{detail or title or 'invalid or revoked'} — regenerate it "
                "in the developer portal and reconnect",
                status_code=401,
                x_error_code=code,
                title=title,
            )
        if resp.status == 403:
            title, detail, code = self._error_detail(resp)
            raise XError(
                "x refused (403): "
                f"{detail or title or 'forbidden'} — posting on a free-tier "
                "token is read-only; upgrade to Basic or higher",
                status_code=403,
                x_error_code=code,
                title=title,
            )
        if resp.status == 429:
            raise XError(
                "x rate limit hit (429): free-tier quotas are tiny — back "
                "off before retrying",
                status_code=429,
            )
        if not resp.ok:
            title, detail, code = self._error_detail(resp)
            raise XError(
                f"x {method} {path} failed ({resp.status}): "
                f"{detail or title or resp.text[:200]}",
                status_code=resp.status,
                x_error_code=code,
                title=title,
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise XError(
                f"x {method} {path} returned invalid JSON"
            ) from exc
        if isinstance(body, dict) and body.get("errors"):
            title, detail, code = self._error_detail(resp)
            raise XError(
                f"x {method} {path} failed: {detail or title}",
                status_code=resp.status,
                x_error_code=code,
                title=title,
            )
        return body

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, str, int]:
        """Unwrap X's two error shapes → (title, detail, code)."""
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return "", resp.text[:200], 0
        if not isinstance(body, dict):
            return "", resp.text[:200], 0
        if body.get("title"):
            detail = str(body.get("detail", ""))[:200]
            try:
                code = int(body.get("status", 0))
            except (TypeError, ValueError):
                code = 0
            return str(body["title"])[:120], detail, code
        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            message = str(first.get("message", ""))[:200]
            try:
                code = int(first.get("code", 0))
            except (TypeError, ValueError):
                code = 0
            title = str(first.get("title", ""))[:120]
            return title, message, code
        return "", resp.text[:200], 0

