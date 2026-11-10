"""X adapter — native text posting on the X API v2, beside Postiz.

The Postiz route stays for owners who run Postiz; this adapter posts
directly with the account's own user-context token (``POST /2/tweets``)
so X is a first-class draft platform like the rest.

The failure shape mirrors the contentops video publisher
(``media/contentops/publish/x.py``) on purpose — one platform truth,
worded the same everywhere: X's write APIs need a PAID tier (Basic or
Pro). Free-tier keys are read-only and app-only tokens cannot post, so
a missing token or a 403 is a *tier/connection refusal* carrying the
exact manual step that unblocks it — never a generic error, never a
fake success, and never retryable (retrying cannot change the tier).

The token is the social :class:`Account`'s credential reference —
connect the account with ``credentials="env:X_ACCESS_TOKEN"`` (the
same env name the video publisher uses) and the adapter resolves it
through :meth:`Account.resolve_token`; no secret is handled in code.

Metrics ARE real here: ``GET /2/tweets/{id}?tweet.fields=
public_metrics`` is a genuine v2 read the same token can make, mapped
to the house engagement keys. Best-effort like Mastodon's: any failure
returns ``{}`` so the measurement spine counts the row ``empty`` and
never overwrites stored data.
"""

from __future__ import annotations

import time
from typing import Any

from ...core.http import HttpClient
from ..base import (
    ERROR_AUTH,
    Account,
    PlatformAdapter,
    PostResult,
    classify_http_error,
)

__all__ = ["XAdapter"]

_API = "https://api.x.com/2"

#: The exact unblock step, mirroring the video publisher's wording.
_MANUAL_STEP = (
    "create an app at developer.x.com, enable OAuth user context with "
    "tweet.write scope, note that X write APIs need a PAID tier "
    "(Basic/Pro) — free keys are read-only — then connect the x account "
    "with the user access token (env X_ACCESS_TOKEN)")

#: Appended to a 403, verbatim in shape from the video publisher.
_TIER_HINT = (
    "403 on post usually means the app is on X's free tier (write "
    "needs Basic/Pro) or the token is app-only instead of user context")


class XAdapter(PlatformAdapter):
    """Native X text posts (+ replies) with honest tier refusals."""

    name = "x"
    max_chars = 280
    supports_media = False  # media upload lives in the video publisher
    supports_thread = True
    supports_reply = True  # post() threads reply.in_reply_to_tweet_id

    def __init__(self, http: HttpClient | None = None) -> None:
        self._http = http

    def _client(self) -> HttpClient:
        return self._http or HttpClient(timeout=30.0)

    @staticmethod
    def _payload(resp: Any) -> dict[str, Any]:
        raw = resp.json() if callable(getattr(resp, "json", None)) else resp.json
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _api_error(resp: Any) -> str:
        data = XAdapter._payload(resp)
        detail = str(data.get("detail") or data.get("title") or "")
        errors = data.get("errors")
        if not detail and isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                detail = str(first.get("message") or first.get("detail") or "")
        return detail or (resp.text or "")[:300] or f"HTTP {resp.status}"

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        token = account.resolve_token()
        if not token:
            return PostResult(
                platform=self.name, ok=False,
                error=("x: no X user access token — the account isn't "
                       f"connected\nManual step: {_MANUAL_STEP}"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        payload: dict[str, Any] = {"text": content}
        reply_to = str(kwargs.get("reply_to") or "")
        if reply_to:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}
        try:
            resp = self._client().request(
                "POST", f"{_API}/tweets",
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                json=payload)
        except Exception as exc:  # noqa: BLE001 - transport failure → words
            return PostResult(
                platform=self.name, ok=False, error=str(exc),
                error_code=classify_http_error(0, str(exc)),
                seconds=time.perf_counter() - started)
        if resp.ok:
            data = self._payload(resp).get("data")
            tweet_id = ""
            if isinstance(data, dict):
                tweet_id = str(data.get("id") or "")
            handle = (account.handle or "").lstrip("@")
            url = (f"https://x.com/{handle}/status/{tweet_id}"
                   if tweet_id and handle else "")
            return PostResult(
                platform=self.name, ok=True, external_id=tweet_id, url=url,
                seconds=time.perf_counter() - started)
        message = f"x post failed: {self._api_error(resp)}"
        if resp.status == 403:
            message += f"\n{_TIER_HINT}"
        return PostResult(
            platform=self.name, ok=False, error=message,
            error_code=classify_http_error(resp.status, message),
            status_code=int(resp.status or 0),
            seconds=time.perf_counter() - started)

    def read_engagement(
        self, account: Account, external_id: str
    ) -> list[dict[str, Any]]:
        """Replies/quotes on one post, via conversation search.

        X has no per-tweet replies endpoint; the documented read is
        recent search over the post's conversation
        (``GET /2/tweets/search/recent?query=conversation_id:{id}``)
        with author expansions — the same token and transport as
        :meth:`metrics`. Pure retweets carry no words of their own
        and are skipped, as are the owner's own thread follow-ups.
        Unlike metrics, a failed read RAISES (with the tier hint on
        a 403): the engagement poll records failures by name, and a
        dead read must never read as an empty thread.
        """
        from ..base import SocialError

        token = account.resolve_token()
        if not token:
            raise SocialError(
                "x: no X user access token — the account isn't "
                f"connected\nManual step: {_MANUAL_STEP}")
        resp = self._client().request(
            "GET", f"{_API}/tweets/search/recent",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "query": f"conversation_id:{external_id}",
                "tweet.fields": "created_at,referenced_tweets",
                "expansions": "author_id",
                "user.fields": "name,username",
                "max_results": "100",
            })
        if not resp.ok:
            message = (f"x engagement read failed for post "
                       f"{external_id}: {self._api_error(resp)}")
            if resp.status == 403:
                message += f"\n{_TIER_HINT}"
            raise SocialError(message)
        payload = self._payload(resp)
        users: dict[str, dict[str, Any]] = {}
        includes = payload.get("includes")
        if isinstance(includes, dict):
            for user in includes.get("users") or []:
                if isinstance(user, dict) and user.get("id"):
                    users[str(user["id"])] = user
        own = (account.handle or "").lstrip("@").strip().lower()
        items: list[dict[str, Any]] = []
        for tweet in payload.get("data") or []:
            if not isinstance(tweet, dict):
                continue
            tweet_id = str(tweet.get("id") or "")
            if not tweet_id or tweet_id == str(external_id):
                continue  # the post itself is not engagement on it
            refs = [r for r in (tweet.get("referenced_tweets") or [])
                    if isinstance(r, dict)]
            ref_types = {str(r.get("type") or "") for r in refs}
            if ref_types == {"retweeted"}:
                continue  # a bare repost has no words to answer
            user = users.get(str(tweet.get("author_id") or ""), {})
            username = str(user.get("username") or "")
            if own and username.lower() == own:
                continue  # his own thread follow-ups are not engagement
            if "replied_to" in ref_types:
                kind = "reply"
            elif "quoted" in ref_types:
                kind = "comment"
            else:
                kind = "reply"
            items.append({
                "item_id": tweet_id,
                "author": str(user.get("name") or username),
                "text": str(tweet.get("text") or ""),
                "kind": kind,
                "created_at": _parse_ts(tweet.get("created_at")),
            })
        return items

    def metrics(self, account: Account, external_id: str) -> dict[str, Any]:
        """Real engagement read via ``public_metrics`` (best-effort)."""
        token = account.resolve_token()
        if not token or not external_id:
            return {}
        try:
            resp = self._client().request(
                "GET", f"{_API}/tweets/{external_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"tweet.fields": "public_metrics"})
        except Exception:  # noqa: BLE001 - metrics are best-effort
            return {}
        if not resp.ok:
            return {}
        data = self._payload(resp).get("data")
        if not isinstance(data, dict):
            return {}
        pub = data.get("public_metrics")
        if not isinstance(pub, dict):
            return {}
        out: dict[str, Any] = {}
        for src, dst in (("like_count", "likes"), ("reply_count", "replies"),
                         ("retweet_count", "reposts"),
                         ("quote_count", "quotes"),
                         ("impression_count", "impressions")):
            val = pub.get(src)
            if isinstance(val, (int, float)):
                out[dst] = val
        return out

    def health(self, account: Account) -> bool:
        return bool(account.resolve_token())


def _parse_ts(raw: Any) -> float:
    """ISO-8601 → epoch float; 0.0 when the platform gave nothing."""
    if not raw:
        return 0.0
    try:
        import datetime

        return datetime.datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0
