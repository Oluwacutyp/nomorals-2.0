"""Bluesky adapter — the AT Protocol HTTP API.

Bluesky needs a session JWT rather than a long-lived token: you authenticate with
a handle and an app password, and the access token expires. So the adapter logs in
on demand and caches the session per handle until it stops working.

The app password is the credential, never the account password. That distinction is
the whole reason Bluesky has app passwords.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from ..base import Account, PlatformAdapter, PostResult

__all__ = ["Adapter"]

_log = get_logger(__name__)

DEFAULT_PDS = "https://bsky.social"

#: A Bluesky post: 300 graphemes AND 3000 UTF-8 bytes.
BSKY_GRAPHEME_LIMIT = 300
BSKY_BYTE_LIMIT = 3000
#: Auto-thread cap: longer content is rejected, not silently eaten.
BSKY_MAX_THREAD_POSTS = 10

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_MENTION_RE = re.compile(
    r"(?:^|(?<=\s))@([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)"
)
# Approximation of the official client's TAG_REGEX: text start or
# whitespace before #, trailing punctuation stripped, ≥1 non-digit
# non-punctuation char, 64-char cap.
_TAG_RE = re.compile(r"(?:^|(?<=\s))#([A-Za-z0-9_][A-Za-z0-9_]{0,63})")
_TRAILING_PUNCT = ".,;:!?…)}]\"'"


def _byte_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Char offsets → UTF-8 byte offsets (what facets use)."""
    raw = text.encode("utf-8")
    return (len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8")))


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(s < span[1] and span[0] < e for s, e in spans)


def detect_facets(text: str,
                  resolve_mention: Any = None) -> list[dict[str, Any]]:
    """Link + mention + hashtag facets for a post, with UTF-8 byte offsets.

    Pure except ``resolve_mention``: an optional ``handle → did`` callable
    used for @mentions (without a resolved DID a mention facet is invalid,
    so unresolved mentions are left as plain text). Hashtag facets never
    overlap link facets; a tag clipped by truncation is dropped.
    """
    text = text or ""
    facets: list[dict[str, Any]] = []
    claimed: list[tuple[int, int]] = []

    def _add(start: int, end: int, feature: dict[str, Any]) -> None:
        span = _byte_span(text, start, end)
        if _overlaps(span, claimed):
            return
        claimed.append(span)
        facets.append({
            "index": {"byteStart": span[0], "byteEnd": span[1]},
            "features": [feature],
        })

    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        if not url:
            continue
        end = m.start() + len(url)
        _add(m.start(), end, {
            "$type": "app.bsky.richtext.facet#link", "uri": url})

    if resolve_mention is not None:
        for m in _MENTION_RE.finditer(text):
            handle = m.group(1)
            try:
                did = resolve_mention(handle)
            except Exception:  # noqa: BLE001 - unresolved → plain text
                did = ""
            if did:
                _add(m.start(), m.end(), {
                    "$type": "app.bsky.richtext.facet#mention", "did": did})

    for m in _TAG_RE.finditer(text):
        tag = m.group(1).rstrip("_")
        if not tag or tag[0].isdigit():
            continue
        _add(m.start(1) - 1, m.start(1) + len(tag), {
            "$type": "app.bsky.richtext.facet#tag", "tag": tag})
    return facets


def _graphemes(text: str) -> int:
    """Best-effort grapheme count (ZWJ/combining marks don't start clusters)."""
    import unicodedata

    return sum(
        1 for ch in text
        if ch != "\u200d" and ch != "\ufe0f" and not unicodedata.combining(ch)
    )


def split_bluesky_text(text: str) -> list[str]:
    """Token-aware split like atproto.dart's ``split()``: never cuts a
    handle, link, or #tag in half, and every chunk respects BOTH the
    300-grapheme and 3000-byte limits. Split BEFORE facet detection —
    facets are measured on each chunk, not the raw input.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    def _fits(chunk: str) -> bool:
        return (_graphemes(chunk) <= BSKY_GRAPHEME_LIMIT
                and len(chunk.encode("utf-8")) <= BSKY_BYTE_LIMIT)

    if _fits(text):
        return [text]

    # Protect tokens (URLs, @mentions, #tags) — a chunk boundary may only
    # fall on whitespace outside a token.
    protected: list[tuple[int, int]] = [
        (m.start(), m.end())
        for pat in (_URL_RE, _MENTION_RE, _TAG_RE)
        for m in pat.finditer(text)
    ]

    def _inside_token(pos: int) -> bool:
        return any(s < pos < e for s, e in protected)

    chunks: list[str] = []
    cur = ""
    for word in text.split(" "):
        # A pathological single token longer than the limit: hard-cut it.
        while not _fits(word):
            cut = len(word) // 2 or 1
            while cut > 1 and _inside_token(0):
                cut -= 1
            chunks.append(word[:cut])
            word = word[cut:]
            if not word:
                break
        if not word:
            continue
        piece = f"{cur} {word}".strip() if cur else word
        if _fits(piece):
            cur = piece
        else:
            if cur:
                chunks.append(cur)
            cur = word
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c] or [text[:BSKY_GRAPHEME_LIMIT]]


def external_embed(uri: str, title: str, description: str = "") -> dict[str, Any]:
    """An ``app.bsky.embed.external`` link card for a post."""
    return {
        "$type": "app.bsky.embed.external",
        "external": {
            "uri": uri, "title": title[:300], "description": description[:1000],
        },
    }


class Adapter(PlatformAdapter):
    name = "bluesky"
    max_chars = 300
    supports_media = True
    supports_thread = True

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._sessions: dict[str, dict[str, Any]] = {}
        self._did_cache: dict[str, str] = {}

    def _pds(self, account: Account) -> str:
        return str(account.limits.get("pds") or DEFAULT_PDS).rstrip("/")

    def _login(self, account: Account) -> dict[str, Any]:
        """Create or reuse a session. App password in, access token out."""
        cached = self._sessions.get(account.handle)
        if cached and cached.get("accessJwt"):
            return cached
        password = account.resolve_token()
        if not password:
            raise ValueError(f"no app password configured for {account.handle}")
        client = HttpClient(timeout=self.timeout)
        response = client.post_json(
            f"{self._pds(account)}/xrpc/com.atproto.server.createSession",
            {"identifier": account.handle, "password": password},
        )
        if not response.ok:
            raise ValueError(f"bluesky login failed ({response.status}): {response.text[:200]}")
        session = json.loads(response.text or "{}")
        self._sessions[account.handle] = session
        return session

    def validate(self, content: str) -> str:
        """Over-limit content is allowed up to the thread cap — ``post()``
        threads it automatically. Beyond the cap it is still rejected."""
        text = (content or "").strip()
        if not text:
            from ..base import ValidationError
            raise ValidationError("post content is empty", field="content")
        chunks = split_bluesky_text(text)
        if len(chunks) > BSKY_MAX_THREAD_POSTS:
            from ..base import ValidationError
            raise ValidationError(
                f"bluesky thread would need {len(chunks)} posts "
                f"(cap {BSKY_MAX_THREAD_POSTS})", field="content")
        return text

    def _resolve_handle(self, account: Account, handle: str) -> str:
        """handle → DID via the PDS, cached per adapter."""
        if handle in self._did_cache:
            return self._did_cache[handle]
        import urllib.parse

        client = HttpClient(timeout=self.timeout)
        query = urllib.parse.urlencode({"handle": handle})
        response = client.get(
            f"{self._pds(account)}/xrpc/com.atproto.identity.resolveHandle?{query}"
        )
        did = ""
        if response.ok:
            try:
                did = str(json.loads(response.text or "{}").get("did", ""))
            except json.JSONDecodeError:
                did = ""
        if did:
            self._did_cache[handle] = did
        return did

    def _create_post(self, account: Account, session: dict[str, Any],
                     record: dict[str, Any]) -> dict[str, Any]:
        """One createRecord call. Returns the parsed response (or {})."""
        client = HttpClient(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {session.get('accessJwt', '')}"},
        )
        response = client.post_json(
            f"{self._pds(account)}/xrpc/com.atproto.repo.createRecord",
            {"repo": session.get("did", ""),
             "collection": "app.bsky.feed.post", "record": record},
        )
        if not response.ok:
            if response.status in {400, 401} and "auth" in response.text.lower():
                self._sessions.pop(account.handle, None)
            raise RuntimeError(f"bluesky post failed ({response.status}): "
                               f"{response.text[:300]}")
        try:
            return json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return {}

    def post_thread(self, account: Account, texts: list[str],
                    **kwargs: Any) -> PostResult:
        """Post a thread: each chunk replies to the previous, all rooted
        at the first. Returns one PostResult for the thread (external_id
        and url point at the first post; metrics carry every post)."""
        from ..base import ERROR_VALIDATION, classify_http_error

        started = time.perf_counter()
        texts = [t for t in (texts or []) if (t or "").strip()]
        if not texts:
            return PostResult(platform=self.name, ok=False,
                              error="thread has no posts",
                              error_code=ERROR_VALIDATION)
        try:
            session = self._login(account)
        except Exception as exc:  # noqa: BLE001 - auth failure is a result
            return PostResult(platform=self.name, ok=False, error=str(exc),
                              error_code=classify_http_error(0, str(exc)),
                              seconds=time.perf_counter() - started)

        langs = account.limits.get("langs") or ["en"]
        link_card = kwargs.get("link_card") or {}
        posted: list[dict[str, str]] = []
        try:
            root_ref: dict[str, str] = {}
            parent_ref: dict[str, str] = {}
            for i, chunk in enumerate(texts):
                facets = detect_facets(
                    chunk,
                    resolve_mention=lambda h, _a=account: self._resolve_handle(_a, h))
                record: dict[str, Any] = {
                    "$type": "app.bsky.feed.post",
                    "text": chunk,
                    "createdAt": _now_iso(),
                    "langs": langs,
                }
                if facets:
                    record["facets"] = facets
                if parent_ref:
                    record["reply"] = {"root": root_ref, "parent": parent_ref}
                elif kwargs.get("reply_to"):
                    record["reply"] = kwargs["reply_to"]
                if i == 0 and link_card.get("uri"):
                    record["embed"] = external_embed(
                        str(link_card["uri"]), str(link_card.get("title", "")),
                        str(link_card.get("description", "")))
                data = self._create_post(account, session, record)
                uri = str(data.get("uri", ""))
                ref = {"uri": uri, "cid": str(data.get("cid", ""))}
                if i == 0:
                    root_ref = ref
                parent_ref = ref
                posted.append({"uri": uri,
                               "url": _post_url(account.handle, uri)})
        except Exception as exc:  # noqa: BLE001 - partial thread is a result
            return PostResult(
                platform=self.name, ok=False,
                error=f"thread stopped after {len(posted)}/{len(texts)} posts: {exc}",
                error_code=classify_http_error(
                    getattr(exc, "status_code", 0) or 0, str(exc)),
                seconds=time.perf_counter() - started,
                metrics={"posted": posted, "total": len(texts)},
            )
        first = posted[0]
        return PostResult(
            platform=self.name, ok=True,
            external_id=first["uri"].rsplit("/", 1)[-1], url=first["url"],
            seconds=time.perf_counter() - started,
            metrics={"posts": posted, "thread_posts": len(posted)},
        )

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        texts = split_bluesky_text(content or "")
        if len(texts) > 1:
            # Over-limit content threads automatically (validate() already
            # enforced the cap) — one logical post, N chained records.
            result = self.post_thread(account, texts, **kwargs)
            result.seconds = time.perf_counter() - started
            return result
        try:
            session = self._login(account)
        except Exception as exc:  # noqa: BLE001 - auth failure is a result
            from ..base import classify_http_error
            return PostResult(platform=self.name, ok=False, error=str(exc),
                              error_code=classify_http_error(0, str(exc)),
                              seconds=time.perf_counter() - started)

        langs = account.limits.get("langs") or ["en"]
        text = texts[0] if texts else ""
        facets = detect_facets(
            text, resolve_mention=lambda h: self._resolve_handle(account, h))
        record: dict[str, Any] = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": _now_iso(),
            "langs": langs,
        }
        if facets:
            record["facets"] = facets
        reply_to = kwargs.get("reply_to") or ""
        if reply_to:
            record["reply"] = reply_to  # caller supplies the full reply ref
        link_card = kwargs.get("link_card") or {}
        if link_card.get("uri"):
            record["embed"] = external_embed(
                str(link_card["uri"]), str(link_card.get("title", "")),
                str(link_card.get("description", "")))

        try:
            data = self._create_post(account, session, record)
        except Exception as exc:  # noqa: BLE001
            from ..base import classify_http_error
            return PostResult(
                platform=self.name, ok=False, error=str(exc),
                error_code=classify_http_error(
                    getattr(exc, "status_code", 0) or 0, str(exc)),
                seconds=time.perf_counter() - started)

        uri = str(data.get("uri", ""))
        return PostResult(
            platform=self.name, ok=True, external_id=uri.rsplit("/", 1)[-1],
            url=_post_url(account.handle, uri),
            seconds=time.perf_counter() - started,
        )

    def health(self, account: Account) -> bool:
        try:
            self._login(account)
            return True
        except Exception:  # noqa: BLE001
            return False


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _post_url(handle: str, uri: str) -> str:
    """Build a bsky.app permalink from an at:// URI."""
    if not uri:
        return ""
    parts = uri.split("/")
    rkey = parts[-1] if parts else ""
    name = handle.split("@")[-1] if "@" in handle else handle
    return f"https://bsky.app/profile/{name}/post/{rkey}"
