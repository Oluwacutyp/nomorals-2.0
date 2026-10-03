"""Notion connector — pages and databases over the Notion API.

Docs: https://developers.notion.com/

Auth: internal integration token (``AuthMethod.API_KEY``) as a Bearer
token, created at https://www.notion.so/my-integrations (starts with
``secret_`` or ``ntn_``). Every request also carries
``Notion-Version: 2022-06-28``. The integration must be shared into
each page/database it should touch (⋯ → Connections) or the API sees
nothing — a 404/empty result usually means "not shared", not "missing".

Writes are consequential: ``create_page`` and ``update_page`` never run
on implied consent — they require ``confirmed=True`` (owner approved the
exact content) or a human checkpoint when ``db`` is given. Reads never
need confirmation.
"""

from __future__ import annotations

import json
import time
from typing import Any

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
from .checkpoints import CheckpointState
from .registry import register_connector

__all__ = ["NotionConnector", "NotionError"]

_log = get_logger(__name__)

API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
TOKEN_ENV = "NOTION_API_TOKEN"


class NotionError(ConnectorError):
    """A Notion API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, error_code: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@register_connector
class NotionConnector(Connector):
    """Devon's Notion adapter: pages, databases, search."""

    id = "notion"
    name = "Notion"
    description = (
        "Notion API: create/get/update pages, query databases, and "
        "search. Internal integration token auth; page writes require "
        "explicit owner confirmation."
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
        """Validate an integration token against /v1/search and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "notion is already connected — one account per service. "
                "Disconnect first to switch integration tokens."
            )
        secret = (token or "").strip() or prompt_secret(
            "Notion integration token", env_var=TOKEN_ENV
        )
        if not secret:
            raise ConnectorError(
                "empty integration token: nothing to connect with"
            )
        data = self._api("POST", "/search", token=secret,
                         body={"page_size": 1})
        n = len(data.get("results", [])) if isinstance(data, dict) else 0
        self._store_credential(
            "notion",
            secret,
            credential_type="api_key",
            scopes=["pages:read", "pages:write", "databases:read"],
            metadata={},
        )
        _log.info("notion connected (search probe ok)")
        return ConnectResult(
            ok=True,
            account="notion",
            scopes=["pages:read", "pages:write", "databases:read"],
            message=(
                "connected to Notion (integration token valid; the probe "
                f"search returned {n} item(s) — if this is 0, share pages "
                "or databases with the integration via ⋯ → Connections). "
                "The token is in the encrypted vault. Every page write "
                "still needs your explicit confirmation at call time."
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
                       "--name notion`",
            )
        try:
            self._api("POST", "/search", body={"page_size": 1})
        except NotionError as exc:
            return ConnectorStatus(
                connected=False,
                account="notion",
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): rotate it at "
                       "notion.so/my-integrations and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account="notion",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="integration token valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("POST", "/search", body={"page_size": 1})
            return True
        except ConnectorError:
            return False

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Execute a confirmed write after the owner resolved it."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the write first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        payload = dict((checkpoint.resume_state or {}).get("payload", {}))
        if stage == "create_page":
            if not payload.get("parent"):
                raise NotionError(
                    "the resolved checkpoint has no page payload — "
                    "it cannot create the page"
                )
            return self._create_now(payload)
        if stage == "update_page":
            if not payload.get("page_id"):
                raise NotionError(
                    "the resolved checkpoint has no update payload"
                )
            return self._update_now(payload)
        raise NotionError(
            f"notion cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def get_page(self, page_id: str) -> dict[str, Any]:
        """Page properties (``GET /v1/pages/{id}``).

        404 usually means the integration was never shared into the page
        (⋯ → Connections), not that the page is gone.
        """
        page_id = self._clean_id(page_id, "page_id")
        data = self._api("GET", f"/pages/{page_id}")
        return data if isinstance(data, dict) else {}

    def get_page_content(self, page_id: str) -> list[dict[str, Any]]:
        """Page blocks (``GET /v1/blocks/{id}/children``) — the body text."""
        page_id = self._clean_id(page_id, "page_id")
        data = self._api("GET", f"/blocks/{page_id}/children")
        results = data.get("results", []) if isinstance(data, dict) else []
        return results if isinstance(results, list) else []

    def query_database(
        self,
        database_id: str,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sorts: list[dict[str, Any]] | None = None,
        page_size: int = 50,
        start_cursor: str = "",
    ) -> dict[str, Any]:
        """Query a database (``POST /v1/databases/{id}/query``).

        ``filter``/``sorts`` are raw Notion query objects, e.g.
        ``{"property": "Status", "status": {"equals": "Done"}}``.
        """
        database_id = self._clean_id(database_id, "database_id")
        body: dict[str, Any] = {"page_size": max(1, min(page_size, 100))}
        if filter:
            body["filter"] = filter
        if sorts:
            body["sorts"] = sorts
        if start_cursor:
            body["start_cursor"] = start_cursor
        data = self._api(
            "POST", f"/databases/{database_id}/query", body=body
        )
        return {
            "results": data.get("results", []),
            "has_more": data.get("has_more", False),
            "next_cursor": data.get("next_cursor"),
        }

    def list_databases(
        self, *, query: str = "", page_size: int = 50
    ) -> list[dict[str, Any]]:
        """Databases the integration can see (``POST /v1/search``).

        ``query`` narrows by title; empty lists everything shared.
        """
        body: dict[str, Any] = {
            "filter": {"property": "object", "value": "database"},
            "page_size": max(1, min(page_size, 100)),
        }
        if query:
            body["query"] = query
        data = self._api("POST", "/search", body=body)
        results = data.get("results", []) if isinstance(data, dict) else []
        return results if isinstance(results, list) else []

    # ── writes (confirmation-gated) ──────────────────────────────

    def create_page(
        self,
        *,
        database_id: str = "",
        parent_page_id: str = "",
        properties: dict[str, Any] | None = None,
        children: list[dict[str, Any]] | None = None,
        title: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Create a page (``POST /v1/pages``).

        Parent is either a database (``database_id`` + ``properties``
        matching its schema) or a page (``parent_page_id`` + ``title``
        and optional ``children`` blocks). Never runs on implied consent:
        pass ``confirmed=True`` only after the owner approved the exact
        content — or pass ``db`` to park it on a human checkpoint.
        """
        database_id = self._clean_id(database_id, "database_id",
                                     allow_empty=True)
        parent_page_id = self._clean_id(parent_page_id, "parent_page_id",
                                        allow_empty=True)
        if bool(database_id) == bool(parent_page_id):
            raise ConnectorError(
                "pass exactly one of database_id / parent_page_id"
            )
        parent: dict[str, Any] = (
            {"database_id": database_id}
            if database_id else {"page_id": parent_page_id}
        )
        props = dict(properties or {})
        if title:
            # Convenience for page parents: Notion expects
            # properties.title.title = [{text: {content}}].
            props.setdefault(
                "title",
                {"title": [{"text": {"content": title}}]},
            )
        payload: dict[str, Any] = {
            "parent": parent,
            "properties": props,
            "children": list(children or []),
        }
        label = (
            f"database {database_id}" if database_id
            else f"page {parent_page_id}"
        )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="create_page",
            title=f"Create Notion page under {label}",
            instructions="\n".join([
                "Devon wants to create this Notion page.",
                f"Parent: {label}",
                f"Title: {title}" if title else "",
                f"Properties: {json.dumps(props)[:400]}" if props else "",
                f"Content blocks: {len(payload['children'])}",
            ]),
            resume_state={"payload": payload},
        )
        return self._create_now(payload)

    def _create_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "parent": payload["parent"],
            "properties": payload.get("properties", {}),
        }
        if payload.get("children"):
            body["children"] = payload["children"]
        data = self._api("POST", "/pages", body=body)
        result = data if isinstance(data, dict) else {}
        _log.info("notion page created: %s", result.get("id", "?"))
        return result

    def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Update page properties (``PATCH /v1/pages/{id}``).

        ``properties`` is the raw Notion properties object, e.g.
        ``{"Status": {"status": {"name": "Done"}}}``. Confirmation-gated
        like creation.
        """
        page_id = self._clean_id(page_id, "page_id")
        if not properties:
            raise ConnectorError("properties is required for update_page")
        payload = {"page_id": page_id, "properties": dict(properties)}
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="update_page",
            title=f"Update Notion page {page_id}",
            instructions="\n".join([
                "Devon wants to update this Notion page's properties.",
                f"Page: {page_id}",
                f"Properties: {json.dumps(properties)[:400]}",
            ]),
            resume_state={"payload": payload},
        )
        return self._update_now(payload)

    def _update_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._api(
            "PATCH",
            f"/pages/{payload['page_id']}",
            body={"properties": payload["properties"]},
        )
        result = data if isinstance(data, dict) else {}
        _log.info("notion page updated: %s", payload["page_id"])
        return result

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_id(value: str, name: str, *, allow_empty: bool = False) -> str:
        cleaned = (value or "").strip().replace("-", "")
        if not cleaned and not allow_empty:
            raise ConnectorError(f"{name} is required")
        return cleaned

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "notion is not connected — run "
                "`nm connectors connect --name notion` first"
            )
        return cred

    def _headers(self, token: str = "") -> dict[str, str]:
        secret = token or self._require_credential().password
        return {
            "Authorization": f"Bearer {secret}",
            "Notion-Version": NOTION_VERSION,
        }

    def _api(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        token: str = "",
    ) -> dict[str, Any]:
        """One Notion API call. Failures become NotionError."""
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(url, headers=self._headers(token))
            elif method == "POST":
                resp = self.http.post_json(
                    url, body or {}, headers=self._headers(token)
                )
            elif method == "PATCH":
                resp = self.http.request(
                    "PATCH", url,
                    data=json.dumps(body or {}).encode("utf-8"),
                    headers={
                        **self._headers(token),
                        "Content-Type": "application/json",
                    },
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise NotionError(f"notion request failed: {exc}") from exc
        if resp.status == 401:
            raise NotionError(
                "notion rejected the integration token (401): it is "
                "invalid or revoked — rotate it at "
                "notion.so/my-integrations and reconnect",
                status_code=401,
            )
        if resp.status == 403:
            raise NotionError(
                "notion refused (403): the integration lacks the needed "
                "capability — check its capabilities at "
                "notion.so/my-integrations",
                status_code=403,
            )
        if resp.status == 404:
            raise NotionError(
                f"notion {method} {path} not found (404): the page/database "
                "does not exist, or the integration was never shared into "
                "it (⋯ → Connections)",
                status_code=404,
            )
        if resp.status == 429:
            raise NotionError(
                "notion rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            code, msg = self._error_detail(resp)
            raise NotionError(
                f"notion {method} {path} failed ({resp.status}): {msg}",
                status_code=resp.status,
                error_code=code,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise NotionError(
                f"notion {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, str]:
        try:
            body = resp.json()
            if isinstance(body, dict):
                return (
                    str(body.get("code", "")),
                    str(body.get("message", body))[:200],
                )
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        return "", resp.text[:200]
