"""Trello connector — boards, lists, and cards over the Trello REST API.

Docs: https://developer.atlassian.com/cloud/trello/rest/

Auth: API key + token (``AuthMethod.API_KEY``) — the key comes from
https://trello.com/app-key, the token from the "Token" link on that
page (generate one with ``read,write`` scope). Both travel as query
parameters (``key`` + ``token``) on every request, per Trello's API
design — they are never logged, and error paths never include the query
string.

Writes are consequential: ``create_card`` and ``move_card`` never run on
implied consent — they require ``confirmed=True`` (owner approved the
exact card/list) or a human checkpoint when ``db`` is given. Reads never
need confirmation.
"""

from __future__ import annotations

import time
import urllib.parse
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

__all__ = ["TrelloConnector", "TrelloError"]

_log = get_logger(__name__)

API_BASE = "https://api.trello.com"
KEY_ENV = "TRELLO_API_KEY"
TOKEN_ENV = "TRELLO_API_TOKEN"


class TrelloError(ConnectorError):
    """A Trello API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, error_code: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@register_connector
class TrelloConnector(Connector):
    """Devon's Trello adapter: boards, lists, cards."""

    id = "trello"
    name = "Trello"
    description = (
        "Trello REST API: who-am-i, boards, board lists, and creating or "
        "moving cards. Key + token auth; card writes require explicit "
        "owner confirmation."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        api_key: str | None = None,
        token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate a key + token pair against /1/members/me and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "trello is already connected — one account per service. "
                "Disconnect first to switch credentials."
            )
        key = (api_key or "").strip() or prompt_secret(
            "Trello API key (trello.com/app-key)", env_var=KEY_ENV
        )
        secret = (token or "").strip() or prompt_secret(
            "Trello API token (the Token link on trello.com/app-key)",
            env_var=TOKEN_ENV,
        )
        if not key or not secret:
            raise ConnectorError(
                "empty API key/token: nothing to connect with"
            )
        me = self._api("GET", "/1/members/me", key=key, token=secret)
        username = str(me.get("username", ""))
        label = f"@{username}" if username else "trello"
        self._store_credential(
            label,
            secret,
            credential_type="api_key",
            scopes=["read", "write"],
            metadata={"api_key": key, "member_id": me.get("id")},
        )
        _log.info("trello connected as %s", label)
        return ConnectResult(
            ok=True,
            account=label,
            scopes=["read", "write"],
            message=(
                f"connected to Trello as {label}. The token is in the "
                "encrypted vault (the key rides with it). Every card "
                "write still needs your explicit confirmation at call time."
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
                       "--name trello`",
            )
        try:
            me = self._api("GET", "/1/members/me")
        except TrelloError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): generate a fresh token at "
                       "trello.com/app-key and reconnect",
            )
        username = me.get("username", cred.username)
        return ConnectorStatus(
            connected=True,
            account=f"@{username}",
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"member {me.get('id', '?')} responding",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/1/members/me")
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
        """Execute a confirmed card write after the owner resolved it."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must approve the card write first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        payload = dict((checkpoint.resume_state or {}).get("payload", {}))
        if stage == "create_card":
            if not payload.get("id_list"):
                raise TrelloError(
                    "the resolved checkpoint has no card payload — "
                    "it cannot create the card"
                )
            return self._create_now(payload)
        if stage == "move_card":
            if not payload.get("card_id"):
                raise TrelloError(
                    "the resolved checkpoint has no move payload"
                )
            return self._move_now(payload)
        raise TrelloError(
            f"trello cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """The token's member (``GET /1/members/me``)."""
        data = self._api("GET", "/1/members/me")
        return data if isinstance(data, dict) else {}

    def list_boards(self) -> list[dict[str, Any]]:
        """Open boards the member can see (``GET /1/members/me/boards``)."""
        data = self._api(
            "GET", "/1/members/me/boards",
            params={"filter": "open", "fields": "id,name,url,closed"},
        )
        return data if isinstance(data, list) else []

    def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        """Lists on a board (``GET /1/boards/{id}/lists``)."""
        board_id = (board_id or "").strip()
        if not board_id:
            raise ConnectorError("board_id is required")
        data = self._api(
            "GET", f"/1/boards/{board_id}/lists",
            params={"cards": "none", "fields": "id,name,pos"},
        )
        return data if isinstance(data, list) else []

    def list_cards(self, list_id: str) -> list[dict[str, Any]]:
        """Cards on a list (``GET /1/lists/{id}/cards``)."""
        list_id = (list_id or "").strip()
        if not list_id:
            raise ConnectorError("list_id is required")
        data = self._api(
            "GET", f"/1/lists/{list_id}/cards",
            params={"fields": "id,name,desc,due,url"},
        )
        return data if isinstance(data, list) else []

    def get_card(self, card_id: str) -> dict[str, Any]:
        """One card (``GET /1/cards/{id}``)."""
        card_id = (card_id or "").strip()
        if not card_id:
            raise ConnectorError("card_id is required")
        data = self._api(
            "GET", f"/1/cards/{card_id}",
            params={"fields": "id,name,desc,due,idList,url"},
        )
        return data if isinstance(data, dict) else {}

    # ── writes (confirmation-gated) ──────────────────────────────

    def create_card(
        self,
        list_id: str,
        name: str,
        *,
        description: str = "",
        due: str = "",
        position: str = "bottom",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Create a card (``POST /1/cards``).

        ``due`` is an ISO date/datetime. Never runs on implied consent:
        pass ``confirmed=True`` only after the owner approved the exact
        card — or pass ``db`` to park it on a human checkpoint.
        """
        list_id = (list_id or "").strip()
        name = (name or "").strip()
        if not list_id:
            raise ConnectorError("list_id is required")
        if not name:
            raise ConnectorError("refusing to create an unnamed card")
        payload: dict[str, Any] = {
            "id_list": list_id,
            "name": name,
            "description": description or "",
            "due": due.strip(),
            "position": position,
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="create_card",
            title=f"Create Trello card: {name}",
            instructions="\n".join([
                "Devon wants to create this Trello card.",
                f"Card: {name}",
                f"List id: {list_id}",
                f"Description: {description[:200]}" if description else "",
                f"Due: {due}" if due else "",
            ]),
            resume_state={"payload": payload},
        )
        return self._create_now(payload)

    def _create_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "idList": payload["id_list"],
            "name": payload["name"],
            "pos": payload.get("position", "bottom"),
        }
        if payload.get("description"):
            params["desc"] = payload["description"]
        if payload.get("due"):
            params["due"] = payload["due"]
        data = self._api("POST", "/1/cards", params=params)
        result = data if isinstance(data, dict) else {}
        _log.info(
            "trello card created: %s (%s)",
            result.get("id", "?"), payload["name"],
        )
        return result

    def move_card(
        self,
        card_id: str,
        list_id: str,
        *,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Move a card to another list (``PUT /1/cards/{id}``).

        Confirmation-gated like creation: moving changes the board's
        visible state, so the owner approves each one.
        """
        card_id = (card_id or "").strip()
        list_id = (list_id or "").strip()
        if not card_id:
            raise ConnectorError("card_id is required")
        if not list_id:
            raise ConnectorError("list_id is required")
        payload = {"card_id": card_id, "id_list": list_id}
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="move_card",
            title=f"Move Trello card {card_id}",
            instructions="\n".join([
                "Devon wants to move this Trello card to another list.",
                f"Card id: {card_id}",
                f"Target list id: {list_id}",
            ]),
            resume_state={"payload": payload},
        )
        return self._move_now(payload)

    def _move_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._api(
            "PUT",
            f"/1/cards/{payload['card_id']}",
            params={"idList": payload["id_list"]},
        )
        result = data if isinstance(data, dict) else {}
        _log.info(
            "trello card moved: %s -> list %s",
            payload["card_id"], payload["id_list"],
        )
        return result

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "trello is not connected — run "
                "`nm connectors connect --name trello` first"
            )
        return cred

    def _credentials(
        self, *, key: str = "", token: str = ""
    ) -> tuple[str, str]:
        if key and token:
            return key, token
        cred = self._require_credential()
        return (
            key or str((cred.metadata or {}).get("api_key", "")),
            token or cred.password,
        )

    def _api(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        key: str = "",
        token: str = "",
    ) -> Any:
        """One Trello call. Failures become TrelloError.

        ``key``/``token`` travel as query parameters per Trello's API
        design; the query string is scrubbed from every error message.
        """
        api_key, api_token = self._credentials(key=key, token=token)
        query = dict(params or {})
        query["key"] = api_key
        query["token"] = api_token
        url = f"{API_BASE}{path}?{urllib.parse.urlencode(query)}"
        safe_url = url.replace(api_token, "<token>").replace(
            api_key, "<key>"
        )
        try:
            if method == "GET":
                resp = self.http.get(url)
            elif method == "POST":
                resp = self.http.request("POST", url)
            elif method == "PUT":
                resp = self.http.request("PUT", url)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise TrelloError(f"trello request failed: {exc}") from exc
        if resp.status == 401:
            raise TrelloError(
                "trello rejected the credentials (401): the key or token "
                "is invalid or revoked — generate fresh ones at "
                "trello.com/app-key and reconnect",
                status_code=401,
            )
        if resp.status == 429:
            raise TrelloError(
                "trello rate limit hit (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            raise TrelloError(
                f"trello {method} {path} failed (HTTP {resp.status}, "
                f"{safe_url}): {resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise TrelloError(
                f"trello {method} {path} returned invalid JSON"
            ) from exc
