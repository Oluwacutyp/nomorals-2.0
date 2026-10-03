"""Google Calendar connector — the owner's calendar over Calendar API v3.

Docs: https://developers.google.com/calendar/api/v3/reference

Auth: OAuth 2.0 (``AuthMethod.OAUTH2``) via the shared Google
authorization-code flow in ``_google_oauth`` — the owner grants the
``calendar`` scope in their own browser and pastes back the code. The
refresh token is vaulted; access tokens auto-refresh.

Writes are consequential: ``create_event`` and ``delete_event`` never run
on implied consent — they require ``confirmed=True`` (owner approved the
exact event) or a human checkpoint when ``db`` is given. A create/update
that adds or changes guests notifies them (``sendUpdates=all``) unless
the caller explicitly passes ``notify=False``. Reads never need
confirmation.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from ._google_oauth import GoogleOAuth
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .checkpoints import (
    CheckpointKind,
    CheckpointState,
    HumanCheckpointPending,
)
from .registry import register_connector

__all__ = ["GCalendarConnector", "GCalendarError"]

_log = get_logger(__name__)

API_BASE = "https://www.googleapis.com/calendar/v3"

_SCOPE_CALENDAR = "https://www.googleapis.com/auth/calendar"


class GCalendarError(ConnectorError):
    """A Google Calendar API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class GCalendarConnector(Connector, GoogleOAuth):
    """Devon's Google Calendar adapter: list, create, update, delete."""

    id = "gcalendar"
    name = "Google Calendar"
    description = (
        "Google Calendar API: list events, read one event, and create, "
        "update, or delete events. OAuth 2.0 (calendar scope); writes "
        "require explicit owner confirmation."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def _google_default_scopes(self) -> list[str]:
        return [_SCOPE_CALENDAR]

    def connect(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        code: str | None = None,
        scopes: list[str] | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Complete the Google OAuth flow and vault the tokens.

        Without ``code``: prints the grant guide + authorization URL. With
        ``db`` the flow pauses at a human checkpoint (the owner grants
        access in their browser, then resolves the checkpoint with the
        code); without it, the owner hands the code back and calls
        ``connect(..., code=<code>)`` again.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "gcalendar is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        cid = self._google_client_id(client_id)
        wanted = list(scopes or self._google_default_scopes())
        if code:
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            account = self._primary_summary(tokens.get("access_token", ""))
            self._store_google_tokens(
                account or "gcalendar", cid, tokens,
                account=account, scopes=wanted,
            )
            return ConnectResult(
                ok=True,
                account=account or "gcalendar",
                scopes=wanted,
                message=(
                    f"connected to Google Calendar "
                    f"({account or 'the granted account'}). Tokens are in "
                    "the encrypted vault."
                ),
            )
        guide = self._google_connect_guide(cid, wanted)
        print(guide)
        if db is None:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "grant access in your browser, then call "
                    "connect(..., code=<code>) with the code Google shows"
                ),
            )
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Grant Google Calendar access",
                guide
                + "\n\nWhen Google shows the authorization code, resolve "
                "this checkpoint with the code, e.g. note "
                "'code=<authorization code>'.",
                db=db,
                context=context,
                resume_state={
                    "stage": "oauth_code",
                    "client_id": cid,
                    "scopes": wanted,
                },
            )
        except HumanCheckpointPending as pending:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "grant access in your browser, then resolve "
                    f"checkpoint {pending.checkpoint.id} with the code"
                ),
            )
        raise ConnectorError(
            "access granted interactively but no code was captured — call "
            "connect(..., code=<code>) with the code Google showed"
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name gcalendar`",
            )
        try:
            account = self._primary_summary(self._google_access_token())
        except GCalendarError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "")),
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh code",
            )
        return ConnectorStatus(
            connected=True,
            account=account,
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="oauth tokens valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._primary_summary(self._google_access_token())
            return True
        except ConnectorError:
            return False

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Continue after a human checkpoint resolved."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — finish the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "oauth_code":
            code = self._code_from_note(checkpoint.result_note or "")
            if not code:
                raise ConnectorError(
                    "the resolved checkpoint has no authorization code — "
                    "resolve it again with note 'code=<authorization code>'"
                )
            cid = str((checkpoint.resume_state or {}).get("client_id", ""))
            scopes = list(
                (checkpoint.resume_state or {}).get(
                    "scopes", self._google_default_scopes()
                )
            )
            secret = self._google_client_secret(client_secret)
            tokens = self._google_exchange_code(cid, secret, code)
            account = self._primary_summary(tokens["access_token"])
            self._store_google_tokens(
                account or "gcalendar", cid, tokens,
                account=account, scopes=scopes,
            )
            return {"connected": True, "account": account}
        if stage in ("create_event", "update_event"):
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("summary"):
                raise ConnectorError(
                    "the resolved checkpoint has no event payload — "
                    "it cannot write the event"
                )
            if stage == "create_event":
                return self._insert_now(payload)
            return self._patch_now(payload)
        if stage == "delete_event":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("event_id"):
                raise ConnectorError(
                    "the resolved checkpoint has no delete payload"
                )
            return self._delete_now(payload)
        raise ConnectorError(
            f"gcalendar cannot resume checkpoint stage {stage!r}"
        )

    # ── reads ────────────────────────────────────────────────────

    def list_calendars(self) -> list[dict[str, Any]]:
        """The owner's calendar list (``calendarList.list``)."""
        data = self._api("GET", "/users/me/calendarList")
        items = data.get("items", [])
        return items if isinstance(items, list) else []

    def list_events(
        self,
        *,
        calendar_id: str = "primary",
        time_min: str = "",
        time_max: str = "",
        query: str = "",
        max_results: int = 50,
        page_token: str = "",
    ) -> dict[str, Any]:
        """Events on a calendar (``events.list``).

        ``time_min``/``time_max`` are RFC3339 bounds (e.g.
        ``2026-10-04T00:00:00+01:00``); ``query`` is free-text search.
        Returns one-off and recurring instances in start order.
        """
        params: dict[str, Any] = {
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": max(1, min(max_results, 2500)),
        }
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        data = self._api(
            "GET",
            f"/calendars/{self._clean_id(calendar_id)}/events",
            params=params,
        )
        return {
            "events": data.get("items", []),
            "next_page_token": data.get("nextPageToken", ""),
            "time_zone": data.get("timeZone", ""),
        }

    def get_event(
        self, event_id: str, *, calendar_id: str = "primary"
    ) -> dict[str, Any]:
        """One event's full detail (``events.get``)."""
        event_id = (event_id or "").strip()
        if not event_id:
            raise ConnectorError("event_id is required")
        return self._api(
            "GET",
            f"/calendars/{self._clean_id(calendar_id)}/events/{event_id}",
        )

    # ── writes (confirmation-gated) ──────────────────────────────

    def create_event(
        self,
        summary: str,
        start: dict[str, str],
        end: dict[str, str],
        *,
        calendar_id: str = "primary",
        description: str = "",
        location: str = "",
        attendees: list[str] | None = None,
        notify: bool = True,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Create an event (``events.insert``).

        ``start``/``end`` are ``{"dateTime": "2026-10-04T14:00:00+01:00"}``
        (timed) or ``{"date": "2026-10-04"}`` (all-day; end is exclusive).
        Attendee emails notify guests unless ``notify=False``. Never runs
        on implied consent: pass ``confirmed=True`` only after the owner
        approved the exact event — or pass ``db`` to park it on a human
        checkpoint.
        """
        payload = self._event_payload(
            summary, start, end, calendar_id,
            description=description, location=location,
            attendees=attendees, notify=notify,
        )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="create_event",
            title=f"Create calendar event: {summary}",
            instructions="\n".join([
                "Devon wants to create this calendar event.",
                f"Summary: {summary}",
                f"Start: {start}",
                f"End: {end}",
                f"Attendees: {', '.join(attendees or []) or 'none'}",
                f"Guests will be notified: {notify and bool(attendees)}",
            ]),
            resume_state={"payload": payload},
        )
        return self._insert_now(payload)

    def _insert_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = {"sendUpdates": "all"} if payload.pop("notify", True) else {}
        data = self._api(
            "POST",
            f"/calendars/{payload['calendar_id']}/events",
            payload=payload["body"],
            params=params or None,
        )
        _log.info(
            "gcalendar event created: %s (%s)",
            data.get("id", "?"), payload["body"].get("summary", "?"),
        )
        return data

    def update_event(
        self,
        event_id: str,
        updates: dict[str, Any],
        *,
        calendar_id: str = "primary",
        notify: bool = True,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Patch an event (``events.patch``) — only ``updates`` change.

        ``updates`` are raw event fields (``summary``, ``start``,
        ``location``, ...). Changing guests notifies them unless
        ``notify=False``. Confirmation-gated like creation.
        """
        event_id = (event_id or "").strip()
        if not event_id:
            raise ConnectorError("event_id is required")
        if not updates:
            raise ConnectorError("updates is required for update_event")
        payload = {
            "event_id": event_id,
            "calendar_id": self._clean_id(calendar_id),
            "summary": updates.get("summary", ""),
            "body": dict(updates),
            "notify": bool(notify),
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="update_event",
            title=f"Update calendar event {event_id}",
            instructions="\n".join([
                "Devon wants to update this calendar event.",
                f"Event: {event_id}",
                f"Changes: {updates}",
                f"Guests will be notified: {notify}",
            ]),
            resume_state={"payload": payload},
        )
        return self._patch_now(payload)

    def _patch_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = {"sendUpdates": "all"} if payload.get("notify") else {}
        data = self._api(
            "PATCH",
            f"/calendars/{payload['calendar_id']}"
            f"/events/{payload['event_id']}",
            payload=payload["body"],
            params=params or None,
        )
        _log.info("gcalendar event updated: %s", payload["event_id"])
        return data

    def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str = "primary",
        notify: bool = True,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Delete an event (``events.delete``).

        Guests get a cancellation notice unless ``notify=False``.
        Confirmation-gated like creation.
        """
        event_id = (event_id or "").strip()
        if not event_id:
            raise ConnectorError("event_id is required")
        payload = {
            "event_id": event_id,
            "calendar_id": self._clean_id(calendar_id),
            "notify": bool(notify),
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="delete_event",
            title=f"Delete calendar event {event_id}",
            instructions="\n".join([
                "Devon wants to delete this calendar event.",
                f"Event: {event_id}",
                f"Calendar: {payload['calendar_id']}",
                f"Guests will be notified: {notify}",
            ]),
            resume_state={"payload": payload},
        )
        return self._delete_now(payload)

    def _delete_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = {"sendUpdates": "all"} if payload.get("notify") else {}
        self._api(
            "DELETE",
            f"/calendars/{payload['calendar_id']}"
            f"/events/{payload['event_id']}",
            params=params or None,
        )
        _log.info("gcalendar event deleted: %s", payload["event_id"])
        return {"deleted": True, "event_id": payload["event_id"]}

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "gcalendar is not connected — run "
                "`nm connectors connect --name gcalendar` first"
            )
        return cred

    @staticmethod
    def _clean_id(calendar_id: str) -> str:
        cleaned = (calendar_id or "").strip()
        if not cleaned:
            raise ConnectorError("calendar_id is required")
        return cleaned

    def _event_payload(
        self,
        summary: str,
        start: dict[str, str],
        end: dict[str, str],
        calendar_id: str,
        *,
        description: str,
        location: str,
        attendees: list[str] | None,
        notify: bool,
    ) -> dict[str, Any]:
        summary = (summary or "").strip()
        if not summary:
            raise ConnectorError("summary is required for create_event")
        for name, bound in (("start", start), ("end", end)):
            if not isinstance(bound, dict) or not (
                bound.get("dateTime") or bound.get("date")
            ):
                raise ConnectorError(
                    f"{name} must be a dict with dateTime (timed) or date "
                    "(all-day), e.g. {'dateTime': '2026-10-04T14:00:00+01:00'}"
                )
        body: dict[str, Any] = {
            "summary": summary,
            "start": dict(start),
            "end": dict(end),
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        guests = [a.strip() for a in (attendees or []) if a.strip()]
        if guests:
            body["attendees"] = [{"email": email} for email in guests]
        return {
            "calendar_id": self._clean_id(calendar_id),
            "summary": summary,
            "body": body,
            "notify": bool(notify and guests),
        }

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._google_access_token()}"}

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One Calendar API call; errors become GCalendarError."""
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(), params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=self._headers(),
                    params=params,
                )
            elif method == "PATCH":
                resp = self.http.request(
                    "PATCH", url,
                    data=_json_bytes(payload or {}),
                    headers={
                        **self._headers(),
                        "Content-Type": "application/json",
                    },
                    params=params,
                )
            elif method == "DELETE":
                resp = self.http.request(
                    "DELETE", url, headers=self._headers(), params=params
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GCalendarError(f"gcalendar request failed: {exc}") from exc
        if resp.status == 401:
            raise GCalendarError(
                "gcalendar rejected the access token (401) — the grant was "
                "revoked or expired; reconnect with a fresh code",
                status_code=401,
            )
        if resp.status == 403:
            reason = self._api_reason(resp)
            raise GCalendarError(
                f"gcalendar refused (403{', ' + reason if reason else ''}): "
                "the grant lacks this scope — reconnect granting the "
                "calendar scope",
                status_code=403,
                reason=reason,
            )
        if resp.status == 404:
            raise GCalendarError(
                f"gcalendar {method} {path} not found (404): the calendar "
                "or event id is wrong",
                status_code=404,
            )
        if resp.status == 429:
            raise GCalendarError(
                "gcalendar rate limit exceeded (429) — back off and retry",
                status_code=429,
            )
        if not resp.ok:
            raise GCalendarError(
                f"gcalendar {method} {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        if method == "DELETE":
            return {}
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise GCalendarError(
                f"gcalendar {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _api_reason(resp: Any) -> str:
        try:
            body = resp.json()
            errors = (body.get("error") or {}).get("errors", [])
            if errors:
                return str(errors[0].get("reason", ""))
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return ""

    def _primary_summary(self, access_token: str) -> str:
        """The primary calendar's summary — the connection probe."""
        try:
            resp = self.http.get(
                f"{API_BASE}/calendars/primary",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GCalendarError(
                f"gcalendar calendar lookup failed: {exc}"
            ) from exc
        if not resp.ok:
            if resp.status == 401:
                raise GCalendarError(
                    "gcalendar rejected the access token (401) — the grant "
                    "was revoked or expired; reconnect with a fresh code",
                    status_code=401,
                )
            raise GCalendarError(
                f"gcalendar calendar lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return str(resp.json().get("summary", ""))
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise GCalendarError(
                "gcalendar calendar lookup returned invalid JSON"
            ) from exc

    @staticmethod
    def _code_from_note(note: str) -> str:
        """Pull the authorization code out of a checkpoint note."""
        text = (note or "").strip()
        if not text:
            return ""
        import re

        match = re.search(
            r"code\s*(?:=|:|\bis\b)?\s*[\"']?([A-Za-z0-9_.\-/]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        tokens = text.split()
        if len(tokens) == 1 and tokens[0].lower() != "code":
            return tokens[0].strip("\"'")
        return ""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")
