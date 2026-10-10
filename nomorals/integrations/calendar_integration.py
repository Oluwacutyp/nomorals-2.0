"""Calendar integration with multiple backends.

Supports:
1. Google Calendar API (OAuth)
2. CalDAV (works with iCloud, Nextcloud, etc.)
3. Browser automation (fallback)

Usage:
    calendar = CalendarIntegration(account_manager, session_manager)
    
    # Add an event
    event_id = await calendar.add_event(
        title="Meeting with Alice",
        start="2026-09-30T14:00:00",
        end="2026-09-30T15:00:00",
        account="bot@gmail.com",
        description="Discuss project",
        location="Zoom"
    )
    
    # List upcoming events
    events = await calendar.list_events(
        account="bot@gmail.com",
        days_ahead=7
    )
    
    # Get event details
    event = await calendar.get_event(event_id, account="bot@gmail.com")
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["CalendarIntegration", "CalendarEvent", "format_agenda",
           "format_event"]

_log = get_logger(__name__)

#: sqlite cursors for incremental sync (syncToken per account+calendar).
_SYNC_DB = os.path.expanduser("~/.nomorals/calendar/sync.db")
_sync_lock = threading.Lock()


def _sync_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_SYNC_DB), exist_ok=True)
    db = sqlite3.connect(_SYNC_DB, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS sync_cursors ("
               " account TEXT NOT NULL, calendar_id TEXT NOT NULL,"
               " sync_token TEXT NOT NULL,"
               " updated REAL NOT NULL,"
               " PRIMARY KEY (account, calendar_id))")
    db.commit()
    return db


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    
    event_id: str
    title: str
    start: str  # ISO 8601 datetime
    end: str  # ISO 8601 datetime
    description: str = ""
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    reminders: list[dict[str, Any]] = field(default_factory=list)
    calendar_id: str = "primary"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "location": self.location,
            "attendees": self.attendees,
            "reminders": self.reminders,
            "calendar_id": self.calendar_id,
        }


def format_event(event: CalendarEvent, *, show_date: bool = False) -> str:
    """One-line agenda rendering: ``⏰ 14:00–15:00  Meeting with Alice 📍 Zoom``."""
    start = event.start or ""
    end = event.end or ""
    clock = f"{start[11:16]}–{end[11:16]}" if len(start) >= 16 else start
    date = f"{start[:10]} " if show_date and len(start) >= 10 else ""
    loc = f" 📍 {event.location}" if event.location else ""
    people = f" 👥 {len(event.attendees)}" if event.attendees else ""
    recur = " 🔁" if event.metadata.get("recurring") else ""
    return f"⏰ {date}{clock}  {event.title}{loc}{people}{recur}"


def format_agenda(events: list[CalendarEvent], *, title: str = "📅 agenda") -> str:
    """God-tier day rendering: grouped by date, relative + absolute times."""
    if not events:
        return f"{title}\n_nothing scheduled — clear skies._"
    by_day: dict[str, list[CalendarEvent]] = {}
    for ev in events:
        by_day.setdefault((ev.start or "")[:10] or "undated", []).append(ev)
    lines = [title]
    today = time.strftime("%Y-%m-%d")
    for day in sorted(by_day):
        label = day
        if day == today:
            label = f"{day} (today)"
        lines.append(f"\n**{label}**")
        for ev in sorted(by_day[day], key=lambda e: e.start):
            lines.append("  " + format_event(ev))
    return "\n".join(lines)


class CalendarIntegration:
    """Multi-backend calendar integration."""
    
    def __init__(
        self,
        account_manager: AccountManager,
        session_manager: SessionManager,
    ) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        _log.info("Calendar integration initialized")
    
    async def add_event(
        self,
        title: str,
        start: str,
        end: str,
        account: str,
        *,
        description: str = "",
        location: str = "",
        attendees: list[str] | None = None,
        reminders: list[dict[str, Any]] | None = None,
        calendar_id: str = "primary",
        recurrence: str = "",
        timezone: str = "",
        send_updates: str = "none",
    ) -> str:
        """Add a calendar event.

        Args:
            title: Event title
            start: Start time (ISO 8601)
            end: End time (ISO 8601)
            account: Calendar account
            description: Event description
            location: Event location
            attendees: List of attendee emails
            reminders: List of reminder configs
            calendar_id: Calendar ID (default: primary)
            recurrence: RRULE string, e.g. ``"RRULE:FREQ=DAILY;COUNT=10"``
            timezone: IANA timezone for start/end (default: account's)
            send_updates: "none" (default — agent writes don't spam
                attendees), "all", or "externalOnly"

        Returns:
            Event ID
        """
        backend = self._detect_backend(account)

        if backend == "google_api":
            return await self._add_google_api(
                title, start, end, account, description, location,
                attendees, reminders, calendar_id,
                recurrence=recurrence, timezone=timezone,
                send_updates=send_updates,
            )
        else:
            raise ValueError(f"No calendar backend available for {account}")

    async def search_events(
        self,
        query: str,
        account: str,
        *,
        days_ahead: int = 30,
        calendar_id: str = "primary",
    ) -> list[CalendarEvent]:
        """Free-text search across events (Google ``q=`` search)."""
        backend = self._detect_backend(account)
        if backend != "google_api":
            raise ValueError(f"No calendar backend available for {account}")
        from datetime import datetime, timedelta
        now = datetime.now().astimezone()
        params = {
            "q": query,
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=days_ahead)).isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        url = ("https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events")
        result = self._request(account, "GET", url, params=params) or {}
        return [self._to_event(item, calendar_id)
                for item in result.get("items", [])]

    async def list_calendars(self, account: str) -> list[dict[str, Any]]:
        """List the account's calendars (id, summary, primary flag)."""
        backend = self._detect_backend(account)
        if backend != "google_api":
            raise ValueError(f"No calendar backend available for {account}")
        url = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
        result = self._request(account, "GET", url) or {}
        return [{"id": c.get("id"), "summary": c.get("summary"),
                 "primary": bool(c.get("primary")),
                 "timezone": c.get("timeZone")}
                for c in result.get("items", [])]

    async def find_free_time(
        self,
        account: str,
        duration_min: int,
        *,
        days_ahead: int = 7,
        work_start: str = "09:00",
        work_end: str = "18:00",
        calendar_id: str = "primary",
    ) -> list[dict[str, str]]:
        """Free slots of ``duration_min`` minutes via the freebusy API."""
        backend = self._detect_backend(account)
        if backend != "google_api":
            raise ValueError(f"No calendar backend available for {account}")
        from datetime import datetime, timedelta
        now = datetime.now().astimezone()
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        busy: list[tuple[datetime, datetime]] = []
        url = "https://www.googleapis.com/calendar/v3/freeBusy"
        for d in range(days_ahead):
            ws = day + timedelta(days=d)
            we = day + timedelta(days=d + 1)
            payload = {"timeMin": ws.isoformat(), "timeMax": we.isoformat(),
                       "items": [{"id": calendar_id}]}
            result = self._request(account, "POST", url, payload) or {}
            for b in result.get("calendars", {}).get(
                    calendar_id, {}).get("busy", []):
                busy.append((datetime.fromisoformat(b["start"]),
                             datetime.fromisoformat(b["end"])))
        slots: list[dict[str, str]] = []
        for d in range(days_ahead):
            day_start = (day + timedelta(days=d)).replace(
                hour=int(work_start[:2]), minute=int(work_start[3:]))
            day_end = (day + timedelta(days=d)).replace(
                hour=int(work_end[:2]), minute=int(work_end[3:]))
            cursor = max(day_start, now + timedelta(minutes=15))
            day_busy = sorted(b for b in busy if b[0] < day_end
                              and b[1] > day_start)
            for bstart, bend in day_busy + [(day_end, day_end)]:
                while cursor + timedelta(minutes=duration_min) <= min(
                        bstart, day_end):
                    slots.append({"start": cursor.isoformat(),
                                  "end": (cursor + timedelta(
                                      minutes=duration_min)).isoformat()})
                    cursor += timedelta(minutes=duration_min)
                cursor = max(cursor, bend)
        return slots

    async def sync_changes(
        self,
        account: str,
        *,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """Incremental sync via syncToken — only what changed since last
        call. Persists the cursor in sqlite. Handles 410 Gone (token
        expired → transparent full re-sync)."""
        backend = self._detect_backend(account)
        if backend != "google_api":
            raise ValueError(f"No calendar backend available for {account}")
        with _sync_lock:
            db = _sync_db()
            row = db.execute(
                "SELECT sync_token FROM sync_cursors WHERE account = ?"
                " AND calendar_id = ?", (account, calendar_id)).fetchone()
            token = row[0] if row else None
        url = ("https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events")
        params: dict[str, str] = {"singleEvents": "true",
                                  "orderBy": "startTime"}
        if token:
            params["syncToken"] = token
        try:
            result = self._request(account, "GET", url, params=params) or {}
        except urllib.error.HTTPError as exc:
            if exc.code == 410:  # sync token expired — full re-sync
                _log.info("calendar sync token expired for %s — re-syncing",
                          account)
                params.pop("syncToken", None)
                result = self._request(account, "GET", url,
                                       params=params) or {}
            else:
                raise
        changed = [self._to_event(i, calendar_id)
                   for i in result.get("items", [])
                   if i.get("status") != "cancelled"]
        deleted = [i.get("id") for i in result.get("items", [])
                   if i.get("status") == "cancelled"]
        new_token = result.get("nextSyncToken")
        if new_token:
            with _sync_lock:
                db.execute(
                    "INSERT INTO sync_cursors (account, calendar_id,"
                    " sync_token, updated) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT (account, calendar_id) DO UPDATE SET"
                    " sync_token = excluded.sync_token,"
                    " updated = excluded.updated",
                    (account, calendar_id, new_token, time.time()))
                db.commit()
            db.close()
        return {"changed": changed, "deleted": deleted,
                "full_sync": token is None}

    @staticmethod
    def _to_event(item: dict[str, Any], calendar_id: str) -> CalendarEvent:
        return CalendarEvent(
            event_id=item["id"],
            title=item.get("summary", ""),
            start=item["start"].get("dateTime", item["start"].get("date", "")),
            end=item["end"].get("dateTime", item["end"].get("date", "")),
            description=item.get("description", ""),
            location=item.get("location", ""),
            attendees=[a["email"] for a in item.get("attendees", [])],
            calendar_id=calendar_id,
            metadata={"recurring": bool(item.get("recurrence")),
                      "status": item.get("status", "")},
        )

    async def list_events(
        self,
        account: str,
        *,
        days_ahead: int = 7,
        calendar_id: str = "primary",
    ) -> list[CalendarEvent]:
        """List upcoming events.
        
        Args:
            account: Calendar account
            days_ahead: How many days ahead to fetch
            calendar_id: Calendar ID
            
        Returns:
            List of CalendarEvent objects
        """
        backend = self._detect_backend(account)
        
        if backend == "google_api":
            return await self._list_google_api(account, days_ahead, calendar_id)
        else:
            raise ValueError(f"No calendar backend available for {account}")
    
    async def get_event(
        self,
        event_id: str,
        account: str,
        *,
        calendar_id: str = "primary",
    ) -> Optional[CalendarEvent]:
        """Get event details.
        
        Args:
            event_id: Event ID
            account: Calendar account
            calendar_id: Calendar ID
            
        Returns:
            CalendarEvent object or None
        """
        backend = self._detect_backend(account)
        
        if backend == "google_api":
            return await self._get_google_api(event_id, account, calendar_id)
        else:
            raise ValueError(f"No calendar backend available for {account}")
    
    async def update_event(
        self,
        event_id: str,
        account: str,
        *,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
    ) -> None:
        """Update an existing event.
        
        Args:
            event_id: Event ID
            account: Calendar account
            title: New title (or None to keep)
            start: New start time (or None to keep)
            end: New end time (or None to keep)
            description: New description (or None to keep)
            location: New location (or None to keep)
            calendar_id: Calendar ID
        """
        backend = self._detect_backend(account)
        
        if backend == "google_api":
            await self._update_google_api(
                event_id, account, title, start, end, description,
                location, calendar_id
            )
        else:
            raise ValueError(f"No calendar backend available for {account}")
    
    async def delete_event(
        self,
        event_id: str,
        account: str,
        *,
        calendar_id: str = "primary",
    ) -> None:
        """Delete an event.
        
        Args:
            event_id: Event ID
            account: Calendar account
            calendar_id: Calendar ID
        """
        backend = self._detect_backend(account)
        
        if backend == "google_api":
            await self._delete_google_api(event_id, account, calendar_id)
        else:
            raise ValueError(f"No calendar backend available for {account}")
    
    def _detect_backend(self, account: str) -> str:
        """Detect the best backend for a calendar account."""
        if account.endswith("@gmail.com"):
            try:
                cred = self.account_manager.get_credential("google_calendar_oauth", account)
                if cred.credential_type == "oauth_token":
                    return "google_api"
            except Exception as e:
                _log.debug("calendar backend probe failed for %s: %s", account, e)
        
        return "none"

    # ── token lifecycle ──────────────────────────────────────────────

    def _access_token(self, account: str) -> str:
        """Fresh access token, refreshing transparently when needed."""
        cred = self.account_manager.get_credential(
            "google_calendar_oauth", account)
        try:
            token_data = json.loads(cred.password)
        except (TypeError, ValueError):
            token_data = {"access_token": cred.password}
        access = token_data.get("access_token", "")
        # Probe the token; refresh on 401.
        if access and not self._token_ok(access):
            _log.info("calendar token expired for %s — refreshing", account)
            access = self._refresh_token(account, token_data)
        return access

    @staticmethod
    def _token_ok(access_token: str) -> bool:
        try:
            req = urllib.request.Request(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                headers={"Authorization": f"Bearer {access_token}"})
            with urllib.request.urlopen(req, timeout=8):
                return True
        except Exception:  # noqa: BLE001 - any failure → try refresh
            return False

    def _refresh_token(self, account: str, token_data: dict) -> str:
        """OAuth refresh_token grant; persists the new token to the vault."""
        refresh = token_data.get("refresh_token", "")
        client_id = token_data.get("client_id", "") or os.environ.get(
            "GOOGLE_CLIENT_ID", "")
        client_secret = token_data.get("client_secret", "") or os.environ.get(
            "GOOGLE_CLIENT_SECRET", "")
        if not refresh or not client_id or not client_secret:
            raise ValueError(
                f"cannot refresh calendar token for {account}: no "
                "refresh_token/client credentials stored")
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode()
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                fresh = json.loads(resp.read().decode())
        except Exception as exc:
            raise ValueError(
                f"calendar token refresh failed for {account}: {exc}") from exc
        token_data["access_token"] = fresh["access_token"]
        token_data["expires_at"] = time.time() + int(
            fresh.get("expires_in", 3600))
        try:
            self.account_manager.store_credential(
                "google_calendar_oauth", account, json.dumps(token_data),
                credential_type="oauth_token")
        except Exception:  # noqa: BLE001 - vault write is best-effort
            _log.debug("could not persist refreshed calendar token",
                       exc_info=True)
        return token_data["access_token"]

    def _request(self, account: str, method: str, url: str,
                 payload: dict | None = None,
                 params: dict | None = None) -> Any:
        """Authenticated Google API call with transparent token refresh."""
        token = self._access_token(account)
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            if exc.code == 401:  # token died mid-flight — one retry
                token_data = json.loads(
                    self.account_manager.get_credential(
                        "google_calendar_oauth", account).password)
                token = self._refresh_token(account, token_data)
                headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(url, data=data, headers=headers,
                                             method=method)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read().decode()
                    return json.loads(raw) if raw else None
            raise
    
    # ── Google Calendar API Backend ────────────────────────────────────────
    # All calls go through _request(): transparent OAuth refresh + 401 retry.

    async def _add_google_api(
        self,
        title: str,
        start: str,
        end: str,
        account: str,
        description: str,
        location: str,
        attendees: list[str] | None,
        reminders: list[dict[str, Any]] | None,
        calendar_id: str,
        *,
        recurrence: str = "",
        timezone: str = "",
        send_updates: str = "none",
    ) -> str:
        """Add event via Google Calendar API."""
        tz = timezone or self._account_timezone(account)
        event: dict[str, Any] = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start, "timeZone": tz},
            "end": {"dateTime": end, "timeZone": tz},
        }
        if attendees:
            event["attendees"] = [{"email": e} for e in attendees]
        if reminders:
            event["reminders"] = {"useDefault": False, "overrides": reminders}
        if recurrence:
            event["recurrence"] = [recurrence] if isinstance(
                recurrence, str) else list(recurrence)
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events")
        try:
            result = self._request(account, "POST", url, event,
                                   params={"sendUpdates": send_updates})
            event_id = result["id"]
            _log.info("Added calendar event: %s", title)
            return event_id
        except Exception as e:
            _log.error("Failed to add event: %s", e)
            raise

    def _account_timezone(self, account: str) -> str:
        """Account timezone: primary calendar's timeZone, else local."""
        try:
            url = ("https://www.googleapis.com/calendar/v3/users/me/"
                   "calendarList")
            result = self._request(account, "GET", url) or {}
            for cal in result.get("items", []):
                if cal.get("primary"):
                    return cal.get("timeZone") or "UTC"
        except Exception:  # noqa: BLE001 - best effort
            _log.debug("calendar timezone lookup failed", exc_info=True)
        try:
            return time.tzname[0] or "UTC"
        except Exception:  # noqa: BLE001
            return "UTC"

    async def _list_google_api(
        self,
        account: str,
        days_ahead: int,
        calendar_id: str,
    ) -> list[CalendarEvent]:
        """List events via Google Calendar API."""
        from datetime import datetime, timedelta
        now = datetime.now().astimezone()
        params = {
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=days_ahead)).isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events")
        try:
            result = self._request(account, "GET", url, params=params) or {}
            return [self._to_event(item, calendar_id)
                    for item in result.get("items", [])]
        except Exception as e:
            _log.error("Failed to list events: %s", e)
            return []

    async def _get_google_api(
        self,
        event_id: str,
        account: str,
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Get event via Google Calendar API."""
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events/{event_id}")
        try:
            item = self._request(account, "GET", url)
            return self._to_event(item, calendar_id) if item else None
        except Exception as e:
            _log.error("Failed to get event: %s", e)
            return None

    async def _update_google_api(
        self,
        event_id: str,
        account: str,
        title: str | None,
        start: str | None,
        end: str | None,
        description: str | None,
        location: str | None,
        calendar_id: str,
        *,
        timezone: str = "",
        send_updates: str = "none",
    ) -> None:
        """Update event via Google Calendar API."""
        tz = timezone or self._account_timezone(account)
        update: dict[str, Any] = {}
        if title is not None:
            update["summary"] = title
        if description is not None:
            update["description"] = description
        if location is not None:
            update["location"] = location
        if start is not None:
            update["start"] = {"dateTime": start, "timeZone": tz}
        if end is not None:
            update["end"] = {"dateTime": end, "timeZone": tz}
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events/{event_id}")
        try:
            self._request(account, "PATCH", url, update,
                          params={"sendUpdates": send_updates})
            _log.info("Updated calendar event: %s", event_id)
        except Exception as e:
            _log.error("Failed to update event: %s", e)
            raise

    async def _delete_google_api(
        self,
        event_id: str,
        account: str,
        calendar_id: str,
        *,
        send_updates: str = "none",
    ) -> None:
        """Delete event via Google Calendar API."""
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{calendar_id}/events/{event_id}")
        try:
            self._request(account, "DELETE", url,
                          params={"sendUpdates": send_updates})
            _log.info("Deleted calendar event: %s", event_id)
        except Exception as e:
            _log.error("Failed to delete event: %s", e)
            raise
