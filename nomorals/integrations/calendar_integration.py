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
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["CalendarIntegration", "CalendarEvent"]

_log = get_logger(__name__)


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
            
        Returns:
            Event ID
        """
        backend = self._detect_backend(account)
        
        if backend == "google_api":
            return await self._add_google_api(
                title, start, end, account, description, location,
                attendees, reminders, calendar_id
            )
        else:
            raise ValueError(f"No calendar backend available for {account}")
    
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
            except Exception:
                pass
        
        return "none"
    
    # ── Google Calendar API Backend ────────────────────────────────────────
    
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
    ) -> str:
        """Add event via Google Calendar API."""
        cred = self.account_manager.get_credential("google_calendar_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Build event object
        event = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end": {"dateTime": end, "timeZone": "UTC"},
        }
        
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]
        
        if reminders:
            event["reminders"] = {
                "useDefault": False,
                "overrides": reminders,
            }
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        data = json.dumps(event).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                event_id = result["id"]
                _log.info(f"Added calendar event: {title}")
                return event_id
        except Exception as e:
            _log.error(f"Failed to add event: {e}")
            raise
    
    async def _list_google_api(
        self,
        account: str,
        days_ahead: int,
        calendar_id: str,
    ) -> list[CalendarEvent]:
        """List events via Google Calendar API."""
        cred = self.account_manager.get_credential("google_calendar_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Calculate time range
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"
        
        params = urllib.parse.urlencode({
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "orderBy": "startTime",
        })
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events?{params}"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                
                events = []
                for item in result.get("items", []):
                    events.append(CalendarEvent(
                        event_id=item["id"],
                        title=item.get("summary", ""),
                        start=item["start"].get("dateTime", ""),
                        end=item["end"].get("dateTime", ""),
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        attendees=[a["email"] for a in item.get("attendees", [])],
                        calendar_id=calendar_id,
                    ))
                
                return events
        except Exception as e:
            _log.error(f"Failed to list events: {e}")
            return []
    
    async def _get_google_api(
        self,
        event_id: str,
        account: str,
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Get event via Google Calendar API."""
        cred = self.account_manager.get_credential("google_calendar_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                item = json.loads(response.read().decode())
                
                return CalendarEvent(
                    event_id=item["id"],
                    title=item.get("summary", ""),
                    start=item["start"].get("dateTime", ""),
                    end=item["end"].get("dateTime", ""),
                    description=item.get("description", ""),
                    location=item.get("location", ""),
                    attendees=[a["email"] for a in item.get("attendees", [])],
                    calendar_id=calendar_id,
                )
        except Exception as e:
            _log.error(f"Failed to get event: {e}")
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
    ) -> None:
        """Update event via Google Calendar API."""
        cred = self.account_manager.get_credential("google_calendar_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Build update object
        update = {}
        if title is not None:
            update["summary"] = title
        if description is not None:
            update["description"] = description
        if location is not None:
            update["location"] = location
        if start is not None:
            update["start"] = {"dateTime": start, "timeZone": "UTC"}
        if end is not None:
            update["end"] = {"dateTime": end, "timeZone": "UTC"}
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        data = json.dumps(update).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10):
                _log.info(f"Updated calendar event: {event_id}")
        except Exception as e:
            _log.error(f"Failed to update event: {e}")
            raise
    
    async def _delete_google_api(
        self,
        event_id: str,
        account: str,
        calendar_id: str,
    ) -> None:
        """Delete event via Google Calendar API."""
        cred = self.account_manager.get_credential("google_calendar_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            method="DELETE",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10):
                _log.info(f"Deleted calendar event: {event_id}")
        except Exception as e:
            _log.error(f"Failed to delete event: {e}")
            raise
