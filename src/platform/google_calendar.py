"""Google Calendar adapter — cross-platform CalendarAdapter using Google Calendar API v3.

Implements the CalendarAdapter interface so the SchedulerAgent and other agents
can create, read, update, and delete calendar events on any platform (macOS,
Linux, Windows) via the Google Calendar REST API.

Authentication is handled by GoogleOAuthTokenManager which supports both
OAuth2 flow and static Bearer token fallback.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import structlog

from src.bridges.google_oauth import GoogleOAuthTokenManager
from src.platform.base import CalendarAdapter, CalendarEvent, PlatformCapability

logger = structlog.get_logger(__name__)


class GoogleCalendarAdapter(CalendarAdapter):
    """Google Calendar API v3 adapter implementing CalendarAdapter.

    Supports:
    - List user's calendars
    - Read events (with time range and search)
    - Create events
    - Update events (partial)
    - Delete events
    - Find free/busy time

    All operations require a valid Google access token (OAuth2 or static).
    """

    BASE_URL = "https://www.googleapis.com/calendar/v3"
    REQUEST_TIMEOUT = 15.0

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        token_path: Optional[Path] = None,
        oauth_manager: Optional[GoogleOAuthTokenManager] = None,
    ) -> None:
        """Initialize the Google Calendar adapter.

        Args:
            client_id: Google OAuth client ID (for token refresh).
            client_secret: Google OAuth client secret (for token refresh).
            token_path: Path to token file (passed to GoogleOAuthTokenManager).
            oauth_manager: Pre-configured token manager (overrides token_path).
        """
        self.client_id = client_id or os.environ.get("AGENTARMY_GOOGLE_OAUTH_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("AGENTARMY_GOOGLE_OAUTH_CLIENT_SECRET", "")
        self.oauth_manager = oauth_manager or GoogleOAuthTokenManager(token_path)

        logger.info("GoogleCalendarAdapter initialized")

    # ── Private helpers ───────────────────────────────────────────

    async def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with a valid Bearer token.

        Returns:
            Dict with Authorization and Content-Type headers.

        Raises:
            ValueError: If no valid token is available.
        """
        token = await self.oauth_manager.get_valid_token(
            self.client_id, self.client_secret
        )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make an authenticated request to the Google Calendar API.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            path: API path (appended to BASE_URL).
            params: Query parameters.
            json_body: JSON request body.

        Returns:
            Parsed JSON response (empty dict for DELETE).

        Raises:
            httpx.HTTPStatusError: On API errors.
            ValueError: If not authenticated.
        """
        headers = await self._get_headers()
        url = f"{self.BASE_URL}{path}"

        async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT) as client:
            resp = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )
            resp.raise_for_status()

            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()

    @staticmethod
    def _to_calendar_event(item: dict[str, Any], calendar_name: str = "") -> CalendarEvent:
        """Convert a Google Calendar API event item to CalendarEvent.

        Args:
            item: Raw event dict from Google Calendar API.
            calendar_name: Name of the calendar this event belongs to.

        Returns:
            Normalized CalendarEvent dataclass.
        """
        start = item.get("start", {})
        end = item.get("end", {})
        return CalendarEvent(
            id=item.get("id", ""),
            title=item.get("summary", "(No title)"),
            start=start.get("dateTime") or start.get("date", ""),
            end=end.get("dateTime") or end.get("date", ""),
            calendar_name=calendar_name,
            description=item.get("description", ""),
            location=item.get("location", ""),
        )

    # ── CalendarAdapter interface ─────────────────────────────────

    async def list_calendars(self) -> List[Dict[str, str]]:
        """List all Google Calendar calendars the user has access to.

        Returns:
            List of dicts with id, name, description keys.
        """
        data = await self._request("GET", "/users/me/calendarList")
        calendars = []
        for item in data.get("items", []):
            calendars.append({
                "id": item.get("id", ""),
                "name": item.get("summary", ""),
                "description": item.get("description", ""),
                "primary": str(item.get("primary", False)),
                "backgroundColor": item.get("backgroundColor", ""),
            })
        logger.info("Listed Google calendars", count=len(calendars))
        return calendars

    async def get_events(
        self,
        *,
        start: str,
        end: str,
        calendar_name: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """Fetch events in an ISO-8601 date range.

        Args:
            start: ISO-8601 start datetime.
            end: ISO-8601 end datetime.
            calendar_name: Calendar ID (default: "primary").

        Returns:
            List of CalendarEvent in chronological order.
        """
        calendar_id = calendar_name or "primary"
        params: dict[str, Any] = {
            "timeMin": start,
            "timeMax": end,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": 250,
        }

        path = f"/calendars/{quote(calendar_id, safe='')}/events"
        data = await self._request("GET", path, params=params)

        events = [
            self._to_calendar_event(item, calendar_id)
            for item in data.get("items", [])
        ]
        logger.info(
            "Fetched Google Calendar events",
            calendar=calendar_id,
            count=len(events),
            start=start,
            end=end,
        )
        return events

    async def create_event(
        self,
        *,
        calendar_name: str,
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
    ) -> CalendarEvent:
        """Create a new calendar event.

        Args:
            calendar_name: Calendar ID (e.g. "primary").
            title: Event title/summary.
            start: ISO-8601 start datetime.
            end: ISO-8601 end datetime.
            description: Optional event description/notes.
            location: Optional event location.

        Returns:
            The created CalendarEvent.
        """
        calendar_id = calendar_name or "primary"
        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location

        path = f"/calendars/{quote(calendar_id, safe='')}/events"
        data = await self._request("POST", path, json_body=body)

        event = self._to_calendar_event(data, calendar_id)
        logger.info(
            "Created Google Calendar event",
            event_id=event.id,
            title=title,
            calendar=calendar_id,
        )
        return event

    async def update_event(
        self,
        event_id: str,
        *,
        calendar_name: Optional[str] = None,
        title: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CalendarEvent:
        """Update an existing calendar event (partial update via PATCH).

        Args:
            event_id: Google Calendar event ID.
            calendar_name: Calendar ID (default: "primary").
            title: New event title (None = keep current).
            start: New start datetime (None = keep current).
            end: New end datetime (None = keep current).
            description: New description (None = keep current).
            location: New location (None = keep current).

        Returns:
            The updated CalendarEvent.
        """
        calendar_id = calendar_name or "primary"
        body: dict[str, Any] = {}
        if title is not None:
            body["summary"] = title
        if start is not None:
            body["start"] = {"dateTime": start}
        if end is not None:
            body["end"] = {"dateTime": end}
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location

        path = f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
        data = await self._request("PATCH", path, json_body=body)

        event = self._to_calendar_event(data, calendar_id)
        logger.info(
            "Updated Google Calendar event",
            event_id=event_id,
            calendar=calendar_id,
        )
        return event

    async def delete_event(
        self,
        event_id: str,
        *,
        calendar_name: Optional[str] = None,
    ) -> bool:
        """Delete an event by ID.

        Args:
            event_id: Google Calendar event ID.
            calendar_name: Calendar ID (default: "primary").

        Returns:
            True on success.
        """
        calendar_id = calendar_name or "primary"
        path = f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"

        try:
            await self._request("DELETE", path)
            logger.info(
                "Deleted Google Calendar event",
                event_id=event_id,
                calendar=calendar_id,
            )
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 410:
                # Already deleted
                return True
            logger.error("Failed to delete event", event_id=event_id, error=str(exc))
            raise

    async def get_event(
        self,
        event_id: str,
        *,
        calendar_name: Optional[str] = None,
    ) -> CalendarEvent:
        """Get a single event by ID.

        Args:
            event_id: Google Calendar event ID.
            calendar_name: Calendar ID (default: "primary").

        Returns:
            CalendarEvent.
        """
        calendar_id = calendar_name or "primary"
        path = f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
        data = await self._request("GET", path)
        return self._to_calendar_event(data, calendar_id)

    async def find_free_time(
        self,
        *,
        calendar_ids: Optional[List[str]] = None,
        time_min: str,
        time_max: str,
    ) -> dict[str, Any]:
        """Find free/busy time across one or more calendars.

        Args:
            calendar_ids: Calendar IDs to check (default: ["primary"]).
            time_min: ISO-8601 start of range.
            time_max: ISO-8601 end of range.

        Returns:
            Dict with calendars (busy times) from Google's freeBusy API.
        """
        ids = calendar_ids or ["primary"]
        body = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": cid} for cid in ids],
        }
        data = await self._request("POST", "/freeBusy", json_body=body)
        return {
            "calendars": data.get("calendars", {}),
            "time_min": time_min,
            "time_max": time_max,
        }

    def capabilities(self) -> List[PlatformCapability]:
        """Report Google Calendar capabilities.

        Returns:
            List of PlatformCapability — all True when configured.
        """
        configured = self.oauth_manager.has_tokens
        reason = "" if configured else "Google Calendar not configured"
        return [
            PlatformCapability("local_calendar", False, "Google Calendar is cloud-based"),
            PlatformCapability("google_calendar", configured, reason),
            PlatformCapability("list_calendars", configured, reason),
            PlatformCapability("create_event", configured, reason),
            PlatformCapability("delete_event", configured, reason),
            PlatformCapability("update_event", configured, reason),
            PlatformCapability("find_free_time", configured, reason),
        ]

    @property
    def is_configured(self) -> bool:
        """Check if Google Calendar is configured (OAuth or static token)."""
        return self.oauth_manager.has_tokens
