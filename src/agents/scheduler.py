"""Scheduler agent — macOS Calendar, Google Calendar, availability & time management."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent
from src.platform.base import CalendarAdapter, PlatformCapability
from src.platform.google_calendar import GoogleCalendarAdapter

logger = structlog.get_logger(__name__)

# ── macOS Calendar AppleScript templates ────────────────────────
MACOS_CALENDAR_SCRIPTS = {
    "get_today_events": {
        "label": "Today's Events (macOS Calendar)",
        "description": "Reads all events for today from the local macOS Calendar.app",
        "script": '''
tell application "Calendar"
    set today to current date
    set todayStart to today - (time of today)
    set todayEnd to todayStart + (1 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= todayStart and start date < todayEnd)
        repeat with evt in evts
            set s to start date of evt
            set e to end date of evt
            set hStart to text -2 thru -1 of ("0" & (hours of s as text))
            set mStart to text -2 thru -1 of ("0" & (minutes of s as text))
            set hEnd to text -2 thru -1 of ("0" & (hours of e as text))
            set mEnd to text -2 thru -1 of ("0" & (minutes of e as text))
            set timeStr to hStart & ":" & mStart & "-" & hEnd & ":" & mEnd
            set output to output & timeStr & " | " & (summary of evt) & " | " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "NO_EVENTS"
    end if
    return output
end tell
''',
    },
    "get_tomorrow_events": {
        "label": "Tomorrow's Events (macOS Calendar)",
        "description": "Reads all events for tomorrow from macOS Calendar.app",
        "script": '''
tell application "Calendar"
    set today to current date
    set tomorrowStart to (today - (time of today)) + (1 * days)
    set tomorrowEnd to tomorrowStart + (1 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= tomorrowStart and start date < tomorrowEnd)
        repeat with evt in evts
            set s to start date of evt
            set e to end date of evt
            set hStart to text -2 thru -1 of ("0" & (hours of s as text))
            set mStart to text -2 thru -1 of ("0" & (minutes of s as text))
            set hEnd to text -2 thru -1 of ("0" & (hours of e as text))
            set mEnd to text -2 thru -1 of ("0" & (minutes of e as text))
            set timeStr to hStart & ":" & mStart & "-" & hEnd & ":" & mEnd
            set output to output & timeStr & " | " & (summary of evt) & " | " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "NO_EVENTS"
    end if
    return output
end tell
''',
    },
    "get_week_events": {
        "label": "This Week's Events (macOS Calendar)",
        "description": "Reads events for the current week from macOS Calendar.app",
        "script": '''
tell application "Calendar"
    set today to current date
    set weekStart to today - (time of today)
    set weekEnd to weekStart + (7 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= weekStart and start date < weekEnd)
        repeat with evt in evts
            set s to start date of evt
            set e to end date of evt
            set hStart to text -2 thru -1 of ("0" & (hours of s as text))
            set mStart to text -2 thru -1 of ("0" & (minutes of s as text))
            set hEnd to text -2 thru -1 of ("0" & (hours of e as text))
            set mEnd to text -2 thru -1 of ("0" & (minutes of e as text))
            set timeStr to hStart & ":" & mStart & "-" & hEnd & ":" & mEnd
            set d to s
            set m to (month of d as integer)
            set dayNum to day of d
            set dateStr to (year of d as text) & "-" & text -2 thru -1 of ("0" & (m as text)) & "-" & text -2 thru -1 of ("0" & (dayNum as text))
            set output to output & dateStr & " " & timeStr & " | " & (summary of evt) & " | " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "NO_EVENTS"
    end if
    return output
end tell
''',
    },
    "create_event": {
        "label": "Create Calendar Event (macOS)",
        "description": "Creates a new event in the default macOS Calendar.app calendar",
        "script": '''
-- Args: title, startISO (YYYY-MM-DDTHH:MM), endISO (YYYY-MM-DDTHH:MM) [, notes]
-- Parses dates from ISO strings — fully locale-independent.
on run argv
    set evtTitle to item 1 of argv
    set startStr to item 2 of argv
    set endStr to item 3 of argv
    set evtNotes to ""
    if (count of argv) > 3 then
        set evtNotes to item 4 of argv
    end if

    -- Parse start: "2026-02-10T14:00"
    set evtStart to current date
    set year of evtStart to (text 1 thru 4 of startStr) as integer
    set month of evtStart to (text 6 thru 7 of startStr) as integer
    set day of evtStart to (text 9 thru 10 of startStr) as integer
    set hours of evtStart to (text 12 thru 13 of startStr) as integer
    set minutes of evtStart to (text 15 thru 16 of startStr) as integer
    set seconds of evtStart to 0

    -- Parse end: "2026-02-10T15:00"
    set evtEnd to current date
    set year of evtEnd to (text 1 thru 4 of endStr) as integer
    set month of evtEnd to (text 6 thru 7 of endStr) as integer
    set day of evtEnd to (text 9 thru 10 of endStr) as integer
    set hours of evtEnd to (text 12 thru 13 of endStr) as integer
    set minutes of evtEnd to (text 15 thru 16 of endStr) as integer
    set seconds of evtEnd to 0

    tell application "Calendar"
        set targetCalendar to default calendar
        make new event at targetCalendar with properties {summary:evtTitle, start date:evtStart, end date:evtEnd, description:evtNotes}
        set calName to name of targetCalendar
    end tell
    return "Created: " & evtTitle & " on " & (evtStart as text) & " (calendar: " & calName & ")"
end run
''',
    },
    "list_calendars": {
        "label": "List Calendars (macOS)",
        "description": "Lists all available calendars in macOS Calendar.app",
        "script": '''
tell application "Calendar"
    set output to ""
    repeat with cal in calendars
        set output to output & "- " & (name of cal) & " (" & (description of cal) & ")" & linefeed
    end repeat
    return output
end tell
''',
    },
    "delete_event": {
        "label": "Delete Calendar Event (macOS)",
        "description": "Deletes a specific event by title from today's events",
        "script": '''
on run argv
    set targetTitle to item 1 of argv
    tell application "Calendar"
        set today to current date
        set todayStart to today - (time of today)
        set todayEnd to todayStart + (1 * days)
        repeat with cal in calendars
            set evts to (every event of cal whose summary is targetTitle and start date >= todayStart and start date < todayEnd)
            repeat with evt in evts
                delete evt
            end repeat
        end repeat
    end tell
    return "Deleted event: " & targetTitle
end run
''',
    },
}

# ── Google Calendar API reference ───────────────────────────────
GCAL_API_REFERENCE = {
    "list_calendars": {
        "label": "List Google Calendars",
        "description": "Fetch all Google Calendar calendars the user has access to",
        "endpoint": "GET /api/calendar/calendars",
    },
    "list_events": {
        "label": "List Events",
        "description": "Fetch events from a calendar within a time range",
        "endpoint": "GET /api/calendar/events",
        "params": ["calendar_id", "time_min", "time_max", "max_results", "query"],
    },
    "get_event": {
        "label": "Get Event Details",
        "description": "Fetch full details of a specific event",
        "endpoint": "GET /api/calendar/events/{event_id}",
        "params": ["calendar_id", "event_id"],
    },
    "create_event": {
        "label": "Create Event",
        "description": "Create a new event on Google Calendar",
        "endpoint": "POST /api/calendar/events",
        "params": ["calendar_id", "title", "start", "end", "description", "location"],
    },
    "update_event": {
        "label": "Update Event",
        "description": "Update an existing event's title, time, description, or location",
        "endpoint": "PUT /api/calendar/events/{event_id}",
        "params": ["calendar_id", "event_id", "title", "start", "end", "description", "location"],
    },
    "delete_event": {
        "label": "Delete Event",
        "description": "Delete an event from Google Calendar",
        "endpoint": "DELETE /api/calendar/events/{event_id}",
        "params": ["calendar_id", "event_id"],
    },
    "find_free_time": {
        "label": "Find Free Time",
        "description": "Find available time slots across calendars",
        "endpoint": "POST /api/calendar/free-time",
        "params": ["calendar_ids", "time_min", "time_max"],
    },
}

# ── Scheduling workflow templates ───────────────────────────────
SCHEDULING_WORKFLOWS = {
    "daily_agenda": {
        "label": "Daily Agenda",
        "description": (
            "Compiles today's events from both macOS Calendar and Google Calendar, "
            "merges them chronologically, identifies gaps/free time, "
            "and generates a formatted daily schedule."
        ),
        "steps": [
            "Read today's events from macOS Calendar.app (AppleScript)",
            "Read today's events from Google Calendar API",
            "Merge and deduplicate events by time",
            "Calculate free time blocks",
            "Format as a clean agenda with times, titles, and locations",
        ],
    },
    "weekly_overview": {
        "label": "Weekly Overview",
        "description": (
            "Generates a 7-day overview combining both calendar sources, "
            "showing busy/free patterns and meeting load per day."
        ),
        "steps": [
            "Read this week's events from macOS Calendar",
            "Read this week's events from Google Calendar",
            "Group by day, calculate daily meeting hours",
            "Identify busiest/freest days",
            "Generate weekly overview with recommendations",
        ],
    },
    "find_meeting_slot": {
        "label": "Find Meeting Slot",
        "description": (
            "Finds available time slots that work across both calendar systems. "
            "Useful for scheduling new meetings without conflicts."
        ),
        "steps": [
            "Query Google Calendar freebusy API",
            "Read macOS Calendar events for overlap",
            "Combine busy times from both sources",
            "Find gaps matching requested duration",
            "Return suggested slots ranked by preference (morning/afternoon)",
        ],
    },
    "timesheet_prep": {
        "label": "Timesheet Preparation",
        "description": (
            "Reads all events for a date range (week/month), categorizes by project/client, "
            "calculates total hours, and formats for Jira/Tempo timesheet entry."
        ),
        "steps": [
            "Read events from macOS Calendar + Google Calendar for the period",
            "Extract project/client tags from event titles",
            "Map events to Jira issue keys where possible",
            "Calculate hours per project/day",
            "Generate timesheet summary (Jira-compatible format)",
        ],
    },
    "meeting_prep": {
        "label": "Meeting Prep",
        "description": (
            "Reads upcoming meeting details (attendees, notes, agenda), "
            "fetches attendee info, and generates a prep brief."
        ),
        "steps": [
            "Get the next meeting from calendar",
            "Extract attendees and their email domains",
            "Research attendee companies/roles if external",
            "Compile meeting topic + past notes if available",
            "Generate a one-page prep brief",
        ],
    },
    "focus_time_blocker": {
        "label": "Focus Time Blocker",
        "description": (
            "Analyzes the week's schedule and creates focus time blocks "
            "in empty slots to protect deep work time."
        ),
        "steps": [
            "Read this week's full schedule from both calendars",
            "Identify gaps longer than 1 hour",
            "Filter for preferred focus times (morning/afternoon)",
            "Create 'Focus Time' events in the chosen calendar",
            "Report on total focus hours secured",
        ],
    },
}


class SchedulerAgent(BaseAgent):
    """Calendar management and scheduling agent.

    Combines macOS Calendar.app (via AppleScript/osascript) with Google Calendar
    API to provide unified calendar intelligence, scheduling, and time management.

    Responsibilities:
    - Reads events from both macOS Calendar and Google Calendar
    - Finds free time slots across all calendars
    - Creates events in either calendar system
    - Generates daily agendas and weekly overviews
    - Prepares timesheet data from calendar events
    - Helps find optimal meeting times
    - Blocks focus time in calendar gaps

    Capabilities:
    - read_calendar: Read events from macOS or Google Calendar
    - find_free_time: Find available time slots across calendars
    - create_event: Create new events in macOS or Google Calendar
    - daily_agenda: Generate a merged daily schedule
    - run_workflow: Execute multi-step scheduling workflows
    """

    def __init__(
        self,
        agent_id: str = "scheduler-001",
        name: str = "Scheduler Agent",
        role: str = "scheduler",
        calendar: Optional[CalendarAdapter] = None,
    ) -> None:
        """Initialize the Scheduler agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
            calendar: Platform-specific calendar adapter (auto-detected if None).
        """
        self._calendar = calendar
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=2,
            capabilities=[
                AgentCapability(
                    name="read_calendar",
                    version="1.0.0",
                    description="Read events from macOS Calendar or Google Calendar",
                    parameters={
                        "source": "str",  # "macos", "google", "both"
                        "range": "str",   # "today", "tomorrow", "week", "custom"
                        "start_date": "str",
                        "end_date": "str",
                    },
                ),
                AgentCapability(
                    name="find_free_time",
                    version="1.0.0",
                    description="Find available time slots across all calendars",
                    parameters={
                        "duration_minutes": "int",
                        "range_days": "int",
                        "preferred_time": "str",  # "morning", "afternoon", "any"
                    },
                ),
                AgentCapability(
                    name="create_event",
                    version="1.0.0",
                    description="Create a new calendar event (Google Calendar or macOS)",
                    parameters={
                        "target": "str",  # "google" (default), "macos"
                        "title": "str",
                        "start": "str",   # ISO 8601
                        "end": "str",     # ISO 8601
                        "calendar_name": "str",
                        "description": "str",
                        "location": "str",
                        "notes": "str",
                    },
                ),
                AgentCapability(
                    name="daily_agenda",
                    version="1.0.0",
                    description="Generate a merged daily schedule from all calendars",
                    parameters={
                        "date": "str",
                        "include_free_time": "bool",
                    },
                ),
                AgentCapability(
                    name="run_workflow",
                    version="1.0.0",
                    description="Execute a multi-step scheduling workflow",
                    parameters={
                        "template": "str",
                        "params": "dict",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._scheduling_history: list[dict[str, Any]] = []

    async def startup(self) -> None:
        """Initialize the Scheduler agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo("scheduler_startup", agent_id=self.identity.id)

    async def shutdown(self) -> None:
        """Shutdown the Scheduler agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "scheduler_shutdown",
            agent_id=self.identity.id,
            scheduling_actions=len(self._scheduling_history),
        )
        await super().shutdown()

    @property
    def calendar(self) -> CalendarAdapter:
        """Lazy-load the calendar adapter if not injected."""
        if self._calendar is None:
            from src.platform import get_calendar_adapter
            self._calendar = get_calendar_adapter()
        return self._calendar

    @property
    def google_calendar(self) -> GoogleCalendarAdapter:
        """Lazy-load a GoogleCalendarAdapter configured from env vars.

        Returns a cached instance so the OAuth token manager isn't re-created
        on every call.
        """
        if not hasattr(self, "_google_calendar_adapter"):
            self._google_calendar_adapter = GoogleCalendarAdapter(
                client_id=os.environ.get("AGENTARMY_GOOGLE_OAUTH_CLIENT_ID", ""),
                client_secret=os.environ.get("AGENTARMY_GOOGLE_OAUTH_CLIENT_SECRET", ""),
            )
        return self._google_calendar_adapter

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Dynamically adjusts the prompt based on the current platform's
        calendar capabilities — local calendar sections are only shown
        when supported.

        Returns:
            System context string describing the agent's role and expertise.
        """
        caps = {c.name: c for c in self.calendar.capabilities()}
        has_local = caps.get("local_calendar", PlatformCapability("local_calendar", False)).available

        sections: list[str] = [
            "You are a smart calendar and scheduling assistant that provides "
            "unified time management across available calendar sources.\n"
        ]

        # ── Local calendar (macOS Calendar.app, etc.) ──
        if has_local:
            macos_templates = "\n".join(
                f"  - **{info['label']}**: {info['description']}"
                for key, info in MACOS_CALENDAR_SCRIPTS.items()
            )
            sections.append(
                "## LOCAL CALENDAR (AppleScript / osascript)\n"
                "You can read and create events in the local Calendar.app:\n"
                f"{macos_templates}\n\n"
                "### HOW TO RUN SCRIPTS\n"
                "For **reading** events, write the exact script from the templates above "
                "to a temp file and run with: `osascript /tmp/calendar_script.scpt`\n\n"
                "For **creating** events, use the create_event template with arguments:\n"
                "```bash\n"
                "cat > /tmp/cal_create.scpt << 'APPLESCRIPT'\n"
                "<paste the create_event script template exactly as provided above>\n"
                "APPLESCRIPT\n"
                "osascript /tmp/cal_create.scpt \"Meeting with Tom\" \"2026-02-10T14:00\" \"2026-02-10T15:00\" \"Notes here\"\n"
                "```\n"
                "Date arguments MUST be ISO format: `YYYY-MM-DDTHH:MM` (e.g. `2026-02-10T14:00`).\n\n"
                "⚠️ CRITICAL RULES:\n"
                "- NEVER write your own AppleScript for creating events — always use the template above.\n"
                "- NEVER use `date \"...\"` coercion — it is locale-dependent and WILL FAIL on non-English macOS.\n"
                "- NEVER hardcode calendar names like \"Calendar\" — always use `default calendar`.\n"
                "- NEVER set date properties (year, month, day, hours, minutes) individually inline — "
                "the template already handles this correctly.\n"
                "- Always capture output with bash command substitution.\n"
            )
        else:
            sections.append(
                "## LOCAL CALENDAR\n"
                "Local calendar access is not available on this platform.\n"
                "All calendar operations should use Google Calendar integration.\n"
            )

        # ── Google Calendar (always available — cross-platform) ──
        gcal_operations = "\n".join(
            f"  - **{info['label']}**: {info['description']}"
            for key, info in GCAL_API_REFERENCE.items()
        )
        sections.append(
            "## GOOGLE CALENDAR API\n"
            "The system has a built-in Google Calendar integration with full CRUD support:\n"
            f"{gcal_operations}\n"
            "Use the built-in `/api/calendar/*` endpoints — DO NOT use raw Google API URLs.\n"
            "Available endpoints:\n"
            "  - `GET  /api/calendar/calendars` — list all calendars\n"
            "  - `GET  /api/calendar/events?calendar_id=...&time_min=...&time_max=...` — list events\n"
            "  - `POST /api/calendar/events` — create a new event (body: calendar_id, title, start, end, description?, location?)\n"
            "  - `GET  /api/calendar/events/{event_id}?calendar_id=...` — get event details\n"
            "  - `PUT  /api/calendar/events/{event_id}` — update event (body: calendar_id, title?, start?, end?, description?, location?)\n"
            "  - `DELETE /api/calendar/events/{event_id}?calendar_id=...` — delete event\n"
            "  - `POST /api/calendar/free-time` — find free time slots\n\n"
            "To create an event on Google Calendar, you can either:\n"
            "1. Call the adapter directly via the `google` target in create_event tasks\n"
            "2. Or instruct the user that the system will POST to /api/calendar/events\n"
        )

        # ── Scheduling workflows ──
        workflow_list = "\n".join(
            f"  - **{info['label']}**: {info['description']}"
            for key, info in SCHEDULING_WORKFLOWS.items()
        )
        sections.append(
            "## SCHEDULING WORKFLOWS\n"
            "Pre-built multi-step workflows:\n"
            f"{workflow_list}\n"
        )

        # ── Calendar capabilities summary ──
        cap_lines = "\n".join(
            f"  - {c.name}: {'✓' if c.available else '✗'}"
            + (f" ({c.reason})" if c.reason else "")
            for c in self.calendar.capabilities()
        )
        sections.append(
            "## PLATFORM CAPABILITIES\n"
            f"{cap_lines}\n"
        )

        # Detect user's local timezone
        import time as _time
        try:
            local_tz = _time.tzname[0]
            utc_offset_sec = -_time.timezone if _time.daylight == 0 else -_time.altzone
            utc_offset_h = utc_offset_sec // 3600
            utc_offset_m = abs(utc_offset_sec % 3600) // 60
            tz_offset_str = f"{utc_offset_h:+03d}:{utc_offset_m:02d}"
            try:
                from datetime import datetime as _dt
                tz_iana = _dt.now().astimezone().tzinfo
                tz_display = f"{tz_iana} (UTC{tz_offset_str})"
            except Exception:
                tz_display = f"{local_tz} (UTC{tz_offset_str})"
        except Exception:
            tz_display = "unknown — ask the user"

        now_local = datetime.now().strftime("%A, %B %d, %Y at %H:%M")

        sections.append(
            "## TIME FORMAT\n"
            "Always use ISO 8601 format for dates: YYYY-MM-DDTHH:MM:SS±HH:MM\n"
            f"User's timezone: **{tz_display}**\n"
            f"Current local time: **{now_local}**\n"
            "When presenting times to the user, use readable format: 'Mon Jan 6, 9:00 AM - 10:00 AM'\n\n"
            "## BEST PRACTICES\n"
            + ("- When asked about 'my calendar', check BOTH local and Google Calendar\n"
               "- Merge and deduplicate events from both sources\n"
               if has_local else
               "- Use Google Calendar for all calendar operations\n") +
            "- Always show events in chronological order\n"
            "- Calculate and display free time blocks between meetings\n"
            "- When creating events, ask which calendar to use\n"
            "- For timesheet workflows, categorize events by project/client based on title\n"
            "- Respect privacy — never share calendar details with other agents without permission\n"
        )

        sections.append(
            "## RESPONSE FORMATTING\n"
            "Always format responses using clean **Markdown**. Never output raw pipe-delimited data.\n"
            "For event lists, use this format:\n\n"
            "### 📅 Monday, Feb 16\n"
            "| Time | Event | Calendar |\n"
            "|------|-------|----------|\n"
            "| 11:00 – 12:00 | Team standup | Work |\n"
            "| 14:00 – 15:00 | Review meeting | Personal |\n\n"
            "### 📅 Tuesday, Feb 17\n"
            "| Time | Event | Calendar |\n"
            "|------|-------|----------|\n"
            "| All day | Holiday — Mardi Gras | Holidays |\n\n"
            "If no events found, show a friendly message like:\n"
            "> ✨ **No events scheduled** — your week is wide open!\n\n"
            "For free time, show available slots clearly.\n"
            "For errors (e.g. Google Calendar not connected), show a brief note with a fix suggestion.\n"
            "Keep responses concise. Don't dump raw command output — always parse and reformat it.\n"
        )

        return "\n\n".join(sections)

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process scheduling-related tasks.

        Supported task types:
        - read_calendar: Read events from calendars
        - find_free_time: Find available slots
        - create_event: Create new events
        - daily_agenda: Generate daily schedule
        - run_workflow: Execute scheduling workflows

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with scheduling data.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "scheduler_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "read_calendar":
                result = await self._handle_read_calendar(task)
            elif task_type == "find_free_time":
                result = await self._handle_find_free_time(task)
            elif task_type == "create_event":
                result = await self._handle_create_event(task)
            elif task_type == "daily_agenda":
                result = await self._handle_daily_agenda(task)
            elif task_type == "run_workflow":
                result = await self._handle_run_workflow(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "scheduler_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    async def _handle_read_calendar(self, task: dict[str, Any]) -> dict[str, Any]:
        """Read events from macOS or Google Calendar.

        Args:
            task: Task with calendar read parameters.

        Returns:
            Dictionary with calendar events.
        """
        params = task.get("context", {})
        source = params.get("source", "both")
        time_range = params.get("range", "today")

        await logger.ainfo("calendar_read", source=source, range=time_range)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._scheduling_history.append({
                "type": "read_calendar",
                "source": source,
                "range": time_range,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "read_calendar",
                "source": source,
                "range": time_range,
                "analysis": analysis,
                "macos_scripts": MACOS_CALENDAR_SCRIPTS,
                "gcal_reference": GCAL_API_REFERENCE,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("calendar_read_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "read_calendar",
                "source": source,
                "range": time_range,
                "macos_scripts": {k: v["label"] for k, v in MACOS_CALENDAR_SCRIPTS.items()},
                "summary": "Calendar read — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_find_free_time(self, task: dict[str, Any]) -> dict[str, Any]:
        """Find available time slots across calendars.

        Args:
            task: Task with free time search parameters.

        Returns:
            Dictionary with available time slots.
        """
        params = task.get("context", {})
        duration = params.get("duration_minutes", 60)
        range_days = params.get("range_days", 5)
        preferred = params.get("preferred_time", "any")

        await logger.ainfo("find_free_time", duration=duration, range_days=range_days)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "find_free_time",
                "duration_minutes": duration,
                "range_days": range_days,
                "preferred_time": preferred,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("find_free_time_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "find_free_time",
                "duration_minutes": duration,
                "range_days": range_days,
                "summary": "Free time search — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_create_event(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create a new calendar event.

        Supports two targets:
        - ``"google"`` — creates the event directly via GoogleCalendarAdapter
        - ``"macos"``  — falls back to AppleScript (LLM generates the command)

        When the target is ``"google"`` and all required fields are present the
        event is created immediately without an extra LLM round-trip.

        Args:
            task: Task with event creation parameters.

        Returns:
            Dictionary with creation result.
        """
        params = task.get("context", {})
        target = params.get("target", "google")  # default to google for cross-platform
        title = params.get("title", "")

        await logger.ainfo("create_event", target=target, title=title)

        # ── Direct Google Calendar creation ────────────────────────
        if target == "google" and self.google_calendar.is_configured:
            start = params.get("start", "")
            end = params.get("end", "")
            calendar_name = params.get("calendar_name") or params.get("calendar_id") or "primary"
            description = params.get("description", "") or params.get("notes", "")
            location = params.get("location", "")

            if title and start and end:
                try:
                    event = await self.google_calendar.create_event(
                        calendar_name=calendar_name,
                        title=title,
                        start=start,
                        end=end,
                        description=description,
                        location=location,
                    )

                    self._scheduling_history.append({
                        "type": "create_event",
                        "target": "google",
                        "title": title,
                        "event_id": event.id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    return {
                        "status": "completed",
                        "type": "create_event",
                        "target": "google",
                        "title": event.title,
                        "event_id": event.id,
                        "start": event.start_time,
                        "end": event.end_time,
                        "calendar": event.calendar_name,
                        "summary": f"Created '{event.title}' on Google Calendar ({event.start_time} → {event.end_time})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as exc:
                    await logger.awarning(
                        "google_create_event_failed",
                        error=str(exc),
                        title=title,
                    )
                    # Fall through to LLM-based approach

        # ── LLM-assisted creation (macOS or incomplete params) ─────
        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._scheduling_history.append({
                "type": "create_event",
                "target": target,
                "title": title,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "create_event",
                "target": target,
                "title": title,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("create_event_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "create_event",
                "target": target,
                "title": title,
                "summary": "Event creation — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_daily_agenda(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a merged daily schedule from all calendars.

        Args:
            task: Task with agenda parameters.

        Returns:
            Dictionary with daily agenda.
        """
        params = task.get("context", {})
        date = params.get("date", "today")
        include_free = params.get("include_free_time", True)

        await logger.ainfo("daily_agenda", date=date, include_free=include_free)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "daily_agenda",
                "date": date,
                "include_free_time": include_free,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("daily_agenda_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "daily_agenda",
                "date": date,
                "summary": "Daily agenda — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_run_workflow(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a multi-step scheduling workflow.

        Args:
            task: Task with workflow parameters.

        Returns:
            Dictionary with workflow execution result.
        """
        params = task.get("context", {})
        template_name = params.get("template", "")

        await logger.ainfo("scheduling_workflow", template=template_name)

        template = SCHEDULING_WORKFLOWS.get(template_name)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._scheduling_history.append({
                "type": "workflow",
                "template": template_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "run_workflow",
                "template": template_name,
                "template_info": template,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("scheduling_workflow_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "run_workflow",
                "template": template_name,
                "template_info": template,
                "available_workflows": list(SCHEDULING_WORKFLOWS.keys()),
                "summary": "Scheduling workflow — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
