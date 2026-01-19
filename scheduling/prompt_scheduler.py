"""
Scheduled Prompts Scheduler.

Polls Google Calendars linked to Smart Aliases and executes prompts
at their scheduled times. Calendar events become LLM prompts that run
through the Smart Alias pipeline.

Event format:
- Event title: Brief description (used for logging)
- Event description: The actual prompt to send to the LLM
- Event time: When to execute the prompt
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import requests as http_requests

logger = logging.getLogger(__name__)


class PromptScheduler:
    """
    Scheduler that polls calendars and executes due prompts.

    For each Smart Alias with scheduled_prompts_enabled:
    1. Fetches upcoming events from the linked calendar
    2. Creates execution records for new events
    3. Executes prompts that are due (scheduled_time <= now)
    4. Tracks execution results
    """

    def __init__(self, poll_interval_seconds: int = 300):
        """
        Initialize the scheduler.

        Args:
            poll_interval_seconds: How often to poll calendars (default: 5 min)
        """
        self.poll_interval = poll_interval_seconds
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._executor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Prompt scheduler already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start calendar polling thread
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="prompt-scheduler-poll",
            daemon=True,
        )
        self._poll_thread.start()

        # Start execution thread
        self._executor_thread = threading.Thread(
            target=self._executor_loop,
            name="prompt-scheduler-exec",
            daemon=True,
        )
        self._executor_thread.start()

        logger.info("Prompt scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        if self._executor_thread:
            self._executor_thread.join(timeout=5)

        logger.info("Prompt scheduler stopped")

    def _poll_loop(self):
        """Background loop that polls calendars for new events."""
        logger.info("Calendar polling loop started")

        while not self._stop_event.is_set():
            try:
                self._poll_all_calendars()
            except Exception as e:
                logger.error(f"Error in calendar poll loop: {e}", exc_info=True)

            # Wait for next poll interval
            self._stop_event.wait(timeout=self.poll_interval)

        logger.info("Calendar polling loop stopped")

    def _executor_loop(self):
        """Background loop that executes due prompts."""
        logger.info("Prompt executor loop started")

        # Check more frequently than polling (every 10 seconds)
        check_interval = 30

        while not self._stop_event.is_set():
            try:
                self._execute_due_prompts()
            except Exception as e:
                logger.error(f"Error in executor loop: {e}", exc_info=True)

            self._stop_event.wait(timeout=check_interval)

        logger.info("Prompt executor loop stopped")

    def _poll_all_calendars(self):
        """Poll calendars for all aliases with scheduled prompts enabled."""
        from db import get_all_smart_aliases

        aliases = get_all_smart_aliases()

        for alias in aliases:
            if not alias.enabled or not alias.scheduled_prompts_enabled:
                continue

            if not alias.scheduled_prompts_account_id:
                logger.warning(
                    f"Smart Alias '{alias.name}' has scheduled prompts enabled "
                    "but no calendar account configured"
                )
                continue

            try:
                self._poll_calendar_for_alias(alias)
            except Exception as e:
                logger.error(
                    f"Error polling calendar for alias '{alias.name}': {e}",
                    exc_info=True,
                )

    def _poll_calendar_for_alias(self, alias):
        """
        Poll the calendar for a specific Smart Alias and create execution records.

        Args:
            alias: SmartAlias with scheduled_prompts_enabled
        """
        from db.oauth_tokens import get_oauth_token_by_id
        from db.scheduled_prompts import create_execution, event_already_scheduled

        # Get OAuth token
        token_data = get_oauth_token_by_id(alias.scheduled_prompts_account_id)
        if not token_data:
            logger.error(
                f"OAuth token {alias.scheduled_prompts_account_id} not found "
                f"for alias '{alias.name}'"
            )
            return

        access_token = self._get_valid_access_token(
            alias.scheduled_prompts_account_id, token_data
        )
        if not access_token:
            logger.error(f"Cannot get valid access token for alias '{alias.name}'")
            return

        # Determine calendar and time range
        calendar_id = alias.scheduled_prompts_calendar_id or "primary"
        lookahead_minutes = alias.scheduled_prompts_lookahead or 15

        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(minutes=lookahead_minutes)).isoformat() + "Z"

        # Fetch upcoming events
        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": 50,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        response = http_requests.get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"Calendar API error for alias '{alias.name}': "
                f"{response.status_code} - {response.text}"
            )
            return

        data = response.json()
        events = data.get("items", [])

        logger.debug(
            f"Found {len(events)} upcoming events for alias '{alias.name}' "
            f"in next {lookahead_minutes} minutes"
        )

        # Create execution records for new events
        for event in events:
            event_id = event.get("id")
            ical_uid = event.get("iCalUID")
            summary = event.get("summary", "(No title)")
            description = event.get("description", "")

            # Parse event start time
            start = event.get("start", {})
            start_time_str = start.get("dateTime") or start.get("date")
            if not start_time_str:
                continue

            # Parse datetime
            try:
                if "T" in start_time_str:
                    # DateTime format
                    if start_time_str.endswith("Z"):
                        scheduled_time = datetime.fromisoformat(
                            start_time_str.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    elif "+" in start_time_str or (
                        "-" in start_time_str and "T" in start_time_str
                    ):
                        # Has timezone offset
                        from datetime import timezone

                        scheduled_time = datetime.fromisoformat(start_time_str)
                        scheduled_time = scheduled_time.astimezone(
                            timezone.utc
                        ).replace(tzinfo=None)
                    else:
                        scheduled_time = datetime.fromisoformat(start_time_str)
                else:
                    # All-day event - skip these for prompts
                    continue
            except ValueError as e:
                logger.warning(f"Cannot parse event time '{start_time_str}': {e}")
                continue

            # Check if already scheduled
            instance_start = scheduled_time if ical_uid else None
            if event_already_scheduled(
                smart_alias_id=alias.id,
                event_id=event_id,
                instance_start=instance_start,
            ):
                continue

            # Skip events without a description (no prompt to execute)
            if not description.strip():
                logger.debug(f"Skipping event '{summary}' - no description/prompt")
                continue

            # Create execution record
            logger.info(
                f"Scheduling prompt '{summary}' for alias '{alias.name}' "
                f"at {scheduled_time}"
            )

            create_execution(
                smart_alias_id=alias.id,
                event_id=event_id,
                event_title=summary,
                scheduled_time=scheduled_time,
                ical_uid=ical_uid,
                instance_start=instance_start,
                event_description=description,
                status="pending",
            )

    def _get_valid_access_token(
        self, account_id: int, token_data: dict
    ) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return None

        # Test the token
        test_response = http_requests.get(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"maxResults": 1},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error("Cannot refresh token - missing credentials")
            return None

        logger.info(f"Refreshing access token for account {account_id}")

        refresh_response = http_requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if refresh_response.status_code != 200:
            logger.error(f"Token refresh failed: {refresh_response.text}")
            return None

        new_token_data = refresh_response.json()
        new_access_token = new_token_data.get("access_token")

        # Update stored token
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {**token_data, "access_token": new_access_token}
        update_oauth_token_data(account_id, updated_data)

        return new_access_token

    def _execute_due_prompts(self):
        """Execute all prompts that are due."""
        from db.scheduled_prompts import (
            get_pending_executions,
            mark_execution_completed,
            mark_execution_failed,
            mark_execution_running,
        )

        # Get prompts that are due (scheduled_time <= now)
        pending = get_pending_executions(before_time=datetime.utcnow())

        if not pending:
            return

        logger.info(f"Found {len(pending)} due prompts to execute")

        for execution in pending:
            try:
                self._execute_single_prompt(execution)
            except Exception as e:
                logger.error(
                    f"Error executing prompt {execution.id}: {e}",
                    exc_info=True,
                )
                mark_execution_failed(
                    execution_id=execution.id,
                    error_message=str(e),
                )

    def _execute_single_prompt(self, execution):
        """
        Execute a single scheduled prompt.

        Args:
            execution: ScheduledPromptExecution record
        """
        from db import get_smart_alias_by_id
        from db.scheduled_prompts import (
            mark_execution_completed,
            mark_execution_failed,
            mark_execution_running,
        )

        start_time = time.time()

        # Mark as running
        mark_execution_running(execution.id)

        # Get the Smart Alias
        alias = get_smart_alias_by_id(execution.smart_alias_id)
        if not alias:
            mark_execution_failed(
                execution_id=execution.id,
                error_message=f"Smart Alias {execution.smart_alias_id} not found",
            )
            return

        if not alias.enabled:
            mark_execution_failed(
                execution_id=execution.id,
                error_message=f"Smart Alias '{alias.name}' is disabled",
            )
            return

        logger.info(
            f"Executing scheduled prompt: '{execution.event_title}' "
            f"via alias '{alias.name}'"
        )

        # Build the prompt from event description
        prompt = execution.event_description
        if not prompt:
            mark_execution_failed(
                execution_id=execution.id,
                error_message="No prompt content in event description",
            )
            return

        # Execute through the relay's own API
        try:
            response_data = self._call_relay_api(alias, prompt)

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract response content
            response_text = ""
            response_model = alias.target_model
            response_tokens = None

            if "message" in response_data:
                # Ollama format
                response_text = response_data.get("message", {}).get("content", "")
                response_model = response_data.get("model", alias.target_model)
            elif "choices" in response_data:
                # OpenAI format
                choices = response_data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                response_model = response_data.get("model", alias.target_model)
                usage = response_data.get("usage", {})
                response_tokens = usage.get("total_tokens")

            # Store results
            mark_execution_completed(
                execution_id=execution.id,
                execution_duration_ms=duration_ms,
                response_model=response_model,
                response_tokens=response_tokens,
                response_preview=response_text[:500] if response_text else None,
                response_full=response_text
                if alias.scheduled_prompts_store_response
                else None,
            )

            logger.info(
                f"Prompt '{execution.event_title}' completed in {duration_ms}ms "
                f"via {response_model}"
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            mark_execution_failed(
                execution_id=execution.id,
                error_message=str(e),
                execution_duration_ms=duration_ms,
            )
            raise

    def _call_relay_api(self, alias, prompt: str) -> dict:
        """
        Call the relay's own API to execute a prompt through a Smart Alias.

        Uses the Ollama-compatible API endpoint.
        """
        # Get the relay's internal URL
        relay_host = os.environ.get("RELAY_INTERNAL_HOST", "127.0.0.1")
        relay_port = os.environ.get("RELAY_PORT", "11434")
        relay_url = f"http://{relay_host}:{relay_port}/api/chat"

        payload = {
            "model": alias.name,  # Use the Smart Alias name as the model
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = http_requests.post(
            relay_url,
            json=payload,
            timeout=300,  # 5 minute timeout for LLM calls
        )

        if response.status_code != 200:
            raise Exception(
                f"Relay API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def poll_now(self, alias_id: Optional[int] = None):
        """
        Manually trigger a calendar poll.

        Args:
            alias_id: Optional specific alias to poll (None = poll all)
        """
        if alias_id:
            from db import get_smart_alias_by_id

            alias = get_smart_alias_by_id(alias_id)
            if alias and alias.scheduled_prompts_enabled:
                self._poll_calendar_for_alias(alias)
        else:
            self._poll_all_calendars()

    def execute_now(self, execution_id: int):
        """
        Manually execute a specific scheduled prompt.

        Args:
            execution_id: ID of the execution to run
        """
        from db.scheduled_prompts import get_execution_by_id

        execution = get_execution_by_id(execution_id)
        if execution and execution.status == "pending":
            self._execute_single_prompt(execution)


# Global scheduler instance
_scheduler: Optional[PromptScheduler] = None


def get_prompt_scheduler() -> PromptScheduler:
    """Get or create the global prompt scheduler instance."""
    global _scheduler
    if _scheduler is None:
        poll_interval = int(os.environ.get("PROMPT_SCHEDULER_INTERVAL", "60"))
        _scheduler = PromptScheduler(poll_interval_seconds=poll_interval)
    return _scheduler


def start_prompt_scheduler():
    """Start the global prompt scheduler."""
    scheduler = get_prompt_scheduler()
    scheduler.start()


def stop_prompt_scheduler():
    """Stop the global prompt scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None
