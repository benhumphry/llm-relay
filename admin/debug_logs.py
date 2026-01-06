"""
Debug log streaming for the admin UI.

Provides a circular buffer for log messages and SSE streaming to clients.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Generator


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: datetime
    level: str
    logger_name: str
    message: str

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
        }


class DebugLogBuffer:
    """
    Thread-safe circular buffer for log entries with SSE streaming support.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque[LogEntry] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._subscribers: list[threading.Event] = []
        self._enabled = False

    def enable(self):
        """Enable log capture."""
        self._enabled = True

    def disable(self):
        """Disable log capture."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if log capture is enabled."""
        return self._enabled

    def add(self, entry: LogEntry):
        """Add a log entry to the buffer."""
        if not self._enabled:
            return

        with self._lock:
            self._buffer.append(entry)
            # Notify all subscribers that new data is available
            for event in self._subscribers:
                event.set()

    def get_recent(self, count: int = 100) -> list[LogEntry]:
        """Get the most recent log entries."""
        with self._lock:
            entries = list(self._buffer)
            return entries[-count:] if len(entries) > count else entries

    def clear(self):
        """Clear all log entries."""
        with self._lock:
            self._buffer.clear()

    def subscribe(self) -> threading.Event:
        """Subscribe to new log events. Returns an Event to wait on."""
        event = threading.Event()
        with self._lock:
            self._subscribers.append(event)
        return event

    def unsubscribe(self, event: threading.Event):
        """Unsubscribe from log events."""
        with self._lock:
            if event in self._subscribers:
                self._subscribers.remove(event)

    def stream(self, timeout: float = 30.0) -> Generator[LogEntry, None, None]:
        """
        Generator that yields new log entries as they arrive.
        Yields entries indefinitely until the connection is closed.
        """
        event = self.subscribe()
        last_index = len(self._buffer)

        try:
            while True:
                # Wait for new entries or timeout
                if event.wait(timeout=1.0):
                    event.clear()

                # Yield any new entries
                with self._lock:
                    current_len = len(self._buffer)
                    if current_len > last_index:
                        # Get new entries
                        entries = list(self._buffer)
                        for entry in entries[last_index:]:
                            yield entry
                        last_index = current_len
                    elif current_len < last_index:
                        # Buffer was cleared or wrapped
                        last_index = current_len

        finally:
            self.unsubscribe(event)


class BufferedLogHandler(logging.Handler):
    """
    Logging handler that sends log records to a DebugLogBuffer.
    """

    def __init__(self, buffer: DebugLogBuffer, level=logging.DEBUG):
        super().__init__(level)
        self.buffer = buffer

    def emit(self, record: logging.LogRecord):
        """Emit a log record to the buffer."""
        try:
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
            self.buffer.add(entry)
        except Exception:
            # Don't let logging errors crash the application
            pass


# Global debug log buffer instance
debug_log_buffer = DebugLogBuffer(max_size=2000)


def install_debug_handler():
    """
    Install the debug log handler on the root logger.
    Call this once during application startup.
    """
    handler = BufferedLogHandler(debug_log_buffer)
    handler.setLevel(logging.DEBUG)

    # Format log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    return handler
