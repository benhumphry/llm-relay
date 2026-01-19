"""
Action Handler Loader.

Initializes and registers all available action handlers.
"""

import logging

from .registry import register_handler

logger = logging.getLogger(__name__)


def load_action_handlers() -> None:
    """
    Load and register all action handlers.

    Call this during application startup.
    """
    logger.info("Loading action handlers...")

    # Import and register handlers
    from .handlers.calendar import CalendarActionHandler
    from .handlers.email import EmailActionHandler
    from .handlers.notification import NotificationActionHandler
    from .handlers.schedule import ScheduleActionHandler

    register_handler(EmailActionHandler())
    register_handler(CalendarActionHandler())
    register_handler(NotificationActionHandler())
    register_handler(ScheduleActionHandler())

    # Future handlers will be added here:
    # from .handlers.task import TaskActionHandler

    logger.info("Action handlers loaded")
