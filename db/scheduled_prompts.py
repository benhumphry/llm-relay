"""
Scheduled Prompt Execution CRUD operations for LLM Relay.

Provides functions to create, read, update, and query scheduled prompt executions.
These track calendar-based prompt events and their execution status.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from .connection import get_db_context
from .models import ScheduledPromptExecution

logger = logging.getLogger(__name__)


def get_execution_by_id(
    execution_id: int, db: Optional[Session] = None
) -> Optional[ScheduledPromptExecution]:
    """Get a scheduled prompt execution by its ID."""
    if db:
        return (
            db.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )

    with get_db_context() as session:
        execution = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )
        if execution:
            session.expunge(execution)
        return execution


def get_execution_by_event(
    smart_alias_id: int,
    event_id: str,
    instance_start: Optional[datetime] = None,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """
    Get an execution record for a specific calendar event.

    For recurring events, instance_start is used to identify the specific instance.
    """

    def query(session: Session):
        q = session.query(ScheduledPromptExecution).filter(
            ScheduledPromptExecution.smart_alias_id == smart_alias_id,
            ScheduledPromptExecution.event_id == event_id,
        )
        if instance_start:
            q = q.filter(ScheduledPromptExecution.instance_start == instance_start)
        return q.first()

    if db:
        return query(db)

    with get_db_context() as session:
        execution = query(session)
        if execution:
            session.expunge(execution)
        return execution


def get_executions_for_alias(
    smart_alias_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    db: Optional[Session] = None,
) -> list[ScheduledPromptExecution]:
    """Get execution records for a smart alias, optionally filtered by status."""

    def query(session: Session):
        q = session.query(ScheduledPromptExecution).filter(
            ScheduledPromptExecution.smart_alias_id == smart_alias_id
        )
        if status:
            q = q.filter(ScheduledPromptExecution.status == status)
        return (
            q.order_by(ScheduledPromptExecution.scheduled_time.desc())
            .limit(limit)
            .all()
        )

    if db:
        return query(db)

    with get_db_context() as session:
        executions = query(session)
        for e in executions:
            session.expunge(e)
        return executions


def get_pending_executions(
    before_time: Optional[datetime] = None,
    smart_alias_id: Optional[int] = None,
    db: Optional[Session] = None,
) -> list[ScheduledPromptExecution]:
    """
    Get pending executions that are due for processing.

    Args:
        before_time: Only return executions scheduled before this time (defaults to now)
        smart_alias_id: Optionally filter to a specific smart alias
    """
    if before_time is None:
        before_time = datetime.utcnow()

    def query(session: Session):
        q = session.query(ScheduledPromptExecution).filter(
            ScheduledPromptExecution.status == "pending",
            ScheduledPromptExecution.scheduled_time <= before_time,
        )
        if smart_alias_id:
            q = q.filter(ScheduledPromptExecution.smart_alias_id == smart_alias_id)
        return q.order_by(ScheduledPromptExecution.scheduled_time.asc()).all()

    if db:
        return query(db)

    with get_db_context() as session:
        executions = query(session)
        for e in executions:
            session.expunge(e)
        return executions


def get_recent_executions(
    limit: int = 50,
    smart_alias_id: Optional[int] = None,
    db: Optional[Session] = None,
) -> list[ScheduledPromptExecution]:
    """Get the most recent executions across all aliases or for a specific alias."""

    def query(session: Session):
        q = session.query(ScheduledPromptExecution)
        if smart_alias_id:
            q = q.filter(ScheduledPromptExecution.smart_alias_id == smart_alias_id)
        return q.order_by(ScheduledPromptExecution.created_at.desc()).limit(limit).all()

    if db:
        return query(db)

    with get_db_context() as session:
        executions = query(session)
        for e in executions:
            session.expunge(e)
        return executions


def create_execution(
    smart_alias_id: int,
    event_id: str,
    event_title: str,
    scheduled_time: datetime,
    ical_uid: Optional[str] = None,
    instance_start: Optional[datetime] = None,
    event_description: Optional[str] = None,
    status: str = "pending",
    db: Optional[Session] = None,
) -> ScheduledPromptExecution:
    """
    Create a new scheduled prompt execution record.

    This is called when the scheduler discovers a new calendar event
    that should trigger a prompt execution.
    """
    execution = ScheduledPromptExecution(
        smart_alias_id=smart_alias_id,
        event_id=event_id,
        ical_uid=ical_uid,
        instance_start=instance_start,
        event_title=event_title,
        event_description=event_description,
        scheduled_time=scheduled_time,
        status=status,
    )

    if db:
        db.add(execution)
        db.flush()
        return execution

    with get_db_context() as session:
        session.add(execution)
        session.commit()
        session.refresh(execution)
        session.expunge(execution)
        return execution


def update_execution_status(
    execution_id: int,
    status: str,
    executed_at: Optional[datetime] = None,
    execution_duration_ms: Optional[int] = None,
    response_model: Optional[str] = None,
    response_tokens: Optional[int] = None,
    response_preview: Optional[str] = None,
    response_full: Optional[str] = None,
    error_message: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """
    Update the status and results of an execution.

    Called by the scheduler after attempting to execute a prompt.
    """

    def update(session: Session):
        execution = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )

        if not execution:
            return None

        execution.status = status
        execution.updated_at = datetime.utcnow()

        if executed_at:
            execution.executed_at = executed_at
        if execution_duration_ms is not None:
            execution.execution_duration_ms = execution_duration_ms
        if response_model:
            execution.response_model = response_model
        if response_tokens is not None:
            execution.response_tokens = response_tokens
        if response_preview:
            execution.response_preview = response_preview
        if response_full is not None:
            execution.response_full = response_full
        if error_message:
            execution.error_message = error_message

        return execution

    if db:
        return update(db)

    with get_db_context() as session:
        execution = update(session)
        if execution:
            session.commit()
            session.refresh(execution)
            session.expunge(execution)
        return execution


def mark_execution_running(
    execution_id: int,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """Mark an execution as currently running."""
    return update_execution_status(
        execution_id=execution_id,
        status="running",
        executed_at=datetime.utcnow(),
        db=db,
    )


def mark_execution_completed(
    execution_id: int,
    execution_duration_ms: int,
    response_model: str,
    response_tokens: Optional[int] = None,
    response_preview: Optional[str] = None,
    response_full: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """Mark an execution as successfully completed."""
    return update_execution_status(
        execution_id=execution_id,
        status="completed",
        execution_duration_ms=execution_duration_ms,
        response_model=response_model,
        response_tokens=response_tokens,
        response_preview=response_preview,
        response_full=response_full,
        db=db,
    )


def mark_execution_failed(
    execution_id: int,
    error_message: str,
    execution_duration_ms: Optional[int] = None,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """Mark an execution as failed."""

    def update(session: Session):
        execution = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )

        if not execution:
            return None

        execution.status = "failed"
        execution.error_message = error_message
        execution.retry_count += 1
        execution.updated_at = datetime.utcnow()

        if execution_duration_ms is not None:
            execution.execution_duration_ms = execution_duration_ms

        return execution

    if db:
        return update(db)

    with get_db_context() as session:
        execution = update(session)
        if execution:
            session.commit()
            session.refresh(execution)
            session.expunge(execution)
        return execution


def mark_execution_skipped(
    execution_id: int,
    reason: str,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """Mark an execution as skipped (e.g., calendar event was cancelled)."""
    return update_execution_status(
        execution_id=execution_id,
        status="skipped",
        error_message=reason,
        db=db,
    )


def reset_execution_to_pending(
    execution_id: int,
    db: Optional[Session] = None,
) -> Optional[ScheduledPromptExecution]:
    """Reset a failed or skipped execution back to pending for retry."""

    def update(session: Session):
        execution = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )

        if not execution:
            return None

        execution.status = "pending"
        execution.error_message = None
        execution.executed_at = None
        execution.execution_duration_ms = None
        execution.response_model = None
        execution.response_tokens = None
        execution.response_preview = None
        execution.response_full = None
        execution.updated_at = datetime.utcnow()

        return execution

    if db:
        return update(db)

    with get_db_context() as session:
        execution = update(session)
        if execution:
            session.commit()
            session.refresh(execution)
            session.expunge(execution)
        return execution


def delete_execution(
    execution_id: int,
    db: Optional[Session] = None,
) -> bool:
    """Delete an execution record."""

    def delete(session: Session):
        execution = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.id == execution_id)
            .first()
        )

        if not execution:
            return False

        session.delete(execution)
        return True

    if db:
        return delete(db)

    with get_db_context() as session:
        result = delete(session)
        if result:
            session.commit()
        return result


def delete_executions_for_alias(
    smart_alias_id: int,
    db: Optional[Session] = None,
) -> int:
    """Delete all execution records for a smart alias. Returns count deleted."""

    def delete(session: Session):
        count = (
            session.query(ScheduledPromptExecution)
            .filter(ScheduledPromptExecution.smart_alias_id == smart_alias_id)
            .delete()
        )
        return count

    if db:
        return delete(db)

    with get_db_context() as session:
        count = delete(session)
        session.commit()
        return count


def cleanup_old_executions(
    days_to_keep: int = 30,
    db: Optional[Session] = None,
) -> int:
    """
    Delete completed/failed executions older than the specified days.
    Returns the count of deleted records.
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=days_to_keep)

    def cleanup(session: Session):
        count = (
            session.query(ScheduledPromptExecution)
            .filter(
                ScheduledPromptExecution.status.in_(["completed", "failed", "skipped"]),
                ScheduledPromptExecution.created_at < cutoff,
            )
            .delete(synchronize_session=False)
        )
        return count

    if db:
        return cleanup(db)

    with get_db_context() as session:
        count = cleanup(session)
        session.commit()
        logger.info(f"Cleaned up {count} old scheduled prompt executions")
        return count


def get_execution_stats(
    smart_alias_id: Optional[int] = None,
    db: Optional[Session] = None,
) -> dict:
    """
    Get statistics about scheduled prompt executions.

    Returns counts by status and other useful metrics.
    """
    from sqlalchemy import func

    def query(session: Session):
        base_query = session.query(ScheduledPromptExecution)
        if smart_alias_id:
            base_query = base_query.filter(
                ScheduledPromptExecution.smart_alias_id == smart_alias_id
            )

        # Count by status
        status_counts = (
            base_query.with_entities(
                ScheduledPromptExecution.status, func.count(ScheduledPromptExecution.id)
            )
            .group_by(ScheduledPromptExecution.status)
            .all()
        )

        counts = {status: count for status, count in status_counts}

        # Get average execution time for completed
        avg_duration = (
            base_query.filter(ScheduledPromptExecution.status == "completed")
            .with_entities(func.avg(ScheduledPromptExecution.execution_duration_ms))
            .scalar()
        )

        return {
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "skipped": counts.get("skipped", 0),
            "total": sum(counts.values()),
            "avg_execution_ms": round(avg_duration) if avg_duration else None,
        }

    if db:
        return query(db)

    with get_db_context() as session:
        return query(session)


def event_already_scheduled(
    smart_alias_id: int,
    event_id: str,
    instance_start: Optional[datetime] = None,
    db: Optional[Session] = None,
) -> bool:
    """
    Check if a calendar event has already been scheduled for execution.

    This prevents duplicate scheduling of the same event instance.
    """
    execution = get_execution_by_event(
        smart_alias_id=smart_alias_id,
        event_id=event_id,
        instance_start=instance_start,
        db=db,
    )
    return execution is not None
