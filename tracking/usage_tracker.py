"""
Async usage tracking service.

Provides a queue-based logging system that processes request logs
in a background thread to avoid impacting request latency.
"""

import logging
import queue
import threading
from datetime import date, datetime
from typing import Optional

from db import DailyStats, RequestLog, Setting, get_db_context

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Async usage tracking service.

    Logs requests to a queue and processes them in a background thread
    to avoid impacting request latency.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Usage tracker started")

    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        self._queue.put(None)  # Poison pill
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Usage tracker stopped")

    def is_running(self) -> bool:
        """Check if the tracker is running."""
        return self._running

    def log_request(
        self,
        timestamp: datetime,
        client_ip: str,
        hostname: Optional[str],
        tag: str,
        provider_id: str,
        model_id: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        response_time_ms: int,
        status_code: int,
        error_message: Optional[str] = None,
        is_streaming: bool = False,
    ):
        """
        Queue a request log entry.

        Args:
            timestamp: When the request was made
            client_ip: Client's IP address
            hostname: Resolved hostname (optional)
            tag: Usage attribution tag
            provider_id: Provider that handled the request
            model_id: Model used
            endpoint: API endpoint called
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            error_message: Error message if request failed
            is_streaming: Whether this was a streaming request
        """
        if not self._is_tracking_enabled():
            return

        self._queue.put(
            {
                "timestamp": timestamp,
                "client_ip": client_ip,
                "hostname": hostname,
                "tag": tag,
                "provider_id": provider_id,
                "model_id": model_id,
                "endpoint": endpoint,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_time_ms": response_time_ms,
                "status_code": status_code,
                "error_message": error_message,
                "is_streaming": is_streaming,
            }
        )

    def _is_tracking_enabled(self) -> bool:
        """Check if tracking is enabled in settings."""
        try:
            with get_db_context() as db:
                setting = (
                    db.query(Setting)
                    .filter(Setting.key == Setting.KEY_TRACKING_ENABLED)
                    .first()
                )
                if setting:
                    return setting.value.lower() == "true"
        except Exception:
            pass
        # Default to enabled
        return True

    def _worker(self):
        """Background worker that processes the log queue."""
        while self._running:
            try:
                entry = self._queue.get(timeout=1)
                if entry is None:  # Poison pill
                    break

                self._save_log(entry)
                self._update_daily_stats(entry)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing usage log: {e}")

    def _save_log(self, entry: dict):
        """Save a request log entry to the database."""
        try:
            with get_db_context() as db:
                log = RequestLog(
                    timestamp=entry["timestamp"],
                    client_ip=entry["client_ip"],
                    hostname=entry["hostname"],
                    tag=entry["tag"],
                    provider_id=entry["provider_id"],
                    model_id=entry["model_id"],
                    endpoint=entry["endpoint"],
                    input_tokens=entry["input_tokens"],
                    output_tokens=entry["output_tokens"],
                    response_time_ms=entry["response_time_ms"],
                    status_code=entry["status_code"],
                    error_message=entry["error_message"],
                    is_streaming=entry["is_streaming"],
                )
                db.add(log)
        except Exception as e:
            logger.error(f"Error saving request log: {e}")

    def _update_daily_stats(self, entry: dict):
        """
        Update pre-aggregated daily statistics.

        Updates multiple aggregation levels:
        1. Overall daily totals (no dimensions)
        2. Per-tag totals (split comma-separated tags into separate rows)
        3. Per-provider totals
        4. Per-model totals
        5. Full dimension combination (tag + provider + model)

        For multi-tag entries like "alice,project-x", creates separate DailyStats
        rows for each tag so filtering by either "alice" OR "project-x" works.
        """
        try:
            entry_date = entry["timestamp"].date()
            is_success = 200 <= entry["status_code"] < 400
            estimated_cost = self._calculate_cost(
                entry["provider_id"],
                entry["model_id"],
                entry["input_tokens"],
                entry["output_tokens"],
            )

            # Split comma-separated tags into individual tags
            raw_tag = entry["tag"]
            individual_tags = [t.strip() for t in raw_tag.split(",") if t.strip()]
            if not individual_tags:
                individual_tags = [raw_tag]  # Fallback to original if no valid splits

            # Define base aggregation levels (tag-independent)
            base_aggregations = [
                # Overall totals
                {"tag": None, "provider_id": None, "model_id": None},
                # Per-provider
                {"tag": None, "provider_id": entry["provider_id"], "model_id": None},
                # Per-model
                {
                    "tag": None,
                    "provider_id": entry["provider_id"],
                    "model_id": entry["model_id"],
                },
            ]

            with get_db_context() as db:
                # Update tag-independent aggregations
                for dims in base_aggregations:
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        dims["tag"],
                        dims["provider_id"],
                        dims["model_id"],
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )

                # Update per-tag aggregations for EACH individual tag
                for tag in individual_tags:
                    # Per-tag total
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        tag,
                        None,
                        None,
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )
                    # Full combination (tag + provider + model)
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        tag,
                        entry["provider_id"],
                        entry["model_id"],
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )

        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")

    def _upsert_daily_stat(
        self,
        db,
        stat_date: date,
        tag: Optional[str],
        provider_id: Optional[str],
        model_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
        response_time_ms: int,
        is_success: bool,
        estimated_cost: float,
    ):
        """Insert or update a daily stats record."""
        # Find existing record
        query = db.query(DailyStats).filter(
            DailyStats.date == datetime.combine(stat_date, datetime.min.time())
        )

        if tag is None:
            query = query.filter(DailyStats.tag.is_(None))
        else:
            query = query.filter(DailyStats.tag == tag)

        if provider_id is None:
            query = query.filter(DailyStats.provider_id.is_(None))
        else:
            query = query.filter(DailyStats.provider_id == provider_id)

        if model_id is None:
            query = query.filter(DailyStats.model_id.is_(None))
        else:
            query = query.filter(DailyStats.model_id == model_id)

        stat = query.first()

        if stat:
            # Update existing
            stat.request_count += 1
            stat.input_tokens += input_tokens
            stat.output_tokens += output_tokens
            stat.total_response_time_ms += response_time_ms
            stat.estimated_cost += estimated_cost
            if is_success:
                stat.success_count += 1
            else:
                stat.error_count += 1
        else:
            # Create new
            stat = DailyStats(
                date=datetime.combine(stat_date, datetime.min.time()),
                tag=tag,
                provider_id=provider_id,
                model_id=model_id,
                request_count=1,
                success_count=1 if is_success else 0,
                error_count=0 if is_success else 1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_response_time_ms=response_time_ms,
                estimated_cost=estimated_cost,
            )
            db.add(stat)

    def _calculate_cost(
        self,
        provider_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate estimated cost for a request.

        Reads cost from the hybrid model loader (YAML + custom models).

        Args:
            provider_id: Provider ID
            model_id: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        try:
            from providers.hybrid_loader import load_hybrid_models

            # Load models from YAML/hybrid system
            models, _ = load_hybrid_models(provider_id)
            model_info = models.get(model_id)

            if model_info and model_info.input_cost is not None:
                input_cost = (input_tokens / 1_000_000) * model_info.input_cost
                output_cost = (output_tokens / 1_000_000) * (
                    model_info.output_cost or 0.0
                )
                return input_cost + output_cost

        except Exception as e:
            logger.debug(f"Error calculating cost: {e}")

        return 0.0


# Global tracker instance
tracker = UsageTracker()
