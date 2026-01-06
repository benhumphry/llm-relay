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
        cost: Optional[float] = None,
        # Extended token tracking (v2.2.3)
        reasoning_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        # Alias tracking (v3.1)
        alias: Optional[str] = None,
        # Smart router tracking (v3.2)
        is_designator: bool = False,
        router_name: Optional[str] = None,
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
            cost: Actual cost from provider (if available, e.g., OpenRouter)
            reasoning_tokens: OpenAI reasoning tokens (o1/o3 models)
            cached_input_tokens: OpenAI cached prompt tokens
            cache_creation_tokens: Anthropic cache creation tokens
            cache_read_tokens: Anthropic cache read tokens
            alias: Alias name if request used an alias (v3.1)
            is_designator: Whether this was a designator call (v3.2)
            router_name: Smart router name if request used a router (v3.2)
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
                "cost": cost,
                "reasoning_tokens": reasoning_tokens,
                "cached_input_tokens": cached_input_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "cache_read_tokens": cache_read_tokens,
                "alias": alias,
                "is_designator": is_designator,
                "router_name": router_name,
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
            # Use provider-reported cost if available, otherwise calculate
            cost = entry.get("cost")
            if cost is None:
                cost = self._calculate_cost(
                    entry["provider_id"],
                    entry["model_id"],
                    entry["input_tokens"],
                    entry["output_tokens"],
                    reasoning_tokens=entry.get("reasoning_tokens"),
                    cached_input_tokens=entry.get("cached_input_tokens"),
                    cache_creation_tokens=entry.get("cache_creation_tokens"),
                    cache_read_tokens=entry.get("cache_read_tokens"),
                )

            with get_db_context() as db:
                log = RequestLog(
                    timestamp=entry["timestamp"],
                    client_ip=entry["client_ip"],
                    hostname=entry["hostname"],
                    tag=entry["tag"],
                    alias=entry.get("alias"),  # v3.1
                    provider_id=entry["provider_id"],
                    model_id=entry["model_id"],
                    endpoint=entry["endpoint"],
                    input_tokens=entry["input_tokens"],
                    output_tokens=entry["output_tokens"],
                    reasoning_tokens=entry.get("reasoning_tokens"),
                    cached_input_tokens=entry.get("cached_input_tokens"),
                    cache_creation_tokens=entry.get("cache_creation_tokens"),
                    cache_read_tokens=entry.get("cache_read_tokens"),
                    response_time_ms=entry["response_time_ms"],
                    status_code=entry["status_code"],
                    error_message=entry["error_message"],
                    is_streaming=entry["is_streaming"],
                    cost=cost,
                    is_designator=entry.get("is_designator", False),  # v3.2
                    router_name=entry.get("router_name"),  # v3.2
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
        6. Per-alias totals (v3.1)
        7. Per-router totals (v3.2)

        For multi-tag entries like "alice,project-x", creates separate DailyStats
        rows for each tag so filtering by either "alice" OR "project-x" works.
        """
        try:
            entry_date = entry["timestamp"].date()
            is_success = 200 <= entry["status_code"] < 400
            alias = entry.get("alias")
            router_name = entry.get("router_name")

            # Use provider-returned cost if available (e.g., OpenRouter),
            # otherwise calculate from static pricing
            estimated_cost = entry.get("cost")
            if estimated_cost is None:
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

            # Define base aggregation levels (tag-independent, alias-independent)
            base_aggregations = [
                # Overall totals
                {"tag": None, "provider_id": None, "model_id": None, "alias": None},
                # Per-provider
                {
                    "tag": None,
                    "provider_id": entry["provider_id"],
                    "model_id": None,
                    "alias": None,
                },
                # Per-model
                {
                    "tag": None,
                    "provider_id": entry["provider_id"],
                    "model_id": entry["model_id"],
                    "alias": None,
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
                        dims["alias"],
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
                        None,  # alias
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
                        None,  # alias
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )

                # Update per-alias aggregation if alias was used (v3.1)
                if alias:
                    # Per-alias total
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        None,  # tag
                        None,  # provider_id
                        None,  # model_id
                        alias,
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )
                    # Alias + model combination
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        None,  # tag
                        entry["provider_id"],
                        entry["model_id"],
                        alias,
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                    )

                # Update per-router aggregation if router was used (v3.2)
                if router_name:
                    # Per-router total
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        None,  # tag
                        None,  # provider_id
                        None,  # model_id
                        None,  # alias
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                        router_name=router_name,
                    )
                    # Router + model combination
                    self._upsert_daily_stat(
                        db,
                        entry_date,
                        None,  # tag
                        entry["provider_id"],
                        entry["model_id"],
                        None,  # alias
                        entry["input_tokens"],
                        entry["output_tokens"],
                        entry["response_time_ms"],
                        is_success,
                        estimated_cost,
                        router_name=router_name,
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
        alias: Optional[str],
        input_tokens: int,
        output_tokens: int,
        response_time_ms: int,
        is_success: bool,
        estimated_cost: float,
        router_name: Optional[str] = None,
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

        if alias is None:
            query = query.filter(DailyStats.alias.is_(None))
        else:
            query = query.filter(DailyStats.alias == alias)

        if router_name is None:
            query = query.filter(DailyStats.router_name.is_(None))
        else:
            query = query.filter(DailyStats.router_name == router_name)

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
                alias=alias,
                router_name=router_name,
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
        reasoning_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
    ) -> float:
        """
        Calculate estimated cost for a request.

        Reads cost from the database model loader.
        Accounts for different token types with different pricing:
        - Reasoning tokens (o1/o3): billed at output rate
        - Cached input tokens (OpenAI): 50% discount on input rate
        - Cache creation tokens (Anthropic): 1.25x input rate
        - Cache read tokens (Anthropic): 0.1x input rate

        Args:
            provider_id: Provider ID
            model_id: Model ID
            input_tokens: Number of input tokens (total, includes cached)
            output_tokens: Number of output tokens (total, includes reasoning)
            reasoning_tokens: OpenAI reasoning tokens (o1/o3 models)
            cached_input_tokens: OpenAI cached prompt tokens
            cache_creation_tokens: Anthropic cache creation tokens
            cache_read_tokens: Anthropic cache read tokens

        Returns:
            Estimated cost in USD
        """
        try:
            from providers.hybrid_loader import load_hybrid_models

            # Load models from database
            models = load_hybrid_models(provider_id)
            model_info = models.get(model_id)

            if model_info and model_info.input_cost is not None:
                input_rate = model_info.input_cost
                output_rate = model_info.output_cost or 0.0

                # Get model-specific cache multipliers, or use defaults
                # Default: OpenAI cached = 0.5x, Anthropic write = 1.25x, read = 0.1x
                cache_read_mult = (
                    model_info.cache_read_multiplier
                    if model_info.cache_read_multiplier is not None
                    else 0.1  # Default for Anthropic cache reads
                )
                cache_write_mult = (
                    model_info.cache_write_multiplier
                    if model_info.cache_write_multiplier is not None
                    else 1.25  # Default for Anthropic cache writes
                )

                # Calculate input cost
                # For OpenAI: cached tokens use cache_read_multiplier (default 0.5)
                if cached_input_tokens:
                    # OpenAI uses 0.5x for cached tokens by default
                    openai_cache_mult = (
                        model_info.cache_read_multiplier
                        if model_info.cache_read_multiplier is not None
                        else 0.5
                    )
                    regular_input = input_tokens - cached_input_tokens
                    input_cost = (regular_input / 1_000_000) * input_rate
                    input_cost += (cached_input_tokens / 1_000_000) * (
                        input_rate * openai_cache_mult
                    )
                # For Anthropic: cache creation and read use model-specific multipliers
                elif cache_creation_tokens or cache_read_tokens:
                    cache_creation = cache_creation_tokens or 0
                    cache_read = cache_read_tokens or 0
                    regular_input = input_tokens - cache_creation - cache_read
                    input_cost = (regular_input / 1_000_000) * input_rate
                    input_cost += (cache_creation / 1_000_000) * (
                        input_rate * cache_write_mult
                    )
                    input_cost += (cache_read / 1_000_000) * (
                        input_rate * cache_read_mult
                    )
                else:
                    input_cost = (input_tokens / 1_000_000) * input_rate

                # Calculate output cost
                # Reasoning tokens are already included in output_tokens and billed
                # at output rate, so no adjustment needed
                output_cost = (output_tokens / 1_000_000) * output_rate

                return input_cost + output_cost

        except Exception as e:
            logger.debug(f"Error calculating cost: {e}")

        return 0.0


# Global tracker instance
tracker = UsageTracker()
