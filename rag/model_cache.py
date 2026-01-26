"""
Unified GPU model cache with TTL-based eviction.

Manages lifecycle of GPU-loaded models (embeddings, reranker, Docling) to prevent
memory leaks while avoiding repeated model loading overhead.

Models are cached after first load and automatically unloaded after a period of
inactivity to free GPU memory.
"""

import atexit
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default TTL: 5 minutes of inactivity before unloading
# Can be overridden via GPU_MODEL_TTL_SECONDS environment variable
DEFAULT_TTL_SECONDS = int(os.environ.get("GPU_MODEL_TTL_SECONDS", 300))

# Cleanup interval: check for expired models every 60 seconds
# Can be overridden via GPU_MODEL_CLEANUP_INTERVAL environment variable
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("GPU_MODEL_CLEANUP_INTERVAL", 60))


@dataclass
class CachedModel:
    """A cached model with usage tracking."""

    key: str
    model: Any
    last_used: float = field(default_factory=time.time)
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    unload_fn: Optional[Callable[[], None]] = None  # Custom cleanup function

    def touch(self):
        """Update last used time."""
        self.last_used = time.time()

    def is_expired(self) -> bool:
        """Check if model has exceeded TTL."""
        return time.time() - self.last_used > self.ttl_seconds

    def unload(self):
        """Unload the model and free resources."""
        logger.info(f"Unloading cached model: {self.key}")
        if self.unload_fn:
            try:
                self.unload_fn()
            except Exception as e:
                logger.warning(f"Error in custom unload for {self.key}: {e}")
        # Clear reference
        self.model = None


class GPUModelCache:
    """
    Singleton cache manager for GPU models with TTL-based eviction.

    Thread-safe cache that:
    - Caches models by key to avoid repeated loading
    - Tracks last usage time for each model
    - Runs background cleanup thread to unload expired models
    - Clears CUDA cache after unloading
    """

    _instance: Optional["GPUModelCache"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: dict[str, CachedModel] = {}
        self._cache_lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._initialized = True

        # Start cleanup thread
        self._start_cleanup_thread()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="GPUModelCache-Cleanup",
        )
        self._cleanup_thread.start()
        logger.debug("Started GPU model cache cleanup thread")

    def _cleanup_loop(self):
        """Background loop that checks for and unloads expired models."""
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

            # Wait for cleanup interval or until shutdown is signaled
            # This is much more efficient than polling every second
            self._shutdown_event.wait(timeout=CLEANUP_INTERVAL_SECONDS)

    def _cleanup_expired(self):
        """Check for and unload expired models."""
        expired_keys = []

        with self._cache_lock:
            for key, cached in self._cache.items():
                if cached.is_expired():
                    expired_keys.append(key)

        if not expired_keys:
            return

        # Unload expired models
        for key in expired_keys:
            with self._cache_lock:
                cached = self._cache.pop(key, None)
            if cached:
                cached.unload()

        # Clear GPU memory after unloading
        if expired_keys:
            self._clear_gpu_memory()
            logger.info(
                f"Cleaned up {len(expired_keys)} expired model(s): {expired_keys}"
            )

    def _clear_gpu_memory(self):
        """Clear CUDA cache and run garbage collection."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to clear CUDA cache: {e}")

        gc.collect()

    def get(
        self,
        key: str,
        loader_fn: Callable[[], Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        unload_fn: Optional[Callable[[], None]] = None,
    ) -> Any:
        """
        Get a model from cache, loading it if not present.

        Args:
            key: Unique cache key for this model
            loader_fn: Function to load the model if not cached
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
            unload_fn: Optional custom cleanup function

        Returns:
            The cached or newly loaded model
        """
        with self._cache_lock:
            if key in self._cache:
                cached = self._cache[key]
                cached.touch()
                return cached.model

            # Load model inside lock to prevent concurrent loading
            # This blocks other threads but prevents the "meta tensor" error
            # that occurs when multiple threads load the same GPU model simultaneously
            logger.info(f"Loading model for cache: {key}")
            model = loader_fn()

            self._cache[key] = CachedModel(
                key=key,
                model=model,
                ttl_seconds=ttl_seconds,
                unload_fn=unload_fn,
            )

            return model

    def touch(self, key: str):
        """Update last used time for a cached model."""
        with self._cache_lock:
            if key in self._cache:
                self._cache[key].touch()

    def invalidate(self, key: str):
        """Remove a specific model from cache."""
        with self._cache_lock:
            cached = self._cache.pop(key, None)

        if cached:
            cached.unload()
            self._clear_gpu_memory()

    def clear(self):
        """Clear all cached models."""
        with self._cache_lock:
            keys = list(self._cache.keys())
            for key in keys:
                cached = self._cache.pop(key, None)
                if cached:
                    cached.unload()

        self._clear_gpu_memory()
        logger.info("Cleared all cached GPU models")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._cache_lock:
            stats = {
                "cached_models": len(self._cache),
                "models": {},
            }
            for key, cached in self._cache.items():
                age = time.time() - cached.last_used
                stats["models"][key] = {
                    "age_seconds": round(age, 1),
                    "ttl_seconds": cached.ttl_seconds,
                    "expires_in": round(cached.ttl_seconds - age, 1),
                }
            return stats

    def shutdown(self):
        """Shutdown the cache manager and cleanup thread."""
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        self.clear()
        logger.info("GPU model cache shutdown complete")


# Global cache instance
_cache: Optional[GPUModelCache] = None


def get_model_cache() -> GPUModelCache:
    """Get the global GPU model cache instance."""
    global _cache
    if _cache is None:
        _cache = GPUModelCache()
    return _cache
