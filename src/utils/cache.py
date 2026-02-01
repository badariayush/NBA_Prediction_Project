"""
Disk-based caching for HTTP API calls.

Reduces API calls and handles rate limiting gracefully.
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Simple disk-based cache for API responses.
    
    Features:
    - TTL (time-to-live) support
    - Automatic cache directory creation
    - JSON and pickle serialization
    - Cache key generation from function args
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl_hours: float = 24,
        enabled: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path for a cache key."""
        return self.cache_dir / f"{key}.meta.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if valid."""
        if not self.enabled:
            return None
        
        path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        if not path.exists() or not meta_path.exists():
            return None
        
        # Check TTL
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            created = datetime.fromisoformat(meta["created"])
            ttl_hours = meta.get("ttl_hours", self.default_ttl.total_seconds() / 3600)
            
            if datetime.now() - created > timedelta(hours=ttl_hours):
                logger.debug(f"Cache expired for key {key}")
                return None
            
            with open(path, "rb") as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_hours: Optional[float] = None) -> None:
        """Store value in cache."""
        if not self.enabled:
            return
        
        path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
            
            meta = {
                "created": datetime.now().isoformat(),
                "ttl_hours": ttl_hours or self.default_ttl.total_seconds() / 3600
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
                
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def cached(
        self,
        prefix: str,
        ttl_hours: Optional[float] = None
    ) -> Callable:
        """
        Decorator for caching function results.
        
        Usage:
            cache = DiskCache()
            
            @cache.cached("player_stats", ttl_hours=6)
            def get_player_stats(player_id: int, season: str):
                # Expensive API call
                ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                key = self._make_key(prefix, *args, **kwargs)
                
                # Try cache first
                cached_value = self.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {prefix}")
                    return cached_value
                
                # Call function and cache result
                logger.debug(f"Cache miss for {prefix}, calling function")
                result = func(*args, **kwargs)
                
                if result is not None:
                    self.set(key, result, ttl_hours)
                
                return result
            
            return wrapper
        return decorator
    
    def clear(self, older_than_hours: Optional[float] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            older_than_hours: If set, only clear entries older than this
            
        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0
        
        cleared = 0
        cutoff = None
        if older_than_hours:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                should_delete = True
                
                if cutoff:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    created = datetime.fromisoformat(meta["created"])
                    should_delete = created < cutoff
                
                if should_delete:
                    data_path = meta_path.with_suffix("").with_suffix(".pkl")
                    meta_path.unlink(missing_ok=True)
                    data_path.unlink(missing_ok=True)
                    cleared += 1
                    
            except Exception as e:
                logger.warning(f"Error clearing cache entry {meta_path}: {e}")
        
        return cleared


# Global cache instance
_cache = None


def get_cache() -> DiskCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = DiskCache()
    return _cache


def with_retry(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying failed API calls with exponential backoff.
    
    Usage:
        @with_retry
        def api_call():
            ...
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                time.sleep(current_delay)
                current_delay *= backoff
        
        raise last_exception
    
    return wrapper

