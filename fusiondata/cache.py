"""
Disk-based response cache for fusiondata.

Stores API responses as JSON files in a platform-appropriate cache directory.
Supports TTL (time-to-live) expiration and manual clearing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("fusiondata")


def _get_cache_dir() -> Path:
    """Get the platform-appropriate cache directory."""
    # Use XDG_CACHE_HOME on Linux, LOCALAPPDATA on Windows, ~/Library/Caches on macOS
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.name == "posix":
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    else:
        base = Path.home() / ".cache"
    return base / "fusiondata"


class DiskCache:
    """Simple disk-based cache with TTL support.

    Args:
        namespace: Subdirectory name for this cache (e.g., source name).
        ttl: Time-to-live in seconds. Entries older than this are stale.
        max_entries: Maximum number of cached entries (LRU eviction).
    """

    def __init__(self, namespace: str = "default", ttl: int = 3600, max_entries: int = 1000):
        self.ttl = ttl
        self.max_entries = max_entries
        self.cache_dir = _get_cache_dir() / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache dir: {self.cache_dir}")

    def make_key(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key from URL and parameters."""
        raw = url
        if params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            raw += "?" + "&".join(f"{k}={v}" for k, v in sorted_params)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, or None if missing/expired."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Corrupted cache entry
            path.unlink(missing_ok=True)
            return None

        # Check TTL
        if time.time() - entry.get("timestamp", 0) > self.ttl:
            path.unlink(missing_ok=True)
            return None

        return entry.get("data")

    def set(self, key: str, data: Any) -> None:
        """Store a value in the cache."""
        entry = {
            "timestamp": time.time(),
            "data": data,
        }
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f)
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to cache: {e}")

        # Simple eviction: if too many files, remove oldest
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Remove oldest entries if cache exceeds max_entries."""
        entries = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(entries) > self.max_entries:
            oldest = entries.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug(f"Evicted cache entry: {oldest.name}")

    def clear(self):
        """Remove all cached entries."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)
            count += 1
        logger.info(f"Cleared {count} cache entries from {self.cache_dir}")

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(list(self.cache_dir.glob("*.json")))

    def __repr__(self) -> str:
        return f"DiskCache(dir='{self.cache_dir}', entries={self.size}, ttl={self.ttl}s)"
