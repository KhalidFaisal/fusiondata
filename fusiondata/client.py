"""
Base client for fusion data sources.

Provides session management, retries, caching, and error handling
that all source-specific clients inherit from.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fusiondata.cache import DiskCache

logger = logging.getLogger("fusiondata")


class FusionDataError(Exception):
    """Base exception for fusiondata errors."""
    pass


class SourceUnavailableError(FusionDataError):
    """Raised when a data source is unreachable."""
    pass


class DataNotFoundError(FusionDataError):
    """Raised when requested data does not exist."""
    pass


class RateLimitError(FusionDataError):
    """Raised when we hit a rate limit."""
    pass


class BaseClient(ABC):
    """Abstract base class for all fusion data source clients.

    Provides:
        - Persistent HTTP session with connection pooling
        - Automatic retries with exponential backoff
        - Disk-based caching for repeated queries
        - Unified logging and error handling

    Args:
        base_url: Root URL of the data source API.
        timeout: Request timeout in seconds.
        cache_enabled: Whether to cache responses to disk.
        cache_ttl: Cache time-to-live in seconds (default: 1 hour).
        max_retries: Maximum number of retry attempts.
    """

    SOURCE_NAME: str = "Unknown"

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # --- HTTP session with retry logic ---
        self._session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({
            "User-Agent": "fusiondata/0.1.0 (https://github.com/fusiondata/fusiondata)",
            "Accept": "application/json",
        })

        # --- Cache ---
        self._cache: Optional[DiskCache] = None
        if cache_enabled:
            self._cache = DiskCache(
                namespace=self.SOURCE_NAME.lower().replace(" ", "_"),
                ttl=cache_ttl,
            )

        logger.debug(f"Initialized {self.SOURCE_NAME} client → {self.base_url}")

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        raw: bool = False,
    ) -> Any:
        """Make a cached GET request.

        Args:
            endpoint: URL path relative to base_url (or full URL).
            params: Query parameters.
            use_cache: Whether to use the cache for this request.
            raw: If True, return the raw Response object.

        Returns:
            Parsed JSON response, or raw Response if raw=True.
        """
        if endpoint.startswith(("http://", "https://")):
            url = endpoint
        else:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Check cache
        if use_cache and self._cache and not raw:
            cache_key = self._cache.make_key(url, params)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {url}")
                return cached

        # Make request
        try:
            logger.debug(f"GET {url} params={params}")
            resp = self._session.get(url, params=params, timeout=self.timeout)

            if resp.status_code == 404:
                raise DataNotFoundError(
                    f"Not found: {url} (HTTP 404)"
                )
            if resp.status_code == 429:
                raise RateLimitError(
                    f"Rate limited by {self.SOURCE_NAME}. Try again later."
                )
            resp.raise_for_status()

        except requests.ConnectionError as e:
            raise SourceUnavailableError(
                f"Cannot connect to {self.SOURCE_NAME} at {self.base_url}. "
                f"Check your internet connection. Details: {e}"
            )
        except requests.Timeout:
            raise SourceUnavailableError(
                f"Request to {self.SOURCE_NAME} timed out after {self.timeout}s."
            )

        if raw:
            return resp

        # Parse & cache
        try:
            data = resp.json()
        except ValueError:
            # Not JSON — return text
            data = resp.text

        if use_cache and self._cache:
            self._cache.set(cache_key, data)

        return data

    def _get_binary(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        """GET request that returns raw bytes (for binary data / downloads)."""
        resp = self._get(endpoint, params=params, raw=True, use_cache=False)
        return resp.content

    def clear_cache(self):
        """Clear the local disk cache for this source."""
        if self._cache:
            self._cache.clear()
            logger.info(f"Cleared cache for {self.SOURCE_NAME}")

    @property
    def is_available(self) -> bool:
        """Check if the data source is reachable."""
        try:
            self._session.head(self.base_url, timeout=5)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_url='{self.base_url}')"
