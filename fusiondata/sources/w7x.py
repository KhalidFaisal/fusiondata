"""
Wendelstein 7-X (W7-X) data source client.

Connects to the W7-X ArchiveDB REST API to retrieve experimental data
from the world's largest optimized stellarator.

API base: http://archive-webapi.ipp-hgw.mpg.de/

The W7-X data hierarchy:
    Project → StreamGroup → DataStream → Signal data
    e.g., ArchiveDB/raw/W7XAnalysis/QSB_ECRH/...
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from fusiondata.client import BaseClient, DataNotFoundError
from fusiondata.models import Signal, Experiment
from fusiondata.utils import validate_signal_data

logger = logging.getLogger("fusiondata")


class W7X(BaseClient):
    """Client for Wendelstein 7-X ArchiveDB.

    The W7-X experiment is the world's largest optimized stellarator,
    operated by the Max Planck Institute for Plasma Physics in Greifswald.

    Example:
        >>> from fusiondata import W7X
        >>> w7x = W7X()
        >>> programs = w7x.list_programs("2023-03-15")
        >>> signal = w7x.get_signal(programs[0].id, "ECE_CECE/Te_profile")
        >>> signal.plot()
    """

    SOURCE_NAME = "W7-X"
    BASE_URL = "http://archive-webapi.ipp-hgw.mpg.de"
    LOGBOOK_URL = "https://logbook-api.ipp-hgw.mpg.de/api"

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url or self.BASE_URL, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    #  Programs (experiment sessions)
    # ═══════════════════════════════════════════════════════════════

    def list_programs(
        self,
        date: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> List[Experiment]:
        """List experiment programs for a given date or time range.

        Args:
            date: Date string "YYYY-MM-DD". Lists all programs on that day.
            from_time: ISO start time (alternative to date).
            to_time: ISO end time (alternative to date).

        Returns:
            List of Experiment objects with program metadata.

        Example:
            >>> w7x = W7X()
            >>> programs = w7x.list_programs("2023-03-15")
            >>> for p in programs:
            ...     print(p.id, p.description)
        """
        if date:
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            # W7-X API uses nanosecond timestamps
            from_ns = int(dt.timestamp() * 1e9)
            to_ns = int((dt + datetime.timedelta(days=1)).timestamp() * 1e9)
        elif from_time and to_time:
            from_dt = datetime.datetime.fromisoformat(from_time)
            to_dt = datetime.datetime.fromisoformat(to_time)
            from_ns = int(from_dt.timestamp() * 1e9)
            to_ns = int(to_dt.timestamp() * 1e9)
        else:
            raise ValueError("Provide either 'date' or both 'from_time' and 'to_time'")

        data = self._get(
            f"/programs.json",
            params={"from": from_ns, "upto": to_ns},
        )

        programs = []
        if isinstance(data, dict) and "_embedded" in data:
            entries = data["_embedded"].get("programs", [])
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        for entry in entries:
            prog = self._parse_program(entry)
            if prog:
                programs.append(prog)

        logger.info(f"Found {len(programs)} W7-X programs for the requested period")
        return programs

    def get_program(self, program_id: str) -> Experiment:
        """Get detailed metadata for a specific program.

        Args:
            program_id: The W7-X program ID (e.g., "20230315.003").

        Returns:
            Experiment object.
        """
        data = self._get(f"/programs/{program_id}.json")
        prog = self._parse_program(data)
        if not prog:
            raise DataNotFoundError(f"Program '{program_id}' not found")
        return prog

    def _parse_program(self, data: dict) -> Optional[Experiment]:
        """Parse a program JSON entry into an Experiment object."""
        if not isinstance(data, dict):
            return None

        prog_id = data.get("id", data.get("name", ""))
        if not prog_id:
            return None

        # Parse times (nanoseconds → datetime)
        from_ns = data.get("from", 0)
        upto_ns = data.get("upto", 0)
        date = None
        duration = None
        if from_ns:
            date = datetime.datetime.fromtimestamp(from_ns / 1e9)
        if from_ns and upto_ns:
            duration = (upto_ns - from_ns) / 1e9

        return Experiment(
            id=str(prog_id),
            source="W7-X",
            date=date,
            duration=duration,
            description=data.get("description", ""),
            parameters={
                k: v for k, v in data.items()
                if k not in ("id", "name", "from", "upto", "description", "_links", "_embedded")
            },
            diagnostics=[],
            metadata=data,
        )

    # ═══════════════════════════════════════════════════════════════
    #  Data streams & signals
    # ═══════════════════════════════════════════════════════════════

    def list_streams(self, path: str = "") -> List[Dict[str, Any]]:
        """Browse the W7-X data hierarchy.

        The data is organized as: Project → StreamGroup → DataStream.
        Use this to discover available signals.

        Args:
            path: Path within the hierarchy (e.g., "ArchiveDB/raw").
                  Empty string returns the top-level.

        Returns:
            List of child stream/group entries with their paths and types.

        Example:
            >>> w7x = W7X()
            >>> top = w7x.list_streams()
            >>> for s in top:
            ...     print(s["name"], s["type"])
        """
        endpoint = f"/{path}.json" if path else "/.json"
        data = self._get(endpoint)

        streams = []
        if isinstance(data, dict):
            # HAL format — extract _embedded links
            embedded = data.get("_embedded", {})
            for key, items in embedded.items():
                if isinstance(items, list):
                    for item in items:
                        streams.append({
                            "name": item.get("name", ""),
                            "type": key,
                            "path": item.get("_links", {}).get("self", {}).get("href", ""),
                            "metadata": item,
                        })
                elif isinstance(items, dict):
                    streams.append({
                        "name": items.get("name", ""),
                        "type": key,
                        "path": items.get("_links", {}).get("self", {}).get("href", ""),
                        "metadata": items,
                    })

            # Also check _links
            links = data.get("_links", {})
            for key, link in links.items():
                if key not in ("self", "curies"):
                    if isinstance(link, dict):
                        streams.append({
                            "name": key,
                            "type": "link",
                            "path": link.get("href", ""),
                            "metadata": link,
                        })

        return streams

    def get_signal(
        self,
        signal_path: str,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        program_id: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> Signal:
        """Retrieve a time-series signal from the W7-X archive.

        Args:
            signal_path: Full data stream path
                         (e.g., "ArchiveDB/codac/W7X/CoDaStationDesc.81/DataModuleDesc.21965_DATASTREAM/0/Scaled").
            from_time: ISO start time or nanosecond timestamp.
            to_time: ISO end time or nanosecond timestamp.
            program_id: If given, automatically set time range from the program.
            resolution: Number of points to return (server-side downsampling).

        Returns:
            Signal object with timestamps and values.

        Example:
            >>> w7x = W7X()
            >>> sig = w7x.get_signal(
            ...     "ArchiveDB/codac/W7X/CoDaStationDesc.81/DataModuleDesc.21965_DATASTREAM/0/Scaled",
            ...     program_id="20230315.003"
            ... )
            >>> sig.plot()
        """
        # If program_id given, resolve time range
        params = {}
        if program_id:
            prog = self.get_program(program_id)
            if prog.metadata:
                params["from"] = prog.metadata.get("from", "")
                params["upto"] = prog.metadata.get("upto", "")
        else:
            if from_time:
                params["from"] = self._parse_time(from_time)
            if to_time:
                params["upto"] = self._parse_time(to_time)

        if resolution:
            params["reduction"] = "minmax"
            params["nSamples"] = resolution

        endpoint = f"/{signal_path}/_signal.json"
        data = self._get(endpoint, params=params)

        return self._parse_signal_data(data, signal_path)

    def _parse_signal_data(self, data: Any, signal_path: str) -> Signal:
        """Parse W7-X signal JSON response into a Signal object."""
        if isinstance(data, dict):
            # Standard signal response format
            dimensions = data.get("dimensions", [])
            values_raw = data.get("values", [])

            if dimensions and len(dimensions) > 0:
                timestamps = np.array(dimensions[0], dtype=np.float64)
            else:
                timestamps = np.arange(len(values_raw), dtype=np.float64)

            values = np.array(values_raw, dtype=np.float64)

            # Flatten if nested
            if values.ndim > 1:
                values = values.flatten()
            if timestamps.ndim > 1:
                timestamps = timestamps.flatten()

            # Ensure same length
            min_len = min(len(timestamps), len(values))
            timestamps = timestamps[:min_len]
            values = values[:min_len]

            units = data.get("unit", data.get("units", ""))
            label = data.get("label", signal_path.split("/")[-1])

        elif isinstance(data, list):
            values = np.array(data, dtype=np.float64)
            timestamps = np.arange(len(values), dtype=np.float64)
            units = ""
            label = signal_path.split("/")[-1]
        else:
            raise DataNotFoundError(f"Unexpected response format for signal: {signal_path}")

        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=label,
            signal_path=signal_path,
            values=values,
            timestamps=timestamps,
            units=units,
            source="W7-X",
            metadata=data if isinstance(data, dict) else {},
        )

    def _parse_time(self, time_str: str) -> int:
        """Convert a time string to W7-X nanosecond timestamp."""
        try:
            # Already a number (nanoseconds)?
            return int(time_str)
        except (ValueError, TypeError):
            pass

        # ISO format
        dt = datetime.datetime.fromisoformat(str(time_str))
        return int(dt.timestamp() * 1e9)

    # ═══════════════════════════════════════════════════════════════
    #  Logbook
    # ═══════════════════════════════════════════════════════════════

    def get_logbook(
        self,
        program_id: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch W7-X logbook entries.

        The logbook contains operator notes, experiment descriptions,
        tags, and contextual metadata.

        Args:
            program_id: Filter by program ID.
            date: Filter by date (YYYY-MM-DD).

        Returns:
            List of logbook entry dicts.

        Example:
            >>> w7x = W7X()
            >>> entries = w7x.get_logbook(date="2023-03-15")
            >>> for e in entries:
            ...     print(e.get("title"), e.get("text"))
        """
        params = {}
        if program_id:
            params["program"] = program_id
        if date:
            params["date"] = date

        try:
            data = self._get(
                f"{self.LOGBOOK_URL}/entries",
                params=params,
            )
        except Exception:
            # Logbook API might be separate/unreachable
            logger.warning("Could not reach W7-X logbook API")
            return []

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("entries", data.get("_embedded", {}).get("entries", []))
        return []

    # ═══════════════════════════════════════════════════════════════
    #  Search
    # ═══════════════════════════════════════════════════════════════

    def search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search the W7-X archive by keyword.

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            List of matching entries with paths and descriptions.
        """
        # The W7-X archive supports browsing more than searching,
        # so we search through the main stream groups
        results = []
        query_lower = query.lower()

        try:
            top_level = self.list_streams()
            for stream in top_level:
                name = stream.get("name", "")
                if query_lower in name.lower():
                    results.append(stream)
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.warning(f"Search failed: {e}")

        return results
