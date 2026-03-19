"""
FAIR MAST data source client.

Connects to the FAIR MAST API to retrieve data from the
Mega Ampere Spherical Tokamak (MAST) at UKAEA.

API: https://mastapp.site/api/
S3:  s3://mast (public, Zarr format)
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


class MAST(BaseClient):
    """Client for FAIR MAST tokamak data.

    MAST (Mega Ampere Spherical Tokamak) was a fusion experiment at the
    UK Atomic Energy Authority (UKAEA). The FAIR MAST initiative provides
    open access to its experimental data.

    Data is served via a JSON REST API for metadata and a public S3 bucket
    (Zarr format) for bulk diagnostic data.

    Example:
        >>> from fusiondata import MAST
        >>> mast = MAST()
        >>> shots = mast.list_shots(campaign="M9")
        >>> shot = mast.get_shot(30420)
        >>> print(shot.summary)
    """

    SOURCE_NAME = "MAST"
    BASE_URL = "https://mastapp.site/api"
    S3_BUCKET = "mast"
    S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url or self.BASE_URL, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    #  Shots
    # ═══════════════════════════════════════════════════════════════

    def list_shots(
        self,
        campaign: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        shot_min: Optional[int] = None,
        shot_max: Optional[int] = None,
    ) -> List[Experiment]:
        """List available MAST shots.

        Args:
            campaign: Filter by campaign name (e.g., "M9").
            limit: Maximum number of results.
            offset: Pagination offset.
            shot_min: Minimum shot number.
            shot_max: Maximum shot number.

        Returns:
            List of Experiment objects with shot metadata.

        Example:
            >>> mast = MAST()
            >>> shots = mast.list_shots(shot_min=30000, shot_max=30100, limit=20)
            >>> for s in shots:
            ...     print(s.id, s.date)
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if campaign:
            params["campaign"] = campaign
        if shot_min is not None:
            params["shot_min"] = shot_min
        if shot_max is not None:
            params["shot_max"] = shot_max

        data = self._get("/shots", params=params)

        shots = []
        entries = data if isinstance(data, list) else data.get("results", data.get("shots", []))
        for entry in entries:
            shot = self._parse_shot(entry)
            if shot:
                shots.append(shot)

        logger.info(f"Found {len(shots)} MAST shots")
        return shots

    def get_shot(self, shot_id: int) -> Experiment:
        """Get metadata for a specific MAST shot.

        Args:
            shot_id: MAST shot number (e.g., 30420).

        Returns:
            Experiment object with detailed shot metadata.
        """
        data = self._get(f"/shots/{shot_id}")
        shot = self._parse_shot(data)
        if not shot:
            raise DataNotFoundError(f"MAST shot {shot_id} not found")
        return shot

    def _parse_shot(self, data: Any) -> Optional[Experiment]:
        """Parse a shot JSON entry into an Experiment."""
        if not isinstance(data, dict):
            return None

        shot_id = data.get("shot_id", data.get("shot", data.get("id", "")))
        if not shot_id:
            return None

        date = None
        timestamp = data.get("timestamp", data.get("date", ""))
        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    date = datetime.datetime.fromtimestamp(timestamp)
                else:
                    date = datetime.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except (ValueError, OSError):
                pass

        return Experiment(
            id=str(shot_id),
            source="MAST",
            date=date,
            duration=data.get("duration"),
            description=data.get("description", data.get("comment", "")),
            parameters={
                k: v for k, v in data.items()
                if k not in ("shot_id", "shot", "id", "timestamp", "date", "description",
                             "comment", "duration", "signals", "diagnostics")
            },
            diagnostics=data.get("diagnostics", data.get("signals", [])),
            metadata=data,
        )

    # ═══════════════════════════════════════════════════════════════
    #  Signals
    # ═══════════════════════════════════════════════════════════════

    def list_diagnostics(self, shot_id: Optional[int] = None) -> List[str]:
        """List available diagnostic signal names.

        Args:
            shot_id: If given, list diagnostics for this specific shot.
                     Otherwise, list all known diagnostics.

        Returns:
            List of diagnostic/signal names.
        """
        if shot_id:
            try:
                data = self._get(f"/shots/{shot_id}/signals")
                if isinstance(data, list):
                    return [s.get("name", s) if isinstance(s, dict) else str(s) for s in data]
                elif isinstance(data, dict):
                    return list(data.get("signals", data.keys()))
            except Exception:
                pass
            # Fallback: get from shot metadata
            shot = self.get_shot(shot_id)
            return shot.diagnostics

        # List all diagnostics
        try:
            data = self._get("/signals")
            if isinstance(data, list):
                return [s.get("name", s) if isinstance(s, dict) else str(s) for s in data]
            elif isinstance(data, dict):
                return list(data.get("signals", data.keys()))
        except Exception:
            logger.warning("Could not list MAST diagnostics")
        return []

    def get_signal(
        self,
        shot_id: int,
        signal_name: str,
        source: str = "api",
    ) -> Signal:
        """Retrieve a diagnostic signal from a MAST shot.

        Can fetch data either via the REST API (JSON) or directly
        from the S3 Zarr store (requires xarray + s3fs).

        Args:
            shot_id: MAST shot number.
            signal_name: Name of the diagnostic signal.
            source: "api" for REST API, "s3" for direct S3/Zarr access.

        Returns:
            Signal object with time-series data.

        Example:
            >>> mast = MAST()
            >>> ne = mast.get_signal(30420, "electron_density")
            >>> ne.plot()
            >>> df = ne.to_dataframe()
        """
        if source == "s3":
            return self._get_signal_s3(shot_id, signal_name)
        return self._get_signal_api(shot_id, signal_name)

    def _get_signal_api(self, shot_id: int, signal_name: str) -> Signal:
        """Fetch signal via REST API."""
        data = self._get(f"/shots/{shot_id}/signals/{signal_name}")

        if isinstance(data, dict):
            time_data = data.get("time", data.get("dimensions", data.get("x", [])))
            value_data = data.get("data", data.get("values", data.get("y", [])))
            units = data.get("units", data.get("unit", ""))
            label = data.get("label", data.get("name", signal_name))

            if isinstance(time_data, list) and len(time_data) > 0 and isinstance(time_data[0], list):
                time_data = time_data[0]

            timestamps = np.array(time_data, dtype=np.float64)
            values = np.array(value_data, dtype=np.float64)

            if values.ndim > 1:
                values = values.flatten()

            min_len = min(len(timestamps), len(values))
            timestamps = timestamps[:min_len]
            values = values[:min_len]
        else:
            raise DataNotFoundError(
                f"Signal '{signal_name}' not found for MAST shot {shot_id}"
            )

        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=label,
            signal_path=f"shots/{shot_id}/signals/{signal_name}",
            values=values,
            timestamps=timestamps,
            units=units,
            source="MAST",
            experiment_id=str(shot_id),
            metadata=data,
        )

    def _get_signal_s3(self, shot_id: int, signal_name: str) -> Signal:
        """Fetch signal directly from S3 Zarr store.

        Requires: pip install fusiondata[full]  (xarray, s3fs, zarr)
        """
        try:
            import xarray as xr
            import s3fs
        except ImportError:
            raise ImportError(
                "S3 access requires extra dependencies. Install with:\n"
                "  pip install fusiondata[full]\n"
                "Or: pip install xarray s3fs zarr"
            )

        s3_path = f"s3://{self.S3_BUCKET}/{shot_id}/{signal_name}"
        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={"endpoint_url": self.S3_ENDPOINT},
        )

        store = s3fs.S3Map(root=s3_path, s3=fs)
        ds = xr.open_zarr(store)

        # Extract the first data variable
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Get time coordinate
        time_dim = None
        for dim in da.dims:
            if "time" in dim.lower() or dim == "t":
                time_dim = dim
                break
        if time_dim is None and len(da.dims) > 0:
            time_dim = da.dims[0]

        timestamps = da.coords[time_dim].values.astype(np.float64)
        values = da.values.astype(np.float64)

        if values.ndim > 1:
            values = values.flatten()

        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=signal_name,
            signal_path=s3_path,
            values=values,
            timestamps=timestamps,
            units=str(da.attrs.get("units", "")),
            source="MAST",
            experiment_id=str(shot_id),
            metadata=dict(da.attrs),
        )

    # ═══════════════════════════════════════════════════════════════
    #  Search
    # ═══════════════════════════════════════════════════════════════

    def search(self, query: str, limit: int = 50) -> List[Experiment]:
        """Search MAST shots by keyword or parameter.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            List of matching Experiment objects.
        """
        params = {"q": query, "limit": limit}
        try:
            data = self._get("/search", params=params)
            entries = data if isinstance(data, list) else data.get("results", [])
            return [self._parse_shot(e) for e in entries if self._parse_shot(e)]
        except Exception:
            logger.warning(f"MAST search not available, falling back to shot listing")
            return self.list_shots(limit=limit)
