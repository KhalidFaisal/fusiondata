"""
JET (Joint European Torus) data source client.

Connects to the JET Data Centre (JDC) to retrieve Processed Pulse Files (PPF)
and JET Pulse Files (JPF) from the EUROfusion facility in Culham, UK.

**Important:** Accessing JET data requires explicit authorization.
You must have a verified EUROfusion account and provide a valid JWT token
to authenticate with the JDC API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from fusiondata.client import BaseClient, DataNotFoundError
from fusiondata.models import Signal, Experiment
from fusiondata.utils import validate_signal_data

logger = logging.getLogger("fusiondata")


class AuthError(Exception):
    """Raised when authentication/network access to a restricted source fails."""
    pass


class JET(BaseClient):
    """Client for JET (Joint European Torus) data.

    The JET tokamak was the world's largest magnetic fusion experiment,
    operated by EUROfusion at the UKAEA site in Culham until Dec 2023.

    **Note:** This client requires you to have a valid EUROfusion JWT token.
    It will fail gracefully with an `AuthError` if the token is missing,
    expired, or invalid.

    Example:
        >>> from fusiondata import JET
        >>> # You must provide your EUROfusion token
        >>> jet = JET(token="eyJhbGciOiJIUzI1NiIsInR5c...")
        >>> # Retrieve electron density (LIDR diagnostic)
        >>> ne = jet.get_signal(99971, "LIDR", "NE")
        >>> ne.plot()
    """

    SOURCE_NAME = "JET"
    # Placeholder base URL — the actual JDC API endpoint depends on the specific
    # access layer (e.g., UDA REST Gateway) provided by EUROfusion IT.
    BASE_URL = "https://data.euro-fusion.org/api/v1/jet"

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url or self.BASE_URL, **kwargs)
        self.token = token

        if self.token:
            self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _check_auth(self):
        """Verify that a token was provided."""
        if not self.token:
            raise AuthError(
                "Accessing JET data requires a EUROfusion JWT token.\n\n"
                "Please log in to the EUROfusion User Portal to generate an API token, "
                "then initialize the client with:\n"
                "  jet = JET(token='your_jwt_token')"
            )

    # ═══════════════════════════════════════════════════════════════
    #  Signals (PPF / JPF)
    # ═══════════════════════════════════════════════════════════════

    def get_signal(
        self,
        pulse: int,
        dda: str,
        dataname: str,
        type: str = "PPF",
    ) -> Signal:
        """Retrieve a signal (PPF or JPF) from a JET pulse.

        Args:
            pulse: JET pulse (shot) number (e.g., 99971 for final D-T campaign).
            dda: Diagnostic Data Area (e.g., 'LIDR', 'EFIT').
            dataname: Name of the data signal (e.g., 'NE', 'TE').
            type: Data type, usually "PPF" (Processed Pulse File) or "JPF" (Raw).

        Returns:
            Signal object with time-series data.

        Example:
            >>> jet = JET(token="...")
            >>> te = jet.get_signal(99971, "LIDR", "TE")
            >>> te.plot()
        """
        self._check_auth()

        endpoint = f"/{type.lower()}/{pulse}/{dda}/{dataname}"

        try:
            data = self._get(endpoint)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (401, 403):
                raise AuthError(
                    f"Authentication failed for JET Data Centre (HTTP {e.response.status_code}).\n"
                    "Your EUROfusion token may be expired or lack permissions for this data."
                )
            raise

        # Determine timestamps and values based on the typical JDC JSON format
        if isinstance(data, dict):
            # Assume time array is 't' or 'time' and data array is 'd' or 'data'
            t_raw = data.get("time", data.get("t", data.get("dimensions", [])))
            y_raw = data.get("data", data.get("d", data.get("values", [])))
            units = data.get("units", data.get("units_data", ""))
        else:
            raise DataNotFoundError(f"Unexpected data format from JET API for {dda}/{dataname}")

        if not t_raw or not y_raw:
            raise DataNotFoundError(f"No valid time/data arrays returned for {dda}/{dataname}")

        # In typical UDA/PPF, the time axis might be a 1D array while data could
        # be multi-dimensional (e.g., profiles). Flattening for now.
        timestamps = np.asarray(t_raw, dtype=np.float64)
        values = np.asarray(y_raw, dtype=np.float64)

        if values.ndim > 1:
            logger.warning(
                f"JET PPF '{dda}/{dataname}' returned a multidimensional array ({values.shape}). "
                "Flattening to 1D, which may not be physically meaningful."
            )
            values = values.flatten()
        if timestamps.ndim > 1:
            timestamps = timestamps.flatten()

        min_len = min(len(timestamps), len(values))
        timestamps = timestamps[:min_len]
        values = values[:min_len]

        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=f"{dda}/{dataname}",
            signal_path=f"{type}/{pulse}/{dda}/{dataname}",
            values=values,
            timestamps=timestamps,
            units=units,
            source=f"JET ({type})",
            experiment_id=str(pulse),
            metadata=data if isinstance(data, dict) else {},
        )

    # ═══════════════════════════════════════════════════════════════
    #  Pulse info
    # ═══════════════════════════════════════════════════════════════

    def get_shot(self, pulse: int) -> Experiment:
        """Get summary metadata for a JET pulse.

        Args:
            pulse: JET pulse number.

        Returns:
            Experiment object.
        """
        self._check_auth()

        try:
            data = self._get(f"/pulse/{pulse}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (401, 403):
                raise AuthError(
                    f"Authentication failed for JET Data Centre (HTTP {e.response.status_code})."
                )
            raise

        if not isinstance(data, dict):
            # Fallback if API doesn't return dict
            data = {}

        return Experiment(
            id=str(pulse),
            source="JET",
            description=data.get("description", data.get("comment", f"JET pulse #{pulse}")),
            date=None,  # Not universally available in basic JDC metadata
            duration=data.get("duration", None),
            metadata=data,
            parameters={
                k: v for k, v in data.items()
                if k not in ("pulse", "id", "description", "comment", "duration")
            },
        )
