"""
DIII-D (General Atomics) data source client.

Connects to the General Atomics MDSplus server (`atlas.gat.com`)
to retrieve experimental data from the DIII-D national fusion facility.

**Important:** Accessing DIII-D data requires explicit authorization.
You must either be on the DIII-D internal network, authenticated via VPN,
or have an active SSH tunnel to the MDSplus server.

Uses the pure-Python `mdsthin` package, avoiding the need for full
MDSplus C++ libraries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from fusiondata.client import BaseClient, DataNotFoundError
from fusiondata.models import Signal, Experiment

logger = logging.getLogger("fusiondata")


class AuthError(Exception):
    """Raised when authentication/network access to a restricted source fails."""
    pass


class D3D(BaseClient):
    """Client for DIII-D MDSplus data.

    The DIII-D National Fusion Facility is operated by General Atomics.
    All data is stored in an MDSplus hierarchical database.

    **Note:** This client requires you to have DIII-D access credentials
    and be on the internal network (or via VPN/SSH tunnel).
    It will fail gracefully with an `AuthError` if it cannot connect.

    Example:
        >>> from fusiondata import D3D
        >>> d3d = D3D()
        >>> # Retrieve electron density from the 'efit01' tree
        >>> ne = d3d.get_signal(165920, "ne")
        >>> ne.plot()
    """

    SOURCE_NAME = "DIII-D"
    SERVER = "atlas.gat.com"
    PORT = 8000

    def __init__(self, server: Optional[str] = None, port: Optional[int] = None, **kwargs):
        # We don't use base_url for HTTP requests in MDSplus, but BaseClient expects it
        super().__init__(base_url=f"mdsplus://{server or self.SERVER}", **kwargs)
        self.server = server or self.SERVER
        self.port = port or self.PORT

        # Check for mdsthin
        try:
            import mdsthin
            self._Connection = mdsthin.Connection
        except ImportError:
            raise ImportError(
                "DIII-D access requires the 'mdsthin' package.\n"
                "Install with: pip install fusiondata[full]\n"
                "Or: pip install mdsthin"
            )

    def _get_connection(self) -> Any:
        """Get an active MDSplus connection. Fails with AuthError if blocked."""
        try:
            conn = self._Connection(self.server, port=self.port)
            return conn
        except Exception as e:
            raise AuthError(
                f"Could not connect to DIII-D MDSplus server at {self.server}:{self.port}.\n"
                "To access DIII-D data, you must have explicit authorization and be connected "
                "to the DIII-D network via VPN or SSH tunnel.\n"
                f"Original error: {e}"
            )

    # ═══════════════════════════════════════════════════════════════
    #  Signals
    # ═══════════════════════════════════════════════════════════════

    def get_signal(
        self,
        shot_id: int,
        pointname: str,
        tree: str = "efit01",
    ) -> Signal:
        """Retrieve a diagnostic signal from DIII-D.

        Args:
            shot_id: DIII-D shot number (e.g., 165920).
            pointname: Name of the point/node in MDSplus (e.g., "ne", "te").
            tree: MDSplus tree name (e.g., "efit01", "ptdata", "pcf").

        Returns:
            Signal object with time-series data.

        Example:
            >>> d3d = D3D()
            >>> ip = d3d.get_signal(165920, "ip", tree="efit01")
            >>> ip.plot()
        """
        conn = self._get_connection()

        try:
            # Open the tree for the specific shot
            conn.openTree(tree, shot_id)

            # Define TDI expression to get data and dimensions (time)
            # In MDSplus, dim_of() retrieves the time axis
            data_expr = f"{pointname}"
            time_expr = f"dim_of({pointname})"

            # Evaluate expressions
            import mdsthin.exceptions
            try:
                y_node = conn.evaluate(data_expr)
                t_node = conn.evaluate(time_expr)
            except mdsthin.exceptions.MdsException as e:
                raise DataNotFoundError(
                    f"Node '{pointname}' not found in DIII-D tree '{tree}' for shot {shot_id}. "
                    f"MDSplus error: {e}"
                )

            # Extract numpy arrays
            values = np.asarray(y_node.data, dtype=np.float64)
            timestamps = np.asarray(t_node.data, dtype=np.float64)

            # Try to fetch units if available
            try:
                units_expr = f"units_of({pointname})"
                units_node = conn.evaluate(units_expr)
                units = str(units_node.data)
            except Exception:
                units = ""

        finally:
            try:
                conn.closeAllTrees()
            except Exception:
                pass

        # Flatten if >1D array (e.g., profiles vs time) — for now we only support 1D time-traces
        # In a real app, you'd handle 2D profiles (R, Z, t) properly
        if values.ndim > 1:
            logger.warning(
                f"DIII-D point '{pointname}' returned a multidimensional array ({values.shape}). "
                "Flattening to 1D, which may not be physically meaningful."
            )
            values = values.flatten()
        if timestamps.ndim > 1:
            timestamps = timestamps.flatten()

        min_len = min(len(timestamps), len(values))
        timestamps = timestamps[:min_len]
        values = values[:min_len]

        # Validation imports here to avoid circular logic
        from fusiondata.utils import validate_signal_data
        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=pointname,
            signal_path=f"\\{tree}::{pointname}",
            values=values,
            timestamps=timestamps,
            units=units,
            source="DIII-D",
            experiment_id=str(shot_id),
            metadata={"tree": tree},
        )

    # ═══════════════════════════════════════════════════════════════
    #  Shot info
    # ═══════════════════════════════════════════════════════════════

    def get_shot(self, shot_id: int) -> Experiment:
        """Get basic metadata for a DIII-D shot.

        Note: Metadata retrieval relies on standard `efit` or `ptdata` summary nodes.
        """
        conn = self._get_connection()
        summary = ""

        try:
            # Let's try to get a basic summary comment if it exists
            # This is illustrative; exact pointnames depend on DIII-D's schema
            try:
                conn.openTree("ptdata", shot_id)
                summary_node = conn.evaluate("shot_summary")
                if summary_node and summary_node.data:
                    summary = str(summary_node.data)
            except Exception:
                pass
        finally:
            try:
                conn.closeAllTrees()
            except Exception:
                pass

        return Experiment(
            id=str(shot_id),
            source="DIII-D",
            description=summary or f"DIII-D shot #{shot_id}",
            metadata={"shot_number": shot_id},
        )
