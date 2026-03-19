"""
LHD (Large Helical Device) data source client.

Accesses the publicly available LHD experiment data hosted on AWS S3.

S3 bucket: s3://lhd-experiment
Registry:  https://registry.opendata.aws/lhd-experiment/

The LHD is a heliotron-type device at the National Institute for
Fusion Science (NIFS), Toki, Japan. All archived data has been
publicly available since 2023.
"""

from __future__ import annotations

import datetime
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

import numpy as np

from fusiondata.client import BaseClient, DataNotFoundError
from fusiondata.models import Signal, Experiment
from fusiondata.utils import validate_signal_data

logger = logging.getLogger("fusiondata")


class LHD(BaseClient):
    """Client for LHD (Large Helical Device) open data.

    LHD data is hosted on AWS S3 as open data. This client provides
    both direct HTTP access (no AWS credentials needed) and optional
    S3 access via s3fs for bulk downloads.

    The data is organized by shot number and diagnostic name.

    Example:
        >>> from fusiondata import LHD
        >>> lhd = LHD()
        >>> # List available diagnostics for a shot
        >>> diags = lhd.list_diagnostics(shot_id=160000)
        >>> # Get a signal
        >>> sig = lhd.get_signal(160000, "ece_fast")
        >>> sig.plot()
    """

    SOURCE_NAME = "LHD"
    BASE_URL = "https://lhd-experiment.s3.amazonaws.com"
    S3_BUCKET = "lhd-experiment"

    # Known LHD diagnostics (partial list — the full list depends on the shot)
    KNOWN_DIAGNOSTICS = [
        "ece_fast",           # Electron Cyclotron Emission (fast sampling)
        "ece_slow",           # ECE slow sampling
        "thomson_scattering", # Thomson scattering (Te, ne profiles)
        "fir",                # Far-Infrared Interferometer (line-integrated density)
        "sxr",                # Soft X-ray array
        "bolometer",          # Bolometer (radiation power)
        "magnetics",          # Magnetic diagnostics
        "wp",                 # Diamagnetic energy (stored energy)
        "ha",                 # H-alpha emission
        "impurity",           # Impurity monitor
        "nbi",                # Neutral Beam Injection parameters
        "icrh",               # Ion Cyclotron Resonance Heating
        "ecrh",               # Electron Cyclotron Resonance Heating
    ]

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url or self.BASE_URL, **kwargs)
        # Remove Accept: application/json since S3 serves XML/binary
        self._session.headers.pop("Accept", None)

    # ═══════════════════════════════════════════════════════════════
    #  Shot listing
    # ═══════════════════════════════════════════════════════════════

    def list_shots(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        cycle: Optional[int] = None,
    ) -> List[Experiment]:
        """List available LHD shots (experiment numbers).

        Note: LHD has over 170,000 shots. Use start/end to filter.

        Args:
            start: Starting shot number (inclusive).
            end: Ending shot number (inclusive).
            cycle: Experiment cycle number (e.g., 23 for the 23rd cycle).
                   Shots are roughly: cycle 1 = 1-~5000, etc.

        Returns:
            List of Experiment objects with basic metadata.

        Example:
            >>> lhd = LHD()
            >>> shots = lhd.list_shots(start=160000, end=160010)
        """
        if cycle is not None:
            # Approximate shot ranges for LHD cycles
            start = (cycle - 1) * 8000 + 1
            end = cycle * 8000

        if start is None:
            start = 160000
        if end is None:
            end = start + 100

        shots = []
        for shot_num in range(start, end + 1):
            shots.append(Experiment(
                id=str(shot_num),
                source="LHD",
                description=f"LHD shot #{shot_num}",
                metadata={"shot_number": shot_num},
            ))

        logger.info(f"Listed {len(shots)} LHD shots ({start}-{end})")
        return shots

    def get_shot(self, shot_id: int) -> Experiment:
        """Get metadata for a specific LHD shot.

        Args:
            shot_id: LHD shot number.

        Returns:
            Experiment object with available metadata.
        """
        # Try to fetch a summary/metadata file from S3
        try:
            params = {"list-type": "2", "prefix": f"{shot_id}/", "delimiter": "/"}
            data = self._get("/", params=params, use_cache=True)

            # Parse XML listing to find diagnostics
            diagnostics = []
            if isinstance(data, str) and "<Key>" in data:
                import re
                keys = re.findall(r"<Key>([^<]+)</Key>", data)
                for key in keys:
                    parts = key.split("/")
                    if len(parts) > 1:
                        diag = parts[1].split(".")[0]
                        if diag and diag not in diagnostics:
                            diagnostics.append(diag)

            return Experiment(
                id=str(shot_id),
                source="LHD",
                description=f"LHD shot #{shot_id}",
                diagnostics=diagnostics,
                metadata={"shot_number": shot_id},
            )
        except Exception as e:
            logger.debug(f"Could not fetch LHD metadata for shot {shot_id}: {e}")
            return Experiment(
                id=str(shot_id),
                source="LHD",
                description=f"LHD shot #{shot_id}",
                metadata={"shot_number": shot_id},
            )

    # ═══════════════════════════════════════════════════════════════
    #  Diagnostics
    # ═══════════════════════════════════════════════════════════════

    def list_diagnostics(self, shot_id: Optional[int] = None) -> List[str]:
        """List available diagnostics.

        Args:
            shot_id: If given, list diagnostics for this specific shot
                     by querying S3. Otherwise, return the known diagnostics
                     list (may not be complete for all shots).

        Returns:
            List of diagnostic names.
        """
        if shot_id:
            shot = self.get_shot(shot_id)
            if shot.diagnostics:
                return shot.diagnostics

        return list(self.KNOWN_DIAGNOSTICS)

    # ═══════════════════════════════════════════════════════════════
    #  Signals
    # ═══════════════════════════════════════════════════════════════

    def get_signal(
        self,
        shot_id: int,
        diagnostic: str,
        channel: int = 0,
        use_s3fs: bool = False,
    ) -> Signal:
        """Retrieve a diagnostic signal from an LHD shot.

        Args:
            shot_id: LHD shot number.
            diagnostic: Diagnostic name (e.g., "ece_fast", "fir").
            channel: Channel index (for multi-channel diagnostics).
            use_s3fs: If True, use s3fs for direct S3 access (requires
                      pip install fusiondata[full]).

        Returns:
            Signal object with time-series data.

        Example:
            >>> lhd = LHD()
            >>> wp = lhd.get_signal(160000, "wp")
            >>> wp.plot()
            >>> print(wp.stats)
        """
        if use_s3fs:
            return self._get_signal_s3fs(shot_id, diagnostic, channel)
        return self._get_signal_http(shot_id, diagnostic, channel)

    def _get_signal_http(self, shot_id: int, diagnostic: str, channel: int = 0) -> Signal:
        """Fetch signal via HTTP from S3."""
        # Try common file patterns used by LHD
        patterns = [
            f"{shot_id}/{diagnostic}.dat",
            f"{shot_id}/{diagnostic}.zip",
            f"{shot_id}/{diagnostic}/{channel}.dat",
            f"{shot_id}/{diagnostic}_ch{channel:02d}.dat",
        ]

        data = None
        used_path = None
        for pattern in patterns:
            try:
                raw = self._get_binary(f"/{pattern}")
                if raw and len(raw) > 0:
                    data = raw
                    used_path = pattern
                    break
            except Exception:
                continue

        if data is None:
            raise DataNotFoundError(
                f"Signal '{diagnostic}' not found for LHD shot {shot_id}.\n"
                f"Tried paths: {patterns}\n"
                f"Use lhd.list_diagnostics({shot_id}) to see available signals."
            )

        timestamps, values = self._parse_dat_file(data)
        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=diagnostic,
            signal_path=used_path or f"{shot_id}/{diagnostic}",
            values=values,
            timestamps=timestamps,
            units="",
            source="LHD",
            experiment_id=str(shot_id),
            metadata={"channel": channel},
        )

    def _parse_dat_file(self, raw_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Parse an LHD .dat file.

        LHD .dat files can be in several formats:
        1. ASCII: tab/space separated columns (time, value)
        2. Binary: packed float32 or float64
        """
        # Try ASCII first
        try:
            text = raw_data.decode("utf-8", errors="ignore")
            lines = [l.strip() for l in text.strip().split("\n") if l.strip() and not l.startswith("#")]

            if lines:
                # Detect separator
                first_line = lines[0]
                if "\t" in first_line:
                    sep = "\t"
                elif "," in first_line:
                    sep = ","
                else:
                    sep = None  # whitespace

                times = []
                vals = []
                for line in lines:
                    parts = line.split(sep) if sep else line.split()
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            v = float(parts[1])
                            times.append(t)
                            vals.append(v)
                        except ValueError:
                            continue

                if len(times) > 0:
                    return np.array(times), np.array(vals)
        except Exception:
            pass

        # Try binary (float32 pairs: time, value, time, value, ...)
        try:
            n_floats = len(raw_data) // 4
            if n_floats >= 2 and n_floats % 2 == 0:
                all_values = struct.unpack(f"<{n_floats}f", raw_data[:n_floats * 4])
                times = np.array(all_values[0::2])
                vals = np.array(all_values[1::2])
                if np.all(np.isfinite(times)) and np.all(np.isfinite(vals)):
                    return times, vals
        except Exception:
            pass

        # Try binary float64
        try:
            n_doubles = len(raw_data) // 8
            if n_doubles >= 2 and n_doubles % 2 == 0:
                all_values = struct.unpack(f"<{n_doubles}d", raw_data[:n_doubles * 8])
                times = np.array(all_values[0::2])
                vals = np.array(all_values[1::2])
                if np.all(np.isfinite(times)) and np.all(np.isfinite(vals)):
                    return times, vals
        except Exception:
            pass

        raise ValueError(
            f"Could not parse LHD data file ({len(raw_data)} bytes). "
            f"Format not recognized as ASCII or binary float."
        )

    def _get_signal_s3fs(self, shot_id: int, diagnostic: str, channel: int = 0) -> Signal:
        """Fetch signal using s3fs for direct S3 access."""
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "S3 access requires s3fs. Install with:\n"
                "  pip install fusiondata[full]\n"
                "Or: pip install s3fs"
            )

        fs = s3fs.S3FileSystem(anon=True)

        # Try to find the file
        prefix = f"{self.S3_BUCKET}/{shot_id}/{diagnostic}"
        try:
            files = fs.ls(prefix)
        except Exception:
            files = []

        if not files:
            # Try broader search
            try:
                all_files = fs.ls(f"{self.S3_BUCKET}/{shot_id}/")
                files = [f for f in all_files if diagnostic.lower() in f.lower()]
            except Exception:
                raise DataNotFoundError(f"Signal '{diagnostic}' not found for LHD shot {shot_id}")

        if not files:
            raise DataNotFoundError(f"Signal '{diagnostic}' not found for LHD shot {shot_id}")

        # Read the first matching file
        target = files[0]
        with fs.open(target, "rb") as f:
            raw_data = f.read()

        timestamps, values = self._parse_dat_file(raw_data)
        timestamps, values = validate_signal_data(timestamps, values)

        return Signal(
            name=diagnostic,
            signal_path=target,
            values=values,
            timestamps=timestamps,
            units="",
            source="LHD",
            experiment_id=str(shot_id),
            metadata={"channel": channel, "s3_path": target},
        )
