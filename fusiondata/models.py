"""
Data models for fusion data.

Provides domain-specific containers for fusion data:
- Signal: time-series data from diagnostics
- Experiment: metadata about a plasma experiment/shot
- CrossSection: nuclear reaction cross-section data
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """A time-series signal from a fusion experiment diagnostic.

    Attributes:
        name: Human-readable signal name (e.g., "Electron Temperature").
        signal_path: Full signal path in the source archive.
        values: Array of measured values.
        timestamps: Array of timestamps (seconds from experiment start, or absolute).
        units: Physical units of the values (e.g., "eV", "m^-3").
        source: Name of the data source (e.g., "W7-X", "MAST").
        experiment_id: Identifier of the parent experiment/shot.
        metadata: Additional metadata from the source.
    """

    name: str
    signal_path: str
    values: np.ndarray
    timestamps: np.ndarray
    units: str = ""
    source: str = ""
    experiment_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame with time as index."""
        df = pd.DataFrame({
            "time": self.timestamps,
            self.name: self.values,
        })
        df = df.set_index("time")
        df.index.name = f"time [{self._time_units}]"
        df.columns = [f"{self.name} [{self.units}]" if self.units else self.name]
        return df

    @property
    def _time_units(self) -> str:
        if len(self.timestamps) > 0 and np.max(self.timestamps) > 1e6:
            return "ns"
        return "s"

    @property
    def duration(self) -> float:
        """Duration of the signal in the native time unit."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def sampling_rate(self) -> float:
        """Estimated sampling rate in Hz."""
        if len(self.timestamps) < 2:
            return 0.0
        dt = np.median(np.diff(self.timestamps))
        if dt <= 0:
            return 0.0
        return 1.0 / dt

    @property
    def stats(self) -> Dict[str, float]:
        """Quick statistics: min, max, mean, std."""
        return {
            "min": float(np.nanmin(self.values)),
            "max": float(np.nanmax(self.values)),
            "mean": float(np.nanmean(self.values)),
            "std": float(np.nanstd(self.values)),
            "samples": len(self.values),
        }

    def slice(self, t_start: float, t_end: float) -> "Signal":
        """Return a new Signal sliced to the given time range."""
        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)
        return Signal(
            name=self.name,
            signal_path=self.signal_path,
            values=self.values[mask].copy(),
            timestamps=self.timestamps[mask].copy(),
            units=self.units,
            source=self.source,
            experiment_id=self.experiment_id,
            metadata=self.metadata.copy(),
        )

    def resample(self, n_points: int) -> "Signal":
        """Downsample the signal to n_points using linear interpolation."""
        new_times = np.linspace(self.timestamps[0], self.timestamps[-1], n_points)
        new_values = np.interp(new_times, self.timestamps, self.values)
        return Signal(
            name=self.name,
            signal_path=self.signal_path,
            values=new_values,
            timestamps=new_times,
            units=self.units,
            source=self.source,
            experiment_id=self.experiment_id,
            metadata=self.metadata.copy(),
        )

    def plot(self, ax=None, **kwargs):
        """Quick matplotlib plot of the signal.

        Args:
            ax: Optional matplotlib axes to plot on.
            **kwargs: Passed to matplotlib's plot().

        Returns:
            matplotlib axes object.
        """
        from fusiondata.plotting import plot_signal
        return plot_signal(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Signal('{self.name}', {len(self.values)} samples, "
            f"source='{self.source}', units='{self.units}')"
        )

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class Experiment:
    """Metadata about a single fusion experiment (shot/program).

    Attributes:
        id: Unique experiment identifier (shot number or program ID).
        source: Data source name.
        date: Date/time of the experiment.
        duration: Duration of the plasma in seconds.
        description: Human-readable description or summary.
        parameters: Key plasma parameters (dict of name → value).
        diagnostics: List of available diagnostic signal names.
        metadata: Additional source-specific metadata.
    """

    id: str
    source: str = ""
    date: Optional[datetime.datetime] = None
    duration: Optional[float] = None
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Formatted summary string."""
        lines = [
            f"{'═' * 60}",
            f"  Experiment: {self.id}",
            f"  Source:     {self.source}",
        ]
        if self.date:
            lines.append(f"  Date:       {self.date.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.duration is not None:
            lines.append(f"  Duration:   {self.duration:.3f} s")
        if self.description:
            lines.append(f"  Info:       {self.description}")
        if self.parameters:
            lines.append(f"  Parameters:")
            for k, v in self.parameters.items():
                lines.append(f"    {k}: {v}")
        if self.diagnostics:
            lines.append(f"  Diagnostics: {len(self.diagnostics)} available")
        lines.append(f"{'═' * 60}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary."""
        d = {
            "id": self.id,
            "source": self.source,
            "date": self.date.isoformat() if self.date else None,
            "duration": self.duration,
            "description": self.description,
        }
        d.update(self.parameters)
        return d

    def __repr__(self) -> str:
        date_str = self.date.strftime('%Y-%m-%d') if self.date else "?"
        return f"Experiment('{self.id}', source='{self.source}', date='{date_str}')"


@dataclass
class CrossSection:
    """Nuclear reaction cross-section data.

    Attributes:
        reaction: Reaction string (e.g., "D(T,n)4He").
        energies: Array of projectile energies (keV, center-of-mass).
        values: Array of cross-section values (barns).
        energy_units: Units for the energy axis.
        cs_units: Units for the cross-section axis.
        source: Data source identifier.
        reference: Bibliographic reference for the data.
        metadata: Additional metadata.
    """

    reaction: str
    energies: np.ndarray
    values: np.ndarray
    energy_units: str = "keV"
    cs_units: str = "b"
    source: str = "IAEA/EXFOR"
    reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.energies = np.asarray(self.energies, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)

    @property
    def peak_cross_section(self) -> float:
        """Maximum cross-section value."""
        return float(np.nanmax(self.values))

    @property
    def peak_energy(self) -> float:
        """Energy at peak cross-section."""
        idx = np.nanargmax(self.values)
        return float(self.energies[idx])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            f"energy [{self.energy_units}]": self.energies,
            f"cross_section [{self.cs_units}]": self.values,
        })

    def at_energy(self, energy: float) -> float:
        """Interpolate cross-section at a given energy."""
        return float(np.interp(energy, self.energies, self.values))

    def plot(self, ax=None, **kwargs):
        """Quick plot of σ vs energy."""
        from fusiondata.plotting import plot_cross_section
        return plot_cross_section(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return (
            f"CrossSection('{self.reaction}', {len(self.energies)} points, "
            f"peak={self.peak_cross_section:.4g} {self.cs_units} "
            f"at {self.peak_energy:.4g} {self.energy_units})"
        )

    def __len__(self) -> int:
        return len(self.energies)
