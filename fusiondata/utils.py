"""
Shared utilities for fusiondata.

Unit conversions, time formatting, and data validation helpers.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  Unit conversions common in fusion science
# ═══════════════════════════════════════════════════════════════════

def ev_to_kelvin(ev: float) -> float:
    """Convert electron-volts to Kelvin. 1 eV ≈ 11604.5 K."""
    return ev * 11604.518

def kelvin_to_ev(k: float) -> float:
    """Convert Kelvin to electron-volts."""
    return k / 11604.518

def kev_to_ev(kev: float) -> float:
    """Convert keV to eV."""
    return kev * 1000.0

def ev_to_kev(ev: float) -> float:
    """Convert eV to keV."""
    return ev / 1000.0

def barn_to_m2(barn: float) -> float:
    """Convert barns to m². 1 barn = 1e-28 m²."""
    return barn * 1e-28

def m2_to_barn(m2: float) -> float:
    """Convert m² to barns."""
    return m2 / 1e-28


# ═══════════════════════════════════════════════════════════════════
#  Time utilities
# ═══════════════════════════════════════════════════════════════════

def ns_to_s(ns: float) -> float:
    """Convert nanoseconds to seconds."""
    return ns * 1e-9

def s_to_ns(s: float) -> float:
    """Convert seconds to nanoseconds."""
    return s * 1e9

def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ═══════════════════════════════════════════════════════════════════
#  Data validation
# ═══════════════════════════════════════════════════════════════════

def validate_signal_data(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and clean signal data arrays.

    - Ensures arrays have the same length
    - Removes NaN/Inf pairs
    - Sorts by timestamp

    Returns:
        Cleaned (timestamps, values) tuple.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(timestamps) != len(values):
        raise ValueError(
            f"Timestamp and value arrays must have same length, "
            f"got {len(timestamps)} and {len(values)}"
        )

    # Remove NaN/Inf
    valid = np.isfinite(timestamps) & np.isfinite(values)
    timestamps = timestamps[valid]
    values = values[valid]

    # Sort by time
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    values = values[sort_idx]

    return timestamps, values


def parse_reaction_string(reaction: str) -> dict:
    """Parse a fusion reaction string into components.

    Supports formats like:
        "D(T,n)4He"  →  {target: "D", projectile: "T", ejectile: "n", residual: "4He"}
        "D+T→n+4He"  →  same
        "2H(3H,n)4He" → same

    Returns:
        Dictionary with target, projectile, ejectile, residual keys.
    """
    reaction = reaction.strip()

    # Try "X(Y,Z)W" format
    if "(" in reaction and ")" in reaction:
        target = reaction.split("(")[0].strip()
        inside = reaction.split("(")[1].split(")")[0]
        parts = inside.split(",")
        projectile = parts[0].strip() if len(parts) > 0 else ""
        ejectile = parts[1].strip() if len(parts) > 1 else ""
        residual = reaction.split(")")[-1].strip() if ")" in reaction else ""
        return {
            "target": target,
            "projectile": projectile,
            "ejectile": ejectile,
            "residual": residual,
        }

    # Try "A+B→C+D" format
    for arrow in ["→", "->", "⟶", "="]:
        if arrow in reaction:
            left, right = reaction.split(arrow, 1)
            left_parts = [x.strip() for x in left.split("+")]
            right_parts = [x.strip() for x in right.split("+")]
            return {
                "target": left_parts[0] if len(left_parts) > 0 else "",
                "projectile": left_parts[1] if len(left_parts) > 1 else "",
                "ejectile": right_parts[0] if len(right_parts) > 0 else "",
                "residual": right_parts[1] if len(right_parts) > 1 else "",
            }

    raise ValueError(f"Cannot parse reaction string: '{reaction}'")


# Common fusion reaction aliases for convenience
FUSION_REACTIONS = {
    "DT": "D(T,n)4He",
    "DD-p": "D(D,p)T",
    "DD-n": "D(D,n)3He",
    "D3He": "D(3He,p)4He",
    "TT": "T(T,2n)4He",
    "pB11": "p(11B,3α)",
    "p-Li7": "p(7Li,α)4He",
}
