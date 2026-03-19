"""
Plotting utilities for fusiondata.

Provides quick-plot functions with a fusion-themed matplotlib style.
All functions return the matplotlib axes object for further customization.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

if TYPE_CHECKING:
    from fusiondata.models import Signal, CrossSection


# ═══════════════════════════════════════════════════════════════════
#  Fusion-themed style
# ═══════════════════════════════════════════════════════════════════

FUSION_COLORS = {
    "plasma_hot": "#FF6B35",      # hot orange
    "plasma_core": "#E8175D",     # hot pink / core
    "plasma_edge": "#3A86FF",     # cool blue / edge
    "neutron": "#8338EC",         # purple / neutron
    "field_line": "#06D6A0",      # green / magnetic field
    "background": "#0A0E17",      # deep navy
    "grid": "#1A2332",            # dark grid
    "text": "#E0E0E0",            # light text
}

FUSION_PALETTE = [
    "#FF6B35", "#E8175D", "#3A86FF", "#06D6A0",
    "#8338EC", "#FB5607", "#FFBE0B", "#FF006E",
]


def apply_fusion_style(ax=None, fig=None):
    """Apply the fusion-themed dark style to axes."""
    if fig is None and ax is not None:
        fig = ax.get_figure()
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    fig.patch.set_facecolor(FUSION_COLORS["background"])
    ax.set_facecolor(FUSION_COLORS["background"])
    ax.tick_params(colors=FUSION_COLORS["text"], which="both")
    ax.xaxis.label.set_color(FUSION_COLORS["text"])
    ax.yaxis.label.set_color(FUSION_COLORS["text"])
    ax.title.set_color(FUSION_COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(FUSION_COLORS["grid"])
    ax.grid(True, alpha=0.2, color=FUSION_COLORS["grid"], linestyle="--")

    return ax


def plot_signal(signal: "Signal", ax=None, color: Optional[str] = None, **kwargs) -> plt.Axes:
    """Plot a time-series Signal.

    Args:
        signal: Signal object to plot.
        ax: Optional matplotlib axes.
        color: Line color (defaults to fusion palette).
        **kwargs: Additional arguments passed to ax.plot().

    Returns:
        matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    else:
        fig = ax.get_figure()

    if color is None:
        color = FUSION_COLORS["plasma_hot"]

    # Downsample if too many points for display
    if len(signal.timestamps) > 10000:
        from fusiondata.models import Signal as Sig
        plot_sig = signal.resample(10000)
        t, v = plot_sig.timestamps, plot_sig.values
    else:
        t, v = signal.timestamps, signal.values

    ax.plot(t, v, color=color, linewidth=0.8, **kwargs)
    ax.set_xlabel(f"Time [{signal._time_units}]")
    ylabel = f"{signal.name}"
    if signal.units:
        ylabel += f" [{signal.units}]"
    ax.set_ylabel(ylabel)

    title = f"{signal.name}"
    if signal.source:
        title = f"[{signal.source}] {title}"
    if signal.experiment_id:
        title += f" — {signal.experiment_id}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    apply_fusion_style(ax, fig)
    fig.tight_layout()
    return ax


def plot_signals(signals: List["Signal"], ax=None, **kwargs) -> plt.Axes:
    """Plot multiple signals on the same axes."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for i, signal in enumerate(signals):
        color = FUSION_PALETTE[i % len(FUSION_PALETTE)]
        label = signal.name if signal.name else f"Signal {i}"
        ax.plot(signal.timestamps, signal.values, color=color, linewidth=0.8,
                label=label, **kwargs)

    ax.legend(facecolor=FUSION_COLORS["background"], edgecolor=FUSION_COLORS["grid"],
              labelcolor=FUSION_COLORS["text"])
    apply_fusion_style(ax)
    ax.get_figure().tight_layout()
    return ax


def plot_cross_section(
    cs: "CrossSection",
    ax=None,
    color: Optional[str] = None,
    log_scale: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a nuclear reaction cross-section (σ vs energy).

    Args:
        cs: CrossSection object.
        ax: Optional matplotlib axes.
        color: Line color.
        log_scale: Use log-log scale (standard for cross-sections).
        **kwargs: Additional arguments for ax.plot().

    Returns:
        matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if color is None:
        color = FUSION_COLORS["neutron"]

    ax.plot(cs.energies, cs.values, color=color, linewidth=1.5, **kwargs)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(f"Energy [{cs.energy_units}]")
    ax.set_ylabel(f"Cross Section [{cs.cs_units}]")
    ax.set_title(f"Cross Section: {cs.reaction}", fontsize=13, fontweight="bold")

    # Mark peak
    peak_e, peak_v = cs.peak_energy, cs.peak_cross_section
    ax.axhline(y=peak_v, color=FUSION_COLORS["plasma_core"], linestyle=":", alpha=0.4)
    ax.annotate(
        f"Peak: {peak_v:.3g} {cs.cs_units}\nat {peak_e:.3g} {cs.energy_units}",
        xy=(peak_e, peak_v),
        xytext=(peak_e * 3, peak_v * 0.3),
        fontsize=9,
        color=FUSION_COLORS["text"],
        arrowprops=dict(arrowstyle="->", color=FUSION_COLORS["plasma_core"], lw=1.2),
    )

    apply_fusion_style(ax, fig)
    fig.tight_layout()
    return ax


def plot_experiment_overview(signals: List["Signal"], experiment=None) -> plt.Figure:
    """Create a multi-panel figure with all signals from an experiment.

    Args:
        signals: List of Signal objects to plot.
        experiment: Optional Experiment object for title metadata.

    Returns:
        matplotlib Figure.
    """
    n = len(signals)
    if n == 0:
        raise ValueError("No signals to plot")

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (signal, ax) in enumerate(zip(signals, axes)):
        color = FUSION_PALETTE[i % len(FUSION_PALETTE)]
        ax.plot(signal.timestamps, signal.values, color=color, linewidth=0.8)
        ylabel = signal.name
        if signal.units:
            ylabel += f" [{signal.units}]"
        ax.set_ylabel(ylabel, fontsize=9)
        apply_fusion_style(ax, fig)

    axes[-1].set_xlabel(f"Time [{signals[0]._time_units}]")

    if experiment:
        fig.suptitle(
            f"Experiment {experiment.id} — {experiment.source}",
            color=FUSION_COLORS["text"],
            fontsize=14,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            "Experiment Overview",
            color=FUSION_COLORS["text"],
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig
