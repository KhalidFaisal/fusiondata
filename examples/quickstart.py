"""
fusiondata quickstart example.

Demonstrates:
1. IAEA cross-sections (works offline — no API needed!)
2. Cross-section comparison plot
3. Signal object features
4. W7-X / MAST / LHD usage patterns (require internet)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import numpy as np


def demo_iaea():
    """IAEA cross-sections — works fully offline."""
    from fusiondata import IAEA

    print("\n" + "=" * 60)
    print("  IAEA Cross-Section Demo (offline)")
    print("=" * 60)

    iaea = IAEA()

    # --- 1. Get D-T cross-section ---
    dt = iaea.get_cross_section("DT")
    print(f"\nD-T reaction: {dt}")
    print(f"  Peak σ:    {dt.peak_cross_section:.4g} {dt.cs_units}")
    print(f"  At energy: {dt.peak_energy:.1f} {dt.energy_units}")
    print(f"  σ(100 keV) = {dt.at_energy(100):.4g} {dt.cs_units}")

    # Convert to DataFrame
    df = dt.to_dataframe()
    print(f"\n  DataFrame preview:")
    print(df.head(10).to_string(index=False))

    # Plot single reaction
    ax = dt.plot()
    ax.get_figure().savefig("dt_cross_section.png", dpi=150, bbox_inches="tight")
    print("\n  → Saved: dt_cross_section.png")

    # --- 2. Compare all major fusion reactions ---
    print("\n  Comparing fusion reactions...")
    ax = iaea.plot_comparison(["DT", "DD-p", "DD-n", "D3He"])
    ax.get_figure().savefig("fusion_comparison.png", dpi=150, bbox_inches="tight")
    print("  → Saved: fusion_comparison.png")

    # --- 3. List available reactions ---
    reactions = iaea.list_reactions()
    print(f"\n  Available reactions ({len(reactions)}):")
    for r in reactions:
        aliases = ", ".join(r["aliases"]) if r["aliases"] else "-"
        print(f"    {r['reaction']:20s} aliases=[{aliases}]  source={r['source']}")

    return dt


def demo_signal_features():
    """Show off Signal object capabilities with synthetic data."""
    from fusiondata.models import Signal
    from fusiondata.plotting import plot_experiment_overview

    print("\n" + "=" * 60)
    print("  Signal Feature Demo (synthetic data)")
    print("=" * 60)

    # Create a synthetic plasma signal (mimics W7-X ECE data)
    t = np.linspace(0, 5, 10000)  # 5 seconds, 2 kHz
    Te = 3.0 + 1.5 * np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(len(t))
    Te *= np.exp(-0.1 * t)  # cooling
    Te = np.maximum(Te, 0.1)

    signal = Signal(
        name="Electron Temperature",
        signal_path="ECE/Te_core",
        values=Te,
        timestamps=t,
        units="keV",
        source="Synthetic",
        experiment_id="DEMO-001",
    )

    print(f"\n  {signal}")
    print(f"  Duration:      {signal.duration:.2f} s")
    print(f"  Sampling rate: {signal.sampling_rate:.0f} Hz")
    stats = signal.stats
    print(f"  Min:  {stats['min']:.3f} keV")
    print(f"  Max:  {stats['max']:.3f} keV")
    print(f"  Mean: {stats['mean']:.3f} keV")
    print(f"  Std:  {stats['std']:.3f} keV")

    # Slice a time window
    sliced = signal.slice(1.0, 3.0)
    print(f"\n  Sliced (1-3s): {sliced}")

    # Resample
    downsampled = signal.resample(500)
    print(f"  Resampled to 500 pts: {downsampled}")

    # DataFrame
    df = signal.to_dataframe()
    print(f"\n  DataFrame:")
    print(df.head(5).to_string())

    # Plot
    ax = signal.plot()
    ax.get_figure().savefig("synthetic_signal.png", dpi=150, bbox_inches="tight")
    print("\n  → Saved: synthetic_signal.png")

    # Multi-panel overview
    ne = Signal(
        name="Electron Density",
        signal_path="interferometer/ne",
        values=2e19 * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))),
        timestamps=t,
        units="m⁻³",
        source="Synthetic",
        experiment_id="DEMO-001",
    )
    P_rad = Signal(
        name="Radiated Power",
        signal_path="bolometer/P_rad",
        values=0.5e6 * (1 + 0.5 * np.abs(np.sin(2 * np.pi * 0.7 * t))) + 1e5 * np.random.randn(len(t)),
        timestamps=t,
        units="W",
        source="Synthetic",
        experiment_id="DEMO-001",
    )

    fig = plot_experiment_overview([signal, ne, P_rad])
    fig.savefig("experiment_overview.png", dpi=150, bbox_inches="tight")
    print("  → Saved: experiment_overview.png")


def demo_sources_list():
    """Show all available data sources."""
    from fusiondata import list_sources
    print()
    list_sources()


def demo_w7x():
    """W7-X demo (requires internet)."""
    from fusiondata import W7X

    print("\n" + "=" * 60)
    print("  W7-X Demo (requires internet)")
    print("=" * 60)

    w7x = W7X()

    if not w7x.is_available:
        print("  ⚠ W7-X archive is not reachable. Skipping.")
        return

    # Browse top-level streams
    print("\n  Top-level data streams:")
    streams = w7x.list_streams()
    for s in streams[:10]:
        print(f"    {s['name']:30s} [{s['type']}]")

    # List programs for a date
    try:
        programs = w7x.list_programs("2018-08-21")
        print(f"\n  Programs on 2018-08-21: {len(programs)}")
        for p in programs[:5]:
            print(f"    {p.id}: {p.description[:60] if p.description else '(no description)'}")
    except Exception as e:
        print(f"  Could not list programs: {e}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        fusiondata — Quickstart Demo                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # These always work (offline):
    demo_sources_list()
    dt = demo_iaea()
    demo_signal_features()

    # These need internet:
    print("\n" + "─" * 60)
    print("  Online demos (skipped if no connection):")
    print("─" * 60)
    try:
        demo_w7x()
    except Exception as e:
        print(f"  W7-X demo failed: {e}")

    print("\n✅ Done! Check the generated .png files.")
