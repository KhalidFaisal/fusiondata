# 🔥 fusiondata

**A unified Python API for fusion reactor data — astropy for fusion energy.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Fusion science produces incredible data — plasma temperatures of 100 million degrees, magnetic confinement diagnostics, neutron yields — but accessing it is a nightmare. Data is scattered across incompatible archives with confusing formats and poor documentation.

**fusiondata** fixes that. One `pip install`, one import, and you have access to the world's public fusion data.

## 🚀 Quick Start

```bash
pip install -e .
```

```python
from fusiondata import IAEA

# Get the D-T fusion cross-section (works offline!)
iaea = IAEA()
dt = iaea.get_cross_section("DT")
dt.plot()
print(f"Peak: {dt.peak_cross_section:.4g} mb at {dt.peak_energy:.1f} keV")
```

## 📡 Supported Sources

| Source | Device | Location | Data |
|--------|--------|----------|------|
| `W7X` | Wendelstein 7-X | Greifswald, DE | Stellarator diagnostics, plasma signals |
| `MAST` | MAST / MAST-U | Culham, UK | Spherical tokamak shots & diagnostics |
| `IAEA` | EXFOR Database | Vienna, AT | Nuclear reaction cross-sections |
| `LHD` | Large Helical Device | Toki, JP | Heliotron diagnostics (AWS S3) |
| `D3D` | DIII-D | San Diego, US | MDSplus tokamak data *(Auth required)* |
| `JET` | Joint European Torus | Culham, UK | JPF/PPF tokamak data *(Auth required)* |

## 📖 Usage Examples

### W7-X — Stellarator Data

```python
from fusiondata import W7X

w7x = W7X()

# Browse the data hierarchy
streams = w7x.list_streams()

# List experiment programs for a date
programs = w7x.list_programs("2023-03-15")
for p in programs:
    print(p.id, p.description)

# Get a signal and plot it
signal = w7x.get_signal("ArchiveDB/codac/W7X/.../Scaled", program_id="20230315.003")
signal.plot()                    # Fusion-themed matplotlib plot
df = signal.to_dataframe()      # → pandas DataFrame
print(signal.stats)             # min, max, mean, std, samples
```

### MAST — Tokamak Shots

```python
from fusiondata import MAST

mast = MAST()

# Browse shots
shots = mast.list_shots(shot_min=30000, shot_max=30100)
shot = mast.get_shot(30420)
print(shot.summary)

# Get diagnostic signals
ne = mast.get_signal(30420, "electron_density")
ne.plot()

# Direct S3/Zarr access (requires: pip install fusiondata[full])
ne = mast.get_signal(30420, "electron_density", source="s3")
```

### IAEA — Nuclear Cross-Sections

```python
from fusiondata import IAEA

iaea = IAEA()

# D-T fusion (the big one)
dt = iaea.get_cross_section("DT")
dt.plot()
print(f"Peak cross-section: {dt.peak_cross_section:.4g} mb")
print(f"At energy: {dt.peak_energy:.1f} keV")

# Get cross-section at a specific energy
sigma_100 = dt.at_energy(100)   # σ at 100 keV

# Compare multiple reactions
iaea.plot_comparison(["DT", "DD-p", "DD-n", "D3He"])

# All available reactions
reactions = iaea.list_reactions()
```

### LHD — Helical Device

```python
from fusiondata import LHD

lhd = LHD()

# List available diagnostics
diags = lhd.list_diagnostics(shot_id=160000)

# Get signals
wp = lhd.get_signal(160000, "wp")
wp.plot()
```

### DIII-D — MDSplus Access (Restricted)

Requires an active VPN or SSH connection to the DIII-D network (`atlas.gat.com`).

```python
from fusiondata import D3D

# Will raise AuthError if you are not authorized/connected
d3d = D3D() 

# Get signal from 'efit01' tree
ip = d3d.get_signal(165920, "ip")
ip.plot()
```

### JET — JDC Authentication (Restricted)

Requires a validated EUROfusion JWT token.

```python
from fusiondata import JET

# Must provide token
jet = JET(token="eyJhbGciOiJIUz...") 

# Get Processed Pulse File (PPF)
ne = jet.get_signal(99971, "LIDR", "NE")
ne.plot()
```

## 🛠️ Signal Object

Every signal comes back as a rich `Signal` object:

```python
signal.values            # numpy array of measurements
signal.timestamps        # numpy array of times
signal.units             # physical units ("eV", "m^-3", etc.)
signal.to_dataframe()    # → pandas DataFrame
signal.plot()            # instant matplotlib plot
signal.stats             # {"min": ..., "max": ..., "mean": ..., "std": ..., "samples": ...}
signal.slice(t0, t1)     # time-slice
signal.resample(1000)    # downsample to N points
signal.duration          # time span
signal.sampling_rate     # Hz
```

## 🧪 CrossSection Object

```python
cs.energies              # energy array (keV)
cs.values                # σ array (mb or b)
cs.peak_cross_section    # max σ
cs.peak_energy           # energy at peak
cs.at_energy(100)        # interpolate σ at 100 keV
cs.to_dataframe()        # → pandas DataFrame
cs.plot()                # log-log plot with peak annotation
```

## 📦 Installation

```bash
# Core (requests, numpy, pandas, matplotlib)
pip install -e .

# Full (adds xarray, s3fs, zarr for direct S3/Zarr access)
pip install -e ".[full]"

# Development
pip install -e ".[dev]"
```

## 🎨 Plotting

All `.plot()` calls use a custom **fusion-themed dark style** with:
- Deep navy background
- Plasma-colored signal traces
- Auto-scaled axes with physical units
- Peak annotations for cross-sections
- Multi-panel experiment overviews

```python
from fusiondata.plotting import plot_experiment_overview

# Multi-panel overview
signals = [signal1, signal2, signal3]
fig = plot_experiment_overview(signals, experiment=shot)
```

## 💾 Caching

Responses are cached locally to avoid repeated API calls:

```python
w7x = W7X(cache_enabled=True, cache_ttl=3600)  # 1-hour cache

# Clear cache
w7x.clear_cache()
```

## 🌡️ Utilities

```python
from fusiondata.utils import ev_to_kelvin, parse_reaction_string

# Unit conversions
ev_to_kelvin(1.0)           # → 11604.5 K
kev_to_ev(10.0)             # → 10000.0 eV

# Parse reaction strings
parse_reaction_string("D(T,n)4He")
# → {"target": "D", "projectile": "T", "ejectile": "n", "residual": "4He"}
```

## 🤖 Automated Publishing

This repository is configured with **PyPI Trusted Publishing** via GitHub Actions.

When you are ready to publish a new release:
1. Update the version inside `pyproject.toml` and `fusiondata/__init__.py`.
2. Commit and tag the release:
   ```bash
   git add .
   git commit -m "Bump version to v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```
3. The GitHub Action `.github/workflows/publish.yml` will automatically build the wheel and upload it to PyPI securely using an OIDC token matching the tag.

*Note: You must configure PyPI to trust your GitHub repository first. Go to PyPI → Account Settings → Publishing → Add a new publisher → GitHub. Set the Environment name to `pypi`.*

## 🤝 Contributing

This is an open-source project. Contributions welcome:
- Add new data sources (DIII-D, EAST, ITER, ...)
- Improve data format parsers
- Add more built-in cross-section parametrizations
- Write tutorials and examples

## 📄 License

MIT License — use it, fork it, build on it.

---

*Because fusion data should be as easy to access as star catalogs.*
