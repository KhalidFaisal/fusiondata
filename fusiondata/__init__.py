"""
fusiondata — A unified Python API for fusion reactor data.

Like astropy for astronomy, but for fusion energy research.

Supports:
    - W7-X  (Wendelstein 7-X ArchiveDB)
    - MAST  (FAIR MAST tokamak data)
    - IAEA  (Nuclear reaction cross-sections from EXFOR)
    - LHD   (Large Helical Device open data on AWS)

Quick start:
    >>> from fusiondata import W7X
    >>> w7x = W7X()
    >>> programs = w7x.list_programs("2023-03-15")
"""

__version__ = "0.1.0"
__author__ = "fusiondata contributors"

from fusiondata.sources.w7x import W7X
from fusiondata.sources.mast import MAST
from fusiondata.sources.iaea import IAEA
from fusiondata.sources.lhd import LHD
from fusiondata.sources.d3d import D3D
from fusiondata.sources.jet import JET
from fusiondata.models import Signal, Experiment, CrossSection


def list_sources():
    """List all available fusion data sources."""
    sources = {
        "W7-X": {
            "class": "W7X",
            "description": "Wendelstein 7-X stellarator (IPP Greifswald, Germany)",
            "api": "ArchiveDB REST API",
            "url": "https://www.ipp.mpg.de/w7x",
            "auth_required": False,
        },
        "MAST": {
            "class": "MAST",
            "description": "Mega Ampere Spherical Tokamak (UKAEA, UK)",
            "api": "FAIR MAST JSON API + S3",
            "url": "https://mastapp.site",
            "auth_required": False,
        },
        "IAEA": {
            "class": "IAEA",
            "description": "IAEA Nuclear Reaction Cross-Sections (EXFOR database)",
            "api": "EXFOR REST API",
            "url": "https://nds.iaea.org",
            "auth_required": False,
        },
        "LHD": {
            "class": "LHD",
            "description": "Large Helical Device (NIFS, Japan)",
            "api": "AWS S3 Open Data",
            "url": "https://registry.opendata.aws/lhd-experiment",
            "auth_required": False,
        },
        "DIII-D": {
            "class": "D3D",
            "description": "DIII-D National Fusion Facility (General Atomics, USA)",
            "api": "MDSplus via mdsthin",
            "url": "https://d3dfusion.org/",
            "auth_required": True,
        },
        "JET": {
            "class": "JET",
            "description": "Joint European Torus (EUROfusion, UK)",
            "api": "JET Data Centre REST API",
            "url": "https://data.euro-fusion.org/",
            "auth_required": True,
        },
    }
    print("=" * 70)
    print("  fusiondata — Available Data Sources")
    print("=" * 70)
    for name, info in sources.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    API:   {info['api']}")
        print(f"    Auth:  {'Required' if info['auth_required'] else 'None (public)'}")
        print(f"    Class: fusiondata.{info['class']}")
    print("\n" + "=" * 70)
    return sources
