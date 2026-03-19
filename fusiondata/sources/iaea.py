"""
IAEA Nuclear Data source client.

Provides access to fusion-relevant nuclear reaction cross-sections
from the IAEA EXFOR database and related nuclear data services.

Primary API: IAEA Nuclear Reactions REST API
Backup: Built-in fundamental fusion cross-section parametrizations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from fusiondata.client import BaseClient, DataNotFoundError
from fusiondata.models import CrossSection
from fusiondata.utils import FUSION_REACTIONS

logger = logging.getLogger("fusiondata")


# ═══════════════════════════════════════════════════════════════════
#  Built-in parametrized cross-sections (Bosch & Hale 1992)
#  These are analytic fits — always available offline.
# ═══════════════════════════════════════════════════════════════════

def _bosch_hale_dt(E_keV: np.ndarray) -> np.ndarray:
    """D(T,n)4He cross-section parametrization (Bosch & Hale 1992).

    Valid for 0.5 keV ≤ E_cm ≤ 550 keV.

    Args:
        E_keV: Center-of-mass energy in keV.

    Returns:
        Cross-section in millibarns.
    """
    # Astrophysical S-factor coefficients
    A1 = 6.927e4
    A2 = 7.454e8
    A3 = 2.050e6
    A4 = 5.2002e4
    A5 = 0.0
    B_G = 34.3827  # Gamow constant (keV^0.5)

    E = np.asarray(E_keV, dtype=np.float64)
    E = np.maximum(E, 1e-10)  # avoid div by zero

    # S-factor (keV·barn)
    S = (A1 + E * (A2 + E * (A3 + E * (A4 + E * A5)))) / \
        (1.0 + E * (7.6822e3 + E * (1.3818e2 + E * (-5.7501e-1 + E * 9.3619e-4))))

    # Cross-section (millibarns)
    sigma = S / (E * np.exp(B_G / np.sqrt(E))) * 1e-3

    return sigma


def _bosch_hale_dd_p(E_keV: np.ndarray) -> np.ndarray:
    """D(D,p)T cross-section parametrization (Bosch & Hale 1992).

    Args:
        E_keV: Center-of-mass energy in keV.

    Returns:
        Cross-section in millibarns.
    """
    A1 = 5.5576e4
    A2 = 2.1054e2
    A3 = -3.2638e-2
    A4 = 1.4987e-6
    A5 = 1.8181e-10
    B_G = 31.3970

    E = np.asarray(E_keV, dtype=np.float64)
    E = np.maximum(E, 1e-10)

    S = (A1 + E * (A2 + E * (A3 + E * (A4 + E * A5)))) / \
        (1.0 + E * (0.0 + E * (0.0 + E * 0.0)))

    sigma = S / (E * np.exp(B_G / np.sqrt(E))) * 1e-3
    return sigma


def _bosch_hale_dd_n(E_keV: np.ndarray) -> np.ndarray:
    """D(D,n)3He cross-section parametrization (Bosch & Hale 1992).

    Args:
        E_keV: Center-of-mass energy in keV.

    Returns:
        Cross-section in millibarns.
    """
    A1 = 5.3701e4
    A2 = 3.3027e2
    A3 = -1.2706e-1
    A4 = 2.9327e-5
    A5 = -2.5151e-9
    B_G = 31.3970

    E = np.asarray(E_keV, dtype=np.float64)
    E = np.maximum(E, 1e-10)

    S = (A1 + E * (A2 + E * (A3 + E * (A4 + E * A5)))) / \
        (1.0 + E * (0.0 + E * (0.0 + E * 0.0)))

    sigma = S / (E * np.exp(B_G / np.sqrt(E))) * 1e-3
    return sigma


def _bosch_hale_d3he(E_keV: np.ndarray) -> np.ndarray:
    """D(3He,p)4He cross-section parametrization (Bosch & Hale 1992).

    Args:
        E_keV: Center-of-mass energy in keV.

    Returns:
        Cross-section in millibarns.
    """
    A1 = 5.7501e6
    A2 = 2.5226e3
    A3 = 4.5566e1
    A4 = 0.0
    A5 = 0.0
    B_G = 68.7508

    E = np.asarray(E_keV, dtype=np.float64)
    E = np.maximum(E, 1e-10)

    S = (A1 + E * (A2 + E * (A3 + E * (A4 + E * A5)))) / \
        (1.0 + E * (-3.1995e-3 + E * (-8.5530e-6 + E * 5.9014e-8)))

    sigma = S / (E * np.exp(B_G / np.sqrt(E))) * 1e-3
    return sigma


BUILTIN_REACTIONS = {
    "D(T,n)4He": _bosch_hale_dt,
    "DT": _bosch_hale_dt,
    "D+T→n+⁴He": _bosch_hale_dt,
    "D+T->n+4He": _bosch_hale_dt,
    "D(D,p)T": _bosch_hale_dd_p,
    "DD-p": _bosch_hale_dd_p,
    "D(D,n)3He": _bosch_hale_dd_n,
    "DD-n": _bosch_hale_dd_n,
    "D(3He,p)4He": _bosch_hale_d3he,
    "D3He": _bosch_hale_d3he,
}


class IAEA(BaseClient):
    """Client for IAEA nuclear reaction cross-section data.

    Provides access to fusion-relevant cross-sections from:
    1. **Built-in parametrizations** (Bosch & Hale 1992) — always available offline
    2. **IAEA EXFOR database** — experimental data via REST API

    The built-in data covers the most important fusion reactions:
    - D-T (deuterium-tritium)
    - D-D (both branches: proton + neutron)
    - D-³He

    Example:
        >>> from fusiondata import IAEA
        >>> iaea = IAEA()
        >>> dt = iaea.get_cross_section("DT")
        >>> dt.plot()
        >>> print(f"Peak: {dt.peak_cross_section:.4g} mb at {dt.peak_energy:.1f} keV")
    """

    SOURCE_NAME = "IAEA"
    BASE_URL = "https://nds.iaea.org/exfor/servlet/X4sGetSubent"

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url or self.BASE_URL, **kwargs)

    def get_cross_section(
        self,
        reaction: str,
        energy_range: Optional[tuple] = None,
        n_points: int = 500,
        prefer_builtin: bool = True,
    ) -> CrossSection:
        """Get cross-section data for a fusion reaction.

        Args:
            reaction: Reaction identifier. Accepts multiple formats:
                      "DT", "D(T,n)4He", "D+T→n+⁴He", "DD-p", "D3He", etc.
            energy_range: (E_min, E_max) in keV. Default: (1, 1000).
            n_points: Number of energy points for parametrized data.
            prefer_builtin: If True, use built-in Bosch & Hale parametrizations
                           (recommended — fast, smooth, offline). If False,
                           try the EXFOR database first.

        Returns:
            CrossSection object.

        Example:
            >>> iaea = IAEA()
            >>> dt = iaea.get_cross_section("DT")
            >>> print(dt)
            CrossSection('D(T,n)4He', 500 points, peak=5.014 b at 64.0 keV)
            >>> dt.to_dataframe().head()
        """
        # Resolve aliases
        canonical = FUSION_REACTIONS.get(reaction, reaction)
        reaction_key = reaction

        # Try built-in first
        if prefer_builtin and reaction_key in BUILTIN_REACTIONS:
            return self._builtin_cross_section(reaction_key, canonical, energy_range, n_points)

        # Also try canonical name
        if prefer_builtin and canonical in BUILTIN_REACTIONS:
            return self._builtin_cross_section(canonical, canonical, energy_range, n_points)

        # Try EXFOR API
        try:
            return self._exfor_cross_section(canonical, energy_range)
        except Exception as e:
            logger.warning(f"EXFOR lookup failed for '{reaction}': {e}")

        # Ultimate fallback: check if we have a builtin
        for key in [reaction_key, canonical]:
            if key in BUILTIN_REACTIONS:
                return self._builtin_cross_section(key, canonical, energy_range, n_points)

        raise DataNotFoundError(
            f"No cross-section data found for '{reaction}'.\n"
            f"Available built-in reactions: {list(set(FUSION_REACTIONS.values()))}\n"
            f"Aliases: {list(FUSION_REACTIONS.keys())}"
        )

    def _builtin_cross_section(
        self,
        key: str,
        canonical: str,
        energy_range: Optional[tuple],
        n_points: int,
    ) -> CrossSection:
        """Generate cross-section from built-in parametrization."""
        func = BUILTIN_REACTIONS[key]

        if energy_range is None:
            energy_range = (1.0, 1000.0)

        energies = np.logspace(
            np.log10(energy_range[0]),
            np.log10(energy_range[1]),
            n_points,
        )
        values = func(energies)

        return CrossSection(
            reaction=canonical,
            energies=energies,
            values=values,
            energy_units="keV",
            cs_units="mb",
            source="Bosch & Hale (1992)",
            reference="H.-S. Bosch and G.M. Hale, Nucl. Fusion 32 (1992) 611",
            metadata={"parametrization": "Bosch-Hale", "type": "analytic"},
        )

    def _exfor_cross_section(
        self,
        reaction: str,
        energy_range: Optional[tuple],
    ) -> CrossSection:
        """Fetch cross-section from IAEA EXFOR database."""
        from fusiondata.utils import parse_reaction_string

        parsed = parse_reaction_string(reaction)
        target = parsed["target"]
        projectile = parsed["projectile"]

        params = {
            "target": target,
            "projectile": projectile,
            "quantity": "SIG",
            "format": "json",
        }

        data = self._get("", params=params)

        if not data or (isinstance(data, dict) and not data.get("datasets")):
            raise DataNotFoundError(f"No EXFOR data found for {reaction}")

        # Parse EXFOR JSON response
        datasets = data.get("datasets", [data]) if isinstance(data, dict) else data
        energies_all = []
        values_all = []

        for ds in datasets:
            if isinstance(ds, dict):
                e = ds.get("energy", ds.get("EN", []))
                v = ds.get("data", ds.get("DATA", ds.get("cross_section", [])))
                energies_all.extend(e)
                values_all.extend(v)

        if not energies_all:
            raise DataNotFoundError(f"EXFOR returned empty data for {reaction}")

        energies = np.array(energies_all, dtype=np.float64)
        values = np.array(values_all, dtype=np.float64)

        # Sort by energy
        sort_idx = np.argsort(energies)
        energies = energies[sort_idx]
        values = values[sort_idx]

        return CrossSection(
            reaction=reaction,
            energies=energies,
            values=values,
            energy_units="keV",
            cs_units="b",
            source="IAEA/EXFOR",
            metadata={"source": "EXFOR", "datasets": len(datasets)},
        )

    def list_reactions(self, category: str = "all") -> List[Dict[str, str]]:
        """List available fusion reactions.

        Args:
            category: Filter — "builtin" for offline parametrizations,
                      "all" for everything including EXFOR queries.

        Returns:
            List of dicts with reaction info.
        """
        results = []

        # Built-in reactions
        seen = set()
        for alias, canonical in FUSION_REACTIONS.items():
            if canonical not in seen:
                seen.add(canonical)
                results.append({
                    "reaction": canonical,
                    "aliases": [k for k, v in FUSION_REACTIONS.items() if v == canonical and k != canonical],
                    "source": "Bosch & Hale (1992)",
                    "type": "builtin",
                    "offline": True,
                })

        if category == "builtin":
            return results

        # Note: EXFOR has thousands of reactions.
        # We list only the most fusion-relevant ones.
        exfor_reactions = [
            {"reaction": "T(T,2n)4He", "aliases": ["TT"], "source": "EXFOR", "type": "experimental", "offline": False},
            {"reaction": "p(11B,3α)", "aliases": ["pB11"], "source": "EXFOR", "type": "experimental", "offline": False},
            {"reaction": "p(7Li,α)4He", "aliases": ["p-Li7"], "source": "EXFOR", "type": "experimental", "offline": False},
        ]
        results.extend(exfor_reactions)

        return results

    def compare(self, reactions: List[str], energy_range: Optional[tuple] = None, n_points: int = 500):
        """Compare cross-sections for multiple reactions.

        Args:
            reactions: List of reaction names/aliases.
            energy_range: (E_min, E_max) in keV.
            n_points: Number of energy points.

        Returns:
            Dict mapping reaction name to CrossSection object.
        """
        result = {}
        for rxn in reactions:
            try:
                cs = self.get_cross_section(rxn, energy_range=energy_range, n_points=n_points)
                result[cs.reaction] = cs
            except DataNotFoundError as e:
                logger.warning(f"Skipping {rxn}: {e}")
        return result

    def plot_comparison(self, reactions: List[str], **kwargs):
        """Plot multiple cross-sections on the same axes.

        Args:
            reactions: List of reaction names.
            **kwargs: Passed to matplotlib.

        Returns:
            matplotlib Axes.
        """
        import matplotlib.pyplot as plt
        from fusiondata.plotting import apply_fusion_style, FUSION_PALETTE

        cross_sections = self.compare(reactions)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, (name, cs) in enumerate(cross_sections.items()):
            color = FUSION_PALETTE[i % len(FUSION_PALETTE)]
            ax.plot(cs.energies, cs.values, color=color, linewidth=1.5, label=name)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Cross Section [mb]")
        ax.set_title("Fusion Cross-Section Comparison", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#0A0E17", edgecolor="#1A2332", labelcolor="#E0E0E0")
        apply_fusion_style(ax, fig)
        fig.tight_layout()
        return ax
