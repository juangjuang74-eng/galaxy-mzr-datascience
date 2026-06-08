"""
src/data_utils.py
-----------------
Synthetic galaxy catalog generator that mirrors GIMIC/SDSS statistics
from De Rossi et al. (2015). Replace generate_* functions with real
data loaders when connecting to SDSS MPA-JHU or EAGLE catalogs.
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng


# ── Physical constants / calibration offsets ──────────────────────────────────
SOLAR_Z = 0.0127          # Wiersma et al. 2009
OH_SOLAR = 8.69           # 12 + log(O/H)_sun

ELEMENTS = ["O", "Mg", "Si", "Fe", "N", "C"]

# Approximate GIMIC MZR parameters (power-law fit to Fig 1)
MZR_SLOPE  = 0.30          # dex per dex in log M★
MZR_ZERO   = 8.40          # 12+log(O/H) at log M★ = 9
MZR_SCATTER = 0.18         # intrinsic scatter (dex)


def generate_central_galaxies(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate a mock catalog of central galaxies matching GIMIC statistics
    at z = 0 (De Rossi et al. 2015, Section 2.2 and Fig. 1).

    Returns
    -------
    pd.DataFrame with columns:
        log_Mstar, log_SFR, OH, Z_total, fg, Vcirc, R50,
        log_Mhalo, redshift, is_satellite
    """
    rng = default_rng(seed)

    log_Mstar = rng.uniform(9.0, 10.5, n)                 # log(M★/M☉)
    log_SFR   = -0.9 + 0.8 * (log_Mstar - 9.5) + rng.normal(0, 0.35, n)

    # MZR with SFR secondary dependence (Fig. 1 right panel)
    OH_mean = MZR_ZERO + MZR_SLOPE * (log_Mstar - 9.0) - 0.08 * log_SFR
    OH       = OH_mean + rng.normal(0, MZR_SCATTER, n)
    OH       = np.clip(OH, 7.8, 9.2)

    # Gas fraction: anti-correlated with M★ (Fig. 3)
    fg_mean = 0.55 - 0.12 * (log_Mstar - 9.0)
    fg      = np.clip(fg_mean + rng.normal(0, 0.08, n), 0.01, 0.95)

    # Circular velocity (Tully-Fisher proxy)
    log_Vcirc = 1.6 + 0.32 * (log_Mstar - 9.5) + rng.normal(0, 0.07, n)
    Vcirc     = 10 ** log_Vcirc

    # Half-mass radius (kpc)
    R50 = 2.5 * 10 ** (0.28 * (log_Mstar - 9.5)) * rng.lognormal(0, 0.20, n)

    # Halo mass (abundance matching approximation)
    log_Mhalo = log_Mstar + 1.6 + rng.normal(0, 0.15, n)

    # Total metallicity (Z/Z☉), correlated with O/H
    Z_total = 10 ** (OH - OH_SOLAR - 0.05 + rng.normal(0, 0.07, n))

    df = pd.DataFrame({
        "log_Mstar":  log_Mstar,
        "log_SFR":    log_SFR,
        "OH":         OH,           # 12 + log(O/H)
        "Z_total":    Z_total,      # Z / Z☉
        "fg":         fg,
        "Vcirc":      Vcirc,
        "R50":        R50,
        "log_Mhalo":  log_Mhalo,
        "redshift":   np.zeros(n),
        "is_satellite": False,
    })
    return df


def generate_satellite_galaxies(n: int = 300, seed: int = 99) -> pd.DataFrame:
    """
    Generate mock satellite galaxies.  Satellites are offset +0.15 dex in O/H,
    have lower fg (gas stripped by ram pressure), lower Vcirc at fixed M★,
    and scatter ~ 2× larger than centrals (Fig. 10–11).
    """
    rng = default_rng(seed)

    log_Mstar = rng.uniform(9.0, 10.5, n)
    log_SFR   = -0.9 + 0.8 * (log_Mstar - 9.5) + rng.normal(0, 0.35, n)

    # Metallicity offset +0.15 dex, scatter doubled
    OH_mean = MZR_ZERO + MZR_SLOPE * (log_Mstar - 9.0) - 0.08 * log_SFR + 0.15
    OH      = OH_mean + rng.normal(0, MZR_SCATTER * 2.0, n)
    OH      = np.clip(OH, 7.8, 9.2)

    # Gas-stripped: fg offset −0.2 relative to centrals
    fg_mean = 0.35 - 0.12 * (log_Mstar - 9.0)
    fg      = np.clip(fg_mean + rng.normal(0, 0.12, n), 0.01, 0.95)

    # Slightly shallower potential at fixed M★ (Section 5)
    log_Vcirc = 1.55 + 0.32 * (log_Mstar - 9.5) + rng.normal(0, 0.09, n)
    Vcirc     = 10 ** log_Vcirc

    R50       = 2.2 * 10 ** (0.28 * (log_Mstar - 9.5)) * rng.lognormal(0, 0.25, n)
    log_Mhalo = log_Mstar + 2.6 + rng.normal(0, 0.20, n)   # ~10× more massive host

    Z_total   = 10 ** (OH - OH_SOLAR - 0.05 + rng.normal(0, 0.09, n))

    # Extra halo properties for gap-3 analysis
    infall_time  = rng.uniform(0.5, 8.0, n)        # Gyr since satellite infall
    ram_pressure = rng.lognormal(0, 0.8, n)        # proxy (arbitrary units)

    df = pd.DataFrame({
        "log_Mstar":    log_Mstar,
        "log_SFR":      log_SFR,
        "OH":           OH,
        "Z_total":      Z_total,
        "fg":           fg,
        "Vcirc":        Vcirc,
        "R50":          R50,
        "log_Mhalo":    log_Mhalo,
        "redshift":     np.zeros(n),
        "is_satellite": True,
        "infall_time":  infall_time,
        "ram_pressure": ram_pressure,
    })
    return df


def generate_redshift_snapshots(
    z_values: list = [0.0, 0.5, 1.0, 2.0, 3.0],
    n_per_snap: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate galaxy snapshots at multiple redshifts.
    O/H for α-elements barely evolves; Fe/H evolves strongly below z=2
    (Fig. 5, bottom-right panel).
    """
    rng   = default_rng(seed)
    snaps = []

    for z in z_values:
        log_Mstar = rng.uniform(9.0, 10.5, n_per_snap)
        log_SFR   = -0.9 + 0.8 * (log_Mstar - 9.5) + rng.normal(0, 0.35, n_per_snap)

        # O/H: negligible evolution (centrals only)
        OH = MZR_ZERO + MZR_SLOPE * (log_Mstar - 9.0) + rng.normal(0, MZR_SCATTER, n_per_snap)

        # Mg/H: same as O (both SNII α-elements)
        MgH = OH - 0.10 + rng.normal(0, 0.05, n_per_snap)

        # Si/H: flattens toward z=0 (metallicity-dependent yield)
        SiH = OH - 0.40 + 0.05 * max(0, 2 - z) + rng.normal(0, 0.06, n_per_snap)

        # Fe/H: strong evolution below z=2 (SNIa delay ~1 Gyr)
        fe_evol = 0.3 * max(0, 2.0 - z) / 2.0
        FeH = OH - 0.90 + fe_evol + rng.normal(0, 0.10, n_per_snap)

        # N/H: AGB dominated, evolves z=3→1
        n_evol = 0.25 * max(0, 3.0 - z) / 3.0
        NH = OH - 1.20 + n_evol + rng.normal(0, 0.08, n_per_snap)

        # C/H: AGB + SNII mix
        CH = OH - 1.05 + 0.15 * max(0, 3.0 - z) / 3.0 + rng.normal(0, 0.08, n_per_snap)

        fg      = np.clip(0.55 - 0.12 * (log_Mstar - 9.0) + 0.08 * z + rng.normal(0, 0.08, n_per_snap), 0.01, 0.99)
        Vcirc   = 10 ** (1.6 + 0.32 * (log_Mstar - 9.5) + rng.normal(0, 0.07, n_per_snap))

        snap = pd.DataFrame({
            "log_Mstar": log_Mstar,
            "log_SFR":   log_SFR,
            "OH": OH, "MgH": MgH, "SiH": SiH,
            "FeH": FeH, "NH": NH, "CH": CH,
            "fg":     fg,
            "Vcirc":  Vcirc,
            "redshift": z,
        })
        snaps.append(snap)

    return pd.concat(snaps, ignore_index=True)


def mannucci_fmr(log_Mstar: np.ndarray, log_SFR: np.ndarray) -> np.ndarray:
    """
    Fundamental Metallicity Relation from Mannucci et al. (2010) — eq. (5)
    of De Rossi et al. 2015.

    Returns 12 + log(O/H).
    """
    m = log_Mstar - 10.0
    s = log_SFR
    return 8.90 + 0.37*m - 0.14*s - 0.19*m**2 + 0.12*m*s - 0.054*s**2
