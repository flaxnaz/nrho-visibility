"""
coverage.py
-----------
Coverage sweep for lunar surface sites over an NRHO trajectory.
Port of MATLAB coverage_sweep_NRHO.m (Flaxon Nazareth, UNSW 2025)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from visibility import los_visibility_link, LinkBudget


@dataclass
class SurfaceSite:
    """Lunar surface ground site."""
    name:    str
    lat_deg: float
    lon_deg: float


# Default sites from your thesis
DEFAULT_SITES = [
    SurfaceSite("Equator_23E",  0.0,   23.0),
    SurfaceSite("SouthPole",   -89.9,   0.0),
    SurfaceSite("NorthPole",    89.9,   0.0),
]

DEFAULT_MASKS = [0, 5, 10]    # elevation masks [deg]


def coverage_sweep(X:          np.ndarray,
                   sites:      List[SurfaceSite] = None,
                   masks_deg:  List[float]        = None,
                   r_moon:     float              = 1737.4,
                   budget:     LinkBudget         = None,
                   verbose:    bool               = True
                   ) -> pd.DataFrame:
    """
    Compute LOS, TX, and SNR coverage for multiple sites and masks.
    Port of MATLAB coverage_sweep_NRHO.m

    Parameters
    ----------
    X : np.ndarray, shape (N, 6)
        State history [km, km/s] — positions are X[:, :3]
    sites : list of SurfaceSite
        Surface sites to evaluate (default: Equator, SouthPole, NorthPole)
    masks_deg : list of float
        Elevation masks in degrees (default: [0, 5, 10])
    r_moon : float
        Lunar radius [km]
    budget : LinkBudget or None
        Link budget for SNR gating
    verbose : bool
        Print progress

    Returns
    -------
    df : pd.DataFrame
        Coverage results with columns:
        Site, MinEl_deg, LOS_pct, TX_pct, SNR_pct
    """
    if sites is None:
        sites = DEFAULT_SITES
    if masks_deg is None:
        masks_deg = DEFAULT_MASKS
    if budget is None:
        budget = LinkBudget()

    positions = X[:, :3]     # shape (N, 3)
    N         = len(positions)
    rows      = []

    for mask in masks_deg:
        if verbose:
            print(f"\nElevation mask: {mask} deg")

        for site in sites:
            vis_los = np.zeros(N, dtype=bool)
            vis_tx  = np.zeros(N, dtype=bool)
            vis_snr = np.zeros(N, dtype=bool)

            for k in range(N):
                vl, vt, vs, _ = los_visibility_link(
                    r_sc       = positions[k],
                    lat_deg    = site.lat_deg,
                    lon_deg    = site.lon_deg,
                    min_el_deg = mask,
                    r_moon     = r_moon,
                    budget     = budget
                )
                vis_los[k] = vl
                vis_tx[k]  = vt
                vis_snr[k] = vs

            cov_los = 100.0 * vis_los.mean()
            cov_tx  = 100.0 * vis_tx.mean()
            cov_snr = 100.0 * vis_snr.mean()

            if verbose:
                print(f"  {site.name:15s}  "
                      f"LOS: {cov_los:6.2f}%  "
                      f"TX: {cov_tx:6.2f}%  "
                      f"SNR: {cov_snr:6.2f}%")

            rows.append({
                "Site":      site.name,
                "MinEl_deg": mask,
                "LOS_pct":   round(cov_los, 3),
                "TX_pct":    round(cov_tx,  3),
                "SNR_pct":   round(cov_snr, 3),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    from propagator import propagate_nrho, MU_MOON

    # CAPSTONE-like NRHO initial state
    y0 = np.array([
        -3200.0,  500.0,  60000.0,
            0.08,   0.30,    0.015
    ])

    P_NRHO   = 5.7444e5
    duration = 4 * P_NRHO

    print("coverage.py — self test")
    print("=" * 50)
    print(f"Propagating {duration/86400:.1f} days...")
    t, X = propagate_nrho(y0, duration, dt_s=30.0)
    print(f"Done. {len(t)} steps.\n")

    print("Running coverage sweep...")
    df = coverage_sweep(X, verbose=True)

    # Save to CSV
    out = "output/coverage_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\nFull results:")
    print(df.to_string(index=False))