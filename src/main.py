"""
NRHO Visibility Analysis Tool
==============================
Python port of MATLAB/HALO thesis pipeline.

Author : Flaxon Nazareth
Thesis : GNSS-Assisted CubeSat Orbit Determination in Cis-Lunar Space
         UNSW Sydney, 2025
Supervisors: Prof. Andrew Dempster, Dr. Yang Yang (HALO framework)

Usage
-----
    python main.py

Outputs
-------
    output/coverage_sweep.csv
    figures/nrho_orbit.png
    figures/nrho_radius.png
    figures/coverage_bars.png
    figures/visibility.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from propagator import propagate_nrho
from coverage  import coverage_sweep
from plots     import (plot_nrho_orbit, plot_radius_vs_time,
                       plot_coverage_bars, plot_visibility_timeseries)

# ── Mission configuration ─────────────────────────────────────
# CAPSTONE-like NRHO initial state, Moon-centred inertial [km, km/s]
Y0 = np.array([
    -3200.0,  500.0,  60000.0,
        0.08,   0.30,    0.015
])

P_NRHO    = 5.7444e5     # NRHO period [s] — ~6.64 days
N_PERIODS = 4            # number of periods to propagate
DT_S      = 30.0         # output timestep [s] — matches HALO default
MIN_EL    = 10.0         # elevation mask [deg]


def main():
    print("=" * 55)
    print("  NRHO Visibility Analysis — Flaxon Nazareth, 2025")
    print("=" * 55)

    # 1 — Propagate
    duration = N_PERIODS * P_NRHO
    print(f"\n[1/4] Propagating {duration/86400:.1f} days "
          f"({N_PERIODS} NRHO periods, dt={DT_S}s)...")
    t, X = propagate_nrho(Y0, duration, dt_s=DT_S)
    r    = np.linalg.norm(X[:, :3], axis=1)
    print(f"      Steps: {len(t):,}")
    print(f"      Moon distance: {r.min():.0f} – {r.max():.0f} km "
          f"(mean {r.mean():.0f} km)")

    # 2 — Coverage sweep
    print(f"\n[2/4] Running coverage sweep "
          f"(3 sites × 3 masks × {len(t):,} steps)...")
    df = coverage_sweep(X, verbose=True)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/coverage_sweep.csv", index=False)
    print(f"      Saved: output/coverage_sweep.csv")

    # 3 — Orbit figures
    print("\n[3/4] Generating orbit figures...")
    os.makedirs("figures", exist_ok=True)
    plot_nrho_orbit(t, X)
    plot_radius_vs_time(t, X)

    # 4 — Visibility figures
    print(f"\n[4/4] Generating visibility figures (minEl={MIN_EL} deg)...")
    print("      This takes a few minutes...")
    plot_coverage_bars(df)
    plot_visibility_timeseries(t, X, min_el_deg=MIN_EL)

    print("\n" + "=" * 55)
    print("  Complete. Outputs:")
    print("    output/coverage_sweep.csv")
    print("    figures/nrho_orbit.png")
    print("    figures/nrho_radius.png")
    print("    figures/coverage_bars.png")
    print("    figures/visibility.png")
    print("=" * 55)


if __name__ == "__main__":
    main()