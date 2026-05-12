"""
visibility.py
-------------
Line-of-sight, TX-gated, and SNR-gated visibility computation.
Port of MATLAB los_visibility_link.m (Flaxon Nazareth, UNSW 2025)
"""

import numpy as np
from dataclasses import dataclass, field
from moon_geometry import moon_fixed_to_inertial, site_elevation_angle


@dataclass
class LinkBudget:
    """
    S-band link budget parameters.
    Mirrors default_link() from MATLAB coverage_sweep_NRHO.m
    """
    freq_hz:        float = 2.2e9       # S-band carrier [Hz]
    ptx_dbw:        float = 10.0        # TX power [dBW] — 10W
    gtx_dbi:        float = 5.0         # TX antenna gain [dBi]
    grx_dbi:        float = 45.0        # RX dish gain [dBi]
    tsys_k:         float = 500.0       # system noise temperature [K]
    rb_bps:         float = 1000.0      # data rate [bps]
    ebno_req_db:    float = 9.0         # required Eb/N0 [dB]
    imp_margin_db:  float = 0.0         # implementation margin [dB]
    misc_loss_db:   float = 2.0         # miscellaneous losses [dB]


def link_margin_db(range_km: float, budget: LinkBudget) -> float:
    """
    Compute S-band link margin in dB.

    Parameters
    ----------
    range_km : float
        Slant range from site to spacecraft [km]
    budget : LinkBudget
        Link budget parameters

    Returns
    -------
    margin : float
        Link margin in dB. Positive = link closes.
    """
    f_ghz   = budget.freq_hz / 1e9
    lfs_db  = 92.45 + 20 * np.log10(f_ghz) + 20 * np.log10(range_km)

    c_dbw   = (budget.ptx_dbw + budget.gtx_dbi
               + budget.grx_dbi - lfs_db - budget.misc_loss_db)

    k_db    = -228.6                              # Boltzmann [dBW/Hz/K]
    n0_dbw  = k_db + 10 * np.log10(budget.tsys_k)

    cno     = c_dbw - n0_dbw
    ebno_db = cno - 10 * np.log10(budget.rb_bps)
    margin  = ebno_db - budget.ebno_req_db - budget.imp_margin_db

    return margin


def los_visibility_link(r_sc:      np.ndarray,
                         lat_deg:   float,
                         lon_deg:   float,
                         min_el_deg: float = 10.0,
                         r_moon:    float = 1737.4,
                         budget:    LinkBudget = None
                         ) -> tuple[bool, bool, bool, float]:
    """
    Compute LOS, TX-gated, and SNR-gated visibility.
    Port of MATLAB los_visibility_link.m

    Parameters
    ----------
    r_sc : np.ndarray, shape (3,)
        Spacecraft position [km], Moon-centred inertial
    lat_deg : float
        Site latitude [deg]
    lon_deg : float
        Site longitude [deg]
    min_el_deg : float
        Elevation mask [deg]
    r_moon : float
        Lunar radius [km]
    budget : LinkBudget or None
        If None, SNR gate == TX gate

    Returns
    -------
    vis_los : bool   — geometric LOS above horizon
    vis_tx  : bool   — LOS and above elevation mask
    vis_snr : bool   — vis_tx and link margin > 0
    el_deg  : float  — elevation angle [deg]
    """
    r_sc = np.asarray(r_sc, dtype=float).flatten()

    r_site, n_local = moon_fixed_to_inertial(lat_deg, lon_deg, r_moon)
    el_deg          = site_elevation_angle(r_sc, r_site, n_local)

    vis_los = el_deg > 0.0
    vis_tx  = vis_los and (el_deg >= min_el_deg)

    if budget is None or not vis_tx:
        vis_snr = vis_tx
    else:
        range_km = float(np.linalg.norm(r_sc - r_site))
        margin   = link_margin_db(range_km, budget)
        vis_snr  = vis_tx and (margin > 0.0)

    return vis_los, vis_tx, vis_snr, el_deg


if __name__ == "__main__":
    # Quick self-test — spacecraft directly above Equator_23E
    import numpy as np
    from moon_geometry import moon_fixed_to_inertial

    r_site, _ = moon_fixed_to_inertial(0.0, 23.0)
    r_sc_overhead = r_site * 40.0          # 40x lunar radius above site

    budget = LinkBudget()

    print("visibility.py — self test")
    print("-" * 50)

    sites = [
        ("Equator_23E",  0.0,   23.0),
        ("SouthPole",   -89.9,   0.0),
        ("NorthPole",    89.9,   0.0),
    ]

    for name, lat, lon in sites:
        los, tx, snr, el = los_visibility_link(
            r_sc_overhead, lat, lon,
            min_el_deg=10.0, budget=budget
        )
        print(f"{name:15s}  el={el:6.2f} deg  "
              f"LOS={int(los)}  TX={int(tx)}  SNR={int(snr)}")

    # Test link margin function
    margin = link_margin_db(60000.0, budget)
    print(f"\nLink margin at 60,000 km: {margin:.2f} dB")