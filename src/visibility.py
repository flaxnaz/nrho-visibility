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
    freq_hz:        float = 2.2e9
    ptx_dbw:        float = 10.0
    gtx_dbi:        float = 5.0
    grx_dbi:        float = 45.0
    tsys_k:         float = 500.0
    rb_bps:         float = 1000.0
    ebno_req_db:    float = 9.0
    imp_margin_db:  float = 0.0
    misc_loss_db:   float = 2.0


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

    k_db    = -228.6
    n0_dbw  = k_db + 10 * np.log10(budget.tsys_k)

    cno     = c_dbw - n0_dbw
    ebno_db = cno - 10 * np.log10(budget.rb_bps)
    margin  = ebno_db - budget.ebno_req_db - budget.imp_margin_db

    return float(margin)


def los_visibility_link(r_sc:       np.ndarray,
                         lat_deg:    float,
                         lon_deg:    float,
                         min_el_deg: float = 10.0,
                         r_moon:     float = 1737.4,
                         budget:     LinkBudget = None
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
    vis_los : bool
    vis_tx  : bool
    vis_snr : bool
    el_deg  : float
    """
    r_sc = np.asarray(r_sc, dtype=float).flatten()

    r_site, n_local = moon_fixed_to_inertial(lat_deg, lon_deg, r_moon)
    el_deg = site_elevation_angle(r_sc, r_site, n_local)

    vis_los = bool(el_deg > 0.0)
    vis_tx  = bool(vis_los and (el_deg >= min_el_deg))

    if budget is None or not vis_tx:
        vis_snr = vis_tx
    else:
        range_km = float(np.linalg.norm(r_sc - r_site))
        if range_km < 1.0:
            vis_snr = vis_tx
        else:
            margin  = link_margin_db(range_km, budget)
            vis_snr = bool(vis_tx and (margin > 0.0))

    return vis_los, vis_tx, vis_snr, float(el_deg)


if __name__ == "__main__":
    import numpy as np
    from moon_geometry import moon_fixed_to_inertial

    r_site, _ = moon_fixed_to_inertial(0.0, 23.0)
    r_sc_overhead = r_site * 40.0

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

    margin = link_margin_db(60000.0, budget)
    print(f"\nLink margin at 60,000 km: {margin:.2f} dB")