"""
moon_geometry.py
----------------
Lunar surface geometry utilities.
Port of MATLAB moon_fixed_to_inertial.m (Flaxon Nazareth, UNSW 2025)

Assumes simplified non-rotating Moon frame — sufficient for approximate
visibility and coverage analysis without SPICE dependency.
"""

import numpy as np


def moon_fixed_to_inertial(lat_deg: float,
                            lon_deg: float,
                            r_moon: float = 1737.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a lunar surface site (lat, lon) to Moon-centred inertial position.

    Assumes Moon is non-rotating relative to the inertial frame.
    Body-fixed and inertial frames are aligned — valid for approximate
    coverage analysis without a lunar PCK kernel.

    Parameters
    ----------
    lat_deg : float
        Site latitude in degrees [-90, 90]
    lon_deg : float
        Site east longitude in degrees [-180, 180]
    r_moon : float
        Lunar radius in km (default 1737.4)

    Returns
    -------
    r_site : np.ndarray, shape (3,)
        Site position vector in km, Moon-centred
    n_local : np.ndarray, shape (3,)
        Outward unit normal at site (radial direction)

    Examples
    --------
    >>> r, n = moon_fixed_to_inertial(0.0, 23.0)
    >>> round(float(np.linalg.norm(r)), 2)
    1737.4
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    r_site = r_moon * np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

    n_local = r_site / np.linalg.norm(r_site)

    return r_site, n_local


def site_elevation_angle(r_sc: np.ndarray,
                          r_site: np.ndarray,
                          n_local: np.ndarray) -> float:
    """
    Compute elevation angle of spacecraft as seen from a surface site.

    Parameters
    ----------
    r_sc : np.ndarray, shape (3,)
        Spacecraft position [km], Moon-centred
    r_site : np.ndarray, shape (3,)
        Site position [km], Moon-centred
    n_local : np.ndarray, shape (3,)
        Site outward unit normal

    Returns
    -------
    el_deg : float
        Elevation angle in degrees. Positive = above horizon.
    """
    v_los = r_sc - r_site
    v_norm = np.linalg.norm(v_los)

    if v_norm == 0:
        return 90.0                        # spacecraft exactly at site

    u_los = v_los / v_norm
    arg   = np.clip(np.dot(u_los, n_local), -1.0, 1.0)
    el_deg = np.degrees(np.arcsin(arg))

    return el_deg


if __name__ == "__main__":
    # Quick self-test
    sites = [
        ("Equator_23E",  0.0,   23.0),
        ("SouthPole",   -89.9,   0.0),
        ("NorthPole",    89.9,   0.0),
    ]

    print("moon_geometry.py — self test")
    print("-" * 40)
    for name, lat, lon in sites:
        r, n = moon_fixed_to_inertial(lat, lon)
        print(f"{name:15s}  |r| = {np.linalg.norm(r):.2f} km  "
              f"n = [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")