"""
test_moon_geometry.py
---------------------
Tests for moon_geometry.py
pytest style — no classes needed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from moon_geometry import moon_fixed_to_inertial, site_elevation_angle


# ── Basic correctness tests ───────────────────────────────────

def test_site_radius_equator():
    """Site on equator should be exactly at lunar radius."""
    r, n = moon_fixed_to_inertial(0.0, 0.0, r_moon=1737.4)
    assert abs(np.linalg.norm(r) - 1737.4) < 1e-6

def test_site_radius_pole():
    """Site at pole should also be at lunar radius."""
    r, n = moon_fixed_to_inertial(90.0, 0.0, r_moon=1737.4)
    assert abs(np.linalg.norm(r) - 1737.4) < 1e-6

def test_normal_is_unit_vector():
    """Local normal must always be a unit vector."""
    for lat in [-89.9, 0.0, 45.0, 89.9]:
        for lon in [0.0, 23.0, 180.0]:
            _, n = moon_fixed_to_inertial(lat, lon)
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10

def test_normal_is_radial():
    """Normal should point in same direction as position vector."""
    r, n = moon_fixed_to_inertial(30.0, 60.0)
    r_hat = r / np.linalg.norm(r)
    assert np.allclose(r_hat, n, atol=1e-10)

def test_equator_23e_position():
    """Equator 23E should have correct x,y,z components."""
    r, _ = moon_fixed_to_inertial(0.0, 23.0)
    assert abs(r[2]) < 1e-6          # z should be ~0 on equator
    assert r[0] > 0                   # x positive
    assert r[1] > 0                   # y positive (east of prime meridian)

def test_north_pole_position():
    """North pole should point along +Z axis."""
    r, n = moon_fixed_to_inertial(89.9, 0.0)
    assert r[2] > 0                   # z positive
    assert abs(n[2] - np.linalg.norm(n) * np.sin(np.radians(89.9))) < 1e-4

def test_south_pole_position():
    """South pole should point along -Z axis."""
    r, _ = moon_fixed_to_inertial(-89.9, 0.0)
    assert r[2] < 0                   # z negative

def test_custom_radius():
    """Custom radius should scale the position correctly."""
    r1, _ = moon_fixed_to_inertial(0.0, 0.0, r_moon=1000.0)
    r2, _ = moon_fixed_to_inertial(0.0, 0.0, r_moon=2000.0)
    assert abs(np.linalg.norm(r1) - 1000.0) < 1e-6
    assert abs(np.linalg.norm(r2) - 2000.0) < 1e-6


# ── Elevation angle tests ─────────────────────────────────────

def test_overhead_elevation_is_90():
    """Spacecraft directly above site should give 90 deg elevation."""
    r_site, n = moon_fixed_to_inertial(0.0, 23.0)
    r_sc = r_site * 40.0             # directly above, 40x further out
    el = site_elevation_angle(r_sc, r_site, n)
    assert abs(el - 90.0) < 1e-3

def test_below_horizon_is_negative():
    """Spacecraft on opposite side of Moon should be below horizon."""
    r_site, n = moon_fixed_to_inertial(0.0, 0.0)
    r_sc = -r_site * 10.0            # opposite side
    el = site_elevation_angle(r_sc, r_site, n)
    assert el < 0

def test_elevation_range():
    """Elevation angle must always be in [-90, 90] degrees."""
    r_site, n = moon_fixed_to_inertial(45.0, 90.0)
    for scale in [1.1, 2.0, 5.0, 40.0]:
        r_sc = r_site * scale
        el = site_elevation_angle(r_sc, r_site, n)
        assert -90.0 <= el <= 90.0


# ── Parametrize — run same test with multiple inputs ──────────
# This is pytest's killer feature vs GTest

@pytest.mark.parametrize("lat,lon", [
    (0.0,   0.0),
    (0.0,  23.0),
    (-89.9, 0.0),
    (89.9,  0.0),
    (45.0, 180.0),
])
def test_site_always_on_surface(lat, lon):
    """All sites should land exactly on lunar surface."""
    r, _ = moon_fixed_to_inertial(lat, lon)
    assert abs(np.linalg.norm(r) - 1737.4) < 1e-5


# ── Edge cases ────────────────────────────────────────────────

def test_spacecraft_at_site_returns_90():
    """Degenerate case: spacecraft exactly at site surface."""
    r_site, n = moon_fixed_to_inertial(0.0, 0.0)
    el = site_elevation_angle(r_site, r_site, n)
    assert el == 90.0