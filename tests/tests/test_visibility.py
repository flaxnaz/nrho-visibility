"""
test_visibility.py
------------------
Tests for visibility.py — LOS, TX, SNR gating and link budget.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from moon_geometry import moon_fixed_to_inertial
from visibility import los_visibility_link, link_margin_db, LinkBudget


# ── Fixtures — reusable test data ────────────────────────────
# pytest fixtures replace GTest SetUp() — cleaner and composable

@pytest.fixture
def default_budget():
    """Standard S-band link budget."""
    return LinkBudget()

@pytest.fixture
def equator_site():
    """Equator 23E site position."""
    r, _ = moon_fixed_to_inertial(0.0, 23.0)
    return r

@pytest.fixture
def spacecraft_overhead(equator_site):
    """Spacecraft directly above Equator 23E."""
    return equator_site * 40.0


# ── LOS tests ─────────────────────────────────────────────────

def test_overhead_spacecraft_has_los(spacecraft_overhead, default_budget):
    """Spacecraft directly above site must have LOS."""
    los, tx, snr, el = los_visibility_link(
        spacecraft_overhead, 0.0, 23.0,
        min_el_deg=0.0, budget=default_budget
    )
    assert los is True

def test_overhead_elevation_near_90(spacecraft_overhead, default_budget):
    """Spacecraft directly above should give ~90 deg elevation."""
    _, _, _, el = los_visibility_link(
        spacecraft_overhead, 0.0, 23.0,
        min_el_deg=0.0, budget=default_budget
    )
    assert abs(el - 90.0) < 0.1

def test_opposite_side_no_los(default_budget):
    """Spacecraft on opposite side of Moon — no LOS."""
    r_site, _ = moon_fixed_to_inertial(0.0, 0.0)
    r_sc = -r_site * 10.0
    los, tx, snr, el = los_visibility_link(
        r_sc, 0.0, 0.0,
        min_el_deg=0.0, budget=default_budget
    )
    assert los is False
    assert tx  is False
    assert snr is False

def test_no_los_means_no_tx(default_budget):
    """TX must be False whenever LOS is False."""
    r_site, _ = moon_fixed_to_inertial(0.0, 0.0)
    r_sc = -r_site * 5.0
    los, tx, snr, el = los_visibility_link(
        r_sc, 0.0, 0.0, min_el_deg=0.0
    )
    if not los:
        assert not tx
        assert not snr


# ── Elevation mask tests ──────────────────────────────────────

def test_zero_mask_same_as_los(spacecraft_overhead, default_budget):
    """At 0 deg mask, TX should equal LOS."""
    los, tx, _, _ = los_visibility_link(
        spacecraft_overhead, 0.0, 23.0,
        min_el_deg=0.0, budget=default_budget
    )
    assert los == tx

def test_strict_mask_reduces_coverage(default_budget):
    """Higher elevation mask should never increase coverage."""
    r_site, _ = moon_fixed_to_inertial(0.0, 23.0)
    r_sc = r_site * 3.0

    _, tx_0,  _, _ = los_visibility_link(r_sc, 0.0, 23.0, 0.0,  budget=default_budget)
    _, tx_10, _, _ = los_visibility_link(r_sc, 0.0, 23.0, 10.0, budget=default_budget)
    _, tx_45, _, _ = los_visibility_link(r_sc, 0.0, 23.0, 45.0, budget=default_budget)

    # Coverage can only stay same or decrease with stricter mask
    assert int(tx_0) >= int(tx_10) >= int(tx_45)

@pytest.mark.parametrize("min_el", [0, 5, 10, 30, 45, 89])
def test_mask_values_dont_crash(min_el, spacecraft_overhead, default_budget):
    """Any elevation mask value should run without error."""
    los, tx, snr, el = los_visibility_link(
        spacecraft_overhead, 0.0, 23.0,
        min_el_deg=float(min_el), budget=default_budget
    )
    assert isinstance(los, bool)
    assert isinstance(el, float)


# ── Link budget tests ─────────────────────────────────────────

def test_link_margin_positive_at_close_range(default_budget):
    """Link should close at short range."""
    margin = link_margin_db(1000.0, default_budget)
    assert margin > 0

def test_link_margin_negative_at_extreme_range(default_budget):
    """Link should not close at extreme range."""
    margin = link_margin_db(1e7, default_budget)
    assert margin < 0

def test_link_margin_decreases_with_range(default_budget):
    """Margin must decrease as range increases."""
    m1 = link_margin_db(10000.0,  default_budget)
    m2 = link_margin_db(50000.0,  default_budget)
    m3 = link_margin_db(100000.0, default_budget)
    assert m1 > m2 > m3

def test_link_margin_at_60000km(default_budget):
    """Known value — 25.75 dB at 60,000 km from Day 4."""
    margin = link_margin_db(60000.0, default_budget)
    assert abs(margin - 25.75) < 0.5

def test_higher_tx_power_improves_margin():
    """Doubling TX power should increase margin by ~3 dB."""
    b1 = LinkBudget(ptx_dbw=10.0)
    b2 = LinkBudget(ptx_dbw=13.0)
    m1 = link_margin_db(60000.0, b1)
    m2 = link_margin_db(60000.0, b2)
    assert abs((m2 - m1) - 3.0) < 0.1


# ── Return type tests ─────────────────────────────────────────

def test_return_types(spacecraft_overhead, default_budget):
    """All return types must be correct."""
    los, tx, snr, el = los_visibility_link(
        spacecraft_overhead, 0.0, 23.0,
        min_el_deg=10.0, budget=default_budget
    )
    assert isinstance(los, bool)
    assert isinstance(tx,  bool)
    assert isinstance(snr, bool)
    assert isinstance(el,  float)

def test_elevation_always_in_valid_range(default_budget):
    """Elevation must always be between -90 and 90 degrees."""
    r_site, _ = moon_fixed_to_inertial(0.0, 23.0)
    for scale in [0.5, 1.0, 2.0, 10.0, 100.0]:
        r_sc = r_site * scale
        _, _, _, el = los_visibility_link(
            r_sc, 0.0, 23.0, budget=default_budget
        )
        assert -90.0 <= el <= 90.0