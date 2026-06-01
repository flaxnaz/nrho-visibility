"""
conftest.py
-----------
Shared pytest fixtures for nrho-visibility test suite.
Fixtures defined here are automatically available to all test files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from moon_geometry import moon_fixed_to_inertial
from visibility import LinkBudget


@pytest.fixture
def default_budget() -> LinkBudget:
    """Standard S-band link budget — shared across test files."""
    return LinkBudget()


@pytest.fixture
def equator_site() -> np.ndarray:
    """Equator 23E site position vector."""
    r, _ = moon_fixed_to_inertial(0.0, 23.0)
    return r


@pytest.fixture
def south_pole_site() -> np.ndarray:
    """South Pole site position vector."""
    r, _ = moon_fixed_to_inertial(-89.9, 0.0)
    return r


@pytest.fixture
def spacecraft_overhead(equator_site) -> np.ndarray:
    """Spacecraft directly above Equator 23E at 40x lunar radius."""
    return equator_site * 40.0


@pytest.fixture
def spacecraft_leo() -> np.ndarray:
    """ISS-like LEO spacecraft state vector [km, km/s]."""
    return np.array([6771.0, 0.0, 0.0, 0.0, 7.6726, 0.0])