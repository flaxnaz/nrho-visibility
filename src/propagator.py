"""
propagator.py
-------------
NRHO orbit propagator for Moon-centred missions.
Replaces HALO prophpop.m with a Python two-body + optional J2 integrator.
(Flaxon Nazareth, UNSW 2025)
"""

import numpy as np
from scipy.integrate import solve_ivp


# Moon gravitational parameter and radius
MU_MOON  = 4902.800118    # km3/s2
R_MOON   = 1737.4         # km
J2_MOON  = 2.0335e-4      # Moon J2 coefficient


def moon_two_body(t: float, y: np.ndarray,
                  mu: float = MU_MOON) -> np.ndarray:
    """
    Moon-centred two-body equations of motion.

    Parameters
    ----------
    t : float
        Time [s] (not used — autonomous system)
    y : np.ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in km and km/s
    mu : float
        Gravitational parameter [km3/s2]

    Returns
    -------
    dydt : np.ndarray, shape (6,)
    """
    r     = y[:3]
    v     = y[3:]
    r_norm = np.linalg.norm(r)
    a     = -mu / r_norm**3 * r
    return np.concatenate([v, a])


def moon_two_body_j2(t: float, y: np.ndarray,
                      mu: float = MU_MOON,
                      j2: float = J2_MOON,
                      re: float = R_MOON) -> np.ndarray:
    """
    Moon-centred two-body + J2 equations of motion.

    Parameters
    ----------
    t : float
        Time [s]
    y : np.ndarray, shape (6,)
        State [x, y, z, vx, vy, vz] km / km/s
    mu, j2, re : float
        Moon gravity parameter, J2 coefficient, radius

    Returns
    -------
    dydt : np.ndarray, shape (6,)
    """
    x, yy, z = y[0], y[1], y[2]
    r        = np.sqrt(x**2 + yy**2 + z**2)
    r2       = r**2
    r5       = r**5

    factor   = (3/2) * j2 * mu * re**2 / r5
    z_r2     = z**2 / r2

    ax = -mu*x/r**3  + factor * x  * (5*z_r2 - 1)
    ay = -mu*yy/r**3 + factor * yy * (5*z_r2 - 1)
    az = -mu*z/r**3  + factor * z  * (5*z_r2 - 3)

    return np.array([y[3], y[4], y[5], ax, ay, az])


def propagate_nrho(y0:        np.ndarray,
                   duration_s: float,
                   dt_s:       float = 30.0,
                   use_j2:     bool  = False,
                   rtol:       float = 1e-9,
                   atol:       float = 1e-12
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate an NRHO (or any Moon-centred orbit) forward in time.

    Parameters
    ----------
    y0 : np.ndarray, shape (6,)
        Initial state [x,y,z,vx,vy,vz] km / km/s
    duration_s : float
        Total propagation time [s]
    dt_s : float
        Output time step [s] (default 30s — matches HALO)
    use_j2 : bool
        Include Moon J2 perturbation (default False)
    rtol, atol : float
        Integrator tolerances

    Returns
    -------
    t : np.ndarray, shape (N,)   — time array [s]
    X : np.ndarray, shape (N,6) — state history [km, km/s]
    """
    t_eval = np.arange(0, duration_s + dt_s, dt_s)
    f      = moon_two_body_j2 if use_j2 else moon_two_body

    sol = solve_ivp(
        f,
        t_span=(0, duration_s),
        y0=y0,
        method='DOP853',           # high-order — matches ode113 accuracy
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")

    return sol.t, sol.y.T         # shape (N,), (N,6)


if __name__ == "__main__":
    # CAPSTONE-like NRHO initial state (Moon-centred J2000, km/km/s)
    # Approximate — real CAPSTONE state requires SPICE kernels
    y0 = np.array([
        -3200.0,   500.0,  60000.0,   # position [km]
         0.08,      0.30,    0.015    # velocity [km/s]
    ])

    P_NRHO    = 5.7444e5              # ~6.64 day NRHO period [s]
    duration  = 4 * P_NRHO            # 4 periods — matches your MATLAB

    print("propagator.py — self test")
    print("-" * 45)
    print(f"Propagating {duration/86400:.1f} days ({duration/P_NRHO:.0f} NRHO periods)...")

    t, X = propagate_nrho(y0, duration, dt_s=30.0)

    r_norms = np.linalg.norm(X[:, :3], axis=1)


    print(f"Steps:           {len(t)}")
    print(f"Duration:        {t[-1]/86400:.2f} days")
    print(f"Min Moon dist:   {r_norms.min():.1f} km")
    print(f"Max Moon dist:   {r_norms.max():.1f} km")
    print(f"Mean Moon dist:  {r_norms.mean():.1f} km")

    # Energy conservation check
    mu = MU_MOON
    KE  = 0.5 * np.linalg.norm(X[:, 3:], axis=1)**2
    PE  = -mu / r_norms
    E   = KE + PE
    print(f"Energy drift:    {abs(E[-1]-E[0]):.2e} km2/s2")