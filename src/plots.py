"""
plots.py
--------
Visualisation for NRHO visibility analysis.
Port of MATLAB plotting functions (Flaxon Nazareth, UNSW 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from moon_geometry import moon_fixed_to_inertial
from visibility import los_visibility_link, LinkBudget


DARK_BG   = '#0F1117'
PANEL_BG  = '#1A1D27'
CLR_LOS   = '#1D9E75'
CLR_TX    = '#7B6FE8'
CLR_SNR   = '#E8836F'
CLR_GRID  = '#333344'
CLR_TEXT  = '#AAAAAA'


def _style(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color='white', fontsize=9, pad=5)
    ax.tick_params(colors=CLR_TEXT, labelsize=8)
    ax.xaxis.label.set_color(CLR_TEXT)
    ax.yaxis.label.set_color(CLR_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(CLR_GRID)
    ax.grid(alpha=0.15, color='white')
    return ax


def plot_nrho_orbit(t: np.ndarray, X: np.ndarray,
                    save_path: str = "figures/nrho_orbit.png"):
    """3D NRHO orbit plot around the Moon."""
    fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(PANEL_BG)

    pos = X[:, :3]

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color=CLR_TX, lw=0.8, alpha=0.9, label='NRHO trajectory')
    ax.scatter(*pos[0],  color='#00FF88', s=80, zorder=5, label='Start')
    ax.scatter(*pos[-1], color='#FF4444', s=80, zorder=5, label='End')

    # Moon — scaled up x8 for visibility at NRHO distances
    u, v   = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    Rm_vis = 1737.4 * 8
    ax.plot_surface(Rm_vis*np.cos(u)*np.sin(v),
                    Rm_vis*np.sin(u)*np.sin(v),
                    Rm_vis*np.cos(v),
                    alpha=0.3, color='#AAAAAA')

    ax.view_init(elev=25, azim=135)
    ax.set_xlabel('X (km)', color=CLR_TEXT, fontsize=8)
    ax.set_ylabel('Y (km)', color=CLR_TEXT, fontsize=8)
    ax.set_zlabel('Z (km)', color=CLR_TEXT, fontsize=8)
    ax.tick_params(colors=CLR_TEXT, labelsize=7)
    ax.set_title('NRHO Orbit — Moon-Centred Inertial Frame\n'
                 '(Moon scaled x8 for visibility)',
                 color='white', fontsize=10)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    print(f"Saved: {save_path}")


def plot_radius_vs_time(t: np.ndarray, X: np.ndarray,
                        save_path: str = "figures/nrho_radius.png"):
    """Spacecraft distance from Moon centre over time."""
    r_norms = np.linalg.norm(X[:, :3], axis=1)
    t_days  = t / 86400

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    _style(ax, 'NRHO: Spacecraft–Moon Distance Over Time')

    ax.plot(t_days, r_norms, color=CLR_LOS, lw=1.0)
    ax.fill_between(t_days, r_norms, alpha=0.15, color=CLR_LOS)
    ax.axhline(r_norms.mean(), color='#FF6600', lw=1, linestyle='--',
               alpha=0.7, label=f'Mean {r_norms.mean():.0f} km')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Distance from Moon (km)')
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor='white')

    plt.tight_layout()
    fig.patch.set_facecolor(DARK_BG)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    print(f"Saved: {save_path}")


def plot_visibility_timeseries(t:          np.ndarray,
                                X:          np.ndarray,
                                min_el_deg: float = 10.0,
                                r_moon:     float = 1737.4,
                                budget:     LinkBudget = None,
                                save_path:  str = "figures/visibility.png"):
    """
    LOS / TX / SNR visibility vs time for all three sites.
    Port of MATLAB plot_visibility_time_series.m
    """
    sites = [
        ("Equator_23E",  0.0,  23.0),
        ("SouthPole",   -89.9,  0.0),
        ("NorthPole",    89.9,  0.0),
    ]

    positions = X[:, :3]
    N         = len(t)
    t_hours   = t / 3600

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor=DARK_BG)
    fig.suptitle(f'NRHO Visibility — minEl = {min_el_deg}°',
                 color='white', fontsize=11)

    for ax, (name, lat, lon) in zip(axes, sites):
        los_arr = np.zeros(N)
        tx_arr  = np.zeros(N)
        snr_arr = np.zeros(N)

        for k in range(N):
            vl, vt, vs, _ = los_visibility_link(
                positions[k], lat, lon, min_el_deg, r_moon, budget)
            los_arr[k] = float(vl)
            tx_arr[k]  = float(vt)
            snr_arr[k] = float(vs)

        _style(ax, f'{name} — LOS / TX / SNR')
        ax.step(t_hours, los_arr,        color=CLR_LOS, lw=1.0,
                label=f'LOS  ({100*los_arr.mean():.1f}%)', where='post')
        ax.step(t_hours, tx_arr  + 0.05, color=CLR_TX,  lw=1.0,
                linestyle='--',
                label=f'TX   ({100*tx_arr.mean():.1f}%)', where='post')
        ax.step(t_hours, snr_arr + 0.10, color=CLR_SNR, lw=1.0,
                linestyle=':',
                label=f'SNR  ({100*snr_arr.mean():.1f}%)', where='post')
        ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel('Visibility (0/1)')
        ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor='white',
                  loc='upper right')

    axes[-1].set_xlabel('Time (hours)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    print(f"Saved: {save_path}")


def plot_coverage_bars(df: pd.DataFrame,
                       save_path: str = "figures/coverage_bars.png"):
    """
    Grouped bar chart — LOS / TX / SNR coverage per site and mask.
    Port of MATLAB plot_coverage_bars.m
    """
    sites  = df['Site'].unique()
    masks  = sorted(df['MinEl_deg'].unique())
    n_s    = len(sites)

    fig, axes = plt.subplots(1, n_s, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle('NRHO Coverage Summary — LOS / TX / SNR vs Elevation Mask',
                 color='white', fontsize=11)

    colors = [CLR_LOS, CLR_TX, CLR_SNR]
    width  = 0.22
    x      = np.arange(len(masks))

    for ax, site in zip(axes, sites):
        _style(ax, site)
        sub = df[df['Site'] == site].sort_values('MinEl_deg')

        for i, (col, label) in enumerate(
                zip(['LOS_pct', 'TX_pct', 'SNR_pct'], ['LOS', 'TX', 'SNR'])):
            ax.bar(x + (i - 1) * width, sub[col].values,
                   width=width, color=colors[i], alpha=0.85, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(m)}°' for m in masks])
        ax.set_xlabel('Elevation mask')
        ax.set_ylabel('Coverage (%)')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.show()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    from propagator import propagate_nrho
    from coverage import coverage_sweep

    # CAPSTONE-like NRHO initial state
    y0 = np.array([
        -3200.0,  500.0,  60000.0,
            0.08,   0.30,    0.015
    ])

    P_NRHO   = 5.7444e5
    duration = 4 * P_NRHO

    print("plots.py — generating all figures...")
    print("Propagating orbit...")
    t, X = propagate_nrho(y0, duration, dt_s=30.0)
    r_norms = np.linalg.norm(X[:, :3], axis=1)
    print(f"Done. {len(t)} steps.")
    print(f"Min Moon dist: {r_norms.min():.1f} km")
    print(f"Max Moon dist: {r_norms.max():.1f} km")

    print("\nPlotting 3D orbit...")
    plot_nrho_orbit(t, X)

    print("Plotting radius vs time...")
    plot_radius_vs_time(t, X)

    print("\nRunning coverage sweep...")
    df = coverage_sweep(X, verbose=False)

    print("Plotting coverage bars...")
    plot_coverage_bars(df)

    print("\nPlotting visibility time series (10 deg mask)...")
    print("This will take a few minutes...")
    plot_visibility_timeseries(t, X, min_el_deg=10.0)

    print("\nAll figures saved to figures/")