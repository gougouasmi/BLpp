"""

Reference:
  [1] Kemp, N.H., Rose, P.H., and Detra, R.W. :
      Laminar Heat Transfer Around Blunt Bodies
      in Dissociated Air, Journal of the Aero/Space
      Sciences, 1958, pp. 421-430.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple


def gen_hemisphere_data(
    pre_shock: Dict[str, float],
    grid_size: int,
) -> np.ndarray:
    """
    Compute (x/R, p/p_stag) grid for hemi-sphere

    """
    gam = pre_shock['gam']
    mach = pre_shock['mach']

    nb_points = grid_size
    x0, x1 = 0, np.pi/2

    # Post shock pressure and mach
    gamp1 = gam + 1
    gamm1 = gam - 1
    
    shock_ratio_p = 1. + 2.0 * gam / gamp1 * (mach ** 2 - 1.0)
    mach_shock = (
        (gamm1 * mach ** 2 + 2.) /
        (2. * gam * mach**2 - gamm1)
    ) ** 0.5

    # Stagnation pressure
    
    stag_ratio_temp = 1. + 0.5 * gamm1 * mach_shock ** 2
    stag_ratio_p = stag_ratio_temp ** (gam / gamm1)
    
    # Grid
    data = np.zeros((nb_points, 2), dtype=float)

    data[:, 0] = np.linspace(x0, x1, nb_points)
    data[:, 1] = 1 - (1. - 1. / (shock_ratio_p * stag_ratio_p)) * np.sin(data[:, 0]) ** 2

    # p1 / p_stag = p1 / p_shock * p_shock / p_stag
    #             = 1 / (shock_ratio_p * stag_ratio_p)

    return data


if __name__ == "__main__":

    ###
    # Inputs
    #
    
    pre_shock = {'gam': 1.19, 'mach': 8.8}
    nb_points = 100

    vary_mach_gamma = False

    # Generate pressure grid
    hemisphere_data = gen_hemisphere_data(pre_shock, nb_points)

    x_grid = hemisphere_data[:, 0]
    p_grid = hemisphere_data[:, 1]

    # Plot    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    ax.set_title(rf"$M = $ {pre_shock['mach']:.2f}, $\gamma = $ {pre_shock['gam']:.2f}")
    ax.plot(x_grid, p_grid, color='k')
    
    ax.set_xlabel(r"$\frac{x}{R}$", fontsize=15)
    ax.set_ylabel(r"$\frac{p}{p_{0}}$", rotation=0, fontsize=15)
    ax.set_xlim([x_grid[0], x_grid[-1]])
    ax.grid(which='both')

    # Write to .csv
    np.savetxt(
        f"hemisphere_pressure_{nb_points:d}.csv",
        hemisphere_data,
        delimiter=", ",
        fmt="%.6e",
    )

    ###
    # Compare profiles at different mach numbers
    #

    if vary_mach_gamma:
        gam_val = 1.4

        for gam_val in [1.1, 1.2, 1.3, 1.5]:

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

            ax.set_title(rf"Hemisphere pressure - $\gamma$ = {gam_val:.2f}")

            for mach_val in [2, 5, 10, 15, 30]:
                data = gen_hemisphere_data(
                    {'gam': gam_val, 'mach': mach_val}, 
                    nb_points,
                )

                x_grid = data[:, 0]
                p_grid = data[:, 1]

                ax.plot(x_grid, p_grid, label=rf"$M={mach_val}$")

            ax.set_xlabel(r"$\frac{x}{R}$", fontsize=15)
            ax.set_ylabel(r"$\frac{p}{p_{0}}$", rotation=0, fontsize=15)
            ax.set_xlim([x_grid[0], x_grid[-1]])
            ax.grid(which='both')
            ax.legend()