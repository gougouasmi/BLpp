"""

Use this script to visualize boundary
layer profiles at different stations.

"""

import numpy as np
import matplotlib.pyplot as plt

from view_edge import read_edge_file
from view_profile import read_eta_file, read_profile, view_state
from view_outputs import read_outputs, view_outputs

from typing import List

###
# Read eta grid
#

eta_grid = read_eta_file()

###
# Read edge conditions
#

edge_grid, edge_indices = read_edge_file()

EDGE_U_ID = edge_indices["ue"]
EDGE_H_ID = edge_indices["he"]
EDGE_P_ID = edge_indices["pe"]
EDGE_XI_ID = edge_indices["xi"]
EDGE_X_ID = edge_indices["x"]
EDGE_DU_DXI_ID = edge_indices["due/dxi"]
EDGE_DH_DXI_ID = edge_indices["dhe/dxi"]
EDGE_DXI_DX_ID = edge_indices["dxi/dx"]
EDGE_RO_ID = edge_indices["roe"]
EDGE_MU_ID = edge_indices["mue"]

ue_grid = edge_grid[EDGE_U_ID,:]
he_grid = edge_grid[EDGE_H_ID,:]
pe_grid = edge_grid[EDGE_P_ID,:]

x_grid = edge_grid[EDGE_X_ID,:]
xi_grid = edge_grid[EDGE_XI_ID,:]

dxi_dx_grid = edge_grid[EDGE_DXI_DX_ID,:]

xi0, xi1 = np.min(xi_grid), np.max(xi_grid)
x0, x1 = np.min(x_grid), np.max(x_grid)

# Compute factors
ue_0 = ue_grid[0]
he_0 = he_grid[0]

due_dx0 = (ue_grid[1] - ue_grid[0]) / (x_grid[1] - x_grid[0])

output_grid_0, output_labels = read_outputs("station_0_outputs.h5")

roe_0 = edge_grid[EDGE_RO_ID, 0]
mue_0 = edge_grid[EDGE_MU_ID, 0]

station_0_factors = {
    'tau_factor': ((roe_0 * mue_0) ** 0.5) * (due_dx0 ** 1.5) * x0,
    'q_factor': he_0 * (roe_0 * mue_0 * due_dx0) ** 0.5,
    'y_factor': (roe_0 * mue_0 / due_dx0) ** 0.5,
}

###
# Profiles at various stations
#

def view_station(station_id: int) -> None:
    assert(station_id >= 0)

    filename = f"station_{station_id:d}.h5"
    state_grid, state_labels = read_profile(filename)
    view_state(state_grid, eta_grid, state_labels, f"Station {station_id:d}")

def view_station_outputs(station_id: int) -> None:
    assert(station_id >= 0)

    ue = ue_grid[station_id]
    he = he_grid[station_id]
    xi = xi_grid[station_id]
    dxi_dx = dxi_dx_grid[station_id]

    station_factors = None
    if station_id > 0:
        station_factors = {
            'y_factor': np.sqrt(2. * xi) / ue,
            'tau_factor': ue * dxi_dx / np.sqrt(2. * xi),
            'q_factor': he * dxi_dx / np.sqrt(2. * xi),
        }
    else:
        station_factors = station_0_factors
        

    filename = f"station_{station_id:d}_outputs.h5"
    output_grid, output_labels = read_outputs(filename)

    view_outputs(
        output_grid,
        eta_grid,
        output_labels,
        station_factors,
        f"Station {station_id:d}",
    )

def compare_stations(station_ids: List[int]) -> None:
    if len(station_ids) > 5:
        print("compare_stations: no more than 5 station ids.\n")
        return 

    # Wrt to xi (get a sense of the signs of the diff-diff terms)
    _, ax_eta_f = plt.subplots(1, figsize=(5,5))
    _, ax_eta_fp = plt.subplots(1, figsize=(5,5))
    _, ax_eta_g = plt.subplots(1, figsize=(5,5))

    # Wrt to y (get sense of physical thickness)
    _, ax_y_f = plt.subplots(1, figsize=(5,5))
    _, ax_y_fp = plt.subplots(1, figsize=(5,5))
    _, ax_y_g = plt.subplots(1, figsize=(5,5))

    for station_id in station_ids:
        # State data
        filename = f"station_{station_id:d}.h5"
        state_grid, state_labels = read_profile(filename)

        F_ID = state_labels["f"]
        FP_ID = state_labels["f'"]
        G_ID = state_labels["g"]
 
        profile_size = state_grid.shape[1]

        ax_eta_f.plot(state_grid[F_ID], eta_grid[:profile_size], label=f"stat. {station_id:d}")
        ax_eta_fp.plot(state_grid[FP_ID], eta_grid[:profile_size], label=f"stat. {station_id:d}")
        ax_eta_g.plot(state_grid[G_ID], eta_grid[:profile_size], label=f"stat. {station_id:d}")

        # Output data
        filename = f"station_{station_id:d}_outputs.h5"
        output_grid, output_labels = read_outputs(filename)

        Y_ID = output_labels["y"]
        y_grid = output_grid[Y_ID, :]

        y_factor = station_0_factors["y_factor"]
        if station_id > 0:
            ue = ue_grid[station_id]
            xi = xi_grid[station_id]

            y_factor = np.sqrt(2. * xi) / ue

        y_grid *= y_factor

        ax_y_f.plot(state_grid[F_ID], y_grid[:profile_size], label=f"stat. {station_id:d}")
        ax_y_fp.plot(state_grid[FP_ID], y_grid[:profile_size], label=f"stat. {station_id:d}")
        ax_y_g.plot(state_grid[G_ID], y_grid[:profile_size], label=f"stat. {station_id:d}")

    for ax in [ax_eta_f, ax_eta_fp, ax_eta_g]:
        ax.set_ylim([0, eta_grid[-1]])
        ax.set_ylabel(r"$\eta$")
        ax.grid(which="both")
        ax.legend()

    for ax in [ax_y_f, ax_y_fp, ax_y_g]:
        ax.set_ylim([0, None])
        ax.set_ylabel(r"$y$")
        ax.grid(which="both")
        ax.legend()

    ax_eta_f.set_title(r"$f(\xi, \eta)$ profiles")
    ax_eta_fp.set_title(r"$f'(\xi, \eta)$ profiles")
    ax_eta_g.set_title(r"$g(\xi, \eta)$ profiles")

    ax_y_f.set_title(r"$f(\xi, y)$ profiles")
    ax_y_fp.set_title(r"$f'(\xi, y)$ profiles")
    ax_y_g.set_title(r"$g(\xi, y)$ profiles")

    plt.show()
    
###
# Compute heat flux values
#

def view_heat_flux(station_0: int, station_1: int) -> None:

    assert(station_0 >= 0)
    assert(station_1 > station_0)

    nb_stations = station_1 - station_0 + 1

    x = x_grid[station_0:station_1+1]
    qw = np.zeros(nb_stations, dtype=float)

    for station_id in range(station_0, station_1+1):
    
        he = he_grid[station_id]
        xi = xi_grid[station_id]
        dxi_dx = dxi_dx_grid[station_id]

        q_factor = 1
        if (xi > 0): 
            q_factor = he * dxi_dx / np.sqrt(2. * xi)
        else:
            q_factor = station_0_factors["q_factor"]

        output_grid, output_labels = read_outputs(f"station_{station_id}_outputs.h5")
    
        Q_ID = output_labels["q"]
    
        qw[station_id-station_0] = output_grid[Q_ID, 0] * q_factor

    fig, ax = plt.subplots(figsize=(5,5))

    fig.suptitle("Heat flux")

    ax.plot(x, qw/qw[0], c='k', marker='x')

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\bar{q}_{w}$")

    ax.set_xlim([x[0], x[-1]])
    ax.grid(which='both')

    fig.show()
    