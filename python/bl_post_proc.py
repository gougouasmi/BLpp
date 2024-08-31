"""

Use this script to visualize boundary
layer profiles at different stations.

"""

import numpy as np
import matplotlib.pyplot as plt

FP_ID, GP_ID, G_ID = 2, 1, 4 

EDGE_U_ID = 0
EDGE_H_ID = 1
EDGE_P_ID = 2
EDGE_XI_ID = 3
EDGE_X_ID = 4
EDGE_DU_DXI_ID = 5
EDGE_DH_DXI_ID = 6

station_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80]

eta_grid = np.loadtxt("eta_grid.csv")

###
# Show edge conditions
#

edge_grid = np.transpose(
    np.loadtxt("edge_grid.csv", delimiter=",", dtype=float)
)

ue_grid = edge_grid[EDGE_U_ID,:]
he_grid = edge_grid[EDGE_H_ID,:]
pe_grid = edge_grid[EDGE_P_ID,:]

x_grid = edge_grid[EDGE_X_ID,:]
xi_grid = edge_grid[EDGE_XI_ID,:]

xi0, xi1 = np.min(xi_grid), np.max(xi_grid)
x0, x1 = np.min(x_grid), np.max(x_grid)

# Edge conditions wrt to xi
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

fig.suptitle(rf"Edge conditions ($\xi$)")

ax1, ax2 = axes

ax1.plot(xi_grid, ue_grid, color='k')
ax1.set_title(r"$u_{e}(\xi)$")

ax2.plot(xi_grid, pe_grid/pe_grid[0], color='k')
ax2.set_title(r"$p_{e}(\xi) / p_{e}(0)$")

for ax in axes:
    ax.grid(which="both")
    ax.set_xlim([xi0, xi1])
    ax.set_xlabel(r"$\xi$")

fig.show()

# Edge conditions wrt to x
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

fig.suptitle(rf"Edge conditions ($x$)")

ax1, ax2 = axes

ax1.plot(x_grid, ue_grid, color='k')
ax1.set_title(r"$u_{e}(x)$")

ax2.plot(x_grid, pe_grid/pe_grid[0], color='k')
ax2.set_title(r"$p_{e}(x) / p_{e}(0)$")

for ax in axes:
    ax.grid(which="both")
    ax.set_xlim([x0, x1])
    ax.set_xlabel(r"$x$")

fig.show()

###
# Profiles at various stations
#

def view_station(station_id: int) -> None:
    assert(station_id >= 0)

    filename = f"station_{station_id:d}.csv"
    profile = np.transpose(np.loadtxt(filename, delimiter=",", dtype=float))

    assert(profile.shape[1] == eta_grid.shape[0])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    fig.suptitle(f"Station #{station_id:d}")

    ax1, ax2 = axes

    ax1.plot(profile[FP_ID, :], eta_grid, color='k')
    ax1.set_title(r"$f'(\eta) \ = \ u / u_{e}$")

    ax2.plot(profile[G_ID, :], eta_grid, color='k')
    ax2.set_title(r"$g(\eta) \ = \ h / h_{e}$")

    for ax in axes:
        ax.grid(which="both")
        ax.set_ylim([0, np.max(eta_grid)])

    fig.show()

###
# Compute heat flux values
#

nb_stations = x_grid.shape[0]
qw = np.zeros(nb_stations, dtype=float)

for station_id in range(nb_stations):

    filename = f"station_{station_id:d}.csv"
    profile = np.loadtxt(filename, delimiter=",", dtype=float)

    qw[station_id] = profile[0][GP_ID]

fig, ax = plt.subplots(1, figsize=(6,6))

ax.set_title(r"$q_{w} / q_{w}(0)$")

ax.plot(x_grid, qw/qw[0])

ax.grid(which="both")
ax.set_xlim([x0, x1])
ax.set_ylim([0, 1.3])
ax.set_xlabel(r"$x$")


