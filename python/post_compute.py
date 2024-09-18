"""

Use this script to visualize boundary
layer profiles at different stations.

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

from view_edge import read_edge_file
from view_profile import read_eta_file, read_profile, view_state
from view_outputs import read_outputs, view_outputs

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
    
ue_grid = edge_grid[EDGE_U_ID,:]
he_grid = edge_grid[EDGE_H_ID,:]
pe_grid = edge_grid[EDGE_P_ID,:]

x_grid = edge_grid[EDGE_X_ID,:]
xi_grid = edge_grid[EDGE_XI_ID,:]

xi0, xi1 = np.min(xi_grid), np.max(xi_grid)
x0, x1 = np.min(x_grid), np.max(x_grid)

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

    filename = f"station_{station_id:d}_outputs.h5"
    output_grid, output_labels = read_outputs(filename)
    view_outputs(output_grid, eta_grid, output_labels, f"Station {station_id:d}")

###
# Compute heat flux values : TODO
#
