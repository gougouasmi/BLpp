"""

Helper functions for post-processing

"""

import numpy as np
import h5py

###
# Indices
#

FP_ID, GP_ID, G_ID = 2, 1, 4 

EDGE_U_ID = 0
EDGE_H_ID = 1
EDGE_P_ID = 2
EDGE_XI_ID = 3
EDGE_X_ID = 4
EDGE_DU_DXI_ID = 5
EDGE_DH_DXI_ID = 6
EDGE_DXI_DX_ID = 7

###
#
#

def read_eta_file(filename: str = "eta_grid.h5") -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        eta_grid = f["eta_grid"][:]
    
    return eta_grid

def read_edge_file(filename: str = "edge_grid.h5") -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        edge_grid = np.transpose(f["edge_data"][:])

    return edge_grid

def read_profile(filename: str) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        profile = np.transpose(f["state_data"][:])

    return profile