"""

Use this script to visualize properties
at the edge of your boundary layer domain.

You can create a symbolic link to your use folder
-> ln -s <path_to_view_edge.py> .

"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

from typing import Tuple, Dict

###
# Helper functions
#

def read_edge_file(
    filename: str = "edge_grid.h5",
) -> Tuple[np.ndarray, Dict[str, int]]:

    with h5py.File(filename, 'r') as f:
        edge_grid = np.transpose(f["data"][:])
        assert(f.attrs["description"] == "edge fields")
        raw = f["field indices"][:]

    edge_indices = {
        key.decode('utf-8'): key_id
        for key_id, key in zip(raw['index'], raw['label'])
    }

    return edge_grid, edge_indices

def view_edge_conditions(
    edge_grid: np.ndarray,
    edge_indices: Dict[str, int],
) -> None:

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
    
    dxi_dx_grid = edge_grid[EDGE_DXI_DX_ID, :]
    due_dxi_grid = edge_grid[EDGE_DU_DXI_ID, :]
    dhe_dxi_grid = edge_grid[EDGE_DH_DXI_ID, :]
    
    x_grid = edge_grid[EDGE_X_ID]
    x0, x1 = np.min(x_grid), np.max(x_grid)
    
    xi_grid = edge_grid[EDGE_XI_ID]
    xi0, xi1 = np.min(xi_grid), np.max(xi_grid)
    
    #
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    ax1, ax2, ax3 = axes
    
    ax1.plot(x_grid, ue_grid, marker='x')
    ax1.set_title(r"$u_{e}(x)$")
    
    ax2.plot(x_grid, he_grid, marker='x')
    ax2.set_title(r"$h_{e}(x)$")
    
    ax3.plot(x_grid, pe_grid, marker='x')
    ax3.set_title(r"$p_{e}(x)$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([x0, x1])
        ax.set_xlabel(r"$x$")
    
    #
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    ax1, ax2, ax3 = axes
    
    ax1.plot(xi_grid, dxi_dx_grid, marker='x')
    ax1.set_title(r"$\frac{d\xi}{dx}$")
    
    ax2.plot(xi_grid, due_dxi_grid, marker='x')
    ax2.set_title(r"$\frac{du_{e}}{d\xi}$")
    
    ax3.plot(xi_grid, dhe_dxi_grid, marker='x')
    ax3.set_title(r"$\frac{dh_{e}}{d\xi}$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([xi0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    # 
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    ax1, ax2, ax3 = axes
    
    ax1.plot(xi_grid, ue_grid, marker='x')
    ax1.set_title(r"$u_{e}(\xi)$")
    
    ax2.plot(xi_grid, he_grid, marker='x')
    ax2.set_title(r"$h_{e}(\xi)$")
    
    ax3.plot(xi_grid, pe_grid, marker='x')
    ax3.set_title(r"$p_{e}(\xi)$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([xi0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    ## Plot ODE coefficients
    due_dxi_grid = edge_grid[EDGE_DU_DXI_ID, :]
    dhe_dxi_grid = edge_grid[EDGE_DH_DXI_ID, :]
    
    c1_grid = 2. * xi_grid[1:] * due_dxi_grid[1:] / ue_grid[1:]
    c2_grid = 2. * xi_grid[1:] * dhe_dxi_grid[1:] / he_grid[1:]
    c3_grid = 2. * xi_grid[1:] * due_dxi_grid[1:] * ue_grid[1:] / he_grid[1:]
    
    #
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    ax11, ax12 = axes
    
    ax11.plot(xi_grid[1:], c1_grid, marker='x')
    ax11.set_title(r"$c_{1}(\xi) := \frac{2 \xi}{u_{e}}\frac{du_{e}}{d\xi}$")
    
    ax12.plot(xi_grid[1:], c2_grid, marker='x')
    ax12.set_title(r"$c_{2}(\xi) = \frac{2 \xi}{h_{e}} \frac{dh_{e}}{d\xi} = - c_{3}(\xi)$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([xi0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    #
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    ax21, ax22 = axes
    
    ax21.plot(xi_grid, ue_grid**2 / he_grid, marker='x')
    ax21.set_title(r"$u_{e}^{2} / h_{e}$")
    
    delta_xi = xi_grid[1:] - xi_grid[:-1]
    ax22.plot(xi_grid[1:], 2. * xi_grid[1:] / delta_xi, marker='x')
    ax22.set_title(r"$2 \xi / \Delta \xi$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([xi0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    #
    grid_size = edge_grid.shape[1]
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    ax1, ax2 = axes
    
    ax1.plot(xi_grid, marker='x')
    ax1.set_title(r"$\xi$")
    
    ax2.plot(x_grid, marker='x')
    ax2.set_title(r"$x$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([0, grid_size-1])
    
    plt.show()


###
#
#

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 2:
        print("Usage: view_edge.py <edge_data_file.h5>")
        sys.exit(1)
    
    filename = "edge_grid.h5"
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    
    try:
        edge_grid, edge_indices = read_edge_file(filename)
        view_edge_conditions(edge_grid, edge_indices)
    except FileNotFoundError:
        print(f"File {filename:s} not found.")
        sys.exit(1)