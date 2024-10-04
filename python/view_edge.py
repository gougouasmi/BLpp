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
    delta_x = x_grid[1:] - x_grid[:-1]
    delta_xi = xi_grid[1:] - xi_grid[:-1]   
    alpha_be = 2. * xi_grid[1:] / delta_xi

    #
    xi0_grid = xi_grid[2:]
    xi1_grid = xi_grid[1:-1]
    xi2_grid = xi_grid[0:-2]

    #
    lg2_0 = (2. * xi0_grid - xi1_grid - xi2_grid) / ((xi0_grid - xi1_grid) * (xi0_grid - xi2_grid))
    lg2_1 = (xi0_grid - xi2_grid) / ((xi0_grid - xi1_grid) * (xi1_grid - xi2_grid))
    lg2_2 = (xi0_grid - xi1_grid) / ((xi0_grid - xi2_grid) * (xi1_grid - xi2_grid))

    #
    x0_grid = x_grid[2:]
    x1_grid = x_grid[1:-1]
    x2_grid = x_grid[0:-2]

    #
    lg2_x0 = (2. * x0_grid - x1_grid - x2_grid) / ((x0_grid - x1_grid) * (x0_grid - x2_grid))
    lg2_x1 = (x0_grid - x2_grid) / ((x0_grid - x1_grid) * (x1_grid - x2_grid))
    lg2_x2 = (x0_grid - x1_grid) / ((x0_grid - x2_grid) * (x1_grid - x2_grid))

    vert_space = 5

    ## Flow conditions in physical space
    fig0, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))
    
    fig0.suptitle("Flow conditions (physical space)")

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

    ## Flow conditions in reference space  
    fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))
    
    fig1.suptitle(r"Flow conditions ($\xi$-space)")

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

    ## Gradients (physical space)
    fig5, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, vert_space))
    
    fig5.suptitle(r"Flow gradients wrt $x$")

    ax1, ax2 = axes

    ax1.plot(x_grid[1:], (ue_grid[1:] - ue_grid[0:-1]) / delta_x, marker='x', label=r"BE")
    ax1.plot(x_grid[2:], lg2_x0 * ue_grid[2:] - lg2_x1 * ue_grid[1:-1] + lg2_x2 * ue_grid[:-2], marker='x', label=r"LG2")

    ax1.legend()
    ax1.set_title(r"$\frac{du_{e}}{dx}$")
    
    ax2.plot(x_grid[1:], (he_grid[1:] - he_grid[0:-1]) / delta_x, marker='x', label=r"BE")
    ax2.plot(x_grid[2:], lg2_x0 * he_grid[2:] - lg2_x1 * he_grid[1:-1] + lg2_x2 * he_grid[:-2], marker='x', label=r"LG2")

    ax2.legend()
    ax2.set_title(r"$\frac{dh_{e}}{dx}$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([0, x1])
        ax.set_xlabel(r"$x$")
 

    ## Gradients 
    fig2, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))
    
    fig2.suptitle(r"Flow gradients wrt $\xi$")

    ax1, ax2, ax3 = axes

    ax1.plot(xi_grid, dxi_dx_grid, marker='x', label="data")
    ax1.plot(xi_grid[1:], (xi_grid[1:] - xi_grid[:-1]) / (x_grid[1:] - x_grid[:-1]), label=r"BE")
    ax1.plot(xi_grid[2:], 1. / (lg2_0 * x_grid[2:] - lg2_1 * x_grid[1:-1] + lg2_2 * x_grid[0:-2]) , label=r"1 / LG2")
    ax1.plot(xi_grid[2:], (lg2_x0 * xi_grid[2:] - lg2_x1 * xi_grid[1:-1] + lg2_x2 * xi_grid[0:-2]) , label=r"LG2")

    ax1.legend()
    ax1.set_title(r"$\frac{d\xi}{dx}$")

    ax2.plot(xi_grid, due_dxi_grid, marker='x', label=r"data")
    ax2.plot(xi_grid[1:], (ue_grid[1:] - ue_grid[0:-1]) / delta_xi, marker='x', label=r"BE")
    ax2.plot(xi_grid[2:], lg2_0 * ue_grid[2:] - lg2_1 * ue_grid[1:-1] + lg2_2 * ue_grid[:-2], marker='x', label=r"LG2")

    ax2.legend()
    ax2.set_title(r"$\frac{du_{e}}{d\xi}$")
    
    ax3.plot(xi_grid, dhe_dxi_grid, marker='x', label=r"data")
    ax3.plot(xi_grid[1:], (he_grid[1:] - he_grid[0:-1]) / delta_xi, marker='x', label=r"BE")
    ax3.plot(xi_grid[2:], lg2_0 * he_grid[2:] - lg2_1 * he_grid[1:-1] + lg2_2 * he_grid[:-2], marker='x', label=r"LG2")

    ax3.legend()
    ax3.set_title(r"$\frac{dh_{e}}{d\xi}$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([xi0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    ## Plot ODE coefficients
    c1_grid = 2. * xi_grid[1:] * due_dxi_grid[1:] / ue_grid[1:] 
    c1_grid_be = alpha_be * (ue_grid[1:] - ue_grid[0:-1])/ ue_grid[1:]
    c1_grid_lg2 = 2. * xi0_grid * (
        lg2_0 * ue_grid[2:] -
        lg2_1 * ue_grid[1:-1] +
        lg2_2 * ue_grid[0:-2]
    ) / ue_grid[2:]

    c2_grid = 2. * xi_grid[1:] * dhe_dxi_grid[1:] / he_grid[1:]
    c2_grid_be = alpha_be * (he_grid[1:] - he_grid[0:-1])/ he_grid[1:]
    c2_grid_lg2 = 2. * xi0_grid * (
        lg2_0 * he_grid[2:] -
        lg2_1 * he_grid[1:-1] +
        lg2_2 * he_grid[0:-2]
    ) / he_grid[2:]

    #c3_grid = 2. * xi_grid[1:] * due_dxi_grid[1:] * ue_grid[1:] / he_grid[1:]

    #
    fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))

    fig3.suptitle("Local-Similarity coefficients")

    ax11, ax12, ax13 = axes
    
    ax11.plot(xi_grid[1:], c1_grid, marker='x', label="data")
    ax11.plot(xi_grid[1:], c1_grid_be, marker='x', label=r"BE ($u_{e}$)")
    ax11.plot(xi_grid[2:], c1_grid_lg2, marker='x', label=r"LG2 ($u_{e}$)")
    ax11.legend()

    ax11.set_title(r"$c_{1}(\xi) := \frac{2 \xi}{u_{e}}\frac{du_{e}}{d\xi}$")
    
    ax12.plot(xi_grid[1:], c2_grid, marker='x', label="data")
    ax12.plot(xi_grid[1:], c2_grid_be, marker='x', label=r"BE ($h_{e}$)")
    ax12.plot(xi_grid[2:], c2_grid_lg2, marker='x', label=r"LG2 ($h_{e}$)")
    ax12.legend()

    ax12.set_title(r"$c_{2}(\xi) = \frac{2 \xi}{h_{e}} \frac{dh_{e}}{d\xi} = - c_{3}(\xi)$")

    ax13.plot(xi_grid, ue_grid**2 / he_grid, marker='x')
    ax13.set_title(r"$Ec = u_{e}^{2} / h_{e}$")
 
    for ax in axes:
        ax.grid(which="both")
        ax.set_xlim([0, xi1])
        ax.set_xlabel(r"$\xi$")
    
    #
    fig4, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, vert_space))

    fig4.suptitle(r"$2\xi \ ( \frac{\partial f}{\partial \xi} )$ approximations.")

    ax1, ax2 = axes
 
    ax1.plot(xi_grid[1:], alpha_be, marker='x', label=r"$\alpha_{0}$")
    ax1.set_title(r"BE : $\alpha_{0} ( f^{n} - f^{n-1} )$")

    ax2.plot(xi0_grid, 2. * xi0_grid * lg2_0, marker='x', label=r"$\alpha_{0}$")
    ax2.plot(xi0_grid, 2. * xi0_grid * lg2_1, marker='x', label=r"$\alpha_{1}$")
    ax2.plot(xi0_grid, 2. * xi0_grid * lg2_2, marker='x', label=r"$\alpha_{2}$")

    ax2.plot(xi0_grid, 2. * xi0_grid * (lg2_0 - lg2_1 + lg2_2), c='k', dashes=(2,2))

    ax2.set_title(r"LG2: $\alpha_{0} f^{n} - \alpha_{1} f^{n-1} + \alpha_{2} f^{n-1}$")

    for ax in axes:
        ax.legend()
        ax.grid(which="both")
        ax.set_xlim([0, xi1])
        ax.set_xlabel(r"$\xi$")
    
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