"""

Use this script to compare properties
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

def compare_edge_conditions(
    edge_conditions_0: Tuple[np.ndarray, Dict[str, int], str],
    edge_conditions_1: Tuple[np.ndarray, Dict[str, int], str], 
) -> None:
    edge_grid_0, edge_indices_0, label_0 = edge_conditions_0
    edge_grid_1, edge_indices_1, label_1 = edge_conditions_1

    ###
    #
    #

    # First dataset
    ue_grid_0 = edge_grid_0[edge_indices_0["ue"], :]
    he_grid_0 = edge_grid_0[edge_indices_0["he"], :]
    pe_grid_0 = edge_grid_0[edge_indices_0["pe"], :]    
    x_grid_0  = edge_grid_0[edge_indices_0["x"] , :]
    xi_grid_0 = edge_grid_0[edge_indices_0["xi"], :]
    
    due_dxi_grid_0 = edge_grid_0[edge_indices_0["due/dxi"], :]
    dhe_dxi_grid_0 = edge_grid_0[edge_indices_0["dhe/dxi"], :]

    c1_grid_0 = 2. * xi_grid_0[1:] * due_dxi_grid_0[1:] / ue_grid_0[1:] 
    c2_grid_0 = 2. * xi_grid_0[1:] * dhe_dxi_grid_0[1:] / he_grid_0[1:]
    eck_0 = ue_grid_0**2 / he_grid_0

    grid_0_size = x_grid_0.shape[0]
    output_factors_0 = {
        'tau': np.zeros(grid_0_size), #,
        'q': np.zeros(grid_0_size), #,
        'y': np.zeros(grid_0_size), #,
    }

    dxi_dx_grid_0 = edge_grid_0[edge_indices_0["dxi/dx"], :]

    romu_stag_0 = (
        edge_grid_0[edge_indices_0["roe"], 0] *
        edge_grid_0[edge_indices_0["mue"], 0]
    )
 
    he_stag_0 = he_grid_0[0]
    due_dx_stag_0 = (ue_grid_0[1] - ue_grid_0[0]) / (x_grid_0[1] - x_grid_0[0])

    output_factors_0["tau"][0] = (romu_stag_0 ** 0.5) * (due_dx_stag_0 ** 1.5) * x_grid_0[0]
    output_factors_0["q"][0]   = he_stag_0 * (romu_stag_0 * due_dx_stag_0) ** 0.5
    output_factors_0["y"][0]   = (romu_stag_0 / due_dx_stag_0) ** 0.5

    output_factors_0["tau"][1:] = ue_grid_0[1:] * dxi_dx_grid_0[1:] / np.sqrt(2. * xi_grid_0[1:])
    output_factors_0["q"][1:]   = he_grid_0[1:] * dxi_dx_grid_0[1:] / np.sqrt(2. * xi_grid_0[1:])
    output_factors_0["y"][1:]   = np.sqrt(2. * xi_grid_0[1:]) / ue_grid_0[1:]

    # Second dataset
    ue_grid_1 = edge_grid_1[edge_indices_1["ue"], :]
    he_grid_1 = edge_grid_1[edge_indices_1["he"], :]
    pe_grid_1 = edge_grid_1[edge_indices_1["pe"], :]    
    x_grid_1  = edge_grid_1[edge_indices_1["x"] , :]
    xi_grid_1 = edge_grid_1[edge_indices_1["xi"], :]

    due_dxi_grid_1 = edge_grid_1[edge_indices_1["due/dxi"], :]
    dhe_dxi_grid_1 = edge_grid_1[edge_indices_1["dhe/dxi"], :]

    c1_grid_1 = 2. * xi_grid_1[1:] * due_dxi_grid_1[1:] / ue_grid_1[1:] 
    c2_grid_1 = 2. * xi_grid_1[1:] * dhe_dxi_grid_1[1:] / he_grid_1[1:]
    eck_1 = ue_grid_1**2 / he_grid_1

    grid_1_size = x_grid_1.shape[0]
    output_factors_1 = {
        'tau': np.zeros(grid_1_size), #,
        'q': np.zeros(grid_1_size), #,
        'y': np.zeros(grid_1_size), #,
    }

    dxi_dx_grid_1 = edge_grid_1[edge_indices_1["dxi/dx"], :]

    romu_stag_1 = (
        edge_grid_1[edge_indices_1["roe"], 0] *
        edge_grid_1[edge_indices_1["mue"], 0]
    )
 
    he_stag_1 = he_grid_1[0]
    due_dx_stag_1 = (ue_grid_1[1] - ue_grid_1[0]) / (x_grid_1[1] - x_grid_1[0])

    output_factors_1["tau"][0] = (romu_stag_1 ** 0.5) * (due_dx_stag_1 ** 1.5) * x_grid_1[0]
    output_factors_1["q"][0]   = he_stag_1 * (romu_stag_1 * due_dx_stag_1) ** 0.5
    output_factors_1["y"][0]   = (romu_stag_1 / due_dx_stag_1) ** 0.5

    output_factors_1["tau"][1:] = ue_grid_1[1:] * dxi_dx_grid_1[1:] / np.sqrt(2. * xi_grid_1[1:])
    output_factors_1["q"][1:]   = he_grid_1[1:] * dxi_dx_grid_1[1:] / np.sqrt(2. * xi_grid_1[1:])
    output_factors_1["y"][1:]   = np.sqrt(2. * xi_grid_1[1:]) / ue_grid_1[1:]

    # Bounds
    x1  = max(np.max(x_grid_0),  np.max(x_grid_1)) 
    xi1 = max(np.max(xi_grid_0), np.max(xi_grid_1))

    # Figure parameters
    vert_space = 5

    ###
    # Flow conditions in physical space
    #

    fig0, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))
    
    fig0.suptitle("Flow conditions (physical space)")

    ax1, ax2, ax3 = axes
    
    ax1.plot(x_grid_0, ue_grid_0, marker='x', label=label_0)
    ax1.plot(x_grid_1, ue_grid_1, marker='x', label=label_1)
    ax1.set_title(r"$u_{e}(x)$")
    
    ax2.plot(x_grid_0, he_grid_0, marker='x', label=label_0)
    ax2.plot(x_grid_1, he_grid_1, marker='x', label=label_1)
    ax2.set_title(r"$h_{e}(x)$")
    
    ax3.plot(x_grid_0, pe_grid_0, marker='x', label=label_0)
    ax3.plot(x_grid_1, pe_grid_1, marker='x', label=label_1)
    ax3.set_title(r"$p_{e}(x)$")
    
    for ax in axes:
        ax.legend()
        ax.grid(which="both")
        ax.set_xlim([0, x1])
        ax.set_xlabel(r"$x$")

    ###
    # Flow conditions in reference space  
    #
        
    fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))
    
    fig1.suptitle(r"Flow conditions ($\xi$-space)")

    ax1, ax2, ax3 = axes
    
    ax1.plot(xi_grid_0, ue_grid_0, marker='x', label=label_0)
    ax1.plot(xi_grid_1, ue_grid_1, marker='x', label=label_1) 
    ax1.set_title(r"$u_{e}(\xi)$")
    
    ax2.plot(xi_grid_0, he_grid_0, marker='x', label=label_0)
    ax2.plot(xi_grid_1, he_grid_1, marker='x', label=label_1)
    ax2.set_title(r"$h_{e}(\xi)$")
    
    ax3.plot(xi_grid_0, pe_grid_0, marker='x', label=label_0)
    ax3.plot(xi_grid_1, pe_grid_1, marker='x', label=label_1)
    ax3.set_title(r"$p_{e}(\xi)$")
    
    for ax in axes:
        ax.legend()
        ax.grid(which="both")
        ax.set_xlim([0, xi1])
        ax.set_xlabel(r"$\xi$")

    ###
    # Local-Similarity Coefficients
    #
        
    fig2, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))

    fig2.suptitle("Local-Similarity coefficients")

    ax1, ax2, ax3 = axes
    
    ax1.plot(xi_grid_0[1:], c1_grid_0, marker='x', label=label_0)
    ax1.plot(xi_grid_1[1:], c1_grid_1, marker='x', label=label_1)

    ax1.set_title(r"$c_{1}(\xi) := \frac{2 \xi}{u_{e}}\frac{du_{e}}{d\xi}$")
    
    ax2.plot(xi_grid_0[1:], c2_grid_0, marker='x', label=label_0)
    ax2.plot(xi_grid_1[1:], c2_grid_1, marker='x', label=label_1)

    ax2.set_title(r"$c_{2}(\xi) = \frac{2 \xi}{h_{e}} \frac{dh_{e}}{d\xi} = - c_{3}(\xi)$")

    ax3.plot(xi_grid_0, eck_0, marker='x', label=label_0)
    ax3.plot(xi_grid_1, eck_1, marker='x', label=label_1)

    ax3.set_title(r"Eckert factor $u_{e}^{2} / h_{e}$")
 
    for ax in axes:
        ax.legend()
        ax.grid(which="both")
        ax.set_xlim([0, xi1])
        ax.set_xlabel(r"$\xi$")

    ###
    # Output factors
    #

    fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, vert_space))

    fig3.suptitle("Output factors")

    ax1, ax2, ax3 = axes

    ax1.plot(x_grid_0, output_factors_0["tau"], marker='x', label=label_0)
    ax1.plot(x_grid_1, output_factors_1["tau"], marker='x', label=label_1)

    ax1.set_title(r"$\tau$")

    ax2.plot(x_grid_0, output_factors_0["q"], marker='x', label=label_0)
    ax2.plot(x_grid_1, output_factors_1["q"], marker='x', label=label_1)

    ax2.set_title(r"$q$")

    ax3.plot(x_grid_0, output_factors_0["y"], marker='x', label=label_0)
    ax3.plot(x_grid_1, output_factors_1["y"], marker='x', label=label_1)

    ax3.set_title(r"$y$")

    for ax in axes:
        ax.legend()
        ax.grid(which="both")
        ax.set_xlim([0, x1])
        ax.set_xlabel(r"$x$")


    plt.show()
 
###
#
#

if __name__ == "__main__":

    import sys, os
    
    if len(sys.argv) != 5:
        print("Usage: compare_edges.py <edge_data_file.h5> <label> <other_edge_data_file.h5> <other_label>")
        sys.exit(1)

    filename0 = sys.argv[1]
    if not os.path.exists(filename0):
        print(f"{filename0:s} not found.")
        sys.exit(1)

    label0 = sys.argv[2]

    filename1 = sys.argv[3]
    if not os.path.exists(filename1):
        print(f"{filename1:s} not found.")
        sys.exit()

    label1 = sys.argv[4]

    edge_grid_0, edge_indices_0 = read_edge_file(filename0)
    edge_grid_1, edge_indices_1 = read_edge_file(filename1)

    compare_edge_conditions(
        (edge_grid_0, edge_indices_0, label0),
        (edge_grid_1, edge_indices_1, label1),
    )

