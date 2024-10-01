"""

Usage: main.py <filename.h5>

Will plot $f'(\eta) = u/u_e$, $g(\eta) = h / h_{e}$
and $C f''(\eta)$.

You can create a symbolic link to your use folder
-> ln -s <path_to_view_edge.py> .

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict

###
# Helper functions
#

def read_eta_file(filename: str = "eta_grid.h5") -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        eta_grid = f["eta_grid"][:]
    
    return eta_grid

def read_outputs(filename: str) -> Tuple[np.ndarray, Dict[str, int]]:
    with h5py.File(filename, 'r') as f:
        output_factors = np.transpose(f["data"][:])
        raw = f["field indices"][:]
        assert(f.attrs["description"] == "output fields")

    output_labels = {
        key.decode('utf-8'): key_id
        for key_id, key in zip(raw['index'], raw['label'])
    }

    return output_factors, output_labels   

def view_outputs(
    outputs: np.ndarray,
    eta_grid: np.ndarray,
    output_labels: Dict[str, int],
    station_factors: Dict[str, float] = None,
    fig_title: str="",
) -> None: 
    TAU_ID = output_labels["tau"]
    Q_ID = output_labels["q"]
    RO_ID = output_labels["ro"]
    Y_ID = output_labels["y"]
    C_ID = output_labels["C"]
    PR_ID = output_labels["Pr"]
    MU_ID = output_labels["mu"]
    
    profile_size = outputs.shape[1]

    # ro
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    fig.suptitle(fig_title)

    ax.plot(outputs[RO_ID, :], eta_grid[:profile_size], color='k')
    ax.set_title(r"$\rho$")
    
    ax.grid(which="both")
    ax.set_ylabel(r"$\eta$")
    ax.set_ylim([0, np.max(eta_grid)])
    
    # C, mu, Pr
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    fig.suptitle(fig_title)

    ax1, ax2, ax3 = axes
    
    ax1.plot(outputs[C_ID, :], eta_grid[:profile_size], color='k')
    ax1.set_title(r"$ C $")
    
    ax2.plot(outputs[MU_ID, :], eta_grid[:profile_size], color='k')
    ax2.set_title(r"$\mu$")
    
    ax3.plot(outputs[PR_ID, :], eta_grid[:profile_size], color='k')
    ax3.set_title(r"$Pr$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_ylabel(r"$\eta$")
        ax.set_ylim([0, np.max(eta_grid)])
 
    # y, q, \tau
    if (station_factors is not None):
        # y
        y_factor = station_factors["y_factor"]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

        fig.suptitle(fig_title)

        ax.plot(y_factor * outputs[Y_ID, :], eta_grid[:profile_size], color='k')
        ax.set_title(r"$y(\eta)$")

        ax.grid(which='both')
        ax.set_ylabel(r"$\eta$")
        ax.set_ylim([0, np.max(eta_grid)])

        # tau
        tau_factor = station_factors["tau_factor"]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

        fig.suptitle(fig_title)

        ax.plot(tau_factor * outputs[TAU_ID, :], eta_grid[:profile_size], color='k')
        ax.set_title(r"$\tau(\eta)$")

        ax.grid(which='both')
        ax.set_ylabel(r"$\eta$")
        ax.set_ylim([0, np.max(eta_grid)])

        # q
        q_factor = station_factors["q_factor"]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

        fig.suptitle(fig_title)

        ax.plot(q_factor * outputs[Q_ID, :], eta_grid[:profile_size], color='k')
        ax.set_title(r"$q(\eta)$")

        ax.grid(which='both')
        ax.set_ylabel(r"$\eta$")
        ax.set_ylim([0, np.max(eta_grid)])

    plt.show()
    
###
#
#

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 2:
        print("Usage: main.py <outputs_file.h5>")
        sys.exit(1)
    
    filename = "debug_outputs.h5"
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    
    try:
        eta_grid = read_eta_file()
    except FileNotFoundError:
        print(f"File eta_grid.h5 not found.")
        sys.exit(1)
    
    try:
        outputs, output_labels = read_outputs(filename)
        view_outputs(outputs, eta_grid, output_labels)
    except FileNotFoundError:
        print(f"File {filename:s} not found.")
        sys.exit(1)