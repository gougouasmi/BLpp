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

def read_profile(filename: str) -> Tuple[np.ndarray, Dict[str, int]]:
    with h5py.File(filename, 'r') as f:
        profile = np.transpose(f["data"][:])
        raw = f["field indices"][:]
        assert(f.attrs["description"] == "state fields")
    
    state_labels = {
        key.decode('utf-8'): key_id
        for key_id, key in zip(raw['index'], raw['label'])
    }

    return profile, state_labels

def view_state(
    state_grid: np.ndarray,
    eta_grid: np.ndarray,
    state_labels: Dict[str, int],
    fig_title: str="",
) -> None:
    FPP_ID = state_labels["C f''"]    
    FP_ID = state_labels["f'"]
    G_ID = state_labels["g"]
    GP_ID = state_labels["(C/Pr) g'"]

    profile_size = state_grid.shape[1]

    # f', g
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    fig.suptitle(fig_title)

    ax1, ax2 = axes
    
    ax1.plot(state_grid[FP_ID, :], eta_grid[:profile_size], color='k')
    ax1.set_title(r"$f'(\eta) \ = \ u / u_{e}$")
    
    ax2.plot(state_grid[G_ID, :], eta_grid[:profile_size], color='k')
    ax2.set_title(r"$g(\eta) \ = \ h / h_{e}$")

    for ax in axes:
        ax.grid(which="both")
        ax.set_ylim([0, np.max(eta_grid)])

    # f'', g'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    fig.suptitle(fig_title)

    ax1, ax2 = axes

    ax1.plot(state_grid[GP_ID, :], eta_grid[:profile_size], color='k')
    ax1.set_title(r"$\frac{C}{Pr} g'(\eta)$")

    ax2.plot(state_grid[FPP_ID, :], eta_grid[:profile_size], color='k')
    ax2.set_title(r"$C(\eta) f''(\eta)$")
 
    for ax in axes:
        ax.grid(which="both")
        ax.set_ylim([0, np.max(eta_grid)])

    plt.show()

###
#
#

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 2:
        print("Usage: main.py <debug_profile.h5>")
        sys.exit(1)
    
    filename = "debug_profile.h5"
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    
    try:
        eta_grid = read_eta_file()
    except FileNotFoundError:
        print(f"File eta_grid.h5 not found.")
        sys.exit(1)
    
    try:
        state_grid, state_labels = read_profile(filename)
        view_state(state_grid, eta_grid, state_labels)
    except FileNotFoundError:
        print(f"File {filename:s} not found.")
        sys.exit(1)