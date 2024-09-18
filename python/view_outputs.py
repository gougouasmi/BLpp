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
        output_data = np.transpose(f["data"][:])
        raw = f["field indices"][:]
        assert(f.attrs["description"] == "output fields")

    output_labels = {
        key.decode('utf-8'): key_id
        for key_id, key in zip(raw['index'], raw['label'])
    }

    return output_data, output_labels   

def view_outputs(
    outputs: np.ndarray,
    eta_grid: np.ndarray,
    output_labels: Dict[str, int],
    fig_title: str="",
) -> None: 
    U_ID = output_labels["u/ue"]
    H_ID = output_labels["h/he"]
    RO_ID = output_labels["ro/roe"]
    Y_ID = output_labels["y"]
    C_ID = output_labels["C"]
    PR_ID = output_labels["Pr"]
    
    profile_size = outputs.shape[1]
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    
    fig.suptitle(fig_title)

    ax1, ax2, ax3 = axes
    
    ax1.plot(outputs[U_ID, :], eta_grid[:profile_size], color='k')
    ax1.set_title(r"$f'(\eta) \ = \ u / u_{e}$")
    
    ax2.plot(outputs[H_ID, :], eta_grid[:profile_size], color='k')
    ax2.set_title(r"$g(\eta) \ = \ h / h_{e}$")
    
    ax3.plot(outputs[C_ID, :], eta_grid[:profile_size], color='k')
    ax3.set_title(r"$C(\eta)$")
    
    for ax in axes:
        ax.grid(which="both")
        ax.set_ylim([0, np.max(eta_grid[:profile_size])])
    
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