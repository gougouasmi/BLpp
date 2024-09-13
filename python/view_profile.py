"""

Usage: main.py <filename.h5>

Will plot $f'(\eta) = u/u_e$, $g(\eta) = h / h_{e}$
and $C f''(\eta)$.

You can create a symbolic link to your use folder
-> ln -s <path_to_view_edge.py> .

"""

import h5py
import numpy as np

###
#
#

FPP_ID, FP_ID, GP_ID, G_ID = 0, 2, 1, 4 

def read_eta_file(filename: str = "eta_grid.h5") -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        eta_grid = f["eta_grid"][:]
    
    return eta_grid

def read_profile(filename: str) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        profile = np.transpose(f["state_data"][:])

    return profile

###
#
#

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
    profile = read_profile(filename)
except FileNotFoundError:
    print(f"File {filename:s} not found.")
    sys.exit(1)

###
#
#

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

ax1, ax2, ax3 = axes

ax1.plot(profile[FP_ID, :], eta_grid, color='k')
ax1.set_title(r"$f'(\eta) \ = \ u / u_{e}$")

ax2.plot(profile[G_ID, :], eta_grid, color='k')
ax2.set_title(r"$g(\eta) \ = \ h / h_{e}$")

ax3.plot(profile[FPP_ID, :], eta_grid, color='k')
ax3.set_title(r"$C(\eta) f''(\eta)$")

for ax in axes:
    ax.grid(which="both")
    ax.set_ylim([0, np.max(eta_grid)])

plt.show()