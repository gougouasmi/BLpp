"""

View outcomes. 

"""

import numpy as np
import matplotlib.pyplot as plt

def view_outcomes(outcome_file: str) -> None:
    outcomes = np.loadtxt(
        outcome_file,
        delimiter=",",
        dtype={
            'names': ('success', 'worker', 'size', 'fpp0', 'gp0'),
            'formats': (bool, int, int, float, float),
        },
    )
    
    stations = np.array(range(0, outcomes.shape[0]))
    
    # Profile sizes
    fig, ax = plt.subplots(1, figsize=(5,5))
    
    ax.plot(stations, outcomes['size'], marker='o', mfc='none', ls='none')
    
    ax.set_title("Profile sizes")
    ax.set_xlabel('Station #')
    ax.grid(which="both")
    
    ax.set_xlim([0, stations[-1]])
    ax.set_ylim([0, None])
    
    # Values found
    fig, ax = plt.subplots(1, figsize=(5,5))
    
    ax.plot(stations, outcomes['fpp0'], marker='o', mfc='none', ls='none', color='b', label=r"$f''(0)$")
    ax.plot(stations, outcomes['gp0'], marker='o', mfc='none', ls='none', color='r', label=r"$g'(0)$")
    
    ax.set_title("Solutions")
    ax.set_xlabel('Station #')
    
    ax.grid(which="both")
    ax.legend()
    
    ax.set_xlim([0, stations[-1]])
    
    plt.show()

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 2:
        print("Usage: view_outcomes.py <search_outcomes.csv>")
        sys.exit(1)
    
    outcome_file = "search_outcomes.csv"
    if len(sys.argv) == 2:
        outcome_file = sys.argv[1]
    
    view_outcomes(outcome_file)