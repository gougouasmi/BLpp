#include <vector>

int develop_profile(std::vector<double> &initial_guess,
                    std::vector<double> &state_grid,
                    std::vector<double> &eta_grid, std::vector<double> &rhs,
                    std::vector<double> &score, bool &converged);