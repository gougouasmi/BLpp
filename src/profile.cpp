#include "profile.h"

#include <algorithm>
#include <cassert>
#include <vector>

/*
 * Evolve the BL equations from an initial guess until it
 * converges and return the resulting profile.
 *
 */

void initialize_state(std::vector<double> &state,
                      ProfileParams &profile_params) {
  double fpp0 = profile_params.fpp0;
  double gp0 = profile_params.gp0;

  double fp0 = 0;
  double g0 = 0.2;

  double romu0 = 1.;
  double prandtl0 = 1.;

  state[FPP_ID] = romu0 * fpp0;
  state[FP_ID] = fp0;
  state[F_ID] = 0;
  state[GP_ID] = (romu0 / prandtl0) * gp0;
  state[G_ID] = g0;
}

double compute_rhs(std::vector<double> &state, std::vector<double> &rhs,
                   int offset) {
  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[offset + FP_ID];
  double g = state[offset + G_ID];

  double fpp = state[offset + FPP_ID] / romu;
  double f = state[offset + F_ID];
  double gp = state[offset + GP_ID] / romu * prandtl;

  rhs[FPP_ID] = -f * fpp;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void print_state(std::vector<double> &state, int offset) {
  printf("%f %f %f %f %f\n", state[offset], state[offset + 1],
         state[offset + 2], state[offset + 3], state[offset + 4]);
}

int develop_profile(ProfileParams &profile_params,
                    std::vector<double> &state_grid,
                    std::vector<double> &eta_grid, std::vector<double> &rhs,
                    std::vector<double> &score, bool &converged) {
  assert(profile_params.valid());

  converged = false;
  int nb_steps = eta_grid.size() - 1;

  assert(state_grid.size() / (nb_steps + 1) == SYSTEM_RANK);

  initialize_state(state_grid, profile_params);

  double min_step = profile_params.min_eta_step;
  double eta_step = 1.;

  int step_id = 0;
  int offset = 0;
  while (step_id < nb_steps) {

    // Compute rhs and limit time step
    eta_step = std::min(min_step, compute_rhs(state_grid, rhs, offset));

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;
    for (int var_id = 0; var_id < SYSTEM_RANK; ++var_id) {
      state_grid[offset + SYSTEM_RANK + var_id] =
          state_grid[offset + var_id] + eta_step * rhs[var_id];
    }

    // Update indexing
    offset += SYSTEM_RANK;
    step_id += 1;

    // Check convergence
    double rate = sqrt(rhs[FP_ID] * rhs[FP_ID] + rhs[G_ID] * rhs[G_ID]);
    converged = rate < 1e-3;
    if (converged)
      break;
  }

  // Compute score
  score[0] = state_grid[offset + FP_ID] - 1.;
  score[1] = state_grid[offset + G_ID] - 1;

  return step_id;
}