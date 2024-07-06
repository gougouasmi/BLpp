#include "profile.h"

#include <algorithm>
#include <cassert>
#include <vector>

/*
 * Evolve the BL equations from an initial guess until it
 * converges and return the resulting profile.
 *
 */

double compute_rhs_default(std::vector<double> &state, std::vector<double> &rhs,
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
