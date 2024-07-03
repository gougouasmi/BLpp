#include <cstdlib>
#include <iostream>
#include <vector>

#include "profile.h"

int main(int argc, char *argv[]) {

  ProfileParams params;
  params.set_default();
  params.parse_cmd_inputs(argc, argv);

  int max_steps = params.nb_steps;
  int profile_size = 0;

  std::vector<double> state_grid(max_steps * SYSTEM_RANK);
  std::vector<double> eta_grid(max_steps);
  std::vector<double> rhs(SYSTEM_RANK);
  std::vector<double> score(2);

  bool converged = false;

  profile_size =
      develop_profile(params, state_grid, eta_grid, rhs, score, converged);

  printf("Profile size: %d, Score: [%.5e, %.5e].\n", profile_size, score[0],
         score[1]);

  return 0;
}
