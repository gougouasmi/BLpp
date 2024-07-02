#include <cstdlib>
#include <iostream>
#include <vector>

#include "profile.h"

int main(int argc, char *argv[]) {

  int max_steps = 1000;
  int profile_size = 0;

  std::vector<double> initial_guess(2);
  if (argc < 2) {
    initial_guess[0] = 0.5;
    initial_guess[1] = 0.5;
  } else {
    initial_guess[0] = std::atof(argv[0]);
    initial_guess[1] = std::atof(argv[1]);
  }

  std::vector<double> state_grid(max_steps * 5);
  std::vector<double> eta_grid(max_steps);

  std::vector<double> rhs(5);
  std::vector<double> score(2);

  bool converged = false;

  profile_size = develop_profile(initial_guess, state_grid, eta_grid, rhs,
                                 score, converged);

  printf("Score: [%.2e, %.2e].\n", score[0], score[1]);

  return 0;
}
