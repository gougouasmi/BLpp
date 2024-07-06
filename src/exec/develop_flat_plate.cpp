#include <cstdlib>
#include <iostream>
#include <vector>

#include "flat_plate.h"
#include "profile.h"

int main(int argc, char *argv[]) {

  ProfileParams params;
  params.SetDefault();
  params.ParseCmdInputs(argc, argv);

  int max_steps = params.nb_steps;
  int profile_size = 0;

  std::vector<double> score(2);
  bool converged = false;

  //
  FlatPlate flat_plate(max_steps);

  profile_size = flat_plate.DevelopProfile(params, score, converged);

  printf("Profile size: %d, Score: [%.5e, %.5e].\n", profile_size, score[0],
         score[1]);

  return 0;
}
