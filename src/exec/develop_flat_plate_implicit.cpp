#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "flat_plate_factory.h"
#include "profile.h"

/*
 *
 * ./devel_profile -flag1 flag1_args ... ... -flagN -flagN_args where
 *    flag : n,     flag_args (1) : nb_steps
 *    flag : eta,   flag_args (1) : min_eta_step
 *    flag : fpp0,  flag_args (1) : f''(0)
 *    flag : gp0,   flag_args (1) : g'(0)
 *    flag : g0,    flag_args (1) : g(0)
 *    flag : wadiab flag_args (0)
 *
 */
int main(int argc, char *argv[]) {

  ProfileParams profile_params;
  profile_params.SetDefault();
  profile_params.ParseCmdInputs(argc, argv);

  int max_steps = profile_params.nb_steps;
  int profile_size = 0;

  std::vector<double> score(2);
  bool converged = false;

  //
  FlatPlate flat_plate = FlatPlateFactory(max_steps, "default");

  // profile_params.fpp0 = 0.45;
  // profile_params.gp0 = 0.38;

  profile_size =
      flat_plate.DevelopProfileImplicit(profile_params, score, converged);

  printf("Profile size: %d, Score: [%.5e, %.5e].\n", profile_size, score[0],
         score[1]);

  return 0;
}
