#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "atmosphere.h"
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

  // Parse Profile parameters
  ProfileParams profile_params;
  profile_params.SetDefault();
  profile_params.ParseCmdInputs(argc, argv);

  double altitude_km = 5., mach_number = 0.2;
  ParseEntryParams(argc, argv, altitude_km, mach_number);
  SetEntryConditions(altitude_km, mach_number, profile_params);

  printf("Entry conditions: altitude=%.2f km, mach_number=%.2f\n\n",
         altitude_km, mach_number);

  // Build FlatPlate instance and develop profile
  FlatPlate flat_plate = FlatPlateFactory(profile_params.nb_steps, "cpg");

  std::vector<double> score(2);
  bool converged = flat_plate.DevelopProfile(profile_params, score);

  if (converged)
    printf("Profile converged, score: [%.5e, %.5e].\n", score[0], score[1]);
  else
    printf("Profile did not converge.\n");

  return 0;
}
