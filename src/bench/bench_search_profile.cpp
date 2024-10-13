#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "atmosphere.hpp"
#include "boundary_layer_factory.hpp"
#include "search_struct.hpp"
#include "timers.hpp"

using std::array;

/*
 *
 * ./search_profile -flag1 flag1_args ... ... -flagN -flagN_args where
 *    flag : fbounds,    flag_args (2) : min_fpp0, max_fpp0
 *    flag : gbounds,    flag_args (2) : min_gp0, max_gp0
 *    flag : boxdims,    flag_args (3) : fdim, gdim
 *    flag : max_iter,   flag_args (1) : max_iter
 *    flag : res_tol,    flag_args (1) : res_tol
 *    flag : v (verbose) flag_args (0)
 *    flag : altitude,   flag_args (1) : altitude_km
 *    flag : mach,       flag_args (1) : mach_number
 *
 */

int main(int argc, char *argv[]) {

  SearchParams search_params;
  search_params.ParseCmdInputs(argc, argv);

  ProfileParams profile_params;
  profile_params.ParseCmdInputs(argc, argv);

  double altitude_km = 5., mach_number = 0.2;
  ParseEntryParams(argc, argv, altitude_km, mach_number);
  SetEntryConditions(altitude_km, mach_number, profile_params);

  printf("Entry conditions: altitude=%.2f km, mach_number=%.2f\n\n",
         altitude_km, mach_number);

  // Construct class instance
  BoundaryLayer boundary_layer =
      BoundaryLayerFactory(profile_params.nb_steps, "cpg");

  // Compare search methods
  std::map<SearchMethod, array<double, 2>> search_configs{
      {SearchMethod::BoxSerial, {0.5, 0.5}},
      {SearchMethod::BoxParallel, {0.5, 0.5}},
      {SearchMethod::BoxParallelQueue, {0.5, 0.5}},
      {SearchMethod::GradientSerial, {0.5, 0.5}},
      {SearchMethod::GradientExp, {0.5, 0.5}},
  };

  for (auto &search_config : search_configs) {
    SearchMethod method = search_config.first;
    array<double, 2> &guess = search_config.second;

    search_params.method = method;

    auto search_task = [&profile_params, &search_params, &guess,
                        &boundary_layer]() {
      boundary_layer.ProfileSearch(profile_params, search_params, guess);
    };

    double avg_duration = timeit(search_task, 1);

    std::cout << to_string(method) << " search took " << avg_duration
              << " seconds. Solution = [" << guess[0] << ", " << guess[1] << "]"
              << std::endl;

    printf("\n");
  }

  return 0;
}