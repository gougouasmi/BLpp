#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

#include "atmosphere.h"
#include "boundary_layer_factory.h"
#include "search_struct.h"
#include "timers.h"

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

  // (1 / 4) Serial solution (box)
  array<double, 2> best_guess{{0.5, 0.5}};

  auto serial_task = [&profile_params, &search_params, &best_guess,
                      &boundary_layer]() {
    boundary_layer.BoxProfileSearch(profile_params, search_params, best_guess);
  };

  double avg_duration = timeit(serial_task, 1);

  std::cout << "Serial search took " << avg_duration << " seconds."
            << std::endl;

  printf("\n");

  // (2 / 4) Parallel solution (box)
  array<double, 2> parallel_best_guess{{0.5, 0.5}};

  auto parallel_task = [&profile_params, &search_params, &parallel_best_guess,
                        &boundary_layer]() {
    boundary_layer.BoxProfileSearchParallel(profile_params, search_params,
                                            parallel_best_guess);
  };

  avg_duration = timeit(parallel_task, 1);

  std::cout << "Parallel search took " << avg_duration << " seconds."
            << std::endl;

  printf("\n");

  // (3 / 4) Parallel solution with queues (box)
  array<double, 2> parallel_queues_best_guess{{0.5, 0.5}};

  auto parallel_queues_task = [&profile_params, &search_params,
                               &parallel_queues_best_guess, &boundary_layer]() {
    boundary_layer.BoxProfileSearchParallelWithQueues(
        profile_params, search_params, parallel_queues_best_guess);
  };

  avg_duration = timeit(parallel_queues_task, 1);

  std::cout << "Parallel search with queues took " << avg_duration
            << " seconds." << std::endl;

  // (4 / 4) Serial solution (gradient)
  array<double, 2> gradient_best_guess{{0.5, 0.5}};

  auto gradient_task = [&profile_params, &search_params, &gradient_best_guess,
                        &boundary_layer]() {
    boundary_layer.GradientProfileSearch(profile_params, search_params,
                                         gradient_best_guess);
  };

  avg_duration = timeit(gradient_task, 1);

  std::cout << "Serial gradient search took " << avg_duration << " seconds."
            << std::endl;

  return 0;
}