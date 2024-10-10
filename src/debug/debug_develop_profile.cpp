#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "atmosphere.hpp"
#include "boundary_data_struct.hpp"
#include "boundary_layer_factory.hpp"
#include "case_functions.hpp"
#include "file_io.hpp"
#include "profile_struct.hpp"
#include "search_struct.hpp"

#include "profile_functions_cpg.hpp"

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
 *
 *  alias run_debug = './debug_devel_profile -implicit -eta 1e-3 -n 2000
 * -station 2 -verbose && python view_profile.py debug_profile.h5'
 *
 *
 */

void ParseStationId(int argc, char *argv[], int &station_id) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-station") {
      if (i + 1 < argc) {
        station_id = std::stoi(argv[++i]);
      } else {
        printf("profile nb_steps spec is incomplete.\n");
      }
    }
  }
}

int main(int argc, char *argv[]) {

  // Parse Profile parameters
  ProfileParams profile_params;

  profile_params.solve_type = SolveType::LocallySimilar;
  profile_params.scheme = TimeScheme::Implicit;

  profile_params.ParseCmdInputs(argc, argv);

  // Build BoundaryLayer instance and develop profile
  BoundaryLayer boundary_layer =
      BoundaryLayerFactory(profile_params.nb_steps, "cpg");

  // Get data to play with
  const char *flow_path = FLAT_NOSED_CONSTANT_RO_COARSE_PATH;
  BoundaryData boundary_data = GenFlatNosedCylinder(50, 2., flow_path);

  boundary_data.WriteEdgeConditions();

  // Set profile params
  int station_id = 1;
  ParseStationId(argc, argv, station_id);
  if (station_id == 0) {
    profile_params.solve_type = SolveType::SelfSimilar;
  }

  profile_params.ReadEdgeConditions(boundary_data.edge_field,
                                    station_id * EDGE_FIELD_RANK);
  profile_params.ReadWallConditions(boundary_data.wall_field, station_id);
  profile_params.PrintODEFactors();

  //
  std::vector<double> score(2);
  int profile_size = boundary_layer.DevelopProfile(profile_params, score);

  double s0 = score[0], s1 = score[1];
  double snorm = vector_norm(score);
  printf("Profile evolved to eta_id=%d, score: [%.5e, %.5e], norm=%.5e.\n",
         profile_size, s0, s1, snorm);

  //
  SearchParams search_params;
  search_params.ParseCmdInputs(argc, argv);

  array<double, 2> guess{{0.5, 0.5}};
  guess[0] = profile_params.fpp0;
  guess[1] = profile_params.gp0;

  SearchOutcome outcome = boundary_layer.GradientProfileSearch(
      profile_params, search_params, guess);

  if (!outcome.success) {
    printf("Search did not converge.\n");
  }

  int worker_id = outcome.worker_id;

  boundary_layer.WriteEtaGrid(worker_id);

  boundary_layer.WriteStateGrid("debug_profile.h5", worker_id);
  boundary_layer.WriteOutputGrid("debug_outputs.h5", profile_params,
                                 outcome.profile_size, worker_id);

  return 0;
}
