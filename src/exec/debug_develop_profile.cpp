#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "atmosphere.h"
#include "boundary_layer_factory.h"
#include "case_functions.h"
#include "profile.h"
#include "search_struct.h"

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

void ParseStationId(int argc, char *argv[], int &station_id) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-station") {
      if (i + 1 < argc) {
        station_id = std::stoi(argv[++i]);
        printf("station id set to %d.\n", station_id);
      } else {
        printf("profile nb_steps spec is incomplete.\n");
      }
    }
  }
}

int main(int argc, char *argv[]) {

  // Parse Profile parameters
  ProfileParams profile_params;
  profile_params.SetDefault();

  profile_params.solve_type = SolveType::LocallySimilar;
  profile_params.scheme = TimeScheme::Implicit;

  profile_params.ParseCmdInputs(argc, argv);

  // Build BoundaryLayer instance and develop profile
  BoundaryLayer boundary_layer =
      BoundaryLayerFactory(profile_params.nb_steps, "cpg");

  // Get data to play with
  BoundaryData boundary_data = GetFlatNosedCylinder(50, 2.);

  // Set profile params
  int station_id = 1;
  ParseStationId(argc, argv, station_id);

  int offset = 6 * station_id;

  profile_params.ue = boundary_data.edge_field[offset + 0];
  profile_params.he = boundary_data.edge_field[offset + 1];
  profile_params.pe = boundary_data.edge_field[offset + 2];
  profile_params.xi = boundary_data.edge_field[offset + 3];
  profile_params.due_dxi = boundary_data.edge_field[offset + 4];
  profile_params.dhe_dxi = boundary_data.edge_field[offset + 5];

  profile_params.g0 = boundary_data.wall_field[station_id];

  //
  std::vector<double> score(2);
  int profile_size =
      boundary_layer.DevelopProfile(profile_params, score, 0, true);

  double s0 = score[0], s1 = score[1];
  double snorm = pow(s0 * s0 + s1 * s1, 0.5);
  printf("Profile evolved to eta_id=%d, score: [%.5e, %.5e], norm=%.5e.\n",
         profile_size, s0, s1, snorm);

  //
  // SearchParams search_params;
  // search_params.SetDefault();

  // vector<double> guess(2, 0.5);

  // int worker_id = boundary_layer.GradientProfileSearch(profile_params,
  //                                                      search_params, guess);

  return 0;
}