#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "boundary_data_struct.hpp"
#include "boundary_layer_factory.hpp"
#include "file_io.hpp"
#include "parsing.hpp"
#include "profile_struct.hpp"
#include "timers.hpp"

#include "profile_functions_cpg.hpp"

using std::vector;

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

bool consistent_outcomes(const vector<double> &score1, int psize_1,
                         const vector<double> &score2, int psize_2) {
  return (score1[0] == score2[0]) && (score1[1] == score2[1]) &&
         (psize_1 == psize_2);
}

struct ProgramParams {
  int station_id{1};
  string edge_file{"edge_grid.h5"};

  int nb_reps{1};

  void Parse(int argc, char *argv[]) {
    ParseValues(argc, argv, {{"-station", &station_id}, {"-nrep", &nb_reps}});
    assert(station_id >= 0);
    assert(nb_reps > 0);

    ParseValues(argc, argv, {{"-e", &edge_file}});
  }
};

int main(int argc, char *argv[]) {

  //
  ProgramParams program_params;
  program_params.Parse(argc, argv);

  // Parse Profile parameters
  ProfileParams profile_params;
  profile_params.scheme = TimeScheme::Implicit;
  profile_params.ParseCmdInputs(argc, argv);

  //
  BoundaryLayer boundary_layer =
      BoundaryLayerFactory(profile_params.nb_steps, "cpg");

  // Set profile params from station data
  BoundaryData boundary_data(program_params.edge_file);

  int station_id = program_params.station_id;
  profile_params.solve_type =
      (station_id == 0) ? SolveType::SelfSimilar : SolveType::LocallySimilar;

  profile_params.ReadEdgeConditions(boundary_data.edge_field,
                                    station_id * EDGE_FIELD_RANK);
  profile_params.ReadWallConditions(boundary_data.wall_field, station_id);

  //
  const int nb_reps = program_params.nb_reps;

  // Run default version
  vector<double> base_score(2);
  int base_psize = boundary_layer.DevelopProfile(profile_params, base_score);

  printf("Profile evolved to eta_id=%f, score: [%.5e, %.5e], norm=%.5e.\n\n",
         boundary_layer.GetEtaGrid()[base_psize], base_score[0], base_score[1],
         vector_norm(base_score));

  // Data structures for different tasks
  std::map<DevelMode, vector<double>> scores;
  scores.insert({DevelMode::Full, {0., 0.}});
  scores.insert({DevelMode::Primal, {0., 0.}});

  std::map<DevelMode, int> psizes;
  psizes.insert({DevelMode::Full, 0});
  psizes.insert({DevelMode::Primal, 0});

  int psize;
  vector<double> score;

  // Run devel for different modes
  for (const DevelMode &dev_mode : {DevelMode::Full, DevelMode::Primal}) {
    profile_params.devel_mode = dev_mode;

    psize = psizes.at(dev_mode);
    score = scores.at(dev_mode);

    auto full_devel_task = [&psize, &profile_params, &score, &boundary_layer,
                            nb_reps]() {
      for (int rep_id = 0; rep_id < nb_reps; rep_id++) {
        psize = boundary_layer.DevelopProfile(profile_params, score);
      }
    };

    auto avg_duration = timeit(full_devel_task, 1);

    std::cout << "\n"
              << to_string(dev_mode) << " devel task took " << avg_duration
              << " seconds.\n";

    scores.at(dev_mode) = score;
    psizes.at(dev_mode) = psize;

    assert(consistent_outcomes(base_score, base_psize, scores.at(dev_mode),
                               psizes.at(dev_mode)));
  }

  return 0;
}
