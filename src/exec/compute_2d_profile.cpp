#include <iostream>

#include "atmosphere.hpp"
#include "boundary_layer_factory.hpp"
#include "case_functions.hpp"
#include "file_io.hpp"
#include "profile_struct.hpp"
#include "search_struct.hpp"
#include "timers.hpp"

#include <sstream>
#include <vector>

/*
 * ./compute_boundary_layer -flag1 flag1_args ... -flagN flagN_args where
 *
 * Flow configuration parameters:
 *   flag: altitude, flag_args(1) altitude_km
 *   flag: mach, flag_args(1) mach_number
 *   flag: wall, flag_args(1) wall_type
 *
 * Initial profile parameters:
 *   flag: fpp0, flag_args(1) f''(0)
 *   flag: gp0,  flag_args(1) g'(0)
 *   flag: g0,   flag_args(1) g(0)
 *
 * Grid parameters
 *   flag: eta_step, flag_args(1) eta step size
 *   flag: neta,     flag_args(1) nb of points along eta
 *   flag: xi_step,  flag_args(1) delta_xi
 *   flag: nxi,      flag_args(1) nb of points long xi
 *
 * Program computes boundary layer flow. It can write to file
 *   - loads (friction, thermal) along the wall.
 *   - the entire flow-field
 */

struct ComputeParams {
  SearchParams search_params;
  ProfileParams profile_params;
  string edge_file{"edge_grid.h5"};

  void Parse(int argc, char *argv[]) {
    search_params.ParseCmdInputs(argc, argv);
    profile_params.ParseCmdInputs(argc, argv);
    ParseValues(argc, argv, {{"-e", &edge_file}});
  }
};

int main(int argc, char *argv[]) {

  ComputeParams compute_params;
  compute_params.Parse(argc, argv);

  SearchParams &search_params = compute_params.search_params;
  ProfileParams &profile_params = compute_params.profile_params;
  const string &edge_file = compute_params.edge_file;

  // Build class instance
  int eta_dim = profile_params.nb_steps;
  BoundaryLayer boundary_layer = BoundaryLayerFactory(eta_dim, "cpg");

  // Compute 2D profile
  BoundaryData boundary_data(edge_file);

  const int xi_dim = boundary_data.xi_dim;

  std::vector<std::vector<double>> bl_state_grid(
      xi_dim, std::vector<double>(BL_RANK * (eta_dim + 1), 0.));

  vector<SearchOutcome> search_outcomes;

  // Solve for 2D profile
  auto compute_task = [&boundary_layer, &boundary_data, &profile_params,
                       &search_params, &bl_state_grid, &search_outcomes]() {
    search_outcomes = boundary_layer.Compute(boundary_data, profile_params,
                                             search_params, bl_state_grid);
  };

  auto compute_duration = timeit(compute_task, 1);

  std::cout << "\nCompute task took " << compute_duration << " seconds.\n";

  // Write to file
  bool write_profiles = true;
  if (write_profiles) {
    WriteOutcomes(search_outcomes);

    boundary_layer.WriteEtaGrid(0);
    boundary_data.WriteEdgeConditions();

    for (int xi_id = 0; xi_id < xi_dim; xi_id++) {
      std::string state_filename("station_" + std::to_string(xi_id) + ".h5");
      boundary_layer.WriteStateGrid(state_filename, bl_state_grid[xi_id],
                                    search_outcomes[xi_id].profile_size);

      std::string output_filename("station_" + std::to_string(xi_id) +
                                  "_outputs.h5");
      boundary_layer.WriteOutputGrid(
          output_filename, bl_state_grid[xi_id], boundary_layer.GetEtaGrid(0),
          profile_params, search_outcomes[xi_id].profile_size);
    }
  }
}