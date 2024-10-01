#include <iostream>

#include "atmosphere.h"
#include "boundary_layer_factory.h"
#include "case_functions.h"
#include "file_io.h"
#include "profile_struct.h"
#include "search_struct.h"
#include "timers.h"

#include <sstream>
#include <vector>

int main(int argc, char *argv[]) {

  SearchParams search_params;
  search_params.ParseCmdInputs(argc, argv);
  search_params.rtol = 1e-4;
  search_params.scoring = Scoring::Exp;

  ProfileParams profile_params;
  profile_params.ParseCmdInputs(argc, argv);
  profile_params.scheme = TimeScheme::Implicit;
  profile_params.max_step = 1e-3;
  profile_params.nb_steps = 20000;

  int eta_dim = profile_params.nb_steps;

  // Build class instance
  BoundaryLayer boundary_layer = BoundaryLayerFactory(eta_dim, "cpg");

  // Compute 2D profile
  const char *flow_path =
      FLAT_NOSED_CONSTANT_RO_COARSE_PATH; // FLAT_NOSED_CONSTANT_RO_PATH;
  BoundaryData boundary_data = GenFlatNosedCylinder(50, 2., flow_path, false);

  const int xi_dim = boundary_data.xi_dim;

  std::vector<std::vector<double>> bl_state_grid(
      xi_dim, std::vector<double>(BL_RANK * (eta_dim + 1), 0.));

  vector<SearchOutcome> search_outcomes;

  // Solve for 2D profile
  int nb_workers = 1;
  auto compute_task = [&boundary_layer, &boundary_data, &profile_params,
                       &search_params, &bl_state_grid, &search_outcomes,
                       &nb_workers]() {
    search_outcomes = boundary_layer.ComputeLocalSimilarityParallel(
        boundary_data, profile_params, search_params, bl_state_grid,
        nb_workers);
  };

  for (const int nb_workers_val : {1, 2, 4, 6, 8}) {
    nb_workers = nb_workers_val;

    auto compute_duration = timeit(compute_task, 1);

    std::cout << "\nCompute task took " << compute_duration << " seconds with "
              << nb_workers << " threads.\n";
  }

  // Write to file
  bool write_profiles = false;
  if (write_profiles) {
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