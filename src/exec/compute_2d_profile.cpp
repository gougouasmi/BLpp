#include <iostream>

#include "atmosphere.h"
#include "boundary_layer_factory.h"
#include "case_functions.h"
#include "file_io.h"
#include "profile_struct.h"
#include "search_struct.h"

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

int main(int argc, char *argv[]) {

  SearchParams search_params;
  search_params.SetDefault();
  search_params.ParseCmdInputs(argc, argv);

  ProfileParams profile_params;
  profile_params.SetDefault();
  profile_params.ParseCmdInputs(argc, argv);

  double altitude_km = 5., mach_number = 0.2;
  ParseEntryParams(argc, argv, altitude_km, mach_number);
  SetEntryConditions(altitude_km, mach_number, profile_params);

  printf("Entry conditions: altitude=%.2f km, mach_number=%.2f\n\n",
         altitude_km, mach_number);

  int eta_dim = profile_params.nb_steps;

  // Build class instance
  BoundaryLayer boundary_layer = BoundaryLayerFactory(eta_dim, "cpg");

  // Compute 2D profile
  BoundaryData boundary_data = GetFlatNosedCylinder(50, 2., true);

  const int xi_dim = boundary_data.xi_dim;

  std::vector<std::vector<double>> bl_state_grid(
      xi_dim, std::vector<double>(BL_RANK * (eta_dim + 1), 0.));

  // Solve for 2D profile
  boundary_layer.Compute(boundary_data, profile_params, search_params,
                         bl_state_grid);

  // Post-process
  //  constexpr int OUTPUT_RANK = 4;
  //  std::vector<std::vector<double>> bl_output_grid(
  //      xi_dim, std::vector<double>(OUTPUT_RANK * (eta_dim + 1), 0.));
  //
  // boundary_layer.PostProcess(boundary_data, bl_state_grid, bl_output_grid);

  // Write to file
  bool write_profiles = true;
  if (write_profiles) {
    vector<double> eta_grid = boundary_layer.GetEtaGrid();

    // WriteCSV("eta_grid.csv", eta_grid, 1, eta_dim + 1);
    // WriteCSV("edge_grid.csv", boundary_data.edge_field, EDGE_FIELD_RANK,
    //          xi_dim);

    WriteH5("eta_grid.h5", eta_grid, "eta_grid");
    WriteH5("edge_grid.h5", boundary_data.edge_field, "edge_data", xi_dim,
            EDGE_FIELD_RANK);

    for (int xi_id = 0; xi_id < xi_dim; xi_id++) {
      // std::string state_filename("station_" + std::to_string(xi_id) +
      // ".csv"); WriteCSV(state_filename, bl_state_grid[xi_id], BL_RANK,
      // eta_dim + 1);

      std::string state_filename("station_" + std::to_string(xi_id) + ".h5");
      WriteH5(state_filename, bl_state_grid[xi_id], "state_data", eta_dim + 1,
              BL_RANK);

      // std::string output_filename("station_" + std::to_string(xi_id) +
      //                             "_outputs.csv");
      // WriteCSV(output_filename, bl_output_grid[xi_id], OUTPUT_RANK,
      //          eta_dim + 1);
    }
  }
}