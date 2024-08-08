#include <iostream>

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

#include "atmosphere.h"
#include "boundary_layer_factory.h"
#include "case_functions.h"
#include "profile_struct.h"
#include "search_struct.h"

#include <vector>

int main(int argc, char *argv[]) {

  SearchWindow search_window;
  search_window.SetDefault();
  search_window.ParseCmdInputs(argc, argv);

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
  int xi_dim = 5;

  BoundaryData boundary_data =
      GenFlatPlateConstant(profile_params.ue, profile_params.he,
                           profile_params.pe, profile_params.g0, xi_dim);

  std::vector<std::vector<double>> bl_state_grid(
      xi_dim, std::vector<double>(BL_RANK * (eta_dim + 1), 0.));

  boundary_layer.ComputeLS(boundary_data.edge_field, boundary_data.wall_field,
                           profile_params, search_params, bl_state_grid);
}