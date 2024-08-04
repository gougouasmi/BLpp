#include <iostream>

/*
 * ./compute_flat_plate -flag1 flag1_args ... -flagN flagN_args where
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
#include "flat_plate.h"
#include "profile.h"
#include "profile_lsim.h"
#include "profile_search.h"
#include "timers.h"

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
  FlatPlate flat_plate =
      FlatPlate(eta_dim, compute_lsim_rhs_cpg, initialize_cpg,
                compute_lsim_rhs_jacobian_cpg);

  // Compute 2D profile
  int xi_dim = 5;

  std::vector<double> edge_field(xi_dim * 6, 0.);
  std::vector<std::vector<double>> bl_state_grid(
      xi_dim, std::vector<double>(FLAT_PLATE_RANK * (eta_dim + 1), 0.));

  for (int xid = 0; xid < xi_dim; xid++) {

    edge_field[6 * xid + 0] = profile_params.ue; // ue
    edge_field[6 * xid + 1] = profile_params.he; // he
    edge_field[6 * xid + 2] = profile_params.pe; // pe

    edge_field[6 * xid + 3] = (double)xid / 10.; // xi
    edge_field[6 * xid + 4] = 0.;                // due_dxi
    edge_field[6 * xid + 5] = 0.;                // dhe_dxi
  }

  flat_plate.ComputeLS(edge_field, search_params, bl_state_grid);
}