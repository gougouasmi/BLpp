#include "case_functions.h"

#include <cassert>
#include <cmath>
#include <utility>

#include "atmosphere.h"
#include "file_io.h"
#include "gas_model.h"
#include "shock_relations.h"
#include "stagnation.h"

BoundaryData GenFlatPlateConstant(double ue, double he, double pe, double g0,
                                  int nb_points) {
  std::vector<double> edge_field(nb_points * 6, 0.);
  std::vector<double> wall_field(nb_points, g0);

  for (int xid = 0; xid < nb_points; xid++) {

    edge_field[6 * xid + 0] = ue; // ue
    edge_field[6 * xid + 1] = he; // he
    edge_field[6 * xid + 2] = pe; // pe

    edge_field[6 * xid + 3] = (double)xid / 10.; // xi
    edge_field[6 * xid + 4] = 0.;                // due_dxi
    edge_field[6 * xid + 5] = 0.;                // dhe_dxi
  }

  return BoundaryData(edge_field, wall_field);
}

BoundaryData GenChapmannRubesinFlatPlate(double mach, int nb_points,
                                         double prandtl) {
  std::vector<double> edge_field(nb_points * 6, 0.);
  std::vector<double> wall_field(nb_points, 0.);

  double dx = 1. / (double)(nb_points - 1);

  const double sound_speed = 340.;
  const double roe = 1.;

  const double gam = 1.4;
  const double ue = mach * sound_speed;
  const double he = sound_speed * sound_speed / (gam - 1);
  const double pe = roe * sound_speed * sound_speed / gam;

  const double mue = AIR_VISC(pe / (R_AIR * roe));

  const double dxi_dx = roe * ue * mue;

  const double recovery = sqrt(prandtl);
  const double gaw = 1 + 0.5 * recovery * (gam - 1) * mach * mach;

  for (int xid = 0; xid < nb_points; xid++) {
    edge_field[6 * xid + 0] = ue; // ue
    edge_field[6 * xid + 1] = he; // he
    edge_field[6 * xid + 2] = pe; // pe

    double xval = (xid * dx);

    edge_field[6 * xid + 3] = xval * dxi_dx; // xi
    edge_field[6 * xid + 4] = 0.;            // due_dxi
    edge_field[6 * xid + 5] = 0.;            // dhe_dxi

    wall_field[xid] = gaw * (1 + 0.25 - 0.83 * xval + 0.33 * xval * xval); // gw

    printf("boundary data at station #%d: xi=%.2e, ue=%.2e, he=%.2e, "
           "pe=%.2e, gw=%.2e.\n",
           xid, xval * dxi_dx, ue, he, pe, wall_field[xid]);
  }

  printf("\n");

  return BoundaryData(edge_field, wall_field);
}

const std::string FLAT_NOSED_PATH =
    "/Users/gouasmia/Documents/Work/Research/BLpp/src/data/flat_nosed_flow.csv";

BoundaryData GetFlatNosedCylinder(double altitude_km, double mach) {
  assert(mach > 1);
  assert(altitude_km > 0);

  double gamma = 1.4;

  // (1 / 5) Compute shock conditions
  double pre_pressure, pre_temperature;

  EarthPT(altitude_km, pre_temperature, pre_pressure);

  double pre_density = pre_pressure / (pre_temperature * R_AIR);
  double pre_sound_speed = sqrt(gamma * pre_pressure / pre_density);
  double pre_velocity = mach * pre_sound_speed;

  ShockRatios ratios = ComputeShockRatiosCPG(mach, 1.4);

  double post_pressure = pre_pressure * ratios.pressure;
  double post_density = pre_density * ratios.density;
  double post_velocity = pre_velocity * ratios.velocity;

  double post_sound_speed2 = gamma * post_pressure / post_density;
  double post_enthalpy = post_sound_speed2 / (gamma - 1.);

  // (2 / 5) Compute values at stagnation point
  double post_mach = post_velocity / sqrt(post_sound_speed2);

  double stag_pressure;
  double stag_density;
  double stag_enthalpy;

  ComputeStagnationRatios(post_mach, 1.4, stag_enthalpy, stag_density,
                          stag_pressure);

  printf("Enthalpy ratio %.2e, Post-shock enthalpy: %.2e, \n\n", stag_enthalpy,
         post_enthalpy);

  stag_enthalpy *= post_enthalpy;
  stag_density *= post_density;
  stag_pressure *= post_pressure;

  double v_scale = sqrt(stag_pressure / stag_density);

  // (3 / 5) Fetch flow along body eddge
  vector<vector<double>> csv_data = ReadCSV(FLAT_NOSED_PATH);

  vector<double> &body_grid = csv_data[0];
  vector<double> &pressure_field = csv_data[1];
  vector<double> &density_field = csv_data[2];
  vector<double> &velocity_field = csv_data[3];

  int grid_size = body_grid.size();

  // (5 / 5) Finish pre-processing
  std::vector<double> edge_field(grid_size * 6, 0.);
  std::vector<double> wall_field(grid_size, 0.2);

  //
  edge_field[6 * 0 + 0] = 0;
  edge_field[6 * 0 + 1] = stag_enthalpy;
  edge_field[6 * 0 + 2] = stag_pressure;

  for (int xid = 1; xid < grid_size; xid++) {
    //
    double dx = body_grid[xid] - body_grid[xid - 1];

    double roe = density_field[xid] * stag_density;
    double ue = velocity_field[xid] * v_scale;
    double he = stag_enthalpy - 0.5 * ue * ue;
    double pe = pressure_field[xid] * stag_pressure;

    double due_dx = (ue - velocity_field[xid - 1] * v_scale) / dx;
    double dhe_dx = -ue * due_dx;

    //
    double mue = AIR_VISC(pe / (roe * R_AIR));
    double dxi_dx = roe * ue * mue;

    edge_field[6 * xid + 0] = ue; // ue
    edge_field[6 * xid + 1] = he; // he
    edge_field[6 * xid + 2] = pe; // pe

    //
    edge_field[6 * xid + 3] = edge_field[6 * (xid - 1) + 3] + dx * dxi_dx; // xi
    edge_field[6 * xid + 4] = due_dx / dxi_dx; // due_dxi
    edge_field[6 * xid + 5] = dhe_dx / dxi_dx; // dhe_dxi

    printf("%d: %.2e, %.2e, %.2e, %.2e, %.2e, %.2e \n", xid, dx, roe, ue, he,
           pe, dxi_dx);
  }

  printf("\n");

  return BoundaryData(edge_field, wall_field);
}