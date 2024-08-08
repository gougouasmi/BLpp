#ifndef CASE_FUNCTIONS_H
#define CASE_FUNCTIONS_H

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

using std::vector;

typedef struct BoundaryData {
  BoundaryData(vector<double> edge_vals, vector<double> wall_vals)
      : edge_field(edge_vals), wall_field(wall_vals) {
    assert(wall_field.size() > 0);
    assert(wall_field.size() * 6 == edge_field.size());
  };
  vector<double> edge_field;
  vector<double> wall_field;
} BoundaryData;

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
                                         double prandtl = 0.72) {
  std::vector<double> edge_field(nb_points * 6, 0.);
  std::vector<double> wall_field(nb_points, 0.);

  double dx = 1. / (double)(nb_points - 1);

  const double gam = 1.4;
  const double ue = mach;
  const double pe = 1. / 1.4;
  const double he = 1. / (gam - 1);

  const double recovery = sqrt(prandtl);
  const double gaw = 1 + 0.5 * recovery * (gam - 1) * mach * mach;

  for (int xid = 0; xid < nb_points; xid++) {
    edge_field[6 * xid + 0] = ue; // ue
    edge_field[6 * xid + 1] = he; // he
    edge_field[6 * xid + 2] = pe; // pe

    const double xval = xid * dx;

    edge_field[6 * xid + 3] = xval; // xi
    edge_field[6 * xid + 4] = 0.;   // due_dxi
    edge_field[6 * xid + 5] = 0.;   // dhe_dxi

    wall_field[xid] = gaw * (1 + 0.25 - 0.83 * xval + 0.33 * xval * xval); // gw
  }

  return BoundaryData(edge_field, wall_field);
}

#endif