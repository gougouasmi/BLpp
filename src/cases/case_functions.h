#ifndef CASE_FUNCTIONS_H
#define CASE_FUNCTIONS_H

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
                                  int nb_points);
BoundaryData GenChapmannRubesinFlatPlate(double mach, int nb_points,
                                         double prandtl = 0.72);
BoundaryData GetFlatNosedCylinder(double altitude, double mach);

#endif