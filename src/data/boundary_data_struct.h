#ifndef BOUNDARY_DATA_STRUCT_H
#define BOUNDARY_DATA_STRUCT_H

#include <cassert>
#include <vector>

using std::vector;

typedef struct BoundaryData {
  BoundaryData(vector<double> edge_vals, vector<double> wall_vals)
      : edge_field(edge_vals), wall_field(wall_vals), xi_dim(wall_vals.size()) {
    assert(wall_field.size() > 0);
    assert(wall_field.size() * 6 == edge_field.size());
  };
  vector<double> edge_field;
  vector<double> wall_field;
  int xi_dim;
} BoundaryData;

#endif