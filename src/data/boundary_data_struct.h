#ifndef BOUNDARY_DATA_STRUCT_H
#define BOUNDARY_DATA_STRUCT_H

#include <cassert>
#include <vector>

using std::vector;

constexpr int EDGE_FIELD_RANK = 8;

constexpr int EDGE_U_ID = 0;
constexpr int EDGE_H_ID = 1;
constexpr int EDGE_P_ID = 2;
constexpr int EDGE_XI_ID = 3;
constexpr int EDGE_X_ID = 4;
constexpr int EDGE_DU_DXI_ID = 5;
constexpr int EDGE_DH_DXI_ID = 6;
constexpr int EDGE_DXI_DX_ID = 7;

typedef struct BoundaryData {
  BoundaryData(vector<double> edge_vals, vector<double> wall_vals)
      : edge_field(edge_vals), wall_field(wall_vals), xi_dim(wall_vals.size()) {
    assert(wall_field.size() > 0);
    assert(wall_field.size() * EDGE_FIELD_RANK == edge_field.size());
  };
  vector<double> edge_field;
  vector<double> wall_field;
  int xi_dim;
} BoundaryData;

#endif