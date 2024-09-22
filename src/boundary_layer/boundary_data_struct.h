#ifndef BOUNDARY_DATA_STRUCT_H
#define BOUNDARY_DATA_STRUCT_H

#include <cassert>
#include <string>
#include <vector>

#include "file_io.h"
#include "indexing.h"

using std::vector;

constexpr int EDGE_FIELD_RANK = 10;

constexpr int EDGE_U_ID = 0;
constexpr int EDGE_H_ID = 1;
constexpr int EDGE_P_ID = 2;
constexpr int EDGE_XI_ID = 3;
constexpr int EDGE_DU_DXI_ID = 4;
constexpr int EDGE_DH_DXI_ID = 5;

constexpr int EDGE_DXI_DX_ID = 6;
constexpr int EDGE_X_ID = 7;
constexpr int EDGE_RO_ID = 8;
constexpr int EDGE_MU_ID = 9;

constexpr std::array<int, EDGE_FIELD_RANK> EDGE_INDICES{
    EDGE_U_ID,      EDGE_H_ID,      EDGE_P_ID, EDGE_XI_ID, EDGE_DU_DXI_ID,
    EDGE_DH_DXI_ID, EDGE_DXI_DX_ID, EDGE_X_ID, EDGE_RO_ID, EDGE_MU_ID,
};
static_assert(complete_indexing(EDGE_INDICES));

struct BoundaryData {
  BoundaryData(vector<double> edge_vals, vector<double> wall_vals)
      : edge_field(edge_vals), wall_field(wall_vals), xi_dim(wall_vals.size()) {
    assert(wall_field.size() > 0);
    assert(wall_field.size() * EDGE_FIELD_RANK == edge_field.size());
  };
  vector<double> edge_field;
  vector<double> wall_field;
  int xi_dim;

  void WriteEdgeConditions(const std::string filepath = "edge_grid.h5") const {
    static vector<LabelIndex> edge_data_labels = {
        {"ue", EDGE_U_ID},           {"he", EDGE_H_ID},
        {"pe", EDGE_P_ID},           {"xi", EDGE_XI_ID},
        {"due/dxi", EDGE_DU_DXI_ID}, {"dhe/dxi", EDGE_DH_DXI_ID},
        {"dxi/dx", EDGE_DXI_DX_ID},  {"x", EDGE_X_ID},
        {"roe", EDGE_RO_ID},         {"mue", EDGE_MU_ID},
    };
    WriteH5(filepath, edge_field, edge_data_labels, xi_dim, EDGE_FIELD_RANK,
            "edge fields");
  }
};

#endif