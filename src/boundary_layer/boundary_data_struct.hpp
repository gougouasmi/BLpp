#ifndef BOUNDARY_DATA_STRUCT_HPP
#define BOUNDARY_DATA_STRUCT_HPP

#include <cassert>
#include <string>
#include <vector>

#include "file_io.hpp"
#include "indexing.hpp"

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

static const vector<LabelIndex> edge_data_labels = {
    {"ue", EDGE_U_ID},           {"he", EDGE_H_ID},
    {"pe", EDGE_P_ID},           {"xi", EDGE_XI_ID},
    {"due/dxi", EDGE_DU_DXI_ID}, {"dhe/dxi", EDGE_DH_DXI_ID},
    {"dxi/dx", EDGE_DXI_DX_ID},  {"x", EDGE_X_ID},
    {"roe", EDGE_RO_ID},         {"mue", EDGE_MU_ID},
};

struct BoundaryData {
  vector<double> edge_field;
  vector<double> wall_field;
  int xi_dim;

  BoundaryData(vector<double> &&edge_vals, vector<double> &&wall_vals)
      : edge_field(std::move(edge_vals)), wall_field(std::move(wall_vals)) {
    xi_dim = wall_field.size();
    assert(wall_field.size() > 0);
    assert(wall_field.size() * EDGE_FIELD_RANK == edge_field.size());
  };

  BoundaryData(const std::string filepath = "edge_grid.h5")
      : edge_field(ReadH5(filepath, edge_data_labels, EDGE_FIELD_RANK)) {
    xi_dim = static_cast<int>(edge_field.size()) / EDGE_FIELD_RANK;
    wall_field.resize(xi_dim);
    std::fill(wall_field.begin(), wall_field.end(), 0.2);
  }

  void WriteEdgeConditions(const std::string filepath = "edge_grid.h5") const {
    WriteH5(filepath, edge_field, edge_data_labels, xi_dim, EDGE_FIELD_RANK,
            "edge fields");
  }
};

#endif