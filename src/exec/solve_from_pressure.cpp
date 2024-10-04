#include <iostream>

#include "boundary_data_struct.h"
#include "case_functions.h"
#include "edge_solvers.h"
#include "file_io.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using std::vector;

/*
 * USAGE: ./edge_solve -p <pressure_file.csv> -e <edge_file.h5> -v
 *
 */

void ParseCmdInputs(int argc, char *argv[], string &p_file, string &e_file,
                    bool &verbose) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-p") {
      if (i + 1 < argc) {
        p_file = argv[++i];
        printf("input pressure file set to %s.\n", p_file.c_str());
      } else {
        printf("input pressure file path spec is incomplete.\n");
      }
    } else if (arg == "-e") {
      if (i + 1 < argc) {
        e_file = argv[++i];
        printf("output edge file set to %s.\n", e_file.c_str());
      } else {
        printf("output edge file path spec is incomplete.\n");
      }
    } else if (arg == "-v") {
      verbose = true;
    }
  }
}

int main(int argc, char *argv[]) {
  string pressure_path = "pressure_distribution.csv";
  string edge_path = "test_edge_grid.h5";
  bool verbose = false;

  ParseCmdInputs(argc, argv, pressure_path, edge_path, verbose);

  vector<vector<double>> csv_data = ReadCSV(pressure_path);

  int nb_rows, nb_cols;
  GetDimsCSV(pressure_path, nb_rows, nb_cols);
  assert(nb_cols == 2);

  if (verbose)
    printf("\nCSV data dims : %d rows, %d columns.\n\n", nb_rows, nb_cols);

  vector<double> body_grid = csv_data[0];
  vector<double> pressure_field = csv_data[1];

  int grid_size = body_grid.size();
  vector<double> density_field(grid_size, 0.);
  vector<double> velocity_field(grid_size, 0.);

  density_field[0] = 1.;

  double gamma_ref = 1.05;
  int solve_size = ComputeFromPressureConstantDensity(
      pressure_field, density_field, velocity_field, gamma_ref);

  for (int xid = 0; xid < solve_size; xid++) {
    double pe = pressure_field[xid];
    double dp_ds;
    if (xid == 0) {
      dp_ds = (pressure_field[1] - pressure_field[0]) /
              (body_grid[1] - body_grid[0]);
    } else {
      dp_ds = (pressure_field[xid + 1] - pressure_field[xid]) /
              (body_grid[xid + 1] - body_grid[xid]);
    }

    double roe = density_field[xid];
    double ue = velocity_field[xid];
    if (verbose)
      printf("%d: p = %.3e (dp/ds = %.2e), ro = %.3e, u = %.3e.\n", xid, pe,
             dp_ds, roe, ue);
  }

  vector<vector<double>> csv_flow_data;
  csv_flow_data.push_back(body_grid);
  csv_flow_data.push_back(pressure_field);
  csv_flow_data.push_back(density_field);
  csv_flow_data.push_back(velocity_field);

  BoundaryData boundary_data = GenFlatNosedCylinder(50, 2.0, csv_flow_data);

  boundary_data.WriteEdgeConditions(edge_path);

  bool write_to_file = false;
  if (write_to_file) {
    const std::string output_path = "output_constant_density.csv";
    WriteCSV(output_path, csv_flow_data);
  }
}