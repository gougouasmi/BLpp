#include <iostream>

#include "boundary_data_struct.hpp"
#include "case_functions.hpp"
#include "edge_solvers.hpp"
#include "file_io.hpp"
#include "parsing.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using std::vector;

/*
 * USAGE: ./edge_solve -p <pressure_file.csv> -e <edge_file.h5> -v
 *
 */
constexpr const char *USAGE =
    "\n** USAGE: ./edge_solve -p <pressure_file.csv> -e "
    "<edge_file.h5> -v **\n\n";

struct ProgramParams {
  string pressure_path;
  string edge_path{"edge_grid.h5"};
  bool verbose;

  void Parse(int argc, char *argv[]) {
    ParseUsage(argc, argv, USAGE);
    ParseValues(argc, argv, {{"-p", &pressure_path}, {"-e", &edge_path}});
    ParseOptions(argc, argv, {{"-v", &verbose}});
  }
};

int main(int argc, char *argv[]) {

  ProgramParams params;
  params.Parse(argc, argv);

  string pressure_path = params.pressure_path;
  string edge_path = params.edge_path;
  bool verbose = params.verbose;

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