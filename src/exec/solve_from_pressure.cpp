#include <iostream>

#include "edge_solvers.h"
#include "file_io.h"

#include <cmath>
#include <string>
#include <vector>

using std::vector;

int main(int argc, char *argv[]) {
  string file_path = "pressure_distribution.csv";
  vector<vector<double>> csv_data = ReadCSV(file_path);

  int nb_rows, nb_cols;
  GetDimsCSV(file_path, nb_rows, nb_cols);
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
    printf("%d: p = %.3e (dp/ds = %.2e), ro = %.3e, u = %.3e.\n", xid, pe,
           dp_ds, roe, ue);
  }

  bool write_to_file = false;
  if (write_to_file) {
    vector<vector<double>> output_data;
    output_data.push_back(body_grid);
    output_data.push_back(pressure_field);
    output_data.push_back(density_field);
    output_data.push_back(velocity_field);

    const std::string output_path = "output_constant_density.csv";
    WriteCSV(output_path, output_data);
  }
}