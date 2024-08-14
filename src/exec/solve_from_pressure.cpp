#include <iostream>

#include "edge_solvers.h"
#include "file_io.h"

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

  ComputeFromPressure(pressure_field, density_field, velocity_field);

  vector<vector<double>> output_data;
  output_data.push_back(body_grid);
  output_data.push_back(pressure_field);
  output_data.push_back(density_field);
  output_data.push_back(velocity_field);

  const std::string output_path = "output.csv";
  WriteCSV(output_path, output_data);
}