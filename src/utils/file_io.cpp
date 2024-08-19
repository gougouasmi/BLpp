#include "file_io.h"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

void GetDimsCSV(const string &file_path, int &nb_rows, int &nb_cols) {
  std::ifstream file(file_path);

  assert(file.is_open());

  string line, cell;

  nb_rows = 0;
  nb_cols = 0;

  // Get number of columns from first line
  if (std::getline(file, line)) {
    nb_rows += 1;

    std::stringstream line_stream(line);
    while (std::getline(line_stream, cell, ',')) {
      nb_cols += 1;
    }

    assert(nb_cols > 0);
  }
  assert(nb_rows > 0);

  // Navigate remaining lines and check that the number of columns
  // doesn't change
  while (std::getline(file, line)) {
    std::stringstream line_stream(line);

    int col_count = 0;
    while (std::getline(line_stream, cell, ',')) {
      col_count += 1;
    }

    assert(col_count == nb_cols);
    nb_rows += 1;
  }
}

vector<vector<double>> ReadCSV(const string &file_path) {
  int nb_rows, nb_cols;
  GetDimsCSV(file_path, nb_rows, nb_cols);

  vector<vector<double>> csv_data(nb_cols, vector<double>(nb_rows, 0.));

  std::ifstream file(file_path);
  string line, cell;

  for (int row_id = 0; row_id < nb_rows; row_id++) {
    std::getline(file, line);
    std::stringstream line_stream(line);

    for (int col_id = 0; col_id < nb_cols; col_id++) {
      std::getline(line_stream, cell, ',');
      csv_data[col_id][row_id] = std::stod(cell);
    }
  }

  return std::move(csv_data);
}

void WriteCSV(const string &file_path,
              const vector<vector<double>> &data_columns) {
  std::ofstream file(file_path);

  assert(file.is_open());

  int nb_cols = data_columns.size();
  assert(nb_cols > 0);

  int nb_rows = data_columns[0].size();
  for (int col_id = 1; col_id < nb_cols; col_id++) {
    assert(data_columns[col_id].size() == nb_rows);
  }

  for (int row_id = 0; row_id < nb_rows; row_id++) {
    for (int col_id = 0; col_id < nb_cols - 1; col_id++) {
      file << std::scientific << std::setprecision(6)
           << data_columns[col_id][row_id] << ", ";
    }
    file << std::scientific << std::setprecision(6)
         << data_columns[nb_cols - 1][row_id] << "\n";
  }

  file.close();
}