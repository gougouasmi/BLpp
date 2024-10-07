#include "file_io.h"

#include "H5Cpp.h"

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

  const int nb_cols = data_columns.size();
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

void WriteCSV(const string &file_path, const vector<double> &data, int rank,
              int nb_points) {
  assert(data.size() >= rank * nb_points);

  std::ofstream file(file_path);
  assert(file.is_open());

  const int nb_cols = rank;
  assert(nb_cols > 0);

  const int nb_rows = nb_points;

  int offset = 0;
  for (int row_id = 0; row_id < nb_rows; row_id++) {
    for (int col_id = 0; col_id < nb_cols - 1; col_id++) {
      file << std::scientific << std::setprecision(6) << data[offset + col_id]
           << ", ";
    }
    file << std::scientific << std::setprecision(6)
         << data[offset + nb_cols - 1] << "\n";

    offset += rank;
  }

  file.close();
}

void WriteCSV(const string &file_path, const vector<double> &data, int rank,
              int nb_points, const vector<int> &var_indices) {
  assert(data.size() >= rank * nb_points);

  const int nb_cols = var_indices.size();
  assert(nb_cols > 0);
  assert(nb_cols <= rank);

  std::ofstream file(file_path);
  assert(file.is_open());

  const int nb_rows = nb_points;

  int offset = 0;
  for (int row_id = 0; row_id < nb_rows; row_id++) {
    for (int col_id = 0; col_id < nb_cols - 1; col_id++) {
      file << std::scientific << std::setprecision(6)
           << data[offset + var_indices[col_id]] << ", ";
    }
    file << std::scientific << std::setprecision(6)
         << data[offset + var_indices[nb_cols - 1]] << "\n";

    offset += rank;
  }

  file.close();
}

void WriteH5(const string &filepath, const vector<double> &data,
             const string &data_label) {
  H5::H5File file(filepath, H5F_ACC_TRUNC);

  hsize_t dims[1] = {data.size()};
  H5::DataSpace dataspace(1, dims); // 1D dataspace
  H5::PredType datatype(H5::PredType::NATIVE_DOUBLE);

  H5::DataSet dataset = file.createDataSet(data_label, datatype, dataspace);

  dataset.write(data.data(), datatype);
}

void WriteH5(const string &filepath, const vector<double> &data,
             const string &data_label, const size_t nb_points,
             const size_t rank) {
  assert(data.size() >= nb_points * rank);
  H5::H5File file(filepath, H5F_ACC_TRUNC);

  hsize_t dims[2] = {nb_points, rank};
  H5::DataSpace dataspace(2, dims); // 2D dataspace
  H5::PredType datatype(H5::PredType::NATIVE_DOUBLE);

  H5::DataSet dataset = file.createDataSet(data_label, datatype, dataspace);

  dataset.write(data.data(), datatype);
}

void WriteH5(const string &filepath, const vector<double> &data,
             const vector<LabelIndex> &data_labels, const size_t nb_points,
             const size_t rank, const string description) {
  assert(data.size() >= nb_points * rank);
  assert(data_labels.size() == rank);

  H5::H5File file(filepath, H5F_ACC_TRUNC);

  // Write data
  hsize_t data_dims[2] = {nb_points, rank};
  H5::DataSpace dataspace(2, data_dims);
  H5::PredType datatype(H5::PredType::NATIVE_DOUBLE);

  H5::DataSet dataset = file.createDataSet("data", datatype, dataspace);
  dataset.write(data.data(), datatype);

  // Write (variable_name, variable_id) pairs
  H5::CompType H5_LABEL_INDEX_PAIR(sizeof(LabelIndex));
  H5_LABEL_INDEX_PAIR.insertMember("index", HOFFSET(LabelIndex, index),
                                   H5::PredType::NATIVE_INT);
  H5_LABEL_INDEX_PAIR.insertMember(
      "label", HOFFSET(LabelIndex, label),
      H5::StrType(H5::PredType::C_S1, LABEL_CSIZE));

  hsize_t label_dims[1] = {rank};
  H5::DataSpace label_dataspace{1, label_dims};

  H5::DataSet label_dataset =
      file.createDataSet("field indices", H5_LABEL_INDEX_PAIR, label_dataspace);
  label_dataset.write(data_labels.data(), H5_LABEL_INDEX_PAIR);

  // Write attribute
  H5::StrType str_dtype(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute str_attribute =
      file.createAttribute("description", str_dtype, H5S_SCALAR);
  str_attribute.write(str_dtype, description);
}

bool inline H5_labels_are_consistent(H5::H5File &file,
                                     const vector<LabelIndex> &data_labels,
                                     const size_t rank) {
  // Open the dataset containing the label-index pairs
  H5::DataSet label_dataset = file.openDataSet("field indices");

  // Create a dataspace for reading the labels
  H5::DataSpace label_dataspace = label_dataset.getSpace();

  // Get the dimension of the label data
  hsize_t label_dims[1];
  label_dataspace.getSimpleExtentDims(label_dims, nullptr);

  assert(label_dims[0] == rank);

  // Read the label-index pairs from the file
  std::vector<LabelIndex> file_labels(rank);

  // Create the HDF5 compound type for LabelIndex
  H5::CompType H5_LABEL_INDEX_PAIR(sizeof(LabelIndex));
  H5_LABEL_INDEX_PAIR.insertMember("index", HOFFSET(LabelIndex, index),
                                   H5::PredType::NATIVE_INT);
  H5_LABEL_INDEX_PAIR.insertMember(
      "label", HOFFSET(LabelIndex, label),
      H5::StrType(H5::PredType::C_S1, LABEL_CSIZE));

  // Read the data
  label_dataset.read(file_labels.data(), H5_LABEL_INDEX_PAIR);

  // Compare the labels with the provided data_labels
  for (size_t i = 0; i < rank; ++i) {
    if (file_labels[i].index != data_labels[i].index ||
        std::strncmp(file_labels[i].label, data_labels[i].label, LABEL_CSIZE) !=
            0) {
      return false; // Inconsistent labels
    }
  }

  return true;
}

vector<double> ReadH5(const string &filepath,
                      const vector<LabelIndex> &data_labels,
                      const size_t rank) {
  //
  H5::H5File file(filepath, H5F_ACC_RDONLY);

  assert(H5_labels_are_consistent(file, data_labels, rank));

  //
  H5::DataSet dataset = file.openDataSet("data");
  H5::DataSpace dataspace = dataset.getSpace();

  //
  int ndims = dataspace.getSimpleExtentNdims();
  bool dataset_is_two_dimensional = (ndims == 2);
  assert(dataset_is_two_dimensional);

  // Get the dimensions of the dataset
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims, nullptr);

  const size_t nb_points = dims[0];
  const size_t file_rank = dims[1];

  //
  bool consistent_data_dimensions = (file_rank == rank);
  assert(consistent_data_dimensions);

  // Read the data into a vector<double>
  std::vector<double> data(nb_points * rank);
  H5::PredType datatype(H5::PredType::NATIVE_DOUBLE);
  dataset.read(data.data(), datatype);

  return std::move(data);
}