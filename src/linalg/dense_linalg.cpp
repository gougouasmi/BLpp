#include "dense_linalg.h"

void DenseMatrixMultiply(const std::vector<double> &matrix_data_rm,
                         const std::vector<double> &input_vector,
                         std::vector<double> &output, int xdim) {
  int offset = 0;
  for (int i = 0; i < xdim; i++) {
    output[i] = 0;
    for (int j = 0; j < xdim; j++) {
      output[i] += matrix_data_rm[offset + j] * input_vector[j];
    }
    offset += xdim;
  }
}

void DenseMatrixMatrixMultiply(const std::vector<double> &matrix_data_rm,
                               const std::vector<double> &input_matrix_cm,
                               std::vector<double> &output_matrix_cm, int xdim,
                               int zdim) {
  assert(input_matrix_cm.size() == xdim * zdim);

  int cm_offset = 0;
  for (int k = 0; k < zdim; k++) {
    int offset = 0;
    for (int i = 0; i < xdim; i++) {
      output_matrix_cm[cm_offset + i] = 0.;
      for (int j = 0; j < xdim; j++) {
        output_matrix_cm[cm_offset + i] +=
            matrix_data_rm[offset + j] * input_matrix_cm[cm_offset + j];
      }
      offset += xdim;
    }

    cm_offset += xdim;
  }
}