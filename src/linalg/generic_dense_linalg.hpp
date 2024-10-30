#ifndef GENERIC_DENSE_LINALG_HPP
#define GENERIC_DENSE_LINALG_HPP

#include "generic_vector.hpp"

namespace Generic {

template <typename T, std::size_t ctime_xdim>
void DenseMatrixMultiply(
    const Generic::Vector<T, ctime_xdim * ctime_xdim> &matrix_data_rm,
    const Generic::Vector<T, ctime_xdim> &input_vector,
    Generic::Vector<T, ctime_xdim> &output_vector, const size_t xdim) {
  const size_t nb_rows = (ctime_xdim == 0) ? xdim : ctime_xdim;

  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    output_vector[i] = 0;
    for (int j = 0; j < nb_rows; j++) {
      output_vector[i] += matrix_data_rm[offset + j] * input_vector[j];
    }
    offset += nb_rows;
  }
}

template <typename T, std::size_t ctime_xdim, std::size_t ctime_zdim>
void DenseMatrixMatrixMultiply(
    const Generic::Vector<T, ctime_xdim * ctime_xdim> &matrix_data_rm,
    const Generic::Vector<T, ctime_xdim * ctime_zdim> &input_matrix_cm,
    Generic::Vector<T, ctime_xdim * ctime_zdim> &output_matrix_cm,
    const size_t xdim, const size_t zdim) {

  const size_t nb_rows = (ctime_xdim == 0) ? xdim : ctime_xdim;
  const size_t nb_input_cols = (ctime_zdim == 0) ? zdim : ctime_zdim;

  assert(input_matrix_cm.size() >= xdim * zdim);
  assert(output_matrix_cm.size() >= xdim * zdim);

  int cm_offset = 0;
  for (int k = 0; k < nb_input_cols; k++) {
    int offset = 0;
    for (int i = 0; i < nb_rows; i++) {
      output_matrix_cm[cm_offset + i] = 0.;
      for (int j = 0; j < nb_rows; j++) {
        output_matrix_cm[cm_offset + i] +=
            matrix_data_rm[offset + j] * input_matrix_cm[cm_offset + j];
      }
      offset += nb_rows;
    }

    cm_offset += nb_rows;
  }
}
} // namespace Generic

#endif