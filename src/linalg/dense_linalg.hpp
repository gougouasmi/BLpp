#ifndef DENSE_LINALG_HPP
#define DENSE_LINALG_HPP

#include <vector>

void DenseMatrixMultiply(const std::vector<double> &matrix_data_rm,
                         const std::vector<double> &input_vector,
                         std::vector<double> &output_vector, const size_t xdim);

void DenseMatrixMatrixMultiply(const std::vector<double> &matrix_data_rm,
                               const std::vector<double> &input_matrix_cm,
                               std::vector<double> &output_matrix_cm,
                               const size_t xdim, const size_t zdim);

#endif