#ifndef DENSE_LINALG_H
#define DENSE_LINALG_H

#include <vector>

void DenseMatrixMultiply(const std::vector<double> &matrix_data_rm,
                         const std::vector<double> &input_vector,
                         std::vector<double> &output_vector, int xdim);

void DenseMatrixMatrixMultiply(const std::vector<double> &matrix_data_rm,
                               const std::vector<double> &input_matrix_cm,
                               std::vector<double> &output_matrix_cm, int xdim,
                               int zdim);

#endif