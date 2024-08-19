#ifndef DENSE_DIRECT_SOLVER_H
#define DENSE_DIRECT_SOLVER_H

#include <vector>

void ReadLowerUpper(const std::vector<double> &matrix_data,
                    std::vector<double> &lower_data,
                    std::vector<double> &upper_data, int xdim);
void FactorizeLU(std::vector<double> &lower_data,
                 std::vector<double> &upper_data, int xdim);
void LowerSolve(const std::vector<double> &lower_data, std::vector<double> &rhs,
                int xdim);
void UpperSolve(const std::vector<double> &upper_data, std::vector<double> &rhs,
                int xdim);

void LUSolve(const std::vector<double> &matrix_data, std::vector<double> &rhs,
             int xdim);
void Multiply(const std::vector<double> &matrix_data,
              const std::vector<double> &input_vector,
              std::vector<double> &output_vector, int xdim);

// Matrix-Matrix
void LUMatrixSolve(const std::vector<double> &matrix_data,
                   std::vector<double> &rhs_matrix_cm, int xdim, int zdim);
void LowerMatrixSolve(const std::vector<double> &lower_data,
                      std::vector<double> &rhs_matrix_cm, int xdim, int zdim);
void UpperMatrixSolve(const std::vector<double> &upper_data,
                      std::vector<double> &rhs_matrix_cm, int xdim, int zdim);
void MatrixMultiply(const std::vector<double> &matrix_data,
                    const std::vector<double> &input_matrix_cm,
                    std::vector<double> &output_matrix_cm, int xdim, int zdim);

#endif