#ifndef DENSE_DIRECT_SOLVER_H
#define DENSE_DIRECT_SOLVER_H

#include <vector>

using std::vector;

void ReadLowerUpper(const vector<double> &matrix_data,
                    vector<double> &lower_data, vector<double> &upper_data,
                    const size_t xdim);
void FactorizeLU(vector<double> &lower_data, vector<double> &upper_data,
                 const size_t xdim);
void LowerSolve(const vector<double> &lower_data, vector<double> &rhs,
                const size_t xdim);
void UpperSolve(const vector<double> &upper_data, vector<double> &rhs,
                const size_t xdim);

void LUSolve(const vector<double> &matrix_data, vector<double> &rhs,
             const size_t xdim, vector<vector<double>> &resources);

// Matrix-Matrix
void LUMatrixSolve(const vector<double> &matrix_data,
                   vector<double> &rhs_matrix_cm, const size_t xdim,
                   const size_t zdim, vector<vector<double>> &resources);
void LowerMatrixSolve(const vector<double> &lower_data,
                      vector<double> &rhs_matrix_cm, const size_t xdim,
                      const size_t zdim);
void UpperMatrixSolve(const vector<double> &upper_data,
                      vector<double> &rhs_matrix_cm, const size_t xdim,
                      const size_t zdim);

vector<vector<double>> AllocateLUResources(const size_t xdim);

#endif