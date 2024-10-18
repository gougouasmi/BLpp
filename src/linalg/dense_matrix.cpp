#include "dense_matrix.hpp"
#include "dense_direct_solver.hpp"

void DenseMatrix::Solve(const std::vector<double> &rhs,
                        std::vector<double> &solution) {
  if (_Nx == 1) {
    solution[0] = rhs[0] / _data[0];
    return;
  }

  for (int idx = 0; idx < _Nx; idx++) {
    solution[idx] = rhs[idx];
  }

  LUSolve(_data, solution, _Nx, _lu_resources);
}

void DenseMatrix::Solve(std::vector<double> &solution) {

  if (_Nx == 1) {
    solution[0] /= _data[0];
    return;
  }

  LUSolve(_data, solution, _Nx, _lu_resources);
}

void DenseMatrix::MatrixSolve(vector<double> &solution_matrix_cm,
                              const size_t xdim, const size_t zdim) {
  LUMatrixSolve(_data, solution_matrix_cm, xdim, zdim, _lu_resources);
}

double DenseMatrix::Determinant() {
  assert(_Nx == _Ny);
  return UpperDeterminant(_lu_resources.second, _Nx);
}