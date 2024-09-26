#include "dense_matrix.h"
#include "dense_direct_solver.h"

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
};
