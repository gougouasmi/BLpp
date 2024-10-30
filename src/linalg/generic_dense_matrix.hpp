#ifndef DENSE_GENERIC_MATRIX_HPP
#define DENSE_GENERIC_MATRIX_HPP

#include <cassert>
#include <utility>
#include <vector>

#include "generic_dense_direct_solver.hpp"
#include "generic_vector.hpp"

namespace Generic {

template <std::size_t ctime_xdim = 0> class DenseMatrix {
  using LUPair =
      std::pair<Generic::Vector<double, ctime_xdim *(ctime_xdim - 1) / 2>,
                Generic::Vector<double, ctime_xdim *(ctime_xdim + 1) / 2>>;

public:
  DenseMatrix(const DenseMatrix &other_matrix) = delete;
  DenseMatrix(int xdim) {
    if constexpr (ctime_xdim == 0) {
      _Nx = xdim;
      _data.resize(xdim * xdim);
      _lu_resources.first.resize(0.5 * xdim * (xdim - 1));
      _lu_resources.second.resize(0.5 * xdim * (xdim + 1));
    } else {
      _Nx = ctime_xdim;
      assert(ctime_xdim == xdim);
    }
  };
  DenseMatrix() {
    if constexpr (ctime_xdim != 0) {
      _Nx = ctime_xdim;
    }
  };

  void Solve(Generic::Vector<double, ctime_xdim> &solution) {
    Generic::LUSolve<double, ctime_xdim>(_data, solution, _lu_resources, _Nx);
  }

  void Solve(const Generic::Vector<double, ctime_xdim> &rhs,
             Generic::Vector<double, ctime_xdim> &solution) {
    for (int idx = 0; idx < _Nx; idx++) {
      solution[idx] = rhs[idx];
    }
    Solve(solution);
  }

  template <std::size_t ctime_zdim>
  void MatrixSolve(
      Generic::Vector<double, ctime_xdim * ctime_zdim> &solution_matrix_cm,
      size_t xdim, size_t zdim) {
    Generic::LUMatrixSolve<double, ctime_xdim, ctime_zdim>(
        _data, solution_matrix_cm, _lu_resources, xdim, zdim);
  }

  Generic::Vector<double, ctime_xdim * ctime_xdim> &GetData() { return _data; };
  LUPair &GetLU() { return _lu_resources; };
  double Determinant() {
    return Generic::UpperDeterminant(_lu_resources.second, _Nx);
  };

private:
  int _Nx;
  Generic::Vector<double, ctime_xdim * ctime_xdim> _data;
  LUPair _lu_resources;
};

} // namespace Generic

#endif