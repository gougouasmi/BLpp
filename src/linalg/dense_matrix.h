#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <cassert>
#include <vector>

using std::vector;

using MatrixDims = std::tuple<int, int>;

class DenseMatrix {
public:
  DenseMatrix(int xdim) : _Nx(xdim), _Ny(xdim), _data(xdim * xdim, 0.0) {
    _resources.emplace_back(0.5 * xdim * (xdim - 1), 0.);
    _resources.emplace_back(0.5 * xdim * (xdim + 1), 0.);
  };
  DenseMatrix(int xdim, int ydim)
      : _Nx(xdim), _Ny(ydim), _data(xdim * ydim, 0.0) {
    assert(xdim == ydim);
    _resources.emplace_back(0.5 * xdim * (xdim - 1), 0.);
    _resources.emplace_back(0.5 * xdim * (xdim + 1), 0.);
  };
  DenseMatrix() = delete;
  DenseMatrix(const DenseMatrix &other_matrix) = delete;

  void Solve(const vector<double> &rhs, vector<double> &solution);

  MatrixDims GetDims() const { return MatrixDims(_Nx, _Ny); };
  vector<double> &GetData() { return _data; };

private:
  int _Nx;
  int _Ny;
  vector<double> _data;
  vector<vector<double>> _resources;
};

#endif