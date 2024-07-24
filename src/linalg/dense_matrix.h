#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <cassert>
#include <vector>

using MatrixDims = std::tuple<int, int>;

class DenseMatrix {
public:
  DenseMatrix(int xdim) : _Nx(xdim), _Ny(xdim), _data(xdim * xdim, 0.0){};
  DenseMatrix(int xdim, int ydim)
      : _Nx(xdim), _Ny(ydim), _data(xdim * ydim, 0.0) {
    assert(xdim == ydim);
  };
  DenseMatrix() = delete;
  DenseMatrix(const DenseMatrix &other_matrix) = delete;

  void Solve(const std::vector<double> &rhs, std::vector<double> &solution);

  MatrixDims GetDims() const { return MatrixDims(_Nx, _Ny); };
  std::vector<double> &GetData() { return _data; };

private:
  int _Nx;
  int _Ny;
  std::vector<double> _data;
};

#endif