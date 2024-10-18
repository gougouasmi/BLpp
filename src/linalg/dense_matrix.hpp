#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <cassert>
#include <utility>
#include <vector>

using std::pair;
using std::vector;

using MatrixDims = std::tuple<int, int>;

class DenseMatrix {
public:
  DenseMatrix() = default;
  DenseMatrix(int xdim)
      : _Nx(xdim), _Ny(xdim), _data(xdim * xdim, 0.0),
        _lu_resources(std::vector<double>(0.5 * xdim * (xdim - 1), 0.),
                      std::vector<double>(0.5 * xdim * (xdim + 1), 0.)){};
  DenseMatrix(int xdim, int ydim)
      : _Nx(xdim), _Ny(ydim), _data(xdim * ydim, 0.0),
        _lu_resources(std::vector<double>(0.5 * xdim * (xdim - 1), 0.),
                      std::vector<double>(0.5 * xdim * (xdim + 1), 0.)) {
    assert(xdim == ydim);
  };
  DenseMatrix(const DenseMatrix &other_matrix) = delete;

  void Solve(const vector<double> &rhs, vector<double> &solution);
  void Solve(vector<double> &solution);

  void MatrixSolve(vector<double> &solution_matrix_cm, const size_t xdim,
                   const size_t zdim);

  MatrixDims GetDims() const { return MatrixDims(_Nx, _Ny); };
  vector<double> &GetData() { return _data; };
  pair<vector<double>, vector<double>> &GetLU() { return _lu_resources; };
  double Determinant();

private:
  int _Nx;
  int _Ny;
  vector<double> _data;
  pair<vector<double>, vector<double>> _lu_resources;
};

#endif