#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

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
      : _Nx(xdim), _data(xdim * xdim, 0.0),
        _lu_resources(std::vector<double>(0.5 * xdim * (xdim - 1), 0.),
                      std::vector<double>(0.5 * xdim * (xdim + 1), 0.)){};
  DenseMatrix(const DenseMatrix &other_matrix) = delete;

  void Solve(const vector<double> &rhs, vector<double> &solution);
  void Solve(vector<double> &solution);

  void MatrixSolve(vector<double> &solution_matrix_cm, const size_t xdim,
                   const size_t zdim);

  vector<double> &GetData() { return _data; };
  pair<vector<double>, vector<double>> &GetLU() { return _lu_resources; };
  double Determinant();

private:
  int _Nx;
  vector<double> _data;
  pair<vector<double>, vector<double>> _lu_resources;
};

#endif