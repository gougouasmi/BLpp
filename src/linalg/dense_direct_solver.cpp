#include "dense_direct_solver.h"

#include <cassert>

/*
 * Read dense matrix into
 * Lower/Upper triangular matrices.
 *
 * Row-Major order.
 */
void ReadLowerUpper(const vector<double> &matrix_data,
                    vector<double> &lower_data, vector<double> &upper_data,
                    const size_t xdim) {
  const size_t nb_rows = xdim;
  const size_t nb_cols = nb_rows;

  int upper_offset = 0;
  int lower_offset = 0;

  for (int row_id = 0; row_id < nb_rows; row_id++) {

    for (int col_id = 0; col_id < row_id; col_id++) {
      lower_data[lower_offset + col_id] =
          matrix_data[row_id * nb_cols + col_id];
    }

    for (int col_id = row_id; col_id < nb_cols; col_id++) {
      upper_data[upper_offset + (col_id - row_id)] =
          matrix_data[row_id * nb_cols + col_id];
    }

    upper_offset += (nb_cols - row_id);
    lower_offset += row_id;
  }
}

/*
 * Perform dense LU factorization in-place.
 * Row-Major order.
 */
void FactorizeLU(vector<double> &lower_data, vector<double> &upper_data,
                 const size_t xdim) {
  const size_t nb_rows = xdim;
  const size_t nb_cols = nb_rows;

  assert(2 * lower_data.size() == nb_rows * (nb_rows - 1));
  assert(upper_data.size() == lower_data.size() + nb_rows);

  int upper_pivot_offset = 0;
  int lower_pivot_offset = 0;

  for (int j = 0; j < nb_cols - 1; j++) {

    // Compute diagonal term
    double pivot1 = 1. / upper_data[upper_pivot_offset];

    int lower_offset = lower_pivot_offset;
    int upper_offset = upper_pivot_offset + (nb_cols - j);

    for (int i = j + 1; i < nb_rows; i++) {

      // Compute L[i,j]
      lower_data[lower_offset] *= pivot1;

      double gauss_coeff = lower_data[lower_offset];

      // Compute L[i, j:i]
      for (int jb = j + 1; jb < i; jb++) {
        lower_data[lower_offset + jb - j] -=
            (gauss_coeff * upper_data[upper_pivot_offset + jb - j]);
      }

      // Compute U[i, i:nb_cols]
      for (int jb = i; jb < nb_cols; jb++) {
        upper_data[upper_offset + (jb - i)] -=
            (gauss_coeff * upper_data[upper_pivot_offset + jb - j]);
      }

      // Update offset
      lower_offset += i;
      upper_offset += (nb_cols - i);
    }

    // Update pivot offset
    upper_pivot_offset += (nb_cols - j);
    lower_pivot_offset += (j + 2);
  }
}

/*
 * Solve dense lower triangular system in-place.
 * Row-Major order.
 */
void LowerSolve(const vector<double> &lower_data, vector<double> &rhs,
                const size_t xdim) {
  const size_t nb_rows = xdim;
  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < i; j++) {
      rhs[i] -= lower_data[offset + j] * rhs[j];
    }
    offset += i;
  }
}

/*
 * Solve dense upper triangular system in-place.
 * Row-Major order.
 */
void UpperSolve(const vector<double> &upper_data, vector<double> &rhs,
                const size_t xdim) {
  const size_t nb_rows = xdim;
  const size_t last_row_id = nb_rows - 1;

  int offset = nb_rows * (nb_rows + 1) / 2 - 1;
  for (int i = 0; i < nb_rows; i++) {

    int row_id = last_row_id - i;
    double pivot1 = 1. / upper_data[offset];

    for (int j = 1; j < i + 1; j++) {
      rhs[row_id] -= upper_data[offset + j] * rhs[row_id + j];
    }
    rhs[row_id] *= pivot1;

    offset -= (i + 2);
  }
}

/*
 * Direct LU solver. Row-major order
 */
void LUSolve(const vector<double> &matrix_data, vector<double> &rhs,
             const size_t xdim, vector<vector<double>> &resources) {
  assert(xdim > 1);
  assert(matrix_data.size() == xdim * xdim);

  const size_t lower_size = xdim * (xdim - 1) / 2;
  const size_t upper_size = lower_size + xdim;

  vector<double> &lower_data = resources[0];
  vector<double> &upper_data = resources[1];

  assert(lower_data.size() == lower_size);
  assert(upper_data.size() == upper_size);

  ReadLowerUpper(matrix_data, lower_data, upper_data, xdim);

  FactorizeLU(lower_data, upper_data, xdim);

  LowerSolve(lower_data, rhs, xdim);
  UpperSolve(upper_data, rhs, xdim);
}

/////
// Matrix-matrix
//

void LUMatrixSolve(const vector<double> &matrix_data,
                   vector<double> &rhs_matrix_cm, const size_t xdim,
                   const size_t zdim, vector<vector<double>> &resources) {
  assert(xdim > 1);
  assert(matrix_data.size() == xdim * xdim);
  assert(rhs_matrix_cm.size() == xdim * zdim);

  const size_t lower_size = xdim * (xdim - 1) / 2;
  const size_t upper_size = lower_size + xdim;

  vector<double> &lower_data = resources[0];
  vector<double> &upper_data = resources[1];

  assert(lower_data.size() == lower_size);
  assert(upper_data.size() == upper_size);

  ReadLowerUpper(matrix_data, lower_data, upper_data, xdim);

  FactorizeLU(lower_data, upper_data, xdim);

  LowerMatrixSolve(lower_data, rhs_matrix_cm, xdim, zdim);
  UpperMatrixSolve(upper_data, rhs_matrix_cm, xdim, zdim);
}

void LowerMatrixSolve(const vector<double> &lower_data,
                      vector<double> &rhs_matrix_cm, const size_t xdim,
                      const size_t zdim) {
  const size_t nb_rows = xdim;
  int cm_offset = 0;

  for (int k = 0; k < zdim; k++) {
    int offset = 0;
    for (int i = 0; i < nb_rows; i++) {
      for (int j = 0; j < i; j++) {
        rhs_matrix_cm[cm_offset + i] -=
            lower_data[offset + j] * rhs_matrix_cm[cm_offset + j];
      }
      offset += i;
    }
    cm_offset += xdim;
  }
}

void UpperMatrixSolve(const vector<double> &upper_data,
                      vector<double> &rhs_matrix_cm, const size_t xdim,
                      const size_t zdim) {
  const size_t nb_rows = xdim;
  const size_t last_row_id = nb_rows - 1;

  int cm_offset = 0;

  for (int k = 0; k < zdim; k++) {
    int offset = nb_rows * (nb_rows + 1) / 2 - 1;

    for (int i = 0; i < nb_rows; i++) {

      int row_id = last_row_id - i;
      double pivot1 = 1. / upper_data[offset];

      for (int j = 1; j < i + 1; j++) {
        rhs_matrix_cm[cm_offset + row_id] -=
            upper_data[offset + j] * rhs_matrix_cm[cm_offset + row_id + j];
      }
      rhs_matrix_cm[cm_offset + row_id] *= pivot1;

      offset -= (i + 2);
    }
    cm_offset += xdim;
  }
}

// Resource allocation
vector<vector<double>> AllocateLUResources(const size_t xdim) {
  vector<vector<double>> resources;

  resources.emplace_back(0.5 * xdim * (xdim - 1), 0.);
  resources.emplace_back(0.5 * xdim * (xdim + 1), 0.);

  return std::move(resources);
}
