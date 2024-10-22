#ifndef DENSE_DIRECT_SOLVER_GENERIC_HPP
#define DENSE_DIRECT_SOLVER_GENERIC_HPP

#include "generic_vector.hpp"
#include <utility>

using std::pair;

namespace Generic {

template <typename T, std::size_t xdim>
void ReadLowerUpper(const Generic::Vector<T, xdim * xdim> &matrix_data,
                    Generic::Vector<T, xdim *(xdim - 1) / 2> &lower_data,
                    Generic::Vector<T, xdim *(xdim + 1) / 2> &upper_data,
                    const size_t rank) {
  const size_t nb_rows = (xdim == 0) ? rank : xdim;
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

template <typename T, std::size_t xdim>
void FactorizeLU(Generic::Vector<T, xdim *(xdim - 1) / 2> &lower_data,
                 Generic::Vector<T, xdim *(xdim + 1) / 2> &upper_data,
                 const size_t rank) {
  const size_t nb_rows = (xdim == 0) ? rank : xdim;
  const size_t nb_cols = nb_rows;

  int upper_pivot_offset = 0;
  int lower_pivot_offset = 0;

  for (int j = 0; j < nb_cols - 1; j++) {

    // Compute diagonal term
    T pivot1 = 1. / upper_data[upper_pivot_offset];

    int lower_offset = lower_pivot_offset;
    int upper_offset = upper_pivot_offset + (nb_cols - j);

    for (int i = j + 1; i < nb_rows; i++) {

      // Compute L[i,j]
      lower_data[lower_offset] *= pivot1;

      T gauss_coeff = lower_data[lower_offset];

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

template <typename T, std::size_t xdim>
void LowerSolve(const Generic::Vector<T, xdim *(xdim - 1) / 2> &lower_data,
                Generic::Vector<T, xdim> &rhs, const size_t rank) {
  const size_t nb_rows = (xdim == 0) ? rank : xdim;

  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < i; j++) {
      rhs[i] -= lower_data[offset + j] * rhs[j];
    }
    offset += i;
  }
}

template <typename T, std::size_t xdim>
void UpperSolve(const Generic::Vector<T, xdim *(xdim + 1) / 2> &upper_data,
                Generic::Vector<T, xdim> &rhs, const size_t rank) {
  const size_t nb_rows = (xdim == 0) ? rank : xdim;
  const size_t last_row_id = nb_rows - 1;

  int offset = nb_rows * (nb_rows + 1) / 2 - 1;
  for (int i = 0; i < nb_rows; i++) {

    int row_id = last_row_id - i;
    T pivot1 = 1. / upper_data[offset];

    for (int j = 1; j < i + 1; j++) {
      rhs[row_id] -= upper_data[offset + j] * rhs[row_id + j];
    }
    rhs[row_id] *= pivot1;

    offset -= (i + 2);
  }
}

template <typename T, std::size_t xdim = 0>
void LUSolve(const Generic::Vector<T, xdim * xdim> &matrix_data,
             Generic::Vector<T, xdim> &rhs,
             pair<Generic::Vector<T, xdim *(xdim - 1) / 2>,
                  Generic::Vector<T, xdim *(xdim + 1) / 2>> &lu_resources,
             const size_t rank) {
  auto &lower_data = lu_resources.first;
  auto &upper_data = lu_resources.second;

  ReadLowerUpper<T, xdim>(matrix_data, lower_data, upper_data, rank);

  FactorizeLU<T, xdim>(lower_data, upper_data, rank);

  LowerSolve<T, xdim>(lower_data, rhs, rank);
  UpperSolve<T, xdim>(upper_data, rhs, rank);
}

template <typename T, std::size_t xdim = 0>
double UpperDeterminant(const Generic::Vector<T, xdim> &upper_data,
                        const size_t rank) {
  const int nb_rows = (xdim == 0) ? rank : xdim;

  assert(2 * upper_data.size() >= nb_rows * (nb_rows + 1));

  T det_val = 1.;
  int diag_id = 0;
  for (int row_id = 0; row_id < nb_rows; row_id++) {
    det_val *= upper_data[diag_id];
    diag_id += (nb_rows - row_id);
  }

  return det_val;
}

} // namespace Generic

#endif