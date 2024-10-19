#ifndef DENSE_DIRECT_SOLVER_ARRAY_HPP
#define DENSE_DIRECT_SOLVER_ARRAY_HPP

#include <array>

namespace ArraySolver {

template <typename T, std::size_t xdim>
void ReadLowerUpper(const array<T, xdim * xdim> &matrix_data,
                    array<T, xdim *(xdim - 1) / 2> &lower_data,
                    array<T, xdim *(xdim + 1) / 2> &upper_data) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t nb_cols = nb_rows;

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
void FactorizeLU(array<T, xdim *(xdim - 1) / 2> &lower_data,
                 array<T, xdim *(xdim + 1) / 2> &upper_data) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t nb_cols = nb_rows;

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
void LowerSolve(const array<T, xdim *(xdim - 1) / 2> &lower_data,
                array<T, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < i; j++) {
      rhs[i] -= lower_data[offset + j] * rhs[j];
    }
    offset += i;
  }
}

template <typename T, std::size_t xdim>
void UpperSolve(const array<T, xdim *(xdim + 1) / 2> &upper_data,
                array<T, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t last_row_id = nb_rows - 1;

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

template <typename T, std::size_t xdim>
void LUSolve(const array<T, xdim * xdim> &matrix_data, array<T, xdim> &rhs,
             pair<array<T, xdim *(xdim - 1) / 2>,
                  array<T, xdim *(xdim + 1) / 2>> &lu_resources) {

  auto &lower_data = lu_resources.first;
  auto &upper_data = lu_resources.second;

  ReadLowerUpper<T, xdim>(matrix_data, lower_data, upper_data);

  FactorizeLU<T, xdim>(lower_data, upper_data);

  LowerSolve<T, xdim>(lower_data, rhs);
  UpperSolve<T, xdim>(upper_data, rhs);
}

} // namespace ArraySolver

#endif